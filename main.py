import io  
import os  
import csv
import pytz  
import time  
import json  
import shutil  
import signal  
import zipfile  
import uvicorn  
import requests  
import threading  
from threading import Thread, Event  
from datetime import datetime  
from watchdog.observers import Observer  
from fastapi import FastAPI, UploadFile, File, Form, Query  
from fastapi.responses import FileResponse  
from api.whisper_finetune import WhisperFineTuner  
from api.threading_api import whisper_finetune, stop_thread, abnormal_file_deletion, LatestRetention  
from api.utils import zip_checkpoint  
from lib.constant import TMPPATH, MODELPATH, AUDIOPATH, TASKPATH, JSONPATH, TESTPATH, ZIP_REQUIRED, SAFETENSORS_REQUIRED  
from lib.base_object import BaseResponse  
from lib.log_config import setup_sys_logging, setup_watchdog_logging  
  
#############################################################################  
# If the "logs" directory does not exist, create it.  
if not os.path.exists("./logs"):  
    os.mkdir("./logs")  
  
# Setup system and watchdog logging.  
logger = setup_sys_logging()  
wg_logger = setup_watchdog_logging()  
  
# Configure UTC+8 time zone.  
utc_now = datetime.now(pytz.utc)  
tz = pytz.timezone('Asia/Taipei')  
local_now = utc_now.astimezone(tz)  
  
# Initialize FastAPI app and other components.  
app = FastAPI()  
whisper = WhisperFineTuner()  
event_handler = LatestRetention()  
  
# Delete abnormal files in TMPPATH.  
abnormal_file_deletion(TMPPATH)  
task_pair = {}  
#############################################################################  
  
@app.get("/")  
def HelloWorld(name: str = None):  
    """  
    Return a greeting message.  
      
    :param name: str  
        Optional name to include in the greeting.  
    :return: dict  
        A dictionary with a greeting message.  
    """  
    return {"Hello": f"World {name}"}  
  
#############################################################################  
  
@app.get("/task_state")  
def task_state(task_id: str = Query(...)):  
    """  
    Get the state of a specific task based on the task ID.  
      
    :param task_id: str  
        The ID of the task.  
    :return: BaseResponse  
        A response object containing task state details.  
    """  
    task_path = os.path.join(TASKPATH, task_id)  
    zip_file = []  
      
    if not os.path.exists(task_path):  
        logger.error(f" | Task ID '{task_id}' not found. | ")  
        return BaseResponse(status="FAILED", message=f" | Task ID '{task_id}' not found. | ", data=False)  
    else:  
        checkout_point = [f for f in os.listdir(task_path) if not f.endswith('.zip')]  
        now_process = whisper.task_flags.get(task_id)  
        for item in os.listdir(task_path):  
            if item.endswith('.zip'):  
                zip_file.append(item)  
        state = "running" if now_process else "stopped"  
        progress = whisper.task_progress.get(task_id, 0)  
        return_info = {  
            "task_id": task_id,  
            "state": state,  
            "progress": f"{progress:.2f}%",  
            "now_process": now_process,  
            "checkout_point": checkout_point,  
            "zip_file": zip_file  
        }  
        logger.info(f" | Task ID '{task_id}' is {state}. | Refer process: {now_process} | finetune progress: {progress:.2f} | Retained checkpoint: {checkout_point} | Retained ZIP: {zip_file} | ")  
        return BaseResponse(  
            status="OK",  
            message=f" | Task ID '{task_id}' is {state}. | Refer process: {now_process} | finetune progress: {progress:.2f} | Retained checkpoint: {checkout_point} | Retained ZIP: {zip_file} | ",  
            data=return_info  
        )  
  
@app.get("/get_model_list")  
def get_model_list():  
    """  
    Get a list of available models.  
      
    :return: BaseResponse  
        A response object containing the list of models.  
    """  
    model_list = []  
    for root, dirs, files in os.walk(MODELPATH):  
        for dir_name in dirs:  
            model_list.append(dir_name)  
    logger.info(f" | Model list: {model_list} | ")  
    return BaseResponse(status="OK", message=f" | Model list: {model_list} | ", data=model_list)  
  
@app.get("/get_task_list")  
def get_task_list():  
    """  
    Get a list of available tasks.  
      
    :return: BaseResponse  
        A response object containing the list of tasks.  
    """  
    task_list = [dir_name for dir_name in os.listdir(TASKPATH) if os.path.isdir(os.path.join(TASKPATH, dir_name))]  
    logger.info(f" | Task list: {task_list} | ")  
    return BaseResponse(status="OK", message=f" | Task list: {task_list} | ", data=task_list)  
  
@app.get("/get_audio")  
async def get_audio(task_id: str, url: str = "http://172.17.0.1:52010/get_audio"):  
    """  
    Download and extract audio files for a specific task.  
      
    :param task_id: str  
        The ID of the task.  
    :param url: str  
        The URL to download the audio files from.  
    :return: BaseResponse  
        A response object indicating the success or failure of the operation.  
    """  
    out_path = os.path.join(AUDIOPATH, task_id)  
    params = {"task_id": task_id}  
      
    try:  
        response = requests.get(url, params=params)  
    except requests.exceptions.RequestException as e:  
        logger.error(f" | Error occurred while making the request: {e} | ")  
        return BaseResponse(status="FAILED", message=" | Error occurred while making the request: {e} | ", data=False)  
      
    if response.status_code == 200:  
        if response.headers.get('Content-Type') == 'application/zip':  
            try:  
                with io.BytesIO(response.content) as zip_memory_file:  
                    with zipfile.ZipFile(zip_memory_file, 'r') as zip_ref:  
                        os.makedirs(out_path, exist_ok=True)  
                        zip_ref.extractall(out_path)  
                logger.info(f" | Files extracted to {out_path} successfully. | ")  
                return BaseResponse(status="OK", message=" | Files downloaded and extracted successfully. | ", data=True)  
            except zipfile.BadZipFile as e:  
                logger.error(f" | Failed to unzip file: {e} | ")  
                return BaseResponse(status="FAILED", message=f" | Error extracting zip file: {e} | ", data=False)  
        else:  
            try:  
                error_message = response.json().get('message', 'Unexpected content type')  
            except ValueError:  
                error_message = 'Unexpected content type and failed to parse JSON response'  
            logger.error(f" | Something error: {error_message} | ")  
            return BaseResponse(status="FAILED", message=error_message, data=False)  
    else:  
        logger.error(f" | Failed to download audio file. Status code: {response.status_code} | ")  
        return BaseResponse(status="FAILED", message=f" | Failed to download audio file. Status code: {response.status_code} | ", data=False)  
  
@app.post("/weight_upload")  
async def weight_upload(file: UploadFile = File(...)):  
    """  
    Upload and save a model weight file.  
      
    :param file: UploadFile  
        The file to be uploaded.  
    :return: BaseResponse  
        A response object indicating the success or failure of the upload.  
    """  
    save_flag = False  
    model_folder = file.filename.replace(".zip", "")  
      
    # Check if the file is a ZIP file.  
    if not file.filename.endswith(".zip"):  
        logger.error(f" | Invalid file type. Only ZIP files are allowed. | ")  
        return BaseResponse(status="FAILED", message=f" | Invalid file type. Only ZIP files are allowed. | ", data=False)  
      
    # Check if the model folder already exists.  
    model_path = os.path.join(MODELPATH, model_folder)  
    if os.path.exists(model_path):  
        logger.error(f" | Model '{model_folder}' already exists. Please delete it first or rename and upload again. | ")  
        return BaseResponse(status="FAILED", message=f" | Model '{model_folder}' already exists. Please delete it first or rename and upload again. | ", data=False)  
      
    logger.info(f" | Get the upload file '{file.filename}' | ")  
    logger.info(f" | Start to parse file | ")  
    file_content = await file.read()  
      
    # Check the contents of the ZIP file.  
    with zipfile.ZipFile(io.BytesIO(file_content)) as zip_file:  
        zip_file_contents = zip_file.namelist()  
          
        # Check if the ZIP file contains a folder structure.  
        if all(item.startswith(model_folder + "/") for item in zip_file_contents):  
            save_flag = True  
            zip_file_contents = [item[len(model_folder) + 1:] for item in zip_file_contents if item.startswith(model_folder + "/")]  
          
        missing_files = [filename for filename in ZIP_REQUIRED if filename not in zip_file_contents]  
        if not any(file.endswith(".safetensors") for file in zip_file_contents):  
            missing_files.append(f"model.safetensors or model-00001-of-0000n.safetensors ...")  
          
        model_chunk_files = [file for file in zip_file_contents if file.startswith("model-00001-of-")]  
        if model_chunk_files and SAFETENSORS_REQUIRED not in zip_file_contents:  
            missing_files.append(SAFETENSORS_REQUIRED)  
          
        if missing_files:  
            logger.info(f" | Model '{model_folder}' has some files missing: {missing_files} | ")  
            return BaseResponse(status="FAILED", message=f" | Model '{file.filename}' has some files missing: {missing_files} | ", data=False)  
        else:  
            logger.info(f" | Model '{model_folder}' has all required files | ")  
          
        if save_flag:  
            config = os.path.join(model_folder, "config.json")  
        else:  
            config = "config.json"  
        with zip_file.open(config) as config_file:  
            config_data = json.load(config_file)  
            name = config_data.get("_name_or_path", "N/A")  
            if name != "N/A":  
                name = name.split("/")[-1]  
            if name.startswith("whisper"):  
                name = name.replace("whisper", "").strip("-")  
            if name.endswith("v3"):  
                logger.error(f" | Model '{model_folder}' is not supported. Please use a different model. (Your model is '{name}' We only support tiny ~ large-v2) | ")  
                return BaseResponse(status="FAILED", message=f" | Model '{model_folder}' is not supported. Please use a different model. (Your model is '{name}' We only support tiny ~ large-v2) | ", data=False)  
            logger.info(f" | model size: '{name}' | ")  
          
        logger.info(f" | Start saving model: '{model_folder}' | ")  
        if save_flag:  
            model_path = MODELPATH  
        zip_file.extractall(model_path)  
        logger.info(f" | Model '{model_folder}' has been saved successfully | ")  
      
    return BaseResponse(status="OK", message=f" | Your model '{model_folder}' has been saved successfully | ", data=True)  
  
@app.delete("/delete_model")  
def delete_task(model_folder: str):  
    """  
    Delete a specific model based on the model folder name.  
      
    :param model_folder: str  
        The name of the model folder to be deleted.  
    :return: BaseResponse  
        A response object indicating the success or failure of the deletion.  
    """  
    output_path = os.path.join(MODELPATH, model_folder)  
      
    if not os.path.exists(output_path):  
        logger.error(f" | Please check the model name is correct | ")  
        return BaseResponse(status="FAILED", message=f" | model not found | ", data=False)  
    else:  
        shutil.rmtree(output_path)  
        logger.info(f" | Model name: '{model_folder}' has been deleted. | ")  
      
    return BaseResponse(status="OK", message=f" | Model name: '{model_folder}' has been deleted. | ", data=True)  
  
@app.post("/check_json_format")  
async def check_json_format(task_id: str, file: UploadFile = File(...)):  
    """  
    Check the format of a JSON file for a specific task.  
      
    :param task_id: str  
        The ID of the task.  
    :param file: UploadFile  
        The JSON file to be checked.  
    :return: BaseResponse  
        A response object indicating whether the JSON format is correct.  
    """  
    audio_path = os.path.join(AUDIOPATH, task_id)  
      
    if not os.path.exists(audio_path):  
        logger.error(f" | audio path not exists. Please check the task ID '{task_id}' is correct | ")  
        return BaseResponse(status="FAILED", message=f" | audio path not exists. Please check the task ID '{task_id}' is correct | ", data=False)  
      
    if not file.filename.endswith(".json"):  
        logger.error(f" | Invalid file type. Only JSON files are allowed. | ")  
        return BaseResponse(status="FAILED", message=f" | Invalid file type. Only JSON files are allowed. | ", data=False)  
      
    logger.info(f" | Get the upload file '{file.filename}' | ")  
    logger.info(f" | Start to parse file | ")  
    file_content = await file.read()  
    try:  
        json_data = json.loads(file_content)  
    except json.JSONDecodeError:  
        logger.error(f" | Invalid JSON format in file | ")  
        return BaseResponse(status="FAILED", message=f" | Invalid JSON format in file | ", data=False)  
      
    if not isinstance(json_data, list):  
        logger.error(f" | JSON content is not a list | ")  
        return BaseResponse(status="FAILED", message=f" | JSON content is not a list | ", data=False)  
      
    for line in json_data:  
        if not isinstance(line, dict):  
            logger.error(f" | Invalid JSON object format in list: {line} | ")  
            return BaseResponse(status="FAILED", message=f" | Invalid JSON object format in list: {line} | ", data=False)  
          
        if 'audio' not in line or 'text' not in line:  
            logger.error(f" | Missing 'audio' or 'text' in object: {line} | ")  
            return BaseResponse(status="FAILED", message=f" | Missing 'audio' or 'text' in object: {line} | ", data=False)  
          
        audio = os.path.join(audio_path, line['audio'])  
        if not os.path.exists(audio):  
            logger.error(f" | 'audio' does not exist: {line['audio']} | ")  
            return BaseResponse(status="FAILED", message=f" | 'audio' does not exist: {line['audio']} | ", data=False)  
      
    logger.info(f" | JSON format is correct. | ")  
    return BaseResponse(status="OK", message=f" | JSON format is correct. | ", data=True)  
  
@app.post("/finetune")  
def finetune(task_id: str,  
             pretrain_model: str = "openai/whisper-tiny",  
             batch_size: int = 6,  
             gradient_accumulation_steps: int = 2,  
             epochs: int = 5,  
             learning_rate: float = 1e-5,  
             warmup_steps: int = 1000,  
             logging_steps: int = 10,  
             save_steps: int = 1000,  
             file: UploadFile = File(...)):  
    """  
    Start the fine-tuning process for a specific task.  
      
    :param task_id: str  
        The ID of the task.  
    :param pretrain_model: str  
        The name of the pre-trained model.  
    :param batch_size: int  
        The batch size for fine-tuning.  
    :param gradient_accumulation_steps: int  
        The number of gradient accumulation steps.  
    :param epochs: int  
        The number of epochs for fine-tuning.  
    :param learning_rate: float  
        The learning rate for fine-tuning.  
    :param warmup_steps: int  
        The number of warmup steps for fine-tuning.  
    :param logging_steps: int  
        The number of steps between logging.  
    :param save_steps: int  
        The number of steps between saving checkpoints.  
    :param file: UploadFile  
        The JSON file containing the fine-tuning data.  
    :return: BaseResponse  
        A response object indicating the success or failure of the fine-tuning process.  
    """  
    if whisper.task_flags.get(task_id):  
        logger.error(f" | Task ID '{task_id}' is already running. | ")  
        return BaseResponse(status="FAILED", message=f" | Task ID '{task_id}' is already running. | ", data=False)  
      
    audio_path = os.path.join(AUDIOPATH, task_id)  
    if not os.path.exists(audio_path):  
        logger.error(f" | audio path not exists. Please check the task ID '{task_id}' is correct | ")  
        return BaseResponse(status="FAILED", message=f" | audio path not exists. Please check the task ID '{task_id}' is correct | ", data=False)  
      
    json_file_path = os.path.join(JSONPATH, file.filename)  
    os.makedirs(JSONPATH, exist_ok=True)  
    with open(json_file_path, "wb") as f:  
        f.write(file.file.read())  
      
    output_path = os.path.join(TASKPATH, task_id)  
    if os.path.exists(output_path):  
        shutil.rmtree(output_path)  
        os.makedirs(output_path, exist_ok=True)  
        logger.info(f" | Task ID '{task_id}' already exists. Old task has been deleted. | ")  
    else:  
        os.makedirs(output_path, exist_ok=True)  
      
    if not pretrain_model.startswith("openai/whisper-"):  
        pretrain_model = os.path.join(MODELPATH, pretrain_model)  
        if not os.path.exists(pretrain_model):  
            logger.error(f" | Model '{pretrain_model}' not found. Please check the model name is correct | ")  
            return BaseResponse(status="FAILED", message=f" | Model '{pretrain_model}' not found. Please check the model name is correct | ", data=False)  
      
    args = {  
        "output_dir": output_path,  
        "pretrain_model": pretrain_model,  
        "batch_size": batch_size,  
        "gradient_accumulation_steps": gradient_accumulation_steps,  
        "num_train_epochs": epochs,  
        "learning_rate": learning_rate,  
        "warmup_steps": warmup_steps,  
        "logging_steps": logging_steps,  
        "save_steps": save_steps  
    }  
    finetune_process = threading.Thread(target=whisper_finetune, args=(whisper, json_file_path, pretrain_model, args, task_id))  
    observer = Observer()  
    observer.schedule(event_handler, path=output_path, recursive=False)  
      
    finetune_process.start()  
    observer.start()  
      
    task_pair[task_id] = {"finetune_process": finetune_process, "watchdog": observer} 
    
    logger.info(f" | Fine Tuning task '{task_id}' has been started. | ")
    return BaseResponse(status="OK", message=f" | Fine Tuning task '{task_id}' has been started. | ", data=True)  


@app.post("/cancel_task")  
def cancel_task(task_id: str = Form(...)):
    """
    Cancel a running fine-tuning task.

    :param task_id: str  
        The ID of the task to cancel.  
    :return: BaseResponse  
        A response object indicating the success or failure of the cancellation.
    """
    if not task_pair.get(task_id) or whisper.task_flags[task_id] is False:
        logger.error(f" | Task ID '{task_id}' not running. | ")
        return BaseResponse(status="FAILED", message=f" | Task ID '{task_id}' not running. | ", data=False)
    
    try:    
        state = stop_thread(task_pair[task_id]["finetune_process"], task_id) 
        if not state:
            logger.error(f" | Try to stop task failed. Please try again latter | ")
            whisper.task_flags[task_id] = "Try to stop task failed"
            return BaseResponse(status="FAILED", message=f" | Try to stop task failed. Please try again latter | ", data=False)
        task_pair[task_id]["watchdog"].stop()  
        task_pair[task_id]["watchdog"].join()  
        whisper.task_flags[task_id] = False
    except Exception as e:
        logger.error(f" | Failed to stop thread: {e} | ")
        return BaseResponse(status="FAILED", message=f" | Failed to stop thread: {e} | ", data=False)
    
    
    logger.info(f" | Task ID '{task_id}' has been canceled | ")
    return BaseResponse(status="OK", message=f" | Task ID '{task_id}' has been canceled | ", data=True)

@app.delete("/delete_task")  
def delete_task(task_id: str = Form(...)):
    """
    Delete a specific task and its related resources.

    :param task_id: str  
        The ID of the task to delete.  
    :return: BaseResponse  
        A response object indicating whether the deletion was successful.
    """
    output_path = os.path.join(TASKPATH, task_id)
    if not os.path.exists(output_path):
        logger.error(f" | No task found. Please check the task ID '{task_id}' is correct. | ")
        return BaseResponse(status="FAILED", message=f" | No task found. Please check the task ID '{task_id}' is correct. | ", data=False)

    if whisper.task_flags.get(task_id):
        logger.info(f" | Task ID '{task_id}' is running. Try to stop task | ")
        try:    
            state = stop_thread(task_pair[task_id]["finetune_process"], task_id) 
            if not state:
                logger.error(f" | Try to stop task failed. Please try again latter | ")
                whisper.task_flags[task_id] = "Try to stop task failed"
                return BaseResponse(status="FAILED", message=f" | Failed to stop thread: {e} | ", data=False)
            task_pair[task_id]["watchdog"].stop()  
            task_pair[task_id]["watchdog"].join()  
            whisper.task_flags[task_id] = False
            logger.info(f" | Task {task_id} has been canceled | ")
        except Exception as e:
            logger.error(f" | Failed to stop thread: {e} | ")
            return BaseResponse(status="FAILED", message=f" | Failed to stop thread: {e} | ", data=False)
    
    output_path = os.path.join(TASKPATH, task_id)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)  
        
    json_path = os.path.join(JSONPATH, task_id+".json")
    if os.path.exists(json_path):
        os.remove(json_path)  
        
    audio_path = os.path.join(AUDIOPATH, task_id)
    if os.path.exists(audio_path):
        shutil.rmtree(audio_path)  
        
    logger.info(f" | Task {task_id} has been deleted | ")
    return BaseResponse(status="OK", message=f" | Task ID '{task_id}' has been deleted | ", data=True)
        
@app.put("/zip_model")
def zip_model(task_id: str = Form(...)):
    """
    Compress the model checkpoints for a specific task into a zip file.

    :param task_id: str  
        The ID of the task whose model checkpoints will be zipped.  
    :return: BaseResponse  
        A response object indicating the success or failure of the zipping process.
    """
    output_path = os.path.join(TASKPATH, task_id)
    if not os.path.exists(output_path):
        logger.error(f" | Task ID '{task_id}' not found. | ")
        return BaseResponse(status="FAILED", message=f" | Task ID '{task_id}' not found. | ", data=False)
    
    if whisper.task_flags.get(task_id):
        logger.error(f" | Task ID '{task_id}' is running. Please stop the task before zipping. | ")
        return BaseResponse(status="FAILED", message=f" | Task ID '{task_id}' is running. Please stop the task before zipping. | ", data=False)
    
    logger.info(f" | Task {task_id}: Start zip checkpoint | ")
    try:
        zip_checkpoint(output_path, task_id)
        if not os.path.isfile(os.path.join(output_path, f"{task_id}.zip")): 
            logger.error(f" | An error occurred while compressing the checkpoint. Please check whether the task fine-tuning has been successful. | ")
            return BaseResponse(status="FAILED", message=f" | An error occurred while compressing the checkpoint. Please check whether the task fine-tuning has been successful. | ", data=False)
        logger.info(f" | Task {task_id}: Successfully zipped | ")
        return BaseResponse(status="OK", message=f" | Task {task_id}: Successfully zipped | ", data=True)
    except Exception as e:
        logger.error(f" | Error zipping checkpoint: {e} | ")
        return BaseResponse(status="FAILED", message=f" | Error zipping checkpoint: {e} | ", data=False)
    
@app.get("/download")
def download(task_id: str):
    """
    Download the zipped model checkpoint for a specific task.

    :param task_id: str  
        The ID of the task to download the model from.  
    :return: FileResponse or BaseResponse  
        The zipped model file if successful, otherwise an error response.
    """
    output_path = os.path.join(TASKPATH, task_id)
    if not os.path.exists(output_path):
        logger.error(f" | Task ID '{task_id}' not found. Please check the task ID is correct | ")
        return BaseResponse(status="FAILED", message=f" | Task ID '{task_id}' not found. Please check the task ID is correct | ", data=False)
    
    zip_file = os.path.join(output_path, f"{task_id}.zip")
    if not os.path.isfile(zip_file):
        logger.info(f" | Task {task_id}: zip file not found. | ")
        logger.info(f" | Start zip checkpoint. (It may need some times) | ")
        start = time.time()
        try:
            zip_checkpoint(output_path, task_id)
            if not os.path.isfile(os.path.join(output_path, f"{task_id}.zip")): 
                logger.error(f" | An error occurred while compressing the checkpoint. Please check whether the task fine-tuning has been successful. | ")
                return BaseResponse(status="FAILED", message=f" | An error occurred while compressing the checkpoint. Please check whether the task fine-tuning has been successful. | ", data=False)
        except Exception as e:
            logger.error(f" | Error zipping checkpoint: {e} | ")
            return BaseResponse(status="FAILED", message=f" | Task ID '{task_id}': Error zipping checkpoint: {e} | ", data=False)
        end = time.time()
        logger.info(f" | ZIP archive {task_id}.zip completed. zip time: {end-start} | ")
    else:
        logger.info(f" | Task {task_id}: zip file found. | ")
    logger.info(f" | Try to transfer '{task_id}.zip'. | ")
    return FileResponse(zip_file, media_type='application/zip', filename=f"{task_id}.zip") 

@app.post("/inference")
def inference(task_id: str, audio_file: UploadFile = File(...)):
    """
    Perform inference using the latest checkpoint of a specific fine-tuned task.

    :param task_id: str  
        The ID of the task to run inference with.  
    :param audio_file: UploadFile  
        The audio file (.wav format) to transcribe.  
    :return: BaseResponse  
        A response object containing the transcription result and timing information.
    """
    task_path = os.path.join(TASKPATH, task_id)
    if not os.path.exists(task_path):
        logger.error(f" | task not exists. Please check the task ID '{task_id}' is correct | ")  
        return BaseResponse(status="FAILED", message=f" | audio path not exists. Please check the task ID '{task_id}' is correct | ", data=False)  
    
    checkpoints = os.listdir(task_path)
    checkpoints.sort(key=lambda x: int(x.split("-")[-1].split(".")[0]))
    if not checkpoints:
        logger.error(f" | task ID {task_id}: No checkpoint found. Please finish finetune first and try again. | ")  
        return BaseResponse(status="FAILED", message=f" | task ID {task_id}: No checkpoint found. Please finish finetune first and try again. | ", data=False) 
    else:
        latest_checkpoint = checkpoints[-1]
        logger.info(f" | The checkpoint is '{latest_checkpoint}' | ")
        checkpoint_path = os.path.join(task_path, latest_checkpoint)
    
    if not audio_file.filename.endswith(".wav"):  
        logger.error(f" | Invalid file type. Only WAV files are allowed. | ")  
        return BaseResponse(status="FAILED", message=f" | Invalid file type. Only WAV files are allowed. | ", data=False)  
    
    audio_file_path = os.path.join(TESTPATH, audio_file.filename)  
    os.makedirs(TESTPATH, exist_ok=True)
    with open(audio_file_path, "wb") as f:  
        f.write(audio_file.file.read())  
    
    logger.info(f" | Audio file '{audio_file.filename}' has been saved successfully | ")
    logger.info(f" | task ID {task_id}: Start inference | ")
    result, loading_time, inference_time = whisper.inference(checkpoint_path, audio_file_path)
    
    if not loading_time and not inference_time:
        logger.error(f" | Failed to load model/audio or inference error. Please check error information. | ")
        return BaseResponse(status="FAILED", message=f" | Failed to load model/audio or inference error. Please check error information | Error: '{result}' | ", data=False)
   
    logger.info(f" | transcription: {result} | model loading time: {loading_time:.2f} seconds | inference time: {inference_time:.2f} seconds | ")
    return BaseResponse(status="OK", message=f" | transcription: {result} | model loading time: {loading_time:.2f} seconds | inference time: {inference_time:.2f} seconds | ", data=result)

##############################################################################  

@app.post("/csv2json")  
async def csv2json(task_id: str, file: UploadFile = File(...)):  
    logger.info(f" | Start to read csv file | ")
    if file.filename.endswith(".csv"):
        content = await file.read()  
        decoded_content = content.decode("utf-8").splitlines()  
    else:
        logger.error(f" | We only support csv file | ")
        return BaseResponse(status="FAILED", message=f" | We only support csv file |", data=False)
  
    csv_reader = csv.DictReader(decoded_content)  
    result = []  
    for row in csv_reader:
        audio = row["output_audio_filename"]+".wav"
        # if os.isfile(os.path.join(AUDIOPATH, task_id, audio)):
        result.append({  
            "audio": audio,  
            "text": row["content_to_synthesize"]  
        })  
    
    logger.info(f" | Start to write json file | ")
    json_filename = os.path.join(JSONPATH, file.filename.replace(".csv", ".json"))  
    with open(json_filename, "w", encoding="utf-8") as json_file:  
        json.dump(result, json_file, ensure_ascii=False, indent=4)  
  
    logger.info(f" | Generate json file successful | Total available pair {len(result)} | ")
    # return BaseResponse(status="OK", message=f" | Generate json file successful | Total available pair {len(result)} | ", data=True)  
    return FileResponse(json_filename, media_type='application/json', filename=file.filename.replace(".csv", ".json"))  


##############################################################################  

# Clean up audio files  
def delete_old_files(directory, exception_files=None, age_limit_seconds=24*60*60):  
    """  
    Delete files older than age_limit_seconds from the specified directory, except for files in exception_files.  
  
    :param directory: Directory to delete files from  
    :param exception_files: List of filenames to skip  
    :param age_limit_seconds: Age limit in seconds for files to be deleted  
    """  
    current_time = time.time()  
    exception_files = exception_files or []  
  
    if os.path.exists(directory):  
        for filename in os.listdir(directory):  
            if filename in exception_files:  
                continue  
            file_path = os.path.join(directory, filename)  
            if os.path.isfile(file_path):  
                file_creation_time = os.path.getctime(file_path)  
                if current_time - file_creation_time > age_limit_seconds:  
                    os.remove(file_path)  
                    logger.info(f"Deleted old file: {file_path}")  
  
def delete_old_audio_files():  
    """  
    Delete audio files older than one day from the audio directory, except for test files.  
    """  
    audio_dir = "./audio"  
    test_dir = "./test_audio"  
    exception_files = ["test.wav"]  
  
    delete_old_files(audio_dir, exception_files)  
    delete_old_files(test_dir, exception_files)
  
# Daily task scheduling  
def schedule_daily_task(stop_event):  
    """
    Schedule a daily cleanup task that removes outdated audio and temporary files at midnight.

    :param stop_event: threading.Event  
        A signal used to stop the scheduled task gracefully.
    """
    while not stop_event.is_set():  
        if local_now.hour == 0 and local_now.minute == 0:  
            delete_old_audio_files()  
            abnormal_file_deletion(TMPPATH)
            time.sleep(60)  # Prevent triggering multiple times within the same minute  
        time.sleep(1)  
  
# Start daily task scheduling  
stop_event = Event()  
task_thread = Thread(target=schedule_daily_task, args=(stop_event,))  
task_thread.start()  

# Signal handler for graceful shutdown  
def handle_exit(sig, frame):  
    """
    Handle system exit signals (SIGINT, SIGTERM) to gracefully shut down the daily task scheduler.
    """
    stop_event.set()  
    task_thread.join()  
    abnormal_file_deletion(TMPPATH)
    logger.info("Scheduled task has been stopped.")  
    os._exit(0)  
  
signal.signal(signal.SIGINT, handle_exit)  
signal.signal(signal.SIGTERM, handle_exit)  
  
@app.on_event("shutdown")  
def shutdown_event():  
    """
    FastAPI shutdown event handler to stop background tasks and clean up temporary files.
    """
    handle_exit(None, None)  
    stop_event.set()  
    task_thread.join()  
    logger.info("Scheduled task has been stopped.")  
    
# def abnormal_file_processing():
#     abnormal_observer = Observer() 
#     event_handler = AbnormalFileDeletion()  
#     abnormal_observer.schedule(event_handler, path="/tmp", recursive=False)  
#     abnormal_observer.start() 
# abnormal_file_processing()


if __name__ == "__main__":  
    port = int(os.environ.get("PORT", 80))  
    uvicorn.run(app, log_level='info', host='0.0.0.0', port=port)  
    
    