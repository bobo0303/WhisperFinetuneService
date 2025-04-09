from fastapi import FastAPI, UploadFile, File, Form, Query
import io
import os  
import csv
import pytz  
import time  
import tqdm
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


from api.whisper_finetune import WhisperFineTuner
from api.threading_api import whisper_finetune, stop_thread, LatestRetention

from lib.constant import MODELPATH, AUDIOPATH, TASKPATH, JSONPATH, ZIP_REQUIRED, SAFETENSORS_REQUIRED
from lib.base_object import BaseResponse  
from lib.log_config import setup_sys_logging, setup_watchdog_logging

#############################################################################  

if not os.path.exists("./logs"):  
    os.mkdir("./logs")  

logger = setup_sys_logging()  
wg_logger = setup_watchdog_logging()  

# Configure UTC+8 time  
utc_now = datetime.now(pytz.utc)  
tz = pytz.timezone('Asia/Taipei')  
local_now = utc_now.astimezone(tz)  
  
app = FastAPI()  
whisper = WhisperFineTuner()
observer = Observer() 
event_handler = LatestRetention()  
task_pair = {}

#############################################################################  

@app.get("/")  
def HelloWorld(name: str = None):  
    return {"Hello": f"World {name}"}  

#############################################################################  

@app.get("/get_audio")
async def get_audio(task_id: str, url: str = "http://172.17.0.1:52010/get_audio"):  
    out_path = os.path.join(AUDIOPATH, task_id)
    
    params = {"task_id": task_id}  
    response = requests.get(url, params=params)  
      
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
    save_flag = False
    model_folder = file.filename.replace(".zip", "")  
    # check .zip
    if not file.filename.endswith(".zip"):  
        return BaseResponse(status="FAILED", message=f" | Invalid file type. Only ZIP files are allowed. | ", data=False)
    
    # check path exist
    model_path = os.path.join(MODELPATH, model_folder)
    if os.path.exists(model_path):
        return BaseResponse(status="FAILED", message=f" | Model '{model_folder}' already exists. Please delete it first or rename and upload again. | ", data=False)
    
    logger.info(f" | Get the upload file '{file.filename}' | ")    
    logger.info(f" | Start to parse file | ")    
    file_content = await file.read()  
    
    # check zip file
    with zipfile.ZipFile(io.BytesIO(file_content)) as zip_file:  
        zip_file_contents = zip_file.namelist()  
        # Check if the zip file contains a folder structure
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
                return BaseResponse(status="FAILED", message=f" | Model '{model_folder}' is not supported. Please use a different model. (We only support tiny ~ large-v2) | ", data=False)
            logger.info(f" | model size: '{name}' | ")  
            
            logger.info(f" | Start saving model: '{model_folder}' | ")
            if save_flag:
                model_path = MODELPATH
            zip_file.extractall(model_path)
            
    logger.info(f" | Model '{model_folder}' has been saved successfully | ")

    return BaseResponse(status="OK", message=f" | Your model '{model_folder}' has been saved successfully | ", data=True)
    
@app.delete("/delete_model")
def delete_task(model_folder: str):
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
        
    if not pretrain_model.startswith("openai/whisper-"):
        pretrain_model = os.path.join(MODELPATH, pretrain_model)
        if not os.path.exists(pretrain_model):
            logger.error(f" | Model '{pretrain_model}' not found. Please check the model name is correct | ")
            return BaseResponse(status="FAILED", message=f" | Model '{pretrain_model}' not found. Please check the model name is correct | ", data=False)
        
    args = {"output_dir": output_path,
            "pretrain_model": pretrain_model,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "num_train_epochs": epochs,
            "learning_rate": learning_rate,
            "warmup_steps": warmup_steps,
            "logging_steps": logging_steps,
            "save_steps": save_steps}
    
    finetune_process = threading.Thread(target=whisper_finetune, args=(whisper, json_file_path, pretrain_model, args, task_id))
    observer.schedule(event_handler, path=output_path, recursive=True)  
    finetune_process.start()
    observer.start() 
    
    task_pair[task_id] = {"finetune_process": finetune_process, "watchdog": observer}
    
    return BaseResponse(status="OK", message=f" | Finetuning task '{task_id}' has been started. | ", data=True)  


@app.post("/cancel_task")  
def cancel_task(task_id: str = Form(...)):
    if task_pair.get(task_id) is None:
        logger.error(f" | Task ID '{task_id}' not found. | ")
        return BaseResponse(status="FAILED", message=f" | Task ID '{task_id}' not found. | ", data=False)
    
    try:    
        stop_thread(task_pair[task_id]["finetune_process"]) 
        task_pair[task_id]["observer"].stop()  
        task_pair[task_id]["observer"].join()  
    except Exception as e:
        logger.error(f" | Failed to stop thread: {e} | ")
        return BaseResponse(status="FAILED", message=f" | Failed to stop thread: {e} | ", data=False)
    
    logger.info(f" | Task ID '{task_id}' has been canceled | ")
           

@app.delete("/delete_task")  
def delete_task(task_id: str = Form(...)):
    if task_pair.get(task_id) is None:
        logger.error(f" | Task ID '{task_id}' not found. | ")
        return BaseResponse(status="FAILED", message=f" | Task ID '{task_id}' not found. | ", data=False)
    
    if not whisper.task_flags.get(task_id):
        logger.info(f" | Task ID '{task_id}' is running. Try to stop task | ")
        try:    
            stop_thread(task_pair[task_id]["finetune_process"]) 
            task_pair[task_id]["observer"].stop()  
            task_pair[task_id]["observer"].join()  
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
        
    
##############################################################################  


    

# Clean up audio files  
def delete_old_audio_files():  
    """  
    The process of deleting old audio files  
    :param  
    ----------  
    None: The function does not take any parameters  
    :rtype  
    ----------  
    None: The function does not return any value  
    :logs  
    ----------  
    Deleted old files  
    """  
    current_time = time.time()  
    audio_dir = "./audio"  
    for filename in os.listdir(audio_dir):  
        if filename == "test.wav":  # Skip specific file  
            continue  
        file_path = os.path.join(audio_dir, filename)  
        if os.path.isfile(file_path):  
            file_creation_time = os.path.getctime(file_path)  
            # Delete files older than a day  
            if current_time - file_creation_time > 24 * 60 * 60:  
                os.remove(file_path)  
                logger.info(f"Deleted old file: {file_path}")  
  
# Daily task scheduling  
def schedule_daily_task(stop_event):  
    while not stop_event.is_set():  
        if local_now.hour == 0 and local_now.minute == 0:  
            delete_old_audio_files()  
            time.sleep(60)  # Prevent triggering multiple times within the same minute  
        time.sleep(1)  
  
# Start daily task scheduling  
stop_event = Event()  
task_thread = Thread(target=schedule_daily_task, args=(stop_event,))  
task_thread.start()  
  
# Signal handler for graceful shutdown  
def handle_exit(sig, frame):  
    stop_event.set()  
    task_thread.join()  
    logger.info("Scheduled task has been stopped.")  
    os._exit(0)  
  
signal.signal(signal.SIGINT, handle_exit)  
signal.signal(signal.SIGTERM, handle_exit)  
  
@app.on_event("shutdown")  
def shutdown_event():  
    handle_exit(None, None)  

    stop_event.set()  
    task_thread.join()  
    logger.info("Scheduled task has been stopped.")  

if __name__ == "__main__":  
    port = int(os.environ.get("PORT", 80))  
    uvicorn.run(app, log_level='info', host='0.0.0.0', port=port)  