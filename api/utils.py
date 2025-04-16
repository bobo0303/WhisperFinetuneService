
import os
import csv
import sys
import json
import shutil
import zipfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.constant import CONFIG 
from lib.log_config import setup_finetune_logging

finetune_logger = setup_finetune_logging()  

def load_csv_data(csv_file_path: str):  
    """
    Load data from a CSV file and return a dictionary mapping filenames to content.

    :param csv_file_path: str
        Path to the CSV file.
    :return: dict
        Dictionary where key is output audio filename and value is the corresponding text.
    """
    data = {}  
    with open(csv_file_path, mode='r', encoding='utf-8') as file:  
        csv_reader = csv.DictReader(file)  
        for row in csv_reader:  
            data[row['output_audio_filename']] = row['content_to_synthesize']  
    return data  

def load_file_list(file_path: str):  
    """
    Load a list of filenames from a text file (one per line).

    :param file_path: str
        Path to the file containing the list.
    :return: set
        A set of filenames loaded from the file.
    """
    if os.path.exists(file_path):  
        with open(file_path, 'r', encoding='utf-8') as file:  
            return set(file.read().splitlines())  
    return set()  

def save_file_list(file_path: str, file_list: list):  
    """
    Save a list of filenames to a file, each on a new line.

    :param file_path: str
        Path where the file will be saved.
    :param file_list: list
        The list or set of filenames to write.
    :return: None
    """
    with open(file_path, 'w', encoding='utf-8') as file:  
        for item in file_list:  
            file.write(f"{item}\n") 

def load_json_data(json_file_path: str):  
    """
    Load JSON data from a file.

    :param json_file_path: str
        Path to the JSON file.
    :return: dict
        Parsed JSON data as dictionary. Returns empty dict if file doesn't exist.
    """
    if os.path.exists(json_file_path):  
        with open(json_file_path, 'r', encoding='utf-8') as file:  
            return json.load(file)  
    return {}  

def load_jsonl_data(jsonl_file_path: str):  
    """
    Load JSONL (JSON Lines) data from a file.

    :param jsonl_file_path: str
        Path to the JSONL file.
    :return: list
        List of parsed JSON objects from each line.
    """
    data = []  
    if os.path.exists(jsonl_file_path):  
        with open(jsonl_file_path, 'r', encoding='utf-8') as file:  
            for line in file:  
                data.append(json.loads(line))  
    return data  

def zip_checkpoint(output_path: str, task_id: str): 
    """
    Compress the latest checkpoint folder into a ZIP file, and copy CONFIG into it if not present.

    :param output_path: str
        Directory where the task checkpoints are stored.
    :param task_id: str
        The ID of the task being zipped.
    :return: None
    """
    for item in os.listdir(output_path):  
            if item.endswith('.zip'):  
                os.remove(os.path.join(output_path, item))  
                finetune_logger.info(f" | Old ZIP file found. Delete file: {item} | ") 
                
    checkpoints = os.listdir(output_path)
    checkpoints.sort(key=lambda x: int(x.split("-")[-1].split(".")[0]))
    if checkpoints:
        latest_checkpoint = checkpoints[-1]
        finetune_logger.info(f" | Task {task_id}: Start zip the '{latest_checkpoint}' | ")
        checkpoint_path = os.path.join(output_path, latest_checkpoint)  
        zip_file_path = os.path.join(output_path, f"{task_id}.zip") 
        if not os.path.exists(os.path.join(checkpoint_path, CONFIG.split("/")[-1])):  
            shutil.copy(CONFIG, checkpoint_path)  
        with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:  
            for root, dirs, files in os.walk(checkpoint_path):  
                for file in files:  
                    file_path = os.path.join(root, file)  
                    arcname = os.path.relpath(file_path, checkpoint_path)  
                    zipf.write(file_path, arcname)  
        finetune_logger.info(f" | Successfully zipped '{checkpoint_path}' to '{zip_file_path}' | ")  
    else:
        finetune_logger.error(f" | No checkpoint found in {output_path} | ")