import os
import sys
import shutil
import logging  
import threading
import ctypes
from watchdog.events import FileSystemEventHandler  

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.log_config import setup_sys_logging

logger = setup_sys_logging()  


def whisper_finetune(whisper, json_file, pretrain_model, args, task_id):  
    whisper.finetune(json_file, pretrain_model, args, task_id)

def get_thread_id(thread):  
    if not thread.is_alive():  
        return None  
    for tid, tobj in threading._active.items():  
        if tobj is thread:  
            return tid  
    logger.debug(" | Could not determine the thread ID | ")
    raise AssertionError("Could not determine the thread ID")  

def stop_thread(thread):  
    thread_id = get_thread_id(thread)  
    if thread_id is not None:
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), ctypes.py_object(SystemExit))  
        if res == 0:  
            logger.debug(" | Invalid thread ID | ")
            raise ValueError("Invalid thread ID")  
        elif res != 1:  
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)  
            logger.debug(" | PyThreadState_SetAsyncExc failed | ")
            raise SystemError("PyThreadState_SetAsyncExc failed")  

class LatestRetention(FileSystemEventHandler):  
    def on_modified(self, event):  
        logger.info(f" | File modified: {event.src_path} | ")
        output_path = os.path.dirname(event.src_path)
        save_folder = ""
        folder_list = os.listdir(output_path)
        for folder in folder_list:
            if not save_folder and folder.startswith("checkpoint-"):
                save_folder = os.path.join(output_path, folder)
            else:
                if folder.startswith("checkpoint-") and save_folder:
                    if folder.split("-")[-1] > save_folder.split("-")[-1]:
                        shutil.rmtree(save_folder)  
                        save_folder = os.path.join(output_path, folder)

        
        
        


