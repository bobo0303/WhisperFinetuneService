import os
import sys
import time
import shutil
import threading
import ctypes
from watchdog.events import FileSystemEventHandler  

from lib.constant import RETRAYTIME  


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.log_config import setup_sys_logging, setup_watchdog_logging

logger = setup_sys_logging()  
wg_logger = setup_watchdog_logging()  


def whisper_finetune(whisper, json_file: str, pretrain_model: str, args: dict, task_id: str):  
    """
    Run the fine-tuning process and zip the model after completion.

    :param whisper: WhisperFineTuner
        The whisper fine-tuner instance.
    :param json_file: str
        Path to the training dataset JSON file.
    :param pretrain_model: str
        Path or name of the pre-trained model.
    :param args: dict
        Fine-tuning argument settings.
    :param task_id: str
        ID of the task.
    :return: None
    """
    whisper.finetune(json_file, pretrain_model, args, task_id)
    whisper.zip_model(task_id)

def get_thread_id(thread: threading.Thread):  
    """
    Get the internal thread ID used for low-level thread operations.

    :param thread: threading.Thread
        A running thread object.
    :return: int or None
        The thread ID if found, otherwise None.
    """
    if not thread.is_alive():  
        return None  
    for tid, tobj in threading._active.items():  
        if tobj is thread:  
            return tid  
    logger.debug(" | Could not determine the thread ID | ")
    raise AssertionError("Could not determine the thread ID")  

def stop_thread(thread: threading.Thread, task_id: str):  
    """
    Attempt to stop a thread forcefully by raising SystemExit in the thread.

    :param thread: threading.Thread
        The thread to be terminated.
    :param task_id: str
        The ID of the task the thread belongs to.
    :return: bool
        True if the thread was stopped, False if retry limit was reached.
    """
    retry_count = 0
    while retry_count < RETRAYTIME:  
        thread_id = get_thread_id(thread)  
        if thread_id is None: 
            return True
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), ctypes.py_object(SystemExit))  
        if res == 0:  
            logger.debug(" | Invalid thread ID | ")  
            raise ValueError("Invalid thread ID")  
        elif res != 1:  
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)  
            logger.debug(" | PyThreadState_SetAsyncExc failed | ")  
            raise SystemError("PyThreadState_SetAsyncExc failed")  

        logger.info(f" | Waiting for task {task_id} to stop... | ")
        time.sleep(5)  
        retry_count += 1  
        
    return False
    
class LatestRetention(FileSystemEventHandler):  
    """
    A watchdog event handler that retains only the latest model checkpoint file.
    """
    def on_created(self, event: FileSystemEvent):  
        """
        Called when a new file is created. Cleans up older checkpoints, retains the latest.

        :param event: FileSystemEvent
            The file system event object containing event information.
        :return: None
        """
        output_path = os.path.dirname(event.src_path)
        if os.path.exists(output_path):
            wg_logger.info(f" | File modified: {output_path} | ")
            folder_list = [f for f in os.listdir(output_path) if not f.endswith('.zip')]  
            folder_list.sort(key=lambda x: int(x.split("-")[-1].split(".")[0]))
            latest_checkpoint = folder_list[-1]
            old_checkpoints = folder_list[:-1]
            wg_logger.info(f" | We found {folder_list} in '{output_path}' | ")
            for folder in old_checkpoints:
                    if folder.startswith("checkpoint-"):
                        wg_logger.info(f" | We remove the old file '{folder}' | ")
                        shutil.rmtree(os.path.join(output_path, folder))  
            if latest_checkpoint:
                wg_logger.info(f" | The latest '{latest_checkpoint}' Stayed in '{output_path}' | ")
            
# class AbnormalFileDeletion(FileSystemEventHandler):  
#     def on_modified(self, event):  
#         tmp_path = event.src_path
#         if os.path.exists(tmp_path):
#             folder_list = os.listdir(tmp_path)
#             for folder in folder_list:
#                 if folder.startswith("tmp"):
#                     folder_path = os.path.join(tmp_path, folder)
#                     shutil.rmtree(folder_path)  
#                     wg_logger.info(f" | The abnormal file '{folder}' has been deleted | ")

def abnormal_file_deletion(tmp_path):
    """
    Delete temporary or abnormal files inside the specified directory.

    :param tmp_path: str
        Path to the directory to be cleaned.
    :return: None
    """
    if os.path.exists(tmp_path):
        folder_list = os.listdir(tmp_path)
        for folder in folder_list:
            if folder.startswith("tmp") or folder.startswith("pymp"):
                folder_path = os.path.join(tmp_path, folder)
                shutil.rmtree(folder_path)  
                wg_logger.info(f" | The abnormal file '{folder}' has been deleted | ")

