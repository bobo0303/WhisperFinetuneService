
import os
import csv
import json
import jiwer

def calculate_cer(reference, hypothesis):  
    return jiwer.cer(reference, hypothesis)  

def load_csv_data(csv_file_path):  
    data = {}  
    with open(csv_file_path, mode='r', encoding='utf-8') as file:  
        csv_reader = csv.DictReader(file)  
        for row in csv_reader:  
            data[row['output_audio_filename']] = row['content_to_synthesize']  
    return data  

def load_file_list(file_path):  
    if os.path.exists(file_path):  
        with open(file_path, 'r', encoding='utf-8') as file:  
            return set(file.read().splitlines())  
    return set()  

def save_file_list(file_path, file_list):  
    with open(file_path, 'w', encoding='utf-8') as file:  
        for item in file_list:  
            file.write(f"{item}\n") 

def load_json_data(json_file_path):  
    if os.path.exists(json_file_path):  
        with open(json_file_path, 'r', encoding='utf-8') as file:  
            return json.load(file)  
    return {}  

def load_jsonl_data(jsonl_file_path):  
    data = []  
    if os.path.exists(jsonl_file_path):  
        with open(jsonl_file_path, 'r', encoding='utf-8') as file:  
            for line in file:  
                data.append(json.loads(line))  
    return data  