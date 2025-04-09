import os
import json
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration, TrainingArguments, Trainer
from transformers.trainer_callback import TrainerCallback  

from lib.constant import AUDIOPATH
from lib.log_config import setup_finetune_logging

finetune_logger = setup_finetune_logging()  

class WhisperFineTuner: 
    def __init__(self):  
        self.task_flags = {}
        
    def _data_collator(self, batch):
        input_features = [torch.tensor(item["input_features"], dtype=torch.float) for item in batch]
        labels = [torch.tensor(item["labels"], dtype=torch.long) for item in batch]
        input_features = pad_sequence(input_features, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        return {"input_features": input_features, "labels": labels}

    def _prepare_example(self, example, processor, audio_path):
        print(example)
        audio = os.path.join(audio_path, example["audio"])
        speech_array, sr = torchaudio.load(audio)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            speech_array = resampler(speech_array)
        
        speech_array = speech_array.squeeze().numpy()
        
        input_features = processor.feature_extractor(speech_array, sampling_rate=16000).input_features[0]
        example["input_features"] = input_features

        labels = processor.tokenizer(example["text"], max_length=448, truncation=True).input_ids
        example["labels"] = labels
        
        return example

    # https://huggingface.co/docs/transformers/v4.51.1/en/main_classes/trainer#transformers.Seq2SeqTrainingArguments.logging_steps
    def _set_training_args(self, args):
        training_args = TrainingArguments(
            output_dir=args['output_dir'],
            per_device_train_batch_size=args['batch_size'],         
            gradient_accumulation_steps=args['gradient_accumulation_steps'],         
            num_train_epochs=args['num_train_epochs'],                     
            learning_rate=args['learning_rate'],                      
            warmup_steps=args['warmup_steps'],    
            logging_dir=args['output_dir'],                   
            logging_steps=args['logging_steps'],
            save_steps=args['save_steps'],
            fp16=True,                             
            eval_strategy="no",     # ["steps", "epoch", "no"]          
            remove_unused_columns=False,
        )
        return training_args

    def finetune(self, json_file, model_name, args, task_id):
        # loading model and processor
        self.task_flags[task_id] = "model loading"
        finetune_logger.info(f" | Task {task_id}: Loading model | ")
        processor = WhisperProcessor.from_pretrained(model_name, language="Chinese", task="transcribe")
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        
        # loading dataset
        self.task_flags[task_id] = "data loading"
        finetune_logger.info(f" | Task {task_id}: Loading data from {json_file} | ")
        dataset = load_dataset("json", data_files={"train": json_file}, split="train")
        audio_path = os.path.join(AUDIOPATH, task_id)
        dataset = dataset.map(self._prepare_example, fn_kwargs={"processor": processor, "audio_path": audio_path}, num_proc=24)  

        # setting training arguments
        self.task_flags[task_id] = "setting training args"
        training_args = self._set_training_args(args)
        
        # Custom callback for logging  
        class CustomCallback(TrainerCallback):  
            def on_log(self, args, state, control, logs=None, **kwargs):  
                if logs is not None:  
                    finetune_logger.info(f" | Task {task_id} | Step {state.global_step} | {logs}")  
        
        self.task_flags[task_id] = "model fine-tuning"
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=self._data_collator,
            tokenizer=processor.tokenizer,
            callbacks=[CustomCallback] 
        )
        
        finetune_logger.info(f" | Task {task_id}: Start finetune | ")
        trainer.train()
        self.task_flags[task_id] = False
        finetune_logger.info(f" | Task {task_id}: finetune completed | ")
        
        