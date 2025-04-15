import os
import time
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration, TrainingArguments, Trainer, AutoProcessor, AutoModelForSpeechSeq2Seq
from transformers.trainer_callback import TrainerCallback  

from lib.constant import AUDIOPATH, TASKPATH
from lib.log_config import setup_finetune_logging

from api.utils import zip_checkpoint

finetune_logger = setup_finetune_logging()  

class WhisperFineTuner: 
    def __init__(self):  
        """
        A class for managing fine-tuning and inference processes using Whisper models.

        Attributes:
            task_flags (dict): Task running state mapping.
            task_progress (dict): Fine-tuning progress mapping.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        self.task_flags = {}
        self.task_progress = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _data_collator(self, batch: list):
        """
        Collate input features and labels from a batch for training.

        :param batch: list  
            A list of preprocessed examples containing input features and labels.  
        :return: dict  
            A dictionary with padded input features and labels for training.
        """
        input_features = [torch.tensor(item["input_features"], dtype=torch.float) for item in batch]
        labels = [torch.tensor(item["labels"], dtype=torch.long) for item in batch]
        input_features = pad_sequence(input_features, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        return {"input_features": input_features, "labels": labels}

    def _prepare_example(self, example: dict, processor: str, audio_path: str):
        """
        Preprocess a single example by loading audio and converting text to token IDs.

        :param example: dict  
            A single data example with audio file name and text.  
        :param processor: WhisperProcessor  
            The processor used for feature extraction and tokenization.  
        :param audio_path: str  
            The directory where audio files are stored.  
        :return: dict  
            The processed example with input features and labels.
        """
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
    def _set_training_args(self, args: dict):
        """
        Configure training arguments for the Whisper model.

        :param args: dict  
            A dictionary of training parameters.  
        :return: TrainingArguments  
            A configured Hugging Face TrainingArguments object.
        """
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

    def finetune(self, json_file: str, model_name: str, args: dict, task_id: str):
        """
        Perform fine-tuning on a Whisper model using the provided dataset.

        :param json_file: str  
            Path to the JSON file containing the training data.  
        :param model_name: str  
            The name or path of the pre-trained Whisper model.  
        :param args: dict  
            Training configuration parameters.  
        :param task_id: str  
            The ID used to track and manage the task.  
        :return: None
        """
        # loading model and processor
        self.task_flags[task_id] = " model loading"
        finetune_logger.info(f" | Task {task_id}: Loading model | ")
        processor = WhisperProcessor.from_pretrained(model_name, language="zh", task="transcribe")
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        
        # loading dataset
        self.task_flags[task_id] = "data loading"
        finetune_logger.info(f" | Task {task_id}: Loading data from {json_file} | ")
        dataset = load_dataset("json", data_files={"train": json_file}, split="train")
        audio_path = os.path.join(AUDIOPATH, task_id)
        dataset = dataset.map(self._prepare_example, fn_kwargs={"processor": processor, "audio_path": audio_path}, num_proc=1)
        num_step_equal_epoch = len(dataset)/args['batch_size']/args['gradient_accumulation_steps'] 
        num_step_equal_epoch = num_step_equal_epoch if num_step_equal_epoch > 1 else 1
        
        # setting training arguments
        self.task_flags[task_id] = "setting training args"
        training_args = self._set_training_args(args)
        
        def get_progress(task_progress: float):
            self.task_progress[task_id] = task_progress
            
        # Custom callback for logging  
        class CustomCallback(TrainerCallback):  
            def on_log(self, args, state, control, logs=None, **kwargs):  
                if logs is not None:  
                    task_progress = state.global_step/num_step_equal_epoch/args.num_train_epochs * 100
                    get_progress(task_progress)
                    finetune_logger.info(f" | Task {task_id} | Step {state.global_step} | Now progress {task_progress:.2f}%| {logs}")
        
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
        finetune_logger.info(f" | Task {task_id}: finetune completed | ")
        
    def zip_model(self, task_id: str):
        """
        Compress the model checkpoints for the specified task into a zip file.

        :param task_id: str  
            The ID of the task to zip.  
        :return: None
        """
        self.task_flags[task_id] = "model zipping"
        output_path = os.path.join(TASKPATH, task_id)
        finetune_logger.info(f" | Task {task_id}: Start zip the latest checkpoint | ")
        try:
            zip_checkpoint(output_path, task_id)
            self.task_flags[task_id] = False
        except Exception as e:
            finetune_logger.error(f" | Error zipping checkpoint: {e} | ")

    def inference(self, checkpoint_path: str, audio_file_path: str): 
        """
        Perform inference using a fine-tuned model checkpoint on a given audio file.

        :param checkpoint_path: str  
            Path to the model checkpoint directory.  
        :param audio_file_path: str  
            Path to the audio file for transcription.  
        :return: tuple  
            Transcription result, model loading time, and inference time.
        """
        try: 
            finetune_logger.info(f" | Loading checkpoint from {checkpoint_path} | ")
            start = time.time()
            processor = AutoProcessor.from_pretrained(checkpoint_path)  
            model = AutoModelForSpeechSeq2Seq.from_pretrained(checkpoint_path).to(self.device)  
            model.eval()  
            end = time.time()
            load_model_time = end - start

            finetune_logger.info(f" | Loading audio from {audio_file_path} | ")
            start = time.time()
            speech_array, sr = torchaudio.load(audio_file_path)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                speech_array = resampler(speech_array)
            speech_array = speech_array.squeeze().numpy()

            input_features = processor.feature_extractor(speech_array, sampling_rate=16000).input_features
            input_features = torch.tensor(input_features).to(self.device)
            
            forced_decoder_ids = processor.get_decoder_prompt_ids(language="zh", task="transcribe")

            finetune_logger.info(f" | Start model inference | ")
            with torch.no_grad():
                predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids, num_beams=10)

            transcription = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            end = time.time()
            inference_time = end - start
        except Exception as e:
            finetune_logger.error(f" | Error during inference: {e} | ")
            transcription = e
            load_model_time = None
            inference_time = None

        return transcription, load_model_time, inference_time
    
    