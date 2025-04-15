#############################################################################
""" path constants """
TMPPATH = "/tmp"
MODELPATH = "/tmp/models"
TASKPATH = "/tmp/tasks"
AUDIOPATH = "/tmp/audio"
JSONPATH = "/tmp/json"
TESTPATH = "/tmp/test"

#############################################################################
""" model include below """
ZIP_REQUIRED = [  
    "added_tokens.json",  
    "config.json",  
    "generation_config.json",  
    "merges.txt",  
    "normalizer.json",  
    "special_tokens_map.json",  
    "tokenizer_config.json",  
    "trainer_state.json",  # maybe not needed
    "training_args.bin",    # maybe not needed
    "vocab.json",
    "preprocessor_config.json", # We give this file when they download
]

SAFETENSORS_REQUIRED = "model.safetensors.index.json"

CONFIG = "lib/preprocessor_config.json"

# url = "http://172.17.0.1:52010/get_audio"  # bv_service

#############################################################################

# shut down retry times
RETRAYTIME = 5