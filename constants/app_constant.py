import logging
from constants.file_constant import DATASET_PATH
from models.recognizer_model import CosplayCharacterRecognizer, ModelConfig

Recognizer = None
try:
    Recognizer = CosplayCharacterRecognizer(ModelConfig(dataset_path=DATASET_PATH))
    logging.info("Model initialized.")
except Exception as e:
    logging.error("Error initializing model: %s", str(e))
    raise e
