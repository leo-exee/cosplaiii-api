import base64
from fastapi import HTTPException
import os
import tempfile
from fastapi import status, Response
from constants.file_constant import DATASET_PATH
from constants.app_constant import Recognizer
from models.recognizer_model import CharacterRecognitionResult


async def recognize_character_service(file) -> CharacterRecognitionResult:
    if Recognizer is None:
        raise HTTPException(
            status_code=500, detail="Model not initialized. Please check logs."
        )

    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(await file.read())
        tmp_file_path = tmp_file.name

    try:
        sanitized_file_path = tmp_file_path.replace(" ", "_")
        character, confidence = Recognizer.predict_character(sanitized_file_path)
        character_image = Recognizer.get_image_for_character(character)
        with open(os.path.join(DATASET_PATH, character_image), "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
        return CharacterRecognitionResult(
            character=character, confidence=float(confidence), image_base64=image_base64
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
    finally:
        os.unlink(tmp_file_path)


async def train_recognizer_service():
    try:
        Recognizer.train_model()
        return Response(
            status_code=status.HTTP_200_OK, content="Model trained successfully"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
