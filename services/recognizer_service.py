import base64
from fastapi import HTTPException, UploadFile
import os
import tempfile
from fastapi import status, Response
from constants.file_constant import DATASET_PATH
from constants.app_constant import Recognizer
from models.recognizer_model import CharacterRecognitionResult
import zipfile
import uuid


async def recognize_character_service(
    file: UploadFile, add_to_dataset: bool
) -> CharacterRecognitionResult:
    if Recognizer is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model not initialized. Please check logs.",
        )

    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="File must be an image"
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(await file.read())
        tmp_file_path = tmp_file.name

    try:
        sanitized_file_path = tmp_file_path.replace(" ", "_")
        character, confidence = Recognizer.predict_character(sanitized_file_path)
        character_image = Recognizer.get_image_for_character(character)
        with open(os.path.join(DATASET_PATH, character_image), "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
        if add_to_dataset and confidence > 0.98:
            random_name = f"{uuid.uuid4()}.jpg"
            dst_file_path = os.path.join(DATASET_PATH, random_name)
            os.rename(sanitized_file_path, dst_file_path)
        return CharacterRecognitionResult(
            character=character, confidence=float(confidence), image_base64=image_base64
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
    finally:
        os.unlink(tmp_file_path)


async def add_character_service(name: str, file: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
        tmp_file.write(await file.read())
        tmp_file_path = tmp_file.name

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with zipfile.ZipFile(tmp_file_path, "r") as zip_ref:
                zip_ref.extractall(tmp_dir)

            for root, _, files in os.walk(tmp_dir):
                for file_name in files:
                    if not file_name.lower().endswith(
                        (
                            ".png",
                            ".jpg",
                            ".jpeg",
                            ".bmp",
                        )
                    ):
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Archive contains non-image files",
                        )

            sanitized_name = name.lower()
            dataset_dir = os.path.join(DATASET_PATH, sanitized_name)
            if os.path.exists(dataset_dir):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Directory with this name already exists",
                )

            os.makedirs(dataset_dir, exist_ok=True)

            for root, _, files in os.walk(tmp_dir):
                for file_name in files:
                    src_file_path = os.path.join(root, file_name)
                    dst_file_path = os.path.join(dataset_dir, file_name)
                    os.rename(src_file_path, dst_file_path)

        return Response(
            status_code=status.HTTP_200_OK, content="Images added successfully"
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
