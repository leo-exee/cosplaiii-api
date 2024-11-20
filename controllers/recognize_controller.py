from fastapi import APIRouter, File, UploadFile

from models.recognizer_model import CharacterRecognitionResult
from services.recognizer_service import (
    recognize_character_service,
    train_recognizer_service,
)

recognize_controller = APIRouter(
    prefix="/recognize",
    tags=["recognize"],
)


@recognize_controller.post(
    "",
    summary="Recognize a character",
    description="Recognize a character from an uploaded image.",
    response_model=CharacterRecognitionResult,
)
async def recognize_character_controller(file: UploadFile = File(...)):
    return await recognize_character_service(file)


@recognize_controller.get(
    "/train",
    summary="Train the model",
    description="Train the model using the dataset.",
    responses={200: {"description": "Model trained successfully"}},
)
async def train_model_controller():
    return await train_recognizer_service()
