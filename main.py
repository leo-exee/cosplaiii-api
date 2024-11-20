from fastapi import FastAPI
import uvicorn

from controllers.recognize_controller import recognize_controller

app = FastAPI(
    title="Cosplay Character Recognition",
    version="1.0.0",
    description="API for recognizing characters from images",
)

app.include_router(recognize_controller)
