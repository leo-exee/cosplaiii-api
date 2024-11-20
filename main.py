from fastapi import FastAPI
import uvicorn

from controllers.recognize_controller import recognize_controller

app = FastAPI(title="Cosplay Character Recognition")

app.include_router(recognize_controller)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
