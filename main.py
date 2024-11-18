import os
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field
from sklearn.preprocessing import LabelEncoder  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
import cv2
import tempfile

DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset/data/test")


class ModelConfig(BaseModel):
    dataset_path: str = Field(..., description="Path to cosplay dataset")
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    n_estimators: int = Field(default=100, gt=0)
    random_state: int = 42
    image_size: tuple = (128, 128)


class CharacterRecognitionResult(BaseModel):
    character: str
    confidence: float


class CosplayCharacterRecognizer:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.label_encoder = LabelEncoder()
        self.train_model()

    def _extract_features(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, self.config.image_size)
        return self._compute_hog_features(img_resized)

    def _compute_hog_features(self, img, cell_size=(8, 8), block_size=(2, 2)):
        hog = cv2.HOGDescriptor(
            _winSize=(img.shape[1], img.shape[0]),
            _blockSize=(block_size[1] * cell_size[1], block_size[0] * cell_size[0]),
            _blockStride=(cell_size[1], cell_size[0]),
            _cellSize=(cell_size[1], cell_size[0]),
            _nbins=9,
        )
        hog_features = hog.compute(img)
        return hog_features.flatten()

    def train_model(self):
        features, labels = [], []
        if not os.path.exists(self.config.dataset_path) or not os.listdir(
            self.config.dataset_path
        ):
            raise ValueError(
                f"Dataset path '{self.config.dataset_path}' is empty or does not exist"
            )

        for character_folder in os.listdir(self.config.dataset_path):
            character_path = os.path.join(self.config.dataset_path, character_folder)

            if os.path.isdir(character_path):
                for image_file in os.listdir(character_path):
                    image_path = os.path.join(character_path, image_file)

                    try:
                        feature_vector = self._extract_features(image_path)
                        features.append(feature_vector)
                        labels.append(character_folder)
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")

        labels_encoded = self.label_encoder.fit_transform(labels)
        X, y = np.array(features), np.array(labels_encoded)

        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state
        )

        self.model = RandomForestClassifier(
            n_estimators=self.config.n_estimators, random_state=self.config.random_state
        )
        self.model.fit(X_train, y_train)

    def predict_character(self, image_path):
        if self.model is None:
            raise ValueError("Model must be trained first")

        features = self._extract_features(image_path)
        proba = self.model.predict_proba([features])[0]
        prediction = self.model.predict([features])[0]

        character = self.label_encoder.inverse_transform([prediction])[0]
        confidence = proba[prediction]

        return character, confidence


# FastAPI Application
app = FastAPI(title="Cosplay Character Recognition")

# Global model instance
recognizer = CosplayCharacterRecognizer(ModelConfig(dataset_path=DATASET_PATH))


@app.post("/recognize", response_model=CharacterRecognitionResult)
async def recognize_character(file: UploadFile = File(...)):
    # Validate file type (optional)
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(await file.read())
        tmp_file_path = tmp_file.name

    try:
        # Predict character
        character, confidence = recognizer.predict_character(tmp_file_path)

        return CharacterRecognitionResult(
            character=character, confidence=float(confidence)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file
        os.unlink(tmp_file_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
