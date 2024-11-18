import os
import numpy as np
import pandas as pd
import uvicorn
import cv2
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class ModelConfig(BaseModel):
    dataset_path: str = Field(..., description="Path to cosplay dataset")
    csv_path: str = Field(..., description="Path to CSV file")
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
        print(f"Extracting features for: {image_path}")
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Image {image_path} could not be read or does not exist")
        img_resized = cv2.resize(img, self.config.image_size)
        return self._compute_hog_features(img_resized)

    def _compute_hog_features(self, img, cell_size=(8, 8), block_size=(2, 2)):
        print(f"Computing HOG features for image of size {img.shape}")
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
        print("Loading dataset...")
        df = pd.read_csv(self.config.csv_path)

        features, labels = [], []

        # Get label columns (from the second column onward)
        label_columns = df.columns[1:]
        print(f"Label columns: {label_columns}")

        for _, row in df.iterrows():
            image_path = os.path.join(
                self.config.dataset_path + "/dataset/data/data/test/", row["filename"]
            )
            print(f"Processing file: {image_path}")

            if os.path.exists(image_path):
                try:
                    label_index = np.where(row[label_columns].values == 1)[0]
                    if len(label_index) > 0:
                        label = label_columns[label_index[0]]
                        print(f"Found label: {label}")

                        feature_vector = self._extract_features(image_path)
                        features.append(feature_vector)
                        labels.append(label)
                    else:
                        print(f"No label found for: {image_path}")
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
            else:
                print(f"File does not exist: {image_path}")

        # Check if we have collected features and labels
        if not features or not labels:
            raise ValueError("No data found. Please check your dataset and CSV file.")

        # Encode labels
        print("Encoding labels...")
        labels_encoded = self.label_encoder.fit_transform(labels)
        X = np.array(features)
        y = np.array(labels_encoded)

        print(f"Dataset size: {X.shape[0]} samples")
        print("Splitting dataset...")
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state
        )

        print("Training model...")
        self.model = RandomForestClassifier(
            n_estimators=self.config.n_estimators, random_state=self.config.random_state
        )
        self.model.fit(X_train, y_train)
        print("Model trained successfully.")

    def predict_character(self, image_path):
        if self.model is None:
            raise ValueError("Model must be trained first")

        print(f"Predicting character for: {image_path}")
        features = self._extract_features(image_path)
        proba = self.model.predict_proba([features])[0]
        prediction = np.argmax(proba)  # Use argmax to avoid out-of-bounds errors

        if prediction >= len(self.label_encoder.classes_):
            raise ValueError(
                f"Prediction index {prediction} is invalid. Classes: {self.label_encoder.classes_}"
            )

        character = self.label_encoder.inverse_transform([prediction])[0]
        confidence = proba[prediction]

        print(f"Prediction: {character}, Confidence: {confidence}")
        return character, confidence


# FastAPI Application
app = FastAPI(title="Cosplay Character Recognition")

# Global model instance
DATASET_PATH = os.path.dirname(__file__)
CSV_PATH = os.path.join(DATASET_PATH, "dataset/data/data/test/labels.csv")

print("Initializing model...")
try:
    recognizer = CosplayCharacterRecognizer(
        ModelConfig(dataset_path=DATASET_PATH, csv_path=CSV_PATH)
    )
    print("Model initialized.")
except Exception as e:
    print(f"Error during model initialization: {e}")
    recognizer = None


@app.get("/train")
async def train_model():
    try:
        recognizer.train_model()
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/recognize", response_model=CharacterRecognitionResult)
async def recognize_character(file: UploadFile = File(...)):
    if recognizer is None:
        raise HTTPException(
            status_code=500, detail="Model not initialized. Please check logs."
        )

    # Validate file type (optional)
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(await file.read())
        tmp_file_path = tmp_file.name

    try:
        character, confidence = recognizer.predict_character(tmp_file_path)
        return CharacterRecognitionResult(
            character=character, confidence=float(confidence)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_file_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
