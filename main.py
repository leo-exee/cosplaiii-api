import base64
import logging
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import tempfile
import uvicorn


# Configuration
class ModelConfig(BaseModel):
    dataset_path: str
    image_size: tuple = (224, 224)  # ResNet50 requires 224x224 images
    batch_size: int = 16
    epochs: int = 20
    learning_rate: float = 0.0001
    test_split: float = 0.2  # Percentage of data for validation


class CharacterRecognitionResult(BaseModel):
    character: str
    confidence: float
    image_url: str


class CosplayCharacterRecognizer:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.label_map: dict = {}
        self.best_model_path = os.path.join(
            self.config.dataset_path, "cosplay_model.keras"
        )
        self.label_map_path = os.path.join(self.config.dataset_path, "label_map.json")
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the model: load the best model or prepare for training."""
        if os.path.exists(self.best_model_path) and os.path.exists(self.label_map_path):
            logging.info("Loading existing model and label map...")
            self.model = load_model(self.best_model_path)
            self._load_label_map()
        else:
            logging.info("No existing model found. Please train the model.")

    def _load_label_map(self):
        """Load the label map from a JSON file."""
        with open(self.label_map_path, "r") as f:
            self.label_map = json.load(f)

    def _save_label_map(self):
        """Save the label map to a JSON file."""
        with open(self.label_map_path, "w") as f:
            json.dump(self.label_map, f)

    def _create_model(self, num_classes):
        """Create a ResNet50-based model for classification."""
        base_model = ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=self.config.image_size + (3,),
        )
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(512, activation="relu")(x)
        x = Dropout(0.5)(x)
        output = Dense(num_classes, activation="softmax")(x)
        model = Model(inputs=base_model.input, outputs=output)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def train_model(self):
        """Train the model using images from the dataset."""
        # Data Augmentation and Data Preparation
        datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            validation_split=self.config.test_split,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
        )

        train_gen = datagen.flow_from_directory(
            self.config.dataset_path,
            target_size=self.config.image_size,
            batch_size=self.config.batch_size,
            subset="training",
            class_mode="categorical",
        )
        val_gen = datagen.flow_from_directory(
            self.config.dataset_path,
            target_size=self.config.image_size,
            batch_size=self.config.batch_size,
            subset="validation",
            class_mode="categorical",
        )

        # Save the label map
        self.label_map = train_gen.class_indices
        self._save_label_map()

        # Create the model
        num_classes = len(train_gen.class_indices)
        self.model = self._create_model(num_classes)

        # Callbacks
        early_stopping = EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True
        )
        model_checkpoint = ModelCheckpoint(
            self.best_model_path, monitor="val_accuracy", save_best_only=True
        )

        logging.info("Starting training...")
        self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=self.config.epochs,
            callbacks=[early_stopping, model_checkpoint],
        )
        logging.info("Training complete. Best model saved.")

    def predict_character(self, image_path):
        """Predict the character from an image."""
        if self.model is None:
            raise ValueError("Model must be trained first")

        # Load and preprocess the image
        img = load_img(image_path, target_size=self.config.image_size)
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make a prediction
        predictions = self.model.predict(img_array)
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
        character = [
            name for name, idx in self.label_map.items() if idx == predicted_idx
        ][0]

        return character, confidence
 
    def get_image_for_character(self, character):
        """Retourne le chemin relatif de l'image associée à un caractère donné."""
        character_folder = os.path.join(self.config.dataset_path, character)
        if not os.path.exists(character_folder):
            raise ValueError(f"No folder found for character: {character}")

        # Récupère une image aléatoire ou la première dans le dossier
        images = [
            f
            for f in os.listdir(character_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if not images:
            raise ValueError(f"No images found for character: {character}")

        # Retourne le chemin relatif depuis le dossier dataset
        return os.path.join(character, images[0])


# FastAPI Application
app = FastAPI(title="Cosplay Character Recognition")

DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset")

try:
    recognizer = CosplayCharacterRecognizer(ModelConfig(dataset_path=DATASET_PATH))
    logging.info("Model initialized.")
except Exception as e:
    logging.error("Error initializing model: %s", str(e))
    raise e


@app.get(
    "/train",
    summary="Train the model",
    description="Train the model using the dataset.",
)
async def train_model():
    """Endpoint to train the model."""
    try:
        recognizer.train_model()
        return {"status": "success", "message": "Model trained successfully."}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post(
    "/recognize",
    response_model=CharacterRecognitionResult,
    summary="Recognize a character",
    description="Recognize a character from an uploaded image.",
)
async def recognize_character(file: UploadFile = File(...)):
    """Endpoint to recognize a character from an uploaded image."""
    if recognizer is None:
        raise HTTPException(
            status_code=500, detail="Model not initialized. Please check logs."
        )

    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(await file.read())
        tmp_file_path = tmp_file.name

    try:
        # Sanitize the file name to handle white spaces and non-encoded characters
        sanitized_file_path = tmp_file_path.replace(" ", "_")

        character, confidence = recognizer.predict_character(sanitized_file_path)
        character_image = recognizer.get_image_for_character(character)
        # Read the image and convert it to base64
        with open(os.path.join(DATASET_PATH, character_image), "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

        # Construct and return the result
        return CharacterRecognitionResult(
            character=character, confidence=float(confidence), image_url=image_base64
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_file_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)