import logging
import os
import json
import numpy as np
import tensorflow as tf  # type: ignore
from tensorflow.keras.models import Model, load_model  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)
from tensorflow.keras.applications import ResNet50  # type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input  # type: ignore
from tensorflow.keras.preprocessing.image import (  # type: ignore
    ImageDataGenerator,
    load_img,
    img_to_array,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore
from pydantic import BaseModel


class Character(BaseModel):
    name: str
    image_base64: str


class ModelConfig(BaseModel):
    dataset_path: str
    image_size: tuple = (224, 224)
    batch_size: int = 16
    epochs: int = 20
    learning_rate: float = 0.0001
    test_split: float = 0.2


class CharacterRecognitionResult(BaseModel):
    character: str
    confidence: float
    image_base64: str


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

        self.label_map = train_gen.class_indices
        self._save_label_map()

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

        img = load_img(image_path, target_size=self.config.image_size)
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

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

        images = [
            f
            for f in os.listdir(character_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if not images:
            raise ValueError(f"No images found for character: {character}")

        return os.path.join(character, images[0])
