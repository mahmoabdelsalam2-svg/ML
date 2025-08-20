# automl_api/routers/image_router.py

import uuid
import shutil
import zipfile
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2  # Import pre-trained model
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np

from automl_api import settings
from automl_api.logger import loguru_logger as logger

router = APIRouter()
model_registry = {}


class TrainRequest(BaseModel):
    dataset_id: str


def train_model_task(model_id: str, dataset_id: str, num_classes: int):
    try:
        dataset_path = settings.STORAGE_PATH / dataset_id
        model_save_path = settings.TRAINED_MODELS_PATH / f"{model_id}.keras"

        model_registry[model_id] = {"status": "training", "progress": 0, "model_path": str(model_save_path)}

        # 1. Setup ImageDataGenerator for data augmentation and preprocessing
        # MobileNetV2 expects pixel values in the range [-1, 1]
        datagen = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2  # Use 20% of data for validation
        )

        img_height, img_width = 224, 224
        batch_size = 32

        train_generator = datagen.flow_from_directory(
            dataset_path,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )

        validation_generator = datagen.flow_from_directory(
            dataset_path,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )

        # 2. --- Build the model using Transfer Learning ---
        # Load the MobileNetV2 model, pre-trained on ImageNet, without the top classification layer
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

        # Freeze the layers of the base model so we don't retrain them
        base_model.trainable = False

        # Create a new model on top
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),  # Pools the features from the base model
            Dense(num_classes, activation='softmax')  # Our custom classification layer
        ])

        # 3. Compile the model
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # 4. Train the model
        # We only train for a few epochs because the base model is already very powerful
        model.fit(
            train_generator,
            epochs=5,
            validation_data=validation_generator
        )

        # Save the trained model
        model.save(model_save_path)

        # Save class indices for later prediction
        class_indices = train_generator.class_indices
        model_registry[model_id]["class_indices"] = class_indices

        # Update status to ready
        model_registry[model_id]["status"] = "ready"
        logger.info(f"Image model {model_id} trained and saved successfully.")

    except Exception as e:
        logger.error(f"Model training failed for {model_id}: {e}", exc_info=True)
        model_registry[model_id]["status"] = "failed"
        model_registry[model_id]["error"] = str(e)


@router.post("/upload-dataset")
async def upload_image_dataset(file: UploadFile = File(...)):
    dataset_id = str(uuid.uuid4())
    dataset_path = settings.STORAGE_PATH / dataset_id
    zip_path = settings.STORAGE_PATH / f"{dataset_id}.zip"

    try:
        dataset_path.mkdir(parents=True, exist_ok=True)

        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_path)

        zip_path.unlink()  # Clean up the zip file

        # Check number of classes (subdirectories)
        subdirs = [p for p in dataset_path.iterdir() if p.is_dir()]
        num_classes = len(subdirs)
        if num_classes < 2:
            shutil.rmtree(dataset_path)
            raise HTTPException(status_code=400,
                                detail="The uploaded ZIP file must contain at least two subdirectories, one for each class.")

        return {"dataset_id": dataset_id, "num_classes": num_classes,
                "message": "Dataset uploaded and extracted successfully."}
    except Exception as e:
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
        logger.error(f"Dataset upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process dataset: {e}")


@router.post("/train-model")
async def train_model_endpoint(request: TrainRequest, background_tasks: BackgroundTasks):
    dataset_path = settings.STORAGE_PATH / request.dataset_id
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found.")

    subdirs = [p for p in dataset_path.iterdir() if p.is_dir()]
    num_classes = len(subdirs)

    model_id = str(uuid.uuid4())
    model_registry[model_id] = {"status": "pending"}

    background_tasks.add_task(train_model_task, model_id, request.dataset_id, num_classes)

    return {"model_id": model_id, "message": "Model training started in the background."}


@router.get("/model-status/{model_id}")
async def get_model_status(model_id: str):
    if model_id not in model_registry:
        raise HTTPException(status_code=404, detail="Model ID not found.")
    return model_registry[model_id]


@router.post("/predict-image/{model_id}")
async def predict_image(model_id: str, file: UploadFile = File(...)):
    if model_id not in model_registry or model_registry[model_id]["status"] != "ready":
        raise HTTPException(status_code=404, detail="Model not ready or not found.")

    try:
        model_path = model_registry[model_id]["model_path"]
        model = load_model(model_path)

        # Load and preprocess the image
        img_bytes = await file.read()
        temp_file_path = settings.STORAGE_PATH / f"temp_{file.filename}"
        with open(temp_file_path, "wb") as f:
            f.write(img_bytes)

        img = keras_image.load_img(temp_file_path, target_size=(224, 224))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

        temp_file_path.unlink()  # Clean up the temporary image file

        # Make prediction
        prediction = model.predict(img_array)

        # Decode prediction
        class_indices = model_registry[model_id]["class_indices"]
        # Invert the dictionary to map from index to class name
        class_labels = {v: k for k, v in class_indices.items()}

        predicted_class_index = np.argmax(prediction[0])
        predicted_class_name = class_labels[predicted_class_index]
        confidence = float(np.max(prediction[0]))

        return {"predicted_class": predicted_class_name, "confidence": confidence}

    except Exception as e:
        logger.error(f"Prediction failed for model {model_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")