# automl_api/settings.py

from pathlib import Path

PROJECT_NAME = "AutoML Backend API"
VERSION = "0.2.0" # Version updated
DEBUG = True

# Extensions relevant to the image classification workflow
ALLOWED_EXTENSIONS = {".zip", ".jpg", ".jpeg", ".png"}

MAX_DATASET_SIZE_MB = 1000
STORAGE_PATH = Path("uploaded_datasets")
MODELS_PATH = Path("trained_models")

# Ensure directories exist
STORAGE_PATH.mkdir(exist_ok=True)
MODELS_PATH.mkdir(exist_ok=True)