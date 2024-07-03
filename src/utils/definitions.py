import os.path
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent

SAVED_MODELS_DIR = os.path.join(ROOT_DIR, "saved_models")
