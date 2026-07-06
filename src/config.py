from pathlib import Path

# Project Root
BASE_DIR = Path(__file__).resolve().parent.parent

# Data
RAW_DATA = BASE_DIR / "data" / "raw" / "Daily.csv"
PROCESSED_DATA = BASE_DIR / "data" / "processed"

# Outputs
FIGURES = BASE_DIR / "outputs" / "figures"
MODELS = BASE_DIR / "outputs" / "models"
PREDICTIONS = BASE_DIR / "outputs" / "predictions"

# Create folders automatically
for folder in [PROCESSED_DATA, FIGURES, MODELS, PREDICTIONS]:
    folder.mkdir(parents=True, exist_ok=True)