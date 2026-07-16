"""
src/convert_to_sar.py
Add SAR-equivalent columns to saved USD prediction files
Gold Price Forecasting project
"""

from pathlib import Path
import pandas as pd

from utils import usd_to_sar

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = PROJECT_ROOT / "outputs" / "predictions"

# Prediction files that contain per-date (date, actual, predicted) columns
FILES_WITH_DATES = [
    "lstm_results.csv",
    "lstm_returns_results.csv",
]

if __name__ == "__main__":
    for filename in FILES_WITH_DATES:
        path = PRED_DIR / filename
        if not path.exists():
            print(f"Skipped (not found): {filename}")
            continue

        df = pd.read_csv(path)
        df["actual_sar"] = usd_to_sar(df["actual"], dates=df["date"])
        df["predicted_sar"] = usd_to_sar(df["predicted"], dates=df["date"])
        df.to_csv(path, index=False)

        print(f"Updated: {filename}")
        print(df[["date", "actual", "predicted", "actual_sar", "predicted_sar"]].head(3))
        print()