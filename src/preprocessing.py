"""
src/preprocessing.py
Cleaning & transformation functions - Gold Price Forecasting project
"""

from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "usd_gold_log_returns.csv"


def add_log_returns(df, price_col="USD"):
    """
    Add log_price and log_return columns to a Date-indexed price DataFrame.
    Drops the first row (NaN from the diff operation).
    """
    df = df.copy()
    df["log_price"] = np.log(df[price_col])
    df["log_return"] = df["log_price"].diff()
    df = df.dropna(subset=["log_return"])
    return df


def save_processed(df, path=DEFAULT_PROCESSED_PATH):
    df.to_csv(path)
    print(f"Saved: {path}")