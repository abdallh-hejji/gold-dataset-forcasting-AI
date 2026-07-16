"""
src/data_loader.py
Raw data loading functions - Gold Price Forecasting project
"""

from pathlib import Path
import pandas as pd

# Project root = parent of the src/ folder, regardless of where the
# script is executed from (PyCharm Run button, terminal, etc.)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "Daily.csv"


def load_daily_usd(raw_path=DEFAULT_RAW_PATH):
    """
    Load the raw Daily.csv file and return a Date-indexed
    DataFrame containing only the cleaned USD column.
    """
    df = pd.read_csv(raw_path, usecols=["Date", "USD"])
    df["Date"] = pd.to_datetime(df["Date"])

    # Some rows use thousand-separator commas once gold crossed $1,000/oz
    # in 2008 (e.g. "1,003.5"). Strip commas before converting to float.
    df["USD"] = df["USD"].astype(str).str.replace(",", "", regex=False).astype(float)

    df = df.sort_values("Date").set_index("Date")
    return df


if __name__ == "__main__":
    df = load_daily_usd()
    print("Shape:", df.shape)
    print("Date range:", df.index.min().date(), "to", df.index.max().date())
    print("Missing values:", df["USD"].isna().sum())