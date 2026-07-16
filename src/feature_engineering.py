"""
src/feature_engineering.py
Lag & rolling-window feature engineering - Gold Price Forecasting project
"""

import pandas as pd

LAG_DAYS = [1, 2, 3, 5, 10]
ROLLING_WINDOWS = [5, 20, 60]


def build_features(series, price_col="USD"):
    """
    Build a supervised-learning feature table from a Date-indexed price series.
    Each row's features use only past values (no leakage) to predict that
    row's price - reframes the time series as a regression problem.
    """
    df = pd.DataFrame({price_col: series})

    # ----- Lag features -----
    for lag in LAG_DAYS:
        df[f"lag_{lag}"] = df[price_col].shift(lag)

    # ----- Rolling mean & std (shifted by 1 to avoid leakage) -----
    for window in ROLLING_WINDOWS:
        df[f"roll_mean_{window}"] = df[price_col].shift(1).rolling(window).mean()
        df[f"roll_std_{window}"] = df[price_col].shift(1).rolling(window).std()

    # ----- Calendar features -----
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month

    df = df.dropna()
    return df


def get_feature_columns(df, price_col="USD"):
    """Return the list of feature column names (everything except the target)."""
    return [c for c in df.columns if c != price_col]