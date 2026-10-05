"""
src/train_macro_comparison.py
Phase 6 - Do established macro drivers of gold (Dollar Index, VIX,
Treasury yield) improve forecast accuracy beyond price history alone?

Uses real Yahoo Finance data (no synthetic values). Common overlap window
is used since VIX data only goes back to ~1990.

NOTE: ^TNX is the NOMINAL 10-Year Treasury yield (not the true real/TIPS
yield, which isn't freely available via yfinance) - documented as a proxy.
"""

import warnings
from pathlib import Path
import pandas as pd
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

from data_loader import load_daily_usd
from train_baseline import evaluate

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "outputs" / "predictions" / "macro_comparison_results.csv"

MACRO_START = "1990-01-01"
MACRO_END = "2023-07-21"  # matches the original dataset's last date
TEST_SIZE = 0.2
ARIMA_ORDER = (1, 1, 1)

TICKERS = {
    "dxy": "DX-Y.NYB",   # ICE US Dollar Index
    "vix": "^VIX",       # CBOE Volatility Index
    "yield10y": "^TNX",  # Nominal 10-Year Treasury yield (proxy for real yield)
}

LAG_DAYS = [1, 2, 3, 5, 10]
ROLLING_WINDOWS = [5, 20, 60]


def fetch_macro_data():
    """Download DXY, VIX, and 10Y yield, aligned to a single DataFrame."""
    frames = {}
    for name, ticker in TICKERS.items():
        print(f"Downloading {name} ({ticker})...")
        data = yf.download(ticker, start=MACRO_START, end=MACRO_END, progress=False)
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        frames[name] = close

    macro = pd.DataFrame(frames)
    macro.index.name = "Date"
    return macro.dropna()


def load_merged_data():
    price = load_daily_usd()["USD"]
    macro = fetch_macro_data()

    df = pd.DataFrame({"USD": price}).join(macro, how="inner").dropna()
    return df


def chrono_split(df, test_size=TEST_SIZE):
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def run_arima(train, test, exog_cols=None):
    endog_train = train["USD"]
    exog_train = train[exog_cols] if exog_cols else None
    exog_test = test[exog_cols] if exog_cols else None

    model = SARIMAX(endog_train, exog=exog_train, order=ARIMA_ORDER,
                     enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False)

    extended = fitted.append(test["USD"], exog=exog_test, refit=False)
    start = len(train)
    end = len(train) + len(test) - 1
    preds = extended.predict(start=start, end=end, exog=exog_test)
    preds.index = test.index
    return preds


def build_features(df, macro_cols):
    feat = pd.DataFrame({"USD": df["USD"]})
    for lag in LAG_DAYS:
        feat[f"lag_{lag}"] = df["USD"].shift(lag)
    for window in ROLLING_WINDOWS:
        feat[f"roll_mean_{window}"] = df["USD"].shift(1).rolling(window).mean()
    for col in macro_cols:
        feat[col] = df[col]
        feat[f"{col}_lag1"] = df[col].shift(1)
    feat["day_of_week"] = df.index.dayofweek
    return feat.dropna()


def run_xgboost(df, macro_cols):
    feat = build_features(df, macro_cols)
    feature_cols = [c for c in feat.columns if c != "USD"]

    train, test = chrono_split(feat)
    X_train, y_train = train[feature_cols], train["USD"]
    X_test, y_test = test[feature_cols], test["USD"]

    model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    preds = pd.Series(model.predict(X_test), index=y_test.index)
    return y_test, preds


if __name__ == "__main__":
    df = load_merged_data()
    print(f"\nMerged data: {len(df)} trading days, {df.index.min().date()} to {df.index.max().date()}")

    train, test = chrono_split(df)
    print(f"Train: {len(train)}  |  Test: {len(test)}\n")

    results = []

    print("--- ARIMA (price only) ---")
    preds = run_arima(train, test, exog_cols=None)
    results.append(evaluate(test["USD"], preds, "ARIMA (price only)"))

    for name in ["dxy", "vix", "yield10y"]:
        print(f"--- ARIMAX (+ {name}) ---")
        preds = run_arima(train, test, exog_cols=[name])
        results.append(evaluate(test["USD"], preds, f"ARIMAX (+ {name})"))

    print("--- ARIMAX (+ DXY + VIX + Yield, combined) ---")
    preds = run_arima(train, test, exog_cols=["dxy", "vix", "yield10y"])
    results.append(evaluate(test["USD"], preds, "ARIMAX (+ all 3 combined)"))

    print("--- XGBoost (price only) ---")
    y_true, preds = run_xgboost(df, macro_cols=[])
    results.append(evaluate(y_true, preds, "XGBoost (price only)"))

    print("--- XGBoost (+ all 3 macro vars) ---")
    y_true, preds = run_xgboost(df, macro_cols=["dxy", "vix", "yield10y"])
    results.append(evaluate(y_true, preds, "XGBoost (+ all 3 combined)"))

    results_df = pd.DataFrame(results).sort_values("rmse")
    print("\n=== Summary (sorted by RMSE) ===")
    print(results_df.to_string(index=False))

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved: {RESULTS_PATH}")
