"""
src/train_arima.py
ARIMA / SARIMA baseline models - Gold Price Forecasting project
"""

from pathlib import Path
import warnings
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from data_loader import load_daily_usd
from train_baseline import train_test_split_series, evaluate

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "outputs" / "predictions" / "arima_results.csv"

# Candidate (p, d, q) orders based on ACF/PACF analysis (near white-noise returns)
ORDERS = [(0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1)]


def one_step_ahead_forecast(train, test, order):
    """
    Fit SARIMAX on train, then extend with the actual test values (refit=False)
    to get one-step-ahead in-sample predictions across the test period.
    At each test day the model "knows" all previous actual prices but
    predicts only the next one - this is the fair, non-drifting comparison
    to the Naive/MA baselines.
    """
    model = SARIMAX(train, order=order,
                     enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False)
    aic = fitted.aic

    extended = fitted.append(test, refit=False)
    start = len(train)
    end = len(train) + len(test) - 1
    preds = extended.predict(start=start, end=end)
    preds.index = test.index

    return preds, aic


if __name__ == "__main__":
    df = load_daily_usd()
    series = df["USD"]

    train, test = train_test_split_series(series)
    print(f"Train size: {len(train)}  |  Test size: {len(test)}\n")

    results = []
    for order in ORDERS:
        preds, aic = one_step_ahead_forecast(train, test, order)
        label = f"ARIMA{order}"
        row = evaluate(test, preds, label)
        row["aic"] = aic
        results.append(row)
        print(f"{'':25s} AIC (train fit): {aic:.2f}\n")

    results_df = pd.DataFrame(results)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"Saved: {RESULTS_PATH}")