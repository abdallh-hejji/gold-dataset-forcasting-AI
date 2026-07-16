"""
src/train_prophet.py
Prophet forecasting model - Gold Price Forecasting project
"""

from pathlib import Path
import pandas as pd
from prophet import Prophet

from data_loader import load_daily_usd
from train_baseline import train_test_split_series, evaluate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "outputs" / "predictions" / "prophet_results.csv"

TEST_SIZE = 0.2


if __name__ == "__main__":
    df = load_daily_usd()
    series = df["USD"]

    train, test = train_test_split_series(series, test_size=TEST_SIZE)
    print(f"Train size: {len(train)}  |  Test size: {len(test)}\n")

    # ----- Prophet requires columns named exactly 'ds' and 'y' -----
    train_prophet = pd.DataFrame({
        "ds": train.index,
        "y": train.values,
    })

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
    )
    model.fit(train_prophet)

    # ----- Forecast exactly over the test period dates -----
    future = pd.DataFrame({"ds": test.index})
    forecast = model.predict(future)

    preds = pd.Series(forecast["yhat"].values, index=test.index)

    evaluate(test, preds, "Prophet")

    # ----- Save predictions -----
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "date": test.index, "actual": test.values, "predicted": preds.values
    }).to_csv(RESULTS_PATH, index=False)
    print(f"Saved: {RESULTS_PATH}")