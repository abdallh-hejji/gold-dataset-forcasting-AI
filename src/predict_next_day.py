"""
src/predict_next_day.py
Production-style inference: fit the best model (ARIMA(1,1,1)) on all available
history and forecast the next trading day's USD gold price, with a 95%
confidence interval.
Gold Price Forecasting project
"""

import argparse
import warnings
from pathlib import Path
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from data_loader import load_daily_usd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTENDED_PATH = PROJECT_ROOT / "data" / "processed" / "usd_gold_extended.csv"
BEST_ORDER = (1, 1, 1)  # selected in Phase 3 based on AIC + RMSE comparison
MAX_GAP_DAYS = 30


def parse_args():
    parser = argparse.ArgumentParser(description="Forecast tomorrow's gold price (USD).")
    parser.add_argument("--price", type=float, default=None,
                         help="Optional: today's actual USD price, for same-day override before market close.")
    parser.add_argument("--date", type=str, default=None,
                         help="Optional: today's date (YYYY-MM-DD), required if --price is given.")
    return parser.parse_args()


def load_best_available_series():
    """
    Use the gap-filled extended series if it exists (run fetch_gap_data.py
    first to create/update it), otherwise fall back to the original dataset.
    """
    if EXTENDED_PATH.exists():
        df = pd.read_csv(EXTENDED_PATH, index_col="Date", parse_dates=True)
        print(f"Using extended series (gap-filled): {EXTENDED_PATH.name}")
        return df["USD"]
    print("Extended series not found - using original dataset only "
          "(run fetch_gap_data.py first to fill the gap with real market data).")
    return load_daily_usd()["USD"]


def append_latest_price(series, price, date):
    """Append a manually-provided latest observation (same-day override)."""
    new_date = pd.Timestamp(date)
    gap_days = (new_date - series.index.max()).days

    if gap_days > MAX_GAP_DAYS:
        warnings.warn(
            f"\n*** WARNING: {gap_days} days between the series' last date "
            f"({series.index.max().date()}) and the provided date ({new_date.date()}). "
            "Run fetch_gap_data.py first to properly fill this gap with real data "
            "before trusting this forecast. ***\n",
            stacklevel=2,
        )

    if new_date in series.index:
        series = series.drop(new_date)
    updated = pd.concat([series, pd.Series([price], index=[new_date])])
    return updated.sort_index()


def predict_next_day(series, order=BEST_ORDER, alpha=0.05):
    """
    Fit SARIMAX on the full series and forecast one step ahead.

    Returns
    -------
    next_date : pd.Timestamp
    pred_price : float
    lower : float - lower bound of the (1-alpha) confidence interval
    upper : float - upper bound of the (1-alpha) confidence interval
    """
    model = SARIMAX(series, order=order,
                     enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False)

    forecast_result = fitted.get_forecast(steps=1)
    pred_price = forecast_result.predicted_mean.iloc[0]

    conf_int = forecast_result.conf_int(alpha=alpha)
    lower = conf_int.iloc[0, 0]
    upper = conf_int.iloc[0, 1]

    next_date = series.index[-1] + pd.tseries.offsets.BDay(1)
    return next_date, pred_price, lower, upper


if __name__ == "__main__":
    args = parse_args()

    series = load_best_available_series()
    print(f"Last known date: {series.index.max().date()}  (price: {series.iloc[-1]:.2f} USD)")

    if args.price is not None:
        if args.date is None:
            raise ValueError("--date is required when --price is provided.")
        series = append_latest_price(series, args.price, args.date)
        print(f"Using manually provided latest price: {args.price:.2f} USD on {args.date}")

    next_date, pred_price, lower, upper = predict_next_day(series)

    print(f"\nForecast for next trading day ({next_date.date()}):")
    print(f"  USD: {pred_price:.2f}  (95% CI: {lower:.2f} - {upper:.2f})")
    print(f"  SAR: {pred_price * 3.75:.2f}  (95% CI: {lower * 3.75:.2f} - {upper * 3.75:.2f})")