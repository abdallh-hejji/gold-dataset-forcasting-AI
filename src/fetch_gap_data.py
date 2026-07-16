"""
src/fetch_gap_data.py
Fetch REAL historical gold prices to fill the gap between the dataset's
last date and today, using free public market data (Yahoo Finance).
Gold Price Forecasting project

IMPORTANT (methodology note for the report): this pulls real COMEX Gold
futures prices (ticker GC=F, USD per troy ounce) - the closest free proxy
to spot price. It is NOT identical to the World Gold Council spot series
used in Daily.csv (futures can differ slightly from spot due to
contango/backwardation). Document this as a methodology note if the
extended series is used for reporting.
"""

from pathlib import Path
import pandas as pd
import yfinance as yf

from data_loader import load_daily_usd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTENDED_PATH = PROJECT_ROOT / "data" / "processed" / "usd_gold_extended.csv"

GAP_TICKER = "GC=F"  # COMEX Gold futures (continuous), USD per troy ounce


def fetch_gap_prices(start_date):
    """Download real daily closing prices for the gap period from Yahoo Finance."""
    data = yf.download(GAP_TICKER, start=start_date, progress=False)

    close = data["Close"]
    # Newer yfinance versions return MultiIndex columns (ticker, field) even
    # for a single ticker - flatten to a plain Series in that case.
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    gap_series = close.rename("USD")
    gap_series.index.name = "Date"
    return gap_series


if __name__ == "__main__":
    original = load_daily_usd()["USD"]
    last_date = original.index.max()
    gap_start = last_date + pd.Timedelta(days=1)

    print(f"Original dataset ends: {last_date.date()}")
    print(f"Fetching real gold futures prices (GC=F) from {gap_start.date()} to today...\n")

    gap_prices = fetch_gap_prices(gap_start.strftime("%Y-%m-%d"))

    if gap_prices.empty:
        print("No new data returned - check internet connection or ticker availability.")
    else:
        print(f"Fetched {len(gap_prices)} new trading days, "
              f"{gap_prices.index.min().date()} to {gap_prices.index.max().date()}")

        extended = pd.concat([original, gap_prices])
        extended = extended[~extended.index.duplicated(keep="last")].sort_index()

        EXTENDED_PATH.parent.mkdir(parents=True, exist_ok=True)
        extended.to_csv(EXTENDED_PATH, header=["USD"], index_label="Date")
        print(f"\nSaved extended series: {EXTENDED_PATH}")
        print(f"Full range: {extended.index.min().date()} -> {extended.index.max().date()}")