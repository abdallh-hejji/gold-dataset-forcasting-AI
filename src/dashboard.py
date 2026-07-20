"""
src/dashboard.py
Streamlit dashboard - Gold Price Forecasting project
Each click on "Update & Predict" re-fetches the latest real gold prices
(filling any new gap via Yahoo Finance) and re-forecasts the next trading day,
with a 95% confidence interval (USD, SAR, and per-gram for both).
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_daily_usd
from fetch_gap_data import fetch_gap_prices, EXTENDED_PATH
from predict_next_day import predict_next_day, BEST_ORDER

OUNCE_TO_GRAM = 31.1034768
USD_TO_SAR = 3.75
HISTORICAL_TEST_MAPE = 0.618  # ARIMA(1,1,1), measured on 2,326-day test set (Phase 3)

st.set_page_config(page_title="Gold Price Forecast (USD)", layout="wide")
st.title("💰 Gold Price Forecasting Dashboard")
st.caption("ARIMA(1,1,1) — trained on 1978-2023 history (World Gold Council) "
           "+ live gap-fill (Yahoo Finance GC=F)")


def load_extended_series():
    """Load the gap-filled series if it exists, else fall back to the original dataset."""
    if EXTENDED_PATH.exists():
        df = pd.read_csv(EXTENDED_PATH, index_col="Date", parse_dates=True)
        return df["USD"]
    return load_daily_usd()["USD"]


def update_and_refetch():
    """Fetch the latest real prices for any gap since the series' last date, and save."""
    series = load_extended_series()
    last_date = series.index.max()
    gap_start = last_date + pd.Timedelta(days=1)

    gap_prices = fetch_gap_prices(gap_start.strftime("%Y-%m-%d"))

    if not gap_prices.empty:
        series = pd.concat([series, gap_prices])
        series = series[~series.index.duplicated(keep="last")].sort_index()
        series.to_csv(EXTENDED_PATH, header=["USD"], index_label="Date")

    return series


def format_ci(lower, upper, symbol="$"):
    """Render a confidence interval with Low in red and High in green
    using Streamlit's markdown color syntax."""
    return (f"95% CI — :red[Low: {symbol}{lower:,.2f}]   "
            f":green[High: {symbol}{upper:,.2f}]")


# ----- Session state: keep the series and forecast across reruns -----
if "series" not in st.session_state:
    st.session_state.series = load_extended_series()
    st.session_state.forecast = None  # will hold (next_date, pred_price, lower, upper)

col1, col2 = st.columns([1, 3])

with col1:
    if st.button("🔄 Update & Predict Tomorrow", type="primary"):
        with st.spinner("Fetching latest prices and refitting model..."):
            st.session_state.series = update_and_refetch()
            next_date, pred_price, lower, upper = predict_next_day(
                st.session_state.series, order=BEST_ORDER
            )
            st.session_state.forecast = (next_date, pred_price, lower, upper)

series = st.session_state.series
last_date = series.index.max()
last_price = series.iloc[-1]

# Compare today's last known price to the previous available day, for a
# color-coded delta (green = up, red = down) shown next to the metric.
prev_price = series.iloc[-2] if len(series) > 1 else None
price_delta = (last_price - prev_price) if prev_price is not None else None

with col1:
    st.metric("Last Known Date", str(last_date.date()))
    st.metric(
        "Last Known Price (USD)",
        f"${last_price:,.2f}",
        delta=f"{price_delta:,.2f}" if price_delta is not None else None,
    )

    if st.session_state.forecast is not None:
        next_date, pred_price, lower, upper = st.session_state.forecast

        pred_price_gram = pred_price / OUNCE_TO_GRAM
        lower_gram = lower / OUNCE_TO_GRAM
        upper_gram = upper / OUNCE_TO_GRAM

        pred_sar = pred_price * USD_TO_SAR
        lower_sar = lower * USD_TO_SAR
        upper_sar = upper * USD_TO_SAR

        pred_sar_gram = pred_sar / OUNCE_TO_GRAM
        lower_sar_gram = lower_sar / OUNCE_TO_GRAM
        upper_sar_gram = upper_sar / OUNCE_TO_GRAM

        st.metric(f"Forecast ({next_date.date()}) USD", f"${pred_price:,.2f}")
        st.markdown(format_ci(lower, upper, "$"))

        st.metric(f"Forecast ({next_date.date()}) USD/gram", f"${pred_price_gram:,.2f}")
        st.markdown(format_ci(lower_gram, upper_gram, "$"))

        st.metric(f"Forecast ({next_date.date()}) SAR", f"﷼{pred_sar:,.2f}")
        st.markdown(format_ci(lower_sar, upper_sar, "﷼"))

        st.metric(f"Forecast ({next_date.date()}) SAR/gram", f"﷼{pred_sar_gram:,.2f}")
        st.markdown(format_ci(lower_sar_gram, upper_sar_gram, "﷼"))

        st.caption(f"📊 Historical test MAPE: {HISTORICAL_TEST_MAPE}% "
                   f"(evaluated on 2,326 trading days, 2014-2023)")
    else:
        st.info("Click 'Update & Predict Tomorrow' to generate a forecast.")

with col2:
    recent = series.loc[series.index >= last_date - pd.Timedelta(days=180)]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(recent.index, recent.values, label="Actual (last 180 days)")

    if st.session_state.forecast is not None:
        next_date, pred_price, lower, upper = st.session_state.forecast
        ax.errorbar([next_date], [pred_price],
                    yerr=[[pred_price - lower], [upper - pred_price]],
                    fmt="o", color="red", capsize=5, label="Forecast (95% CI)")

    ax.set_xlabel("Date")
    ax.set_ylabel("USD")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

st.caption("Model refits on-demand on the full gap-filled history each time you click Update.")