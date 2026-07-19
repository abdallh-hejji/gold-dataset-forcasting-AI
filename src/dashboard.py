"""
src/dashboard.py
Streamlit dashboard - Gold Price Forecasting project
Each click on "Update & Predict" re-fetches the latest real gold prices
(filling any new gap via Yahoo Finance) and re-forecasts the next trading day.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_daily_usd
from fetch_gap_data import fetch_gap_prices, EXTENDED_PATH
from predict_next_day import predict_next_day, BEST_ORDER

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


# ----- Session state: keep the series and forecast across reruns -----
if "series" not in st.session_state:
    st.session_state.series = load_extended_series()
    st.session_state.forecast = None

col1, col2 = st.columns([1, 3])

with col1:
    if st.button("🔄 Update & Predict Tomorrow", type="primary"):
        with st.spinner("Fetching latest prices and refitting model..."):
            st.session_state.series = update_and_refetch()
            next_date, pred_price = predict_next_day(st.session_state.series, order=BEST_ORDER)
            st.session_state.forecast = (next_date, pred_price)

series = st.session_state.series
last_date = series.index.max()
last_price = series.iloc[-1]

with col1:
    st.metric("Last Known Date", str(last_date.date()))
    st.metric("Last Known Price (USD)", f"${last_price:,.2f}")

    if st.session_state.forecast is not None:
        next_date, pred_price = st.session_state.forecast
        st.metric(f"Forecast ({next_date.date()}) USD", f"${pred_price:,.2f}")
        st.metric(f"Forecast ({next_date.date()}) SAR", f"﷼{pred_price * 3.75:,.2f}")
    else:
        st.info("Click 'Update & Predict Tomorrow' to generate a forecast.")

with col2:
    recent = series.loc[series.index >= last_date - pd.Timedelta(days=180)]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(recent.index, recent.values, label="Actual (last 180 days)")
    if st.session_state.forecast is not None:
        next_date, pred_price = st.session_state.forecast
        ax.scatter([next_date], [pred_price], color="red", zorder=5, label="Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("USD")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

st.caption("Model refits on-demand on the full gap-filled history each time you click Update.")