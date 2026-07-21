"""
src/dashboard.py
Streamlit dashboard - Gold Price Forecasting project
Each click on "Update & Predict" re-fetches the latest real gold prices
(filling any new gap via Yahoo Finance) and re-forecasts the next trading day,
with a 95% confidence interval. Currency display (USD/SAR) is togglable.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_daily_usd
from fetch_gap_data import fetch_gap_prices, EXTENDED_PATH
from predict_next_day import predict_next_day, BEST_ORDER

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "predictions" / "all_models_summary.csv"
FIG_PATH = PROJECT_ROOT / "outputs" / "figures" / "model_comparison.png"
GITHUB_REPO_URL = "https://github.com/abdallh-hejji/gold-dataset-forcasting-AI"

OUNCE_TO_GRAM = 31.1034768
USD_TO_SAR = 3.75
HISTORICAL_TEST_MAPE = 0.618  # ARIMA(1,1,1), measured on 2,326-day test set (Phase 3)
SELECTED_MODEL_LABEL = "ARIMA(1, 1, 1)"  # must match the label used in all_models_summary.csv

# Emoji written as Unicode escapes instead of literal characters - avoids
# encoding/display issues in some editors while rendering identically in browser.
EMOJI_MONEY_BAG = "\U0001F4B0"     # 💰
EMOJI_CHART_UP = "\U0001F4C8"      # 📈
EMOJI_BAR_CHART = "\U0001F4CA"     # 📊
EMOJI_REFRESH = "\U0001F504"       # 🔄
EMOJI_LINK = "\U0001F517"          # 🔗

st.set_page_config(page_title="Gold Price Forecast", layout="wide")
st.title(f"{EMOJI_MONEY_BAG} Gold Price Forecasting Dashboard")
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


def convert(value_usd_per_ounce, currency, unit):
    """Convert a USD-per-ounce value to the selected currency/unit combo."""
    value = value_usd_per_ounce
    if currency == "SAR":
        value = value * USD_TO_SAR
    if unit == "gram":
        value = value / OUNCE_TO_GRAM
    return value


# ----- Session state: keep the series and forecast across reruns -----
if "series" not in st.session_state:
    st.session_state.series = load_extended_series()
    st.session_state.forecast = None  # will hold (next_date, pred_price, lower, upper)

col1, col2 = st.columns([1, 3])

with col1:
    if st.button(f"{EMOJI_REFRESH} Update & Predict Tomorrow", type="primary"):
        with st.spinner("Fetching latest prices and refitting model..."):
            st.session_state.series = update_and_refetch()
            next_date, pred_price, lower, upper = predict_next_day(
                st.session_state.series, order=BEST_ORDER
            )
            st.session_state.forecast = (next_date, pred_price, lower, upper)

    # ----- Currency / unit toggle -----
    currency = st.radio("Currency", ["USD", "SAR"], horizontal=True)
    unit = st.radio("Unit", ["ounce", "gram"], horizontal=True)
    symbol = "$" if currency == "USD" else "\uFDFC"  # ﷼

series = st.session_state.series
last_date = series.index.max()
last_price = series.iloc[-1]

prev_price = series.iloc[-2] if len(series) > 1 else None
price_delta = (last_price - prev_price) if prev_price is not None else None

with col1:
    st.metric("Last Known Date", str(last_date.date()))

    last_price_conv = convert(last_price, currency, unit)
    delta_conv = convert(price_delta, currency, unit) if price_delta is not None else None
    st.metric(
        f"Last Known Price ({currency}/{unit})",
        f"{symbol}{last_price_conv:,.2f}",
        delta=f"{delta_conv:,.2f}" if delta_conv is not None else None,
    )

    if st.session_state.forecast is not None:
        next_date, pred_price, lower, upper = st.session_state.forecast

        pred_conv = convert(pred_price, currency, unit)
        lower_conv = convert(lower, currency, unit)
        upper_conv = convert(upper, currency, unit)
        forecast_delta = pred_conv - last_price_conv

        st.metric(
            f"Forecast ({next_date.date()}) {currency}/{unit}",
            f"{symbol}{pred_conv:,.2f}",
            delta=f"{forecast_delta:,.2f} vs last known",
        )
        st.markdown(format_ci(lower_conv, upper_conv, symbol))

        st.caption(f"{EMOJI_BAR_CHART} Historical test MAPE: {HISTORICAL_TEST_MAPE}% "
                   f"(evaluated on 2,326 trading days, 2014-2023)")
    else:
        st.info("Click 'Update & Predict Tomorrow' to generate a forecast.")

with col2:
    recent = series.loc[series.index >= last_date - pd.Timedelta(days=180)]
    recent_conv = recent.apply(lambda v: convert(v, currency, unit))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(recent.index, recent_conv.values, label="Actual (last 180 days)")

    if st.session_state.forecast is not None:
        next_date, pred_price, lower, upper = st.session_state.forecast
        pred_conv = convert(pred_price, currency, unit)
        lower_conv = convert(lower, currency, unit)
        upper_conv = convert(upper, currency, unit)
        ax.errorbar([next_date], [pred_conv],
                    yerr=[[pred_conv - lower_conv], [upper_conv - pred_conv]],
                    fmt="o", color="red", capsize=5, label="Forecast (95% CI)")

    ax.set_xlabel("Date")
    ax.set_ylabel(f"{currency} / {unit}")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

st.caption("Model refits on-demand on the full gap-filled history each time you click Update.")

# ----- Model comparison table -----
st.divider()
st.subheader(f"{EMOJI_CHART_UP} Why ARIMA(1,1,1)?")
st.caption("Selected from 11 models compared in Phase 3, based on RMSE/MAPE on a "
           "2,326-day held-out test set. Lower is better.")

if SUMMARY_PATH.exists():
    summary_df = pd.read_csv(SUMMARY_PATH).sort_values("rmse").head(5).reset_index(drop=True)
    summary_df["rmse"] = summary_df["rmse"].round(3)
    summary_df["mape"] = summary_df["mape"].round(3)
    summary_df = summary_df.rename(
        columns={"model": "Model", "rmse": "RMSE", "mape": "MAPE (%)"}
    )

    def highlight_selected(row):
        is_selected = row["Model"] == SELECTED_MODEL_LABEL
        return ["background-color: #1a3d1a" if is_selected else "" for _ in row]

    st.dataframe(
        summary_df.style.apply(highlight_selected, axis=1),
        hide_index=True,
        use_container_width=True,
    )
else:
    st.caption("Model comparison table not found - run compare_models.py to generate it.")

# ----- Full model comparison chart (expandable) -----
with st.expander(f"{EMOJI_BAR_CHART} Show full model comparison chart (RMSE & MAPE, all models)"):
    if FIG_PATH.exists():
        st.image(str(FIG_PATH), use_container_width=True)
    else:
        st.caption("Chart not found - run compare_models.py to generate it.")

# ----- Footer: link to full source code and analysis -----
st.divider()
st.caption(
    f"{EMOJI_LINK} [View full analysis & code on GitHub]({GITHUB_REPO_URL})"
)