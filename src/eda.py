"""
src/eda.py
Exploratory Data Analysis - Gold Price Forecasting project
Runs the full chain: raw Daily.csv -> clean -> log returns -> ADF -> ACF/PACF
"""

from pathlib import Path
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from data_loader import load_daily_usd
from preprocessing import add_log_returns, save_processed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIG_PATH = PROJECT_ROOT / "outputs" / "figures" / "acf_pacf_plot.png"


def run_adf_test(series, label="Series"):
    """Run Augmented Dickey-Fuller test and print results."""
    result = adfuller(series.dropna(), autolag="AIC")
    print(f"--- ADF Test: {label} ---")
    print(f"ADF Statistic : {result[0]:.4f}")
    print(f"p-value       : {result[1]:.6f}")
    verdict = "STATIONARY" if result[1] < 0.05 else "NON-STATIONARY"
    print(f"Verdict       : {verdict}\n")
    return result[1]


def plot_acf_pacf(series, lags=40, save_path=DEFAULT_FIG_PATH):
    """Plot ACF and PACF for a time series and save the figure."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    plot_acf(series, lags=lags, ax=axes[0])
    axes[0].set_title("ACF - Gold Log Returns (USD)")
    plot_pacf(series, lags=lags, ax=axes[1], method="ywm")
    axes[1].set_title("PACF - Gold Log Returns (USD)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot: {save_path}")
    plt.show()


if __name__ == "__main__":
    # 1. Load raw data
    raw_df = load_daily_usd()

    # 2. Compute log returns
    df = add_log_returns(raw_df)
    save_processed(df)

    # 3. Stationarity tests
    run_adf_test(raw_df["USD"], "Raw USD Price")
    run_adf_test(df["log_return"], "Log Returns")

    # 4. ACF / PACF
    plot_acf_pacf(df["log_return"])