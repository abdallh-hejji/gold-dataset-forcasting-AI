"""
src/evaluation.py
Visual & statistical validation of model predictions - Gold Price Forecasting project
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = PROJECT_ROOT / "outputs" / "predictions"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"

RESULTS_FILE = PRED_DIR / "lstm_returns_results.csv"  # best model with daily predictions saved


def load_results(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date")
    df["error"] = df["actual"] - df["predicted"]
    return df


def plot_actual_vs_predicted(df, save_path):
    plt.figure(figsize=(14, 5))
    plt.plot(df.index, df["actual"], label="Actual", linewidth=1)
    plt.plot(df.index, df["predicted"], label="Predicted", linewidth=1, alpha=0.7)
    plt.title("Actual vs Predicted - Gold Price (USD)")
    plt.xlabel("Date")
    plt.ylabel("USD")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")


def plot_residuals(df, save_path):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].plot(df.index, df["error"], linewidth=0.6)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Prediction Error Over Time (Actual - Predicted)")
    axes[0].set_ylabel("Error (USD)")
    axes[0].grid(alpha=0.3)

    axes[1].hist(df["error"], bins=50)
    axes[1].set_title("Error Distribution")
    axes[1].set_xlabel("Error (USD)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")


def plot_residual_acf(df, save_path, lags=20):
    """
    Plot ACF of residuals AND print the raw correlation values.
    A statistically 'significant' Ljung-Box result can still be
    practically meaningless if the actual correlation values are tiny -
    this checks the magnitude, not just the p-value.
    """
    plt.figure(figsize=(10, 5))
    plot_acf(df["error"], lags=lags, ax=plt.gca())
    plt.title("ACF of Residuals (Prediction Errors)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")

    acf_values = acf(df["error"], nlags=lags)
    print("\nResidual autocorrelation values (first 10 lags):")
    for lag, value in enumerate(acf_values[1:11], start=1):
        flag = " <- largest" if abs(value) == max(abs(v) for v in acf_values[1:11]) else ""
        print(f"  Lag {lag:2d}: {value:+.4f}{flag}")


def check_residual_bias(df):
    mean_error = df["error"].mean()
    print(f"Mean error (bias): {mean_error:.4f} USD")
    if abs(mean_error) < 0.5:
        print("-> No meaningful systematic bias (model doesn't consistently over/under-predict).")
    else:
        print("-> Possible systematic bias - investigate further.")

    lb_result = acorr_ljungbox(df["error"], lags=[10], return_df=True)
    p_value = lb_result["lb_pvalue"].iloc[0]
    print(f"Ljung-Box p-value (lag=10): {p_value:.4f}")
    if p_value > 0.05:
        print("-> Residuals behave like white noise (good - no leftover pattern missed).")
    else:
        print("-> Statistically significant autocorrelation detected - check magnitude below "
              "(with N > 2000, tiny correlations can still be 'significant').")


if __name__ == "__main__":
    df = load_results(RESULTS_FILE)

    plot_actual_vs_predicted(df, FIG_DIR / "actual_vs_predicted.png")
    plot_residuals(df, FIG_DIR / "residuals_diagnostic.png")
    check_residual_bias(df)
    plot_residual_acf(df, FIG_DIR / "residuals_acf.png")