"""
src/compare_models.py
Combine results from all trained models into one comparison table and chart
Gold Price Forecasting project
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = PROJECT_ROOT / "outputs" / "predictions"
FIG_PATH = PROJECT_ROOT / "outputs" / "figures" / "model_comparison.png"
SUMMARY_PATH = PRED_DIR / "all_models_summary.csv"

# Excluded from the chart only (not the table/CSV) - its multi-step
# methodology isn't a fair visual comparison against one-step-ahead models,
# and its scale (RMSE ~777) would flatten every other bar.
EXCLUDE_FROM_CHART = ["Prophet (multi-step, not one-step-ahead)"]


def metrics_from_predictions(path, label):
    """Compute RMSE/MAPE from a (date, actual, predicted) predictions file."""
    df = pd.read_csv(path)
    rmse = np.sqrt(mean_squared_error(df["actual"], df["predicted"]))
    mape = mean_absolute_percentage_error(df["actual"], df["predicted"]) * 100
    return {"model": label, "rmse": rmse, "mape": mape}


if __name__ == "__main__":
    all_results = []

    # ----- Already-summarized results (baseline + ARIMA) -----
    baseline_df = pd.read_csv(PRED_DIR / "baseline_results.csv")
    arima_df = pd.read_csv(PRED_DIR / "arima_results.csv")[["model", "rmse", "mape"]]
    all_results.append(baseline_df)
    all_results.append(arima_df)

    # ----- Results computed from raw prediction files -----
    file_labels = [
        ("lstm_results.csv", "LSTM (raw price)"),
        ("lstm_returns_results.csv", "LSTM (log returns)"),
        ("xgboost_results.csv", "XGBoost (lag/rolling features)"),
        ("prophet_results.csv", "Prophet (multi-step, not one-step-ahead)"),
    ]
    computed = [metrics_from_predictions(PRED_DIR / fname, label) for fname, label in file_labels]
    all_results.append(pd.DataFrame(computed))

    # ----- Combine, sort, save (full table includes Prophet) -----
    summary = pd.concat(all_results, ignore_index=True)
    summary = summary.sort_values("rmse").reset_index(drop=True)

    print("Full summary (all models):")
    print(summary.to_string(index=False))

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"\nSaved: {SUMMARY_PATH}")

    # ----- Chart data: exclude Prophet for visual clarity -----
    chart_data = summary[~summary["model"].isin(EXCLUDE_FROM_CHART)].reset_index(drop=True)

    def bar_color(m):
        if m == chart_data.loc[chart_data["rmse"].idxmin(), "model"]:
            return "#2ca02c"
        if "raw price" in m:
            return "#d62728"
        return "#1f77b4"

    colors = [bar_color(m) for m in chart_data["model"]]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].barh(chart_data["model"], chart_data["rmse"], color=colors)
    axes[0].set_xlabel("RMSE (lower is better)")
    axes[0].set_title("Model Comparison - RMSE\n(Prophet excluded - see table for full results)")
    axes[0].invert_yaxis()

    axes[1].barh(chart_data["model"], chart_data["mape"], color=colors)
    axes[1].set_xlabel("MAPE % (lower is better)")
    axes[1].set_title("Model Comparison - MAPE\n(Prophet excluded - see table for full results)")
    axes[1].invert_yaxis()

    plt.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_PATH, dpi=150)
    print(f"Saved plot: {FIG_PATH}")
    plt.show()