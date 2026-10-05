"""
src/dm_test.py
Diebold-Mariano test - formally tests whether the difference in forecast
accuracy between two models is statistically significant, rather than just
comparing point RMSE/MAPE values.
Gold Price Forecasting project - Phase 6

Reference: Diebold, F.X. and Mariano, R.S. (1995), "Comparing Predictive
Accuracy," Journal of Business & Economic Statistics.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = PROJECT_ROOT / "outputs" / "predictions"


def diebold_mariano(actual, pred1, pred2, h=1):
    """
    DM test comparing squared-error loss of two forecasts.

    H0: both forecasts have equal predictive accuracy (mean loss difference = 0)
    H1: forecasts differ in accuracy

    Returns (dm_statistic, p_value). A large |DM stat| / small p-value means
    the accuracy difference is unlikely to be due to chance.
    """
    e1 = actual - pred1
    e2 = actual - pred2
    d = e1**2 - e2**2  # loss differential (squared error)

    T = len(d)
    d_mean = d.mean()

    # Newey-West long-run variance (h-1 lags - h=1 means just the sample variance)
    gamma0 = np.var(d, ddof=0)
    lr_var = gamma0
    for lag in range(1, h):
        gamma = np.cov(d[:-lag], d[lag:])[0, 1]
        lr_var += 2 * gamma

    dm_stat = d_mean / np.sqrt(lr_var / T)
    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
    return dm_stat, p_value


def load_predictions(label):
    path = PRED_DIR / f"macro_arima_{label}_predictions.csv"
    df = pd.read_csv(path)
    return df["actual"].values, df["predicted"].values


def report(name1, name2):
    actual1, pred1 = load_predictions(name1)
    actual2, pred2 = load_predictions(name2)

    assert len(actual1) == len(actual2), "Prediction files have different lengths - re-run train_macro_comparison.py"

    dm_stat, p_value = diebold_mariano(actual1, pred1, pred2, h=1)

    rmse1 = np.sqrt(np.mean((actual1 - pred1) ** 2))
    rmse2 = np.sqrt(np.mean((actual2 - pred2) ** 2))

    print(f"--- {name1} vs {name2} ---")
    print(f"  RMSE ({name1}): {rmse1:.4f}")
    print(f"  RMSE ({name2}): {rmse2:.4f}")
    print(f"  DM statistic  : {dm_stat:.4f}")
    print(f"  p-value       : {p_value:.4f}")

    if p_value < 0.05:
        better = name2 if rmse2 < rmse1 else name1
        print(f"  -> STATISTICALLY SIGNIFICANT (p < 0.05): {better} is significantly more accurate.")
    else:
        print("  -> NOT statistically significant (p >= 0.05): "
              "the accuracy difference could plausibly be due to chance.")
    print()


if __name__ == "__main__":
    report("baseline", "dxy")
    report("baseline", "combined")
