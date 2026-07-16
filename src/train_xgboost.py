"""
src/train_xgboost.py
XGBoost forecasting model (lag & rolling features) - Gold Price Forecasting project
"""

from pathlib import Path
import pandas as pd
from xgboost import XGBRegressor

from data_loader import load_daily_usd
from feature_engineering import build_features, get_feature_columns
from train_baseline import evaluate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "outputs" / "predictions" / "xgboost_results.csv"
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "xgboost_model.json"

TEST_SIZE = 0.2


if __name__ == "__main__":
    raw_df = load_daily_usd()
    features_df = build_features(raw_df["USD"])

    feature_cols = get_feature_columns(features_df)
    X = features_df[feature_cols]
    y = features_df["USD"]

    # Chronological split - same principle as train_baseline.py, no shuffling
    split_idx = int(len(features_df) * (1 - TEST_SIZE))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")
    print(f"Train range: {y_train.index.min().date()} -> {y_train.index.max().date()}")
    print(f"Test range : {y_test.index.min().date()} -> {y_test.index.max().date()}\n")

    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = pd.Series(model.predict(X_test), index=y_test.index)
    evaluate(y_test, preds, "XGBoost (lag/rolling features)")

    # ----- Feature importance -----
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False)
    print("\nTop 5 feature importances:")
    print(importances.head(5))

    # ----- Save model & predictions -----
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    print(f"\nSaved model: {MODEL_PATH}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "date": y_test.index, "actual": y_test.values, "predicted": preds.values
    }).to_csv(RESULTS_PATH, index=False)
    print(f"Saved: {RESULTS_PATH}")