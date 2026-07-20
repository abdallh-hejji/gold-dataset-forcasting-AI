"""
src/train_lstm_returns.py
LSTM on log returns (stationary target) - Gold Price Forecasting project
Reconstructs predicted price using the true previous-day price (fair,
one-step-ahead comparison with ARIMA/Naive, same principle as train_arima.py).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from data_loader import load_daily_usd
from preprocessing import add_log_returns
from train_baseline import train_test_split_series, evaluate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "outputs" / "predictions" / "lstm_returns_results.csv"
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "lstm_returns_model.keras"

SEQUENCE_LENGTH = 60
EPOCHS = 50
BATCH_SIZE = 32
TEST_SIZE = 0.2


def make_sequences(scaled_values, seq_len):
    X, y = [], []
    for i in range(seq_len, len(scaled_values)):
        X.append(scaled_values[i - seq_len:i, 0])
        y.append(scaled_values[i, 0])
    return np.array(X), np.array(y)


def build_model(seq_len):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_len, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


if __name__ == "__main__":
    raw_df = load_daily_usd()
    df = add_log_returns(raw_df)

    returns = df["log_return"]
    train_ret, test_ret = train_test_split_series(returns, test_size=TEST_SIZE)
    print(f"Train size: {len(train_ret)}  |  Test size: {len(test_ret)}\n")

    # ----- Scale returns (fit on train only, avoid leakage) -----
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_ret.values.reshape(-1, 1))
    test_scaled = scaler.transform(test_ret.values.reshape(-1, 1))

    full_scaled = np.concatenate([train_scaled, test_scaled])
    X_all, y_all = make_sequences(full_scaled, SEQUENCE_LENGTH)

    n_train = len(train_ret)
    split_point = n_train - SEQUENCE_LENGTH

    X_train, y_train = X_all[:split_point], y_all[:split_point]
    X_test, y_test = X_all[split_point:], y_all[split_point:]

    X_train = X_train.reshape(-1, SEQUENCE_LENGTH, 1)
    X_test = X_test.reshape(-1, SEQUENCE_LENGTH, 1)

    print(f"Training sequences: {X_train.shape}  |  Test sequences: {X_test.shape}\n")

    # ----- Build & train -----
    model = build_model(SEQUENCE_LENGTH)
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1,
    )

    # ----- Predict returns & inverse-transform -----
    preds_scaled = model.predict(X_test)
    pred_returns = scaler.inverse_transform(preds_scaled).flatten()

    target_index = test_ret.index[-len(pred_returns):]

    # ----- Reconstruct predicted price using the TRUE previous-day price -----
    # (one-step-ahead: same fairness principle used in train_arima.py)
    prev_price = raw_df["USD"].shift(1).loc[target_index]
    pred_price = prev_price.values * np.exp(pred_returns)
    pred_price_series = pd.Series(pred_price, index=target_index)

    actual_price = raw_df["USD"].loc[target_index]

    evaluate(actual_price, pred_price_series, "LSTM (on log returns)")

    # ----- Save model & predictions -----
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Saved model: {MODEL_PATH}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "date": target_index, "actual": actual_price.values, "predicted": pred_price
    }).to_csv(RESULTS_PATH, index=False)
    print(f"Saved: {RESULTS_PATH}")