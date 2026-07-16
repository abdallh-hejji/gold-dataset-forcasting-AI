"""
src/train_lstm.py
LSTM forecasting model - Gold Price Forecasting project
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from data_loader import load_daily_usd
from train_baseline import train_test_split_series, evaluate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "outputs" / "predictions" / "lstm_results.csv"
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "lstm_model.keras"

SEQUENCE_LENGTH = 60  # ~3 trading months of history per prediction
EPOCHS = 50
BATCH_SIZE = 32


def make_sequences(scaled_values, seq_len):
    """Turn a scaled 1-column array into (X, y) sliding-window sequences."""
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
    df = load_daily_usd()
    series = df["USD"]

    train, test = train_test_split_series(series)
    print(f"Train size: {len(train)}  |  Test size: {len(test)}\n")

    # ----- Scale: fit on train only to avoid leakage -----
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
    test_scaled = scaler.transform(test.values.reshape(-1, 1))

    # Combine so test sequences can use the tail of train as history
    full_scaled = np.concatenate([train_scaled, test_scaled])
    X_all, y_all = make_sequences(full_scaled, SEQUENCE_LENGTH)

    # Sequence i predicts the value at position (SEQUENCE_LENGTH + i) of the
    # combined series. Split by whether that target index falls in train or test.
    n_train = len(train)
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

    # ----- Predict & inverse-transform back to USD scale -----
    preds_scaled = model.predict(X_test)
    preds = scaler.inverse_transform(preds_scaled).flatten()

    target_index = test.index[-len(preds):]
    preds_series = pd.Series(preds, index=target_index)
    y_true_series = test.loc[target_index]

    evaluate(y_true_series, preds_series, "LSTM")

    # ----- Save model & predictions -----
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Saved model: {MODEL_PATH}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "date": target_index, "actual": y_true_series.values, "predicted": preds
    }).to_csv(RESULTS_PATH, index=False)
    print(f"Saved: {RESULTS_PATH}")