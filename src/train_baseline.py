"""
src/train_baseline.py
Baseline forecasting models (Naive & Moving Average) - Gold Price Forecasting project
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

from data_loader import load_daily_usd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "outputs" / "predictions" / "baseline_results.csv"

TEST_SIZE = 0.2  # last 20% of the series used as the test set
MA_WINDOWS = [5, 20, 60]  # ~1 week, ~1 month, ~1 quarter of trading days


def train_test_split_series(series, test_size=TEST_SIZE):
    """Chronological split - no shuffling, since this is time series data."""
    split_idx = int(len(series) * (1 - test_size))
    train, test = series.iloc[:split_idx], series.iloc[split_idx:]
    return train, test


def naive_forecast(train, test):
    """Predict each test-day value as the previous actual value (random walk)."""
    full = pd.concat([train, test])
    preds = full.shift(1).loc[test.index]
    return preds


def moving_average_forecast(train, test, window):
    """Predict each test-day value as the rolling mean of the previous `window` days."""
    full = pd.concat([train, test])
    preds = full.rolling(window=window).mean().shift(1).loc[test.index]
    return preds


def evaluate(y_true, y_pred, label):
    """Print RMSE and MAPE, skipping any NaN rows from warm-up windows."""
    mask = y_pred.notna()
    rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
    mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100
    print(f"{label:25s} RMSE: {rmse:8.3f}   MAPE: {mape:6.3f}%")
    return {"model": label, "rmse": rmse, "mape": mape}


if __name__ == "__main__":
    df = load_daily_usd()
    series = df["USD"]

    train, test = train_test_split_series(series)
    print(f"Train size: {len(train)}  |  Test size: {len(test)}")
    print(f"Train range: {train.index.min().date()} -> {train.index.max().date()}")
    print(f"Test range : {test.index.min().date()} -> {test.index.max().date()}\n")

    results = []

    # ----- Naive baseline (random walk) -----
    naive_preds = naive_forecast(train, test)
    results.append(evaluate(test, naive_preds, "Naive (t-1)"))

    # ----- Moving average baselines -----
    for w in MA_WINDOWS:
        ma_preds = moving_average_forecast(train, test, w)
        results.append(evaluate(test, ma_preds, f"Moving Average ({w}d)"))

    # ----- Save results -----
    results_df = pd.DataFrame(results)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved: {RESULTS_PATH}")
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