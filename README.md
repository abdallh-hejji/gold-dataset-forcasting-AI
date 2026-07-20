# gold-dataset-forcasting-AI
# 💰 Gold Price Forecasting Dashboard

An end-to-end time series forecasting project analyzing 45+ years of daily gold prices (1978–2023), comparing 11 forecasting models, and deploying a live dashboard that self-updates with real market data.

**🔗 Live Demo:** https://gold-dataset-forcasting-ai-ixzijm2ixbmiaf5gtiih5b.streamlit.app

---

## Key Finding

Gold's daily USD price closely follows a **Random Walk** (weak-form market efficiency). Across 11 models tested — Naive, 3 Moving Average variants, 4 ARIMA configurations, 2 LSTM variants, XGBoost, and Prophet — the best-performing models (ARIMA, Naive, LSTM on log returns) all converge to nearly identical performance:

| Model | RMSE | MAPE |
|---|---|---|
| **ARIMA(1,1,1)** *(selected)* | 13.722 | 0.618% |
| Naive (t-1) | 13.726 | 0.618% |
| LSTM (log returns) | 13.728 | 0.618% |
| Moving Average (5d) | 20.063 | 0.968% |
| LSTM (raw price) | 28.866 | 1.479% |

The one model trained on the **wrong data representation** (LSTM on raw, non-stationary prices instead of log returns) underperformed significantly — a key lesson: **correct data representation matters more than model complexity.**

---

## Live Dashboard

The deployed dashboard:
- Fetches real gold futures prices (Yahoo Finance, `GC=F`) to auto-fill any gap since the original dataset's last date (2023-07-21)
- Refits ARIMA(1,1,1) on the full updated history on each click
- Forecasts the next trading day with a **95% confidence interval**, in USD, USD/gram, SAR, and SAR/gram
- Shows why ARIMA(1,1,1) was selected, with a live comparison table against the other 10 models

---

## Project Pipeline

**Phase 1 — Data Cleaning:** Loaded 11,626 daily USD price records (1978–2023), handled thousand-separator formatting inconsistencies (post-2008 values), verified zero missing values and zero duplicate dates.

**Phase 2 — Stationarity Analysis:** Confirmed non-stationarity in raw prices via ADF test (p = 0.98), converted to log returns (p ≈ 0.00 after transformation), and used ACF/PACF analysis to identify near-white-noise behavior in returns.

**Phase 3 — Model Comparison:** Trained and evaluated 11 models using a chronological 80/20 train-test split (no data leakage), with consistent one-step-ahead evaluation across statistical, machine learning, and deep learning approaches.

**Phase 4 — Validation:** Verified model residuals show no systematic bias (mean error ≈ 0.16 USD) and no practically significant autocorrelation (Ljung-Box test interpreted alongside actual correlation magnitudes, not p-values alone).

**Phase 5 — Production Pipeline:** Built a gap-filling pipeline (`fetch_gap_data.py`) using free live market data, and a Streamlit dashboard for real-time next-day forecasting.

---

## Tech Stack

`Python` · `pandas` · `NumPy` · `statsmodels` (ARIMA/SARIMAX) · `TensorFlow/Keras` (LSTM) · `XGBoost` · `Prophet` · `scikit-learn` · `yfinance` · `Streamlit` · `matplotlib`

---

## Project Structure
gold-dataset-forcasting-AI/
├── data/
│ ├── raw/ # Original World Gold Council datasets (8 frequencies)
│ └── processed/ # Cleaned & gap-filled series
├── notebooks/
│ └── eda.ipynb
├── outputs/
│ ├── figures/ # ACF/PACF, residual diagnostics, model comparison charts
│ ├── models/ # Saved LSTM & XGBoost models
│ └── predictions/ # Per-model prediction CSVs + summary table
├── src/
│ ├── data_loader.py
│ ├── preprocessing.py
│ ├── eda.py
│ ├── train_baseline.py
│ ├── train_arima.py
│ ├── train_lstm.py
│ ├── train_lstm_returns.py
│ ├── train_xgboost.py
│ ├── train_prophet.py
│ ├── compare_models.py
│ ├── evaluation.py
│ ├── fetch_gap_data.py
│ ├── predict_next_day.py
│ ├── convert_to_sar.py
│ ├── utils.py
│ └── dashboard.py
├── requirements.txt
└── README.md
---

## Running Locally

```bash
git clone https://github.com/abdallh-hejji/gold-dataset-forcasting-AI.git
cd gold-dataset-forcasting-AI
pip install -r requirements.txt
streamlit run src/dashboard.py
```

---

## Author

**Abdullah hejji alnuwaysir** — AI student, Imam Abdulrahman Bin Faisal University, College of Computer Science and Information Technology