from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# Project Paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "Daily.csv"
# Figures path
FIGURES_PATH = BASE_DIR / "outputs" / "figures"

# Create folder if it doesn't exist
FIGURES_PATH.mkdir(parents=True, exist_ok=True)
# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv(DATA_PATH)

# ==========================================
# Data Cleaning
# ==========================================

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Convert all price columns to numeric
for col in df.columns[1:]:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", "", regex=False)
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ==========================================
# Basic Information
# ==========================================

print("\n===== Dataset Information =====")
print(df.info())

print("\n===== Missing Values =====")
print(df.isnull().sum())

print("\n===== Statistical Summary =====")
print(df.describe())

# ==========================================
# Gold Price (USD) Over Time
# ==========================================

plt.figure(figsize=(15, 6))
plt.plot(df["Date"], df["USD"])
plt.title("Gold Price in USD (1978 - 2023)")
plt.xlabel("Date")
plt.ylabel("Gold Price (USD)")
plt.grid(True)
plt.tight_layout()
plt.savefig(FIGURES_PATH / "gold_price_usd.png", dpi=300)
plt.close()

# ==========================================
# Distribution of USD Prices
# ==========================================

plt.figure(figsize=(8, 5))
sns.histplot(df["USD"], bins=50, kde=True)
plt.title("Distribution of Gold Prices (USD)")
plt.xlabel("Gold Price")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(FIGURES_PATH / "usd_distribution.png", dpi=300)
plt.close()

# ==========================================
# Boxplot
# ==========================================

plt.figure(figsize=(6, 6))
sns.boxplot(y=df["USD"])
plt.title("Boxplot of Gold Prices (USD)")
plt.tight_layout()
plt.savefig(FIGURES_PATH / "usd_boxplot.png", dpi=300)
plt.close()

# ==========================================
# 30-Day Moving Average
# ==========================================

df["MA30"] = df["USD"].rolling(window=30).mean()

plt.figure(figsize=(15, 6))
plt.plot(df["Date"], df["USD"], label="USD Price")
plt.plot(df["Date"], df["MA30"], label="30-Day Moving Average", linewidth=2)

plt.title("Gold Price with 30-Day Moving Average")
plt.xlabel("Date")
plt.ylabel("Gold Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(FIGURES_PATH / "moving_average_30.png", dpi=300)
plt.close()
