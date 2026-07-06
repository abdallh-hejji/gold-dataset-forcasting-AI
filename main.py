import pandas as pd

# Read dataset
df = pd.read_csv("data/raw/Daily.csv")

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"])

# Convert all price columns to numeric
for col in df.columns[1:]:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", "", regex=False)
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.describe())