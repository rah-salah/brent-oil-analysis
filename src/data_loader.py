import pandas as pd
import numpy as np
import os


def load_brent_data(path="data/raw/BrentOilPrices.csv"):
    """Load and clean Brent oil price data."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True, dayfirst=True)
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.dropna(subset=["Price"])
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna()
    df["log_return"] = np.log(df["Price"]).diff()
    df = df.dropna()
    print(f"Loaded {len(df)} rows from {df.Date.min().date()} to {df.Date.max().date()}")
    return df


if __name__ == "__main__":
    df = load_brent_data()
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/brent_clean.csv", index=False)
    print(df.describe().round(2))
    print("Saved to data/processed/brent_clean.csv")
