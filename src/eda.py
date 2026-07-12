import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import os

os.makedirs("data/processed", exist_ok=True)

df = pd.read_csv("data/processed/brent_clean.csv", parse_dates=["Date"])

# Key geopolitical events
EVENTS = [
    ("1990-08-02", "Gulf War starts"),
    ("1997-07-01", "Asian Financial Crisis"),
    ("2001-09-11", "9/11 Attacks"),
    ("2003-03-20", "Iraq War"),
    ("2005-08-29", "Hurricane Katrina"),
    ("2008-07-11", "Oil Peak $147"),
    ("2008-09-15", "Global Financial Crisis"),
    ("2010-12-18", "Arab Spring"),
    ("2011-02-17", "Libya Civil War"),
    ("2014-11-27", "OPEC No Cut Decision"),
    ("2016-01-16", "Iran Sanctions Lifted"),
    ("2020-03-06", "COVID-19 Oil Price War"),
    ("2022-02-24", "Russia invades Ukraine"),
]

# Figure 1: Price history with events
fig, ax = plt.subplots(figsize=(16, 7))
ax.plot(df.Date, df.Price, color="#2980b9", linewidth=0.8, label="Brent Oil Price")
for date_str, label in EVENTS:
    d = pd.to_datetime(date_str)
    ax.axvline(d, color="red", alpha=0.4, linewidth=0.8)
    ax.text(d, df.Price.max() * 0.95, label, rotation=90, fontsize=6, color="darkred", va="top")
ax.set_title("Brent Oil Prices 1987-2022 with Key Geopolitical Events", fontweight="bold", fontsize=13)
ax.set_ylabel("Price (USD/barrel)")
ax.set_xlabel("Date")
ax.legend()
plt.tight_layout()
plt.savefig("data/processed/price_history.png", dpi=150, bbox_inches="tight")
plt.close()
print("Figure 1 saved: price history")

# Figure 2: Log returns
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(df.Date, df.log_return, color="#e74c3c", linewidth=0.5, alpha=0.8)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_title("Brent Oil Log Returns (Daily)", fontweight="bold")
ax.set_ylabel("Log Return")
plt.tight_layout()
plt.savefig("data/processed/log_returns.png", dpi=150, bbox_inches="tight")
plt.close()
print("Figure 2 saved: log returns")

# Figure 3: Rolling mean and std
fig, axes = plt.subplots(2, 1, figsize=(16, 8))
roll = df.set_index("Date")["Price"].rolling(90)
axes[0].plot(df.Date, df.Price, color="#95a5a6", linewidth=0.5, label="Price")
axes[0].plot(df.Date, roll.mean().values, color="#2980b9", linewidth=1.5, label="90-day Rolling Mean")
axes[0].set_title("Rolling Mean", fontweight="bold")
axes[0].legend()
axes[1].plot(df.Date, roll.std().values, color="#e67e22", linewidth=1)
axes[1].set_title("Rolling Standard Deviation (Volatility)", fontweight="bold")
plt.tight_layout()
plt.savefig("data/processed/rolling_stats.png", dpi=150, bbox_inches="tight")
plt.close()
print("Figure 3 saved: rolling stats")

# Figure 4: Distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(df.Price, bins=50, color="#3498db", edgecolor="white")
axes[0].set_title("Price Distribution", fontweight="bold")
axes[0].set_xlabel("Price (USD)")
axes[1].hist(df.log_return, bins=100, color="#e74c3c", edgecolor="white")
axes[1].set_title("Log Return Distribution", fontweight="bold")
axes[1].set_xlabel("Log Return")
plt.tight_layout()
plt.savefig("data/processed/distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("Figure 4 saved: distributions")

# ADF stationarity tests
print("\n=== ADF Stationarity Tests ===")
for col, name in [("Price", "Raw Price"), ("log_return", "Log Returns")]:
    result = adfuller(df[col].dropna())
    print(f"{name}: ADF={result[0]:.4f}, p={result[1]:.4f}, Stationary={result[1] < 0.05}")

# Summary stats
print("\n=== Price Summary ===")
print(f"Mean: ${df.Price.mean():.2f}")
print(f"Max:  ${df.Price.max():.2f} on {df.loc[df.Price.idxmax(), 'Date'].date()}")
print(f"Min:  ${df.Price.min():.2f} on {df.loc[df.Price.idxmin(), 'Date'].date()}")
print(f"Volatility (std log return): {df.log_return.std():.4f}")
