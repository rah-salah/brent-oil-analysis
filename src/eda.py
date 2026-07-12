"""
Exploratory Data Analysis for Brent Oil Price Data.

Loads cleaned Brent oil price data, generates EDA visualizations
(price history, log returns, rolling statistics, distributions),
and runs stationarity tests (ADF) on price and log returns.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller


DATA_PATH = "data/processed/brent_clean.csv"
OUTPUT_DIR = "data/processed"

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


def load_data(path=DATA_PATH):
    """Load the cleaned Brent oil price CSV, parsing the Date column.

    Raises a clear, actionable error if the file is missing or malformed,
    rather than letting pandas' raw traceback surface to the user.
    """
    try:
        df = pd.read_csv(path, parse_dates=["Date"])
    except FileNotFoundError:
        print(f"ERROR: {path} not found. Run the data cleaning step first.")
        sys.exit(1)
    except pd.errors.ParserError as e:
        print(f"ERROR: could not parse {path} - {e}")
        sys.exit(1)

    if df.empty:
        print(f"ERROR: {path} loaded but contains no rows.")
        sys.exit(1)

    return df


def plot_price_history(df, events=EVENTS, output_dir=OUTPUT_DIR):
    """Plot the full price series with key geopolitical/OPEC events overlaid."""
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(df.Date, df.Price, color="#2980b9", linewidth=0.8, label="Brent Oil Price")

    for date_str, label in events:
        d = pd.to_datetime(date_str)
        ax.axvline(d, color="red", alpha=0.4, linewidth=0.8)
        ax.text(d, df.Price.max() * 0.95, label, rotation=90, fontsize=6, color="darkred", va="top")

    ax.set_title("Brent Oil Prices 1987-2022 with Key Geopolitical Events", fontweight="bold", fontsize=13)
    ax.set_ylabel("Price (USD/barrel)")
    ax.set_xlabel("Date")
    ax.legend()
    plt.tight_layout()

    out_path = os.path.join(output_dir, "price_history.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure 1 saved: price history -> {out_path}")


def plot_log_returns(df, output_dir=OUTPUT_DIR):
    """Plot the daily log return series."""
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(df.Date, df.log_return, color="#e74c3c", linewidth=0.5, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Brent Oil Log Returns (Daily)", fontweight="bold")
    ax.set_ylabel("Log Return")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "log_returns.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure 2 saved: log returns -> {out_path}")


def plot_rolling_stats(df, window=90, output_dir=OUTPUT_DIR):
    """Plot rolling mean price and rolling volatility (std of price) over a window."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    roll = df.set_index("Date")["Price"].rolling(window)

    axes[0].plot(df.Date, df.Price, color="#95a5a6", linewidth=0.5, label="Price")
    axes[0].plot(df.Date, roll.mean().values, color="#2980b9", linewidth=1.5, label=f"{window}-day Rolling Mean")
    axes[0].set_title("Rolling Mean", fontweight="bold")
    axes[0].legend()

    axes[1].plot(df.Date, roll.std().values, color="#e67e22", linewidth=1)
    axes[1].set_title("Rolling Standard Deviation (Volatility)", fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "rolling_stats.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure 3 saved: rolling stats -> {out_path}")


def plot_distributions(df, output_dir=OUTPUT_DIR):
    """Plot histograms of raw price and log return distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(df.Price, bins=50, color="#3498db", edgecolor="white")
    axes[0].set_title("Price Distribution", fontweight="bold")
    axes[0].set_xlabel("Price (USD)")

    axes[1].hist(df.log_return, bins=100, color="#e74c3c", edgecolor="white")
    axes[1].set_title("Log Return Distribution", fontweight="bold")
    axes[1].set_xlabel("Log Return")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "distributions.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure 4 saved: distributions -> {out_path}")


def run_stationarity_tests(df):
    """Run Augmented Dickey-Fuller tests on raw price and log returns.

    Returns a dict of results so callers (e.g. a dashboard or test suite)
    can use the numbers programmatically instead of only printing them.
    """
    print("\n=== ADF Stationarity Tests ===")
    results = {}

    for col, name in [("Price", "Raw Price"), ("log_return", "Log Returns")]:
        series = df[col].dropna()
        if series.empty:
            print(f"WARNING: no data available to test stationarity for {name}")
            continue

        stat, pvalue, *_ = adfuller(series)
        is_stationary = pvalue < 0.05
        results[name] = {"adf_stat": stat, "p_value": pvalue, "stationary": is_stationary}
        print(f"{name}: ADF={stat:.4f}, p={pvalue:.4f}, Stationary={is_stationary}")

    return results


def print_summary_stats(df):
    """Print summary statistics for the price series."""
    print("\n=== Price Summary ===")
    print(f"Mean: ${df.Price.mean():.2f}")
    print(f"Max:  ${df.Price.max():.2f} on {df.loc[df.Price.idxmax(), 'Date'].date()}")
    print(f"Min:  ${df.Price.min():.2f} on {df.loc[df.Price.idxmin(), 'Date'].date()}")
    print(f"Volatility (std log return): {df.log_return.std():.4f}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data()

    plot_price_history(df)
    plot_log_returns(df)
    plot_rolling_stats(df)
    plot_distributions(df)

    run_stationarity_tests(df)
    print_summary_stats(df)

    print("\nEDA complete!")


if __name__ == "__main__":
    main()
