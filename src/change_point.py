"""
Task 2: Bayesian Change Point Detection for Brent Oil Log Returns.

Fits a single change point model using PyMC to detect a structural break
in the mean and volatility of Brent oil daily log returns, then matches
the detected change point to the nearest event in the compiled event
dataset (data/events.csv) for contextual interpretation.

Usage:
    python src/change_point.py
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pymc as pm
    import arviz as az
except ImportError:
    print("ERROR: pymc and arviz are required. Run: pip install pymc arviz")
    sys.exit(1)


DATA_PATH = "data/processed/brent_clean.csv"
EVENTS_PATH = "data/events.csv"
OUTPUT_DIR = "data/processed"


def load_data(path=DATA_PATH):
    """Load the cleaned Brent oil price series with error handling."""
    try:
        df = pd.read_csv(path, parse_dates=["Date"])
    except FileNotFoundError:
        print(f"ERROR: {path} not found. Run src/data_loader.py first.")
        sys.exit(1)
    except pd.errors.ParserError as e:
        print(f"ERROR: could not parse {path} - {e}")
        sys.exit(1)

    if "log_return" not in df.columns:
        print("ERROR: expected a 'log_return' column in the cleaned data.")
        sys.exit(1)

    return df.dropna(subset=["log_return"]).reset_index(drop=True)


def load_events(path=EVENTS_PATH):
    """Load the structured event dataset used to contextualize change points."""
    try:
        events = pd.read_csv(path, parse_dates=["date"])
    except FileNotFoundError:
        print(f"WARNING: {path} not found - change points will be reported without event context.")
        return None
    return events


def build_and_sample_model(log_returns, draws=2000, tune=1000, chains=4, target_accept=0.9, cores=1):
    """Build a single change point model over the mean and volatility of log returns.

    The model assumes log returns are Normally distributed with a mean and
    volatility that shift at an unknown change point `tau`. Priors are
    weakly informative given the observed scale of the data.

    Returns the fitted PyMC model and its InferenceData trace.
    """
    n = len(log_returns)
    idx = np.arange(n)

    with pm.Model() as model:
        tau = pm.DiscreteUniform("tau", lower=0, upper=n - 1)

        mu1 = pm.Normal("mu1", mu=0, sigma=0.05)
        mu2 = pm.Normal("mu2", mu=0, sigma=0.05)
        sigma1 = pm.HalfNormal("sigma1", sigma=0.05)
        sigma2 = pm.HalfNormal("sigma2", sigma=0.05)

        mu = pm.math.switch(idx < tau, mu1, mu2)
        sigma = pm.math.switch(idx < tau, sigma1, sigma2)

        pm.Normal("observed", mu=mu, sigma=sigma, observed=log_returns)

        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            return_inferencedata=True,
            progressbar=True,
        )

    return model, trace


def check_convergence(trace):
    """Print R-hat and effective sample size diagnostics for the key parameters."""
    summary = az.summary(trace, var_names=["tau", "mu1", "mu2", "sigma1", "sigma2"])
    print("\n=== Convergence Diagnostics ===")
    print(summary)

    rhat_numeric = pd.to_numeric(summary["r_hat"], errors="coerce")
    max_rhat = rhat_numeric.max()
    if pd.isna(max_rhat):
        print("WARNING: could not compute R-hat (NaN) - check sampler output above.")
        return summary
    if max_rhat > 1.01:
        print(f"WARNING: max R-hat = {max_rhat:.4f} - chains may not have converged. "
              f"Consider increasing `tune` or `draws`.")
    else:
        print(f"R-hat looks good (max = {max_rhat:.4f} <= 1.01).")

    return summary


def extract_change_point(df, trace):
    """Extract the most probable change point index/date and before/after parameter estimates."""
    tau_samples = trace.posterior["tau"].values.flatten()
    tau_mode = int(pd.Series(tau_samples).mode()[0])
    change_date = df.loc[tau_mode, "Date"]

    mu1 = float(trace.posterior["mu1"].mean())
    mu2 = float(trace.posterior["mu2"].mean())
    sigma1 = float(trace.posterior["sigma1"].mean())
    sigma2 = float(trace.posterior["sigma2"].mean())

    before = df.iloc[:tau_mode]
    after = df.iloc[tau_mode:]

    result = {
        "change_point_index": tau_mode,
        "change_point_date": str(change_date.date()),
        "mean_log_return_before": mu1,
        "mean_log_return_after": mu2,
        "volatility_before": sigma1,
        "volatility_after": sigma2,
        "avg_price_before": float(before["Price"].mean()) if len(before) else None,
        "avg_price_after": float(after["Price"].mean()) if len(after) else None,
    }
    return result


def match_nearest_event(change_date, events):
    """Find the nearest event in the event dataset to a detected change point date."""
    if events is None or events.empty:
        return None

    change_date = pd.to_datetime(change_date)
    events = events.copy()
    events["days_diff"] = (events["date"] - change_date).abs().dt.days
    nearest = events.loc[events["days_diff"].idxmin()]

    return {
        "event": nearest["event"],
        "event_date": str(nearest["date"].date()),
        "days_from_change_point": int(nearest["days_diff"]),
        "category": nearest.get("category", None),
    }


def plot_tau_posterior(trace, output_dir=OUTPUT_DIR):
    """Plot the posterior distribution of the change point location tau."""
    tau_samples = trace.posterior["tau"].values.flatten()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(tau_samples, bins=50, color="#2980b9", edgecolor="white")
    ax.axvline(np.median(tau_samples), color="red", linestyle="--",
               label=f"Median tau = {int(np.median(tau_samples))}")
    ax.set_title("Posterior Distribution of Change Point (tau)", fontweight="bold")
    ax.set_xlabel("Index position in time series")
    ax.set_ylabel("Posterior sample count")
    ax.legend()
    plt.tight_layout()

    out_path = os.path.join(output_dir, "tau_posterior.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved tau posterior plot -> {out_path}")


def plot_change_point(df, change_point_index, output_dir=OUTPUT_DIR):
    """Plot the price series with the detected change point marked."""
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(df.Date, df.Price, color="#2980b9", linewidth=0.8, label="Brent Oil Price")

    change_date = df.loc[change_point_index, "Date"]
    ax.axvline(change_date, color="red", linewidth=1.5, linestyle="--",
               label=f"Detected change point: {change_date.date()}")

    ax.set_title("Brent Oil Price with Detected Change Point", fontweight="bold", fontsize=13)
    ax.set_ylabel("Price (USD/barrel)")
    ax.set_xlabel("Date")
    ax.legend()
    plt.tight_layout()

    out_path = os.path.join(output_dir, "change_point.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved change point plot -> {out_path}")


def save_summary(result, matched_event, output_dir=OUTPUT_DIR):
    """Save the change point result and matched event as a JSON summary."""
    summary = {**result, "matched_event": matched_event}
    out_path = os.path.join(output_dir, "change_point_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary -> {out_path}")
    return summary


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data()
    events = load_events()

    print(f"Fitting change point model on {len(df)} log return observations...")
    model, trace = build_and_sample_model(df["log_return"].values, draws=1000, tune=1000, chains=4, cores=1)

    check_convergence(trace)

    result = extract_change_point(df, trace)
    matched_event = match_nearest_event(result["change_point_date"], events)

    print("\n=== Change Point Result ===")
    for k, v in result.items():
        print(f"{k}: {v}")

    if matched_event:
        print("\n=== Nearest Matching Event ===")
        for k, v in matched_event.items():
            print(f"{k}: {v}")
    else:
        print("\nNo event dataset available - change point reported without event context.")

    plot_tau_posterior(trace)
    plot_change_point(df, result["change_point_index"])
    save_summary(result, matched_event)

    print("\nChange point analysis complete!")


if __name__ == "__main__":
    main()
