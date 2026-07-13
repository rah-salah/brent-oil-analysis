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
        print(f"ERROR: could not parse {path} — {e}")
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
        print(f"WARNING: {path} not found — change points will be reported without event context.")
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
        print("WARNING: could not compute R-hat (NaN) — check sampler output above.")
        return summary
    if max_rhat > 1.01:
        print(f"WARNING: max R-hat = {max_rhat:.4f} — chains may not have converged. "
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


def plot_trace_diagnostics(trace, output_dir=OUTPUT_DIR):
    """Save trace plots for all parameters, used to visually check MCMC convergence
    (alongside the numeric R-hat check in check_convergence)."""
    pc = az.plot_trace(trace, var_names=["tau", "mu1", "mu2", "sigma1", "sigma2"])

    out_path = os.path.join(output_dir, "trace_plots.png")
    try:
        pc.savefig(out_path)
    except AttributeError:
        # Fallback for older arviz versions where plot_trace returns a numpy array of axes
        fig = pc.ravel()[0].figure
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved trace plots -> {out_path}")


def plot_parameter_posteriors(trace, output_dir=OUTPUT_DIR):
    """Plot posterior distributions for before/after mean and volatility, and report
    probabilistic statements (e.g. P(volatility increased after the change point))."""
    mu1 = trace.posterior["mu1"].values.flatten()
    mu2 = trace.posterior["mu2"].values.flatten()
    sigma1 = trace.posterior["sigma1"].values.flatten()
    sigma2 = trace.posterior["sigma2"].values.flatten()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(mu1, bins=40, alpha=0.6, label="mu1 (before)", color="#2980b9")
    axes[0].hist(mu2, bins=40, alpha=0.6, label="mu2 (after)", color="#e74c3c")
    axes[0].set_title("Posterior: Mean Log Return, Before vs After", fontweight="bold", fontsize=11)
    axes[0].legend()

    axes[1].hist(sigma1, bins=40, alpha=0.6, label="sigma1 (before)", color="#2980b9")
    axes[1].hist(sigma2, bins=40, alpha=0.6, label="sigma2 (after)", color="#e74c3c")
    axes[1].set_title("Posterior: Volatility, Before vs After", fontweight="bold", fontsize=11)
    axes[1].legend()

    plt.tight_layout()
    out_path = os.path.join(output_dir, "parameter_posteriors.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved parameter posterior plot -> {out_path}")

    # Probabilistic statements directly from the posterior samples (not just point estimates)
    p_vol_increased = float(np.mean(sigma2 > sigma1))
    p_mean_increased = float(np.mean(mu2 > mu1))

    print("\n=== Probabilistic Statements ===")
    print(f"P(volatility after > volatility before) = {p_vol_increased:.3f}")
    print(f"P(mean log return after > mean log return before) = {p_mean_increased:.3f}")

    return {"p_volatility_increased": p_vol_increased, "p_mean_increased": p_mean_increased}


def quantify_local_impact(df, change_point_index, window=90):
    """Quantify the price/volatility shift in a symmetric window immediately around
    the change point, rather than full-sample before/after averages (which get diluted
    by a 35-year dataset). This is the number that belongs in an impact statement like
    'price shifted from $X to $Y, a Z% change'.
    """
    start = max(0, change_point_index - window)
    end = min(len(df), change_point_index + window)

    before = df.iloc[start:change_point_index]
    after = df.iloc[change_point_index:end]

    if before.empty or after.empty:
        return None

    avg_price_before = float(before["Price"].mean())
    avg_price_after = float(after["Price"].mean())
    pct_change = ((avg_price_after - avg_price_before) / avg_price_before) * 100

    vol_before = float(before["log_return"].std())
    vol_after = float(after["log_return"].std())

    result = {
        "window_days": window,
        "avg_price_before_window": round(avg_price_before, 2),
        "avg_price_after_window": round(avg_price_after, 2),
        "pct_price_change": round(pct_change, 2),
        "volatility_before_window": round(vol_before, 5),
        "volatility_after_window": round(vol_after, 5),
    }

    print(f"\n=== Quantified Local Impact (+/- {window} days around change point) ===")
    print(f"Avg price before: ${result['avg_price_before_window']}  ->  after: ${result['avg_price_after_window']}"
          f"  ({'+' if pct_change >= 0 else ''}{result['pct_price_change']}%)")
    print(f"Volatility before: {result['volatility_before_window']}  ->  after: {result['volatility_after_window']}")

    return result


def save_summary(result, matched_event, local_impact=None, probabilistic=None, output_dir=OUTPUT_DIR):
    """Save the change point result, matched event, local impact, and probabilistic
    statements as a single JSON summary (consumed by the Task 3 dashboard API)."""
    summary = {**result, "matched_event": matched_event}
    if local_impact:
        summary["local_impact"] = local_impact
    if probabilistic:
        summary["probabilistic"] = probabilistic
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
    model, trace = build_and_sample_model(df["log_return"].values)

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
        print("\nNo event dataset available — change point reported without event context.")

    plot_tau_posterior(trace)
    plot_change_point(df, result["change_point_index"])
    plot_trace_diagnostics(trace)
    probabilistic = plot_parameter_posteriors(trace)
    local_impact = quantify_local_impact(df, result["change_point_index"])

    save_summary(result, matched_event, local_impact, probabilistic)

    print("\nChange point analysis complete!")


if __name__ == "__main__":
    main()
