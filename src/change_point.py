import os
os.environ["PYTENSOR_FLAGS"] = "mode=NUMBA"

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import json
import sys
import warnings
warnings.filterwarnings("ignore")

DRAWS = 1500
TUNE = 1000
CHAINS = 4

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


def load_data(path="data/processed/brent_clean.csv"):
    try:
        df = pd.read_csv(path, parse_dates=["Date"])
    except FileNotFoundError:
        print(f"ERROR: {path} not found. Run src/data_loader.py first.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: failed to load {path}: {e}")
        sys.exit(1)
    return df


def resample_monthly(df):
    """Resample to monthly log returns, computed correctly as the log price
    change over the month (log(price_end / previous price_end)) - NOT the
    average of daily log returns, which would dilute the true monthly signal
    by roughly the number of trading days in a month (~21x)."""
    monthly_price = df.set_index("Date")["Price"].resample("ME").last().dropna()
    monthly = pd.DataFrame({"Price": monthly_price})
    monthly["log_return"] = np.log(monthly["Price"]).diff()
    monthly = monthly.dropna().reset_index()
    print(f"Resampled to {len(monthly)} monthly points (from {len(df)} daily rows)")
    return monthly


def build_and_sample_model(returns):
    """Single change-point model: log returns are Normal(mu_1, sigma) before tau,
    Normal(mu_2, sigma) after tau. tau's posterior tells us the most probable
    location of a regime shift in average daily returns."""
    n = len(returns)
    try:
        with pm.Model() as model:
            tau = pm.DiscreteUniform("tau", lower=0, upper=n - 1)
            mu_1 = pm.Normal("mu_1", mu=0, sigma=0.1)
            mu_2 = pm.Normal("mu_2", mu=0, sigma=0.1)
            sigma = pm.HalfNormal("sigma", sigma=0.1)

            idx = np.arange(n)
            mu = pm.math.switch(tau >= idx, mu_1, mu_2)
            pm.Normal("obs", mu=mu, sigma=sigma, observed=returns)

            trace = pm.sample(
                draws=DRAWS, tune=TUNE, chains=CHAINS, cores=1,
                target_accept=0.95,
                return_inferencedata=True, progressbar=True
            )
        return trace
    except Exception as e:
        print(f"ERROR: PyMC sampling failed: {e}")
        sys.exit(1)


def nearest_event(change_date, events=EVENTS, max_days=180):
    change_date = pd.to_datetime(change_date)
    best = None
    best_diff = None
    for date_str, label in events:
        d = pd.to_datetime(date_str)
        diff = abs((change_date - d).days)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best = (date_str, label, diff)
    if best and best[2] <= max_days:
        return {"event": best[1], "event_date": best[0], "days_apart": int(best[2])}
    return {"event": None, "event_date": None, "days_apart": int(best[2]) if best else None}


def analyze_results(trace, df):
    try:
        tau_samples = trace.posterior["tau"].values.flatten()
        mu_1_samples = trace.posterior["mu_1"].values.flatten()
        mu_2_samples = trace.posterior["mu_2"].values.flatten()

        tau_mode = int(pd.Series(tau_samples).mode()[0])
        change_date = df.iloc[tau_mode]["Date"]

        mu_1_mean = float(mu_1_samples.mean())
        mu_2_mean = float(mu_2_samples.mean())

        # Convert average daily log return to an approximate daily % change
        pct_1 = (np.exp(mu_1_mean) - 1) * 100
        pct_2 = (np.exp(mu_2_mean) - 1) * 100

        price_before = float(df.iloc[max(0, tau_mode - 3):tau_mode]["Price"].mean())
        price_after = float(df.iloc[tau_mode:tau_mode + 3]["Price"].mean())

        event = nearest_event(change_date)

        summary = {
            "change_point_index": tau_mode,
            "change_point_date": str(pd.to_datetime(change_date).date()),
            "avg_daily_log_return_before": round(mu_1_mean, 5),
            "avg_daily_log_return_after": round(mu_2_mean, 5),
            "avg_daily_pct_change_before": round(pct_1, 4),
            "avg_daily_pct_change_after": round(pct_2, 4),
            "avg_price_30d_before": round(price_before, 2),
            "avg_price_30d_after": round(price_after, 2),
            "matched_event": event,
        }
        return summary, tau_samples
    except Exception as e:
        print(f"ERROR: result analysis failed: {e}")
        return {}, None


def plot_change_point(df, summary, path="data/processed/change_point.png"):
    try:
        fig, ax = plt.subplots(figsize=(16, 7))
        ax.plot(df.Date, df.Price, color="#2980b9", linewidth=0.8)
        change_date = pd.to_datetime(summary["change_point_date"])
        ax.axvline(change_date, color="red", linewidth=2, label=f"Detected Change Point ({summary['change_point_date']})")
        if summary["matched_event"]["event"]:
            ax.set_title(
                f"Brent Oil Price with Detected Change Point\nMatched Event: {summary['matched_event']['event']} "
                f"({summary['matched_event']['days_apart']} days apart)",
                fontweight="bold"
            )
        else:
            ax.set_title("Brent Oil Price with Detected Change Point", fontweight="bold")
        ax.set_ylabel("Price (USD/barrel)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to {path}")
    except Exception as e:
        print(f"WARNING: could not save change point plot: {e}")


def plot_posterior(trace, df, path="data/processed/tau_posterior.png"):
    try:
        tau_samples = trace.posterior["tau"].values.flatten()
        mu_1_samples = trace.posterior["mu_1"].values.flatten()
        mu_2_samples = trace.posterior["mu_2"].values.flatten()

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].hist(tau_samples, bins=40, color="#2980b9", edgecolor="white")
        axes[0].set_title("Posterior: Change Point Index (tau)", fontweight="bold")
        axes[1].hist(mu_1_samples, bins=40, color="#27ae60", edgecolor="white")
        axes[1].set_title("Posterior: Mean Return Before (mu_1)", fontweight="bold")
        axes[2].hist(mu_2_samples, bins=40, color="#e74c3c", edgecolor="white")
        axes[2].set_title("Posterior: Mean Return After (mu_2)", fontweight="bold")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to {path}")
    except Exception as e:
        print(f"WARNING: could not save posterior plot: {e}")


def save_summary(summary, path="data/processed/change_point_summary.json"):
    try:
        json.dump(summary, open(path, "w"), indent=2)
        print(f"Summary saved to {path}")
    except Exception as e:
        print(f"WARNING: could not save summary: {e}")


def main():
    df_daily = load_data()
    df = resample_monthly(df_daily)
    returns = df["log_return"].values
    print(f"Modeling on {len(returns)} monthly log returns")

    print("Building and sampling PyMC model (this may take a few minutes)...")
    trace = build_and_sample_model(returns)
    print("Sampling complete")

    summary, tau_samples = analyze_results(trace, df)
    print(json.dumps(summary, indent=2))

    save_summary(summary)
    plot_change_point(df, summary)
    plot_posterior(trace, df)
    print("Task 2 change point analysis complete!")


if __name__ == "__main__":
    main()