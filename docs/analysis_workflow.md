# Analysis Workflow — Brent Oil Price Change Point Analysis

**Project:** Birhan Energies — Impact of Geopolitical and Economic Events on Brent Oil Prices
**Data range:** 1987–2022 (daily Brent crude prices, 9,010 observations)

## 1. Data Loading & Cleaning
- Load raw Brent oil price series (`data/raw/BrentOilPrices.csv`).
- Parse dates, sort chronologically, check for duplicates and missing values.
- Compute log returns (`log(P_t / P_t-1)`) as the primary series for volatility/stationarity analysis.
- Save cleaned output to `data/processed/brent_clean.csv`.

## 2. Event Dataset Compilation
- Research and compile at least 10–15 major geopolitical, OPEC policy, and macroeconomic
  events likely to have influenced oil prices (wars, sanctions, OPEC decisions, financial
  crises, pandemics).
- Store as a structured table (`data/events.csv`) with columns: `date`, `event`, `category`,
  `description`.
- This dataset is used only to contextualize detected change points — it is not used to
  fit the model directly (see Assumptions & Limitations).

## 3. Exploratory Data Analysis (EDA)
- Plot raw price series with event markers overlaid.
- Plot log returns to visualize volatility clustering.
- Compute rolling mean and rolling standard deviation (90-day window) to inspect
  time-varying trend and volatility.
- Run Augmented Dickey-Fuller (ADF) tests on both price levels and log returns to assess
  stationarity.
- Document distributional properties (histograms of price and returns).

## 4. Modeling Choice Rationale
- Raw price series is non-stationary (ADF fails to reject unit root) → not directly
  suitable for standard time-series modeling without differencing/transformation.
- Log returns are stationary (ADF strongly rejects unit root) → appropriate series for
  detecting structural breaks in volatility/mean behavior.
- A **Bayesian change point model** (implemented in PyMC) is used to detect one or more
  points in time where the statistical properties (mean, volatility) of the log return
  series shift abruptly — indicating a structural break potentially associated with a
  major event.

## 5. Change Point Detection
- Fit a Bayesian change point model to the log return series.
- Extract the posterior distribution over the change point location (`tau`) and the
  before/after parameter estimates (mean, volatility).
- Identify the most probable change point date(s) from the posterior.

## 6. Insight Generation
- Match detected change point date(s) to the closest entries in the event dataset
  (nearest-date matching, reporting the time gap in days).
- Quantify the shift in average price and volatility before vs. after the change point.
- Frame findings as a plausible, evidence-based association — explicitly not a causal
  claim (see Assumptions & Limitations).

## 7. Reporting
- Summarize methodology, findings, and limitations in the interim/final report.
- Visualize change point location and posterior distribution alongside the price series.
