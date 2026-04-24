# Predicting Intraday Trading Volume and Volume Percentages
**Authors:** Venkatesh Satish, Abhay Saxena, Max Palmer
**Year:** 2014
**DOI:** N/A (Published in The Journal of Trading, Summer 2014)

## Core Contribution

This paper presents two separate forecasting models developed at FlexTrade Systems:
one for predicting intraday raw volume and one for predicting intraday volume
percentages, both operating over 15-minute bins. The raw volume model combines a
rolling historical average with dual ARMA components (inter-day and intraday) and
a dynamic regime-switching weight overlay. The volume percentage model extends the
"dynamic VWAP" framework of Humphery-Jenner (2011), refining it through in-sample
optimization on U.S. equity data.

The paper also establishes — through simulation — that reducing volume percentage
forecast error leads to a statistically significant reduction in realized VWAP
tracking error. This validation is notable because it connects the modeling problem
directly to a measurable execution quality metric, providing a practical benchmark
for evaluating volume forecasting improvements.

## Model/Algorithm Description

### Raw Volume Forecast Model (four components)

1. Rolling historical average — the classical baseline: arithmetic mean of volume
   in a given 15-minute bin over the prior N days (called Historical Window Average
   or Rolling Mean).

2. Inter-day ARMA — a per-symbol, per-bin ARMA(p, q) model fitted to daily volume
   series, capturing serial correlation in total daily volume. Model selection uses
   the corrected AIC (AICc, Hurvich and Tsai 1989/1993) to penalize overfitting in
   small samples. All p, q in {0,...,5} and a constant term are considered.

3. Intraday ARMA — a per-symbol ARMA(p, q) fitted to deseasonalized intraday bin
   volumes. Deseasonalization divides each bin's volume by the trailing six-month
   average volume for that bin. AR coefficients decay quickly, so lags are kept
   below five; the combined dual ARMA model has fewer than 11 terms total. Fitted
   on a rolling one-month window. Forecasts are re-seasonalized (multiplied back by
   the seasonal factor) before use.

4. Dynamic weight overlay — combines the three components above via weights
   optimized in-sample. Regime switching is incorporated by training separate
   weight sets for different historical volume percentile cutoffs; during the
   out-of-sample period the appropriate weight set is selected intraday based on
   the historical percentile of observed cumulative volume.

### Volume Percentage Forecast Model

Built on Humphery-Jenner's (2011) dynamic VWAP approach. The core idea is that
volume surprises — deviations from a naive historical forecast — are regressed in a
rolling regression to adjust future participation rates. Extensions developed in
this paper include: identifying the optimal number of model terms for U.S. equities,
a separate method for computing deviation bounds, and using the more sophisticated
raw volume model to compute the base volume surprises. Safety constraints (deviation
limits from historical VWAP curve, switch-off once 80% of day's volume is reached)
are preserved from the original formulation.

## Key Parameters

- Bin size: 15 minutes (26 bins per trading day); 5- and 30-minute bins also tested.
- Historical window (N): number of prior days for the rolling mean baseline
  (treated as a tunable parameter; exact value not disclosed).
- ARMA orders (p, q): selected via AICc; p, q ∈ {0,...,5} considered; effective
  lags in intraday model kept below 5.
- Intraday deseasonalization window: trailing six months.
- Intraday ARMA fitting window: rolling one month.
- Regime-switch thresholds: defined by historical volume percentile cutoffs
  (specific values not disclosed).
- Volume percentage deviation limit: no more than 10% departure from historical
  VWAP curve (inherited from Humphery-Jenner).
- Dynamic VWAP switch-off: revert to historical curve once 80% of the day's volume
  is reached.

## Data Requirements

- Universe: top 500 U.S. stocks by dollar volume (NYSE TAQ data).
- Period: two years of TAQ data; out-of-sample results reported on the final year
  (~250 trading days).
- VWAP simulation: 600+ day-long VWAP orders across Dow 30 components, midcap
  names (second 1,000 by dollar volume), and 30 highest intraday volume-variance
  stocks from the Russell 3000.
- Order size in simulation: 10% of 30-day average daily volume (ADV).
- Validation of percentage-error-to-VWAP-error relationship: May 2011 data for
  Dow 30 and 30 high-volume-variance Russell 3000 names, run through the
  FlexTRADER tick simulator.

## Results

### Raw Volume Forecasting

- Median MAPE reduction vs. historical window baseline: 24% across all intraday
  intervals.
- Bottom-95% mean MAPE reduction: 29% across all intraday intervals.
- Improvements are consistent across industry groups (SIC two-digit codes) and
  beta deciles, validating breadth of the gain.

### Volume Percentage Forecasting (15-minute bins)

- Median absolute error: 0.00874 (historical) vs. 0.00808 (dynamic) — 7.55%
  reduction, significant at << 1% level (Wilcoxon signed-rank test).
- Bottom-95% average absolute error: 0.00986 (historical) vs. 0.00924 (dynamic)
  — 6.29% reduction.
- 5- and 30-minute bins show smaller but statistically significant improvements
  (2.25% and 2.95% median reduction, respectively).

### VWAP Tracking Error

- Regression of VWAP tracking error on absolute volume percentage error: R^2 > 50%
  for both Dow 30 (R^2 = 0.51, coefficient = 220.9 bps per unit error) and
  high-variance names (R^2 = 0.59, coefficient = 454.3 bps per unit error).
- Simulation result: mean VWAP tracking error reduced from 9.62 bps (historical
  curve) to 8.74 bps (dynamic curve) — a 9.1% reduction, significant at p < 0.01
  via paired t-test.
- Per-category VWAP reductions: 7%–10% across Dow 30, midcap, and high-variance
  groups.

## Relationships

This paper builds directly on three prior works:

- Bialkowski, Darolles, Le Fol (2008): first study focused on VWAP tracking error
  via volume turnover forecasting (ARMA and SETAR on CAC 40 data). This paper
  surpasses their 4.8%–7% VWAP error reduction with a 9.1% reduction and on a
  larger dataset.
- Brownlees, Cipollini, Gallo (2011): component memory error model for intraday
  volume on DIA/QQQ/SPY. They achieved 6.5% VWAP tracking error reduction and
  12.7% volume MSE reduction. The dual ARMA approach here achieves comparable
  VWAP improvement on a much broader universe (500 names vs. 3 ETFs).
- Humphery-Jenner (2011): dynamic VWAP volume percentage model on 200 ASX names,
  achieving 3.4%–4.8% volume percentage error reduction without extending to VWAP
  tracking error. This paper both extends and validates that work, optimizes it
  for U.S. equities, and demonstrates the VWAP tracking error link.

The paper also cites Pragma Trading (2009) on static VWAP approaches and Hurvich
and Tsai (1989, 1993) for the corrected AIC used in model selection.

## Implementation Notes

- Two entirely separate models are recommended: one for raw volume, one for volume
  percentages. The raw volume model must forecast all remaining bins simultaneously
  (needed by scheduling tools and participation models), while the percentage model
  only needs to forecast the next bin (used step-by-step by VWAP algorithms).
- Custom curves for special calendar days (option expiry, Fed events) are
  recommended rather than ARMAX models, due to insufficient historical occurrences
  to reliably estimate exogenous effects.
- Performance metrics: MAPE for raw volume; mean absolute deviation (no
  normalization) for volume percentages. Using consistent metrics across studies
  is emphasized — prior literature used different metrics, making direct comparison
  difficult.
- Regime switching (via volume percentile cutoffs) is a key design decision that
  allows the weight overlay to adapt to high- vs. low-volume days intraday.
- The paper does not disclose the specific values of the dynamic weighting
  coefficients, the exact number of regime buckets, or the optimal regression
  terms identified for U.S. equities in the dynamic VWAP extension — these are
  treated as proprietary. Replicators would need to rediscover these through
  in-sample grid search.
- VWAP algorithm simulation is recommended as the ultimate validation test;
  pure forecast error metrics alone may not capture all execution-quality benefits.
