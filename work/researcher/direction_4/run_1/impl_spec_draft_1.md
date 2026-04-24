# Implementation Specification: Dual-Mode Volume Forecast (Raw + Percentage)

## Overview

This direction implements a two-model system for intraday volume forecasting based on Satish, Saxena, and Palmer (2014). The system comprises:

1. A **raw volume forecast model** that combines four components -- a rolling historical average, an inter-day ARMA model, a deseasonalized intraday ARMA model, and a dynamic regime-switching weight overlay -- to produce bin-level volume predictions for all remaining bins in a trading day.

2. A **volume percentage forecast model** that extends the dynamic VWAP framework of Humphery-Jenner (2011), using volume surprises (deviations from a baseline forecast) in a rolling regression to adjust next-bin participation rates, with safety constraints.

The raw volume model produces full-day forecasts for scheduling tools. The volume percentage model produces next-bin-only forecasts for step-by-step VWAP execution. The two models are coupled: the raw model provides the surprise signal that drives the percentage model's adjustments.

Reported performance: 24% median MAPE reduction for raw volume and 9.1% VWAP tracking error reduction in simulation on top 500 U.S. stocks (Satish et al. 2014, Exhibits 6 and 10).

## Algorithm

### Model Description

The system operates on fixed-length intraday bins (default: 15-minute bins, I = 26 bins per 6.5-hour U.S. equity trading day, from 9:30 to 16:00 ET). Each bin is indexed by i in {1, ..., I}.

**Model A (Raw Volume Forecast)** takes as input historical bin-level volume data for a single stock and produces, at each bin during the trading day, forecasts for all remaining bins. It assumes that intraday volume has (a) a slowly-varying daily level, (b) a stable seasonal intraday shape, and (c) short-lived intraday dynamics. These three aspects are captured by three forecast components combined through a weighted overlay with regime-dependent weights.

**Model B (Volume Percentage Forecast)** takes as input the cumulative volume observed so far today and a baseline volume percentage curve (derived from historical averages), and produces a forecast of the next bin's volume percentage. It adjusts the baseline using a rolling regression on volume surprises computed from Model A. Safety constraints prevent excessive deviation from the baseline.

### Pseudocode

#### Model A: Raw Volume Forecast

```
INPUTS:
  V[s, t, i]       -- historical volume for stock s, day t, bin i
  I                 -- number of bins per day (default 26 for 15-min)
  N_hist            -- rolling historical window in days (default 21)
  N_arma_intraday   -- rolling intraday ARMA fitting window in days (default 21)
  N_seasonal        -- trailing deseasonalization window in days (default 126, ~6 months)
  max_p, max_q      -- maximum ARMA orders to search (default 5 each)

OUTPUT:
  V_hat[s, t, i]    -- forecast volume for stock s, day t, bin i

PROCEDURE for stock s, day t, bin i:

  # ---- Component 1: Rolling Historical Average ----
  # Satish et al. 2014, Section "Raw Volume Forecast Methodology", p.17
  # Exhibit 1 shows "Prior 21 days" feeding into Historical Window
  C1[i] = (1 / N_hist) * SUM(V[s, t-d, i] for d in 1..N_hist)

  # ---- Component 2: Inter-day ARMA ----
  # Satish et al. 2014, Section "Raw Volume Forecast Methodology", p.17
  # Per-symbol, per-bin ARMA(p,q) fitted to the daily volume series for bin i
  # Exhibit 1 shows "Prior 5 days" feeding ARMA Daily
  #
  # For each bin i, form the time series: {V[s, t-D, i], V[s, t-D+1, i], ..., V[s, t-1, i]}
  # where D is the training window length (e.g., 250 days or more for model selection)
  
  daily_series_i = [V[s, d, i] for d in training_window]
  
  # Select ARMA(p,q) order via AICc (Hurvich and Tsai, 1989/1993)
  best_p, best_q = argmin over p in {0,...,max_p}, q in {0,...,max_q}:
      AICc(fit_ARMA(daily_series_i, p, q, include_constant=True))
  
  # AICc = AIC + 2*k*(k+1) / (n-k-1), where k = p+q+1 (including constant)
  # Satish et al. 2014, p.17: "We depart from the standard technique in using
  # the corrected AIC, symbolized by AICc, as detailed by Hurvich and Tsai
  # [1989, 1993]. AICc adds a penalty term to AIC for extra AR and MA terms."
  
  model_interday_i = fit_ARMA(daily_series_i, best_p, best_q, include_constant=True)
  C2[i] = forecast(model_interday_i, steps_ahead=1)

  # ---- Component 3: Intraday ARMA (deseasonalized) ----
  # Satish et al. 2014, Section "Raw Volume Forecast Methodology", pp.17-18
  # Exhibit 1 shows "Current Bin" and "4 Bins Prior to Current Bin" feeding ARMA Intraday
  
  # Step 3a: Compute seasonal factors from trailing N_seasonal days
  S[i] = (1 / N_seasonal) * SUM(V[s, t-d, i] for d in 1..N_seasonal)
  # S[i] is the average volume for bin i over the trailing ~6 months
  
  # Step 3b: Deseasonalize current day's observed bins
  # For each observed bin j on day t (j = 1, ..., current_bin):
  V_deseas[s, t, j] = V[s, t, j] / S[j]
  
  # Step 3c: Fit intraday ARMA to deseasonalized series
  # Use rolling 1-month window of intraday deseasonalized data
  # Concatenate intraday deseasonalized observations from recent N_arma_intraday days
  # plus today's observed bins
  intraday_series = []
  for d in (t - N_arma_intraday) .. (t - 1):
      for j in 1..I:
          intraday_series.append(V[s, d, j] / S[j])
  for j in 1..current_bin:
      intraday_series.append(V_deseas[s, t, j])
  
  # Select ARMA order via AICc; effective lags kept below 5
  # Satish et al. 2014, p.18: "AR coefficients quickly decayed, so that we
  # used AR lags with a value less than five. As a result, we fit each
  # symbol with a dual ARMA model having fewer than 11 terms."
  best_p_intra, best_q_intra = argmin over p in {0,...,4}, q in {0,...,4}:
      AICc(fit_ARMA(intraday_series, p, q, include_constant=True))
      subject to: p + q + 1 + best_p + best_q + 1 < 11  # total terms < 11
  
  model_intraday = fit_ARMA(intraday_series, best_p_intra, best_q_intra, include_constant=True)
  
  # Forecast remaining bins in deseasonalized space, then re-seasonalize
  for future_bin in (current_bin + 1)..I:
      steps = future_bin - current_bin
      C3[future_bin] = forecast(model_intraday, steps_ahead=steps) * S[future_bin]

  # ---- Component 4: Dynamic Weight Overlay with Regime Switching ----
  # Satish et al. 2014, Section "Raw Volume Forecast Methodology", p.18
  # "a dynamic weight overlay on top of these three components ... that minimizes
  # the error on in-sample data. We incorporate a notion of regime switching by
  # training several weight models for different historical volume percentile
  # cutoffs."
  
  # Step 4a: Determine current regime
  # Compute cumulative volume observed today
  cum_vol_today = SUM(V[s, t, j] for j in 1..current_bin)
  
  # Compute historical percentile of cum_vol_today at this point in the day
  historical_cum_vols = [SUM(V[s, t-d, j] for j in 1..current_bin)
                         for d in 1..N_hist]
  percentile = percentile_rank(cum_vol_today, historical_cum_vols)
  
  # Map percentile to regime bucket
  # [Researcher inference: specific bucket boundaries not disclosed;
  #  recommend 3 regimes: low (0-33rd pctile), medium (33-67th), high (67-100th)]
  regime = classify_regime(percentile, bucket_boundaries)
  
  # Step 4b: Apply regime-specific weights
  # Weights w1[regime], w2[regime], w3[regime] are pre-trained in-sample
  # by minimizing forecast error for each regime
  V_hat[s, t, i] = w1[regime] * C1[i] + w2[regime] * C2[i] + w3[regime] * C3[i]
```

#### Model B: Volume Percentage Forecast

```
INPUTS:
  V[s, t, i]           -- historical volume for stock s, day t, bin i
  V_hat_raw[s, t, i]   -- raw volume forecast from Model A for stock s, day t, bin i
  I                     -- number of bins per day
  N_pct_hist            -- historical window for percentage baseline
  max_deviation         -- maximum allowed deviation from historical curve (default 0.10)
  switchoff_threshold   -- cumulative volume fraction to switch off (default 0.80)
  K_regression          -- number of regression terms (lags of surprise)

OUTPUT:
  pct_hat[s, t, i]     -- forecast volume percentage for stock s, day t, next bin i

PROCEDURE for stock s, day t, after observing bin (i-1):

  # ---- Step 1: Compute historical volume percentage curve ----
  # Satish et al. 2014, Section "Volume Percentage Forecast Methodology", pp.18-19
  # This is the baseline curve, computed the same way as the rolling mean
  # but normalized to sum to 1.
  
  for j in 1..I:
      hist_vol[j] = (1 / N_pct_hist) * SUM(V[s, t-d, j] for d in 1..N_pct_hist)
  total_hist = SUM(hist_vol[j] for j in 1..I)
  pct_hist[j] = hist_vol[j] / total_hist   for j in 1..I

  # ---- Step 2: Compute volume surprises ----
  # Satish et al. 2014, Section "Volume Percentage Forecast Methodology", p.19
  # "volume surprises -- deviations from a naive historical forecast -- are
  # regressed in a rolling regression to adjust future participation rates"
  #
  # The raw volume model provides the base forecast for computing surprises.
  # Surprise at bin j of day t:
  
  for j in 1..(i-1):
      actual_pct[j] = V[s, t, j] / V_total_est
      # where V_total_est is the estimated total daily volume
      # (from Model A's full-day forecast at market open or updated intraday)
      surprise[j] = actual_pct[j] - pct_hist[j]

  # ---- Step 3: Rolling regression to adjust participation ----
  # Satish et al. 2014, Section "Volume Percentage Forecast Methodology", p.19
  # Based on Humphery-Jenner (2011) dynamic VWAP framework.
  # Regress volume surprise on K lagged surprises.
  # [Researcher inference: exact regression specification not fully disclosed;
  #  recommend OLS regression of surprise[j] on {surprise[j-1], ..., surprise[j-K]}
  #  with K selected in-sample, no intercept]
  
  # Build regression dataset from recent history
  # For each recent day d and bin j:
  #   y = surprise[d, j]
  #   X = [surprise[d, j-1], surprise[d, j-2], ..., surprise[d, j-K]]
  
  # Fit rolling regression (OLS)
  beta = fit_rolling_regression(surprise_history, K=K_regression)
  
  # Predict next-bin surprise adjustment
  surprise_hat = beta[1]*surprise[i-1] + beta[2]*surprise[i-2] + ... + beta[K]*surprise[i-K]
  
  # ---- Step 4: Compute adjusted percentage forecast ----
  pct_raw = pct_hist[i] + surprise_hat

  # ---- Step 5: Apply safety constraints ----
  # Satish et al. 2014, Section "Volume Percentage Forecast Methodology", p.19
  # Constraint 1: deviation limit from historical VWAP curve
  # "depart no more than 10% away from a historical VWAP curve"
  # Humphery-Jenner (2011)
  
  deviation = pct_raw - pct_hist[i]
  if abs(deviation) > max_deviation * pct_hist[i]:
      pct_raw = pct_hist[i] + sign(deviation) * max_deviation * pct_hist[i]
  
  # Constraint 2: switch-off once cumulative fraction exceeds threshold
  # "once 80% of the day's volume is reached, return to historical curve"
  # Humphery-Jenner (2011)
  cum_pct = SUM(actual_pct[j] for j in 1..(i-1))
  if cum_pct >= switchoff_threshold:
      pct_raw = pct_hist[i]
  
  # ---- Step 6: Renormalize remaining bins ----
  # Ensure remaining percentage forecasts sum to (1 - cum_pct)
  remaining_pct = 1.0 - cum_pct
  remaining_hist = SUM(pct_hist[j] for j in i..I)
  pct_hat[s, t, i] = pct_raw * (remaining_pct / remaining_hist)
```

#### Weight Calibration Procedure (for Model A Component 4)

```
INPUTS:
  V[s, t, i]          -- historical volume data (in-sample period)
  C1[t, i], C2[t, i], C3[t, i]  -- component forecasts computed in-sample
  bucket_boundaries    -- regime percentile cutoffs to evaluate

OUTPUT:
  w1[r], w2[r], w3[r] -- optimal weights per regime r

PROCEDURE:

  # Satish et al. 2014, Section "Raw Volume Forecast Methodology", p.18
  # "a dynamic weight overlay ... that minimizes the error on in-sample data"
  
  # Step 1: For each day t and bin i in in-sample, compute:
  #   - cumulative volume percentile (for regime classification)
  #   - component forecast errors
  
  # Step 2: Group observations by regime
  for each regime r in {low, medium, high}:
      observations_r = {(t, i) : regime_of(t, i) == r}
      
      # Step 3: Solve weighted least squares
      # min_{w1, w2, w3} SUM over (t,i) in observations_r:
      #   (V[s, t, i] - w1*C1[t,i] - w2*C2[t,i] - w3*C3[t,i])^2
      #
      # [Researcher inference: paper does not specify whether weights are
      #  constrained (e.g., sum to 1 or non-negative). Recommend unconstrained
      #  OLS as default, with optional non-negativity constraint if out-of-sample
      #  performance degrades.]
      
      w1[r], w2[r], w3[r] = solve_OLS(V, C1, C2, C3, observations_r)
```

### Data Flow

```
Input Data:
  Raw volume time series V[stock, day, bin]
    - Shape: (S stocks) x (T days) x (I bins)
    - Type: float64 (shares traded per bin)
    - Source: NYSE TAQ or equivalent tick data, aggregated to bin intervals

Preprocessing:
  1. Aggregate tick data to fixed bins (15-min default)
     - Input:  raw trades with timestamps and sizes
     - Output: V[s, t, i] as float64, shape (S, T, I)
  
  2. Compute seasonal factors S[s, i]
     - Input:  V[s, t-1..t-N_seasonal, i]
     - Output: S[s, i] as float64, shape (S, I)
     - Transform: arithmetic mean over trailing N_seasonal days per bin

Model A Pipeline:
  Component 1: V[s, t-N_hist..t-1, i] -> mean -> C1[i]
    - Shape: scalar per bin
  
  Component 2: V[s, training_window, i] -> ARMA(p,q) -> forecast -> C2[i]
    - Shape: scalar per bin
    - ARMA orders stored as (p_i, q_i) per stock per bin
  
  Component 3: V[s, recent, 1..I] / S[j] -> ARMA(p,q) -> forecast -> reseasonalize -> C3[i]
    - Shape: scalar per bin
    - Single ARMA model across all bins (deseasonalized)
  
  Combination: (C1[i], C2[i], C3[i]) -> weighted sum with regime weights -> V_hat[i]
    - Shape: scalar per bin
    - Regime selected by current cumulative volume percentile

Model B Pipeline:
  Historical curve: V[s, t-N..t-1, 1..I] -> normalize -> pct_hist[1..I]
    - Shape: (I,) float64, sums to 1.0
  
  Surprises: (V[s, t, 1..j] / V_total_est) - pct_hist[1..j] -> surprise[1..j]
    - Shape: (j,) float64
  
  Rolling regression: surprise_history -> OLS -> beta[1..K]
    - Shape: (K,) float64
  
  Adjusted forecast: pct_hist[i] + beta . surprise_lags -> clamp -> renormalize -> pct_hat[i]
    - Shape: scalar, in [0, 1]

Output:
  Model A: V_hat[s, t, i] for i in {current_bin+1, ..., I}
    - Shape: (I - current_bin,) float64
    - Units: shares
  
  Model B: pct_hat[s, t, i] for i = current_bin + 1 (next bin only)
    - Shape: scalar float64
    - Units: fraction of daily volume (0 to 1)
```

### Variants

This implementation follows the single variant described in Satish et al. (2014). The paper does not describe multiple model variants; rather, the four components of Model A and the dynamic VWAP extension in Model B represent a single, integrated system.

The comparison paper Chen, Feng, and Palomar (2016) offers an alternative approach (Kalman filter state-space model) for the same forecasting problem. We implement the Satish et al. approach because:
1. It is the foundational paper for this direction.
2. The practitioner-oriented design is directly usable in production VWAP execution.
3. The two-model architecture (raw + percentage) serves distinct use cases that a single model cannot.

Chen et al. (2016) results serve as comparison benchmarks for validation (see Validation section).

## Parameters

| Parameter | Description | Recommended Value | Sensitivity | Range |
|-----------|-------------|-------------------|-------------|-------|
| I | Number of bins per trading day | 26 (15-min bins for 6.5h session) | Low (structural) | 13 (30-min), 26 (15-min), 78 (5-min) |
| N_hist | Rolling historical window for Component 1 (days) | 21 | Medium | 10-60 |
| N_seasonal | Trailing window for deseasonalization (days) | 126 (~6 months) | Low-Medium | 63-252 |
| N_arma_intraday | Rolling window for intraday ARMA fitting (days) | 21 (~1 month) | Medium | 10-42 |
| max_p | Maximum AR order for ARMA search | 5 | Low | 3-7 |
| max_q | Maximum MA order for ARMA search | 5 | Low | 3-7 |
| max_p_intra | Maximum AR order for intraday ARMA | 4 | Medium | 2-5 |
| max_q_intra | Maximum MA order for intraday ARMA | 4 | Medium | 2-5 |
| n_regimes | Number of regime buckets for weight overlay | 3 | Medium-High | 2-5 |
| regime_boundaries | Percentile cutoffs for regime classification | [33, 67] | High | Grid search required |
| N_pct_hist | Historical window for percentage baseline (days) | 21 | Medium | 10-60 |
| K_regression | Number of lagged surprise terms in percentage regression | 3 | Medium-High | 1-5 |
| max_deviation | Maximum fractional deviation from historical curve | 0.10 | Medium | 0.05-0.20 |
| switchoff_threshold | Cumulative volume fraction to switch off dynamic adjustments | 0.80 | Low-Medium | 0.70-0.90 |

### Parameter Sources and Notes

- **I = 26**: Satish et al. 2014, p.16: "The work in this article is based on 15-minute bins, and there are 26 such bins in a trading day." Also tested with 5-min (78 bins) and 30-min (13 bins).

- **N_hist = 21**: Researcher inference. The paper does not disclose the exact value. Exhibit 1 shows "Prior 21 days" feeding into the Historical Window component. This is the best available evidence for this parameter.

- **N_seasonal = 126**: Satish et al. 2014, p.17: "The intraday data are deseasonalized by dividing by the intraday amount of volume traded in that bin over the trailing six months."

- **N_arma_intraday = 21**: Satish et al. 2014, p.18: "We compute this model on a rolling basis over the most recent month."

- **max_p, max_q = 5**: Satish et al. 2014, p.17: "In fitting the ARMA model, we consider all values of p and q lags through five."

- **max_p_intra, max_q_intra = 4**: Satish et al. 2014, p.18: "we used AR lags with a value less than five." Constraint: combined inter-day + intraday ARMA has fewer than 11 terms total.

- **n_regimes, regime_boundaries**: Researcher inference. Satish et al. 2014, p.18: "training several weight models for different historical volume percentile cutoffs." Specific values are proprietary. Three regimes with boundaries at 33rd and 67th percentiles is a reasonable starting point; grid search over {2,3,4,5} regimes and percentile boundaries is recommended.

- **max_deviation = 0.10**: Satish et al. 2014, p.19 / Humphery-Jenner (2011): "depart no more than 10% away from a historical VWAP curve."

- **switchoff_threshold = 0.80**: Satish et al. 2014, p.19 / Humphery-Jenner (2011): "once 80% of the day's volume is reached, return to a historical approach."

- **K_regression**: Researcher inference. The paper does not disclose the optimal number of regression terms for U.S. equities. Humphery-Jenner (2011) used a rolling regression on surprise lags. K=3 is a reasonable default; the paper states this was identified through in-sample optimization.

### Initialization

1. **Historical data requirement**: Minimum N_seasonal (126) + N_hist (21) = 147 trading days of history before the first forecast can be produced. In practice, 250+ days recommended for stable ARMA estimation.

2. **ARMA model initialization**: Use standard MLE initialization (conditional or exact MLE via statsmodels or equivalent). Per Satish et al. 2014, p.17: "We use nearly standard ARMA model-fitting techniques relying on maximum-likelihood estimation."

3. **Weight initialization**: Before in-sample calibration, initialize all weights to equal: w1 = w2 = w3 = 1/3 for each regime.

4. **Seasonal factor initialization**: Compute S[i] from the first N_seasonal days of available data. If fewer days are available, use all available days (minimum ~21 days for any stability).

5. **First-day-of-trading behavior**: On the first bin of the day (i=1), no intraday observations are available. Component 3 uses only prior-day intraday data. The percentage model defaults to the historical curve for the first bin.

### Calibration

The calibration procedure operates in two phases:

**Phase 1: Component Model Fitting (daily rolling)**
1. For each stock s and each bin i:
   a. Extract the daily volume series for bin i: {V[s, 1, i], ..., V[s, T, i]}
   b. Fit ARMA(p,q) with all orders p,q in {0,...,5} with constant term
   c. Select (p,q) by minimum AICc
   d. Store fitted model for one-step-ahead forecasting
   
2. For each stock s:
   a. Compute seasonal factors S[i] for each bin
   b. Build deseasonalized intraday series over rolling 1-month window
   c. Fit intraday ARMA(p,q) with orders p,q in {0,...,4}, AICc selection
   d. Verify combined inter-day + intraday terms < 11

**Phase 2: Weight Calibration (in-sample optimization)**
1. Using in-sample period (e.g., first year of two-year dataset):
   a. Generate component forecasts C1, C2, C3 for all in-sample bins
   b. Classify each (day, bin) observation into a regime based on cumulative volume percentile
   c. For each candidate regime configuration (number of buckets, bucket boundaries):
      - Solve OLS for weights within each regime
      - Compute in-sample MAPE
   d. Select regime configuration with lowest in-sample MAPE
   
2. For Model B:
   a. Using same in-sample period, compute surprises based on Model A forecasts
   b. Grid search over K_regression in {1, 2, 3, 4, 5}
   c. Select K with lowest in-sample absolute percentage error

**Phase 3: Validation**
- Apply calibrated models to out-of-sample period (e.g., second year)
- Evaluate using MAPE (raw volume) and mean absolute deviation (percentages)

**Re-calibration schedule**: Satish et al. do not specify explicit re-calibration frequency. Researcher inference: re-calibrate ARMA orders monthly (inter-day) and weekly (intraday). Re-calibrate regime weights quarterly. Seasonal factors update daily by construction (trailing window).

## Validation

### Expected Behavior

**Raw volume forecast (Model A):**
- Median MAPE reduction vs. historical rolling mean: ~24% across all intraday bins (Satish et al. 2014, p.20, Exhibit 6).
- Bottom 95% mean MAPE reduction: ~29% (Satish et al. 2014, p.20).
- MAPE reduction should be consistent across industry groups (SIC 2-digit codes) and beta deciles (Satish et al. 2014, Exhibits 7 and 8).
- Error reduction is largest in the 12:00-15:00 period (midday, where historical averages are weakest) and smallest near market open and close (Satish et al. 2014, Exhibit 6).

**Volume percentage forecast (Model B):**
- Median absolute error: ~0.00808 (dynamic) vs. ~0.00874 (historical) for 15-min bins, a 7.55% reduction (Satish et al. 2014, Exhibit 9).
- Bottom 95% average absolute error: ~0.00924 (dynamic) vs. ~0.00986 (historical), a 6.29% reduction (Satish et al. 2014, Exhibit 9).
- Smaller but significant improvements for 5-min bins (2.25%) and 30-min bins (2.95%) (Satish et al. 2014, Exhibit 9).

**VWAP tracking error:**
- Dynamic VWAP curve should reduce VWAP tracking error by ~9.1% relative to historical curve (mean 8.74 bps vs. 9.62 bps) (Satish et al. 2014, Exhibit 10).
- Improvement should be 7%-10% across Dow 30, midcap, and high-variance stock groups.
- Regression of VWAP tracking error on volume percentage error: R^2 > 0.50, coefficient ~220.9 bps/unit for Dow 30 stocks, ~454.3 bps/unit for high-variance stocks (Satish et al. 2014, Exhibits 3 and 5).

**Comparison benchmark (Chen et al. 2016):**
- The Kalman filter model of Chen et al. achieves average MAPE of 0.46 (dynamic) and 0.61 (static) across 30 securities, and VWAP tracking error of 6.38 bps (dynamic) (Chen et al. 2016, Table 3 and Table 4). The Satish et al. model should produce comparable or better results on similar data, particularly for U.S. liquid equities.
- Note: direct comparison is imperfect because the datasets differ (500 U.S. stocks vs. 30 multi-market securities) and the time periods differ.

### Sanity Checks

1. **Component 1 (historical average) baseline**: Compute MAPE using only C1 as the forecast. This should match the "historical window" baseline reported in the paper (the denominator for all improvement calculations).

2. **Seasonal factors**: The seasonal factor vector S[1..I] should exhibit the well-known U-shaped pattern: high values at market open (bins 1-2) and close (bins 25-26), low values in the midday period (bins 10-16).

3. **Deseasonalized volume**: After dividing by S[i], the deseasonalized bin volumes should have approximately constant mean across bins within a day. Verify: std(mean(V_deseas[i]) for i=1..I) should be small relative to the grand mean.

4. **ARMA order distribution**: Most stocks should select low-order ARMA models (p+q <= 4). If many stocks select p=5 or q=5 (boundary of search space), consider expanding the search range.

5. **Weight reasonableness**: Calibrated weights w1, w2, w3 for each regime should all be positive and of similar magnitude (within the same order of magnitude). If any weight is large and negative, suspect overfitting.

6. **Percentage constraint verification**: The volume percentage forecasts from Model B should satisfy: (a) each pct_hat[i] is in [0, 1], (b) the sum of actual observed percentages plus remaining forecasted percentages is approximately 1.0.

7. **Regime classification stability**: For a given stock, the regime classification should not oscillate rapidly between bins within the same day. If it does, the regime boundaries may be too narrow.

8. **Surprise magnitude**: Volume surprises (actual_pct - hist_pct) should have mean approximately zero and standard deviation on the order of 0.005-0.015 for 15-minute bins on liquid stocks.

### Edge Cases

1. **Zero-volume bins**: Some bins may have zero volume (illiquid stocks, trading halts). This creates division-by-zero in MAPE and deseasonalization. Handle by: (a) excluding zero-volume bins from MAPE calculation, (b) using a floor value (e.g., 1 share) when computing seasonal factors, (c) flagging stocks with >5% zero-volume bins as potentially unsuitable for this model.

2. **First bin of the day (i=1)**: No intraday information is available yet. Component 3 must rely entirely on prior-day intraday data. Model B defaults to the historical percentage curve. This is the least informative forecast point.

3. **Last bin of the day (i=I)**: The closing auction can produce volume spikes. The historical average handles this implicitly. The intraday ARMA may not capture closing auction effects well. Consider using only Component 1 (historical average) for the last bin.

4. **Half-day trading sessions**: U.S. markets close early on certain days (e.g., day before Thanksgiving, Christmas Eve). The number of bins I changes. Handle by: (a) maintaining separate seasonal factors for half-day sessions, or (b) excluding half-day sessions from the model.

5. **Stock splits and corporate actions**: Volume series have structural breaks at splits. Adjust historical volume by split ratios before computing seasonal factors and fitting ARMA models. Alternatively, normalize volume by shares outstanding to compute turnover.

6. **Missing data**: If trading is halted intraday, some bins will have zero volume. Skip these bins in the ARMA fitting and percentage calculations. Resume normal forecasting after the halt ends.

7. **Extreme volume events**: Earnings releases, index rebalancing, option expiry can cause volume spikes >5x normal. Satish et al. 2014, p.18: "custom curves for special days (option expiry dates and Fed calendar events)" are recommended rather than ARMAX models. Implement a special-day detection module that overrides the standard forecast with day-type-specific historical averages.

8. **Regime boundary instability**: Near percentile boundaries (e.g., 33rd percentile), small changes in cumulative volume could cause regime switches between adjacent bins. Implement hysteresis: once a regime is selected, require the percentile to cross the boundary by a margin (e.g., 5 percentile points) before switching.

9. **ARMA convergence failure**: MLE optimization may fail to converge for some (stock, bin, order) combinations. Implement fallback: if convergence fails, reduce ARMA order by 1 (e.g., try ARMA(p-1, q) or ARMA(p, q-1)). If all orders fail, fall back to Component 1 (historical average) only.

### Known Limitations

1. **Proprietary parameters**: Many key parameter values are not disclosed in the paper (regime thresholds, weighting coefficients, optimal regression terms for U.S. equities). These must be rediscovered through grid search. The paper's reported performance may partly reflect these proprietary choices that a replicator may not exactly reproduce (Satish et al. 2014, pp.17-18).

2. **Static VWAP limitation**: Model A produces forecasts for all remaining bins, but these multi-step-ahead ARMA forecasts decay toward the unconditional mean. Satish et al. 2014 note this indirectly via Model B's design: "techniques that predict only the next interval will perform better than those attempting to predict volume percentages for the remainder of the trading day" (p.18). Dynamic (intraday-updated) forecasting is essential.

3. **No formal out-of-sample model selection**: The paper uses in-sample AICc for ARMA order selection and in-sample optimization for weights. There is no cross-validation or walk-forward validation described for the weight calibration. Overfitting risk exists, particularly for the regime-specific weights with small regime-specific sample sizes.

4. **Single-stock model**: Unlike the BDF approach (Direction 2), this model is fitted independently per stock. It does not exploit cross-sectional information. Stocks with short histories or unusual volume patterns may not benefit.

5. **No outlier handling**: Unlike Chen et al. (2016), which includes a robust Lasso extension for outlier detection, this model has no explicit mechanism for handling outliers in volume data. Extreme observations can distort ARMA parameter estimates and seasonal factors.

6. **Percentage model limitations**: The volume percentage model requires an estimate of total daily volume (V_total_est) to compute actual percentages during the day. This creates a circular dependency: the percentage model needs the raw volume model's daily total forecast, but that forecast is itself uncertain. Errors in V_total_est propagate into surprise calculations.

7. **Computational cost for large universes**: ARMA order selection via AICc requires fitting 36 models (6x6 grid) per stock per bin for the inter-day model, plus additional models for the intraday component. For 500 stocks x 26 bins, this is ~468,000 ARMA fits per calibration cycle. Parallelization and caching of results across days are recommended.

8. **Humphery-Jenner framework limitations**: The percentage model inherits limitations from Humphery-Jenner (2011): the deviation bound and switch-off threshold are hard constraints that may be suboptimal for all stocks. Adaptive thresholds (varying by stock volatility or liquidity) would be an extension.

## Paper References

| Specification Section | Paper Source | Section/Exhibit |
|----------------------|-------------|-----------------|
| Model A: Component 1 (Rolling Historical Average) | Satish et al. 2014 | p.17, "Historical Window Average/Rolling Means" |
| Model A: Component 1 window (21 days) | Satish et al. 2014 | Exhibit 1 diagram ("Prior 21 days") |
| Model A: Component 2 (Inter-day ARMA) | Satish et al. 2014 | p.17, "Raw Volume Forecast Methodology" |
| Model A: AICc order selection | Satish et al. 2014 | p.17, referencing Hurvich and Tsai (1989, 1993) |
| Model A: Component 3 (Intraday ARMA) | Satish et al. 2014 | pp.17-18, "Raw Volume Forecast Methodology" |
| Model A: Deseasonalization (6-month trailing) | Satish et al. 2014 | p.17, "The intraday data are deseasonalized..." |
| Model A: Intraday ARMA rolling window (1 month) | Satish et al. 2014 | p.18, "on a rolling basis over the most recent month" |
| Model A: AR lags < 5, total terms < 11 | Satish et al. 2014 | p.18, "AR lags with a value less than five... fewer than 11 terms" |
| Model A: Component 4 (Dynamic Weight Overlay) | Satish et al. 2014 | p.18, "a dynamic weight overlay..." |
| Model A: Regime switching | Satish et al. 2014 | p.18, "regime switching by training several weight models..." |
| Model A: Regime bucket count/boundaries | Researcher inference | Not disclosed; 3 regimes at [33, 67] pctile recommended |
| Model A: Weight constraints | Researcher inference | Not disclosed; unconstrained OLS recommended |
| Model A: Architecture diagram | Satish et al. 2014 | Exhibit 1 |
| Model B: Dynamic VWAP framework | Satish et al. 2014 / Humphery-Jenner 2011 | p.18-19 |
| Model B: 10% deviation limit | Humphery-Jenner 2011 | Referenced in Satish et al. 2014, p.19 |
| Model B: 80% switch-off threshold | Humphery-Jenner 2011 | Referenced in Satish et al. 2014, p.19 |
| Model B: Optimal regression terms for U.S. equities | Researcher inference | Not disclosed; K=3 recommended |
| Model B: Surprise computation using raw model | Satish et al. 2014 | p.19, "we could apply our more sophisticated volume forecasting model... to compute volume surprises" |
| Bin size: 15 minutes | Satish et al. 2014 | p.16 |
| Special-day custom curves | Satish et al. 2014 | p.18, "custom curves for special days" |
| MAPE metric definition | Satish et al. 2014 | p.17, "Measuring Raw Volume Predictions -- MAPE" |
| Percentage error metric definition | Satish et al. 2014 | p.17, "Measuring Percentage Volume Predictions -- Absolute Deviation" |
| VWAP tracking error definition | Satish et al. 2014 | p.16, "VWAP Tracking Error" |
| Raw volume results: 24% median MAPE reduction | Satish et al. 2014 | p.20, Exhibit 6 |
| Percentage results: 7.55% median reduction (15-min) | Satish et al. 2014 | p.21, Exhibit 9 |
| VWAP simulation: 9.1% reduction | Satish et al. 2014 | p.23, Exhibit 10 |
| VWAP regression: R^2=0.51, coef=220.9 (Dow 30) | Satish et al. 2014 | p.20, Exhibit 3 |
| VWAP regression: R^2=0.59, coef=454.3 (high-variance) | Satish et al. 2014 | p.21, Exhibit 5 |
| Comparison: Kalman filter MAPE 0.46 (dynamic) | Chen et al. 2016 | Table 3 (average across 30 securities) |
| Comparison: Kalman filter VWAP 6.38 bps | Chen et al. 2016 | Table 4 (average across 30 securities) |
