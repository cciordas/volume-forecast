# Implementation Specification: Dual-Mode Intraday Volume Forecast (Raw Volume + Volume Percentage)

## Overview

This specification describes a dual-model system for intraday volume forecasting, following Satish, Saxena, and Palmer (2014). **Model A** forecasts raw bin-level trading volume by combining three components -- a historical rolling average, an inter-day ARMA, and an intraday ARMA -- via a regime-switching dynamic weight overlay. **Model B** forecasts the next-bin volume percentage (fraction of daily volume), extending the "dynamic VWAP" framework of Humphery-Jenner (2011) with volume surprises derived from Model A. The two models serve different use cases: Model A provides absolute volume forecasts for all remaining bins (needed by scheduling tools, participation models, and portfolio trade allocation), while Model B provides a one-bin-ahead percentage forecast for VWAP execution algorithms.

The system targets U.S. equities using 15-minute bins (26 bins per 6.5-hour trading day, 9:30--16:00 ET). The paper reports a 24% median MAPE reduction for raw volume and a 9.1% VWAP tracking error reduction in simulation.

---

## Algorithm

### Model Description

**Model A (Raw Volume)** takes as input a symbol's historical intraday volume data and produces a forecast of raw share volume for each remaining 15-minute bin of the current trading day. It operates by:

1. Computing a rolling historical average of volume for each bin over recent days.
2. Fitting a per-bin inter-day ARMA model to capture day-to-day serial correlation in volume.
3. Fitting a single intraday ARMA model (per symbol) on deseasonalized bin-level volume within the day to capture within-day momentum.
4. Combining these three forecasts via regime-dependent weights that adapt based on where the current day's cumulative volume falls relative to its historical distribution.

**Model B (Volume Percentage)** takes as input the historical volume percentage profile and volume surprises computed from Model A, and produces a one-bin-ahead forecast of what fraction of the day's total volume will trade in the next bin. It adjusts a historical baseline using a rolling regression on recent volume surprises, subject to deviation limits and an end-of-day switch-off rule.

**Inputs:** Historical intraday volume data (shares traded per 15-minute bin) for a universe of symbols, with at least 1 year of history.

**Outputs:**
- Model A: Raw volume forecast V_hat[s, t, i] (shares) for symbol s, day t, bin i, for all remaining bins.
- Model B: Volume percentage forecast p_hat[t, next_bin] for the next bin.

### Pseudocode

#### Function 1: ComputeSeasonalFactors

Computes the trailing average volume per bin, used to deseasonalize intraday volume for the intraday ARMA.

```
function ComputeSeasonalFactors(symbol s, day t):
    # Reference: Satish et al. 2014, p.17, para 3
    # "dividing by the intraday amount of volume traded in that bin
    #  over the trailing six months"

    for i in 1..I:  # I = 26 bins
        volumes = [volume[s, d, i] for d in trailing N_seasonal trading days before t]
        seasonal_factor[i] = max(mean(volumes), epsilon)

    return seasonal_factor[1..I]
```

**Paper reference:** Satish et al. 2014, p.17, paragraph 3. The paper specifies "trailing six months" for the deseasonalization window.

#### Function 2: FitInterDayARMA

Fits a per-bin ARMA model on the daily volume series for that bin, selecting orders via AICc.

```
function FitInterDayARMA(symbol s, bin i, day t):
    # Reference: Satish et al. 2014, p.17, para 2
    # "a per-symbol, per-bin ARMA(p,q) model ... serial correlation
    #  observable across daily volumes"
    # "We consider all values of p and q lags through five, as well as
    #  a constant term"
    # "corrected AIC, symbolized by AIC_c, as detailed by Hurvich and
    #  Tsai [1989, 1993]"

    daily_series = [volume[s, d, i] for d in training window of N_interday days before t]

    best_model = None
    best_aicc = +infinity

    for p in 0..p_max_interday:
        for q in 0..q_max_interday:
            if p == 0 and q == 0:
                continue  # At least one AR or MA term required
            n = length(daily_series)
            k = p + q + 1  # AR params + MA params + constant
            if n <= k + 1:
                continue  # Insufficient data for AICc correction

            model = fit_ARMA(daily_series, order=(p, q), include_constant=True)

            if not model.converged or not model.is_stationary or not model.is_invertible:
                continue

            aic = -2 * model.log_likelihood + 2 * k
            aicc = aic + 2 * k * (k + 1) / (n - k - 1)

            if aicc < best_aicc:
                best_aicc = aicc
                best_model = model

    return best_model  # May be None if all fits fail
```

**Paper reference:** Satish et al. 2014, p.17, paragraph 2. AICc formula from Hurvich and Tsai (1989, 1993).

#### Function 3: FitIntraDayARMA

Fits a single ARMA model per symbol on the deseasonalized intraday volume series from a rolling window of recent trading days.

```
function FitIntraDayARMA(symbol s, day t, seasonal_factor):
    # Reference: Satish et al. 2014, p.17-18
    # "we fit an additional ARMA(p,q) model over deseasonalized
    #  intraday bin volume data"
    # "We compute this model on a rolling basis over the most recent month"
    # "AR lags with a value less than five"
    # "a dual ARMA model having fewer than 11 terms"

    # Collect deseasonalized intraday series from trailing N_intraday days
    deseas_series = []
    for d in trailing N_intraday trading days before t:
        for i in 1..I:
            deseas_series.append(volume[s, d, i] / seasonal_factor[i])

    # Also include today's observed bins (bins 1..current_bin) if forecasting intraday
    # This is handled by the caller; here we fit on the training window only

    best_model = None
    best_aicc = +infinity

    for p in 0..p_max_intraday:  # p_max_intraday = 4 (AR lags < 5)
        for q in 0..q_max_intraday:
            if p == 0 and q == 0:
                continue

            n = length(deseas_series)
            k = p + q + 1
            if n <= k + 1:
                continue

            model = fit_ARMA(deseas_series, order=(p, q), include_constant=True)

            if not model.converged or not model.is_stationary or not model.is_invertible:
                continue

            aic = -2 * model.log_likelihood + 2 * k
            aicc = aic + 2 * k * (k + 1) / (n - k - 1)

            if aicc < best_aicc:
                best_aicc = aicc
                best_model = model

    return best_model  # May be None if all fits fail
```

**Paper reference:** Satish et al. 2014, p.17 para 3 -- p.18 para 1. "Trailing six months" for deseasonalization window, "rolling basis over the most recent month" for ARMA fitting window, "AR lags with a value less than five" (p.18).

**Note on "fewer than 11 terms":** The paper states "we fit each symbol with a dual ARMA model having fewer than 11 terms" (p.18). This is an observed outcome of AICc selection and AR coefficient decay, not a hard constraint. The phrasing "As a result, we fit..." indicates it describes what happened empirically. In practice, AICc-selected models with p_max_interday=5 and p_max_intraday=4 will typically have low total parameter counts. If a developer wishes to enforce this strictly, the combined parameter count across both ARMA models should satisfy: (p_inter + q_inter + 1) + (p_intra + q_intra + 1) < 11. [Researcher inference for the interpretation; the exact phrasing is from Satish et al. 2014, p.18, para 1.]

#### Function 4: ClassifyRegime

Determines which volume regime the current (day, bin) observation falls into, based on cumulative volume percentile.

```
function ClassifyRegime(symbol s, day t, bin i, regime_thresholds):
    # Reference: Satish et al. 2014, p.18, para 4
    # "a notion of regime switching by training several weight models
    #  for different historical volume percentile cutoffs, and, in our
    #  out-of-sample period, dynamically apply the appropriate weights
    #  intraday based on the historical percentile of the observed
    #  cumulative volume"

    if i < min_regime_bins:
        return DEFAULT_REGIME  # Too early to classify reliably

    # Compute cumulative volume observed so far today
    cum_vol_today = sum(volume[s, t, j] for j in 1..i)

    # Build percentile distribution of historical cumulative volume at this bin
    historical_cum_vols = []
    for d in trailing regime_threshold_window trading days before t:
        historical_cum_vols.append(sum(volume[s, d, j] for j in 1..i))

    percentile = percentile_rank(cum_vol_today, historical_cum_vols)

    # Map percentile to regime index
    for r in 0..n_regimes-1:
        if percentile <= regime_thresholds[r]:
            return r
    return n_regimes - 1
```

**Paper reference:** Satish et al. 2014, p.18, paragraph 4. The paper does not disclose the number of regimes or the specific percentile cutoffs -- these are proprietary. [Researcher inference: 3 regimes with tercile boundaries at [33, 67] is the simplest scheme consistent with "several weight models."]

#### Function 5: OptimizeWeights

Trains regime-specific weights that combine the three forecast components to minimize prediction error on in-sample data.

```
function OptimizeWeights(symbol s, training_days, regime_labels):
    # Reference: Satish et al. 2014, p.18, para 3-4
    # "a dynamic weight overlay on top of these three components
    #  (historical, inter-, and intraday ARMA) that minimizes the error
    #  on in-sample data"

    weights = {}  # dict: regime_index -> (w1, w2, w3)

    for r in 0..n_regimes-1:
        # Collect (actual, H, D, A) tuples where regime == r
        data_r = []
        for (d, i) in training_days x bins:
            if regime_label[d, i] == r:
                actual = volume[s, d, i]
                H = historical_avg[s, d, i]
                D = interday_forecast[s, d, i]
                A = intraday_forecast[s, d, i]
                data_r.append((actual, H, D, A))

        if len(data_r) < min_samples_per_regime:
            weights[r] = (1/3, 1/3, 1/3)  # Fallback to equal weights
            continue

        # Minimize MSE subject to simplex constraint
        # min_w sum((actual - w1*H - w2*D - w3*A)^2)
        # subject to: w1 + w2 + w3 = 1, w1 >= 0, w2 >= 0, w3 >= 0
        w_opt = scipy.optimize.minimize(
            fun=lambda w: MSE(data_r, w),
            x0=[1/3, 1/3, 1/3],
            method='SLSQP',
            constraints=[{'type': 'eq', 'fun': lambda w: sum(w) - 1}],
            bounds=[(0, 1), (0, 1), (0, 1)]
        )
        weights[r] = w_opt.x

    return weights
```

**Paper reference:** Satish et al. 2014, p.18, paragraphs 3-4. The paper says "minimizes the error" without specifying the loss function. [Researcher inference: MSE with non-negative simplex constraint. The constraint w1+w2+w3=1 ensures the combined forecast is a convex combination of the components. The paper does not explicitly state this constraint, but it is the standard approach for combining forecasts and prevents scale drift. An alternative is unconstrained weights with MAPE minimization.]

#### Function 6: ForecastRawVolume (Model A)

Produces the raw volume forecast for a given symbol, day, and bin by combining the three components with regime-dependent weights.

```
function ForecastRawVolume(symbol s, day t, bin i, current_bin):
    # Reference: Satish et al. 2014, p.17-18, Exhibit 1
    #
    # TRAINING PHASE (run once per re-estimation cycle):
    #   - seasonal_factor = ComputeSeasonalFactors(s, t)
    #   - interday_model[i] = FitInterDayARMA(s, i, t)  for each bin i
    #   - intraday_model = FitIntraDayARMA(s, t, seasonal_factor)
    #   - weights = OptimizeWeights(s, training_days, regime_labels)
    #
    # PREDICTION PHASE (run each bin):

    # Component 1: Historical rolling average
    # Reference: Satish et al. 2014, Exhibit 1 "Prior 21 days"
    H = mean(volume[s, d, i] for d in trailing N_hist days before t)

    # Component 2: Inter-day ARMA forecast
    # Reference: Satish et al. 2014, p.17, Exhibit 1 "Prior 5 days"
    if interday_model[i] is not None:
        D = interday_model[i].forecast(steps=1)
        # The model has been updated with observations through day t-1
    else:
        D = H  # Fallback if ARMA fitting failed

    # Component 3: Intraday ARMA forecast
    # Reference: Satish et al. 2014, p.17-18
    if intraday_model is not None and current_bin >= 1:
        # Condition the intraday ARMA on today's observed deseasonalized bins
        deseas_observed = [volume[s, t, j] / seasonal_factor[j] for j in 1..current_bin]
        # Update ARMA state (not re-estimate -- just condition on new observations)
        intraday_model.update_state(deseas_observed)
        # Forecast (i - current_bin) steps ahead in deseasonalized space
        steps_ahead = i - current_bin
        A_deseas = intraday_model.forecast(steps=steps_ahead)
        # Re-seasonalize
        A = A_deseas * seasonal_factor[i]
    else:
        A = H  # Fallback

    # Combine via regime-specific weights
    regime = ClassifyRegime(s, t, current_bin, regime_thresholds)
    w1, w2, w3 = weights[regime]
    V_hat = w1 * H + w2 * D + w3 * A

    # Clamp to non-negative
    V_hat = max(V_hat, 0)

    return V_hat
```

**Paper reference:** Satish et al. 2014, pp.17-18, Exhibit 1. The exhibit shows the data flow: "Prior 21 days" feeds the Historical Window, "Prior 5 days" feeds ARMA Daily, "Current Bin / 4 Bins Prior to Current Bin" feeds ARMA Intraday, and all three flow into the "Dynamic Weights Engine" to produce the "Next Bin Forecast."

**Ambiguity -- "Prior 5 days" for inter-day ARMA:** Exhibit 1 labels the input to "ARMA Daily" as "Next Bin (Prior 5 days)." This could mean: (a) the ARMA is trained on only the most recent 5 daily observations, or (b) 5 days' worth of daily volume values serve as the AR prediction inputs (i.e., the effective AR memory is at most 5). Interpretation (a) is problematic because fitting ARMA(p,q) with p,q up to 5 on only 5 data points is statistically unsound -- the number of parameters would exceed or equal the number of observations, and AICc correction would dominate. Interpretation (b) is more defensible: the ARMA is trained on a longer window (N_interday trading days) but the AR coefficients effectively capture patterns over the prior ~5 days. The Exhibit's annotation likely describes what data the model "looks at" for prediction, not the training window. I adopt interpretation (b) with N_interday = 126 (6 months) as the training window and p_max_interday = 5. [Researcher inference for the interpretation; Exhibit 1 caption and p.17 para 2 are the source.]

#### Function 7: ForecastVolumePercentage (Model B)

Produces a one-bin-ahead volume percentage forecast by adjusting a historical baseline with a regression on recent volume surprises.

```
function ForecastVolumePercentage(symbol s, day t, next_bin, observed_volumes):
    # Reference: Satish et al. 2014, pp.18-19
    # Based on Humphery-Jenner (2011) dynamic VWAP
    #
    # TRAINING PHASE (run once per re-estimation cycle):
    #   - Fit rolling OLS regression of volume surprises on lagged surprises
    #   - Compute historical volume percentage curve p_hist[i] for each bin
    #
    # PREDICTION PHASE (run each bin):

    current_bin = next_bin - 1  # Last observed bin

    # Step 1: Compute estimated daily total volume
    # Reference: Researcher inference; required for converting raw forecasts to percentages
    observed_total = sum(observed_volumes[1..current_bin])
    remaining_forecasts = [ForecastRawVolume(s, t, i, current_bin) for i in next_bin..I]
    V_total_est = observed_total + sum(remaining_forecasts)

    # Step 2: Check switch-off condition
    # Reference: Satish et al. 2014, p.24
    # "once 80% of the day's volume is reached, return to a historical approach"
    # Also Humphery-Jenner (2011)
    cum_pct = observed_total / V_total_est
    if cum_pct >= pct_switchoff:
        # Switch off dynamic adjustment; use scaled historical baseline
        remaining_hist = sum(p_hist[j] for j in next_bin..I)
        remaining_fraction = 1.0 - cum_pct
        if remaining_hist > 0:
            scale = remaining_fraction / remaining_hist
        else:
            scale = 0.0
        return scale * p_hist[next_bin]

    # Step 3: Compute volume surprises for recent bins
    # Reference: Satish et al. 2014, p.18-19
    # Surprise = actual_pct - expected_pct
    surprises = []
    for k in 1..K_reg:
        lag_bin = current_bin - k + 1
        if lag_bin < 1:
            surprises.append(0.0)  # Pad with zero for early bins
        else:
            actual_pct = observed_volumes[lag_bin] / V_total_est
            expected_pct = ForecastRawVolume(s, t, lag_bin, lag_bin - 1) / V_total_est
            surprises.append(actual_pct - expected_pct)

    # Step 4: Compute dynamic adjustment
    # Reference: Satish et al. 2014, pp.18-19
    # delta = sum(beta_k * surprise[k]) for k in 1..K_reg
    # No intercept: "we perform both regressions without the inclusion
    #   of a constant term" (p.19)
    delta = sum(beta[k] * surprises[k] for k in 0..K_reg-1)

    # Step 5: Scale historical baseline for remaining volume
    # Reference: Researcher inference for renormalization
    remaining_hist = sum(p_hist[j] for j in next_bin..I)
    remaining_fraction = 1.0 - cum_pct
    if remaining_hist > 0:
        scale = remaining_fraction / remaining_hist
    else:
        scale = 0.0
    scaled_base = scale * p_hist[next_bin]

    # Step 6: Apply deviation constraint
    # Reference: Satish et al. 2014, p.24 / Humphery-Jenner (2011)
    # "depart no more than 10% away from a historical VWAP curve"
    max_delta = max_deviation * scaled_base
    delta = clip(delta, -max_delta, +max_delta)

    # Step 7: Final forecast
    p_hat = scaled_base + delta
    p_hat = max(p_hat, 0.0)  # Non-negative

    return p_hat
```

**Paper reference:** Satish et al. 2014, pp.18-19, p.24. The dynamic VWAP methodology is based on Humphery-Jenner (2011). The paper states "we could apply our more extensive volume forecasting model" to compute volume surprises (p.19), indicating the raw model's forecasts should serve as the expected volume for computing surprises.

**Note on "could":** The paper uses "we could apply our more extensive volume forecasting model" (p.19) -- the word "could" is aspirational, not confirmative. It is possible the published results used a simpler historical baseline for surprises (as in the original Humphery-Jenner). I recommend using the raw model (Model A) forecasts as the primary approach and providing a configuration option to fall back to historical averages. [Researcher inference.]

**Note on no-intercept regression:** The paper states: "we perform both regressions without the inclusion of a constant term (indicating a non-zero y intercept). This means our model does not assume a positive amount of VWAP error if our volume predictions are 100% accurate" (p.19). While this quote refers to the validation regressions (VWAP error vs. percentage error), the same principle applies to the surprise regression: when all surprises are zero, the adjustment delta should be zero. [Researcher inference for applying the no-intercept principle to the prediction regression; the explicit quote is about validation regressions.]

#### Function 8: TrainPercentageRegression

Trains the rolling OLS regression coefficients for the volume percentage model.

```
function TrainPercentageRegression(symbol s, training_days):
    # Reference: Satish et al. 2014, pp.18-19
    # Humphery-Jenner (2011) rolling regression

    # Collect training samples: (delta_actual, surprise_lags)
    X = []  # Each row: [surprise[lag_1], surprise[lag_2], ..., surprise[lag_K_reg]]
    y = []  # Each element: actual_pct - expected_pct for target bin

    for d in training_days:
        for i in 2..I:  # Start at bin 2 (need at least 1 lag)
            # Compute target: actual percentage deviation
            actual_pct_d_i = volume[s, d, i] / sum(volume[s, d, j] for j in 1..I)
            expected_pct_d_i = raw_forecast[s, d, i] / sum(raw_forecast[s, d, j] for j in 1..I)
            target = actual_pct_d_i - expected_pct_d_i

            # Compute lagged surprises
            row = []
            for k in 1..K_reg:
                lag_bin = i - k
                if lag_bin < 1:
                    row.append(0.0)
                else:
                    actual_pct_lag = volume[s, d, lag_bin] / sum(volume[s, d, j] for j in 1..I)
                    expected_pct_lag = raw_forecast[s, d, lag_bin] / sum(raw_forecast[s, d, j] for j in 1..I)
                    row.append(actual_pct_lag - expected_pct_lag)

            X.append(row)
            y.append(target)

    # OLS without intercept
    beta = OLS_fit(X, y, fit_intercept=False)

    return beta  # Array of K_reg coefficients
```

**Paper reference:** Satish et al. 2014, pp.18-19. The paper does not specify the exact number of lag terms or whether the regression is pooled across bins or fit per-bin. [Researcher inference: pooled across all bins and days (single regression), K_reg = 3 lagged surprise terms. A pooled regression is more parsimonious and more robust than 26 per-bin regressions with limited data.]

#### Function 9: DailyOrchestration

Orchestrates the full forecasting pipeline for a single symbol on a single trading day.

```
function DailyOrchestration(symbol s, day t):
    # STEP 0: Pre-market setup
    # Load pre-trained models (or re-estimate if scheduled)
    seasonal_factor = ComputeSeasonalFactors(s, t)
    interday_models = {i: FitInterDayARMA(s, i, t) for i in 1..I}  # or load cached
    intraday_model = FitIntraDayARMA(s, t, seasonal_factor)  # or load cached
    weights = load_or_compute_weights(s, t)
    beta = load_or_compute_regression(s, t)
    p_hist = ComputeHistoricalPercentages(s, t)  # trailing N_hist days

    # STEP 1: Pre-market forecasts (static, all bins)
    # Used as baseline expected percentages for surprise computation
    static_forecasts = []
    for i in 1..I:
        V_hat = ForecastRawVolume(s, t, i, current_bin=0)  # No observed bins yet
        static_forecasts.append(V_hat)
    V_total_static = sum(static_forecasts)

    # STEP 2: Intraday loop -- process each bin as it completes
    observed_volumes = []
    for current_bin in 1..I:
        # Wait for bin current_bin to complete, then observe volume
        observed_volumes.append(volume[s, t, current_bin])

        # Model A: Update raw forecasts for all remaining bins
        raw_forecasts_remaining = []
        for i in (current_bin + 1)..I:
            V_hat = ForecastRawVolume(s, t, i, current_bin)
            raw_forecasts_remaining.append(V_hat)

        # Model B: Forecast volume percentage for next bin
        if current_bin < I:
            next_bin = current_bin + 1
            p_hat = ForecastVolumePercentage(s, t, next_bin, observed_volumes)
            # Output p_hat to VWAP execution algorithm

    return
```

**Paper reference:** Satish et al. 2014, p.15 (introduction) and p.18 (methodology). The paper emphasizes that raw volume forecasts are for "all remaining bins" while percentage forecasts are "only for the subsequent interval" (p.18).

### Data Flow

```
Input: Historical volume data volume[s, d, i] for symbol s, day d, bin i
       At least N_seasonal + max(N_interday, N_intraday) days of history
       26 bins per day (15-minute intervals, 9:30-16:00 ET)

  |
  v

[ComputeSeasonalFactors]
  Input:  volume[s, d, i] for trailing 126 days, all bins
  Output: seasonal_factor[1..26]  (float array, shape (26,))

  |
  v

[FitInterDayARMA] x 26 bins            [FitIntraDayARMA] x 1 per symbol
  Input:  daily volume series             Input:  deseasonalized intraday series
          for bin i, shape (N_interday,)           shape (N_intraday * 26,)
  Output: ARMA model object               Output: ARMA model object
          (coefficients, state)                    (coefficients, state)

  |                                        |
  +----------+----------------------------+
             |
             v

[ClassifyRegime]
  Input:  cumulative volume up to current bin, historical distribution
  Output: regime index r (integer, 0..n_regimes-1)

             |
             v

[ForecastRawVolume] (Model A)
  Input:  trained models, current_bin, target bin i
  Output: V_hat (float, shares)
          Combines H, D, A with weights[regime]

             |
             v

[ForecastVolumePercentage] (Model B)
  Input:  V_hat from Model A for remaining bins,
          observed volumes, p_hist, beta coefficients
  Output: p_hat (float, fraction of daily volume for next bin)
```

### Variants

This specification implements the **full dual-model system** as described in Satish et al. (2014), which represents the most complete treatment in the assigned papers. The key variant decision is the choice of surprise baseline in Model B:

- **Primary (implemented):** Use Model A (raw volume) forecasts to compute expected volume percentages for surprise calculation. This is the more sophisticated approach suggested by the paper ("we could apply our more extensive volume forecasting model," p.19).
- **Alternative (configuration option):** Use simple historical rolling averages as the surprise baseline, matching the original Humphery-Jenner (2011) formulation. This is simpler and may have been what the published results actually used.

The paper also tested 5-minute and 30-minute bin sizes (Exhibit 9), showing smaller improvements (2.25% and 2.95% respectively vs. 7.55% for 15-minute). The 15-minute bin size is implemented as the primary configuration.

---

## Parameters

| Parameter | Description | Recommended Value | Sensitivity | Range |
|-----------|-------------|-------------------|-------------|-------|
| I | Number of intraday bins | 26 | N/A (structural) | {13, 26, 78} for {30, 15, 5}-min |
| N_hist | Rolling window for historical average (Component 1) | 21 trading days | Medium | [10, 60] |
| N_seasonal | Window for seasonal factor computation | 126 trading days (6 months) | Low | [63, 252] |
| N_interday | Training window for inter-day ARMA | 126 trading days | Low-Medium | [63, 252] |
| N_intraday | Rolling window for intraday ARMA fitting | 21 trading days (1 month) | Medium | [10, 42] |
| p_max_interday | Maximum AR order for inter-day ARMA | 5 | Low | [3, 5] |
| q_max_interday | Maximum MA order for inter-day ARMA | 5 | Low | [3, 5] |
| p_max_intraday | Maximum AR order for intraday ARMA | 4 | Low | [2, 4] |
| q_max_intraday | Maximum MA order for intraday ARMA | 5 | Low | [3, 5] |
| n_regimes | Number of volume regimes for weight switching | 3 | Medium-High | [2, 5] |
| regime_percentiles | Percentile cutoffs between regimes | [33, 67] | High | Depends on n_regimes |
| regime_threshold_window | Window for building percentile distribution | 60 trading days | Medium | [21, 126] |
| min_regime_bins | Bins before regime classification activates | 3 | Low | [1, 5] |
| min_samples_per_regime | Minimum training samples per regime for weight optimization | 50 | Low | [20, 100] |
| epsilon | Floor for seasonal factor to prevent division by zero | 1.0 (shares) | Low | [0.1, 10.0] |
| K_reg | Number of lagged surprise terms in percentage regression | 3 | Medium | [1, 5] |
| max_deviation | Maximum percentage deviation from scaled baseline | 0.10 (10%) | Medium | [0.05, 0.20] |
| pct_switchoff | Cumulative volume fraction triggering switch-off | 0.80 (80%) | Medium | [0.70, 0.90] |

**Parameter sources:**
- N_hist = 21: Satish et al. 2014, Exhibit 1 caption "Prior 21 days."
- N_seasonal = 126: Satish et al. 2014, p.17 "trailing six months."
- N_intraday = 21: Satish et al. 2014, p.18 "rolling basis over the most recent month."
- p_max_intraday = 4: Satish et al. 2014, p.18 "AR lags with a value less than five."
- p_max_interday = 5, q_max_interday = 5: Satish et al. 2014, p.17 "all values of p and q lags through five."
- max_deviation = 0.10: Satish et al. 2014, p.24 / Humphery-Jenner (2011) "depart no more than 10% away." Note: the paper says the authors "developed a separate method for computing the deviation bounds" (p.19) which is proprietary. The 10% is from Humphery-Jenner's original formulation.
- pct_switchoff = 0.80: Satish et al. 2014, p.24 "once 80% of the day's volume is reached."
- N_interday = 126: [Researcher inference.] Exhibit 1 says "Prior 5 days" for the inter-day ARMA input, which I interpret as AR prediction memory, not training window size (see ambiguity discussion in Function 6).
- n_regimes = 3, regime_percentiles = [33, 67]: [Researcher inference.] Not disclosed in paper.
- regime_threshold_window = 60: [Researcher inference.]
- min_regime_bins = 3: [Researcher inference.]
- K_reg = 3: [Researcher inference.] Not disclosed; the paper refers to "identifying the optimal number of model terms" (p.19) but does not report the value.
- epsilon = 1.0: [Researcher inference.]
- min_samples_per_regime = 50: [Researcher inference.]

### Initialization

1. **Historical data requirement:** At minimum, max(N_seasonal, N_interday) + N_intraday = 126 + 21 = 147 trading days of intraday volume data per symbol before the first forecast can be produced.

2. **Seasonal factors:** Initialize from the first 126 trading days. Update daily by sliding the window forward one day.

3. **Inter-day ARMA models:** Fit 26 models (one per bin) on the first N_interday days. Cache fitted models. On subsequent days, update ARMA state with the new observation (conditioning, not re-estimation).

4. **Intraday ARMA model:** Fit one model per symbol on the most recent 21 trading days of deseasonalized data (21 * 26 = 546 observations). Re-fit daily at market open.

5. **Regime weights:** Require a training period for weight optimization. Use the first N_weight_train days (after ARMA warm-up) to train initial weights. [Researcher inference: N_weight_train = 63 (3 months).]

6. **Percentage regression:** Train on a window of N_reg_train days using volume surprises computed from in-sample raw forecasts. [Researcher inference: N_reg_train = 252 (1 year).]

7. **Initial weight guess:** (1/3, 1/3, 1/3) for all regimes.

### Calibration

**Model A calibration** involves three concurrent procedures:

1. **ARMA order selection:** Grid search over (p, q) for each of the 26 inter-day models and the single intraday model. AICc selects the best model per bin/symbol. For 500 symbols: 500 * 26 * 36 = 468,000 inter-day ARMA fits + 500 * 36 = 18,000 intraday fits per calibration cycle. This is computationally intensive but parallelizable across symbols.

2. **Weight optimization:** For each regime, minimize MSE over in-sample forecast errors. SLSQP with simplex constraint. Convergence is fast (3 parameters per regime).

3. **Regime threshold computation:** Compute cumulative volume percentile distributions from historical data.

**Model B calibration:**

1. **Percentage regression:** OLS (closed-form) with K_reg predictors and no intercept. Fast to compute.

**Re-estimation schedule:** [Researcher inference; paper does not specify.]
- Daily: seasonal factors (window slides by 1 day), intraday ARMA (re-fit on latest 21 days), historical percentage curve.
- Weekly (every 5 trading days): inter-day ARMA models (full re-estimation with AICc selection).
- Monthly (every 21 trading days): regime weights, percentage regression coefficients.
- Between inter-day re-estimations: update ARMA state by conditioning on new daily observations without re-running MLE.

---

## Validation

### Expected Behavior

**Model A (Raw Volume):**
- Median MAPE reduction of 24% vs. historical rolling average baseline across all 26 intraday bins. (Satish et al. 2014, p.20, text below Exhibit 6.)
- Bottom-95% MAPE reduction of 29% vs. baseline. (Same source.)
- Error reduction varies by time of day: approximately 10-12% at 9:30 (first bin), increasing to 30-33% by 15:30 (last few bins). The bottom-95% reduction increases more smoothly from approximately 15% to 35-40%. (Satish et al. 2014, Exhibit 6.)
- Improvements are consistent across SIC industry groups (20-30% median reduction for most groups) and beta deciles (20-35% median reduction across all deciles). (Satish et al. 2014, Exhibits 7 and 8.)

**Model B (Volume Percentage):**
- Median absolute error: 0.00874 (historical) vs. 0.00808 (dynamic) -- 7.55% reduction for 15-minute bins. Significant at << 1% level via Wilcoxon signed-rank test. (Satish et al. 2014, Exhibit 9, p.23.)
- Bottom-95% average absolute error: 0.00986 (historical) vs. 0.00924 (dynamic) -- 6.29% reduction. (Same source.)
- For 5-minute bins: 2.25% median reduction. For 30-minute bins: 2.95% median reduction. (Satish et al. 2014, Exhibit 9.)

**VWAP tracking error (simulation):**
- Mean VWAP tracking error reduced from 9.62 bps (historical curve) to 8.74 bps (dynamic curve) -- 9.1% reduction. Standard deviation: 11.18 bps (historical) vs. 10.08 bps (dynamic). (Satish et al. 2014, Exhibit 10, p.23.)
- Paired t-test statistic: 2.34, p < 0.01. (Same source.)
- Per-category reductions: 7-10% across Dow 30, midcap, and high-variance stock groups. (Satish et al. 2014, p.23.)
- Simulation used 600+ day-long VWAP orders with 10% of 30-day ADV order size. (Satish et al. 2014, p.23.)

**Relationship between percentage error and VWAP tracking error:**
- Dow 30: coefficient 220.9 bps per unit error, R^2 = 0.5146, t-stat 7.496. (Satish et al. 2014, Exhibit 3, p.20.)
- High-variance stocks: coefficient 454.3 bps per unit error, R^2 = 0.5886, t-stat 6.329. (Satish et al. 2014, Exhibit 5, p.21.)
- Both regressions fit without an intercept, through the origin. (Satish et al. 2014, p.19.)

**Comparison with Chen et al. (2016) Kalman filter approach:**
- Chen et al. report average dynamic MAPE of 0.46 across 30 securities, vs. 0.61 for static prediction, vs. 1.28 for rolling means. (Chen et al. 2016, Table 3.)
- Chen et al. VWAP tracking: average 6.38 bps (dynamic robust Kalman) vs. 7.48 bps (rolling means) -- 15% improvement. (Chen et al. 2016, Table 4.)
- **Direct comparison is imperfect:** different datasets (Satish: 500 U.S. equities; Chen: 30 global securities), different time periods, and different MAPE normalization (Satish uses percentage, Chen uses fractional). Chen et al. claim to outperform Satish et al., but the evaluation conditions differ substantially. (Chen et al. 2016, p.8.)

### Sanity Checks

1. **Regime weights:** All weights should be non-negative and sum to 1.0 per regime. Expect w1 (historical) to be larger for early bins where ARMA has less intraday data to leverage, and w3 (intraday ARMA) to increase for later bins as more intraday observations become available. [Researcher inference.]

2. **Seasonal factor U-shape:** The seasonal_factor array should exhibit the well-known U-shaped intraday volume pattern: higher values at market open (bin 1) and close (bin 26), lower values mid-day. The ratio of max to min seasonal factor should typically be 2x-4x for liquid stocks. [Researcher inference; U-shape is well-established in literature, e.g., Brownlees et al. 2011.]

3. **Deseasonalized series stationarity:** The deseasonalized intraday series (volume[s,d,i] / seasonal_factor[i]) should be approximately stationary. An Augmented Dickey-Fuller test should reject the unit root null. Expected mean of the deseasonalized series is approximately 1.0. [Researcher inference.]

4. **AICc model selection:** Expect AICc to typically select low-order models (p <= 2, q <= 2). Monitor the distribution of selected orders across bins. [Researcher inference.]

5. **Regime bucket populations:** All regime buckets should be populated with sufficient training samples. If any bucket has fewer than min_samples_per_regime, the regime scheme may be too fine-grained. [Researcher inference.]

6. **Deviation constraint binding frequency:** The deviation constraint in Model B should bind on no more than 10-20% of bins. If it binds too frequently, the constraint may be too tight; if it never binds, max_deviation may be too generous. [Researcher inference.]

7. **Switch-off engagement:** The 80% switch-off should engage for the last 2-4 bins of a typical day. If it engages much earlier, V_total_est may be systematically too high. [Researcher inference.]

8. **Forecast non-negativity:** Raw volume forecasts should never be negative after clamping. If negative values occur frequently before clamping, the ARMA component may be producing poor extrapolations. [Researcher inference.]

9. **Surprise mean:** The mean of volume surprises across all training (symbol, day, bin) observations should be approximately zero (indicating the raw model is unbiased on average). Standard deviation of surprises for 15-minute bins is expected to be approximately 0.005-0.015. [Researcher inference.]

10. **Regression coefficient magnitudes:** The absolute values of beta coefficients in the percentage regression should be small (|beta_k| < 0.5). Large coefficients indicate instability. [Researcher inference.]

11. **Historical baseline MAPE:** Component 1 alone (historical rolling average) should produce MAPE values consistent with the paper's baseline (which the dual ARMA improves upon by 24% median). This establishes that the baseline implementation is correct before adding ARMA components. [Researcher inference.]

### Edge Cases

1. **ARMA convergence failures:** Some ARMA(p,q) fits will fail to converge, especially for illiquid stocks or unusual bin patterns. Detection: check optimizer convergence flag, verify Hessian positive definiteness, confirm finite standard errors. Fallback: if interday_model[i] is None, set D = H (use historical average). If intraday_model is None, set A = H. For 500 symbols * 26 bins * 36 order combinations, expect 1-5% convergence failures per calibration cycle. [Researcher inference.]

2. **Zero-volume bins:** Bins with zero traded volume may occur for illiquid stocks. The seasonal factor floor (epsilon = 1.0) prevents division by zero in deseasonalization. MAPE is undefined for zero-volume bins; exclude them from error computation. Raw forecasts may produce zero or slightly negative values; clamp at zero. [Researcher inference.]

3. **Early-day bins and surprise padding:** At bins 1 through K_reg, fewer than K_reg lagged surprises are available. Pad missing lags with zeros. This means the percentage model provides minimal adjustment for the first few bins of each day. [Researcher inference.]

4. **Day-boundary handling for intraday ARMA:** The intraday ARMA is fit on a concatenated series of deseasonalized volumes from multiple days. At day boundaries (transition from bin 26 of day d to bin 1 of day d+1), there is an overnight gap. Options: (a) Naive concatenation -- accept that overnight transitions weaken AR coefficient estimates slightly. This is the simplest approach and likely what the paper used. (b) Insert NaN at day boundaries and use state-space models with missing observations. Approximately 5% of training observations (21 day-boundaries / 546 total observations) are affected. [Researcher inference; the paper does not discuss overnight handling.]

5. **Half-day trading sessions:** Days before holidays have 13 bins instead of 26 (9:30--13:00 ET). Exclude these days from training and evaluation, or configure a separate bin count. [Researcher inference; the paper does not address this explicitly.]

6. **Special calendar days:** Option expiration days, Fed announcement days, and index rebalancing days have distinctive volume patterns. The paper recommends "custom curves for special calendar days... rather than ARMAX models, due to insufficient historical occurrences" (Satish et al. 2014, p.18, paragraph 4). Implementation: detect these days via a calendar and fall back to a special-day historical average rather than the dynamic model.

7. **Lookahead bias in percentage model training:** The raw volume model is trained on data through train_end_date but then used to compute forecasts for earlier days d < train_end_date. This introduces a small look-ahead bias. The bias is negligible because ARMA parameters estimated on 126+ days change minimally when dropping one day, and seasonal factors are 126-day averages. To eliminate completely: use expanding-window re-estimation (but this multiplies computational cost by N_reg_train). [Researcher inference.]

8. **Regime boundary instability:** On days where cumulative volume oscillates near a regime percentile boundary, the model could switch rapidly between regimes, causing forecast discontinuities. [Researcher inference. Optional mitigation: hysteresis -- require the percentile to cross the boundary by a margin (e.g., 5 percentile points) before switching regimes.]

### Known Limitations

1. **Proprietary parameters:** The paper does not disclose the specific values of the dynamic weighting coefficients, the exact number of regime buckets, the regime percentile cutoffs, or the optimal regression terms for U.S. equities (Satish et al. 2014, p.18-19). These must be rediscovered via in-sample optimization.

2. **No price information:** The model is purely volume-driven and agnostic to price dynamics. It cannot exploit volume-price correlations or adjust for price impact. (Satish et al. 2014, implicitly; also noted by Bialkowski et al. 2008.)

3. **Small order size assumption:** The VWAP simulation uses 10% of 30-day ADV as order size (Satish et al. 2014, p.23). For larger orders, price impact becomes significant and the model's volume-only framework becomes insufficient.

4. **Metric incomparability:** Different papers use different MAPE normalization conventions, making direct cross-paper comparison difficult. Satish et al. express MAPE as a percentage; Chen et al. (2016) express it as a fraction. (Satish et al. 2014, p.17; Chen et al. 2016, p.7 Equation 37.)

5. **Linear ARMA assumptions:** ARMA models assume linear dynamics and Gaussian-like residuals. Volume distributions are positively skewed with heavy tails. This could be mitigated by log-transforming volume before ARMA fitting (as Chen et al. 2016 do), but the paper does not describe this step.

6. **No multivariate effects:** The model treats each symbol independently. Cross-stock volume correlations (e.g., sector or market-wide volume surges) are not captured. Bialkowski et al. (2008) address this with PCA-based factor decomposition.

7. **Dataset specificity:** Results are reported on the top 500 U.S. equities by dollar volume over one specific year. Performance on less liquid stocks, other markets, or different time periods may differ.

---

## Paper References

| Spec Section | Paper Source |
|---|---|
| Overview, dual-model architecture | Satish et al. 2014, pp.15-18 |
| Bin structure (26 bins, 15-min) | Satish et al. 2014, p.16 |
| Component 1: Historical rolling average | Satish et al. 2014, p.17 para 1, Exhibit 1 "Prior 21 days" |
| Component 2: Inter-day ARMA | Satish et al. 2014, p.17 para 2 |
| AICc formula | Hurvich and Tsai (1989, 1993), cited in Satish et al. 2014, p.17 |
| Component 3: Intraday ARMA | Satish et al. 2014, pp.17-18 |
| AR lags < 5 | Satish et al. 2014, p.18 para 1 |
| Fewer than 11 terms | Satish et al. 2014, p.18 para 1 |
| Deseasonalization (trailing 6 months) | Satish et al. 2014, p.17 para 3 |
| Intraday ARMA window (1 month) | Satish et al. 2014, p.18 para 1 |
| Component 4: Dynamic weight overlay | Satish et al. 2014, p.18 para 3 |
| Regime switching | Satish et al. 2014, p.18 para 4 |
| Custom curves for special days | Satish et al. 2014, p.18 para 4 |
| Volume percentage model (dynamic VWAP) | Satish et al. 2014, pp.18-19; Humphery-Jenner (2011) |
| Deviation limits (10%) | Satish et al. 2014, p.24; Humphery-Jenner (2011) |
| Switch-off at 80% | Satish et al. 2014, p.24; Humphery-Jenner (2011) |
| No-intercept regressions | Satish et al. 2014, p.19 |
| MAPE metric | Satish et al. 2014, p.17 |
| Percentage error metric | Satish et al. 2014, p.17 |
| MAPE reduction (24% median, 29% bottom-95%) | Satish et al. 2014, p.20, below Exhibit 6 |
| MAPE by time-of-day | Satish et al. 2014, Exhibit 6 |
| MAPE by SIC group | Satish et al. 2014, Exhibit 7 |
| MAPE by beta decile | Satish et al. 2014, Exhibit 8 |
| Volume percentage results (Exhibit 9) | Satish et al. 2014, Exhibit 9, p.23 |
| VWAP tracking error simulation | Satish et al. 2014, Exhibit 10, p.23 |
| VWAP error vs. percentage error regression | Satish et al. 2014, Exhibits 2-5, pp.20-21 |
| Comparison: Chen et al. Kalman filter | Chen et al. 2016, Tables 3-4, pp.9-10 |
| Comparison: Brownlees et al. CMEM | Brownlees et al. 2011; cited in Satish et al. 2014, p.24 |
| Comparison: Bialkowski et al. PCA-SETAR | Bialkowski et al. 2008; cited in Satish et al. 2014, p.24 |
| Regime count, percentile thresholds | Researcher inference |
| N_interday training window | Researcher inference |
| K_reg value | Researcher inference |
| MSE objective and simplex constraint for weights | Researcher inference |
| Re-estimation schedule | Researcher inference |
| Surprise normalization details | Researcher inference |
| Renormalization of scaled baseline | Researcher inference |
| Epsilon floor value | Researcher inference |
