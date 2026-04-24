# Implementation Specification: Dual-Mode Volume Forecast (Raw + Percentage)

## Overview

This specification describes a two-model intraday volume forecasting system based on Satish, Saxena, and Palmer (2014). The system produces two outputs:

1. **Model A (Raw Volume Forecast):** Predicts the absolute number of shares that will trade in each 15-minute bin of a U.S. equity trading day. It combines three forecast components -- a rolling historical average (H), an inter-day ARMA model (D), and a deseasonalized intraday ARMA model (A) -- using regime-dependent optimized weights. The regime is selected dynamically based on the historical percentile of cumulative volume observed so far in the day.

2. **Model B (Volume Percentage Forecast):** Predicts the fraction of daily volume expected to trade in the next bin, extending Humphery-Jenner's (2011) dynamic VWAP framework. It adjusts a naive historical percentage curve using volume surprises computed from Model A's output, via a rolling OLS regression without intercept. Safety constraints (deviation limits, switch-off threshold) prevent excessive departures from the historical curve.

Model A forecasts all remaining bins simultaneously (for portfolio scheduling), while Model B forecasts one bin ahead (for step-by-step VWAP execution). Model A's output feeds Model B's surprise signal, making them a tightly coupled sequential pair.

Chen, Feng, and Palomar (2016) provide a benchmark comparison target (Kalman filter approach with 0.46 average MAPE on 30 securities) but are not implemented here.

## Algorithm

### Model Description

**Problem 1 -- Raw volume forecasting:** Given the history of bin-level volumes for a stock, predict the raw volume (share count) in each of I = 26 fifteen-minute bins (9:30-16:00 ET) of the current trading day, updating predictions intraday as new bins are observed. Three information sources are exploited:
- (a) The typical volume for that bin over recent history (historical average H).
- (b) Day-to-day serial correlation in volume at the same bin index (inter-day ARMA D).
- (c) Within-day momentum from recently observed bins today (intraday ARMA on deseasonalized data A).

These three forecasts are combined via weights optimized per regime, where regimes are defined by the day's cumulative volume percentile relative to historical norms.

**Problem 2 -- Volume percentage forecasting:** Given a historical estimate of the percentage of daily volume in each bin, adjust the participation rate for the next bin based on observed volume surprises earlier in the day. This produces a dynamic VWAP execution curve that adapts to unusual volume patterns.

**Assumptions:**
- Volume exhibits serial correlation both across days (same bin) and within days (adjacent bins). [Satish et al. 2014, p.17, "serial correlation observable across daily volumes"]
- The intraday seasonal pattern (U-shape) is stable over a trailing six-month window. [Satish et al. 2014, p.17, "dividing by the intraday amount of volume traded in that bin over the trailing six months"]
- Regime switching based on cumulative volume percentile captures high-volume vs. low-volume day dynamics. [Satish et al. 2014, p.18, "regime switching... training several weight models for different historical volume percentile cutoffs"]
- Volume percentage errors are linearly related to VWAP tracking error (R^2 > 0.50). [Satish et al. 2014, Exhibits 2-5, pp.20-21]

**Inputs:** Historical intraday volume time series at 15-minute resolution, per stock. Split-adjusted share counts.

**Outputs:** (1) Predicted raw volume for each of the 26 bins (Model A). (2) Predicted volume percentage for the next bin (Model B, for VWAP execution).

### Pseudocode

The system is organized into 10 functions: 5 for Model A (training, state update, forecasting, regime classification, weight optimization), 3 for Model B (training, forecasting, evaluation), and 2 for orchestration (daily workflow, re-estimation scheduling).

---

#### Function 1: ComputeSeasonalFactors

```
FUNCTION ComputeSeasonalFactors(volume_data, stock, reference_date, N_seasonal) -> seasonal_factor[1..I]:
    """
    Compute per-bin seasonal factors as trailing arithmetic mean of raw volume.
    Used for deseasonalization of intraday ARMA input.

    Reference: Satish et al. 2014, p.17, "dividing by the intraday amount of
      volume traded in that bin over the trailing six months"
    """

    FOR each bin i in 1..I:
        volumes_i = [volume_data[stock, bin=i, day=d]
                     for d in (reference_date - N_seasonal + 1)..reference_date
                     if is_full_trading_day(d)]  # exclude half-days
        seasonal_factor[i] = mean(volumes_i) if len(volumes_i) > 0 else 0.0

    # Handle zero seasonal factors: replace with minimum non-zero value
    # across all bins for this stock. This is data-adaptive and avoids
    # division-by-zero in deseasonalization.
    # [Researcher inference: paper does not address zero-volume bins.
    #  Min-non-zero is more robust than a fixed constant because it
    #  scales with the stock's volume level.]
    nonzero_values = [sf for sf in seasonal_factor if sf > 0]
    IF len(nonzero_values) == 0:
        RAISE InsufficientDataError("All bins have zero seasonal factor")
    min_nonzero = min(nonzero_values)
    FOR each bin i in 1..I:
        IF seasonal_factor[i] <= 0:
            seasonal_factor[i] = min_nonzero

    RETURN seasonal_factor
```

---

#### Function 2: FitInterDayARMA

```
FUNCTION FitInterDayARMA(volume_data, stock, reference_date, N_interday_fit,
                          p_max, q_max, seasonal_factor) -> interday_models[1..I]:
    """
    Fit one ARMA(p,q) model per bin on the daily volume time series for that bin.
    Model selection via AICc. Returns array of I fitted models.

    Reference: Satish et al. 2014, p.17, "a per-symbol, per-bin ARMA(p, q)
      model reflecting the serial correlation observable across daily volumes.
      We use nearly standard ARMA model-fitting techniques relying on
      maximum-likelihood estimation, which selects an ARMA(p, q) model
      minimizing an Akaike information criterion (AIC) as the test for the
      best model."
    Reference: Satish et al. 2014, p.17, "we consider all values of p and q
      lags through five, as well as a constant term. We depart from the
      standard technique in using the corrected AIC, symbolized by AIC_c,
      as detailed by Hurvich and Tsai [1989, 1993]."
    """

    interday_models = [None] * (I + 1)  # 1-indexed

    FOR each bin i in 1..I:
        daily_series = [volume_data[stock, bin=i, day=d]
                        for d in (reference_date - N_interday_fit + 1)..reference_date
                        if is_full_trading_day(d)]
        n = len(daily_series)

        best_aicc = +infinity
        best_model = None

        FOR p in 0..p_max:
            FOR q in 0..q_max:
                IF p == 0 AND q == 0:
                    CONTINUE  # skip constant-only; it is the baseline (H)
                k = p + q + 1  # +1 for the constant term
                IF n <= k + 1:
                    CONTINUE  # AICc denominator (n-k-1) would be <= 0

                TRY:
                    model = fit_ARMA(daily_series, order=(p, q),
                                     include_constant=True,
                                     enforce_stationarity=True,
                                     enforce_invertibility=True)
                    # Verify convergence and model validity
                    IF NOT model.converged:
                        CONTINUE
                    IF NOT model.is_stationary:
                        CONTINUE
                    IF NOT model.is_invertible:
                        CONTINUE

                    # AICc: Satish et al. 2014, p.17; Hurvich & Tsai 1989, 1993
                    aic = -2 * model.log_likelihood + 2 * k
                    aicc = aic + 2 * k * (k + 1) / (n - k - 1)

                    IF aicc < best_aicc:
                        best_aicc = aicc
                        best_model = model
                CATCH (ConvergenceError, LinAlgError):
                    CONTINUE

        interday_models[i] = best_model  # None if all fits failed

    RETURN interday_models
```

---

#### Function 3: FitIntraDayARMA

```
FUNCTION FitIntraDayARMA(volume_data, stock, reference_date, N_intraday_fit,
                          p_max_intra, q_max_intra, seasonal_factor) -> intraday_model:
    """
    Fit a single ARMA(p,q) model per stock on deseasonalized intraday volume
    from the most recent N_intraday_fit trading days. Model selection via AICc.

    Day-boundary handling: treat each day's bins as an independent segment.
    The ARMA likelihood resets at each day boundary so that lag-1 of day d+1
    does NOT connect to the last bin of day d. This prevents the model from
    learning spurious overnight correlations.

    Reference: Satish et al. 2014, p.17-18, "we fit an additional ARMA(p, q)
      model over deseasonalized intraday bin volume data... The intraday data
      are deseasonalized by dividing by the intraday amount of volume traded
      in that bin over the trailing six months."
    Reference: Satish et al. 2014, p.18, "We compute this model on a rolling
      basis over the most recent month."
    Reference: Satish et al. 2014, p.18, "we used AR lags with a value less
      than five" (constrains AR order only, not MA).
    """

    # Build deseasonalized intraday series with day boundaries
    day_segments = []  # list of lists; each inner list = one day's bins
    FOR each day d in (reference_date - N_intraday_fit + 1)..reference_date:
        IF NOT is_full_trading_day(d):
            CONTINUE
        day_bins = []
        FOR each bin i in 1..I:
            deseas = volume_data[stock, bin=i, day=d] / seasonal_factor[i]
            day_bins.append(deseas)
        day_segments.append(day_bins)

    IF len(day_segments) < 5:
        RETURN None  # insufficient data

    # Concatenate for fitting, but track day boundaries
    full_series = concatenate(day_segments)
    day_boundary_indices = []  # indices where new days start
    offset = 0
    FOR seg in day_segments:
        day_boundary_indices.append(offset)
        offset += len(seg)

    best_aicc = +infinity
    best_model = None

    FOR p in 0..p_max_intra:   # p_max_intra = 4 (AR < 5)
        FOR q in 0..q_max_intra:  # q_max_intra = 5
            IF p == 0 AND q == 0:
                CONTINUE
            k = p + q + 1

            # Effective sample size: each day-segment contributes
            # (I - p) effective observations after initial conditions.
            # This is more precise than the naive len(full_series).
            n_eff = sum(max(len(seg) - p, 0) for seg in day_segments)
            IF n_eff <= k + 1:
                CONTINUE

            TRY:
                # Fit with day-boundary awareness: each day-segment is
                # treated as an independent realization. In practice:
                # reset the initial AR/MA state at each day boundary.
                # Implementation: use statsmodels SARIMAX with
                # missing='drop' and insert NaN at boundaries, OR
                # compute per-segment likelihoods and sum them.
                model = fit_ARMA_segmented(day_segments, order=(p, q),
                                            include_constant=True,
                                            enforce_stationarity=True,
                                            enforce_invertibility=True)
                IF NOT model.converged:
                    CONTINUE

                aic = -2 * model.log_likelihood + 2 * k
                aicc = aic + 2 * k * (k + 1) / (n_eff - k - 1)

                IF aicc < best_aicc:
                    best_aicc = aicc
                    best_model = model
            CATCH (ConvergenceError, LinAlgError):
                CONTINUE

    # Log combined term count as soft guardrail
    # Reference: Satish et al. 2014, p.18, "As a result, we fit each symbol
    #   with a dual ARMA model having fewer than 11 terms."
    # [Researcher inference: "As a result" indicates this is an observed
    #  outcome of AICc selection and AR coefficient decay, not an imposed
    #  constraint. We treat it as a warning threshold. AICc naturally
    #  penalizes excess complexity, so the constraint rarely binds.]
    IF best_model is not None:
        intra_terms = best_model.p + best_model.q + 1
        # The inter-day model terms vary per bin; check worst case
        # when inter-day models are available
        LOG_INFO(f"Intraday ARMA selected: ({best_model.p},{best_model.q}), "
                 f"{intra_terms} terms")

    RETURN best_model
```

---

#### Function 4: ClassifyRegime

```
FUNCTION ClassifyRegime(observed_volumes, current_bin, cumvol_distribution,
                         regime_percentiles, min_regime_bins, default_regime) -> regime_index:
    """
    Determine the regime for the current (day, bin) based on cumulative
    volume percentile rank in the historical distribution.

    Reference: Satish et al. 2014, p.18, "dynamically apply the appropriate
      weights intraday based on the historical percentile of the observed
      cumulative volume"
    """

    # For early bins, regime classification is unreliable because
    # cumulative volume is too small to be meaningful.
    # [Researcher inference: paper does not specify start-of-day behavior.
    #  Defaulting to the middle regime is conservative and avoids extreme
    #  weight sets when little information is available.]
    IF current_bin < min_regime_bins:
        RETURN default_regime

    # Compute cumulative volume up to current_bin
    cumvol = sum(observed_volumes[j] for j in 1..current_bin)

    # Look up percentile rank in the historical distribution for this bin
    # cumvol_distribution[current_bin] is a sorted array of historical
    # cumulative volumes at this bin index
    sorted_dist = cumvol_distribution[current_bin]
    rank = bisect_left(sorted_dist, cumvol)
    pctile = 100.0 * rank / len(sorted_dist)

    # Map percentile to regime index
    # regime_percentiles is sorted, e.g., [33.3, 66.7] for 3 regimes
    FOR idx, cutoff in enumerate(regime_percentiles):
        IF pctile < cutoff:
            RETURN idx
    RETURN len(regime_percentiles)  # highest regime
```

---

#### Function 5: OptimizeRegimeWeights

```
FUNCTION OptimizeRegimeWeights(volume_data, stock, reference_date,
                                N_regime_window, interday_models, intraday_model,
                                seasonal_factor, H, regime_candidates) -> best_config:
    """
    Grid search over regime configurations. For each configuration, optimize
    per-regime component weights and evaluate on a held-out validation set.

    Reference: Satish et al. 2014, p.18, "a dynamic weight overlay on top
      of these three components... that minimizes the error on in-sample
      data. We incorporate a notion of regime switching by training several
      weight models for different historical volume percentile cutoffs"
    [Researcher inference: the paper does not specify the loss function,
     constraint set, or optimizer for weight optimization. We implement
     both MSE and MAPE objectives (see design decision below).]
    """

    # Split into train and validation periods
    val_days = 21  # holdout last 21 trading days for validation
    train_start = reference_date - N_regime_window + 1
    train_end = reference_date - val_days
    val_start = reference_date - val_days + 1
    val_end = reference_date

    best_config = None
    best_val_error = +infinity

    FOR n_reg in regime_candidates:  # e.g., [3, 4, 5]
        # Compute equally-spaced percentile cutoffs
        percentiles = [100.0 * k / n_reg for k in 1..n_reg-1]
        # e.g., n_reg=3 -> [33.33, 66.67]
        # e.g., n_reg=4 -> [25, 50, 75]

        # Build cumulative volume distribution for regime assignment
        cumvol_dist = BuildCumVolDistribution(
            volume_data, stock, train_start, train_end)

        # --- Collect training samples per regime ---
        regime_samples = {r: [] for r in 0..n_reg-1}
        FOR each day d in train_start..train_end:
            IF NOT is_full_trading_day(d): CONTINUE
            FOR each bin j in 1..I:
                regime = ClassifyRegime_Training(
                    volume_data, stock, d, j, cumvol_dist, percentiles)

                actual = volume_data[stock, bin=j, day=d]
                h_val = compute_H_asof(volume_data, stock, j, d, N_hist)
                d_val = predict_interday(interday_models[j], volume_data, stock, j, d)
                a_val = predict_intraday_for_training(
                    intraday_model, volume_data, stock, d, j, seasonal_factor)
                regime_samples[regime].append((actual, h_val, d_val, a_val))

        # --- Optimize weights per regime ---
        weights = {}
        FOR r in 0..n_reg-1:
            samples = regime_samples[r]
            IF len(samples) < min_samples_per_regime:
                # Insufficient data for this regime; use equal weights
                weights[r] = [1/3, 1/3, 1/3]
                CONTINUE

            # Primary objective: minimize MSE with simplex constraint
            # w1 + w2 + w3 = 1, all >= 0
            # This is a constrained quadratic program with a closed-form
            # solution (non-negative least squares on the simplex).
            #
            # Design decision: MSE vs MAPE
            # - MSE: smooth, convex, unique minimum, fast solvers (SLSQP/NNLS).
            #   But: trains on MSE, evaluates on MAPE (metric mismatch).
            # - MAPE: non-differentiable, may have local minima, needs
            #   derivative-free optimizer (Nelder-Mead). Matches evaluation.
            # We implement MSE as the primary and MAPE as a variant.
            # If MAPE variant produces lower out-of-sample MAPE, switch.
            # [Researcher inference throughout this function.]

            actuals = [s[0] for s in samples]
            H_vals = [s[1] for s in samples]
            D_vals = [s[2] for s in samples]
            A_vals = [s[3] for s in samples]

            # MSE optimization with simplex constraint
            # X is (n_samples, 3) matrix of component forecasts
            # y is (n_samples,) vector of actual volumes
            X = column_stack(H_vals, D_vals, A_vals)
            y = array(actuals)

            result = minimize(
                fun=lambda w: mean((y - X @ w)**2),
                x0=[1/3, 1/3, 1/3],
                method='SLSQP',
                constraints=[{'type': 'eq', 'fun': lambda w: sum(w) - 1.0}],
                bounds=[(0, None), (0, None), (0, None)]
            )
            weights[r] = result.x

        # --- Evaluate on validation set ---
        val_mape = EvaluateWeights(
            volume_data, stock, val_start, val_end,
            weights, cumvol_dist, percentiles,
            H, interday_models, intraday_model, seasonal_factor)

        IF val_mape < best_val_error:
            best_val_error = val_mape
            best_config = RegimeConfig(
                n_regimes=n_reg,
                percentiles=percentiles,
                weights=weights,
                cumvol_distribution=cumvol_dist
            )

    RETURN best_config


FUNCTION compute_H_asof(volume_data, stock, bin_i, day_d, N_hist) -> float:
    """
    Compute historical rolling average for bin i as of day d-1
    (using data strictly before day d to avoid lookahead).
    """
    past_volumes = [volume_data[stock, bin=bin_i, day=d_prev]
                    for d_prev in (day_d - N_hist)..(day_d - 1)
                    if is_full_trading_day(d_prev)]
    RETURN mean(past_volumes) if len(past_volumes) > 0 else 0.0


FUNCTION predict_interday(model, volume_data, stock, bin_i, day_d) -> float:
    """
    Produce the inter-day ARMA one-step-ahead forecast for bin i on day d.
    Uses only data through day d-1.
    """
    IF model is None:
        RETURN compute_H_asof(volume_data, stock, bin_i, day_d, N_hist)
    # The model was fitted on historical data. Produce forecast using
    # the model's internal state updated through the most recent observations.
    RETURN model.forecast(steps=1)[0]


FUNCTION predict_intraday_for_training(intraday_model, volume_data, stock,
                                        day_d, target_bin, seasonal_factor) -> float:
    """
    Produce intraday ARMA forecast for target_bin on day d during training.
    Condition on observed bins 1..target_bin-1 of day d (dynamic forecast,
    matching prediction-time behavior).

    [Researcher inference: training should use dynamic forecasts matching
     prediction behavior to avoid train/predict mismatch. The paper says
     "the volume in the next 15-minute bin from the current 15-minute bin"
     (p.17), implying sequential conditioning.]
    """
    IF intraday_model is None:
        RETURN seasonal_factor[target_bin]

    # Deseasonalize observed bins
    observed_deseas = []
    FOR j in 1..target_bin-1:
        observed_deseas.append(
            volume_data[stock, bin=j, day=day_d] / seasonal_factor[j])

    # Condition and forecast (pure operation, does not mutate model)
    state = intraday_model.condition(observed_deseas)  # returns new state
    forecast_deseas = state.forecast(steps=1)[0]

    # Re-seasonalize
    RETURN forecast_deseas * seasonal_factor[target_bin]
```

---

#### Function 6: ForecastRawVolume

```
FUNCTION ForecastRawVolume(model_a, stock, date, current_bin,
                            observed_volumes) -> forecasts[current_bin..I]:
    """
    Produce raw volume forecasts for all remaining bins on the given date.
    Called at market open (current_bin=1, observed_volumes={}) and updated
    intraday as each bin completes.

    This function is PURE: it does not modify model_a's persistent state.
    All conditioning operates on copies.

    Reference: Satish et al. 2014, Exhibit 1 (p.18) data flow diagram.
    """

    forecasts = {}

    # --- Component 1: Historical average (H) ---
    # Pre-computed, does not change intraday.
    # Reference: Satish et al. 2014, p.17, "a rolling historical average"
    #   and Exhibit 1 caption "Prior 21 days"
    H = model_a.H  # array[1..I], computed as of yesterday

    # --- Component 2: Inter-day ARMA forecast (D) ---
    # One-step-ahead prediction for today, per bin. Does not change
    # intraday (depends only on prior days' observations at that bin).
    # Reference: Satish et al. 2014, p.17, "per-symbol, per-bin ARMA" and
    #   Exhibit 1 "Next Bin (Prior 5 days)"
    # [Researcher inference: "Prior 5 days" describes the effective AR
    #  memory depth (p_max=5), not the fitting window. The model is fitted
    #  on N_interday_fit days but only uses the most recent p observations
    #  as AR lags for prediction. See p.18 Exhibit 1.]
    D = {}
    FOR each bin i in current_bin..I:
        IF model_a.interday_models[i] is not None:
            D[i] = model_a.interday_models[i].forecast(steps=1)[0]
        ELSE:
            D[i] = H[i]  # fallback to historical average

    # --- Component 3: Intraday ARMA forecast (A) ---
    # Condition on today's observed deseasonalized volumes, then forecast
    # forward. Forecasts are in deseasonalized space; re-seasonalize.
    #
    # Reference: Satish et al. 2014, p.18, "before passing intraday ARMA
    #   forecasts, we re-seasonalize these forecasts via multiplication"
    # Reference: Exhibit 1, "Current Bin" + "4 Bins Prior to Current Bin"
    #   feed into "ARMA Intraday".
    # [Researcher inference: "4 Bins Prior" refers to the AR context
    #  (p < 5), not a hard window. We pass all observed bins to the
    #  conditioner; only the most recent p affect the AR component, but
    #  passing all bins preserves the MA residual state accumulated from
    #  earlier bins. See Exhibit 1 annotation discussion.]

    # Deseasonalize observed volumes
    observed_deseas = []
    FOR j in 1..current_bin - 1:
        observed_deseas.append(
            observed_volumes[j] / model_a.seasonal_factor[j])

    A = {}
    IF model_a.intraday_model is not None AND len(observed_deseas) > 0:
        # Condition: pure operation returning a new state object.
        # Does NOT mutate model_a.intraday_model.
        # Implementation: reset AR/MA buffers to post-training state,
        # then sequentially process each observed bin:
        #   for each obs in observed_deseas:
        #     pred = compute_prediction(ar_buffer, ma_buffer, coeffs)
        #     resid = obs - pred
        #     shift ar_buffer (append obs, drop oldest if > p)
        #     shift ma_buffer (append resid, drop oldest if > q)
        # This is the "reset-and-reprocess" (stateless) design.
        # Calling condition([1,2,3]) then condition([1,2,3,4]) re-processes
        # bins 1-3 -- no state corruption risk.
        conditioned = model_a.intraday_model.condition(observed_deseas)
        steps_ahead = I - current_bin + 1
        # forecast() is PURE: operates on internal copies of buffers,
        # does NOT modify conditioned's persistent state.
        deseas_fcst = conditioned.forecast(steps=steps_ahead)
        FOR idx, bin_i in enumerate(current_bin..I):
            A[bin_i] = deseas_fcst[idx] * model_a.seasonal_factor[bin_i]
    ELIF model_a.intraday_model is not None AND len(observed_deseas) == 0:
        # Pre-market or first bin: unconditional forecast
        steps_ahead = I - current_bin + 1
        deseas_fcst = model_a.intraday_model.forecast(steps=steps_ahead)
        FOR idx, bin_i in enumerate(current_bin..I):
            A[bin_i] = deseas_fcst[idx] * model_a.seasonal_factor[bin_i]
    ELSE:
        # No intraday model available: fall back to seasonal pattern
        FOR bin_i in current_bin..I:
            A[bin_i] = model_a.seasonal_factor[bin_i]

    # --- Regime classification and weight lookup ---
    regime = ClassifyRegime(
        observed_volumes, current_bin - 1,  # bins observed so far
        model_a.regime_config.cumvol_distribution,
        model_a.regime_config.percentiles,
        min_regime_bins=model_a.min_regime_bins,
        default_regime=model_a.default_regime)

    w = model_a.regime_config.weights[regime]  # (w1, w2, w3)

    # --- Weighted combination ---
    # Reference: Satish et al. 2014, p.17-18, Exhibit 1:
    #   V_hat[i] = w1 * H[i] + w2 * D[i] + w3 * A[i]
    FOR each bin i in current_bin..I:
        raw = w[0] * H[i] + w[1] * D[i] + w[2] * A[i]
        forecasts[i] = max(raw, 0.0)  # non-negative clamp

    RETURN forecasts
```

---

#### Function 7: ForecastVolumePercentage

```
FUNCTION ForecastVolumePercentage(model_b, model_a, stock, date,
                                   current_bin, observed_volumes) -> pct_forecast:
    """
    Predict the volume percentage for the next bin (current_bin + 1).
    Uses Model A to compute volume surprises from observed bins.

    Reference: Satish et al. 2014, pp.18-19, "Volume Percentage Forecast
      Methodology"; extends Humphery-Jenner (2011).
    """

    target_bin = current_bin + 1
    IF target_bin > I:
        RETURN None  # no more bins to predict

    # --- Step 1: Estimate total daily volume ---
    # V_total_est = observed volume + forecasted remaining volume
    # Model A must run FIRST to provide remaining-bin forecasts.
    raw_fcst_conditioned = ForecastRawVolume(
        model_a, stock, date, target_bin, observed_volumes)
    cumulative_observed = sum(observed_volumes[j] for j in 1..current_bin)
    forecasted_remaining = sum(raw_fcst_conditioned[j] for j in target_bin..I)
    V_total_est = cumulative_observed + forecasted_remaining

    IF V_total_est <= 0:
        RETURN model_b.hist_pct[target_bin]  # safety fallback

    # --- Step 2: Check switch-off condition ---
    # Reference: Satish et al. 2014, p.24, referencing Humphery-Jenner (2011):
    #   "once 80% of the day's volume is reached, return to a historical
    #   approach"
    # [Note: in practice, this only activates in the last 2-4 bins because
    #  cumulative volume at earlier bins is well below 80% of the daily total.]
    observed_fraction = cumulative_observed / V_total_est
    IF observed_fraction >= pct_switchoff:
        # Switch off dynamic adjustment: return scaled historical percentage
        remaining_frac = 1.0 - observed_fraction
        remaining_hist = sum(model_b.hist_pct[j] for j in target_bin..I)
        IF remaining_hist > 0:
            RETURN model_b.hist_pct[target_bin] * remaining_frac / remaining_hist
        ELSE:
            RETURN remaining_frac  # last bin edge case

    # --- Step 3: Compute surprises from observed bins ---
    # Use Model A's unconditional forecasts (current_bin=1) as the surprise
    # baseline. This gives a consistent baseline across all observed bins.
    # Reference: Satish et al. 2014, p.19, "we could apply our more
    #   extensive volume forecasting model" as the surprise base.
    # [Researcher inference: the paper uses "could" (aspirational), so the
    #  published results may have used simpler historical averages. We use
    #  Model A as primary, consistent with the paper's stated direction.
    #  A configuration flag could fall back to historical averages.]
    raw_fcst_unconditional = ForecastRawVolume(
        model_a, stock, date, current_bin=1, observed_volumes={})

    surprises = {}
    FOR j in 1..current_bin:
        expected = raw_fcst_unconditional[j]
        actual = observed_volumes[j]
        # Surprise in percentage space:
        #   surprise = actual_pct - expected_pct
        #            = actual/V_total_est - expected/V_total_est
        #            = (actual - expected) / V_total_est
        # [Researcher inference: computing surprise in percentage space
        #  (both terms share V_total_est as denominator) ensures the
        #  regression output (delta) is in the same units as the percentage
        #  forecast. Alternative: relative surprise = (actual-expected)/expected.
        #  We use percentage-space because it aligns with the regression target.]
        IF V_total_est > 0:
            surprises[j] = (actual - expected) / V_total_est
        ELSE:
            surprises[j] = 0.0

    # --- Step 4: Apply surprise regression ---
    # Predict delta from lagged surprises
    # Reference: Satish et al. 2014, p.18-19, surprise-based rolling regression
    L = model_b.optimal_L

    IF current_bin < L:
        # Insufficient surprise lags: fall back to no adjustment
        # [Researcher inference: paper does not address early-bin case.
        #  Zero-padding produces same result: delta = 0 when all lags are 0.]
        delta = 0.0
    ELSE:
        # Construct lag vector: most recent L surprises
        lag_vector = []
        FOR lag in 1..L:
            lag_bin = current_bin - lag + 1
            IF lag_bin >= 1:
                lag_vector.append(surprises[lag_bin])
            ELSE:
                lag_vector.append(0.0)  # zero-pad if before bin 1
        delta = dot(model_b.beta, lag_vector)

    # --- Step 5: Scale historical percentage to remaining fraction ---
    # The historical percentage applies to a full day. We need to rescale
    # the target bin's percentage to account for the fraction already consumed.
    remaining_frac = 1.0 - observed_fraction
    remaining_hist = sum(model_b.hist_pct[j] for j in target_bin..I)
    IF remaining_hist > 0:
        scaled_base = model_b.hist_pct[target_bin] * remaining_frac / remaining_hist
    ELSE:
        scaled_base = remaining_frac if target_bin == I else model_b.hist_pct[target_bin]

    # --- Step 6: Apply deviation constraint ---
    # Reference: Satish et al. 2014, p.24, "depart no more than 10% away
    #   from a historical VWAP curve"; Humphery-Jenner (2011).
    # [Researcher inference: the paper says "we developed a separate method
    #  for computing the deviation bounds" (p.19), meaning the production
    #  system used adaptive/proprietary bounds. The 10% is a simplification
    #  from Humphery-Jenner. max_deviation is a primary tuning candidate.]
    max_delta = max_deviation * scaled_base
    delta = clip(delta, -max_delta, max_delta)

    # --- Step 7: Compute adjusted percentage ---
    adjusted_pct = scaled_base + delta
    adjusted_pct = max(adjusted_pct, 0.0)  # non-negative

    # --- Step 8: Handle last-bin edge case ---
    # [Researcher inference: when target_bin == I, there are no subsequent
    #  bins to redistribute to. The forecast is the entire remaining fraction.]
    IF target_bin == I:
        pct_forecast = remaining_frac
    ELSE:
        pct_forecast = adjusted_pct

    RETURN pct_forecast
```

---

#### Function 8: TrainPercentageRegression

```
FUNCTION TrainPercentageRegression(volume_data, stock, reference_date,
                                    N_regression_fit, L_max, model_a,
                                    hist_pct) -> (beta, optimal_L):
    """
    Train the surprise regression for Model B using historical data.
    For each training day, compute Model A's dynamic forecasts (conditioned
    on observed bins) and derive surprises, then regress percentage deviations
    on lagged surprises.

    Reference: Satish et al. 2014, pp.18-19, "volume surprises based on
      a naive volume forecast model can be used to train a rolling regression
      model that adjusts market participation"
    Reference: Satish et al. 2014, p.19, "identifying the optimal number
      of model terms for U.S. equities" (value not disclosed)
    """

    # --- Step 1: Build training data ---
    # For each historical day and each bin, compute:
    #   - The surprise at that bin (using dynamic Model A forecasts)
    #   - The percentage deviation from historical (actual_pct - hist_pct)
    #
    # Design decision: dynamic vs static forecasts for training
    # [Researcher inference: training should use DYNAMIC forecasts
    #  (conditioned on bins 1..j-1 for target bin j) to match prediction-time
    #  behavior. Static (market-open) forecasts create a train/predict
    #  mismatch because they lack intraday conditioning. The cost is
    #  higher: ~N_reg * I * I/2 ForecastRawVolume calls per symbol,
    #  but this is manageable with vectorized ARMA code.]

    all_X = []  # list of lag vectors (length L_max)
    all_y = []  # list of percentage deviations

    FOR each day d in (reference_date - N_regression_fit + 1)..reference_date:
        IF NOT is_full_trading_day(d):
            CONTINUE

        daily_total = sum(volume_data[stock, bin=k, day=d] for k in 1..I)
        IF daily_total <= 0:
            CONTINUE

        # Pre-compute dynamic surprises for all bins of this day
        # At each bin j, condition Model A on bins 1..j-1
        day_surprises = {}
        FOR j in 1..I:
            observed_so_far = {k: volume_data[stock, bin=k, day=d] for k in 1..j-1}
            # Dynamic raw forecast conditioned on observed bins
            raw_fcst = ForecastRawVolume(model_a, stock, d, j, observed_so_far)
            expected = raw_fcst[j]
            actual = volume_data[stock, bin=j, day=d]

            # Surprise in percentage space using daily total as denominator
            # [Researcher inference: during training we use the actual daily
            #  total as denominator (known in hindsight). During prediction,
            #  V_total_est is used instead. This creates a small statistical
            #  mismatch that diminishes late in the day as V_total_est
            #  converges to the actual total. The mismatch is acceptable
            #  because regression coefficients are small and the deviation
            #  constraint limits impact.]
            day_surprises[j] = (actual - expected) / daily_total

        # Build (X, y) pairs: for each bin j >= 2, regress percentage
        # deviation on the L_max most recent surprises
        FOR j in 2..I:
            # Response: actual percentage minus historical percentage
            actual_pct = volume_data[stock, bin=j, day=d] / daily_total
            y_val = actual_pct - hist_pct[j]

            # Predictor: lagged surprises [surprise[j-1], ..., surprise[j-L_max]]
            x_val = []
            FOR lag in 1..L_max:
                lag_bin = j - lag
                IF lag_bin >= 1:
                    x_val.append(day_surprises[lag_bin])
                ELSE:
                    x_val.append(0.0)  # zero-pad for early bins
            all_X.append(x_val)
            all_y.append(y_val)

    X_matrix = array(all_X)  # shape: (n_samples, L_max)
    y_vector = array(all_y)  # shape: (n_samples,)

    # --- Step 2: Select optimal number of lag terms ---
    # Use time-series cross-validation (blocked by day) to avoid
    # temporal leakage.
    # [Researcher inference: the paper says they "identified the optimal
    #  number of model terms" (p.19) but does not disclose the value or
    #  the selection method. Time-series CV is standard and respects
    #  temporal ordering, unlike random K-fold which would create leakage
    #  from within-day correlations.]
    best_L = 1
    best_cv_error = +infinity
    FOR L in 1..L_max:
        # Blocked time-series cross-validation: split at day boundaries
        # Expanding window: train on first k days, validate on day k+1
        cv_errors = []
        # ... (standard expanding-window CV implementation)
        cv_error = mean(cv_errors)
        IF cv_error < best_cv_error:
            best_cv_error = cv_error
            best_L = L

    # --- Step 3: Fit final OLS regression with optimal L, no intercept ---
    # Reference: Satish et al. 2014, p.19, "we perform both regressions
    #   without the inclusion of a constant term"
    # [Researcher inference: the p.19 quote refers to VWAP-error validation
    #  regressions, not the surprise regression. We apply the no-intercept
    #  principle by analogy: when all recent surprises are zero, the
    #  adjustment delta should be zero. This is a natural constraint for
    #  a surprise-based correction model.]
    X_final = X_matrix[:, 0:best_L]
    beta = ols_no_intercept(X_final, y_vector)
    # beta = (X^T X)^{-1} X^T y, computed via pseudoinverse for stability

    RETURN (beta, best_L)
```

---

#### Function 9: DailyOrchestration

```
FUNCTION DailyOrchestration(stock, date, model_a, model_b, market_data_feed):
    """
    Top-level daily workflow: initialize at market open, update at each bin,
    produce forecasts, and close at end of day.

    This function describes the complete interaction protocol between
    Models A and B during a trading day.

    [Researcher inference: the paper does not provide an explicit orchestration
     procedure. This function synthesizes the interaction from the descriptions
     of Models A and B and Exhibit 1's data flow.]
    """

    # === PRE-MARKET (before 9:30 ET) ===
    # Step 1: Update inter-day ARMA state with yesterday's observed volumes
    FOR each bin i in 1..I:
        IF model_a.interday_models[i] is not None:
            yesterday_vol = volume_data[stock, bin=i, day=date-1]
            UpdateInterDayState(model_a.interday_models[i], yesterday_vol)
            # UpdateInterDayState:
            #   1. Compute prediction error: resid = yesterday_vol - predicted
            #   2. Shift AR buffer: append yesterday_vol, drop oldest if > p
            #   3. Shift MA buffer: append resid, drop oldest if > q
            #   4. Next call to forecast() uses fixed phi/theta with updated buffers
            # This is a STATE UPDATE, not re-estimation (parameters unchanged).
            # Full re-estimation (new MLE, possibly new order) happens on
            # the weekly schedule.

    # Step 2: Produce pre-market (unconditional) forecasts
    pre_market_forecasts = ForecastRawVolume(
        model_a, stock, date, current_bin=1, observed_volumes={})

    observed_volumes = {}

    # === INTRADAY LOOP (9:30-16:00 ET, 26 bins) ===
    FOR current_bin in 1..I:
        # Step 3: Observe actual volume for this bin
        actual_volume = market_data_feed.get_volume(stock, date, current_bin)
        observed_volumes[current_bin] = actual_volume

        # Step 4: Update Model A forecasts for remaining bins
        # Conditioning happens inside ForecastRawVolume via the
        # intraday ARMA's condition() method (pure, stateless).
        IF current_bin < I:
            updated_forecasts = ForecastRawVolume(
                model_a, stock, date, current_bin + 1, observed_volumes)

            # Step 5: Produce Model B percentage forecast for next bin
            pct_forecast = ForecastVolumePercentage(
                model_b, model_a, stock, date, current_bin, observed_volumes)

            # Step 6: Output forecasts to execution algorithms
            EMIT(stock, date, current_bin,
                 raw_forecasts=updated_forecasts,
                 next_bin_pct=pct_forecast)

    # === END OF DAY ===
    # Step 7: Record today's data for tomorrow's model updates
    # Inter-day state update happens in next day's pre-market step.
    # No additional end-of-day processing needed for Model A.


FUNCTION UpdateInterDayState(interday_model, new_observation):
    """
    Update inter-day ARMA model state with a new daily observation
    WITHOUT re-estimating parameters. The AR and MA coefficients
    (phi, theta, constant) remain fixed; only the internal buffers
    (recent observations and residuals) are updated.

    [Researcher inference: implied by daily ARMA forecasting in
     Satish et al. 2014, p.17. The model must consume each new day's
     observation to produce an updated forecast for the next day.
     Full re-estimation (new MLE, potentially new order selection)
     happens on the weekly cycle.]
    """
    # 1. Compute one-step-ahead prediction using current buffers
    predicted = interday_model.predict_next()
    residual = new_observation - predicted

    # 2. Update AR observation buffer
    interday_model.ar_buffer.append(new_observation)
    IF len(interday_model.ar_buffer) > interday_model.p:
        interday_model.ar_buffer.pop(0)  # drop oldest

    # 3. Update MA residual buffer
    interday_model.ma_buffer.append(residual)
    IF len(interday_model.ma_buffer) > interday_model.q:
        interday_model.ma_buffer.pop(0)  # drop oldest
```

---

#### Function 10: ReEstimationSchedule

```
FUNCTION ReEstimationSchedule(stock, current_date, model_a, model_b,
                               volume_data, last_refit_dates):
    """
    Manage the re-estimation schedule for all model components.
    Called daily before market open.

    [Researcher inference: the paper does not discuss re-estimation
     frequency. This tiered schedule balances computational cost with
     model freshness. Daily updates are lightweight (seasonal factors,
     H, intraday ARMA). Weekly updates are moderate (inter-day ARMA
     re-estimation). Monthly updates are expensive (regime grid search,
     weight optimization, percentage regression).]
    """

    # --- DAILY re-estimation ---
    # These are cheap and benefit from fresh data
    model_a.seasonal_factor = ComputeSeasonalFactors(
        volume_data, stock, current_date - 1, N_seasonal)
    FOR each bin i in 1..I:
        model_a.H[i] = compute_H_asof(
            volume_data, stock, i, current_date, N_hist)
    model_a.intraday_model = FitIntraDayARMA(
        volume_data, stock, current_date - 1, N_intraday_fit,
        p_max_intraday, q_max_intraday, model_a.seasonal_factor)
    # Note: hist_pct for Model B also updated daily
    model_b.hist_pct = compute_hist_pct(volume_data, stock, current_date - 1, N_hist)

    # --- WEEKLY re-estimation (every 5 trading days) ---
    # Full ARMA re-estimation with AICc (may change order)
    IF trading_days_since(last_refit_dates['interday']) >= 5:
        model_a.interday_models = FitInterDayARMA(
            volume_data, stock, current_date - 1, N_interday_fit,
            p_max_interday, q_max_interday, model_a.seasonal_factor)
        last_refit_dates['interday'] = current_date
    # Between weekly re-estimations, inter-day ARMA gets daily STATE
    # UPDATES (UpdateInterDayState in DailyOrchestration) with fixed
    # parameters but updated buffers.

    # --- MONTHLY re-estimation (every 21 trading days) ---
    # These are computationally expensive
    IF trading_days_since(last_refit_dates['regime']) >= 21:
        model_a.regime_config = OptimizeRegimeWeights(
            volume_data, stock, current_date - 1, N_regime_window,
            model_a.interday_models, model_a.intraday_model,
            model_a.seasonal_factor, model_a.H, regime_candidates)
        last_refit_dates['regime'] = current_date

    IF trading_days_since(last_refit_dates['regression']) >= 21:
        model_b.beta, model_b.optimal_L = TrainPercentageRegression(
            volume_data, stock, current_date - 1, N_regression_fit,
            L_max, model_a, model_b.hist_pct)
        last_refit_dates['regression'] = current_date
```

### Data Flow

```
Input: Historical volume[stock, bin, day] matrix (split-adjusted)
  |
  v
[ComputeSeasonalFactors] -- trailing 126 days, per bin
  |                                |
  v                                v
[Historical Avg H]      [Deseasonalize] --> [FitIntraDayARMA]
  |     per-bin mean         divide            single model/stock
  |     trailing 21d         by SF             trailing 21d
  |                                              |
  |    [FitInterDayARMA]                         |
  |     26 models, one per bin                   |
  |     trailing 126d                            |
  |         |                                    |
  v         v                                    v
H[1..I]  D[1..I]                             A[1..I]
  |     (one-step-ahead                   (multi-step forecast,
  |      per-bin forecast)                 re-seasonalized)
  |         |                                    |
  +----+----+------------------------------------+
       |
       v
[ClassifyRegime] -- cumvol percentile -> regime index r
       |
       v
[Weight Lookup] -- weights[r] = (w1, w2, w3)
       |
       v
[Weighted Combination] -- V_hat[i] = w1*H[i] + w2*D[i] + w3*A[i]
       |                   clamp >= 0
       v
  Model A Output: V_hat[i] for i in current_bin..I
       |
       +----> [V_total_est] = observed + sum(remaining V_hat)
       |           |
       |           v
       +----> [Surprise Computation] -- surprise[j] = (actual[j] - V_hat_uncond[j]) / V_total_est
                    |
                    v
              [Surprise Regression] -- delta = beta . [surprise lags]
                    |                   OLS, no intercept
                    v
              [Deviation Clipping] -- |delta| <= max_deviation * scaled_base
                    |
                    v
              [Scale + Adjust] -- scaled_base + clipped delta
                    |
                    v
              [Switch-off Check] -- if cum_obs / V_total_est >= 80%: use historical
                    |
                    v
              Model B Output: pct_forecast for next bin
```

**Shapes and types at each step:**

| Step | Variable | Shape | Type | Unit |
|------|----------|-------|------|------|
| Input | volume[s, d, i] | scalar | float | shares |
| Seasonal | seasonal_factor[i] | (I,) = (26,) | float | shares |
| Hist Avg | H[i] | (26,) | float | shares |
| Inter-day | D[i] | (26,) | float | shares |
| Intraday deseas | deseasonalized series | (N_intraday_fit * I,) | float | dimensionless ratio |
| Intraday reseas | A[i] | (26,) | float | shares |
| Regime | regime_index | scalar | int | 0..n_regimes-1 |
| Weights | w | (3,) | float | dimensionless |
| Raw forecast | V_hat[i] | (26,) | float | shares (>= 0) |
| V_total_est | V_total_est | scalar | float | shares |
| Surprise | surprises[j] | scalar | float | fraction |
| Lag vector | lag_vector | (L,) | float | fraction |
| Regression coef | beta | (L,) | float | dimensionless |
| Delta | delta | scalar | float | fraction |
| Scaled base | scaled_base | scalar | float | fraction (0-1) |
| Pct forecast | pct_forecast | scalar | float | fraction (0-1) |

### Variants

This specification implements the full dual-model system from Satish et al. (2014), which is the most complete variant described in the paper:

- **Raw Volume Model:** All three components (historical average, inter-day ARMA, intraday ARMA) plus regime-dependent weight overlay. The paper evaluates the full model against the historical-average baseline; no simpler variant is formally presented. [Satish et al. 2014, pp.17-18]

- **Volume Percentage Model:** The extended dynamic VWAP approach with Model A's raw forecasts as the surprise baseline. The paper mentions using simpler historical averages as an alternative surprise baseline ("departures from a historical average approach," p.18) but recommends the Model A approach: "we could apply our more extensive volume forecasting model" (p.19). [Researcher inference: "could" is aspirational, not confirmative -- the published results may have used either baseline. We implement Model A as primary with a configuration option to fall back to historical averages.]

- **Not implemented:** The ARMAX variant with calendar event exogenous inputs. The paper explicitly advises against it: "there is scant historical data represented by the special cases to construct such formal models reliably" (p.18). Custom curves for special days are handled as a lookup table, not a model.

## Parameters

| Parameter | Description | Recommended Value | Sensitivity | Range |
|-----------|-------------|-------------------|-------------|-------|
| I | Number of intraday bins (15-min) | 26 | N/A (fixed) | {13, 26, 78} |
| N_hist | Days for historical rolling average (Component 1) | 21 | Medium-High | [10, 60] |
| N_seasonal | Days for seasonal factor computation | 126 | Low | [63, 252] |
| N_intraday_fit | Days for intraday ARMA fitting window | 21 | Medium | [15, 42] |
| N_interday_fit | Days for inter-day ARMA fitting window | 126 | Medium | [63, 252] |
| p_max_interday | Maximum AR order for inter-day ARMA | 5 | Low | [3, 5] |
| q_max_interday | Maximum MA order for inter-day ARMA | 5 | Low | [3, 5] |
| p_max_intraday | Maximum AR order for intraday ARMA | 4 | Low | [2, 4] |
| q_max_intraday | Maximum MA order for intraday ARMA | 5 | Low | [3, 5] |
| n_regimes | Number of regime buckets | Grid search {3, 4, 5} | High | [2, 6] |
| regime_percentiles | Percentile cutoffs for regime assignment | Equally spaced | High | Data-driven |
| N_regime_window | Days for building cumvol percentile distribution | 63 | Medium | [21, 126] |
| min_regime_bins | Minimum observed bins before regime classification | 3 | Low | [1, 5] |
| default_regime | Regime index for early bins / pre-market | 1 (middle tercile) | Low | [0, n_regimes-1] |
| min_samples_per_regime | Min training samples per regime for weight optimization | 50 | Low | [20, 100] |
| N_regression_fit | Days for surprise regression training | 63 | Medium | [21, 126] |
| L_max | Maximum number of surprise lag terms | 5 | Medium | [3, 7] |
| max_deviation | Maximum proportional deviation from scaled historical pct | 0.10 (10%) | Medium-High | [0.05, 0.20] |
| pct_switchoff | Cumulative volume fraction to switch off dynamic model | 0.80 (80%) | Medium | [0.70, 0.90] |
| min_volume_floor | Minimum volume for MAPE/surprise computation (shares) | 100 | Low | [50, 500] |

### Parameter Sources

- **I = 26:** Satish et al. 2014, p.16, "26 such bins in a trading day."
- **N_hist = 21:** Satish et al. 2014, Exhibit 1 caption, "Prior 21 days." Note: this appears in the diagram annotation, not the methodology text. The methodology (p.16) introduces N as "a variable that we shall call N" without disclosing its value. The 21-day value may be illustrative. Sensitivity upgraded to Medium-High to encourage testing alternatives.
- **N_seasonal = 126:** Satish et al. 2014, p.17, "trailing six months." 126 trading days ~ 6 calendar months.
- **N_intraday_fit = 21:** Satish et al. 2014, p.18, "rolling basis over the most recent month." 21 trading days ~ 1 calendar month.
- **N_interday_fit = 126:** [Researcher inference] The paper says "Prior 5 days" in Exhibit 1, which refers to the effective AR memory depth (p_max=5), not the fitting window. Fitting an ARMA(5,5) on only 5 observations is statistically infeasible (11 parameters for 5 data points). A 126-day window (6 months) provides sufficient data for stable estimation and aligns with the seasonal factor window. Range: [63, 252].
- **p_max_interday = 5, q_max_interday = 5:** Satish et al. 2014, p.17, "we consider all values of p and q lags through five."
- **p_max_intraday = 4:** Satish et al. 2014, p.18, "AR lags with a value less than five." The constraint "less than five" applies to AR order only (p in {0,1,2,3,4}). The paper does not constrain MA order for the intraday model.
- **q_max_intraday = 5:** [Researcher inference] Paper constrains only AR ("less than five"), not MA, for intraday. We set q_max = 5 for consistency with inter-day. AICc penalizes excess orders.
- **n_regimes:** [Researcher inference] Paper says "training several weight models" (p.18) but does not disclose the number of regimes. Grid search over {3, 4, 5} is recommended because the optimal number depends on the stock universe and time period.
- **max_deviation = 0.10:** Satish et al. 2014, p.24, "depart no more than 10% away from a historical VWAP curve," referencing Humphery-Jenner (2011). However, the paper also states "we developed a separate method for computing the deviation bounds" (p.19), meaning the production system used adaptive/proprietary bounds, NOT the fixed 10%. The 10% is a simplification; max_deviation is a primary tuning candidate.
- **pct_switchoff = 0.80:** Satish et al. 2014, p.24, "once 80% of the day's volume is reached, return to a historical approach," referencing Humphery-Jenner (2011).
- **N_regression_fit = 63:** [Researcher inference] 3-month window provides ~63 * 25 = 1575 training samples (pooled across bins starting from bin 2 and all training days) for the low-dimensional OLS. Adapts faster to changing market conditions than a longer window. The paper does not disclose this value.
- **min_volume_floor = 100:** [Researcher inference] Prevents division-by-zero instability in MAPE and surprise computation. Not in the paper.

### Initialization

1. **Data preparation:** Load split-adjusted intraday volume data at 15-minute resolution. Exclude half-day trading sessions (13 bins instead of 26) from training. Exclude days with zero total volume. Ensure data is adjusted for stock splits and corporate actions (use split-adjusted share counts from the data vendor, or normalize by daily shares outstanding as in Chen et al. 2016, Section 4.1). [Researcher inference on split adjustment]

2. **Seasonal factors:** ComputeSeasonalFactors from trailing N_seasonal = 126 days. If fewer than 63 days available, use all available data (minimum 63 trading days required for meaningful seasonal estimation). Replace zero seasonal factors with the minimum non-zero value across all bins for that stock.

3. **Historical averages:** Compute H[i] from trailing N_hist = 21 days for each bin. Compute hist_pct[i] similarly for percentage model.

4. **Inter-day ARMA:** FitInterDayARMA: 26 separate models (one per bin) on trailing N_interday_fit = 126 days. If a model fails to converge for all (p, q) combinations at a bin, fall back to H[i] for that bin.

5. **Intraday ARMA:** FitIntraDayARMA: single model on concatenated deseasonalized intraday volume from trailing N_intraday_fit = 21 days, with day-boundary likelihood breaks. Save the post-training AR/MA buffer state for use in condition() reset-and-reprocess.

6. **Regime configuration:** OptimizeRegimeWeights: grid search over regime candidates using trailing N_regime_window = 63 days, with last 21 days held out for validation. Per-regime weight optimization via constrained MSE minimization (SLSQP with simplex constraint).

7. **Percentage regression:** TrainPercentageRegression: compute dynamic Model A forecasts for trailing N_regression_fit = 63 days, derive surprises, fit OLS (no intercept) with time-series cross-validated lag count.

### Calibration

All parameters are calibrated from historical data. No manual tuning is required for initial deployment, though monitoring and periodic adjustment of max_deviation and pct_switchoff are recommended.

1. **ARMA model selection:** Automatic via AICc grid search over (p, q) combinations. No manual tuning required. [Satish et al. 2014, p.17]

2. **Regime count and boundaries:** Grid search over n_regimes in {3, 4, 5} with equally-spaced percentile cutoffs. Select configuration minimizing out-of-sample MAPE on held-out validation period. [Researcher inference]

3. **Component weights:** Per-regime optimization minimizing MSE with simplex constraint (w1+w2+w3=1, all >= 0), using scipy SLSQP. Variant: MAPE minimization with variable transformation (w_raw = exp(w_log)) and Nelder-Mead. Switch to MAPE variant if it produces lower out-of-sample MAPE. [Researcher inference; paper says "minimizes the error" (p.18) but does not specify loss function or optimizer]

4. **Surprise regression lag count:** Time-series cross-validation (expanding window, blocked by day) over L in {1, 2, ..., L_max}. Select L minimizing cross-validated MAE of percentage deviation predictions. [Researcher inference; paper says they "identified the optimal number of model terms" (p.19) but does not disclose it or the selection method]

5. **Re-estimation schedule:**
   - Daily: seasonal factors, H, hist_pct, intraday ARMA (full re-fit).
   - Daily (state update only): inter-day ARMA buffers (UpdateInterDayState).
   - Weekly (every 5 trading days): inter-day ARMA full re-estimation (new MLE, may change order).
   - Monthly (every 21 trading days): regime grid search + weight optimization, percentage regression.
   [Researcher inference; paper does not discuss re-estimation frequency]

## Validation

### Expected Behavior

**Model A (Raw Volume):**
- Median MAPE reduction vs. historical-only baseline: 24% across all intraday intervals. [Satish et al. 2014, p.20, "we reduce the median volume error by 24%"]
- Bottom-95% mean MAPE reduction: 29%. [Satish et al. 2014, p.20]
- Error reduction varies by time of day: ~10% at 9:30 (bin 1) increasing to ~33% at 15:30 (bin 25). Bottom-95% profile increases more smoothly from ~15% to ~40%. [Satish et al. 2014, Exhibit 6, p.22]
- Consistent across SIC industry groups (~15-35% reduction). [Satish et al. 2014, Exhibit 7, p.22]
- Consistent across beta deciles (~20-35% reduction). [Satish et al. 2014, Exhibit 8, p.23]

**Model B (Volume Percentage):**
- Median absolute deviation: 0.00874 (historical) vs. 0.00808 (dynamic) -- 7.55% reduction, significant at << 1% level via Wilcoxon signed-rank test. [Satish et al. 2014, Exhibit 9, p.23]
- Bottom-95% average absolute error: 0.00986 vs. 0.00924 -- 6.29% reduction. [Satish et al. 2014, Exhibit 9, p.23]
- 5-minute bins: 2.25% median reduction. 30-minute bins: 2.95% median reduction. 15-minute bins show largest improvement. [Satish et al. 2014, Exhibit 9, p.23]
- Note: the percentage error metric is MAD (mean absolute deviation), NOT MAPE. Percentage predictions are already normalized (sum to ~1), so no division by actual is needed. Formula: Error = (1/I) * sum_i |Predicted_Percentage_i - Actual_Percentage_i|. [Satish et al. 2014, p.17, "Measuring Percentage Volume Predictions -- Absolute Deviation"]

**VWAP Tracking Error:**
- Mean tracking error reduced from 9.62 bps (historical) to 8.74 bps (dynamic) -- 9.1% reduction, p < 0.01 via paired t-test (t = 2.34). Std dev: 11.18 bps (historical) vs. 10.08 bps (dynamic). [Satish et al. 2014, Exhibit 10, p.23]
- 7-10% reduction within each stock category (Dow 30, midcap, high-variance). [Satish et al. 2014, p.23]
- Simulation: 600+ day-long VWAP orders, order size 10% of 30-day ADV. [Satish et al. 2014, p.23]

**Benchmark comparison (different dataset/period):**
- Chen et al. (2016) robust Kalman filter: average MAPE 0.46, VWAP tracking 6.38 bps on 30 securities. [Chen et al. 2016, Tables 3-4]
- Chen et al. (2016) rolling mean baseline: average MAPE 1.28, VWAP 7.48 bps. [Chen et al. 2016, Table 3]
- Direct comparison is imprecise: different datasets (500 vs. 30 securities), time periods, exchanges, and MAPE normalization conventions (Satish uses percentage, Chen uses fraction). [Researcher inference]

### Sanity Checks

1. **Historical-only baseline reproduction:** With w1=1.0, w2=0.0, w3=0.0 for all regimes, Model A should exactly reproduce the historical rolling average. Verify MAPE matches the baseline.

2. **Seasonal U-shape:** Plot seasonal_factor[i] for i=1..26. Should show elevated volume at bins 1-2 (9:30-10:00) and bins 24-26 (15:00-16:00), with a trough around bins 10-16 (midday). [Standard intraday volume pattern; Satish et al. 2014, p.17]

3. **Deseasonalized stationarity:** Run ADF test on the deseasonalized intraday volume series. Should reject unit root null at p < 0.05. If non-stationary, seasonal factor window or deseasonalization may need adjustment.

4. **ARMA order distribution:** Across all 26 inter-day ARMA models for a stock, selected (p, q) orders should predominantly be low: most models p+q <= 4. "AR coefficients quickly decayed" [Satish et al. 2014, p.18]. If many select p=5 or q=5, AICc penalty may be miscalibrated.

5. **Combined term count:** For each stock, compute (max interday terms across bins) + (intraday terms). Should be fewer than 11 for most stocks. If not, log a warning. [Satish et al. 2014, p.18, observed outcome]

6. **Weight non-negativity and sum-to-1:** All optimized weights should satisfy w1, w2, w3 >= 0 and w1+w2+w3 = 1 (within numerical tolerance) for every regime.

7. **Regime bucket population:** Each regime should contain at least 5% of total (day, bin) training observations. Nearly empty regimes produce unreliable weights.

8. **Deviation constraint binding frequency:** The max_deviation constraint should bind (clip delta) on no more than 10-20% of bin forecasts. If it binds much more frequently, regression coefficients may be too large, or max_deviation too tight.

9. **Switch-off activation timing:** The 80% switch-off should activate for the last 2-4 bins on a typical day. If it never activates, V_total_est may be systematically too high. If it activates before bin 20, V_total_est may be too low.

10. **Cumulative percentage monotonicity:** Sum of observed + forecasted percentages for bins 1..j should increase monotonically and approach 1.0 as j approaches I.

11. **Surprise coefficient signs:** For L=1, beta[0] should be positive (positive surprise at bin j implies positive adjustment at bin j+1). Typical magnitude: |beta[0]| < 0.5.

12. **Surprise distribution:** Average surprise across all (day, bin) pairs should be approximately zero (unbiased model). Std dev of surprises typically in [0.005, 0.015] for percentage-space surprises.

13. **Monotonic improvement with components:** Adding each component should reduce out-of-sample MAPE: H only < H + D < H + D + A (approximately). If adding a component worsens MAPE, its weight should be driven to near-zero by the optimizer.

14. **Inter-day ARMA state consistency:** After UpdateInterDayState with a new observation, the next forecast should differ from the prior forecast. If it does not change, the state update may not be working.

15. **Intraday ARMA multi-step convergence:** For a deseasonalized multi-step forecast far beyond observed bins (e.g., 20+ steps ahead), the forecast should converge toward the unconditional mean (~1.0 in deseasonalized space), re-seasonalizing to the seasonal factor. This is expected ARMA behavior, not a bug.

16. **Intraday ARMA conditioning consistency:** Calling condition([1,2,3]) followed by condition([1,2,3,4]) should produce the same result as a single call to condition([1,2,3,4]). The reset-and-reprocess design guarantees this.

17. **Evaluation MAPE formula:** MAPE = 100% * (1/N) * SUM_i(|Predicted_Volume_i - Raw_Volume_i| / Raw_Volume_i). Exclude bins where Raw_Volume_i < min_volume_floor. Aggregation hierarchy: per-symbol-day MAPE -> per-symbol MAPE (mean across days) -> cross-symbol median and bottom-95% average. [Satish et al. 2014, p.17]

### Edge Cases

1. **ARMA convergence failure:** If an inter-day ARMA fails to converge for all (p, q) candidates at a given bin, fall back to H[i]. If the intraday ARMA fails, fall back to seasonal factors. Expected failure rate: 1-5% of per-bin fits across a 500-stock universe. Detection: check optimizer convergence flag, verify finite standard errors, confirm stationarity and invertibility. [Researcher inference]

2. **Insufficient surprise lags (early bins):** At bins 1 through L, fewer than L lagged surprises are available. Pad missing lags with 0.0. This produces delta = 0 (no adjustment), equivalent to falling back to scaled historical percentage. Matches prediction-time behavior. [Researcher inference]

3. **Half-day trading sessions:** Days before certain holidays have 13 bins instead of 26. Exclude from training and evaluation. Maintain a calendar marking half-days. [Researcher inference; paper uses full days only]

4. **Special calendar days:** Option expiration, Fed announcements, index rebalancing. "Custom curves for special calendar days... rather than ARMAX models, due to insufficient historical occurrences" (Satish et al. 2014, p.18). Maintain a calendar lookup; substitute pre-computed curves for these days and bypass the dynamic model.

5. **Zero-volume bins:** Seasonal factor floor prevents division by zero in deseasonalization. Bins with actual volume < min_volume_floor are excluded from MAPE computation. Raw forecasts are clamped at zero. [Researcher inference]

6. **Multi-step intraday ARMA degradation:** Recursive multi-step ARMA predictions degrade with forecast horizon. After p steps, the AR component uses only past forecasts; after q steps, MA contributes nothing. For distant bins, the forecast converges to the unconditional mean (~1.0 deseasonalized), re-seasonalizing to the seasonal factor. Paper acknowledges: "techniques that predict only the next interval will perform better than those attempting to predict volume percentages for the remainder of the trading day" (Satish et al. 2014, p.18). Regime weights should naturally adapt -- the intraday component gets less weight for distant bins. [Satish et al. 2014, p.18]

7. **Regime transition at start of day:** No cumulative volume observed at bin 1. Default to middle regime (default_regime = 1 for n_regimes = 3). Regime classification becomes reliable after min_regime_bins (default 3, ~45 minutes). [Researcher inference]

8. **Stock splits / corporate actions:** Volume data must be split-adjusted. A 2:1 split doubles apparent volume. Use split-adjusted volume from the data vendor, or normalize by daily shares outstanding as in Chen et al. (2016, Section 4.1). [Researcher inference]

9. **Last-bin edge case in Model B:** When target_bin == I (predicting last bin), there are no subsequent bins to redistribute. The percentage forecast is the entire remaining fraction (1.0 minus cumulative observed fraction). [Researcher inference]

10. **Lookahead bias in percentage model training:** Model A is trained on data through train_end_date and used to compute forecasts for earlier days d < train_end_date. Bias is negligible: ARMA parameters on 126+ days change minimally when dropping one day, and seasonal factors are 126-day averages. To eliminate entirely: use expanding-window re-estimation (multiplies computational cost by N_regression_fit). [Researcher inference]

11. **Train/predict denominator mismatch in surprises:** During prediction, surprises use V_total_est (observed + forecasted remaining) as denominator, which is noisy and evolves intraday. During training, surprises use the exact daily total (known in hindsight). This creates a statistical mismatch that diminishes late in the day. Two approaches: (a) accept mismatch (implemented -- regression coefficients are small, deviation constraint limits impact); (b) use leave-future-out estimated totals during training (eliminates mismatch but multiplies training compute by I). [Researcher inference]

12. **Pre-market regime handling (current_bin=0):** When no bins are observed (pre-market forecasts), min_regime_bins check forces default_regime for all bins. Pre-market Model A forecasts are therefore not regime-adapted. [Researcher inference]

### Known Limitations

1. **Point forecasts only:** Unlike CMEM or Kalman filter models, this approach does not specify a noise distribution and cannot produce prediction intervals or density forecasts. Volume distributions are positively skewed with heavy tails, but the ARMA framework assumes linear dynamics. A log-transformation (as Chen et al. 2016 use) could reduce skewness, but the paper does not describe this step. [Researcher inference]

2. **No outlier robustness:** Unlike Chen et al.'s robust Kalman filter with Lasso regularization, no built-in outlier handling. Outlier bins directly affect historical averages, ARMA estimates, and surprise computations. [Researcher inference]

3. **Single-stock model:** Each stock is modeled independently. Cross-sectional information (e.g., sector-wide volume surges from ETF flows) is not captured. [Researcher inference; also noted by Bialkowski et al. 2008]

4. **Static seasonal assumption:** The 6-month trailing average for seasonal factors assumes the intraday volume shape is constant over that window. Structural changes (e.g., shifts in closing auction participation, changes in electronic trading) are captured slowly. [Researcher inference]

5. **No price information:** The model is purely volume-driven and agnostic to price dynamics. Cannot exploit volume-price correlations or adjust for price impact. [Researcher inference]

6. **Small order size assumption:** VWAP simulation uses 10% of 30-day ADV as order size (Satish et al. 2014, p.23). For larger orders, price impact becomes significant and the model's volume-only framework becomes insufficient. [Researcher inference]

7. **Proprietary components not disclosed:** The paper does not disclose specific values for regime thresholds, optimized weights, the optimal number of regression terms, or the proprietary deviation bounds method. Replicators must rediscover these through grid search. [Satish et al. 2014, p.18-19, implicit throughout]

8. **Metric incomparability across papers:** Different papers use different MAPE normalization conventions, making direct cross-paper comparison difficult. Satish et al. express MAPE as percentage; Chen et al. (2016) express it as fraction. [Researcher inference]

9. **Dataset specificity:** Results are reported on the top 500 U.S. equities by dollar volume over one specific year. Performance on less liquid stocks, other markets, or different time periods may differ. [Researcher inference]

## Paper References

| Spec Section | Paper Source |
|-------------|-------------|
| Overall architecture (dual model) | Satish et al. 2014, pp.17-18, full methodology section |
| Bin structure (I=26) | Satish et al. 2014, p.16, "26 such bins in a trading day" |
| VWAP tracking error definition | Satish et al. 2014, p.16, formula |
| Historical Window Average | Satish et al. 2014, p.16-17, "Historical Window Average/Rolling Means" |
| N_hist = 21 | Satish et al. 2014, Exhibit 1 diagram annotation, "Prior 21 days" |
| MAPE formula | Satish et al. 2014, p.17, formula |
| MAD formula (percentage metric) | Satish et al. 2014, p.17, "Measuring Percentage Volume Predictions -- Absolute Deviation" |
| Inter-day ARMA | Satish et al. 2014, p.17, "per-symbol, per-bin ARMA(p,q)" |
| AICc selection | Satish et al. 2014, p.17, "corrected AIC... Hurvich and Tsai [1989, 1993]" |
| Deseasonalization | Satish et al. 2014, p.17, "dividing by the intraday amount of volume traded in that bin over the trailing six months" |
| Intraday ARMA | Satish et al. 2014, pp.17-18, rolling one-month fit on deseasonalized data |
| AR < 5 constraint | Satish et al. 2014, p.18, "AR lags with a value less than five" |
| Fewer than 11 terms (observed) | Satish et al. 2014, p.18, "As a result, we fit each symbol with a dual ARMA model having fewer than 11 terms" |
| Re-seasonalization | Satish et al. 2014, p.18, "we re-seasonalize these forecasts via multiplication" |
| Exhibit 1 data flow | Satish et al. 2014, p.18, Exhibit 1 diagram |
| Regime switching | Satish et al. 2014, p.18, "regime switching... historical volume percentile cutoffs" |
| Dynamic weight overlay | Satish et al. 2014, p.18, "dynamic weight overlay... minimizes the error on in-sample data" |
| Custom curves for special days | Satish et al. 2014, p.18, "custom curves for special calendar days" |
| Volume percentage methodology | Satish et al. 2014, pp.18-19; Humphery-Jenner (2011) |
| Model A as surprise baseline | Satish et al. 2014, p.19, "we could apply our more extensive volume forecasting model" |
| Proprietary deviation bounds | Satish et al. 2014, p.19, "we developed a separate method for computing the deviation bounds" |
| No-intercept regressions | Satish et al. 2014, p.19, "we perform both regressions without the inclusion of a constant term" |
| VWAP-percentage error relationship | Satish et al. 2014, Exhibits 2-5, pp.20-21 |
| Raw volume MAPE results | Satish et al. 2014, p.20, Exhibit 6 (p.22), Exhibits 7-8 (pp.22-23) |
| Percentage volume results | Satish et al. 2014, Exhibit 9, p.23 |
| VWAP simulation results | Satish et al. 2014, Exhibit 10, p.23 |
| Deviation limit (10%) | Satish et al. 2014, p.24; Humphery-Jenner (2011) |
| Switch-off (80%) | Satish et al. 2014, p.24; Humphery-Jenner (2011) |
| Chen et al. benchmark | Chen et al. 2016, Tables 3-4 |
| N_interday_fit = 126 | Researcher inference (see Parameters section) |
| N_regression_fit = 63 | Researcher inference (see Parameters section) |
| Regime count grid search | Researcher inference (see Parameters section) |
| min_volume_floor = 100 | Researcher inference (see Parameters section) |
| Zero seasonal factor handling | Researcher inference |
| Day-boundary handling for intraday ARMA | Researcher inference |
| Surprise formula (percentage space) | Researcher inference |
| Weight optimization (MSE/SLSQP primary) | Researcher inference |
| Renormalization in Model B | Researcher inference |
| DailyOrchestration workflow | Researcher inference |
| ReEstimationSchedule | Researcher inference |
| UpdateInterDayState mechanism | Researcher inference |
| Intraday ARMA conditioning (reset-and-reprocess) | Researcher inference |
| Forecast purity requirement | Researcher inference |
