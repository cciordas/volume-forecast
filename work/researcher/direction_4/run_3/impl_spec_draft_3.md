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

#### Function 2b: UpdateInterDayState

Updates the ARMA state for a given bin when a new daily observation arrives, without re-estimating the model parameters. This is used between full re-estimation cycles.

```
function UpdateInterDayState(model, new_observation):
    # Reference: Researcher inference; required for daily state updates
    # between weekly re-estimation cycles.
    # The paper does not detail this step but it is implied by the use of
    # ARMA models that forecast one step ahead daily.

    if model is None:
        return  # No model to update

    # Step 1: Compute the prediction error (residual) for the new observation
    predicted = model.one_step_forecast()
    residual = new_observation - predicted

    # Step 2: Update the AR observation buffer
    # Shift observations: drop the oldest, append the new one
    model.ar_buffer.append(new_observation)
    if length(model.ar_buffer) > model.p:
        model.ar_buffer.pop_front()

    # Step 3: Update the MA residual buffer
    # The MA component uses past residuals e[t-1], ..., e[t-q]
    model.ma_buffer.append(residual)
    if length(model.ma_buffer) > model.q:
        model.ma_buffer.pop_front()

    # Step 4: The next one-step forecast is now:
    # predicted[t+1] = constant
    #                  + sum(phi[j] * ar_buffer[end - j + 1] for j in 1..p)
    #                  + sum(theta[j] * ma_buffer[end - j + 1] for j in 1..q)
    #
    # NOTE: The AR/MA coefficients (phi, theta) and constant remain FIXED.
    # Only the buffers (recent observations and residuals) are updated.
    # Full re-estimation (new MLE, possibly new order selection) happens
    # on the weekly cycle.

    return
```

**Paper reference:** Researcher inference. The paper describes daily forecasting with ARMA models (Satish et al. 2014, p.17) which implicitly requires state updates as new daily observations arrive. The Calibration section specifies weekly re-estimation; between re-estimations, only the internal state (observation and residual buffers) is updated while model parameters remain fixed.

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
    # N_intraday = 21 trading days. Note: the paper says "most recent month"
    # (p.18) which is ambiguous between calendar month and trading month.
    # We use 21 trading days as the implementation choice (approximately
    # one calendar month of trading activity).
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

    # Soft constraint: check combined parameter count across both ARMA models.
    # The paper states "a dual ARMA model having fewer than 11 terms" (p.18).
    # While this is an empirical observation (preceded by "As a result"), it
    # serves as a useful overfitting guard. If the combined parameter count
    # (inter-day + intraday) exceeds 10, log a warning. Do not reject the
    # model, but flag it for investigation.
    # Combined count = (p_inter + q_inter + 1) + (p_intra + q_intra + 1)

    return best_model  # May be None if all fits fail
```

**Paper reference:** Satish et al. 2014, p.17 para 3 -- p.18 para 1. "Trailing six months" for deseasonalization window, "rolling basis over the most recent month" for ARMA fitting window, "AR lags with a value less than five" (p.18).

**Note on "fewer than 11 terms":** The paper states "we fit each symbol with a dual ARMA model having fewer than 11 terms" (p.18). This is an observed outcome of AICc selection and AR coefficient decay, not a hard constraint. The phrasing "As a result, we fit..." indicates it describes what happened empirically. In practice, AICc-selected models with p_max_interday=5 and p_max_intraday=4 will typically have low total parameter counts. The developer should implement this as a **soft constraint**: if the combined parameter count across both ARMA models exceeds 10, log a warning but do not reject the model. This provides a useful safety valve against overfitting while preserving AICc's flexibility. [Researcher inference for the interpretation; the exact phrasing is from Satish et al. 2014, p.18, para 1.]

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
        # Too early to classify reliably; use default regime.
        # DEFAULT_REGIME = 1 (the middle tercile for n_regimes=3),
        # representing "typical" volume. This is the safest default
        # since it avoids extreme-regime weights when there is
        # insufficient intraday data to classify.
        # [Researcher inference.]
        return 1  # DEFAULT_REGIME (middle tercile)

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

**Note on pre-market forecasts:** When `current_bin=0` (pre-market), `i < min_regime_bins` is always true, so all pre-market forecasts use the default regime (middle tercile). This is by design: no intraday volume has been observed yet, so regime classification is impossible. The default regime weights should represent "typical" day behavior. Pre-market Model A forecasts are therefore not regime-adapted. [Researcher inference.]

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

        # ---- DESIGN DECISION: Loss function for weight optimization ----
        #
        # The paper says "minimizes the error" (p.18, para 3) without
        # specifying which error metric. Two defensible choices exist:
        #
        # OPTION A (recommended primary): MSE minimization
        #   min_w sum((actual - w1*H - w2*D - w3*A)^2)
        #   subject to: w1 + w2 + w3 = 1, w1 >= 0, w2 >= 0, w3 >= 0
        #   Advantages: convex objective with unique solution; SLSQP
        #   converges reliably; equivalent to constrained least squares.
        #   Disadvantage: penalizes large absolute errors quadratically,
        #   biasing weights toward high-volume bins/days. The paper
        #   evaluates using MAPE, so there is a train-eval metric mismatch.
        #
        # OPTION B (recommended variant to test): MAPE minimization
        #   min_w (1/N) * sum(|actual - w1*H - w2*D - w3*A| / actual)
        #   subject to: w1 + w2 + w3 = 1, w1 >= 0, w2 >= 0, w3 >= 0
        #   Advantages: directly optimizes the evaluation metric; treats
        #   relative errors equally across volume scales.
        #   Disadvantage: non-smooth (absolute value), requires gradient-
        #   free optimization (e.g., Nelder-Mead, differential evolution).
        #   May have multiple local minima. Undefined when actual = 0
        #   (exclude zero-volume bins).
        #
        # The developer should implement BOTH and compare:
        #   1. MSE weights as default (robust, fast convergence).
        #   2. MAPE weights as alternative, evaluated on held-out data.
        #   If MAPE-optimal weights produce lower out-of-sample MAPE,
        #   switch to MAPE optimization.
        # [Researcher inference for both options; the paper's "error"
        #  is ambiguous.]

        # Primary implementation: MSE with simplex constraint
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

**Paper reference:** Satish et al. 2014, p.18, paragraphs 3-4. The paper says "minimizes the error" without specifying the loss function. [Researcher inference: MSE with non-negative simplex constraint as primary, MAPE minimization as variant. The simplex constraint w1+w2+w3=1 ensures the combined forecast is a convex combination of the components. The paper does not explicitly state this constraint, but it prevents scale drift and is standard for forecast combination.]

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
        # via UpdateInterDayState()
    else:
        D = H  # Fallback if ARMA fitting failed

    # Component 3: Intraday ARMA forecast
    # Reference: Satish et al. 2014, p.17-18, Exhibit 1
    #
    # --- EXHIBIT 1 ANNOTATION DISCUSSION ---
    # Exhibit 1 labels the input to "ARMA Intraday" as "Current Bin"
    # and "4 Bins Prior to Current Bin" -- a total of 5 bins.
    #
    # Two interpretations:
    # (1) Only the 5 most recent observed bins are passed to the ARMA
    #     state updater at prediction time (hard window of 5 bins).
    # (2) The AR order is at most 4 (consistent with "AR lags with a
    #     value less than five," p.18), so regardless of how many
    #     observations are fed to the state updater, only the last ~4
    #     matter for the AR prediction. The Exhibit's "4 Bins" annotation
    #     describes the effective prediction memory, not a data filter.
    #
    # We adopt interpretation (2) because:
    # - The ARMA model's AR order p <= 4 means only the p most recent
    #   deseasonalized observations influence the forecast. Passing all
    #   observed bins is mathematically equivalent to passing only the
    #   last p bins when the model makes its AR prediction.
    # - For the MA component, past residuals are needed. These are
    #   computed recursively from the full observed series during state
    #   updates, so truncating the input to 5 bins could lose MA state
    #   that accumulated from earlier bins.
    # - The Exhibit annotation is a simplified schematic, not pseudocode.
    #
    # Under interpretation (2), passing all observed bins is harmless
    # and preserves MA state. If the developer prefers interpretation (1),
    # they should slice deseas_observed to the last min(current_bin, 5)
    # entries; the AR forecast will be identical for p <= 4.
    # [Researcher inference for this interpretation; Exhibit 1 caption
    #  and p.18 are the sources.]

    if intraday_model is not None and current_bin >= 1:
        # Condition the intraday ARMA on today's observed deseasonalized bins
        deseas_observed = [volume[s, t, j] / seasonal_factor[j] for j in 1..current_bin]
        # Update ARMA state using ConditionIntraDayARMA (Function 6b)
        ConditionIntraDayARMA(intraday_model, deseas_observed)
        # Forecast (i - current_bin) steps ahead in deseasonalized space
        steps_ahead = i - current_bin
        A_deseas = intraday_model.forecast(steps=steps_ahead)
        # Re-seasonalize
        A = A_deseas * seasonal_factor[i]
        #
        # NOTE ON MULTI-STEP FORECAST DEGRADATION:
        # For ARMA(p,q) models, multi-step forecasts degrade rapidly.
        # After p steps, the AR component uses only its own past forecasts
        # (not observations), converging toward the unconditional mean.
        # After q steps, the MA component contributes nothing (past
        # residuals are zero in the forecast horizon).
        # For ARMA(2,2) with current_bin=1 forecasting bin 26 (steps_ahead=25),
        # the forecast is essentially the unconditional mean (~1.0 in
        # deseasonalized space), which re-seasonalizes to the seasonal
        # factor -- i.e., the historical average for that bin.
        # This is expected behavior: the intraday ARMA component A provides
        # diminishing incremental value for distant bins. The regime weights
        # should adapt accordingly (w3 may be lower for early current_bin).
        # The model's performance for distant bins relies primarily on H
        # (historical) and D (inter-day ARMA).
        # [Researcher inference.]
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

#### Function 6b: ConditionIntraDayARMA

Conditions the intraday ARMA model on today's observed deseasonalized bin volumes, building up the AR observation buffer and MA residual buffer sequentially. This is the intraday analogue of Function 2b (UpdateInterDayState), but differs in two ways: (1) it processes an array of observations rather than a single scalar, and (2) it resets to the model's training-window state before processing, so it is stateless across calls within the day.

```
function ConditionIntraDayARMA(model, deseas_observed):
    # Reference: Researcher inference; required for intraday ARMA
    # state conditioning. The inter-day analogue (Function 2b) takes
    # a single scalar; the intraday case takes an array because
    # multiple bins may have been observed since the model was last
    # used.
    #
    # DESIGN CHOICE: Reset-and-reprocess (stateless) vs. incremental.
    #
    # (a) RESET-AND-REPROCESS (implemented): Each time this function
    #     is called, reset the ARMA buffers to their post-training
    #     state (the state after processing the entire training window),
    #     then sequentially process all observed bins from scratch.
    #     At bin 5, process bins [1,2,3,4,5]; at bin 6, process
    #     [1,2,3,4,5,6] -- re-processing bins 1-5.
    #     Advantage: stateless, no risk of state corruption across bins.
    #     Disadvantage: O(current_bin^2) total work across a day (trivial
    #     for 26 bins).
    #
    # (b) INCREMENTAL (alternative): Maintain state across calls within
    #     the day. At bin 6, only process the new observation (bin 6).
    #     Advantage: O(current_bin) total work.
    #     Disadvantage: requires the caller to guarantee that the function
    #     is called exactly once per new bin, in order. If any call is
    #     missed or repeated, the state becomes inconsistent.
    #
    # We implement (a) for robustness. The computational cost of
    # reprocessing at most 26 observations per call is negligible.

    if model is None:
        return

    # Step 1: Save the post-training state (buffers from end of training window)
    # This is set once when the model is fitted and never modified.
    # model.training_ar_buffer: last p observations from training series
    # model.training_ma_buffer: last q residuals from training series
    saved_ar = copy(model.training_ar_buffer)
    saved_ma = copy(model.training_ma_buffer)

    # Step 2: Reset buffers to post-training state
    model.ar_buffer = copy(saved_ar)
    model.ma_buffer = copy(saved_ma)

    # Step 3: Sequentially process each observed bin
    for obs in deseas_observed:
        # Compute one-step-ahead prediction from current state
        predicted = model.one_step_forecast()
        residual = obs - predicted

        # Update AR buffer (shift in new observation)
        model.ar_buffer.append(obs)
        if length(model.ar_buffer) > model.p:
            model.ar_buffer.pop_front()

        # Update MA buffer (shift in new residual)
        model.ma_buffer.append(residual)
        if length(model.ma_buffer) > model.q:
            model.ma_buffer.pop_front()

    # After processing, model.ar_buffer and model.ma_buffer reflect
    # the state conditioned on all observed bins. The next call to
    # model.forecast(steps=N) will use this updated state.
    return
```

**Paper reference:** Researcher inference. The paper describes intraday ARMA conditioning via Exhibit 1's "Current Bin / 4 Bins Prior to Current Bin" flow (Satish et al. 2014, Exhibit 1), but does not specify the implementation. The reset-and-reprocess approach is chosen for robustness: it is mathematically equivalent to the incremental approach but eliminates state management complexity. The sequential processing order matters because each residual depends on the AR/MA state from all prior observations.

**Note on training state:** When `FitIntraDayARMA` (Function 3) fits the model, it must save the final AR and MA buffer states as `training_ar_buffer` and `training_ma_buffer`. These represent the model's state after processing the entire training series (N_intraday * I observations). The developer must ensure the ARMA fitting routine exposes these final buffer states.

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
    #
    # --- TRAIN/PREDICT DENOMINATOR DISCUSSION ---
    # During prediction (here), surprises are computed using V_total_est
    # (observed total + forecasted remaining), which is a noisy estimate
    # that evolves as bins are observed.
    # During training (Function 8), surprises use the exact daily total
    # (sum of all bins for that day), which is known in hindsight.
    # This creates a statistical mismatch: early-day prediction surprises
    # have different noise properties than training surprises because
    # V_total_est can deviate significantly from the true daily total.
    # Late in the day, V_total_est converges toward the actual total,
    # reducing the mismatch.
    #
    # Two approaches to handle this:
    # (a) SIMPLER (implemented): Accept the mismatch. Justification:
    #     the regression coefficients are small (|beta| < 0.5) and the
    #     deviation constraint (Step 6) limits the impact of noisy
    #     surprises. Early-day surprises are also padded with zeros
    #     (Step 3 below), further reducing their influence.
    # (b) ADVANCED (variant): During training, mimic prediction by using
    #     "leave-future-out" estimated totals: for training day d at bin i,
    #     use sum of actuals through bin i plus forecasts for bins i+1..I
    #     as the denominator. This eliminates the mismatch but multiplies
    #     training compute by I (must re-forecast remaining bins at each
    #     training bin).
    #
    # The developer should implement (a) first and test (b) if the
    # percentage model underperforms early in the day.
    # [Researcher inference for both the mismatch identification and
    #  the recommended approach.]

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
    #
    # IMPORTANT: The paper says "we developed a separate method for
    # computing the deviation bounds" (p.19), indicating the actual
    # implementation uses adaptive or proprietary bounds, NOT the
    # fixed 10% from Humphery-Jenner (2011). The 10% is a simplification
    # of what the paper actually used. The developer should treat
    # max_deviation as a primary tuning candidate. The paper's published
    # results may have used tighter or data-adaptive bounds.
    # [Researcher inference for the warning about simplification.]
    max_delta = max_deviation * scaled_base
    delta = clip(delta, -max_delta, +max_delta)

    # Step 7: Final forecast
    p_hat = scaled_base + delta
    p_hat = max(p_hat, 0.0)  # Non-negative

    return p_hat
```

**Paper reference:** Satish et al. 2014, pp.18-19, p.24. The dynamic VWAP methodology is based on Humphery-Jenner (2011). The paper states "we could apply our more extensive volume forecasting model" to compute volume surprises (p.19), indicating the raw model's forecasts should serve as the expected volume for computing surprises.

**Note on "could":** The paper uses "we could apply our more extensive volume forecasting model" (p.19) -- the word "could" is aspirational, not confirmative. It is possible the published results used a simpler historical baseline for surprises (as in the original Humphery-Jenner). I recommend using the raw model (Model A) forecasts as the primary approach and providing a configuration option to fall back to historical averages. [Researcher inference.]

**Note on no-intercept regression:** The paper states: "we perform both regressions without the inclusion of a constant term (indicating a non-zero y intercept). This means our model does not assume a positive amount of VWAP error if our volume predictions are 100% accurate" (p.19). This quote explicitly refers to the validation regressions (VWAP error vs. percentage error), not the surprise prediction regression. However, the same principle applies: when all surprises are zero, the dynamic adjustment delta should be zero, which is automatically satisfied by a no-intercept model. We apply no-intercept to the surprise regression by analogy with the stated reasoning. [Researcher inference for applying the no-intercept principle to the prediction regression; the explicit quote is about validation regressions.]

#### Function 8: TrainPercentageRegression

Trains the rolling OLS regression coefficients for the volume percentage model.

```
function TrainPercentageRegression(symbol s, training_days):
    # Reference: Satish et al. 2014, pp.18-19
    # Humphery-Jenner (2011) rolling regression

    # --- STEP 0: Pre-compute raw forecasts for all training days ---
    #
    # DESIGN CHOICE: Dynamic vs. static training forecasts.
    #
    # At prediction time (Function 7), raw forecasts are DYNAMIC:
    # ForecastRawVolume(s, t, lag_bin, lag_bin - 1) uses all observations
    # through lag_bin - 1 to forecast lag_bin. To match this at training
    # time, we must also use dynamic forecasts -- i.e., at each bin i of
    # training day d, the raw forecast uses observations through bin i-1.
    #
    # Using static forecasts (current_bin=0, pre-market only) would be
    # simpler but creates a train/predict mismatch: the surprise signal
    # would have different statistical properties because static forecasts
    # do not incorporate intraday information that dynamic forecasts use.
    # Early-bin surprises would differ most (static forecasts lack any
    # intraday conditioning).
    #
    # To manage computational cost, we pre-compute and cache dynamic
    # raw forecasts: for each training day d, run the Model A pipeline
    # at each current_bin (0 through I-1), storing all raw forecasts.
    # This is done once per re-estimation cycle.
    #
    # Cost: For 252 training days * 26 bins * 26 target bins = ~170K
    # ForecastRawVolume calls per symbol. With 500 symbols, this is ~85M
    # calls total. Most of the cost is in the intraday ARMA conditioning
    # (Function 6b), but each call processes at most 26 observations.
    # Total wall-clock time: ~minutes with vectorized ARMA code.
    #
    # [Researcher inference for the dynamic training forecast choice and
    #  caching strategy. The paper does not specify which approach is used.]

    # Pre-compute: raw_forecast_cache[d][current_bin][target_bin] = V_hat
    raw_forecast_cache = {}
    for d in training_days:
        raw_forecast_cache[d] = {}
        for cb in 0..I-1:  # current_bin from 0 (pre-market) to I-1
            raw_forecast_cache[d][cb] = {}
            for target in (cb+1)..I:
                raw_forecast_cache[d][cb][target] = ForecastRawVolume(s, d, target, cb)

    # Collect training samples: (delta_actual, surprise_lags)
    X = []  # Each row: [surprise[lag_1], surprise[lag_2], ..., surprise[lag_K_reg]]
    y = []  # Each element: actual_pct - expected_pct for target bin

    for d in training_days:
        # --- DENOMINATOR DISCUSSION ---
        # The actual percentage denominator uses the exact daily total
        # (known in hindsight during training), while at prediction time
        # V_total_est (observed + forecasted remaining) is used. See
        # Function 7, Step 3 for the full discussion of this mismatch.
        #
        # Additionally, the EXPECTED percentage denominator also differs:
        # - Training: sum of all raw forecasts for the day, using dynamic
        #   forecasts at current_bin = i-1:
        #   sum(raw_forecast_cache[d][i-1][j] for j in 1..I), where
        #   bins 1..i-1 use actual observations and bins i..I use forecasts.
        #   However, for simplicity we use: sum of dynamic forecasts for
        #   ALL bins generated at current_bin = i-1.
        # - Prediction: V_total_est = observed_total + sum(remaining forecasts),
        #   which evolves as bins are observed.
        # Both denominators converge as more bins are observed: late in
        # the day, both are dominated by actual observations. The mismatch
        # is largest in early bins. This is a second-order effect compared
        # to the actual-percentage denominator mismatch and does not
        # warrant a separate correction.
        # [Researcher inference.]
        daily_total = sum(volume[s, d, j] for j in 1..I)

        # --- EARLY-BIN PADDING DESIGN CHOICE ---
        # The loop starts at i=2 (need at least 1 lag). For K_reg=3,
        # bins 2 and 3 have only 1 and 2 "real" surprise lags, with
        # the remainder padded as zero. This means the regression is
        # trained on a mixture of real and zero-padded data.
        #
        # Two alternatives:
        # (a) INCLUDE PADDED (implemented): More training data but
        #     zero-padded early-bin points may slightly bias coefficients
        #     toward zero. Consistent with Function 7's prediction-time
        #     padding (which also pads with zeros for early bins).
        # (b) EXCLUDE EARLY BINS (start at i = K_reg + 1): Cleaner
        #     coefficient estimates, losing K_reg - 1 bins per day
        #     (~8% of data for K_reg=3). The developer can test this
        #     as a variant.
        #
        # Option (a) is recommended because it matches prediction-time
        # behavior and the data loss from (b) is modest.
        # [Researcher inference.]
        for i in 2..I:  # Start at bin 2 (need at least 1 lag)
            # current_bin for this training observation is i-1
            # (we have observed bins 1..i-1 and are forecasting bin i)
            cb = i - 1

            # Compute target: actual percentage deviation
            actual_pct_d_i = volume[s, d, i] / daily_total

            # Expected percentage uses dynamic raw forecasts at current_bin = i-1
            # Denominator: sum of raw forecasts for all bins, generated at cb
            forecast_total_d = sum(
                raw_forecast_cache[d][cb][j] for j in (cb+1)..I
            ) + sum(volume[s, d, j] for j in 1..cb)
            # NOTE: for bins 1..cb, we use actual observed volumes
            # (they are already known at current_bin = cb).
            # For bins cb+1..I, we use dynamic forecasts from cache.
            expected_pct_d_i = raw_forecast_cache[d][cb][i] / forecast_total_d
            target = actual_pct_d_i - expected_pct_d_i

            # Compute lagged surprises (also using dynamic forecasts)
            row = []
            for k in 1..K_reg:
                lag_bin = i - k
                if lag_bin < 1:
                    row.append(0.0)
                else:
                    actual_pct_lag = volume[s, d, lag_bin] / daily_total
                    # For the lagged surprise, the current_bin was lag_bin - 1
                    lag_cb = lag_bin - 1
                    if lag_cb < 0:
                        expected_pct_lag = 0.0
                    else:
                        lag_forecast_total = sum(
                            raw_forecast_cache[d][lag_cb][j] for j in (lag_cb+1)..I
                        ) + sum(volume[s, d, j] for j in 1..lag_cb)
                        expected_pct_lag = raw_forecast_cache[d][lag_cb][lag_bin] / lag_forecast_total
                    row.append(actual_pct_lag - expected_pct_lag)

            X.append(row)
            y.append(target)

    # OLS without intercept
    beta = OLS_fit(X, y, fit_intercept=False)

    return beta  # Array of K_reg coefficients
```

**Paper reference:** Satish et al. 2014, pp.18-19. The paper does not specify the exact number of lag terms, whether the regression is pooled across bins or fit per-bin, or whether training uses static or dynamic raw forecasts. [Researcher inference: pooled across all bins and days (single regression), K_reg = 3 lagged surprise terms. A pooled regression is more parsimonious and more robust than 26 per-bin regressions with limited data. Dynamic forecasts are used to match prediction-time behavior, with caching to manage computational cost.]

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
    # Used as baseline expected percentages for surprise computation.
    # NOTE: All pre-market forecasts use the default regime (middle
    # tercile) because no intraday volume has been observed to
    # classify the regime. See Function 4 note on pre-market regime.
    static_forecasts = []
    for i in 1..I:
        V_hat = ForecastRawVolume(s, t, i, current_bin=0)
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

    # STEP 3: End-of-day state update
    # Update inter-day ARMA state for each bin with today's actual volume
    for i in 1..I:
        UpdateInterDayState(interday_models[i], volume[s, t, i])

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
  +---------- Daily state update ----------+
  | [UpdateInterDayState]                  |
  |   Input:  model + new daily obs        |
  |   Output: updated model state          |
  |   (fixed params, updated buffers)      |
  +----------+----------------------------+
             |
             v

[ClassifyRegime]
  Input:  cumulative volume up to current bin, historical distribution
  Output: regime index r (integer, 0..n_regimes-1)

             |
             v

[ConditionIntraDayARMA] (Function 6b)
  Input:  intraday ARMA model + deseas_observed (array, shape (current_bin,))
  Output: updated model state (AR/MA buffers conditioned on observations)
          Resets to training state, then processes observations sequentially

             |
             v

[ForecastRawVolume] (Model A)
  Input:  trained models, current_bin, target bin i
  Output: V_hat (float, shares)
          Combines H, D, A with weights[regime]
          Calls ConditionIntraDayARMA internally for component A

             |
             v

[ForecastVolumePercentage] (Model B)
  Input:  V_hat from Model A for remaining bins,
          observed volumes, p_hist, beta coefficients
  Output: p_hat (float, fraction of daily volume for next bin)
```

### Variants

This specification implements the **full dual-model system** as described in Satish et al. (2014), which represents the most complete treatment in the assigned papers. The key variant decisions are:

1. **Surprise baseline in Model B:**
   - **Primary (implemented):** Use Model A (raw volume) forecasts to compute expected volume percentages for surprise calculation. This is the more sophisticated approach suggested by the paper ("we could apply our more extensive volume forecasting model," p.19).
   - **Alternative (configuration option):** Use simple historical rolling averages as the surprise baseline, matching the original Humphery-Jenner (2011) formulation. This is simpler and may have been what the published results actually used.

2. **Weight optimization loss function:**
   - **Primary (implemented):** MSE minimization with simplex constraint.
   - **Variant (recommended to test):** MAPE minimization to match the evaluation metric. See Function 5 for detailed discussion.

The paper also tested 5-minute and 30-minute bin sizes (Exhibit 9), showing smaller improvements (2.25% and 2.95% respectively vs. 7.55% for 15-minute). The 15-minute bin size is implemented as the primary configuration.

---

## Parameters

| Parameter | Description | Recommended Value | Sensitivity | Range |
|-----------|-------------|-------------------|-------------|-------|
| I | Number of intraday bins | 26 | N/A (structural) | {13, 26, 78} for {30, 15, 5}-min |
| N_hist | Rolling window for historical average (Component 1) | 21 trading days | Medium-High | [10, 60] |
| N_seasonal | Window for seasonal factor computation | 126 trading days (6 months) | Low | [63, 252] |
| N_interday | Training window for inter-day ARMA | 126 trading days | Low-Medium | [63, 252] |
| N_intraday | Rolling window for intraday ARMA fitting | 21 trading days | Medium | [10, 42] |
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
| max_deviation | Maximum percentage deviation from scaled baseline | 0.10 (10%) | Medium-High | [0.05, 0.20] |
| pct_switchoff | Cumulative volume fraction triggering switch-off | 0.80 (80%) | Medium | [0.70, 0.90] |

**Parameter sources:**
- N_hist = 21: Satish et al. 2014, Exhibit 1 caption "Prior 21 days." Note: the paper explicitly treats N as a tunable parameter ("One chooses the number of days of historical data to use, a variable that we shall call N," p.16). The value 21 from Exhibit 1 is a reasonable default but should be tuned. Sensitivity is Medium-High because the historical average is one of the three forecast components and its accuracy depends directly on window length.
- N_seasonal = 126: Satish et al. 2014, p.17 "trailing six months."
- N_intraday = 21: Satish et al. 2014, p.18 "rolling basis over the most recent month." Note: "month" is ambiguous between calendar month (20-23 trading days) and trading month. We use 21 trading days as a fixed implementation choice.
- p_max_intraday = 4: Satish et al. 2014, p.18 "AR lags with a value less than five."
- p_max_interday = 5, q_max_interday = 5: Satish et al. 2014, p.17 "all values of p and q lags through five."
- max_deviation = 0.10: Satish et al. 2014, p.24 / Humphery-Jenner (2011) "depart no more than 10% away." **Important caveat:** The paper says "we developed a separate method for computing the deviation bounds" (p.19), meaning the actual production system uses adaptive or proprietary bounds, not this fixed 10%. The fixed 10% is a simplification from Humphery-Jenner's original formulation. This parameter is a **primary tuning candidate** and the paper's published results may have used tighter or data-adaptive bounds.
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

3. **Inter-day ARMA models:** Fit 26 models (one per bin) on the first N_interday days. Cache fitted models. On subsequent days, update ARMA state with the new observation using `UpdateInterDayState()` (conditioning, not re-estimation). The distinction: re-estimation runs full MLE and possibly changes the model order (p, q); state update keeps the current AR/MA coefficients fixed and only shifts the observation and residual buffers forward.

4. **Intraday ARMA model:** Fit one model per symbol on the most recent 21 trading days of deseasonalized data (21 * 26 = 546 observations). Re-fit daily at market open.

5. **Regime weights:** Require a training period for weight optimization. Use the first N_weight_train days (after ARMA warm-up) to train initial weights. [Researcher inference: N_weight_train = 63 (3 months).]

6. **Percentage regression:** Train on a window of N_reg_train days using volume surprises computed from in-sample raw forecasts. [Researcher inference: N_reg_train = 252 (1 year).]

7. **Initial weight guess:** (1/3, 1/3, 1/3) for all regimes.

### Calibration

**Model A calibration** involves three concurrent procedures:

1. **ARMA order selection:** Grid search over (p, q) for each of the 26 inter-day models and the single intraday model. AICc selects the best model per bin/symbol. For 500 symbols: 500 * 26 * 36 = 468,000 inter-day ARMA fits + 500 * 36 = 18,000 intraday fits per calibration cycle. This is computationally intensive but parallelizable across symbols. After AICc selection, check the combined parameter count against the soft constraint of < 11 total terms; log a warning if exceeded.

2. **Weight optimization:** For each regime, minimize MSE (primary) or MAPE (variant) over in-sample forecast errors. SLSQP with simplex constraint for MSE; Nelder-Mead or differential evolution for MAPE. Convergence is fast (3 parameters per regime).

3. **Regime threshold computation:** Compute cumulative volume percentile distributions from historical data.

**Model B calibration:**

1. **Percentage regression:** OLS (closed-form) with K_reg predictors and no intercept. Fast to compute.

**Re-estimation schedule:** [Researcher inference; paper does not specify.]
- Daily: seasonal factors (window slides by 1 day), intraday ARMA (re-fit on latest 21 days), historical percentage curve.
- Daily (state update only): inter-day ARMA state updated via `UpdateInterDayState()` with the new daily observation. Model parameters (AR/MA coefficients) remain fixed.
- Weekly (every 5 trading days): inter-day ARMA models (full re-estimation with AICc selection, possibly changing model order).
- Monthly (every 21 trading days): regime weights, percentage regression coefficients.

---

## Validation

### Metrics

The developer must implement the following exact formulas to match the paper's evaluation. Implementing the wrong metric will produce misleading comparisons.

**MAPE (Mean Absolute Percentage Error) -- for raw volume (Model A):**

```
MAPE = 100% * (1/N) * sum_{i=1}^{N} |Predicted_Volume_i - Raw_Volume_i| / Raw_Volume_i
```

where i runs over all bins, N is the total number of bins. Each bin's error is normalized by the **actual** raw volume for that bin. Bins with zero actual volume should be excluded (MAPE is undefined for zero denominators).

Reference: Satish et al. 2014, p.17, "Measuring Raw Volume Predictions -- MAPE."

**Percentage prediction error -- for volume percentage (Model B):**

```
Error = (1/N) * sum_{i=1}^{N} |Predicted_Percentage_i - Actual_Percentage_i|
```

where i runs over all bins, N is the total number of bins. This is a mean absolute error **without** normalization by the actual value, since percentages are already normalized (they sum to 100). Do not confuse this with MAPE -- there is no division by actual percentage.

Reference: Satish et al. 2014, p.17, "Measuring Percentage Volume Predictions -- Absolute Deviation."

**Aggregation:** The paper computes these metrics across "the 500-name universe over 250 trading days" with "26 forecasts for each of the 15-minute intervals" (Satish et al. 2014, p.20). The primary aggregate statistic is the **median** across symbols (for the "24% median MAPE reduction" headline number). The "bottom 95%" statistic averages after excluding the worst 5% of symbols. The developer should compute:
1. Per-symbol-day MAPE (average across 26 bins for one symbol-day).
2. Per-symbol MAPE (average across all days for one symbol).
3. Cross-symbol median of per-symbol MAPE.
4. Cross-symbol bottom-95% average of per-symbol MAPE.

[Researcher inference for the aggregation hierarchy; the paper does not state the exact aggregation steps but the quoted statistics imply per-symbol aggregation followed by cross-symbol median/percentile summary.]

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
- Per-category reductions: 7%-10% when computed separately within each stock category (Dow 30, midcap, high-variance); the aggregate reduction across all simulated orders is 9.1%. (Satish et al. 2014, p.23.)
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

4. **AICc model selection:** Expect AICc to typically select low-order models (p <= 2, q <= 2). Monitor the distribution of selected orders across bins. If the combined parameter count across both ARMA models exceeds 10 for any symbol, log a warning (soft constraint from "fewer than 11 terms," p.18). [Researcher inference.]

5. **Regime bucket populations:** All regime buckets should be populated with sufficient training samples. If any bucket has fewer than min_samples_per_regime, the regime scheme may be too fine-grained. [Researcher inference.]

6. **Deviation constraint binding frequency:** The deviation constraint in Model B should bind on no more than 10-20% of bins. If it binds too frequently, the constraint may be too tight (and may differ from the paper's adaptive bounds); if it never binds, max_deviation may be too generous. [Researcher inference.]

7. **Switch-off engagement:** The 80% switch-off should engage for the last 2-4 bins of a typical day. If it engages much earlier, V_total_est may be systematically too high. [Researcher inference.]

8. **Forecast non-negativity:** Raw volume forecasts should never be negative after clamping. If negative values occur frequently before clamping, the ARMA component may be producing poor extrapolations. [Researcher inference.]

9. **Surprise mean:** The mean of volume surprises across all training (symbol, day, bin) observations should be approximately zero (indicating the raw model is unbiased on average). Standard deviation of surprises for 15-minute bins is expected to be approximately 0.005-0.015. [Researcher inference.]

10. **Regression coefficient magnitudes:** The absolute values of beta coefficients in the percentage regression should be small (|beta_k| < 0.5). Large coefficients indicate instability. [Researcher inference.]

11. **Historical baseline MAPE:** Component 1 alone (historical rolling average) should produce MAPE values consistent with the paper's baseline (which the dual ARMA improves upon by 24% median). This establishes that the baseline implementation is correct before adding ARMA components. [Researcher inference.]

12. **Inter-day ARMA state consistency:** After calling UpdateInterDayState(), the model's one-step-ahead forecast should differ from the previous forecast. If the forecast does not change, the state update is not working correctly. Verify by comparing forecast before and after update on a few test cases. [Researcher inference.]

13. **Intraday ARMA multi-step convergence:** For steps_ahead > 10, the intraday ARMA forecast (in deseasonalized space) should converge toward the unconditional mean of the deseasonalized series (approximately 1.0). Verify: with current_bin=1 and an ARMA(2,2) model, the forecast for bin 26 (steps_ahead=25) should be within 5% of 1.0 in deseasonalized space. After re-seasonalization, this means the forecast approximately equals the seasonal factor for that bin. If the forecast does NOT converge, the multi-step forecasting logic may have a bug (e.g., the forecast might be recursively amplifying rather than damping). [Researcher inference.]

14. **Intraday ARMA conditioning consistency:** After calling ConditionIntraDayARMA with bins [1,2,3], calling it again with [1,2,3,4] should produce the same AR/MA buffer state for bins 1-3 as the previous call, plus the state update from bin 4. Since the function resets and reprocesses from scratch, the result must be identical to incremental processing. Verify by comparing buffer contents. [Researcher inference.]

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

1. **Proprietary parameters:** The paper does not disclose the specific values of the dynamic weighting coefficients, the exact number of regime buckets, the regime percentile cutoffs, the adaptive deviation bounds, or the optimal regression terms for U.S. equities (Satish et al. 2014, p.18-19). These must be rediscovered via in-sample optimization.

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
| Fewer than 11 terms (soft constraint) | Satish et al. 2014, p.18 para 1 |
| Deseasonalization (trailing 6 months) | Satish et al. 2014, p.17 para 3 |
| Intraday ARMA window (1 month) | Satish et al. 2014, p.18 para 1 |
| Component 4: Dynamic weight overlay | Satish et al. 2014, p.18 para 3 |
| Regime switching | Satish et al. 2014, p.18 para 4 |
| Custom curves for special days | Satish et al. 2014, p.18 para 4 |
| Volume percentage model (dynamic VWAP) | Satish et al. 2014, pp.18-19; Humphery-Jenner (2011) |
| Deviation limits (10%, simplified from proprietary) | Satish et al. 2014, p.19 ("separate method"), p.24; Humphery-Jenner (2011) |
| Switch-off at 80% | Satish et al. 2014, p.24; Humphery-Jenner (2011) |
| No-intercept regressions | Satish et al. 2014, p.19 |
| Inter-day ARMA state update | Researcher inference (implied by daily ARMA forecasting) |
| Exhibit 1 "4 Bins Prior" interpretation | Satish et al. 2014, Exhibit 1; Researcher inference |
| Intraday ARMA state conditioning (Function 6b) | Researcher inference (implied by Exhibit 1 data flow) |
| Multi-step ARMA forecast degradation | Researcher inference (standard ARMA theory) |
| Dynamic vs. static training forecasts | Researcher inference |
| Early-bin padding design choice | Researcher inference |
| Expected percentage denominator mismatch | Researcher inference |
| MAPE formula | Satish et al. 2014, p.17 |
| Percentage error formula | Satish et al. 2014, p.17 |
| MAPE reduction (24% median, 29% bottom-95%) | Satish et al. 2014, p.20, below Exhibit 6 |
| MAPE by time-of-day | Satish et al. 2014, Exhibit 6 |
| MAPE by SIC group | Satish et al. 2014, Exhibit 7 |
| MAPE by beta decile | Satish et al. 2014, Exhibit 8 |
| Volume percentage results (Exhibit 9) | Satish et al. 2014, Exhibit 9, p.23 |
| VWAP tracking error simulation | Satish et al. 2014, Exhibit 10, p.23 |
| VWAP error vs. percentage error regression | Satish et al. 2014, Exhibits 2-5, pp.20-21 |
| N_hist = 21 and its tunability | Satish et al. 2014, p.16 ("a variable that we shall call N"), Exhibit 1 |
| Comparison: Chen et al. Kalman filter | Chen et al. 2016, Tables 3-4, pp.9-10 |
| Comparison: Brownlees et al. CMEM | Brownlees et al. 2011; cited in Satish et al. 2014, p.24 |
| Comparison: Bialkowski et al. PCA-SETAR | Bialkowski et al. 2008; cited in Satish et al. 2014, p.24 |
| MSE vs MAPE weight optimization | Researcher inference |
| Train/predict denominator mismatch | Researcher inference |
| Regime count, percentile thresholds | Researcher inference |
| N_interday training window | Researcher inference |
| K_reg value | Researcher inference |
| Simplex constraint for weights | Researcher inference |
| Re-estimation schedule | Researcher inference |
| Surprise normalization details | Researcher inference |
| Renormalization of scaled baseline | Researcher inference |
| Epsilon floor value | Researcher inference |
| DEFAULT_REGIME = 1 (middle tercile) | Researcher inference |
