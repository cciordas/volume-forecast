# Implementation Specification: Dual-Mode Intraday Volume Forecast (Raw Volume + Volume Percentage)

## Overview

This specification describes a dual-model system for intraday volume forecasting
based on Satish, Saxena, and Palmer (2014). Model A forecasts raw bin-level
trading volume by combining three components — a rolling historical average, an
inter-day ARMA, and an intraday ARMA — through a regime-switching dynamic weight
overlay. Model B forecasts next-bin volume percentages by applying a
surprise-regression adjustment (extending Humphery-Jenner 2011's "dynamic VWAP"
framework) on top of a historical volume percentage curve. The two models are
coupled: Model A's historical average component serves as one of Model A's three
signal components, while Model B operates independently in percentage space.

The system operates on 15-minute bins (I = 26 per 6.5-hour U.S. equity trading
day, 9:30-16:00 ET) and is fitted per individual stock. Each stock's model is
independent — no cross-sectional information is used.

### Revision Notes (Draft 3)

Changes from Draft 2 in response to Critique 2:

- **M1 resolved:** Redefined surprises in percentage space throughout Functions 7
  and 8. Surprises are now percentage-point departures: `surprise = actual_pct -
  hist_pct[i]`, where `actual_pct = volume[d,i] / total_volume[d]`. This ensures
  domain consistency: delta (a predicted percentage-point departure) is added
  directly to scaled_base (also a percentage). The deviation clamp now operates
  on quantities in the same domain, preserving the regression signal. This matches
  Humphery-Jenner's original formulation where "volume surprises" are departures
  of actual participation rates from historical participation rates. Updated
  surprise baseline documentation, sanity check 9, and all related references.
- **M2 resolved:** Made hist_avg rolling per training day in Function 5 (weight
  optimization). For each training day d, H_d is computed as the N_hist-day
  average ending at d, rather than using a single static hist_avg from
  train_end_date. This eliminates systematic bias for trending stocks. Function 7
  no longer uses H for surprises (resolved by M1 fix — surprises are now in
  percentage space using hist_pct), so the M2 rolling-H issue in Function 7 is
  moot. Updated Edge Case 10 to reflect the fix.
- **m1 resolved:** Validation evaluation in the regime grid search now uses a
  separate hist_avg computed from data ending before the validation period, not
  from val_days. This prevents the validation from unfairly benefiting from H
  computed on validation data.
- **m2 resolved:** Updated sanity check 9 surprise std dev estimate to ~0.005-0.015
  for percentage-space surprises, consistent with the M1 domain fix.
- **m3 resolved:** Added verification note to the no-intercept regression
  justification: mean surprise must be approximately zero for unbiased betas.
  Added diagnostic check.
- **m4 resolved:** Documented the zero-default assumption for missing/illiquid-bin
  surprises in build_surprise_regression, noting it is benign for the target
  universe (top 500 by dollar volume).

---

## Algorithm

### Model Description

**Model A (Raw Volume Forecast)** takes as input a stock symbol, a date, a target
bin index, and the volumes observed so far today. It produces a point forecast of
the number of shares that will trade in the target bin. Three signal components
are combined with weights that depend on the stock's current volume regime
(determined by the historical percentile of cumulative volume observed up to the
last observed bin).

**Model B (Volume Percentage Forecast)** takes the same inputs and produces a
forecast of what fraction of the day's total volume will trade in the next bin.
It adjusts a static historical percentage curve using recent volume-percentage
surprises (deviations of actual bin participation rates from historical
participation rates), subject to deviation constraints and a late-day switch-off
rule.

**Inputs:**
- Historical daily volume time series per bin: volume[stock, day, bin] — split-adjusted share counts.
- Current day's observed volumes through the current bin.

**Outputs:**
- Model A: V_hat[stock, day, bin] — forecasted raw volume (shares) for each target bin.
- Model B: pct_hat[stock, day, next_bin] — forecasted volume fraction for the next bin.

### Pseudocode

The system is organized into 11 functions: 7 for Model A training and prediction,
2 for Model B training and prediction, and 2 for orchestration.

#### Function 1: compute_seasonal_factors

Computes the per-bin rolling average over a long trailing window (6 months).
Used exclusively for deseasonalization of the intraday ARMA input series. This
is NOT the same as Component 1 (historical average); see Function 1a.

```
FUNCTION compute_seasonal_factors(volume_history, N_seasonal) -> seasonal_factors[1..I]
    # Input:  volume_history[day, bin] for trailing N_seasonal days
    # Output: seasonal_factors array of length I (used for deseasonalization only)

    FOR i = 1 TO I:
        values = [volume_history[d, i] FOR d IN trailing N_seasonal days]
        seasonal_factors[i] = mean(values)

    # Guard against zero seasonal factors (prevents division-by-zero in deseasonalization)
    # Replace zeros with minimum non-zero value across all bins
    nonzero_values = [sf FOR sf IN seasonal_factors IF sf > 0]
    IF len(nonzero_values) > 0:
        min_nonzero = min(nonzero_values)
    ELSE:
        min_nonzero = 1.0   # absolute fallback; should never trigger on liquid stocks

    FOR i = 1 TO I:
        IF seasonal_factors[i] == 0:
            seasonal_factors[i] = min_nonzero

    RETURN seasonal_factors
```

**References:** Satish et al. 2014, p.17 para 5 ("dividing by the intraday
amount of volume traded in that bin over the trailing six months"). The
zero-floor replacement is Researcher inference — the paper does not address
zero-volume bins.

#### Function 1a: compute_historical_average

Computes the per-bin rolling average over a shorter trailing window (N_hist
days). This is Model A Component 1 — the "Historical Window" in Exhibit 1.

```
FUNCTION compute_historical_average(volume_history, N_hist) -> hist_avg[1..I]
    # Input:  volume_history[day, bin] for trailing N_hist days
    # Output: hist_avg array of length I (Component 1 of Model A)

    FOR i = 1 TO I:
        values = [volume_history[d, i] FOR d IN trailing N_hist days]
        hist_avg[i] = mean(values)

    # Guard against zero averages (same logic as seasonal factors)
    nonzero_values = [v FOR v IN hist_avg IF v > 0]
    IF len(nonzero_values) > 0:
        min_nonzero = min(nonzero_values)
    ELSE:
        min_nonzero = 1.0

    FOR i = 1 TO I:
        IF hist_avg[i] == 0:
            hist_avg[i] = min_nonzero

    RETURN hist_avg
```

**References:** Satish et al. 2014, p.17 para 1 ("a rolling historical average
for the volume trading in a given 15-minute bin"), p.16 ("the average of daily
volume over the prior N days"), Exhibit 1 ("Next Bin (Prior 21 days)" feeding
into "Historical Window"). The 21-day window comes from Exhibit 1's label. The
paper introduces N as "a variable that we shall call N" (p.16) without
disclosing the optimized value; 21 is the illustrative value from Exhibit 1.
The zero-floor guard is Researcher inference.

**Distinction from Function 1:** Function 1 computes a 126-day (6-month) average
used only for deseasonalizing the intraday ARMA input. Function 1a computes a
21-day average used as Component 1 (H) in the weighted combination. The paper
treats these as separate quantities: Exhibit 1 shows "Prior 21 days" for the
Historical Window and the text says "trailing six months" for deseasonalization
(p.17 para 5). A shorter window for Component 1 makes it more responsive to
recent volume changes, while a longer window for deseasonalization provides a
more stable reference for removing the intraday pattern.

#### Function 2: fit_interday_arma

Fits I independent ARMA(p,q) models (one per bin) on daily volume series using
AICc for model selection.

```
FUNCTION fit_interday_arma(volume_history, N_interday_fit, p_max, q_max)
        -> interday_models[1..I]
    # Input:  volume_history[day, bin] for trailing N_interday_fit days
    # Output: array of I fitted ARMA model objects (or FALLBACK sentinel)

    FOR i = 1 TO I:
        series = [volume_history[d, i] FOR d IN trailing N_interday_fit days]
        n = len(series)

        best_aicc = +infinity
        best_model = None

        FOR p = 0 TO p_max:
            FOR q = 0 TO q_max:
                k = p + q + 1           # +1 for constant term
                IF n <= k + 1:
                    CONTINUE            # AICc denominator non-positive; skip

                model = fit_ARMA(series, order=(p, q), include_constant=True,
                                 method="MLE", enforce_stationarity=True,
                                 enforce_invertibility=True)

                IF NOT model.converged OR NOT model.is_stationary OR NOT model.is_invertible:
                    CONTINUE

                aic = -2 * model.log_likelihood + 2 * k
                aicc = aic + 2 * k * (k + 1) / (n - k - 1)

                IF aicc < best_aicc:
                    best_aicc = aicc
                    best_model = model

        IF best_model IS None:
            interday_models[i] = FALLBACK    # no valid model found
        ELSE:
            interday_models[i] = best_model

    RETURN interday_models
```

**References:** Satish et al. 2014, p.17 para 2 ("a per-symbol, per-bin ARMA(p, q)
model... We use nearly standard ARMA model-fitting techniques relying on
maximum-likelihood estimation, which selects an ARMA(p, q) model minimizing an
Akaike information criterion (AIC) as the test for the best model. In fitting
the ARMA model, we consider all values of p and q lags through five, as well as
a constant term. We depart from the standard technique in using the corrected
AIC, symbolized by AICc, as detailed by Hurvich and Tsai [1989, 1993]."). The
AICc formula is from Hurvich and Tsai (1989, 1993) as cited in Satish et al.
The convergence/stationarity/invertibility guards and the FALLBACK sentinel are
Researcher inference — the paper does not address convergence failures.

**ARMA model interface (applicable to both inter-day and intraday models):**

Each fitted ARMA model object exposes the following interface:

- `predict_at(d) -> float`: Returns the one-step-ahead forecast for day d,
  using information available through day d-1. Concretely, this is the
  conditional mean E[Y_d | Y_{d-1}, Y_{d-2}, ...] computed from the fitted
  ARMA coefficients and the observed series up through d-1. Used during
  training (Function 5) to get the forecast that would have been available
  on each historical day.

- `predict_next() -> float`: Returns the one-step-ahead forecast for the
  next unobserved time step after the last appended observation. Equivalent
  to `predict_at(d_last + 1)` where `d_last` is the most recently appended
  day. Used during live prediction (Function 6).

- `append_observation(value)`: Extends the model's observation buffer with
  a new data point and updates the internal state (AR/MA lag values). This
  is a Kalman-filter-style state update — the ARMA coefficients are NOT
  re-estimated; only the recursion state is advanced. After calling
  `append_observation(y_t)`, `predict_next()` returns the forecast for t+1.

- `make_state(observed_values) -> state`: (Intraday model only.) Creates a
  fresh prediction state from a list of deseasonalized observations WITHOUT
  mutating the model's internal state. Returns a state object that can be
  passed to `predict(state, steps)`.

- `predict(state, steps) -> list[float]`: (Intraday model only.) Produces
  multi-step-ahead forecasts from the given state. Returns a list of length
  `steps` where element k is the (k+1)-step-ahead forecast.

Researcher inference — the paper does not specify the ARMA prediction interface.
These semantics are standard for ARMA state-space implementations.

#### Function 3: fit_intraday_arma

Fits a single ARMA model per stock on deseasonalized intraday volume from the
most recent N_intraday_fit trading days. Each day is treated as an independent
segment to avoid spurious lag connections across the overnight gap.

```
FUNCTION fit_intraday_arma(volume_history, seasonal_factors, N_intraday_fit,
                           p_max_intra, q_max_intra) -> intraday_model
    # Input:  volume_history[day, bin], seasonal_factors[1..I], window size
    # Output: fitted ARMA model object (or FALLBACK sentinel)

    # Build deseasonalized series as independent daily segments
    segments = []       # list of daily deseasonalized volume arrays
    FOR d IN trailing N_intraday_fit days:
        daily_deseas = []
        FOR i = 1 TO I:
            daily_deseas.append(volume_history[d, i] / seasonal_factors[i])
        segments.append(daily_deseas)

    # Fitting approach: independent-segment (panel) ARMA
    # Each day is a separate segment of length I sharing the same ARMA
    # coefficients. The likelihood is the sum of per-segment log-likelihoods.
    # At each day boundary, the ARMA state (AR lag values and MA residuals)
    # resets to the unconditional mean.
    #
    # Concretely:
    #   L_total = SUM over segments s: L_s(phi, theta, sigma^2)
    # where L_s is the Gaussian log-likelihood for segment s of length I,
    # with the first max(p,q) observations in each segment used to initialize
    # the AR/MA recursion (exact or conditional MLE; conditional is simpler).
    #
    # Implementation note: this can be done by concatenating all segments
    # into a single series and zeroing out AR/MA lag contributions for indices
    # that cross segment boundaries. Equivalently, use a panel ARMA fitting
    # routine or manually sum per-segment likelihoods.

    n_segments = len(segments)
    n_usable_per_segment_max = I   # max observations per segment

    best_aicc = +infinity
    best_model = None

    FOR p = 0 TO p_max_intra:
        FOR q = 0 TO q_max_intra:
            k = p + q + 1
            # Effective sample size: each segment contributes I - max(p,q)
            # usable observations (the first max(p,q) initialize the state)
            usable_per_segment = I - max(p, q)
            IF usable_per_segment <= 0:
                CONTINUE
            n_eff = n_segments * usable_per_segment
            IF n_eff <= k + 1:
                CONTINUE

            model = fit_ARMA_panel(segments, order=(p, q),
                                   include_constant=True,
                                   method="conditional_MLE",
                                   enforce_stationarity=True,
                                   enforce_invertibility=True)

            IF NOT model.converged OR NOT model.is_stationary OR NOT model.is_invertible:
                CONTINUE

            aic = -2 * model.log_likelihood + 2 * k
            aicc = aic + 2 * k * (k + 1) / (n_eff - k - 1)

            IF aicc < best_aicc:
                best_aicc = aicc
                best_model = model

    IF best_model IS None:
        RETURN FALLBACK
    RETURN best_model
```

**Day-boundary handling detail:**

The independent-segment approach works as follows:

1. **During fitting:** Each day is a separate segment. The ARMA state
   (AR lags and MA residuals) is initialized at the start of each segment
   using the unconditional mean of the deseasonalized series (approximately
   1.0, since dividing by the mean removes the level). The first max(p,q)
   bins of each day are "burned" for state initialization — their prediction
   errors contribute to the likelihood but with reduced information.

2. **During prediction (start of new day):** When forecasting bin 1 of a
   new day with no observed bins (observed_deseas is empty), make_state([])
   initializes the ARMA state to the unconditional mean. The forecast for
   bin 1 is therefore the unconditional mean (~1.0 in deseasonalized space),
   which re-seasonalizes to the seasonal factor. This is the correct
   behavior: with no intraday information, the intraday ARMA contributes
   no signal beyond the seasonal pattern.

3. **During prediction (mid-day):** make_state(observed_deseas) runs the
   AR/MA recursion through the observed values, building up the ARMA state.
   The forecast for the next bin uses this accumulated state. After ~4 bins,
   the ARMA state is fully conditioned.

**Exhibit 1 "4 Bins Prior to Current Bin" note:** Exhibit 1 shows "4 Bins Prior
to Current Bin" as the input to "ARMA Intraday." This likely reflects the
diagram notation for the maximum AR lag (p_max_intra = 4), not a strict
limitation to using only the 4 most recent bins. The ARMA model uses all
observed bins to build its state via the recursion, but only the last p bins
directly enter the AR equation, and only the last q residuals enter the MA
equation. Using all observed bins (not just the last 4) is the standard and
correct ARMA prediction procedure.

**References:** Satish et al. 2014, p.17-18 ("Next, we fit an additional ARMA (p, q)
model over deseasonalized intraday bin volume data. The intraday data are
deseasonalized by dividing by the intraday amount of volume traded in that bin
over the trailing six months... we used AR lags with a value less than five...
We compute this model on a rolling basis over the most recent month."). The
independent-segment approach is Researcher inference — the paper says "rolling
basis over the most recent month" but does not specify overnight-boundary
handling. Treating each day as an independent segment prevents spurious lag-1
connections across the overnight gap and is standard practice for intraday
time-series models with overnight breaks. The AR constraint "less than five"
means p_max_intra = 4 (Satish et al. 2014, p.18). The MA order range is
unconstrained by the paper; we use q_max_intra = 5 to match the inter-day
search range. Researcher inference for q_max_intra.

#### Function 4: build_regime_classifier

Builds a regime classification system based on historical percentiles of
cumulative intraday volume.

```
FUNCTION build_regime_classifier(volume_history, N_regime_window, n_regimes)
        -> regime_classifier
    # Input:  volume_history[day, bin], lookback window, number of regimes
    # Output: object containing cumulative volume distributions and percentile cutoffs

    # Build cumulative volume distribution per bin
    cumvol_distributions = {}   # bin -> list of historical cumulative volumes
    FOR d IN trailing N_regime_window days:
        cumvol = 0
        FOR i = 1 TO I:
            cumvol += volume_history[d, i]
            cumvol_distributions.setdefault(i, []).append(cumvol)

    # Compute percentile cutoffs: equally spaced
    # For n_regimes=3: cutoffs = [33.3, 66.7]
    # For n_regimes=4: cutoffs = [25, 50, 75]
    cutoffs = [100 * k / n_regimes FOR k = 1 TO n_regimes - 1]

    # Precompute thresholds per bin
    thresholds = {}   # bin -> list of (n_regimes - 1) volume thresholds
    FOR i = 1 TO I:
        dist = sorted(cumvol_distributions[i])
        thresholds[i] = [percentile(dist, c) FOR c IN cutoffs]

    RETURN RegimeClassifier(thresholds=thresholds, n_regimes=n_regimes)
```

```
FUNCTION assign_regime(regime_classifier, last_observed_bin, cumulative_volume)
        -> regime_index
    # Returns integer in [0, n_regimes - 1]
    #
    # last_observed_bin: the most recent bin for which volume has been observed.
    #   This is the "information bin" — the bin index at which we look up the
    #   threshold vector. In training: i-1 (the bin before the target). In
    #   prediction: current_bin (the last observed bin). Both represent the
    #   same information state.
    # cumulative_volume: sum of observed volumes through last_observed_bin.

    IF last_observed_bin < 1:
        # No volume observed yet; default to middle regime
        RETURN regime_classifier.n_regimes // 2

    thresholds = regime_classifier.thresholds[last_observed_bin]
    regime = 0
    FOR t IN thresholds:
        IF cumulative_volume > t:
            regime += 1
        ELSE:
            BREAK
    RETURN regime
```

**References:** Satish et al. 2014, p.18 para 4 ("The final component of the model is
a dynamic weight overlay... We incorporate a notion of regime switching by
training several weight models for different historical volume percentile cutoffs
and, in our out-of-sample period, dynamically apply the appropriate weights
intraday based on the historical percentile of the observed cumulative volume.").
The specific percentile-based approach and the number of regimes are not disclosed
("treated as proprietary" per summary, p.155-157 of paper). The equally-spaced
percentile scheme and n_regimes as a tunable parameter are Researcher inference.
The default-to-middle-regime for last_observed_bin < 1 is Researcher inference —
the paper does not address pre-market regime assignment.

**Regime assignment standardization (Draft 2 fix for M2):** Both training
(Function 5) and prediction (Function 6) now call assign_regime with the same
convention: the bin index passed is the last bin for which volume has been
observed, and cumulative_volume includes that bin. In training, when forecasting
bin i, the last observed bin is i-1 and cumvol covers bins 1..(i-1). In
prediction, the last observed bin is current_bin and cumvol covers bins
1..current_bin. For next-bin forecasts (target_bin = current_bin + 1), these are
identical. For multi-step forecasts, prediction still uses current_bin (the true
last-observed bin), which is consistent with the information available at
prediction time.

#### Function 5: optimize_regime_weights

Finds optimal combination weights for each regime by minimizing MAPE on
in-sample data. Uses a time-varying (rolling) historical average H_d for each
training day d, rather than a single static hist_avg, to avoid systematic bias
for stocks with trending volume.

```
FUNCTION optimize_regime_weights(volume_history, N_hist, seasonal_factors,
                                  interday_models, intraday_model,
                                  regime_classifier, N_weight_train,
                                  min_volume_floor)
        -> weights[0..n_regimes-1][3]
    # Input:  trained components, regime classifier, training window
    #         N_hist: window for rolling historical average computation
    # Output: per-regime weight vectors [w_H, w_D, w_A]
    #
    # NOTE (Draft 3 fix for M2): H is computed per training day d as a
    # rolling N_hist-day average ending at d. This ensures that each training
    # day uses the H that would have been available at that time, avoiding
    # lookahead bias from using a future H for past training days.

    n_regimes = regime_classifier.n_regimes

    # Collect (H, D, A, actual, regime) tuples from training period
    samples_by_regime = {r: [] FOR r = 0 TO n_regimes - 1}

    FOR d IN trailing N_weight_train days:
        # Compute rolling hist_avg for day d (N_hist-day average ending at d)
        H_d = compute_historical_average(volume_history[:d], N_hist)

        cumvol = 0
        FOR i = 1 TO I:
            actual = volume_history[d, i]
            IF actual < min_volume_floor:
                cumvol += actual
                CONTINUE

            # Component 1: Historical rolling average (H) — rolling per day d
            H = H_d[i]

            # Component 2: Inter-day ARMA forecast (D)
            IF interday_models[i] IS FALLBACK:
                D = H               # fallback: use historical average
            ELSE:
                D = interday_models[i].predict_at(d)

            # Component 3: Intraday ARMA forecast (A)
            IF intraday_model IS FALLBACK:
                A = seasonal_factors[i]   # fallback: raw seasonal factor
            ELSE:
                observed_deseas = [volume_history[d, j] / seasonal_factors[j]
                                   FOR j = 1 TO i - 1]
                state = intraday_model.make_state(observed_deseas)
                A_deseas = intraday_model.predict(state, steps=1)[0]
                A = A_deseas * seasonal_factors[i]   # re-seasonalize

            # Regime assignment: use last-observed bin (i-1) and cumvol
            # through that bin. This matches prediction-time convention.
            regime = assign_regime(regime_classifier, i - 1, cumvol)
            samples_by_regime[regime].append((H, D, A, actual))
            cumvol += actual

    # Optimize weights per regime using multi-restart Nelder-Mead
    weights = {}
    FOR r = 0 TO n_regimes - 1:
        samples = samples_by_regime[r]
        IF len(samples) < min_samples_per_regime:
            weights[r] = [1/3, 1/3, 1/3]   # insufficient data; equal weights
            CONTINUE

        # MAPE loss function with exp-transformation for non-negativity
        FUNCTION mape_loss(w_log):
            w = exp(w_log)          # element-wise exp; w has 3 elements
            total_ape = 0
            count = 0
            FOR (H, D, A, actual) IN samples:
                forecast = w[0] * H + w[1] * D + w[2] * A
                forecast = max(forecast, 0)     # non-negativity clamp
                total_ape += abs(forecast - actual) / actual
                count += 1
            RETURN total_ape / count

        # Multi-restart optimization to avoid local minima
        best_result = None
        best_loss = +infinity

        # Starting points: equal weights, plus component-dominant initializations
        starting_points = [
            [log(1/3), log(1/3), log(1/3)],    # equal weights
            [log(0.8), log(0.1), log(0.1)],     # H-dominant
            [log(0.1), log(0.8), log(0.1)],     # D-dominant
            [log(0.1), log(0.1), log(0.8)],     # A-dominant
        ]

        FOR w_log_init IN starting_points:
            result = scipy.optimize.minimize(mape_loss, w_log_init,
                                             method="Nelder-Mead",
                                             options={maxiter: 1000,
                                                      xatol: 1e-4, fatol: 1e-6})
            IF result.fun < best_loss:
                best_loss = result.fun
                best_result = result

        # Convergence fallback: if optimized weights are worse than equal,
        # use equal weights
        equal_loss = mape_loss([log(1/3), log(1/3), log(1/3)])
        IF best_loss >= equal_loss:
            weights[r] = [1/3, 1/3, 1/3]
        ELSE:
            weights[r] = exp(best_result.x)   # convert back from log-space

    RETURN weights
```

**Computational cost of rolling H (Draft 3 M2 fix):** Computing H_d for each
training day requires N_weight_train calls to compute_historical_average, each
averaging over N_hist days and I bins. Total: O(N_weight_train * N_hist * I) =
O(63 * 21 * 26) ~ 34K operations. This is negligible relative to the ARMA
fitting and Nelder-Mead optimization. The alternative — precompute all H_d
values into a matrix hist_avg_rolling[d, i] before the training loop — is more
memory-efficient and avoids redundant computation.

**References:** Satish et al. 2014, p.18 para 4 ("a dynamic weight overlay on top of
these three components... that minimizes the error on in-sample data"). The
paper does not disclose the optimizer, objective function (MSE vs MAPE), or
whether weights are constrained to sum to 1. Researcher inference: we use MAPE
as objective because (a) the paper uses MAPE as its primary evaluation metric
for raw volume, and (b) MAPE is non-differentiable, motivating a
derivative-free optimizer (Nelder-Mead). The exp-transformation trick for
non-negativity and the absence of a sum-to-1 constraint are Researcher inference.
Omitting sum-to-1 allows the model to correct for scale bias in individual
components. The min_volume_floor exclusion in MAPE is Researcher inference.
Multi-restart with 4 starting points and the convergence fallback to equal
weights are Researcher inference — added to improve robustness of the
3-parameter optimization. The rolling H_d per training day is Researcher
inference based on the paper's description of H as a "rolling historical average"
(p.17 para 1), which implies time-varying computation.

**Conditioning note (m4 clarification):** The intraday ARMA component in
Function 5 conditions on observed bins 1..(i-1) from the same day. This is
intentional and matches the operational prediction setup: at prediction time,
the model also conditions on earlier-same-day bins. This is NOT data leakage
because the information used (earlier bins) would be available at the time the
forecast is made. The training MAPE thus measures the model's expected
real-world performance.

#### Function 6: forecast_raw_volume

Produces a raw volume forecast for a specific (stock, day, target_bin) given
observed volumes. This is the main Model A prediction function.

```
FUNCTION forecast_raw_volume(model_a, stock, day, target_bin, observed_volumes)
        -> V_hat
    # Input:  trained Model A state, stock id, date, target bin (1..I),
    #         observed_volumes dict {bin_index: volume} for bins seen so far today
    # Output: forecasted volume (shares) for target_bin

    current_bin = max(observed_volumes.keys()) IF observed_volumes ELSE 0

    # Component 1: Historical rolling average (from N_hist window)
    H = model_a.hist_avg[target_bin]

    # Component 2: Inter-day ARMA
    IF model_a.interday_models[target_bin] IS FALLBACK:
        D = H
    ELSE:
        D = model_a.interday_models[target_bin].predict_next()

    # Component 3: Intraday ARMA (conditioned on today's observations)
    IF model_a.intraday_model IS FALLBACK:
        A = model_a.seasonal_factors[target_bin]
    ELSE:
        # Deseasonalize observed volumes
        observed_deseas = [observed_volumes[j] / model_a.seasonal_factors[j]
                           FOR j = 1 TO current_bin IF j IN observed_volumes]
        # Create fresh state from observations (pure — no mutation of model)
        state = model_a.intraday_model.make_state(observed_deseas)
        # Multi-step forecast to reach target_bin
        steps_ahead = target_bin - current_bin
        forecasts_deseas = model_a.intraday_model.predict(state, steps=steps_ahead)
        A_deseas = forecasts_deseas[-1]      # last element is target_bin forecast
        A = A_deseas * model_a.seasonal_factors[target_bin]   # re-seasonalize

    # Regime classification — use last-observed bin and cumvol through it
    cumvol = sum(observed_volumes.values())
    regime = assign_regime(model_a.regime_classifier, current_bin, cumvol)

    # Weighted combination
    w = model_a.weights[regime]
    V_hat = w[0] * H + w[1] * D + w[2] * A

    # Non-negativity clamp
    V_hat = max(V_hat, 0.0)

    RETURN V_hat
```

**References:** Satish et al. 2014, p.17-18, Exhibit 1 (data flow diagram showing
three components feeding into Dynamic Weights Engine). The formula is implicit
in the paper's description of "a dynamic weight overlay on top of these three
components." The re-seasonalization step: "before passing intraday ARMA forecasts,
we re-seasonalize these forecasts via multiplication" (p.18 para 3). The
non-negativity clamp is Researcher inference — volume cannot be negative.

**Purity requirement (Researcher inference):** make_state() creates a NEW state
object from the observed deseasonalized values. It does NOT mutate the
intraday_model's internal state. This is critical because: (a) during live
prediction, forecast_raw_volume is called multiple times per day with different
current_bin values, and mutating state would double-condition on earlier
observations; (b) during Model B training, the function is called across
multiple historical days, and state leakage across days would corrupt forecasts.

#### Function 7: train_percentage_model

Trains the Model B surprise regression: fits an OLS (no intercept) of future
percentage surprise on lagged percentage surprises. Surprises are computed in
percentage space as departures of actual bin participation rates from historical
participation rates, following Humphery-Jenner's (2011) formulation.

```
FUNCTION train_percentage_model(volume_history, N_regression_fit,
                                 L_max, min_volume_floor)
        -> percentage_model
    # Input:  volume history, regression window, max lags, volume floor
    # Output: fitted percentage model (beta coefficients, optimal L, hist_pct curve)
    #
    # NOTE (Draft 3 M1 fix): Surprises are now defined in percentage space:
    #   surprise = actual_pct - hist_pct[i]
    # where actual_pct = volume[d,i] / total_volume[d]. This ensures domain
    # consistency with the percentage-space application in Function 8.
    # Previous drafts used raw-volume surprises (actual - H) / H, which
    # produced values ~25x larger than the percentage quantities they were
    # added to, effectively disabling the regression signal via the deviation
    # clamp.

    # Step 1: Compute historical percentage curve (static baseline)
    hist_pct = ARRAY[1..I]
    FOR i = 1 TO I:
        total_per_day = [sum(volume_history[d, j] FOR j = 1 TO I)
                         FOR d IN trailing N_regression_fit days]
        hist_pct[i] = mean(volume_history[d, i] / total_per_day[d]
                           FOR d IN trailing N_regression_fit days)

    # Step 2: Compute percentage-space surprises for training data
    # Surprise = actual participation rate - historical participation rate
    # This follows Humphery-Jenner (2011): "volume surprises" are departures
    # of actual participation rates from a naive historical forecast (the
    # historical VWAP curve).
    surprise_data = []   # list of (day, bin, surprise_value)
    FOR d IN trailing N_regression_fit days:
        total_vol_d = sum(volume_history[d, j] FOR j = 1 TO I)
        IF total_vol_d < min_volume_floor:
            CONTINUE    # skip days with negligible total volume
        FOR i = 1 TO I:
            actual_pct = volume_history[d, i] / total_vol_d
            surprise = actual_pct - hist_pct[i]
            surprise_data.append((d, i, surprise))

    # Step 3: Select optimal number of lags via blocked time-series CV
    best_L = 1
    best_cv_error = +infinity

    FOR L_candidate = 1 TO L_max:
        # Blocked K-fold CV: partition days into K contiguous blocks
        K = 5
        days = [distinct days in trailing N_regression_fit]
        block_size = len(days) // K
        # Remainder days: assign to the last fold
        # E.g., 63 days / 5 folds = 12 days each, last fold gets 15 days
        cv_errors = []

        FOR fold = 0 TO K - 1:
            IF fold < K - 1:
                test_days = days[fold * block_size : (fold + 1) * block_size]
            ELSE:
                test_days = days[fold * block_size :]   # last fold gets remainder
            train_days = days NOT IN test_days

            # Build regression matrices from train_days
            X_train, y_train = build_surprise_regression(
                surprise_data, train_days, L_candidate, L_max)
            beta = ols_no_intercept(X_train, y_train)

            # Evaluate on test_days
            X_test, y_test = build_surprise_regression(
                surprise_data, test_days, L_candidate, L_max)
            y_pred = X_test @ beta
            cv_errors.append(mean(abs(y_test - y_pred)))

        mean_cv_error = mean(cv_errors)
        IF mean_cv_error < best_cv_error:
            best_cv_error = mean_cv_error
            best_L = L_candidate

    # Step 4: Fit final model with optimal L on all training data
    X_all, y_all = build_surprise_regression(
        surprise_data, all_days, best_L, L_max)
    beta_final = ols_no_intercept(X_all, y_all)

    RETURN PercentageModel(beta=beta_final, L=best_L, hist_pct=hist_pct)
```

```
FUNCTION build_surprise_regression(surprise_data, days, L, L_max)
        -> (X, y)
    # Build regression: y[t] = beta_1 * surprise[t-1] + ... + beta_L * surprise[t-L]
    # Only use bins > L_max to avoid edge effects (early bins lack sufficient lags)
    #
    # NOTE (m4 documentation): surprises_today.get(i - lag, 0.0) returns 0.0
    # for missing bins. In practice, surprise_data contains entries for all
    # (day, bin) combinations in the training window (days with total volume
    # below min_volume_floor are excluded entirely in Step 2, so partial-day
    # gaps should not occur). For the target universe (top 500 by dollar volume),
    # all 26 bins have meaningful volume. If applying to illiquid stocks where
    # some bins may have zero volume, consider excluding bins with
    # actual_pct < epsilon from the lag vector, or using NaN-aware regression.
    X_rows = []
    y_values = []
    FOR d IN days:
        surprises_today = {i: s FOR (dd, i, s) IN surprise_data IF dd == d}
        FOR i = (L_max + 1) TO I:
            lags = [surprises_today.get(i - lag, 0.0) FOR lag = 1 TO L]
            X_rows.append(lags)
            y_values.append(surprises_today.get(i, 0.0))
    RETURN (array(X_rows), array(y_values))
```

```
FUNCTION ols_no_intercept(X, y) -> beta
    # OLS without intercept: beta = (X^T X)^{-1} X^T y
    RETURN solve(X.T @ X, X.T @ y)
```

**References:** Satish et al. 2014, p.18-19 ("we focused on an approach developed by
Humphery-Jenner [2011]... The key result is that training a model on decomposed
volume, or departures from a historical average approach, aids the volume
percentage forecasting problem... we were able to identify the optimal number of
model terms for U.S. stocks in our in-sample data"). The Humphery-Jenner (2011)
method works with "volume surprises — deviations from a naive historical
forecast" where the "naive historical forecast" is a percentage curve (VWAP
profile) and the surprises are percentage-point departures of actual
participation rates from historical participation rates.

**Percentage-space surprise rationale (Draft 3 M1 fix):**

In Humphery-Jenner's original framework, surprises operate in percentage space:
the naive historical forecast IS a percentage (historical VWAP curve), and
surprises are departures from that percentage. Satish et al. describe this as
"departures from a historical average approach" (p.18-19) and "computing
decomposed volumes (volume surprises)" (p.19). In the volume percentage context,
"decomposed volumes" are the difference between actual and historical
participation rates.

Using percentage-space surprises ensures domain consistency throughout Model B:
- hist_pct[i] ~ 1/26 ~ 0.038 (a percentage)
- surprise = actual_pct - hist_pct[i] ~ 0.005-0.015 std dev (a percentage-point departure)
- delta = dot(beta, lag_vector) ~ 0.001-0.005 (a predicted percentage-point departure)
- scaled_base ~ 0.038 (a percentage)
- adjusted = scaled_base + delta (percentage + percentage-point = percentage)

All quantities are in the same domain. The deviation clamp max_delta = 0.10 *
0.038 = 0.0038 is comparable to typical delta values, allowing the regression
signal through.

**Surprise baseline alternatives (documented for completeness):**
- **(a) Percentage-space surprise (implemented):** surprise = actual_pct - hist_pct[i].
  Matches Humphery-Jenner's original formulation. Ensures domain consistency.
  Produces small, well-scaled surprises.
- **(b) Raw-volume surprise with multiplicative application:**
  surprise = (actual - H) / H, applied as adjusted = scaled_base * (1 + delta).
  Preserves raw-volume surprise definition but changes the application to
  multiplicative. Deviation clamp becomes |delta| <= max_deviation = 0.10.
- **(c) Raw-volume surprise with scaled additive application:**
  surprise = (actual - H) / H, applied as adjusted = scaled_base + delta * hist_pct.
  Mathematically equivalent to (a) if regression coefficients adapt.
- **(d) Raw-volume surprise (Draft 2 approach, deprecated):**
  surprise = (actual - H) / H, applied as adjusted = scaled_base + delta.
  Domain mismatch: delta ~ 0.1 vs. scaled_base ~ 0.038. The deviation clamp
  absorbs 96% of the regression signal. Not recommended.

Option (a) is cleanest and most consistent with Humphery-Jenner. Option (b)
is a viable alternative requiring the least conceptual change from Draft 2.

The no-intercept choice: p.19 ("we perform both regressions without the
inclusion of a constant term") — this quote explicitly refers to the VWAP-error
validation regressions. We apply the same principle to the surprise regression:
omitting the intercept forces zero adjustment when all recent surprises are zero,
which is the correct behavior (if the historical percentage curve has been
accurate recently, no dynamic adjustment is needed). Researcher inference on
applying this to the surprise regression.

**No-intercept validity requirement (Draft 3 m3 fix):** The no-intercept
assumption requires that mean surprise is approximately zero. This holds when
hist_pct is computed from the same or a similar window as the surprise training
data, since hist_pct is the mean of actual_pct by construction. If hist_pct
were computed from a substantially different period (e.g., a very long window
while surprises use a short window), mean surprise could be nonzero, biasing the
slope coefficients. Verify via Sanity Check 9: if mean surprise deviates
significantly from zero (e.g., |mean| > 0.001 in percentage-point units),
consider adding an intercept term or recomputing hist_pct from a window that
better matches the surprise training window.

The blocked time-series CV methodology is Researcher inference — the paper says
they "identified the optimal number of model terms" (p.19) but does not disclose
the selection procedure.

#### Function 8: forecast_volume_percentage

Produces a forecast of the volume fraction for the next bin using the dynamic
VWAP adjustment. Surprises are computed in percentage space as departures of
actual bin participation rates from historical participation rates.

```
FUNCTION forecast_volume_percentage(model_a, pct_model, stock, day,
                                     current_bin, observed_volumes,
                                     max_deviation, pct_switchoff,
                                     min_volume_floor)
        -> pct_forecast
    # Input:  trained models, current state, constraint parameters
    # Output: forecasted volume fraction for bin (current_bin + 1)

    next_bin = current_bin + 1
    IF next_bin > I:
        RETURN 0.0    # no more bins

    # Step 1: Compute V_total_est (estimated total daily volume)
    #   = observed volume + sum of conditioned raw forecasts for remaining bins
    observed_total = sum(observed_volumes.values())
    remaining_forecast = 0
    FOR j = next_bin TO I:
        remaining_forecast += forecast_raw_volume(
            model_a, stock, day, target_bin=j, observed_volumes=observed_volumes)
    V_total_est = observed_total + remaining_forecast

    # Step 2: Compute percentage-space surprise vector
    # Surprise = actual participation rate - historical participation rate
    # This is consistent with training (Function 7): both use percentage-space
    # surprises, ensuring the regression coefficients are applied to the same
    # type of input they were trained on.
    surprises = []
    FOR j = 1 TO current_bin:
        actual_pct = observed_volumes[j] / V_total_est IF V_total_est > 0 ELSE 0.0
        surprise = actual_pct - pct_model.hist_pct[j]
        surprises.append(surprise)

    # Step 3: Compute surprise-based adjustment (delta)
    L = pct_model.L
    IF current_bin < L:
        delta = 0.0        # insufficient lags for regression
    ELSE:
        # Check switch-off condition
        observed_frac = observed_total / V_total_est IF V_total_est > 0 ELSE 0
        IF observed_frac >= pct_switchoff:
            delta = 0.0    # revert to historical curve
        ELSE:
            lag_vector = [surprises[current_bin - lag]
                          FOR lag = 1 TO L]  # most recent L surprises
            delta = dot(pct_model.beta, lag_vector)

    # Step 4: Scale baseline to remaining volume fraction
    observed_hist_frac = sum(pct_model.hist_pct[j] FOR j = 1 TO current_bin)
    remaining_hist_frac = 1.0 - observed_hist_frac
    actual_remaining_frac = 1.0 - (observed_total / V_total_est) IF V_total_est > 0 ELSE 1.0

    IF remaining_hist_frac > 0:
        scale = actual_remaining_frac / remaining_hist_frac
    ELSE:
        scale = 1.0

    scaled_base = scale * pct_model.hist_pct[next_bin]

    # Step 5: Apply adjustment with deviation constraint
    # delta is in percentage-point space (same domain as scaled_base),
    # so direct addition is valid. The clamp limits the departure to
    # max_deviation * scaled_base in either direction.
    adjusted = scaled_base + delta
    max_delta = max_deviation * scaled_base
    adjusted = clip(adjusted, scaled_base - max_delta, scaled_base + max_delta)

    # Step 6: Non-negativity
    pct_forecast = max(adjusted, 0.0)

    # Step 7: Last-bin special case
    IF next_bin == I:
        pct_forecast = actual_remaining_frac   # assign all remaining fraction

    RETURN pct_forecast
```

**Train/predict surprise consistency note (Draft 3):**

In training (Function 7), surprises use the true total daily volume:
  `actual_pct = volume[d, i] / total_volume[d]`

In prediction (Function 8), surprises use estimated total volume:
  `actual_pct = observed_volumes[j] / V_total_est`

This introduces a mild inconsistency: the training surprises use exact total
volume (known after market close), while prediction surprises use an estimate
(observed + forecasted remaining). The inconsistency diminishes as the day
progresses (V_total_est becomes more accurate). Early-day prediction surprises
may be noisier than their training counterparts, but the deviation clamp
provides a safety bound. This is inherent to any online forecasting system
and is consistent with Humphery-Jenner's framework.

**Percentage coherence discussion (M5 resolution from Draft 2, retained):**

The percentage forecasts produced bin-by-bin do NOT strictly sum to 1.0 over the
full day. This is because: (a) V_total_est changes as more bins are observed,
causing the scale factor to shift between calls; (b) the deviation clamp may
distort the scaled baseline; (c) there is no explicit renormalization step.

**Why this is acceptable for VWAP algorithms:** VWAP execution algorithms
typically use only the next-bin forecast at each step to decide what fraction of
the remaining order to execute in the upcoming interval. They do not require
full-day coherent percentage schedules. The paper evaluates Model B's
performance using per-bin absolute deviation (MAD), not cumulative deviation,
confirming the per-bin focus.

**Mitigation mechanisms already in place:**
- The scaled_base recomputation at each call implicitly redistributes remaining
  volume proportionally across future bins.
- The last-bin special case (Step 7) ensures the final bin absorbs any residual.
- The deviation constraint (10%) limits per-bin distortion.
- The switch-off rule (80%) reverts to the historical curve for late-day bins,
  when coherence matters most.

**Optional renormalization (for applications requiring full-day coherence):**
If a downstream consumer needs a full-day coherent schedule, add the following
post-processing step after computing all pct_hat values for remaining bins:

```
# Optional: renormalize remaining-bin forecasts to sum to actual_remaining_frac
raw_forecasts = [pct_hat[j] FOR j = next_bin TO I]   # batch compute all
total_raw = sum(raw_forecasts)
IF total_raw > 0:
    FOR j = next_bin TO I:
        pct_hat[j] = pct_hat[j] * (actual_remaining_frac / total_raw)
```

This is Researcher inference. The paper does not describe a renormalization step.

**References:** Satish et al. 2014, p.18-19 (Volume Percentage Forecast Methodology),
p.24 ("depart no more than 10% away from a historical VWAP curve", referencing
Humphery-Jenner 2011; "once 80% of the day's volume is reached, return to a
historical approach"). The paper notes: "we developed a separate method for
computing the deviation bounds" (p.19), indicating their production system used
proprietary adaptive bounds, not the fixed 10%. The scaled_base renormalization
approach is Researcher inference — the paper does not describe an explicit
renormalization step. The dual raw-forecast use (conditioned for V_total_est
in Step 1, percentage-space for surprises in Step 2) is Researcher inference.
V_total_est uses the full conditioned Model A because it needs the best available
volume estimate; surprises use percentage departures because that matches the
training formulation.

#### Function 9: train_full_model

Top-level training function that fits all Model A and Model B components.

```
FUNCTION train_full_model(volume_history, stock, train_end_date, params)
        -> (model_a, pct_model)
    # Step 1: Seasonal factors (deseasonalization basis — 6-month window)
    seasonal_factors = compute_seasonal_factors(
        volume_history[:train_end_date], params.N_seasonal)

    # Step 1a: Historical average (Component 1 — N_hist window, at train_end_date)
    # This is used for prediction-time H. For training, rolling H_d is
    # computed inside Function 5.
    hist_avg = compute_historical_average(
        volume_history[:train_end_date], params.N_hist)

    # Step 2: Inter-day ARMA models (Component 2)
    interday_models = fit_interday_arma(
        volume_history[:train_end_date], params.N_interday_fit,
        params.p_max_inter, params.q_max_inter)

    # Step 3: Intraday ARMA model (Component 3)
    intraday_model = fit_intraday_arma(
        volume_history[:train_end_date], seasonal_factors,
        params.N_intraday_fit, params.p_max_intra, params.q_max_intra)

    # Step 4: Regime classifier
    # Grid search over candidate regime counts
    best_config = None
    best_oos_mape = +infinity

    FOR n_reg IN params.regime_candidates:
        regime_classifier = build_regime_classifier(
            volume_history[:train_end_date], params.N_regime_window, n_reg)

        # Split training window: last 21 days as validation
        val_days = last 21 days before train_end_date
        train_days = N_weight_train days before val_days

        # NOTE (Draft 3 m1 fix): optimize_regime_weights now takes N_hist
        # and computes rolling H_d internally. No external hist_avg is passed.
        weights = optimize_regime_weights(
            volume_history[train_days], params.N_hist, seasonal_factors,
            interday_models, intraday_model, regime_classifier,
            len(train_days), params.min_volume_floor)

        # Evaluate on validation period
        # NOTE (Draft 3 m1 fix): Use a separate hist_avg for validation,
        # computed from data ending BEFORE val_days start. This prevents
        # the validation from unfairly benefiting from H computed on
        # validation data.
        val_hist_avg = compute_historical_average(
            volume_history[:val_days_start], params.N_hist)
        temp_model_a = ModelA(val_hist_avg, seasonal_factors, interday_models,
                              intraday_model, regime_classifier, weights)
        val_mape = compute_evaluation_mape(
            temp_model_a, volume_history, val_days, params.min_volume_floor)

        IF val_mape < best_oos_mape:
            best_oos_mape = val_mape
            best_config = (n_reg, regime_classifier, weights)

    n_reg, regime_classifier, weights = best_config

    # Log soft constraint check (11-term observation)
    max_interday_k = max(interday_models[i].k FOR i = 1 TO I
                         IF interday_models[i] IS NOT FALLBACK)
    intraday_k = intraday_model.k IF intraday_model IS NOT FALLBACK ELSE 0
    combined_terms = max_interday_k + intraday_k
    IF combined_terms > 10:
        LOG_WARNING("Combined ARMA terms = %d exceeds 10 (soft limit)", combined_terms)

    model_a = ModelA(hist_avg, seasonal_factors, interday_models, intraday_model,
                     regime_classifier, weights)

    # Step 5: Percentage model (Model B)
    # NOTE (Draft 3 M1 fix): train_percentage_model no longer needs model_a
    # as input — surprises are computed in percentage space from volume_history
    # directly, without using H.
    pct_model = train_percentage_model(
        volume_history[:train_end_date],
        params.N_regression_fit, params.L_max, params.min_volume_floor)

    RETURN (model_a, pct_model)
```

```
FUNCTION compute_evaluation_mape(model_a, volume_history, eval_days,
                                  min_volume_floor) -> mape
    # Compute MAPE over evaluation period using conditioned forecasts.
    # This is the OPERATIONAL evaluation: each bin's forecast uses observed
    # bins 1..(i-1) from the same day, matching real-world prediction setup.
    # This is NOT data leakage because earlier-same-day bins are available
    # at the time the forecast would be made in production.
    total_ape = 0
    count = 0
    FOR d IN eval_days:
        FOR i = 1 TO I:
            actual = volume_history[d, i]
            IF actual < min_volume_floor:
                CONTINUE
            # Conditioned forecast using observed bins 1..(i-1)
            observed = {j: volume_history[d, j] FOR j = 1 TO i - 1}
            predicted = forecast_raw_volume(model_a, stock, d,
                                            target_bin=i, observed_volumes=observed)
            total_ape += abs(predicted - actual) / actual
            count += 1
    IF count == 0:
        RETURN +infinity
    RETURN total_ape / count
```

**References:** The training pipeline order (seasonal -> historical avg -> inter-day
-> intraday -> regime -> weights -> percentage) follows the dependency structure
implied by Satish et al. 2014, p.17-19. The regime grid search over {3, 4, 5}
and validation split are Researcher inference — the paper treats regime thresholds
as proprietary. The 11-term soft constraint: Satish et al. 2014, p.18 ("As a
result, we fit each symbol with a dual ARMA model having fewer than 11 terms").
The "As a result" phrasing indicates this is an observed empirical outcome, not
a prescriptive constraint. Researcher inference on treating it as a soft guardrail.
The separate validation hist_avg (m1 fix) is Researcher inference — the paper
does not discuss validation methodology.

#### Function 10: run_daily

Daily orchestration function for live operation.

```
FUNCTION run_daily(model_a, pct_model, stock, today, params)
    # Called at market open. Runs throughout the trading day.

    observed_volumes = {}

    # Pre-market: generate initial forecasts for all bins (unconditional)
    initial_forecasts = {}
    FOR i = 1 TO I:
        initial_forecasts[i] = forecast_raw_volume(
            model_a, stock, today, target_bin=i, observed_volumes={})

    # Intraday loop: process each bin as volume is observed
    FOR current_bin = 1 TO I:
        # Record observed volume for current_bin (from market data feed)
        observed_volumes[current_bin] = get_actual_volume(stock, today, current_bin)

        # Update conditioned raw forecasts for remaining bins
        updated_forecasts = {}
        FOR j = current_bin + 1 TO I:
            updated_forecasts[j] = forecast_raw_volume(
                model_a, stock, today, target_bin=j,
                observed_volumes=observed_volumes)

        # Forecast volume percentage for next bin
        IF current_bin < I:
            pct_next = forecast_volume_percentage(
                model_a, pct_model, stock, today, current_bin,
                observed_volumes, params.max_deviation,
                params.pct_switchoff, params.min_volume_floor)

        # Emit forecasts to downstream consumers (VWAP algo, scheduling, etc.)
        emit_forecasts(updated_forecasts, pct_next)

    # End of day: update Model A state for next day
    # Inter-day ARMA: append today's observation (advances recursion state,
    # does NOT re-estimate coefficients)
    FOR i = 1 TO I:
        IF model_a.interday_models[i] IS NOT FALLBACK:
            model_a.interday_models[i].append_observation(observed_volumes[i])

    # Historical average: recompute (cheap rolling update — add today, drop oldest)
    model_a.hist_avg = compute_historical_average(
        volume_history_including_today, params.N_hist)

    # Seasonal factors: recompute (cheap rolling update)
    model_a.seasonal_factors = compute_seasonal_factors(
        volume_history_including_today, params.N_seasonal)

    # Check re-estimation schedule
    IF days_since_last_full_reestimation >= params.reestimation_interval:
        model_a, pct_model = train_full_model(
            volume_history_including_today, stock, today, params)
```

**References:** The daily workflow structure (pre-market unconditional forecasts,
intraday conditioned updates, end-of-day state update) is inferred from the
paper's discussion of the model's online usage pattern. Satish et al. 2014,
Exhibit 1 shows the data flow from "Current Bin" through the model to "Next Bin
Forecast." The distinction between daily state updates (append_observation for
inter-day ARMA, rolling update for hist_avg) and periodic full re-estimation is
Researcher inference — the paper says "We compute this model on a rolling basis"
(p.18) without specifying the update frequency. The re-estimation schedule is
Researcher inference.

### Data Flow

```
INPUT: volume_history[stock, day, bin] — split-adjusted share counts
       Shape: (n_stocks, n_days, I=26)
       Type: float64 (share counts can be fractional after split adjustment)

                           TRAINING PHASE
                           ==============

volume_history ──► compute_seasonal_factors(N_seasonal=126) ──► seasonal_factors[26]
       │                                                              │ (deseasonalization)
       │                                                              │
       ├──► compute_historical_average(N_hist=21) ──► hist_avg[26]    │
       │         (at train_end_date; for prediction)                  │
       │                                                              │
       ├──────────────► fit_interday_arma() ──► interday_models[26]   │
       │                                              │               │
       │                                              ▼               ▼
       ├──────────────► fit_intraday_arma() ──► intraday_model (single ARMA)
       │
       ├──────────────► build_regime_classifier() ──► regime_classifier
       │                                                    │
       │                                                    ▼
       ├──────────────► optimize_regime_weights() ──► weights[n_regimes][3]
       │                    (rolling H_d per training day)
       │
       ╰──────────── ModelA(hist_avg, seasonal_factors, interday_models,
                            intraday_model, regime_classifier, weights)

                   train_percentage_model() ◄──── volume_history
                     (surprises = actual_pct - hist_pct)
                         │
                         ▼
                   PercentageModel (beta[L], hist_pct[26])

                           PREDICTION PHASE
                           ================

observed_volumes ──► forecast_raw_volume()
                         │
                         ├──► V_hat (raw volume forecast per bin)
                         │
                         ╰──► forecast_volume_percentage()
                                   │ (pct-space surprises from observed_pct - hist_pct)
                                   ╰──► pct_hat (volume fraction for next bin)
```

**Intermediate types and shapes:**

| Variable | Shape | Type | Description |
|----------|-------|------|-------------|
| seasonal_factors | (26,) | float64 | Per-bin 126-day average volume (deseasonalization) |
| hist_avg | (26,) | float64 | Per-bin 21-day average volume (Component 1, H) |
| interday_models | (26,) | ARMA or FALLBACK | Per-bin fitted ARMA models |
| intraday_model | scalar | ARMA or FALLBACK | Single deseasonalized ARMA |
| regime_classifier | object | RegimeClassifier | Thresholds per bin |
| weights | (n_regimes, 3) | float64 | Per-regime [w_H, w_D, w_A] |
| beta | (L,) | float64 | Surprise regression coefficients |
| hist_pct | (26,) | float64 | Historical volume percentage curve (sums to ~1.0) |
| V_hat | scalar | float64 | Raw volume forecast (shares) |
| pct_hat | scalar | float64 | Volume fraction forecast [0, 1] |

### Variants

We implement the full dual-model system as described in Satish et al. (2014).
This is the most complete variant presented in the paper, incorporating all four
components of Model A (historical average + inter-day ARMA + intraday ARMA +
regime-switching weights) and the dynamic VWAP extension for Model B.

Simpler variants exist:
- **Historical average only**: Component 1 alone (hist_avg). This is the baseline
  the paper benchmarks against.
- **Single ARMA**: Either inter-day or intraday ARMA alone, without regime
  switching. Not tested in the paper as a standalone variant.
- **Static VWAP**: Model B without the surprise regression (hist_pct alone).
  This is the Humphery-Jenner (2011) baseline.

We chose the full variant because: (a) it achieves the best reported performance
(24% MAPE reduction, 9.1% VWAP improvement); (b) the components are modular, so
simpler variants can be obtained by setting specific weights to zero.

---

## Parameters

| Parameter | Description | Recommended Value | Sensitivity | Range |
|-----------|-------------|-------------------|-------------|-------|
| I | Number of bins per trading day (15-min bins, 9:30-16:00 ET) | 26 | N/A (structural) | {13, 26, 78} |
| N_seasonal | Trailing window for deseasonalization (trading days) | 126 | Low | [63, 252] |
| N_hist | Rolling window for Component 1 historical average (trading days) | 21 | Medium-high | [10, 63] |
| N_interday_fit | Fitting window for inter-day ARMA (trading days) | 63 | Medium | [42, 126] |
| p_max_inter | Maximum AR order for inter-day ARMA | 5 | Low | [3, 5] |
| q_max_inter | Maximum MA order for inter-day ARMA | 5 | Low | [3, 5] |
| N_intraday_fit | Rolling window for intraday ARMA (trading days) | 21 | Medium | [15, 42] |
| p_max_intra | Maximum AR order for intraday ARMA | 4 | Low | [2, 4] |
| q_max_intra | Maximum MA order for intraday ARMA | 5 | Low | [3, 5] |
| N_regime_window | Lookback for cumulative volume distribution (trading days) | 63 | Medium | [21, 126] |
| regime_candidates | Candidate regime counts for grid search | {3, 4, 5} | High | [2, 6] |
| N_weight_train | Training window for weight optimization (trading days) | 63 | Medium | [42, 126] |
| min_samples_per_regime | Minimum samples per regime bucket for weight fitting | 50 | Low | [20, 100] |
| min_volume_floor | Minimum volume for MAPE computation / day exclusion (shares) | 100 | Low | [50, 500] |
| N_regression_fit | Training window for surprise regression (trading days) | 63 | Medium | [21, 126] |
| L_max | Maximum number of lagged surprise terms to consider | 5 | Medium | [3, 8] |
| max_deviation | Maximum relative departure from historical pct curve | 0.10 | Medium-high | [0.05, 0.20] |
| pct_switchoff | Cumulative volume fraction triggering switch to historical curve | 0.80 | Medium | [0.70, 0.90] |
| reestimation_interval | Days between full model re-estimation | 21 | Low | [5, 63] |

**Parameter sources:**
- I = 26: Satish et al. 2014, p.16 ("based on 15-minute bins, and there are 26 such bins in a trading day").
- N_seasonal = 126: Satish et al. 2014, p.17 ("trailing six months"). 126 trading days ~ 6 calendar months. Used for deseasonalization only.
- N_hist = 21: Satish et al. 2014, Exhibit 1 caption ("Prior 21 days"). This is the window for Component 1 (H), the historical average used in the weighted combination. The paper introduces N as "a variable that we shall call N" (p.16) without disclosing the optimized value; 21 is the illustrative value from Exhibit 1. Note: N_hist and N_seasonal serve different purposes and should generally differ — N_hist should be shorter (more responsive) while N_seasonal should be longer (more stable deseasonalization).
- N_interday_fit = 63: Researcher inference. The paper says "prior 5 days" in Exhibit 1, which refers to the AR lag memory (p_max=5), not the fitting window. Fitting an ARMA(5,5) on 5 observations is statistically infeasible. 63 trading days (3 months) provides sufficient data while adapting to regime changes.
- p_max_inter = 5, q_max_inter = 5: Satish et al. 2014, p.17 ("we consider all values of p and q lags through five").
- N_intraday_fit = 21: Satish et al. 2014, p.18 ("the most recent month"). 21 trading days ~ 1 calendar month.
- p_max_intra = 4: Satish et al. 2014, p.18 ("AR lags with a value less than five"). "Less than five" means p in {0,1,2,3,4}.
- q_max_intra = 5: Researcher inference. The paper constrains only AR order, not MA. We use q_max = 5 to match the inter-day search range.
- N_regime_window = 63: Researcher inference. The paper does not disclose the lookback for percentile computation.
- regime_candidates = {3, 4, 5}: Researcher inference. The paper says "training several weight models" (p.18) without specifying how many. Grid search finds the best configuration on held-out data.
- max_deviation = 0.10: Satish et al. 2014, p.24 ("depart no more than 10% away"), referencing Humphery-Jenner (2011). Important caveat: the paper states "we developed a separate method for computing the deviation bounds" (p.19), meaning the actual production system used adaptive/proprietary bounds, not this fixed value. The 10% is a conservative starting point.
- pct_switchoff = 0.80: Satish et al. 2014, p.24 ("once 80% of the day's volume is reached, return to a historical approach"), referencing Humphery-Jenner (2011).
- reestimation_interval = 21: Researcher inference. Monthly full re-estimation balances computational cost with adaptation speed.

### Initialization

1. **Data requirement:** Minimum 126 trading days (N_seasonal) of historical
   volume data per stock before the model can be trained. If fewer than 126 days
   are available, use all available data (minimum 63 days absolute floor).

2. **Seasonal factors:** Computed first from the 126-day window. If any bin has
   zero average volume across the full window, replace with the minimum non-zero
   value across all bins.

3. **Historical average:** Computed from the shorter N_hist (21-day) window. Same
   zero-floor guard as seasonal factors.

4. **ARMA models:** Fitted after seasonal factors (which are needed for
   deseasonalization of the intraday series). Models that fail to converge are
   replaced with a FALLBACK sentinel. The forecast functions check for FALLBACK
   and substitute the historical average (for inter-day: D = H) or seasonal
   factor (for intraday: A = seasonal_factors[i]) as the component value.

5. **Regime classifier:** Built from cumulative volume distributions. If the
   grid search yields a best configuration where any regime bucket has fewer
   than min_samples_per_regime observations, that regime gets equal weights
   [1/3, 1/3, 1/3].

6. **Percentage model:** Trained last. As of Draft 3, it depends only on
   volume_history (not on Model A's hist_avg), since surprises are computed in
   percentage space directly from volume ratios.

7. **Initial weights:** Optimizer uses 4 starting points (equal weights plus
   3 component-dominant initializations). Best result across restarts is used.
   Falls back to equal weights if optimization does not improve over them.

### Calibration

The system does not have a separate calibration phase in the traditional sense.
All parameters are either set to literature-recommended values or selected via
in-sample optimization:

1. **ARMA order selection:** Automatic via AICc grid search over (p, q) combinations.
2. **Regime configuration:** Grid search over {3, 4, 5} regime counts with
   validation MAPE as selection criterion.
3. **Component weights:** Per-regime optimization minimizing MAPE with multi-restart
   Nelder-Mead, with convergence fallback to equal weights.
4. **Surprise lag count:** Blocked time-series cross-validation over L in {1, ..., L_max}.
   Remainder days in the fold partition are assigned to the last fold.
5. **User-tunable parameters:** N_hist, max_deviation, and pct_switchoff are the
   primary candidates for manual tuning. N_hist has the highest sensitivity among
   the user-set parameters.

---

## Validation

### Expected Behavior

**Model A (Raw Volume):**
- Median MAPE reduction of ~24% relative to the historical rolling average
  baseline across all 26 intraday bins. (Satish et al. 2014, p.20, Exhibit 6.)
- Bottom-95% mean MAPE reduction of ~29%. (Satish et al. 2014, p.20.)
- Error reduction increases through the day: approximately 10% at 9:30 to 33% at
  15:30. (Satish et al. 2014, Exhibit 6, p.22.) This is expected because the
  intraday ARMA accumulates more conditioning information as the day progresses.
- Improvements should be consistent across SIC industry groups (~15-35%
  reduction) and beta deciles (~20-35%). (Satish et al. 2014, Exhibits 7-8,
  pp.22-23.)

**Model B (Volume Percentage):**
- Median absolute error: 0.00808 (dynamic) vs 0.00874 (historical) — a 7.55%
  reduction. (Satish et al. 2014, Exhibit 9, p.23.)
- Bottom-95% average absolute error: 0.00924 vs 0.00986 — a 6.29% reduction.
  (Satish et al. 2014, Exhibit 9, p.23.)
- Statistical significance: Wilcoxon signed-rank test at << 1% level. (Satish
  et al. 2014, Exhibit 9 footnote.)
- Note: the percentage error metric is MAD (mean absolute deviation), NOT MAPE.
  Volume percentages are already normalized (sum to ~1.0), so no division by
  actual is needed. (Satish et al. 2014, p.17, "Measuring Percentage Volume
  Predictions — Absolute Deviation".)

**VWAP Tracking Error:**
- Mean tracking error: 8.74 bps (dynamic) vs 9.62 bps (historical) — 9.1%
  reduction. Std dev: 10.08 bps vs 11.18 bps. (Satish et al. 2014, Exhibit 10,
  p.23.)
- Paired t-test statistic: 2.34, p < 0.01. (Satish et al. 2014, Exhibit 10.)
- 7-10% improvement across Dow 30, midcap, and high-variance stock groups.
  (Satish et al. 2014, p.23.)
- Regression of VWAP tracking error on volume percentage error: R^2 = 0.5146,
  coefficient = 220.9 bps/unit error (Dow 30); R^2 = 0.5886, coefficient = 454.3
  bps/unit error (high-variance names). Both regressions through origin.
  (Satish et al. 2014, Exhibits 3 and 5, pp.20-21.)
- **Important (m7 clarification):** The absolute bps values (8.74, 9.62) are
  specific to the FlexTRADER simulation conditions: 10% of 30-day ADV order size,
  day-long orders, May 2011 data (Satish et al. 2014, p.19). A developer should
  NOT expect to match these absolute numbers. The meaningful benchmark is the
  relative improvement (~9% reduction) and the statistical significance of the
  paired test. Absolute VWAP tracking error depends on order size, stock
  liquidity, market conditions, and execution simulator details.

### Sanity Checks

1. **Seasonal factor U-shape:** seasonal_factors should exhibit the well-known
   intraday volume U-shape (high at open, low at midday, high at close). Plot
   seasonal_factors[1..26] and verify visually.

2. **Historical average vs. seasonal factor ratio:** hist_avg[i] / seasonal_factors[i]
   should be close to 1.0 for most bins (both are per-bin volume averages, just
   over different windows). Large deviations (> 2x or < 0.5x) indicate a recent
   volume regime shift.

3. **Deseasonalized series stationarity:** Run an Augmented Dickey-Fuller (ADF)
   test on the deseasonalized intraday series. Expect rejection of the unit-root
   null at 5% significance for most stocks.

4. **ARMA order parsimony:** AICc should select low-order models. Expect median
   p + q <= 4 across stocks and bins. If most models select p=5, q=5, the
   search range may be too narrow or the data is non-stationary.

5. **Weight non-negativity:** All optimized weights should be non-negative (by
   construction via exp-transformation). Verify exp(w_log) > 0 for all regimes.

6. **Regime bucket population:** Each regime bucket should contain a reasonable
   number of observations (>= min_samples_per_regime). If any bucket has < 10
   samples, log a warning.

7. **Component 1 baseline MAPE:** forecast_raw_volume with weights = [1, 0, 0]
   (historical average only) should reproduce the baseline MAPE. This verifies
   the pipeline is correct.

8. **Monotonic improvement:** Adding each component should not worsen MAPE.
   Expected ordering: historical only > + inter-day ARMA > + intraday ARMA >
   + regime weights. If a component worsens MAPE, its weight should be near zero.

9. **Surprise mean near zero:** The mean of percentage-space surprise values
   across the training set should be approximately zero (balanced over- and
   under-predictions relative to the historical curve). Researcher inference:
   std dev should be approximately 0.005-0.015 for liquid stocks (since surprise
   = actual_pct - hist_pct, and individual bin participation rates typically
   deviate by 0.5-1.5 percentage points from their historical mean). If mean
   surprise deviates significantly from zero (e.g., |mean| > 0.001), verify
   that hist_pct is computed from a representative window. A nonzero mean
   surprise would bias the no-intercept regression coefficients — see Function 7
   no-intercept validity requirement.

10. **Surprise regression coefficients:** |beta_k| < 1.0 for all k. Positive
    beta_1 expected (positive autocorrelation in percentage surprises: a bin
    that exceeded its historical participation rate suggests the next bin will
    also exceed it). Note: with percentage-space surprises, beta values are
    dimensionless ratios of percentage-point departures, and their typical
    magnitude may be larger than with raw-volume surprises because the input
    and output are in the same (smaller) scale.

11. **Deviation constraint binding rate:** The max_deviation clamp should bind
    on approximately 5-15% of bins. If it never binds, max_deviation may be too
    loose. If it binds on > 30% of bins, the surprise regression may be producing
    unreasonably large adjustments, or max_deviation may be too tight.

12. **Switch-off activation timing:** The pct_switchoff condition (80% cumulative
    volume) should activate in the last 2-4 bins of the day. If it activates
    before bin 20, V_total_est may be systematically low.

13. **Cumulative percentage monotonicity:** The cumulative sum of pct_hat values
    should approach 1.0 monotonically. Large discontinuities indicate issues
    with the scaling logic.

14. **MAPE formula consistency:** Verify that compute_evaluation_mape matches
    the paper's MAPE formula: 100% * (1/N) * SUM(|Predicted - Actual| / Actual).
    (Satish et al. 2014, p.17.)

### Edge Cases

1. **ARMA convergence failure:** 1-5% of inter-day models may fail to converge,
   especially for illiquid bins. Fall back to historical average component
   (set D = H for that bin). For intraday ARMA failure, set A = seasonal_factor
   for all bins. Log all fallbacks for monitoring.

2. **Zero-volume bins:** Can occur for illiquid stocks in midday bins. Floor
   seasonal factors and hist_avg via min-nonzero replacement (Functions 1, 1a).
   Exclude from MAPE computation. Clamp raw forecast at zero.

3. **Insufficient surprise lags (early bins):** For bins 1 through L, fewer than
   L lagged surprises are available. Set delta = 0 (no adjustment, use scaled
   historical percentage). This is functionally equivalent to zero-padding
   missing lags.

4. **Half-day trading sessions:** Days before holidays have 13 bins (9:30-13:00
   ET) instead of 26. Exclude from training data or dynamically adjust I. The
   paper does not address this; Researcher inference is to exclude.

5. **Special calendar days:** Option expiration, Fed announcements, index
   rebalancing days have atypical volume patterns. The paper recommends "custom
   curves for special calendar days... rather than ARMAX models, due to
   insufficient historical occurrences" (Satish et al. 2014, p.18 para 4).
   Maintain a calendar of special days and bypass the dynamic model.

6. **Stock splits and corporate actions:** Volume data must be split-adjusted.
   A 2:1 split doubles apparent volume. Use split-adjusted share counts as
   input, or normalize by daily shares outstanding.

7. **V_total_est approaching zero:** If V_total_est is very small or zero
   (unlikely for liquid stocks), the observed_frac computation in
   forecast_volume_percentage can produce NaN. Guard with IF V_total_est > 0.

8. **Regime boundary noise:** A stock near a percentile boundary may oscillate
   between regimes as bins accumulate. Optional enhancement: require the
   percentile to cross the boundary by a hysteresis margin (e.g., 5 percentile
   points) before switching. Not required for baseline implementation. Researcher
   inference.

9. **Last-bin percentage forecast:** When predicting the last bin (next_bin == I),
   assign all remaining volume fraction rather than applying the normal
   regression adjustment. This ensures cumulative percentages sum to 1.0.

10. **Rolling H in weight optimization (Draft 3 M2 fix):** Function 5 now
    computes H_d as a rolling N_hist-day average for each training day d. This
    requires volume_history to extend at least N_hist + N_weight_train days
    before train_end_date (21 + 63 = 84 days). If insufficient history exists
    for the earliest training days, use whatever history is available (minimum
    N_hist days is preferred; if fewer, the average is computed from the available
    days). This edge case is relevant only during the initial warm-up period.

11. **Empty regime bucket in grid search:** If a regime candidate (e.g., 5
    regimes) produces buckets with fewer than min_samples_per_regime observations,
    those buckets get equal weights [1/3, 1/3, 1/3]. This may make the
    configuration score poorly in the grid search, naturally favoring fewer
    regimes.

12. **Near-zero total volume days in percentage surprise computation:** Days
    where total_vol_d < min_volume_floor are excluded from surprise training
    (Function 7, Step 2). This prevents division-by-near-zero in actual_pct
    computation. Such days are extremely rare for the target universe.

### Known Limitations

1. **Point forecasts only:** Unlike CMEM or Kalman filter approaches, this
   model does not specify a noise distribution and cannot produce prediction
   intervals or density forecasts. (Researcher inference; cf. Chen et al. 2016.)

2. **No outlier robustness:** Unlike Chen et al.'s robust Kalman filter with
   Lasso regularization, no built-in mechanism for outlier handling. Outlier
   bins directly affect the historical average, ARMA estimates, and surprises.
   (Researcher inference.)

3. **Single-stock independence:** Each stock is modeled independently. Cross-
   sectional information (e.g., sector-wide volume surges, market-wide events)
   is not captured. (Satish et al. 2014, implicitly; cf. Bialkowski et al. 2008
   PCA-based market/stock decomposition.)

4. **Static intraday shape assumption:** The 6-month trailing average for
   seasonal factors assumes the intraday volume shape is approximately constant
   over that window. Structural changes (e.g., growth of closing auctions) are
   captured slowly. (Researcher inference.)

5. **No price information:** The model is purely volume-driven and agnostic to
   price dynamics. Volume-price correlations and price impact effects are not
   exploited. (Researcher inference.)

6. **Dataset specificity:** Reported results are on the top 500 U.S. equities
   by dollar volume over one specific year. Performance on less liquid stocks,
   other markets, or different time periods may differ. (Satish et al. 2014,
   pp.19-20.)

7. **Multi-step forecast degradation:** Recursive ARMA predictions degrade with
   forecast horizon. After p steps, the AR component uses only past forecasts;
   after q steps, MA contributes nothing. For distant bins, the intraday ARMA
   forecast converges to the unconditional mean (~1.0 in deseasonalized space),
   re-seasonalizing to the seasonal factor. The regime-switching weights should
   naturally adapt by giving more weight to H for distant bins. (Researcher
   inference; Satish et al. 2014, p.18 indirectly acknowledges: "techniques
   that predict only the next interval will perform better.")

8. **Metric incomparability:** Different papers use different MAPE normalization
   conventions. Satish et al. express MAPE as a percentage; Chen et al. (2016)
   express it as a fraction. Direct cross-paper performance comparisons are
   imprecise. (Satish et al. 2014, p.17; Chen et al. 2016, p.7.)

9. **Train/predict surprise denominator mismatch:** In Model B training
   (Function 7), percentage surprises use exact daily total volume (known
   after market close). In prediction (Function 8), they use estimated total
   volume (observed + forecasted remaining). Early-day predictions have noisier
   surprises due to V_total_est uncertainty. This is inherent to any online
   volume percentage forecasting system and is mitigated by the deviation clamp.
   (Researcher inference.)

---

## Paper References

| Spec Section | Paper Source |
|---|---|
| Overall architecture (dual model) | Satish et al. 2014, p.15-19 (full methodology) |
| Bin structure (I=26, 15-min) | Satish et al. 2014, p.16 ("26 such bins") |
| Component 1: Historical average (N_hist) | Satish et al. 2014, p.17 para 1 ("rolling historical average"), Exhibit 1 ("Prior 21 days") |
| Deseasonalization (N_seasonal) | Satish et al. 2014, p.17 para 5 ("dividing by the intraday amount... trailing six months") |
| Component 2: Inter-day ARMA + AICc | Satish et al. 2014, p.17 para 2; Hurvich & Tsai 1989, 1993 |
| Component 3: Intraday ARMA | Satish et al. 2014, p.17-18 para 5 + p.18 para 1-2 |
| Re-seasonalization | Satish et al. 2014, p.18 para 3 ("re-seasonalize these forecasts via multiplication") |
| AR lag constraint (< 5) | Satish et al. 2014, p.18 para 1 ("AR lags with a value less than five") |
| 11-term observation | Satish et al. 2014, p.18 para 2 ("fewer than 11 terms") |
| Component 4: Regime-switching weights | Satish et al. 2014, p.18 para 4 ("dynamic weight overlay") |
| Exhibit 1 data flow | Satish et al. 2014, p.18, Exhibit 1 |
| Exhibit 1 "4 bins" notation | Satish et al. 2014, p.18, Exhibit 1 (reflects p_max_intra = 4) |
| Volume percentage methodology | Satish et al. 2014, p.18-19; Humphery-Jenner 2011 |
| Surprise = pct-space departures | Satish et al. 2014, p.18-19 ("departures from a historical average approach"); Humphery-Jenner 2011 ("volume surprises — deviations from a naive historical forecast" where the forecast is a percentage/VWAP curve) |
| Deviation constraint (10%) | Satish et al. 2014, p.24; Humphery-Jenner 2011 |
| Proprietary deviation bounds | Satish et al. 2014, p.19 ("separate method") |
| Switch-off (80%) | Satish et al. 2014, p.24; Humphery-Jenner 2011 |
| No-intercept regressions | Satish et al. 2014, p.19 ("without the inclusion of a constant term") — applied by analogy to surprise regression |
| MAPE formula | Satish et al. 2014, p.17 ("Measuring Raw Volume Predictions") |
| MAD formula (pct error) | Satish et al. 2014, p.17 ("Measuring Percentage Volume Predictions") |
| Custom curves for special days | Satish et al. 2014, p.18 para 4 (footnote 1, Pragma Trading 2009) |
| Validation: 24%/29% MAPE reduction | Satish et al. 2014, p.20, Exhibit 6 |
| Validation: 7.55% MAD reduction | Satish et al. 2014, p.23, Exhibit 9 |
| Validation: 9.1% VWAP reduction | Satish et al. 2014, p.23, Exhibit 10 |
| Validation: VWAP-error regression | Satish et al. 2014, pp.20-21, Exhibits 2-5 |
| Validation: simulation conditions | Satish et al. 2014, p.19 (FlexTRADER, Dow 30, midcap, May 2011) |
| N_hist = 21 (illustrative) | Satish et al. 2014, Exhibit 1 caption ("Prior 21 days") |
| N_seasonal = 126 (6 months) | Satish et al. 2014, p.17 ("trailing six months") |
| N_intraday_fit = 21 (1 month) | Satish et al. 2014, p.18 ("most recent month") |
| p_max, q_max = 5 | Satish et al. 2014, p.17 ("p and q lags through five") |

**Researcher inference items** (not directly from any paper):
- Zero seasonal factor and hist_avg replacement (min-nonzero)
- N_interday_fit = 63 (paper ambiguous; "prior 5 days" is lag memory, not fit window)
- q_max_intra = 5 (paper constrains only AR)
- Independent-segment approach for intraday ARMA day-boundary handling
- Regime grid search methodology
- Default-to-middle-regime for pre-market
- Optimizer choice (multi-restart Nelder-Mead with exp-transformation and convergence fallback)
- Absence of sum-to-1 weight constraint
- min_volume_floor for MAPE/surprise/day exclusion
- Blocked time-series CV for lag selection (remainder days to last fold)
- Implicit renormalization via per-call scaled_base (optional explicit renormalization documented)
- Re-estimation schedule (21-day interval)
- Convergence failure handling (FALLBACK sentinel)
- Regime boundary hysteresis (optional)
- No-intercept in surprise regression (by analogy from VWAP-error regressions)
- Surprise std dev range (~0.005-0.015 for liquid stocks in percentage space)
- Rolling H_d per training day in weight optimization (based on "rolling historical average" description)
- Separate validation hist_avg to prevent validation data leakage
- Percentage-space surprise definition (consistent with Humphery-Jenner's percentage-space framework)
- Train/predict surprise denominator mismatch documentation
