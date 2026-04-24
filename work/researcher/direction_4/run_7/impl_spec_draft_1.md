# Implementation Specification: Dual-Mode Intraday Volume Forecast (Raw Volume + Volume Percentage)

## Overview

This specification describes a dual-model system for intraday volume forecasting
based on Satish, Saxena, and Palmer (2014). Model A forecasts raw bin-level
trading volume by combining three components — a rolling historical average, an
inter-day ARMA, and an intraday ARMA — through a regime-switching dynamic weight
overlay. Model B forecasts next-bin volume percentages by applying a
surprise-regression adjustment (extending Humphery-Jenner 2011's "dynamic VWAP"
framework) on top of a historical volume percentage curve. The two models are
coupled: Model A's raw forecasts provide the surprise signal consumed by Model B.

The system operates on 15-minute bins (I = 26 per 6.5-hour U.S. equity trading
day, 9:30-16:00 ET) and is fitted per individual stock. Each stock's model is
independent — no cross-sectional information is used.

---

## Algorithm

### Model Description

**Model A (Raw Volume Forecast)** takes as input a stock symbol, a date, a target
bin index, and the volumes observed so far today. It produces a point forecast of
the number of shares that will trade in the target bin. Three signal components
are combined with weights that depend on the stock's current volume regime
(determined by the historical percentile of cumulative volume observed up to the
current bin).

**Model B (Volume Percentage Forecast)** takes the same inputs plus Model A's
output and produces a forecast of what fraction of the day's total volume will
trade in the next bin. It adjusts a static historical percentage curve using
recent volume surprises (deviations of actual volume from Model A's forecast),
subject to deviation constraints and a late-day switch-off rule.

**Inputs:**
- Historical daily volume time series per bin: volume[stock, day, bin] — split-adjusted share counts.
- Current day's observed volumes through the current bin.

**Outputs:**
- Model A: V_hat[stock, day, bin] — forecasted raw volume (shares) for each target bin.
- Model B: pct_hat[stock, day, next_bin] — forecasted volume fraction for the next bin.

### Pseudocode

The system is organized into 10 functions: 6 for Model A training and prediction,
2 for Model B training and prediction, and 2 for orchestration.

#### Function 1: compute_seasonal_factors

Computes the per-bin rolling historical average used for deseasonalization and as
Model A Component 1.

```
FUNCTION compute_seasonal_factors(volume_history, N_seasonal) -> seasonal_factors[1..I]
    # Input:  volume_history[day, bin] for trailing N_seasonal days
    # Output: seasonal_factors array of length I

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

**References:** Satish et al. 2014, p.17 para 3 ("a rolling historical average for the
volume trading in a given 15-minute bin"), p.17 para 5 ("dividing by the intraday
amount of volume traded in that bin over the trailing six months"). The zero-floor
replacement is Researcher inference — the paper does not address zero-volume bins.

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

#### Function 3: fit_intraday_arma

Fits a single ARMA model per stock on deseasonalized intraday volume from the
most recent N_intraday_fit trading days.

```
FUNCTION fit_intraday_arma(volume_history, seasonal_factors, N_intraday_fit,
                           p_max_intra, q_max_intra) -> intraday_model
    # Input:  volume_history[day, bin], seasonal_factors[1..I], window size
    # Output: fitted ARMA model object (or FALLBACK sentinel)

    # Build deseasonalized series with day-boundary tracking
    deseas_series = []
    day_boundaries = []     # indices where new days start
    idx = 0
    FOR d IN trailing N_intraday_fit days:
        day_boundaries.append(idx)
        FOR i = 1 TO I:
            deseas_series.append(volume_history[d, i] / seasonal_factors[i])
            idx += 1

    n = len(deseas_series)
    best_aicc = +infinity
    best_model = None

    FOR p = 0 TO p_max_intra:
        FOR q = 0 TO q_max_intra:
            k = p + q + 1
            # Effective sample size accounts for day-boundary resets
            n_eff = N_intraday_fit * (I - max(p, q))
            IF n_eff <= k + 1:
                CONTINUE

            model = fit_ARMA(deseas_series, order=(p, q), include_constant=True,
                             method="MLE", enforce_stationarity=True,
                             enforce_invertibility=True,
                             day_breaks=day_boundaries)

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

**References:** Satish et al. 2014, p.17-18 ("Next, we fit an additional ARMA (p, q)
model over deseasonalized intraday bin volume data. The intraday data are
deseasonalized by dividing by the intraday amount of volume traded in that bin
over the trailing six months... we used AR lags with a value less than five...
We compute this model on a rolling basis over the most recent month."). The
day_breaks mechanism is Researcher inference — the paper says "rolling basis over
the most recent month" but does not specify overnight-boundary handling. Treating
each day as an independent segment prevents spurious lag-1 connections across the
overnight gap. The AR constraint "less than five" means p_max_intra = 4 (Satish
et al. 2014, p.18). The MA order range is unconstrained by the paper; we use
q_max_intra = 5 to match the inter-day search range. Researcher inference for
q_max_intra.

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
FUNCTION assign_regime(regime_classifier, current_bin, cumulative_volume) -> regime_index
    # Returns integer in [0, n_regimes - 1]

    IF current_bin < 1:
        # No volume observed yet; default to middle regime
        RETURN regime_classifier.n_regimes // 2

    thresholds = regime_classifier.thresholds[current_bin]
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
The default-to-middle-regime for current_bin < 1 is Researcher inference — the
paper does not address pre-market regime assignment.

#### Function 5: optimize_regime_weights

Finds optimal combination weights for each regime by minimizing MAPE on
in-sample data.

```
FUNCTION optimize_regime_weights(volume_history, seasonal_factors,
                                  interday_models, intraday_model,
                                  regime_classifier, N_weight_train,
                                  min_volume_floor)
        -> weights[0..n_regimes-1][3]
    # Input:  trained components, regime classifier, training window
    # Output: per-regime weight vectors [w_H, w_D, w_A]

    n_regimes = regime_classifier.n_regimes

    # Collect (H, D, A, actual, regime) tuples from training period
    samples_by_regime = {r: [] FOR r = 0 TO n_regimes - 1}

    FOR d IN trailing N_weight_train days:
        cumvol = 0
        FOR i = 1 TO I:
            actual = volume_history[d, i]
            IF actual < min_volume_floor:
                cumvol += actual
                CONTINUE

            # Component 1: Historical rolling average (H)
            H = seasonal_factors[i]

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

            regime = assign_regime(regime_classifier, i, cumvol)
            samples_by_regime[regime].append((H, D, A, actual))
            cumvol += actual

    # Optimize weights per regime
    weights = {}
    FOR r = 0 TO n_regimes - 1:
        samples = samples_by_regime[r]
        IF len(samples) < min_samples_per_regime:
            weights[r] = [1/3, 1/3, 1/3]   # insufficient data; equal weights
            CONTINUE

        # Minimize MAPE using exp-transformation for guaranteed non-negativity
        # w_actual = exp(w_log), so any real-valued w_log produces positive weights
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

        w_log_init = [log(1/3), log(1/3), log(1/3)]
        result = scipy.optimize.minimize(mape_loss, w_log_init,
                                         method="Nelder-Mead",
                                         options={maxiter: 1000,
                                                  xatol: 1e-4, fatol: 1e-6})
        weights[r] = exp(result.x)   # convert back from log-space

    RETURN weights
```

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

    # Component 1: Historical rolling average
    H = model_a.seasonal_factors[target_bin]

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

    # Regime classification
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
surprise on lagged surprises.

```
FUNCTION train_percentage_model(model_a, volume_history, N_regression_fit,
                                 L_max, min_volume_floor)
        -> percentage_model
    # Input:  trained Model A, volume history, regression window, max lags
    # Output: fitted percentage model (beta coefficients, optimal L, hist_pct curve)

    # Step 1: Compute historical percentage curve (static baseline)
    hist_pct = ARRAY[1..I]
    FOR i = 1 TO I:
        total_per_day = [sum(volume_history[d, j] FOR j = 1 TO I)
                         FOR d IN trailing N_regression_fit days]
        hist_pct[i] = mean(volume_history[d, i] / total_per_day[d]
                           FOR d IN trailing N_regression_fit days)

    # Step 2: Compute surprises for training data
    # Use STATIC (unconditional, pre-market) raw forecasts for consistency
    # with prediction-time surprise computation.
    surprise_data = []   # list of (day, bin, surprise_value)
    FOR d IN trailing N_regression_fit days:
        FOR i = 1 TO I:
            raw_fcst = forecast_raw_volume(model_a, stock, d,
                                           target_bin=i, observed_volumes={})
            actual = volume_history[d, i]
            IF raw_fcst <= min_volume_floor:
                surprise = 0.0
            ELSE:
                surprise = (actual - raw_fcst) / raw_fcst
            surprise_data.append((d, i, surprise))

    # Step 3: Select optimal number of lags via blocked time-series CV
    best_L = 1
    best_cv_error = +infinity

    FOR L_candidate = 1 TO L_max:
        # Blocked K-fold CV: partition days into K contiguous blocks
        K = 5
        days = [distinct days in trailing N_regression_fit]
        block_size = len(days) // K
        cv_errors = []

        FOR fold = 0 TO K - 1:
            test_days = days[fold * block_size : (fold + 1) * block_size]
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
model terms for U.S. stocks in our in-sample data"). The no-intercept choice:
p.19 ("we perform both regressions without the inclusion of a constant term") —
this quote explicitly refers to the VWAP-error validation regressions, but we
apply the same principle by analogy: when all surprises are zero, the adjustment
should be zero. Researcher inference on the analogy. The use of static
(unconditional) raw forecasts during training is Researcher inference — it ensures
consistency with prediction time (where surprises also use unconditional
forecasts as baseline). The blocked time-series CV methodology is Researcher
inference — the paper says they "identified the optimal number of model terms"
(p.19) but does not disclose the selection procedure.

#### Function 8: forecast_volume_percentage

Produces a forecast of the volume fraction for the next bin using the dynamic
VWAP adjustment.

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

    # Step 2: Compute surprise vector from unconditional (pre-market) raw forecasts
    # Use unconditional forecasts as surprise baseline — consistent with training
    surprises = []
    FOR j = 1 TO current_bin:
        raw_fcst_uncond = forecast_raw_volume(
            model_a, stock, day, target_bin=j, observed_volumes={})
        actual = observed_volumes[j]
        IF raw_fcst_uncond <= min_volume_floor:
            surprises.append(0.0)
        ELSE:
            surprises.append((actual - raw_fcst_uncond) / raw_fcst_uncond)

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

**References:** Satish et al. 2014, p.18-19 (Volume Percentage Forecast Methodology),
p.24 ("depart no more than 10% away from a historical VWAP curve", referencing
Humphery-Jenner 2011; "once 80% of the day's volume is reached, return to a
historical approach"). The paper notes: "we developed a separate method for
computing the deviation bounds" (p.19), indicating their production system used
proprietary adaptive bounds, not the fixed 10%. The scaled_base renormalization
approach is Researcher inference — the paper does not describe an explicit
renormalization step. The implicit approach via per-call recomputation of
scaled_base produces self-consistent percentage sequences without inter-call
state propagation. The dual raw-forecast calls (conditioned for V_total_est,
unconditional for surprises) are Researcher inference — the paper does not
distinguish these two uses of raw forecasts.

#### Function 9: train_full_model

Top-level training function that fits all Model A and Model B components.

```
FUNCTION train_full_model(volume_history, stock, train_end_date, params)
        -> (model_a, pct_model)
    # Step 1: Seasonal factors (Component 1 / deseasonalization basis)
    seasonal_factors = compute_seasonal_factors(
        volume_history[:train_end_date], params.N_seasonal)

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

        weights = optimize_regime_weights(
            volume_history[train_days], seasonal_factors,
            interday_models, intraday_model, regime_classifier,
            len(train_days), params.min_volume_floor)

        # Evaluate on validation period
        temp_model_a = ModelA(seasonal_factors, interday_models,
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

    model_a = ModelA(seasonal_factors, interday_models, intraday_model,
                     regime_classifier, weights)

    # Step 5: Percentage model (Model B)
    pct_model = train_percentage_model(
        model_a, volume_history[:train_end_date],
        params.N_regression_fit, params.L_max, params.min_volume_floor)

    RETURN (model_a, pct_model)
```

```
FUNCTION compute_evaluation_mape(model_a, volume_history, eval_days,
                                  min_volume_floor) -> mape
    # Compute MAPE over evaluation period
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

**References:** The training pipeline order (seasonal -> inter-day -> intraday ->
regime -> weights -> percentage) follows the dependency structure implied by
Satish et al. 2014, p.17-19. The regime grid search over {3, 4, 5} and
validation split are Researcher inference — the paper treats regime thresholds
as proprietary. The 11-term soft constraint: Satish et al. 2014, p.18 ("As a
result, we fit each symbol with a dual ARMA model having fewer than 11 terms").
The "As a result" phrasing indicates this is an observed empirical outcome, not
a prescriptive constraint. Researcher inference on treating it as a soft guardrail.

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
    # Inter-day ARMA: append today's observation (no re-estimation)
    FOR i = 1 TO I:
        IF model_a.interday_models[i] IS NOT FALLBACK:
            model_a.interday_models[i].append_observation(observed_volumes[i])

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
Forecast." The distinction between daily state updates (append_observation) and
periodic full re-estimation is Researcher inference — the paper says "We compute
this model on a rolling basis" (p.18) without specifying the update frequency.
The re-estimation schedule is Researcher inference.

### Data Flow

```
INPUT: volume_history[stock, day, bin] — split-adjusted share counts
       Shape: (n_stocks, n_days, I=26)
       Type: float64 (share counts can be fractional after split adjustment)

                           TRAINING PHASE
                           ==============

volume_history ──► compute_seasonal_factors() ──► seasonal_factors[26]
                         │                              │
                         ▼                              ▼
                   fit_interday_arma() ──► interday_models[26] (ARMA objects)
                         │                              │
                         ▼                              ▼
                   fit_intraday_arma() ──► intraday_model (single ARMA object)
                         │
                         ▼
                   build_regime_classifier() ──► regime_classifier
                         │                              │
                         ▼                              ▼
                   optimize_regime_weights() ──► weights[n_regimes][3]
                         │
                         ╰──────────── ModelA ──────────╮
                                                        │
                   train_percentage_model() ◄───────────╯
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
                                   │
                                   ╰──► pct_hat (volume fraction for next bin)
```

**Intermediate types and shapes:**

| Variable | Shape | Type | Description |
|----------|-------|------|-------------|
| seasonal_factors | (26,) | float64 | Per-bin 6-month average volume |
| interday_models | (26,) | ARMA or FALLBACK | Per-bin fitted ARMA models |
| intraday_model | scalar | ARMA or FALLBACK | Single deseasonalized ARMA |
| regime_classifier | object | RegimeClassifier | Thresholds per bin |
| weights | (n_regimes, 3) | float64 | Per-regime [w_H, w_D, w_A] |
| beta | (L,) | float64 | Surprise regression coefficients |
| hist_pct | (26,) | float64 | Historical volume percentage curve |
| V_hat | scalar | float64 | Raw volume forecast (shares) |
| pct_hat | scalar | float64 | Volume fraction forecast [0, 1] |

### Variants

We implement the full dual-model system as described in Satish et al. (2014).
This is the most complete variant presented in the paper, incorporating all four
components of Model A (historical average + inter-day ARMA + intraday ARMA +
regime-switching weights) and the dynamic VWAP extension for Model B.

Simpler variants exist:
- **Historical average only**: Component 1 alone. This is the baseline the paper
  benchmarks against.
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
| N_seasonal | Trailing window for seasonal factors (trading days) | 126 | Low | [63, 252] |
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
| min_volume_floor | Minimum volume for MAPE/surprise computation (shares) | 100 | Low | [50, 500] |
| N_regression_fit | Training window for surprise regression (trading days) | 63 | Medium | [21, 126] |
| L_max | Maximum number of lagged surprise terms to consider | 5 | Medium | [3, 8] |
| max_deviation | Maximum relative departure from historical pct curve | 0.10 | Medium-high | [0.05, 0.20] |
| pct_switchoff | Cumulative volume fraction triggering switch to historical curve | 0.80 | Medium | [0.70, 0.90] |
| reestimation_interval | Days between full model re-estimation | 21 | Low | [5, 63] |

**Parameter sources:**
- I = 26: Satish et al. 2014, p.16 ("based on 15-minute bins, and there are 26 such bins in a trading day").
- N_seasonal = 126: Satish et al. 2014, p.17 ("trailing six months"). 126 trading days ~ 6 calendar months.
- N_hist = 21: Satish et al. 2014, Exhibit 1 caption ("Prior 21 days"). Caveat: this appears only in the diagram and may be illustrative rather than the actual tuned value. The paper introduces N as "a variable that we shall call N" (p.16) without disclosing its value.
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

2. **Seasonal factors:** Computed first. If any bin has zero average volume across
   the full window, replace with the minimum non-zero value across all bins.

3. **ARMA models:** Fitted after seasonal factors. Models that fail to converge
   are replaced with a FALLBACK sentinel. The forecast functions check for
   FALLBACK and substitute the historical average (for inter-day) or seasonal
   factor (for intraday) as the component value.

4. **Regime classifier:** Built from cumulative volume distributions. If the
   grid search yields a best configuration where any regime bucket has fewer
   than min_samples_per_regime observations, that regime gets equal weights
   [1/3, 1/3, 1/3].

5. **Percentage model:** Trained last because it depends on Model A's forecasts
   for surprise computation.

6. **Initial weights:** Optimizer starts at log(1/3) in log-space for each weight
   component (equivalent to equal 1/3 weights in real space).

### Calibration

The system does not have a separate calibration phase in the traditional sense.
All parameters are either set to literature-recommended values or selected via
in-sample optimization:

1. **ARMA order selection:** Automatic via AICc grid search over (p, q) combinations.
2. **Regime configuration:** Grid search over {3, 4, 5} regime counts with
   validation MAPE as selection criterion.
3. **Component weights:** Per-regime optimization minimizing MAPE with Nelder-Mead.
4. **Surprise lag count:** Blocked time-series cross-validation over L in {1, ..., L_max}.
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

### Sanity Checks

1. **Seasonal factor U-shape:** seasonal_factors should exhibit the well-known
   intraday volume U-shape (high at open, low at midday, high at close). Plot
   seasonal_factors[1..26] and verify visually.

2. **Deseasonalized series stationarity:** Run an Augmented Dickey-Fuller (ADF)
   test on the deseasonalized intraday series. Expect rejection of the unit-root
   null at 5% significance for most stocks.

3. **ARMA order parsimony:** AICc should select low-order models. Expect median
   p + q <= 4 across stocks and bins. If most models select p=5, q=5, the
   search range may be too narrow or the data is non-stationary.

4. **Weight non-negativity:** All optimized weights should be non-negative (by
   construction via exp-transformation). Verify exp(w_log) > 0 for all regimes.

5. **Regime bucket population:** Each regime bucket should contain a reasonable
   number of observations (>= min_samples_per_regime). If any bucket has < 10
   samples, log a warning.

6. **Component 1 baseline MAPE:** forecast_raw_volume with weights = [1, 0, 0]
   (historical average only) should reproduce the baseline MAPE. This verifies
   the pipeline is correct.

7. **Monotonic improvement:** Adding each component should not worsen MAPE.
   Expected ordering: historical only > + inter-day ARMA > + intraday ARMA >
   + regime weights. If a component worsens MAPE, its weight should be near zero.

8. **Surprise mean near zero:** The mean of surprise values across the training
   set should be approximately zero (balanced over- and under-predictions).
   Std dev should be ~0.005-0.015 for liquid stocks.

9. **Surprise regression coefficients:** |beta_k| < 0.5 for all k. Positive
   beta_1 expected (positive autocorrelation in surprises: a bin that exceeded
   forecast suggests the next bin will also exceed).

10. **Deviation constraint binding rate:** The max_deviation clamp should bind
    on no more than 10-20% of bins. If it binds frequently, the surprise
    regression is producing unreasonably large adjustments.

11. **Switch-off activation timing:** The pct_switchoff condition (80% cumulative
    volume) should activate in the last 2-4 bins of the day. If it activates
    before bin 20, V_total_est may be systematically low.

12. **Cumulative percentage monotonicity:** The cumulative sum of pct_hat values
    should approach 1.0 monotonically. Large discontinuities indicate issues
    with the scaling logic.

13. **MAPE formula consistency:** Verify that compute_evaluation_mape matches
    the paper's MAPE formula: 100% * (1/N) * SUM(|Predicted - Actual| / Actual).
    (Satish et al. 2014, p.17.)

### Edge Cases

1. **ARMA convergence failure:** 1-5% of inter-day models may fail to converge,
   especially for illiquid bins. Fall back to historical average component
   (set D = H for that bin). For intraday ARMA failure, set A = seasonal_factor
   for all bins. Log all fallbacks for monitoring.

2. **Zero-volume bins:** Can occur for illiquid stocks in midday bins. Floor
   seasonal factors via min-nonzero replacement (Function 1). Exclude from MAPE
   computation. Clamp raw forecast at zero.

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

10. **Lookahead bias in Model B training:** Model A is trained on data through
    train_end_date, then used to compute forecasts for earlier days in the
    surprise regression training window. The bias is negligible because ARMA
    parameters on 63+ day windows change minimally when dropping one day.
    To eliminate: use expanding-window re-estimation (multiplies cost by
    N_regression_fit). Researcher inference.

11. **Empty regime bucket in grid search:** If a regime candidate (e.g., 5
    regimes) produces buckets with fewer than min_samples_per_regime observations,
    those buckets get equal weights [1/3, 1/3, 1/3]. This may make the
    configuration score poorly in the grid search, naturally favoring fewer
    regimes.

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

---

## Paper References

| Spec Section | Paper Source |
|---|---|
| Overall architecture (dual model) | Satish et al. 2014, p.15-19 (full methodology) |
| Bin structure (I=26, 15-min) | Satish et al. 2014, p.16 ("26 such bins") |
| Component 1: Historical average | Satish et al. 2014, p.17 para 1 ("rolling historical average") |
| Component 2: Inter-day ARMA + AICc | Satish et al. 2014, p.17 para 2; Hurvich & Tsai 1989, 1993 |
| Component 3: Intraday ARMA | Satish et al. 2014, p.17-18 para 5 + p.18 para 1-2 |
| Deseasonalization | Satish et al. 2014, p.17 para 5 ("dividing by the intraday amount") |
| Re-seasonalization | Satish et al. 2014, p.18 para 3 ("re-seasonalize these forecasts via multiplication") |
| AR lag constraint (< 5) | Satish et al. 2014, p.18 para 1 ("AR lags with a value less than five") |
| 11-term observation | Satish et al. 2014, p.18 para 2 ("fewer than 11 terms") |
| Component 4: Regime-switching weights | Satish et al. 2014, p.18 para 4 ("dynamic weight overlay") |
| Exhibit 1 data flow | Satish et al. 2014, p.18, Exhibit 1 |
| Volume percentage methodology | Satish et al. 2014, p.18-19; Humphery-Jenner 2011 |
| Deviation constraint (10%) | Satish et al. 2014, p.24; Humphery-Jenner 2011 |
| Proprietary deviation bounds | Satish et al. 2014, p.19 ("separate method") |
| Switch-off (80%) | Satish et al. 2014, p.24; Humphery-Jenner 2011 |
| No-intercept regressions | Satish et al. 2014, p.19 ("without the inclusion of a constant term") |
| MAPE formula | Satish et al. 2014, p.17 ("Measuring Raw Volume Predictions") |
| MAD formula (pct error) | Satish et al. 2014, p.17 ("Measuring Percentage Volume Predictions") |
| Custom curves for special days | Satish et al. 2014, p.18 para 4 (footnote 1, Pragma Trading 2009) |
| Validation: 24%/29% MAPE reduction | Satish et al. 2014, p.20, Exhibit 6 |
| Validation: 7.55% MAD reduction | Satish et al. 2014, p.23, Exhibit 9 |
| Validation: 9.1% VWAP reduction | Satish et al. 2014, p.23, Exhibit 10 |
| Validation: VWAP-error regression | Satish et al. 2014, pp.20-21, Exhibits 2-5 |
| N_hist = 21 (illustrative) | Satish et al. 2014, Exhibit 1 caption ("Prior 21 days") |
| N_seasonal = 126 (6 months) | Satish et al. 2014, p.17 ("trailing six months") |
| N_intraday_fit = 21 (1 month) | Satish et al. 2014, p.18 ("most recent month") |
| p_max, q_max = 5 | Satish et al. 2014, p.17 ("p and q lags through five") |

**Researcher inference items** (not directly from any paper):
- Zero seasonal factor replacement (min-nonzero)
- N_interday_fit = 63 (paper ambiguous; "prior 5 days" is lag memory, not fit window)
- q_max_intra = 5 (paper constrains only AR)
- Day-boundary handling for intraday ARMA
- Regime grid search methodology
- Default-to-middle-regime for pre-market
- Optimizer choice (Nelder-Mead with exp-transformation)
- Absence of sum-to-1 weight constraint
- min_volume_floor for MAPE/surprise
- Static (unconditional) surprise baseline for train/predict consistency
- Blocked time-series CV for lag selection
- Implicit renormalization via per-call scaled_base
- Re-estimation schedule (21-day interval)
- Convergence failure handling (FALLBACK sentinel)
- Regime boundary hysteresis (optional)
