# Implementation Specification: Dual-Mode Volume Forecast (Raw + Percentage)

## Overview

This direction implements a two-model intraday volume forecasting system from Satish, Saxena, and Palmer (2014). The system comprises:

1. **Model A (Raw Volume Forecast):** Predicts the absolute number of shares traded in each 15-minute bin of a trading day by combining three forecast components -- a rolling historical average, an inter-day ARMA model, and a deseasonalized intraday ARMA model -- via regime-dependent optimized weights. The regime is determined dynamically based on the historical percentile of cumulative volume observed so far in the day.

2. **Model B (Volume Percentage Forecast):** Predicts the fraction of daily volume that will trade in the next bin, extending the dynamic VWAP framework of Humphery-Jenner (2011). It adjusts a naive historical percentage curve using volume surprises (deviations of actual from forecast volume) computed from Model A's output, via a rolling OLS regression without intercept.

Model A forecasts all remaining bins simultaneously (for portfolio scheduling tools), while Model B forecasts one bin ahead (for step-by-step VWAP execution algorithms). Model A's output feeds Model B's surprise signal, making them a tightly coupled pair.

Chen, Feng, and Palomar (2016) provide a benchmark comparison target (Kalman filter approach with 0.46 average MAPE, 6.38 bps VWAP tracking error) but are not implemented here.

## Algorithm

### Model Description

**Problem 1 -- Raw volume forecasting:** Given the history of bin-level volumes for a stock, predict the raw volume (share count) that will trade in each of the I=26 fifteen-minute bins (9:30-16:00 ET) of the current trading day, updating predictions intraday as new bins are observed. The model exploits three information sources:
- (a) The typical volume for that bin over recent history (historical average).
- (b) Day-to-day serial correlation in volume at the same bin index (inter-day ARMA).
- (c) Within-day momentum from recently observed bins today (intraday ARMA on deseasonalized data).

These three forecasts are combined via weights optimized per regime, where regimes are defined by the day's cumulative volume percentile.

**Problem 2 -- Volume percentage forecasting:** Given a naive historical estimate of the percentage of daily volume in each bin, adjust the participation rate for the next bin based on observed volume surprises earlier in the day. This produces a dynamic VWAP execution curve that adapts to unusual volume patterns.

**Assumptions:**
- Volume exhibits serial correlation both across days (same bin) and within days (adjacent bins). [Satish et al. 2014, p.17]
- The intraday seasonal pattern (U-shape) is stable over a trailing six-month window. [Satish et al. 2014, p.17, "dividing by the intraday amount of volume traded in that bin over the trailing six months"]
- Regime switching based on cumulative volume percentile captures high-volume vs. low-volume day dynamics. [Satish et al. 2014, p.18, "regime switching... training several weight models for different historical volume percentile cutoffs"]
- Volume percentage errors are linearly related to VWAP tracking error (R^2 > 0.50). [Satish et al. 2014, Exhibits 2-5, pp.20-21]

**Inputs:** Historical intraday volume time series at 15-minute resolution, per stock. Split-adjusted share counts.
**Outputs:** (1) Predicted raw volume for each of the 26 bins; (2) Predicted volume percentage for the next bin (for VWAP execution).

### Pseudocode

#### Part A: Raw Volume Forecast Model

```
FUNCTION train_raw_volume_model(stock, train_end_date, historical_data):
    """
    Fit all components of the raw volume model up to train_end_date.
    Called once at initialization and periodically for re-estimation.
    Returns a trained model object containing all fitted parameters.
    """

    # === COMPONENT 1: Seasonal Factors ===
    # Reference: Satish et al. 2014, p.17, "dividing by the intraday amount
    #   of volume traded in that bin over the trailing six months"
    # These are used for deseasonalization in Component 3.

    FOR each bin i in 1..I:  # I = 26 for 15-minute bins
        seasonal_factor[i] = arithmetic_mean(
            volume[stock, bin=i, day]
            for day in (train_end_date - N_seasonal + 1)..train_end_date
        )
        # Handle zero seasonal factors: replace with minimum non-zero
        # seasonal factor across all bins for this stock.
        # [Researcher inference: paper does not address zero-volume bins
        #  explicitly. Using min non-zero is data-adaptive and avoids
        #  division-by-zero in deseasonalization.]

    min_nonzero = min(sf for sf in seasonal_factor if sf > 0)
    FOR each bin i in 1..I:
        IF seasonal_factor[i] == 0:
            seasonal_factor[i] = min_nonzero

    # === COMPONENT 2: Historical Window Average (H) ===
    # Reference: Satish et al. 2014, p.17, "a rolling historical average for
    #   the volume of the trading in a 15-minute bin" and Exhibit 1 caption
    #   "Prior 21 days"

    FOR each bin i in 1..I:
        H[i] = arithmetic_mean(
            volume[stock, bin=i, day]
            for day in (train_end_date - N_hist + 1)..train_end_date
        )

    # === COMPONENT 3: Inter-day ARMA (D) ===
    # Reference: Satish et al. 2014, p.17, "a per-symbol, per-bin ARMA(p, q)
    #   model reflecting the serial correlation observable across daily volumes"
    # One ARMA model per bin, fitted to the daily time series of volume at
    # that bin index. Model selection via AICc.

    FOR each bin i in 1..I:
        daily_series_i = [volume[stock, bin=i, day]
                          for day in (train_end_date - N_interday_fit + 1)..train_end_date]

        best_aic = +infinity
        best_model = None
        FOR p in 0..p_max_interday:  # p_max_interday = 5
            FOR q in 0..q_max_interday:  # q_max_interday = 5
                IF p == 0 AND q == 0: CONTINUE  # constant-only model is baseline
                k = p + q + 1  # +1 for constant term
                IF k > max_interday_budget: CONTINUE  # see joint constraint below
                IF len(daily_series_i) <= k + 1: CONTINUE  # insufficient data

                TRY:
                    model = fit_ARMA(daily_series_i, order=(p, q),
                                     include_constant=True,
                                     enforce_stationarity=True,
                                     enforce_invertibility=True)
                    n = len(daily_series_i)
                    aic = -2 * model.log_likelihood + 2 * k
                    aicc = aic + 2 * k * (k + 1) / (n - k - 1)
                    IF aicc < best_aic:
                        best_aic = aicc
                        best_model = model
                        interday_terms[i] = k
                CATCH ConvergenceError:
                    CONTINUE  # skip non-convergent models

        interday_model[i] = best_model
        IF best_model is None:
            interday_model[i] = FALLBACK_CONSTANT  # use H[i] as fallback
            interday_terms[i] = 0

    # === COMPONENT 4: Intraday ARMA (A) ===
    # Reference: Satish et al. 2014, p.17-18, "an additional ARMA(p, q)
    #   model over deseasonalized intraday bin volume data... computed on a
    #   rolling basis over the most recent month"
    #
    # Joint term constraint: "a dual ARMA model having fewer than 11 terms"
    # (Satish et al. 2014, p.18). This is interpreted as:
    #   max_i(interday_terms[i]) + intraday_terms < 11
    # [Researcher inference: the paper states this as an observed outcome
    #  but treating it as a constraint provides a safety guardrail. AICc
    #  naturally selects low orders, so the constraint rarely binds.]

    max_interday_k = max(interday_terms[i] for i in 1..I)
    intraday_budget = 11 - max_interday_k - 1  # subtract 1 for safety margin

    # Build deseasonalized intraday series from recent N_intraday_fit days
    # Each day contributes I observations; day boundaries are respected.
    deseasonalized_series = []
    day_boundary_indices = []
    FOR each day d in (train_end_date - N_intraday_fit + 1)..train_end_date:
        day_boundary_indices.append(len(deseasonalized_series))
        FOR each bin i in 1..I:
            deseasonalized_series.append(
                volume[stock, bin=i, day=d] / seasonal_factor[i]
            )

    best_aic = +infinity
    best_model = None
    FOR p in 0..p_max_intraday:  # p_max_intraday = 4
        FOR q in 0..q_max_intraday:  # q_max_intraday = 5
            IF p == 0 AND q == 0: CONTINUE
            k = p + q + 1
            IF k > intraday_budget: CONTINUE

            TRY:
                # Fit with day-boundary awareness: treat each day's bins
                # as an independent segment. In practice, concatenate days
                # but insert likelihood breaks at day boundaries so that
                # lag-1 of day d+1 does NOT connect to last bin of day d.
                # [Researcher inference: the paper says "intraday" suggesting
                #  within-day data only. Concatenation with breaks is the
                #  practical implementation approach.]
                model = fit_ARMA(deseasonalized_series, order=(p, q),
                                 include_constant=True,
                                 enforce_stationarity=True,
                                 day_breaks=day_boundary_indices)
                n_eff = len(deseasonalized_series) - len(day_boundary_indices) * p
                aic = -2 * model.log_likelihood + 2 * k
                aicc = aic + 2 * k * (k + 1) / (n_eff - k - 1)
                IF aicc < best_aic:
                    best_aic = aicc
                    best_model = model
            CATCH ConvergenceError:
                CONTINUE

    intraday_model = best_model
    IF best_model is None:
        intraday_model = FALLBACK_CONSTANT  # fallback: unconditional mean

    # === COMPONENT 5: Regime Classification & Weight Optimization ===
    # Reference: Satish et al. 2014, p.18, "regime switching... training
    #   several weight models for different historical volume percentile
    #   cutoffs... dynamically apply the appropriate weights intraday based
    #   on the historical percentile of the observed cumulative volume"

    # Step 5a: Build cumulative volume percentile distribution
    # For each (day, bin), compute cumulative volume up to that bin,
    # then determine its percentile rank in the historical distribution.
    cumvol_distribution = {}  # key: bin index, value: sorted array of cumvols
    FOR each bin j in 1..I:
        cumvols = []
        FOR each day d in (train_end_date - N_regime_window + 1)..train_end_date:
            cumvol = sum(volume[stock, bin=k, day=d] for k in 1..j)
            cumvols.append(cumvol)
        cumvol_distribution[j] = sorted(cumvols)

    # Step 5b: Define regime boundaries
    # Grid search over regime configurations
    best_config = None
    best_oos_error = +infinity
    FOR n_reg in regime_candidates:  # e.g., {3, 4, 5}
        percentiles = equally_spaced_percentiles(n_reg)
        # e.g., n_reg=3 -> [33.3, 66.7]; n_reg=4 -> [25, 50, 75]

        # Step 5c: For each regime, optimize component weights
        # Split training data into train/validation (e.g., last 21 days = validation)
        train_days = (train_end_date - N_regime_window + 1)..(train_end_date - 21)
        val_days = (train_end_date - 20)..train_end_date

        weights = {}  # key: regime index 0..n_reg-1
        FOR r in 0..n_reg-1:
            # Collect all (day, bin) pairs assigned to regime r
            training_samples = []
            FOR each day d in train_days:
                FOR each bin j in 1..I:
                    cumvol = sum(volume[stock, bin=k, day=d] for k in 1..j)
                    pctile = percentile_rank(cumvol, cumvol_distribution[j])
                    regime = assign_regime(pctile, percentiles)
                    IF regime == r:
                        actual = volume[stock, bin=j, day=d]
                        h_val = H_for_day(stock, j, d)  # rolling mean as of day d
                        d_val = D_for_day(stock, j, d)  # inter-day ARMA forecast
                        a_val = A_for_day(stock, j, d)  # intraday ARMA forecast
                        training_samples.append((actual, h_val, d_val, a_val))

            # Optimize: minimize MAPE over training samples
            # V_hat = w1 * H + w2 * D + w3 * A
            # subject to: w1, w2, w3 >= 0 (no sum-to-1 constraint)
            # Reference: Satish et al. 2014, p.18, "a dynamic weight overlay
            #   on top of these three components... that minimizes the error"
            # [Researcher inference: MAPE is the paper's stated metric; no
            #  sum-to-1 constraint allows systematic bias correction; the paper
            #  does not specify the loss function or constraints for weight
            #  optimization explicitly.]
            def mape_loss(w):
                errors = []
                FOR (actual, h, d, a) in training_samples:
                    IF actual < min_volume_floor: CONTINUE
                    predicted = w[0]*h + w[1]*d + w[2]*a
                    errors.append(abs(predicted - actual) / actual)
                RETURN mean(errors)

            result = minimize(mape_loss, x0=[1/3, 1/3, 1/3],
                              method='Nelder-Mead',
                              bounds=[(0, None), (0, None), (0, None)])
            weights[r] = result.x

        # Evaluate on validation days
        val_error = compute_mape_on_validation(val_days, weights, ...)
        IF val_error < best_oos_error:
            best_oos_error = val_error
            best_config = (n_reg, percentiles, weights)

    RETURN ModelA(seasonal_factor, H, interday_model, intraday_model,
                  best_config, cumvol_distribution)


FUNCTION forecast_raw_volume(model_a, stock, date, current_bin, observed_volumes):
    """
    Produce raw volume forecasts for bins current_bin..I on the given date.
    observed_volumes: dict mapping bin index -> actual volume for bins 1..current_bin-1.
    Called at market open (current_bin=1) and updated intraday.

    Reference: Satish et al. 2014, Exhibit 1 (p.18) data flow diagram.
    """

    forecasts = {}

    # Component 1: Historical average (precomputed, does not change intraday)
    H = model_a.H  # array of I values

    # Component 2: Inter-day ARMA forecast
    # Use fitted model to produce one-step-ahead prediction for today.
    # This does not change intraday (it depends only on prior days).
    # Reference: Satish et al. 2014, p.17, "per-bin ARMA" and Exhibit 1
    #   "Next Bin (Prior 5 days)" -- the 5 most recent daily observations
    #   for this bin serve as the lagged inputs for prediction.
    D = {}
    FOR each bin i in current_bin..I:
        IF model_a.interday_model[i] is not FALLBACK:
            D[i] = model_a.interday_model[i].predict(steps=1)
        ELSE:
            D[i] = H[i]

    # Component 3: Intraday ARMA forecast
    # Feed observed deseasonalized volumes for today's bins 1..current_bin-1
    # into the intraday ARMA to condition its state, then forecast forward.
    # Reference: Satish et al. 2014, Exhibit 1 (p.18): "Current Bin" and
    #   "4 Bins Prior to Current Bin" feed into "ARMA Intraday".
    # [Researcher inference: "4 Bins Prior" refers to the AR lag context,
    #  not a hard limit. The model uses up to p lagged observations where
    #  p is the selected AR order (p < 5). At forecast time, we condition
    #  on all observed bins, but only the most recent p matter for the
    #  AR component.]
    #
    # Forecasts are in deseasonalized space; re-seasonalize before output.
    # Reference: Satish et al. 2014, p.18, "before passing intraday ARMA
    #   forecasts, we re-seasonalize these forecasts via multiplication"

    observed_deseas = []
    FOR each bin j in 1..current_bin-1:
        observed_deseas.append(observed_volumes[j] / model_a.seasonal_factor[j])

    A = {}
    IF model_a.intraday_model is not FALLBACK:
        # Condition the ARMA on observed deseasonalized values (not re-estimated,
        # just state update). This is reconditioning, not re-fitting.
        # Reference: Satish et al. 2014, p.18, implicit in Exhibit 1 flow.
        # [Researcher inference: the paper describes forecasting "the volume in
        #  the next 15-minute bin from the current 15-minute bin" (p.17), which
        #  implies updating the ARMA state with each new observation.]
        conditioned_state = model_a.intraday_model.condition(observed_deseas)
        steps_ahead = I - current_bin + 1
        deseas_forecasts = conditioned_state.predict(steps=steps_ahead)
        FOR idx, bin_i in enumerate(current_bin..I):
            A[bin_i] = deseas_forecasts[idx] * model_a.seasonal_factor[bin_i]
    ELSE:
        FOR each bin i in current_bin..I:
            A[i] = model_a.seasonal_factor[i]  # fallback to seasonal pattern

    # Determine regime from cumulative observed volume
    # Reference: Satish et al. 2014, p.18, "dynamically apply the appropriate
    #   weights intraday based on the historical percentile of the observed
    #   cumulative volume"
    IF current_bin == 1:
        # No volume observed yet; use default (median) regime
        # [Researcher inference: paper does not specify what to do at bin 1.
        #  Using median regime is conservative -- avoids extreme weight sets
        #  when no information is available.]
        regime = model_a.n_regimes // 2  # middle regime
    ELSE:
        cumvol = sum(observed_volumes[j] for j in 1..current_bin-1)
        pctile = percentile_rank(cumvol, model_a.cumvol_distribution[current_bin-1])
        regime = assign_regime(pctile, model_a.regime_percentiles)

    w = model_a.weights[regime]

    # Combine components
    FOR each bin i in current_bin..I:
        raw_forecast = w[0] * H[i] + w[1] * D[i] + w[2] * A[i]
        forecasts[i] = max(raw_forecast, 0)  # floor at zero

    RETURN forecasts
```

#### Part B: Volume Percentage Forecast Model

```
FUNCTION train_percentage_model(stock, train_end_date, model_a, historical_data):
    """
    Fit the volume percentage model (Model B).
    Requires a trained Model A to compute volume surprises.

    Reference: Satish et al. 2014, pp.18-19, "Volume Percentage Forecast
      Methodology"; extends Humphery-Jenner (2011) dynamic VWAP.
    """

    # Step 1: Compute historical volume percentages
    # Reference: Satish et al. 2014, p.17, historical window average applied
    #   to percentages
    FOR each bin i in 1..I:
        hist_pct[i] = arithmetic_mean(
            volume[stock, bin=i, day=d] / sum_k(volume[stock, bin=k, day=d])
            for d in (train_end_date - N_hist + 1)..train_end_date
            if sum_k(volume[stock, bin=k, day=d]) > 0  # skip zero-volume days
        )

    # Step 2: Build training data for surprise regression
    # For each historical day, compute Model A's raw forecast and the
    # resulting surprise at each bin.
    #
    # Reference: Satish et al. 2014, p.18-19, "volume surprises based on
    #   a naive volume forecast model can be used to train a rolling
    #   regression model"
    #
    # surprise[d, j] = (actual_volume[d, j] - raw_forecast[d, j]) / raw_forecast[d, j]
    # [Researcher inference: the paper does not specify the exact surprise
    #  formula. We use relative surprise (percentage deviation from raw
    #  forecast) because it normalizes across stocks and bins of different
    #  volume levels. The paper says "departures from a historical average
    #  approach" (p.18) which could also mean absolute difference, but
    #  relative is more robust.]

    training_X = []  # predictor: lagged surprises
    training_y = []  # response: percentage deviation

    FOR each day d in (train_end_date - N_regression_fit + 1)..train_end_date:
        # Get Model A forecasts for this day (unconditional, as if at market open)
        raw_fcst = forecast_raw_volume(model_a, stock, d, current_bin=1, {})
        daily_total = sum(volume[stock, bin=k, day=d] for k in 1..I)

        surprises = {}
        FOR each bin j in 1..I:
            IF raw_fcst[j] > min_volume_floor:
                surprises[j] = (volume[stock, bin=j, day=d] - raw_fcst[j]) / raw_fcst[j]
            ELSE:
                surprises[j] = 0.0

        FOR each bin j in (L_max + 1)..I:
            # Predictor: L lagged surprises [surprise[d, j-1], ..., surprise[d, j-L]]
            x = [surprises[j - lag] for lag in 1..L_max]
            # Response: actual percentage minus historical percentage
            actual_pct = volume[stock, bin=j, day=d] / daily_total
            y = actual_pct - hist_pct[j]
            training_X.append(x)
            training_y.append(y)

    # Step 3: Select optimal number of lag terms via cross-validation
    # Reference: Satish et al. 2014, p.19, "identifying the optimal number
    #   of model terms for U.S. equities"
    # [Researcher inference: the paper says they identified the optimal
    #  number but does not disclose it. Cross-validation is the standard
    #  approach for this selection.]

    best_L = 1
    best_cv_error = +infinity
    FOR L in 1..L_max:
        # K-fold cross-validation (K=5) on training data
        cv_error = cross_validate_ols_no_intercept(
            training_X[:, 0:L], training_y, K=5
        )
        IF cv_error < best_cv_error:
            best_cv_error = cv_error
            best_L = L

    # Step 4: Fit final OLS regression with optimal L, no intercept
    # Reference: Satish et al. 2014, p.19, "we perform both regressions
    #   without the inclusion of a constant term"
    beta = ols_no_intercept(training_X[:, 0:best_L], training_y)

    RETURN ModelB(hist_pct, beta, best_L)


FUNCTION forecast_volume_percentage(model_b, model_a, stock, date,
                                     current_bin, observed_volumes):
    """
    Predict the volume percentage for the next bin (current_bin + 1).
    Uses Model A to compute surprises from observed bins.

    Reference: Satish et al. 2014, pp.18-19, volume percentage methodology.
    """

    target_bin = current_bin + 1
    IF target_bin > I: RETURN None  # no more bins to predict

    # Step 1: Check switch-off condition
    # Reference: Satish et al. 2014, p.24, referencing Humphery-Jenner (2011):
    #   "once 80% of the day's volume is reached, return to a historical approach"
    #
    # Estimate total daily volume using observed + forecasted remaining
    raw_fcst = forecast_raw_volume(model_a, stock, date, current_bin + 1,
                                    observed_volumes)
    V_total_est = (sum(observed_volumes[j] for j in 1..current_bin)
                   + sum(raw_fcst[j] for j in current_bin+1..I))
    cumulative_observed = sum(observed_volumes[j] for j in 1..current_bin)

    IF V_total_est > 0 AND cumulative_observed / V_total_est >= pct_switchoff:
        # Switch off: return historical percentage, renormalized
        remaining_frac = 1.0 - sum(
            observed_volumes[j] / V_total_est for j in 1..current_bin
        )
        pct_forecast = model_b.hist_pct[target_bin]
        remaining_hist = sum(model_b.hist_pct[j] for j in target_bin..I)
        IF remaining_hist > 0:
            pct_forecast = model_b.hist_pct[target_bin] * remaining_frac / remaining_hist
        RETURN pct_forecast

    # Step 2: Compute surprises from observed bins
    # Get unconditional Model A forecasts (as baseline for surprises)
    raw_fcst_uncond = forecast_raw_volume(model_a, stock, date, current_bin=1, {})
    surprises = {}
    FOR each bin j in 1..current_bin:
        IF raw_fcst_uncond[j] > min_volume_floor:
            surprises[j] = (observed_volumes[j] - raw_fcst_uncond[j]) / raw_fcst_uncond[j]
        ELSE:
            surprises[j] = 0.0

    # Step 3: Apply surprise regression
    # If fewer than L lags available (early in day), set delta = 0
    # [Researcher inference: paper does not address this edge case. Zero
    #  delta means falling back to historical percentage, which is safe.]
    L = model_b.L
    IF current_bin < L:
        delta = 0.0
    ELSE:
        x = [surprises[current_bin - lag + 1] for lag in 1..L]
        delta = dot(model_b.beta, x)

    # Step 4: Apply deviation constraint
    # Reference: Satish et al. 2014, p.24, "depart no more than 10% away
    #   from a historical VWAP curve"; Humphery-Jenner (2011).
    # [Researcher inference: the paper says they developed a "separate
    #  method to compute the deviation bounds" that is proprietary. We use
    #  a simple proportional constraint on the historical percentage.]

    # Scale historical percentage to account for fraction already consumed
    observed_total_pct = cumulative_observed / V_total_est if V_total_est > 0 else 0
    remaining_frac = 1.0 - observed_total_pct
    remaining_hist = sum(model_b.hist_pct[j] for j in target_bin..I)
    IF remaining_hist > 0:
        scaled_base = model_b.hist_pct[target_bin] * remaining_frac / remaining_hist
    ELSE:
        scaled_base = model_b.hist_pct[target_bin]

    max_delta = max_deviation * scaled_base
    delta = clip(delta, -max_delta, max_delta)

    # Step 5: Compute adjusted percentage
    adjusted_pct = scaled_base + delta

    # Step 6: Renormalize remaining bins
    # Ensure all remaining bins (target_bin..I) still sum to remaining_frac.
    # Reference: Satish et al. 2014, implicit in percentage forecast
    #   methodology (percentages must sum to 1.0 over the full day).
    # [Researcher inference: explicit renormalization is not described in the
    #  paper but is mathematically necessary to maintain the sum constraint.]

    IF target_bin == I:
        # Last bin: no redistribution needed
        pct_forecast = remaining_frac
    ELSE:
        remaining_after_target = remaining_frac - adjusted_pct
        remaining_hist_after_target = sum(model_b.hist_pct[j] for j in target_bin+1..I)
        IF remaining_hist_after_target > 0:
            # Scale factor for bins target_bin+1..I
            scale = remaining_after_target / remaining_hist_after_target
            # (This scale factor would be applied to subsequent bin forecasts;
            #  only the current target_bin forecast is returned here.)
        pct_forecast = adjusted_pct

    RETURN pct_forecast
```

#### Helper Functions

```
FUNCTION assign_regime(pctile, regime_percentiles):
    """
    Map a percentile value to a regime index.
    regime_percentiles: sorted list of cutoff percentiles, e.g., [33.3, 66.7].
    Returns regime index 0 to len(regime_percentiles).
    """
    FOR idx, cutoff in enumerate(regime_percentiles):
        IF pctile < cutoff:
            RETURN idx
    RETURN len(regime_percentiles)  # highest regime


FUNCTION percentile_rank(value, sorted_distribution):
    """
    Compute the percentile rank of value within sorted_distribution.
    Returns value in [0, 100].
    """
    rank = bisect_left(sorted_distribution, value)
    RETURN 100.0 * rank / len(sorted_distribution)
```

### Data Flow

```
Input: Historical volume[stock, bin, day] matrix
  |
  v
[Seasonal Factor Computation] -- trailing 126 days, per bin
  |                                |
  v                                v
[Historical Avg (H)]    [Deseasonalize] --> [Intraday ARMA fit (A)]
  |                                              |
  |    [Inter-day ARMA fit (D)]                  |
  |         |                                    |
  v         v                                    v
[H[i]]   [D[i]]                              [A[i]] (re-seasonalized)
  |         |                                    |
  +----+----+------------------------------------+
       |
       v
[Regime Classification] -- cumulative volume percentile -> regime index
       |
       v
[Weight Lookup] -- weights[regime] = (w1, w2, w3)
       |
       v
[Weighted Combination] -- V_hat[i] = w1*H[i] + w2*D[i] + w3*A[i]
       |
       v
  Model A Output: V_hat[i] for i in current_bin..I
       |
       +-----> [Surprise Computation] -- surprise[j] = (actual - V_hat[j]) / V_hat[j]
                     |
                     v
               [Surprise Regression] -- delta = beta . [surprise lags]
                     |
                     v
               [Deviation Clipping] -- |delta| <= max_deviation * scaled_base
                     |
                     v
               [Renormalization] -- ensure remaining percentages sum correctly
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
| Intraday deseas | deseasonalized[j] | (N_intraday_fit * I,) | float | ratio |
| Intraday reseas | A[i] | (26,) | float | shares |
| Regime | regime_index | scalar | int | 0..n_regimes-1 |
| Weights | w | (3,) | float | dimensionless |
| Raw forecast | V_hat[i] | (26,) | float | shares |
| Surprise | surprise[j] | scalar | float | ratio |
| Lag vector | x | (L,) | float | ratio |
| Regression coef | beta | (L,) | float | pct/ratio |
| Delta | delta | scalar | float | fraction |
| Pct forecast | pct_forecast | scalar | float | fraction (0-1) |

### Variants

This specification implements the full dual-model system from Satish et al. (2014), which is the most complete variant described in the paper. Specifically:

- **Raw Volume Model:** All four components (historical average, inter-day ARMA, intraday ARMA, regime-dependent weights). The paper does not present simpler variants; it evaluates the full model against the historical-average baseline. [Satish et al. 2014, pp.17-18]

- **Volume Percentage Model:** The extended dynamic VWAP approach with Model A's raw forecasts as the surprise baseline. The paper also mentions using simpler historical averages as the surprise baseline ("departures from a historical average approach," p.18) but recommends the Model A approach: "we could apply our more extensive volume forecasting model" (p.19). [Satish et al. 2014, p.19]

- **Not implemented:** The ARMAX variant with calendar event exogenous inputs is explicitly recommended against by the paper: "there is scant historical data represented by the special cases to construct such formal models reliably" (p.18). Custom curves for special days are handled as a lookup table, not a model.

## Parameters

| Parameter | Description | Recommended Value | Sensitivity | Range |
|-----------|-------------|-------------------|-------------|-------|
| I | Number of intraday bins (15-min) | 26 | N/A (fixed) | {13, 26, 78} |
| N_hist | Days for historical rolling average (Component 1) | 21 | Medium | [10, 60] |
| N_seasonal | Days for seasonal factor computation | 126 | Low | [63, 252] |
| N_intraday_fit | Days for intraday ARMA fitting window | 21 | Medium | [15, 42] |
| N_interday_fit | Days for inter-day ARMA fitting window | 63 | Medium | [42, 126] |
| p_max_interday | Maximum AR order for inter-day ARMA | 5 | Low | [3, 5] |
| q_max_interday | Maximum MA order for inter-day ARMA | 5 | Low | [3, 5] |
| p_max_intraday | Maximum AR order for intraday ARMA | 4 | Low | [2, 4] |
| q_max_intraday | Maximum MA order for intraday ARMA | 5 | Low | [3, 5] |
| max_interday_budget | Max terms for inter-day ARMA (p+q+1) | 8 | Low | [7, 9] |
| n_regimes | Number of regime buckets | Grid search {3,4,5} | High | [2, 6] |
| regime_percentiles | Percentile cutoffs for regime assignment | Equally spaced | High | Data-driven |
| N_regime_window | Days for building regime percentile distributions | 63 | Medium | [21, 126] |
| N_regression_fit | Days for surprise regression training | 63 | Medium | [21, 126] |
| L_max | Maximum number of surprise lag terms | 5 | Medium | [3, 7] |
| max_deviation | Maximum percentage deviation from historical VWAP curve | 0.10 (10%) | Medium | [0.05, 0.20] |
| pct_switchoff | Cumulative volume fraction to switch off dynamic model | 0.80 (80%) | Medium | [0.70, 0.90] |
| min_volume_floor | Minimum volume for MAPE computation (shares) | 100 | Low | [50, 500] |

### Sources

- **I = 26:** Satish et al. 2014, p.16, "26 such bins in a trading day."
- **N_hist = 21:** Satish et al. 2014, Exhibit 1 caption, "Prior 21 days."
- **N_seasonal = 126:** Satish et al. 2014, p.17, "trailing six months."
- **N_intraday_fit = 21:** Satish et al. 2014, p.18, "rolling basis over the most recent month." 21 trading days ~ 1 calendar month.
- **N_interday_fit = 63:** [Researcher inference] The paper says "Prior 5 days" in Exhibit 1, which refers to the lagged inputs for prediction, not the fitting window. Fitting an ARMA(5,5) on only 5 observations is statistically infeasible. 63 trading days (3 months) provides sufficient data while adapting to regime changes. The paper does not disclose this value.
- **p_max_interday = 5, q_max_interday = 5:** Satish et al. 2014, p.17, "we consider all values of p and q lags through five."
- **p_max_intraday = 4:** Satish et al. 2014, p.18, "AR lags with a value less than five." The constraint applies to AR order only; MA order is unconstrained by the paper.
- **q_max_intraday = 5:** [Researcher inference] Paper constrains AR ("less than five") but does not constrain MA for intraday. We allow up to 5 for consistency with the inter-day ARMA. AICc will penalize excessive orders.
- **max_interday_budget = 8:** [Researcher inference] From "fewer than 11 terms" (p.18) with the interpretation that the intraday model needs at least 2 terms (AR(1) + constant) to provide dynamic information. 11 - 2 - 1 = 8. The paper states "fewer than 11 terms" as an observed outcome, not a constraint.
- **n_regimes:** [Researcher inference] Paper says "training several weight models" (p.18) but does not disclose the number. Grid search over {3, 4, 5} is recommended because the optimal number depends on the stock universe and time period.
- **max_deviation = 0.10:** Satish et al. 2014, p.24, "depart no more than 10% away from a historical VWAP curve," referencing Humphery-Jenner (2011).
- **pct_switchoff = 0.80:** Satish et al. 2014, p.24, "once 80% of the day's volume is reached, return to a historical approach," referencing Humphery-Jenner (2011).
- **N_regression_fit = 63:** [Researcher inference] 3-month window provides ~63 * 22 = 1386 training samples (pooled across bins and days) for the low-dimensional OLS. The paper does not disclose this value.
- **min_volume_floor = 100:** [Researcher inference] Prevents division-by-zero instability in MAPE and surprise computation. Not in the paper.

### Initialization

1. **Data preparation:** Load split-adjusted intraday volume data at 15-minute resolution. Exclude half-day trading sessions (13 bins instead of 26) from training. Exclude days with zero total volume.

2. **Seasonal factors:** Compute from the trailing N_seasonal = 126 days. If fewer than 63 days are available, use all available data. Replace zero seasonal factors with the minimum non-zero value across all bins for that stock.

3. **Historical averages:** Compute H[i] from trailing N_hist = 21 days for each bin.

4. **Inter-day ARMA:** Fit 26 separate ARMA models (one per bin) on the trailing N_interday_fit = 63 days. If a model fails to converge for a (p, q) combination, skip that combination. If no model converges for a bin, fall back to H[i].

5. **Intraday ARMA:** Fit one ARMA model on concatenated deseasonalized intraday volume from trailing N_intraday_fit = 21 days, with day-boundary breaks.

6. **Regime weights:** Run grid search over regime configurations using the trailing N_regime_window = 63 days, holding out the last 21 days for validation. Optimize weights per regime using derivative-free minimization of MAPE.

7. **Percentage regression:** Using Model A's unconditional forecasts on the trailing N_regression_fit = 63 days, compute surprises and fit OLS regression (no intercept) with cross-validated lag count.

### Calibration

All parameters are calibrated from historical data. The calibration procedure is:

1. **ARMA model selection:** Automatic via AICc grid search over (p, q) combinations. No manual tuning required. [Satish et al. 2014, p.17]

2. **Regime count and boundaries:** Grid search over n_regimes in {3, 4, 5} with equally-spaced percentile cutoffs. Select configuration minimizing out-of-sample MAPE on held-out validation period. [Researcher inference]

3. **Component weights:** Per-regime optimization minimizing MAPE, using Nelder-Mead (derivative-free, appropriate for non-differentiable MAPE objective). [Researcher inference; paper says "minimizes the error" but does not specify optimizer]

4. **Surprise regression lag count:** Cross-validation over L in {1, 2, ..., L_max} to select optimal number of lagged surprise terms. [Researcher inference; paper says they "identified the optimal number of model terms" (p.19) but does not disclose it]

5. **Re-estimation schedule:** Re-estimate all component models daily on a rolling basis. Re-run regime grid search and weight optimization monthly (every ~21 trading days) as this is computationally expensive. Between monthly re-estimations, use existing weights and regime boundaries. [Researcher inference; paper does not discuss re-estimation frequency]

## Validation

### Expected Behavior

**Model A (Raw Volume):**
- Median MAPE reduction vs. historical-only baseline: 24% across all intraday intervals. [Satish et al. 2014, p.20, "we reduce the median volume error by 24%"]
- Bottom-95% mean MAPE reduction: 29%. [Satish et al. 2014, p.20]
- Error reduction varies by time of day: ~10% at 9:30 to ~33% at 15:30. [Satish et al. 2014, Exhibit 6, p.22]
- Consistent across SIC industry groups (~15-35% reduction). [Satish et al. 2014, Exhibit 7, p.22]
- Consistent across beta deciles (~20-35% reduction). [Satish et al. 2014, Exhibit 8, p.23]

**Model B (Volume Percentage):**
- Median absolute deviation: 0.00874 (historical) vs. 0.00808 (dynamic) -- 7.55% reduction, significant at << 1% level via Wilcoxon signed-rank test. [Satish et al. 2014, Exhibit 9, p.23]
- Bottom-95% average absolute error: 0.00986 vs. 0.00924 -- 6.29% reduction. [Satish et al. 2014, Exhibit 9, p.23]

**VWAP Tracking Error:**
- Mean tracking error reduced from 9.62 bps (historical) to 8.74 bps (dynamic) -- 9.1% reduction, p < 0.01 via paired t-test. [Satish et al. 2014, Exhibit 10, p.23]
- 7-10% reduction within each stock category (Dow 30, midcap, high-variance). [Satish et al. 2014, p.23]

**Benchmark comparison (different dataset/period):**
- Chen et al. (2016) robust Kalman filter: average MAPE 0.46, VWAP tracking 6.38 bps on 30 securities. [Chen et al. 2016, Tables 3-4]
- Chen et al. (2016) rolling mean baseline: average MAPE 1.28. [Chen et al. 2016, Table 3]
- Direct comparison is imprecise due to different datasets (500 vs. 30 securities), time periods, and exchanges.

### Sanity Checks

1. **Historical-only baseline reproduction:** With w1=1.0, w2=0.0, w3=0.0 for all regimes, Model A should exactly reproduce the historical rolling average. Verify that MAPE matches the baseline.

2. **Seasonal U-shape:** Plot seasonal_factor[i] for i=1..26. Should show elevated volume at bins 1-2 (9:30-10:00) and bins 24-26 (15:00-16:00), with a trough around bins 10-16 (midday). [Satish et al. 2014, p.17, standard intraday volume pattern]

3. **ARMA order distribution:** Across all 26 inter-day ARMA models for a stock, the selected (p, q) orders should predominantly be low: most models should have p+q <= 4. If many models select p=5 or q=5, the AICc penalty may be miscalibrated. [Satish et al. 2014, p.18, "AR coefficients quickly decayed"]

4. **Stationarity of deseasonalized series:** Run ADF test on the deseasonalized intraday volume series. Should reject the null of a unit root at p < 0.05. If non-stationary, deseasonalization or the seasonal factor window may need adjustment.

5. **Weight non-negativity:** All optimized weights w1, w2, w3 should be >= 0 for every regime. If optimizer returns negative weights despite constraints, check the optimization implementation.

6. **Regime bucket population:** Each regime bucket should contain at least 5% of the total (day, bin) observations. If a regime is nearly empty, the weight optimization for that regime is unreliable.

7. **Deviation constraint binding frequency:** The max_deviation constraint should bind (clip delta) on no more than 10-20% of bin forecasts. If it binds much more frequently, the regression is producing implausibly large adjustments.

8. **Switch-off behavior:** The 80% switch-off should activate for the last 2-4 bins on a typical day. If it never activates or activates before bin 20, check V_total_est computation.

9. **Cumulative percentage monotonicity:** Sum of percentage forecasts for bins 1..j should be monotonically increasing and approach 1.0 as j approaches I.

10. **Surprise coefficient signs:** For L=1, the regression coefficient beta[0] should be positive (positive surprise at bin j implies positive adjustment at bin j+1). Typical magnitude: |beta[0]| < 0.5.

11. **Surprise mean near zero:** Average surprise across all (day, bin) pairs should be approximately zero (model is unbiased on average). Standard deviation of surprises typically in [0.005, 0.015] for percentage-space surprises.

12. **Monotonic improvement with components:** Adding each component should reduce out-of-sample MAPE:
    - H only < H + D < H + D + A (approximately).
    If adding a component worsens MAPE, its weight should be driven to near-zero by the optimizer.

13. **Day-boundary check:** Verify that the intraday ARMA does not exhibit spurious correlation across day boundaries. Compute the correlation between the last deseasonalized bin of day d and first deseasonalized bin of day d+1. If the day-break implementation is correct, this correlation should be used only through the lag structure, not as a direct input.

### Edge Cases

1. **ARMA convergence failure:** If an inter-day ARMA fails to converge for all (p, q) candidates at a given bin, set its weight contribution to the historical average H[i]. Expected failure rate: 1-5% of per-bin fits across a 500-stock universe. Detection: check optimizer convergence flag, verify finite standard errors. [Researcher inference]

2. **Insufficient surprise lags (early bins):** At bins 1 through L, fewer than L lagged surprises are available. Set delta = 0 (fall back to historical percentage). [Researcher inference]

3. **Half-day trading sessions:** Days before holidays have 13 bins instead of 26. Exclude from training and evaluation. Mark in a calendar table. [Researcher inference; paper uses full days only]

4. **Special calendar days:** Option expiration, Fed announcements, index rebalancing. "Custom curves for special calendar days... rather than ARMAX models, due to insufficient historical occurrences" (Satish et al. 2014, p.18). Maintain a calendar lookup; substitute pre-computed curves for these days and bypass the dynamic model.

5. **Zero-volume bins:** Floor prevents division by zero in deseasonalization and MAPE. Replace seasonal factor with minimum non-zero value. Exclude bins with actual volume < min_volume_floor from MAPE numerator and denominator. Raw forecasts are clamped at zero. [Researcher inference]

6. **Multi-step intraday ARMA degradation:** Recursive multi-step ARMA predictions degrade with forecast horizon. For bins far from the last observation, the intraday ARMA converges to the unconditional mean. Paper acknowledges: "techniques that predict only the next interval will perform better than those attempting to predict volume percentages for the remainder of the trading day" (p.18). Regime weights may naturally down-weight the intraday component for distant bins. [Satish et al. 2014, p.18]

7. **Joint term constraint infeasibility:** If the max-terms inter-day ARMA uses 8 terms, intraday is limited to 2 (constant + one AR lag). This is acceptable; very few bins should require such high inter-day orders. If the constraint frequently binds, increase max_interday_budget to 9. [Researcher inference]

8. **Regime transition at start of day (bin 1):** No cumulative volume observed. Default to the middle regime. An alternative is to use an unconditional (all-regime-averaged) weight set. The choice has minimal impact because bin 1 is typically high-volume with low forecasting leverage. [Researcher inference]

9. **Stock splits / corporate actions:** Volume data must be split-adjusted. A 2:1 split doubles apparent volume. Use split-adjusted share counts from the data vendor, or normalize by daily shares outstanding as in Chen et al. (2016, Section 4.1). [Researcher inference]

10. **Last-bin edge case in renormalization (Model B):** When predicting the last bin (target_bin == I), there are no remaining bins to redistribute. The percentage forecast is simply the remaining fraction (1.0 minus sum of observed percentages). [Researcher inference]

11. **Lookahead bias in percentage model training:** Model A is trained on all data through train_end_date, but used to compute forecasts for earlier days. Bias is negligible: ARMA parameters on 63+ days change minimally when dropping one day, and seasonal factors are 126-day averages. To eliminate entirely: use expanding-window re-estimation (multiplies computational cost by N_regression_fit). [Researcher inference]

### Known Limitations

1. **Point forecasts only:** Unlike CMEM or Kalman filter models, this approach does not specify a noise distribution and cannot produce prediction intervals or density forecasts. [Researcher inference; paper only reports point forecast metrics]

2. **No outlier robustness:** Unlike Chen et al.'s robust Kalman filter with Lasso regularization, no built-in outlier handling. Outlier bins directly affect historical averages, ARMA estimates, and surprise computations. [Researcher inference]

3. **Single-stock model:** Each stock is modeled independently. Cross-sectional information (e.g., sector-wide volume surges) is not exploited. [Researcher inference]

4. **Static seasonal assumption:** The 6-month trailing average for seasonal factors assumes the intraday volume shape is constant over that window. Structural changes (e.g., shifts in closing auction participation) are captured slowly. [Researcher inference]

5. **Proprietary components not disclosed:** The paper does not disclose specific values for regime thresholds, optimized weights, or the optimal number of regression terms for U.S. equities. Replicators must rediscover these through grid search. [Satish et al. 2014, p.18, implicit throughout]

## Paper References

| Spec Section | Paper Source |
|-------------|-------------|
| Overall architecture | Satish et al. 2014, pp.17-18, full methodology section |
| Bin structure (I=26) | Satish et al. 2014, p.16, "26 such bins in a trading day" |
| Historical Window Average | Satish et al. 2014, p.17, "Historical Window Average/Rolling Means" |
| N_hist = 21 | Satish et al. 2014, Exhibit 1, "Prior 21 days" |
| Inter-day ARMA | Satish et al. 2014, p.17, "per-symbol, per-bin ARMA(p,q)" |
| AICc selection | Satish et al. 2014, p.17, "corrected AIC... Hurvich and Tsai [1989, 1993]" |
| Intraday ARMA | Satish et al. 2014, pp.17-18, deseasonalization and rolling month fit |
| AR < 5 constraint | Satish et al. 2014, p.18, "AR lags with a value less than five" |
| Fewer than 11 terms | Satish et al. 2014, p.18, "a dual ARMA model having fewer than 11 terms" |
| Regime switching | Satish et al. 2014, p.18, "regime switching... historical volume percentile cutoffs" |
| Dynamic weight overlay | Satish et al. 2014, p.18, "dynamic weight overlay... minimizes the error" |
| Custom curves | Satish et al. 2014, p.18, "custom curves for special calendar days" |
| Volume percentage model | Satish et al. 2014, pp.18-19; Humphery-Jenner (2011) |
| No-intercept regression | Satish et al. 2014, p.19, "we perform both regressions without the inclusion of a constant term" |
| Deviation limit (10%) | Satish et al. 2014, p.24; Humphery-Jenner (2011) |
| Switch-off (80%) | Satish et al. 2014, p.24; Humphery-Jenner (2011) |
| MAPE formula | Satish et al. 2014, p.17, "Measuring Raw Volume Predictions -- MAPE" |
| MAD formula | Satish et al. 2014, p.17, "Measuring Percentage Volume Predictions -- Absolute Deviation" |
| Raw volume results | Satish et al. 2014, p.20, Exhibit 6 (p.22), Exhibits 7-8 (pp.22-23) |
| Percentage results | Satish et al. 2014, Exhibit 9 (p.23) |
| VWAP simulation results | Satish et al. 2014, Exhibit 10 (p.23) |
| VWAP-error regression | Satish et al. 2014, Exhibits 2-5 (pp.20-21) |
| Chen et al. benchmark | Chen et al. 2016, Tables 3-4 |
| N_interday_fit = 63 | Researcher inference (see Parameters section) |
| N_regression_fit = 63 | Researcher inference (see Parameters section) |
| Regime count grid search | Researcher inference (see Parameters section) |
| min_volume_floor = 100 | Researcher inference (see Parameters section) |
| Zero seasonal factor handling | Researcher inference |
| Day-boundary handling for intraday ARMA | Researcher inference |
| Surprise formula (relative) | Researcher inference |
| Weight optimization (MAPE, Nelder-Mead) | Researcher inference |
| Renormalization in Model B | Researcher inference |
