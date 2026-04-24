# Implementation Specification: Dual-Mode Intraday Volume Forecast (Raw Volume + Volume Percentage)

## Overview

This specification describes a dual-model system for intraday volume forecasting
based on Satish, Saxena, and Palmer (2014). **Model A** forecasts raw bin-level
trading volume by combining four components — a rolling historical average (H),
an inter-day ARMA forecast (D), an intraday ARMA forecast (A), and a
regime-dependent dynamic weight engine — into a single point estimate. The
paper explicitly counts these as "four components" (Satish et al. 2014, p.17:
"The raw volume forecast model consists of four components, see Exhibit 1").
The first three are signal sources; the fourth (the dynamic weight engine) is
the combination mechanism that produces the final forecast. **Model B**
forecasts next-bin volume percentages by applying a surprise-regression
adjustment (extending Humphery-Jenner 2011's "dynamic VWAP" framework) on top
of a historical volume percentage curve.

The models are loosely coupled: Model A's raw volume forecasts can provide the
surprise signal that drives Model B (the "sophisticated" variant described on
p.19), or Model B can operate independently using simple historical-average-based
surprises (the "naive" variant). Both variants are specified below.

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
(determined by the historical percentile of cumulative volume observed so far).

The three signal components are:
1. **H (Historical average):** Rolling mean of volume in the target bin over the
   prior N_hist days. (Satish et al. 2014, p.17 para 1, Exhibit 1.)
2. **D (Inter-day ARMA):** One-step-ahead forecast from a per-bin ARMA(p,q) model
   fitted on daily volume series for that bin. Captures serial correlation in
   daily volumes. (Satish et al. 2014, p.17 para 2.)
3. **A (Intraday ARMA):** Forecast from a single ARMA(p,q) model fitted on
   deseasonalized intraday bin volumes. Captures within-day momentum. The
   forecast is re-seasonalized before use. (Satish et al. 2014, p.17 paras 3-5.)

The combination formula is:

    V_hat[i] = w_H[r] * H[i] + w_D[r] * D[i] + w_A[r] * A[i]

where r is the current regime index determined by the cumulative volume
percentile. (Satish et al. 2014, p.18 para 1.)

**Note on Exhibit 1 labels:** Exhibit 1 (p.18) labels the inputs as "Next Bin
(Prior 21 days)" for H, "Next Bin (Prior 5 days)" for D, and "Current Bin /
4 Bins Prior to Current Bin" for A. The "Prior 5 days" label for D refers to
the maximum AR lag order (p_max = 5), not the fitting window length — the
inter-day ARMA is fitted on a much longer window (N_interday_fit ~ 252 days)
to provide sufficient data for stable parameter estimation. The paper confirms
on p.17: "we consider all values of p and q lags through five." Similarly,
"4 Bins Prior to Current Bin" for A refers to the maximum AR lag order for
the intraday ARMA (p_max_intra = 4), which is the upper bound for AICc-based
model selection — the actual selected order may be fewer lags.
(Researcher inference — the Exhibit 1 labels are ambiguous but this
interpretation is consistent with the text on p.17-18.)

**Model B (Volume Percentage Forecast)** takes the same inputs and produces a
forecast of what fraction of the day's total volume will trade in the next bin.
It adjusts a static historical percentage curve using recent volume surprises
(deviations of actual bin volumes from a baseline forecast), subject to
adaptive deviation constraints and a late-day switch-off rule. (Satish et al.
2014, pp.18-19, referencing Humphery-Jenner 2011.)

**Inputs:**
- Historical daily volume time series per bin: volume[stock, day, bin] — split-adjusted share counts.
- Current day's observed volumes through the current bin.

**Outputs:**
- Model A: V_hat[stock, day, bin] — forecasted raw volume (shares) for each target bin.
- Model B: pct_hat[stock, day, next_bin] — forecasted volume fraction for the next bin.

### Pseudocode

The system is organized into 12 functions: 7 for Model A, 2 for Model B, and
3 for orchestration/training.

---

#### Function 1: compute_seasonal_factors

Computes the per-bin rolling average over a long trailing window (6 months).
Used exclusively for deseasonalizing the intraday ARMA input series. This is
NOT the same as Component 1 (historical average); see Function 2.

```
FUNCTION compute_seasonal_factors(volume_history, N_seasonal) -> seasonal_factors[1..I]
    # Input:  volume_history[day, bin] for trailing N_seasonal trading days
    # Output: seasonal_factors array of length I

    FOR i = 1 TO I:
        values = [volume_history[d, i] FOR d IN trailing N_seasonal days]
        seasonal_factors[i] = mean(values)

    # Guard: replace zero seasonal factors with minimum non-zero value
    nonzero_min = min([sf FOR sf IN seasonal_factors IF sf > 0], default=1.0)
    FOR i = 1 TO I:
        IF seasonal_factors[i] == 0:
            seasonal_factors[i] = nonzero_min

    RETURN seasonal_factors
```

**References:** Satish et al. 2014, p.17 para 5 ("dividing by the intraday amount
of volume traded in that bin over the trailing six months"). The zero-floor
guard is Researcher inference — the paper does not address zero-volume bins.

---

#### Function 2: compute_historical_average

Computes the per-bin rolling average over a shorter trailing window (N_hist
days). This is Model A Component 1 — the "Historical Window" in Exhibit 1.

```
FUNCTION compute_historical_average(volume_history, N_hist) -> hist_avg[1..I]
    # Input:  volume_history[day, bin] for trailing N_hist days
    # Output: hist_avg array of length I (Component 1 of Model A)

    FOR i = 1 TO I:
        values = [volume_history[d, i] FOR d IN trailing N_hist days]
        hist_avg[i] = mean(values)

    # Same zero-floor guard as seasonal factors
    nonzero_min = min([v FOR v IN hist_avg IF v > 0], default=1.0)
    FOR i = 1 TO I:
        IF hist_avg[i] == 0:
            hist_avg[i] = nonzero_min

    RETURN hist_avg
```

**References:** Satish et al. 2014, p.17 para 1 ("a rolling historical average for
the volume trading in a given 15-minute bin"), p.16 ("the average of daily volume
over the prior N days"), Exhibit 1 ("Next Bin (Prior 21 days)"). The 21-day
window is from Exhibit 1's label; N is described as tunable ("a variable that we
shall call N", p.16). The zero-floor guard is Researcher inference.

**Distinction from Function 1:** Function 1 uses a ~126-day window for
deseasonalization; Function 2 uses a ~21-day window for Component 1. A shorter
window makes H more responsive to recent volume shifts, while the longer
deseasonalization window provides a stable reference for removing the
intraday shape.

---

#### Function 3: fit_interday_arma

Fits I independent ARMA(p,q) models (one per bin) on daily volume series.
Model selection uses AICc. Returns models and their term counts (needed for
the joint dual-ARMA constraint with the intraday model — see Function 4).

```
FUNCTION fit_interday_arma(volume_history, N_interday_fit, p_max, q_max)
        -> (interday_models[1..I], interday_term_counts[1..I])
    # Input:  volume_history[day, bin] for trailing N_interday_fit days
    # Output: array of I fitted ARMA model objects (or FALLBACK sentinel)
    #         array of I term counts (p + q + 1 for each model, 0 for FALLBACK)

    FOR i = 1 TO I:
        series = [volume_history[d, i] FOR d IN trailing N_interday_fit days]
        n = len(series)

        best_aicc = +infinity
        best_model = None
        best_k = 0

        FOR p = 0 TO p_max:
            FOR q = 0 TO q_max:
                k = p + q + 1         # +1 for constant term
                IF n <= k + 1:
                    CONTINUE           # AICc denominator must be positive

                model = fit_ARMA(series, order=(p, q), include_constant=True,
                                 method="MLE", enforce_stationarity=True,
                                 enforce_invertibility=True)

                IF NOT model.converged:
                    CONTINUE

                aic = -2 * model.log_likelihood + 2 * k
                aicc = aic + (2 * k * (k + 1)) / (n - k - 1)

                IF aicc < best_aicc:
                    best_aicc = aicc
                    best_model = model
                    best_k = k

        IF best_model IS None:
            interday_models[i] = FALLBACK    # use hist_avg[i] as fallback
            interday_term_counts[i] = 0
        ELSE:
            interday_models[i] = best_model
            interday_term_counts[i] = best_k

    RETURN (interday_models, interday_term_counts)
```

**References:** Satish et al. 2014, p.17 para 2 ("a per-symbol, per-bin ARMA(p,q)
model... We use nearly standard ARMA model-fitting techniques relying on
maximum-likelihood estimation, which selects an ARMA(p,q) model minimizing an
Akaike information criterion... We depart from the standard technique in using
the corrected AIC, symbolized by AICc, as detailed by Hurvich and Tsai [1989,
1993]... we consider all values of p and q lags through five, as well as a
constant term."). The AICc formula follows Hurvich and Tsai (1989, 1993). The
FALLBACK sentinel and convergence guards are Researcher inference.

**ARMA Model Interface** (applies to both inter-day and intraday models):

Each fitted ARMA model object exposes:

- `predict_next() -> float`: One-step-ahead forecast for the next unobserved
  time step, based on the fitted coefficients and the observation history
  stored in the model. Used during live prediction. This is a pure query
  that does NOT modify the model's internal state. It can be called multiple
  times and will return the same value until append_observation() is called.

- `append_observation(value)`: Extends the model's observation buffer with a
  new data point and updates the AR/MA recursion state. The ARMA coefficients
  are NOT re-estimated; only the recursion state is advanced (Kalman-filter
  style update).

- `predict_at(d) -> float`: Returns the one-step-ahead conditional mean
  E[Y_d | Y_{d-1}, ...] for a given historical index d. Used during training
  to reconstruct what the model would have predicted on each historical day.

- `make_state(observations) -> state`: (Intraday model only.) Creates a fresh
  prediction state from a sequence of deseasonalized observations WITHOUT
  mutating the model's stored state. Returns a state object. The state is
  initialized with the unconditional mean for AR lags and zero for MA
  residuals. Observations are then processed sequentially through the Kalman
  filter to build up the conditional state. For the first few observations
  (fewer than max(p,q)), the prediction relies on the unconditional
  initialization; accuracy improves as more observations are incorporated.

- `predict_from_state(state, steps) -> list[float]`: (Intraday model only.)
  Produces multi-step-ahead forecasts from the given state. Returns a list of
  length `steps` where element k is the (k+1)-step-ahead forecast.

- `term_count -> int`: Returns p + q + 1 (number of parameters including the
  constant term).

Researcher inference — the paper does not specify the prediction interface.
These semantics are standard for ARMA state-space implementations (e.g.,
statsmodels SARIMAX). The make_state initialization procedure (unconditional
mean for AR lags, zero for MA residuals) is Researcher inference — this is
the standard unconditional initialization used in Kalman filter implementations
of ARMA models.

---

#### Function 4: fit_intraday_arma

Fits a single ARMA model per stock on deseasonalized intraday volume from the
most recent N_intraday_fit trading days. Each day is treated as an independent
segment to avoid spurious lag connections across the overnight gap.

The intraday model is subject to a **joint term budget** with the inter-day
models: the paper states "we fit each symbol with a dual ARMA model having
fewer than 11 terms" (Satish et al. 2014, p.18 para 1). This means the total
number of parameters across the inter-day model for a given bin AND the
intraday model must be fewer than 11. Since the intraday model is shared
across all bins but the inter-day models are per-bin, the intraday model must
satisfy this constraint for ALL bins simultaneously. Therefore, the intraday
term budget is determined by the maximum inter-day term count across all bins:

    intraday_budget = max_dual_arma_terms - max(interday_term_counts)

This ensures that even the bin with the most complex inter-day model stays
within the joint budget.

```
FUNCTION fit_intraday_arma(volume_history, seasonal_factors, N_intraday_fit,
                           p_max_intra, q_max_intra,
                           interday_term_counts) -> intraday_model
    # Input:  volume_history[day, bin], seasonal_factors[1..I]
    #         interday_term_counts[1..I]: term count for each inter-day model
    # Output: fitted ARMA model object (or FALLBACK sentinel)

    # Compute intraday term budget from joint constraint
    max_interday_terms = max(interday_term_counts)
    # If all inter-day models are FALLBACK (term count 0), allow full budget
    IF max_interday_terms == 0:
        max_interday_terms = 0
    intraday_budget = max_dual_arma_terms - max_interday_terms
    # intraday_budget is the maximum number of terms (p + q + 1) allowed

    IF intraday_budget < 2:
        # Even ARMA(0,0) with constant = 1 term needs budget >= 1,
        # but meaningful models need >= 2 terms. If budget is too tight,
        # fall back.
        RETURN FALLBACK

    # Build deseasonalized series as independent daily segments
    segments = []
    FOR d IN trailing N_intraday_fit days:
        daily_deseas = []
        FOR i = 1 TO I:
            daily_deseas.append(volume_history[d, i] / seasonal_factors[i])
        segments.append(daily_deseas)

    # Independent-segment (panel) ARMA fitting:
    # Each day is a separate segment of length I = 26, sharing the same ARMA
    # coefficients. The total log-likelihood is the sum of per-segment
    # log-likelihoods. At each segment boundary, the AR lag values and MA
    # residuals reset to the unconditional mean / zero.
    #
    # Implementation: concatenate segments into a single series but zero out
    # AR/MA lag contributions that cross segment boundaries. Or use a panel
    # ARMA routine, or manually sum per-segment likelihoods.

    best_aicc = +infinity
    best_model = None
    n_total = len(segments) * I   # total observations

    FOR p = 0 TO p_max_intra:
        FOR q = 0 TO q_max_intra:
            k = p + q + 1
            # Joint dual-ARMA constraint: intraday terms must fit within budget
            IF k > intraday_budget:
                CONTINUE

            usable_per_segment = I - max(p, q)
            IF usable_per_segment <= 0:
                CONTINUE
            n_eff = len(segments) * usable_per_segment

            IF n_eff <= k + 1:
                CONTINUE

            model = fit_panel_ARMA(segments, order=(p, q),
                                   include_constant=True,
                                   method="MLE",
                                   enforce_stationarity=True,
                                   enforce_invertibility=True)

            IF NOT model.converged:
                CONTINUE

            aic = -2 * model.log_likelihood + 2 * k
            aicc = aic + (2 * k * (k + 1)) / (n_eff - k - 1)

            IF aicc < best_aicc:
                best_aicc = aicc
                best_model = model

    IF best_model IS None:
        RETURN FALLBACK
    RETURN best_model
```

**References:** Satish et al. 2014, p.17 paras 3-5 ("we fit an additional ARMA(p,q)
model over deseasonalized intraday bin volume data. The intraday data are
deseasonalized by dividing by the intraday amount of volume traded in that bin
over the trailing six months... the autoregressive (AR) coefficients quickly
decayed, so that we used AR lags with a value less than five... we fit each
symbol with a dual ARMA model having fewer than 11 terms. We compute this model
on a rolling basis over the most recent month.").

The joint term-budget enforcement is based on the paper's "dual ARMA model having
fewer than 11 terms" (p.18 para 1), where "dual" refers to the combined inter-day
+ intraday ARMA system. The use of max(interday_term_counts) to determine the
budget is Researcher inference — the paper does not specify how to handle the
fact that inter-day models are per-bin while the intraday model is shared. Using
the maximum ensures the joint constraint is satisfied for every bin.

The independent-segment approach (resetting state at day boundaries) is
Researcher inference — the paper does not specify how overnight gaps are handled,
but treating each day as a fresh segment is the natural approach since the
intraday model captures within-day patterns, not overnight dynamics.

---

#### Function 5: optimize_regime_weights

Trains the regime-switching dynamic weight overlay. Optimizes per-regime weight
vectors [w_H, w_D, w_A] by minimizing MAPE on in-sample data, using
historical volume percentile cutoffs to assign each observation to a regime.

Weights are constrained to be non-negative and sum to 1 using a softmax
parameterization. This ensures interpretable proportional weights and matches
the natural meaning of a "weight overlay" (Satish et al. 2014, p.18 para 1).

```
FUNCTION optimize_regime_weights(
        volume_history,             # [day, bin] array for training window
        hist_avg_per_day,           # hist_avg[d][i] or ability to compute H for each training day
        interday_models,            # fitted inter-day ARMA models (1..I)
        intraday_model,             # fitted intraday ARMA model
        seasonal_factors,           # for re-seasonalizing intraday forecasts
        N_hist,                     # for computing rolling H per training day
        N_weight_train,             # number of training days for weight optimization
        N_regimes,                  # number of regime buckets
        N_regime_valid              # validation days held out for regime count selection
    ) -> (regime_thresholds[1..N_regimes-1], weights[1..N_regimes][3])

    # ---- Step 1: Build component forecasts for each (day, bin) in training ----
    # For each training day d and each bin i, compute what H, D, and A
    # would have predicted at that point.
    #
    # NOTE: This uses in-sample fitted values rather than true out-of-sample
    # predictions for the ARMA components — the inter-day ARMA models were
    # fitted on data that may include days within the weight training window.
    # This is an approximation that produces slightly optimistic "predictions"
    # during weight training. A walk-forward approach (refitting ARMA models
    # for each training day using only prior data) would avoid this look-ahead
    # bias but is computationally expensive. The approximation is acceptable
    # because: (a) the ARMA models contribute only one of three signals, so
    # weight optimization is robust to moderate ARMA prediction bias; (b) the
    # regime count selection uses a held-out validation set that partially
    # mitigates overfitting; and (c) the paper's own methodology likely uses
    # a similar approach given the described "rolling" re-estimation.
    # (Researcher inference — the paper does not specify whether the weight
    # training uses in-sample or out-of-sample ARMA predictions.)

    training_days = trailing N_weight_train days from volume_history
    records = []    # list of (actual, H_pred, D_pred, A_pred, cum_vol_pctile)

    FOR d IN training_days:
        # H for day d: rolling average of bin volumes over [d-N_hist .. d-1]
        H_d = compute_historical_average(volume_history[d-N_hist:d], N_hist)

        # D for day d: inter-day ARMA one-step-ahead forecast
        D_d = [interday_models[i].predict_at(d) FOR i = 1 TO I]
        # For FALLBACK models, D_d[i] = H_d[i]

        # A for day d: intraday ARMA predictions
        # Build deseasonalized observations for bins already seen on day d,
        # then predict remaining bins.
        deseas_today = [volume_history[d, i] / seasonal_factors[i] FOR i = 1 TO I]
        state = intraday_model.make_state([])  # fresh state at day start

        FOR i = 1 TO I:
            # Predict bin i using bins [1..i-1] as observed
            IF i == 1:
                A_raw = intraday_model.predict_from_state(state, 1)[0]
            ELSE:
                state = intraday_model.make_state(deseas_today[1:i])
                A_raw = intraday_model.predict_from_state(state, 1)[0]
            A_pred_i = A_raw * seasonal_factors[i]   # re-seasonalize

            actual_i = volume_history[d, i]

            # Cumulative volume percentile through bin i-1
            # (what would be known at prediction time)
            IF i == 1:
                cum_vol_pctile = 0.5   # no info yet, assign to median regime
            ELSE:
                cum_vol_today = sum(volume_history[d, 1:i])
                # Compare to historical cumulative volumes at the same bin
                hist_cum_vols = [sum(volume_history[dd, 1:i])
                                 FOR dd IN trailing N_hist days before d]
                cum_vol_pctile = percentile_rank(cum_vol_today, hist_cum_vols)

            records.append((actual_i, H_d[i], D_d[i], A_pred_i, cum_vol_pctile))

    # ---- Step 2: Select optimal N_regimes via cross-validation ----
    # Split training records into fit and validation portions.
    # The last N_regime_valid days are validation.
    fit_records = records from first (N_weight_train - N_regime_valid) days
    valid_records = records from last N_regime_valid days

    best_n_regimes = None
    best_valid_mape = +infinity

    FOR n_r IN [1, 2, 3, 4, 5]:   # candidate regime counts
        # Define regime thresholds as evenly spaced percentile cutoffs
        thresholds = [k / n_r FOR k = 1 TO n_r - 1]
        # e.g., n_r=3 -> thresholds = [0.333, 0.667]

        # Assign each fit record to a regime based on cum_vol_pctile
        regime_records = {r: [] FOR r = 1 TO n_r}
        FOR (actual, H, D, A, pctile) IN fit_records:
            r = assign_regime(pctile, thresholds)
            regime_records[r].append((actual, H, D, A))

        # Optimize weights per regime using MAPE objective
        regime_weights = {}
        FOR r = 1 TO n_r:
            IF len(regime_records[r]) == 0:
                regime_weights[r] = [1/3, 1/3, 1/3]   # default
                CONTINUE
            regime_weights[r] = minimize_mape(regime_records[r])

        # Evaluate on validation set
        valid_mape = compute_mape_with_regimes(valid_records, thresholds,
                                                regime_weights)

        IF valid_mape < best_valid_mape:
            best_valid_mape = valid_mape
            best_n_regimes = n_r

    # ---- Step 3: Re-optimize on full training set with chosen N_regimes ----
    thresholds = [k / best_n_regimes FOR k = 1 TO best_n_regimes - 1]
    regime_records = {r: [] FOR r = 1 TO best_n_regimes}
    FOR (actual, H, D, A, pctile) IN records:   # all records, not just fit
        r = assign_regime(pctile, thresholds)
        regime_records[r].append((actual, H, D, A))

    final_weights = {}
    FOR r = 1 TO best_n_regimes:
        IF len(regime_records[r]) == 0:
            final_weights[r] = [1/3, 1/3, 1/3]
            CONTINUE
        final_weights[r] = minimize_mape(regime_records[r])

    RETURN (thresholds, final_weights)


# ---- Helper: assign_regime ----
FUNCTION assign_regime(pctile, thresholds) -> regime_index
    # Returns 1-indexed regime: regime 1 for pctile <= thresholds[1], etc.
    FOR k = 1 TO len(thresholds):
        IF pctile <= thresholds[k]:
            RETURN k
    RETURN len(thresholds) + 1


# ---- Helper: percentile_rank ----
FUNCTION percentile_rank(value, reference_values) -> float in [0, 1]
    # Returns the fraction of reference values strictly less than `value`.
    # Equivalent to scipy.stats.percentileofscore(reference_values, value,
    # kind='strict') / 100.
    #
    # percentile_rank(x, ref) = count(ref < x) / len(ref)
    #
    # Edge cases:
    # - If len(ref) == 0, return 0.5 (no information, assign to median).
    # - If value < min(ref), returns 0.0 (lowest regime).
    # - If value > max(ref), returns 1.0 (highest regime).
    IF len(reference_values) == 0:
        RETURN 0.5
    count_below = sum(1 FOR r IN reference_values IF r < value)
    RETURN count_below / len(reference_values)

# (Researcher inference — the paper does not specify the percentile rank
# function. The strict-less-than definition is chosen because it produces
# 0.0 when a stock's volume is at its historical minimum, naturally placing
# it in the lowest regime.)


# ---- Helper: minimize_mape ----
FUNCTION minimize_mape(records) -> weights[3]
    # records: list of (actual, H, D, A)
    # Minimize: mean(|actual - (w_H*H + w_D*D + w_A*A)| / actual) over records
    # Subject to: w_H, w_D, w_A >= 0 and w_H + w_D + w_A = 1
    #
    # Approach: use Nelder-Mead on softmax-transformed parameters.
    #   w_j = exp(theta_j) / sum(exp(theta_k)),  j = 1..3
    # This ensures non-negativity AND sum-to-1 without explicit constraints.
    #
    # Multiple restarts (e.g., 5) with different random initializations
    # to avoid local minima. Return weights from best restart.

    def objective(theta):
        # Softmax: weights are non-negative and sum to 1
        max_theta = max(theta)   # for numerical stability
        exp_theta = [exp(t - max_theta) for t in theta]
        w_sum = sum(exp_theta)
        w = [e / w_sum for e in exp_theta]
        errors = []
        FOR (actual, H, D, A) IN records:
            IF actual <= 0:
                CONTINUE    # skip zero-volume bins
            pred = w[0]*H + w[1]*D + w[2]*A
            errors.append(abs(actual - pred) / actual)
        RETURN mean(errors)

    best_w = [1/3, 1/3, 1/3]
    best_obj = +infinity

    FOR restart = 1 TO N_optimizer_restarts:
        theta_init = [random_normal(0, 0.5) FOR _ = 1 TO 3]
        result = nelder_mead(objective, theta_init, max_iter=1000)
        IF result.fun < best_obj:
            best_obj = result.fun
            # Extract final weights via softmax
            max_t = max(result.x)
            exp_t = [exp(t - max_t) for t in result.x]
            w_sum = sum(exp_t)
            best_w = [e / w_sum for e in exp_t]

    RETURN best_w
```

**References:** Satish et al. 2014, p.18 para 1 ("a dynamic weight overlay on top
of these three components... that minimizes the error on in-sample data. We
incorporate a notion of regime switching by training several weight models for
different historical volume percentile cutoffs and, in our out-of-sample period,
dynamically apply the appropriate weights intraday based on the historical
percentile of the observed cumulative volume.").

The paper does not disclose the number of regimes, the weight optimization
algorithm, or the percentile threshold construction. The following are
Researcher inference:
- Softmax parameterization for sum-to-1 non-negative weights.
- Nelder-Mead optimizer.
- Multiple random restarts for the optimizer.
- Evenly spaced percentile thresholds for regime boundaries.
- Grid search over N_regimes in {1..5} with cross-validation.
- MAPE as the weight optimization objective (consistent with the paper's use
  of MAPE as the primary raw volume metric).

---

#### Function 6: predict_raw_volume

Produces the Model A raw volume forecast for a target bin on the current day.

```
FUNCTION predict_raw_volume(
        target_bin,                 # bin index to forecast (1..I)
        observed_volumes,           # volumes observed today, bins [1..current_bin]
        current_bin,                # last completed bin (0 if none)
        hist_avg,                   # from Function 2, computed at start of day
        interday_models,            # fitted inter-day models
        intraday_model,             # fitted intraday model
        seasonal_factors,           # for re-seasonalizing
        regime_thresholds,          # from Function 5
        regime_weights,             # from Function 5
        volume_history_for_pctile   # recent daily cum-vol data for percentile
    ) -> V_hat

    # Component 1: Historical average (already computed for today)
    H = hist_avg[target_bin]

    # Component 2: Inter-day ARMA forecast
    IF interday_models[target_bin] IS FALLBACK:
        D = H                       # fallback to historical average
    ELSE:
        D = interday_models[target_bin].predict_next()

    # Component 3: Intraday ARMA forecast
    IF intraday_model IS FALLBACK:
        A = H                       # fallback to historical average
    ELSE:
        # Build deseasonalized observation sequence for today
        deseas_today = [observed_volumes[j] / seasonal_factors[j]
                        FOR j = 1 TO current_bin]
        state = intraday_model.make_state(deseas_today)
        steps_ahead = target_bin - current_bin
        forecasts = intraday_model.predict_from_state(state, steps_ahead)
        A = forecasts[-1] * seasonal_factors[target_bin]   # re-seasonalize

    # Determine current regime
    IF current_bin == 0:
        regime = assign_regime(0.5, regime_thresholds)   # no info, use median
    ELSE:
        cum_vol_today = sum(observed_volumes[1:current_bin+1])
        hist_cum_vols = [sum(volume_history_for_pctile[dd, 1:current_bin+1])
                         FOR dd IN recent N_hist days]
        pctile = percentile_rank(cum_vol_today, hist_cum_vols)
        regime = assign_regime(pctile, regime_thresholds)

    # Combine components with regime-specific weights (sum to 1 by construction)
    w = regime_weights[regime]    # [w_H, w_D, w_A]
    V_hat = w[0] * H + w[1] * D + w[2] * A

    # Floor: forecast should not be negative
    V_hat = max(V_hat, 0)

    RETURN V_hat
```

**References:** Satish et al. 2014, p.17-18 (combination of three components),
p.18 para 1 (regime switching via cumulative volume percentile). The
non-negativity floor is Researcher inference. The multi-step intraday
forecasting (when target_bin > current_bin + 1) is Researcher inference
derived from the model structure.

---

#### Function 7: compute_historical_percentages

Computes the historical volume percentage curve used as the baseline for Model B.

```
FUNCTION compute_historical_percentages(volume_history, N_hist_pct)
        -> hist_pct[1..I]
    # Input:  volume_history[day, bin] for trailing N_hist_pct days
    # Output: hist_pct[i] = average fraction of daily volume in bin i

    daily_pcts = []
    FOR d IN trailing N_hist_pct days:
        daily_total = sum(volume_history[d, 1:I+1])
        IF daily_total > 0:
            pcts = [volume_history[d, i] / daily_total FOR i = 1 TO I]
        ELSE:
            pcts = [1.0 / I FOR i = 1 TO I]   # uniform fallback
        daily_pcts.append(pcts)

    hist_pct = [mean([daily_pcts[d][i] FOR d IN range(len(daily_pcts))])
                FOR i = 1 TO I]

    RETURN hist_pct
```

**References:** Satish et al. 2014, p.16 ("the average of daily volume over the
prior N days... the same technique applies to constructing volume percentage
bins"), p.18 para 4 ("volume surprises — deviations from a naive historical
forecast"). The uniform fallback for zero-total days is Researcher inference.

---

#### Function 8: fit_surprise_regression

Fits Model B's rolling surprise regression. For each training day, computes
volume surprises (actual percentage minus baseline percentage for each bin),
then fits a regression predicting next-bin surprise from recent surprise lags.

This function supports two surprise baselines:
- **Naive:** Surprises are computed relative to hist_pct (historical average
  percentages). This is the simpler variant. (Satish et al. 2014, p.18-19:
  "volume surprises based on a naive volume forecast model.")
- **Sophisticated (Model A-based):** Surprises are computed relative to
  Model A's raw volume forecasts converted to percentages. (Satish et al.
  2014, p.19: "we could apply our more extensive volume forecasting model
  described previously as the base model from which to compute volume
  surprises.") See function signature for the optional model_a_forecasts
  parameter.

```
FUNCTION fit_surprise_regression(
        volume_history,             # [day, bin] for training window
        N_hist_pct,                 # window for computing hist percentages
        N_surprise_train,           # number of training days for regression
        N_surprise_lags,            # number of lagged surprise terms
        model_a_forecasts=None      # optional: [day, bin] raw volume forecasts
                                    # from Model A for the training window.
                                    # If provided, uses "sophisticated" variant.
    ) -> regression_coefficients[1..N_surprise_lags]

    # For each training day d and each bin i (where i >= N_surprise_lags + 1),
    # build a regression observation:
    #   target: surprise[d, i] = actual_pct[d, i] - baseline_pct[d, i]
    #   features: [surprise[d, i-1], surprise[d, i-2], ..., surprise[d, i-N_surprise_lags]]

    X = []   # feature matrix
    y = []   # target vector

    training_days = trailing N_surprise_train days from volume_history

    FOR d IN training_days:
        # Determine baseline percentages for day d
        IF model_a_forecasts IS NOT None:
            # Sophisticated variant: use Model A forecasts as baseline
            model_a_total = sum(model_a_forecasts[d, 1:I+1])
            IF model_a_total > 0:
                baseline_pct_d = [model_a_forecasts[d, i] / model_a_total
                                  FOR i = 1 TO I]
            ELSE:
                # Fallback to historical if Model A total is zero
                baseline_pct_d = compute_historical_percentages(
                    volume_history[d - N_hist_pct : d], N_hist_pct)
        ELSE:
            # Naive variant: use historical percentages as baseline
            baseline_pct_d = compute_historical_percentages(
                volume_history[d - N_hist_pct : d], N_hist_pct)

        # Compute actual percentages for day d
        daily_total = sum(volume_history[d, 1:I+1])
        IF daily_total <= 0:
            CONTINUE
        actual_pct_d = [volume_history[d, i] / daily_total FOR i = 1 TO I]

        # Compute surprises
        surprise_d = [actual_pct_d[i] - baseline_pct_d[i] FOR i = 1 TO I]

        # Build regression observations for bins with enough lag history
        FOR i = (N_surprise_lags + 1) TO I:
            features = [surprise_d[i - lag] FOR lag = 1 TO N_surprise_lags]
            X.append(features)
            y.append(surprise_d[i])

    # Fit OLS regression WITHOUT intercept.
    # Justification: When using the naive baseline (hist_pct), mean surprise
    # is approximately zero by construction since hist_pct is the average of
    # actual_pct over the lookback window. When using the sophisticated
    # baseline (Model A), the mean surprise is expected to be small if Model A
    # is reasonably accurate. In either case, no intercept is needed.
    # (Researcher inference — the paper does not explicitly specify whether
    # the surprise regression uses an intercept. The p.19 statement "we perform
    # both regressions without the inclusion of a constant term" refers to
    # the VWAP tracking error vs. volume percentage error validation
    # regressions in Exhibits 3 and 5, NOT to this surprise regression.
    # The no-intercept choice here is justified by the mean-zero-surprise
    # property of the naive baseline.)
    coefficients = ols_no_intercept(X, y)

    RETURN coefficients
```

**References:** Satish et al. 2014, pp.18-19 ("volume surprises based on a naive
volume forecast model can be used to train a rolling regression model that
adjusts market participation... After identifying the key drivers of forecasting
performance in our in-sample data, we were able to identify the optimal number
of model terms for U.S. stocks."), referencing Humphery-Jenner (2011).

Satish et al. 2014, p.19: "For example, we could apply our more extensive volume
forecasting model described previously as the base model from which to compute
volume surprises" — this justifies the sophisticated variant.

The paper does not disclose the optimal number of regression terms for U.S.
equities. The no-intercept choice is Researcher inference (see detailed
justification in code comments above). The training procedure (rolling surprise
computation, lagged features) is Researcher inference based on the described
methodology.

---

#### Function 9: predict_volume_percentage

Produces the Model B forecast for the next bin's volume percentage.
Supports both naive and sophisticated surprise baselines.

The deviation limit and switch-off threshold are adaptive per stock, following
Humphery-Jenner (2011)'s "self-updating" design described in Satish et al.
2014, p.24: "self-updating deviation limits and switch-off parameters."

```
FUNCTION predict_volume_percentage(
        next_bin,                   # bin to forecast (2..I)
        observed_volumes,           # volumes seen today, bins [1..current_bin]
        current_bin,                # last completed bin (= next_bin - 1)
        hist_pct,                   # from Function 7, computed at start of day
        regression_coefficients,    # from Function 8
        deviation_limit,            # max allowed departure from baseline
        switchoff_threshold,        # cumulative fraction for switch-off
        baseline_pct=None           # optional: Model A-based baseline percentages.
                                    # If None, uses hist_pct as baseline (naive).
    ) -> pct_hat

    # Select baseline: use Model A-based if provided, otherwise hist_pct
    base = baseline_pct IF baseline_pct IS NOT None ELSE hist_pct

    # Compute actual percentages for observed bins
    # Use a "remaining fraction" scaling approach:
    cum_vol_today = sum(observed_volumes[1:current_bin+1])

    # Estimate total daily volume (for computing current percentages)
    # Assume remaining volume follows the active baseline (Model A-based
    # in sophisticated mode, hist_pct in naive mode). This ensures the
    # total-volume estimate is consistent with the baseline used for
    # surprise computation.
    remaining_pct = sum(base[current_bin+1 : I+1])
    IF remaining_pct > 0:
        estimated_total = cum_vol_today / (1.0 - remaining_pct)
    ELSE:
        estimated_total = cum_vol_today   # late in day, almost all volume seen

    # Compute actual participation rates for observed bins
    actual_pct = []
    FOR j = 1 TO current_bin:
        IF estimated_total > 0:
            actual_pct.append(observed_volumes[j] / estimated_total)
        ELSE:
            actual_pct.append(base[j])

    # Compute surprises for observed bins relative to baseline
    surprises = [actual_pct[j] - base[j] FOR j = 1 TO current_bin]

    # Switch-off check: if cumulative volume exceeds threshold, revert to hist
    cum_fraction = cum_vol_today / estimated_total IF estimated_total > 0 ELSE 0
    IF cum_fraction >= switchoff_threshold:
        # Scale historical percentage for remaining fraction
        RETURN hist_pct[next_bin] * (1.0 - cum_fraction) / remaining_pct
            IF remaining_pct > 0 ELSE hist_pct[next_bin]

    # Build feature vector from most recent surprises
    N_lags = len(regression_coefficients)
    IF current_bin < N_lags:
        # Not enough observed bins for full feature vector; pad with zeros
        features = [0.0] * (N_lags - current_bin) + surprises[-current_bin:]
    ELSE:
        features = surprises[-N_lags:]

    # Predict surprise adjustment (delta)
    delta = dot_product(regression_coefficients, features)

    # Apply deviation clamp
    delta = clamp(delta, -deviation_limit, +deviation_limit)

    # Adjusted percentage for next bin
    base_pct = base[next_bin]
    adjusted_pct = base_pct + delta

    # Scale for remaining fraction:
    # The base values assume full-day context. Scale by the fraction
    # of volume remaining to get the conditional participation rate.
    IF remaining_pct > 0:
        scale = (1.0 - cum_fraction) / remaining_pct
    ELSE:
        scale = 1.0

    pct_hat = scale * adjusted_pct

    # Floor: percentage should not be negative
    pct_hat = max(pct_hat, 0.0)

    RETURN pct_hat
```

**References:** Satish et al. 2014, pp.18-19 ("deviation limits... depart no more
than 10% away from a historical VWAP curve", "once 80% of the day's volume is
reached, return to a historical approach"), p.24 ("self-updating deviation limits
and switch-off parameters", describing Humphery-Jenner 2011). The
remaining-fraction scaling is Researcher inference — the paper does not detail
how conditional percentages are computed from unconditional hist_pct, but this
is the natural approach to ensure percentages sum to 1 over remaining bins. The
zero-padding for early bins is Researcher inference.

**Self-updating deviation limits and switch-off:** See Function 10a
(calibrate_adaptive_limits) for the per-stock calibration procedure.

---

#### Function 9a: calibrate_adaptive_limits

Calibrates per-stock deviation limits and switch-off thresholds, implementing
the "self-updating" mechanism described in Satish et al. 2014, p.24 (attributed
to Humphery-Jenner 2011).

The paper describes that Humphery-Jenner's model has "self-updating deviation
limits and switch-off parameters" but does not detail the update mechanism.
The following procedure is Researcher inference, designed to adapt these
parameters to each stock's characteristics:

```
FUNCTION calibrate_adaptive_limits(
        volume_history,             # [day, bin] for calibration window
        hist_pct_series,            # historical percentage curves for each day
        N_calibration,              # number of days for calibration
        base_deviation_limit,       # initial deviation limit (0.10)
        base_switchoff_threshold    # initial switch-off threshold (0.80)
    ) -> (deviation_limit, switchoff_threshold)

    # ---- Adaptive deviation limit ----
    # Compute the distribution of actual single-bin surprise magnitudes
    # for this stock. Set deviation_limit as the p-th percentile of
    # absolute surprise magnitudes, clipped to [0.5 * base, 2.0 * base].
    # This ensures illiquid stocks with large surprises get wider limits,
    # while stable large-caps get tighter limits.

    surprise_magnitudes = []
    calibration_days = trailing N_calibration days from volume_history

    FOR d IN calibration_days:
        daily_total = sum(volume_history[d, 1:I+1])
        IF daily_total <= 0:
            CONTINUE
        actual_pct_d = [volume_history[d, i] / daily_total FOR i = 1 TO I]
        FOR i = 1 TO I:
            surprise_mag = abs(actual_pct_d[i] - hist_pct_series[d][i])
            surprise_magnitudes.append(surprise_mag)

    IF len(surprise_magnitudes) > 0:
        # Use the 95th percentile of surprise magnitudes as deviation limit
        p95 = percentile(surprise_magnitudes, 95)
        deviation_limit = clamp(p95, 0.5 * base_deviation_limit,
                                     2.0 * base_deviation_limit)
    ELSE:
        deviation_limit = base_deviation_limit

    # ---- Adaptive switch-off threshold ----
    # Compute the typical cumulative volume fraction reached by each bin.
    # Set switchoff_threshold so that the switch-off activates when
    # the stock has historically completed most of its daily volume.
    # Use the bin at which the median cumulative fraction first exceeds
    # the base threshold.

    cum_fractions_by_bin = {i: [] FOR i = 1 TO I}
    FOR d IN calibration_days:
        daily_total = sum(volume_history[d, 1:I+1])
        IF daily_total <= 0:
            CONTINUE
        running_sum = 0
        FOR i = 1 TO I:
            running_sum += volume_history[d, i]
            cum_fractions_by_bin[i].append(running_sum / daily_total)

    # Find the bin where median cumulative fraction crosses the threshold
    switchoff_bin = I   # default: last bin (effectively no switch-off)
    FOR i = 1 TO I:
        IF len(cum_fractions_by_bin[i]) > 0:
            median_cum = median(cum_fractions_by_bin[i])
            IF median_cum >= base_switchoff_threshold:
                switchoff_bin = i
                BREAK

    # Set threshold to the median cumulative fraction at the crossover bin
    # itself, so the switch-off activates when the stock typically reaches
    # approximately the base threshold level. (Draft 2 used switchoff_bin - 1,
    # which systematically produced thresholds BELOW the base value — e.g.,
    # 0.76 instead of 0.82 — causing premature switch-off. Using the
    # crossover bin directly gives the first bin whose median crosses the
    # base threshold.)
    IF len(cum_fractions_by_bin[switchoff_bin]) > 0:
        switchoff_threshold = median(cum_fractions_by_bin[switchoff_bin])
    ELSE:
        switchoff_threshold = base_switchoff_threshold

    # Clamp to reasonable range
    switchoff_threshold = clamp(switchoff_threshold, 0.70, 0.95)

    RETURN (deviation_limit, switchoff_threshold)
```

**References:** Satish et al. 2014, p.24: "self-updating deviation limits and
switch-off parameters" — attributed to Humphery-Jenner (2011). The specific
adaptation mechanism (95th percentile of surprise magnitudes for deviation
limit; median cumulative volume for switch-off) is entirely Researcher
inference. The paper does not describe how these parameters self-update;
Humphery-Jenner (2011) should be consulted for the original specification.
The base values of 10% (deviation) and 80% (switch-off) are from Satish et al.
2014, p.24, describing Humphery-Jenner (2011) model attributes.

---

#### Function 10: train_full_model

Orchestrates the full training pipeline for one stock.

```
FUNCTION train_full_model(
        volume_history,             # complete [day, bin] array for stock
        train_end_date,             # last date in training window
        params                      # all hyperparameters (see Parameters section)
    ) -> model_state

    # Volume history slicing: need enough data for all components.
    # The longest lookback is max(N_seasonal, N_interday_fit) + N_weight_train
    # + N_hist (for rolling H computation during weight training).

    # Step 1: Compute seasonal factors (6-month window ending at train_end_date)
    vol_hist_seasonal = volume_history[train_end_date - N_seasonal + 1 : train_end_date + 1]
    seasonal_factors = compute_seasonal_factors(vol_hist_seasonal, params.N_seasonal)

    # Step 2: Compute historical average (N_hist-day window ending at train_end_date)
    vol_hist_short = volume_history[train_end_date - N_hist + 1 : train_end_date + 1]
    hist_avg = compute_historical_average(vol_hist_short, params.N_hist)

    # Step 3: Fit inter-day ARMA models (returns models AND term counts)
    vol_hist_interday = volume_history[train_end_date - N_interday_fit + 1 : train_end_date + 1]
    (interday_models, interday_term_counts) = fit_interday_arma(
        vol_hist_interday, params.N_interday_fit, params.p_max, params.q_max)

    # Step 4: Fit intraday ARMA model (uses inter-day term counts for joint budget)
    vol_hist_intraday = volume_history[train_end_date - N_intraday_fit + 1 : train_end_date + 1]
    intraday_model = fit_intraday_arma(vol_hist_intraday, seasonal_factors,
                                        params.N_intraday_fit,
                                        params.p_max_intra, params.q_max_intra,
                                        interday_term_counts)

    # Step 5: Optimize regime weights
    # Need extended history: N_weight_train days, plus N_hist days before that
    # for computing rolling H on the first training day.
    vol_hist_weights = volume_history[
        train_end_date - N_weight_train - N_hist + 1 : train_end_date + 1]
    (regime_thresholds, regime_weights) = optimize_regime_weights(
        vol_hist_weights, hist_avg, interday_models, intraday_model,
        seasonal_factors, params.N_hist, params.N_weight_train,
        params.N_regimes, params.N_regime_valid)

    # Step 6: Compute historical percentages for Model B
    vol_hist_pct = volume_history[train_end_date - N_hist_pct + 1 : train_end_date + 1]
    hist_pct = compute_historical_percentages(vol_hist_pct, params.N_hist_pct)

    # Step 7: Fit surprise regression for Model B
    # Need N_surprise_train days + N_hist_pct days of lookback
    vol_hist_surprise = volume_history[
        train_end_date - N_surprise_train - N_hist_pct + 1 : train_end_date + 1]

    # For sophisticated variant: generate Model A forecasts for training days
    model_a_forecasts = None
    IF params.use_model_a_baseline:
        model_a_forecasts = generate_model_a_training_forecasts(
            volume_history, interday_models, intraday_model, seasonal_factors,
            hist_avg, regime_thresholds, regime_weights, params,
            train_end_date, params.N_surprise_train)

    regression_coefficients = fit_surprise_regression(
        vol_hist_surprise, params.N_hist_pct,
        params.N_surprise_train, params.N_surprise_lags,
        model_a_forecasts=model_a_forecasts)

    # Step 8: Calibrate adaptive deviation limits and switch-off
    # Use the same calibration window as surprise training
    hist_pct_series = {}
    FOR d IN trailing N_calibration days ending at train_end_date:
        hist_pct_series[d] = compute_historical_percentages(
            volume_history[d - N_hist_pct : d], params.N_hist_pct)
    (deviation_limit, switchoff_threshold) = calibrate_adaptive_limits(
        vol_hist_surprise, hist_pct_series, params.N_calibration,
        params.base_deviation_limit, params.base_switchoff_threshold)

    RETURN ModelState(
        seasonal_factors=seasonal_factors,
        hist_avg=hist_avg,
        interday_models=interday_models,
        interday_term_counts=interday_term_counts,
        intraday_model=intraday_model,
        regime_thresholds=regime_thresholds,
        regime_weights=regime_weights,
        hist_pct=hist_pct,
        regression_coefficients=regression_coefficients,
        deviation_limit=deviation_limit,
        switchoff_threshold=switchoff_threshold,
        train_end_date=train_end_date,
        params=params
    )


# ---- Helper: generate_model_a_training_forecasts ----
FUNCTION generate_model_a_training_forecasts(
        volume_history, interday_models, intraday_model, seasonal_factors,
        hist_avg, regime_thresholds, regime_weights, params,
        train_end_date, N_surprise_train
    ) -> model_a_forecasts[day, bin]
    # Generates Model A's raw volume forecasts for each training day and bin.
    # Uses the same approach as Function 5 (in-sample fitted values).
    # Returns a [day, bin] array where model_a_forecasts[d, i] is Model A's
    # forecast for day d, bin i.
    #
    # For each (day, bin), the regime is reconstructed from historical
    # cumulative volume percentiles, matching the live-prediction regime
    # assignment. This avoids using a fixed median regime for all bins,
    # which would create a systematic training/prediction mismatch.
    #
    # To convert to percentage baseline for Model B surprises:
    #   baseline_pct[d, i] = model_a_forecasts[d, i] / sum(model_a_forecasts[d, :])
    #
    # This is Researcher inference — the paper does not specify the exact
    # mechanism for using Model A as the surprise baseline.

    training_days = trailing N_surprise_train days ending at train_end_date
    forecasts = {}

    FOR d IN training_days:
        H_d = compute_historical_average(volume_history[d-N_hist:d], params.N_hist)
        D_d = [interday_models[i].predict_at(d) IF interday_models[i] IS NOT FALLBACK
               ELSE H_d[i] FOR i = 1 TO I]
        deseas_today = [volume_history[d, i] / seasonal_factors[i] FOR i = 1 TO I]

        FOR i = 1 TO I:
            IF intraday_model IS FALLBACK:
                A_i = H_d[i]
            ELSE:
                IF i == 1:
                    state = intraday_model.make_state([])
                ELSE:
                    state = intraday_model.make_state(deseas_today[1:i])
                A_i = intraday_model.predict_from_state(state, 1)[0] * seasonal_factors[i]

            # Reconstruct the regime that would have been assigned in live
            # prediction for this (day, bin) pair. During training, historical
            # cumulative volumes are available, so we can compute the volume
            # percentile and assign the correct regime rather than defaulting
            # to the median. This mirrors Function 5's approach and avoids a
            # systematic training/prediction mismatch where all training-time
            # Model A forecasts use median-regime weights while live forecasts
            # use the actual regime.
            IF i == 1:
                # No prior bins observed; default to median percentile
                regime = assign_regime(0.5, regime_thresholds)
            ELSE:
                cum_vol_d = sum(volume_history[d, 1:i])
                # Compute percentile vs trailing N_hist days
                hist_cum_vols = [sum(volume_history[dd, 1:i])
                                 FOR dd IN trailing N_hist days before d]
                pctile = percentile_rank(cum_vol_d, hist_cum_vols)
                regime = assign_regime(pctile, regime_thresholds)
            w = regime_weights[regime]
            forecasts[d, i] = w[0] * H_d[i] + w[1] * D_d[i] + w[2] * A_i

    RETURN forecasts
```

**References:** The training sequence follows the component dependency order
from Satish et al. 2014, pp.17-19. The specific orchestration logic is
Researcher inference — the paper describes each component but does not provide
a combined training procedure. The sophisticated variant integration
(generate_model_a_training_forecasts) is Researcher inference based on the
paper's statement that "we could apply our more extensive volume forecasting
model described previously as the base model from which to compute volume
surprises" (p.19).

---

#### Function 11: run_daily_prediction

Orchestrates prediction for one stock on one trading day.

```
FUNCTION run_daily_prediction(
        model_state,                # from train_full_model
        today_date,                 # current trading day
        volume_history              # includes today's data as it arrives
    ) -> (raw_forecasts[1..I], pct_forecasts[1..I])

    # ---- Pre-market (before bin 1): ----

    # Update hist_avg and hist_pct for today using latest N_hist / N_hist_pct days
    hist_avg_today = compute_historical_average(
        volume_history[today_date - N_hist : today_date], model_state.params.N_hist)
    hist_pct_today = compute_historical_percentages(
        volume_history[today_date - N_hist_pct : today_date], model_state.params.N_hist_pct)

    # Update seasonal factors (recompute or use cached if recent enough)
    seasonal_factors = model_state.seasonal_factors   # typically stable; recompute weekly

    raw_forecasts = array of length I
    pct_forecasts = array of length I

    # Initialize pct_forecasts[1] = hist_pct[1] since Model B cannot forecast
    # bin 1 (needs at least one observed bin for surprise computation).
    pct_forecasts[1] = hist_pct_today[1]

    # Prepare Model A-based baseline for sophisticated variant (if enabled).
    # IMPORTANT: The baseline must use Model A's PRE-OBSERVATION forecasts
    # (what Model A would have predicted before seeing any actual volume today),
    # NOT the actual observed volumes for completed bins. This matches the
    # training procedure (generate_model_a_training_forecasts), where surprises
    # are computed as actual_pct - model_a_forecast_pct for ALL bins. If we
    # used actual volumes for observed bins, the surprise for those bins would
    # reduce to observed * (1/est_total - 1/forecast_total), which captures
    # only the total-volume estimation error rather than the bin-level
    # prediction error that the regression was trained on.
    model_a_baseline_pct = None

    IF model_state.params.use_model_a_baseline:
        # Compute Model A's pre-observation forecasts for ALL bins at the
        # start of the day (current_bin = 0, no observations yet).
        pre_obs_raw_forecasts = array of length I
        FOR target_bin = 1 TO I:
            pre_obs_raw_forecasts[target_bin] = predict_raw_volume(
                target_bin, [], 0,   # empty observations, current_bin=0
                hist_avg_today, model_state.interday_models,
                model_state.intraday_model, seasonal_factors,
                model_state.regime_thresholds, model_state.regime_weights,
                volume_history_for_pctile=volume_history[today_date - N_hist : today_date])
        total_pre_obs = sum(pre_obs_raw_forecasts[1 : I+1])
        IF total_pre_obs > 0:
            model_a_baseline_pct = [pre_obs_raw_forecasts[j] / total_pre_obs
                                    FOR j = 1 TO I]
        # This baseline is fixed for the entire day — it represents Model A's
        # "prior" before any intraday observations. It is NOT updated as bins
        # complete, ensuring consistency with the training-time surprise
        # computation in generate_model_a_training_forecasts.
        #
        # KNOWN TRADE-OFF: This pre-observation approach creates a secondary
        # context mismatch for later bins. During training, the Model A forecast
        # for bin i uses intraday ARMA context from bins 1..i-1 and a regime
        # assignment based on cumulative volume through bin i-1. Here, ALL
        # forecasts use context-free predictions (empty observations, median
        # regime). For early bins (i=1,2), this difference is negligible. For
        # later bins (i=20+), the live baseline is less accurate than the
        # training baseline, producing systematically larger surprise magnitudes.
        # The deviation clamp bounds the practical impact, and this direction
        # (larger live surprises) is far less harmful than using actual volumes
        # (which would produce near-zero surprises and nullify Model B).
        #
        # A potential refinement is to build the baseline iteratively during
        # the day: at each bin, compute Model A's forecast using observations
        # from bins 1..current_bin-1 (BEFORE observing the current bin's actual
        # volume), storing each forecast as it is produced. This would match
        # the training procedure exactly but adds complexity. See Known
        # Limitation #11. (Researcher inference.)

    # ---- Intraday: iterate through bins ----
    observed_volumes = []

    FOR current_bin = 0 TO I-1:
        # Produce raw volume forecasts for all remaining bins
        FOR target_bin = current_bin + 1 TO I:
            raw_forecasts[target_bin] = predict_raw_volume(
                target_bin, observed_volumes, current_bin,
                hist_avg_today, model_state.interday_models,
                model_state.intraday_model, seasonal_factors,
                model_state.regime_thresholds, model_state.regime_weights,
                volume_history_for_pctile=volume_history[today_date - N_hist : today_date])

        # Produce volume percentage forecast for next bin
        IF current_bin >= 1 AND current_bin < I:
            next_bin = current_bin + 1

            pct_forecasts[next_bin] = predict_volume_percentage(
                next_bin, observed_volumes, current_bin,
                hist_pct_today, model_state.regression_coefficients,
                model_state.deviation_limit,
                model_state.switchoff_threshold,
                baseline_pct=model_a_baseline_pct)

        # Wait for bin to complete, then observe actual volume
        actual_volume = observe_volume(today_date, current_bin + 1)
        observed_volumes.append(actual_volume)

        # Update inter-day models (append observation for today)
        # This happens at end of day (see below)

    # ---- End-of-day: ----
    # Append today's volumes to inter-day ARMA models for next-day prediction
    FOR i = 1 TO I:
        IF model_state.interday_models[i] IS NOT FALLBACK:
            model_state.interday_models[i].append_observation(observed_volumes[i])

    RETURN (raw_forecasts, pct_forecasts)
```

**References:** The intraday prediction loop follows from Satish et al. 2014,
p.15 ("forecasting intraday volume and volume percentages for fixed intervals
of time from the present moment"), p.18 ("dynamically apply the appropriate
weights intraday"). The end-of-day ARMA update is Researcher inference —
the paper says "rolling basis" which implies the model state advances daily.
The entire orchestration structure is Researcher inference.

The pct_forecasts[1] = hist_pct_today[1] initialization handles the edge case
where Model B cannot produce a forecast for the first bin (no prior observed
bins for surprise computation). This is Researcher inference.

The sophisticated variant live-prediction mechanism (computing Model A's
pre-observation forecasts once at start of day and using them as a fixed
baseline throughout the day) is Researcher inference — the paper states the
option of using Model A forecasts as the surprise baseline (p.19) but does not
specify the real-time integration procedure. The pre-observation approach
ensures consistency with the training procedure: during training,
generate_model_a_training_forecasts computes Model A forecasts for each
(day, bin) without using that day's actual volumes, so surprises capture
genuine bin-level prediction error. The live baseline must match this to
avoid a train/predict mismatch. A secondary context mismatch remains (see
inline comment and Known Limitation #11): the pre-observation baseline uses
context-free forecasts for all bins, while training forecasts benefit from
intraday ARMA conditioning. This is bounded by the deviation clamp and is
a pragmatic trade-off for simplicity.

---

#### Function 12: compute_evaluation_metrics

Computes MAPE for raw volume and mean absolute error for volume percentages.

```
FUNCTION compute_evaluation_metrics(
        actual_volumes,             # [day, bin] array for evaluation period
        predicted_volumes,          # [day, bin] array from Model A
        actual_pcts,                # [day, bin] actual percentages
        predicted_pcts              # [day, bin] from Model B
    ) -> (mape_raw, mae_pct)

    # MAPE for raw volume (Satish et al. 2014, p.17)
    # MAPE = (100% / N) * SUM_i |predicted - actual| / actual
    N_raw = 0
    sum_ape = 0.0
    FOR d, i IN actual_volumes:
        IF actual_volumes[d, i] > 0:
            sum_ape += abs(predicted_volumes[d, i] - actual_volumes[d, i]) / actual_volumes[d, i]
            N_raw += 1
    mape_raw = 100.0 * sum_ape / N_raw IF N_raw > 0 ELSE NaN

    # Mean absolute error for volume percentages (Satish et al. 2014, p.17)
    # Error = (1/N) * SUM_i |predicted_pct - actual_pct|
    N_pct = 0
    sum_ae = 0.0
    FOR d, i IN actual_pcts:
        sum_ae += abs(predicted_pcts[d, i] - actual_pcts[d, i])
        N_pct += 1
    mae_pct = sum_ae / N_pct IF N_pct > 0 ELSE NaN

    RETURN (mape_raw, mae_pct)
```

**References:** Satish et al. 2014, p.17 (MAPE definition: "100% x (1/N) x
SUM |Predicted_Volume - Raw_Volume| / Raw_Volume"), p.17 (percentage error:
"(1/N) x SUM |Predicted_Percentage - Actual_Percentage|"). The paper
explicitly notes that percentage predictions are "already normalized, since
they sum to 100" so no denominator normalization is needed for the percentage
metric.

---

### Data Flow

```
Input: volume_history[stock, day, bin] — raw share counts

Training Phase:
  volume_history ──┬── Function 1 ──> seasonal_factors[1..I]
                   ├── Function 2 ──> hist_avg[1..I]
                   ├── Function 3 ──> (interday_models[1..I], interday_term_counts[1..I])
                   ├── Function 4 ──> intraday_model
                   │   (uses seasonal_factors AND interday_term_counts for joint budget)
                   ├── Function 5 ──> (regime_thresholds, regime_weights)
                   │   (uses hist_avg, interday_models, intraday_model, seasonal_factors)
                   ├── Function 7 ──> hist_pct[1..I]
                   ├── Function 8 ──> regression_coefficients[1..N_surprise_lags]
                   │   (uses hist_pct via Function 7; optionally Model A forecasts)
                   └── Function 9a ──> (deviation_limit, switchoff_threshold)
                       (uses hist_pct_series for adaptive calibration)

Prediction Phase (per bin):
  observed_volumes + model_state ──┬── Function 6 ──> V_hat (raw volume)
                                   └── Function 9 ──> pct_hat (volume %)
                                       (optionally uses V_hat for sophisticated baseline)
```

**Shapes and Types:**

| Variable | Shape | Type | Description |
|----------|-------|------|-------------|
| volume_history | [D, I] | float | D days x I=26 bins, raw share counts |
| seasonal_factors | [I] | float | 26 bin averages (6-month) |
| hist_avg | [I] | float | 26 bin averages (N_hist-day) |
| interday_models | [I] | ARMA or FALLBACK | 26 fitted models |
| interday_term_counts | [I] | int | 26 term counts (0 for FALLBACK) |
| intraday_model | 1 | ARMA or FALLBACK | single fitted model |
| regime_thresholds | [N_regimes-1] | float in (0,1) | percentile cutoffs |
| regime_weights | [N_regimes, 3] | float in [0,1], sum=1 | per-regime [w_H, w_D, w_A] |
| hist_pct | [I] | float in [0,1] | 26 historical volume fractions |
| regression_coefficients | [N_surprise_lags] | float | surprise regression betas |
| deviation_limit | scalar | float > 0 | adaptive per-stock deviation clamp |
| switchoff_threshold | scalar | float in [0.70, 0.95] | adaptive per-stock switch-off |
| observed_volumes | [current_bin] | float | today's volumes so far |
| V_hat | scalar | float >= 0 | raw volume forecast (shares) |
| pct_hat | scalar | float >= 0 | volume fraction forecast |

### Variants

This specification implements the full dual-model system from Satish et al.
(2014): Model A with all three signal components + regime-switching weights,
and Model B with the surprise regression. This is the most complete variant
described in the paper and yields the best reported results.

Two sub-variants of Model B are specified:
1. **Naive variant** (default): surprises computed relative to historical
   average percentages (hist_pct). Simpler and self-contained.
2. **Sophisticated variant** (use_model_a_baseline=True): surprises computed
   relative to Model A's raw forecasts converted to percentages. Potentially
   more accurate since the baseline is already improved over the naive
   historical average. (Satish et al. 2014, p.19.)

Simpler variants of Model A could omit the intraday ARMA (using only H + D
with regime weights) or omit regime switching entirely (using a single weight
set), but these are strictly inferior per the paper's results. The full model
is selected because:
1. It achieves the best MAPE reduction (24% median, Exhibit 6).
2. All components are well-specified in the paper.
3. The regime-switching overlay is the key differentiator over simpler baselines.

---

## Parameters

| Parameter | Description | Recommended Value | Sensitivity | Range |
|-----------|-------------|-------------------|-------------|-------|
| I | Bins per trading day | 26 | High — determines forecast granularity | {13, 26, 78} |
| N_hist | Historical average window (Component 1) | 21 days | Medium — too short is noisy, too long misses trends | 10-63 |
| N_seasonal | Deseasonalization window | 126 days (~6 months) | Low — stable seasonal patterns | 63-252 |
| N_interday_fit | Inter-day ARMA fitting window | 252 days (~1 year) | Medium — needs enough for stable ARMA | 126-504 |
| N_intraday_fit | Intraday ARMA fitting window | 21 days (~1 month) | Medium — paper says "rolling basis over the most recent month" | 15-42 |
| p_max | Max AR order for inter-day ARMA | 5 | Low — AICc handles selection | 3-7 |
| q_max | Max MA order for inter-day ARMA | 5 | Low — AICc handles selection | 3-7 |
| p_max_intra | Max AR order for intraday ARMA | 4 | Medium — paper says "lags less than five" | 2-5 |
| q_max_intra | Max MA order for intraday ARMA | 4 | Medium — constrained by joint dual-ARMA budget | 2-5 |
| max_dual_arma_terms | Max combined terms in dual ARMA system | 10 | Medium — paper says "fewer than 11" | 8-12 |
| N_weight_train | Weight optimization training window | 252 days (~1 year) | Medium | 126-504 |
| N_regime_valid | Validation days for regime count selection | 63 days (~3 months) | Low | 42-126 |
| N_regimes | Number of regime buckets (selected by CV) | 3 (typical) | High — too many overfits, too few misses regime shifts | 1-5 |
| N_optimizer_restarts | Random restarts for weight optimizer | 5 | Low | 3-10 |
| N_hist_pct | Historical window for percentage curve | 21 days | Medium | 10-63 |
| N_surprise_train | Training window for surprise regression | 252 days | Medium | 126-504 |
| N_surprise_lags | Number of lagged surprise terms | 5 | Medium — fixed globally per universe; paper says "optimal number" found for U.S. equities but not disclosed | 1-10 |
| base_deviation_limit | Base deviation limit (before adaptation) | 0.10 (10%) | Medium — safety constraint | 0.05-0.20 |
| base_switchoff_threshold | Base cumulative fraction for switch-off | 0.80 (80%) | Medium — safety constraint | 0.70-0.90 |
| N_calibration | Days for adaptive limit calibration | 126 (~6 months) | Low | 63-252 |
| use_model_a_baseline | Use Model A for surprise baseline (sophisticated) | False | Medium — set True for coupled mode | {True, False} |
| re_estimation_frequency | How often to re-train full model | Monthly | Low — paper says "rolling basis" | Weekly to Quarterly |
| daily_update_components | Components updated daily vs. at re-estimation | hist_avg, hist_pct, ARMA state | N/A | N/A |

### Initialization

**Model A:**
- seasonal_factors: computed from the trailing 126 trading days of volume data.
- hist_avg: computed from the trailing 21 trading days.
- interday_models: fitted on trailing ~252 days; initialized with full MLE.
  If fitting fails for any bin, that bin uses FALLBACK (hist_avg substitution).
  Term counts recorded for each bin.
- intraday_model: fitted on trailing ~21 days of deseasonalized data, subject
  to the joint dual-ARMA term budget (max_dual_arma_terms minus the maximum
  inter-day term count across all bins).
- regime_thresholds and regime_weights: computed by Function 5. Weights are
  non-negative and sum to 1 per regime (softmax parameterization).
- On the first prediction day, all ARMA models have their state initialized
  from the training data (no cold-start gap).

**Model B:**
- hist_pct: computed from trailing 21 days of percentage data.
- regression_coefficients: fitted on trailing ~252 days of surprise data.
- deviation_limit and switchoff_threshold: calibrated per stock via
  Function 9a using the trailing N_calibration days.
- On the first prediction day, if fewer than N_surprise_lags bins have been
  observed, the feature vector is zero-padded.
- pct_forecasts[1] is initialized to hist_pct[1] (Model B cannot forecast
  bin 1 since no prior bins have been observed).

### Calibration

**Per-stock calibration procedure:**

1. Choose a historical dataset of at least 504 trading days (~2 years).
   The first year is in-sample (training), the second year is out-of-sample.
   (Satish et al. 2014, pp.19-20: "two years of TAQ data; out-of-sample
   results reported on the final year.")

2. For each out-of-sample day, train the model using all data up to (but
   not including) that day, with rolling windows as specified by the
   parameter values.

3. **Regime count selection:** Use expanding-window cross-validation within
   the weight training window. For each candidate N_regimes in {1,2,3,4,5},
   hold out the last N_regime_valid days of the training window as validation,
   fit weights on the remainder, and evaluate MAPE on the held-out portion.
   Select the N_regimes with the lowest validation MAPE. Then re-fit weights
   on the full training window with the chosen N_regimes.

4. **Surprise lag count:** N_surprise_lags is a fixed hyperparameter
   (recommended: 5), not selected per stock. The paper states "we were able
   to identify the optimal number of model terms for U.S. stocks" (Satish
   et al. 2014, p.19), implying a single value was chosen globally across
   the universe rather than per-stock cross-validation. If a developer
   wishes to tune this value, an offline study across a representative stock
   panel using expanding-window cross-validation (fit regression on earlier
   portion, evaluate MAE on held-out portion, for each candidate in
   {1,...,10}) is recommended, but this is a one-time exercise rather than
   part of the per-stock calibration loop. (Researcher inference — the
   paper does not disclose the chosen value or the selection method.)

5. **Adaptive limit calibration:** Run Function 9a on the trailing
   N_calibration days to set per-stock deviation_limit and
   switchoff_threshold.

6. **Evaluation:** Compute MAPE for Model A and MAE for Model B over the full
   out-of-sample period using Function 12.

7. **Re-estimation schedule:** In production, re-estimate ARMA models and
   weights monthly. Update hist_avg, hist_pct, and ARMA observation states
   daily. Seasonal factors and adaptive limits can be updated weekly or
   monthly (they change slowly). (Satish et al. 2014, p.17: "We compute this
   model on a rolling basis over the most recent month.")

---

## Validation

### Expected Behavior

On typical liquid U.S. equities:

1. **Model A MAPE reduction:** Expect ~24% median MAPE reduction vs.
   historical-window-only baseline across all bins. Bottom 95% average
   reduction ~29%. (Satish et al. 2014, p.20, Exhibit 6.)

2. **Model A MAPE by bin:** Error reduction increases through the day (from
   ~10% for early bins to ~30% for late bins), as intraday ARMA gains more
   conditioning information. (Satish et al. 2014, Exhibit 6.)

3. **Model B MAE reduction:** Expect ~7.55% median reduction in absolute
   volume percentage error vs. historical curve (0.00874 -> 0.00808 for
   15-minute bins). (Satish et al. 2014, Exhibit 9.)

4. **VWAP tracking error:** If used to drive a VWAP execution algorithm,
   expect ~9.1% reduction in mean VWAP tracking error (9.62 -> 8.74 bps).
   (Satish et al. 2014, Exhibit 10.)

5. **Cross-sectional consistency:** Improvements should be consistent across
   industry groups (SIC codes) and beta deciles. (Satish et al. 2014,
   Exhibits 7 and 8.)

### Sanity Checks

1. **Model A historical-only baseline:** With all regime weights set to
   [1, 0, 0] (H only), Model A should reproduce the simple rolling average
   baseline exactly: V_hat = H for every bin. No direct absolute MAPE
   benchmark is available from the paper (Exhibit 6 reports reduction
   percentages, not absolute MAPE values). Verify that V_hat matches H for
   all bins when weights are [1, 0, 0].

2. **Model B historical-only baseline:** With all regression coefficients
   set to zero, Model B should reproduce hist_pct (scaled for remaining
   fraction). The Exhibit 9 "HVWAP" column (0.00874 for 15-minute bins)
   is the correct benchmark for this check — it represents the median
   volume percentage error of the historical-only approach.

3. **Seasonal factors positive:** All seasonal_factors[i] > 0 after the
   zero-floor guard.

4. **Volume percentages sum to ~1:** hist_pct[1] + ... + hist_pct[I] should
   be approximately 1.0 (within floating-point tolerance).

5. **Deseasonalized mean ~1:** The mean of deseasonalized intraday values
   should be approximately 1.0 by construction (each bin divided by its
   mean over the deseasonalization window).

6. **ARMA stationarity:** All fitted ARMA models should have AR roots outside
   the unit circle. The enforce_stationarity flag ensures this.

7. **Regime assignment deterministic:** For the same cumulative volume and
   historical data, assign_regime should always return the same regime.

8. **Deviation clamp active:** On a stock with stable volume patterns,
   |delta| should rarely approach deviation_limit. Typical delta values
   should be in [-0.03, +0.03] for liquid stocks.

9. **Switch-off fires:** For a stock that reaches 80% of daily volume by
   bin ~21 (typical for U.S. equities), the switch-off should activate
   for the last ~5 bins.

10. **No-intercept mean-zero check:** At training time, the mean of all
    surprise values in the regression (naive variant) should be approximately
    zero (since hist_pct is the mean of actual_pct over the training window).

11. **Weight sum-to-1:** All optimized weights w_H + w_D + w_A = 1.0
    (guaranteed by softmax parameterization, within floating-point tolerance).

12. **Fallback equivalence:** When all ARMA models are FALLBACK, Model A
    should degenerate to a simple weighted historical average (effectively
    V_hat = w_H * H + w_D * H + w_A * H = (w_H + w_D + w_A) * H = H,
    since weights sum to 1).

13. **Monotonic improvement:** Adding the inter-day ARMA (H+D) should improve
    over H alone; adding the intraday ARMA (H+D+A) should improve further.
    If not, check ARMA fitting or weight optimization.

14. **Percentage forecast non-negative:** pct_hat >= 0 for all bins
    (guaranteed by the floor in Function 9).

15. **Joint term constraint:** For every bin i, verify that
    interday_term_counts[i] + intraday_model.term_count <= max_dual_arma_terms.
    This enforces the paper's "fewer than 11 terms" constraint (Satish et al.
    2014, p.18 para 1), since max_dual_arma_terms = 10.

16. **Bin 1 percentage:** pct_forecasts[1] should always equal
    hist_pct_today[1] (set during initialization, since Model B cannot
    forecast bin 1).

### Edge Cases

1. **Zero-volume bins:** Can occur for illiquid stocks. Guard with zero-floor
   in seasonal factors and historical averages. MAPE is undefined when
   actual = 0; skip these observations in evaluation.

2. **First bin of the day (bin 1):** No prior intraday observations.
   Intraday ARMA uses unconditional forecast. Model B cannot produce a
   percentage forecast for bin 1 (needs at least one observed bin for
   surprise computation). pct_forecasts[1] is explicitly set to
   hist_pct[1] before the main loop in Function 11.

3. **ARMA convergence failure:** If MLE fails for all (p,q) combinations for
   a given bin, use FALLBACK. The prediction function substitutes hist_avg.

4. **New stock / insufficient history:** If fewer than N_seasonal days of
   history are available, use whatever history exists. If fewer than 10 days,
   fall back to a pure historical average model (no ARMA, no regime switching).

5. **Extreme volume days:** Corporate events, index rebalances, etc. can
   produce outlier volumes. The regime-switching mechanism partially handles
   this (high-volume regime has different weights), but extreme outliers may
   still produce large forecast errors. Consider winsorizing the ARMA input
   series at the 1st/99th percentile.

6. **Half-trading days:** The market closes early on some days (e.g., day
   before Thanksgiving, 1:00 PM close). On these days, I is reduced.
   Either skip these days or use a separate set of seasonal factors computed
   only from half-days.

7. **Stock splits / corporate actions:** Volume data must be split-adjusted.
   A failure to adjust will produce discontinuities in the historical average
   and ARMA series.

8. **Late-day switch-off boundary:** When cum_fraction is exactly at the
   switchoff_threshold, the switch-off activates. This is a discrete
   transition; no smoothing is applied.

9. **Single-regime result:** If cross-validation selects N_regimes = 1, the
   model degenerates to a single weight set with no regime switching. This is
   valid and expected for stocks with stable volume patterns.

10. **Empty regime bucket:** If no training observations fall in a particular
    regime, use equal weights [1/3, 1/3, 1/3] as default.

11. **Intraday ARMA multi-step degradation:** For target bins far ahead of
    the current bin, multi-step ARMA forecasts converge to the unconditional
    mean. The model relies more heavily on H and D for distant bins, which
    the regime weights should learn to accommodate.

12. **Joint term budget exhausted:** If max(interday_term_counts) >=
    max_dual_arma_terms, the intraday ARMA cannot be fitted at all. In this
    case, Function 4 returns FALLBACK and the intraday component falls back
    to hist_avg. This is expected to be rare since typical inter-day models
    use ARMA(1,1) or ARMA(2,1) = 3-4 terms, leaving 6-7 terms for the
    intraday model.

### Known Limitations

1. **Single-stock model:** No cross-sectional information is used. A stock
   that suddenly changes character (e.g., due to index inclusion) will have
   stale model parameters until re-estimation.

2. **No exogenous variables:** The model does not incorporate news events,
   earnings calendars, option expiry, or macroeconomic releases. Satish
   et al. (2014, p.18) recommend custom curves for special days rather than
   ARMAX models.

3. **Proprietary parameters:** The paper does not disclose the exact values
   of regime thresholds, weight coefficients, or the optimal number of
   surprise regression terms for U.S. equities. These must be rediscovered
   via calibration.

4. **Stale intraday ARMA:** The intraday ARMA is fitted on the most recent
   month. If the stock's intraday pattern changes (e.g., due to a
   fundamental shift), the model will lag until the next re-estimation.

5. **Percentage coherence:** Model B produces one-bin-ahead forecasts
   sequentially. There is no mechanism to ensure that the sum of all
   predicted bin percentages equals 1.0 over the full day. Each forecast
   is conditioned only on past observed bins, not on future bin forecasts.

6. **Volume denominator for surprises:** The surprise computation requires
   estimating total daily volume (to convert raw volumes to percentages)
   before the day is complete. The remaining-fraction scaling is an
   approximation that becomes more accurate as the day progresses.

7. **Regime assignment at day start:** Before any bins are observed
   (current_bin = 0), the regime assignment defaults to the median
   percentile. This is a heuristic; the model has no information about
   today's volume level until the first bin completes.

8. **MAPE sensitivity to small volumes:** MAPE gives disproportionate weight
   to low-volume bins (since dividing by a small actual volume amplifies the
   error). The paper uses bottom 95% statistics to mitigate this, but the
   weight optimization still uses full MAPE and may over-optimize for
   low-volume regimes.

9. **Humphery-Jenner details unavailable:** The self-updating mechanism for
   deviation limits and switch-off parameters is attributed to Humphery-Jenner
   (2011) but that paper was not available for detailed review. The adaptive
   calibration in Function 9a is Researcher inference that approximates the
   intent described in Satish et al. 2014, p.24. Consulting the original
   Humphery-Jenner (2011) paper would improve the fidelity of this component.

10. **Weight training look-ahead:** The weight optimization in Function 5
    uses in-sample ARMA predictions (the ARMA models are fitted on data that
    includes the weight training period). This produces slightly optimistic
    "predictions" during weight training. See the detailed note in Function 5.

11. **Sophisticated variant baseline context mismatch:** When
    use_model_a_baseline=True, the live-prediction baseline computes all
    Model A forecasts with context-free inputs (current_bin=0, no
    observations, median regime), while the training baseline
    (generate_model_a_training_forecasts) computes each bin's forecast with
    intraday ARMA context from prior bins and an accurate regime assignment.
    For later bins (i=20+), this causes systematically larger surprise
    magnitudes during live prediction than during training, because the
    context-free forecast is less accurate. The deviation clamp bounds the
    practical impact. A potential refinement is to compute the live baseline
    iteratively (forecasting each bin using only observations from earlier
    bins, before the current bin's actual volume is observed), which would
    match the training procedure exactly but add implementation complexity.
    (Researcher inference — this trade-off was identified during adversarial
    review and is not discussed in the paper.)

---

## Paper References

| Spec Section | Paper Source |
|-------------|-------------|
| Function 1 (seasonal factors) | Satish et al. 2014, p.17 para 5 |
| Function 2 (historical average) | Satish et al. 2014, p.17 para 1, p.16, Exhibit 1 |
| Function 3 (inter-day ARMA) | Satish et al. 2014, p.17 para 2 |
| Function 4 (intraday ARMA) | Satish et al. 2014, p.17 paras 3-5, p.18 para 1 |
| Function 4 (joint term constraint) | Satish et al. 2014, p.18 para 1 ("dual ARMA model having fewer than 11 terms") |
| Function 5 (regime weights) | Satish et al. 2014, p.18 para 1 |
| Function 6 (raw prediction) | Satish et al. 2014, pp.17-18 |
| Function 7 (historical pct) | Satish et al. 2014, p.16, p.18 para 4 |
| Function 8 (surprise regression) | Satish et al. 2014, pp.18-19; Humphery-Jenner 2011 |
| Function 8 (sophisticated variant) | Satish et al. 2014, p.19 ("apply our more extensive volume forecasting model") |
| Function 9 (pct prediction) | Satish et al. 2014, pp.18-19; Humphery-Jenner 2011 |
| Function 9a (adaptive limits) | Satish et al. 2014, p.24, describing Humphery-Jenner (2011) model attributes |
| AICc formula | Hurvich and Tsai 1989, 1993 (cited in Satish et al. 2014, p.17) |
| MAPE definition | Satish et al. 2014, p.17 |
| MAE (pct) definition | Satish et al. 2014, p.17 |
| Deviation limit base (10%) | Satish et al. 2014, p.24 (referencing Humphery-Jenner 2011) |
| Switch-off base (80%) | Satish et al. 2014, p.24 (referencing Humphery-Jenner 2011) |
| Exhibit 1 (model diagram) | Satish et al. 2014, p.18 |
| Exhibit 1 ("Prior 5 days" = p_max) | Satish et al. 2014, p.18 Exhibit 1, reconciled with p.17 para 2 |
| Exhibit 1 ("4 Bins Prior" = p_max_intra) | Satish et al. 2014, p.18 Exhibit 1, reconciled with p.17 para 4 |
| Exhibit 6 (MAPE reduction by bin) | Satish et al. 2014, p.22 |
| Exhibit 9 (pct results) | Satish et al. 2014, p.23 |
| Exhibit 10 (VWAP simulation) | Satish et al. 2014, p.23 |
| Exhibits 7-8 (SIC/beta) | Satish et al. 2014, p.22 |
| "Four components" framing | Satish et al. 2014, p.17 ("four components, see Exhibit 1") |
| 21-day historical window | Satish et al. 2014, Exhibit 1 ("Prior 21 days") |
| "fewer than 11 terms" | Satish et al. 2014, p.18 para 1 |
| "lags less than five" | Satish et al. 2014, p.17 para 4 |
| 6-month deseasonalization | Satish et al. 2014, p.17 para 5 |
| 1-month intraday fit window | Satish et al. 2014, p.18 para 1 |
| Regime switching via percentiles | Satish et al. 2014, p.18 para 1 |
| No-intercept (VWAP validation regs) | Satish et al. 2014, p.19 (Exhibits 3, 5 regressions only) |

### Researcher Inference Items

The following algorithmic choices are NOT directly from the papers and are
marked as Researcher inference throughout the spec:

1. Zero-floor guard for seasonal factors and historical averages.
2. FALLBACK sentinel for failed ARMA fitting.
3. ARMA model interface (predict_next, append_observation, make_state, etc.).
4. ARMA make_state initialization (unconditional mean for AR lags, zero for
   MA residuals, sequential Kalman filter update).
5. Softmax parameterization for sum-to-1 non-negative weight optimization.
6. Nelder-Mead optimizer with multiple random restarts.
7. Evenly spaced percentile thresholds for regime boundaries.
8. Grid search over N_regimes in {1..5}.
9. Cross-validation for regime count selection.
10. Independent-segment (panel) ARMA fitting for intraday model.
11. Zero-padding for early bins in Model B feature construction.
12. Remaining-fraction scaling for conditional percentage forecasts.
13. Non-negativity floor on V_hat and pct_hat.
14. Daily orchestration structure (pre-market / intraday / end-of-day).
15. Re-estimation schedule (monthly for ARMA, daily for hist_avg).
16. Default regime assignment (median percentile) when no bins observed.
17. Winsorization suggestion for extreme volume days.
18. Estimated total daily volume approach in Function 9.
19. No-intercept choice for surprise regression (justified by mean-zero
    property, NOT by p.19 citation which refers to VWAP validation regressions).
20. percentile_rank definition: count(ref < value) / len(ref) (strict less-than).
21. Exhibit 1 label interpretation ("Prior 5 days" = p_max, "4 Bins Prior" =
    p_max_intra).
22. Joint term budget enforcement via max(interday_term_counts).
23. Adaptive deviation limit calibration (95th percentile of surprise magnitudes).
24. Adaptive switch-off calibration (median cumulative volume threshold).
25. pct_forecasts[1] = hist_pct[1] initialization for bin 1.
26. In-sample ARMA predictions for weight training (look-ahead approximation).
27. Sophisticated variant: Model A-based surprise baseline procedure, using
    pre-observation forecasts as a fixed daily baseline for train/predict
    consistency.
28. generate_model_a_training_forecasts helper for sophisticated variant,
    including per-(day, bin) regime reconstruction from historical cumulative
    volume percentiles.
29. predict_next() specified as a pure query (no state mutation). Standard for
    ARMA state-space implementations but not stated in the paper.
30. Switch-off calibration uses crossover bin's median directly (not the
    previous bin's median, which would bias below the base threshold).
31. Sophisticated variant pre-observation baseline trade-off: acknowledged
    secondary context mismatch between context-free live baseline and
    context-aware training baseline for later bins. Deviation clamp bounds
    the impact. Iterative baseline construction identified as potential
    refinement. See Known Limitation #11.
32. N_surprise_lags treated as a fixed global hyperparameter (not per-stock
    CV), consistent with the paper's statement that an optimal number was
    identified globally for U.S. equities.
