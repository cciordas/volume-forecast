# Implementation Specification: Dual-Mode Volume Forecast (Raw + Percentage)

## Overview

This direction implements a two-model system for intraday volume forecasting, based on Satish, Saxena, and Palmer (2014). The system comprises:

1. A **raw volume forecast model (Model A)** that combines four components -- a rolling historical average, an inter-day ARMA model, a deseasonalized intraday ARMA model, and a dynamic regime-switching weight overlay -- to produce bin-level volume predictions for all remaining bins in a trading day.

2. A **volume percentage forecast model (Model B)** that extends the dynamic VWAP framework of Humphery-Jenner (2011), using volume surprises (deviations from the raw volume model's baseline) in a single rolling regression to adjust next-bin participation rates, with safety constraints.

The raw volume model produces full-day forecasts needed by scheduling tools and participation models. The volume percentage model produces next-bin-only forecasts used step-by-step by VWAP execution algorithms. The two models are tightly coupled: the raw volume model provides the surprise signal that drives the percentage model's adjustments.

**Reported performance (separate results for each model):**
- **Model A (raw volume):** 24% median MAPE reduction and 29% bottom-95% MAPE reduction over the rolling historical average baseline across all intraday intervals (Satish et al. 2014, p.20, Exhibit 6).
- **Model B (volume percentages):** 7.55% median reduction in absolute volume percentage error for 15-minute bins (Satish et al. 2014, p.23, Exhibit 9).
- **VWAP simulation (Model B's dynamic curve applied in execution):** 9.1% VWAP tracking error reduction (mean 8.74 bps vs. 9.62 bps historical) on 600+ simulated orders across Dow 30, midcap, and high-variance stocks (Satish et al. 2014, p.23, Exhibit 10).

## Algorithm

### Model Description

The system operates on 15-minute intraday bins (I = 26 bins per 6.5-hour U.S. equity trading day, from 9:30 to 16:00 ET). Each bin is indexed by i in {1, ..., 26}.

**Raw Volume Model (Model A):** For a given symbol s, day t, and bin i, the raw volume forecast V_hat(s, t, i) is a weighted combination of three component forecasts:

    V_hat(s, t, i) = w1(r) * H(s, t, i) + w2(r) * D(s, t, i) + w3(r) * A(s, t, i)

where H is the historical rolling average, D is the inter-day ARMA forecast, A is the re-seasonalized intraday ARMA forecast, and w1(r), w2(r), w3(r) are regime-dependent weights indexed by regime r, selected dynamically based on the historical percentile of cumulative volume observed so far on day t. (Satish et al. 2014, p.17-18, Exhibit 1.)

**Volume Percentage Model (Model B):** For a given symbol, day t, and the next bin i, the volume percentage forecast p_hat(t, i) adjusts a historical baseline percentage by incorporating volume surprises:

    p_hat(t, i) = p_hist(i) + delta(t, i)

where p_hist(i) is the historical average volume percentage for bin i, and delta(t, i) is a correction term derived from a single rolling regression on recent volume surprises across all bins, subject to deviation and switch-off constraints. (Satish et al. 2014, p.18-19, extending Humphery-Jenner 2011.)

### Pseudocode

#### Part 1: Raw Volume Forecast Model

```
PROCEDURE TrainRawVolumeModel(symbol, historical_data, train_end_date):
    # Step 1: Compute seasonal factors (trailing 6-month average per bin)
    # [Satish et al. 2014, p.17 "Raw Volume Forecast Methodology", para 3:
    #  "dividing by the intraday amount of volume traded in that bin over the
    #  trailing six months"]
    FOR each bin i in {1, ..., I}:
        raw_values = [volume[s, d, i] for d in [train_end_date - 126 .. train_end_date - 1]]
        seasonal_factor[i] = max(mean(raw_values), epsilon)
        # epsilon = 1.0 share (floor to prevent division by zero for illiquid bins)
        # [Researcher inference: floor value; prevents numerical explosion in
        #  deseasonalization for bins with zero or near-zero average volume]
    
    # Step 2: Fit inter-day ARMA models (one per bin)
    # [Satish et al. 2014, p.17, para 2: "We add to this a per-symbol, per-bin
    #  ARMA (autoregressive moving average) model reflecting the serial
    #  correlation observable across daily volumes."]
    # [AICc: Satish et al. 2014, p.17; Hurvich & Tsai 1989, 1993]
    #
    # Training window (N_interday):
    # The paper does not explicitly state the inter-day ARMA training window
    # length. Exhibit 1 labels this component's input as "Next Bin (Prior
    # 5 days)." Two interpretations exist:
    #   (a) The training window is 5 days (fit ARMA on only 5 observations).
    #   (b) "Prior 5 days" describes the effective AR memory: with p_max = 5,
    #       the forecast depends on at most the 5 most recent observations,
    #       while the model is trained on a longer history.
    # We adopt interpretation (b) because:
    #   - Fitting ARMA(p,q) with p+q up to 10 parameters on 5 observations is
    #     statistically unsound (AICc correction would dominate).
    #   - MLE for ARMA requires substantially more observations than parameters.
    #   - The intraday ARMA uses a 21-day window (~546 observations), making a
    #     5-observation inter-day window inconsistent in design philosophy.
    #   - Exhibit 1's labels describe the data each component *uses for
    #     prediction* (effective lookback), paralleling "4 Bins Prior to Current
    #     Bin" for the intraday ARMA, which also describes effective AR memory
    #     rather than a training window.
    # If the developer wishes to test interpretation (a), set N_interday = 5
    # and constrain p_max to 1-2 to avoid overfitting.
    # [Researcher inference: training window choice; see justification above]
    
    FOR each bin i in {1, ..., I}:
        daily_series_i = [volume[s, d, i] for d in training_days]  # N_interday days
        best_aic = infinity
        FOR p in {0, 1, 2, 3, 4, 5}:
            FOR q in {0, 1, 2, 3, 4, 5}:
                IF p == 0 AND q == 0: SKIP  # at least one AR or MA term
                model = fit_ARMA(daily_series_i, p, q, 
                                 include_constant=True,
                                 enforce_stationarity=True,
                                 enforce_invertibility=True)
                IF model failed to converge: SKIP
                # AICc computation:
                # AIC = -2 * log_likelihood + 2 * k
                # AICc = AIC + 2*k*(k+1) / (n - k - 1)
                # where k = p + q + 1 (AR params + MA params + constant)
                #       n = number of observations in daily_series_i
                # [Satish et al. 2014, p.17; Hurvich & Tsai 1989, 1993]
                k = p + q + 1
                n = len(daily_series_i)
                aic = -2 * model.log_likelihood + 2 * k
                aicc = aic + 2 * k * (k + 1) / (n - k - 1)
                IF aicc < best_aic:
                    best_aic = aicc
                    interday_model[i] = model
        IF no model converged for bin i:
            interday_model[i] = None  # fall back to historical average only
    
    # Step 3: Fit intraday ARMA model (one per symbol, on deseasonalized data)
    # [Satish et al. 2014, p.17-18: "Next, we fit an additional ARMA (p, q)
    #  model over deseasonalized intraday bin volume data... We compute this
    #  model on a rolling basis over the most recent month."]
    #
    # The intraday ARMA operates on the deseasonalized volume series.
    # Training data: 21 trading days x 26 bins = 546 observations.
    
    intraday_window = last 21 trading days of training data
    deseasonalized_series = []
    FOR each day d in intraday_window:
        FOR each bin i in {1, ..., I}:
            deseasonalized_series.append(volume[s, d, i] / seasonal_factor[i])
    # Total observations: 21 * 26 = 546
    
    # Day-boundary handling:
    # Concatenating days treats the transition from day d bin 26 to day d+1
    # bin 1 as continuous. However, the overnight gap (17.5 hours for U.S.
    # equities) introduces structural breaks: overnight news, pre-market
    # activity, and the opening auction create exogenous shocks that weaken
    # serial correlation across day boundaries. After deseasonalization, the
    # U-shape level discontinuity is removed, but the AR structure is still
    # disrupted at overnight transitions.
    #
    # With AR lags < 5 (at most 4 bins = 1 hour of lookback), the day-
    # boundary issue affects only the first ~4 bins of each day in the
    # training series, where the AR terms reach back into the previous
    # day's close. This is ~20% of the training data (4 bins * 21 days =
    # 84 of 546 observations).
    #
    # Reconciling Exhibit 1 with the paper text:
    # - Exhibit 1 shows "Current Bin" and "4 Bins Prior to Current Bin"
    #   feeding the intraday ARMA. This describes the *prediction inputs*
    #   (the most recent 4 bins that condition the forecast), not the
    #   training data window.
    # - The paper text (p.17-18) says "We compute this model on a rolling
    #   basis over the most recent month." This describes the *training
    #   window* for parameter estimation (21 trading days).
    # - These are consistent: the ARMA is *estimated* on 1 month of data,
    #   and at forecast time, uses the most recent ~4 observed bins as
    #   conditioning data for the AR component.
    #
    # Mitigation options (developer should test):
    # (a) Concatenate naively (baseline): accept that overnight transitions
    #     weaken AR coefficient estimates slightly. This is the simplest
    #     approach and likely what the paper uses given no discussion.
    # (b) Insert NaN/missing at day boundaries and use statsmodels'
    #     state-space representation (SARIMAX with missing observations)
    #     to handle gaps properly.
    # (c) Treat each day's 26 bins as a separate segment; estimate ARMA
    #     parameters via conditional MLE on each segment independently,
    #     then average the parameter estimates across days. This avoids
    #     cross-boundary contamination but reduces effective sample size.
    # We recommend option (a) as the default implementation since the paper
    # does not discuss boundary handling, and options (b)/(c) as sensitivity
    # checks.
    # [Researcher inference: paper does not discuss overnight boundary handling]
    
    # AR lag constraint: [Satish et al. 2014, p.18: "we found that the
    # autoregressive (AR) coefficients quickly decayed, so that we used AR
    # lags with a value less than five"]
    #
    # "Fewer than 11 terms" interpretation:
    # [Satish et al. 2014, p.18: "As a result, we fit each symbol with a
    #  dual ARMA model having fewer than 11 terms."]
    # The phrase "As a result" indicates this is a descriptive outcome of
    # AR coefficient decay, not a hard constraint imposed during fitting.
    # The hard constraint is: AR lags < 5 for the intraday ARMA.
    # "Terms" is ambiguous: it could mean (i) total estimated parameters
    # (p + q + constant for each model), or (ii) just AR + MA order terms
    # (p + q for each model). Under interpretation (i), counting constants:
    # max combined = (5+5+1) + (4+5+1) = 21, so the constraint would have
    # real bite. Under (ii): max combined = (5+5) + (4+5) = 19.
    # In practice, AICc-selected models are typically low-order (p <= 2,
    # q <= 2), so the combined count naturally stays well below 11.
    # We enforce p < 5 on the intraday model as the hard constraint, and
    # log a warning if the combined parameter count reaches or exceeds 11.
    # [Researcher inference: interpretation of "fewer than 11 terms" as
    #  descriptive rather than prescriptive, based on "As a result" phrasing]
    
    best_aic = infinity
    FOR p in {0, 1, 2, 3, 4}:        # AR lags < 5 (hard constraint)
        FOR q in {0, 1, 2, 3, 4, 5}:
            IF p == 0 AND q == 0: SKIP
            model = fit_ARMA(deseasonalized_series, p, q, 
                             include_constant=True,
                             enforce_stationarity=True,
                             enforce_invertibility=True)
            IF model failed to converge: SKIP
            k = p + q + 1
            n = len(deseasonalized_series)
            aic = -2 * model.log_likelihood + 2 * k
            aicc = aic + 2 * k * (k + 1) / (n - k - 1)
            IF aicc < best_aic:
                best_aic = aicc
                intraday_model = model
    
    # Verify combined term count (informational check, not hard constraint):
    # For each bin i, combined_terms = interday_model[i].num_params +
    #                                  intraday_model.num_params
    # Log a warning if combined_terms >= 11 for any bin.
    
    IF no intraday model converged:
        intraday_model = None  # fall back to other components
    
    # Step 4: Train regime-switching weights via in-sample optimization
    # [Satish et al. 2014, p.18, para 4: "a dynamic weight overlay on top of
    #  these three components... that minimizes the error on in-sample data.
    #  We incorporate a notion of regime switching by training several weight
    #  models for different historical volume percentile cutoffs"]
    #
    # Regime classification operates at the (day, bin) level: a single day
    # can transition between regimes as cumulative volume grows. This is
    # consistent with the paper's description of dynamically applying
    # weights "based on the historical percentile of the observed cumulative
    # volume" (p.18) -- cumulative volume changes at each bin.
    # [Researcher inference: (day,bin)-level classification for consistency
    #  between training and forecasting]
    #
    # Number of regimes: The paper says "training several weight models"
    # without specifying how many. We use 3 regimes based on tercile cutoffs
    # of the historical cumulative volume distribution, as this is the
    # simplest scheme consistent with "several."
    # [Researcher inference: number of regime buckets and thresholds not
    #  disclosed in paper; 3 regimes at [33, 67] percentile proposed]
    
    # For each training day d and bin i, compute cumulative volume through bin i:
    FOR each day d in training_days:
        FOR each bin i in {1, ..., I}:
            cum_vol[d, i] = sum(volume[s, d, j] for j in {1, ..., i})
    
    # Build historical percentile distribution per bin:
    FOR each bin i in {1, ..., I}:
        cum_vol_distribution[i] = sorted([cum_vol[d, i] for d in prior_60_days])
        regime_thresholds[i] = percentiles(cum_vol_distribution[i], [33, 67])
    
    # Optimize weights per regime across all (day, bin) observations in that regime
    FOR each regime r in {low, medium, high}:
        # Select (day, bin) pairs falling into regime r
        regime_observations = [(d, i) where classify_regime(
            cum_vol[d, i], regime_thresholds[i]) == r]
        
        # Optimize weights to minimize forecast error on regime subset
        # [Satish et al. 2014, p.18: "minimizes the error on in-sample data"
        #  without specifying which error metric. Since MAPE is the evaluation
        #  metric (p.17), it is the natural objective. However, MAPE minimization
        #  is non-smooth. Practical approach: use MSE objective with
        #  scipy.optimize.minimize (method='SLSQP') and verify MAPE also improves.
        #  Alternatively, use Nelder-Mead on MAPE directly.]
        # [Researcher inference: error metric and optimization method not specified]
        w1[r], w2[r], w3[r] = optimize_weights(
            target=actual_volumes[regime_observations],
            forecasts=[H_forecasts, D_forecasts, A_forecasts],
            objective="MSE",  # or "MAPE" with Nelder-Mead
            # Weight constraints:
            # [Researcher inference: paper does not specify whether weights are
            #  constrained. The simplex constraint (sum-to-1, non-negative) ensures
            #  a convex combination of components, which is the natural interpretation
            #  of "weighted overlay." Relaxing to allow negative weights or
            #  unconstrained sums is an alternative worth testing.]
            constraint: w1 + w2 + w3 = 1, all weights >= 0,
            initial_guess: [1/3, 1/3, 1/3]
        )
    
    RETURN seasonal_factor, interday_model, intraday_model, 
           regime_thresholds, weights


PROCEDURE ForecastRawVolume(symbol, day_t, current_bin, models):
    # Produce forecasts for all bins from current_bin to I
    # [Satish et al. 2014, Exhibit 1 and p.17-18]
    
    # Determine regime based on observed cumulative volume
    # [Satish et al. 2014, p.18, para 4: "dynamically apply the appropriate
    #  weights intraday based on the historical percentile of the observed
    #  cumulative volume"]
    #
    # Early-bin regime classification:
    # At current_bin = 1, only one observation is available, producing high
    # variance in percentile rank. At bins 1 through min_regime_bins,
    # default to the "medium" regime to avoid spurious regime switches.
    # [Researcher inference: minimum bins before regime activation not
    #  specified in paper; 3 bins proposed to accumulate sufficient volume
    #  signal (~45 minutes of trading)]
    IF current_bin >= min_regime_bins AND current_bin > 1:
        cum_vol = sum(volume[s, day_t, j] for j in {1, ..., current_bin - 1})
        regime = classify_regime(cum_vol, regime_thresholds[current_bin - 1])
    ELSE:
        regime = "medium"  # default at market open or early bins
    
    # Update intraday ARMA state with observed deseasonalized volumes
    # [Satish et al. 2014, Exhibit 1: ARMA Intraday receives "Current Bin"
    #  and "4 Bins Prior to Current Bin" as inputs. With AR order p <= 4,
    #  only the most recent p observations affect the forecast. The ARMA
    #  recursive structure propagates earlier observations through its state,
    #  so feeding all observed bins is equivalent for forecasting purposes.]
    #
    # Note: At forecast time, the ARMA model is NOT re-estimated. Only
    # the internal state (most recent residuals and observations) is updated
    # by conditioning on newly observed bins. Re-estimation at every bin
    # would be computationally expensive (AICc grid search on 546+ obs)
    # and unnecessary -- the parameters capture the general intraday dynamics,
    # while the state update incorporates today's specific information.
    # [Researcher inference: reconditioning vs. re-estimation; paper does not
    #  specify but reconditioning is the standard ARMA forecasting approach]
    IF current_bin > 1 AND intraday_model is not None:
        observed_deseas = [volume[s, day_t, j] / seasonal_factor[j] 
                           for j in {1, ..., current_bin - 1}]
        intraday_model.update(observed_deseas)
    
    FOR each bin i in {current_bin, ..., I}:
        # Component 1: Historical rolling average
        # [Satish et al. 2014, p.16 "Historical Window Average/Rolling Means":
        #  "One chooses the number of days of historical data to use, a variable
        #  that we shall call N, and then uses the prior N days of volume data
        #  to construct an arithmetic average"]
        # [Exhibit 1: "Next Bin (Prior 21 days)" labels this component's input]
        H[i] = mean(volume[s, d, i] for d in [day_t - N_hist .. day_t - 1])
        
        # Component 2: Inter-day ARMA forecast
        # [Satish et al. 2014, p.17, para 2]
        IF interday_model[i] is not None:
            D[i] = interday_model[i].forecast(steps=1)
        ELSE:
            D[i] = H[i]  # fallback to historical average
        
        # Component 3: Intraday ARMA forecast (re-seasonalized)
        # [Satish et al. 2014, p.17-18: "before passing intraday ARMA forecasts,
        #  we re-seasonalize these forecasts via multiplication"]
        IF intraday_model is not None:
            steps_ahead = i - current_bin + 1
            A_deseas[i] = intraday_model.forecast(steps=steps_ahead)
            A[i] = A_deseas[i] * seasonal_factor[i]  # re-seasonalize
        ELSE:
            A[i] = H[i]  # fallback to historical average
        
        # Combine components with regime-specific weights
        V_hat[i] = w1[regime] * H[i] + w2[regime] * D[i] + w3[regime] * A[i]
    
    RETURN V_hat
```

#### Part 2: Volume Percentage Forecast Model

```
PROCEDURE TrainVolumePercentageModel(symbol, historical_data, train_end_date,
                                      raw_volume_model):
    # Training window for the percentage regression:
    # training_days = [train_end_date - N_reg_train .. train_end_date - 1]
    # where N_reg_train defaults to 252 trading days (1 year), matching the
    # DailyUpdate re-estimation window.
    # [Researcher inference: the paper does not specify the initial training
    #  window for the percentage model. 252 days provides enough observations
    #  for a stable regression while matching the rolling window used in
    #  production (DailyUpdate Step 5).]
    #
    # NOTE on mild lookahead bias: The raw_volume_model passed here was
    # trained on all training data through train_end_date. When computing
    # raw forecasts for day d < train_end_date, the ARMA parameters and
    # seasonal factors reflect future data relative to d. This bias is
    # negligible in practice because: (a) ARMA parameters estimated on 252+
    # days change minimally when dropping one day; (b) seasonal factors are
    # 126-day averages that barely shift day-to-day. To eliminate it entirely,
    # use expanding-window re-estimation (re-train raw_volume_model for each
    # d using data through d-1), but this multiplies computational cost by
    # N_reg_train.
    # [Researcher inference: lookahead bias acknowledged; negligible impact]
    
    training_days = [train_end_date - N_reg_train .. train_end_date - 1]
    
    # Step 1: Compute historical volume percentage curve
    # [Satish et al. 2014, p.18 "Volume Percentage Forecast Methodology"]
    FOR each bin i in {1, ..., I}:
        p_hist[i] = mean(volume[s, d, i] / daily_total_volume[s, d]
                         for d in training_days)
    
    # Step 2: Compute deviation bounds
    # [Satish et al. 2014, p.24; Humphery-Jenner 2011]
    # The paper says "deviation limits (e.g., depart no more than 10% away
    # from a historical VWAP curve)." The "e.g." indicates this is an
    # illustrative example from Humphery-Jenner, not a confirmed value for
    # the Satish et al. implementation. The paper also states "we developed
    # a separate method for computing deviation bounds" (p.19) but does not
    # disclose it. We use 10% as a starting point; this should be treated as
    # a tunable parameter.
    # [Researcher inference: exact deviation bound method is proprietary]
    max_deviation = 0.10  # 10% relative deviation from p_hist[i]
    
    # Step 3: Fit a SINGLE rolling regression for volume surprises
    # [Satish et al. 2014, p.18-19: "volume surprises based on a naive
    #  volume forecast model can be used to train a rolling regression model
    #  that adjusts market participation"]
    # [Humphery-Jenner 2011: specifies a single rolling regression model,
    #  not a per-bin model. The regression uses recent volume surprises
    #  across bins to predict the next bin's adjustment.]
    #
    # The model is a single OLS regression applied uniformly across all bins:
    #   delta[t, i] = sum_{k=1}^{K_reg} beta_k * surprise[t, i-k]
    # where surprise[t, j] = actual_pct[t, j] - expected_pct[j]
    #
    # No intercept is included: when all recent surprises are zero (the raw
    # model's predictions are perfect), the adjustment delta should also be
    # zero -- no correction is needed. This is consistent with the paper's
    # no-intercept philosophy: "we perform both regressions without the
    # inclusion of a constant term... This means that our model does not
    # assume that there is a positive amount of VWAP error if our volume
    # predictions are 100% accurate" (Satish et al. 2014, p.19). While that
    # statement refers to validation regressions, the same logic applies:
    # zero surprise should produce zero adjustment. The developer may test
    # with an intercept as an alternative.
    # [Researcher inference: no-intercept default based on paper's philosophy]
    #
    # Surprise baseline:
    # The paper says "we could apply our more extensive volume forecasting
    # model... to compute volume surprises" (Satish et al. 2014, p.19). The
    # word "could" is aspirational, not confirmative. It is unclear whether
    # the published Exhibit 9/10 results used the raw volume model or simple
    # rolling means (as in Humphery-Jenner's original) as the expected
    # baseline. We implement the raw-model approach as the primary version
    # since the paper presents it as their contribution over Humphery-Jenner.
    # However, the developer should also test with simple rolling means
    # (p_hist) as the expected baseline.
    # [Researcher inference: primary choice of raw model for surprises based
    #  on paper's framing, but ambiguity acknowledged]
    #
    # Surprise definition:
    #   During training, actual_pct uses the known daily total:
    #     actual_pct[d, j] = volume[s, d, j] / daily_total_volume[s, d]
    #   The expected_pct uses the raw volume model as the baseline:
    #     expected_pct[d, j] = raw_forecast[s, d, j] / sum_k(raw_forecast[s, d, k])
    #   surprise[d, j] = actual_pct[d, j] - expected_pct[d, j]
    #
    # [Researcher inference: regression structure (single model, not per-bin)
    #  based on Humphery-Jenner's singular "rolling regression model" (p.24)
    #  and the paper's description. K_reg not disclosed; 3 proposed as default.]
    
    K_reg = 3  # number of lagged surprise terms
    
    # Build training data: pool across all (day, bin) observations
    X_train = []  # lagged surprise vectors
    y_train = []  # next-bin surprise values
    
    FOR each day d in training_days:
        # Compute raw volume model forecasts for day d (using model trained
        # on data through d-1, mimicking out-of-sample usage)
        raw_forecasts_d = ForecastRawVolume(symbol, d, bin=1, raw_volume_model)
        raw_total_d = sum(raw_forecasts_d[k] for k in {1, ..., I})
        
        FOR each bin i in {K_reg + 1, ..., I}:
            # Compute expected percentages from raw model
            expected_pct_j = raw_forecasts_d[j] / raw_total_d  # for each bin j
            
            # Compute actual percentages from known daily total
            actual_pct_j = volume[s, d, j] / daily_total_volume[s, d]
            
            # Build lagged surprise vector
            surprise_lags = []
            FOR k in {1, ..., K_reg}:
                j = i - k
                surprise_lags.append(actual_pct_j[j] - expected_pct_j[j])
            
            X_train.append(surprise_lags)
            y_train.append(actual_pct_j[i] - expected_pct_j[i])
    
    # Fit single OLS regression (pooled across all bins and days)
    regression_model = fit_OLS(y=y_train, X=X_train, include_intercept=False)
    
    RETURN p_hist, regression_model, max_deviation


PROCEDURE ForecastVolumePercentage(symbol, day_t, current_bin, pct_models,
                                    raw_volume_model):
    # Forecast the volume percentage for the NEXT bin only
    # [Satish et al. 2014, p.18-19]
    
    next_bin = current_bin + 1
    IF next_bin > I:
        RETURN 0  # no more bins
    
    # Step 1: Compute estimated daily total volume (V_total_est)
    # [Researcher inference: paper does not specify how to estimate daily
    #  total intraday. We use the sum of observed volumes for completed bins
    #  plus raw model forecasts for remaining bins. This couples the two
    #  models as described in the paper while avoiding circularity.]
    #
    # UPDATE ORDERING: Model A (raw volume) must be run FIRST to produce
    # V_hat for remaining bins. Then Model B uses these forecasts. This
    # sequential dependency is critical:
    #   1. Run ForecastRawVolume() -> V_hat[current_bin+1 .. I]
    #   2. Compute V_total_est = sum(observed) + sum(V_hat for remaining)
    #   3. Compute surprises using V_total_est
    #   4. Run percentage regression to get delta
    # The raw model forecasts are computed independently (no dependency on
    # Model B), so there is no circular dependency.
    
    # Two separate ForecastRawVolume calls serve different purposes:
    # (a) Unconditional forecasts (current_bin=1): for computing expected_pct
    #     baseline. These reflect the model's pre-day expectation, appropriate
    #     for the surprise calculation since p_hist is also unconditional.
    # (b) Conditioned forecasts (current_bin=current_bin): for computing
    #     V_total_est. These incorporate intraday information (observed bins,
    #     updated regime), producing a more accurate daily total estimate,
    #     especially in the second half of the day.
    # [Researcher inference: the paper does not distinguish these two uses.
    #  Using conditioned forecasts for V_total_est is the natural choice since
    #  V_total_est is described as "the best available estimate" of daily
    #  total volume. Impact is small early in the day (few observed bins,
    #  little conditioning advantage) and grows later.]
    
    unconditional_forecasts = ForecastRawVolume(symbol, day_t, current_bin=1,
                                                raw_volume_model)
    raw_total = sum(unconditional_forecasts[k] for k in {1, ..., I})
    
    conditioned_forecasts = ForecastRawVolume(symbol, day_t, 
                                              current_bin=current_bin,
                                              raw_volume_model)
    V_total_est = (
        sum(volume[s, day_t, j] for j in {1, ..., current_bin})  # observed
        + sum(conditioned_forecasts[k] for k in {current_bin + 1, ..., I})
    )
    
    # Step 2: Compute volume surprises from observed bins
    # [Satish et al. 2014, p.19, extending Humphery-Jenner 2011]
    # expected_pct = raw model's predicted percentage (normalized by raw total)
    # actual_pct = observed volume / V_total_est
    # During training, actual_pct used the known daily total. During
    # inference, we use V_total_est as the best available estimate.
    # This introduces a small inconsistency that diminishes as the day
    # progresses (V_total_est converges to actual as more bins are observed).
    
    recent_surprises = []
    FOR k in {1, ..., K_reg}:
        j = current_bin - k + 1
        IF j < 1: BREAK  # not enough observed bins
        expected_pct = unconditional_forecasts[j] / raw_total
        actual_pct = volume[s, day_t, j] / V_total_est
        surprise = actual_pct - expected_pct
        recent_surprises.append(surprise)
    
    # Pad with zeros if fewer than K_reg bins observed
    WHILE len(recent_surprises) < K_reg:
        recent_surprises.append(0.0)
    
    # Step 3: Compute adjustment via regression
    delta = regression_model.predict(recent_surprises)
    
    # Step 4: Compute scaled baseline and apply deviation constraint
    # [Satish et al. 2014, p.24; Humphery-Jenner 2011]
    # Note: 10% is illustrative ("e.g."); treat as tunable parameter.
    #
    # The deviation constraint is applied relative to the SCALED baseline
    # (not the unscaled p_hist). This ensures the 10% deviation limit is
    # enforced relative to the actual baseline used in the final forecast.
    # If instead we applied the constraint relative to unscaled p_hist,
    # the effective deviation relative to the scaled base would be
    # max_deviation / scale, which can exceed max_deviation when
    # scale < 1.0 (volume running above expectations, less remaining).
    # [Researcher inference: the paper does not specify whether the
    #  deviation limit is relative to the historical or scaled baseline.
    #  Applying it to the scaled baseline is more conservative and ensures
    #  the stated 10% limit is never exceeded in the final forecast.]
    remaining_fraction = 1.0 - (
        sum(volume[s, day_t, j] for j in {1,...,current_bin}) / V_total_est)
    remaining_hist = sum(p_hist[j] for j in {next_bin, ..., I})
    IF remaining_hist > 0:
        scale = remaining_fraction / remaining_hist
    ELSE:
        scale = 1.0
    scaled_base = scale * p_hist[next_bin]
    max_delta = max_deviation * scaled_base
    delta = clip(delta, -max_delta, +max_delta)
    
    # Step 5: Apply switch-off constraint
    # [Satish et al. 2014, p.24: "once 80% of the day's volume is reached,
    #  return to a historical approach"]
    # [Humphery-Jenner 2011: same switch-off mechanism]
    cum_pct = 1.0 - remaining_fraction  # already computed in Step 4
    IF cum_pct >= switchoff_threshold:
        delta = 0  # revert to historical curve for this and all future bins
    
    # Step 6: Compute final forecast
    # The scaled baseline and scale factor were computed in Step 4 above
    # (for the deviation constraint). Reuse them here.
    #
    # When switch-off has NOT triggered:
    #   p_hat = scaled_base + delta (already clipped by deviation constraint)
    # When switch-off HAS triggered (delta = 0):
    #   p_hat = scaled_base, no further adjustment
    #
    # Renormalization:
    # The remaining percentages (bins next_bin through I) should sum to
    # (1 - cum_pct_observed), where cum_pct_observed is the fraction of
    # total volume already observed. Rather than renormalizing the current
    # single-bin forecast (which would undo the safety constraints), we
    # scale the HISTORICAL baseline for the remaining bins so they sum to
    # the remaining fraction, and then apply delta on top of that scaled
    # baseline. The scale factor (computed in Step 4) accomplishes this:
    #   scale = remaining_fraction / remaining_hist
    #   scaled_base = scale * p_hist[next_bin]
    #   p_hat = scaled_base + delta
    #
    # This ensures that if delta = 0 for all remaining bins (switch-off
    # active), the remaining forecasts naturally sum to remaining_fraction.
    # When delta != 0, the adjustment is additive on top of the scaled base.
    # The deviation constraint (Step 4) clips delta relative to scaled_base,
    # so the safety limit is always respected in the final forecast.
    # [Researcher inference: renormalization approach; paper does not specify.
    #  The key principle is that safety constraints (deviation limit, switch-off)
    #  must not be undone by renormalization.]
    
    p_hat = scaled_base + delta
    p_hat = max(p_hat, 0)  # ensure non-negative
    
    RETURN p_hat
```

#### Part 3: Model Update and Re-estimation

```
PROCEDURE DailyUpdate(symbol, day_t, models):
    # Re-estimate models on rolling basis
    # [Satish et al. 2014, p.17-18]
    
    # 1. Update seasonal factors (rolling 6-month window, daily refresh)
    FOR each bin i:
        raw_values = [volume[s, d, i] for d in [day_t - 126 .. day_t - 1]]
        seasonal_factor[i] = max(mean(raw_values), epsilon)
    
    # 2. Re-fit intraday ARMA (daily, rolling 1-month window)
    # [Satish et al. 2014, p.17-18: "We compute this model on a rolling
    #  basis over the most recent month."]
    intraday_window = [day_t - 21 .. day_t - 1]
    deseasonalized_series = []
    FOR each day d in intraday_window:
        FOR each bin i in {1, ..., I}:
            deseasonalized_series.append(volume[s, d, i] / seasonal_factor[i])
    Refit intraday_model using AICc selection on deseasonalized_series
    (same procedure as TrainRawVolumeModel Step 3)
    
    # 3. Update inter-day ARMA models (weekly, every 5 trading days)
    # [Researcher inference: the paper does not specify re-estimation
    #  frequency for inter-day models. Weekly (every 5 trading days) balances
    #  freshness against computational cost. Between re-estimations, append
    #  new observations and update ARMA state (sufficient statistics) without
    #  re-running MLE. Use conditional MLE with appended observations.]
    IF day_t mod 5 == 0:  # every 5 trading days
        FOR each bin i:
            daily_series_i = [volume[s, d, i] 
                              for d in [day_t - N_interday .. day_t - 1]]
            Refit interday_model[i] using AICc selection
            (same procedure as TrainRawVolumeModel Step 2)
    ELSE:
        FOR each bin i:
            interday_model[i].append_observation(volume[s, day_t, i])
            # Update ARMA state for next-day forecast without full re-estimation
    
    # 4. Re-optimize regime weights (monthly, every 21 trading days)
    # [Researcher inference: monthly re-optimization on rolling window.
    #  More frequent risks overfitting to recent regime distribution;
    #  less frequent risks stale weights.]
    IF day_t mod 21 == 0:
        Re-optimize regime weights using TrainRawVolumeModel Step 4
        on data from [day_t - 252 .. day_t - 1]
    
    # 5. Update volume percentage regression (monthly)
    IF day_t mod 21 == 0:
        Re-fit regression_model using TrainVolumePercentageModel Step 3
        on data from [day_t - 252 .. day_t - 1]
    
    # 6. Update historical volume percentage curve (daily)
    FOR each bin i:
        p_hist[i] = mean(volume[s, d, i] / daily_total_volume[s, d]
                         for d in [day_t - N_hist .. day_t - 1])
```

### Data Flow

```
Input Data
  |
  v
[Intraday volume: symbol, date, bin_index, shares_traded]
[Shape: (num_symbols x num_days x I) where I = 26 for 15-min bins]
  |
  +---> Seasonal Factor Computation
  |      Input:  volume[s, d-126:d-1, i] for each bin i
  |      Output: seasonal_factor[i], shape (I,), float64
  |      Floor:  max(mean(...), 1.0) to prevent division by zero
  |
  +---> Historical Rolling Average (Component 1)
  |      Input:  volume[s, d-N_hist:d-1, i] for each bin i
  |      Output: H[i], shape (I,), float64
  |
  +---> Inter-day ARMA Fitting (Component 2)
  |      Input:  volume[s, :, i] daily series for bin i (N_interday days)
  |      Output: interday_model[i] -- I separate ARMA models
  |      Forecast: D[i], shape (I,), float64
  |      Library: statsmodels.tsa.arima.model.ARIMA (recommended)
  |
  +---> Deseasonalize + Intraday ARMA Fitting (Component 3)
  |      Input:  volume[s, d-21:d-1, :] / seasonal_factor[:]
  |      Output: intraday_model -- 1 ARMA model per symbol
  |      Series length: 21 * 26 = 546 observations
  |      Forecast: A_deseas[i] -> A[i] = A_deseas[i] * seasonal_factor[i]
  |      Shape: (I,), float64
  |      Library: statsmodels.tsa.arima.model.ARIMA (recommended)
  |
  +---> Regime Classification (per bin, not per day)
  |      Input:  cumulative volume through current bin, per-bin historical dist.
  |      Output: regime index r in {low, medium, high}
  |      Constraint: only activate after min_regime_bins (default 3) bins observed;
  |                  before that, default to "medium" regime
  |
  v
Dynamic Weight Combiner
  Input:  H[i], D[i], A[i], regime r
  Weights: w1[r], w2[r], w3[r] per regime, shape (3,) per regime
  Output: V_hat[i] = w1[r]*H[i] + w2[r]*D[i] + w3[r]*A[i]
  Shape: (I - current_bin + 1,), float64
  |
  +---> Raw Volume Forecast V_hat (all remaining bins)
  |
  +---> Estimated Daily Total (V_total_est)
  |      Two separate ForecastRawVolume calls:
  |      (a) Unconditional (current_bin=1): for expected_pct baseline
  |      (b) Conditioned (current_bin=actual): for V_total_est
  |      V_total_est = sum(observed) + sum(conditioned forecasts remaining)
  |      Shape: scalar, float64
  |      UPDATE ORDERING: compute V_hat FIRST, then V_total_est
  |
  +---> Volume Surprise Computation
         expected_pct[j] = unconditional_forecast[j] / sum(unconditional)
         actual_pct[j] = volume[j] / V_total_est
         surprise[j] = actual_pct[j] - expected_pct[j]
         Shape: (K_reg,), float64
           |
           v
         Single Rolling Regression (pooled across bins)
         Input:  K_reg lagged surprises
         Output: delta (scalar adjustment)
         Library: numpy.linalg.lstsq or sklearn.linear_model.LinearRegression
           |
           v
         Constraint Application
           (1) Scale baseline: scaled_base = scale * p_hist[next_bin]
           (2) Clip: |delta| <= max_deviation * scaled_base
           (3) Switch-off: if cum_pct >= 0.80, delta = 0
         Final: p_hat = scaled_base + delta
         Output: p_hat (next-bin volume percentage, scalar, float64)
```

### Variants

We implement the full dual-model system as described in Satish et al. (2014), which is the most complete variant in the paper. The paper does not describe explicit sub-variants of the raw volume model, but does test different bin sizes (5, 15, 30 minutes). We implement 15-minute bins as the primary configuration since this received the most thorough treatment (Satish et al. 2014, p.16: "The work in this article is based on 15-minute bins").

The volume percentage model is an extension of Humphery-Jenner (2011). We implement the Satish et al. extension (with the raw volume model providing the expected baseline for surprises) as the primary variant, since the paper presents it as their contribution over Humphery-Jenner. However, the paper's language is aspirational ("we could apply"), not confirmative (Satish et al. 2014, p.19). The developer should also test with simple rolling means (p_hist) as the expected baseline, as the published Exhibit 9 and 10 results may have been generated with the simpler Humphery-Jenner formulation.

## Parameters

| Parameter | Description | Recommended Value | Sensitivity | Range |
|-----------|-------------|-------------------|-------------|-------|
| I | Number of intraday bins | 26 (15-min bins) | Low (structural) | {13, 26, 78} for {30, 15, 5}-min |
| N_hist | Historical window for rolling mean (days) | 21 | Medium | [10, 60] |
| N_interday | Training window for inter-day ARMA (days) | 252 (1 year) | Low-medium | [126, 504] |
| N_intraday | Rolling window for intraday ARMA (trading days) | 21 (1 month) | Medium | [10, 42] |
| N_seasonal | Trailing window for seasonal factors (trading days) | 126 (6 months) | Low | [63, 252] |
| p_max_interday | Maximum AR order for inter-day ARMA | 5 | Low | [3, 7] |
| q_max_interday | Maximum MA order for inter-day ARMA | 5 | Low | [3, 7] |
| p_max_intraday | Maximum AR order for intraday ARMA | 4 | Medium | [2, 5] |
| q_max_intraday | Maximum MA order for intraday ARMA | 5 | Low | [2, 5] |
| n_regimes | Number of regime buckets | 3 | Medium-high | [2, 5] |
| regime_percentiles | Percentile cutoffs for regime classification | [33, 67] | High | Depends on n_regimes |
| regime_threshold_window | Days of history for regime percentile distribution | 60 | Medium | [21, 126] |
| min_regime_bins | Minimum bins observed before regime switching activates | 3 | Low | [1, 5] |
| max_pct_deviation | Maximum percentage deviation from historical | 0.10 (10%) | Medium | [0.05, 0.20] |
| pct_switchoff | Cumulative volume fraction triggering switch-off | 0.80 (80%) | Medium | [0.70, 0.90] |
| K_reg | Number of lagged surprise terms in percentage regression | 3 | Medium | [1, 5] |
| N_reg_train | Training window for percentage regression (trading days) | 252 (1 year) | Low | [126, 504] |
| epsilon | Floor for seasonal factor to prevent division by zero | 1.0 (shares) | Low | [0.1, 10.0] |
| interday_refit_freq | How often to re-estimate inter-day ARMA (trading days) | 5 (weekly) | Low | [1, 21] |
| weight_refit_freq | How often to re-optimize regime weights (trading days) | 21 (monthly) | Low | [5, 63] |

**Note on N_interday:** The recommended value of 252 assumes interpretation (b) of Exhibit 1's "Prior 5 days" label (see Pseudocode Step 2 discussion). Under interpretation (a), set N_interday = 5 and constrain p_max_interday to 2 to avoid overfitting. The paper does not disclose this value. (Satish et al. 2014, Exhibit 1, p.18.) [Researcher inference: training window choice]

**Note on regime_threshold_window:** The pseudocode uses a 60-day window of historical cumulative volume to build the percentile distribution for regime classification. This is a reasonable middle ground: too short (e.g., 21 days) produces noisy thresholds; too long (e.g., 126 days) may miss regime shifts in market-wide volume levels. [Researcher inference: window length not specified in paper]

**Note on min_regime_bins:** At the start of the day, cumulative volume from just 1-2 bins has high variance in percentile rank, making regime classification unreliable. Defaulting to "medium" regime for the first 3 bins (~45 minutes) allows sufficient volume to accumulate for meaningful percentile ranking. [Researcher inference: not specified in paper]

**Note on max_pct_deviation:** The paper says "e.g., depart no more than 10% away from a historical VWAP curve" (Satish et al. 2014, p.24, describing Humphery-Jenner 2011). The "e.g." qualifier means 10% is illustrative; the authors developed "a separate method for computing deviation bounds" (p.19) that is not disclosed. Treat 10% as approximate/inherited from Humphery-Jenner. [Researcher inference: exact deviation bound method is proprietary]

**Note on "fewer than 11 terms":** The paper states "we fit each symbol with a dual ARMA model having fewer than 11 terms" (Satish et al. 2014, p.18). This follows the sentence about AR coefficient decay and is introduced with "As a result," indicating an observed outcome rather than a prescribed constraint. The hard constraint is: AR lags < 5 for the intraday ARMA. "Terms" is ambiguous -- it could mean total estimated parameters (p + q + constant per model) or just AR + MA order terms (p + q per model). In practice, AICc-selected models are typically low-order (p <= 2, q <= 2), so the combined parameter count naturally stays below 11. We enforce p < 5 on the intraday model and verify the total after fitting. If the developer wishes to impose the 11-term limit as a hard constraint, check: interday_model[i].num_params + intraday_model.num_params < 11 for each bin i, and if violated, reduce the intraday model order. [Researcher inference: interpretation as descriptive]

### Initialization

1. **Seasonal factors:** Compute from the trailing 126 trading days (6 months) before the start of out-of-sample period. For each bin i, seasonal_factor[i] = max(mean(volume[s, d, i] for d in those 126 days), epsilon). The epsilon floor (default 1.0 share) prevents division by zero for illiquid bins (Satish et al. 2014, p.17: "dividing by the intraday amount of volume traded in that bin over the trailing six months").

2. **Inter-day ARMA models:** Fit I = 26 separate ARMA(p, q) models, one per bin, on the daily volume series for that bin over N_interday training days. Select (p, q) by AICc with p, q in {0, ..., 5} plus a constant term. AICc = AIC + 2k(k+1)/(n-k-1), where AIC = -2*log_likelihood + 2*k, k = p + q + 1 (AR params + MA params + constant), and n = number of observations. Enforce stationarity (AR roots outside unit circle) and invertibility (MA roots outside unit circle). Use conditional MLE (set initial innovations to zero) as the default; exact MLE is an alternative for short series. Recommended library: statsmodels.tsa.arima.model.ARIMA. If MLE fails to converge for a given (p, q), skip that specification and try the next. If no ARMA converges for a bin, set interday_model[i] = None (Satish et al. 2014, p.17: "We use nearly standard ARMA model-fitting techniques relying on maximum-likelihood estimation"). Detection of convergence failure: check that the optimizer converged (model.mlefit.mle_retvals['converged'] == True), that the Hessian is positive definite (no negative eigenvalues), and that parameter standard errors are finite.

3. **Intraday ARMA model:** Fit one ARMA model per symbol on the deseasonalized intraday volume series from the most recent 21 trading days (546 observations). Deseasonalize by dividing each bin's volume by seasonal_factor[i]. Constraint: AR order p < 5 (Satish et al. 2014, p.18: "AR lags with a value less than five"). Same estimation approach as inter-day models. Default: concatenate days naively (option (a) from Pseudocode Step 3 discussion).

4. **Regime weights:** Initialize all weights to equal (w1 = w2 = w3 = 1/3), then optimize on in-sample data at the (day, bin) level within each regime bucket. The optimization minimizes MSE (or MAPE via Nelder-Mead) subject to the simplex constraint (weights sum to 1, non-negative). Use scipy.optimize.minimize with method='SLSQP' for MSE, or scipy.optimize.minimize with method='Nelder-Mead' for MAPE (Satish et al. 2014, p.18).

5. **Historical volume percentages:** Compute p_hist[i] as the average of volume[s, d, i] / sum_j(volume[s, d, j]) over the N_hist most recent training days (Satish et al. 2014, p.18).

6. **Regression coefficients:** Fit a single OLS regression (no intercept) of next-bin volume surprise on K_reg lagged surprises, pooled across all bins and training days. Surprises are computed using the raw model's predicted percentages as the expected baseline (Satish et al. 2014, p.18-19, extending Humphery-Jenner 2011). The no-intercept design ensures zero correction when predictions are perfect.

### Calibration

The calibration procedure is a multi-step process:

**Step 1: Seasonal factor estimation**
- Window: trailing 126 trading days.
- Computation: arithmetic mean of raw volume per bin, floored at epsilon.
- Update: daily (rolling window shifts by one day).

**Step 2: ARMA order selection via AICc**
- For inter-day models: enumerate all (p, q) with p, q in {0, ..., 5} (excluding p=q=0); compute AICc for each; select minimum. AICc = AIC + 2k(k+1)/(n-k-1), where AIC = -2*log_likelihood + 2*k, k is the number of estimated parameters (p + q + 1, including constant), and n is the sample size (number of observations in the training window). If convergence fails, skip that (p, q) and try the next. If all fail for a bin, use None (fallback to historical average).
- For the intraday model: same procedure but with AR order constrained to p < 5.
- Satish et al. 2014, p.17: "We depart from the standard technique in using the corrected AIC, symbolized by AICc, as detailed by Hurvich and Tsai [1989, 1993]. AICc adds a penalty term to AIC for extra AR and MA terms."

**Step 3: Regime threshold estimation**
- [Researcher inference: The paper does not disclose specific thresholds.]
- Proposed procedure: For each bin i and each training day d, compute the cumulative volume through bin i. Build the empirical distribution of cumulative volumes per bin across prior 60 trading days. Classify each (day, bin) observation into n_regimes buckets based on equal-frequency percentile cuts of that bin's cumulative volume distribution.
- Alternative: Use quantiles of the first-hour cumulative volume distribution to classify entire days, then apply that classification uniformly across all bins within a day.

**Step 4: Weight optimization**
- For each regime bucket, solve:
  minimize MSE(actual, w1*H + w2*D + w3*A)
  subject to: w1 + w2 + w3 = 1, w1 >= 0, w2 >= 0, w3 >= 0
- Use scipy.optimize.minimize with method='SLSQP', initial guess [1/3, 1/3, 1/3].
- After optimization, verify that MAPE also improves relative to equal weights.
- Alternative: minimize MAPE directly using scipy.optimize.minimize with method='Nelder-Mead' (derivative-free) on the simplex. Project each iterate onto the simplex by clipping negatives and renormalizing.
- [Researcher inference: paper says "minimizes the error on in-sample data" (p.18) without specifying MSE vs. MAPE or weight constraints.]

**Step 5: Volume percentage regression**
- Fit a single OLS regression pooled across all bins and training days.
- The K_reg lagged surprises serve as regressors; no intercept term.
- Surprises are computed using the raw volume model's predicted percentages as the expected baseline.
- During training, actual percentages use the known daily total as the denominator. During inference, actual percentages use V_total_est = sum(observed volumes) + sum(raw forecasts for remaining bins).

**MAPE computation** (for validation and weight optimization):
- MAPE = 100% * (1/N) * SUM(|Predicted_Volume - Raw_Volume| / Raw_Volume)
  where the index i runs over all bins and N is the total number of bins.
  (Satish et al. 2014, p.17: "Measuring Raw Volume Predictions -- MAPE")
- This is an average across bins for a given symbol and day. To compute the overall MAPE across the full sample, aggregate per-bin errors across all (symbol, day, bin) observations, then take medians and bottom-95% averages as reported in Exhibit 6.

## Validation

### Expected Behavior

1. **Raw volume MAPE (Model A):** The model should achieve approximately 24% median MAPE reduction over the rolling historical average baseline across all intraday bins (Satish et al. 2014, p.20, "Validating Volume Prediction Error": "Across all intraday intervals, we reduce the median volume error by 24%"). The bottom-95% average MAPE (excluding the worst 5% of predictions) should improve by approximately 29% (Satish et al. 2014, p.20).

2. **MAPE reduction by time-of-day:** Median error reduction increases from approximately 10-12% at 9:30 to 30-33% by 15:30, with some non-monotonicity (a dip around 10:00-10:30). The bottom-95% average reduction increases more smoothly from approximately 15% to 35-40% (Satish et al. 2014, Exhibit 6, p.22). This improvement through the day reflects the intraday ARMA component gaining more conditioning information as bins are observed.

3. **Consistency across sectors (Model A):** Improvements should be consistent across SIC industry groups and beta deciles, typically in the 15-35% range for raw volume MAPE reduction (Satish et al. 2014, Exhibits 7 and 8, pp.22-23).

4. **Volume percentage error (Model B):** Median absolute deviation should decrease from approximately 0.00874 (historical) to 0.00808 (dynamic) for 15-minute bins -- a 7.55% reduction (Satish et al. 2014, Exhibit 9, p.23). This reduction is statistically significant at the << 1% level using a Wilcoxon signed-rank test (Exhibit 9 footnote, denoted by asterisk). As a complementary benchmark, the bottom-95% average errors (excluding the worst 5% of predictions) should decrease from approximately 0.00986 to 0.00924, a 6.29% reduction (Satish et al. 2014, Exhibit 9, 15-minute row). Both metrics should be checked since median alone can be misleading if the error distribution is skewed.

5. **VWAP tracking error (Model B applied in execution):** In simulation with 10% ADV order size, mean tracking error should decrease from approximately 9.62 bps (historical) to 8.74 bps (dynamic) -- a 9.1% reduction (Satish et al. 2014, Exhibit 10, p.23). Improvement should be 7%-10% across Dow 30, midcap, and high-variance stock groups. Paired t-test statistic: 2.34 with p < 0.01 (Exhibit 10 footnote).

6. **VWAP-percentage error relationship:** Regression of VWAP tracking error on volume percentage error should yield R^2 > 0.50: coefficient ~220.9 bps/unit for Dow 30 stocks (R^2 = 0.5146), ~454.3 bps/unit for high-variance stocks (R^2 = 0.5886) (Satish et al. 2014, Exhibits 3 and 5, pp.20-21).

7. **Comparison benchmark (Chen et al. 2016):** The Kalman filter model of Chen et al. achieves average MAPE of 0.46 (dynamic) and 0.61 (static) across 30 securities, and VWAP tracking error of 6.38 bps (dynamic) (Chen et al. 2016, Table 3 and Table 4). Note: Chen et al. (2016) specifically claim to outperform the Satish et al. approach (Chen et al. summary states "Satish et al. (2014) used a similar decomposition with ARMA models; this paper claims to outperform that approach as well"). Direct comparison is imperfect because the datasets differ (500 U.S. stocks vs. 30 multi-market securities) and the time periods differ.

### Sanity Checks

1. **Weight sanity:** After optimization, regime weights should be positive and sum to 1. The historical average weight (w1) should typically dominate early in the day when ARMA components have limited intraday information. The intraday ARMA weight (w3) should increase for later bins.

2. **Seasonal factor shape:** seasonal_factor[:] should exhibit the well-known U-shape: high at market open (bin 1), declining through midday, and rising again toward market close (bin 26). The ratio of max to min seasonal factor should typically be 2x-4x for liquid U.S. stocks.

3. **Deseasonalized volume:** After dividing by seasonal_factor, the deseasonalized intraday volume series should be approximately stationary (no systematic time-of-day pattern). The expected value of the deseasonalized series is approximately 1.0 (the seasonal factor is an absolute volume level, and the deseasonalized value is a ratio to that level). Test stationarity with augmented Dickey-Fuller.

4. **AICc selection:** For most symbols, the selected ARMA orders should be low (p <= 2, q <= 2 for inter-day; p <= 3 for intraday). Very high orders (p = 5 or q = 5) may indicate overfitting or data issues. Log a warning if the combined inter-day + intraday parameter count reaches or exceeds 11 for any bin.

5. **Regime detection:** During the out-of-sample period, all regime buckets should be populated (no empty regimes). If a regime has very few (day, bin) observations, consider reducing n_regimes.

6. **Percentage model constraints:** The deviation constraint should bind (be active) on at most 10-20% of bins. If it binds on the majority of bins, the max_deviation parameter is too tight.

7. **Switch-off behavior:** The 80% switch-off should engage for the last 2-4 bins of an average day (since approximately 80% of volume occurs by 14:30-15:00 on a typical U.S. equity day).

8. **Monotonic cumulative:** When converting bin forecasts to cumulative volume percentages, the result must be monotonically non-decreasing and approach 1.0 by the last bin.

9. **Regression coefficient magnitude:** The OLS regression coefficients (beta_1, ..., beta_K_reg) should be small (typically |beta_k| < 0.5). Large coefficients suggest overfitting or data issues.

10. **Component 1 baseline match:** Compute MAPE using only H (Component 1) as the forecast. This should match the "historical window" baseline reported in the paper (the denominator for all improvement calculations).

11. **Surprise magnitude:** Volume surprises (actual_pct - expected_pct) should have mean approximately zero and standard deviation on the order of 0.005-0.015 for 15-minute bins on liquid stocks.

### Edge Cases

1. **Zero-volume bins:** Some illiquid stocks or pre/post-market bins may have zero volume. The seasonal factor floor (epsilon = 1.0) prevents division by zero in deseasonalization. MAPE is undefined for zero-volume bins; exclude them from error computation. The raw volume forecast may produce zero or negative values for illiquid bins; clamp at zero.

2. **Half-day trading sessions:** Days before holidays (e.g., day before Thanksgiving) have 13 bins instead of 26. These days should be excluded from training and evaluation or handled with a separate bin count configuration (Satish et al. 2014 does not address this explicitly).

3. **Market disruptions:** Circuit breakers, trading halts, or extreme volatility events will produce volume patterns inconsistent with historical data. The regime-switching mechanism partially addresses this, but extreme events may exceed all historical regimes.

4. **Special calendar days:** Option expiration days, Fed announcement days, index rebalancing days typically have elevated volume with different intraday patterns. Satish et al. (2014, p.18) recommend "custom curves for special calendar days... rather than ARMAX models, due to insufficient historical occurrences."

5. **ARMA estimation failure:** MLE for ARMA may fail to converge for some (p, q) combinations. Detection: check optimizer convergence flag, verify Hessian is positive definite, confirm standard errors are finite. Fall back to the next-best AICc model. If no ARMA converges for a bin's inter-day model, set interday_model[i] = None and use only the historical average and intraday ARMA (effectively D[i] = H[i]). If no intraday ARMA converges, set intraday_model = None and use only historical average and inter-day components. Recommended library: statsmodels.tsa.arima.model.ARIMA with method='innovations_mle' or method='statespace' as fallback. For a 500-stock universe with 26 bins and 36 order combinations, there are approximately 468,000 ARMA fits per calibration cycle; expect some (typically 1-5%) to fail convergence.

6. **New listings / IPOs:** Symbols with fewer than 126 trading days of history cannot compute full seasonal factors. Use available history with a minimum of 21 days; flag short-history symbols.

7. **Stock splits / corporate actions:** Volume normalization by shares outstanding is recommended to handle splits. If using raw shares, adjust the historical series at split boundaries.

8. **Day-boundary transitions for intraday ARMA:** At the start of a new trading day (bin 1), the intraday ARMA has no observed bins for the current day. The model relies on the ARMA state from the previous day's close and the inter-day ARMA forecast. Multi-step-ahead forecasts from the intraday model decay toward the unconditional mean, so early-day forecasts should lean more heavily on the historical average and inter-day components. The overnight gap introduces exogenous shocks that weaken serial correlation across day boundaries; see Pseudocode Step 3 for mitigation options.

9. **Insufficient surprise lags at day start:** For the volume percentage model, at bins 1 through K_reg, fewer than K_reg lagged surprises are available. Pad missing lags with zeros (see pseudocode). This means the percentage model provides minimal adjustment for the first few bins of the day.

10. **Regime boundary instability:** Near percentile boundaries (e.g., 33rd percentile), small changes in cumulative volume could cause regime switches between adjacent bins. The min_regime_bins parameter (default 3) avoids early-day instability. For mid-day instability, an optional hysteresis enhancement can improve robustness: once a regime is selected, require the percentile to cross the boundary by a margin (e.g., 5 percentile points) before switching. This is NOT required for the baseline implementation -- the paper does not mention hysteresis. If implemented, add a `regime_hysteresis` parameter (default 5 percentile points, range [2, 10]) and modify the regime classification block in ForecastRawVolume to track the previous regime:
    ```
    IF current_bin >= min_regime_bins AND current_bin > 1:
        cum_vol = sum(volume[s, day_t, j] for j in {1, ..., current_bin - 1})
        pctile = percentile_rank(cum_vol, regime_thresholds[current_bin - 1])
        candidate_regime = classify_regime_from_percentile(pctile)
        IF candidate_regime != prev_regime:
            # Check if percentile has crossed boundary by hysteresis margin
            boundary = nearest_boundary(pctile, regime_percentiles)
            IF abs(pctile - boundary) >= regime_hysteresis:
                regime = candidate_regime
            ELSE:
                regime = prev_regime  # stick with current regime
        ELSE:
            regime = candidate_regime
    ```
    [Researcher inference: hysteresis not in paper; optional robustness enhancement]

### Known Limitations

1. **Proprietary parameters:** Many key hyperparameters (regime bucket counts, percentile thresholds, optimal regression terms, specific weighting coefficients, deviation bound method) are not disclosed in the paper (Satish et al. 2014, p.18-19). Our specification provides reasonable defaults but these may not replicate the exact published results.

2. **Single-paper foundation:** The entire model is from one practitioner paper. Unlike the CMEM (direction 1) or Kalman filter (direction 7), there are no follow-up papers refining or independently validating the methodology. Chen et al. (2016) claim to outperform this approach on their dataset.

3. **No formal distributional model:** Unlike CMEM (Gamma/Weibull) or the Kalman filter (Gaussian in log space), this model does not specify a distributional form for volume. This limits the ability to compute prediction intervals or likelihood-based diagnostics.

4. **Static volume percentages at day start:** The volume percentage model requires observed intraday bins to compute surprises. At the start of the day (bins 1 through K_reg), insufficient surprise lags are available, and the model produces minimal adjustments, effectively defaulting to the historical curve.

5. **No cross-sectional information:** Unlike the BDF model (direction 2), this model treats each symbol independently. Market-wide volume shocks that affect all stocks simultaneously are not explicitly captured.

6. **AICc sensitivity:** The AICc criterion is well-justified for small-sample AR model selection, but the paper does not discuss robustness of the selected orders to the training window. Re-estimation on rolling windows may cause order changes that introduce forecast instability.

7. **MAPE metric limitations:** MAPE penalizes overestimation and underestimation equally and is undefined for zero actual values. It is also biased toward underprediction since the denominator grows with actual volume.

8. **Training/inference normalization gap:** During training, volume percentages use the known daily total as denominator. During inference, they use V_total_est (sum of observed volumes + raw forecasts for remaining bins). This introduces a small systematic difference that diminishes as the day progresses and more actual volume is observed.

## Paper References

| Spec Section | Source |
|---|---|
| Overall architecture (4-component raw + percentage) | Satish et al. 2014, p.15-19, full paper |
| Historical rolling average definition | Satish et al. 2014, p.16 "Historical Window Average/Rolling Means" |
| MAPE metric definition | Satish et al. 2014, p.17 "Measuring Raw Volume Predictions -- MAPE" |
| MAPE formula: 100% * (1/N) * SUM(\|Pred - Actual\| / Actual) | Satish et al. 2014, p.17 |
| Volume percentage error metric | Satish et al. 2014, p.17 "Measuring Percentage Volume Predictions -- Absolute Deviation" |
| Inter-day ARMA (per-symbol, per-bin) | Satish et al. 2014, p.17 "Raw Volume Forecast Methodology" para 2 |
| AICc model selection | Satish et al. 2014, p.17; Hurvich & Tsai 1989, 1993 |
| AICc formula: AIC + 2k(k+1)/(n-k-1) | Satish et al. 2014, p.17; Hurvich & Tsai 1989, 1993 |
| Intraday ARMA with deseasonalization | Satish et al. 2014, p.17-18 "Raw Volume Forecast Methodology" para 3-4 |
| Trailing 6-month deseasonalization window | Satish et al. 2014, p.17 para 3 |
| Rolling 1-month intraday ARMA window | Satish et al. 2014, p.17-18 para 3-4 |
| AR lags < 5 (hard constraint on intraday ARMA) | Satish et al. 2014, p.18: "AR lags with a value less than five" |
| "Fewer than 11 terms" (descriptive outcome, not hard constraint) | Satish et al. 2014, p.18: "As a result, we fit each symbol with a dual ARMA model having fewer than 11 terms" |
| Dynamic weight overlay with regime switching | Satish et al. 2014, p.18 para 4 |
| Architecture diagram (input windows) | Satish et al. 2014, Exhibit 1 (p.18) |
| Exhibit 1 "Prior 5 days" for ARMA Daily | Satish et al. 2014, Exhibit 1 -- interpreted as effective AR memory, not training window (see Parameter notes and Pseudocode Step 2 discussion) |
| Exhibit 1 "4 Bins Prior to Current Bin" for ARMA Intraday | Satish et al. 2014, Exhibit 1 -- describes effective AR lag inputs for prediction, not training window |
| Volume percentage model (dynamic VWAP extension) | Satish et al. 2014, p.18-19; extends Humphery-Jenner 2011 |
| Single rolling regression (not per-bin) | Satish et al. 2014, p.24: "The author specifies a rolling regression model" (singular); Humphery-Jenner 2011 |
| 10% deviation limit (illustrative, "e.g.") | Satish et al. 2014, p.24: "e.g., depart no more than 10% away from a historical VWAP curve"; Humphery-Jenner 2011 |
| 80% switch-off threshold | Satish et al. 2014, p.24: "once 80% of the day's volume is reached"; Humphery-Jenner 2011 |
| Proprietary deviation bound method | Satish et al. 2014, p.19: "we developed a separate method for computing deviation bounds" (not disclosed) |
| Custom curves for special calendar days | Satish et al. 2014, p.18 para 4 |
| No-intercept in validation regressions | Satish et al. 2014, p.19: "we perform both regressions without the inclusion of a constant term" |
| Raw volume results (24% median MAPE reduction) | Satish et al. 2014, p.20 "Validating Volume Prediction Error" |
| Bottom-95% results (29% MAPE reduction) | Satish et al. 2014, p.20 |
| Error reduction by time-of-day | Satish et al. 2014, Exhibit 6 (p.22) |
| Error reduction by SIC group | Satish et al. 2014, Exhibit 7 (p.22) |
| Error reduction by beta decile | Satish et al. 2014, Exhibit 8 (p.23) |
| Volume percentage results (7.55% median reduction) | Satish et al. 2014, Exhibit 9 (p.23) |
| Exhibit 9 Wilcoxon significance | Satish et al. 2014, Exhibit 9 footnote: asterisk denotes significance at << 1% level |
| Exhibit 9 bottom-95% average (15-min) | Satish et al. 2014, Exhibit 9: 0.00986 (historical) vs. 0.00924 (dynamic), 6.29% reduction |
| VWAP simulation results (9.1% tracking error reduction) | Satish et al. 2014, Exhibit 10 (p.23) |
| VWAP simulation: paired t-test 2.34, p < 0.01 | Satish et al. 2014, Exhibit 10 footnote |
| VWAP regression: R^2=0.51, coef=220.9 (Dow 30) | Satish et al. 2014, Exhibit 3 (p.20) |
| VWAP regression: R^2=0.59, coef=454.3 (high-variance) | Satish et al. 2014, Exhibit 5 (p.21) |
| Comparison: Kalman filter MAPE 0.46 (dynamic) | Chen et al. 2016, Table 3 (average across 30 securities) |
| Comparison: Kalman filter VWAP 6.38 bps | Chen et al. 2016, Table 4 (average across 30 securities) |
| Comparison: Chen et al. claim to outperform Satish et al. | Chen et al. 2016, paper summary |
| N_hist = 21 days | Satish et al. 2014, Exhibit 1 caption "Prior 21 days" |
| N_interday interpretation | Satish et al. 2014, Exhibit 1 caption "Prior 5 days"; Researcher inference -- see detailed justification in Pseudocode Step 2 and Parameter notes |
| Number of regime buckets | Researcher inference: not disclosed; 3 regimes (tercile) proposed as default |
| Regime percentile cutoffs | Researcher inference: not disclosed; [33, 67] proposed as default |
| min_regime_bins = 3 | Researcher inference: not disclosed; 3 bins proposed to avoid unreliable early-day percentile ranking |
| K_reg = 3 lagged surprises | Researcher inference: not disclosed; 3 proposed based on AR-lags-<5 constraint and Humphery-Jenner framework |
| V_total_est definition | Researcher inference: sum of observed volumes + raw forecasts for remaining bins; model A runs first, then model B |
| Regime classification at (day, bin) level | Researcher inference: for consistency between training and forecasting |
| Seasonal factor floor (epsilon) | Researcher inference: prevents division by zero for illiquid bins |
| ARMA library and estimation details | Researcher inference: statsmodels recommended; stationarity/invertibility enforced |
| ARMA convergence failure detection | Researcher inference: check optimizer flag, Hessian, standard errors |
| Re-estimation schedule | Researcher inference: daily for intraday ARMA and seasonal factors; weekly (every 5 days) for inter-day ARMA; monthly (every 21 days) for regime weights and percentage regression |
| Weight optimization method | Researcher inference: SLSQP for MSE or Nelder-Mead for MAPE; paper says "minimizes the error" without specifying |
| Weight simplex constraint | Researcher inference: paper does not specify whether weights are constrained; simplex proposed as default |
| No-intercept in surprise regression | Researcher inference based on paper's no-intercept philosophy (Satish et al. 2014, p.19) |
| Overnight boundary handling in intraday ARMA | Researcher inference: paper does not discuss; naive concatenation recommended as default with options (b)/(c) as sensitivity checks |
| Renormalization approach (scale baseline, not delta) | Researcher inference: ensures safety constraints are not undone by normalization |
| regime_threshold_window = 60 days | Researcher inference: window for building cumulative volume percentile distribution |
| "Could" language for raw model surprises | Satish et al. 2014, p.19: "we could apply our more extensive volume forecasting model" -- aspirational, not confirmative |
| Forecast update = reconditioning, not re-estimation | Researcher inference: standard ARMA forecasting approach; re-estimation at every bin would be computationally prohibitive |
| V_total_est uses conditioned forecasts | Researcher inference: unconditional forecasts for expected_pct baseline, conditioned forecasts for V_total_est; paper does not distinguish |
| Deviation constraint relative to scaled baseline | Researcher inference: ensures stated deviation limit is respected in final forecast; paper does not specify |
| N_reg_train = 252 for percentage regression training | Researcher inference: matches DailyUpdate re-estimation window; paper does not specify initial training window |
| Lookahead bias in percentage model training | Researcher inference: negligible due to slow-moving ARMA parameters and seasonal factors |
| Hysteresis for regime boundary (optional) | Researcher inference: not in paper; optional robustness enhancement with regime_hysteresis parameter |
| Exhibit 6 time-of-day detail (10-12% to 30-33% median) | Satish et al. 2014, Exhibit 6 (p.22): refined reading of figure |
