# Implementation Specification: Dual-Mode Volume Forecast (Raw + Percentage)

## Overview

This direction implements a two-model volume forecasting system from Satish, Saxena, and Palmer (2014), developed at FlexTrade Systems. The system comprises:

1. A **Raw Volume Forecast Model** that combines four components -- a rolling historical average, an inter-day ARMA model, a deseasonalized intraday ARMA model, and a dynamic regime-switching weight overlay -- to predict the absolute number of shares traded in each 15-minute bin.

2. A **Volume Percentage Forecast Model** that extends Humphery-Jenner's (2011) dynamic VWAP framework, using volume surprises (computed from the raw model) in a rolling regression to adjust participation rates for step-by-step VWAP execution.

The raw model forecasts all remaining bins simultaneously (for scheduling tools), while the percentage model forecasts one bin ahead (for step-by-step VWAP algorithms). The raw model's output feeds the percentage model's surprise signal, making them a tightly coupled pair.

The Chen, Feng, and Palomar (2016) Kalman filter model serves as a benchmark comparison; its architecture (log-space state-space decomposition) is not implemented here but informs validation targets.

## Algorithm

### Model Description

The system solves two distinct but related problems:

**Raw volume forecasting:** Given the history of bin-level volumes for a stock, predict the raw volume (number of shares) that will trade in each of the 26 fifteen-minute bins of the next trading day, and update these predictions intraday as new bins are observed. The model exploits three sources of information: (a) the typical volume for that bin over recent history (historical average), (b) day-to-day serial correlation in volume for the same bin (inter-day ARMA), and (c) within-day momentum from recently observed bins (intraday ARMA). These three forecasts are combined via optimized weights that switch across regimes defined by the day's cumulative volume percentile.

**Volume percentage forecasting:** Given a naive historical estimate of the percentage of daily volume in each bin, adjust the participation rate for the next bin based on observed volume surprises (deviations of actual volume from the historical forecast) earlier in the day. This produces a dynamic VWAP execution curve that adapts to unusual volume patterns.

**Assumptions:**
- Volume exhibits serial correlation both across days (same bin) and within days (adjacent bins). [Satish et al. 2014, p. 17]
- The intraday seasonal pattern (U-shape) is stable over a trailing six-month window. [Satish et al. 2014, p. 17]
- Regime switching based on cumulative volume percentile captures high-volume vs. low-volume day dynamics. [Satish et al. 2014, p. 18]
- Volume percentage errors are linearly related to VWAP tracking error (R^2 > 0.50). [Satish et al. 2014, Exhibits 2-5, pp. 20-21]

**Inputs:** Historical intraday volume time series at 15-minute resolution, per stock.
**Outputs:** (1) Predicted raw volume for each of the 26 bins; (2) Predicted volume percentage for the next bin (for VWAP execution).

### Pseudocode

#### Part A: Raw Volume Forecast Model

```
FUNCTION build_raw_volume_model(stock, date, historical_data):
    # === COMPONENT 1: Historical Window Average ===
    # Reference: Satish et al. 2014, p. 17, "Historical Window Average/Rolling Means"
    
    FOR each bin i in 1..I:  # I = 26 for 15-minute bins
        hist_avg[i] = arithmetic_mean(volume[stock, bin=i, day] 
                                       for day in (date - N_hist)...(date - 1))
    
    # === COMPONENT 2: Inter-day ARMA ===
    # Reference: Satish et al. 2014, p. 17, "Raw Volume Forecast Methodology" para 2
    # Fit a per-symbol, per-bin ARMA(p, q) to the daily time series of volume
    # at each bin index. Model selection via AICc.
    # 
    # The fitting window (N_interday_fit) is the number of historical days used
    # to estimate ARMA parameters. This must be substantially longer than the
    # model order to allow reliable estimation.
    # [Researcher inference: Exhibit 1 "Prior 5 days" refers to the most recent
    #  daily observations used as lagged inputs for prediction, not the fitting
    #  window. Fitting on only 5 observations is statistically infeasible for
    #  ARMA models with up to 11 parameters. The fitting window should be
    #  63-126 days (3-6 months) to ensure stable parameter estimates.]
    
    FOR each bin i in 1..I:
        daily_series_i = [volume[stock, bin=i, day]
                          for day in (date - N_interday_fit)...(date - 1)]
        
        best_aic = infinity
        FOR p in 0..5:
            FOR q in 0..5:
                # Joint constraint: combined inter-day + intraday terms < 11
                # [Researcher inference: cap of 8 terms reserves at least 2 for
                #  the intraday ARMA (e.g., AR(1) + constant). A cap of 9 would
                #  allow a constant-only intraday model, which provides no
                #  dynamic information. We prefer to guarantee at least one AR
                #  lag for intraday adaptation. The paper does not specify this
                #  budget allocation.]
                IF p + q + 1 > 8: CONTINUE
                
                model = fit_ARMA(daily_series_i, order=(p, q), include_constant=True)
                aicc = compute_AICc(model)
                IF aicc < best_aic:
                    best_aic = aicc
                    best_model_i = model
                    interday_terms_i = p + q + 1
        
        interday_forecast[i] = best_model_i.predict(steps=1)
    
    # === COMPONENT 3: Intraday ARMA ===
    # Reference: Satish et al. 2014, pp. 17-18, "Raw Volume Forecast Methodology" para 3
    # Reference: Satish et al. 2014, Exhibit 1, p. 18 -- inputs are "Current Bin,"
    #   "4 Bins Prior to Current Bin," and "Today"
    #
    # The intraday ARMA operates on the WITHIN-DAY deseasonalized bin sequence.
    # It does NOT concatenate bins across days (the overnight gap is not a
    # continuous transition). Parameters are estimated from the past month's
    # complete intraday sequences, then applied to the current day's partial
    # sequence for one-step-ahead forecasting.
    #
    # Exhibit 1 shows "Today" as a separate input label alongside "Current Bin"
    # and "4 Bins Prior to Current Bin." We interpret "Today" as the data source
    # qualifier: the AR lag inputs come from today's observed deseasonalized bins,
    # not from historical days. It is not a separate feature (e.g., today's
    # cumulative volume or date characteristics). [Researcher inference]
    
    # Step 3a: Compute seasonal factors from trailing 6-month average
    # Reference: Satish et al. 2014, p. 17, "deseasonalized by dividing by the
    #   intraday amount of volume traded in that bin over the trailing six months"
    FOR each bin i in 1..I:
        seasonal_factor[i] = arithmetic_mean(volume[stock, bin=i, day]
                                              for day in trailing_6_months)
    
    # Step 3b: Collect within-day training sequences (rolling 1-month window)
    # Reference: Satish et al. 2014, p. 18, "compute this model on a rolling
    #   basis over the most recent month"
    # Each day provides one independent intraday sequence of I deseasonalized values.
    training_sequences = []
    FOR each day d in rolling_1_month_window:  # ~21 most recent trading days
        day_seq = []
        FOR each bin i in 1..I:
            deseasonal_vol = volume[stock, bin=i, day=d] / seasonal_factor[i]
            day_seq.append(deseasonal_vol)
        training_sequences.append(day_seq)  # list of I-length sequences
    
    # Step 3c: Fit ARMA to within-day sequences
    # Reference: Satish et al. 2014, p. 18, "AR lags with a value less than five"
    # Reference: Satish et al. 2014, p. 18, "fit each symbol with a dual ARMA
    #   model having fewer than 11 terms" -- this is a JOINT constraint on the
    #   combined inter-day + intraday ARMA system.
    #
    # [Researcher inference on fitting approach: The paper does not specify how
    #  day boundaries are handled during ARMA estimation. We use approach (b):
    #  concatenate the within-day sequences but insert break points at day
    #  boundaries so the likelihood computation resets at each new day. This
    #  prevents lag-1 from connecting the last bin of one day to the first bin
    #  of the next. In statsmodels, this can be approximated by setting the
    #  first p observations of each day's sequence as initial conditions rather
    #  than likelihood contributions. Alternatively, fit ARMA parameters on each
    #  day independently and average the resulting coefficients across days.]
    
    # Determine maximum intraday terms given inter-day model size
    # Joint constraint: interday_terms + intraday_terms < 11
    #
    # [Researcher inference: The paper's "fewer than 11 terms" is per-symbol,
    #  but the inter-day ARMA is per-bin (26 separate models per stock) while
    #  the intraday ARMA is a single model per stock. We use the maximum
    #  inter-day term count across all bins as the binding constraint, which is
    #  the most conservative interpretation: it ensures the joint constraint is
    #  satisfied for the worst-case bin. An alternative is to use the median
    #  inter-day term count, or to apply the constraint per-bin (allowing a
    #  more complex intraday model when most bins have simple inter-day models).
    #  The conservative approach is safer for a first implementation.]
    max_interday_terms = max(interday_terms_i for i in 1..I)
    max_intraday_terms = 10 - max_interday_terms  # total < 11, so max = 10 - interday
    
    best_aic = infinity
    FOR p in 0..4:  # AR lags < 5 per paper
        FOR q in 0..4:
            IF p + q + 1 > max_intraday_terms: CONTINUE
            
            model = fit_ARMA_with_day_breaks(
                training_sequences,  # list of day sequences
                order=(p, q),
                include_constant=True
            )
            aicc = compute_AICc(model)
            IF aicc < best_aic:
                best_aic = aicc
                intraday_model = model
    
    # Step 3d: Forecast using current day's observed bins
    # Reference: Exhibit 1, p. 18 -- "Current Bin" and "4 Bins Prior to Current Bin"
    # The intraday ARMA uses the most recent min(p, j) bins as its AR context,
    # where j is the number of bins observed so far today and p is the AR order.
    
    # Deseasonalize today's observed bins
    today_deseasonal = []
    FOR each bin i in 1..j:  # j = number of bins observed so far
        today_deseasonal.append(volume[stock, bin=i, day=date] / seasonal_factor[i])
    
    # Predict forward and re-seasonalize
    # [Researcher inference: recursive multi-step ARMA predictions degrade with
    #  forecast horizon. For bins far from the last observation, the intraday
    #  ARMA forecast converges to the unconditional mean and adds little value
    #  beyond the historical average and inter-day components. The paper
    #  acknowledges this indirectly: "techniques that predict only the next
    #  interval will perform better than those attempting to predict volume
    #  percentages for the remainder of the trading day" (p. 18). The regime
    #  weights may naturally down-weight the intraday component for distant bins
    #  if trained on data that reflects this degradation. A developer may also
    #  consider capping the intraday forecast horizon (e.g., only use intraday
    #  ARMA for the next 4-5 bins and fall back to the unconditional mean for
    #  more distant bins), though this is not specified in the paper.]
    FOR each bin i in (j+1)..I:
        # The ARMA uses bins max(1, i-p)..i-1 as AR context
        deseasonal_pred = intraday_model.predict_one_step(
            recent_observations=today_deseasonal[max(0, len(today_deseasonal)-p):],
            steps_ahead=(i - j)
        )
        intraday_forecast[i] = deseasonal_pred * seasonal_factor[i]
        # Append prediction to today_deseasonal for multi-step recursive forecasting
        today_deseasonal.append(deseasonal_pred)
    
    # For bins already observed today, use actual volume
    FOR each bin i in 1..j:
        intraday_forecast[i] = volume[stock, bin=i, day=date]
    
    # === COMPONENT 4: Dynamic Weight Overlay with Regime Switching ===
    # Reference: Satish et al. 2014, p. 18, "Raw Volume Forecast Methodology" para 4
    
    # Step 4a: Determine current regime based on cumulative volume percentile
    IF j > 0:
        cumulative_vol_today = sum(volume[stock, bin=k, day=date] for k in 1..j)
        percentile = compute_historical_percentile(cumulative_vol_today, 
                                                    bin_index=j, stock=stock)
        regime = assign_regime(percentile, regime_thresholds)
    ELSE:
        regime = default_regime  # pre-market, use unconditional weights
    
    # Step 4b: Look up pre-trained weights for this regime
    w = regime_weights[regime]  # w = (w_hist, w_interday, w_intraday)
    
    # Step 4c: Combine forecasts
    FOR each bin i in (j+1)..I:
        raw_forecast[i] = w.hist * hist_avg[i] 
                        + w.interday * interday_forecast[i] 
                        + w.intraday * intraday_forecast[i]
    
    RETURN raw_forecast


FUNCTION train_regime_weights(stock, in_sample_data):
    # Reference: Satish et al. 2014, p. 18, para 4
    # Train separate weight sets for different volume percentile regimes.
    
    # Step 1: Define regime boundaries via percentile cutoffs
    # [Researcher inference: the paper does not disclose specific cutoffs.
    #  Reasonable starting point: quartile-based, i.e., 4 regimes at
    #  percentiles 0-25, 25-50, 50-75, 75-100. Grid search over
    #  number of regimes {3, 4, 5} and cutoff positions.]
    
    regime_thresholds = grid_search_regime_cutoffs(in_sample_data)
    
    # Step 2: For each regime, optimize weights to minimize MAPE
    FOR each regime r:
        subset = filter_data_by_regime(in_sample_data, r, regime_thresholds)
        
        # Minimize MAPE over the subset
        # [Researcher inference: Non-negativity constraints are not specified
        #  in the paper (p. 18 says only "minimizes the error on in-sample
        #  data"). We add them because negative weights would invert a
        #  component's forecast, which is not physically sensible.
        #  We do NOT constrain weights to sum to 1. Unconstrained-scale
        #  weights allow the combination to compensate for systematic bias
        #  in individual components (e.g., if all three underpredict).
        #  Since the paper optimizes on MAPE, which penalizes both over- and
        #  under-prediction, unconstrained positive weights are plausible.
        #  If overfitting is observed, add a sum-to-1 constraint as a
        #  regularization measure.]
        #
        # [Researcher inference on optimizer: MAPE is non-differentiable due
        #  to the absolute value and can have numerical issues when predicted
        #  values are near zero. Use a derivative-free optimizer
        #  (Nelder-Mead or Powell) rather than gradient-based SLSQP.
        #  Alternatively, grid search over a discretized weight simplex is
        #  viable given only 3 weights per regime. Exclude or floor bins
        #  with near-zero volume when computing MAPE to avoid division
        #  instability.]
        
        w_opt = minimize(
            objective = mean_absolute_percentage_error,
            variables = (w_hist, w_interday, w_intraday),
            constraints = [w_hist >= 0, w_interday >= 0, w_intraday >= 0],
            method = 'Nelder-Mead',  # derivative-free
            initial_guess = (1/3, 1/3, 1/3),
            exclude_bins_with_volume_below = min_volume_floor
        )
        regime_weights[r] = w_opt
    
    RETURN regime_weights, regime_thresholds
```

#### Part B: Volume Percentage Forecast Model

```
FUNCTION predict_volume_percentage(stock, date, bin_j, raw_model):
    # Reference: Satish et al. 2014, pp. 18-19, "Volume Percentage Forecast Methodology"
    # Extends Humphery-Jenner (2011) dynamic VWAP approach.
    
    # === Step 1: Compute historical volume percentages ===
    FOR each bin i in 1..I:
        hist_pct[i] = arithmetic_mean(
            volume[stock, bin=i, day=d] / daily_total_volume[stock, day=d]
            for d in trailing_N_days
        )
    
    # === Step 2: Compute volume surprises using the raw volume model ===
    # Reference: Satish et al. 2014, p. 19, "Volume Percentage Forecast Methodology"
    # The raw volume model provides the base forecast from which
    # "surprises" (deviations) are computed.
    
    FOR each observed bin k in 1..j:
        raw_forecast_k = raw_model.predict(stock, date, bin=k)
        actual_vol_k = volume[stock, bin=k, day=date]
        surprise[k] = actual_vol_k - raw_forecast_k
        # Normalize surprise as percentage of historical forecast
        surprise_pct[k] = surprise[k] / raw_forecast_k
    
    # === Step 3: Rolling regression of surprises ===
    # Reference: Satish et al. 2014, p. 19
    # Use recent surprises to predict the adjustment for the next bin.
    #
    # [Researcher inference: the paper does not disclose the exact regression
    #  form. We specify OLS without intercept, consistent with p. 19's
    #  statement "we perform both regressions without the inclusion of a
    #  constant term." While that statement directly refers to the VWAP-error
    #  regressions (volume pct error vs. VWAP tracking error), the paper's
    #  general design philosophy favors no-intercept models: "our model does
    #  not assume that there is a positive amount of VWAP error if our volume
    #  predictions are 100% accurate" (p. 19). Applying no-intercept to the
    #  surprise regression is consistent: zero surprise should produce zero
    #  adjustment.
    #
    #  The regression is re-estimated daily using the prior N_regression_fit
    #  days of complete intraday surprise sequences.]
    
    # Step 3a: Training -- estimate regression coefficients
    # For each historical day d in the training window, compute the full
    # intraday surprise sequence surprise_pct_d[1..I] using that day's
    # raw model forecasts vs. actual volumes.
    # Pool all (predictor, response) pairs across training days:
    #   For each bin k = L+1..I on each training day d:
    #     X_row = [surprise_pct_d[k-1], surprise_pct_d[k-2], ..., surprise_pct_d[k-L]]
    #     y_val = surprise_pct_d[k]
    # Fit OLS without intercept: y = X * beta
    
    training_X = []
    training_y = []
    FOR each day d in (date - N_regression_fit)...(date - 1):
        # Compute full-day surprise sequence for historical day d
        surprise_pct_d = []
        FOR each bin k in 1..I:
            raw_pred_k = raw_model.historical_predict(stock, day=d, bin=k)
            actual_k = volume[stock, bin=k, day=d]
            surprise_pct_d.append((actual_k - raw_pred_k) / raw_pred_k)
        
        # Create regression samples from this day's sequence
        FOR each bin k in (L_optimal + 1)..I:
            x_row = [surprise_pct_d[k - lag] for lag in 1..L_optimal]
            training_X.append(x_row)
            training_y.append(surprise_pct_d[k])
    
    # Fit OLS without intercept
    # [Researcher inference: OLS without intercept, consistent with paper's
    #  preference stated on p. 19. Using numpy: beta = lstsq(X, y)]
    beta = fit_ols_no_intercept(X=training_X, y=training_y)
    # beta has shape (L_optimal,)
    
    # Step 3b: Prediction -- apply coefficients to today's surprises
    IF j >= L_optimal:
        x_today = [surprise_pct[j - lag + 1] for lag in 1..L_optimal]
        predicted_adjustment = dot(beta, x_today)
    ELSE:
        # Not enough observed bins for full lag vector; use zero adjustment
        predicted_adjustment = 0.0
    
    # === Step 4: Apply adjustment with safety constraints ===
    # Reference: Satish et al. 2014, p. 24, referencing Humphery-Jenner (2011)
    # "depart no more than 10% away from a historical VWAP curve"
    # "once 80% of the day's volume is reached, return to a historical approach"
    
    raw_pct_forecast = hist_pct[j + 1]
    adjusted_pct = raw_pct_forecast * (1 + predicted_adjustment)
    
    # Safety constraint 1: Deviation limit
    # No more than 10% departure from historical VWAP curve
    max_deviation = 0.10 * hist_pct[j + 1]
    adjusted_pct = clip(adjusted_pct, 
                        hist_pct[j + 1] - max_deviation, 
                        hist_pct[j + 1] + max_deviation)
    
    # Safety constraint 2: Switch-off threshold
    # Revert to historical curve once 80% of daily volume is reached
    cumulative_pct = sum(actual_pct[k] for k in 1..j)
    IF cumulative_pct >= 0.80:
        adjusted_pct = hist_pct[j + 1]
    
    # === Step 5: Renormalize remaining percentages ===
    # [Researcher inference: The paper (p. 15) states "the day's forecasts
    #  must total to 100% to be meaningful" but does not describe how to
    #  redistribute after applying the surprise-based adjustment. The
    #  proportional scaling approach below ensures the constraint is
    #  satisfied while preserving the relative shape of the historical curve
    #  for unadjusted bins. This is the simplest approach consistent with
    #  the paper's constraint.]
    
    # Ensure remaining percentages sum to (1 - cumulative_pct)
    remaining = 1.0 - cumulative_pct
    remaining_hist = sum(hist_pct[k] for k in (j+1)..I)
    
    # Redistribute: keep adjusted_pct for bin j+1, scale the rest proportionally
    pct_forecast[j + 1] = adjusted_pct
    scale_factor = (remaining - adjusted_pct) / (remaining_hist - hist_pct[j + 1])
    FOR each bin i in (j+2)..I:
        pct_forecast[i] = hist_pct[i] * scale_factor
    
    RETURN pct_forecast
```

#### Part C: Corrected AIC (AICc) Computation

```
FUNCTION compute_AICc(model):
    # Reference: Hurvich and Tsai (1989, 1993), cited in Satish et al. 2014, p. 17
    # AICc adds a penalty term to AIC for small sample sizes.
    
    n = number_of_observations
    k = number_of_estimated_parameters  # includes constant term
    aic = -2 * log_likelihood + 2 * k
    aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
    
    RETURN aicc
```

#### Part D: Volume Percentage Error Metric

```
FUNCTION compute_volume_percentage_error(predicted_pct, actual_pct, I):
    # Reference: Satish et al. 2014, p. 17, "Measuring Percentage Volume
    #   Predictions -- Absolute Deviation"
    # This is mean absolute deviation (MAD), NOT mean absolute percentage error.
    # The percentage predictions are already normalized (sum to ~1), so no
    # further normalization by actual values is needed.
    
    error = (1 / I) * sum(|predicted_pct[i] - actual_pct[i]| for i in 1..I)
    
    RETURN error
```

#### Part E: VWAP Execution Simulation

```
FUNCTION simulate_vwap_execution(stock, date, volume_pct_model):
    # Reference: Satish et al. 2014, pp. 19, 23, Exhibit 10
    
    order_size = 0.10 * adv_30day[stock]  # 10% of 30-day ADV
    
    executed_shares = 0
    executed_cost = 0.0
    
    FOR each bin i in 1..I:
        # Dynamic: recompute participation using volume_pct_model
        pct = volume_pct_model.predict(stock, date, bin_j=i)
        shares_this_bin = pct[i] * order_size
        
        # Adjust for what's left
        shares_remaining = order_size - executed_shares
        shares_this_bin = min(shares_this_bin, shares_remaining)
        
        # Execute at bin's VWAP (approximated by last transaction price)
        executed_cost += shares_this_bin * price[stock, bin=i, day=date]
        executed_shares += shares_this_bin
    
    executed_avg_price = executed_cost / executed_shares
    market_vwap = compute_market_vwap(stock, date)
    
    tracking_error_bps = abs(market_vwap - executed_avg_price) / market_vwap * 10000
    
    RETURN tracking_error_bps
```

### Data Flow

```
Input: Historical volume time series V[stock, bin, day]
       shape: (num_stocks, I=26, num_days)

                    +------------------+
                    |  Historical Data |
                    | V[s, i, d]       |
                    +--------+---------+
                             |
              +--------------+--------------+
              |              |              |
              v              v              v
    +------------------+ +------------------+ +------------------+
    | Component 1:     | | Component 2:     | | Component 3:     |
    | Hist. Window Avg | | Inter-day ARMA   | | Intraday ARMA    |
    | N_hist=21 days   | | Fit: 63-126 days | | Fit: 1 month     |
    | shape: (I,)      | | Predict: 5-day   | | Within-day only  |
    |                  | | lag input         | | AR context: up   |
    |                  | | shape: (I,)      | | to 4 prior bins  |
    |                  | | Per-bin: up to 8 | | ("Today" = data  |
    |                  | | terms (Res. inf.)| | source qualifier)|
    |                  | |                  | | shape: (I,)      |
    +--------+---------+ +--------+---------+ +--------+---------+
              |              |              |
              |              |     Joint constraint:
              |              |     interday_terms + intraday_terms < 11
              |              |     (per-symbol; binding = max across bins)
              |              |              |
              +--------------+--------------+
                             |
                             v
                  +---------------------+
                  | Component 4:        |
                  | Dynamic Weights     |
                  | w = f(regime)       |
                  | shape: (3,) per     |
                  |         regime      |
                  +----------+----------+
                             |
                             v
                  +---------------------+
                  | Raw Volume Forecast |
                  | shape: (I,)         |
                  +----------+----------+
                             |
                   +---------+---------+
                   |                   |
                   v                   v
         +------------------+  +------------------+
         | Output 1:        |  | Volume Surprise  |
         | Raw vol forecast |  | = actual - pred  |
         | (all bins)       |  | shape: (j,)      |
         +------------------+  +--------+---------+
                                        |
                                        v
                               +------------------+
                               | Rolling Regress. |
                               | OLS, no intercept|
                               | L lagged surprise|
                               | terms; fit on    |
                               | N_regression_fit |
                               | days (Res. inf.) |
                               +--------+---------+
                                        |
                                        v
                               +------------------+
                               | Safety Clipping  |
                               | +/- 10% limit    |
                               | 80% switch-off   |
                               +--------+---------+
                                        |
                                        v
                               +------------------+
                               | Renormalize      |
                               | (Researcher inf.)|
                               +--------+---------+
                                        |
                                        v
                               +------------------+
                               | Output 2:        |
                               | Volume % forecast|
                               | (next bin)       |
                               +------------------+
```

**Type specifications:**

| Stage | Shape | Type | Notes |
|-------|-------|------|-------|
| Input volume | (num_stocks, 26, num_days) | float64 | Raw share count |
| Seasonal factors | (26,) per stock | float64 | 6-month trailing average |
| Historical average | (26,) per stock | float64 | N_hist-day rolling mean |
| Inter-day ARMA forecast | (26,) per stock | float64 | One per bin per stock |
| Intraday ARMA forecast | (26,) per stock | float64 | Deseasonalized then re-seasonalized |
| Regime weights | (num_regimes, 3) | float64 | (w_hist, w_interday, w_intraday); non-negative, no sum-to-1 constraint |
| Raw forecast | (26,) per stock | float64 | Weighted combination |
| Volume surprises | (j,) per stock | float64 | Observed bins so far |
| Surprise regression beta | (L_optimal,) per stock | float64 | OLS coefficients, no intercept |
| Volume percentage | (26,) per stock | float64 | Sums to 1.0 within day |
| Volume pct error | scalar per stock-day | float64 | MAD (not MAPE) |

### Variants

**Implemented variant:** The full dual-model system as described in Satish et al. (2014), Section "Raw Volume Forecast Methodology" (pp. 17-18) and "Volume Percentage Forecast Methodology" (pp. 18-19). This is the most complete version presented in the paper.

**Not implemented:**
- Static VWAP (no intraday updating) -- the paper explicitly argues dynamic is superior (Satish et al. 2014, p. 18).
- ARMAX with calendar event dummies -- the paper recommends custom curves for special days instead, due to insufficient event occurrences for reliable estimation (Satish et al. 2014, p. 18).
- 5-minute and 30-minute bin variants -- 15-minute shows the largest improvement (Exhibit 9, p. 23: 7.55% median reduction vs. 2.25% and 2.95% for 5- and 30-minute).

**Rationale:** The 15-minute bin dynamic model is the paper's primary contribution and the most thoroughly validated variant, tested on 500 stocks over 250 out-of-sample days.

## Parameters

| Parameter | Description | Recommended Value | Sensitivity | Range |
|-----------|-------------|-------------------|-------------|-------|
| I | Number of intraday bins | 26 (15-min bins, 6.5-hour day) | Low (structural) | {13, 26, 78} for 30/15/5 min |
| N_hist | Historical window for rolling mean (days) | 21 days [Researcher inference from Exhibit 1 "Prior 21 days" label, p. 18] | Medium -- affects baseline stability | 10-60 days |
| N_interday_fit | Fitting window for inter-day ARMA parameters (days) | 63 days (3 months) [Researcher inference: must be substantially longer than max model order; 5 observations from Exhibit 1 are prediction inputs, not fitting window] | Medium -- too short gives unstable parameters, too long adds structural breaks | 42-126 days |
| N_interday_predict | Number of recent daily observations used as ARMA lag inputs for prediction | 5 days [Exhibit 1, p. 18, "Prior 5 days" label for ARMA Daily] | Low -- determined by AR order p | max(p, 5) |
| N_intraday_fit | Fitting window for intraday ARMA (trading days) | 21 days (~1 month) [Satish et al. 2014, p. 18, "rolling basis over the most recent month"] | Medium | 10-40 trading days |
| N_seasonal | Trailing window for seasonal factors | 126 trading days (~6 months) [Satish et al. 2014, p. 17, "trailing six months"] | Low -- seasonal pattern is stable | 63-252 trading days |
| p_max_interday | Maximum AR order for inter-day ARMA | 5 [Satish et al. 2014, p. 17, "all values of p and q lags through five"] | Low -- AICc handles selection | Fixed at 5 |
| q_max_interday | Maximum MA order for inter-day ARMA | 5 [Satish et al. 2014, p. 17] | Low -- AICc handles selection | Fixed at 5 |
| p_max_intraday | Maximum AR order for intraday ARMA | 4 [Satish et al. 2014, p. 18, "AR lags with a value less than five"] | Low -- AICc handles selection | 3-5 |
| q_max_intraday | Maximum MA order for intraday ARMA | 4 | Low | 3-5 |
| max_dual_terms | Maximum total terms across BOTH the inter-day and intraday ARMA models combined | 11 (including constants) [Satish et al. 2014, p. 18, "fit each symbol with a dual ARMA model having fewer than 11 terms"] | Low -- constraint is structural | 8-15 |
| max_interday_budget | Maximum terms allowed for any single inter-day ARMA model | 8 [Researcher inference: reserves at least 2 terms for the intraday ARMA so it has at least one AR lag + constant; paper does not specify this budget split] | Low | 7-9 |
| num_regimes | Number of volume percentile regimes | 4 [Researcher inference: paper does not disclose; quartile-based is a reasonable starting point] | High -- key design choice | 2-6 |
| regime_thresholds | Percentile cutoffs for regime switching | [25, 50, 75] for 4 regimes [Researcher inference: equally spaced; grid search recommended] | High | Determined by grid search |
| deviation_limit | Max allowed departure from historical VWAP curve (pct model) | 0.10 (10%) [Satish et al. 2014, p. 24, referencing Humphery-Jenner 2011] | Medium -- too tight prevents adaptation, too loose adds risk | 0.05-0.20 |
| switchoff_threshold | Cumulative volume fraction at which to revert to historical curve (pct model) | 0.80 [Satish et al. 2014, p. 24, referencing Humphery-Jenner 2011] | Medium | 0.70-0.90 |
| L_optimal | Number of lagged surprise terms in rolling regression (pct model) | 1-3 [Researcher inference: paper does not disclose; determine by in-sample grid search] | Medium | 1-5 |
| N_pct_hist | Historical window for volume percentage baseline | 21 days [Researcher inference: same as N_hist] | Medium | 10-60 days |
| N_regression_fit | Training window for surprise regression (days of historical complete intraday surprise sequences) | 63 days [Researcher inference: provides ~63 * 26 = 1638 training samples when L=1, sufficient for stable OLS estimates; paper does not specify] | Medium -- too short gives noisy beta, too long misses regime changes | 21-126 days |
| min_volume_floor | Minimum volume threshold for including a bin in MAPE computation | 100 shares [Researcher inference: prevents division-by-zero in MAPE; paper does not specify] | Low | 50-500 shares |

### Initialization

1. **Seasonal factors:** Compute the arithmetic mean of volume in each bin over the trailing 6 months of data. If fewer than 6 months are available, use all available data (minimum 63 trading days). [Satish et al. 2014, p. 17, "deseasonalized by dividing by the intraday amount of volume traded in that bin over the trailing six months"]

2. **Historical averages:** Compute the arithmetic mean of volume in each bin over the prior N_hist days. Exhibit 1 (p. 18) shows 21 days as the lookback for the historical window component. [Satish et al. 2014, p. 17]

3. **Inter-day ARMA models:** For each stock and each bin index, fit ARMA(p,q) models to the daily volume series at that bin, selecting the best (p,q) by AICc. The fitting window (N_interday_fit) should be 63-126 trading days to allow reliable parameter estimation. The "Prior 5 days" label in Exhibit 1 (p. 18) refers to the most recent daily observations used as lagged inputs for the ARMA's one-step-ahead prediction, not the parameter estimation window. [Satish et al. 2014, pp. 17-18, Exhibit 1 p. 18. Researcher inference: fitting on 5 observations is infeasible for models with up to 11 parameters.]

4. **Intraday ARMA model:** Fit on a rolling 1-month window of within-day deseasonalized intraday sequences. Each complete trading day provides one sequence of I=26 deseasonalized bin values. Parameters are estimated with day-boundary awareness (the likelihood resets at each day boundary so lag-1 does not cross the overnight gap). At the start of each trading day, the model uses only data up to the previous day's close. During the day, observed bins are fed in for dynamic one-step-ahead prediction, using up to 4 prior bins as AR context (consistent with Exhibit 1's "4 Bins Prior to Current Bin" input). [Satish et al. 2014, pp. 17-18, Exhibit 1 p. 18]

5. **Regime weights:** Train on the first year of data (in-sample period). Optimize weights by minimizing MAPE for each regime bucket separately, using a derivative-free optimizer (Nelder-Mead). Initialize weight optimization with equal weights (1/3, 1/3, 1/3). Weights are constrained to be non-negative but are not constrained to sum to 1. [Satish et al. 2014, p. 18. Researcher inference: non-negativity, no sum-to-1, optimizer choice, and initialization are not specified in the paper.]

6. **Volume percentage baseline:** Compute historical volume percentages as the average of volume[bin_i] / daily_total across the prior N_pct_hist days. [Satish et al. 2014, p. 18]

7. **Surprise regression coefficients:** Estimate OLS regression coefficients (without intercept) by pooling intraday surprise sequences from the prior N_regression_fit trading days. Each day contributes (I - L_optimal) predictor-response pairs (one per bin from L_optimal+1 to I). Re-estimate daily before market open using the most recent N_regression_fit complete days. [Researcher inference: the paper does not specify the regression form, training window, or update frequency; OLS without intercept is consistent with the paper's general preference (p. 19).]

### Calibration

The calibration procedure has three phases:

**Phase 1: Component model fitting (daily rolling)**

1. Each trading day before market open:
   a. Update seasonal factors using trailing 6-month data. [Satish et al. 2014, p. 17]
   b. Update historical window averages using prior N_hist days. [Satish et al. 2014, p. 17]
   c. Refit inter-day ARMA(p,q) models for each bin using AICc selection on the trailing N_interday_fit days. The joint term constraint (inter-day + intraday < 11 terms) must be enforced during model selection, with a per-bin inter-day cap of 8 terms to reserve budget for the intraday model. [Satish et al. 2014, pp. 17-18. Researcher inference: budget split.]
   d. Refit intraday ARMA model on rolling 1-month of within-day deseasonalized sequences, with day-boundary-aware likelihood estimation. [Satish et al. 2014, pp. 17-18]
   e. Re-estimate surprise regression coefficients using the prior N_regression_fit days of complete intraday surprise sequences (OLS without intercept). [Researcher inference]

2. During the trading day:
   a. After each bin observation, feed the new deseasonalized observation into the intraday ARMA and predict the next bin. The ARMA uses up to 4 prior bins as AR context (from Exhibit 1, "4 Bins Prior to Current Bin"). Note that recursive multi-step predictions degrade with horizon; for distant bins the intraday ARMA converges to the unconditional mean. [Satish et al. 2014, Exhibit 1, p. 18. Researcher inference: degradation note.]
   b. Recompute the cumulative volume percentile and select the appropriate regime.
   c. Apply the corresponding regime weights to produce updated forecasts.

**Phase 2: Weight optimization (periodic, e.g., monthly or quarterly)**

1. Collect a validation dataset of (forecast_components, actual_volume) tuples.
2. Partition the data by cumulative volume percentile into regime buckets.
3. For each regime bucket, solve:
   ```
   minimize over (w1, w2, w3):
       MAPE = mean( |w1*hist + w2*interday + w3*intraday - actual| / actual )
   subject to:
       w1, w2, w3 >= 0
   using:
       Nelder-Mead or Powell (derivative-free, since MAPE is non-differentiable)
       Exclude bins with actual volume < min_volume_floor from MAPE computation
   ```
   [Satish et al. 2014, p. 18. Researcher inference: the paper describes optimizing weights on in-sample data but does not specify the exact optimization procedure, constraints, or handling of near-zero volume bins.]

4. Grid search over regime configuration:
   - Number of regimes: {3, 4, 5}
   - Cutoff positions: equally spaced, or data-driven quantiles
   - Select configuration minimizing out-of-sample MAPE on a held-out validation period.
   [Researcher inference: not specified in paper; necessary because regime thresholds are proprietary.]

**Phase 3: Volume percentage model calibration (periodic)**

1. Identify optimal number of lagged surprise terms L via in-sample cross-validation over L in {1, 2, 3, 4, 5}. For each candidate L, fit the OLS surprise regression on a training subset and evaluate MAD on a held-out validation subset.
2. Train rolling regression coefficients on in-sample data using the selected L, pooling intraday surprise sequences and fitting OLS without intercept.
3. Validate that deviation limits and switch-off threshold produce stable behavior.
[Satish et al. 2014, pp. 18-19. Researcher inference on cross-validation procedure and OLS specification.]

## Validation

### Expected Behavior

**Raw volume forecasting (15-minute bins, 500 U.S. stocks, 250 out-of-sample days):**
- Median MAPE reduction vs. historical window baseline: 24%. [Satish et al. 2014, p. 20, "Validating Volume Prediction Error"]
- Bottom-95% mean MAPE reduction: 29%. [Satish et al. 2014, p. 20]
- Error reduction is consistent across all 26 intraday bins (Exhibit 6, p. 22), with median reduction ranging from ~10% (early morning) to ~33% (late afternoon). [Satish et al. 2014, Exhibit 6, p. 22]
- Error reduction is consistent across SIC industry groups (~15-35% median reduction) and beta deciles (~20-35% median reduction). [Satish et al. 2014, Exhibits 7-8, pp. 22-23]

**Volume percentage forecasting (15-minute bins):**
- Metric: mean absolute deviation (MAD), defined as: Error = (1/N) * sum_i |Predicted_Percentage_i - Actual_Percentage_i|, where i runs over all bins and N is the total number of bins. This is NOT MAPE -- percentage predictions are already normalized, so no division by actual is needed. [Satish et al. 2014, p. 17, "Measuring Percentage Volume Predictions -- Absolute Deviation"]
- Median absolute error: 0.00874 (historical) vs. 0.00808 (dynamic) -- 7.55% reduction, significant at << 1% (Wilcoxon signed-rank test). [Satish et al. 2014, Exhibit 9, p. 23]
- Bottom-95% average absolute error: 0.00986 (historical) vs. 0.00924 (dynamic) -- 6.29% reduction. [Satish et al. 2014, Exhibit 9, p. 23]

**VWAP tracking error (simulation, 600+ orders):**
- Mean VWAP tracking error: 9.62 bps (historical) vs. 8.74 bps (dynamic) -- 9.1% reduction, significant at p < 0.01 (paired t-test). [Satish et al. 2014, Exhibit 10, p. 23]
- Per-category reductions: 7-10% across Dow 30, midcap, and high-variance groups. [Satish et al. 2014, p. 23]

**Comparison with Chen et al. (2016) Kalman filter:**
- Chen et al. report average dynamic MAPE of 0.46 (robust Kalman) vs. 0.65 (CMEM) vs. 1.28 (rolling mean) on 30 securities across multiple exchanges. [Chen et al. 2016, Section 4.2, Table 3, p. 10]
- Chen et al. report dynamic VWAP tracking error of 6.38 bps (robust Kalman) vs. 7.48 bps (rolling mean). [Chen et al. 2016, Table 4, p. 12]
- Direct comparison is imprecise because datasets differ (500 U.S. stocks vs. 30 multi-market securities), but the Satish model's 24% MAPE reduction and 9.1% VWAP reduction should be broadly comparable to Kalman filter performance.

### Sanity Checks

1. **Historical average baseline:** With regime weights all set to (1.0, 0.0, 0.0), the model should exactly reproduce the rolling historical average. Verify MAPE matches the baseline. [Researcher inference]

2. **Seasonal pattern recovery:** The seasonal factors should exhibit a U-shape (high volume at open and close, lower midday). Plot seasonal_factor[i] for i=1..26 and verify U-shape. [Well-established empirical fact, cited across all papers]

3. **ARMA order stability:** For liquid stocks, the selected ARMA orders should be low (p,q in {0,1,2} for most bins). If AICc consistently selects high orders (4,5), the data may have issues. [Satish et al. 2014, p. 18: "AR coefficients quickly decayed" and "lags less than five"]

4. **Joint term constraint:** Verify that for every stock, the sum of inter-day ARMA terms (p + q + 1) and intraday ARMA terms (p + q + 1) is strictly less than 11. [Satish et al. 2014, p. 18]

5. **Weight non-negativity:** All regime weights should be non-negative. If the optimizer returns negative weights despite the constraint, the optimization may have failed. [Researcher inference; the paper does not specify this constraint but negative weights are not sensible]

6. **Volume percentage sum:** For any day, the predicted volume percentages across all bins should sum to approximately 1.0 (within floating-point tolerance). [Structural constraint, Satish et al. 2014, p. 15]

7. **Deviation limit effective:** When the deviation limit is active, adjusted_pct should be within +/- 10% of hist_pct for each bin. [Satish et al. 2014, p. 24]

8. **Switch-off behavior:** After cumulative observed volume exceeds 80% of daily volume, the percentage model should revert to the historical curve. Verify this by checking that adjusted_pct == hist_pct for late bins on high-volume days. [Satish et al. 2014, p. 24]

9. **Monotonic improvement with components:** Adding each component should improve (or not significantly worsen) forecast accuracy:
   - Historical average alone: baseline MAPE
   - + Inter-day ARMA: MAPE should decrease
   - + Intraday ARMA: MAPE should decrease further
   - + Regime switching: MAPE should decrease further
   [Researcher inference based on the paper's design philosophy]

10. **Intraday ARMA day-boundary check:** Verify that the intraday ARMA's lag-1 residual does not correlate with the previous day's last bin residual. If it does, day boundaries are not being handled correctly. [Researcher inference]

11. **Surprise regression coefficients sign:** For L=1, the regression coefficient beta[0] should typically be positive (positive serial correlation in surprises: if volume was higher than expected in the previous bin, it is likely higher in the next). If consistently negative across stocks, investigate data issues. [Researcher inference]

### Edge Cases

1. **Zero-volume bins:** Some bins may have zero traded volume, especially for illiquid stocks or around market open/close. The intraday ARMA deseasonalization divides by the seasonal factor; if seasonal_factor[i] = 0, this produces division by zero. **Handling:** Replace zero seasonal factors with a small positive value (e.g., the minimum non-zero seasonal factor across all bins) or exclude zero-volume bins from the intraday ARMA fitting. Similarly, exclude bins with volume below min_volume_floor from MAPE computation to avoid division instability. [Researcher inference; the paper works on top 500 stocks by dollar volume where this is rare]

2. **Insufficient history:** If fewer than N_hist days of data are available, fall back to whatever is available. If fewer than 6 months for seasonal factors, use all available data. Minimum viable: 63 trading days for seasonal factors, N_interday_fit days for inter-day ARMA. For the surprise regression, if fewer than N_regression_fit days are available, use all available complete days (minimum ~21 days to avoid severely underdetermined OLS). [Researcher inference]

3. **ARMA convergence failure:** Maximum likelihood estimation for ARMA may fail to converge for some bin/stock combinations (e.g., near-constant volume series). **Handling:** Fall back to the historical average component for that bin (set w_interday or w_intraday to 0). [Researcher inference; analogous to Szucs (2017) fallback strategy for CMEM]

4. **First bin of day:** The intraday ARMA has no within-day observations yet. The raw forecast for bin 1 relies only on the historical average and inter-day ARMA components. The intraday component produces a static prediction (unconditional mean of the deseasonalized series, re-seasonalized). For the volume percentage model, no surprises are available at bin 1, so predicted_adjustment = 0 and the historical percentage is used unchanged. [Satish et al. 2014, Exhibit 1, p. 18: "Current Bin" starts the intraday feed. Researcher inference on fallback to unconditional mean.]

5. **Half-trading days / early closes:** Days with fewer than 26 bins (e.g., day before holidays, 13 bins for a half-day). **Handling:** Either exclude these days from training and evaluation, or adjust I dynamically. The paper does not address this explicitly. [Researcher inference]

6. **Stock splits / corporate actions:** Volume data must be adjusted for splits. The paper uses absolute share counts; a 2:1 split would double the apparent volume. **Handling:** Use split-adjusted volume or, as Chen et al. (2016) suggest, normalize by daily shares outstanding. [Chen et al. 2016, Section 4.1, p. 8]

7. **Regime assignment at start of day:** Before any bins are observed (j=0), cumulative volume is zero. The regime cannot be determined from percentile. **Handling:** Use the unconditional (all-data) weight set or the median-regime weights. [Researcher inference]

8. **Special calendar days:** Option expiry, Fed meeting days, and index rebalancing dates produce atypical volume patterns. The paper recommends custom VWAP curves rather than ARMAX models for these. **Handling:** Maintain a calendar of special days and substitute pre-computed custom curves. [Satish et al. 2014, p. 18]

9. **Joint term constraint infeasible:** If the inter-day ARMA for a bin requires many terms (e.g., p=4, q=4, constant = 9 terms), the intraday ARMA would be limited to 1 term (constant only), which is useless. **Handling:** Set a maximum inter-day complexity (8 terms, via max_interday_budget parameter) to always leave room for a minimal intraday model. If the inter-day ARMA needs more terms than the budget allows, cap it and accept the AICc suboptimality. [Researcher inference]

10. **Surprise regression with insufficient bins:** If j < L_optimal at prediction time (fewer observed bins than lag terms needed), the regression cannot produce a full-rank predictor vector. **Handling:** Use predicted_adjustment = 0 (fall back to historical percentage). This is handled in the pseudocode (Part B, Step 3b). [Researcher inference]

### Known Limitations

1. **Proprietary parameters:** The specific values of regime-switching thresholds, optimal weighting coefficients, and the number of regression terms for the dynamic VWAP model are not disclosed. Replicators must rediscover these through in-sample optimization, and the resulting model may not exactly match the paper's reported results. [Satish et al. 2014, pp. 18-19; explicitly noted on p. 18]

2. **No distributional framework:** Unlike the CMEM or Kalman filter models, this approach does not specify a noise distribution, so it cannot produce prediction intervals or density forecasts. Only point forecasts are generated. [Researcher inference from paper structure]

3. **Linear combination only:** The weight overlay combines components linearly. Nonlinear interactions between components (e.g., the intraday ARMA being more useful when inter-day volume is unusual) are captured only crudely through regime switching. [Researcher inference]

4. **Volume percentage model limited to next-bin forecasts:** The percentage model is designed for one-step-ahead predictions. Multi-step-ahead percentage forecasts degrade because the rolling regression on surprises has no new observations to feed on. [Satish et al. 2014, p. 18: "techniques that predict only the next interval will perform better"]

5. **No outlier robustness mechanism:** Unlike Chen et al.'s robust Kalman filter with Lasso regularization, this model has no built-in mechanism for handling volume outliers. Outlier bins directly affect the historical average, ARMA estimates, and surprise calculations. [Researcher inference; contrast with Chen et al. 2016, Section 3]

6. **Single-stock model:** Each stock is modeled independently. Cross-sectional information (e.g., sector-wide volume surges) is not exploited, unlike the BDF factor model approach. [Researcher inference]

7. **Static seasonal assumption:** The 6-month trailing average assumes the intraday volume shape is constant over that window. Structural changes (e.g., shift to electronic trading, changes in closing auction share) would be captured slowly. [Researcher inference]

8. **Evaluation on U.S. equities only:** The paper validates on top 500 U.S. stocks by dollar volume. Performance on international markets, ETFs, or illiquid stocks is not tested. Chen et al. (2016) demonstrate the Kalman filter works across multiple exchanges (NYSE, NASDAQ, EPA, LON, ETR, TYO, HKEX). [Satish et al. 2014 vs. Chen et al. 2016, Section 4.1]

9. **Day-boundary handling in intraday ARMA is underspecified:** The paper does not explain how the 1-month rolling estimation handles the boundary between consecutive trading days. Our specification uses likelihood resets at day boundaries, but the exact method may affect parameter estimates. [Researcher inference; see Pseudocode Part A, Step 3c for details]

10. **Multi-step intraday ARMA prediction degradation:** Recursive multi-step ARMA predictions grow increasingly uncertain with forecast horizon. For bins far from the last observation (e.g., bin 2 predicting bins 3-26 requires 24-step-ahead recursive forecasts), the ARMA converges to the unconditional mean. The regime weights may partially compensate if trained on data reflecting this degradation, but there is no explicit mechanism to down-weight the intraday component as a function of forecast horizon. [Researcher inference; paper acknowledges indirectly on p. 18]

## Paper References

| Spec Section | Paper Source | Specific Location |
|-------------|-------------|-------------------|
| Algorithm: Component 1 (Historical Average) | Satish et al. 2014 | p. 17, "Historical Window Average/Rolling Means" |
| Algorithm: Component 2 (Inter-day ARMA) | Satish et al. 2014 | p. 17, "Raw Volume Forecast Methodology" para 2 |
| Algorithm: Component 2 fitting window | Researcher inference | Exhibit 1 "Prior 5 days" is prediction input, not fitting window |
| Algorithm: Component 2 inter-day budget cap (8 terms) | Researcher inference | Reserves at least 2 terms for intraday ARMA; paper does not specify budget split |
| Algorithm: Component 3 (Intraday ARMA) | Satish et al. 2014 | pp. 17-18, "Raw Volume Forecast Methodology" para 3 |
| Algorithm: Component 3 within-day operation | Satish et al. 2014 | Exhibit 1, p. 18 ("Current Bin," "4 Bins Prior to Current Bin," "Today") |
| Algorithm: Component 3 "Today" input interpretation | Researcher inference | Interpreted as data source qualifier (today's bins), not a separate feature |
| Algorithm: Component 3 day-boundary handling | Researcher inference | Paper does not specify; likelihood reset at day boundaries chosen |
| Algorithm: Component 3 multi-step degradation | Researcher inference | Paper acknowledges indirectly (p. 18); no explicit mitigation specified |
| Algorithm: Component 4 (Dynamic Weights) | Satish et al. 2014 | p. 18, "Raw Volume Forecast Methodology" para 4 |
| Algorithm: Joint term constraint | Satish et al. 2014 | p. 18, "dual ARMA model having fewer than 11 terms" |
| Algorithm: Joint constraint per-bin vs per-symbol semantics | Researcher inference | Paper says per-symbol, but inter-day is per-bin; we use max across bins (conservative) |
| Algorithm: Weight non-negativity | Researcher inference | Not specified in paper; added for physical interpretability |
| Algorithm: Weight scale (no sum-to-1) | Researcher inference | Paper says "minimizes the error"; unconstrained scale allows bias correction |
| Algorithm: Volume Percentage Model | Satish et al. 2014 | pp. 18-19, "Volume Percentage Forecast Methodology" |
| Algorithm: Surprise regression (OLS, no intercept) | Researcher inference | Paper does not specify form; OLS without intercept consistent with p. 19 preference |
| Algorithm: Surprise regression training window | Researcher inference | N_regression_fit = 63 days; paper does not specify |
| Algorithm: Surprise regression daily re-estimation | Researcher inference | Paper does not specify update frequency |
| Algorithm: Renormalization step | Researcher inference | Paper states sums-to-100% constraint (p. 15) but not redistribution method |
| Algorithm: AICc model selection | Hurvich and Tsai (1989, 1993) | Cited in Satish et al. 2014, p. 17 |
| Algorithm: Dynamic VWAP framework | Humphery-Jenner (2011) | Cited in Satish et al. 2014, pp. 18-19, p. 24 |
| Algorithm: Flow diagram | Satish et al. 2014 | Exhibit 1, p. 18 |
| Algorithm: No-intercept regressions | Satish et al. 2014 | p. 19, "we perform both regressions without the inclusion of a constant term" |
| Metrics: Volume pct error (MAD) | Satish et al. 2014 | p. 17, "Measuring Percentage Volume Predictions -- Absolute Deviation" |
| Metrics: MAPE definition | Satish et al. 2014 | p. 17, "Measuring Raw Volume Predictions -- MAPE" |
| Metrics: VWAP tracking error definition | Satish et al. 2014 | p. 16, "VWAP Tracking Error" |
| Parameters: Bin size | Satish et al. 2014 | p. 16, "Interval Selection" |
| Parameters: N_hist = 21 | Satish et al. 2014 | Exhibit 1, p. 18 (Researcher inference from "Prior 21 days" label) |
| Parameters: N_interday_predict = 5 | Satish et al. 2014 | Exhibit 1, p. 18 ("Prior 5 days" label for ARMA Daily) |
| Parameters: N_interday_fit = 63 | Researcher inference | Minimum viable for reliable ARMA estimation |
| Parameters: max_interday_budget = 8 | Researcher inference | Budget split to guarantee meaningful intraday model |
| Parameters: N_regression_fit = 63 | Researcher inference | Sufficient pooled samples for OLS; paper does not specify |
| Parameters: Intraday ARMA fitting = 1 month | Satish et al. 2014 | p. 18, "compute this model on a rolling basis over the most recent month" |
| Parameters: Seasonal window = 6 months | Satish et al. 2014 | p. 17, "trailing six months" |
| Parameters: AR lags < 5 | Satish et al. 2014 | p. 18, "AR lags with a value less than five" |
| Parameters: Dual total terms < 11 | Satish et al. 2014 | p. 18, "fit each symbol with a dual ARMA model having fewer than 11 terms" |
| Parameters: Deviation limit = 10% | Satish et al. 2014 | p. 24, referencing Humphery-Jenner (2011) |
| Parameters: Switch-off = 80% | Satish et al. 2014 | p. 24, referencing Humphery-Jenner (2011) |
| Parameters: Regime thresholds (proprietary) | Satish et al. 2014 | p. 18, "different historical volume percentile cutoffs" |
| Parameters: Weighting coefficients (proprietary) | Satish et al. 2014 | pp. 18, noted as undisclosed |
| Parameters: Optimizer choice (Nelder-Mead) | Researcher inference | MAPE is non-differentiable; derivative-free optimizer appropriate |
| Parameters: min_volume_floor | Researcher inference | Prevents MAPE division instability |
| Validation: 24% MAPE reduction | Satish et al. 2014 | p. 20, "Validating Volume Prediction Error" |
| Validation: 29% bottom-95% MAPE reduction | Satish et al. 2014 | p. 20 |
| Validation: 7.55% pct error reduction | Satish et al. 2014 | Exhibit 9, p. 23 |
| Validation: 9.1% VWAP tracking reduction | Satish et al. 2014 | Exhibit 10, p. 23 |
| Validation: SIC/beta consistency | Satish et al. 2014 | Exhibits 7-8, pp. 22-23 |
| Validation: VWAP-error regression R^2 | Satish et al. 2014 | Exhibits 2-5, pp. 20-21 |
| Comparison: Kalman filter MAPE | Chen et al. 2016 | Table 3, p. 10 |
| Comparison: Kalman filter VWAP tracking | Chen et al. 2016 | Table 4, p. 12 |
