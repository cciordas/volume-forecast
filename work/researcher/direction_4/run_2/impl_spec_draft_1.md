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
    
    FOR each bin i in 1..I:
        daily_series_i = [volume[stock, bin=i, day] for day in training_window]
        
        best_aic = infinity
        FOR p in 0..5:
            FOR q in 0..5:
                model = fit_ARMA(daily_series_i, order=(p, q), include_constant=True)
                aicc = compute_AICc(model)
                IF aicc < best_aic:
                    best_aic = aicc
                    best_model_i = model
        
        interday_forecast[i] = best_model_i.predict(steps=1)
    
    # === COMPONENT 3: Intraday ARMA ===
    # Reference: Satish et al. 2014, p. 17-18, "Raw Volume Forecast Methodology" para 3
    # Deseasonalize, fit ARMA, then re-seasonalize.
    
    # Step 3a: Compute seasonal factors from trailing 6-month average
    FOR each bin i in 1..I:
        seasonal_factor[i] = arithmetic_mean(volume[stock, bin=i, day]
                                              for day in trailing_6_months)
    
    # Step 3b: Deseasonalize recent intraday data (rolling 1-month window)
    intraday_series = []
    FOR each day d in rolling_1_month_window:
        FOR each bin i in 1..I:
            deseasonal_vol = volume[stock, bin=i, day=d] / seasonal_factor[i]
            intraday_series.append(deseasonal_vol)
    
    # Step 3c: Fit ARMA to deseasonalized series
    # Effective AR lags < 5; combined model < 11 terms
    best_aic = infinity
    FOR p in 0..4:
        FOR q in 0..4:
            IF p + q + 1 > 10: CONTINUE  # keep total terms < 11
            model = fit_ARMA(intraday_series, order=(p, q), include_constant=True)
            aicc = compute_AICc(model)
            IF aicc < best_aic:
                best_aic = aicc
                intraday_model = model
    
    # Step 3d: Forecast and re-seasonalize
    # For the current day, we have observed bins 1..j already.
    # Feed observed deseasonalized values, then predict forward.
    FOR each bin i in (j+1)..I:
        deseasonal_pred = intraday_model.predict(steps=(i - j))
        intraday_forecast[i] = deseasonal_pred * seasonal_factor[i]
    
    # For bins already observed today, intraday_forecast[i] = actual volume
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
    # (Researcher inference: the paper does not disclose specific cutoffs.
    #  Reasonable starting point: quartile-based, i.e., 4 regimes at
    #  percentiles 0-25, 25-50, 50-75, 75-100. Grid search over
    #  number of regimes {3, 4, 5} and cutoff positions.)
    
    regime_thresholds = grid_search_regime_cutoffs(in_sample_data)
    
    # Step 2: For each regime, optimize weights to minimize MAPE
    FOR each regime r:
        subset = filter_data_by_regime(in_sample_data, r, regime_thresholds)
        
        # Minimize MAPE over the subset
        w_opt = minimize(
            objective = mean_absolute_percentage_error,
            variables = (w_hist, w_interday, w_intraday),
            constraints = [w_hist >= 0, w_interday >= 0, w_intraday >= 0],
            data = subset
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
    # The optimal number of model terms is identified via in-sample search.
    # (Researcher inference: the paper does not disclose exact regression
    #  specification. Based on description, this is a linear regression
    #  of next-bin surprise on lagged surprises.)
    
    # Fit rolling regression: surprise_pct[k] ~ f(surprise_pct[k-1], ..., surprise_pct[k-L])
    # L = number of lagged surprise terms (to be determined by grid search, likely 1-3)
    
    predicted_adjustment = rolling_regression_predict(
        surprise_history=surprise_pct[1..j],
        num_terms=L_optimal
    )
    
    # === Step 4: Apply adjustment with safety constraints ===
    # Reference: Satish et al. 2014, p. 18-19; inherited from Humphery-Jenner (2011)
    
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

#### Part D: VWAP Execution Simulation

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
    | shape: (I,)      | | shape: (I,)      | | shape: (I,)      |
    +--------+---------+ +--------+---------+ +--------+---------+
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
                               | on surprises     |
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
| Historical average | (26,) per stock | float64 | N-day rolling mean |
| Inter-day ARMA forecast | (26,) per stock | float64 | One per bin per stock |
| Intraday ARMA forecast | (26,) per stock | float64 | Deseasonalized then re-seasonalized |
| Regime weights | (num_regimes, 3) | float64 | (w_hist, w_interday, w_intraday) |
| Raw forecast | (26,) per stock | float64 | Weighted combination |
| Volume surprises | (j,) per stock | float64 | Observed bins so far |
| Volume percentage | (26,) per stock | float64 | Sums to 1.0 within day |

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
| N_hist | Historical window for rolling mean (days) | Not disclosed; Researcher inference: 21 days (Exhibit 1 shows "Prior 21 days") | Medium -- affects baseline stability | 10-60 days |
| N_interday | Training window for inter-day ARMA (days) | Not disclosed; Researcher inference: 5 days (Exhibit 1 shows "Prior 5 days" for ARMA Daily) | High -- too short misses patterns, too long adds noise | 5-60 days |
| N_intraday_fit | Fitting window for intraday ARMA | 1 month (~21 trading days) | Medium | 10-40 trading days |
| N_seasonal | Trailing window for seasonal factors | 6 months (~126 trading days) | Low -- seasonal pattern is stable | 63-252 trading days |
| p_max_interday | Maximum AR order for inter-day ARMA | 5 | Low -- AICc handles selection | Fixed at 5 |
| q_max_interday | Maximum MA order for inter-day ARMA | 5 | Low -- AICc handles selection | Fixed at 5 |
| p_max_intraday | Maximum AR order for intraday ARMA | 4 | Low -- lags kept < 5 | 3-5 |
| q_max_intraday | Maximum MA order for intraday ARMA | 4 | Low | 3-5 |
| max_total_terms | Maximum total ARMA terms (intraday) | 11 (including constant) | Low | 8-15 |
| num_regimes | Number of volume percentile regimes | Not disclosed; Researcher inference: 3-5 | High -- key design choice | 2-6 |
| regime_thresholds | Percentile cutoffs for regime switching | Not disclosed; Researcher inference: equally spaced (e.g., [25, 50, 75] for 4 regimes) | High | Determined by grid search |
| deviation_limit | Max allowed departure from historical VWAP curve (pct model) | 10% (0.10) | Medium -- too tight prevents adaptation, too loose adds risk | 5%-20% |
| switchoff_threshold | Cumulative volume at which to revert to historical curve (pct model) | 80% (0.80) | Medium | 70%-90% |
| L_optimal | Number of lagged surprise terms in rolling regression (pct model) | Not disclosed; Researcher inference: 1-3 | Medium | 1-5 |
| N_pct_hist | Historical window for volume percentage baseline | Same as N_hist; Researcher inference: 21 days | Medium | 10-60 days |

### Initialization

1. **Seasonal factors:** Compute the arithmetic mean of volume in each bin over the trailing 6 months of data. If fewer than 6 months are available, use all available data (minimum 63 trading days). [Satish et al. 2014, p. 17]

2. **Historical averages:** Compute the arithmetic mean of volume in each bin over the prior N_hist days. Exhibit 1 (p. 18) shows 21 days as the lookback for the historical window component. [Satish et al. 2014, p. 17]

3. **Inter-day ARMA models:** For each stock and each bin index, fit ARMA(p,q) models to the daily volume series at that bin, selecting the best (p,q) by AICc. The training window for the inter-day ARMA appears from Exhibit 1 to be 5 days ("Prior 5 days"). [Satish et al. 2014, p. 17-18, Exhibit 1 p. 18. Researcher inference: the "Prior 5 days" label in Exhibit 1 likely refers to the lookback for the inter-day ARMA input rather than fitting window; the fitting window is plausibly longer, perhaps matching N_hist=21 days. Developers should experiment.]

4. **Intraday ARMA model:** Fit on a rolling 1-month window of deseasonalized intraday data. At the start of each trading day, the model uses only data up to the previous day's close. During the day, observed bins are fed in for dynamic prediction. [Satish et al. 2014, p. 17-18]

5. **Regime weights:** Train on the first year of data (in-sample period). Optimize weights by minimizing MAPE for each regime bucket separately. Initialize weight optimization with equal weights (1/3, 1/3, 1/3). [Satish et al. 2014, p. 18; Researcher inference on initialization.]

6. **Volume percentage baseline:** Compute historical volume percentages as the average of volume[bin_i] / daily_total across the prior N_pct_hist days. [Satish et al. 2014, p. 18]

### Calibration

The calibration procedure has two phases:

**Phase 1: Component model fitting (daily rolling)**

1. Each trading day before market open:
   a. Update seasonal factors using trailing 6-month data. [Satish et al. 2014, p. 17]
   b. Update historical window averages using prior N_hist days. [Satish et al. 2014, p. 17]
   c. Refit inter-day ARMA(p,q) models for each bin using AICc selection. [Satish et al. 2014, p. 17]
   d. Refit intraday ARMA model on rolling 1-month deseasonalized data. [Satish et al. 2014, pp. 17-18]

2. During the trading day:
   a. After each bin observation, update the intraday ARMA predictions by feeding in the new deseasonalized observation.
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
   ```
   [Satish et al. 2014, p. 18. Researcher inference: the paper describes optimizing weights on in-sample data but does not specify the exact optimization procedure. Constrained nonlinear optimization (e.g., scipy.optimize.minimize with method='SLSQP') or grid search are both viable.]

4. Grid search over regime configuration:
   - Number of regimes: {3, 4, 5}
   - Cutoff positions: equally spaced, or data-driven quantiles
   - Select configuration minimizing out-of-sample MAPE on a held-out validation period.
   [Researcher inference: not specified in paper; necessary because regime thresholds are proprietary.]

**Phase 3: Volume percentage model calibration (periodic)**

1. Identify optimal number of lagged surprise terms L via in-sample cross-validation.
2. Train rolling regression coefficients on in-sample data.
3. Validate that deviation limits and switch-off threshold produce stable behavior.
[Satish et al. 2014, pp. 18-19. Researcher inference on cross-validation procedure.]

## Validation

### Expected Behavior

**Raw volume forecasting (15-minute bins, 500 U.S. stocks, 250 out-of-sample days):**
- Median MAPE reduction vs. historical window baseline: 24%. [Satish et al. 2014, p. 20, "Validating Volume Prediction Error"]
- Bottom-95% mean MAPE reduction: 29%. [Satish et al. 2014, p. 20]
- Error reduction is consistent across all 26 intraday bins (Exhibit 6, p. 22), with median reduction ranging from ~10% (early morning) to ~33% (late afternoon). [Satish et al. 2014, Exhibit 6, p. 22]
- Error reduction is consistent across SIC industry groups (~15-35% median reduction) and beta deciles (~20-35% median reduction). [Satish et al. 2014, Exhibits 7-8, pp. 22-23]

**Volume percentage forecasting (15-minute bins):**
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

4. **Weight non-negativity:** All regime weights should be non-negative (a negative weight would mean the model inverts a component's forecast, which is not sensible). [Researcher inference; the optimization should be constrained]

5. **Volume percentage sum:** For any day, the predicted volume percentages across all bins should sum to approximately 1.0 (within floating-point tolerance). [Structural constraint]

6. **Deviation limit effective:** When the deviation limit is active, adjusted_pct should be within +/- 10% of hist_pct for each bin. [Satish et al. 2014, p. 19]

7. **Switch-off behavior:** After cumulative observed volume exceeds 80% of daily volume, the percentage model should revert to the historical curve. Verify this by checking that adjusted_pct == hist_pct for late bins on high-volume days. [Satish et al. 2014, p. 18]

8. **Monotonic improvement with components:** Adding each component should improve (or not significantly worsen) forecast accuracy:
   - Historical average alone: baseline MAPE
   - + Inter-day ARMA: MAPE should decrease
   - + Intraday ARMA: MAPE should decrease further
   - + Regime switching: MAPE should decrease further
   [Researcher inference based on the paper's design philosophy]

### Edge Cases

1. **Zero-volume bins:** Some bins may have zero traded volume, especially for illiquid stocks or around market open/close. The intraday ARMA deseasonalization divides by the seasonal factor; if seasonal_factor[i] = 0, this produces division by zero. **Handling:** Replace zero seasonal factors with a small positive value (e.g., the minimum non-zero seasonal factor across all bins) or exclude zero-volume bins from the intraday ARMA fitting. [Researcher inference; the paper works on top 500 stocks by dollar volume where this is rare]

2. **Insufficient history:** If fewer than N_hist days of data are available, fall back to whatever is available. If fewer than 6 months for seasonal factors, use all available data. Minimum viable: 21 trading days. [Researcher inference]

3. **ARMA convergence failure:** Maximum likelihood estimation for ARMA may fail to converge for some bin/stock combinations (e.g., near-constant volume series). **Handling:** Fall back to the historical average component for that bin (set w_interday or w_intraday to 0). [Researcher inference; analogous to Szucs (2017) fallback strategy for CMEM]

4. **First bin of day:** The intraday ARMA has no within-day observations yet. The raw forecast for bin 1 relies only on the historical average and inter-day ARMA components. The intraday component produces a static prediction (no within-day update). [Satish et al. 2014, Exhibit 1, p. 18: "Current Bin" starts the intraday feed]

5. **Half-trading days / early closes:** Days with fewer than 26 bins (e.g., day before holidays, 13 bins for a half-day). **Handling:** Either exclude these days or adjust I dynamically. The paper does not address this explicitly. [Researcher inference]

6. **Stock splits / corporate actions:** Volume data must be adjusted for splits. The paper uses absolute share counts; a 2:1 split would double the apparent volume. **Handling:** Use split-adjusted volume or, as Chen et al. (2016) suggest, normalize by daily shares outstanding. [Chen et al. 2016, Section 4.1, p. 8]

7. **Regime assignment at start of day:** Before any bins are observed (j=0), cumulative volume is zero. The regime cannot be determined from percentile. **Handling:** Use the unconditional (all-data) weight set or the median-regime weights. [Researcher inference]

8. **Special calendar days:** Option expiry, Fed meeting days, and index rebalancing dates produce atypical volume patterns. The paper recommends custom VWAP curves rather than ARMAX models for these. **Handling:** Maintain a calendar of special days and substitute pre-computed custom curves. [Satish et al. 2014, p. 18]

### Known Limitations

1. **Proprietary parameters:** The specific values of regime-switching thresholds, optimal weighting coefficients, and the number of regression terms for the dynamic VWAP model are not disclosed. Replicators must rediscover these through in-sample optimization, and the resulting model may not exactly match the paper's reported results. [Satish et al. 2014, pp. 18-19; explicitly noted on p. 18]

2. **No distributional framework:** Unlike the CMEM or Kalman filter models, this approach does not specify a noise distribution, so it cannot produce prediction intervals or density forecasts. Only point forecasts are generated. [Researcher inference from paper structure]

3. **Linear combination only:** The weight overlay combines components linearly. Nonlinear interactions between components (e.g., the intraday ARMA being more useful when inter-day volume is unusual) are captured only crudely through regime switching. [Researcher inference]

4. **Volume percentage model limited to next-bin forecasts:** The percentage model is designed for one-step-ahead predictions. Multi-step-ahead percentage forecasts degrade because the rolling regression on surprises has no new observations to feed on. [Satish et al. 2014, p. 18: "techniques that predict only the next interval will perform better"]

5. **No outlier robustness mechanism:** Unlike Chen et al.'s robust Kalman filter with Lasso regularization, this model has no built-in mechanism for handling volume outliers. Outlier bins directly affect the historical average, ARMA estimates, and surprise calculations. [Researcher inference; contrast with Chen et al. 2016, Section 3]

6. **Single-stock model:** Each stock is modeled independently. Cross-sectional information (e.g., sector-wide volume surges) is not exploited, unlike the BDF factor model approach. [Researcher inference]

7. **Static seasonal assumption:** The 6-month trailing average assumes the intraday volume shape is constant over that window. Structural changes (e.g., shift to electronic trading, changes in closing auction share) would be captured slowly. [Researcher inference]

8. **Evaluation on U.S. equities only:** The paper validates on top 500 U.S. stocks by dollar volume. Performance on international markets, ETFs, or illiquid stocks is not tested. Chen et al. (2016) demonstrate the Kalman filter works across multiple exchanges (NYSE, NASDAQ, EPA, LON, ETR, TYO, HKEX). [Satish et al. 2014 vs. Chen et al. 2016, Section 4.1]

## Paper References

| Spec Section | Paper Source | Specific Location |
|-------------|-------------|-------------------|
| Algorithm: Component 1 (Historical Average) | Satish et al. 2014 | p. 17, "Historical Window Average/Rolling Means" |
| Algorithm: Component 2 (Inter-day ARMA) | Satish et al. 2014 | p. 17, "Raw Volume Forecast Methodology" para 2 |
| Algorithm: Component 3 (Intraday ARMA) | Satish et al. 2014 | pp. 17-18, "Raw Volume Forecast Methodology" para 3 |
| Algorithm: Component 4 (Dynamic Weights) | Satish et al. 2014 | p. 18, "Raw Volume Forecast Methodology" para 4 |
| Algorithm: Volume Percentage Model | Satish et al. 2014 | pp. 18-19, "Volume Percentage Forecast Methodology" |
| Algorithm: AICc model selection | Hurvich and Tsai (1989, 1993) | Cited in Satish et al. 2014, p. 17 |
| Algorithm: Dynamic VWAP framework | Humphery-Jenner (2011) | Cited in Satish et al. 2014, pp. 18-19 |
| Algorithm: Flow diagram | Satish et al. 2014 | Exhibit 1, p. 18 |
| Parameters: Bin size | Satish et al. 2014 | p. 16, "Interval Selection" |
| Parameters: N_hist = 21 | Satish et al. 2014 | Exhibit 1, p. 18 (Researcher inference from "Prior 21 days" label) |
| Parameters: N_interday = 5 | Satish et al. 2014 | Exhibit 1, p. 18 (Researcher inference from "Prior 5 days" label) |
| Parameters: Intraday ARMA fitting = 1 month | Satish et al. 2014 | p. 18, "compute this model on a rolling basis over the most recent month" |
| Parameters: Seasonal window = 6 months | Satish et al. 2014 | p. 17, "trailing six months" |
| Parameters: AR lags < 5 | Satish et al. 2014 | p. 18, "AR lags with a value less than five" |
| Parameters: Total terms < 11 | Satish et al. 2014 | p. 18, "fewer than 11 terms" |
| Parameters: Deviation limit = 10% | Satish et al. 2014 | p. 18, "depart no more than 10% away" |
| Parameters: Switch-off = 80% | Satish et al. 2014 | p. 18, "once 80% of the day's volume is reached" |
| Parameters: Regime thresholds (proprietary) | Satish et al. 2014 | p. 18, "different historical volume percentile cutoffs" |
| Parameters: Weighting coefficients (proprietary) | Satish et al. 2014 | pp. 18, noted as undisclosed |
| Validation: 24% MAPE reduction | Satish et al. 2014 | p. 20, "Validating Volume Prediction Error" |
| Validation: 29% bottom-95% MAPE reduction | Satish et al. 2014 | p. 20 |
| Validation: 7.55% pct error reduction | Satish et al. 2014 | Exhibit 9, p. 23 |
| Validation: 9.1% VWAP tracking reduction | Satish et al. 2014 | Exhibit 10, p. 23 |
| Validation: SIC/beta consistency | Satish et al. 2014 | Exhibits 7-8, pp. 22-23 |
| Validation: VWAP-error regression R^2 | Satish et al. 2014 | Exhibits 2-5, pp. 20-21 |
| Comparison: Kalman filter MAPE | Chen et al. 2016 | Table 3, p. 10 |
| Comparison: Kalman filter VWAP tracking | Chen et al. 2016 | Table 4, p. 12 |
| Metrics: MAPE definition | Satish et al. 2014 | p. 17, "Measuring Raw Volume Predictions -- MAPE" |
| Metrics: Volume percentage error definition | Satish et al. 2014 | p. 17, "Measuring Percentage Volume Predictions -- Absolute Deviation" |
| Metrics: VWAP tracking error definition | Satish et al. 2014 | p. 16, "VWAP Tracking Error" |
