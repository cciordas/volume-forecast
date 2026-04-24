# Implementation Specification: PCA Factor Decomposition (BDF)

## Overview

The BDF model (Bialkowski, Darolles, Le Fol 2008) decomposes intraday stock
turnover into a market-wide common component and a stock-specific idiosyncratic
component using principal components analysis (PCA) applied to a cross-section
of stocks. The common component captures the well-known U-shaped intraday
volume pattern shared across all stocks. The specific component captures
stock-level deviations and is modeled by either an AR(1) or a Self-Exciting
Threshold Autoregressive (SETAR) process. Forecasts are produced by combining a
historical average of the common component with one-step-ahead time-series
predictions of the specific component, updated dynamically throughout the
trading day.

The model is designed for VWAP (Volume Weighted Average Price) execution: a
trader who accurately predicts the intraday volume profile can replicate the
end-of-day VWAP by distributing order execution proportionally across bins.

## Algorithm

### Model Description

The model takes as input a panel of intraday turnover observations for N stocks
over T trading days, each day divided into k equal-length bins. It decomposes
each stock's turnover into two additive components:

    x_{i,t} = c_{i,t} + e_{i,t}

where x_{i,t} is stock i's turnover at intraday time index t, c_{i,t} is the
common (market-wide) component, and e_{i,t} is the specific (stock-level)
component.

The common component is extracted via PCA on a rolling window of the
cross-sectional turnover matrix. It captures the shared intraday seasonal
pattern (U-shape) and is forecast by a simple historical time-of-day average.
The specific component captures stock-level deviations from the market pattern
and is forecast by an AR(1) or SETAR time-series model.

**Inputs:**
- Intraday turnover x_{i,t} for N stocks, k bins per day, over a rolling
  estimation window of L days (N x (k*L) matrix).
- Total shares outstanding (TSO) per stock for turnover computation.

**Outputs:**
- One-step-ahead turnover forecast x_hat_{i,t+1} for each stock i at each
  future bin t+1.
- Intraday volume profile (sequence of forecasts for all remaining bins in
  the current trading day) for VWAP scheduling.

**Assumptions:**
- The cross-sectional dimension N is large enough (N >= 20 stocks) for PCA
  factor estimation to be consistent (Bai 2003).
- All stocks have non-zero volume in every bin within the estimation window.
- The trader's order size is small relative to market volume (no price impact).
- The common component is stable enough that a historical average is a good
  forecast.

### Pseudocode

The algorithm operates in two phases: an overnight estimation phase (run once
per day before market open) and an intraday forecasting phase (run after each
observed bin).

#### Phase 1: Overnight Estimation (run once before each trading day)

```
FUNCTION estimate_model(turnover_data, day_index, L, k, N):
    # turnover_data: array of shape (N, total_bins) containing turnover for all stocks
    # day_index: the current day (we are preparing forecasts for this day)
    # L: number of prior days in rolling estimation window
    # k: number of intraday bins per trading day
    # N: number of stocks

    # Step 1: Extract the estimation window
    # Use the L most recent completed trading days
    # This gives a matrix X of shape (P, N) where P = k * L
    start_bin = (day_index - L) * k
    end_bin = day_index * k
    X = turnover_data[:, start_bin:end_bin].T   # shape (P, N)
    # (Paper: BDF 2008, Section 2.2, Eq. 4-6; Szucs 2017, Section 4.1, Eq. 4)

    # Step 2: Center the data by column (stock) means
    # (Researcher inference: Neither BDF 2008 nor Szucs 2017 explicitly
    #  prescribes centering before eigendecomposition. The Bai (2003) procedure
    #  can handle an intercept as an additional estimated factor. However,
    #  centering is standard PCA practice and ensures the extracted factors
    #  capture intraday shape variation rather than the mean turnover level.
    #  The column means are added back when computing the common component
    #  forecast, so the final turnover forecast is in the original scale.)
    col_means = mean(X, axis=0)          # shape (N,)
    X_centered = X - col_means           # shape (P, N)

    # Step 3: Select the number of common factors r
    # IMPORTANT: factor selection must use the same centered data as PCA
    r = select_num_factors(X_centered, N, P)
    # Uses Bai & Ng (2002) information criterion IC_p1
    # (Paper: BDF 2008, Section 2.2, referencing Bai & Ng 2002)

    # Step 4: Extract common and specific components via PCA
    # Use SVD on X_centered for computational efficiency.
    # BDF 2008, Section 2.2 states: "The estimated factors matrix F_hat is
    # proportional (up to T^{1/2}) to the eigenvectors corresponding to the
    # r-largest eigenvalues of the X'X matrix." Since X is (P, N) and
    # N << P (typically N=33-39, P=500-520), the eigendecomposition of
    # X'X (N x N) is far more efficient than XX' (P x P).
    #
    # SVD approach: X_centered = U @ diag(S) @ V^T
    #   where U is (P, min(P,N)), S is (min(P,N),), V is (N, min(P,N))
    #   F_hat = U[:, :r] * sqrt(P)  (normalized so F'F/P = I_r)
    #   Lambda_hat = V[:, :r] * S[:r] / sqrt(P)
    # This is equivalent to eigendecomposing (1/P) * X'X and recovering
    # factors, but avoids forming the covariance matrix entirely.
    # (Paper: BDF 2008, Section 2.2, Eq. 5-6; Bai 2003 normalization F'F/T = I_r)
    U, S, Vt = svd(X_centered, full_matrices=False)   # thin SVD
    F_hat = U[:, :r] * sqrt(P)                         # shape (P, r)
    Lambda_hat = Vt[:r, :].T * S[:r] / sqrt(P)         # shape (N, r)

    # Common component: K_hat = F_hat * Lambda_hat^T
    K_hat = F_hat @ Lambda_hat.T                        # shape (P, N)

    # Specific component: e_hat = X_centered - K_hat
    e_hat = X_centered - K_hat                          # shape (P, N)

    # Step 5: Compute common component forecast for each bin of the target day
    # Average the common component at the same time-of-day over L prior days,
    # then add back the column means to produce forecasts in the original
    # (uncentered) turnover scale.
    # (Paper: BDF 2008, Section 2.3, Eq. 9)
    #
    # Data layout: X (and hence K_hat) is organized as
    #   [day_0_bin_0, ..., day_0_bin_{k-1}, day_1_bin_0, ...],
    # i.e., days are concatenated chronologically with k bins per day.
    # Therefore bin_j of day d is at row index bin_j + d * k.
    c_forecast = zeros(k, N)
    for bin_j in range(k):
        # Gather common component values at bin_j across the L days in window
        indices = [bin_j + d * k for d in range(L)]
        c_forecast[bin_j, :] = mean(K_hat[indices, :], axis=0) + col_means

    # Step 6: Fit time-series model to specific component for each stock
    # OLS estimation is used because it is equivalent to conditional MLE under
    # Gaussian errors, which is what BDF 2008 Section 2.3 prescribes ("estimate
    # Eqs. (10) or (11) by maximum likelihood"). For AR(1), conditional MLE =
    # OLS. For SETAR, conditional MLE under Gaussian errors reduces to grid
    # search over tau + OLS within each regime (conditional least squares).
    ts_models = {}
    for stock_i in range(N):
        e_series = e_hat[:, stock_i]   # length P = k * L
        ts_models[stock_i] = fit_specific_model(e_series)
        # Fits AR(1) or SETAR; see subroutines below

    RETURN c_forecast, ts_models, e_hat, col_means
```

#### Phase 1a: Select Number of Factors

```
FUNCTION select_num_factors(X_centered, N, P):
    # Implements Bai & Ng (2002) IC_p1 criterion
    # (Paper: BDF 2008, Section 2.2, referencing Bai & Ng 2002)
    #
    # IMPORTANT: X_centered must already be column-mean-centered, the same
    # preprocessing used for the actual PCA extraction. This ensures the
    # factor count is selected consistently with the decomposition.

    # Upper bound on number of factors.
    # (Researcher inference: BDF 2008 does not specify r_max. The value 20 is
    #  a practical choice, generous for volume data where r = 1-3 is typical.
    #  Bai & Ng 2002 recommend searching up to a "reasonable upper bound";
    #  20 is conservative given the expected factor count.)
    r_max = min(20, min(N, P) - 1)
    best_r = 1
    best_ic = infinity

    # Compute thin SVD once; reuse for all truncation levels
    U, S, Vt = svd(X_centered, full_matrices=False)

    for r_candidate in range(1, r_max + 1):
        # Truncate SVD to r_candidate factors
        F_r = U[:, :r_candidate] * sqrt(P)
        Lambda_r = Vt[:r_candidate, :].T * S[:r_candidate] / sqrt(P)
        K_r = F_r @ Lambda_r.T
        e_r = X_centered - K_r

        # Residual variance: V(r) = (1/(N*P)) * sum(e_r^2)
        V_r = sum(e_r ** 2) / (N * P)

        # Penalty: IC_p1 uses ((N + P) / (N * P)) * ln(N * P / (N + P))
        penalty = r_candidate * ((N + P) / (N * P)) * ln((N * P) / (N + P))

        ic = ln(V_r) + penalty

        if ic < best_ic:
            best_ic = ic
            best_r = r_candidate

    RETURN best_r
```

#### Phase 1b: Fit Specific Component Models

**AR(1) variant:**

```
FUNCTION fit_ar1(e_series):
    # Fits: e_t = c + theta_1 * e_{t-1} + epsilon_t
    # by OLS regression of e_t on (1, e_{t-1})
    # (Paper: BDF 2008, Section 2.3, Eq. 10; Szucs 2017, Section 4.1, Eq. 5)
    # Note: OLS is equivalent to conditional MLE under Gaussian errors
    # (BDF 2008, Section 2.3: "estimate ... by maximum likelihood").

    y = e_series[1:]       # dependent variable: e_2, ..., e_P
    X_reg = column_stack([ones(len(y)), e_series[:-1]])   # intercept + lag

    # OLS: beta = (X^T X)^{-1} X^T y
    beta = lstsq(X_reg, y)
    c = beta[0]
    theta_1 = beta[1]
    residuals = y - X_reg @ beta
    sigma_eps = sqrt(sum(residuals ** 2) / (len(y) - 2))

    # Stationarity check
    # (Researcher inference: if |theta_1| >= 1, the AR(1) model is
    #  non-stationary and multi-step forecasts will diverge. Neither paper
    #  addresses this edge case. Constraining to |theta_1| < 1 is the
    #  safest approach; alternatively, fall back to the U-method for that
    #  stock. Clamping at +/- 0.99 preserves the estimated sign/direction
    #  while ensuring stationarity.)
    if abs(theta_1) >= 1.0:
        theta_1 = sign(theta_1) * 0.99
        # Log warning: non-stationary AR(1) coefficient clamped

    RETURN AR1Model(c=c, theta_1=theta_1, sigma_eps=sigma_eps)
```

**SETAR variant:**

```
FUNCTION fit_setar(e_series):
    # Fits two-regime threshold AR model:
    # e_t = (c_1 + theta_{1,2} * e_{t-1}) * I(e_{t-1} <= tau)
    #     + (c_2 + theta_{2,2} * e_{t-1}) * (1 - I(e_{t-1} <= tau))
    #     + epsilon_t
    # (Paper: BDF 2008, Section 2.3, Eq. 11; Szucs 2017, Section 4.1, Eq. 6-7)
    # Note: OLS is equivalent to conditional MLE under Gaussian errors
    # (BDF 2008, Section 2.3: "estimate ... by maximum likelihood").

    y = e_series[1:]
    x_lag = e_series[:-1]

    # Grid search over tau: use percentiles of the lagged specific component
    # Restrict to [15th, 85th] percentile to ensure sufficient obs in each regime
    # (Researcher inference: BDF 2008 and Szucs 2017 do not specify the
    #  percentile grid. Restricting to [15th, 85th] is standard practice for
    #  threshold AR models, ensuring each regime has enough observations for
    #  reliable OLS estimation.)
    tau_candidates = percentiles(x_lag, range(15, 86))
    best_tau = None
    best_ssr = infinity

    for tau in tau_candidates:
        indicator = (x_lag <= tau).astype(float)

        # Regime 1: observations where e_{t-1} <= tau
        # Regime 2: observations where e_{t-1} > tau
        X_reg = column_stack([
            indicator,                    # c_1 coefficient
            indicator * x_lag,            # theta_{1,2} coefficient
            (1 - indicator),              # c_2 coefficient
            (1 - indicator) * x_lag       # theta_{2,2} coefficient
        ])

        beta = lstsq(X_reg, y)
        residuals = y - X_reg @ beta
        ssr = sum(residuals ** 2)

        if ssr < best_ssr:
            best_ssr = ssr
            best_tau = tau
            best_beta = beta

    c_1 = best_beta[0]
    theta_12 = best_beta[1]
    c_2 = best_beta[2]
    theta_22 = best_beta[3]
    sigma_eps = sqrt(best_ssr / (len(y) - 4))

    # Stationarity check for each regime
    # (Researcher inference: same rationale as AR(1) stationarity check.)
    if abs(theta_12) >= 1.0:
        theta_12 = sign(theta_12) * 0.99
    if abs(theta_22) >= 1.0:
        theta_22 = sign(theta_22) * 0.99

    RETURN SETARModel(c_1=c_1, theta_12=theta_12,
                      c_2=c_2, theta_22=theta_22,
                      tau=best_tau, sigma_eps=sigma_eps)
```

#### Phase 2: Intraday Dynamic Forecasting

```
FUNCTION forecast_intraday(c_forecast, ts_models, e_hat, observed_bins,
                           stock_i, k):
    # Produces multi-step chained forecasts for ALL remaining bins in the day.
    # The chaining (setting e_last = e_forecast at each step) is needed for
    # two reasons:
    #   1. The participation fraction for each bin requires a denominator that
    #      sums all remaining bin forecasts.
    #   2. Static (full-day) forecasting at market open uses the chained
    #      multi-step predictions.
    #
    # In dynamic execution (execute_dynamic_vwap), this function is called
    # after each observed bin, but only remaining_forecasts[0] is used for
    # the current bin's participation fraction. The remaining elements are
    # used only for the denominator. This effectively performs one-step-ahead
    # forecasting with respect to the specific component, which is why
    # dynamic execution outperforms static execution.
    # (Paper: BDF 2008, Section 4.2.2 "Dynamic VWAP execution";
    #  Section 4.2.1: "long-horizon ARMA forecasts collapse to zero")
    #
    # c_forecast: pre-computed common component forecast, shape (k, N)
    # ts_models: fitted time-series models per stock
    # e_hat: specific component from estimation window, shape (P, N)
    # observed_bins: list of (bin_index, actual_turnover) observed so far today
    # stock_i: which stock to forecast
    # k: bins per day

    model = ts_models[stock_i]
    forecasts = []

    if len(observed_bins) == 0:
        # Before market open: use last specific component value from prior day
        # (Researcher inference: BDF 2008 does not explicitly state how e_last
        #  is initialized at market open. Using the end-of-prior-day value is
        #  the natural choice for a dynamic model, consistent with the
        #  one-step-ahead forecasting framework.)
        e_last = e_hat[-1, stock_i]
    else:
        # Use the most recent observed specific component residual
        last_bin_idx, last_actual = observed_bins[-1]
        e_last = last_actual - c_forecast[last_bin_idx, stock_i]

    # Forecast remaining bins
    for future_bin in range(len(observed_bins), k):
        # One-step-ahead forecast of specific component
        e_forecast = model.predict(e_last)

        # Combined forecast
        x_forecast = c_forecast[future_bin, stock_i] + e_forecast

        # Ensure non-negative turnover
        # (Researcher inference: turnover is non-negative by definition.
        #  The additive decomposition can produce negative combined forecasts
        #  when the specific component is large and negative.)
        x_forecast = max(x_forecast, 0.0)

        forecasts.append(x_forecast)

        # Chain the forecast for multi-step ahead prediction
        e_last = e_forecast

    RETURN forecasts


FUNCTION ar1_predict(model, e_last):
    # (Paper: BDF 2008, Section 2.3, Eq. 10)
    RETURN model.c + model.theta_1 * e_last


FUNCTION setar_predict(model, e_last):
    # (Paper: BDF 2008, Section 2.3, Eq. 11)
    if e_last <= model.tau:
        RETURN model.c_1 + model.theta_12 * e_last
    else:
        RETURN model.c_2 + model.theta_22 * e_last
```

#### Phase 3: VWAP Execution Scheduling

```
FUNCTION compute_vwap_schedule(forecasts, total_order_shares, k):
    # Convert volume forecasts to participation schedule
    # (Paper: BDF 2008, Section 4.1.3, 4.2)

    total_forecast = sum(forecasts)
    if total_forecast <= 0:
        # Fallback: uniform distribution
        RETURN [total_order_shares / k] * k

    schedule = []
    for f in forecasts:
        fraction = f / total_forecast
        shares_this_bin = fraction * total_order_shares
        schedule.append(shares_this_bin)

    RETURN schedule


FUNCTION execute_dynamic_vwap(c_forecast, ts_models, e_hat,
                               total_order_shares, stock_i, k):
    # Dynamic execution: update forecast after each observed bin
    # (Paper: BDF 2008, Section 4.2.2)

    remaining_shares = total_order_shares
    observed_bins = []
    executed = []

    for bin_j in range(k):
        # Forecast remaining bins (all bins from current through end of day)
        remaining_forecasts = forecast_intraday(
            c_forecast, ts_models, e_hat, observed_bins, stock_i, k)

        # Compute fraction for this bin out of remaining forecast
        total_remaining_forecast = sum(remaining_forecasts)
        if total_remaining_forecast > 0:
            fraction = remaining_forecasts[0] / total_remaining_forecast
        else:
            remaining_bins = k - bin_j
            fraction = 1.0 / remaining_bins

        shares_this_bin = fraction * remaining_shares
        executed.append(shares_this_bin)
        remaining_shares -= shares_this_bin

        # Observe actual turnover (provided by market data feed)
        actual_turnover = observe_market(stock_i, bin_j)
        observed_bins.append((bin_j, actual_turnover))

    RETURN executed
```

### Data Flow

```
Input: Raw intraday volume V_{i,t} and total shares outstanding TSO_i
       for N stocks, k bins/day, over L+1 days (L estimation + 1 forecast)

  |
  v
[Turnover Computation]
  x_{i,t} = V_{i,t} / TSO_i
  Output: Turnover matrix X of shape (P, N), where P = k * L

  |
  v
[Column-Mean Centering]
  (Researcher inference: standard PCA preprocessing; not explicitly in papers)
  col_means = mean(X, axis=0)              (N,)
  X_centered = X - col_means               (P, N)
  Output: X_centered, col_means

  |
  v
[Factor Count Selection]
  Input: X_centered (P x N)
  Bai-Ng IC_p1 criterion evaluated for r = 1, ..., r_max
  SVD computed once, truncated at each candidate r
  Output: r (integer, typically 1-3)

  |
  v
[PCA Factor Extraction via SVD]
  Input: X_centered (P x N)
  1. Thin SVD: X_centered = U @ diag(S) @ V^T
     U: (P, min(P,N)),  S: (min(P,N),),  V: (N, min(P,N))
  2. F_hat = U[:, :r] * sqrt(P)                        (P x r)
  3. Lambda_hat = V[:, :r] * S[:r] / sqrt(P)            (N x r)
  4. K_hat = F_hat @ Lambda_hat^T                        (P x N)
  5. e_hat = X_centered - K_hat                          (P x N)
  Output: K_hat (centered common component), e_hat (specific component)

  |
  v
[Common Component Forecast]
  Input: K_hat (P x N), col_means (N,)
  For each bin j in {0, ..., k-1}:
    c_forecast[j, :] = mean of K_hat[j + d*k, :] for d in 0..L-1, + col_means
  Output: c_forecast (k x N) -- predicted common component for each bin
          (in original uncentered turnover scale)

  |
  v
[Specific Component Model Fitting]
  Input: e_hat (P x N)
  For each stock i in 0..N-1:
    Extract e_hat[:, i] as a univariate time series of length P
    Fit AR(1) or SETAR by OLS (equivalent to conditional MLE under Gaussian
    errors; BDF 2008, Section 2.3)
  Output: ts_models -- dictionary of fitted AR1 or SETAR models per stock

  |
  v
[Intraday Dynamic Forecasting Loop]
  At market open:
    e_last = e_hat[-1, stock_i]  (last specific component from prior day)
    Generate full-day forecast: x_hat = c_forecast + e_forecast for each bin

  After each observed bin j:
    Update e_last = actual_turnover_j - c_forecast[j, stock_i]
    Re-forecast bins j+1 through k-1 using updated e_last
    Only remaining_forecasts[0] is used for the current bin's execution;
    the rest form the denominator for the participation fraction.
  Output: Updated x_hat (k-j-1 remaining forecasts)

  |
  v
[VWAP Schedule]
  Convert forecasts to participation fractions
  Output: shares to trade per remaining bin
```

### Variants

**Implemented variant: BDF with SETAR specific component** (primary) and
**BDF with AR(1) specific component** (secondary).

The SETAR variant is the primary recommendation because:
- It consistently outperforms AR(1) across 39 CAC40 stocks (BDF 2008,
  Section 3.2, Table 2) and across 33 DJIA stocks over 11 years (Szucs 2017,
  Section 5, full sample results).
- By MAPE, SETAR is best overall and beats all models on 26-30 out of 33
  individual stocks (Szucs 2017, Section 5).
- The nonlinear regime-switching captures distinct behavior during calm vs.
  turbulent volume periods.

The AR(1) variant should also be implemented as a baseline because:
- It wins on MSE criterion (Szucs 2017, full sample: BDF_AR MSE = 6.49E-4
  vs. BDF_SETAR MSE = 6.60E-4).
- It is simpler and faster to estimate (no threshold search).
- Szucs shows AR(1) produces fewer extreme squared errors (lower Q95).

Both variants share identical PCA extraction and common component forecasting;
they differ only in how the specific component e_{i,t} is modeled.

**Note on BDF 2008 Eq. 10 ambiguity:** The paper text says "ARMA(1,1)" but
Equation 10 shows e_{i,t} = psi_1 * e_{i,t-1} + psi_2 + epsilon_{i,t}, which
is AR(1) with intercept (no moving-average term). Szucs 2017 explicitly
implements it as AR(1) (Eq. 5: e_p = c + theta_1 * e_{p-1} + epsilon_p). We
follow the equations rather than the text label. (Researcher inference: the
"ARMA(1,1)" label in BDF 2008 is a misnomer for what is actually AR(1) with
intercept.)

## Parameters

| Parameter | Description | Recommended Value | Sensitivity | Range |
|-----------|-------------|-------------------|-------------|-------|
| k | Number of intraday bins per trading day | 26 (15-min bins, 6.5h session) or 25 (20-min bins, 8.3h session) | Low -- determined by market structure | 13-52 depending on bin size and market hours |
| L | Rolling estimation window length (trading days) | 20 | Low-medium -- BDF 2008 and Szucs 2017 both use 20 days; stable results | 10-60 |
| r | Number of common factors | Estimated by Bai-Ng IC_p1 | Medium -- too few misses common variation, too many overfits | 1-10 (typically 1-3 for volume data) |
| r_max | Maximum candidate factor count for IC search | min(20, min(N, P) - 1) | Low -- just an upper bound for search (Researcher inference; see Phase 1a note) | 10-30 |
| tau | SETAR threshold on lagged specific component | Estimated per stock by grid search over percentiles | Medium-high -- determines regime-switching boundary | [15th, 85th] percentile of e_{t-1} |
| tau_grid | Percentile range for SETAR threshold search | 15th to 85th percentile (Researcher inference) | Low -- standard practice to avoid extreme quantiles | [10th, 90th] to [20th, 80th] |
| c (AR1) | AR(1) intercept | Estimated by OLS | Low | Unconstrained |
| theta_1 (AR1) | AR(1) autoregressive coefficient | Estimated by OLS; typically small positive | Medium -- controls persistence of specific component | (-1, 1) for stationarity; clamped if |theta_1| >= 1 |
| c_1, c_2 (SETAR) | SETAR regime intercepts | Estimated by OLS | Low | Unconstrained |
| theta_12, theta_22 (SETAR) | SETAR regime AR coefficients | Estimated by OLS | Medium -- control dynamics in each regime | (-1, 1) for stationarity in each regime; clamped if |theta| >= 1 |

### Initialization

1. **Turnover computation:** For each stock i and each intraday bin t, compute
   x_{i,t} = V_{i,t} / TSO_i. TSO should be the most recent value available
   before the trading day to avoid look-ahead bias. (BDF 2008, Section 3.1;
   Szucs 2017, Section 2: "TSO data was downloaded from a Bloomberg terminal.")

2. **Estimation window:** Use the most recent L = 20 completed trading days.
   Exclude partial trading days (early closes, holidays). (BDF 2008, Section
   3.1: "The 24th and 31st of December 2003 were excluded as partial trading
   days.")

3. **PCA initialization:** No special initialization needed -- SVD is
   deterministic. Center the turnover matrix by column means before
   applying PCA. (Researcher inference: centering is standard PCA practice;
   see Step 2 of Phase 1 pseudocode for rationale.)

4. **AR(1)/SETAR initialization:** OLS estimation is deterministic and requires
   no initialization. For the first intraday forecast of the day, the initial
   e_last value is the last specific component observation from the prior
   trading day: e_last = e_hat[P-1, stock_i]. (Researcher inference: the paper
   does not explicitly state how e_last is initialized at market open, but
   using the end-of-prior-day value is the natural choice for a dynamic model
   and is consistent with the one-step-ahead forecasting framework.)

5. **First estimation day:** The model requires at least L = 20 trading days of
   historical data before it can produce its first forecast. The very first
   PCA estimation uses exactly L days. There is no warm-up or burn-in beyond
   this.

### Calibration

The model uses a rolling daily re-estimation procedure:

1. **Daily recalibration (overnight):**
   a. Construct the rolling window: most recent L trading days of turnover data
      for all N stocks, producing matrix X of shape (k*L, N). (BDF 2008,
      Section 3.2: "rolling 20-day estimation window")
   b. Center X by column means.
   c. Select the number of factors r using Bai-Ng IC_p1 on X_centered.
   d. Run SVD-based PCA to extract common component K_hat and specific
      component e_hat.
   e. Compute common component forecast c_forecast for each bin by averaging
      K_hat at the same time-of-day across L days and adding back column
      means. (BDF 2008, Section 2.3, Eq. 9)
   f. Fit AR(1) or SETAR to each stock's specific component time series.
      (BDF 2008, Section 2.3, Eqs. 10-11; Szucs 2017, Section 4.1, Eqs. 5-7)

2. **Intraday updating (no re-estimation):**
   Model parameters remain fixed during the day. Only the conditioning
   information (e_last) is updated as new bins are observed. (Szucs 2017,
   Section 3: "While parameters are updated daily, the information base for
   the forecast is updated every 15 minutes.")

3. **No cross-validation or hyperparameter tuning** is required beyond the
   Bai-Ng factor count selection and SETAR threshold grid search, both of
   which are embedded in the estimation procedure.

## Validation

### Evaluation Metrics

The two primary evaluation metrics are defined in Szucs 2017, Section 3,
Eqs. 1-2:

**Mean Squared Error (MSE):**

    MSE = (1/N) * sum_{t=1}^{N} (Y_t - Y_t^f)^2

where Y_t is the actual turnover and Y_t^f is the forecast turnover, and N
is the number of observations being evaluated.

**Mean Absolute Percentage Error (MAPE):**

    MAPE = (1/N) * sum_{t=1}^{N} |Y_t - Y_t^f| / Y_t

**Zero-division note:** MAPE has Y_t in the denominator, so bins with zero
actual turnover produce undefined values. This is avoided by the data
requirement that all stocks have non-zero volume in every bin (Szucs 2017,
Section 2: "every stock had trades and thus a volume record larger than zero
in every 15-minute interval"). If zero-volume bins occur in practice,
exclude them from the MAPE calculation and log a warning.

Both metrics are computed per-stock and then averaged across stocks for
aggregate comparisons. (Szucs 2017, Section 3: "Both measures are calculated
for each share, and also in the average of all shares.")

### U-Method Benchmark

The U-method is the naive baseline against which BDF is compared. It uses the
simple historical time-of-day average of raw turnover (no PCA decomposition):

    x_hat_U[bin_j, stock_i] = (1/L) * sum_{d=0}^{L-1} x[bin_j + d*k, stock_i]

This is equivalent to replacing c by x in BDF 2008 Eq. 9 (footnote 5:
"Replacing c by x in Eq. (9), we get the classical prediction of volume that
we will call classical approach"). (Szucs 2017, Section 3, Eq. 3; BDF 2008,
Section 3.2 "classical approach".)

A developer implementing the validation suite should compute U-method forecasts
as the primary baseline for all accuracy comparisons.

### Evaluation Protocol

All MAPE and MSE benchmarks cited below are computed using one-step-ahead
(dynamic) forecasts: at each intraday bin, the forecast is conditioned on the
actual observed turnover of the immediately preceding bin, not on chained
multi-step predictions. This matches the "dynamic adjustment" procedure in
BDF 2008, Section 4.2.2 and the one-step-ahead protocol in Szucs 2017,
Section 3 ("While parameters are updated daily, the information base for the
forecast is updated every 15 minutes. [...] This approach is often called
one-step-ahead forecasting.").

To reproduce benchmark numbers, the developer should:
1. Run the Phase 1 overnight estimation for each day in the evaluation period.
2. For each day, execute the Phase 2 dynamic forecasting loop (one bin at a
   time, updating e_last with actual observed turnover after each bin).
3. Compute MSE and MAPE on the one-step-ahead forecasts (i.e., comparing each
   bin's forecast with the actual turnover that was observed one step later).

Static (multi-step) forecasts will produce substantially higher error rates and
should not be compared against the benchmarks below.

### Expected Behavior

1. **Common component shape:** The common component c_{i,t} should exhibit a
   clear U-shaped pattern across the trading day: high at market open, declining
   through midday, rising again toward close. This shape should be broadly
   similar across stocks. (BDF 2008, Section 1, Section 3.1, Fig. 3)

2. **Specific component properties:** The specific component e_{i,t} should be
   mean-zero (by construction of PCA with centering), weakly autocorrelated,
   and exhibit no strong intraday periodicity. Its ACF should decay quickly
   (within a few lags) compared to raw turnover. (BDF 2008, Section 3.1,
   Fig. 2)

3. **Factor count:** For a typical universe of 30-40 liquid stocks, the Bai-Ng
   criterion should select r = 1 to 3 factors. The first factor alone
   typically explains the majority of cross-sectional variation. (BDF 2008,
   Section 2.2; Researcher inference based on the paper's use of "common
   component" in singular throughout most of the text.)

4. **Forecast accuracy (MAPE):**
   - BDF-SETAR in-sample: ~0.075 MAPE on CAC40 (BDF 2008, Table 2, first
     panel "Performance of model for intraday volume", PC-SETAR row, Mean
     column). This is computed over the in-sample estimation period
     (September-December 2003).
   - BDF-SETAR rolling out-of-sample: ~0.399 MAPE on DJIA (Szucs 2017, full
     sample results with 20-day rolling window, 2,648 forecast days). Note:
     the higher MAPE reflects the 11-year sample (2001-2012) including the
     2008 financial crisis and broader market heterogeneity.
   - BDF-AR rolling out-of-sample: ~0.403 MAPE on DJIA (Szucs 2017, full
     sample results).
   - Both should beat the U-method benchmark (historical average only):
     ~0.091 MAPE on CAC40 in-sample (BDF 2008, Table 2, first panel,
     Classical row), ~0.503 MAPE on DJIA rolling out-of-sample (Szucs 2017).

5. **VWAP tracking error (out-of-sample):**
   - Dynamic PCA-SETAR: ~0.090 MAPE on CAC40 (BDF 2008, Table 2, third panel
     "Result of out-of-sample estimation for VWAP execution", PC/SETAR with
     dynamical adjustment row, Mean = 0.0898).
   - Theoretical PCA-SETAR (static): ~0.098 MAPE on CAC40 (BDF 2008, Table 2,
     third panel, PC/SETAR theoretical row, Mean = 0.0975).
   - Classical approach: ~0.102 MAPE on CAC40 out-of-sample (BDF 2008,
     Table 2, third panel, Classical row, Mean = 0.1016).
   - The dynamic execution should reduce VWAP tracking error vs. classical,
     with larger improvements for high-volatility stocks.
     (BDF 2008, Section 4.3.2)

6. **Estimation speed:** The full pipeline (PCA + AR/SETAR) for 33 stocks over
   2,668 total days (2,648 forecast days after the initial 20-day window)
   should run in approximately 2 hours. (Szucs 2017, Section 5: estimation
   speed for the BDF model. Note: this timing is consistent with the efficient
   N x N eigendecomposition, not the P x P approach.)

### Sanity Checks

1. **Decomposition additivity:** For every observation, verify that
   K_hat[t, i] + e_hat[t, i] equals X_centered[t, i] up to floating-point
   precision. Equivalently, c_forecast[j, i] + e_forecast should approximate
   the original (uncentered) turnover when e_forecast = 0 and c_forecast
   includes col_means. (Direct consequence of BDF 2008, Eq. 4)

2. **Common component forecast ~ U-method:** If r is large enough, the common
   component forecast should approximate the U-method (simple historical
   average of turnover at the same time-of-day). Compare c_forecast[j, i]
   with the U-method benchmark for a few stocks. They should be similar but
   not identical (the PCA decomposition separates common from specific before
   averaging). (BDF 2008, Section 2.3)

3. **Specific component mean:** The average of e_{i,t} across the estimation
   window should be approximately zero for each stock (by PCA construction
   with centering). Verify abs(mean(e_hat[:, i])) < 0.001 * mean(abs(e_hat[:, i])).
   (Consequence of column-mean centering before PCA.)

4. **AR(1) stationarity:** The estimated theta_1 should satisfy |theta_1| < 1
   for each stock. If clamping is triggered (|theta_1| >= 1), log a warning
   and investigate whether the stock's specific component is anomalous.
   (Standard AR stationarity condition; handling per Researcher inference in
   fit_ar1.)

5. **SETAR regime balance:** Each regime should contain at least 15% of
   observations (by construction of the tau grid search over [15th, 85th]
   percentile). Verify that the threshold tau produces a reasonable split.
   (Researcher inference from the percentile grid restriction.)

6. **Multi-step forecast decay:** For the AR(1) model, multi-step-ahead
   forecasts should decay toward the unconditional mean c / (1 - theta_1) as
   the horizon increases. At horizon h, the forecast is approximately
   c / (1 - theta_1) + theta_1^h * (e_last - c / (1 - theta_1)). Verify
   that 5-step-ahead forecasts are substantially attenuated. This is why
   static execution performs poorly. (BDF 2008, Section 4.2.1: "long-horizon
   ARMA forecasts collapse to zero and the dynamic component contributes
   nothing.")

7. **Non-negative forecasts:** The combined forecast x_hat = c_forecast + e_forecast
   should be non-negative for all bins. If negative forecasts occur, clamp to
   zero. Check what fraction of forecasts require clamping -- it should be
   very small (<1%) for liquid stocks. (Researcher inference: turnover is
   non-negative by definition.)

### Edge Cases

1. **Zero-volume bins:** The model assumes all stocks have non-zero volume in
   every bin. If a stock has zero volume in a bin, its turnover is zero and the
   specific component will reflect this. This is not problematic for the
   additive decomposition (unlike log-based models). However, zero-volume bins
   within the estimation window may destabilize the SETAR threshold estimation
   and will produce division-by-zero in MAPE evaluation.
   **Handling:** Exclude illiquid stocks (those with zero-volume bins) from the
   cross-section used for PCA. (BDF 2008, Section 3.1 implicitly assumes
   non-zero volume; Szucs 2017, Section 2: "every stock had trades and thus a
   volume record larger than zero in every 15-minute interval.")

2. **Stock universe changes:** When a stock enters or exits the index (e.g., due
   to corporate actions), the cross-sectional dimension N changes. The PCA
   must be re-estimated with the current universe. Stocks with fewer than L
   days of history in the estimation window should be excluded.
   **Handling:** Use the intersection of stocks available for the full L-day
   estimation window.

3. **Partial trading days:** Early market closes (e.g., day before holidays)
   produce fewer than k bins. These days must be excluded from the estimation
   window. (BDF 2008, Section 3.1: Christmas Eve and New Year's Eve excluded.)

4. **Market open/close effects:** The first and last bins of the day typically
   have much higher volume than midday bins. The common component should
   capture this pattern. Column-mean centering before PCA ensures the first
   factor captures shape variation rather than the overall level.
   (Researcher inference: centering rationale.)

5. **Large return days:** On days with extreme market moves, the specific
   component may be unusually large, potentially causing the AR/SETAR model
   to produce extreme forecasts for the following day. The SETAR model is
   better equipped to handle this via regime switching. (BDF 2008, Section 3.2:
   SETAR outperforms especially for high-volatility stocks.)

6. **Negative combined forecasts:** The additive decomposition can produce
   negative forecasts when the specific component is large and negative.
   **Handling:** Clamp to zero. This is rare for liquid stocks. (Researcher
   inference.)

7. **Non-stationary AR/SETAR coefficients:** If OLS estimation produces
   |theta_1| >= 1 (or |theta_12| or |theta_22| >= 1 for SETAR), the model
   is non-stationary and multi-step forecasts will diverge.
   **Handling:** Clamp the coefficient to sign(theta) * 0.99 and log a warning.
   If clamping is frequent for a stock, consider excluding it or falling back
   to the U-method. (Researcher inference: neither paper addresses this edge
   case.)

### Known Limitations

1. **Requires a cross-section:** The model cannot be applied to a single stock
   in isolation. It requires a universe of at least ~20 stocks for PCA factor
   estimation to be reliable (Bai 2003 consistency requires both N and T to be
   large). (BDF 2008, Section 2.2)

2. **Static execution is inferior:** Using the model's full-day forecast at
   market open (without intraday updating) is worse than the simple historical
   average. Multi-step AR(1) forecasts decay toward zero, neutralizing the
   dynamic component. Dynamic execution is essential. (BDF 2008, Section 4.2.1)

3. **No price information:** The model is purely volume-driven and does not
   incorporate price dynamics. It cannot "beat" VWAP -- only track it. A
   bivariate volume-price model would be needed to exploit price predictability.
   (BDF 2008, Section 5)

4. **Common component stability assumption:** The model assumes the U-shaped
   pattern is stable enough to be forecast by a historical average. During
   market stress or structural breaks (e.g., market microstructure changes),
   this assumption may fail.

5. **SETAR threshold is fixed within estimation window:** The threshold tau is
   re-estimated daily but remains fixed within each day. Rapid intraday regime
   changes may not be captured. (BDF 2008, Section 2.3)

6. **Small order assumption:** The VWAP execution framework assumes the trader's
   order is small relative to market volume (no price impact). For large
   orders, participation-rate limits or multi-day execution is needed.
   (BDF 2008, Section 4.1.3)

7. **No handling of market events:** The model has no mechanism for special
   calendar days (option expiry, earnings, FOMC), circuit breakers, or trading
   halts. These would need to be addressed by excluding affected days or
   adding event-specific overlays. (Researcher inference: neither paper
   addresses event-driven volume anomalies.)

8. **Overnight discontinuity in specific component series:** The AR(1)/SETAR
   model is fit to a concatenated series where bin k of day d is immediately
   followed by bin 1 of day d+1. The overnight gap between these observations
   is fundamentally different from the 15-20 minute gap within a day, but the
   time-series model treats them identically. The estimated AR coefficient
   reflects a mix of intraday persistence and overnight dynamics. This is
   consistent with both BDF 2008 and Szucs 2017's implementations (neither
   paper discusses this discontinuity), but the developer should be aware
   that it may slightly bias the persistence estimate. (Researcher inference:
   practical observation about the concatenated series structure.)

## Paper References

| Spec Section | Paper Source |
|---|---|
| Model decomposition: x = c + e | BDF 2008, Section 2.2, Eq. 4 |
| PCA estimation (SVD-based) | BDF 2008, Section 2.2, Eq. 5-6; referencing Bai (2003). "The estimated factors matrix F_hat is proportional (up to T^{1/2}) to the eigenvectors corresponding to the r-largest eigenvalues of the X'X matrix." |
| Factor count selection (IC_p1) | BDF 2008, Section 2.2; referencing Bai & Ng (2002) |
| Common component forecast (time-of-day average) | BDF 2008, Section 2.3, Eq. 9 |
| AR(1) specific component model | BDF 2008, Section 2.3, Eq. 10; Szucs 2017, Section 4.1, Eq. 5 |
| SETAR specific component model | BDF 2008, Section 2.3, Eq. 11; Szucs 2017, Section 4.1, Eq. 6-7 |
| SETAR threshold indicator I(z) | BDF 2008, Section 2.3, Eq. 11; Szucs 2017, Section 4.1, Eq. 7 |
| OLS equivalent to conditional MLE | BDF 2008, Section 2.3: "estimate Eqs. (10) or (11) by maximum likelihood" |
| Dynamic VWAP execution strategy | BDF 2008, Section 4.2.2 |
| Static vs. dynamic execution comparison | BDF 2008, Section 4.2.1 |
| MSE evaluation metric | Szucs 2017, Section 3, Eq. 1 |
| MAPE evaluation metric | Szucs 2017, Section 3, Eq. 2 |
| U-method benchmark formula | Szucs 2017, Section 3, Eq. 3; BDF 2008, Section 3.2, footnote 5 |
| Evaluation protocol (one-step-ahead) | BDF 2008, Section 4.2.2; Szucs 2017, Section 3 |
| Full-sample BDF vs. CMEM comparison | Szucs 2017, Section 5 |
| Estimation window = 20 days | BDF 2008, Section 3.2; Szucs 2017, Section 3 |
| Turnover = volume / TSO | BDF 2008, Section 3.1; Szucs 2017, Section 2 |
| In-sample MAPE benchmarks (CAC40) | BDF 2008, Table 2, first panel |
| Out-of-sample VWAP MAPE benchmarks (CAC40) | BDF 2008, Table 2, third panel |
| Per-stock VWAP results (theoretical) | BDF 2008, Table 4 |
| Per-stock VWAP results (dynamic) | BDF 2008, Table 5 |
| DJIA full-sample MAPE benchmarks | Szucs 2017, Section 5 |
| Sample size: 2,668 days | Szucs 2017, Table 1 |
| Column-mean centering before PCA | Researcher inference: standard practice; ensures factors capture shape variation rather than mean level |
| ARMA(1,1) label is actually AR(1) | Researcher inference: BDF 2008 Eq. 10 vs. text; confirmed by Szucs 2017 Eq. 5 |
| e_last initialization at market open | Researcher inference: using end-of-prior-day specific component |
| SETAR tau grid [15th, 85th] percentile | Researcher inference: standard practice for threshold AR models |
| Non-negative forecast clamping | Researcher inference: turnover is non-negative by definition |
| r_max = 20 upper bound | Researcher inference: practical choice, generous for volume data (r typically 1-3) |
| Stationarity clamping (|theta| >= 1) | Researcher inference: prevents divergent forecasts from non-stationary estimates |
| Overnight discontinuity in concatenated series | Researcher inference: practical observation about the time-series structure |
