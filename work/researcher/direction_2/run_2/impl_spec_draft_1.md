# Implementation Specification: PCA Factor Decomposition (BDF) for Intraday Volume

## Overview

The BDF model (Bialkowski, Darolles, Le Fol 2008) decomposes intraday stock turnover into
a market-wide common component and a stock-specific idiosyncratic component using
principal components analysis (PCA) on a cross-section of stocks. The common component
captures the U-shaped intraday seasonality shared across all stocks; the specific
component captures individual stock deviations and is modeled by either an AR(1) or
SETAR time-series process. Forecasts are produced dynamically: the common component is
forecast by a time-of-day historical average, and the specific component is forecast
one step ahead using the fitted AR/SETAR model, updating after each observed intraday
bin.

The model is designed for VWAP execution: given an accurate intraday volume profile
forecast, a trader can distribute order slices proportionally across bins to track the
daily VWAP. The dynamic variant updates forecasts intraday as actual volumes arrive,
reducing tracking error by ~10% versus historical-average baselines (BDF 2008, Table 2).

**Inputs:** Intraday turnover matrix for N stocks over a rolling window of L trading days,
each day divided into k bins.

**Outputs:** Per-stock per-bin turnover forecasts for the next trading day (or remaining
bins of the current day, updated dynamically).

## Algorithm

### Model Description

The model has two phases, executed daily:

**Phase 1 (Offline estimation, run once per day before market open):**
1. Construct the turnover matrix X from the rolling estimation window.
2. Extract common factors via PCA (eigendecomposition of X'X/T).
3. Recover common component C and specific component E = X - C.
4. Forecast the common component for the next day via time-of-day averaging.
5. Fit AR(1) or SETAR model to each stock's specific component series.

**Phase 2 (Online forecasting, run intraday after each observed bin):**
1. Observe actual turnover for the current bin.
2. Extract the observed specific component as actual minus common forecast.
3. Produce one-step-ahead specific component forecast via AR/SETAR.
4. Combine common and specific forecasts for the next bin.
5. Optionally compute VWAP execution weights for remaining bins.

Reference: BDF 2008, Sections 2.1-2.3 for Phase 1; Section 4.2.2 for Phase 2.

### Pseudocode

#### Phase 1: Offline Estimation

```
FUNCTION estimate_model(turnover_data, L_days, k_bins, r_max):
    """
    turnover_data: dict mapping stock_id -> array of shape (L_days, k_bins)
                   containing turnover values for the estimation window
    L_days:        number of trading days in rolling window (e.g., 20)
    k_bins:        number of intraday bins per day (e.g., 26)
    r_max:         maximum number of factors to consider (e.g., 10)

    Returns: model_params dict containing all fitted parameters
    """

    # Step 1: Construct turnover matrix X
    # X has shape (T, N) where T = L_days * k_bins, N = number of stocks
    # Rows ordered chronologically: day1_bin1, day1_bin2, ..., dayL_binK
    # Columns ordered by stock
    N = len(turnover_data)
    T = L_days * k_bins
    X = zeros(T, N)
    stock_ids = sorted(turnover_data.keys())
    for col_idx, stock_id in enumerate(stock_ids):
        X[:, col_idx] = turnover_data[stock_id].flatten()  # flatten (L_days, k_bins) -> (T,)
    # Reference: BDF 2008, Section 2.2, Eq. (5): X = FA' + e

    # Step 2: Select number of factors r using Bai & Ng (2002) IC
    r = select_num_factors(X, r_max)
    # Reference: BDF 2008, Section 2.2; Bai & Ng (2002)

    # Step 3: Extract factors and loadings via PCA
    F_hat, A_hat, eigenvalues = extract_factors(X, r)
    # Reference: BDF 2008, Section 2.2, Eq. (6)

    # Step 4: Compute common and specific components
    C_hat = F_hat @ A_hat                 # shape (T, N)
    E_hat = X - C_hat                      # shape (T, N)
    # Reference: BDF 2008, Section 2.2

    # Step 5: Forecast common component for next day
    # For each stock and each bin position j = 1..k_bins:
    # c_forecast[j] = mean of C_hat values at bin position j across all L_days days
    c_forecast = zeros(N, k_bins)
    for j in range(k_bins):
        bin_indices = [d * k_bins + j for d in range(L_days)]
        c_forecast[:, j] = mean(C_hat[bin_indices, :], axis=0)
    # Reference: BDF 2008, Section 2.3, Eq. (9)

    # Step 6: Fit specific component models per stock
    ar_params = {}       # stock_id -> (psi_1, psi_2, sigma_eps)
    setar_params = {}    # stock_id -> (phi_11, phi_12, phi_21, phi_22, tau, sigma_eps)
    for col_idx, stock_id in enumerate(stock_ids):
        e_series = E_hat[:, col_idx]   # shape (T,)
        ar_params[stock_id] = fit_ar1(e_series)
        setar_params[stock_id] = fit_setar(e_series)
    # Reference: BDF 2008, Section 2.3, Eqs. (10)-(11)

    return {
        'stock_ids': stock_ids,
        'r': r,
        'eigenvalues': eigenvalues,
        'F_hat': F_hat,
        'A_hat': A_hat,
        'C_hat': C_hat,
        'E_hat': E_hat,
        'c_forecast': c_forecast,        # shape (N, k_bins)
        'ar_params': ar_params,
        'setar_params': setar_params,
        'k_bins': k_bins,
        'L_days': L_days,
    }
```

#### Step 2 Detail: Factor Count Selection (Bai & Ng IC)

```
FUNCTION select_num_factors(X, r_max):
    """
    X:     turnover matrix, shape (T, N)
    r_max: maximum number of factors to evaluate

    Returns: optimal r in {1, ..., r_max}
    """
    T, N = X.shape

    # Compute eigendecomposition of X'X / T
    Sigma = X.T @ X / T                  # shape (N, N)
    eigenvalues, eigenvectors = eig_descending(Sigma)
    # eigenvalues sorted descending: lambda_1 >= lambda_2 >= ... >= lambda_N
    # eigenvectors[:, j] corresponds to lambda_j

    best_r = 1
    best_ic = +inf

    for r in range(1, r_max + 1):
        # Compute residual variance for r factors
        V_r = eigenvectors[:, :r]                        # shape (N, r)
        D_r = diag(1.0 / sqrt(eigenvalues[:r]))          # shape (r, r)
        F_r = X @ V_r @ D_r                               # shape (T, r)
        A_r = F_r.T @ X / T                               # shape (r, N)
        C_r = F_r @ A_r                                   # shape (T, N)
        resid = X - C_r
        sigma_sq_r = sum(resid ** 2) / (N * T)

        # IC_p2 criterion (recommended by Bai & Ng 2002)
        penalty = r * ((N + T) / (N * T)) * ln(min(N, T))
        ic = ln(sigma_sq_r) + penalty

        if ic < best_ic:
            best_ic = ic
            best_r = r

    return best_r
    # Reference: Bai & Ng (2002), IC_p2; BDF 2008, Section 2.2
```

#### Step 3 Detail: Factor Extraction

```
FUNCTION extract_factors(X, r):
    """
    X: turnover matrix, shape (T, N)
    r: number of factors

    Returns: F_hat (T, r), A_hat (r, N), eigenvalues (r,)
    """
    T, N = X.shape

    # Eigendecomposition of X'X / T
    Sigma = X.T @ X / T                          # (N, N)
    eigenvalues_all, V_all = eig_descending(Sigma)

    # Take top r eigenvectors and eigenvalues
    V_r = V_all[:, :r]                            # (N, r)
    lambda_r = eigenvalues_all[:r]                 # (r,)

    # Compute normalized factors: F_hat'F_hat / T = I_r
    # F_hat = X @ V_r @ diag(1/sqrt(lambda_1..r))
    D_r = diag(1.0 / sqrt(lambda_r))              # (r, r)
    F_hat = X @ V_r @ D_r                          # (T, r)

    # Recover loadings
    A_hat = F_hat.T @ X / T                        # (r, N)

    return F_hat, A_hat, lambda_r
    # Reference: BDF 2008, Section 2.2, Eq. (6) and text below it
    # Normalization: concentrating out A and using F'F/T = I_r, the problem
    # reduces to maximizing tr(F'(Y)(Y')F), solved by eigenvectors of X'X.
    # The loadings are recovered as A_hat = (F_hat'F_hat)^{-1} F_hat' X = F_hat' X / T
    # since F_hat'F_hat/T = I_r implies F_hat'F_hat = T * I_r.
```

**Note on SVD alternative:** The eigendecomposition of X'X/T can equivalently be computed
via the thin SVD of X: if X = U S V', then F_hat = U[:, :r] * sqrt(T) (after appropriate
scaling) and A_hat = V[:, :r]' * S[:r] / T. SVD is numerically more stable and avoids
forming X'X explicitly. Either approach produces identical results up to floating-point
precision. (Researcher inference -- BDF 2008 does not discuss SVD but both yield the same
eigenspace.)

#### Step 6a Detail: AR(1) Estimation

```
FUNCTION fit_ar1(e_series):
    """
    e_series: specific component time series, shape (T,)

    Returns: (psi_1, psi_2, sigma_eps)
    """
    # AR(1) with intercept: e_t = psi_1 * e_{t-1} + psi_2 + epsilon_t
    # Estimate by OLS (equivalent to conditional MLE under Gaussian errors)
    T = len(e_series)

    # Construct regression: y = e[1:], x = [e[0:-1], ones]
    y = e_series[1:]                       # shape (T-1,)
    X_reg = column_stack([e_series[:-1], ones(T - 1)])   # shape (T-1, 2)

    # OLS: beta = (X'X)^{-1} X'y
    beta = solve(X_reg.T @ X_reg, X_reg.T @ y)
    psi_1 = beta[0]
    psi_2 = beta[1]

    # Residual standard deviation
    residuals = y - X_reg @ beta
    sigma_eps = sqrt(sum(residuals ** 2) / (T - 1 - 2))

    return (psi_1, psi_2, sigma_eps)
    # Reference: BDF 2008, Section 2.3, Eq. (10)
    # The paper says "estimate by maximum likelihood." Under Gaussian errors,
    # conditional MLE is equivalent to OLS. Reference: standard time series
    # result; BDF 2008 Section 2.3.
```

**Note on BDF's "ARMA(1,1)" label:** BDF 2008 Eq. (10) is labeled "ARMA(1,1)" but the
equation contains no moving average term -- it is e_{i,t} = psi_1 * e_{i,t-1} + psi_2 +
epsilon_{i,t}, which is AR(1) with an intercept. Szucs (2017) correctly labels this as
"AR(1)." We implement it as AR(1) with intercept. Reference: BDF 2008, Eq. (10); Szucs
2017, Section 4.1.

#### Step 6b Detail: SETAR Estimation

```
FUNCTION fit_setar(e_series):
    """
    e_series: specific component time series, shape (T,)

    Returns: (phi_11, phi_12, phi_21, phi_22, tau, sigma_eps)
    """
    # SETAR: e_t = (phi_11 * e_{t-1} + phi_12) * I(e_{t-1} <= tau)
    #            + (phi_21 * e_{t-1} + phi_22) * (1 - I(e_{t-1} <= tau))
    #            + epsilon_t
    # Estimation: profile MLE via grid search over tau
    # Under Gaussian errors, this reduces to minimizing total SSR

    T = len(e_series)
    y = e_series[1:]                         # shape (T-1,)
    e_lag = e_series[:-1]                    # shape (T-1,)

    # Grid of tau candidates: 15th to 85th percentile of lagged values
    # in 1-percentile steps (71 candidates)
    tau_candidates = [percentile(e_lag, p) for p in range(15, 86)]

    # Minimum regime size: at least max(15, 0.15 * (T-1)) observations
    min_obs = max(15, int(0.15 * (T - 1)))

    best_ssr = +inf
    best_params = None

    for tau in tau_candidates:
        # Split into two regimes
        regime_low = (e_lag <= tau)          # boolean mask
        regime_high = ~regime_low

        n_low = sum(regime_low)
        n_high = sum(regime_high)

        # Skip if either regime has too few observations
        if n_low < min_obs or n_high < min_obs:
            continue

        # Fit AR(1)+intercept in each regime by OLS
        # Regime 1 (low): y_low = phi_11 * e_lag_low + phi_12 + eps
        X_low = column_stack([e_lag[regime_low], ones(n_low)])
        y_low = y[regime_low]
        beta_low = solve(X_low.T @ X_low, X_low.T @ y_low)
        ssr_low = sum((y_low - X_low @ beta_low) ** 2)

        # Regime 2 (high): y_high = phi_21 * e_lag_high + phi_22 + eps
        X_high = column_stack([e_lag[regime_high], ones(n_high)])
        y_high = y[regime_high]
        beta_high = solve(X_high.T @ X_high, X_high.T @ y_high)
        ssr_high = sum((y_high - X_high @ beta_high) ** 2)

        total_ssr = ssr_low + ssr_high

        if total_ssr < best_ssr:
            best_ssr = total_ssr
            best_params = (beta_low[0], beta_low[1],
                           beta_high[0], beta_high[1],
                           tau)

    phi_11, phi_12, phi_21, phi_22, tau = best_params

    # Compute overall residual standard deviation
    residuals = zeros(T - 1)
    for t in range(T - 1):
        if e_lag[t] <= tau:
            residuals[t] = y[t] - (phi_11 * e_lag[t] + phi_12)
        else:
            residuals[t] = y[t] - (phi_21 * e_lag[t] + phi_22)
    sigma_eps = sqrt(sum(residuals ** 2) / (T - 1 - 5))
    # 5 = number of estimated parameters (phi_11, phi_12, phi_21, phi_22, tau)

    return (phi_11, phi_12, phi_21, phi_22, tau, sigma_eps)
    # Reference: BDF 2008, Section 2.3, Eq. (11)
    # Grid search approach: Hansen (1997), Tong (1990)
    # Researcher inference for grid resolution (1-percentile steps)
    # and minimum regime size (15% or 15 obs).
```

#### Phase 2: Online Dynamic Forecasting

```
FUNCTION forecast_dynamic(model_params, observed_turnover, use_setar=True):
    """
    model_params:      output of estimate_model()
    observed_turnover: dict mapping stock_id -> list of observed turnover values
                       for bins completed so far today (may be empty at day start)
    use_setar:         if True, use SETAR; if False, use AR(1)

    Returns: forecasts dict mapping stock_id -> array of shape (k_bins,)
             containing turnover forecasts for ALL bins today.
             For observed bins, the forecast equals the actual.
             For future bins, the forecast is common + specific forecast.
    """
    k_bins = model_params['k_bins']
    stock_ids = model_params['stock_ids']
    c_forecast = model_params['c_forecast']   # shape (N, k_bins)

    forecasts = {}
    for idx, stock_id in enumerate(stock_ids):
        obs = observed_turnover.get(stock_id, [])
        j_current = len(obs)   # number of bins observed so far (0 at day start)

        forecast_full = zeros(k_bins)

        # Copy observed actuals into forecast array
        for j in range(j_current):
            forecast_full[j] = obs[j]

        # Extract most recent specific component observation (if any)
        if j_current > 0:
            # e_observed = x_actual - c_forecast for the last observed bin
            e_last = obs[j_current - 1] - c_forecast[idx, j_current - 1]
        else:
            # Day start: no observations yet, initialize e = 0
            # Justification: unconditional mean of specific component is ~0
            # by PCA construction (factors absorb the mean structure)
            e_last = 0.0
            # Researcher inference: BDF 2008 does not specify day-start
            # initialization. Setting e=0 is consistent with Section 4.2.1
            # (static forecast uses common component only at day start).

        # Produce forecasts for remaining bins
        for j in range(j_current, k_bins):
            if use_setar:
                params = model_params['setar_params'][stock_id]
                phi_11, phi_12, phi_21, phi_22, tau, _ = params
                if e_last <= tau:
                    e_forecast = phi_11 * e_last + phi_12
                else:
                    e_forecast = phi_21 * e_last + phi_22
            else:
                params = model_params['ar_params'][stock_id]
                psi_1, psi_2, _ = params
                e_forecast = psi_1 * e_last + psi_2

            # Combine common + specific
            x_forecast = c_forecast[idx, j] + e_forecast

            # Floor negative forecasts to small positive value
            if x_forecast <= 0:
                x_forecast = 1e-8

            forecast_full[j] = x_forecast

            # For multi-step forecasts, feed forecast as input to next step
            # (deterministic iteration)
            e_last = e_forecast

        forecasts[stock_id] = forecast_full

    return forecasts
    # Reference: BDF 2008, Section 4.2.2
    # Multi-step iteration: BDF 2008 describes dynamic execution as
    # updating after each observed interval. For future bins beyond
    # the next one, we iterate the AR/SETAR recursion using point
    # forecasts as pseudo-observations. This is standard for AR models.
    # For SETAR, this is a deterministic approximation -- the exact
    # conditional expectation would require integrating over the error
    # distribution. (Researcher inference.)
```

#### VWAP Execution Weights

```
FUNCTION compute_vwap_weights(forecasts, stock_id, j_current, k_bins):
    """
    Given forecasts for a stock, compute the fraction of remaining order
    to execute in each future bin.

    forecasts:  output of forecast_dynamic()
    stock_id:   the stock to compute weights for
    j_current:  number of bins already observed/executed
    k_bins:     total bins per day

    Returns: weights array of shape (k_bins - j_current,)
             weights[0] = fraction to execute in bin j_current+1
             weights sum to 1.0
    """
    f = forecasts[stock_id]
    remaining = f[j_current:]                      # forecast for remaining bins
    total_remaining = sum(remaining)

    if total_remaining <= 0:
        # Fallback: uniform distribution
        n_remaining = k_bins - j_current
        return ones(n_remaining) / n_remaining

    weights = remaining / total_remaining

    return weights
    # Reference: BDF 2008, Section 4.2.2-4.2.3
    # The weight for bin j+1 is x_hat[j+1] / sum(x_hat[j+1:k]).
    # At each observed bin, the trader executes weight[0] fraction of
    # remaining order volume, then re-forecasts for subsequent bins.
```

#### Daily Rolling Update

```
FUNCTION daily_update(all_turnover_history, day_index, L_days, k_bins, r_max):
    """
    Run at end of each trading day to prepare next day's model.

    all_turnover_history: full historical turnover data, shape per stock (n_days, k_bins)
    day_index:            index of today (0-based)
    L_days:               rolling window length
    k_bins:               bins per day
    r_max:                max factors

    Returns: model_params for forecasting day_index + 1
    """
    # Construct estimation window: day_index - L_days + 1 through day_index
    start_day = day_index - L_days + 1
    if start_day < 0:
        raise Error("Insufficient history for estimation window")

    turnover_data = {}
    for stock_id, full_history in all_turnover_history.items():
        turnover_data[stock_id] = full_history[start_day:day_index + 1, :]

    return estimate_model(turnover_data, L_days, k_bins, r_max)
    # Reference: BDF 2008, Section 3.1; Szucs 2017, Section 3
    # Parameters re-estimated daily on rolling 20-day window.
```

### Data Flow

```
Input:
  turnover_data: dict[stock_id -> ndarray(L_days, k_bins)]
  Each value is float64, representing shares_traded / total_shares_outstanding
  Typical values: 1e-5 to 5e-2

Phase 1 Pipeline (offline):
  turnover_data                                     dict[str -> (L, k) float64]
    |
    v
  X = flatten and stack columns                     (T, N) float64, T = L*k
    |
    v
  Sigma = X'X / T                                   (N, N) float64
    |
    v
  eigendecompose(Sigma) -> eigenvalues, V           (N,) float64, (N, N) float64
    |
    v
  select_num_factors -> r                           int, typically 1-3
    |
    v
  V_r = V[:, :r]                                    (N, r) float64
  D_r = diag(1/sqrt(lambda[:r]))                    (r, r) float64
  F_hat = X @ V_r @ D_r                             (T, r) float64
  A_hat = F_hat.T @ X / T                           (r, N) float64
    |
    v
  C_hat = F_hat @ A_hat                             (T, N) float64
  E_hat = X - C_hat                                 (T, N) float64
    |
    v
  c_forecast = time-of-day mean of C_hat            (N, k) float64
    |
    v
  Per-stock: fit_ar1(E_hat[:, i]) -> (psi_1, psi_2, sigma)
  Per-stock: fit_setar(E_hat[:, i]) -> (phi_11, phi_12, phi_21, phi_22, tau, sigma)

Phase 2 Pipeline (online, after each bin j):
  observed_turnover[stock_id][:j]                   list of j float64 values
    |
    v
  e_last = obs[j-1] - c_forecast[stock, j-1]       float64 (or 0.0 if j=0)
    |
    v
  For bins j+1 to k:
    e_forecast = AR1(e_last) or SETAR(e_last)       float64
    x_forecast = c_forecast[stock, bin] + e_forecast float64
    floor(x_forecast, 1e-8) if negative
    e_last = e_forecast                             (feed forward)
    |
    v
  forecasts[stock_id] = [obs_1, ..., obs_j, x_hat_{j+1}, ..., x_hat_k]
                                                     (k,) float64

VWAP weights:
  weights = forecasts[j_current:] / sum(forecasts[j_current:])
                                                     (k - j_current,) float64, sum = 1.0
```

### Variants

**Implemented variant: SETAR (recommended).** BDF 2008 shows SETAR outperforms AR(1) for
36 of 39 CAC40 stocks (Section 3.2). Szucs 2017 confirms on 33 DJIA stocks: BDF-SETAR
achieves MAPE 0.399 vs BDF-AR 0.403 (Section 4). The AR(1) variant should also be
implemented as a comparison baseline and fallback (simpler, no threshold estimation).

**Excluded variant: Static execution.** BDF 2008 Section 4.2.1 demonstrates that static
execution (forecasting all bins at market open without intraday updates) performs worse
than the classical historical-average approach. Multi-step AR/SETAR forecasts decay toward
the unconditional mean, neutralizing the specific component. Only dynamic execution is
implemented.

**Excluded variant: Theoretical execution.** BDF 2008 Section 4.2 defines a "theoretical"
strategy using true one-step-ahead predictions (requiring knowledge of end-of-day total
volume). This is an unattainable upper bound, not implementable in real time.

## Parameters

| Parameter | Description | Recommended Value | Sensitivity | Range |
|-----------|-------------|-------------------|-------------|-------|
| k_bins | Number of intraday bins per trading day | 26 (15-min bins, 9:30-16:00) | Low | 13-52 (30-min to 5-min) |
| L_days | Rolling estimation window in trading days | 20 | Medium | 10-60 |
| r_max | Maximum number of factors to evaluate | 10 | Low | 5-20 |
| r | Selected number of factors (output of IC) | Typically 1-3 | Medium | 1-r_max |
| psi_1 | AR(1) autoregressive coefficient | Estimated per stock | High | (-1, 1) for stationarity |
| psi_2 | AR(1) intercept | Estimated per stock | Low | Unrestricted |
| phi_11 | SETAR low-regime AR coefficient | Estimated per stock | Medium | (-1, 1) |
| phi_12 | SETAR low-regime intercept | Estimated per stock | Low | Unrestricted |
| phi_21 | SETAR high-regime AR coefficient | Estimated per stock | Medium | (-1, 1) |
| phi_22 | SETAR high-regime intercept | Estimated per stock | Low | Unrestricted |
| tau | SETAR threshold | Estimated per stock | High | [15th, 85th percentile of e_lag] |
| sigma_eps | Innovation standard deviation | Estimated per stock | Informational | (0, inf) |
| N | Number of stocks in cross-section | >= 30 | Medium | 30-500+ |
| min_regime_frac | Minimum fraction of obs per SETAR regime | 0.15 | Low | 0.10-0.20 |
| min_regime_obs | Minimum absolute obs per SETAR regime | 15 | Low | 10-30 |

### Initialization

**PCA initialization:** No special initialization needed. The eigendecomposition of X'X/T
is a deterministic computation. The first estimation requires at least L_days of historical
turnover data.

**AR(1) initialization:** OLS estimation is deterministic given the data. No iterative
procedure or starting values required.

**SETAR initialization:** Grid search over tau is deterministic. Each candidate tau produces
a deterministic OLS fit.

**Day-start forecast initialization:** At j_current = 0 (no bins observed), set the
specific component forecast to zero for all stocks. The initial day forecast uses only the
common component. This is consistent with BDF 2008 Section 4.2.1 (static forecast as the
starting point before any intraday observations arrive). (Researcher inference.)

**Factor sign convention:** Eigenvector signs are indeterminate -- they may flip between
daily re-estimations. This does not affect the common component C = F @ A because sign
flips cancel in the product. No special handling required. Optionally, enforce a convention
(e.g., require the first element of each loading vector to be positive) for interpretability
only. (Researcher inference.)

### Calibration

The model is re-calibrated daily on a rolling window. The calibration procedure is:

1. At end of trading day d, collect turnover data for days [d - L_days + 1, ..., d].
2. Run `estimate_model()` to produce parameters for day d + 1.
3. Store model_params for use during day d + 1's dynamic forecasting.

Reference: BDF 2008, Section 3.1; Szucs 2017, Section 3.

There is no iterative calibration, no gradient-based optimization, and no convergence
criterion. All estimation steps are closed-form (eigendecomposition + OLS).

**Initial warm-up:** The first forecast day requires L_days of prior turnover history.
Before that, no forecasts can be produced.

**Re-estimation frequency:** Daily (after market close). Parameters are fixed within
each trading day; only the specific component forecast is updated intraday as new
observations arrive. Reference: BDF 2008, Section 3.1; Szucs 2017, Section 3 ("parameters
of the models are re-estimated daily").

## Validation

### Expected Behavior

**Per-stock per-bin MAPE (Szucs 2017 scale):**
- BDF-SETAR: ~0.399 (Szucs 2017, Section 4, full-sample results for 33 DJIA stocks)
- BDF-AR: ~0.403 (Szucs 2017, Section 4)
- U-method benchmark: ~0.503 (Szucs 2017, Section 4)
- Both BDF variants reduce MAPE by ~20% relative to U-method.
- Reference: Szucs 2017, Section 4, full-sample MSE/MAPE table.

**Per-stock per-bin MSE (Szucs 2017 scale):**
- BDF-AR: 6.49e-4 (Szucs 2017, Section 4)
- BDF-SETAR: 6.60e-4 (Szucs 2017, Section 4)
- U-method: 1.02e-3 (Szucs 2017, Section 4)
- Note: By MSE, AR(1) slightly outperforms SETAR (lower MSE despite higher MAPE).
  This is because MSE penalizes large errors more, and SETAR occasionally produces
  larger outlier errors despite lower typical errors.
- Reference: Szucs 2017, Section 4.

**Portfolio-level MAPE (BDF 2008 scale):**
- PCA-SETAR: mean 0.0752, std 0.0869, Q95 0.2010 (BDF 2008, Table 2, panel 1)
- PCA-ARMA: mean 0.0829, std 0.0973, Q95 0.2330 (BDF 2008, Table 2, panel 1)
- Classical: mean 0.0905, std 0.1050, Q95 0.2490 (BDF 2008, Table 2, panel 1)
- Note: Portfolio-level MAPE is ~5x lower than per-stock because averaging turnover
  across stocks diversifies away idiosyncratic noise.

**VWAP tracking error (BDF 2008 scale, out-of-sample):**
- Dynamic PCA-SETAR: mean 0.0898 (BDF 2008, Table 2, panel 3)
- Dynamic PCA-ARMA: mean 0.0922 (BDF 2008, Table 2, panel 3)
- Classical: mean 0.1006 (BDF 2008, Table 2, panel 3)
- Dynamic PCA-SETAR reduces tracking error by ~10% vs classical.
- Per-stock reductions up to ~36% for high-volatility stocks (BDF 2008, Tables 6-7;
  e.g., CAP GEMINI: classical 0.2323, dynamic SETAR 0.1491).
- Reference: BDF 2008, Table 2 panel 3, Tables 6-7.

**MAPE formula (Szucs 2017, Eq. 2):**
```
MAPE = (1/N) * sum_{t=1}^{N} |Y_t - Y_t^f| / Y_t
```
where Y_t is actual turnover, Y_t^f is forecast, N is number of observations.
Computed per stock per bin, then averaged.

**MSE formula (Szucs 2017, Eq. 1):**
```
MSE = (1/N) * sum_{t=1}^{N} (Y_t - Y_t^f)^2
```

### Sanity Checks

1. **Common component U-shape:** Plot c_forecast (averaged across stocks) versus bin
   index. Should show clear U-shape: high at open (bin 1), declining through midday
   (bins 10-16), rising into close (bin 26). Reference: BDF 2008, Section 3.1, Fig. 3.

2. **Common component variance ratio:** For each stock, compute Var(C_hat[:, i]) /
   Var(X[:, i]). Should be > 0.5 for most liquid stocks. If < 0.30 for most stocks,
   PCA is misconfigured or cross-section is too narrow. (Researcher inference -- motivated
   by BDF 2008 Section 3.1 which shows common component captures the dominant seasonal
   pattern.)

3. **Specific component mean near zero:** For each stock, mean(E_hat[:, i]) should be
   approximately zero (within a few percent of mean(X[:, i])). By PCA construction,
   the common component absorbs the level. Reference: BDF 2008, Section 2.2.

4. **Eigenvalue scree plot:** The first r eigenvalues should be clearly separated from
   the remaining eigenvalues (visible gap in scree plot). A smooth decay without gap
   suggests weak factor structure. If IC selects r > 5, investigate data quality.
   (Researcher inference.)

5. **AR(1) stationarity:** For all stocks, |psi_1| < 1. If |psi_1| >= 1, the specific
   component is non-stationary, suggesting a data issue or improper decomposition.
   Reference: standard AR stationarity condition.

6. **SETAR regime coverage:** After fitting, verify that both regimes have substantial
   observations. If one regime has < 15% of data, the SETAR model degenerates to AR(1).
   Reference: BDF 2008, Section 2.3, Eq. (11).

7. **SETAR beats AR(1) for most stocks:** SETAR should produce lower out-of-sample MAPE
   than AR(1) for at least 70% of stocks. BDF 2008 shows only 3 of 39 stocks favor AR;
   Szucs 2017 shows 26-30 of 33 stocks favor SETAR by MAPE. Reference: BDF 2008,
   Section 3.2; Szucs 2017, Section 4.

8. **Forecast positivity:** After combining common + specific forecasts, the vast majority
   of bin forecasts should be positive without needing the floor. If > 10% of forecasts
   hit the floor, something is wrong with the decomposition or model parameters.
   (Researcher inference.)

9. **Toy PCA verification:** Construct a synthetic (T=200, N=5) matrix X = F @ A + noise,
   where F is a known (200, 1) factor with U-shape pattern and A is (1, 5) loadings.
   Run PCA with r=1. Verify F_hat is proportional to F (up to sign), C_hat approximates
   F @ A, and E_hat approximates noise. (Researcher inference.)

10. **Forecast improvement over U-method:** Compute MAPE for both the BDF dynamic forecast
    and the U-method (time-of-day average). BDF should produce lower MAPE for a clear
    majority of stocks. Reference: Szucs 2017, Section 4 (33/33 stocks by MAPE).

### Edge Cases

1. **Zero-volume bins:** If a stock has zero volume in a bin, turnover is zero and MAPE
   is undefined (division by zero). Exclude such bins from MAPE computation. For PCA
   estimation, replace isolated zero-volume bins with a small positive value (e.g.,
   1 share / TSO). Stocks with frequent zero-volume bins should be excluded entirely
   from the cross-section. Reference: Szucs 2017, Section 2 ("every stock had trades
   and thus a volume record larger than zero in every 15-minute interval").

2. **Half-days and early closures:** Days with fewer than k_bins intervals must be
   excluded from the estimation window entirely. They would create shape mismatches in
   the turnover matrix. Reference: BDF 2008, Section 3.1 ("24th and 31st of December
   2003 were excluded from the sample").

3. **Negative forecasts:** When the specific component forecast is large and negative,
   the combined forecast c + e may be negative. Floor at 1e-8 (effectively zero
   volume). If more than 50% of remaining bins produce negative raw forecasts, fall
   back to common-component-only forecasts (set e_forecast = 0) to prevent pathological
   concentration of VWAP weights. (Researcher inference.)

4. **Cross-section changes (IPOs/delistings):** Maintain a stable stock universe within
   each estimation window. Remove stocks that delist during the window (drop from all
   days). Wait for newly listed stocks to accumulate a full L_days window before
   including them. (Researcher inference.)

5. **Extreme volume days:** Days with extreme volume spikes (e.g., index rebalancing,
   earnings, circuit breakers) may distort PCA estimation. No special handling is
   specified in BDF 2008. The 20-day rolling window limits exposure to any single extreme
   day. Optionally, Winsorize turnover at the 99th percentile before PCA. (Researcher
   inference.)

6. **Cross-day boundary contamination:** The specific component series E_hat is
   concatenated across days (day1_bin25, day2_bin1, ...). Overnight jumps between the
   last bin of one day and the first bin of the next create artificial dynamics in the
   AR/SETAR model. BDF 2008 does not address this. The effect is: inflated innovation
   variance and attenuated AR coefficients. An alternative (not in the paper) is to fit
   AR/SETAR only on within-day observations pooled across days, treating each day as an
   independent segment. (Researcher inference for limitation description; concatenated
   fitting follows BDF 2008.)

7. **Singular or near-singular X'X:** If N is very large relative to T, or if some
   stocks have near-identical turnover patterns, X'X/T may be ill-conditioned. Use
   SVD instead of eigendecomposition for numerical stability. (Researcher inference.)

### Known Limitations

1. **Requires cross-section:** The model cannot be applied to a single stock. PCA needs
   a cross-section of N >= 30 stocks for Bai (2003) asymptotics to provide consistent
   factor estimation. Reference: BDF 2008, Section 2.2; Bai (2003).

2. **No daily volume forecast:** The model forecasts the intraday distribution of volume
   (how volume is spread across bins), not the absolute daily total. To convert turnover
   forecasts into share quantities for order slicing, the trader needs an external
   estimate of total daily volume. Reference: BDF 2008, Section 4.2.2 ("to predict the
   future sequence of volume").

3. **No price impact modeling:** The model assumes the trader's order is small relative
   to market volume. Large orders that move the price require a bivariate volume-price
   model. Reference: BDF 2008, Section 5.

4. **Linear factor model:** The common component is a linear combination of factors.
   Nonlinear cross-sectional structure (e.g., sector-specific patterns different from
   market-wide) is captured only approximately. Reference: BDF 2008, Section 2.2.

5. **SETAR threshold static within estimation window:** The threshold tau is estimated
   once per window and held fixed for the next day. Abrupt regime changes within a day
   are captured only through the threshold comparison, not through tau adaptation.
   Reference: BDF 2008, Section 2.3.

6. **Requires total shares outstanding (TSO) data:** Turnover computation requires
   current TSO from an external data source (e.g., Bloomberg). Reference: Szucs 2017,
   Section 2.

7. **Overnight boundary contamination:** The concatenated time series used for AR/SETAR
   fitting includes cross-day transitions, which may distort the fitted dynamics.
   See Edge Cases item 6. Reference: researcher inference based on BDF 2008 methodology.

## Paper References

| Spec Section | Paper Source |
|---|---|
| Overview, model description | BDF 2008, Sections 1-2 |
| PCA decomposition: X = FA' + e | BDF 2008, Section 2.2, Eqs. (4)-(5) |
| Eigendecomposition and normalization | BDF 2008, Section 2.2, Eq. (6) and surrounding text |
| Factor count selection (IC_p2) | BDF 2008, Section 2.2; Bai & Ng (2002) |
| Common component forecast | BDF 2008, Section 2.3, Eq. (9) |
| AR(1) specific component model | BDF 2008, Section 2.3, Eq. (10); Szucs 2017, Section 4.1 |
| SETAR specific component model | BDF 2008, Section 2.3, Eq. (11) |
| MLE = OLS equivalence | BDF 2008, Section 2.3 |
| Dynamic VWAP execution | BDF 2008, Section 4.2.2 |
| Static execution excluded | BDF 2008, Section 4.2.1 |
| MAPE/MSE formulas | Szucs 2017, Section 3, Eqs. (1)-(2) |
| U-method benchmark | Szucs 2017, Section 4, Eq. (3) |
| Per-stock MAPE targets | Szucs 2017, Section 4, results table |
| Portfolio-level MAPE targets | BDF 2008, Table 2 |
| VWAP tracking error targets | BDF 2008, Table 2 panel 3, Tables 6-7 |
| Turnover definition | Szucs 2017, Section 2; BDF 2008, Section 3.1 |
| Data parameters (26 bins, 20 days) | Szucs 2017, Section 2-3; BDF 2008, Section 3.1 |
| SETAR grid search | Hansen (1997); Tong (1990); BDF 2008, Section 2.3 |
| Bai (2003) PCA consistency | BDF 2008, Section 2.2 |
| Demeaning not required | Bai (2003); BDF 2008, Section 2.2 (absence of demeaning) |
| Computation time comparison | Szucs 2017, Section 4 |
| Day-start initialization (e=0) | Researcher inference; consistent with BDF 2008, Section 4.2.1 |
| Multi-step SETAR deterministic iteration | Researcher inference; standard approximation |
| Negative forecast floor | Researcher inference |
| Factor sign indeterminacy | Researcher inference; standard PCA property |
| Cross-day boundary contamination | Researcher inference |
| SVD alternative to eigendecomposition | Researcher inference |
| SETAR grid resolution and min regime size | Researcher inference; Hansen (1997), Tong (1990) |
| Zero-volume handling | Researcher inference; Szucs 2017, Section 2 |
