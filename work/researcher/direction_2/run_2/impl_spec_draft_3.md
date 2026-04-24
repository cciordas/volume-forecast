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
1. Preprocess raw volume data into turnover matrix X.
2. Construct the turnover matrix X from the rolling estimation window.
3. Extract common factors via PCA (eigendecomposition of X'X/T).
4. Recover common component C and specific component E = X - C.
5. Forecast the common component for the next day via time-of-day averaging.
6. Fit AR(1) or SETAR model to each stock's specific component series.

**Phase 2 (Online forecasting, run intraday after each observed bin):**
1. Observe actual turnover for the current bin.
2. Extract the observed specific component as actual minus common forecast.
3. Produce one-step-ahead specific component forecast via AR/SETAR.
4. Combine common and specific forecasts for the next bin.
5. Compute VWAP execution weights for remaining bins.

Reference: BDF 2008, Sections 2.1-2.3 for Phase 1; Section 4.2.2 for Phase 2.

### Data Preprocessing

Before constructing the turnover matrix X, raw market data must be converted and cleaned:

```
FUNCTION preprocess_turnover(raw_volume_data, shares_data, k_bins, L_days):
    """
    raw_volume_data: dict mapping stock_id -> array of shape (n_days, k_bins)
                     containing raw share volumes per bin
    shares_data:     dict mapping stock_id -> array of shape (n_days,)
                     containing total shares outstanding (TSO) per day
                     Alternatively, use float-adjusted shares if available.
    k_bins:          number of intraday bins per day
    L_days:          estimation window length

    Returns: turnover_data dict mapping stock_id -> array of shape (L_days, k_bins)
    """

    # Step 1: Compute turnover = shares_traded / shares_outstanding
    # BDF 2008 Section 3.1 uses "traded shares / float shares" adjusted for
    # splits and dividends. Szucs 2017 Section 2 uses V_t / TSO_t.
    # Either denominator works; be consistent. For US equities, float-adjusted
    # shares are available from CRSP or Bloomberg. Using TSO is simpler and
    # matches Szucs 2017.
    turnover_data = {}
    for stock_id in raw_volume_data:
        volume = raw_volume_data[stock_id]       # (n_days, k_bins)
        tso = shares_data[stock_id]               # (n_days,)
        # Broadcast: divide each bin's volume by that day's TSO
        turnover = volume / tso[:, newaxis]       # (n_days, k_bins)
        turnover_data[stock_id] = turnover
    # Reference: Szucs 2017, Section 2 (x_t = V_t / TSO_t);
    #            BDF 2008, Section 3.1 ("traded shares / float shares")

    # Step 2: Exclude half-days and early closures
    # Days with fewer than k_bins intervals must be excluded entirely.
    # They create shape mismatches in the turnover matrix.
    # Reference: BDF 2008, Section 3.1 ("24th and 31st of December 2003
    # were excluded from the sample")
    # Implementation: the caller should filter these days before passing
    # raw_volume_data. A day is a half-day if the exchange closes early
    # (e.g., day before Thanksgiving, Christmas Eve in US markets).

    # Step 3: Handle stock splits and corporate actions
    # TSO should reflect post-split share counts. If using raw volume
    # from before a split, adjust volume by the split ratio.
    # Reference: BDF 2008, Section 3.1 ("data is adjusted for the stock's
    # splits and dividends")

    # Step 4: Handle zero-volume bins
    # If a stock has zero volume in a bin, turnover is zero.
    # For PCA estimation, isolated zeros are acceptable (they are valid
    # data points indicating no trading). Stocks with frequent zero-volume
    # bins (e.g., > 5% of bins) should be excluded from the cross-section
    # as they indicate insufficient liquidity for reliable estimation.
    # Reference: Szucs 2017, Section 2 ("every stock had trades and thus
    # a volume record larger than zero in every 15-minute interval")

    # Step 5 (optional): Winsorize extreme turnover values
    # Days with extreme volume spikes (index rebalancing, earnings, circuit
    # breakers) may distort PCA estimation. Optionally clip turnover at the
    # 99th percentile per stock before PCA. The 20-day rolling window limits
    # exposure to any single extreme day. (Researcher inference.)

    return turnover_data
```

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

    # IMPORTANT: Do NOT mean-center the columns of X.
    # PCA is applied to the raw turnover matrix directly. Bai (2003) shows
    # that PCA estimation of factors is consistent without demeaning when
    # factors capture the mean structure (which they do here -- the common
    # component includes the U-shaped level). If using a library PCA
    # implementation (e.g., sklearn.decomposition.PCA), you MUST set
    # center=False or equivalent to disable automatic mean centering.
    # The eigendecomposition of X'X/T below operates on raw (uncentered) data.
    # Reference: BDF 2008, Section 2.2 (no demeaning step);
    #            Bai (2003), cited in BDF Section 2.2

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
    # NOTE: This averages the ESTIMATED common component (C_hat), not raw
    # turnover (X). The raw-turnover average is the "U-method" baseline.
    # The BDF common component forecast is thus the U-method applied to C_hat
    # rather than X. The predictive improvement over U-method comes entirely
    # from the specific component AR/SETAR forecast.
    # NOTE: This forecast is not truly out-of-sample within the estimation
    # window -- the same data used to estimate factors provides the averaged
    # common component. This is how BDF describes it (Section 2.3, Eq. 9)
    # and is a design choice, not a bug.
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
    X:     turnover matrix, shape (T, N). Must NOT be mean-centered.
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
        # Optimization: sigma_sq_r can equivalently be computed from eigenvalues
        # as (sum(eigenvalues_all) - sum(eigenvalues_all[:r])) / N, avoiding
        # the O(r * N * T) matrix reconstructions above. This works because
        # F_hat'F_hat/T = I_r by construction, so the residual sum of squares
        # equals N * T * (mean eigenvalue of omitted components). The explicit
        # reconstruction above is retained for pedagogical clarity.

        # IC_p2 criterion (Bai & Ng 2002)
        # NOTE: BDF 2008 Section 2.2 cites Bai & Ng (2002) for factor selection
        # but does not specify which of the three IC criteria (IC_p1, IC_p2,
        # IC_p3) they used. IC_p2 is the most commonly used in practice and
        # is our default. For robustness, consider computing all three and
        # selecting r by majority vote or the most conservative (largest r).
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
    X: turnover matrix, shape (T, N). Must NOT be mean-centered.
    r: number of factors

    Returns: F_hat (T, r), A_hat (r, N), eigenvalues (r,)
    """
    T, N = X.shape

    # Validate dimension requirements
    assert r <= min(N, T), (
        f"Number of factors r={r} must be <= min(N={N}, T={T})={min(N, T)}"
    )

    # Choose computation path based on N vs T
    if N <= T:
        # Standard path: eigendecompose X'X/T, shape (N, N)
        # Efficient when N < T (typical: N=30-50, T=520)
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

    else:
        # Large-N path: eigendecompose XX'/T, shape (T, T)
        # More efficient when N > T (e.g., N=500, T=260)
        # The top r eigenvectors of XX'/T give F_hat directly (up to scaling).
        Sigma_alt = X @ X.T / T                        # (T, T)
        eigenvalues_all, U_all = eig_descending(Sigma_alt)

        lambda_r = eigenvalues_all[:r]                 # (r,)
        U_r = U_all[:, :r]                             # (T, r)

        # F_hat = U_r * sqrt(T) to ensure F_hat'F_hat / T = I_r
        F_hat = U_r * sqrt(T)                          # (T, r)

        # Recover loadings
        A_hat = F_hat.T @ X / T                        # (r, N)

    return F_hat, A_hat, lambda_r
    # Reference: BDF 2008, Section 2.2, Eq. (6) and text below it
    # Normalization: concentrating out A and using F'F/T = I_r, the problem
    # reduces to maximizing tr(F'(Y)(Y')F), solved by eigenvectors of X'X.
    # The loadings are recovered as A_hat = (F_hat'F_hat)^{-1} F_hat' X = F_hat' X / T
    # since F_hat'F_hat/T = I_r implies F_hat'F_hat = T * I_r.
    # The large-N path (XX'/T) is researcher inference -- BDF 2008 uses X'X
    # but both yield identical common components C = F @ A.
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
    #
    # Notation mapping to papers:
    #   spec phi_11 = BDF Eq. (11) phi_1 (regime 1 AR coeff) = Szucs Eq. (6) theta_{1,2}
    #   spec phi_12 = BDF Eq. (11) phi_2 (regime 1 intercept) = Szucs Eq. (6) c_{1,1}
    #   spec phi_21 = BDF Eq. (11) phi_3 (regime 2 AR coeff) = Szucs Eq. (6) theta_{2,2}
    #   spec phi_22 = BDF Eq. (11) phi_4 (regime 2 intercept) = Szucs Eq. (6) c_{2,1}
    # NOTE: In Szucs Eq. (6) the intercept comes FIRST (c_{i,1}) and the AR coefficient
    # SECOND (theta_{i,2}), whereas this spec puts the AR coefficient first (phi_i1) and
    # intercept second (phi_i2). The ordering in the tuple returned by this function is
    # (AR_coeff, intercept, AR_coeff, intercept, tau, sigma). When cross-checking against
    # Szucs, do NOT confuse the coefficient ordering.
    #
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

    # Fallback: if no valid tau found (all candidates violate minimum regime
    # size), degenerate to AR(1) -- use the same parameters in both regimes
    if best_params is None:
        psi_1, psi_2, sigma_eps_ar = fit_ar1(e_series)
        tau_fallback = median(e_lag)
        return (psi_1, psi_2, psi_1, psi_2, tau_fallback, sigma_eps_ar)
        # Researcher inference: fallback needed for very short windows or
        # extreme distributions where all tau candidates violate regime size.

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
    # NOTE: tau is selected by grid search, not by continuous optimization,
    # so treating it as a standard estimated parameter for degrees of freedom
    # is an approximation. See Hansen (1997) for discussion of SETAR
    # degrees-of-freedom issues. This has negligible practical impact since
    # sigma_eps is informational and not used in forecasting.

    return (phi_11, phi_12, phi_21, phi_22, tau, sigma_eps)
    # Reference: BDF 2008, Section 2.3, Eq. (11)
    # Grid search approach: Hansen (1997), Tong (1990)
    # Researcher inference for grid resolution (1-percentile steps),
    # minimum regime size (15% or 15 obs), and AR(1) fallback.
```

#### Phase 2: Online Dynamic Forecasting

The dynamic forecasting phase serves the VWAP execution loop. The key distinction is
between two types of forecasts:

1. **One-step-ahead forecast:** After observing bin j, forecast bin j+1 only. This is what
   BDF Section 4.2.2 describes and what the validated accuracy numbers measure. The trader
   executes a fraction of the order in bin j+1, then observes the actual volume, then
   re-forecasts bin j+2, and so on.

2. **Multi-step weight computation:** To decide what fraction of remaining order to execute
   NOW (in bin j+1) versus later, we need forecasts for ALL remaining bins (j+1 through k).
   For bins j+2 onward, we iterate the AR/SETAR forward using point forecasts as
   pseudo-observations. These multi-step forecasts are approximations with no direct
   benchmark -- their accuracy degrades with horizon. They are replaced by fresh
   one-step-ahead forecasts as new actuals arrive.

```
FUNCTION forecast_next_bin(model_params, stock_id, e_last, j_next, use_setar=True):
    """
    Produce a one-step-ahead forecast for a single stock at a single bin.
    This is the high-accuracy forecast used for VWAP execution.

    model_params: output of estimate_model()
    stock_id:     the stock to forecast
    e_last:       the most recent observed specific component value
                  (or 0.0 at day start)
    j_next:       the bin index to forecast (0-based)
    use_setar:    if True, use SETAR; if False, use AR(1)

    Returns: (x_forecast, e_forecast)
             x_forecast = turnover forecast for bin j_next
             e_forecast = specific component forecast (for use as e_last
                          in subsequent calls if actual is not yet available)
    """
    idx = model_params['stock_ids'].index(stock_id)
    c_forecast = model_params['c_forecast']   # shape (N, k_bins)

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
    x_forecast = c_forecast[idx, j_next] + e_forecast

    # Floor negative forecasts to small positive value
    if x_forecast <= 0:
        x_forecast = 1e-8

    return (x_forecast, e_forecast)
    # Reference: BDF 2008, Section 4.2.2
```

```
FUNCTION run_dynamic_execution(model_params, actual_turnover_feed, use_setar=True):
    """
    Main intraday execution loop for one stock on one trading day.
    Called by the VWAP execution engine.

    model_params:         output of estimate_model()
    actual_turnover_feed: callable that returns the actual turnover for
                          the most recently completed bin (blocks until
                          the bin is complete)
    use_setar:            if True, use SETAR; if False, use AR(1)

    Returns: execution_record (list of per-bin actions taken)
    """
    k_bins = model_params['k_bins']
    stock_ids = model_params['stock_ids']
    c_forecast = model_params['c_forecast']

    execution_record = []

    # Track per-stock state: most recent observed specific component
    e_last = {}  # stock_id -> float
    for stock_id in stock_ids:
        e_last[stock_id] = 0.0
        # Researcher inference: BDF 2008 does not specify day-start
        # initialization. Setting e=0 is consistent with Section 4.2.1
        # (static forecast uses common component only at day start).

    # IMPORTANT: The bin loop is OUTER and the stock loop is INNER.
    # All stocks' bin j weights are computed simultaneously, then all
    # observe bin j actuals, then all forecast bin j+1. This reflects
    # the temporal reality of live execution: bin j closes for all stocks
    # at the same time. If the stock loop were outer, stock 2's bin 0
    # weight would be computed after stock 1's final bin is observed,
    # which is temporally impossible during live execution.
    for j in range(k_bins):
        for stock_id in stock_ids:
            idx = stock_ids.index(stock_id)

            # Compute VWAP weights for remaining bins (j through k-1)
            weights = compute_vwap_weights(
                model_params, stock_id, e_last[stock_id], j, use_setar
            )
            # weights[0] = fraction of remaining order to execute in bin j

            # Execute weights[0] fraction of remaining order in bin j
            # (execution logic is external to this model)
            execution_record.append({
                'stock_id': stock_id,
                'bin': j,
                'weight': weights[0],
            })

            # Observe actual turnover for bin j (blocks until bin completes)
            x_actual = actual_turnover_feed(stock_id, j)

            # Extract observed specific component
            e_last[stock_id] = x_actual - c_forecast[idx, j]

    return execution_record
    # Reference: BDF 2008, Section 4.2.2
    # The trader executes a fraction of remaining order in each bin,
    # then observes the actual, then re-forecasts. The multi-step
    # forecasts (bins j+1 onward) computed inside compute_vwap_weights
    # are approximations that are REPLACED by fresh one-step-ahead
    # forecasts at the next iteration when the actual arrives.
```

#### VWAP Execution Weights

```
FUNCTION compute_vwap_weights(model_params, stock_id, e_last, j_current, use_setar=True):
    """
    Given the current state, compute the fraction of remaining order
    to execute in each future bin. Uses multi-step iterated forecasts
    for bins beyond j_current.

    model_params: output of estimate_model()
    stock_id:     the stock to compute weights for
    e_last:       most recent observed specific component (or 0.0 at start)
    j_current:    current bin index (0-based)
    use_setar:    if True, use SETAR; if False, use AR(1)

    Returns: weights array of shape (k_bins - j_current,)
             weights[0] = fraction to execute in bin j_current
             weights sum to 1.0

    NOTE: These weights use multi-step iterated forecasts for bins
    j_current+1 onward. These forecasts are approximations whose accuracy
    degrades with horizon. In practice, only weights[0] is acted upon;
    the remaining weights are recalculated at the next bin after observing
    the actual. For SETAR, multi-step iteration is a deterministic
    approximation -- the exact conditional expectation would require
    integrating over the error distribution. (Researcher inference.)
    """
    k_bins = model_params['k_bins']
    idx = model_params['stock_ids'].index(stock_id)
    c_forecast = model_params['c_forecast']
    n_remaining = k_bins - j_current

    if n_remaining <= 0:
        return array([])

    forecasts_remaining = zeros(n_remaining)
    e_current = e_last

    for i in range(n_remaining):
        j = j_current + i
        x_forecast, e_forecast = forecast_next_bin(
            model_params, stock_id, e_current, j, use_setar
        )
        forecasts_remaining[i] = x_forecast
        e_current = e_forecast   # feed forward for multi-step

    # Negative forecast safeguard: if more than 50% of remaining bins
    # produced negative raw forecasts (before the 1e-8 floor in
    # forecast_next_bin), fall back to common-component-only forecasts.
    # This prevents pathological VWAP weight concentration when the
    # specific component model produces large systematic negative forecasts.
    # Count how many bins had their forecast floored at 1e-8:
    n_floored = sum(1 for f in forecasts_remaining if f <= 1e-8 + 1e-12)
    if n_floored > n_remaining / 2:
        # Fall back: recompute using common component only (e_forecast = 0)
        for i in range(n_remaining):
            j = j_current + i
            forecasts_remaining[i] = max(c_forecast[idx, j], 1e-8)
        # Researcher inference: the 50% threshold and common-only fallback
        # are a safeguard not described in BDF 2008. The 1e-8 floor in
        # forecast_next_bin handles individual negatives; this handles
        # systematic failure of the specific component model.

    total_remaining = sum(forecasts_remaining)

    if total_remaining <= 0:
        # Fallback: uniform distribution (all forecasts effectively zero)
        return ones(n_remaining) / n_remaining

    weights = forecasts_remaining / total_remaining

    return weights
    # Reference: BDF 2008, Section 4.2.2-4.2.3
    # The weight for bin j is x_hat[j] / sum(x_hat[j:k]).
```

#### U-Method Baseline

The U-method is the standard benchmark against which BDF is compared. It forecasts
turnover for each stock and bin as the simple historical average at that time-of-day
position.

```
FUNCTION forecast_u_method(turnover_data, L_days, k_bins):
    """
    Compute the U-method (historical time-of-day average) forecast.
    This is the naive baseline -- no PCA decomposition, no AR/SETAR.

    turnover_data: dict mapping stock_id -> array of shape (L_days, k_bins)
    L_days:        number of trading days in window
    k_bins:        bins per day

    Returns: u_forecast dict mapping stock_id -> array of shape (k_bins,)
    """
    u_forecast = {}
    for stock_id, data in turnover_data.items():
        # Average turnover at each bin position across all L_days days
        u_forecast[stock_id] = mean(data, axis=0)   # shape (k_bins,)

    return u_forecast
    # Reference: Szucs 2017, Section 4, Eq. (3):
    #   y_hat_{p+1} = (1/L) * sum_{l=1}^{L} y_{p+1 - m*l}
    # where m = number of intraday bins (26), L = number of days (20).
    # Also equivalent to BDF 2008, Eq. (9) with raw turnover X
    # replacing common component C (BDF footnote 5: "Replacing c by x
    # in Eq. (9), we get the classical prediction").
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
  raw_volume_data: dict[stock_id -> ndarray(n_days, k_bins)]
  Each value is int or float, representing raw shares traded per bin

  shares_data: dict[stock_id -> ndarray(n_days)]
  Total shares outstanding (or float-adjusted shares) per day per stock

Preprocessing:
  turnover = raw_volume / shares_outstanding       float64
  Typical turnover values: 1e-5 to 5e-2
  BDF 2008 uses float shares; Szucs 2017 uses TSO. Be consistent.

Phase 1 Pipeline (offline):
  turnover_data                                     dict[str -> (L, k) float64]
    |
    v
  X = flatten and stack columns                     (T, N) float64, T = L*k
  DO NOT mean-center columns.
    |
    v
  If N <= T:
    Sigma = X'X / T                                 (N, N) float64
    eigendecompose(Sigma) -> eigenvalues, V         (N,) float64, (N, N) float64
  Else (N > T):
    Sigma_alt = XX' / T                             (T, T) float64
    eigendecompose(Sigma_alt) -> eigenvalues, U     (T,) float64, (T, T) float64
    |
    v
  select_num_factors -> r                           int, typically 1-3
    |
    v
  F_hat: (T, r) float64, normalized so F_hat'F_hat / T = I_r
  A_hat = F_hat.T @ X / T: (r, N) float64
    |
    v
  C_hat = F_hat @ A_hat                             (T, N) float64
  E_hat = X - C_hat                                 (T, N) float64
    |
    v
  c_forecast = time-of-day mean of C_hat            (N, k) float64
  u_forecast = time-of-day mean of X                (N, k) float64 [baseline]
    |
    v
  Per-stock: fit_ar1(E_hat[:, i]) -> (psi_1, psi_2, sigma)
  Per-stock: fit_setar(E_hat[:, i]) -> (phi_11, phi_12, phi_21, phi_22, tau, sigma)

Phase 2 Pipeline (online, executed per bin):
  At bin j:
    1. Compute VWAP weights for bins [j, j+1, ..., k-1]
       using e_last from previous observation (or 0.0 at day start)
    2. Execute weights[0] fraction of remaining order in bin j
    3. Observe actual turnover x_actual for bin j
    4. e_last = x_actual - c_forecast[stock, j]          float64
    5. Repeat from step 1 for bin j+1

  One-step-ahead forecast (high accuracy):
    e_forecast = AR1(e_last) or SETAR(e_last)            float64
    x_forecast = c_forecast[stock, j+1] + e_forecast     float64
    floor(x_forecast, 1e-8) if negative

  Multi-step forecasts (bins j+2 onward, approximation only):
    Iterate AR/SETAR using e_forecast as pseudo-observation
    Accuracy degrades with horizon; replaced by fresh 1-step at next bin

VWAP weights:
  weights = forecasts_remaining / sum(forecasts_remaining)
                                                     (k - j,) float64, sum = 1.0
  Only weights[0] is acted upon; rest are recalculated next bin.
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
| k_bins | Number of intraday bins per trading day | 26 (15-min, NYSE 9:30-16:00) or 25 (20-min, Euronext 9:20-17:20) | Low | 13-52 (30-min to 5-min) |
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

**Note on k_bins:** BDF 2008 uses k=25 (20-min bins on Euronext Paris, 9:20-17:20). Szucs
2017 uses k=26 (15-min bins on NYSE, 9:30-16:00). The algorithm is parameterized by k and
works with any value. When comparing to published benchmarks, use the k value matching
the source: k=25 for BDF Table 2 numbers, k=26 for Szucs MAPE/MSE numbers. For US
equities, k=26 is the recommended default.
Reference: BDF 2008, Section 3.1; Szucs 2017, Section 2.

### Initialization

**PCA initialization:** No special initialization needed. The eigendecomposition of X'X/T
is a deterministic computation. The first estimation requires at least L_days of historical
turnover data. Do NOT mean-center the columns of X before eigendecomposition. The raw
turnover matrix is used directly. If using a library PCA implementation (e.g., sklearn),
disable automatic mean centering.
Reference: BDF 2008, Section 2.2; Bai (2003).

**AR(1) initialization:** OLS estimation is deterministic given the data. No iterative
procedure or starting values required.

**SETAR initialization:** Grid search over tau is deterministic. Each candidate tau produces
a deterministic OLS fit. If no valid tau is found (all candidates violate minimum regime
size), the SETAR degenerates to AR(1) as a fallback.

**Day-start forecast initialization:** At j_current = 0 (no bins observed), set the
specific component forecast to zero for all stocks. The initial day forecast uses only the
common component. This is consistent with BDF 2008 Section 4.2.1 (static forecast as the
starting point before any intraday observations arrive). (Researcher inference.)

**Factor sign convention:** Eigenvector signs are indeterminate -- they may flip between
daily re-estimations. This does NOT affect the common component C = F @ A because sign
flips cancel in the product. Furthermore, sign flips do NOT affect the specific component
E = X - C (since C is sign-invariant), so the AR/SETAR parameters fitted to E are also
unaffected. The full chain from factors through to forecasts is sign-invariant. No special
handling required. Optionally, enforce a convention (e.g., require the first element of
each loading vector to be positive) for interpretability and debugging visualization only.
(Researcher inference.)

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

**Example fitted parameter values (typical liquid stock):** For a DJIA component (e.g.,
a large-cap stock with consistent intraday trading), fitted parameters on a 20-day
estimation window with k=26 might look like: r = 1-2 factors selected by IC, psi_1 ~
0.3-0.7 (substantial AR(1) persistence in specific component, consistent with BDF 2008
Fig. 2 showing strong PACF at lag 1 for the TOTAL stock), psi_2 ~ 0.001-0.005 (small
intercept near zero since E is centered by construction), SETAR tau ~ 0 (near median of
specific component), Var(C)/Var(X) ~ 0.6-0.8 (common component captures 60-80% of
variance for liquid stocks). If during debugging psi_1 is near 0 or > 0.95 for most
stocks, or Var(C)/Var(X) < 0.3, investigate data preprocessing or PCA configuration.
(Researcher inference -- ranges inferred from BDF 2008 Fig. 2 ACF/PACF plots and general
PCA properties; no paper reports individual fitted parameter values.)

**Re-estimation frequency:** Daily (after market close). Parameters are fixed within
each trading day; only the specific component forecast is updated intraday as new
observations arrive. Reference: BDF 2008, Section 3.1; Szucs 2017, Section 3 ("parameters
of the models are re-estimated daily").

## Validation

### Expected Behavior

#### One-Step-Ahead Forecast Accuracy (Szucs 2017 benchmarks)

These benchmarks validate **one-step-ahead** forecasts, where the actual value of the
previous bin is known before forecasting the next bin. Szucs 2017 Section 3 states:
"the information base for the forecast is updated every 15 minutes... This approach is
often called one-step-ahead forecasting." These numbers are computed on 33 DJIA stocks,
k=26 (15-min bins), L=20 days, over 2648 forecast days (= 2668 total trading days in
sample per Szucs 2017 Table 1, minus 20 initial estimation days).

**Per-stock per-bin MAPE (Szucs 2017, Section 4):**
- BDF-SETAR: ~0.399
- BDF-AR: ~0.403
- U-method benchmark: ~0.503
- Both BDF variants reduce MAPE by ~20% relative to U-method.

**Per-stock per-bin MSE (Szucs 2017, Section 4):**
- BDF-AR: 6.49e-4
- BDF-SETAR: 6.60e-4
- U-method: 1.02e-3
- Note: By MSE, AR(1) slightly outperforms SETAR (lower MSE despite higher MAPE).
  This is because MSE penalizes large errors more, and SETAR occasionally produces
  larger outlier errors despite lower typical errors.

#### Portfolio-Level MAPE (BDF 2008 benchmarks)

These benchmarks are from BDF 2008, Table 2, computed on 39 CAC40 stocks, k=25
(20-min bins), L=20 days, out-of-sample period Sept 2 - Dec 16, 2003.

**Portfolio-level MAPE (BDF 2008, Table 2, panel 1):**
- PCA-SETAR: mean 0.0752, std 0.0869, Q95 0.2010
- PCA-ARMA: mean 0.0829, std 0.0973, Q95 0.2330
- Classical: mean 0.0905, std 0.1050, Q95 0.2490
- Note: Portfolio-level MAPE is ~5x lower than per-stock because averaging turnover
  across stocks diversifies away idiosyncratic noise.

#### VWAP Tracking Error (BDF 2008 benchmarks)

These benchmarks validate the **full dynamic execution strategy**, where one-step-ahead
forecasts are applied sequentially (observe bin j, forecast bin j+1, execute, repeat).
This is NOT multi-step iterated forecasting -- each forecast uses the actual previous
observation. Computed on 39 CAC40 stocks, k=25, L=20.

**VWAP tracking error (BDF 2008, Table 2, panel 3, out-of-sample):**
- Dynamic PCA-SETAR: mean 0.0898
- Dynamic PCA-ARMA: mean 0.0922
- Classical: mean 0.1006
- Dynamic PCA-SETAR reduces tracking error by ~10% vs classical.
- Per-stock reductions up to ~36% for high-volatility stocks (BDF 2008, Tables 5-6;
  e.g., CAP GEMINI: classical 0.2323, dynamic SETAR 0.1491).
- Reference: BDF 2008, Table 2 panel 3, Tables 5-7.

#### Note on Multi-Step Forecast Accuracy

The multi-step iterated forecasts (for bins j+2, j+3, ..., computed inside
`compute_vwap_weights`) have **no direct published benchmark**. Their accuracy degrades
with horizon because forecast errors accumulate. In practice, these forecasts only affect
the weight allocation for the current bin (weights[0]); they are replaced by fresh
one-step-ahead forecasts at the next bin. Do not use Szucs MAPE/MSE or BDF Table 2
tracking error to validate multi-step iterated forecast accuracy.

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
    and the U-method (time-of-day average from `forecast_u_method`). BDF should produce
    lower MAPE for a clear majority of stocks. Reference: Szucs 2017, Section 4
    (33/33 stocks by MAPE).

11. **No-centering verification:** Verify that X is NOT mean-centered before PCA. Run PCA
    both with and without centering on the same data. The uncentered version should produce
    lower MAPE when used in the full BDF pipeline. (Researcher inference.)

### Edge Cases

1. **Zero-volume bins:** If a stock has zero volume in a bin, turnover is zero and MAPE
   is undefined (division by zero). Exclude such bins from MAPE computation. For PCA
   estimation, isolated zero-volume bins are acceptable as valid data points. Stocks with
   frequent zero-volume bins (> 5% of bins) should be excluded entirely from the
   cross-section. Reference: Szucs 2017, Section 2 ("every stock had trades
   and thus a volume record larger than zero in every 15-minute interval").

2. **Half-days and early closures:** Days with fewer than k_bins intervals must be
   excluded from the estimation window entirely. They would create shape mismatches in
   the turnover matrix. Identify half-days by checking whether the exchange closed early
   (e.g., US markets close at 1:00 PM on day before Thanksgiving, Christmas Eve, etc.).
   Reference: BDF 2008, Section 3.1 ("24th and 31st of December 2003 were excluded
   from the sample").

3. **Negative forecasts:** When the specific component forecast is large and negative,
   the combined forecast c + e may be negative. Handled at two levels: (a)
   `forecast_next_bin` floors individual forecasts at 1e-8; (b) `compute_vwap_weights`
   checks if more than 50% of remaining bins were floored and falls back to
   common-component-only forecasts (e_forecast = 0) to prevent pathological
   concentration of VWAP weights. See the pseudocode in both functions for details.
   (Researcher inference.)

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
   SVD instead of eigendecomposition for numerical stability, or use the XX'/T path
   when N > T (see extract_factors). (Researcher inference.)

8. **SETAR fallback triggered:** If the SETAR fallback to AR(1) is triggered (no valid
   tau found), log a warning. This indicates the estimation window is too short or the
   specific component distribution is extreme. Consider increasing L_days or
   min_regime_obs. (Researcher inference.)

9. **Volume participation limits:** In practice, if observed actual volume in a bin is
   much lower than forecast, the trader may not be able to execute the planned fraction
   without exceeding participation rate limits (e.g., no more than 20% of bin volume).
   This practical constraint is external to the model but affects execution. The VWAP
   weight should be interpreted as a target, capped by participation limits, with
   unexecuted volume carried to subsequent bins. (Researcher inference.)

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
   current TSO (or float-adjusted shares) from an external data source (e.g.,
   Bloomberg, CRSP). BDF 2008 uses float shares; Szucs 2017 uses TSO. The choice
   affects absolute turnover levels but not the relative performance of the model.
   Reference: BDF 2008, Section 3.1; Szucs 2017, Section 2.

7. **Overnight boundary contamination:** The concatenated time series used for AR/SETAR
   fitting includes cross-day transitions, which may distort the fitted dynamics.
   See Edge Cases item 6. Reference: researcher inference based on BDF 2008 methodology.

8. **Multi-step forecast degradation:** The multi-step iterated forecasts used for VWAP
   weight computation (bins beyond j+1) degrade with horizon. Their accuracy is not
   validated by any published benchmark. This is mitigated by the dynamic execution
   strategy, which replaces multi-step forecasts with fresh one-step-ahead forecasts
   as actuals arrive. Reference: researcher inference.

## Computational Complexity

For a universe of N stocks, L trading days, k bins per day:
- T = L * k total observations per stock (e.g., 20 * 26 = 520)

**PCA (Phase 1, Steps 1-4):**
- If N <= T: X'X costs O(N^2 * T); eigendecomposition costs O(N^3). Total: O(N^2 * T + N^3).
- If N > T: XX' costs O(T^2 * N); eigendecomposition costs O(T^3). Total: O(T^2 * N + T^3).
- For typical parameters (N=33, T=520): dominated by O(N^2 * T) ~ 566,000 operations.

**AR/SETAR fitting (Phase 1, Step 6):**
- AR(1) per stock: O(T) for OLS on (T-1) observations. Total: O(T * N).
- SETAR per stock: O(T * n_tau) where n_tau ~ 71 grid candidates. Total: O(T * N * n_tau).
- For typical parameters: O(520 * 33 * 71) ~ 1.2 million operations.

**Phase 2 (online, per bin):**
- Per stock per bin: O(k) for VWAP weight computation (iterate AR/SETAR k times).
- Per day: O(k^2 * N) total across all bins and stocks.

**Total per day:** Dominated by PCA for large N, by SETAR grid search for small N.

**Reference benchmark:** Szucs 2017, Section 4 reports the entire BDF pipeline runs in
~2 hours for 33 stocks over 2648 days (i.e., 2648 daily re-estimations). This corresponds
to ~2.7 seconds per day, or ~82 milliseconds per stock per day.

## Paper References

| Spec Section | Paper Source |
|---|---|
| Overview, model description | BDF 2008, Sections 1-2 |
| Data preprocessing, turnover definition | BDF 2008, Section 3.1; Szucs 2017, Section 2 |
| Float shares vs TSO | BDF 2008, Section 3.1 ("traded shares / float shares"); Szucs 2017, Section 2 (V_t / TSO_t) |
| PCA decomposition: X = FA' + e | BDF 2008, Section 2.2, Eqs. (4)-(5) |
| No mean centering required | Bai (2003), cited in BDF 2008, Section 2.2 |
| Eigendecomposition and normalization | BDF 2008, Section 2.2, Eq. (6) and surrounding text |
| Factor count selection (IC_p2) | BDF 2008, Section 2.2; Bai & Ng (2002) |
| IC variant unspecified in BDF | BDF 2008, Section 2.2 (cites Bai & Ng but does not specify IC_p1/p2/p3) |
| Common component forecast | BDF 2008, Section 2.3, Eq. (9) |
| Common forecast = U-method on C_hat | BDF 2008, footnote 5 ("Replacing c by x in Eq. (9), we get the classical prediction") |
| AR(1) specific component model | BDF 2008, Section 2.3, Eq. (10); Szucs 2017, Section 4.1 |
| SETAR specific component model | BDF 2008, Section 2.3, Eq. (11) |
| SETAR notation mapping (spec vs papers) | BDF 2008 Eq. (11); Szucs 2017 Eq. (6) -- see fit_setar comment |
| MLE = OLS equivalence | BDF 2008, Section 2.3 |
| Dynamic VWAP execution | BDF 2008, Section 4.2.2 |
| Static execution excluded | BDF 2008, Section 4.2.1 |
| MAPE/MSE formulas | Szucs 2017, Section 3, Eqs. (1)-(2) |
| U-method benchmark | Szucs 2017, Section 4, Eq. (3) |
| One-step-ahead forecasting | Szucs 2017, Section 3 ("one-step-ahead forecasting") |
| Per-stock MAPE targets (k=26, DJIA) | Szucs 2017, Section 4, results table |
| Portfolio-level MAPE targets (k=25, CAC40) | BDF 2008, Table 2 |
| VWAP tracking error targets (k=25, CAC40) | BDF 2008, Table 2 panel 3, Tables 5-7 |
| Data parameters (26 bins, 20 days) | Szucs 2017, Section 2-3 |
| Data parameters (25 bins, 20 days) | BDF 2008, Section 3.1 |
| Half-day exclusion | BDF 2008, Section 3.1 |
| Split/dividend adjustment | BDF 2008, Section 3.1 |
| SETAR grid search | Hansen (1997); Tong (1990); BDF 2008, Section 2.3 |
| Bai (2003) PCA consistency | BDF 2008, Section 2.2 |
| Computation time comparison | Szucs 2017, Section 4 |
| Day-start initialization (e=0) | Researcher inference; consistent with BDF 2008, Section 4.2.1 |
| Multi-step SETAR deterministic iteration | Researcher inference; standard approximation |
| Multi-step forecast has no benchmark | Researcher inference |
| Negative forecast floor | Researcher inference |
| Factor sign indeterminacy (full chain) | Researcher inference; standard PCA property |
| Cross-day boundary contamination | Researcher inference |
| SVD alternative to eigendecomposition | Researcher inference |
| Large-N path (XX'/T) | Researcher inference |
| SETAR grid resolution and min regime size | Researcher inference; Hansen (1997), Tong (1990) |
| SETAR AR(1) fallback | Researcher inference |
| SETAR DOF approximation | Researcher inference; Hansen (1997) |
| Zero-volume handling | Researcher inference; Szucs 2017, Section 2 |
| Volume participation limits | Researcher inference |
| Computational complexity analysis | Researcher inference; timing from Szucs 2017, Section 4 |
| Data preprocessing steps | Researcher inference; data details from BDF 2008 Section 3.1, Szucs 2017 Section 2 |
| Eigenvalue shortcut for IC residual variance | Researcher inference; standard PCA eigenvalue decomposition property |
| Loop order (bin outer, stock inner) | Researcher inference; temporal consistency for live execution |
| 50% negative fallback in compute_vwap_weights | Researcher inference |
| Example fitted parameter ranges | Researcher inference; BDF 2008 Fig. 2 ACF/PACF plots |
| 2648 forecast days derivation | Szucs 2017, Table 1 (2668 total) minus 20 estimation days |
