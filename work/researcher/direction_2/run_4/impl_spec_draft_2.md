# Implementation Specification: PCA Factor Decomposition (BDF) for Intraday Volume

## Overview

This specification describes the BDF intraday volume forecasting model from
Bialkowski, Darolles, and Le Fol (2008), validated by Szucs (2017). The model
decomposes intraday stock turnover into a market-wide common component (capturing the
U-shaped seasonal pattern) and a stock-specific component (capturing idiosyncratic
deviations). The common component is extracted via principal components analysis (PCA)
on a cross-section of stocks, and the specific component is modeled with either an
AR(1) or SETAR time-series model. The primary application is dynamic VWAP execution:
updating volume forecasts intraday as actual volumes arrive, and computing optimal
trade weights for each remaining time bin.

Inputs: a (P x N) matrix of intraday turnover values for N stocks over P = L * k
time bins (L trading days, k bins per day), plus total shares outstanding (or float
shares) for each stock.

Outputs: for each stock, a forecast of turnover in each remaining intraday bin, and
corresponding VWAP execution weights.

Key assumption: the trader's order size is small relative to market volume (no
significant price impact). The model is purely volume-driven and does not incorporate
price dynamics.

## Revision Notes (Draft 2)

Critique 1 raised 3 major and 7 minor issues. All are addressed in this revision:

- **M1 (TSO vs. float)**: Expanded turnover definition to document the discrepancy
  between BDF 2008 (float) and Szucs 2017 (TSO). Added explicit guidance on which
  to use and impact on benchmark comparability.
- **M2 (ARMA vs. AR justification)**: Strengthened the AR(1) justification with three
  independent arguments including the OLS/MLE consistency check.
- **M3 (Dynamic execution explanation)**: Added a dedicated subsection explaining the
  multi-step forecast normalization role in dynamic execution.
- **m1 (IC_p2 SVD unification)**: Unified factor-count selection to use SVD singular
  values directly, eliminating the P>=N branch.
- **m2 (Variance degrees of freedom)**: Changed to MLE-consistent denominator (T-1)
  with explicit note on the choice.
- **m3 (SETAR notation)**: Rewrote parenthetical notes to use "(regime N intercept)"
  and "(regime N AR coeff)" labels.
- **m4 (MAPE computation)**: Added `compute_mape` and `compute_mse_star` validation
  utility functions in pseudocode.
- **m5 (First bin overnight)**: Added comment at initialization step about overnight
  transition forecast quality.
- **m6 (Stationarity check)**: Implemented as explicit code block with defined U-method
  fallback.
- **m7 (day_index precondition)**: Added assert guard.

## Algorithm

### Model Description

The model has three stages:

1. **Factor extraction**: Apply PCA to the (P x N) turnover matrix X (NOT centered) to
   extract r common factors and per-stock loadings. The common component C = F * A'
   captures the market-wide seasonal pattern. The specific component E = X - C captures
   stock-level deviations.

2. **Forecasting**: The common component forecast for the next trading day is the
   time-of-day average of the common component over the L-day estimation window. The
   specific component forecast uses an AR(1) or SETAR model fitted to the concatenated
   specific component series.

3. **Dynamic execution**: At the start of the trading day, produce a full-day forecast.
   After each observed bin, update the specific component forecast using actual observed
   turnover, and recompute VWAP weights for remaining bins. At each step, only the
   first weight (for the immediate next bin) is executed. However, the full multi-step
   forecast for all remaining bins is still essential: it serves as the normalizing
   denominator that determines the magnitude of the first weight. Poor multi-step
   forecasts distort this normalization and hence the executed weight. See the "Dynamic
   Execution Weight Normalization" subsection below for details.

### Pseudocode

The implementation consists of 13 functions organized in five groups: data
preparation, PCA decomposition, time-series modeling, dynamic execution, and
validation utilities.

#### Group 1: Data Preparation

```
function prepare_turnover_matrix(volume_data, shares_data, k, L_days,
                                 shares_measure="TSO"):
    """
    Convert raw volume data into a turnover matrix suitable for PCA.

    Parameters:
        volume_data:     array of shape (n_total_bins, N) — raw share volumes
                         for N stocks, ordered chronologically, k bins per day
        shares_data:     array of shape (N,) — shares outstanding per stock
                         (or per-day if time-varying, shape (n_days, N))
                         This is TSO or float depending on shares_measure.
        k:               int — number of intraday bins per trading day
        L_days:          int — number of days in the estimation window
        shares_measure:  string — "TSO" or "float", specifying the denominator

    Returns:
        X: array of shape (P, N) where P = L_days * k
           Turnover values: volume / shares. NOT centered, NOT standardized.
    """
    # Turnover definition — PAPERS DISAGREE:
    #
    # BDF 2008, Section 2.1, p. 1711: "the turnover for stock i at date t,
    # i.e. the number of traded shares S_i divided by the number of floated
    # shares X_i." BDF uses FLOAT shares as the denominator.
    #
    # Szucs 2017, Section 2, p. 4: "x_t = V_t / TSO_t, where x stands for
    # turnover, V for volume, and TSO for the total shares outstanding."
    # Szucs uses TSO as the denominator. TSO data downloaded from Bloomberg.
    #
    # IMPLEMENTATION CHOICE: Default to TSO for data availability (TSO is
    # readily available from standard data vendors; float requires additional
    # data on insider/institutional holdings). Szucs 2017 achieves strong
    # results (MAPE = 0.399) using TSO, validating this choice.
    #
    # IMPORTANT: Using TSO instead of float will produce different turnover
    # magnitudes, especially for stocks with large insider/institutional
    # holdings (where float << TSO). As a result:
    #   - The SETAR threshold tau will differ from BDF 2008's reported values
    #   - The specific component magnitudes will differ
    #   - Benchmark numbers from BDF 2008 Table 2 (which used float) are NOT
    #     directly comparable to this implementation's output
    #   - Benchmark numbers from Szucs 2017 Tables 2a-c (which used TSO) ARE
    #     directly comparable
    # The relative ordering of model performance (SETAR > AR > U-method)
    # is robust to this choice.

    P = L_days * k
    X = zeros(P, N)

    for each stock i in 0..N-1:
        for each bin t in 0..P-1:
            X[t, i] = volume_data[t, i] / shares_data[i]

    # Critical: do NOT subtract column means. In the Bai (2003) large-
    # dimensional factor framework, the intercept is absorbed into the
    # factor loadings. BDF 2008 Eq. (6) does not include centering.
    # Centering would strip level information and produce a biased
    # decomposition of the U-shaped seasonal pattern.
    # Reference: BDF 2008, Section 2.2; Bai (2003)

    return X
```

#### Group 2: PCA Decomposition

```
function select_factor_count(X, r_max=10):
    """
    Select the number of factors r using the Bai & Ng (2002) IC_p2 criterion.

    Parameters:
        X:     array of shape (P, N) — turnover matrix (NOT centered)
        r_max: int — maximum number of factors to evaluate

    Returns:
        r_star: int — optimal number of factors (typically 1-3)
    """
    # Reference: BDF 2008, Section 2.2; Bai & Ng (2002)

    P, N = X.shape

    # Use SVD to compute singular values — avoids branching on P vs N
    # and is numerically more stable than eigendecomposition of X'X or XX'.
    # Only the singular values are needed here, not the full decomposition.
    # (Revision note: unified per critique m1 to avoid P>=N branch)
    s = singular_values(X, n_components=r_max)  # top r_max singular values, descending
    total_ss = sum(X ** 2)  # total sum of squares

    ic_values = []
    for r in 1..r_max:
        # Residual variance from rank-r approximation.
        # The sum of squared singular values for the top r components equals
        # the variance explained: sum(s[0:r]^2) = sum of top r eigenvalues
        # of X'X (or XX'). So V(r) = (total_ss - sum(s[0:r]^2)) / (N*P).
        V_r = (1 / (N * P)) * (total_ss - sum(s[0:r] ** 2))

        # Floating-point guard: when top r components capture nearly all
        # variance, V_r can become <= 0
        if V_r <= 0:
            V_r = 1e-15  # Researcher inference: defensive guard

        # IC_p2 penalty
        # Reference: Bai & Ng (2002), Eq. IC_p2
        penalty = r * ((N + P) / (N * P)) * ln(min(N, P))

        ic_values.append(ln(V_r) + penalty)

    r_star = argmin(ic_values) + 1  # +1 because r starts at 1
    return r_star


function extract_factors(X, r):
    """
    Extract r common factors and loadings from X using truncated SVD.

    Parameters:
        X: array of shape (P, N) — turnover matrix (NOT centered)
        r: int — number of factors

    Returns:
        F_hat:      array of shape (P, r) — estimated factors
                    Normalized so F_hat.T @ F_hat / P = I_r
        Lambda_hat: array of shape (N, r) — estimated factor loadings
        C_hat:      array of shape (P, N) — common component = F_hat @ Lambda_hat.T
        E_hat:      array of shape (P, N) — specific component = X - C_hat
    """
    # Reference: BDF 2008, Section 2.2, Eq. (6); Bai (2003)
    #
    # BDF 2008 Eq. (6): minimize sum_{i,t} (X_{it} - lambda_i' F_t)^2
    # subject to F'F/T = I_r  (using BDF's "T" = our "P")
    #
    # Solution: F_hat columns are sqrt(P) times the first r left singular
    # vectors of X. This is equivalent to the eigenvector approach in
    # BDF but avoids branching on P vs N dimensions.

    P, N = X.shape

    # Truncated SVD: X = U @ diag(s) @ Vt
    # Use scipy.sparse.linalg.svds or sklearn.utils.extmath.randomized_svd
    U, s, Vt = truncated_svd(X, n_components=r)
    # U: (P, r), s: (r,), Vt: (r, N)
    # Convention: singular values in descending order

    # Factor matrix with normalization F'F/P = I_r
    F_hat = sqrt(P) * U          # shape (P, r)

    # Loading matrix
    Lambda_hat = (Vt.T * s) / sqrt(P)   # shape (N, r)
    # Equivalently: Lambda_hat = X.T @ F_hat / P

    # Verification: F_hat.T @ F_hat / P should equal I_r (up to floating point)
    # This holds because U.T @ U = I_r, so (sqrt(P)*U).T @ (sqrt(P)*U) / P = I_r

    # Common and specific components
    C_hat = F_hat @ Lambda_hat.T   # shape (P, N)
    E_hat = X - C_hat              # shape (P, N)

    return F_hat, Lambda_hat, C_hat, E_hat
```

**Note on sign/rotation indeterminacy**: The sign and rotation of individual factors
and loadings are not uniquely determined by PCA. However, the common component
C_hat = F_hat @ Lambda_hat.T is unique regardless of sign flips or rotations, because
any orthogonal transformation cancels through the product. This means E_hat is also
unique, and therefore the AR(1)/SETAR parameters fitted to E_hat are unaffected.
No special handling is required.
(Reference: BDF 2008; Bai 2003. Individual factors are not interpreted.)

#### Group 3: Time-Series Modeling of Specific Component

```
function forecast_common(C_hat, k, L_days):
    """
    Compute the common component forecast for the next trading day by
    averaging each time-of-day bin across L days.

    Parameters:
        C_hat:  array of shape (P, N) where P = L_days * k
        k:      int — bins per day
        L_days: int — number of days in estimation window

    Returns:
        c_forecast: array of shape (k, N) — forecast common component
                    for each bin of the next trading day
    """
    # Reference: BDF 2008, Section 2.3, Eq. (9)
    # c_hat_{i,t+1} = (1/L) * sum_{l=1}^{L} c_{i,t+1-k*l}
    # This averages the common component at the same time-of-day across L days.

    # Reshape C_hat from (P, N) to (L_days, k, N)
    C_3d = reshape(C_hat, (L_days, k, N))

    # Average across days (axis 0)
    c_forecast = mean(C_3d, axis=0)   # shape (k, N)

    # This forecast is computed BEFORE the trading day begins.
    # It is fixed for the entire day and NOT updated intraday.
    # Only the specific component updates intraday.
    # Reference: BDF 2008, Section 2.3, paragraph below Eq. (9)

    return c_forecast


function fit_ar1(e_series):
    """
    Fit an AR(1) model with intercept to a specific component series.

    Model: e_t = psi_1 * e_{t-1} + psi_2 + epsilon_t

    Parameters:
        e_series: array of shape (T,) — specific component for one stock,
                  concatenated across L days (T = L_days * k)

    Returns:
        psi_1:     float — AR(1) coefficient
        psi_2:     float — intercept
        sigma2:    float — innovation variance estimate (MLE)
    """
    # Reference: BDF 2008, Section 2.3, Eq. (10)
    #
    # AR(1) JUSTIFICATION — BDF 2008 labels Eq. (10) as "ARMA(1,1)" in the
    # text (p. 1712: "The first one is an ARMA(1,1) with white noise"), but
    # the equation itself contains no MA term:
    #   e_{i,t} = psi_1 * e_{i,t-1} + psi_2 + epsilon_{i,t}
    #
    # Three independent lines of evidence confirm AR(1) is correct:
    #
    # (a) The equation as written in BDF 2008 Eq. (10) has no MA term.
    #     The equation is authoritative over the text label.
    #
    # (b) Szucs 2017, Eq. (5) independently implements the same model and
    #     correctly labels it AR(1): e_p = c + theta_1 * e_{p-1} + eps_p.
    #     Szucs reproduces strong results (MAPE = 0.399) with this AR(1).
    #
    # (c) BDF 2008 states estimation is by "maximum likelihood." For a pure
    #     AR(1) with Gaussian errors, conditional MLE is equivalent to OLS.
    #     If an MA(1) term were present, this equivalence would NOT hold —
    #     MA models require iterative optimization (e.g., innovations
    #     algorithm or exact MLE). The fact that BDF uses the OLS/MLE
    #     equivalence is consistent only with AR(1), not ARMA(1,1).
    #
    # Therefore we implement AR(1). The "ARMA(1,1)" label in BDF's text
    # is a misnomer.

    T = length(e_series)

    # Dependent variable: e_series[1:T]
    y = e_series[1:]         # shape (T-1,)
    # Regressors: [e_series[0:T-1], ones]
    X_reg = column_stack([e_series[:-1], ones(T-1)])  # shape (T-1, 2)

    # OLS: beta = (X'X)^{-1} X'y
    beta = solve(X_reg.T @ X_reg, X_reg.T @ y)   # shape (2,)
    psi_1 = beta[0]
    psi_2 = beta[1]

    # Innovation variance — MLE estimator (biased)
    # Using T-1 (number of observations) as denominator for consistency
    # with maximum likelihood estimation as specified by BDF 2008.
    # For T-1 = L*k - 1 = 519, the difference from the unbiased estimator
    # (denominator T-1-2 = 517) is negligible (~0.4%).
    # Reference: BDF 2008, Section 2.3 ("estimate by maximum likelihood")
    residuals = y - X_reg @ beta
    sigma2 = sum(residuals ** 2) / (T - 1)

    return psi_1, psi_2, sigma2


function fit_setar(e_series, n_grid=100, tau_lo_pct=0.15, tau_hi_pct=0.85,
                   min_regime_frac=0.10, min_regime_obs=20):
    """
    Fit a two-regime SETAR model to a specific component series.

    Model: e_t = (phi_11 * e_{t-1} + phi_12) * I(e_{t-1} <= tau)
               + (phi_21 * e_{t-1} + phi_22) * (1 - I(e_{t-1} <= tau))
               + epsilon_t

    Parameters:
        e_series:          array of shape (T,) — specific component for one stock
        n_grid:            int — number of threshold candidates
        tau_lo_pct:        float — lower quantile bound for tau search
        tau_hi_pct:        float — upper quantile bound for tau search
        min_regime_frac:   float — minimum fraction of obs in each regime
        min_regime_obs:    int — minimum absolute count in each regime

    Returns:
        phi_11, phi_12:    floats — regime 1 (below threshold) AR coeff and intercept
        phi_21, phi_22:    floats — regime 2 (above threshold) AR coeff and intercept
        tau:               float — estimated threshold
        sigma2_1:          float — innovation variance, regime 1 (MLE)
        sigma2_2:          float — innovation variance, regime 2 (MLE)
        success:           bool — whether estimation converged
    """
    # Reference: BDF 2008, Section 2.3, Eq. (11); Hansen (1997)
    # Estimation: profile MLE via grid search over tau. For each candidate
    # tau, fit separate OLS regressions in each regime, compute SSR.
    # Select tau that minimizes total SSR.

    T = length(e_series)
    y = e_series[1:]              # shape (T-1,)
    e_lag = e_series[:-1]         # shape (T-1,)
    n = T - 1

    min_regime = max(min_regime_frac * n, min_regime_obs)

    # Grid of tau candidates
    tau_lo = quantile(e_lag, tau_lo_pct)
    tau_hi = quantile(e_lag, tau_hi_pct)
    tau_candidates = linspace(tau_lo, tau_hi, n_grid)

    # Phase 1: Grid search — minimize total SSR only
    best_ssr = +inf
    best_tau = None

    for tau_c in tau_candidates:
        # Indicator: regime 1 if e_lag <= tau_c
        mask1 = (e_lag <= tau_c)
        mask2 = ~mask1
        n1 = sum(mask1)
        n2 = sum(mask2)

        # Skip if either regime has too few observations
        if n1 < min_regime or n2 < min_regime:
            continue

        # Regime 1 OLS: y[mask1] ~ e_lag[mask1], 1
        X1 = column_stack([e_lag[mask1], ones(n1)])
        beta1 = solve(X1.T @ X1, X1.T @ y[mask1])
        ssr1 = sum((y[mask1] - X1 @ beta1) ** 2)

        # Regime 2 OLS: y[mask2] ~ e_lag[mask2], 1
        X2 = column_stack([e_lag[mask2], ones(n2)])
        beta2 = solve(X2.T @ X2, X2.T @ y[mask2])
        ssr2 = sum((y[mask2] - X2 @ beta2) ** 2)

        total_ssr = ssr1 + ssr2
        if total_ssr < best_ssr:
            best_ssr = total_ssr
            best_tau = tau_c

    if best_tau is None:
        # No valid tau found (all candidates violate min regime size)
        return None, None, None, None, None, None, None, False

    # Phase 2: Final fit at best_tau — compute coefficients and variances
    tau = best_tau
    mask1 = (e_lag <= tau)
    mask2 = ~mask1
    n1 = sum(mask1)
    n2 = sum(mask2)

    X1 = column_stack([e_lag[mask1], ones(n1)])
    beta1 = solve(X1.T @ X1, X1.T @ y[mask1])
    phi_11 = beta1[0]
    phi_12 = beta1[1]
    resid1 = y[mask1] - X1 @ beta1
    sigma2_1 = sum(resid1 ** 2) / n1   # MLE estimator (biased), consistent with fit_ar1

    X2 = column_stack([e_lag[mask2], ones(n2)])
    beta2 = solve(X2.T @ X2, X2.T @ y[mask2])
    phi_21 = beta2[0]
    phi_22 = beta2[1]
    resid2 = y[mask2] - X2 @ beta2
    sigma2_2 = sum(resid2 ** 2) / n2   # MLE estimator (biased), consistent with fit_ar1

    return phi_11, phi_12, phi_21, phi_22, tau, sigma2_1, sigma2_2, True
```

**Notation mapping across papers:**

| This spec | BDF 2008 Eq. (10)/(11) | Szucs 2017 Eq. (5)/(6) |
|-----------|------------------------|------------------------|
| psi_1     | psi_1                  | theta_1                |
| psi_2     | psi_2                  | c                      |
| phi_11    | phi_11                 | theta_{1,2} (regime 1 AR coeff)  |
| phi_12    | phi_12                 | c_{1,1} (regime 1 intercept)     |
| phi_21    | phi_21                 | theta_{2,2} (regime 2 AR coeff)  |
| phi_22    | phi_22                 | c_{2,1} (regime 2 intercept)     |
| tau       | tau                    | tau                    |

**Warning**: Szucs 2017, Eq. (6) places the intercept FIRST: (c_{j,1} + theta_{j,2} * e_{p-1}),
opposite to BDF's ordering (phi * e + intercept). The indicator function I(z) is
identical: I(z) = 1 if z <= tau, 0 otherwise.
(Reference: BDF 2008, Eq. (11); Szucs 2017, Eq. (6)-(7))

#### Group 4: Dynamic Execution

```
function forecast_specific(e_last, model_type, params, n_steps):
    """
    Produce n_steps of specific component forecasts by iterating the
    AR(1) or SETAR recursion forward.

    Parameters:
        e_last:     float — last observed (or predicted) specific component value
        model_type: string — "ar1" or "setar"
        params:     dict — model parameters
                    For ar1: {psi_1, psi_2}
                    For setar: {phi_11, phi_12, phi_21, phi_22, tau}
        n_steps:    int — number of steps to forecast

    Returns:
        e_forecast: array of shape (n_steps,) — specific component forecasts
    """
    # Reference: BDF 2008, Section 4.2.2
    # Multi-step forecasts use deterministic iteration: each forecast is
    # used as the input to the next step's threshold comparison.
    # This is the standard approximation; exact conditional expectation
    # would require integrating over the error distribution.
    # (Researcher inference for the approximation choice; BDF does not
    # specify the multi-step method explicitly.)

    e_forecast = zeros(n_steps)
    e_prev = e_last

    for step in 0..n_steps-1:
        if model_type == "ar1":
            e_next = params.psi_1 * e_prev + params.psi_2
            # For stationary AR(1), this decays geometrically toward
            # psi_2 / (1 - psi_1), the unconditional mean.
        elif model_type == "setar":
            if e_prev <= params.tau:
                e_next = params.phi_11 * e_prev + params.phi_12
            else:
                e_next = params.phi_21 * e_prev + params.phi_22

        e_forecast[step] = e_next
        e_prev = e_next

    return e_forecast


function compute_vwap_weights(forecasts_remaining):
    """
    Convert turnover forecasts for remaining bins into VWAP execution weights.

    Parameters:
        forecasts_remaining: array of shape (n_remaining,) — forecast turnover
                             for each remaining bin (for one stock)

    Returns:
        weights: array of shape (n_remaining,) — VWAP weights, non-negative,
                 sum to 1.0
    """
    # Reference: BDF 2008, Section 4.2.2
    # w[j] = x_hat[j] / sum(x_hat[j:end])
    # Since we normalize, this is equivalent to:
    # weights = forecasts_remaining / sum(forecasts_remaining)

    # Clip negative forecasts to zero
    clipped = maximum(forecasts_remaining, 0.0)

    total = sum(clipped)

    if total <= 0:
        # All forecasts non-positive: fall back to equal weighting
        # Researcher inference: this indicates model failure for this
        # stock on this day
        n = length(clipped)
        return ones(n) / n

    weights = clipped / total
    return weights


function run_dynamic_execution(X_estimation, shares_data, k, L_days,
                               x_actual_today, r_max=10, model_type="setar"):
    """
    Run the full dynamic VWAP execution for one trading day.

    This is the main entry point. It performs:
    1. PCA decomposition on the estimation window
    2. Model fitting (AR1/SETAR) per stock
    3. Common component forecasting
    4. Bin-by-bin dynamic updating and weight computation

    Parameters:
        X_estimation:   array of shape (P, N) — turnover matrix for estimation
                        window, P = L_days * k, already computed
        shares_data:    array of shape (N,) — shares outstanding (TSO or float)
        k:              int — bins per day
        L_days:         int — days in estimation window
        x_actual_today: array of shape (k, N) — actual turnover for today
                        (used for evaluation; in live trading, arrives bin by bin)
        r_max:          int — max factors for IC
        model_type:     string — "ar1" or "setar"

    Returns:
        weights_history: list of length k, each entry is array of shape
                         (k - j, N) — weights for remaining bins at step j
        forecasts_history: list of forecasts at each step
    """
    P, N = X_estimation.shape

    # Step 1: Select factor count and extract factors
    r = select_factor_count(X_estimation, r_max)
    F_hat, Lambda_hat, C_hat, E_hat = extract_factors(X_estimation, r)

    # Step 2: Common component forecast (fixed for entire day)
    c_forecast = forecast_common(C_hat, k, L_days)   # shape (k, N)

    # Step 3: Fit time-series models per stock
    models = []
    use_u_method = zeros(N, dtype=bool)   # fallback flags

    for i in 0..N-1:
        e_i = E_hat[:, i]   # shape (P,)

        if model_type == "setar":
            result = fit_setar(e_i)
            if result.success:
                models.append({"type": "setar", "params": result})
            else:
                # Fallback to AR(1) if SETAR fails
                ar_result = fit_ar1(e_i)
                models.append({"type": "ar1", "params": ar_result})
        else:
            ar_result = fit_ar1(e_i)
            models.append({"type": "ar1", "params": ar_result})

        # Stationarity check: if |AR coeff| >= 1, the specific component is
        # non-stationary and multi-step forecasts will diverge. Fall back to
        # U-method for this stock on this day.
        # Researcher inference: BDF 2008 does not specify this guard, but it
        # is necessary for robustness.
        if models[i].type == "ar1":
            if abs(models[i].params.psi_1) >= 1.0:
                use_u_method[i] = True
        elif models[i].type == "setar":
            if abs(models[i].params.phi_11) >= 1.0 or abs(models[i].params.phi_21) >= 1.0:
                use_u_method[i] = True

    # Step 4: Compute U-method baseline for fallback stocks
    # U-method: simple time-of-day average of raw turnover (not PCA-decomposed)
    X_3d = reshape(X_estimation, (L_days, k, N))
    u_forecast = mean(X_3d, axis=0)   # shape (k, N)
    # Reference: Szucs 2017, Section 4, Eq. (3); BDF 2008, footnote 5

    # Step 5: Initialize specific component state
    # Use the last in-sample residual from the final bin of the final day
    # in the estimation window. This is the correct initial condition for
    # the AR/SETAR recursion: e_t = f(e_{t-1}) requires an actual e_{t-1}.
    # All estimation data precedes the forecast day, so no look-ahead bias.
    # Reference: BDF 2008, Section 2.3
    #
    # NOTE ON OVERNIGHT TRANSITION: This initialization uses E_hat[-1, :],
    # which is the last bin of the last day in the estimation window (e.g.,
    # 15:45-16:00). The first forecast is for the first bin of the next
    # trading day (e.g., 9:30-9:45). This overnight gap introduces dynamics
    # not captured by the AR/SETAR model, which was fitted to both intraday
    # and overnight transitions indiscriminately. BDF 2008 Fig. 2 shows
    # periodic ACF spikes at multiples of k, indicating overnight transitions
    # are structurally different from intraday transitions. The first bin's
    # forecast may therefore be less accurate than subsequent intraday bins.
    # BDF 2008 does not treat overnight transitions differently.
    last_specific = E_hat[-1, :]   # shape (N,) — last bin of last day

    # Step 6: Dynamic bin-by-bin execution
    weights_history = []
    e_state = last_specific.copy()   # current specific component state per stock

    for j in 0..k-1:
        # Forecast remaining bins (j through k-1)
        k_remaining = k - j

        all_weights = zeros(k_remaining, N)

        for i in 0..N-1:
            if use_u_method[i]:
                # Non-stationary stock: use U-method forecast
                x_forecasts = u_forecast[j:k, i]   # shape (k_remaining,)
            else:
                # Forecast specific component for remaining bins
                e_forecasts = forecast_specific(
                    e_state[i], models[i].type, models[i].params, k_remaining
                )

                # Total turnover forecast = common + specific
                x_forecasts = c_forecast[j:k, i] + e_forecasts   # shape (k_remaining,)

            # Compute VWAP weights
            all_weights[:, i] = compute_vwap_weights(x_forecasts)

        weights_history.append(all_weights)

        # Execute: trade weight[0] fraction of remaining order for each stock
        # (Implementation detail: actual order execution is outside this model)

        # Observe actual turnover for bin j and update specific component state
        if j < k - 1:
            for i in 0..N-1:
                # Extract observed specific component
                e_observed = x_actual_today[j, i] - c_forecast[j, i]
                e_state[i] = e_observed
                # Next iteration will use this as e_{t-1} for one-step-ahead

    return weights_history


function daily_rolling_update(X_full, k, day_index, L_days):
    """
    Extract the estimation window for a given day index.

    Parameters:
        X_full:    array of shape (total_bins, N) — full turnover history
        k:         int — bins per day
        day_index: int — 0-based index of the forecast day (must be >= L_days)
        L_days:    int — estimation window length

    Returns:
        X_estimation: array of shape (L_days * k, N) — estimation window
    """
    # Reference: BDF 2008, Section 3.1; Szucs 2017, Section 3
    # At the end of each trading day, slide the window forward by one day.
    # Re-run PCA and model fitting for the next day's forecasts.

    # Precondition: must have enough history for the estimation window
    assert day_index >= L_days, (
        f"day_index ({day_index}) must be >= L_days ({L_days}). "
        f"The first valid forecast day is L_days."
    )

    start_bin = (day_index - L_days) * k
    end_bin = day_index * k
    X_estimation = X_full[start_bin:end_bin, :]   # shape (L_days * k, N)
    return X_estimation


function u_method_benchmark(X_estimation, k, L_days):
    """
    Compute the U-method (naive historical average) forecast as a baseline.

    The U-method averages raw turnover at each time-of-day bin across L days.
    This is equivalent to the common component forecast applied to X instead
    of C_hat. The BDF model's improvement over U-method comes entirely from
    the specific component AR/SETAR forecast.

    Parameters:
        X_estimation: array of shape (P, N) — turnover matrix
        k:            int — bins per day
        L_days:       int — days in estimation window

    Returns:
        u_forecast: array of shape (k, N)
    """
    # Reference: Szucs 2017, Section 4, Eq. (3); BDF 2008, footnote 5
    X_3d = reshape(X_estimation, (L_days, k, N))
    u_forecast = mean(X_3d, axis=0)   # shape (k, N)
    return u_forecast
```

#### Dynamic Execution Weight Normalization

This subsection explains a key subtlety of the dynamic execution algorithm that is
critical for correct implementation.
(Reference: BDF 2008, Section 4.2.2)

At each step j of the dynamic loop:

1. The model forecasts k - j bins (all remaining bins from j to k-1).
2. These forecasts are normalized into VWAP weights via `compute_vwap_weights`.
3. **Only the first weight** `weights[0]` is acted upon — it determines what fraction
   of the remaining order to execute in the immediately next bin.
4. The actual turnover for bin j is then observed, the specific component state is
   updated, and the process repeats at step j+1 with a fresh forecast.

This means the multi-step forecasts for bins j+1, j+2, ..., k-1 are never directly
executed. However, they are **not wasted**. They serve as the normalizing denominator:

```
weight[0] = forecast[0] / sum(forecast[0:k_remaining])
```

If the multi-step forecasts are inaccurate (e.g., they sum to a value much larger or
smaller than the true remaining volume), the denominator is wrong and the first
weight — the only one that matters — will be distorted. Concretely:

- If multi-step forecasts are too high, `sum(forecast)` is inflated, so `weight[0]`
  is too small. The algorithm under-trades early and must catch up later.
- If multi-step forecasts are too low, `weight[0]` is too large. The algorithm
  over-trades early and has insufficient order remaining for later bins.

This is why multi-step forecast quality matters even in dynamic mode, and why SETAR
(with better multi-step forecasts) outperforms AR(1) for dynamic execution despite
the state being updated at every step.
(Reference: BDF 2008, Section 4.2.2 — "we can only update our strategy for the rest
of the trading day" and "the proportion is only applied on the remaining volume to
trade after interval t.")

#### Group 5: Validation Utilities

```
function compute_mape(actual, forecast):
    """
    Compute Mean Absolute Percentage Error per stock and across stocks.

    This matches Szucs 2017, Eq. (2): MAPE = (1/N) * sum |Y_t - Y_t^f| / Y_t
    where N is the number of observations, Y_t is actual, Y_t^f is forecast.

    Parameters:
        actual:   array of shape (n_obs,) or (n_obs, N_stocks) — actual turnover
        forecast: array of shape (n_obs,) or (n_obs, N_stocks) — forecast turnover

    Returns:
        mape_per_stock: float or array of shape (N_stocks,) — per-stock MAPE
        mape_overall:   float — average MAPE across stocks
    """
    # Reference: Szucs 2017, Section 3, Eq. (2)
    #
    # For per-stock per-bin MAPE (Szucs approach):
    #   For each stock i, compute MAPE over all bins across all forecast days:
    #   MAPE_i = (1/T) * sum_{t=1}^{T} |actual[t,i] - forecast[t,i]| / actual[t,i]
    #   Then overall MAPE = (1/N_stocks) * sum_i MAPE_i
    #
    # Exclude bins where actual == 0 (division by zero).
    # Reference: Edge Case 1

    if actual.ndim == 1:
        mask = (actual > 0)
        mape = mean(abs(actual[mask] - forecast[mask]) / actual[mask])
        return mape, mape

    N_stocks = actual.shape[1]
    mape_per_stock = zeros(N_stocks)

    for i in 0..N_stocks-1:
        mask = (actual[:, i] > 0)
        if sum(mask) > 0:
            ape = abs(actual[mask, i] - forecast[mask, i]) / actual[mask, i]
            mape_per_stock[i] = mean(ape)

    mape_overall = mean(mape_per_stock)
    return mape_per_stock, mape_overall


function compute_mse_star(actual, forecast, avg_turnover):
    """
    Compute the scale-adjusted MSE* metric from Szucs 2017 Eq. (14).

    MSE* normalizes each stock's MSE by its average turnover level,
    preventing high-turnover stocks from dominating the aggregate.

    Parameters:
        actual:       array of shape (n_obs, N_stocks) — actual turnover
        forecast:     array of shape (n_obs, N_stocks) — forecast turnover
        avg_turnover: array of shape (N_stocks,) — average turnover per stock
                      (a_i in Szucs notation)

    Returns:
        mse_star: float — scale-adjusted MSE*
    """
    # Reference: Szucs 2017, Section 5, Eq. (14)
    # MSE* = (1/N) * sum_{i=1}^{N} MSE_i / (a_i / a_min)^2
    # where MSE_i is the average MSE of stock i,
    # a_i is the average turnover of stock i,
    # and a_min is the smallest a_i across all stocks.

    N_stocks = actual.shape[1]
    a_min = min(avg_turnover)

    mse_star = 0.0
    for i in 0..N_stocks-1:
        mse_i = mean((actual[:, i] - forecast[:, i]) ** 2)
        scale = (avg_turnover[i] / a_min) ** 2
        mse_star += mse_i / scale

    mse_star /= N_stocks
    return mse_star
```

### Data Flow

```
Input: volume_data (n_bins, N), shares_data (N,), shares_measure ("TSO" or "float")
  |
  v
prepare_turnover_matrix  -->  X: float64 (P, N), P = L*k
  |                           NOT centered
  v
select_factor_count(X)  -->  r: int (typically 1-3)
  |                          [via SVD singular values + IC_p2 criterion]
  |
  v
extract_factors(X, r)   -->  F_hat: float64 (P, r)    [F'F/P = I_r]
                             Lambda_hat: float64 (N, r)
                             C_hat: float64 (P, N)     [common component]
                             E_hat: float64 (P, N)     [specific component]
  |
  +--- forecast_common(C_hat) --> c_forecast: float64 (k, N)
  |                                [fixed for entire day, time-of-day avg]
  |
  +--- fit_ar1(E_hat[:,i]) or fit_setar(E_hat[:,i])  (per stock i)
  |    --> AR1: (psi_1, psi_2, sigma2) all float64
  |    --> SETAR: (phi_11, phi_12, phi_21, phi_22, tau, sigma2_1, sigma2_2)
  |    --> stationarity check: |AR coeff| >= 1 => use_u_method[i] = True
  |
  +--- u_method fallback: u_forecast: float64 (k, N)
  |
  +--- last_specific = E_hat[-1, :]  --> float64 (N,)
  |    [overnight transition: last bin of estimation to first bin of forecast]
  |
  v
Dynamic loop (j = 0 to k-1):
  |
  +--- For non-U-method stocks:
  |    forecast_specific(e_state[i], model, n_steps=k-j)
  |    --> e_forecast: float64 (k-j,)
  |    x_forecast = c_forecast[j:k, i] + e_forecast
  |
  +--- For U-method fallback stocks:
  |    x_forecast = u_forecast[j:k, i]
  |
  +--- compute_vwap_weights(x_forecast)
  |    --> weights: float64 (k-j,), non-negative, sum to 1.0
  |
  +--- EXECUTE: only weights[0] is acted upon at this step.
  |    The remaining weights[1:] serve as the normalizing denominator
  |    that determines the magnitude of weights[0].
  |    (See "Dynamic Execution Weight Normalization" above)
  |
  +--- Observe actual: e_state[i] = x_actual[j,i] - c_forecast[j,i]
  |
  v
Output: weights_history — list of k arrays, weights_history[j] has shape (k-j, N)

Validation: compute_mape(actual, forecast) and compute_mse_star(actual, forecast, avg)
```

### Variants

This specification implements the **BDF-SETAR dynamic execution** variant, which is
the most complete and best-performing model described across both papers:

- **SETAR over AR(1)**: SETAR outperforms AR(1) on 36 of 39 CAC40 stocks (BDF 2008,
  Section 3.2) and 30 of 33 DJIA stocks by MAPE (Szucs 2017, Table 2c). AR(1) is
  retained only as a fallback when SETAR estimation fails.

- **Dynamic over static execution**: Static execution is explicitly shown to be worse
  than the classical (U-method) approach because multi-step AR/SETAR forecasts decay
  toward the unconditional mean, neutralizing the specific component's contribution
  (BDF 2008, Section 4.2, penultimate paragraph of p. 1717).

- **Additive over multiplicative**: BDF's additive decomposition (x = c + e) is
  preferred over the multiplicative BCG decomposition. Szucs 2017 finds BDF more
  accurate by both MSE and MAPE and orders of magnitude faster to estimate.

## Parameters

| Parameter | Description | Recommended Value | Sensitivity | Range |
|-----------|-------------|-------------------|-------------|-------|
| k | Intraday bins per trading day | 26 (15-min, US equities 9:30-16:00) | Low | 13-52 |
| L_days | Rolling estimation window in trading days | 20 | Medium | 10-40 |
| r_max | Maximum number of factors to evaluate in IC | 10 | Low | 5-15 |
| r | Number of common factors (selected by IC_p2) | Typically 1-3 | Medium | 1-r_max |
| n_grid | Number of SETAR threshold candidates | 100 | Low | 50-200 |
| tau_lo_pct | Lower quantile bound for tau grid search | 0.15 | Low | 0.10-0.20 |
| tau_hi_pct | Upper quantile bound for tau grid search | 0.85 | Low | 0.80-0.90 |
| min_regime_frac | Minimum fraction of obs per SETAR regime | 0.10 | Low | 0.05-0.20 |
| min_regime_obs | Minimum absolute obs per SETAR regime | 20 | Low | 10-30 |
| shares_measure | Denominator for turnover: "TSO" or "float" | "TSO" | Medium | n/a |

**Parameter sources:**
- k = 26: Szucs 2017, Section 2 (15-min bins, 9:30 to 16:00 = 6.5 hours = 26 bins).
  BDF 2008 uses k = 25 (20-min bins, CAC40 9:20-17:20).
- L_days = 20: BDF 2008, Section 3.1; Szucs 2017, Section 3. Both use L = 20.
- r_max = 10: Researcher inference. BDF 2008 does not specify r_max but reports
  r is typically small (1-3).
- n_grid, tau bounds, min_regime: Researcher inference based on Hansen (1997).
  BDF 2008, Section 2.3 does not specify grid search details.
- shares_measure = "TSO": Szucs 2017 uses TSO (Section 2, p. 4). BDF 2008 uses
  float (Section 2.1, p. 1711). TSO chosen as default for data availability.
  Sensitivity raised to "Medium" because the choice affects turnover magnitudes
  and benchmark comparability — see discussion in `prepare_turnover_matrix`.

### Initialization

1. **Estimation window**: Collect at least L_days full trading days of intraday volume
   data for N stocks in the same market/index. Exclude half-days and early closures
   (days with fewer than k bins).

2. **Turnover matrix**: Compute X[t, i] = volume[t, i] / shares_data[i] for all bins
   and stocks. The denominator is TSO (default) or float shares — see the detailed
   discussion in `prepare_turnover_matrix` on this choice and its implications.
   Do NOT center or standardize.

3. **Cross-section**: Maintain a stable universe of N >= 30 stocks (for PCA
   asymptotics). Remove stocks that are delisted or have insufficient history within
   the window. If the universe changes (e.g., index reconstitution), re-estimate the
   full model from scratch on the new universe — the specific component time series
   must be re-extracted under the new factor structure.
   (Reference: BDF 2008, Section 3.1; Bai 2003 for large-N requirement.)

4. **First forecast day**: The first forecast is produced for day L_days + 1 (after
   accumulating L_days of history). Szucs 2017 reports 2668 total days - 20 initial
   estimation = 2648 forecast days.
   (Reference: Szucs 2017, Table 1.)

### Calibration

Parameters are re-estimated daily on a rolling L_days window. No manual calibration
is required — all parameters are determined by the data through PCA (r via IC) and
OLS/grid-search (AR1/SETAR coefficients, tau).

The daily re-estimation procedure:
1. Slide the estimation window forward by one day.
2. Recompute the turnover matrix X for the new window.
3. Run select_factor_count to determine r.
4. Run extract_factors to get C_hat, E_hat.
5. Run forecast_common to get the next day's common forecast.
6. Run fit_ar1 or fit_setar per stock on the new E_hat.
7. Run stationarity check; flag non-stationary stocks for U-method fallback.
8. Store the last-bin specific component value as the initial state for the next day.

(Reference: BDF 2008, Section 3.1; Szucs 2017, Section 3, Fig. 1)

## Validation

### Expected Behavior

**Volume prediction accuracy (per-stock per-bin MAPE, 33 DJIA stocks, 15-min bins):**

| Model      | MSE      | MSE*     | MAPE  |
|------------|----------|----------|-------|
| U-method   | 1.02E-03 | 3.65E-05 | 0.503 |
| BDF-AR     | 6.49E-04 | 2.30E-05 | 0.403 |
| BDF-SETAR  | 6.60E-04 | 2.38E-05 | 0.399 |

(Reference: Szucs 2017, Section 5, Table 2a)

Note: 40% MAPE is typical for intraday volume forecasting at 15-minute granularity.
Volume is inherently noisy at fine time scales. The value of the model is relative to
the U-method baseline: ~20% MAPE reduction.

These benchmarks use TSO as the turnover denominator (Szucs 2017, Section 2).
Implementations using TSO should be directly comparable. Implementations using float
shares as the denominator should NOT expect these exact values but should see the
same relative ordering: SETAR > AR > U-method.

MSE and MAPE are computed per stock across all bins and forecast days, then averaged
across stocks. MSE: Szucs 2017, Eq. (1). MAPE: Szucs 2017, Eq. (2):

  MAPE_i = (1/T) * sum_{t=1}^{T} |Y_t - Y_t^f| / Y_t

  MAPE_overall = (1/N_stocks) * sum_i MAPE_i

MSE* is a scale-adjusted metric: MSE* = (1/N) * sum_{i=1}^N MSE_i / (a_i / a_min)^2,
where a_i is the average turnover of stock i and a_min is the smallest average turnover
across all stocks. This normalizes MSE by turnover scale, preventing high-turnover
stocks from dominating. (Reference: Szucs 2017, Section 5, Eq. 14.)

**Volume prediction accuracy (portfolio-level MAPE, 39 CAC40 stocks, 20-min bins):**

| Model              | Mean MAPE | Std    | Q95    |
|--------------------|-----------|--------|--------|
| PCA-SETAR          | 0.0752    | 0.0869 | 0.2010 |
| PCA-ARMA (= AR(1)) | 0.0829   | 0.0973 | 0.2330 |
| Classical (U-method)| 0.0905   | 0.1050 | 0.2490 |

(Reference: BDF 2008, Table 2, panel 1)

NOTE: These benchmarks use float shares as the turnover denominator (BDF 2008,
Section 2.1, p. 1711). An implementation using TSO should NOT expect to match these
exact values. The relative ordering (SETAR > AR > U-method) should hold regardless.

**Scale discrepancy**: The ~5x difference between Szucs MAPE (~0.40) and BDF MAPE
(~0.08) is explained by aggregation level. BDF computes the MAPE of the
portfolio-level error, which diversifies away idiosyncratic forecast errors. Szucs
computes the average of per-stock per-bin MAPE values. Both show the same relative
ordering: SETAR > AR > U-method.
(Reference: Szucs 2017, Eq. 2 vs BDF 2008, Table 2)

**VWAP execution cost (out-of-sample MAPE, 39 CAC40 stocks, 20-min bins):**

| Model                  | Mean   | Std    | Q95    |
|------------------------|--------|--------|--------|
| PCA-SETAR dynamic      | 0.0898 | 0.0954 | 0.2854 |
| PCA-ARMA dynamic       | 0.0922 | 0.0994 | 0.2854 |
| Classical              | 0.1006 | 0.1171 | 0.3427 |

(Reference: BDF 2008, Table 2, panel 3)

Dynamic PCA-SETAR reduces tracking error by ~10% on average versus the classical
approach. Improvements can reach 50% for high-volatility stocks (e.g., CAP GEMINI,
EADS).
(Reference: BDF 2008, Section 4.3.2, Tables 6-7)

### Sanity Checks

1. **PCA reconstruction**: Verify C_hat + E_hat == X (up to floating-point tolerance).
   max(abs(C_hat + E_hat - X)) < 1e-10.

2. **Factor normalization**: Verify F_hat.T @ F_hat / P is approximately I_r.
   max(abs(F_hat.T @ F_hat / P - eye(r))) < 1e-10.

3. **Common component U-shape**: Average c_forecast across stocks. The resulting
   (k,) vector should exhibit a clear U-shape: high at the open, declining through
   midday, rising toward the close.
   (Reference: BDF 2008, Fig. 3, top panels)

4. **Specific component statistics**: E_hat should be approximately mean-zero
   (mean(E_hat) < 0.01 * mean(abs(X))) and have much lower variance than X.
   The values should be roughly in [-0.025, 0.010] while common component values
   are in [0, 0.035] for typical liquid stocks.
   (Reference: BDF 2008, Fig. 3, bottom panels)

5. **Variance ratio**: Var(C_hat) / Var(X) should be > 0.5 for most stocks. If below
   0.30, the PCA is likely misconfigured. Typical range: 0.6-0.8 for liquid stocks.
   (Researcher inference; consistent with BDF 2008, Section 3.1)

6. **AR(1) stationarity**: |psi_1| should be positive and less than 1.0 (stationary).
   Typical range: 0.1-0.6. If |psi_1| >= 1, the specific component is non-stationary
   and should be investigated.
   (Reference: BDF 2008, Eq. (10); psi_2 should be near zero)

7. **SETAR dominance**: When running both models on the same data, SETAR should
   produce lower MAPE than AR(1) for >= 70% of stocks. BDF 2008 reports 36/39
   (92%); Szucs 2017 reports 30/33 (91%) by MAPE.
   (Reference: BDF 2008, Section 3.2; Szucs 2017, Table 2c)

8. **Eigenvalue scree plot**: The first r eigenvalues should show a clear gap from the
   rest. If the optimal r exceeds 5, suspect data quality issues.
   (Researcher inference)

9. **Forecast positivity**: The vast majority (>95%) of bin-level turnover forecasts
   (c_forecast + e_forecast) should be positive without needing clipping. A high rate
   of negative forecasts indicates model failure.
   (Researcher inference)

10. **Toy verification**: Construct a synthetic (P=200, N=5) matrix with one known
    U-shaped factor: X = F @ Lambda.T + noise. Run extract_factors with r=1. The
    recovered common component should correlate > 0.95 with the true factor pattern.
    (Researcher inference)

11. **No-centering verification**: Run PCA on X with and without mean-centering.
    The uncentered version should produce lower out-of-sample MAPE.
    (Researcher inference)

### Edge Cases

1. **Zero-volume bins**: Isolated zero-volume bins (turnover = 0) are valid data points,
   not missing data. PCA handles them naturally. For MAPE computation, exclude bins
   where actual volume is zero (division by zero).
   Stocks with frequent zero-volume bins (>5% of bins across the estimation window)
   should be excluded from the cross-section.
   (Reference: Szucs 2017, Section 2 — all stocks had non-zero volume in every bin)

2. **Half-days and early closures**: Days with fewer than k bins should be excluded
   entirely from the estimation window. They would corrupt the bin-alignment structure.
   (Reference: BDF 2008, Section 3.1 — December 24 and 31 excluded)

3. **Cross-section changes**: If stocks enter or leave the universe (IPOs, delistings,
   index reconstitutions), re-estimate the full model from scratch on the new
   cross-section. The specific component series must be re-extracted under the new
   factor structure because changing N alters the PCA decomposition.
   (Researcher inference)

4. **Non-stationary specific component**: If |psi_1| >= 1 (AR) or |phi_11| >= 1 or
   |phi_21| >= 1 (SETAR), the specific component is non-stationary. This can occur
   during extreme volume events (earnings, M&A). Fallback: use U-method forecast
   (time-of-day average of raw turnover from `u_method_benchmark`) for that stock on
   that day. This is implemented as an explicit stationarity check in
   `run_dynamic_execution`.
   (Researcher inference)

5. **Negative total forecast**: If clipping individual bin forecasts to zero results in
   total remaining forecast <= 0, fall back to equal weighting (1 / k_remaining) for
   that stock.
   (Researcher inference)

6. **Cross-day boundary contamination**: The AR(1)/SETAR model is fit to the full
   concatenated multi-day series, including cross-day transitions (bin k of day d to
   bin 1 of day d+1). Overnight jumps may inflate innovation variance and attenuate
   the AR coefficient. The ACF of the specific component shows periodic spikes at
   multiples of k (BDF 2008, Fig. 2). This is a known limitation; do not add seasonal
   AR terms unless extending beyond the papers.
   (Reference: BDF 2008, Fig. 2; researcher inference on the consequences)

7. **Singular or ill-conditioned matrices**: If N is very large relative to P, the
   covariance matrix may be ill-conditioned. Using SVD (as specified) rather than
   eigendecomposition of X'X avoids this issue by design.
   (Researcher inference)

### Known Limitations

1. **Cross-section required**: The model requires a panel of N >= 30 stocks. It cannot
   forecast a single stock in isolation; the common component is a cross-sectional
   object. Minimum practical N is approximately 10-20 for stable factor estimation.
   (Reference: BDF 2008, Section 2.2; Bai 2003)

2. **No daily volume forecast**: The model forecasts the intraday volume *profile*
   (how volume is distributed across bins), not the total daily volume. To convert
   VWAP weights into share quantities, the trader needs a separate daily volume
   forecast or must work in participation-rate terms.
   (Reference: BDF 2008, Section 4.2.2 — "without knowing the K turnovers of the day
   or the equivalent total volume")

3. **No price model**: The model is purely volume-driven. To "beat" the VWAP
   benchmark (rather than merely track it), a bivariate volume-price model is required.
   (Reference: BDF 2008, Section 5)

4. **Small order assumption**: The model assumes the trader's order size is small
   relative to market volume (no significant price impact). For large orders,
   participation rate limits should be imposed externally.
   (Reference: BDF 2008, Section 4.1.3)

5. **Linear factor model**: The PCA extracts linear factors only. Nonlinear
   cross-sectional dependencies are not captured.
   (Reference: BDF 2008, Section 2.2)

6. **SETAR threshold static within estimation window**: The threshold tau is constant
   across all L_days in the estimation window. Regime-switching dynamics within
   a single day are captured, but slow drift in the threshold over longer horizons
   requires the rolling window to adapt.
   (Reference: BDF 2008, Section 2.3)

7. **Overnight boundary contamination**: Cross-day transitions in the concatenated
   series introduce artificial dynamics. The model does not distinguish overnight
   from intraday transitions.
   (Reference: BDF 2008, Fig. 2; researcher inference)

8. **No intraday events**: The model does not handle scheduled intraday events (FOMC
   announcements, earnings releases) that dramatically alter volume profiles on
   specific days.
   (Researcher inference)

9. **Additive decomposition**: Volume is strictly non-negative but the additive model
   can produce negative forecasts. The multiplicative BCG decomposition avoids this
   but is less accurate and much slower.
   (Reference: Szucs 2017, Section 4)

10. **Factor rotation indeterminacy**: PCA factors are identified only up to rotation.
    Individual factors and loadings are not uniquely interpretable, though the common
    component C = F @ Lambda.T is unique.
    (Reference: Bai 2003)

## Paper References

| Spec Section | Paper Source |
|---|---|
| Model decomposition (x = c + e) | BDF 2008, Section 2.1-2.2, Eq. (2)-(5) |
| Turnover = volume / float | BDF 2008, Section 2.1, p. 1711 |
| Turnover = volume / TSO | Szucs 2017, Section 2, p. 4 |
| PCA estimation (Eq. 6) | BDF 2008, Section 2.2, Eq. (6); Bai (2003) |
| Factor count selection (IC_p2) | BDF 2008, Section 2.2; Bai & Ng (2002) |
| Common component forecast | BDF 2008, Section 2.3, Eq. (9) |
| AR(1) specific component | BDF 2008, Section 2.3, Eq. (10); Szucs 2017, Eq. (5) |
| AR(1) justification (not ARMA) | BDF 2008, Eq. (10) vs text; Szucs 2017, Eq. (5); OLS/MLE consistency |
| SETAR specific component | BDF 2008, Section 2.3, Eq. (11); Szucs 2017, Eq. (6)-(7) |
| Indicator function I(z) | BDF 2008, Eq. (11); Szucs 2017, Eq. (7) |
| MLE = OLS equivalence | BDF 2008, Section 2.3 (valid only for AR(1), not ARMA) |
| Dynamic VWAP execution | BDF 2008, Section 4.2.2 |
| Dynamic weight normalization | BDF 2008, Section 4.2.2; researcher inference for explanation |
| Static execution failure | BDF 2008, Section 4.2, p. 1717 |
| SETAR outperforms AR(1) | BDF 2008, Section 3.2; Szucs 2017, Table 2c |
| U-method baseline | Szucs 2017, Section 4, Eq. (3); BDF 2008, footnote 5 |
| Per-stock MAPE/MSE benchmarks | Szucs 2017, Section 5, Tables 2a-2c |
| MAPE formula | Szucs 2017, Section 3, Eq. (2) |
| MSE formula | Szucs 2017, Section 3, Eq. (1) |
| MSE* formula | Szucs 2017, Section 5, Eq. (14) |
| Portfolio-level benchmarks | BDF 2008, Table 2 |
| VWAP execution cost benchmarks | BDF 2008, Tables 2, 4-7 |
| Data: 33 DJIA, 26 bins, 2668 days | Szucs 2017, Table 1 |
| Data: 39 CAC40, 25 bins, ~250 days | BDF 2008, Section 3.1 |
| No-centering rationale | Bai (2003); BDF 2008, Section 2.2 |
| SETAR grid search | Hansen (1997); BDF 2008, Section 2.3 |
| Sign/rotation indeterminacy | Bai (2003) |
| Computational cost | Szucs 2017, Section 5 (last paragraph) |
| SETAR notation mapping | BDF 2008, Eq. (11) vs Szucs 2017, Eq. (6) |
| Overnight transition dynamics | BDF 2008, Fig. 2; researcher inference |
| Day-start initialization | BDF 2008, Section 2.3 (researcher inference for specific approach) |
| Multi-step forecast method | Researcher inference (BDF does not specify) |
| Stationarity check + U-method fallback | Researcher inference |
| Negative forecast handling | Researcher inference |
| Cross-section change handling | Researcher inference |
| SETAR min regime parameters | Researcher inference based on Hansen (1997) |
| Floating-point V(r) guard | Researcher inference |
| MLE variance estimator choice | BDF 2008, Section 2.3; researcher inference for consistency |
