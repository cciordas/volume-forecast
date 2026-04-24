# Implementation Specification: PCA Factor Decomposition for Intraday Volume (BDF Model)

## Overview

The BDF model forecasts intraday stock volume by decomposing turnover into two additive
components: a market-wide common component (the shared U-shaped seasonal pattern) extracted
via principal components analysis (PCA) on a cross-section of stocks, and a stock-specific
idiosyncratic component modeled by a low-order autoregressive process (AR(1) or SETAR).
The combined forecast feeds a dynamic VWAP execution algorithm that updates weights
intraday as actual volumes are observed.

The model was introduced by Bialkowski, Darolles, and Le Fol (2008) using CAC40 data
and independently validated by Szucs (2017) on 33 DJIA stocks over 11 years, confirming
BDF's superiority over both the naive U-method benchmark and the competing BCG
multiplicative model.

## Algorithm

### Model Description

**What it does:** Given intraday turnover data for N stocks observed at k equal-length bins
per trading day, the model produces one-step-ahead volume forecasts for each stock at each
bin. These forecasts determine a VWAP execution schedule that minimizes tracking error
against the end-of-day VWAP benchmark.

**Core assumption:** Intraday volume is driven by a small number of latent market-wide
factors (primarily the U-shaped pattern) that are common across all stocks. After removing
this common pattern, each stock's residual volume dynamics can be captured by a simple
autoregressive model with one or two regimes.

**Inputs:**
- Intraday turnover matrix: shape (P, N) where P = L * k. Each row is one bin for one day,
  each column is one stock. Values are raw turnover (volume / shares outstanding).
- For dynamic execution: realized turnover values for already-completed bins on the
  current trading day.

**Outputs:**
- One-step-ahead turnover forecast for each stock at each future bin.
- VWAP execution weights: fraction of remaining order to execute in each remaining bin.

### Data Preprocessing

**Source:** BDF 2008, Section 3.1; Szucs 2017, Section 2.

1. **Obtain raw data.** Tick or minute-level trade data for N stocks over an estimation
   window plus an out-of-sample evaluation period. BDF uses N=39 CAC40 stocks
   (BDF 2008, Section 3.1); Szucs uses N=33 DJIA stocks (Szucs 2017, Section 2).

2. **Aggregate into fixed-width intraday bins.**
   - BDF: 20-minute bins, k=25, covering 9:20-17:20 Paris time (BDF 2008, Section 3.1,
     Fig. 1: "25 (k = 25) 20-min intervals").
   - Szucs: 15-minute bins, k=26, covering 9:30-16:00 US market hours (Szucs 2017,
     Section 2: "26 observations every day").
   - The implementation should parameterize k and bin width.

3. **Compute turnover for each bin.**
   - BDF definition: turnover = shares_traded / float_shares (BDF 2008, Section 2.2,
     text below Eq. 3: "the turnover service for stock i at date t, i.e. the number of
     traded shares divided by the panel of floated shares").
   - Szucs definition: turnover = volume / total_shares_outstanding (Szucs 2017,
     Section 2, page 4: "x_t = V_t / TSO_t").
   - Both definitions are valid. TSO is more readily available from standard data
     providers. Use TSO as the default denominator.
   - Handle stock splits: adjust the denominator for split events so that turnover
     remains comparable across the window (BDF 2008, Section 3.1: "adjusted for stock's
     splits and dividends").
   - **TSO data handling (Researcher inference):** Use the most recent TSO value for each
     trading day. If TSO is missing, carry forward the last known value. Flag stock-days
     where TSO changes by more than 10% for manual review (likely corporate action).

4. **Exclude partial trading days.** Half-days before holidays have fewer than k bins and
   must be excluded from the estimation window. BDF excludes Dec 24 and Dec 31, 2003
   (BDF 2008, Section 3.1).

5. **Verify non-zero volume.** All stocks in the cross-section should have non-zero volume
   in every bin throughout the estimation window. Szucs 2017, Section 2: "every stock
   had trades and thus a volume record larger than zero in every 15-minute interval."
   Stocks with frequent zero-volume bins (more than 5% of bins) should be excluded from
   the cross-section. (Researcher inference: practical liquidity filter.)

**Critical: do NOT center the data.** The turnover matrix must NOT be demeaned (no
column-mean subtraction, no row-mean subtraction) before PCA. The Bai (2003) factor
model operates on raw data. The model equation x_{i,t} = lambda_i' * F_t + e_{i,t} has
no intercept term, and the common component is meant to capture both the level and shape
of the market-wide volume pattern. Centering would strip level information and produce a
biased decomposition. (BDF 2008, Section 2.2, Eq. 4-6: no demeaning step mentioned;
Bai 2003, Section 2: model defined without demeaning.)

### Pseudocode

#### Function 1: estimate_model

Top-level daily estimation routine. Estimates the full BDF model on a rolling window of
historical data for one estimation day.

**Source:** BDF 2008, Sections 2.2-2.3; Szucs 2017, Section 4.1.

```
function estimate_model(turnover_matrix, k, L, model_type):
    """
    Parameters
    ----------
    turnover_matrix: array of shape (P, N), where P = L * k
        Raw turnover values. NOT centered. Rows are chronological intraday bins:
        [day_1_bin_0, day_1_bin_1, ..., day_1_bin_{k-1}, day_2_bin_0, ...].
    k: int
        Number of intraday bins per trading day.
    L: int
        Number of trading days in the estimation window.
    model_type: string, one of "AR1" or "SETAR"

    Returns
    -------
    model: dict containing:
        factors:          array (P, r) -- estimated factors F_hat
        loadings:         array (N, r) -- estimated loadings Lambda_hat
        common:           array (P, N) -- common component C_hat
        specific:         array (P, N) -- specific component e_hat
        r:                int -- number of factors
        ts_params:        list of N dicts -- AR(1) or SETAR parameters per stock
        common_forecast:  array (k, N) -- next-day common component forecast
    """
    P, N = turnover_matrix.shape
    assert P == L * k

    # Step 1: Determine number of factors and extract them in one pass
    # [Optimization: a single truncated SVD with r_max components serves both
    # factor-count selection and factor extraction. See extract_and_select_factors.]
    r, factors, loadings, common, specific = extract_and_select_factors(
        turnover_matrix, k, L, r_max=10
    )

    # Step 2: Forecast common component for next day
    common_forecast = forecast_common(common, k, L)

    # Step 3: Fit time-series model to each stock's specific component
    ts_params = []
    for i in range(N):
        if model_type == "SETAR":
            params = fit_setar(specific[:, i])
            if params is None:
                # Fallback to AR(1) if SETAR estimation fails
                params = fit_ar1(specific[:, i])
        else:
            params = fit_ar1(specific[:, i])
        ts_params.append(params)

    return {
        "factors": factors, "loadings": loadings,
        "common": common, "specific": specific,
        "r": r, "ts_params": ts_params,
        "common_forecast": common_forecast,
    }
```

#### Function 2: extract_and_select_factors

Combined factor extraction and factor-count selection using a single truncated SVD.
Determines the number of factors r via the Bai and Ng (2002) information criterion, then
extracts the common and specific components.

**Source:** BDF 2008, Section 2.2, Equations 4-6, reference to Bai & Ng (2002);
Bai (2003), Econometrica 71(1), pp. 135-171; Bai & Ng (2002), Econometrica 70(1),
pp. 191-221.

```
function extract_and_select_factors(turnover_matrix, k, L, r_max=10):
    """
    Parameters
    ----------
    turnover_matrix: array of shape (P, N), P = L * k. Raw, NOT centered.
    k: int -- bins per day
    L: int -- days in window
    r_max: int -- maximum factors to consider (default 10)

    Returns
    -------
    r:        int -- selected number of factors
    F_hat:    array (P, r) -- factors, normalized so F_hat.T @ F_hat / P = I_r
    Lambda:   array (N, r) -- factor loadings
    C_hat:    array (P, N) -- common component
    e_hat:    array (P, N) -- specific component
    """
    P, N = turnover_matrix.shape
    X = turnover_matrix  # (P, N), NOT centered

    # --- Phase 1: Compute truncated SVD for r_max components ---
    # Use scipy.sparse.linalg.svds or sklearn.utils.extmath.randomized_svd.
    # These compute only the top r_max singular triplets efficiently.
    # The SVD internally works with whichever of XX' or X'X is smaller.
    #
    # X = U @ diag(s) @ Vt   (truncated to r_max components)
    # U: (P, r_max), s: (r_max,) descending, Vt: (r_max, N)
    U_full, s_full, Vt_full = truncated_svd(X, n_components=r_max)

    # --- Phase 2: Select r via Bai-Ng IC_p2 ---
    # The total sum of squares of X:
    # Use np.linalg.norm(X, 'fro')**2 for total_ss to avoid materializing X**2.
    # The full turnover matrix X must reside in memory for both the SVD and
    # total_ss computation. For practical cross-sections (N < 500, L < 250),
    # this is under 100 MB and not a concern.
    # (Addressed per critic feedback N4.)
    total_ss = norm(X, 'fro') ** 2   # scalar, equivalent to sum(X ** 2)

    # For each candidate r, the residual variance is:
    #   V(r) = (1 / (N * P)) * (total_ss - sum(s[:r]^2))
    #
    # Derivation: the rank-r SVD approximation X_r = U[:,:r] @ diag(s[:r]) @ Vt[:r,:]
    # satisfies ||X_r||_F^2 = sum(s[:r]^2). Therefore:
    #   ||X - X_r||_F^2 = ||X||_F^2 - sum(s[:r]^2) = total_ss - sum(s[:r]^2)
    #
    # This is a standard property of the Eckart-Young theorem: the truncated SVD
    # minimizes the Frobenius norm of the residual, and the captured variance equals
    # the sum of squared singular values retained.
    # (Researcher inference: standard linear algebra, not stated in BDF 2008.)
    #
    # Note: This formulation is INDEPENDENT of whether P <= N or P > N.
    # The singular values of X are the same regardless of which dimension is larger.
    # No branching on matrix dimensions is needed.

    best_IC = +infinity
    best_r = 1

    for r in range(1, r_max + 1):
        explained_ss = sum(s_full[:r] ** 2)
        residual_ss = total_ss - explained_ss

        # Guard against floating-point underflow when top r components
        # capture nearly all variance (e.g., synthetic low-rank data).
        V_r = max(residual_ss / (N * P), 1e-15)

        # Bai-Ng IC_p2 criterion:
        # IC_p2(r) = ln(V(r)) + r * ((N + P) / (N * P)) * ln(min(N, P))
        # (Bai & Ng 2002, Econometrica 70(1), Section 3, Criterion IC_p2)
        penalty = r * ((N + P) / (N * P)) * ln(min(N, P))
        IC = ln(V_r) + penalty

        if IC < best_IC:
            best_IC = IC
            best_r = r

    r = best_r

    # --- Phase 3: Extract factors with Bai (2003) normalization ---
    # From the truncated SVD, take only the top r components:
    U = U_full[:, :r]       # (P, r)
    s = s_full[:r]          # (r,)
    Vt = Vt_full[:r, :]     # (r, N)

    # Bai (2003) normalization: F_hat.T @ F_hat / P = I_r
    # Set F_hat = sqrt(P) * U, which gives:
    #   F_hat.T @ F_hat / P = P * (U.T @ U) / P = I_r  (since U has orthonormal columns)
    #
    # Then Lambda_hat = (Vt.T * s) / sqrt(P), so that:
    #   F_hat @ Lambda_hat.T = sqrt(P) * U @ diag(s) @ Vt / sqrt(P)
    #                        = U @ diag(s) @ Vt = X_r   (the rank-r approximation)
    #
    # (BDF 2008, Section 2.2, Eq. 6: "concentrating out A and using the normalization
    #  F'F/T = I_r"; Bai 2003, Section 2.)

    F_hat = sqrt(P) * U                    # (P, r)
    Lambda_hat = (Vt.T * s) / sqrt(P)      # (N, r)

    C_hat = F_hat @ Lambda_hat.T           # (P, N) common component
    e_hat = X - C_hat                      # (P, N) specific component

    return r, F_hat, Lambda_hat, C_hat, e_hat
```

**Note on IC variant:** BDF 2008 references Bai & Ng (2002) for factor count selection but
does not specify which of the six IC variants they use (Section 2.2: "using the criteria of
Bai and Ng (2002)"). IC_p2 is the most commonly used in empirical work and has good
finite-sample properties. Its penalty term is r * ((N+P)/(N*P)) * ln(min(N,P)).
(Researcher inference: IC_p2 chosen as default based on prevalence in applied factor
analysis literature.)

**Alternative:** IC_p1 uses penalty r * ((N+P)/(N*P)) * ln((N*P)/(N+P)). If results
are sensitive to the IC variant, test both.

**Note on V(r) from singular values vs. eigenvalues:** Previous formulations branch on
P <= N to decide whether to eigendecompose XX'/N or X'X/P. The SVD-based formulation
above avoids this entirely. The singular values of X are unique (independent of which
dimension is larger), and V(r) = (total_ss - sum(s[:r]^2)) / (N*P) is always correct.
(Researcher inference: standard SVD property, Eckart-Young theorem.)

#### Function 3: forecast_common

Forecasts the common component for the next trading day by averaging the common
component at the same time-of-day across the L prior days.

**Source:** BDF 2008, Section 2.3, Equation 9: "c_hat_{i,t+1} = (1/L) * sum_{l=1}^{L}
c_{i,t+1-k*l}". Also footnote 5: replacing c by x in Eq. 9 gives the "classical approach."

```
function forecast_common(common, k, L):
    """
    Parameters
    ----------
    common: array of shape (P, N), P = L * k
        Estimated common component C_hat from extract_and_select_factors.
        Row layout: [day_1_bin_0, ..., day_1_bin_{k-1}, day_2_bin_0, ...].
    k: int -- bins per day
    L: int -- days in window

    Returns
    -------
    common_forecast: array of shape (k, N)
        Forecast common component for the next day.
    """
    N = common.shape[1]

    # Reshape from (L*k, N) to (L, k, N), then average across the L days
    # for each bin position. This computes the time-of-day average.
    common_3d = reshape(common, (L, k, N))
    common_forecast = mean(common_3d, axis=0)  # (k, N)

    return common_forecast
```

**Key design points:**
- The common forecast is computed BEFORE the trading day begins (BDF 2008, Section 2.3,
  text below Eq. 9). No look-ahead bias: only estimation-window data is used.
- The common forecast is FIXED for the entire trading day. It is NOT updated intraday.
  Only the specific component updates as actual volumes arrive. (BDF 2008, Section 2.3:
  the common component captures the "stable seasonal shape across time" which "has no
  impact on the VWAP".)
- The common forecast of BDF is structurally the U-method applied to C_hat instead of
  raw turnover X. The predictive improvement over the U-method comes entirely from the
  specific component's AR/SETAR forecast. (BDF 2008, footnote 5.)

#### Function 4: fit_ar1

Fits an AR(1) model with constant to one stock's specific component series.

**Source:** BDF 2008, Section 2.3, Equation 10: "e_{i,t} = psi_1 * e_{i,t-1} + psi_2 +
epsilon_{i,t}". Szucs 2017, Section 4.1, Equation 5: "e_p = c + theta_1 * e_{p-1} +
epsilon_p" (same model, different notation).

**Note on ARMA(1,1) vs AR(1):** BDF 2008 refers to this as "ARMA(1,1)" in the text, but
Equation 10 contains no moving-average term -- it is an AR(1) with intercept. Szucs 2017
explicitly labels it AR(1) (Section 4.1). We implement AR(1) as written in the equations.

```
function fit_ar1(e_series):
    """
    Parameters
    ----------
    e_series: array of shape (P,)
        Specific component series for one stock over the full estimation window.
        Treated as a single contiguous series across day boundaries (see note below).

    Returns
    -------
    params: dict with keys:
        psi_1:  float -- AR(1) coefficient
        psi_2:  float -- intercept
        sigma2: float -- innovation variance (unbiased estimate, diagnostic only)
    """
    # Model: e_t = psi_1 * e_{t-1} + psi_2 + epsilon_t
    # Estimate by OLS (equivalent to conditional MLE under Gaussian innovations).
    # BDF 2008, Section 2.3: "estimate ... by maximum likelihood."
    # For AR(1), conditional MLE = OLS. (Researcher inference: standard result.)

    n = len(e_series) - 1          # number of usable observations
    y = e_series[1:]               # (n,): e_t for t = 1..P-1
    X_reg = column_stack([e_series[:-1], ones(n)])  # (n, 2): [e_{t-1}, 1]

    # OLS: beta = (X'X)^{-1} X'y
    beta = solve(X_reg.T @ X_reg, X_reg.T @ y)     # (2,)
    psi_1 = beta[0]
    psi_2 = beta[1]

    # Residuals and unbiased variance estimate
    residuals = y - X_reg @ beta
    # Unbiased: divide by (n - 2) because 2 parameters are estimated.
    # (Researcher inference: standard OLS residual variance.)
    # sigma2 is computed for diagnostic purposes only (e.g., testing residual
    # normality, comparing regime volatilities, computing confidence intervals).
    # It is NOT used in point forecasting. (Addressed per critic feedback P2.)
    sigma2 = sum(residuals ** 2) / (n - 2)

    return {"psi_1": psi_1, "psi_2": psi_2, "sigma2": sigma2}
```

**Contiguous series treatment:** The AR(1)/SETAR is fit on the full contiguous series of
length P = L*k, treating overnight gaps as ordinary lag-1 transitions. This means bin
k-1 of day d is followed by bin 0 of day d+1 in the autoregressive structure. This is the
approach used by both BDF 2008 and Szucs 2017. Do not segment the series by day.
(BDF 2008, Section 2.3, Eq. 10: the model is written on a single time index t running
from 1 to T, with no day-boundary segmentation. Fig. 2: ACF of the specific component
shows spikes at multiples of 25, which are artifacts of the overnight gap in a contiguous
series. Szucs 2017, Section 4.1: same single-index formulation in Eq. 5.)

The AR(1) captures only lag-1 dependence. The periodic ACF spikes at multiples of k are
NOT modeled. Do not add seasonal AR terms unless extending beyond the published model.
(Researcher inference based on BDF 2008, Fig. 2.)

**Note on estimation method:** BDF 2008 specifies "maximum likelihood" (Section 2.3). For
AR(1) with Gaussian innovations, conditional MLE is identical to OLS. OLS is simpler
and numerically stable. (Researcher inference: standard econometric equivalence.)

#### Function 5: fit_setar

Fits a two-regime Self-Exciting Threshold Autoregressive model to one stock's specific
component series.

**Source:** BDF 2008, Section 2.3, Equation 11; Szucs 2017, Section 4.1, Equation 6.

```
function fit_setar(e_series, n_grid=100):
    """
    Parameters
    ----------
    e_series: array of shape (P,)
        Specific component series for one stock.
        Treated as a single contiguous series across day boundaries.
    n_grid: int
        Number of grid points for threshold search (default 100).

    Returns
    -------
    params: dict or None
        If estimation succeeds:
            phi_11:   float -- AR coeff, regime 1 (e_{t-1} <= tau)
            phi_12:   float -- intercept, regime 1
            phi_21:   float -- AR coeff, regime 2 (e_{t-1} > tau)
            phi_22:   float -- intercept, regime 2
            tau:      float -- threshold
            sigma2_1: float -- innovation variance, regime 1 (diagnostic only)
            sigma2_2: float -- innovation variance, regime 2 (diagnostic only)
        If no valid threshold found: None (caller should fall back to AR(1)).
    """
    # Model (BDF 2008 Eq. 11):
    # e_t = (phi_11 * e_{t-1} + phi_12) * I(e_{t-1} <= tau)
    #     + (phi_21 * e_{t-1} + phi_22) * (1 - I(e_{t-1} <= tau))
    #     + epsilon_t
    #
    # Notation mapping:
    #   Spec         BDF Eq. 11        Szucs Eq. 6
    #   phi_11       phi_11            theta_{1,2}
    #   phi_12       phi_12            c_{1,1}
    #   phi_21       phi_21            theta_{2,2}
    #   phi_22       phi_22            c_{2,1}
    #   tau          tau               tau
    # WARNING: Szucs puts intercept first (c_{j,1}) and AR coeff second (theta_{j,2}),
    # opposite of this spec's ordering.

    y = e_series[1:]          # (n,): e_t for t = 1..P-1
    x_lag = e_series[:-1]     # (n,): e_{t-1}
    n = len(y)

    # --- Grid search over tau to minimize total SSR ---
    # Search quantiles of the lagged values. Restrict to [15th, 85th] percentile
    # to ensure both regimes have adequate data.
    # (Researcher inference: standard SETAR estimation practice, following Hansen 1997.)
    tau_candidates = linspace(quantile(x_lag, 0.15), quantile(x_lag, 0.85), n_grid)

    min_regime_obs = max(int(0.10 * n), 20)  # minimum observations per regime

    best_ssr = +infinity
    best_tau = None
    best_beta1 = None   # Store winning coefficients to avoid re-computation
    best_beta2 = None
    best_resid1 = None
    best_resid2 = None
    best_n1 = None
    best_n2 = None

    for tau in tau_candidates:
        mask1 = (x_lag <= tau)
        mask2 = ~mask1
        n1 = sum(mask1)
        n2 = sum(mask2)

        if n1 < min_regime_obs or n2 < min_regime_obs:
            continue

        # Conditional OLS in each regime
        # Regime 1: y_1 = phi_11 * x_1 + phi_12 + eps_1
        y1, x1 = y[mask1], x_lag[mask1]
        X1 = column_stack([x1, ones(n1)])
        beta1 = solve(X1.T @ X1, X1.T @ y1)
        resid1 = y1 - X1 @ beta1
        ssr1 = sum(resid1 ** 2)

        # Regime 2: y_2 = phi_21 * x_2 + phi_22 + eps_2
        y2, x2 = y[mask2], x_lag[mask2]
        X2 = column_stack([x2, ones(n2)])
        beta2 = solve(X2.T @ X2, X2.T @ y2)
        resid2 = y2 - X2 @ beta2
        ssr2 = sum(resid2 ** 2)

        total_ssr = ssr1 + ssr2
        if total_ssr < best_ssr:
            best_ssr = total_ssr
            best_tau = tau
            best_beta1 = beta1
            best_beta2 = beta2
            best_resid1 = resid1
            best_resid2 = resid2
            best_n1 = n1
            best_n2 = n2

    if best_tau is None:
        return None  # no valid threshold; caller falls back to AR(1)

    # Per-regime unbiased variance: each regime has 2 parameters (AR coeff + intercept)
    # sigma2 values are diagnostic only -- not used in point forecasting.
    # Useful for comparing regime volatilities and computing confidence intervals.
    # (Researcher inference: standard OLS residual variance.)
    sigma2_1 = sum(best_resid1 ** 2) / (best_n1 - 2)
    sigma2_2 = sum(best_resid2 ** 2) / (best_n2 - 2)

    return {
        "phi_11": best_beta1[0], "phi_12": best_beta1[1],
        "phi_21": best_beta2[0], "phi_22": best_beta2[1],
        "tau": best_tau,
        "sigma2_1": sigma2_1, "sigma2_2": sigma2_2,
    }
```

**Estimation details:** BDF 2008 states estimation by "maximum likelihood" (Section 2.3,
paragraph below Eq. 11). For SETAR with Gaussian innovations, profile MLE conditional on
tau reduces to regime-specific OLS. The threshold tau is selected by grid search
minimizing total SSR. This is the standard approach (Hansen 1997, Econometrica).
(Researcher inference: BDF does not detail the SETAR estimation procedure beyond "MLE".)

**Single-pass estimation:** The grid search stores the winning coefficients (beta1, beta2)
and residuals alongside the SSR, so no second pass over the data is needed for the
selected tau. This eliminates redundant OLS computation and reduces the risk of bugs
from duplicated code paths. (Researcher inference: efficiency optimization. Addressed
per critic feedback N3.)

**Fallback to AR(1):** If SETAR estimation fails (no tau candidate produces regimes above
the minimum size), fall back to AR(1). When SETAR succeeds, always use SETAR -- do NOT
apply an improvement test (F-test, likelihood ratio, etc.). This is consistent with BDF's
approach: BDF 2008 Section 3.2 reports SETAR results for all 39 stocks, implying SETAR
was used uniformly wherever estimation succeeded. (Researcher inference: BDF does not
mention any per-stock model selection criterion between AR and SETAR.)

**Minimum regime size:** The 15%-85% quantile range and the min(0.10*n, 20) constraint
prevent degenerate regimes with too few observations for stable OLS estimation.
(Researcher inference: not specified in either paper.)

#### Function 6: forecast_specific

Produces a one-step-ahead specific component forecast given the most recent observed value.

**Source:** BDF 2008, Section 2.3, Equations 10-11; Section 4.2.2 (dynamic updating).

```
function forecast_specific(ts_params, last_specific):
    """
    Parameters
    ----------
    ts_params: dict -- AR(1) or SETAR parameters for one stock
    last_specific: float -- most recent observed specific component e_{t-1}

    Returns
    -------
    e_forecast: float -- forecast specific component for next bin
    """
    if "tau" in ts_params:
        # SETAR forecast (BDF 2008 Eq. 11)
        if last_specific <= ts_params["tau"]:
            e_forecast = ts_params["phi_11"] * last_specific + ts_params["phi_12"]
        else:
            e_forecast = ts_params["phi_21"] * last_specific + ts_params["phi_22"]
    else:
        # AR(1) forecast (BDF 2008 Eq. 10)
        e_forecast = ts_params["psi_1"] * last_specific + ts_params["psi_2"]

    return e_forecast
```

#### Function 7: forecast_turnover

Produces a one-step-ahead turnover forecast combining common and specific components.

**Source:** BDF 2008, Section 2.3, Equation 8: "x_{i,t+1} = c_hat_{i,t+1} + e_hat_{i,t+1}".

```
function forecast_turnover(model, stock_idx, bin_idx, last_specific):
    """
    Parameters
    ----------
    model: dict -- output of estimate_model
    stock_idx: int -- stock index (0..N-1)
    bin_idx: int -- next bin to forecast (0..k-1)
    last_specific: float -- most recent specific component value e_{t-1}

    Returns
    -------
    x_hat: float -- forecast turnover
    e_hat: float -- forecast specific component (for chaining)
    """
    c_hat = model["common_forecast"][bin_idx, stock_idx]
    e_hat = forecast_specific(model["ts_params"][stock_idx], last_specific)
    x_hat = c_hat + e_hat
    return x_hat, e_hat
```

#### Function 8: compute_vwap_weights

Converts turnover forecasts for remaining bins into VWAP execution weights.

**Source:** BDF 2008, Section 4.2.2: the proportion of the order to trade at each interval
is determined by the forecast volume share.

```
function compute_vwap_weights(forecasts_remaining):
    """
    Parameters
    ----------
    forecasts_remaining: array of shape (k_remaining,)
        Forecast turnover for each remaining bin for one stock.

    Returns
    -------
    weights: array of shape (k_remaining,)
        Fraction of remaining order to execute in each bin. Sums to 1.0.
        All non-negative.
    """
    # Clip negative forecasts to zero. The additive decomposition can produce
    # c_hat + e_hat < 0 when the specific component is large and negative,
    # or in rare cases when the common component itself is negative (since PCA
    # does not enforce non-negativity).
    # Volume cannot be negative, so clip at 0. (Researcher inference.)
    forecasts_clipped = maximum(forecasts_remaining, 0.0)
    total = sum(forecasts_clipped)

    if total <= 0.0:
        # All forecasts are zero or negative: model failure.
        # Fall back to equal weighting across remaining bins.
        # (Researcher inference: safest fallback when model provides no signal.)
        k_remaining = len(forecasts_remaining)
        return ones(k_remaining) / k_remaining

    return forecasts_clipped / total
```

**Note on negative forecasts:** If more than 50% of remaining bin forecasts are negative
before clipping, this indicates a model failure for this stock-day. The system should log
a warning. The equal-weight fallback is equivalent to the "classical approach" without
the U-shape -- a conservative default. (Researcher inference.)

#### Function 9: execute_one_bin (Event-Driven Dynamic Execution)

Executes one bin of the dynamic VWAP strategy for one stock. This function is called
once per completed bin by an external event loop, and returns the shares to trade in the
next bin along with updated state.

**Source:** BDF 2008, Section 4.2.2: "we include the information on intraday volume after
each time interval" and re-forecast remaining bins. The dynamic strategy updates the
specific component forecast after each observed bin. (BDF 2008, page 1717: "The
prediction e_{i,t}, t = 1,...,25 is still the one-step ahead prediction of the dynamic
model... The proportion w_t is only applied on the remaining volume to trade after
interval t.")

```
function execute_one_bin(model, stock_idx, bin_idx, shares_remaining,
                         last_specific, actual_turnover=None):
    """
    Event-driven dynamic execution for one stock at one bin.

    Call sequence for a full trading day:
      1. Before market open: state = init_execution_state(model, stock_idx, order_shares)
      2. For each bin j = 0..k-1:
         a. Call execute_one_bin to get shares_to_trade for bin j.
         b. Submit the order.
         c. After bin j completes, observe actual_turnover[j].
         d. Call execute_one_bin again with actual_turnover to update state,
            OR update last_specific directly:
            last_specific = actual_turnover - model["common_forecast"][j, stock_idx]

    Parameters
    ----------
    model: dict -- output of estimate_model (estimated the night before)
    stock_idx: int -- which stock to trade
    bin_idx: int -- current bin index (0..k-1)
    shares_remaining: float -- shares left to execute (including this bin)
    last_specific: float -- most recent specific component value e_{t-1}
        For bin 0: model["specific"][-1, stock_idx] (last in-sample value).
        For bin j > 0: actual_turnover[j-1] - model["common_forecast"][j-1, stock_idx].
    actual_turnover: float or None
        If provided, this is the observed turnover for bin_idx (used to update
        last_specific for the next call). If None, only the execution decision
        is returned without state update.

    Returns
    -------
    result: dict with keys:
        shares_to_trade: float -- shares to execute in bin_idx
        updated_last_specific: float -- updated specific component for next bin
            (equals last_specific if actual_turnover is None)
        shares_after: float -- shares remaining after this bin
    """
    k = model["common_forecast"].shape[0]

    if shares_remaining <= 0 or bin_idx >= k:
        return {
            "shares_to_trade": 0.0,
            "updated_last_specific": last_specific,
            "shares_after": shares_remaining,
        }

    # Forecast all remaining bins bin_idx..k-1 using iterated AR/SETAR
    forecasts_remaining = []
    e_chain = last_specific
    for future_j in range(bin_idx, k):
        x_hat, e_chain = forecast_turnover(model, stock_idx, future_j, e_chain)
        forecasts_remaining.append(x_hat)

    # Convert forecasts to execution weights
    weights = compute_vwap_weights(array(forecasts_remaining))

    # Execute: trade weight[0] fraction of remaining shares in this bin
    shares_this_bin = weights[0] * shares_remaining

    # Update state if actual turnover is observed
    if actual_turnover is not None:
        # Extract actual specific component:
        # e_actual = actual_turnover - c_hat_forecast[bin_idx]
        # NOTE: the common forecast is NOT updated -- it stays fixed.
        updated_specific = (
            actual_turnover - model["common_forecast"][bin_idx, stock_idx]
        )
    else:
        updated_specific = last_specific

    return {
        "shares_to_trade": shares_this_bin,
        "updated_last_specific": updated_specific,
        "shares_after": shares_remaining - shares_this_bin,
    }
```

**Example usage (external event loop for one stock):**

```
# Nightly: estimate model
model = daily_update(all_data, today, k, L, "SETAR")

# Market open: initialize execution state
order_shares = 10000
last_specific = model["specific"][-1, stock_idx]
shares_remaining = order_shares
schedule = []

# Intraday: event-driven loop
for j in range(k):
    # Get execution decision for bin j
    result = execute_one_bin(
        model, stock_idx, j, shares_remaining, last_specific
    )
    schedule.append((j, result["shares_to_trade"]))

    # ... submit order, wait for bin to complete, observe actual volume ...
    actual = get_actual_turnover(stock_idx, j)  # from market data feed

    # Update state with observed data
    result = execute_one_bin(
        model, stock_idx, j, shares_remaining, last_specific,
        actual_turnover=actual
    )
    last_specific = result["updated_last_specific"]
    shares_remaining = result["shares_after"]
```

**Critical insight: why dynamic, not static.** BDF 2008, Section 4.2, explicitly shows
that static execution (predicting all k bins at market open with no intraday updates)
performs WORSE than the naive classical approach. Multi-step AR/SETAR forecasts decay
exponentially toward the unconditional mean, effectively neutralizing the specific
component's contribution beyond a few bins. Dynamic execution -- re-forecasting after
each observed bin -- is essential for the model to add value. (BDF 2008, Section 4.2.2,
page 1717: "the simplicity of this strategy is offset by the poor quality of the
long-term estimates given by the ARMA model. Briefly, the specific volume prediction will
become zero and the dynamic part of the model will have no effect on VWAP implementation.")

**Multi-step forecast interpretation:** Although we forecast all remaining bins j..k-1,
only weights[0] (the first weight) is acted upon. The remaining forecasts exist solely
to compute the weight denominator. After observing actual volume for bin j, we re-forecast
from scratch using the fresh observed specific component. (BDF 2008, Section 4.2.2.)

**Loop order for multi-stock execution:** When executing multiple stocks simultaneously,
the bin loop should be OUTER and the stock loop INNER. All stocks' bin-j weights must be
computed before any stock observes its actual bin-j volume, to maintain temporal
consistency. (Researcher inference: execution-timing requirement.)

#### Function 10: daily_update

Top-level daily workflow: re-estimate the model on a rolling window for the next day.

**Source:** BDF 2008, Section 3.2; Szucs 2017, Section 3, Figure 1.

```
function daily_update(all_turnover_data, day_index, k, L, model_type):
    """
    Parameters
    ----------
    all_turnover_data: array of shape (total_days * k, N)
        Full historical turnover dataset.
    day_index: int -- current day (0-indexed); we forecast for day_index + 1.
    k: int -- bins per day
    L: int -- estimation window in days
    model_type: string -- "AR1" or "SETAR"

    Returns
    -------
    model: dict -- estimated model for tomorrow's execution
    """
    # Extract the most recent L days ending at day_index
    start_row = (day_index - L + 1) * k
    end_row = (day_index + 1) * k
    turnover_window = all_turnover_data[start_row:end_row, :]  # (L*k, N)

    model = estimate_model(turnover_window, k, L, model_type)
    return model
```

#### Function 11: u_method_benchmark

Implements the U-method (simple historical average) as a baseline for comparison.

**Source:** Szucs 2017, Section 4, Equation 3: "y_hat_{p+1} = (1/L) * sum_{l=1}^{L}
y_{p+1-m*l}". Also BDF 2008, footnote 5: replacing c by x in Eq. 9 gives the classical
approach.

```
function u_method_benchmark(all_turnover_data, day_index, stock_idx, k, L):
    """
    Parameters
    ----------
    all_turnover_data: array of shape (total_days * k, N)
    day_index: int -- current day
    stock_idx: int
    k: int -- bins per day
    L: int -- averaging window

    Returns
    -------
    forecast: array of shape (k,) -- forecast turnover per bin for next day
    """
    # Vectorized: reshape the stock's estimation window into (L, k) and average.
    # This is equivalent to averaging the same time-of-day bin across L days.
    # (Addressed per critic feedback P5: vectorized for consistency with
    # forecast_common.)
    start_row = (day_index - L + 1) * k
    end_row = (day_index + 1) * k
    stock_data = all_turnover_data[start_row:end_row, stock_idx]  # (L*k,)
    stock_3d = reshape(stock_data, (L, k))
    forecast = mean(stock_3d, axis=0)  # (k,)

    return forecast
```

### Data Flow

```
Raw tick/minute data
    |
    v
[Aggregate to k-bin turnover per day: turnover = volume / shares_outstanding]
    |
    v
Turnover matrix X: shape (P, N), P = L * k
    |  (raw values, NOT centered/demeaned)
    |
    v
[extract_and_select_factors(X, k, L, r_max=10)]
    |  Single truncated SVD of X with r_max components
    |
    +---> Phase 1: IC_p2 criterion selects r from singular values
    |     V(r) = (total_ss - sum(s[:r]^2)) / (N*P)
    |     total_ss = norm(X, 'fro')^2
    |
    +---> Phase 2: From top-r singular triplets:
    |     F_hat = sqrt(P) * U[:, :r]           shape (P, r), F'F/P = I_r
    |     Lambda = (Vt[:r, :].T * s[:r]) / sqrt(P)   shape (N, r)
    |     C_hat = F_hat @ Lambda.T              shape (P, N)
    |     e_hat = X - C_hat                     shape (P, N)
    |
    +--> [forecast_common(C_hat, k, L)]
    |        Reshape to (L, k, N), average over L days --> c_forecast: (k, N)
    |        Fixed for the entire trading day.
    |
    +--> [fit_ar1(e_hat[:, i])] or [fit_setar(e_hat[:, i])] for each stock i
    |        --> ts_params: list of N dicts
    |
    v
Model dict (estimated nightly)
    |
    v
[execute_one_bin] -- event-driven, called once per bin per stock:
    Initialize: last_specific = e_hat[-1, stock_idx]
    For each bin j = 0..k-1 (external event loop):
        [forecast_turnover] for bins j..k-1 via iterated AR/SETAR
            --> forecasts_remaining: (k-j,)
        [compute_vwap_weights]
            --> weights: (k-j,), non-negative, sums to 1.0
        Execute weights[0] * shares_remaining
        Observe actual turnover for bin j
        Update: last_specific = actual_turnover - c_forecast[j, stock_idx]
        shares_remaining -= shares_this_bin
```

**Type summary:**
- Turnover values: float64 (typical range 1e-5 to 1e-2 for individual 15-20 min bins
  under normal conditions; values above 1e-2 are rare and suggest extreme events)
- Factor matrix F_hat: float64, shape (P, r), r typically 1-3
- Loading matrix Lambda: float64, shape (N, r)
- Common component C_hat: float64, shape (P, N), typically positive but not guaranteed
  non-negative (PCA does not enforce non-negativity constraints; for well-behaved turnover
  data the first factor captures the positive level, but negative values are
  mathematically possible)
- Specific component e_hat: float64, shape (P, N), can be negative
- AR(1) parameters: 3 floats (psi_1, psi_2, sigma2). sigma2 is diagnostic only.
- SETAR parameters: 7 floats (phi_11, phi_12, phi_21, phi_22, tau, sigma2_1, sigma2_2).
  sigma2_1 and sigma2_2 are diagnostic only.
- VWAP weights: float64, shape (k_remaining,), in [0, 1], sums to 1.0

### Variants

**Implemented variant:** PCA-SETAR with AR(1) fallback, dynamic VWAP execution.

**Rationale:**
- SETAR outperforms AR(1) for 36 of 39 CAC40 stocks in estimation (BDF 2008, Section 3.2:
  "the SETAR model seems to be better...for 36 stocks out of 39").
- Szucs 2017, Table 2c: BDF_SETAR beats BDF_AR on 30/3 stocks by MAPE in pairwise
  comparison.
- Dynamic execution is the only operationally viable strategy; static execution is
  dominated (BDF 2008, Section 4.2.2).

**NOT implemented:**
- **Static execution:** Explicitly shown to be inferior to the classical approach. Multi-step
  forecasts decay to zero. (BDF 2008, Section 4.2, page 1717.)
- **Theoretical execution:** Requires end-of-day total volume, unavailable in real time. Used
  only as an upper bound in the paper. (BDF 2008, Section 4.1.3: "a strategy it is just
  impossible to implement without knowing the k turnovers of the day".)
- **ARMA(1,1):** The equation in BDF 2008 is actually AR(1) with intercept (no MA term in
  Eq. 10). Szucs 2017 confirms AR(1).

## Parameters

| Parameter | Description | Recommended Value | Sensitivity | Range |
|-----------|-------------|-------------------|-------------|-------|
| k | Intraday bins per trading day | 26 (Szucs) or 25 (BDF) | Low -- set by market hours and bin width | 13-52 |
| L | Estimation window (trading days) | 20 | Medium -- shorter adapts faster, longer reduces noise | 10-40 |
| delta_t | Bin width (minutes) | 15 (Szucs) or 20 (BDF) | Low -- determines k given market hours | 5-30 |
| r_max | Maximum factors to test in IC | 10 | Low -- must exceed true r | 5-20 |
| r | Number of factors (data-driven) | Typically 1-3 | High -- too few misses patterns, too many overfits | 1-10 |
| model_type | Specific component model | "SETAR" | High -- SETAR consistently wins on MAPE | {"AR1", "SETAR"} |
| n_grid | SETAR threshold grid points | 100 | Low -- 50-200 gives similar results | 50-500 |
| tau_quantile_range | Quantile bounds for threshold search | [0.15, 0.85] | Low-Medium | [0.10, 0.90] to [0.20, 0.80] |
| min_regime_obs | Min observations per SETAR regime | max(0.10 * n, 20) | Low -- robustness guard | 10-50 |
| shares_denom | Turnover denominator | "TSO" | Low -- scale factor only | {"TSO", "float"} |

### Initialization

**Source:** BDF 2008, Section 3.2; Szucs 2017, Section 3.

1. **First estimation day:** Requires at least L trading days of historical turnover data for
   all N stocks. With L=20 and k=26, the first estimation uses P=520 observations per
   stock. (Szucs 2017, Section 3, Figure 1: "20 days (520 observations)".)

2. **PCA initialization:** No special initialization needed. The truncated SVD is
   deterministic (up to sign flips of singular vectors, which cancel in
   C_hat = F_hat @ Lambda_hat.T). (Bai 2003; BDF 2008, Section 2.2.)

3. **AR(1)/SETAR initialization:** OLS is deterministic; no starting point needed. The
   SETAR grid search is exhaustive within the specified quantile range.

4. **Dynamic execution initialization:** At market open (bin 0), last_specific is set to the
   in-sample specific component from the final bin of the estimation window:
   model["specific"][-1, stock_idx]. This is the most recent available estimate of the
   stock's idiosyncratic state. Using this in-sample value is correct and intended -- all
   estimation data precedes the forecast day. (BDF 2008, Section 2.3: forecasts conditioned
   on information through end of estimation window.)

5. **Factor count r:** Re-estimated daily via the Bai-Ng IC. In practice, r is stable
   (typically 1-3) and rarely changes day-to-day. (Researcher inference based on BDF 2008,
   Fig. 3 showing stable common component shape.)

### Calibration

**Source:** BDF 2008, Sections 2.2-2.3, 3.2; Szucs 2017, Section 3.

The model is fully re-estimated daily on a rolling L-day window. The nightly calibration
procedure for trading day d+1:

1. Form turnover matrix X from the most recent L days (day d-L+1 through day d), shape
   (L*k, N).
2. Run extract_and_select_factors to determine r, and compute F_hat, Lambda_hat, C_hat,
   e_hat.
3. Run forecast_common to get next-day common forecast c_hat: shape (k, N).
4. For each stock i, run fit_setar (with AR(1) fallback) on e_hat[:, i].
5. Package the model dict for use during day d+1's dynamic execution.

**No cross-validation or hyperparameter tuning** is needed beyond the Bai-Ng IC for r. All
other parameters (AR coefficients, SETAR coefficients and threshold) are estimated directly
by OLS on the rolling window. (BDF 2008, Section 3.2.)

**Computational cost:** BDF model estimation for 33 stocks over 2648 days takes approximately
2 hours on a single machine (Szucs 2017, Section 5, last paragraph: "the BDF model took
about two hours to run"). Per-day estimation is on the order of seconds, dominated by the
SVD step.

## Validation

### Evaluation Protocol

**Source:** Szucs 2017, Section 3.

The validation benchmarks in this spec use **one-step-ahead intraday-updating** evaluation.
This is critical: a developer who evaluates using static next-day forecasts (all k bins
predicted at market open without intraday updates) will get substantially worse numbers
and incorrectly conclude the implementation is wrong.

Szucs 2017, Section 3, page 5: "While parameters are updated daily, the information base
for the forecast is updated every 15 minutes. This is because 26 data points are to be
forecasted each day. Although the parameters of the models are unchanged during the day,
it makes sense to take advantage of the actuals that unfold during the day. This approach
is often called one-step-ahead forecasting."

**Step-by-step evaluation protocol:**

1. For each forecast day d (d = L+1 to D, where D is the total number of days):
   a. Estimate the model on the prior L-day window (days d-L through d-1).
   b. Compute the common forecast: c_forecast (k, N).
   c. For each stock i:
      - **Bin j=0 forecast:** Use common + specific component initialized from the last
        in-sample residual:
        `last_specific = model["specific"][-1, i]`
        `x_hat[0] = c_forecast[0, i] + forecast_specific(ts_params[i], last_specific)`
        This conditions on end-of-estimation-window information, consistent with
        BDF 2008 Section 2.3.
      - **Bin j>0 forecast:** Use common + specific component initialized from the
        ACTUAL observed turnover at bin j-1:
        `last_specific = actual_turnover[j-1, i] - c_forecast[j-1, i]`
        `x_hat[j] = c_forecast[j, i] + forecast_specific(ts_params[i], last_specific)`
      - Compute forecast error: `error[j] = actual_turnover[j, i] - x_hat[j]`

2. Compute MSE and MAPE per stock over all (day, bin) pairs.
3. Report cross-stock averages.

**The Szucs Table 2a benchmark numbers use this protocol.** The BDF Table 2 VWAP execution
numbers use the dynamic execution strategy with an additional VWAP weight computation step
applied to the same one-step-ahead forecasts.

### Validation Metrics

**Source:** Szucs 2017, Section 3, Equations 1-2.

**Mean Squared Error (MSE):**

```
MSE_i = (1 / n_i) * sum_{t=1}^{n_i} (Y_t - Y_t^f)^2
```

where Y_t is actual turnover, Y_t^f is forecast turnover, and n_i is the total number of
bin-level forecasts for stock i (= number_of_forecast_days * k).
(Szucs 2017, Section 3, Eq. 1.)

**Mean Absolute Percentage Error (MAPE):**

```
MAPE_i = (1 / n_i) * sum_{t=1}^{n_i} |Y_t - Y_t^f| / Y_t
```

(Szucs 2017, Section 3, Eq. 2.)

**MAPE exclusion:** Exclude observations where Y_t = 0 from MAPE computation (division by
zero). For datasets pre-filtered for non-zero volume (as Szucs does), this is not needed.

**Aggregation:**
1. Compute MSE_i and MAPE_i per stock, averaging over all bins and all out-of-sample
   forecast days.
2. Report cross-stock average: MSE = (1/N) * sum MSE_i, same for MAPE.
(Szucs 2017, Section 3: "Both measures are calculated for each share, and also in the
average of all shares.")

**Scale-adjusted MSE (MSE*):**

```
MSE* = (1/N) * sum_{i=1}^{N} MSE_i / (a_i / a_min)^2
```

where a_i is stock i's average turnover and a_min is the smallest average turnover.
Normalizes MSE by turnover scale, since high-turnover stocks produce mechanically larger
MSE. (Szucs 2017, Section 5, Eq. 14.)

### Expected Behavior

**Per-stock volume prediction accuracy (one-step-ahead with intraday updating):**

From Szucs 2017, Table 2a, 33 DJIA stocks, 2648 out-of-sample forecast days (= 2668
total days minus 20-day initial window), k=26, L=20:

| Model | MSE | MAPE |
|-------|-----|------|
| U-method | 1.02e-03 | 0.503 |
| BDF_AR | 6.49e-04 | 0.403 |
| BDF_SETAR | 6.60e-04 | 0.399 |

(Szucs 2017, Section 5, Table 2a. These numbers use one-step-ahead intraday-updating
evaluation as described in the Evaluation Protocol section above.)

**MAPE interpretation:** A per-stock per-bin MAPE of ~0.40 means the average absolute
percentage error for individual 15-minute bin forecasts is 40%. This is typical for
intraday volume, which is inherently noisy at fine time scales. The model's value is
relative to the U-method (0.503): a ~20% reduction in MAPE.

**MSE vs MAPE discrepancy:** BDF_AR wins on MSE (6.49e-04) while BDF_SETAR wins on MAPE
(0.399). MSE is scale-sensitive and penalizes large errors more; MAPE penalizes
proportional errors equally. For VWAP execution, MAPE is more relevant since it measures
proportional forecast accuracy across all bin sizes. (Szucs 2017, Section 5, below
Table 2a.)

**Portfolio-level MAPE (BDF 2008):**

From BDF 2008, Table 2, panel 1 (CAC40, 39 stocks, k=25, L=20, 50 out-of-sample days):

| Model | Mean MAPE | Std | Q95 |
|-------|-----------|-----|-----|
| PCA-SETAR (one-step) | 0.0752 | 0.0869 | 0.2010 |
| PCA-ARMA (one-step) | 0.0829 | 0.0973 | 0.2330 |
| Classical average | 0.0905 | 0.1050 | 0.2490 |

(BDF 2008, Table 2, "Prediction of model for intraday volume" panel.)

**Portfolio vs per-stock MAPE:** The BDF portfolio MAPE (0.075) is much lower than Szucs's
per-stock MAPE (0.40) because portfolio aggregation diversifies away idiosyncratic
forecast errors. Different datasets, different aggregation, different markets -- not
directly comparable.

**VWAP execution cost (dynamic, out-of-sample):**

From BDF 2008, Table 2, panel 3:

| Model | Mean MAPE | Std | Q95 |
|-------|-----------|-----|-----|
| PCA-SETAR dynamic | 0.0898 | 0.0954 | 0.2854 |
| PCA-ARMA dynamic | 0.0922 | 0.0994 | 0.2854 |
| Classical approach | 0.1006 | 0.1171 | 0.3427 |

(BDF 2008, Table 2, "Out-of-sample estimation for VWAP execution" panel.)

Dynamic PCA-SETAR reduces VWAP tracking error by ~10% on average vs the classical
approach (0.0898 vs 0.1006). For high-error stocks, improvements can reach 50%.
(BDF 2008, Section 4.3.3: "use of our method allows for a reduction of the error by 10%".)

**Clarification on BDF's "8 bp" figure:** BDF 2008 Section 4.3.3 states tracking error
"is lower (8 bp)." This is a portfolio-level execution cost under a different aggregation
than the Table 2 MAPEs (where 0.0898 = 898 bp). For implementation validation, use the
Table 2 MAPE values. Do not expect 8 bp absolute tracking error.

### Sanity Checks

1. **Factor count r:** For 20-50 liquid stocks, Bai-Ng IC should select r between 1 and 5.
   r > 5 suggests miscalibration or unusual data structure. (Researcher inference based on
   typical factor model results.)

2. **Common component U-shape:** Average the common component across stocks, reshape to
   (L, k), average over days. The resulting (k,) vector should show a U-shape: high at
   market open and close, low at midday. (BDF 2008, Fig. 3, top panels: characteristic
   U-shape for TOTAL stock.)

3. **Specific component magnitude:** |e_{i,t}| should be much smaller than |c_{i,t}| for
   most observations. The variance ratio Var(e)/Var(x) should be substantially below 1
   for most stocks. Typical common-component variance ratio: 0.6-0.8. (BDF 2008, Fig. 3,
   bottom panels: specific values in [-0.025, 0.010] vs common in [0, 0.035].)

4. **AR(1) coefficient range:** psi_1 should be positive and strictly less than 1
   (stationarity). Typical range: 0.1-0.6. If |psi_1| >= 1, the series is non-stationary
   and should be investigated. (Researcher inference; BDF 2008, Fig. 2, PACF: significant
   first lag.)

5. **SETAR threshold tau:** Should be near the median of the specific component. If tau is
   at an extreme quantile, one regime has very few observations and estimates may be
   unstable. (Researcher inference.)

6. **Turnover non-negativity:** Combined forecasts (c_hat + e_hat) should be non-negative
   for >95% of bins. A high rate of negative forecasts indicates model failure.
   (Researcher inference: volume is non-negative by definition.)

7. **VWAP weights shape:** Weights should be non-negative, sum to 1.0, and roughly follow
   the U-shape (more weight at open and close). (Researcher inference.)

8. **U-method pairwise (MSE):** BDF should produce lower MSE than U-method for all or
   nearly all stocks. Szucs 2017, Table 2b: BDF_AR beats U-method 33/0 by MSE; BDF_SETAR
   beats U-method 33/0 by MSE.

9. **U-method pairwise (MAPE):** Szucs 2017, Table 2c: BDF_AR beats U-method 32/1 by MAPE;
   BDF_SETAR beats U-method 32/1 by MAPE. (Verified from Table 2c: BDF_SETAR row, U
   column = 32/1.)

10. **SETAR vs AR(1) pairwise:** Szucs 2017, Table 2b: BDF_SETAR beats BDF_AR on only 6/27
    stocks by MSE (AR better by MSE). Table 2c: BDF_SETAR beats BDF_AR on 30/3 stocks by
    MAPE (SETAR better by MAPE). This asymmetry is expected.

11. **Eigenvalue scree plot:** Should show a clear gap between the first r eigenvalues and
    the rest. Useful diagnostic for validating r selection. (Researcher inference.)

12. **No-centering verification:** Run the model once with centered data and once without.
    The uncentered version should produce lower forecast MAPE, confirming that centering
    degrades performance. (Researcher inference.)

13. **Computational time:** Full estimation for ~30 stocks over ~2500 days should complete in
    under 2 hours. Per-day estimation: seconds. (Szucs 2017, Section 5.)

### Edge Cases

1. **Zero-volume bins:** A stock with zero volume in a bin has turnover = 0. This is valid
   data (not missing). PCA and AR/SETAR handle zeros naturally. MAPE is undefined when
   actual = 0; exclude such bins from MAPE. (Szucs 2017, Section 2: dataset filtered so
   this does not arise.)

2. **Cross-section changes:** If a stock enters or leaves the universe (IPO, delisting, index
   reconstitution), maintain a fixed cross-section N within each estimation window. For
   entering stocks, wait until they have L days of history. For exiting stocks, exclude for
   the entire window if they delist mid-window. If N changes between windows, re-estimate
   the model from scratch -- changing N alters the PCA decomposition and makes factor
   loadings from the old N incomparable. (Researcher inference: PCA factor stability.)

3. **Single-stock application:** The model CANNOT be applied to a single stock. PCA requires
   a cross-section. Minimum practical N: 10-20 for stable factor estimation. BDF uses
   N=39; Szucs uses N=33. (Bai 2003 asymptotics require both N and T large.)

4. **Short estimation windows:** If L < 10, PCA estimation has very few time observations
   per stock (P < 250 for k=25). Factor estimates become noisy. (Researcher inference.)

5. **Non-stationary specific component:** If |psi_1| >= 1, the AR(1) specific component is
   non-stationary. Can occur during extreme volume events (earnings, M&A). Fallback: use
   the U-method for that stock on that day. (Researcher inference.)

6. **Market half-days:** Shortened trading sessions have fewer than k bins. Exclude from
   the estimation window. (BDF 2008, Section 3.1.)

7. **Overnight gaps and periodic ACF:** The series is contiguous across day boundaries, so
   bin k-1 of day d is followed by bin 0 of day d+1. This produces periodic ACF spikes
   at multiples of k (BDF 2008, Fig. 2). The AR(1)/SETAR captures only lag-1 dependence.
   Do NOT add seasonal AR terms unless extending beyond the published model.

8. **Singular/ill-conditioned matrices:** If the turnover matrix is rank-deficient (e.g.,
   two stocks have identical volume profiles, or N is very large relative to P), the SVD
   still works but some singular values will be zero or near-zero. The IC will correctly
   avoid selecting r beyond the effective rank. (Researcher inference.)

9. **Volume participation limits:** In practice, VWAP execution weights represent target
   participation rates, which may be capped (e.g., 20% of bin volume) by trading
   constraints. Unexecuted shares should carry over to subsequent bins. (Researcher
   inference: practical execution constraint.)

### Known Limitations

1. **Volume only, no price model:** The BDF model forecasts volume, not price. To "beat"
   VWAP (execute at a price better than VWAP), a bivariate volume-price model is needed.
   The current model can only track VWAP. (BDF 2008, Section 5: "in order to beat the
   VWAP, our price-adjusted-volume model is not sufficient and it is essential to derive
   a bivariate model for volume and price".)

2. **Small order assumption:** The model assumes the trader's order is small relative to
   market volume (no market impact). Large orders require impact-aware execution models.
   (BDF 2008, Section 4.1.3: "we suppose that he is able to trade without impact on
   prices".)

3. **Cross-section required:** Cannot forecast volume for a single stock in isolation.
   Requires N >= 10 liquid stocks for PCA. (BDF 2008, Section 2.2; Bai 2003.)

4. **Stationarity within window:** The model assumes the specific component is stationary
   within the L-day window. During regime changes (e.g., financial crisis), the rolling
   window may mix different regimes. (Researcher inference.)

5. **No intraday event handling:** The model does not account for scheduled intraday events
   (FOMC announcements, earnings releases, economic data releases) that can dramatically
   alter the volume profile on specific days. (Researcher inference.)

6. **Linear factor model:** PCA assumes a linear factor structure. If the common volume
   pattern varies multiplicatively with daily volume level, the additive model is
   misspecified. This is a known limitation relative to multiplicative models like BCG.
   (Szucs 2017, Section 4: "additive decomposition" vs BCG's "multiplicative
   decomposition".)

7. **Factor rotation indeterminacy:** PCA factors are identified only up to rotation. The
   individual factors and loadings are not uniquely interpretable, but the common component
   C_hat = F_hat @ Lambda_hat.T is unique. (Bai 2003; BDF 2008 does not interpret
   individual factors.)

8. **Multi-step forecast degradation:** Iterated AR/SETAR forecasts degrade rapidly with
   horizon. The dynamic execution strategy mitigates this by using one-step-ahead forecasts
   refreshed after each observed bin. (BDF 2008, Section 4.2.)

## Paper References

| Spec Section | Paper Source | Section/Equation |
|-------------|-------------|-----------------|
| Data Preprocessing: bin aggregation | BDF 2008 | Section 3.1, Fig. 1 |
| Data Preprocessing: turnover (float) | BDF 2008 | Section 2.2, below Eq. 3 |
| Data Preprocessing: turnover (TSO) | Szucs 2017 | Section 2, page 4 |
| Data Preprocessing: no centering | Bai 2003 | Section 2; BDF 2008, Eq. 4-6 |
| Data Preprocessing: exclude half-days | BDF 2008 | Section 3.1 |
| Data Preprocessing: non-zero volume | Szucs 2017 | Section 2 |
| Data Preprocessing: TSO handling | Researcher inference | Practical data handling |
| extract_and_select_factors: PCA procedure | BDF 2008 | Section 2.2, Eq. 4-6 |
| extract_and_select_factors: Bai normalization | Bai 2003 | Econometrica 71(1), pp. 135-171 |
| extract_and_select_factors: IC formula | Bai & Ng 2002 | Econometrica 70(1), pp. 191-221 |
| extract_and_select_factors: V(r) from singular values | Researcher inference | Eckart-Young theorem |
| extract_and_select_factors: SVD implementation | Researcher inference | Numerically equivalent to eigendecomposition |
| extract_and_select_factors: IC_p2 default choice | Researcher inference | Applied factor analysis convention |
| extract_and_select_factors: total_ss via Frobenius norm | Researcher inference | Standard linear algebra |
| forecast_common: historical average | BDF 2008 | Section 2.3, Eq. 9 |
| forecast_common: fixed intraday | BDF 2008 | Section 2.3, text below Eq. 9 |
| forecast_common: = U-method on C_hat | BDF 2008 | Footnote 5 |
| fit_ar1: AR(1) model | BDF 2008 | Section 2.3, Eq. 10 |
| fit_ar1: AR(1) confirmation | Szucs 2017 | Section 4.1, Eq. 5 |
| fit_ar1: ARMA(1,1) mislabel | BDF 2008 | Section 2.3, text vs Eq. 10 |
| fit_ar1: MLE = OLS | Researcher inference | Standard econometrics |
| fit_ar1: contiguous series across day boundaries | BDF 2008 | Section 2.3, Eq. 10; Fig. 2 |
| fit_ar1: contiguous series confirmation | Szucs 2017 | Section 4.1, Eq. 5 |
| fit_ar1: overnight gap ACF | BDF 2008 | Fig. 2 |
| fit_ar1: unbiased variance (n-2) | Researcher inference | Standard OLS |
| fit_ar1: sigma2 diagnostic only | Researcher inference | Not used in point forecasting |
| fit_setar: SETAR model | BDF 2008 | Section 2.3, Eq. 11 |
| fit_setar: SETAR confirmation | Szucs 2017 | Section 4.1, Eq. 6-7 |
| fit_setar: grid search estimation | Researcher inference | Hansen 1997 |
| fit_setar: single-pass estimation | Researcher inference | Efficiency optimization |
| fit_setar: per-regime variance (diagnostic) | Researcher inference | Standard OLS |
| fit_setar: fallback to AR(1) | Researcher inference | BDF Section 3.2 uses SETAR universally |
| fit_setar: min regime size | Researcher inference | Standard SETAR practice |
| forecast_turnover: combined forecast | BDF 2008 | Section 2.3, Eq. 8 |
| compute_vwap_weights: negative clipping | Researcher inference | Volume non-negativity |
| compute_vwap_weights: C_hat can be negative | Researcher inference | PCA non-negativity not guaranteed |
| compute_vwap_weights: equal-weight fallback | Researcher inference | Model failure safeguard |
| execute_one_bin: event-driven dynamic strategy | BDF 2008 | Section 4.2.2, page 1717 |
| execute_one_bin: static dominated | BDF 2008 | Section 4.2, page 1717 |
| execute_one_bin: initialization | BDF 2008 | Section 2.3 |
| execute_one_bin: loop order | Researcher inference | Execution timing |
| daily_update: rolling re-estimation | BDF 2008 | Section 3.2 |
| daily_update: rolling window | Szucs 2017 | Section 3, Figure 1 |
| u_method_benchmark | Szucs 2017 | Section 4, Eq. 3 |
| u_method_benchmark | BDF 2008 | Footnote 5 |
| Evaluation Protocol: one-step-ahead updating | Szucs 2017 | Section 3, page 5 |
| Evaluation Protocol: first-bin initialization | BDF 2008 | Section 2.3 |
| Validation: MSE formula | Szucs 2017 | Section 3, Eq. 1 |
| Validation: MAPE formula | Szucs 2017 | Section 3, Eq. 2 |
| Validation: MSE* formula | Szucs 2017 | Section 5, Eq. 14 |
| Validation: per-stock MSE/MAPE | Szucs 2017 | Section 5, Table 2a |
| Validation: pairwise MSE | Szucs 2017 | Section 5, Table 2b |
| Validation: pairwise MAPE | Szucs 2017 | Section 5, Table 2c |
| Validation: portfolio MAPE | BDF 2008 | Table 2, panel 1 |
| Validation: VWAP execution cost | BDF 2008 | Table 2, panel 3 |
| Validation: SETAR wins 36/39 | BDF 2008 | Section 3.2 |
| Validation: ~10% TE reduction | BDF 2008 | Section 4.3.3 |
| Validation: "8 bp" clarification | BDF 2008 | Section 4.3.3 |
| Limitation: no price model | BDF 2008 | Section 5 |
| Limitation: small order | BDF 2008 | Section 4.1.3 |
| Limitation: additive vs multiplicative | Szucs 2017 | Section 4 |
| Edge case: partial days | BDF 2008 | Section 3.1 |
| Edge case: zero-volume filtering | Szucs 2017 | Section 2 |
| Edge case: fixed cross-section | Researcher inference | PCA factor stability |
| Computational cost | Szucs 2017 | Section 5, last paragraph |
