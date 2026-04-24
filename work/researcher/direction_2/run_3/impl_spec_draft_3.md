# Implementation Specification: PCA Factor Decomposition for Intraday Volume (BDF Model)

## Overview

The BDF model decomposes intraday stock turnover into a market-wide common component
(the shared U-shaped seasonal pattern) and a stock-specific idiosyncratic component,
using principal components analysis (PCA) on a cross-section of stocks. The common
component is forecast by a simple historical average at the same time-of-day. The
specific component is modeled by either an AR(1) or a two-regime Self-Exciting Threshold
Autoregressive (SETAR) process. The combined forecast drives a dynamic VWAP execution
strategy that updates intraday as actual volumes arrive.

The model was introduced by Bialkowski, Darolles, and Le Fol (2008) and independently
validated by Szucs (2017) on US equity data, confirming superior accuracy and
computational efficiency versus alternatives.

## Algorithm

### Model Description

**What it does:** Given a cross-section of N stocks observed at k intraday intervals per
day, the model produces one-step-ahead volume forecasts for each stock at each interval.
These forecasts can be used to construct a VWAP execution schedule that minimizes
tracking error against the end-of-day VWAP benchmark.

**Key assumption:** Intraday volume patterns are driven by a small number of latent
market-wide factors (the U-shape and its variants) that are common across all stocks.
After removing this common pattern, each stock's residual volume dynamics are simple
enough to be captured by a low-order autoregressive model.

**Inputs:**
- Intraday turnover matrix: N stocks x P intraday observations (rolling window of L
  trading days, each with k bins, so P = L * k).
- For dynamic execution: realized volumes for bins already observed on the current day.

**Outputs:**
- One-step-ahead turnover forecast for each stock at each future intraday bin.
- VWAP execution weights: fraction of total order to execute in each remaining bin.

### Data Preprocessing

**Source:** BDF 2008, Section 3.1; Szucs 2017, Section 2.

1. Obtain raw trade data (tick or minute-level) for N stocks over the estimation window.
2. Aggregate into fixed-width intraday bins:
   - BDF uses 20-minute bins, k=25, covering 9:20-17:20 Paris time (BDF 2008, Section 3.1).
   - Szucs uses 15-minute bins, k=26, covering 9:30-16:00 US market hours (Szucs 2017, Section 2).
   - Implementation should parameterize k and bin width.
3. Compute turnover for each bin:
   - BDF definition: turnover = shares_traded / float_shares (BDF 2008, Section 2.2, text below Eq. 3).
   - Szucs definition: turnover = volume / total_shares_outstanding (Szucs 2017, Section 2, page 4).
   - Both are valid; TSO is more readily available from data providers.
   - Adjust for stock splits and dividends (BDF 2008, Section 3.1).
   - **TSO data handling (Researcher inference: practical data handling):** Use the most
     recent available TSO value for each trading day. If TSO changes intraday (e.g.,
     stock split takes effect at open), use the post-event TSO for all bins on that day.
     If TSO is missing for a date, carry forward the last known value. Flag any stock-day
     where TSO changes by more than 10% from the previous day for manual review.
4. Exclude partial trading days (e.g., half-days before holidays). BDF excludes Dec 24 and Dec 31, 2003 (BDF 2008, Section 3.1).
5. Verify all stocks have non-zero volume in every bin (Szucs 2017, Section 2: "every stock had trades and thus a volume record larger than zero in every 15-minute interval").

**Critical note on centering:** Do NOT subtract the cross-sectional or time-series mean
from the turnover matrix before PCA. The Bai (2003) factor model operates on raw
(uncentered) data. The common component is meant to capture both the level and shape
of the market-wide volume pattern. Centering would strip out level information and
produce a biased decomposition. (Researcher inference: BDF 2008 does not explicitly
address centering, but the model equation x_{i,t} = lambda_i' * F_t + e_{i,t} has no
intercept term, and Bai (2003) Section 2 defines the model without demeaning.)

### Pseudocode

#### Function 1: estimate_model

Estimates the full BDF model on a rolling window of historical data for one estimation
day. This is the top-level daily estimation routine.

**Source:** BDF 2008, Sections 2.2-2.3; Szucs 2017, Section 4.1.

```
function estimate_model(turnover_matrix, k, L, model_type):
    """
    Parameters
    ----------
    turnover_matrix: array of shape (P, N)
        P = L * k intraday observations (rows are time, columns are stocks).
        Row ordering: day 1 bin 1, day 1 bin 2, ..., day 1 bin k, day 2 bin 1, ...
        Values are raw turnover (shares traded / shares outstanding). NOT centered.
    k: int
        Number of intraday bins per trading day.
    L: int
        Number of trading days in the estimation window.
    model_type: string
        "AR1" or "SETAR"

    Returns
    -------
    model: dict containing
        - factors: array of shape (P, r) -- estimated common factors F_hat
        - loadings: array of shape (N, r) -- estimated factor loadings Lambda_hat
        - common: array of shape (P, N) -- estimated common component C_hat = F_hat @ Lambda_hat.T
        - specific: array of shape (P, N) -- estimated specific component e_hat = turnover_matrix - common
        - r: int -- number of factors selected
        - ts_params: list of N dicts -- AR(1) or SETAR parameters per stock
        - common_forecast: array of shape (k, N) -- next-day common component forecast
    """

    # Step 1: Select number of factors
    r = select_num_factors(turnover_matrix, k, L)

    # Step 2: Extract common and specific components via PCA
    factors, loadings, common, specific = extract_factors(turnover_matrix, r)

    # [Implementation optimization per critique round 2, m1]:
    # Steps 1 and 2 can share a single truncated SVD with r_max components.
    # Compute U, s, Vt = truncated_svd(X, n_components=r_max) once. Then:
    #   - In select_num_factors: use V(r) = (1/(N*P)) * (total_ss - sum(s[:r]^2))
    #     for each r, avoiding eigendecomposition entirely. The P<=N vs P>N branching
    #     is also unnecessary since SVD is symmetric in P and N.
    #   - In extract_factors: take the top r columns from U and Vt.
    # This halves the cost of model estimation. The modular function signatures above
    # are kept for clarity; in implementation, a combined _estimate_factors(X, r_max)
    # function that returns (r, F_hat, Lambda_hat, C_hat, e_hat) is recommended.

    # Step 3: Forecast common component for next day
    common_forecast = forecast_common(common, k, L)

    # Step 4: Fit time-series model to each stock's specific component
    ts_params = []
    for i in range(N):
        if model_type == "AR1":
            params = fit_ar1(specific[:, i])
        elif model_type == "SETAR":
            params = fit_setar(specific[:, i])
        ts_params.append(params)

    return {
        "factors": factors,
        "loadings": loadings,
        "common": common,
        "specific": specific,
        "r": r,
        "ts_params": ts_params,
        "common_forecast": common_forecast,
    }
```

#### Function 2: select_num_factors

Determines the number of latent factors r using the Bai and Ng (2002) information
criterion. Uses a single eigendecomposition for efficiency.

**Source:** BDF 2008, Section 2.2, reference to Bai & Ng (2002); Bai & Ng (2002),
Econometrica 70(1), pp. 191-221.

```
function select_num_factors(turnover_matrix, k, L, r_max=10):
    """
    Parameters
    ----------
    turnover_matrix: array of shape (P, N), P = L * k
    k: int -- bins per day
    L: int -- days in window
    r_max: int -- maximum number of factors to consider (default 10)

    Returns
    -------
    r: int -- selected number of factors (1 <= r <= r_max)
    """
    P, N = turnover_matrix.shape
    X = turnover_matrix

    # Perform ONE eigendecomposition for all r_max eigenvectors.
    # [Revised per critique M2: avoid redundant eigendecompositions]
    #
    # Choose the smaller matrix for efficiency:
    if P <= N:
        M = (X @ X.T) / N       # (P, P)
        eigenvalues, eigenvectors = eig_descending(M, num=r_max)
        # eigenvalues[j] = mu_j, the j-th largest eigenvalue of XX'/N
        # F_hat_full = sqrt(P) * eigenvectors  (P, r_max)
        # Lambda_hat_full = X.T @ F_hat_full / P  (N, r_max)
    else:
        M = (X.T @ X) / P       # (N, N)
        eigenvalues, eigenvectors = eig_descending(M, num=r_max)

    # Total sum of squares (constant, independent of r)
    total_ss = sum(X^2)

    # For each candidate r, compute V(r) from the rank-r approximation.
    #
    # V(r) = (1/(N*P)) * ||X - C_hat_r||_F^2
    #
    # Key relationship [revised per critique M4]:
    # If mu_j are eigenvalues of XX'/N (P <= N case), then:
    #     ||C_hat_r||_F^2 = sum_{j=1}^{r} mu_j * P
    #     (because C_hat_r = F_hat_r @ Lambda_hat_r.T where F_hat_r has columns
    #      sqrt(P)*v_j, and Lambda_hat_r = X.T @ F_hat_r / P, so
    #      ||C_hat_r||_F^2 = tr(F_r F_r.T X X.T F_r F_r.T ... ) -- more directly,
    #      the rank-r SVD approximation captures sum of top r singular values squared,
    #      which equals sum of top r eigenvalues of XX' = N * sum of top r eigenvalues of XX'/N,
    #      so ||C_hat_r||_F^2 = N * sum_{j=1}^{r} mu_j  where mu_j = eigenvalues of XX'/N)
    #
    # Therefore: V(r) = (1/(N*P)) * (total_ss - N * sum_{j=1}^{r} mu_j)
    #
    # If mu_j are eigenvalues of X'X/P (P > N case), then:
    #     V(r) = (1/(N*P)) * (total_ss - P * sum_{j=1}^{r} mu_j)
    #
    # Note: eigenvalues of XX'/N and X'X/P share the same nonzero values (up to
    # multiplicity), but their magnitudes differ by a factor of P/N. The formulas
    # above are correct for the respective matrix used.

    best_IC = infinity
    best_r = 1

    for r in range(1, r_max + 1):
        if P <= N:
            V_r = (1.0 / (N * P)) * (total_ss - N * sum(eigenvalues[:r]))
        else:
            V_r = (1.0 / (N * P)) * (total_ss - P * sum(eigenvalues[:r]))

        # [Guard per critique round 2, m5]: Protect against V_r <= 0 from
        # floating-point arithmetic when the top r components capture nearly all
        # variance (e.g., synthetic low-rank data, or r_max set too high).
        if V_r <= 0:
            V_r = 1e-15  # machine-epsilon floor; this r likely overfits

        # Bai-Ng IC_p2 penalty (most commonly used):
        # penalty(r) = r * ((N + P) / (N * P)) * ln(min(N, P))
        penalty_r = r * ((N + P) / (N * P)) * ln(min(N, P))

        IC_r = ln(V_r) + penalty_r

        if IC_r < best_IC:
            best_IC = IC_r
            best_r = r

    return best_r
```

**Note on IC variant:** Bai & Ng (2002) define six information criteria (IC_p1, IC_p2,
IC_p3, PC_p1, PC_p2, PC_p3). BDF 2008 does not specify which one they use. IC_p2 is
the most commonly used in empirical work and has good finite-sample properties for
selecting the true number of factors. The penalty term is
r * ((N+P)/(N*P)) * ln(min(N,P)). (Researcher inference: IC_p2 chosen as default based
on prevalence in applied factor analysis literature.)

**Alternative IC_p1 penalty:** r * ((N+P)/(N*P)) * ln((N*P)/(N+P)). If results are
sensitive to the IC variant, both should be tested.

**Note on V(r) from eigenvalues vs. from residuals:** The eigenvalue-based formula
above is algebraically equivalent to V(r) = (1/(N*P)) * ||X - C_hat_r||_F^2 but avoids
reconstructing the full (P, N) residual matrix for each r. If numerical precision is a
concern, the developer can verify by computing V(r) directly from the residual matrix
for a few values of r and confirming they match the eigenvalue formula.
(Researcher inference: standard numerical optimization.)

#### Function 3: extract_factors

Extracts latent factors and loadings via SVD of the turnover matrix. This unified
approach avoids the P <= N vs P > N branching and normalization pitfalls.

**Source:** BDF 2008, Section 2.2, Equations 4-6; Bai (2003), Econometrica 71(1),
pp. 135-171.

```
function extract_factors(turnover_matrix, r):
    """
    Parameters
    ----------
    turnover_matrix: array of shape (P, N). Raw turnover, NOT centered.
    r: int -- number of factors to extract

    Returns
    -------
    factors: array of shape (P, r) -- estimated factors F_hat (normalized: F_hat.T @ F_hat / P = I_r)
    loadings: array of shape (N, r) -- estimated factor loadings Lambda_hat
    common: array of shape (P, N) -- common component C_hat = F_hat @ Lambda_hat.T
    specific: array of shape (P, N) -- specific component e_hat = turnover_matrix - C_hat
    """
    P, N = turnover_matrix.shape
    X = turnover_matrix  # (P, N), NOT centered

    # [Revised per critique M1: use SVD directly to avoid normalization errors
    #  in the dual-case eigendecomposition approach.]
    #
    # The SVD of X is: X = U @ diag(s) @ V.T
    # where U is (P, min(P,N)), s contains singular values in descending order,
    # and V is (N, min(P,N)).
    #
    # The rank-r approximation is: C_hat_r = U[:,:r] @ diag(s[:r]) @ V[:,:r].T
    #
    # To satisfy the Bai (2003) normalization F'F/P = I_r:
    #   F_hat = sqrt(P) * U[:, :r]                  # (P, r), orthonormal columns scaled by sqrt(P)
    #   Lambda_hat = V[:, :r] @ diag(s[:r]) / sqrt(P)  # (N, r)
    #
    # Verification: F_hat.T @ F_hat / P = P * U.T @ U / P = I_r  (correct)
    # And: F_hat @ Lambda_hat.T = sqrt(P) * U @ diag(s) @ V.T / sqrt(P)
    #                           = U @ diag(s) @ V.T = C_hat_r  (correct)
    #
    # Implementation: use truncated SVD (scipy.sparse.linalg.svds or
    # sklearn.utils.extmath.randomized_svd) to compute only the top r singular
    # triplets. This is efficient for both P <= N and P > N cases.
    # Internally, most truncated SVD implementations choose the smaller matrix
    # (XX' or X'X) automatically.

    U, s, Vt = truncated_svd(X, n_components=r)
    # U: (P, r), s: (r,), Vt: (r, N)

    F_hat = sqrt(P) * U                          # (P, r)
    Lambda_hat = (Vt.T * s) / sqrt(P)            # (N, r) -- equivalent to V @ diag(s) / sqrt(P)

    C_hat = F_hat @ Lambda_hat.T    # (P, N) common component
    e_hat = X - C_hat               # (P, N) specific component

    return F_hat, Lambda_hat, C_hat, e_hat
```

**Implementation note:** Use `scipy.sparse.linalg.svds(X, k=r)` or
`sklearn.utils.extmath.randomized_svd(X, n_components=r)`. Both compute only the top r
singular triplets efficiently without forming the full SVD. The matrix X is dense, but
truncated SVD is still faster than full eigendecomposition when r << min(P, N).
(Researcher inference: standard numerical optimization, not specified in the papers.)

**Normalization convention:** The Bai (2003) normalization is F_hat.T @ F_hat / P = I_r.
This means F_hat columns are orthogonal with norm sqrt(P). The common component
C_hat = F_hat @ Lambda_hat.T is invariant to the normalization choice (rotating F and
Lambda inversely leaves C unchanged). The SVD-based approach guarantees the correct
normalization without manual adjustment.
(BDF 2008, Section 2.2, text after Eq. 6.)

#### Function 4: forecast_common

Forecasts the common component for the next trading day by averaging the common
component at the same time-of-day across the L prior days.

**Source:** BDF 2008, Section 2.3, Equation 9.

```
function forecast_common(common, k, L):
    """
    Parameters
    ----------
    common: array of shape (P, N), P = L * k
        Estimated common component from extract_factors.
        Row layout: [day_1_bin_1, ..., day_1_bin_k, day_2_bin_1, ..., day_L_bin_k]
    k: int -- bins per day
    L: int -- number of days in window

    Returns
    -------
    common_forecast: array of shape (k, N)
        Forecast common component for next day, one row per intraday bin.
    """
    # Reshape common into (L, k, N)
    common_3d = reshape(common, (L, k, N))

    # Average across L days for each bin position j = 1..k
    # BDF 2008 Eq. 9: c_hat_{i,t+1} = (1/L) * sum_{l=1}^{L} c_{i,t+1 - k*l}
    # This averages the same intraday bin across all L days in the window.
    common_forecast = mean(common_3d, axis=0)  # (k, N)

    return common_forecast
```

**Important:** The common component forecast is computed BEFORE the trading day begins
(the "night before"), using factor loadings and factors estimated on the most recent
L-day window. There is no look-ahead bias because only past data is used. (BDF 2008,
Section 2.3, paragraph below Eq. 9; also Implementation Notes in the summary.)

**Note on intraday updates:** The common component forecast is fixed for the entire
trading day. It is NOT updated intraday as actual volumes arrive. Only the specific
component updates intraday via observed actuals. This is by design -- the common
component captures the stable seasonal shape, which does not change within a day.
(BDF 2008, Section 2.3, paragraph below Eq. 9: the common forecast uses the historical
average, computed once before the day begins.)

#### Function 5: fit_ar1

Fits an AR(1) model with constant to the specific component of one stock.

**Source:** BDF 2008, Section 2.3, Equation 10; Szucs 2017, Section 4.1, Equation 5.

```
function fit_ar1(e_series):
    """
    Parameters
    ----------
    e_series: array of shape (P,)
        Specific component time series for one stock over the estimation window.
        This is a single contiguous series (day boundaries are NOT treated specially).

    Returns
    -------
    params: dict
        - psi_1: float -- AR(1) coefficient
        - psi_2: float -- constant (intercept)
        - sigma2: float -- innovation variance
    """
    # Model: e_t = psi_1 * e_{t-1} + psi_2 + epsilon_t
    # where epsilon_t ~ N(0, sigma2)
    #
    # BDF 2008 Eq. 10: e_{i,t} = psi_1 * e_{i,t-1} + psi_2 + epsilon_{i,t}
    # Szucs 2017 Eq. 5: e_p = c + theta_1 * e_{p-1} + epsilon_p
    # (same model, different notation)

    # Estimate by OLS:
    n = len(e_series) - 1          # number of regression observations
    y = e_series[1:]               # dependent: e_t for t = 2..P
    X = [e_series[:-1], ones(n)]   # regressors: [e_{t-1}, 1]

    beta = (X.T @ X)^{-1} @ X.T @ y
    psi_1 = beta[0]
    psi_2 = beta[1]

    residuals = y - X @ beta
    # [Revised per critique m2: use unbiased variance estimator]
    # 2 parameters estimated (psi_1, psi_2), so denominator is n - 2.
    sigma2 = sum(residuals^2) / (n - 2)

    return {"psi_1": psi_1, "psi_2": psi_2, "sigma2": sigma2}
```

**Note on estimation method:** BDF 2008 states "maximum likelihood" for both AR and
SETAR (Section 2.3, paragraph below Eq. 11). For the AR(1) with Gaussian innovations,
MLE is equivalent to OLS. OLS is simpler and numerically stable. (Researcher inference.)

**Note on ARMA(1,1) vs AR(1):** BDF 2008, Eq. 10 writes e_{i,t} = psi_1 * e_{i,t-1} +
psi_2 + epsilon_{i,t}. Despite BDF's text referring to "ARMA(1,1)", the equation itself
has no MA term -- it is an AR(1) with intercept. Szucs 2017 explicitly calls it AR(1)
(Section 4.1, Eq. 5). We implement AR(1) as written in the equations.

**Note on overnight gaps:** The AR(1) series treats the intraday time series as
contiguous across day boundaries: bin k of day d is followed by bin 1 of day d+1. This
introduces periodic autocorrelation at lag k (visible in BDF 2008, Fig. 2: ACF of the
specific component shows periodic spikes at multiples of 25). BDF and Szucs do NOT
address this periodic structure -- the simple AR(1)/SETAR captures only the lag-1
dependence. Do not add seasonal AR terms unless extending the model beyond what the
papers describe. (Researcher inference: BDF Fig. 2 confirms the periodic pattern; the
lag-1 model is the deliberate design choice.)

#### Function 6: fit_setar

Fits a two-regime Self-Exciting Threshold Autoregressive model to the specific component
of one stock.

**Source:** BDF 2008, Section 2.3, Equation 11; Szucs 2017, Section 4.1, Equation 6.

```
function fit_setar(e_series, n_grid=100):
    """
    Parameters
    ----------
    e_series: array of shape (P,)
        Specific component time series for one stock.
    n_grid: int
        Number of grid points for threshold search (default 100).

    Returns
    -------
    params: dict
        - phi_11: float -- AR coefficient, regime 1 (e_{t-1} <= tau)
        - phi_12: float -- intercept, regime 1
        - phi_21: float -- AR coefficient, regime 2 (e_{t-1} > tau)
        - phi_22: float -- intercept, regime 2
        - tau: float -- threshold value
        - sigma2_1: float -- innovation variance, regime 1
        - sigma2_2: float -- innovation variance, regime 2
    """
    # Model: e_t = (phi_11 * e_{t-1} + phi_12) * I(e_{t-1} <= tau)
    #            + (phi_21 * e_{t-1} + phi_22) * (1 - I(e_{t-1} <= tau))
    #            + epsilon_t
    #
    # BDF 2008, Eq. 11; Szucs 2017, Eq. 6.
    #
    # [Revised per critique m3: regime labels are neutral ("regime 1" and "regime 2")
    #  rather than interpretive ("calm"/"turbulent"), since the interpretation depends
    #  on the sign of tau relative to the specific component values.]

    y = e_series[1:]           # e_t for t = 2..P
    x = e_series[:-1]          # e_{t-1} for t = 2..P
    n = len(y)

    # Step 1: Grid search over tau
    # Search over quantiles of x to avoid extreme regimes with too few observations.
    # Exclude bottom 15% and top 15% to ensure each regime has adequate data.
    tau_candidates = quantiles(x, linspace(0.15, 0.85, n_grid))

    best_ssr = infinity
    best_tau = None
    best_params = None

    for tau in tau_candidates:
        # Partition observations into two regimes
        regime1 = (x <= tau)   # regime 1: below or at threshold
        regime2 = (x > tau)    # regime 2: above threshold

        # Minimum observations per regime: at least 10% of sample or 20 obs
        n1 = sum(regime1)
        n2 = sum(regime2)
        if n1 < max(0.10 * n, 20):
            continue
        if n2 < max(0.10 * n, 20):
            continue

        # Fit OLS in each regime separately (conditional OLS)
        # Regime 1: y_1 = phi_11 * x_1 + phi_12 + eps_1
        y1, x1 = y[regime1], x[regime1]
        X1 = [x1, ones_like(x1)]
        beta1 = OLS(y1, X1)
        resid1 = y1 - X1 @ beta1

        # Regime 2: y_2 = phi_21 * x_2 + phi_22 + eps_2
        y2, x2 = y[regime2], x[regime2]
        X2 = [x2, ones_like(x2)]
        beta2 = OLS(y2, X2)
        resid2 = y2 - X2 @ beta2

        # Total sum of squared residuals
        ssr = sum(resid1^2) + sum(resid2^2)

        if ssr < best_ssr:
            best_ssr = ssr
            best_tau = tau

    # [Restructured per critique round 2, m3]: Defer variance computation to
    # after the grid search. Only the winning tau needs sigma2 estimates.
    # Phase 2: Final estimation for best_tau
    if best_tau is None:
        return None  # no valid threshold found; caller falls back to AR(1)

    regime1 = (x <= best_tau)
    regime2 = (x > best_tau)

    y1, x1 = y[regime1], x[regime1]
    X1 = [x1, ones_like(x1)]
    beta1 = OLS(y1, X1)
    resid1 = y1 - X1 @ beta1
    n1 = sum(regime1)

    y2, x2 = y[regime2], x[regime2]
    X2 = [x2, ones_like(x2)]
    beta2 = OLS(y2, X2)
    resid2 = y2 - X2 @ beta2
    n2 = sum(regime2)

    best_params = {
        "phi_11": beta1[0], "phi_12": beta1[1],
        "phi_21": beta2[0], "phi_22": beta2[1],
        "tau": best_tau,
        # [Revised per critique m2: unbiased variance, 2 params per regime]
        "sigma2_1": sum(resid1^2) / (n1 - 2),
        "sigma2_2": sum(resid2^2) / (n2 - 2),
    }

    return best_params
```

**Estimation details:** BDF 2008 states estimation by "maximum likelihood" (Section 2.3).
For SETAR with Gaussian innovations, MLE conditional on tau is equivalent to
regime-specific OLS. The threshold tau is found by grid search minimizing the total
residual sum of squares. This is the standard approach for SETAR estimation (Hansen
1997, Econometrica). (Researcher inference: grid search + conditional OLS is the standard
SETAR estimation method; BDF does not detail their procedure beyond "MLE".)

**Minimum regime size:** The 15%-85% quantile range and the minimum 10%/20-obs constraint
are standard safeguards to prevent degenerate regimes. (Researcher inference: not
specified in BDF 2008 or Szucs 2017.)

**Fallback to AR(1):** If the SETAR estimation fails (e.g., no valid threshold found
because all tau candidates produce regimes smaller than the minimum size), fall back to
AR(1). When SETAR estimation succeeds, always use the SETAR model -- do not apply an
improvement test. This is consistent with BDF's approach where SETAR is the default
model and is used for all stocks where estimation converges. (Researcher inference:
BDF 2008 Section 3.2 reports SETAR results for all 39 stocks without mentioning any
AR(1) fallback criterion. The simplest and most faithful approach is to use SETAR
whenever estimation succeeds.)

#### Function 7: forecast_next_bin

Produces a one-step-ahead turnover forecast for the next intraday bin, combining the
common component forecast with the specific component forecast.

**Source:** BDF 2008, Section 2.3, Equation 8; Szucs 2017, Section 3, "one-step-ahead forecasting".

```
function forecast_next_bin(model, stock_idx, bin_idx, last_specific):
    """
    Parameters
    ----------
    model: dict -- output of estimate_model
    stock_idx: int -- index of the stock (0..N-1)
    bin_idx: int -- index of the next bin to forecast (0..k-1)
    last_specific: float -- most recent observed specific component value e_{t-1}

    Returns
    -------
    x_hat: float -- forecast turnover for stock stock_idx at bin bin_idx
    e_hat: float -- forecast specific component (for chaining forecasts)
    """
    # Common component forecast for this bin (pre-computed, fixed for the day)
    c_hat = model["common_forecast"][bin_idx, stock_idx]

    # Specific component forecast
    params = model["ts_params"][stock_idx]

    if "tau" in params:
        # SETAR forecast
        # BDF 2008 Eq. 11
        if last_specific <= params["tau"]:
            e_hat = params["phi_11"] * last_specific + params["phi_12"]
        else:
            e_hat = params["phi_21"] * last_specific + params["phi_22"]
    else:
        # AR(1) forecast
        # BDF 2008 Eq. 10
        e_hat = params["psi_1"] * last_specific + params["psi_2"]

    # Combined forecast
    # BDF 2008 Eq. 8: x_{i,t+1} = c_hat_{i,t+1} + e_hat_{i,t+1}
    x_hat = c_hat + e_hat

    return x_hat, e_hat
```

#### Function 8: compute_vwap_weights

Converts turnover forecasts for remaining bins into VWAP execution weights.

**Source:** BDF 2008, Section 4.2, text after Eq. (section 4.2.2 for dynamic VWAP).

```
function compute_vwap_weights(forecasts_remaining):
    """
    Parameters
    ----------
    forecasts_remaining: array of shape (k_remaining,)
        Forecast turnover for each remaining bin of the trading day for one stock.

    Returns
    -------
    weights: array of shape (k_remaining,)
        Fraction of remaining order to execute in each bin. Sums to 1.0.
    """
    # Handle negative forecasts: floor at zero
    # Turnover is non-negative by definition; negative forecasts are artifacts
    # of the additive model and should be clipped.
    forecasts_clipped = maximum(forecasts_remaining, 0.0)

    total = sum(forecasts_clipped)

    if total <= 0:
        # Fallback: equal weight across remaining bins
        weights = ones(k_remaining) / k_remaining
    else:
        weights = forecasts_clipped / total

    return weights
```

**Note on negative forecasts:** The additive decomposition can produce negative turnover
forecasts (common + specific < 0), especially when the specific component is large and
negative. BDF 2008 does not explicitly address this. Clipping at zero is the natural
remedy since volume cannot be negative. If more than 50% of remaining forecasts are
negative, this indicates a model failure and the system should fall back to equal
weighting or the U-method benchmark. (Researcher inference: practical handling of
model artifacts.)

#### Function 9: run_dynamic_execution

Implements the dynamic VWAP execution strategy for one stock on one trading day.
This is the main real-time loop.

**Source:** BDF 2008, Section 4.2.2 (Dynamic VWAP execution).

```
function run_dynamic_execution(model, stock_idx, order_shares, observed_volumes):
    """
    Parameters
    ----------
    model: dict -- output of estimate_model (estimated the night before)
    stock_idx: int -- which stock to trade
    order_shares: float -- total shares to execute over the day
    observed_volumes: list -- grows as the day progresses; initially empty

    Returns
    -------
    execution_schedule: list of (bin_idx, shares_to_trade) tuples
    """
    k = len(model["common_forecast"])  # bins per day
    shares_remaining = order_shares
    schedule = []

    # Initialize last_specific from the estimation window's last observation.
    # [Revised per critique M5: explicit bootstrapping confirmation]
    # This is the in-sample residual from the last bin of the last day in the
    # estimation window (day L, bin k). Using this in-sample value as the initial
    # e_{t-1} for forecasting is correct and intended: it provides the most recent
    # observation of the specific component to seed the AR(1)/SETAR forecast chain.
    # Day L's observed data contributes to BOTH the PCA estimation AND this
    # initialization -- this is by design, not a look-ahead issue, because all
    # estimation data precedes the forecast day.
    # (BDF 2008, Section 2.3: the forecast for day L+1 is conditioned on information
    # available at the end of day L.)
    last_specific = model["specific"][-1, stock_idx]

    for j in range(k):
        if shares_remaining <= 0:
            break

        # Forecast all remaining bins from j to k-1
        forecasts_remaining = []
        e_chain = last_specific
        for future_j in range(j, k):
            x_hat, e_chain_next = forecast_next_bin(model, stock_idx, future_j, e_chain)
            forecasts_remaining.append(x_hat)
            e_chain = e_chain_next

        # Compute weights for remaining bins
        weights = compute_vwap_weights(array(forecasts_remaining))

        # Trade the first weight's fraction of remaining shares
        shares_this_bin = weights[0] * shares_remaining
        schedule.append((j, shares_this_bin))

        # After bin j completes, observe actual volume and update state
        if j < len(observed_volumes):
            actual_turnover = observed_volumes[j]
            # Compute actual specific component:
            # e_actual = actual_turnover - common_forecast[j]
            # Note: the common forecast is NOT updated -- it remains fixed for the day.
            actual_specific = actual_turnover - model["common_forecast"][j, stock_idx]
            last_specific = actual_specific
            shares_remaining -= shares_this_bin

    return schedule
```

**Key insight about dynamic vs static execution:** BDF 2008, Section 4.2, explicitly
demonstrates that static execution (predicting all k bins at market open with no
intraday updates) performs WORSE than the naive classical approach. This is because
multi-step AR/SETAR forecasts decay exponentially toward the unconditional mean (zero
for centered specific component), effectively neutralizing the dynamic model's
contribution. Dynamic execution -- re-forecasting after each observed bin -- is essential.
(BDF 2008, Section 4.2.2, penultimate paragraph of page 1717.)

#### Function 10: daily_update

Top-level daily workflow: re-estimate the model on a rolling window and prepare for
the next trading day.

**Source:** BDF 2008, Section 3.2 (estimation results methodology); Szucs 2017, Section 3,
Figure 1.

```
function daily_update(all_turnover_data, day_index, k, L, model_type):
    """
    Parameters
    ----------
    all_turnover_data: array of shape (total_days * k, N)
        Full historical turnover dataset.
    day_index: int -- current day (0-indexed); we forecast for day_index + 1.
    k: int -- bins per day
    L: int -- estimation window length in days
    model_type: string -- "AR1" or "SETAR"

    Returns
    -------
    model: dict -- estimated model for tomorrow's trading
    """
    # Extract rolling window: most recent L days ending at day_index
    start_row = (day_index - L + 1) * k
    end_row = (day_index + 1) * k
    turnover_window = all_turnover_data[start_row:end_row, :]  # (L*k, N)

    # Estimate model
    model = estimate_model(turnover_window, k, L, model_type)

    return model
```

#### Function 11: u_method_benchmark

Implements the U-method (simple historical average) benchmark for comparison.

**Source:** Szucs 2017, Section 4, Equation 3; BDF 2008, Section 2.3, Equation 9 (when
common is replaced by raw turnover).

```
function u_method_benchmark(all_turnover_data, day_index, stock_idx, k, L):
    """
    Parameters
    ----------
    all_turnover_data: array of shape (total_days * k, N)
    day_index: int -- current day
    stock_idx: int -- stock index
    k: int -- bins per day
    L: int -- averaging window

    Returns
    -------
    forecast: array of shape (k,) -- forecast turnover per bin for next day
    """
    # Average raw turnover at each bin position across L prior days
    # Szucs 2017 Eq. 3: y_hat_{p+1} = (1/L) * sum_{l=1}^{L} y_{p+1 - m*l}
    forecast = zeros(k)
    for j in range(k):
        values = []
        for l in range(1, L + 1):
            row = (day_index - l + 1) * k + j
            values.append(all_turnover_data[row, stock_idx])
        forecast[j] = mean(values)

    return forecast
```

### Data Flow

```
Raw tick/minute data
    |
    v
[Aggregate to k-bin turnover per day, compute turnover = volume / shares_outstanding]
    |
    v
Turnover matrix X: shape (P, N) where P = L * k
    |  (NOT centered / demeaned)
    v
[select_num_factors] --> r (number of factors)
    |  (single eigendecomposition for r_max, then IC computed from eigenvalues)
    |
    v
[extract_factors(X, r)]
    |  (truncated SVD, top r singular triplets)
    |
    +---> F_hat: (P, r) factors [F'F/P = I_r]
    +---> Lambda_hat: (N, r) loadings
    +---> C_hat = F_hat @ Lambda_hat.T: (P, N) common component
    +---> e_hat = X - C_hat: (P, N) specific component
    |
    +--> [forecast_common(C_hat, k, L)] --> c_hat_next: (k, N)
    |        (fixed for the entire trading day, NOT updated intraday)
    |
    +--> [fit_ar1(e_hat[:, i])] or [fit_setar(e_hat[:, i])] for each stock i
         --> ts_params: list of N parameter dicts
    |
    v
Model dict (estimated nightly)
    |
    v
[run_dynamic_execution] -- intraday loop:
    Initialize: last_specific = e_hat[-1, stock_idx]  (last in-sample residual)
    For each bin j = 0..k-1:
        [forecast_next_bin] for bins j..k-1
            --> turnover forecasts: (k-j,)
        [compute_vwap_weights]
            --> weights: (k-j,), sums to 1.0
        Execute weight[0] * shares_remaining
        Observe actual volume for bin j
        Update last_specific = actual_turnover - c_hat[j]
        (only specific component updated; common forecast stays fixed)
```

**Type summary:**
- Turnover values: float64 (typically in range 1e-5 to 1e-1)
- Factor matrix: float64, shape (P, r) with r typically 1-5
- Loading matrix: float64, shape (N, r)
- AR(1) parameters: 3 floats (psi_1, psi_2, sigma2)
- SETAR parameters: 7 floats (phi_11, phi_12, phi_21, phi_22, tau, sigma2_1, sigma2_2)
- VWAP weights: float64, shape (k_remaining,), non-negative, sums to 1.0

### Variants

**Implemented variant:** PCA-SETAR with AR(1) fallback, dynamic VWAP execution.

**Rationale:** BDF 2008 demonstrates SETAR outperforms AR(1)/ARMA in 36 of 39 stocks
(Section 3.2, paragraph: "the SETAR model seems to be better...for 36 stocks out of
39"). Szucs 2017 confirms SETAR wins on MAPE for 30 of 33 stocks in pairwise comparison
against AR(1) (Table 2c: BDF_SETAR vs BDF_AR = 30/3). The AR(1) is retained as a
fallback for stocks where SETAR estimation fails.

**Not implemented:**
- Static VWAP execution: BDF 2008, Section 4.2, explicitly shows it is dominated by
  the classical approach. Multi-step forecasts decay to zero.
- Theoretical VWAP execution: requires knowing end-of-day total volume (not available
  in real time). Used only as an upper bound in the paper. (BDF 2008, Section 4.1.3,
  "the information about the overall volume" ... "unknown before the market closes".)
- ARMA(1,1): The equation in BDF 2008 is actually AR(1) with intercept (no MA term).

## Parameters

| Parameter | Description | Recommended Value | Sensitivity | Range |
|-----------|-------------|-------------------|-------------|-------|
| k | Number of intraday bins per trading day | 25 (BDF) or 26 (Szucs) | Low -- determined by market hours and bin width | 13-52 (30-min to 5-min bins) |
| L | Estimation window length (trading days) | 20 | Medium -- shorter windows adapt faster but increase estimation noise | 10-40 |
| delta_t | Bin width in minutes | 20 (BDF) or 15 (Szucs) | Low -- determines k given market hours | 5-30 |
| r_max | Maximum number of factors to consider in IC | 10 | Low -- as long as it exceeds the true r | 5-20 |
| r | Number of common factors (estimated by Bai-Ng IC) | Data-driven (typically 1-3) | High -- too few misses common patterns, too many overfits | 1-10 |
| model_type | Time-series model for specific component | "SETAR" | High -- SETAR consistently outperforms AR(1) | {"AR1", "SETAR"} |
| n_grid | Grid points for SETAR threshold search | 100 | Low -- 50-200 gives similar results | 50-500 |
| tau_quantile_range | Quantile bounds for SETAR threshold search | [0.15, 0.85] | Low-Medium -- prevents degenerate regimes | [0.10, 0.90] to [0.20, 0.80] |
| min_regime_obs | Minimum observations per SETAR regime | max(0.10 * P, 20) | Low -- mainly a robustness safeguard | 10-50 |
| shares_measure | Denominator for turnover calculation | "TSO" (total shares outstanding) | Low -- affects scale but not relative forecasts | {"float", "TSO"} |

### Initialization

**Source:** BDF 2008, Section 3.2; Szucs 2017, Section 3.

1. **First estimation day:** Requires at least L trading days of historical turnover data
   for all N stocks. With L=20 and k=26, the first estimation uses 520 observations per
   stock.

2. **PCA initialization:** No special initialization needed -- the truncated SVD is
   deterministic given the data matrix (up to sign flips of singular vectors, which do
   not affect C_hat).

3. **AR(1) initialization:** OLS requires no initialization. The first forecast uses
   the last observed specific component value as e_{t-1}.

4. **SETAR initialization:** The grid search over tau is exhaustive within the specified
   quantile range, so no starting point is needed. Conditional OLS in each regime is
   deterministic.

5. **Dynamic execution initialization:** At market open (bin 0), last_specific is set to
   the last specific component value from the estimation window's final bin:
   model["specific"][-1, stock_idx]. This is the in-sample residual from the last
   observation used in PCA estimation. Using this in-sample value is correct and intended
   -- it is the most recent available estimate of the stock's idiosyncratic deviation, and
   all estimation data precedes the forecast day, so there is no look-ahead bias.
   (BDF 2008, Section 2.3: forecasts are conditioned on information available through
   the end of the estimation window.)

6. **Factor count:** r is re-estimated daily via the Bai-Ng IC. In practice, r tends to be
   stable (typically 1-3) and rarely changes from day to day. (Researcher inference based
   on the stability of the common component shown in BDF 2008, Fig. 3.)

### Calibration

**Source:** BDF 2008, Sections 2.2-2.3, 3.2; Szucs 2017, Section 3.

The model is fully re-estimated daily on a rolling L-day window. The calibration
procedure for each new trading day d+1 is:

1. Form the turnover matrix X from the most recent L days (day d-L+1 to day d),
   with shape (L*k, N).
2. Run select_num_factors to determine r.
3. Run extract_factors to get F_hat, Lambda_hat, C_hat, e_hat.
4. Run forecast_common to get c_hat for day d+1.
5. For each stock i, run fit_ar1 or fit_setar on e_hat[:, i].
6. Store the model for use during day d+1's dynamic execution.

**No cross-validation or hyperparameter tuning is needed** beyond the Bai-Ng IC for r.
All other parameters (AR coefficients, SETAR coefficients and threshold) are estimated
directly from the rolling window. (BDF 2008, Section 3.2.)

**Computational cost:** BDF model estimation for 33 stocks over 2648 days takes
approximately 2 hours (Szucs 2017, Section 5, last paragraph). Per-day estimation is
on the order of seconds, dominated by the SVD step. (Szucs 2017, Section 5.)

## Validation

### Validation Metrics

**Source:** Szucs 2017, Section 3, Equations 1-2.

The following metrics are used to evaluate forecast accuracy. Both are computed
per-stock first, then averaged across stocks.

**Mean Squared Error (MSE):**

```
MSE_i = (1 / n_i) * sum_{t=1}^{n_i} (Y_t - Y_t^f)^2
```

where Y_t is the actual turnover, Y_t^f is the forecast turnover, and n_i is the total
number of bin-level forecasts for stock i (= number_of_forecast_days * k).
(Szucs 2017, Section 3, Eq. 1.)

**Mean Absolute Percentage Error (MAPE):**

```
MAPE_i = (1 / n_i) * sum_{t=1}^{n_i} |Y_t - Y_t^f| / Y_t
```

(Szucs 2017, Section 3, Eq. 2.)

**MAPE exclusion rule:** Exclude observations where Y_t = 0 from the MAPE calculation
(division by zero). If using a dataset pre-filtered for non-zero volume (as Szucs does),
this is not an issue. Otherwise, count only observations with Y_t > 0 in both the
numerator sum and the denominator n_i.

**Aggregation procedure:**
1. Compute MSE_i and MAPE_i for each stock i, averaging over all bins and all
   out-of-sample forecast days for that stock.
2. Report the cross-stock average: MSE = (1/N) * sum_{i=1}^{N} MSE_i, and similarly
   for MAPE.
(Szucs 2017, Section 3: "Both measures are calculated for each share, and also in the
average of all shares.")

**Optional: Scale-adjusted MSE (MSE*):**

```
MSE* = (1/N) * sum_{i=1}^{N} MSE_i / (a_i / a_min)^2
```

where a_i is the average turnover of stock i and a_min is the smallest average turnover
across all stocks. This normalizes MSE by turnover scale, addressing the fact that
high-turnover stocks mechanically produce larger MSE values.
(Szucs 2017, Section 5, Eq. 14.)

### Expected Behavior

**Volume prediction accuracy (one-step-ahead, per-stock averages):**

From Szucs 2017, Table 2a (Section 5), full sample of 33 DJIA stocks, 2648 out-of-sample
forecast days (from 2668 total trading days minus the 20-day initial estimation window),
k=26 bins, L=20 days:

| Model | MSE | MAPE |
|-------|-----|------|
| U-method | 1.02e-03 | 0.503 |
| BDF_AR | 6.49e-04 | 0.403 |
| BDF_SETAR | 6.60e-04 | 0.399 |

(Szucs 2017, Section 5, Table 2a.)

**Notes on MAPE interpretation:** The per-stock MAPE of ~0.40 means the average
absolute percentage error for individual bin forecasts is 40%. This is typical for
intraday volume -- volume is inherently noisy at fine time scales. The value of the model
is relative to the U-method baseline (0.503), showing a ~20% reduction in MAPE.

**Note on MSE vs MAPE best model:** BDF_AR produces the lowest MSE (6.49e-04) while
BDF_SETAR produces the lowest MAPE (0.399). These lead to different "best model"
conclusions depending on the metric. MSE is scale-sensitive and penalizes large errors
more heavily; MAPE treats all percentage errors equally. For VWAP execution, MAPE is
typically more relevant since it measures proportional forecast accuracy across all bins.
(Szucs 2017, Section 5, text below Table 2a.)

From BDF 2008, Table 2, portfolio-level MAPE (CAC40 basket, 39 stocks, k=25, L=20):

| Model | Mean MAPE | Std | Q95 |
|-------|-----------|-----|-----|
| PCA-SETAR (one-step-ahead) | 0.0752 | 0.0869 | 0.2010 |
| PCA-ARMA (one-step-ahead) | 0.0829 | 0.0973 | 0.2330 |
| Classical average | 0.0905 | 0.1050 | 0.2490 |

(BDF 2008, Table 2, "Prediction of model for intraday volume" panel.)

**Note on per-stock vs portfolio MAPE:** The BDF portfolio MAPE (0.075) is much lower
than the Szucs per-stock MAPE (0.40) because portfolio-level aggregation diversifies
away idiosyncratic forecast errors. These are not comparable metrics -- different
datasets, different aggregation levels, different markets.

**VWAP execution cost (out-of-sample, dynamic execution):**

From BDF 2008, Table 2, third panel:

| Model | Mean MAPE | Std | Q95 |
|-------|-----------|-----|-----|
| PCA-SETAR dynamic | 0.0898 | 0.0954 | 0.2854 |
| PCA-ARMA dynamic | 0.0922 | 0.0994 | 0.2854 |
| Classical approach | 0.1006 | 0.1171 | 0.3427 |

(BDF 2008, Table 2, "Out-of-sample estimation for VWAP execution" panel.)

The dynamic PCA-SETAR reduces VWAP tracking error by approximately 10% on average
versus the classical approach (MAPE 0.0898 vs 0.1006). Improvements can reach up to 50%
for high-volatility stocks. (BDF 2008, Section 4.3.3, last paragraph: "use of our method
allows for a reduction of the error by 10%".)

**Note on BDF's "8 bp" figure:** BDF 2008 Section 4.3.3 states the tracking error "is
lower (8 bp)". This refers to a portfolio-level execution cost under a different
aggregation than the MAPE values in Table 2 (where MAPE 0.0898 = 898 bp, not 8 bp).
For implementation validation, use the Table 2 MAPE values directly. Do not expect the
absolute tracking error level to be 8 bp -- that figure is an artifact of BDF's
portfolio-weighted calculation method. [Revised per critique round 2, m2.]

### Sanity Checks

1. **Factor count r:** For a typical cross-section of 20-50 liquid stocks, the Bai-Ng IC
   should select r between 1 and 5. If r > 5, the IC may be miscalibrated or the data
   has unusual structure. (Researcher inference based on typical factor model results.)

2. **Common component shape:** The average common component across stocks should
   resemble a U-shape (high at market open and close, low at midday). Plot
   mean(C_hat, axis=1) reshaped as (L, k) and averaged over L days. (BDF 2008, Fig. 3,
   top panels showing common component with characteristic U-shape.)

3. **Specific component magnitude:** The specific component should be much smaller than
   the common component in absolute terms. Typically |e_{i,t}| << |c_{i,t}| for most
   observations. The ratio var(e) / var(x) should be substantially less than 1 for most
   stocks. (BDF 2008, Fig. 3, bottom panels: specific component values in [-0.025, 0.010]
   versus common component in [0, 0.035].)

4. **AR(1) coefficient:** psi_1 should be positive and less than 1 (stationary). Typical
   range: 0.1-0.6. If |psi_1| >= 1, the AR(1) is non-stationary and the specific
   component series should be inspected for anomalies. (Researcher inference based on
   typical autocorrelation structure of residual volumes; BDF 2008, Fig. 2, PACF of
   specific component shows significant first lag.)

5. **SETAR threshold tau:** Should be near the median of the specific component. If tau is
   at an extreme quantile, one regime has very few observations and the model may be
   poorly estimated. (Researcher inference.)

6. **Turnover non-negativity:** Combined forecasts (c_hat + e_hat) should be non-negative
   for the vast majority of bins (>95%). A high rate of negative forecasts indicates
   model failure. (Researcher inference: volume is non-negative by definition.)

7. **VWAP weights:** Should sum to 1.0, all non-negative. The weight profile should
   roughly follow the U-shape (more execution at open and close). (Researcher inference
   from the volume profile shape.)

8. **U-method comparison (MSE pairwise):** The BDF model should produce lower MSE than the
   U-method for all or nearly all stocks. Szucs 2017, Table 2b: BDF_AR beats U-method
   on 33/0 stocks by MSE; BDF_SETAR beats U-method on 33/0 stocks by MSE. If the BDF
   model does not beat U-method for most stocks, check data preprocessing (centering
   error, turnover calculation, etc.).

9. **U-method comparison (MAPE pairwise):** Szucs 2017, Table 2c: BDF_AR beats U-method
   on 32/1 stocks by MAPE; BDF_SETAR beats U-method on 32/1 stocks by MAPE. Near-total
   dominance is expected.

10. **SETAR vs AR(1) pairwise:** Szucs 2017, Table 2b: BDF_SETAR beats BDF_AR on 6/27
    stocks by MSE (AR is better by MSE). Table 2c: BDF_SETAR beats BDF_AR on 30/3 stocks
    by MAPE (SETAR is better by MAPE). This asymmetry (AR wins MSE, SETAR wins MAPE)
    is expected -- SETAR's regime-switching helps proportional accuracy (MAPE) more than
    squared-error accuracy (MSE). (BDF 2008, Section 3.2: SETAR wins for 36/39 stocks
    in estimation.)

11. **Computational performance:** Full model estimation for ~30 stocks over ~2500 days
    should complete in under 2 hours on modern hardware. Per-day estimation should be
    on the order of seconds. (Szucs 2017, Section 5.)

### Edge Cases

1. **Zero-volume bins:** If a stock has zero volume in a bin, turnover is zero. This is
   valid data (not missing). The PCA and AR/SETAR will handle it naturally. However, MAPE
   is undefined when the actual is zero (division by zero). Exclude zero-actual bins from
   MAPE calculation. (Szucs 2017, Section 2 filters for stocks with "volume record larger
   than zero in every 15-minute interval", so this should not arise in a properly filtered
   dataset.)

2. **Stock entering/leaving the cross-section:** If a stock has insufficient history (less
   than L days), it cannot be included in the PCA. The cross-section N should be stable
   over the estimation window. For IPOs or delistings, exclude the stock from the factor
   estimation for that window. **For stable results, maintain a fixed cross-section N
   throughout the evaluation period.** If the cross-section must change (e.g., index
   reconstitution), re-estimate the full model from scratch on the new cross-section
   rather than treating the new N as a continuation. The specific component time series
   should be re-extracted under the new factor structure before fitting AR/SETAR models,
   because changing N alters the PCA decomposition and can introduce structural breaks
   in the specific component series. (Researcher inference: PCA factor stability requires
   a fixed cross-section; the factor loadings and common component are incomparable across
   different N.)

3. **Single-stock application:** The model CANNOT be applied to a single stock (N=1). PCA
   requires a cross-section. Minimum practical N is approximately 10-20 for stable factor
   estimation. (BDF 2008 uses N=39; Szucs 2017 uses N=33. Bai 2003 asymptotics require
   both N and T large.)

4. **Very short estimation windows:** If L < 10, the PCA estimation will have very few
   time observations per stock (P = L*k < 250 for k=25). Factor estimates become noisy.
   (Researcher inference.)

5. **Non-stationary specific component:** If the AR(1) coefficient |psi_1| >= 1, the
   specific component is non-stationary. This can happen during periods of extreme
   volume events (earnings, M&A). Fallback: use the U-method for that stock on that day.
   (Researcher inference.)

6. **Market half-days:** Some trading days have shortened hours (e.g., day before holidays).
   These days have fewer than k bins. Options: (a) exclude from estimation window, (b)
   pad with NaN and handle in PCA. BDF 2008 excludes partial days (Dec 24, Dec 31).
   Recommendation: exclude partial days from the estimation window. (BDF 2008,
   Section 3.1.)

7. **Overnight gaps:** The model treats the intraday time series as contiguous across days
   (bin k of day d is followed by bin 1 of day d+1 in the AR/SETAR series). This
   introduces an overnight "gap" where the AR dynamics may not be meaningful. The periodic
   autocorrelation at lag k (visible in BDF 2008, Fig. 2: ACF spikes at multiples of 25)
   is a known artifact of this treatment. BDF and Szucs do NOT address this periodic
   structure -- the simple AR(1)/SETAR captures only the lag-1 dependence. Do not add
   seasonal AR terms unless extending the model beyond what the papers describe.

### Known Limitations

1. **Volume only, no price model:** The BDF model forecasts volume only. To "beat" VWAP
   (execute at better than VWAP), a bivariate volume-price model is needed. The current
   model can only track VWAP, not systematically outperform it. (BDF 2008, Section 5,
   Conclusion: "in order to beat the VWAP, our price-adjusted-volume model is not
   sufficient and it is essential to derive a bivariate model for volume and price".)

2. **Small order assumption:** The model assumes the trader's order is small relative to
   market volume (no market impact). For large orders that move the price, the execution
   schedule would need to account for price impact, which the model does not address.
   (BDF 2008, Section 4.1.3: "we suppose that he is able to trade without impact on
   prices".)

3. **Cross-section required:** Cannot be used for single-stock forecasting. Requires a
   stable cross-section of liquid stocks for PCA factor estimation. (BDF 2008, Section
   2.2; Bai 2003 asymptotics.)

4. **Stationarity assumption:** The model assumes the specific component is stationary
   within the estimation window. During regime changes (e.g., financial crisis onset),
   the rolling window may mix different regimes, degrading forecasts. (Researcher
   inference.)

5. **No intraday events:** The model does not handle scheduled intraday events (e.g.,
   FOMC announcements, earnings releases) that can dramatically alter the volume
   profile on specific days. (Researcher inference.)

6. **Linear factor model:** The PCA decomposition assumes a linear factor structure. If
   the common volume pattern is nonlinear (e.g., the U-shape amplitude varies
   multiplicatively with daily volume level), the additive model may be misspecified.
   This is a known limitation relative to multiplicative models like BCG. (Szucs 2017,
   Section 4, describing the difference between additive BDF and multiplicative BCG.)

7. **Factor rotation indeterminacy:** PCA factors are identified only up to rotation. The
   individual factors F_hat and loadings Lambda_hat are not uniquely interpretable, though
   the common component C_hat = F_hat @ Lambda_hat.T is unique. (Bai 2003, standard PCA
   property; BDF 2008 does not interpret individual factors.)

## Paper References

| Spec Section | Paper Source | Section/Equation |
|-------------|-------------|-----------------|
| Data Preprocessing: bin aggregation | BDF 2008 | Section 3.1 |
| Data Preprocessing: turnover definition | BDF 2008 | Section 2.2, below Eq. 3 |
| Data Preprocessing: turnover (TSO variant) | Szucs 2017 | Section 2, page 4 |
| Data Preprocessing: no centering | Bai 2003 | Section 2 (model definition) |
| Data Preprocessing: TSO handling | Researcher inference | Practical data handling |
| select_num_factors: IC formula | Bai & Ng 2002 | Econometrica 70(1), pp. 191-221 |
| select_num_factors: use in BDF | BDF 2008 | Section 2.2, text ref. to Bai & Ng |
| select_num_factors: V(r) from eigenvalues | Researcher inference | Standard linear algebra (eigenvalue-SSR relationship) |
| extract_factors: PCA procedure | BDF 2008 | Section 2.2, Equations 4-6 |
| extract_factors: Bai normalization | Bai 2003 | Econometrica 71(1), pp. 135-171 |
| extract_factors: SVD implementation | Researcher inference | Numerically equivalent to eigendecomposition |
| forecast_common: historical average | BDF 2008 | Section 2.3, Equation 9 |
| forecast_common: fixed intraday | BDF 2008 | Section 2.3, paragraph below Eq. 9 |
| fit_ar1: AR(1) model | BDF 2008 | Section 2.3, Equation 10 |
| fit_ar1: AR(1) confirmation | Szucs 2017 | Section 4.1, Equation 5 |
| fit_ar1: variance estimator (n-2) | Researcher inference | Standard unbiased OLS residual variance |
| fit_ar1: overnight gap periodic ACF | BDF 2008 | Fig. 2 (ACF of specific component) |
| fit_setar: SETAR model | BDF 2008 | Section 2.3, Equation 11 |
| fit_setar: SETAR confirmation | Szucs 2017 | Section 4.1, Equations 6-7 |
| fit_setar: grid search estimation | Researcher inference | Standard SETAR estimation (Hansen 1997) |
| fit_setar: fallback to AR(1) | Researcher inference | BDF Section 3.2 uses SETAR for all 39 stocks |
| forecast_next_bin: combined forecast | BDF 2008 | Section 2.3, Equation 8 |
| forecast_next_bin: one-step-ahead | Szucs 2017 | Section 3, "one-step-ahead forecasting" |
| compute_vwap_weights: negative clipping | Researcher inference | Volume non-negativity constraint |
| run_dynamic_execution: dynamic strategy | BDF 2008 | Section 4.2.2 |
| run_dynamic_execution: static dominated | BDF 2008 | Section 4.2, penultimate paragraph p. 1717 |
| run_dynamic_execution: bootstrapping | BDF 2008 | Section 2.3, forecast conditioned on estimation window |
| daily_update: rolling re-estimation | BDF 2008 | Section 3.2 |
| daily_update: rolling window procedure | Szucs 2017 | Section 3, Figure 1 |
| u_method_benchmark | Szucs 2017 | Section 4, Equation 3 |
| Validation metrics: MSE, MAPE formulas | Szucs 2017 | Section 3, Equations 1-2 |
| Validation metrics: MSE* formula | Szucs 2017 | Section 5, Equation 14 |
| Validation metrics: aggregation | Szucs 2017 | Section 3, text below Eq. 2 |
| Validation: per-stock MSE/MAPE | Szucs 2017 | Section 5, Table 2a |
| Validation: pairwise MSE | Szucs 2017 | Section 5, Table 2b |
| Validation: pairwise MAPE | Szucs 2017 | Section 5, Table 2c |
| Validation: portfolio MAPE | BDF 2008 | Table 2, first panel |
| Validation: VWAP execution cost | BDF 2008 | Table 2, third panel |
| Validation: SETAR wins 36/39 stocks | BDF 2008 | Section 3.2 |
| Validation: ~10% TE reduction | BDF 2008 | Section 4.3.3, last paragraph |
| Limitation: no price model | BDF 2008 | Section 5, Conclusion |
| Limitation: small order assumption | BDF 2008 | Section 4.1.3 |
| Limitation: additive vs multiplicative | Szucs 2017 | Section 4 |
| Edge case: partial days excluded | BDF 2008 | Section 3.1 |
| Edge case: zero-volume filtering | Szucs 2017 | Section 2 |
| Edge case: fixed cross-section | Researcher inference | PCA factor stability |
| estimate_model: shared SVD optimization | Researcher inference | Standard numerical optimization |
| select_num_factors: V(r) guard | Researcher inference | Defensive floating-point handling |
| fit_setar: two-phase estimation | Researcher inference | Efficiency optimization |
| Validation: "8 bp" clarification | BDF 2008 | Section 4.3.3, portfolio-level metric |
