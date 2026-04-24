# Implementation Specification: PCA Factor Decomposition (BDF) for Intraday Volume

## Overview

The BDF model (Bialkowski, Darolles, and Le Fol 2008) decomposes intraday stock
turnover into a market-wide common component and a stock-specific idiosyncratic
component using principal components analysis (PCA) across a cross-section of
stocks. The common component captures the shared U-shaped intraday volume
pattern; the specific component captures stock-level deviations and is modeled
by either AR(1) or SETAR time-series processes. SETAR is the recommended
variant based on both the original paper and the independent validation by
Szucs (2017), which found it more accurate than AR(1) and the competing CMEM
model on 11 years of DJIA data while being orders of magnitude faster to
estimate.

Note on model labeling: BDF 2008 calls the specific component model "ARMA(1,1)"
but Eq. (10) contains no moving average term -- it is an AR(1) with intercept.
Szucs 2017 correctly labels it "AR(1)". This specification uses "AR(1)"
throughout to match the actual equation.

The model produces one-step-ahead intraday volume forecasts updated dynamically
as each bin's actual volume is observed, enabling dynamic VWAP execution
strategies that reduce tracking error by approximately 10% on average versus
historical averages.

## Algorithm

### Model Description

The model takes as input a matrix of intraday turnover values for N stocks over
T intraday time intervals, and produces one-step-ahead volume forecasts for
each stock at each interval. The key assumption is that much of the intraday
volume variation is driven by common market-wide factors (the U-shaped pattern)
that can be extracted by PCA, leaving simpler stock-specific residual dynamics.

Inputs:
- X: a (T x N) matrix of intraday turnover values, where T = number of
  intraday observations in the estimation window (L_days * k intervals/day)
  and N = number of stocks in the cross-section.
- Turnover is defined as x_{i,t} = V_{i,t} / TSO_{i,t}, where V is shares traded
  and TSO is total shares outstanding.

Outputs:
- For each stock i and each intraday interval, a one-step-ahead turnover
  forecast x_hat_{i,t+1}.
- These can be converted to volume shares (fraction of daily volume) for VWAP
  scheduling by normalizing: w_hat_{i,j} = x_hat_{i,j} / sum_j(x_hat_{i,j}).

Assumptions:
- The cross-section N must be large enough for PCA to reliably extract common
  factors (N >= 30 recommended, per Bai 2003 asymptotics).
- All stocks must have non-zero volume in every intraday bin within the
  estimation window.
- The trader's order size is small relative to market volume (no price impact).
- The common component is approximately stable over the estimation window
  (20 trading days).

### Pseudocode

The algorithm operates in two phases: an overnight estimation phase (run once
per day before market open) and an intraday forecasting phase (run at each
interval during the trading day).

#### Phase 1: Overnight Estimation (run once per day, before market open)

```
FUNCTION estimate_model(turnover_history, n_stocks, k_bins, L_days):
    // turnover_history: array of shape (L_days * k_bins, n_stocks)
    //   containing the most recent L_days of intraday turnover data
    //   for all n_stocks stocks, ordered chronologically by bin
    //   (day1_bin1, day1_bin2, ..., day1_binK, day2_bin1, ..., dayL_binK)

    // Step 1: Construct the data matrix X
    // X has shape (T, N) where T = L_days * k_bins, N = n_stocks
    // Reference: BDF 2008, Section 2.2, Eq. (4)-(5)
    X = turnover_history  // shape: (T, N)
    T = L_days * k_bins
    N = n_stocks

    // Note on demeaning: Standard PCA often subtracts column means before
    // eigendecomposition. In the Bai (2003) large-dimensional factor
    // framework used here, demeaning is NOT required -- the column means
    // are absorbed into the factor loadings. BDF 2008 does not mention
    // demeaning and the formulation in Eq. (6) does not include it.
    // Demeaning before PCA would not materially change results (the mean
    // would shift from the common to the specific component) but is not
    // necessary and deviates from the paper's formulation.
    // (Reference: Bai 2003; BDF 2008, Section 2.2)

    // Step 2: Select number of common factors r using Bai & Ng (2002) IC
    // Reference: BDF 2008, Section 2.2; Bai & Ng (2002)
    r = select_number_of_factors(X, T, N)

    // Step 3: Extract factors and loadings via PCA (Bai 2003)
    // Reference: BDF 2008, Section 2.2, Eq. (6), text below Eq. (6)
    // Solve: min_{F,A} (NT)^{-1} sum_i sum_t (X_{it} - A_i' F_t)^2
    // with normalization F'F/T = I_r
    //
    // BDF 2008 states: "The estimated factors matrix F is proportional
    // (up to T^{1/2}) to the eigenvectors corresponding to the r-largest
    // eigenvalues of the X'X matrix" (p.1712, text below Eq. 6).
    //
    // X'X is an (N x N) matrix. This is more efficient than using XX'
    // (T x T) when N < T, which is the typical case (BDF 2008: N=39,
    // T=500; Szucs 2017: N=33, T=520).
    //
    // Procedure:
    //   1. Compute X'X / T  (an N x N matrix)
    //   2. Eigendecompose to get eigenvalues and eigenvectors
    //   3. Take the r eigenvectors corresponding to the r largest eigenvalues
    //   4. Recover F_hat = X @ eigenvectors[:, :r]  (shape: T x r)
    //   5. Normalize: F_hat = F_hat * sqrt(T) / norm(F_hat columns)
    //      such that F_hat' @ F_hat / T = I_r
    //   6. Recover A_hat = F_hat' @ X / T  (shape: r x N)
    //
    // Equivalently, using SVD of X: X = U S V', then F_hat = U[:,:r] * sqrt(T)
    // and A_hat = S[:r,:r] @ V'[:r,:] / sqrt(T). This avoids forming X'X
    // explicitly and is numerically more stable.

    XtX = X.T @ X / T                           // shape: (N, N)
    eigenvalues, eigvecs = eigendecomposition(XtX)
    // eigvecs columns sorted by decreasing eigenvalue
    // Take r largest
    V_r = eigvecs[:, :r]                         // shape: (N, r)
    F_hat = X @ V_r                              // shape: (T, r)
    // Normalize so F_hat' F_hat / T = I_r
    // Since X'X/T eigenvectors satisfy V_r' (X'X/T) V_r = diag(lambda_1..r),
    // we have F_hat' F_hat / T = V_r' X'X/T V_r = diag(lambda), so
    // F_hat_normalized = F_hat @ diag(1/sqrt(lambda_1..r))
    F_hat = F_hat @ diag(1.0 / sqrt(eigenvalues[:r]))  // now F'F/T = I_r
    A_hat = F_hat.T @ X / T                     // shape: (r, N)

    // Step 4: Compute common and specific components
    // Reference: BDF 2008, Section 2.2, Eq. (2)-(3)
    C_hat = F_hat @ A_hat         // common component, shape: (T, N)
    E_hat = X - C_hat             // specific component, shape: (T, N)

    // Step 5: Compute common component forecast for tomorrow
    // Reference: BDF 2008, Section 2.3, Eq. (9)
    // Average the same time-of-day bin across L_days prior days
    FOR j = 1 TO k_bins:
        FOR i = 1 TO N:
            // Gather bin j values across all L_days days
            bin_values = [C_hat[(d-1)*k_bins + j - 1, i] for d in 1..L_days]
            c_forecast[i, j] = mean(bin_values)

    // Step 6: Fit time-series model to specific component for each stock
    // Reference: BDF 2008, Section 2.3, Eq. (10)-(11)
    // BDF 2008 specifies estimation "by maximum likelihood" (Section 2.3,
    // paragraph after Eq. 11).
    FOR i = 1 TO N:
        e_series = E_hat[:, i]    // shape: (T,)

        // Option A: AR(1) with intercept
        // Reference: BDF 2008, Eq. (10)
        // e_{i,t} = psi_1 * e_{i,t-1} + psi_2 + epsilon_{i,t}
        //   where epsilon_{i,t} is white noise
        //
        // Note: BDF 2008 labels this "ARMA(1,1)" but the equation has no
        // moving average term -- it is AR(1) with intercept. Szucs 2017
        // correctly calls it "AR(1)" (Eq. 5). This spec uses "AR(1)".
        //
        // Estimation: MLE for AR(1) with Gaussian errors is equivalent to
        // OLS (conditional on the first observation). Use OLS for simplicity:
        // regress e_{i,t} on [e_{i,t-1}, 1] for t = 2..T.
        // (Reference: BDF 2008, Section 2.3 specifies MLE; OLS yields
        // identical point estimates under Gaussian assumptions.)
        ar_params[i] = fit_ar1_by_mle(e_series)
        // ar_params[i] = (psi_1, psi_2, sigma_epsilon)

        // Option B: SETAR -- Reference: BDF 2008, Eq. (11)
        // e_{i,t} = (phi_11 * e_{i,t-1} + phi_12) * I(e_{i,t-1} <= tau)
        //         + (phi_21 * e_{i,t-1} + phi_22) * (1 - I(e_{i,t-1} <= tau))
        //         + epsilon_{i,t}
        //
        // Estimation: BDF 2008 specifies MLE. Standard SETAR estimation
        // uses profile likelihood / conditional MLE with grid search over
        // tau (equivalent to conditional OLS grid search under Gaussian
        // errors; see Hansen 1997, Tong 1990). Procedure:
        //   1. Sort e_series values to form candidate threshold grid
        //   2. For each candidate tau in [15th, 85th] percentile (see below):
        //      a. Split observations into two regimes based on I(e_{t-1} <= tau)
        //      b. Fit AR(1) with intercept in each regime by OLS
        //      c. Compute total sum of squared residuals (= neg log-likelihood)
        //   3. Select tau minimizing total SSR
        //   4. Final parameters are those from the optimal tau split
        //
        // Grid search range: use the 15th to 85th percentile of e_series
        // values as candidate thresholds (not 10th-90th) to ensure each
        // regime contains at least ~15% of observations for reliable
        // parameter estimation. Use 1-percentile steps (71 candidates).
        // If either regime has fewer than max(15, 0.10 * T) observations,
        // skip that candidate.
        // (Reference: BDF 2008, Section 2.3, Eq. 11 specifies model form;
        // grid search methodology follows Hansen 1997, Tong 1990)
        setar_params[i] = fit_setar_by_profile_mle(e_series)
        // setar_params[i] = (phi_11, phi_12, phi_21, phi_22, tau, sigma_eps)

    RETURN (c_forecast, ar_params OR setar_params, A_hat, F_hat, r)
```

#### Phase 2: Intraday Dynamic Forecasting (run at each bin during the day)

```
FUNCTION forecast_intraday(observed_turnover_today, j_current,
                           c_forecast, ts_params, model_type,
                           k_bins):
    // observed_turnover_today: array of shape (j_current, N)
    //   actual turnover observed so far today (bins 1..j_current)
    // j_current: the most recently completed bin index (1-based)
    //   j_current = 0 means no bins observed yet (start of day)
    // c_forecast: precomputed common component forecast, shape (N, k_bins)
    // ts_params: fitted AR(1) or SETAR parameters per stock
    // model_type: "AR" or "SETAR"

    // Step 1: Handle start-of-day initialization
    // Reference: BDF 2008, Section 4.2.2 -- dynamic execution starts
    //   with static forecast and updates after each observed bin
    IF j_current == 0:
        // No bins observed yet. Use common component forecast only
        // (specific component forecast = 0, its unconditional mean).
        // This is equivalent to the static/classical forecast.
        // (Researcher inference: BDF 2008 does not explicitly specify
        // initialization of e_forecast at day start. Setting to 0 is
        // consistent with Section 4.2.1 which describes starting with
        // static predictions, and with the unconditional mean of the
        // specific component being ~0 by construction of PCA.)
        FOR i = 1 TO N:
            x_forecast[i, 1:k_bins] = c_forecast[i, 1:k_bins]
        RETURN x_forecast, x_forecast / sum(x_forecast, axis=1)

    // Step 2: Extract today's specific component from most recent observation
    // Reference: BDF 2008, Section 2.2 (filtering step, Eq. 7-8)
    // Use yesterday's factor loadings to decompose today's observations
    FOR i = 1 TO N:
        e_observed[i, j_current] = observed_turnover_today[j_current, i]
                                   - c_forecast[i, j_current]

    // Step 3: Produce one-step-ahead specific component forecast
    // Reference: BDF 2008, Section 2.3, Eq. (10) or (11)
    FOR i = 1 TO N:
        e_last = e_observed[i, j_current]

        IF model_type == "AR":
            psi_1, psi_2 = ts_params[i]
            e_forecast[i, j_current + 1] = psi_1 * e_last + psi_2

        ELSE IF model_type == "SETAR":
            phi_11, phi_12, phi_21, phi_22, tau = ts_params[i]
            IF e_last <= tau:
                e_forecast[i, j_current + 1] = phi_11 * e_last + phi_12
            ELSE:
                e_forecast[i, j_current + 1] = phi_21 * e_last + phi_22

    // Step 4: Produce multi-step forecasts for all remaining bins
    // Reference: BDF 2008, Section 4.2.2 -- dynamic VWAP requires
    //   estimating the remaining daily volume to compute shares.
    //   At each step, the trader needs forecasts for bins j_current+1
    //   through k_bins to compute the denominator (estimated daily total).
    //
    // For AR(1): iterate the recursion forward.
    //   e_hat[j+1] = psi_1 * e_hat[j] + psi_2
    //   This decays geometrically toward psi_2 / (1 - psi_1) as horizon
    //   increases (the unconditional mean of the specific component).
    //
    // For SETAR: multi-step forecasts are more complex because the regime
    //   at future steps is unknown. Use deterministic iteration (plug the
    //   point forecast into the threshold comparison at each step).
    //   This is a standard approximation; the exact conditional expectation
    //   would require integrating over the error distribution, which is
    //   computationally expensive and offers marginal improvement.
    //   (Researcher inference: BDF 2008 does not specify how to handle
    //   multi-step SETAR forecasts. Deterministic iteration is standard.)
    //
    // Note: As the forecast horizon increases, multi-step forecasts for
    //   both AR(1) and SETAR decay toward the unconditional mean (~0).
    //   For bins far into the future, the forecast effectively reduces to
    //   the common component alone. This is why static (all-bins-at-once)
    //   execution performs poorly (BDF 2008, Section 4.2.1).
    FOR i = 1 TO N:
        e_prev = e_forecast[i, j_current + 1]
        FOR j = j_current + 2 TO k_bins:
            IF model_type == "AR":
                psi_1, psi_2 = ts_params[i]
                e_forecast[i, j] = psi_1 * e_prev + psi_2
            ELSE IF model_type == "SETAR":
                phi_11, phi_12, phi_21, phi_22, tau = ts_params[i]
                IF e_prev <= tau:
                    e_forecast[i, j] = phi_11 * e_prev + phi_12
                ELSE:
                    e_forecast[i, j] = phi_21 * e_prev + phi_22
            e_prev = e_forecast[i, j]

    // Step 5: Combine common and specific forecasts for all remaining bins
    // Reference: BDF 2008, Section 2.3, below Eq. (9)
    FOR i = 1 TO N:
        FOR j = j_current + 1 TO k_bins:
            x_forecast[i, j] = c_forecast[i, j] + e_forecast[i, j]
            // Floor negative forecasts to a small positive value
            x_forecast[i, j] = max(x_forecast[i, j], 1e-8)

    // Step 6: Compute volume shares for VWAP execution
    // Reference: BDF 2008, Section 4.2.2 (dynamic VWAP)
    // The proportion to trade in the next bin is:
    //   w[j+1] = x_hat[j+1] / sum_{l=j+1}^{k} x_hat[l]
    // applied to the remaining order volume.
    // (Reference: BDF 2008, Section 4.2.3 -- proportion calculation)
    FOR i = 1 TO N:
        remaining_est = sum(x_forecast[i, j_current+1 : k_bins])
        w_forecast[i, j_current + 1] = x_forecast[i, j_current + 1]
                                        / remaining_est

    RETURN x_forecast[:, j_current+1 : k_bins], w_forecast[:, j_current + 1]
```

#### Phase 3: Daily Rolling Update

```
FUNCTION daily_update(all_turnover_data, day_index, L_days, k_bins):
    // At end of each trading day, slide the estimation window forward
    // and re-run Phase 1 for the next day's forecasts

    // Extract most recent L_days of data
    start = (day_index - L_days) * k_bins
    end = day_index * k_bins
    window_data = all_turnover_data[start:end, :]

    // Re-estimate model
    model = estimate_model(window_data, N, k_bins, L_days)
    RETURN model
```

### Data Flow

```
Raw tick data (N stocks)
    |
    v
Aggregate to fixed-width bins (k bins per day)
    |
    v
Compute turnover: x_{i,t} = V_{i,t} / TSO_{i,t}
    |
    v
Assemble matrix X of shape (L_days * k, N) on rolling window
    |
    v
[Phase 1: Overnight Estimation]
    |
    +---> PCA eigendecomposition of X'X / T  (N x N matrix)
    |         |
    |         +---> V_r: (N, r) eigenvector matrix (r largest eigenvalues)
    |         +---> F_hat = X @ V_r @ diag(1/sqrt(lambda)): (T, r) factors
    |         +---> A_hat = F_hat' @ X / T: (r, N) loadings
    |         |
    |         +---> C_hat = F @ A: (T, N) common component
    |         +---> E_hat = X - C: (T, N) specific component
    |
    +---> Common forecast: c_forecast[i,j] = mean of bin j across L_days
    |         shape: (N, k)
    |
    +---> Specific model fit: AR(1) or SETAR per stock on E_hat[:,i]
    |         AR(1): 3 params per stock (psi_1, psi_2, sigma)
    |         SETAR: 6 params per stock (phi_11,phi_12,phi_21,phi_22,tau,sigma)
    |
    v
[Phase 2: Intraday Forecast Loop]
    |
    j_current = 0: use common-only forecast (static baseline)
    |
    For each completed bin j = 1, 2, ..., k-1:
        observed x_{i,j} --> e_{i,j} = x_{i,j} - c_forecast[i,j]
            --> e_hat_{i,j+1} via AR(1)/SETAR (one-step)
            --> e_hat_{i,j+2..k} via iterated AR(1)/SETAR (multi-step)
            --> x_hat_{i,l} = c_forecast[i,l] + e_hat_{i,l} for l=j+1..k
            --> w_hat_{i,j+1} = x_hat_{i,j+1} / sum(x_hat_{i,j+1..k})
    |
    v
Output: per-stock, per-bin turnover forecasts and volume shares
    shape: scalar per stock per bin (streamed one bin at a time)
```

Type information:
- All turnover values: float64 (typically in range 1e-5 to 5e-2; BDF 2008
  Table 1 shows CAC40 mean turnover 0.0166 with Q95 at 0.0445)
- Factor matrix F_hat: float64 array of shape (T, r) where r is typically 1-3
- Loading matrix A_hat: float64 array of shape (r, N)
- AR(1) params per stock: tuple of 3 float64
- SETAR params per stock: tuple of 6 float64 (including threshold tau)
- Volume share forecasts: float64 in [0, 1], sum to 1 across bins within a day

### Variants

**Implemented variant: BDF with SETAR specific component (BDF-SETAR).**

Rationale: The SETAR variant is chosen as the primary implementation because:

1. BDF 2008 (Section 3.2) shows SETAR outperforms AR(1) for the majority of
   stocks tested. The paper text states "there are only three, of the 31,
   companies for which the ARMA slightly surpasses the SETAR model."
   Note: BDF 2008 contains an internal discrepancy -- Table 1 lists 39 CAC40
   stocks, but Section 3.2 results text says "31". The MAPE comparison may
   have used a subset of the 39 stocks. This spec cites the paper's own
   numbers without resolving this inconsistency.
   (Reference: BDF 2008, Section 3.2; Table 1)
2. Szucs 2017 (Section 4, full-sample results table) independently confirms on
   33 DJIA stocks over 11 years that BDF-SETAR achieves the best MAPE (0.399)
   versus BDF-AR (0.403) and the competing CMEM model (0.402).
3. By MSE, BDF-AR slightly edges BDF-SETAR (6.49e-4 vs 6.60e-4 in Szucs 2017),
   but the difference is small and MAPE is the more relevant metric for VWAP
   execution (it penalizes proportional errors equally across volume levels).

The BDF-AR(1) variant should also be implemented as a simpler fallback, as it
requires fewer parameters and avoids the threshold estimation step.

**Not implemented:** The static VWAP execution strategy is explicitly excluded.
BDF 2008 (Section 4.2.1) demonstrates that static execution (forecasting all
bins at market open) performs worse than even the naive historical average
because multi-step AR/SETAR forecasts decay toward zero, neutralizing the
specific component. Only dynamic execution (updating after each observed bin)
is viable.

## Parameters

| Parameter | Description | Recommended Value | Sensitivity | Range |
|-----------|-------------|-------------------|-------------|-------|
| k | Number of intraday bins per trading day | 26 (15-min bins, 9:30-16:00) | Low -- determined by bin width choice; 25 (20-min) also validated | 13-52 (30-min to 5-min) |
| L_days | Rolling window length for PCA estimation and common component averaging (trading days) | 20 | Medium -- too short loses seasonal stability, too long misses regime changes | 10-60 |
| r | Number of common factors extracted by PCA | Estimated via Bai & Ng (2002) IC; typically 1-3 | Medium -- over-extraction adds noise to specific component, under-extraction leaves common structure in residuals | 1-5 |
| r_max | Maximum number of factors to evaluate in Bai & Ng IC search | 10 | Low -- just an upper bound for the search; results insensitive as long as r_max > true r | 5-20 |
| psi_1 | AR(1) coefficient for specific component | Estimated per stock per window; typically 0.3-0.8 | High -- controls persistence of specific component | (-1, 1) for stationarity |
| psi_2 | Intercept for specific component (AR(1) variant) | Estimated per stock per window; typically near zero | Low -- specific component is mean-zero by construction | (-inf, inf) |
| tau | SETAR threshold on lagged specific component | Estimated per stock per window via grid search | High -- determines regime boundary between calm and turbulent volume states | [15th, 85th] percentile of e_{i,t} |
| phi_11 | AR coefficient, low regime (SETAR) | Estimated per stock | Medium | (-1, 1) |
| phi_12 | Intercept, low regime (SETAR) | Estimated per stock | Low | (-inf, inf) |
| phi_21 | AR coefficient, high regime (SETAR) | Estimated per stock | Medium | (-1, 1) |
| phi_22 | Intercept, high regime (SETAR) | Estimated per stock | Low | (-inf, inf) |
| N | Number of stocks in cross-section | >= 30 | Medium -- PCA asymptotics require large N; Bai (2003) consistency holds as N,T -> inf jointly | 30-500+ |
| min_regime_frac | Minimum fraction of observations per SETAR regime | 0.15 | Low -- prevents degenerate regime estimates; standard practice | 0.10-0.20 |

### Initialization

1. **Data preparation:** Collect at least L_days + 1 trading days of tick data
   for all N stocks. Aggregate into k bins per day. Compute turnover by
   dividing each bin's volume by total shares outstanding.
   (Reference: BDF 2008, Section 3.1; Szucs 2017, Section 2)

2. **PCA initialization:** No special initialization needed. The
   eigendecomposition of X'X / T is a deterministic computation with a
   unique solution (up to sign/rotation of eigenvectors, which does not affect
   the common component C = F A). Use standard numerical linear algebra
   (e.g., numpy.linalg.eigh or scipy.linalg.eigh for symmetric matrices).
   Demeaning of X is NOT required in the Bai (2003) framework -- column means
   are absorbed into the factor loadings.
   (Reference: BDF 2008, Section 2.2, Eq. 6, text below Eq. 6; Bai 2003)

3. **AR(1) initialization:** BDF 2008 (Section 2.3, after Eq. 11) specifies
   estimation by maximum likelihood. For AR(1) with Gaussian errors,
   conditional MLE is equivalent to OLS regression of e_{i,t} on
   [e_{i,t-1}, 1] for t = 2..T. Use OLS for simplicity; the point estimates
   are identical.
   (Reference: BDF 2008, Section 2.3 -- "estimate ... by maximum likelihood")

4. **SETAR initialization:** BDF 2008 specifies MLE but does not detail the
   estimation procedure. The standard approach (Hansen 1997, Tong 1990) is
   conditional MLE via grid search, which under Gaussian errors reduces to
   conditional OLS with grid search over tau:
   - Compute candidate thresholds: percentiles of e_{i,t} from the 15th to
     85th percentile in 1-percentile steps (71 candidates).
   - For each candidate tau, split observations at the threshold and fit
     AR(1) with intercept by OLS in each regime.
   - Require each regime to have at least max(15, min_regime_frac * T)
     observations; skip candidates that violate this.
   - Select tau minimizing the total sum of squared residuals.
   - Final parameters are the regime-specific AR(1) coefficients from the
     optimal split.
   (Reference: BDF 2008, Section 2.3, Eq. 11; estimation methodology per
   Hansen 1997, Tong 1990. Note: BDF 2008 states MLE, this procedure is
   equivalent under Gaussian errors.)

5. **Day-start specific component initialization:** At the start of each
   trading day (before any bins are observed), set the specific component
   forecast to zero for all stocks. This means the initial forecast uses
   only the common component (equivalent to the classical/static forecast).
   As bins are observed during the day, the specific component forecast
   updates via the AR(1)/SETAR recursion.
   (Researcher inference: BDF 2008 does not explicitly specify this
   initialization. Setting e_forecast = 0 is consistent with: (a) the
   specific component having unconditional mean ~0 by PCA construction,
   (b) Section 4.2.1 describing the static forecast as the starting point
   before dynamic updates begin, and (c) avoiding carrying overnight
   information from the previous day's final bin, which would span an
   overnight gap where the AR/SETAR dynamics do not apply.)

6. **First-day forecast:** On the very first day of live operation, no prior
   model exists. Use the historical average (U-method) as the forecast:
   x_hat_{i,j} = (1/L_days) * sum of x_{i,j} across prior L_days days at the
   same bin j. This is equivalent to the common component forecast alone with
   zero specific component contribution.

### Calibration

The model is re-calibrated daily on a rolling window. No long-horizon
calibration or burn-in period is needed beyond the initial L_days trading days.

**Daily calibration procedure:**

1. At end of trading day d, assemble the turnover matrix X for days
   (d - L_days + 1) through d, yielding a (L_days * k, N) matrix.

2. Run PCA on X to extract r factors (r selected by Bai & Ng IC).

3. Compute the common component forecast for day d+1 by averaging each bin's
   common component across the L_days days in the window.

4. Fit AR(1) or SETAR to each stock's specific component series.

5. Store the model parameters for use during day d+1's intraday forecast loop.

**Cross-day boundary handling for specific component estimation:**
The specific component series E_hat[:,i] spans multiple days (L_days * k_bins
observations). The AR(1)/SETAR model is fit to this concatenated series as if
it were continuous. At day boundaries (between bin k of day d and bin 1 of day
d+1), there is typically a large overnight jump in the specific component due
to overnight information arrival. BDF 2008 does not discuss this issue.

This spec follows the paper and fits the AR(1)/SETAR to the full concatenated
series, including cross-day transitions. This is a known limitation: overnight
jumps may inflate the estimated innovation variance and attenuate the AR
coefficient toward zero. An alternative (not in the paper) would be to fit
only within-day observations, pooling across days for parameter estimation.
(Researcher inference for the limitation note; fitting to concatenated series
follows BDF 2008's stated procedure.)

**Computational cost:** BDF 2008 reports the entire pipeline is fast. Szucs 2017
(Section 4) reports approximately 2 hours for 33 stocks over 2,648 days
(approximately 87,000 daily re-estimations), implying ~80ms per daily
re-estimation per stock. This makes real-time daily re-estimation trivially
feasible.
(Reference: Szucs 2017, Section 4)

### Factor Count Selection: Bai & Ng (2002) Information Criteria

The number of factors r is selected by minimizing an information criterion
that balances fit against model complexity.

**Procedure:**
(Reference: BDF 2008, Section 2.2; Bai & Ng 2002)

1. For each candidate r from 1 to r_max (default r_max = 10):
   a. Compute the PCA decomposition with r factors.
   b. Compute the residual variance (denoted sigma_sq(r) to distinguish from
      the PCA objective function V(r) in Eq. 6):
      sigma_sq(r) = (NT)^{-1} * sum_i sum_t (X_{it} - A_i' F_t)^2
   c. Compute the penalty term. Bai & Ng (2002) propose several criteria;
      the most commonly used are:
      - IC_p1(r) = ln(sigma_sq(r)) + r * ((N + T) / (NT)) * ln(NT / (N + T))
      - IC_p2(r) = ln(sigma_sq(r)) + r * ((N + T) / (NT)) * ln(min(N, T))

2. Select r* = argmin_r IC(r).

3. In practice, r is typically 1-3 for intraday volume data. The original BDF
   paper states the number of factors is "estimated from data" using Bai & Ng
   IC but does not report the specific r values obtained for CAC40.
   (Reference: BDF 2008, Section 2.2)

**Researcher inference:** The papers do not specify which of the several Bai & Ng
criteria to use. IC_p2 is the most commonly used in practice and is recommended
as the default. If the selected r appears unstable across rolling windows
(varying by more than 1 across consecutive days), fix r at the mode of recent
selections to avoid day-to-day instability.

## Validation

### Expected Behavior

On typical inputs (liquid U.S. or European equities with 15-20 minute bins and
a 20-day estimation window):

1. **Common component shape:** The common component c_{i,t} should exhibit a
   clear U-shaped intraday pattern -- high volume at market open, declining
   through midday, and rising into the close. This U-shape should be visually
   similar across all stocks in the cross-section.
   (Reference: BDF 2008, Section 3.1, Fig. 3 -- showing common component for
   TOTAL stock on two different days)

2. **Specific component properties:** The specific component e_{i,t} should be
   approximately mean-zero and have much lower variance than the raw turnover.
   Its ACF should show low-order serial correlation (significant at lag 1,
   decaying quickly) without the strong seasonal periodicity visible in raw
   turnover.
   (Reference: BDF 2008, Section 3.1, Fig. 2 -- ACF/PACF plots)

3. **Forecast accuracy (MAPE):** For liquid U.S. stocks at 15-minute resolution:
   - BDF-SETAR: MAPE around 0.40 (Reference: Szucs 2017, full-sample results)
   - BDF-AR: MAPE around 0.40 (Reference: Szucs 2017, full-sample results)
   - U-method benchmark: MAPE around 0.50 (Reference: Szucs 2017)
   For CAC40 at 20-minute resolution:
   - BDF-SETAR: mean MAPE 0.0752 (Reference: BDF 2008, Table 2, column 1)
   - BDF-AR(1): mean MAPE 0.0829 (Reference: BDF 2008, Table 2, column 1)
   - Classical average: mean MAPE 0.0905 (Reference: BDF 2008, Table 2)

   Note: The large difference in MAPE scale between the two papers (0.07-0.09
   vs 0.40-0.50) likely reflects different MAPE definitions or data
   characteristics. BDF 2008 reports per-interval MAPE across the portfolio;
   Szucs reports per-interval per-stock MAPE. Both show the same relative
   ordering: BDF-SETAR > BDF-AR(1) > U-method.

4. **MSE:** For 33 DJIA stocks at 15-minute resolution:
   - BDF-AR: MSE 6.49e-4 (Reference: Szucs 2017, full-sample table)
   - BDF-SETAR: MSE 6.60e-4 (Reference: Szucs 2017, full-sample table)
   - U-method: MSE 1.02e-3 (Reference: Szucs 2017, full-sample table)
   Both BDF variants reduce MSE by approximately 35% versus the U-method.

5. **VWAP tracking error:** Dynamic PCA-SETAR execution reduces out-of-sample
   VWAP tracking error (MAPE) from 10.06% (classical) to 8.98% (dynamic
   BDF-SETAR) at the portfolio level -- approximately a 10% relative
   reduction.
   (Reference: BDF 2008, Table 2, panel 3, out-of-sample section;
   values 0.1006 and 0.0898 expressed as fractions, converted to percentages)

   Per-stock reductions vary widely. BDF 2008's conclusion (Section 5, p.1722)
   states "the reduction is greater than 10% and can even reach 50% for some
   stocks." However, examining Tables 6-7 (per-stock VWAP execution results),
   the largest individual stock improvements from classical to dynamic SETAR
   are approximately 8 bp for CAP GEMINI and 5 bp for EADS. The 50% figure
   appears to refer to relative reduction for stocks with large classical
   tracking errors (e.g., CAP GEMINI: 23.23% to 14.91%, a 35.8% relative
   reduction). Nine stocks show slight deterioration, but for 7 of these the
   deterioration is less than 1 bp.
   (Reference: BDF 2008, Tables 6-7; Section 5, p.1722)

6. **Factor structure:** Typically r = 1 to 3 common factors should suffice. The
   first factor alone explains most of the common variation (the U-shape). If
   r > 5 is selected, suspect data quality issues or insufficient
   cross-sectional breadth.
   (Researcher inference based on standard PCA results for financial data)

### Sanity Checks

1. **Common component accounts for most variance:** The ratio
   Var(C_hat) / Var(X) should be substantial (> 0.5 for most stocks). If the
   common component explains less than 30% of variance for most stocks, the
   PCA extraction is likely misconfigured or the cross-section is too narrow.
   (Researcher inference; motivated by BDF 2008 Section 3.1 showing the common
   component closely tracks the overall turnover shape)

2. **Specific component is mean-zero:** mean(E_hat[:, i]) should be
   approximately zero for each stock i. This follows mechanically from PCA
   (the common component absorbs the mean), but verifying it catches data
   alignment errors.
   (Reference: BDF 2008, Eq. 2-3 -- by construction)

3. **U-method baseline reproduces known results:** Before testing the full model,
   verify the U-method benchmark: x_hat_{i,j} = mean of turnover at bin j
   across prior L_days days. This should produce the standard U-shaped
   intraday profile.
   (Reference: Szucs 2017, Section 3, Eq. for U-method benchmark)

4. **SETAR threshold is interior:** The estimated threshold tau for each stock
   should lie strictly between the 15th and 85th percentile of the specific
   component series (by construction of the grid search). If tau clusters near
   the grid boundaries for many stocks, consider widening the grid.
   (Researcher inference; standard SETAR diagnostic)

5. **SETAR outperforms AR(1) on most stocks:** Across a diversified
   cross-section, SETAR should produce lower MAPE than AR(1) for at least 70%
   of stocks.
   (Reference: BDF 2008, Section 3.2 -- SETAR wins for the large majority of
   stocks tested. Note: BDF 2008 text says "three of the 31" stocks favor
   ARMA, but Table 1 lists 39 stocks -- see Variants section for discussion
   of this discrepancy.)

6. **Dynamic execution outperforms static:** The dynamic VWAP strategy
   (re-forecasting after each bin) should have lower tracking error than both
   static execution and the classical U-method. Static execution should
   actually be worse than classical.
   (Reference: BDF 2008, Section 4.2.1-4.2.2)

7. **Eigenvalue spectrum:** Plot the eigenvalues from PCA. There should be a
   clear gap between the first r eigenvalues and the remaining ones (the
   "scree plot" elbow). If the spectrum decays smoothly without a gap, the
   factor structure may be weak.
   (Researcher inference; standard PCA diagnostic, consistent with Bai & Ng
   2002 methodology)

8. **Toy example for verification:** Verify the PCA step on a small synthetic
   dataset. For example, construct a (T=8, N=3) matrix X where:
   - X = F @ A + E, with F a (8,1) column of known values [1,2,3,4,4,3,2,1]',
     A = [0.5, 0.3, 0.4], and E a small random noise matrix.
   - Run PCA with r=1. The recovered F_hat should be proportional to F (up to
     sign), and C_hat = F_hat @ A_hat should closely approximate the true F@A.
   - The specific component E_hat should be close to the true E.
   This verifies the eigendecomposition, normalization, and factor recovery
   steps are correctly implemented.
   (Researcher inference)

### Edge Cases

1. **Zero-volume bins:** If any stock has zero volume in a bin, its turnover is
   zero. This does not cause mathematical errors (unlike log-based models) but
   may distort PCA results if many zeros are present. The papers require all
   bins to have nonzero volume.
   (Reference: Szucs 2017, Section 2 -- "every stock had trades and thus a
   volume record larger than zero in every 15-minute interval")

   **Handling:** Exclude stocks with frequent zero-volume bins from the
   cross-section. Alternatively, replace isolated zeros with a small positive
   value (e.g., 1 share traded) or interpolate from adjacent bins. Log a
   warning when this occurs.

2. **Negative turnover forecasts:** The additive structure means x_hat = c + e
   can produce negative values if the specific component forecast is large
   and negative. Raw turnover cannot be negative.
   **Handling:** Floor negative forecasts to a small positive value (e.g.,
   1e-8) and re-normalize volume shares. Log a warning.
   (Researcher inference -- BDF 2008 does not address this explicitly, but it
   is an inherent limitation of additive decompositions on non-negative data)

3. **Regime change in cross-section (stock additions/removals):** If the
   composition of the stock universe changes (IPOs, delistings, index
   rebalancing), the PCA factor loadings change. This is handled automatically
   by the rolling window since the window covers only 20 days, but care is
   needed to ensure the data matrix X has consistent columns across the window.
   **Handling:** Maintain a stable universe within each estimation window. When
   a stock exits (delisting, halt), remove it from the cross-section for the
   entire window. When a stock enters, wait until it has a full window of data
   before including it.
   (Researcher inference)

4. **Half-days and early closures (e.g., day before holidays):** These days have
   fewer than k bins.
   **Handling:** Exclude half-days from the estimation window entirely, as the
   BDF paper does (BDF 2008, Section 3.1 -- "The 24th and 31st of December
   2003 were excluded from the sample").

5. **Extreme volume events (earnings, index rebalancing):** On days with
   extremely unusual volume patterns, the specific component may be very large,
   pushing the SETAR into the high regime persistently.
   **Handling:** No special handling is needed -- the SETAR model is designed to
   capture exactly this kind of regime switching. However, consider excluding
   such extreme days from the estimation window if they are rare enough to
   distort parameter estimates.
   (Researcher inference)

6. **Cross-sectional dimension too small:** If N < 20, PCA factor extraction may
   be unreliable. The Bai (2003) consistency results require N -> infinity.
   **Handling:** Ensure the cross-section includes at least 30 stocks. If
   fewer stocks are available, consider augmenting with sector ETFs or using
   a longer estimation window (more rows in X).
   (Reference: BDF 2008 uses N=39; Szucs 2017 uses N=33; Bai 2003 requires
   large N)

### Known Limitations

1. **Requires a cross-section of stocks:** The model cannot be applied to a
   single stock in isolation. It fundamentally requires a panel of N stocks
   to extract common factors. This is a structural limitation, not a
   parameter choice.
   (Reference: BDF 2008, Section 2.2 -- the PCA requires a cross-sectional
   dimension)

2. **Static execution is useless:** The model cannot produce useful full-day
   forecasts at market open. Multi-step AR/SETAR forecasts decay to the
   unconditional mean (zero for the specific component), making the static
   forecast equivalent to using only the common component.
   (Reference: BDF 2008, Section 4.2.1 -- "the simplicity of this strategy
   is offset by the poor quality of the long-term estimates")

3. **No daily volume forecast:** The model forecasts intraday volume shares
   (how volume is distributed within a day) but does not forecast the total
   daily volume level. For full VWAP execution, a separate daily volume
   forecast is needed to convert shares into absolute quantities.
   (Reference: BDF 2008, Section 4.1.3 -- the theoretical VWAP execution
   requires knowing end-of-day total volume, which is unavailable in real time)

4. **No price impact modeling:** The model assumes the trader's order is small
   relative to market volume. For large orders, price impact effects are not
   captured.
   (Reference: BDF 2008, Section 5 -- "in order to beat the VWAP, our
   price-adjusted-volume model is not sufficient")

5. **Linear factor model:** The PCA imposes a linear factor structure. Nonlinear
   common effects (e.g., contagion during market crashes) are not captured in
   the common component and instead contaminate the specific component.
   (Researcher inference)

6. **SETAR threshold is static within each window:** The threshold tau is
   estimated once per stock per daily re-estimation and does not adapt
   intraday. Intraday regime changes in the specific component's behavior are
   not captured beyond what a fixed two-regime model provides.
   (Reference: BDF 2008, Eq. 11 -- tau is a fixed parameter per estimation)

7. **Turnover normalization requires shares outstanding data:** Computing
   turnover requires total shares outstanding (TSO), which must be sourced
   from an external provider (Bloomberg, CRSP, etc.) and adjusted for splits
   and corporate actions.
   (Reference: BDF 2008, Section 3.1; Szucs 2017, Section 2)

8. **Overnight boundary contamination:** The AR(1)/SETAR is fit to the
   concatenated multi-day specific component series without special handling
   at day boundaries. Overnight jumps may inflate innovation variance and
   attenuate persistence estimates. See Calibration section for discussion.
   (Researcher inference; BDF 2008 does not address this issue)

## Paper References

| Spec Section | Paper Source | Specific Location |
|---|---|---|
| Model decomposition x = c + e | BDF 2008 | Section 2.1-2.2, Eq. (2)-(5) |
| PCA estimation via eigendecomposition of X'X | BDF 2008 | Section 2.2, Eq. (6), text below Eq. (6); citing Bai (2003) |
| Factor count selection | BDF 2008 | Section 2.2; citing Bai & Ng (2002) |
| Common component forecast (time-of-day average) | BDF 2008 | Section 2.3, Eq. (9) |
| AR(1) specific component model (labeled "ARMA(1,1)" in paper) | BDF 2008 | Section 2.3, Eq. (10) |
| SETAR specific component model | BDF 2008 | Section 2.3, Eq. (11) |
| Estimation by maximum likelihood | BDF 2008 | Section 2.3, paragraph after Eq. (11) |
| Dynamic VWAP execution strategy | BDF 2008 | Section 4.2.2 |
| Static execution is dominated | BDF 2008 | Section 4.2.1 |
| SETAR outperforms AR(1) for majority of stocks | BDF 2008 | Section 3.2 ("three of the 31 companies") |
| 31 vs 39 stock count discrepancy | BDF 2008 | Table 1 (39 stocks) vs Section 3.2 text ("31") |
| VWAP tracking: 10.06% classical, 8.98% dynamic SETAR | BDF 2008 | Table 2, panel 3, out-of-sample |
| Per-stock VWAP results, up to ~36% relative reduction | BDF 2008 | Tables 6-7; Section 5 (p.1722) |
| BDF outperforms CMEM (BCG) | Szucs 2017 | Section 4, full-sample results |
| Computational advantage of BDF | Szucs 2017 | Section 4 (~2 hours vs ~60 machine-days) |
| 15-minute bins, 26 per day, DJIA validation | Szucs 2017 | Section 2, Table 1 |
| 20-minute bins, 25 per day, CAC40 validation | BDF 2008 | Section 3.1 |
| Turnover = V / TSO normalization | BDF 2008 | Section 3.1; Szucs 2017, Section 2 |
| Rolling 20-day estimation window | BDF 2008 | Section 3.1; Szucs 2017, Section 3 |
| MSE and MAPE error metrics | Szucs 2017 | Section 3 |
| Demeaning not required (Bai 2003 framework) | Researcher inference | Based on Bai (2003) and BDF 2008 Eq. (6) formulation |
| Day-start e_forecast = 0 initialization | Researcher inference | Consistent with BDF 2008 Section 4.2.1 |
| Multi-step SETAR via deterministic iteration | Researcher inference | Standard approximation |
| Negative forecast handling | Researcher inference | -- |
| Factor count stability heuristic | Researcher inference | -- |
| Zero-volume bin handling | Researcher inference | Based on Szucs 2017 Section 2 requirement |
| Cross-day boundary contamination | Researcher inference | -- |
| SETAR grid search range and min regime size | Researcher inference | Based on Hansen 1997, Tong 1990 |
| Toy example for verification | Researcher inference | -- |

## Revision History

### Draft 2 Changes (in response to Critique 1)

**Major issues addressed:**

- **M1 (Eigendecomposition matrix):** Rewrote PCA Step 3 to use X'X / T (an
  N x N matrix) per BDF 2008 p.1712, instead of XX' / (NT). Added detailed
  normalization procedure showing how to recover F_hat from X'X eigenvectors
  with F'F/T = I_r normalization. Also noted SVD as a numerically stable
  alternative. Updated Data Flow diagram accordingly.

- **M2 (AR vs ARMA labeling):** Replaced all instances of "ARMA(1,1)" with
  "AR(1)" throughout the document. Added explicit note in Overview explaining
  that BDF 2008's "ARMA(1,1)" label is a misnomer (no MA term in Eq. 10),
  and that Szucs 2017 correctly uses "AR(1)". Paper References table notes
  the discrepancy.

- **M3 (Estimation method):** Updated Initialization points 3-4 to note that
  BDF 2008 explicitly specifies MLE. Added justification that OLS is
  equivalent to conditional MLE for AR(1) under Gaussian errors. For SETAR,
  described the procedure as "profile MLE / conditional MLE with grid search"
  and noted equivalence to conditional OLS under Gaussian assumptions, citing
  Hansen 1997 and Tong 1990.

- **M4 (Dynamic VWAP multi-step forecast gap):** Added Step 4 to Phase 2
  pseudocode computing multi-step forecasts for all remaining bins
  (j_current+2 through k_bins) via iterated AR(1)/SETAR recursion. Added
  Step 5 combining common and specific forecasts for all remaining bins.
  Rewrote Step 6 (volume shares) to use the complete set of remaining-bin
  forecasts as denominator. Documented the deterministic iteration
  approximation for multi-step SETAR as Researcher inference. Updated Data
  Flow to show multi-step forecast computation.

- **M5 (VWAP tracking error claims):** Removed unverified ">50% for
  high-volatility stocks" claim. Replaced with precise per-stock data from
  Tables 6-7 (CAP GEMINI: ~8 bp improvement, ~36% relative reduction). Cited
  the conclusion's "can even reach 50%" statement with the caveat that
  Table data shows max ~36%. Noted that 9 stocks show deterioration but 7
  of these by less than 1 bp.

- **M6 (31 vs 39 stock count):** Added explicit note in Variants section
  flagging the internal discrepancy in BDF 2008 (Table 1 lists 39 stocks,
  Section 3.2 text says "31"). Stated that the spec cites the paper's own
  numbers without resolving the inconsistency.

**Minor issues addressed:**

- **m1 (Turnover range):** Widened stated typical range from "1e-5 to 1e-2"
  to "1e-5 to 5e-2" with citation to BDF 2008 Table 1 (mean 0.0166, Q95
  0.0445).

- **m2 (Demeaning before PCA):** Added explicit note in Phase 1 Step 1 and
  Initialization point 2 clarifying that demeaning is NOT required in the
  Bai (2003) framework and that BDF 2008 does not include it.

- **m3 (First bin initialization):** Added Step 1 to Phase 2 handling
  j_current = 0 case (start of day). Added Initialization point 5 specifying
  e_forecast = 0 at day start with justification.

- **m4 (V(r) notation):** Renamed the residual variance in the IC section
  from V(r) to sigma_sq(r) to distinguish from the PCA objective function
  V(r) in BDF 2008 Eq. (6).

- **m5 (r_max in parameter table):** Added r_max to the parameter table
  with recommended value 10, low sensitivity, range 5-20.

- **m6 (SETAR grid search range):** Narrowed grid search range from
  10th-90th to 15th-85th percentile. Added min_regime_frac parameter
  (default 0.15) to the parameter table. Added minimum regime size
  constraint to SETAR estimation procedure.

- **m7 (Cross-day boundary):** Added "Cross-day boundary handling" subsection
  in Calibration section documenting the issue and the decision to follow
  the paper (fit concatenated series). Added Known Limitation #8 about
  overnight boundary contamination.

- **m8 (Worked example):** Added Sanity Check #8 with a toy PCA verification
  example (T=8, N=3 matrix with known factor structure).
