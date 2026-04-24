# Implementation Specification: PCA Factor Decomposition (BDF) for Intraday Volume

## Overview

The BDF model (Bialkowski, Darolles, and Le Fol 2008) decomposes intraday stock
turnover into a market-wide common component and a stock-specific idiosyncratic
component using principal components analysis (PCA) across a cross-section of
stocks. The common component captures the shared U-shaped intraday volume
pattern; the specific component captures stock-level deviations and is modeled
by either ARMA(1,1) or SETAR time-series processes. SETAR is the recommended
variant based on both the original paper and the independent validation by
Szucs (2017), which found it more accurate than ARMA and the competing CMEM
model on 11 years of DJIA data while being orders of magnitude faster to
estimate.

The model produces one-step-ahead intraday volume forecasts updated dynamically
as each bin's actual volume is observed, enabling dynamic VWAP execution
strategies that reduce tracking error by 10% on average versus historical
averages.

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
    // Reference: BDF 2008, Section 2.2, Eq. (4)-(6)
    X = turnover_history  // shape: (T, N)
    T = L_days * k_bins
    N = n_stocks

    // Step 2: Select number of common factors r using Bai & Ng (2002) IC
    // Reference: BDF 2008, Section 2.2; Bai & Ng (2002)
    r = select_number_of_factors(X, T, N)

    // Step 3: Extract factors and loadings via PCA (Bai 2003)
    // Reference: BDF 2008, Section 2.2, Eq. (6)
    // Solve: min_{F,A} (NT)^{-1} sum_i sum_t (X_{it} - A_i' F_t)^2
    // with normalization F'F/T = I_r
    // Solution: F_hat columns are sqrt(T) times the eigenvectors
    //   corresponding to the r largest eigenvalues of X X' / (NT)
    // A_hat = (F_hat' F_hat)^{-1} F_hat' X = F_hat' X / T
    
    eigenvalues, eigenvectors = eigendecomposition(X @ X.T / (N * T))
    // Take r largest eigenvalues
    F_hat = sqrt(T) * eigenvectors[:, :r]          // shape: (T, r)
    A_hat = (F_hat.T @ F_hat)^{-1} @ F_hat.T @ X  // shape: (r, N)
    // Simplifies to: A_hat = F_hat.T @ X / T

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
    FOR i = 1 TO N:
        e_series = E_hat[:, i]    // shape: (T,)

        // Option A: ARMA(1,1) -- Reference: BDF 2008, Eq. (10)
        // e_{i,t} = psi_1 * e_{i,t-1} + psi_2 + epsilon_{i,t}
        //   where epsilon_{i,t} is white noise
        // Note: This is actually an AR(1) with intercept as written in
        //   the paper. The "ARMA(1,1)" label in the paper summary may
        //   refer to the presence of the constant term.
        arma_params[i] = fit_ar1(e_series)
        // arma_params[i] = (psi_1, psi_2, sigma_epsilon)

        // Option B: SETAR -- Reference: BDF 2008, Eq. (11)
        // e_{i,t} = (phi_11 * e_{i,t-1} + phi_12) * I(e_{i,t-1} <= tau)
        //         + (phi_21 * e_{i,t-1} + phi_22) * (1 - I(e_{i,t-1} <= tau))
        //         + epsilon_{i,t}
        setar_params[i] = fit_setar(e_series)
        // setar_params[i] = (phi_11, phi_12, phi_21, phi_22, tau, sigma_eps)

    RETURN (c_forecast, arma_params OR setar_params, A_hat, F_hat, r)
```

#### Phase 2: Intraday Dynamic Forecasting (run at each bin during the day)

```
FUNCTION forecast_intraday(observed_turnover_today, j_current,
                           c_forecast, ts_params, model_type,
                           A_hat_last, r, k_bins):
    // observed_turnover_today: array of shape (j_current, N)
    //   actual turnover observed so far today (bins 1..j_current)
    // j_current: the most recently completed bin index (1-based)
    // c_forecast: precomputed common component forecast, shape (N, k_bins)
    // ts_params: fitted ARMA or SETAR parameters per stock
    // model_type: "ARMA" or "SETAR"

    // Step 1: Extract today's specific component from observed data
    // Reference: BDF 2008, Section 2.2 (filtering step, Eq. 7-8)
    // Use yesterday's factor loadings to decompose today's observations
    // The common component forecast was already computed overnight
    FOR i = 1 TO N:
        e_observed[i, j_current] = observed_turnover_today[j_current, i]
                                   - c_forecast[i, j_current]

    // Step 2: Produce one-step-ahead specific component forecast
    // Reference: BDF 2008, Section 2.3, Eq. (10) or (11)
    FOR i = 1 TO N:
        e_last = e_observed[i, j_current]

        IF model_type == "ARMA":
            psi_1, psi_2 = ts_params[i]
            e_forecast[i] = psi_1 * e_last + psi_2

        ELSE IF model_type == "SETAR":
            phi_11, phi_12, phi_21, phi_22, tau = ts_params[i]
            IF e_last <= tau:
                e_forecast[i] = phi_11 * e_last + phi_12
            ELSE:
                e_forecast[i] = phi_21 * e_last + phi_22

    // Step 3: Combine common and specific forecasts
    // Reference: BDF 2008, Section 2.3, below Eq. (9)
    FOR i = 1 TO N:
        x_forecast[i, j_current + 1] = c_forecast[i, j_current + 1]
                                        + e_forecast[i]

    // Step 4: Recompute volume shares for remaining bins
    // For VWAP execution, convert turnover forecasts to shares
    // Reference: BDF 2008, Section 4.2 (dynamic VWAP)
    // Remaining bins j_current+1 .. k_bins use the forecast;
    // completed bins 1..j_current use actuals
    FOR i = 1 TO N:
        total_est = sum(observed_turnover_today[:, i])  // actual so far
                  + sum(x_forecast_remaining[i, j_current+1 : k_bins])
        // Volume share for next bin:
        w_forecast[i, j_current + 1] = x_forecast[i, j_current + 1] / total_est

    RETURN x_forecast[:, j_current + 1], w_forecast[:, j_current + 1]
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
    +---> PCA eigendecomposition of X X' / (NT)
    |         |
    |         +---> F_hat: (T, r) factor matrix
    |         +---> A_hat: (r, N) loading matrix
    |         |
    |         +---> C_hat = F @ A: (T, N) common component
    |         +---> E_hat = X - C: (T, N) specific component
    |
    +---> Common forecast: c_forecast[i,j] = mean of bin j across L_days
    |         shape: (N, k)
    |
    +---> Specific model fit: ARMA or SETAR per stock on E_hat[:,i]
    |         ARMA: 3 params per stock (psi_1, psi_2, sigma)
    |         SETAR: 6 params per stock (phi_11,phi_12,phi_21,phi_22,tau,sigma)
    |
    v
[Phase 2: Intraday Forecast Loop]
    |
    For each completed bin j = 1, 2, ..., k-1:
        observed x_{i,j} --> e_{i,j} = x_{i,j} - c_forecast[i,j]
            --> e_hat_{i,j+1} via ARMA/SETAR
            --> x_hat_{i,j+1} = c_forecast[i,j+1] + e_hat_{i,j+1}
            --> w_hat_{i,j+1} = x_hat / estimated_daily_total
    |
    v
Output: per-stock, per-bin turnover forecasts and volume shares
    shape: scalar per stock per bin (streamed one bin at a time)
```

Type information:
- All turnover values: float64 (typically in range 1e-5 to 1e-2)
- Factor matrix F_hat: float64 array of shape (T, r) where r is typically 1-3
- Loading matrix A_hat: float64 array of shape (r, N)
- ARMA params per stock: tuple of 3 float64
- SETAR params per stock: tuple of 6 float64 (including threshold tau)
- Volume share forecasts: float64 in [0, 1], sum to 1 across bins within a day

### Variants

**Implemented variant: BDF with SETAR specific component (BDF-SETAR).**

Rationale: The SETAR variant is chosen as the primary implementation because:

1. BDF 2008 (Section 3.2, Tables 2-5) shows SETAR outperforms ARMA for 36 of
   39 CAC40 stocks on MAPE, with the 3 stocks where ARMA wins having negligible
   margins.
2. Szucs 2017 (Section 4, full-sample results table) independently confirms on
   33 DJIA stocks over 11 years that BDF-SETAR achieves the best MAPE (0.399)
   versus BDF-AR (0.403) and the competing CMEM model (0.402).
3. By MSE, BDF-AR slightly edges BDF-SETAR (6.49e-4 vs 6.60e-4 in Szucs 2017),
   but the difference is small and MAPE is the more relevant metric for VWAP
   execution (it penalizes proportional errors equally across volume levels).

The BDF-ARMA variant should also be implemented as a simpler fallback, as it
requires fewer parameters and avoids the threshold estimation step.

**Not implemented:** The static VWAP execution strategy is explicitly excluded.
BDF 2008 (Section 4.2.1) demonstrates that static execution (forecasting all
bins at market open) performs worse than even the naive historical average
because multi-step ARMA/SETAR forecasts decay toward zero, neutralizing the
specific component. Only dynamic execution (updating after each observed bin)
is viable.

## Parameters

| Parameter | Description | Recommended Value | Sensitivity | Range |
|-----------|-------------|-------------------|-------------|-------|
| k | Number of intraday bins per trading day | 26 (15-min bins, 9:30-16:00) | Low -- determined by bin width choice; 25 (20-min) also validated | 13-52 (30-min to 5-min) |
| L_days | Rolling window length for PCA estimation and common component averaging (trading days) | 20 | Medium -- too short loses seasonal stability, too long misses regime changes | 10-60 |
| r | Number of common factors extracted by PCA | Estimated via Bai & Ng (2002) IC; typically 1-3 | Medium -- over-extraction adds noise to specific component, under-extraction leaves common structure in residuals | 1-5 |
| psi_1 | AR(1) coefficient for specific component (ARMA variant) | Estimated per stock per window | High -- controls persistence of specific component; typically 0.3-0.8 | (-1, 1) for stationarity |
| psi_2 | Intercept for specific component (ARMA variant) | Estimated per stock per window | Low -- typically near zero since specific component is mean-zero by construction | (-inf, inf) |
| tau | SETAR threshold on lagged specific component | Estimated per stock per window | High -- determines regime boundary between calm and turbulent volume states | Range of e_{i,t} values |
| phi_11 | AR coefficient, low regime (SETAR) | Estimated per stock | Medium | (-1, 1) |
| phi_12 | Intercept, low regime (SETAR) | Estimated per stock | Low | (-inf, inf) |
| phi_21 | AR coefficient, high regime (SETAR) | Estimated per stock | Medium | (-1, 1) |
| phi_22 | Intercept, high regime (SETAR) | Estimated per stock | Low | (-inf, inf) |
| N | Number of stocks in cross-section | >= 30 | Medium -- PCA asymptotics require large N; Bai (2003) consistency holds as N,T -> inf jointly | 30-500+ |

### Initialization

1. **Data preparation:** Collect at least L_days + 1 trading days of tick data
   for all N stocks. Aggregate into k bins per day. Compute turnover by
   dividing each bin's volume by total shares outstanding.
   (Reference: BDF 2008, Section 3.1; Szucs 2017, Section 2)

2. **PCA initialization:** No special initialization needed. The
   eigendecomposition of X X' / (NT) is a deterministic computation with a
   unique solution (up to sign/rotation of eigenvectors, which does not affect
   the common component C = F A). Use standard numerical linear algebra
   (e.g., numpy.linalg.eigh or scipy.linalg.eigh for symmetric matrices).
   (Reference: BDF 2008, Section 2.2, Eq. 6; Bai 2003)

3. **ARMA initialization:** Standard maximum likelihood initialization. The AR(1)
   model has only 2 parameters (psi_1, psi_2) and is trivially estimated by
   OLS regression of e_{i,t} on e_{i,t-1}.
   (Reference: BDF 2008, Section 2.3, Eq. 10)

4. **SETAR initialization:** The threshold tau must be searched over a grid of
   candidate values. A standard approach:
   - Sort the specific component series e_{i,1}, ..., e_{i,T}.
   - Consider candidate thresholds at each unique value (or a percentile grid,
     e.g., 10th to 90th percentile in 1% steps).
   - For each candidate tau, estimate the two sets of AR(1) coefficients by OLS
     on the two subsamples split at tau.
   - Select tau that minimizes the sum of squared residuals across both regimes.
   (Reference: BDF 2008, Section 2.3, Eq. 11; standard SETAR estimation per
   Hansen 1997, Tong 1990)

5. **First-day forecast:** On the very first day of live operation, no prior
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

4. Fit ARMA(1,1) or SETAR to each stock's specific component series.

5. Store the model parameters for use during day d+1's intraday forecast loop.

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

1. For each candidate r from 1 to r_max (e.g., r_max = 10):
   a. Compute the PCA decomposition with r factors.
   b. Compute the residual variance:
      V(r) = (NT)^{-1} * sum_i sum_t (X_{it} - A_i' F_t)^2
   c. Compute the penalty term. Bai & Ng (2002) propose several criteria;
      the most commonly used are:
      - IC_p1(r) = ln(V(r)) + r * ((N + T) / (NT)) * ln(NT / (N + T))
      - IC_p2(r) = ln(V(r)) + r * ((N + T) / (NT)) * ln(min(N, T))

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
   - BDF-ARMA: mean MAPE 0.0829 (Reference: BDF 2008, Table 2, column 1)
   - Classical average: mean MAPE 0.0905 (Reference: BDF 2008, Table 2)

   Note: The large difference in MAPE scale between the two papers (0.07-0.09
   vs 0.40-0.50) likely reflects different MAPE definitions or data
   characteristics. BDF 2008 reports per-interval MAPE across the portfolio;
   Szucs reports per-interval per-stock MAPE. Both show the same relative
   ordering: BDF-SETAR > BDF-ARMA > U-method.

4. **MSE:** For 33 DJIA stocks at 15-minute resolution:
   - BDF-AR: MSE 6.49e-4 (Reference: Szucs 2017, full-sample table)
   - BDF-SETAR: MSE 6.60e-4 (Reference: Szucs 2017, full-sample table)
   - U-method: MSE 1.02e-3 (Reference: Szucs 2017, full-sample table)
   Both BDF variants reduce MSE by approximately 35% versus the U-method.

5. **VWAP tracking error:** Dynamic PCA-SETAR execution reduces out-of-sample
   VWAP tracking error (MAPE) from 10.06% (classical) to 8.98% (dynamic
   BDF-SETAR) -- approximately a 10% relative reduction. For high-volatility
   stocks, reductions can exceed 50%.
   (Reference: BDF 2008, Table 2, panel 3; Section 4.3)

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
   across prior L_days days. This should produce the standard U-shaped intraday
   profile.
   (Reference: Szucs 2017, Section 3, Eq. for U-method benchmark)

4. **SETAR threshold is interior:** The estimated threshold tau for each stock
   should lie strictly between the minimum and maximum of the specific component
   series (not at an endpoint). If tau is at an endpoint, the SETAR degenerates
   to AR(1) -- this is not necessarily wrong but should be logged.
   (Researcher inference; standard SETAR diagnostic)

5. **SETAR outperforms ARMA on most stocks:** Across a diversified cross-section,
   SETAR should produce lower MAPE than ARMA for at least 70% of stocks.
   (Reference: BDF 2008, Section 3.2 -- SETAR wins for 36/39 CAC40 stocks;
   Szucs 2017 -- SETAR wins on aggregate MAPE for DJIA)

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
   forecasts at market open. Multi-step ARMA/SETAR forecasts decay to the
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

## Paper References

| Spec Section | Paper Source | Specific Location |
|---|---|---|
| Model decomposition x = c + e | BDF 2008 | Section 2.1-2.2, Eq. (2)-(5) |
| PCA estimation via eigendecomposition | BDF 2008 | Section 2.2, Eq. (6); citing Bai (2003) |
| Factor count selection | BDF 2008 | Section 2.2; citing Bai & Ng (2002) |
| Common component forecast (time-of-day average) | BDF 2008 | Section 2.3, Eq. (9) |
| ARMA(1,1) specific component model | BDF 2008 | Section 2.3, Eq. (10) |
| SETAR specific component model | BDF 2008 | Section 2.3, Eq. (11) |
| Dynamic VWAP execution strategy | BDF 2008 | Section 4.2.2 |
| Static execution is dominated | BDF 2008 | Section 4.2.1 |
| SETAR outperforms ARMA | BDF 2008 | Section 3.2, Table 2 |
| BDF outperforms CMEM (BCG) | Szucs 2017 | Section 4, full-sample results |
| Computational advantage of BDF | Szucs 2017 | Section 4 (~2 hours vs ~60 machine-days) |
| 15-minute bins, 26 per day, DJIA validation | Szucs 2017 | Section 2, Table 1 |
| 20-minute bins, 25 per day, CAC40 validation | BDF 2008 | Section 3.1 |
| Turnover = V / TSO normalization | BDF 2008 | Section 3.1; Szucs 2017, Section 2 |
| Rolling 20-day estimation window | BDF 2008 | Section 3.1; Szucs 2017, Section 3 |
| MSE and MAPE error metrics | Szucs 2017 | Section 3 |
| VWAP tracking error results | BDF 2008 | Section 4.3, Tables 2-7 |
| Negative forecast handling | Researcher inference | -- |
| Factor count stability heuristic | Researcher inference | -- |
| Zero-volume bin handling | Researcher inference | Based on Szucs 2017 Section 2 requirement |
