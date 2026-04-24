# Implementation Specification: PCA Factor Decomposition (BDF) for Intraday Volume

## Overview

The BDF model (Bialkowski, Darolles, Le Fol 2008) decomposes intraday trading
volume into a market-wide common component and a stock-specific residual using
Principal Component Analysis (PCA). The common component captures the well-known
U-shaped intraday seasonal pattern shared across stocks. The specific component
captures each stock's idiosyncratic deviation from this pattern and is modeled
dynamically using either an AR(1) or SETAR time-series model. The combined
forecast enables dynamic VWAP execution strategies that update predictions
intraday as new volume observations arrive.

The model operates on turnover (volume normalized by shares outstanding or float)
aggregated into fixed intraday bins. It uses a rolling estimation window and
produces one-step-ahead forecasts that are updated within the trading day. Szucs
(2017) independently validated the BDF model on 11 years of US equity data,
confirming it outperforms the competing BCG (multiplicative error) model on both
accuracy and computational efficiency.

## Algorithm

### Model Description

The model takes as input a panel of intraday turnover observations for N stocks
over L trading days, with k bins per day, producing per-stock, per-bin volume
forecasts for the next trading day.

**Inputs:**
- Intraday volume data for N stocks, aggregated into k equally-spaced bins per
  trading day.
- Shares outstanding (or float) for each stock, to compute turnover.

**Outputs:**
- Per-stock, per-bin turnover forecasts for the upcoming trading day.
- During the day: updated forecasts incorporating observed volume.

**Key assumption:** The cross-section of intraday turnover shares a low-dimensional
common factor structure that can be extracted by PCA. The remaining stock-specific
residual exhibits short-memory serial correlation amenable to autoregressive
modeling. Researcher inference: typical factor counts are 1-3 for intraday volume
data, based on the low-dimensional structure of intraday volume patterns and the
dominance of the U-shaped seasonal.

### Pseudocode

The algorithm has four phases: data preparation, PCA factor extraction,
component forecasting, and dynamic VWAP execution.

#### Phase 1: Data Preparation

```
FUNCTION prepare_data(raw_volume, shares_outstanding, k):
    """
    Convert raw volume into turnover and organize into the matrix X.

    Reference: BDF 2008, Section 2.2; Szucs 2017, Section 2, p.4.
    """
    # Step 1.1: Compute turnover for each stock i, each bin
    # BDF 2008 uses float (panel structure shares N_s);
    # Szucs 2017 uses TSO from Bloomberg. Either works.
    # TSO is recommended for implementation as it is more widely available.
    FOR each stock i, each bin observation:
        turnover[i] = volume[i] / shares_outstanding[i]

    # Step 1.2: Organize into matrix X of shape (P, N)
    # P = L * k = total number of intraday observations per stock
    # N = number of stocks
    # Rows are time-ordered: day 1 bin 1, day 1 bin 2, ..., day 1 bin k,
    #                        day 2 bin 1, ..., day L bin k
    # Columns are stocks
    X = array of shape (P, N)
    FOR each stock i (column):
        X[:, i] = concatenated turnover for stock i across all L days and k bins

    # Step 1.3: Critical — do NOT demean columns of X
    # In the Bai (2003) large-dimensional factor framework, column means
    # are absorbed into the factor loadings. Demeaning would strip the
    # level information needed to capture the U-shaped seasonal pattern.
    # Reference: Bai 2003, Section 2; BDF 2008, Eq. (6) — no demeaning step.

    RETURN X  # shape (P, N)
```

#### Phase 2: PCA Factor Extraction and Factor Count Selection

```
FUNCTION extract_factors(X, r_max=10):
    """
    Extract r common factors from X using truncated SVD, selecting r via
    the Bai & Ng (2002) IC_p2 information criterion.

    Reference: BDF 2008, Section 2.2, Eq. (6); Bai & Ng 2002, Eq. (IC_p2).
    """
    P, N = X.shape

    # Step 2.1: Compute total sum of squares for later use
    total_ss = sum(X ** 2)

    # Step 2.2: Compute truncated SVD of X up to r_max components
    # X = U @ diag(s) @ Vt, where:
    #   U has shape (P, r_max), columns are left singular vectors
    #   s has shape (r_max,), singular values in descending order
    #   Vt has shape (r_max, N), rows are right singular vectors
    U, s, Vt = truncated_svd(X, n_components=r_max)

    # Step 2.3: Select factor count r by minimizing IC_p2
    # Reference: Bai & Ng 2002; BDF 2008, Section 2.2.
    best_ic = +infinity
    best_r = 1
    FOR r = 1, 2, ..., r_max:
        # Residual variance from rank-r approximation
        # V(r) = (1 / (N * P)) * (total_ss - sum of top r squared singular values)
        V_r = (total_ss - sum(s[0:r] ** 2)) / (N * P)

        # Guard against floating-point issues
        IF V_r <= 0:
            V_r = 1e-15

        # IC_p2 penalty term
        # Reference: Bai & Ng 2002, Eq. for IC_p2
        penalty = r * ((N + P) / (N * P)) * ln(min(N, P))

        ic = ln(V_r) + penalty

        IF ic < best_ic:
            best_ic = ic
            best_r = r

    r = best_r
    # Researcher inference: for liquid equity markets, expect r = 1-3,
    # reflecting the dominance of the U-shaped seasonal and possibly
    # a second factor for open/close asymmetry.

    # Step 2.4: Extract normalized factors and loadings for chosen r
    # Normalization: F_hat' @ F_hat / P = I_r
    # This is achieved by: F_hat = sqrt(P) * U[:, 0:r]
    # Verification: F_hat.T @ F_hat / P = P * U.T @ U / P = I_r
    # (since U has orthonormal columns)
    # Reference: BDF 2008, Eq. (6); Bai 2003.
    F_hat = sqrt(P) * U[:, 0:r]          # shape (P, r)
    Lambda_hat = (Vt[0:r, :].T * s[0:r]) / sqrt(P)  # shape (N, r)

    # Step 2.5: Compute common and specific components
    # C_hat = F_hat @ Lambda_hat.T, shape (P, N)
    # e_hat = X - C_hat, shape (P, N)
    C_hat = F_hat @ Lambda_hat.T          # shape (P, N)
    e_hat = X - C_hat                     # shape (P, N)

    RETURN F_hat, Lambda_hat, C_hat, e_hat, r
```

#### Phase 3: Component Forecasting

```
FUNCTION forecast_common(C_hat, L, k, N):
    """
    Forecast the common component for the next trading day by averaging
    each time-of-day bin across the L days in the estimation window.

    Reference: BDF 2008, Section 2.3, Eq. (9).
    """
    # Step 3.1: Reshape C_hat from (P, N) to (L, k, N)
    # P = L * k
    C_3d = reshape(C_hat, (L, k, N))

    # Step 3.2: Average across days (axis 0) for each bin j and stock i
    # c_forecast[j, i] = (1/L) * sum_{l=1}^{L} C_3d[l, j, i]
    c_forecast = mean(C_3d, axis=0)       # shape (k, N)

    # This forecast is computed BEFORE the trading day begins.
    # It is FIXED for the entire day and NOT updated intraday.

    RETURN c_forecast  # shape (k, N)


FUNCTION fit_ar1(e_series):
    """
    Fit AR(1) with intercept to a specific component time series.

    Model: e_t = psi_1 * e_{t-1} + psi_2 + epsilon_t

    Despite BDF 2008 labeling this "ARMA(1,1)" in text, Eq. (10) contains
    no MA term. Szucs (2017) Eq. (5) independently confirms this is AR(1).

    Reference: BDF 2008, Section 2.3, Eq. (10); Szucs 2017, Section 4.1, Eq. (5).
    """
    # Step 3.3: Estimate by OLS (equivalent to MLE for AR(1) with Gaussian errors)
    # Construct regression: e[1:] = psi_1 * e[:-1] + psi_2
    y = e_series[1:]                    # dependent variable
    x = e_series[:-1]                   # lagged value
    ones = array_of_ones(len(y))        # intercept

    # OLS: [psi_1, psi_2] = (X'X)^{-1} X'y, where X = [x, ones]
    design = column_stack(x, ones)      # shape (T-1, 2)
    coeffs = lstsq(design, y)           # [psi_1, psi_2]

    psi_1 = coeffs[0]
    psi_2 = coeffs[1]

    # Compute residual variance for model comparison
    residuals = y - design @ coeffs
    sigma2 = var(residuals)

    RETURN psi_1, psi_2, sigma2


FUNCTION fit_setar(e_series, n_grid=100, min_regime_obs=10):
    """
    Fit SETAR (Self-Exciting Threshold AR) to a specific component time series.

    Model: e_t = (phi_11 * e_{t-1} + phi_12) * I(e_{t-1} <= tau)
               + (phi_21 * e_{t-1} + phi_22) * (1 - I(e_{t-1} <= tau))
               + epsilon_t

    Two regimes separated by threshold tau on the lagged value e_{t-1}.

    Reference: BDF 2008, Section 2.3, Eq. (11); Szucs 2017, Section 4.1, Eq. (6).

    NOTE on notation ordering between papers:
    - BDF 2008 Eq. (11) writes: (phi_11 * e_{t-1} + phi_12) — AR coefficient
      FIRST, intercept SECOND.
    - Szucs 2017 Eq. (6) writes: (c_{1,1} + theta_{1,2} * e_{p-1}) — intercept
      FIRST, AR coefficient SECOND.
    The two papers use OPPOSITE ordering of intercept and slope within each
    regime. This does not affect the mathematics (addition is commutative),
    but developers cross-referencing the papers must be aware.

    Parameter name mapping (this implementation follows BDF ordering):
      Regime 1 (below threshold):
        phi_11 (AR coeff) = Szucs' theta_{1,2}
        phi_12 (intercept) = Szucs' c_{1,1}
      Regime 2 (above threshold):
        phi_21 (AR coeff) = Szucs' theta_{2,2}
        phi_22 (intercept) = Szucs' c_{2,1}
    """
    y = e_series[1:]
    x_lag = e_series[:-1]
    T = len(y)

    # Step 3.4: Grid search over candidate thresholds
    # Use quantiles of e_{t-1} to define the search grid, excluding
    # extreme quantiles to ensure each regime has sufficient observations.
    # Reference: Researcher inference — standard SETAR estimation practice.
    tau_candidates = quantiles(x_lag, probs=linspace(0.15, 0.85, n_grid))

    best_ssr = +infinity
    best_tau = None
    best_params = None

    FOR tau in tau_candidates:
        # Partition observations into two regimes
        mask_low = (x_lag <= tau)
        mask_high = ~mask_low

        n_low = sum(mask_low)
        n_high = sum(mask_high)

        # Skip if either regime has too few observations
        IF n_low < min_regime_obs OR n_high < min_regime_obs:
            CONTINUE

        # OLS for regime 1 (below threshold): y = phi_11 * x + phi_12
        design_low = column_stack(x_lag[mask_low], ones(n_low))
        coeffs_low = lstsq(design_low, y[mask_low])

        # OLS for regime 2 (above threshold): y = phi_21 * x + phi_22
        design_high = column_stack(x_lag[mask_high], ones(n_high))
        coeffs_high = lstsq(design_high, y[mask_high])

        # Total sum of squared residuals
        resid_low = y[mask_low] - design_low @ coeffs_low
        resid_high = y[mask_high] - design_high @ coeffs_high
        ssr = sum(resid_low ** 2) + sum(resid_high ** 2)

        IF ssr < best_ssr:
            best_ssr = ssr
            best_tau = tau
            best_params = (coeffs_low, coeffs_high)

    # Step 3.5: Handle case where no valid threshold was found
    # This can occur when the estimation window is very short or the
    # residual distribution is highly concentrated, causing all candidates
    # to have too few observations in one regime.
    IF best_tau is None:
        RETURN None  # Signal failure; caller must fall back to AR(1)

    # Step 3.6: Final fit at best threshold
    phi_11, phi_12 = best_params[0]
    phi_21, phi_22 = best_params[1]
    tau = best_tau
    sigma2 = best_ssr / T

    RETURN phi_11, phi_12, phi_21, phi_22, tau, sigma2


FUNCTION forecast_specific_ar1(psi_1, psi_2, e_prev):
    """
    One-step-ahead AR(1) forecast of specific component.

    Reference: BDF 2008, Section 2.3, Eq. (10).
    """
    RETURN psi_1 * e_prev + psi_2


FUNCTION forecast_specific_setar(phi_11, phi_12, phi_21, phi_22, tau, e_prev):
    """
    One-step-ahead SETAR forecast of specific component.

    Reference: BDF 2008, Section 2.3, Eq. (11).
    """
    IF e_prev <= tau:
        RETURN phi_11 * e_prev + phi_12
    ELSE:
        RETURN phi_21 * e_prev + phi_22
```

#### Phase 4: Dynamic VWAP Execution

```
FUNCTION dynamic_vwap_execution(c_forecast, ar_params_or_setar_params,
                                 e_last_observed, k, N, model_type,
                                 total_order_quantity):
    """
    Produce volume forecasts for all remaining bins within a trading day,
    updating as new observations arrive. Allocate shares to trade in each
    bin proportionally to the forecast volume.

    Three strategies exist (BDF 2008, Section 4.2):
    1. Theoretical: uses true one-step-ahead (requires end-of-day total volume;
       serves only as upper bound, not implementable).
    2. Static: all k bins forecast at open; performs WORSE than classical
       because multi-step AR forecasts decay to the unconditional mean.
       NOT recommended. Reference: BDF 2008, Section 4.2.1.
    3. Dynamic: forecast at open, then revise after each observed bin.
       This is the ONLY operationally viable strategy.

    Reference: BDF 2008, Section 4.2.2.

    Parameters:
        total_order_quantity: array of shape (N,), total shares to trade per stock.
    """

    # Step 4.0: Initialize remaining quantity tracking
    # remaining_quantity[i] tracks how many shares of stock i remain to be traded.
    remaining_quantity = copy(total_order_quantity)  # shape (N,)

    # Step 4.1: At market open, produce initial full-day forecast
    # c_forecast is already computed (fixed for the day)
    # For the specific component, compute multi-step forecasts:
    full_day_forecast = array of shape (k, N)
    FOR j = 0, 1, ..., k-1:
        FOR stock i:
            IF j == 0:
                e_prev = e_last_observed[i]  # last bin of previous day
            ELSE:
                e_prev = e_hat_forecast[j-1, i]  # previous forecast

            IF model_type[i] == "AR1":
                e_hat_forecast[j, i] = forecast_specific_ar1(
                    psi_1[i], psi_2[i], e_prev)
            ELSE:  # SETAR
                e_hat_forecast[j, i] = forecast_specific_setar(
                    phi_11[i], phi_12[i], phi_21[i], phi_22[i], tau[i], e_prev)

            full_day_forecast[j, i] = c_forecast[j, i] + e_hat_forecast[j, i]

    # Step 4.2: Compute initial volume proportions for VWAP slicing
    # p[j, i] = full_day_forecast[j, i] / sum(full_day_forecast[:, i])
    proportions = full_day_forecast / sum(full_day_forecast, axis=0)

    # Step 4.3: Trade the first bin based on initial proportions
    shares_to_trade_bin0 = proportions[0, :] * remaining_quantity
    # Execute trade for bin 0
    remaining_quantity -= shares_to_trade_bin0

    # Step 4.4: As each bin j_obs completes, revise remaining forecasts
    FOR j_obs = 0, 1, ..., k-2:
        # Observe actual turnover x_actual[j_obs, i]
        # Compute actual specific component:
        e_actual[j_obs, i] = x_actual[j_obs, i] - c_forecast[j_obs, i]

        # Re-forecast remaining bins j_obs+1, ..., k-1
        FOR j = j_obs+1, ..., k-1:
            FOR stock i:
                IF j == j_obs + 1:
                    e_prev = e_actual[j_obs, i]  # use ACTUAL residual
                ELSE:
                    e_prev = e_hat_revised[j-1, i]

                IF model_type[i] == "AR1":
                    e_hat_revised[j, i] = forecast_specific_ar1(
                        psi_1[i], psi_2[i], e_prev)
                ELSE:
                    e_hat_revised[j, i] = forecast_specific_setar(
                        phi_11[i], phi_12[i], phi_21[i], phi_22[i], tau[i], e_prev)

                revised_forecast[j, i] = c_forecast[j, i] + e_hat_revised[j, i]

        # Recompute proportions for remaining bins
        remaining_volume = sum(revised_forecast[j_obs+1:, i]) for each i
        FOR j = j_obs+1, ..., k-1:
            revised_proportions[j, i] = revised_forecast[j, i] / remaining_volume[i]

        # Execute: trade in the next bin
        shares_next_bin = revised_proportions[j_obs+1, :] * remaining_quantity
        # Execute trade
        remaining_quantity -= shares_next_bin
```

#### Top-Level Daily Pipeline

```
FUNCTION daily_pipeline(historical_data, day_to_forecast, N, k, L):
    """
    End-to-end pipeline run once per day, before market open.

    Reference: BDF 2008, Sections 2.2-2.3, 4.2.
    """
    # Step 0: Extract the trailing L-day estimation window
    # For day d, use days d-L, d-L+1, ..., d-1
    # Each day has k bins, so P = L * k observations per stock
    # NOTE: In this implementation, the same L-day window is used for both
    # PCA estimation and common component averaging, following BDF 2008
    # Section 3.1 ("we chose a 20-day window to construct the common
    # component and L = 20 in Eq. (9)").
    X = prepare_data(historical_data[d-L:d], shares_outstanding, k)
    # X has shape (P, N) where P = L * k

    # Step 1: PCA factor extraction
    F_hat, Lambda_hat, C_hat, e_hat, r = extract_factors(X)

    # Step 2: Forecast common component
    c_forecast = forecast_common(C_hat, L, k, N)  # shape (k, N)

    # Step 3: Fit specific component model per stock with model selection
    # Default strategy: Use SETAR as primary. Fall back to AR(1) if SETAR
    # estimation fails or produces worse fit than AR(1).
    # Researcher inference — BDF 2008 reports both models side by side without
    # providing an explicit selection procedure. This rule is a practical
    # engineering choice motivated by the observation that SETAR outperforms
    # AR(1) in 36 of 39 stocks in BDF's in-sample volume prediction
    # (BDF 2008, Section 3.2) and 26-30 of 33 stocks in Szucs 2017.
    model_type = array of strings, shape (N,)
    FOR each stock i:
        e_series_i = e_hat[:, i]   # shape (P,)

        # Always fit AR(1) as baseline
        psi_1[i], psi_2[i], sigma2_ar[i] = fit_ar1(e_series_i)

        # Attempt SETAR fit
        setar_result = fit_setar(e_series_i)

        IF setar_result is None:
            # SETAR threshold search failed (all candidates had
            # insufficient regime observations) — use AR(1)
            model_type[i] = "AR1"
        ELSE:
            phi_11[i], phi_12[i], phi_21[i], phi_22[i], tau[i], sigma2_setar[i] = \
                setar_result

            # Model selection: use SETAR unless it overfits
            # (i.e., SETAR residual variance exceeds AR(1) residual variance)
            IF sigma2_setar[i] > sigma2_ar[i]:
                model_type[i] = "AR1"  # SETAR overfitting; revert to AR(1)
            ELSE:
                model_type[i] = "SETAR"

    # Step 4: Get last observed specific component value
    # (last bin of previous day, for each stock)
    e_last = e_hat[-1, :]  # shape (N,)

    # Step 5: Dynamic execution during the day
    dynamic_vwap_execution(c_forecast, params, e_last, k, N, model_type,
                           total_order_quantity)
```

### Data Flow

```
Input:                                           Output:
raw_volume  ─────────┐                           per-stock per-bin
  shape: (N, L*k)    │                           turnover forecast
                      ▼                           shape: (k, N)
shares_outstanding ──► prepare_data()
  shape: (N,)         │
                      ▼
                 X: (P, N)     ◄── P = L * k, no demeaning
                      │
                      ▼
               extract_factors()
                  │       │
                  ▼       ▼
            C_hat: (P,N) e_hat: (P,N)
                  │       │
                  ▼       │
         forecast_common  │    ──► c_forecast: (k, N)
                          │
                          ▼
                  fit_ar1 + fit_setar   (per stock, on e_hat[:, i])
                          │
                          ▼
                  model_selection → model_type[i] per stock
                          │
                          ▼
               dynamic_vwap_execution()
                          │
                          ▼
                 proportions: (k, N)   ◄── updated each bin intraday
                          │
                          ▼
                 remaining_quantity: (N,) ◄── decremented each bin
```

**Type details:**
- All turnover values are float64 (small positive numbers, typically 1e-3 to 5e-2).
  Reference: BDF 2008, Table 1 — overall mean turnover 0.0116, Std 0.0146, Q95 0.0380.
- X matrix: float64, shape (P, N). P = L * k (e.g., 20 * 26 = 520 rows for 20 days
  of 15-min bars). N = number of stocks (e.g., 33-39).
- SVD outputs: U (P, r_max) float64, s (r_max,) float64, Vt (r_max, N) float64.
- Factor matrices: F_hat (P, r) float64, Lambda_hat (N, r) float64.
- Common/specific components: same shape as X, float64.
- AR/SETAR parameters: float64 scalars per stock.
- model_type: string per stock, either "AR1" or "SETAR".
- Output proportions: float64, shape (k, N), each column sums to 1.
- remaining_quantity: float64, shape (N,), initialized to total order size,
  decremented after each bin's trade execution.

### Variants

**Implemented variant:** Dynamic PCA-SETAR (primary) with AR(1) as secondary/fallback.

**Model selection procedure (Researcher inference):** For each stock on each day,
SETAR is the primary model. Fall back to AR(1) if any of the following conditions
hold:
  (a) No valid SETAR threshold is found (all candidates have fewer than
      min_regime_obs observations in one regime).
  (b) SETAR residual variance exceeds AR(1) residual variance, indicating
      the two-regime model is overfitting relative to the simpler model.
This is re-evaluated daily since the estimation window shifts. BDF 2008 does
not provide an explicit model selection procedure — they report both models
side by side. This rule is a practical engineering choice.

The SETAR model is preferred because:
1. BDF 2008 Table 2: SETAR achieves lower in-sample volume prediction MAPE
   (0.0752 vs 0.0829 for ARMA) and lower VWAP tracking error (0.0706 vs 0.0772).
   Reference: BDF 2008, Section 3.2, Table 2 (in-sample period: September 2 to
   December 16, 2003).
2. BDF 2008 in-sample volume prediction: SETAR outperforms ARMA for 36 of 39
   stocks (ARMA beats SETAR for only 3 of 39, and by negligible margins).
   Reference: BDF 2008, Section 3.2.
3. BDF 2008 out-of-sample VWAP execution: Dynamic PCA-SETAR reduces tracking
   error vs classical for 30 of 39 stocks. Reference: BDF 2008, Table 6.
4. Szucs 2017 full-sample results: BDF_SETAR has lowest MAPE (0.399) across 33
   DJIA stocks over 11 years. Reference: Szucs 2017, Section 5, Table 2a.
5. Computational cost: SETAR grid search adds negligible overhead over AR(1).

The AR(1) variant should also be implemented for comparison and as a fallback
for cases where SETAR estimation fails or overfits.

**Not implemented:**
- Static execution: explicitly dominated by both dynamic and classical approaches.
  Reference: BDF 2008, Section 4.2.1 — "the static method is therefore worse."
- Theoretical execution: requires end-of-day total volume known in advance.
  Serves only as an upper bound on achievable performance.

## Parameters

| Parameter | Description | Recommended Value | Sensitivity | Range |
|-----------|-------------|-------------------|-------------|-------|
| k | Number of intraday bins per trading day | 26 (15-min) or 25 (20-min) | Low — determined by exchange hours and desired granularity | 13-78 (30-min to 5-min) |
| L | Rolling estimation window in trading days. Used for BOTH PCA estimation and common component averaging (BDF 2008, Section 3.1). | 20 | Moderate — shorter windows adapt faster but increase noise; longer windows smooth but may miss regime changes | 10-60 |
| N | Number of stocks in the cross-section. BDF 2008: N=39 (CAC40). Szucs 2017: N=33 (DJIA, after filtering 3 tickers for short history). | 30+ | Low-moderate — larger N improves factor estimation consistency per Bai (2003), but model works with as few as ~30 stocks | 20-500 |
| r | Number of common factors | Data-driven via IC_p2. Researcher inference: typically 1-3 for intraday volume, reflecting the low-dimensional U-shaped seasonal structure. | Moderate — too few factors leave common variation in residuals; too many overfit | 1-5 |
| r_max | Maximum candidate factor count for IC_p2 search | 10 | Low — only affects search range | 5-15 |
| n_grid | Number of threshold candidates in SETAR grid search | 100 | Low — 50-200 gives similar results | 50-200 |
| tau_quantile_range | Quantile range for SETAR threshold search | [0.15, 0.85] | Low-moderate — excluding extremes prevents degenerate regimes | [0.10, 0.90] |
| min_regime_obs | Minimum observations per SETAR regime | 10 | Low — ensures OLS stability | 5-20 |

### Initialization

1. **First run:** Requires at least L trading days of historical volume data for all
   N stocks before the model can produce forecasts. On the first day of trading,
   use the classical U-method (simple time-of-day average) as the benchmark.

2. **Rolling window:** Each day, the estimation window shifts forward by one day.
   The oldest day is dropped and the newest completed day is added. All model
   components (PCA, AR/SETAR parameters) are re-estimated from scratch on
   the updated window. Reference: BDF 2008, Section 3.1; Szucs 2017, Section 3.

3. **SVD initialization:** No special initialization needed — truncated SVD is a
   deterministic algorithm that produces the same result regardless of starting
   point.

4. **SETAR threshold:** Initialized via exhaustive grid search over quantiles of the
   lagged residual series. No warm-starting from the previous day's estimate is
   needed (or recommended), since the residual distribution can shift day to day.

5. **Intraday state:** At market open, the "previous specific component" value
   e_prev is set to e_hat[-1, i] — the last estimated residual from the final
   bin of the previous trading day. This provides the initial condition for the
   AR/SETAR recursion. Reference: BDF 2008, Section 4.2.2.

6. **Remaining quantity:** At market open, remaining_quantity[i] is initialized to
   the total order size for stock i. It is decremented by the number of shares
   traded after each bin's execution.

### Calibration

The model has no hyperparameters requiring manual calibration beyond the
estimation window length L and bin size k, which are design choices set once.

**Automated calibration steps (run daily):**
1. IC_p2 determines r automatically from the data.
2. AR(1) parameters estimated by OLS (closed-form solution).
3. SETAR parameters estimated by OLS conditional on each threshold candidate,
   with the best threshold selected by minimizing total residual sum of squares.
4. Model selection (AR(1) vs SETAR) is performed per stock based on residual
   variance comparison (Researcher inference — see Variants section).

**No iterative optimization is required.** This is a key practical advantage over
competing models (BCG requires GMM with convergence issues; Szucs 2017 reports
BCG took 60 machine-days vs. 2 hours for BDF on comparable data).

## Validation

### Expected Behavior

**Important note on metrics:** Two fundamentally different MAPE metrics are used
in the literature for this model. They CANNOT be directly compared:

1. **Turnover forecast MAPE** (Szucs 2017): Measures per-bin percentage error of
   the one-step-ahead turnover prediction vs actual turnover. Higher values
   (~0.40) because it measures individual bin-level percentage errors, which
   are naturally noisy. Formula: MAPE = (1/N) * sum |Y_t - Y_t^f| / Y_t
   (Szucs 2017, Eq. (2)). Bins where actual turnover Y_t = 0 must be excluded.

2. **VWAP execution cost MAPE** (BDF 2008): Measures the percentage deviation
   between the trader's achieved average price and the true end-of-day VWAP.
   Lower values (~0.09) because it measures an aggregate price deviation that
   benefits from averaging across many bins within a day.

**Turnover forecast accuracy (one-step-ahead, Szucs metric):**
   - BDF_AR: 0.403 MAPE, 6.49E-04 MSE.
   - BDF_SETAR: 0.399 MAPE, 6.60E-04 MSE.
   - Benchmark U-method: 0.503 MAPE, 1.02E-03 MSE.
   - This represents roughly a 20% MAPE reduction over the naive benchmark.
   - NOTE: BDF_AR beats BDF_SETAR on MSE (6.49E-04 vs 6.60E-04) despite
     SETAR winning on MAPE (0.399 vs 0.403). This means a developer using MSE
     as primary metric should not be surprised if AR(1) appears to win.
   Source: Szucs 2017, Section 5, Table 2a (33 DJIA stocks, 11 years,
   one-step-ahead forecasting with actuals fed back each bin).

**In-sample volume prediction accuracy (BDF data):**
   - PCA-SETAR: 0.0752 MAPE.
   - PCA-ARMA: 0.0829 MAPE.
   - Classical approach: 0.0905 MAPE.
   These are IN-SAMPLE volume prediction MAPE values, not VWAP execution cost.
   Source: BDF 2008, Section 3.2, Table 2 (in-sample period: September 2 to
   December 16, 2003, 39 CAC40 stocks). BDF's out-of-sample analysis evaluates
   VWAP execution cost (below), not raw volume prediction accuracy.

**VWAP tracking error (out-of-sample, BDF data):**
   - Dynamic PCA-SETAR: 0.0898 MAPE.
   - Dynamic PCA-ARMA: 0.0922 MAPE.
   - Classical approach: 0.1006 MAPE.
   - Approximately 10% reduction in VWAP tracking error versus classical.
   Source: BDF 2008, Section 4.3.2, Table 5 (out-of-sample panel, 39 CAC40 stocks).

**Common component shape:** The common component forecast should exhibit the
   well-known U-shape — higher volume at market open and close, lower in the
   middle of the day. Visual inspection of c_forecast should confirm this.
   Source: BDF 2008, Section 3.2, Figure 3.

**Specific component ACF:** The autocorrelation of the specific component
   e_hat should show significant lag-1 correlation but decay quickly thereafter.
   The PACF should show a single significant spike at lag 1 (consistent with
   AR(1) structure). Source: BDF 2008, Section 3.2, Figure 2 (TOTAL stock ACF/PACF).

**Factor count:** For liquid equity markets, IC_p2 should typically select
   r = 1 to 3 factors. Researcher inference: based on the low-dimensional
   structure of intraday volume; BDF 2008 does not explicitly report the
   IC_p2-selected r values for their data.

**Factor normalization check:** After extraction, verify F_hat.T @ F_hat / P
   is approximately I_r (identity matrix of size r). Deviation > 1e-10 indicates
   a numerical issue. Source: Bai 2003; BDF 2008, Eq. (6).

### Sanity Checks

1. **Reconstruction error:** C_hat + e_hat should exactly equal X (up to floating-
   point precision). Check: max(abs(X - C_hat - e_hat)) < 1e-10.

2. **Proportion sum:** For each stock, the dynamic VWAP proportions for remaining
   bins should sum to 1.0. Check: abs(sum(proportions[j:, i]) - 1.0) < 1e-10
   for all stocks i after each intraday update.

3. **Turnover positivity:** Turnover is non-negative by definition. If any
   forecast produces negative turnover (possible since the model is additive and
   the specific component can be negative), floor the forecast at zero and
   redistribute proportionally. This should be rare for liquid stocks.
   Reference: Researcher inference — BDF 2008 does not discuss negative forecasts
   explicitly, but the additive structure makes them possible.

4. **AR(1) stationarity:** psi_1 should satisfy |psi_1| < 1 for each stock. If
   |psi_1| >= 1, the AR(1) model is non-stationary and the forecast will diverge.
   Fallback: use the common component forecast only (set specific forecast to 0).

5. **SETAR regime balance:** Both regimes should have a meaningful number of
   observations. If either regime has fewer than min_regime_obs data points,
   the fit_setar function returns None and the caller falls back to AR(1).
   Reference: Researcher inference — standard SETAR practice.

6. **U-method benchmark comparison:** On random historical data, the BDF model
   should produce lower MAPE than the U-method. If it does not, something is
   wrong with the factor extraction or residual modeling. The U-method forecast
   for bin j, stock i is simply the average of turnover in bin j across the L
   days: u_forecast[j, i] = mean(X[j::k, i]). Reference: BDF 2008, Section 3.2;
   Szucs 2017, Eq. (3).

7. **MAPE computation formula:** For validation comparisons, use the Szucs 2017
   formula (Eq. (2)): MAPE = (1/N) * sum_{t=1}^{N} |Y_t - Y_t^f| / Y_t,
   where Y_t is the actual value and Y_t^f is the forecast. Exclude bins where
   Y_t = 0 (see Edge Cases). Both per-stock and cross-stock average MAPE should
   be computed per Szucs 2017, Section 3.

### Edge Cases

1. **Zero-volume bins:** Some stocks may have zero volume in certain bins
   (especially illiquid stocks or the first/last bins). The model handles this
   naturally since turnover of zero is a valid observation. However, MAPE is
   undefined when actual volume is zero; exclude such bins from MAPE computation.
   Reference: Szucs 2017 ensures all stocks have non-zero volume in every bin
   (Section 2, data description).

2. **Missing data / halted stocks:** If a stock has no trading in the entire
   estimation window (e.g., halted), exclude it from the cross-section before
   PCA. If a stock is halted mid-day, its bins should be treated as missing
   and excluded from the intraday update step.

3. **Short trading days:** Half-days (e.g., December 24, July 3 in the US) have
   fewer bins. BDF 2008 excludes these days (Section 3.1: "The 24th and 31st of
   December 2003 were excluded"). Recommended: exclude partial trading days from
   both the estimation window and forecasting.

4. **Index reconstitution:** When stocks enter or leave the universe, the matrix X
   changes dimensionality. Re-estimate from scratch with the new universe. Factor
   structure is generally robust to small changes in the cross-section.

5. **Very high volume days:** Earnings, index rebalancing, or major news can cause
   volume spikes of 5-10x normal. The AR(1)/SETAR specific component will lag
   behind such spikes. The model does not have explicit event-day handling.
   For production, consider augmenting with event-day adjustments as in Markov
   et al. (2019). Reference: Researcher inference — BDF 2008 does not address
   event days.

6. **Numerical stability of SVD:** When N and P are both large, the full SVD of X
   is expensive. Use a truncated/randomized SVD (e.g., scipy.sparse.linalg.svds
   or sklearn.utils.extmath.randomized_svd) that computes only the top r_max
   components. This is both faster and numerically stable.

7. **SETAR fallback cascade:** If fit_setar returns None for a stock, the daily
   pipeline automatically uses AR(1) for that stock on that day. If AR(1) also
   fails (|psi_1| >= 1), fall back to common-component-only forecasting (specific
   forecast = 0). This ensures the pipeline never crashes. Reference: Researcher
   inference.

### Known Limitations

1. **Additive decomposition can produce negative forecasts.** Unlike multiplicative
   models (BCG/CMEM), the BDF additive structure allows negative turnover
   forecasts when the specific component is strongly negative. This must be
   handled by flooring at zero. Reference: Researcher inference — inherent to
   the additive model structure.

2. **No daily volume component.** The model does not explicitly model day-to-day
   total volume variation. The common component absorbs some of this via the
   rolling window average, but a day with 3x normal volume will not be well
   predicted by the morning forecast. The dynamic update partially compensates.
   Reference: BDF 2008, Section 4.2 — acknowledged limitation.

3. **No price/return effects.** The model is purely volume-driven and does not
   incorporate price movements, volatility, or return asymmetry effects. To
   "beat" the VWAP (rather than track it), a bivariate volume-price model is
   required. Reference: BDF 2008, Section 5 (conclusion).

4. **Cross-sectional requirement.** The model requires a panel of N >= ~20 stocks
   to reliably extract factors via PCA. It cannot be applied to a single stock
   in isolation. Reference: Bai 2003 — consistency requires both N and T large.

5. **Static execution is dominated.** Multi-step AR/SETAR forecasts decay toward
   the unconditional mean, making the specific component contribution negligible
   for horizon > 3-4 bins. Only dynamic (one-step-ahead with updates) execution
   is operationally useful. Reference: BDF 2008, Section 4.2.1.

6. **No closing auction modeling.** The model treats all bins equally and does not
   separately handle the closing auction, which can represent 10-15% of daily
   volume in modern markets. Reference: Researcher inference — BDF 2008 data
   predates the rise of closing auctions.

## Paper References

| Spec Section | Paper Source |
|---|---|
| Overview, model description | BDF 2008, Abstract and Section 1 |
| Turnover definition (shares / float) | BDF 2008, Section 2.2, above Eq. (4) |
| Turnover definition (shares / TSO) | Szucs 2017, Section 2, p.4 |
| Additive decomposition x = c + e | BDF 2008, Section 2.2, Eq. (2)-(5) |
| PCA via eigendecomposition / SVD | BDF 2008, Section 2.2, Eq. (6); Bai 2003 |
| No demeaning before PCA | Bai 2003, Section 2; BDF 2008, Eq. (6) — no demeaning step present |
| Factor count selection IC_p2 | Bai & Ng 2002; BDF 2008, Section 2.2 |
| Factor normalization F'F/T = I_r | BDF 2008, Section 2.2, below Eq. (6) |
| Common component forecast (time-of-day average) | BDF 2008, Section 2.3, Eq. (9) |
| AR(1) specific component model | BDF 2008, Section 2.3, Eq. (10); Szucs 2017, Eq. (5) |
| SETAR specific component model | BDF 2008, Section 2.3, Eq. (11); Szucs 2017, Eq. (6) |
| SETAR notation difference between BDF and Szucs | BDF 2008, Eq. (11) vs Szucs 2017, Eq. (6) — opposite intercept/slope ordering |
| Dynamic VWAP execution strategy | BDF 2008, Section 4.2.2 |
| Static execution is dominated | BDF 2008, Section 4.2.1 |
| Turnover forecast accuracy (MAPE, MSE) | Szucs 2017, Section 5, Table 2a (one-step-ahead, 33 DJIA stocks, 11 years) |
| MAPE formula | Szucs 2017, Section 3, Eq. (2) |
| MSE formula | Szucs 2017, Section 3, Eq. (1) |
| In-sample volume prediction MAPE | BDF 2008, Section 3.2, Table 2 (Sep 2 to Dec 16, 2003) |
| VWAP tracking error results (out-of-sample) | BDF 2008, Section 4.3, Tables 4-7 |
| SETAR outperforms AR: 36/39 stocks (in-sample volume prediction) | BDF 2008, Section 3.2 |
| Dynamic SETAR outperforms classical: 30/39 stocks (out-of-sample VWAP) | BDF 2008, Table 6 |
| Estimation window = 20 days, used for both PCA and common component | BDF 2008, Section 3.1 |
| Parameters re-estimated daily on rolling window | BDF 2008, Section 3.1; Szucs 2017, Section 3 |
| Szucs data: 36 tickers, 33 after filtering for short history | Szucs 2017, Section 2 |
| Turnover value range (mean 0.0116, Q95 0.0380) | BDF 2008, Table 1 |
| Computational efficiency vs BCG | Szucs 2017, Section 5 (2 hours vs 60 machine-days) |
| Negative forecast handling | Researcher inference — not discussed in papers |
| Event-day adjustments suggestion | Researcher inference, motivated by Markov et al. 2019 |
| Closing auction limitation | Researcher inference — BDF 2008 predates modern closing auctions |
| SETAR grid search procedure | Researcher inference — standard SETAR estimation; BDF 2008 does not detail the threshold search |
| SETAR fallback when no valid threshold found | Researcher inference — not addressed in BDF 2008 |
| Model selection procedure (AR1 vs SETAR) | Researcher inference — BDF 2008 reports both without selection rule |
| Minimum regime observations | Researcher inference — standard SETAR practice |
| Factor count "typically 1-3" | Researcher inference — based on low-dimensional intraday volume structure |
| SVD-based extraction vs eigendecomposition | Researcher inference — algebraically equivalent, SVD avoids branching on N vs P |
| Remaining quantity tracking in VWAP execution | Researcher inference — standard VWAP execution accounting |
