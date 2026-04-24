# Implementation Specification: Kalman Filter State-Space Model for Intraday Volume

## Overview

This model forecasts intraday trading volume by decomposing log-volume into three
additive components -- a daily average, an intraday periodic pattern, and an intraday
dynamic component -- within a linear Gaussian state-space framework. The Kalman filter
provides optimal one-step-ahead and multi-step-ahead predictions. The EM algorithm
calibrates all parameters with closed-form updates. A robust extension adds
Lasso-penalized sparse noise detection for automatic outlier handling in real-time
market data.

The model is based entirely on Chen, Feng, and Palomar (2016).

## Algorithm

### Model Description

The model operates on intraday volume data aggregated into fixed-width time bins
(e.g., 15-minute intervals). Raw volume in each bin is normalized by shares outstanding,
then log-transformed. The log-transform is the key design choice: it converts the
multiplicative three-component decomposition (from the CMEM literature) into a linear
additive model, eliminating positiveness constraints, reducing right-skewness, and
enabling exact Kalman filter recursions.

**Inputs:**
- Historical intraday volume time series, aggregated into I bins per trading day.
- Number of shares outstanding per day (for normalization).
- Training window length N (number of total bins = T_train * I).
- For robust variant: Lasso regularization parameter lambda.

**Outputs:**
- One-bin-ahead log-volume forecasts (dynamic mode).
- Full-day log-volume forecasts for day t+1 (static mode).
- Volume share weights for VWAP execution (static and dynamic strategies).

**Assumptions:**
- All bin volumes are strictly positive (log(0) is undefined). Zero-volume bins must
  be excluded or imputed before processing.
- Process noise and observation noise are Gaussian.
- The daily component eta is constant within each trading day, changing only at day
  boundaries.
- The intraday periodic component phi is deterministic and constant across days
  (estimated as a fixed vector, one entry per bin).
- The intraday dynamic component mu follows an AR(1) process within and across days.

### Pseudocode

The system consists of three coupled algorithms: the Kalman filter for prediction,
the Kalman smoother for calibration, and the EM algorithm for parameter estimation.
A fourth procedure adds Lasso-based outlier handling for the robust variant.

#### Data Preprocessing

```
PREPROCESS(raw_volume, shares_outstanding, I):
    # raw_volume[t,i]: raw volume for day t, bin i (i = 1..I)
    # shares_outstanding[t]: shares outstanding for day t
    # I: number of bins per trading day

    for each day t, bin i:
        normalized_volume[t,i] = raw_volume[t,i] / shares_outstanding[t]
        y[t,i] = ln(normalized_volume[t,i])

    # Flatten to single time index tau = (t-1)*I + i
    # tau = 1, 2, ..., N where N = T * I
    y_flat[tau] = y[t,i]  where tau = (t-1)*I + i

    return y_flat, I
```
(Paper Section 2, Equation 3. Normalization by shares outstanding described in
Section 4.1, paragraph 2.)

#### Algorithm 1: Kalman Filter (Prediction)

```
KALMAN_FILTER(y[1..N], theta, I):
    # theta = {a_eta, a_mu, sigma_eta_sq, sigma_mu_sq, r, phi[1..I], pi_1, Sigma_1}
    # I = number of bins per trading day
    # N = total number of observed bins

    # Initialize
    x_hat[1|0] = pi_1                    # 2x1 vector: [eta_0, mu_0]
    Sigma[1|0] = Sigma_1                 # 2x2 matrix

    C = [1, 1]                           # 1x2 observation vector (row vector)

    for tau = 1, 2, ..., N:
        # Determine current bin index within day
        i = ((tau - 1) mod I) + 1

        # --- PREDICTION STEP ---
        # (x_hat[tau+1|tau] and Sigma[tau+1|tau] already computed
        #  from previous iteration, or from initialization for tau=1)

        # Compute innovation (prediction error)
        e[tau] = y[tau] - C @ x_hat[tau|tau-1] - phi[i]

        # Compute innovation variance (scalar)
        W_inv[tau] = C @ Sigma[tau|tau-1] @ C^T + r      # scalar
        # Note: W_inv is actually S (innovation variance), not W.
        # The paper uses W = S^{-1} in the robust section.

        # Compute Kalman gain (2x1 vector)
        K[tau] = Sigma[tau|tau-1] @ C^T / W_inv[tau]      # 2x1 / scalar

        # --- CORRECTION STEP ---
        x_hat[tau|tau] = x_hat[tau|tau-1] + K[tau] * e[tau]
        Sigma[tau|tau] = Sigma[tau|tau-1] - K[tau] * W_inv[tau] * K[tau]^T
        # Equivalently: Sigma[tau|tau] = (I_2 - K[tau] @ C) @ Sigma[tau|tau-1]

        # --- PROPAGATION TO NEXT STEP ---
        # Build transition matrix A[tau] based on whether next step is a day boundary
        if (tau mod I) == 0:
            # Day boundary: eta transitions with AR coefficient and noise
            A[tau] = [[a_eta, 0],
                       [0,     a_mu]]
            Q[tau] = [[sigma_eta_sq, 0],
                       [0,           sigma_mu_sq]]
        else:
            # Within day: eta is constant (no transition noise)
            A[tau] = [[1,     0],
                       [0, a_mu]]
            Q[tau] = [[0,            0],
                       [0, sigma_mu_sq]]

        x_hat[tau+1|tau] = A[tau] @ x_hat[tau|tau]
        Sigma[tau+1|tau] = A[tau] @ Sigma[tau|tau] @ A[tau]^T + Q[tau]

    # Store one-step-ahead predictions
    y_hat[tau] = C @ x_hat[tau|tau-1] + phi[i]   # for each tau

    return x_hat, Sigma, K, y_hat
```
(Paper Section 2.2, Algorithm 1, Equations 4-5, 7-8. The time-varying A_tau structure
is defined in Section 2, page 3, bullet points for A_tau and Q_tau.)

**Important note on indexing:** The paper uses a unified time index tau that runs
across all bins in all days. Day boundaries occur at tau = kI for k = 1, 2, ....
The bin-within-day index is i = ((tau-1) mod I) + 1. The transition matrix A_tau
applied at step tau governs the transition from x_tau to x_{tau+1}. At day boundaries
(tau = kI), eta transitions with coefficient a_eta; within a day, eta stays constant
(coefficient 1, zero noise).

#### Algorithm 2: Kalman Smoother (RTS Backward Pass)

```
KALMAN_SMOOTHER(x_hat_filt, Sigma_filt, x_hat_pred, Sigma_pred, A, N):
    # Inputs from the forward Kalman filter pass:
    #   x_hat_filt[tau] = x_hat[tau|tau] for tau = 1..N
    #   Sigma_filt[tau] = Sigma[tau|tau] for tau = 1..N
    #   x_hat_pred[tau] = x_hat[tau|tau-1] for tau = 1..N
    #   Sigma_pred[tau] = Sigma[tau|tau-1] for tau = 1..N
    #   A[tau] = transition matrix at step tau

    # Initialize smoother at last time step
    x_hat_smooth[N] = x_hat_filt[N]
    Sigma_smooth[N] = Sigma_filt[N]

    for tau = N, N-1, ..., 2:
        # Smoother gain
        L[tau-1] = Sigma_filt[tau-1] @ A[tau-1]^T @ inv(Sigma_pred[tau])

        # Smoothed state estimate
        x_hat_smooth[tau-1] = x_hat_filt[tau-1]
            + L[tau-1] @ (x_hat_smooth[tau] - x_hat_pred[tau])

        # Smoothed covariance
        Sigma_smooth[tau-1] = Sigma_filt[tau-1]
            + L[tau-1] @ (Sigma_smooth[tau] - Sigma_pred[tau]) @ L[tau-1]^T

    # Compute cross-covariance P_{tau,tau-1} needed for EM
    # Initialize: Sigma_{N,N-1|N} = (I - K_N @ C) @ A_{N-1} @ Sigma_filt[N-1]
    Sigma_cross[N] = (I_2 - K[N] @ C) @ A[N-1] @ Sigma_filt[N-1]

    for tau = N, N-1, ..., 3:
        Sigma_cross[tau-1] = Sigma_filt[tau-1] @ L[tau-2]^T
            + L[tau-1] @ (Sigma_cross[tau] - A[tau-1] @ Sigma_filt[tau-1]) @ L[tau-2]^T

    return x_hat_smooth, Sigma_smooth, Sigma_cross
```
(Paper Section 2.3.1, Algorithm 2, Equations 10-11. Cross-covariance computation
from Appendix A, Equations A.20-A.22. The smoother gain L_tau is defined in
Algorithm 2, line 2.)

#### Algorithm 3: EM Parameter Estimation

```
EM_ALGORITHM(y[1..N], I, theta_init, max_iter, tol):
    theta = theta_init
    j = 0

    repeat:
        j = j + 1

        # --- E-STEP ---
        # Run forward Kalman filter with current theta
        x_hat_filt, Sigma_filt, x_hat_pred, Sigma_pred, K = KALMAN_FILTER(y, theta, I)

        # Run backward Kalman smoother
        x_hat_s, Sigma_s, Sigma_cross = KALMAN_SMOOTHER(
            x_hat_filt, Sigma_filt, x_hat_pred, Sigma_pred, A, N)

        # Compute sufficient statistics from smoothed estimates
        # x_hat_s[tau] = [eta_hat_tau, mu_hat_tau]^T (smoothed state)
        # Sigma_s[tau] = smoothed covariance (2x2)
        # Sigma_cross[tau] = E[x_tau @ x_{tau-1}^T | y_{1:N}] cross term

        # Define P matrices (sufficient statistics):
        # P_tau = Sigma_s[tau] + x_hat_s[tau] @ x_hat_s[tau]^T
        # P_{tau,tau-1} = Sigma_cross[tau] + x_hat_s[tau] @ x_hat_s[tau-1]^T
        for tau = 1..N:
            P[tau] = Sigma_s[tau] + x_hat_s[tau] @ x_hat_s[tau]^T
        for tau = 2..N:
            P_cross[tau] = Sigma_cross[tau] + x_hat_s[tau] @ x_hat_s[tau-1]^T

        # Notation for element access:
        # P[tau]^{(k,l)} means element (k,l) of matrix P[tau]
        # P_cross[tau]^{(k,l)} means element (k,l) of matrix P_cross[tau]

        # --- M-STEP (closed-form updates) ---

        # D = set of day-boundary indices {I, 2I, 3I, ...}
        D = {tau : tau mod I == 0, tau >= I, tau <= N}
        # D_plus = {kI+1 : k=1,2,...} = set of first-bin-of-day indices (excluding day 1)
        D_plus = {tau+1 : tau in D, tau+1 <= N}

        # Initial state mean (Eq A.32)
        pi_1 = x_hat_s[1]

        # Initial state covariance (Eq A.33)
        Sigma_1 = P[1] - x_hat_s[1] @ x_hat_s[1]^T

        # Daily AR coefficient a_eta (Eq A.34)
        # Uses only day-boundary transitions (tau in D_plus, i.e., tau = kI+1)
        numerator_a_eta = sum over tau in D_plus of P_cross[tau]^{(1,1)}
        denominator_a_eta = sum over tau in D_plus of P[tau-1]^{(1,1)}
        a_eta = numerator_a_eta / denominator_a_eta

        # Intraday AR coefficient a_mu (Eq A.35)
        # Uses all transitions tau = 2..N
        numerator_a_mu = sum_{tau=2}^{N} P_cross[tau]^{(2,2)}
        denominator_a_mu = sum_{tau=2}^{N} P[tau-1]^{(2,2)}
        a_mu = numerator_a_mu / denominator_a_mu

        # Daily process noise variance (Eq A.36)
        # |D| = number of day boundaries = T-1
        sigma_eta_sq = (1 / |D|) * sum over tau in D_plus of
            [P[tau]^{(1,1)} + a_eta^2 * P[tau-1]^{(1,1)} - 2*a_eta * P_cross[tau]^{(1,1)}]

        # Intraday process noise variance (Eq A.37)
        sigma_mu_sq = (1 / (N-1)) * sum_{tau=2}^{N}
            [P[tau]^{(2,2)} + a_mu^2 * P[tau-1]^{(2,2)} - 2*a_mu * P_cross[tau]^{(2,2)}]

        # Observation noise variance (Eq A.38)
        r = (1/N) * sum_{tau=1}^{N}
            [y[tau]^2 + C @ P[tau] @ C^T - 2*y[tau]*C @ x_hat_s[tau]
             + phi[i_tau]^2 - 2*y[tau]*phi[i_tau] + 2*phi[i_tau]*C @ x_hat_s[tau]]
        # where i_tau = ((tau-1) mod I) + 1

        # Seasonality vector (Eq A.39 / Eq 24)
        for i = 1..I:
            phi[i] = (1/T) * sum_{t=1}^{T} (y[t,i] - C @ x_hat_s[(t-1)*I + i])

        # Check convergence
        delta = |log_likelihood(theta_new) - log_likelihood(theta_old)|
    until delta < tol or j >= max_iter

    return theta
```
(Paper Section 2.3.2, Algorithm 3. M-step closed-form updates from Appendix A,
Equations A.32-A.39. The sufficient statistics P and P_cross are defined in
Appendix A, Equations A.15-A.17.)

**Critical detail on a_eta update:** The sum for a_eta runs only over day-boundary
transitions. Specifically, the set D in the paper (Equations A.26, A.28) corresponds
to indices tau = kI+1 for k = 1, 2, ..., T-1, i.e., the first bin of each day
starting from day 2. At these transitions, eta actually changes (with coefficient
a_eta). The denominator sums P_{tau-1}^{(1,1)} at the last bin of the preceding day.
(Paper Appendix A, Equations A.26 and A.34.)

**Critical detail on sigma_eta_sq:** The divisor is T-1 (number of day transitions),
not N-1. The paper defines |D| = T-1 in the context of Equation A.28/A.36.
(Paper Appendix A, Equation A.36.)

#### Algorithm 4: Robust Kalman Filter (Lasso Extension)

```
ROBUST_KALMAN_FILTER(y[1..N], theta, lambda, I):
    # Same as standard Kalman filter, but correction step is modified.
    # theta includes all standard parameters.
    # lambda = Lasso regularization parameter (scalar > 0).

    # Initialize same as standard filter
    x_hat[1|0] = pi_1
    Sigma[1|0] = Sigma_1
    C = [1, 1]

    for tau = 1, 2, ..., N:
        i = ((tau - 1) mod I) + 1

        # PREDICTION (same as standard)
        # ... (omitted, identical to Algorithm 1)

        # CORRECTION with Lasso
        # Compute innovation
        e[tau] = y[tau] - phi[i] - C @ x_hat[tau|tau-1]

        # Compute W (inverse innovation variance, scalar)
        S[tau] = C @ Sigma[tau|tau-1] @ C^T + r           # innovation variance
        W[tau] = 1.0 / S[tau]                              # W = S^{-1}

        # Solve Lasso: soft-thresholding on innovation
        # (Paper Eq 33-34)
        threshold = lambda / (2.0 * W[tau])

        if e[tau] > threshold:
            z_star[tau] = e[tau] - threshold
        elif e[tau] < -threshold:
            z_star[tau] = e[tau] + threshold
        else:
            z_star[tau] = 0.0

        # Corrected innovation (outlier removed)
        e_corrected[tau] = e[tau] - z_star[tau]

        # Standard Kalman correction using cleaned innovation
        K[tau] = Sigma[tau|tau-1] @ C^T / S[tau]
        x_hat[tau|tau] = x_hat[tau|tau-1] + K[tau] * e_corrected[tau]
        Sigma[tau|tau] = Sigma[tau|tau-1] - K[tau] * S[tau] * K[tau]^T

        # PROPAGATION (same as standard)
        # ... (omitted, identical to Algorithm 1)

    return x_hat, Sigma, K, z_star
```
(Paper Section 3.1, Equations 25-34. The soft-thresholding solution is Equation 33
for z* and Equation 34 for e - z*. The modified correction step follows from
Equations 31-32.)

#### Robust EM Modifications

```
ROBUST_EM_M_STEP_ADJUSTMENTS(y, x_hat_s, z_star, phi, theta, I, T):
    # Only r and phi updates differ from the standard EM.
    # All other M-step updates (a_eta, a_mu, sigma_eta_sq, sigma_mu_sq,
    #   pi_1, Sigma_1) remain identical.

    # Modified observation noise variance (Eq 35)
    r = (1/N) * sum_{tau=1}^{N}
        [y[tau]^2 + C @ P[tau] @ C^T + phi[i_tau]^2 + z_star[tau]^2
         - 2*y[tau]*C @ x_hat_s[tau] - 2*z_star[tau]*y[tau]
         + 2*z_star[tau]*phi[i_tau] + 2*z_star[tau]*C @ x_hat_s[tau]
         - 2*y[tau]*phi[i_tau] + 2*phi[i_tau]*C @ x_hat_s[tau]]

    # Modified seasonality vector (Eq 36)
    for i = 1..I:
        phi[i] = (1/T) * sum_{t=1}^{T} (y[t,i] - C @ x_hat_s[(t-1)*I+i] - z_star[(t-1)*I+i])

    return r, phi
```
(Paper Section 3.2, Equations 35-36.)

#### Prediction Modes

```
DYNAMIC_PREDICTION(y_observed, theta, I):
    # One-bin-ahead forecasting: after observing bin tau, predict bin tau+1.
    # Run standard (or robust) Kalman filter.
    # At each step, the prediction is:
    y_hat[tau+1] = C @ x_hat[tau+1|tau] + phi[i_{tau+1}]

    # Convert to volume space:
    volume_hat[tau+1] = exp(y_hat[tau+1]) * shares_outstanding


STATIC_PREDICTION(theta, I, last_day_state):
    # Predict all I bins of the next day without any intraday corrections.
    # Start from the end-of-day state of the previous day.

    x = last_day_state   # x_hat[tau_last | tau_last] from previous day's close

    # Day-boundary transition
    A_boundary = [[a_eta, 0], [0, a_mu]]
    x = A_boundary @ x

    for h = 1..I:
        y_hat[h] = C @ x + phi[h]

        # Propagate within day (no correction step)
        A_within = [[1, 0], [0, a_mu]]
        x = A_within @ x

    return y_hat[1..I]


MULTI_STEP_PREDICTION(theta, I, h, x_current):
    # h-step-ahead prediction from current state (Paper Eq 9)
    # Used for static prediction at arbitrary horizon h

    # Propagate state h steps without correction
    x = x_current
    for step = 1..h:
        # Determine if step crosses a day boundary
        A_step = <appropriate A_tau for this step>
        x = A_step @ x

    y_hat = C @ x + phi[i_{current+h}]
    return y_hat
```
(Paper Section 2.2, paragraphs on dynamic vs. static prediction, and Equation 9.)

#### VWAP Execution Strategies

```
STATIC_VWAP(volume_forecast_static[1..I]):
    # Compute VWAP weights from static volume forecasts (Eq 40)
    total = sum(volume_forecast_static[1..I])
    for i = 1..I:
        w[i] = volume_forecast_static[i] / total
    return w[1..I]


DYNAMIC_VWAP(volume_forecast_dynamic, observed_volume, i_current, I):
    # Revise VWAP weights at each bin using dynamic forecasts (Eq 41)
    # After observing bins 1..i_current, redistribute remaining execution

    # Cumulative fraction already allocated
    cum_allocated = sum_{j=1}^{i_current} w_realized[j]

    for i = i_current+1 .. I-1:
        # Predicted fraction of remaining volume in bin i
        w[i] = (volume_forecast_dynamic[i] / sum_{j=i_current+1}^{I} volume_forecast_dynamic[j])
               * (1 - cum_allocated)

    # Last bin: absorb residual
    w[I] = 1 - sum_{j=1}^{I-1} w[j]

    return w
```
(Paper Section 4.3, Equations 39-41.)

### Data Flow

```
Raw volume data (T days x I bins)
    |
    v
[Preprocessing]
    - Normalize by shares outstanding
    - Take natural log
    - Flatten to single time series y[1..N], N = T*I
    |
    v
[EM Calibration] (on training window of N bins)
    - Initialize theta_0 (see Initialization section)
    - Repeat until convergence:
        - E-step: Forward Kalman filter -> {x_hat_filt, Sigma_filt}
                  Backward RTS smoother -> {x_hat_smooth, Sigma_smooth, Sigma_cross}
                  Compute sufficient statistics P, P_cross
        - M-step: Closed-form updates for all parameters
    - Output: calibrated theta = {a_eta, a_mu, sigma_eta_sq, sigma_mu_sq, r, phi, pi_1, Sigma_1}
    |
    v
[Prediction] (on out-of-sample data)
    - Dynamic mode: Run Kalman filter bin-by-bin, outputting y_hat[tau+1] at each step
    - Static mode: From prior day's end state, propagate without corrections for all I bins
    - For robust: use Robust Kalman Filter with calibrated lambda
    |
    v
[Post-processing]
    - Convert log-volume forecasts to volume: vol_hat = exp(y_hat) * shares_outstanding
    - Compute VWAP weights from volume forecasts (static or dynamic strategy)
```

**Types and shapes at each stage:**

| Stage | Variable | Shape | Type |
|-------|----------|-------|------|
| Input | raw_volume | (T, I) | float, non-negative |
| Input | shares_outstanding | (T,) | float, positive |
| Preprocessing | y | (N,) where N=T*I | float (log-space) |
| State | x_tau | (2,) | float: [eta, mu] |
| State covariance | Sigma_tau | (2, 2) | float, positive semi-definite |
| Transition | A_tau | (2, 2) | float, diagonal |
| Observation | C | (1, 2) = [1, 1] | float, constant |
| Seasonality | phi | (I,) | float (log-space) |
| Kalman gain | K_tau | (2,) or (2, 1) | float |
| Innovation variance | S_tau | scalar | float, positive |
| Prediction | y_hat | (N,) | float (log-space) |
| Output | volume_hat | (T, I) | float, positive |
| VWAP weights | w | (I,) | float in [0,1], sums to 1 |

### Variants

**Implemented variant: Robust Kalman Filter with EM calibration.**

The paper describes two variants:
1. **Standard Kalman Filter:** The base state-space model with EM calibration.
2. **Robust Kalman Filter:** Adds Lasso-penalized sparse noise detection.

We implement both, with the robust variant as the primary model. The robust
variant subsumes the standard one (setting lambda = infinity recovers the standard
filter, since the threshold becomes infinite and z* = 0 always). The robust variant
is preferred because:
- It achieves marginally better MAPE (0.46 vs 0.47 average across 30 securities).
  (Paper Section 4.2, Table 3.)
- It degrades gracefully under data contamination while CMEM fails entirely at
  medium/large outlier levels. (Paper Section 3.3, Table 1.)
- Real-time market data routinely contains outliers from erroneous prints, trading
  halts, and other anomalies.

## Parameters

| Parameter | Description | Recommended Value | Sensitivity | Range |
|-----------|-------------|-------------------|-------------|-------|
| a_eta | AR(1) coefficient for daily component at day boundaries | Data-driven via EM | High -- controls how quickly the daily level mean-reverts | (0, 1) for stationarity |
| a_mu | AR(1) coefficient for intraday dynamic component | Data-driven via EM | High -- controls decay of intraday shocks | (0, 1) for stationarity |
| sigma_eta_sq | Process noise variance for daily component | Data-driven via EM | Medium -- larger values make eta more volatile day-to-day | (0, inf) |
| sigma_mu_sq | Process noise variance for intraday dynamic component | Data-driven via EM | Medium -- larger values allow faster intraday adaptation | (0, inf) |
| r | Observation noise variance | Data-driven via EM | Medium -- too small overfits; too large smooths away signal | (0, inf) |
| phi[1..I] | Intraday seasonality vector (one per bin) | Data-driven via EM (simple mean of residuals per bin) | Low after estimation -- captures the stable U-shaped pattern | (-inf, inf) in log-space |
| pi_1 | Initial state mean (2x1) | Data-driven via EM (set to smoothed x_hat[1]) | Low -- EM rapidly overrides initialization | R^2 |
| Sigma_1 | Initial state covariance (2x2) | Data-driven via EM (set to smoothed P[1] - x_hat[1]*x_hat[1]^T) | Low -- EM rapidly overrides initialization | Positive semi-definite |
| lambda | Lasso regularization parameter (robust variant only) | Selected by cross-validation | High for robust variant -- controls outlier rejection threshold | (0, inf); larger = less outlier removal |
| N | Training window length (number of bins = T_train * I) | Selected by cross-validation | Medium -- too short = unstable estimates; too long = stale parameters | Practical range: 50-500 trading days worth of bins |
| I | Number of bins per trading day | Exchange-dependent (e.g., 26 for NYSE with 15-min bins over 6.5 hours) | N/A -- determined by data granularity | Typically 20-60 |

### Initialization

**EM initialization (theta_0):**

The paper demonstrates that EM convergence is robust to initialization choice
(Section 2.3.3, Figure 4). Nevertheless, reasonable initial values speed convergence:

1. **phi_init[i]:** Compute the mean of y[t,i] across all training days t for each bin i.
   This captures the gross seasonality pattern.

2. **Compute residuals:** r_init[t,i] = y[t,i] - phi_init[i].

3. **eta_init:** For each day t, compute the mean of r_init[t,i] across bins i.
   Then set a_eta_init from the sample autocorrelation of the eta_init series.
   Set sigma_eta_sq_init from the innovation variance of the eta_init series.

4. **mu_init:** mu_init[t,i] = r_init[t,i] - eta_init[t]. Set a_mu_init from the
   sample autocorrelation of the flattened mu_init series. Set sigma_mu_sq_init
   from the innovation variance.

5. **r_init:** Variance of (y - phi_init - eta_init - mu_init), the unexplained
   residual.

6. **pi_1:** [mean(eta_init), 0].

7. **Sigma_1:** diag([var(eta_init), var(mu_init)]).

8. **lambda_init (robust):** Start with a value proportional to sqrt(r_init),
   then refine via cross-validation.

(Researcher inference: The paper does not specify an initialization procedure
beyond noting insensitivity to initial values in Section 2.3.3. The above procedure
is a reasonable heuristic based on the model structure. Any initialization that
places parameters in roughly the right order of magnitude should work given the
EM's demonstrated robustness.)

### Calibration

**Rolling window EM calibration procedure:**

```
CALIBRATE(y_all, I, N_train, N_val, lambda_grid):
    # y_all: full log-volume time series
    # I: bins per day
    # N_train: training window length (in bins)
    # N_val: validation window length (in bins)
    # lambda_grid: candidate values for Lasso parameter

    # Step 1: Select N and lambda by cross-validation
    # Use a held-out validation period (Paper Section 4.1: Jan-May 2015)
    best_mape = inf
    for N_candidate in N_grid:
        for lambda_candidate in lambda_grid:
            # Train on N_candidate bins before validation period
            theta = EM_ALGORITHM(y_train[1..N_candidate], I, theta_init, max_iter=100, tol=1e-6)
            # Predict on validation period
            mape = evaluate_mape(y_val, theta, lambda_candidate, I)
            if mape < best_mape:
                best_N, best_lambda = N_candidate, lambda_candidate
                best_mape = mape

    # Step 2: Rolling window out-of-sample prediction
    for each out-of-sample day d:
        # Use the most recent best_N bins for training
        y_train = y_all[start_of_window .. start_of_window + best_N - 1]
        theta = EM_ALGORITHM(y_train, I, theta_init, max_iter=100, tol=1e-6)

        # Predict day d using calibrated parameters
        forecasts[d] = ROBUST_KALMAN_FILTER(y_day_d, theta, best_lambda, I)

    return forecasts
```
(Paper Section 4.1, paragraph on cross-validation: "data between January 2015
and May 2015 are considered as a cross-validation set. We repeat performing the
rolling window forecasting on the cross-validation set using different values of
N and lambda, and choose the pair that gives the minimum predictive error rate
as the optimal pair.")

**EM convergence:** The paper shows convergence within "a few iterations" from
various initializations (Section 2.3.3, Figure 4). A practical maximum of 50-100
iterations with a relative log-likelihood tolerance of 1e-6 should suffice.

## Validation

### Expected Behavior

**Volume prediction MAPE (out-of-sample, dynamic mode):**
- Robust Kalman Filter: average MAPE of 0.46 across 30 securities.
- Standard Kalman Filter: average MAPE of 0.47.
- For comparison: CMEM achieves 0.65; rolling mean achieves 1.28.
(Paper Section 4.2, Table 3, bottom row "Average".)

**Volume prediction MAPE (out-of-sample, static mode):**
- Robust Kalman Filter: average MAPE of 0.61.
- Standard Kalman Filter: average MAPE of 0.62.
(Paper Section 4.2, Table 3, bottom row "Average".)

**VWAP tracking error (out-of-sample, dynamic strategy):**
- Robust Kalman Filter: average 6.38 basis points.
- Standard Kalman Filter: 6.39 bps.
- CMEM: 7.01 bps. RM: 7.48 bps.
(Paper Section 4.3, Table 4, bottom row "Average".)

**VWAP tracking error (out-of-sample, static strategy):**
- Robust Kalman Filter: average 6.85 bps.
- Standard Kalman Filter: 8.98 bps.
- CMEM: 8.97 bps. RM: 7.48 bps.
(Paper Section 4.3, Table 4, bottom row "Average".)

**EM convergence behavior:**
- Parameters should converge within 5-20 iterations from reasonable initializations.
- Convergence should be qualitatively similar regardless of initialization
  (different starting points converge to the same values).
(Paper Section 2.3.3, Figure 4.)

**Qualitative behavior of estimated parameters:**
- a_eta should be close to 1 (daily component is highly persistent).
- a_mu should be smaller than a_eta but still positive (intraday dynamics
  mean-revert faster).
- phi should exhibit the characteristic U-shape: high values at market open
  and close, lower values mid-day.
- sigma_eta_sq should be much smaller than sigma_mu_sq (daily level changes
  slowly; intraday dynamics are faster).

### Sanity Checks

1. **Seasonality shape:** After EM calibration, plot phi[1..I]. It should show the
   well-known U-shaped intraday volume pattern: elevated at market open (bins 1-3),
   declining to a trough mid-day, rising again at market close (bins I-2 to I).
   (Paper Section 2, general market microstructure knowledge, and implicit in all
   intraday volume papers in this collection.)

2. **AR coefficient stationarity:** Both a_eta and a_mu should be in (0, 1).
   If either exceeds 1, the model is non-stationary and the EM may have diverged.

3. **Synthetic data recovery:** Generate synthetic data from known parameter values
   {a_eta=0.95, a_mu=0.7, sigma_eta_sq=0.01, sigma_mu_sq=0.05, r=0.1} with a
   U-shaped phi. Run EM and verify that estimated parameters converge to the
   true values within a tolerance (e.g., 10% relative error). This test is inspired
   by the paper's own synthetic validation (Section 2.3.3, Figure 4).

4. **Prediction monotonic improvement:** Dynamic predictions should have lower MAPE
   than static predictions for the same model configuration, since dynamic mode
   incorporates intraday observations. (Paper Table 3: dynamic MAPE < static MAPE
   for all securities.)

5. **Robust vs. standard on clean data:** On clean (curated) data, the robust and
   standard Kalman filters should produce nearly identical results. The z* vector
   should be mostly zeros. (Paper Section 3.3, Table 1, "No outliers" rows show
   comparable performance.)

6. **Robust vs. standard on contaminated data:** Artificially inject outliers into
   10% of bins (multiply volume by 10x or 0.1x). The robust filter MAPE should
   increase only modestly while the standard filter MAPE should degrade
   substantially. (Paper Section 3.3, Table 1.)

7. **MAPE reference values for individual securities:** For SPY with dynamic
   prediction, expect robust Kalman MAPE around 0.24 (Paper Table 3, SPY row).
   For IBM, expect around 0.24 (Paper Table 3). For AAPL, expect around 0.21
   (Paper Table 3).

### Edge Cases

1. **Zero-volume bins:** The model cannot handle bins with zero volume (log(0)
   is undefined). These must be detected and handled before processing. Options:
   - Exclude zero-volume bins and skip the Kalman correction step at those
     time points (treat as missing data -- predict without correcting).
   - Impute with a small positive value (e.g., 1 share).
   - For liquid securities, zero bins are rare and exclusion is acceptable.
   (Paper Section 4.1: "bins with zero volume are excluded.")

2. **Half-day trading sessions:** Sessions with fewer than I bins (e.g., day
   before holidays in U.S. markets) should be excluded entirely, as the
   periodicity model assumes exactly I bins per day. (Paper Section 4.1:
   "excluding half-day sessions.")

3. **Day boundary handling:** The transition from the last bin of day t to the
   first bin of day t+1 must use the day-boundary A_tau (with a_eta coefficient
   and sigma_eta_sq noise) rather than the within-day A_tau. Off-by-one errors
   here will corrupt the daily component estimates.

4. **Numerical stability of covariance updates:** The Joseph form of the
   covariance update (Sigma = (I - K*C) * Sigma * (I - K*C)^T + K*r*K^T)
   is more numerically stable than the standard form (Sigma = Sigma - K*S*K^T)
   for longer time series. Consider using Joseph form if numerical issues arise.
   (Researcher inference: standard Kalman filter best practice, not from paper.)

5. **EM log-likelihood decrease:** In theory the EM log-likelihood is non-decreasing.
   If it decreases, this indicates a numerical bug. Monitor the log-likelihood
   at each iteration as a correctness check.

6. **Very large or very small volumes:** After normalization by shares outstanding,
   some securities may have very small normalized volumes. The log transform
   amplifies differences at small scales. Ensure floating-point precision is
   adequate (use float64).

7. **Cross-market differences in I:** Different exchanges have different trading
   hours, hence different I values. The model and EM must be parameterized per
   security (or at least per exchange). Do not mix securities with different I
   values in a single model instance.

8. **Lambda = 0 in robust variant:** Setting lambda to 0 means every innovation
   is treated as an outlier (z* = e always, no correction to state). This is
   degenerate and should be prevented. Lambda should be strictly positive.

### Known Limitations

1. **Cannot handle zero-volume bins.** Illiquid securities with many zero-volume
   bins are not suitable for this model without an extension for missing data.
   (Paper Section 4.1.)

2. **Single-security model.** Unlike the BDF model (Direction 2), this model
   operates on one security at a time. Cross-sectional information is not used.

3. **No exogenous variables.** The model has no mechanism to incorporate external
   covariates (volatility, spreads, events, overnight gaps). Incorporating such
   information would require extending the state-space formulation.
   (Paper Section 5, identified as future work.)

4. **Fixed bin granularity.** The model assumes a fixed number of bins I per day.
   Adaptive or irregular time grids are not supported.

5. **Gaussian noise assumption.** While the log-transform makes this more
   defensible than raw volume, heavy-tailed residuals can still occur. The robust
   variant partially mitigates this but does not fully model non-Gaussian noise.

6. **No comparison with BDF or GAS-Dirichlet.** The paper benchmarks against
   CMEM and rolling means only. Relative performance against Direction 2 (BDF)
   or Direction 3 (GAS-Dirichlet) is unknown.
   (Paper Section 5, Conclusion.)

## Paper References

| Spec Section | Paper Source |
|---|---|
| Model decomposition (y = eta + phi + mu + v) | Chen et al. (2016), Section 2, Equation 3 |
| State-space formulation | Chen et al. (2016), Section 2, Equations 4-5 |
| Time-varying A_tau and Q_tau | Chen et al. (2016), Section 2, page 3 (bullet points defining A_tau, Q_tau) |
| C = [1, 1] observation matrix | Chen et al. (2016), Section 2, page 3 |
| Kalman filter algorithm | Chen et al. (2016), Section 2.2, Algorithm 1 |
| Dynamic vs. static prediction | Chen et al. (2016), Section 2.2, paragraph following Algorithm 1 |
| Multi-step prediction formula | Chen et al. (2016), Section 2.2, Equation 9 |
| Kalman smoother (RTS) | Chen et al. (2016), Section 2.3.1, Algorithm 2 |
| Cross-covariance computation | Chen et al. (2016), Appendix A, Equations A.20-A.22 |
| EM algorithm structure | Chen et al. (2016), Section 2.3.2, Algorithm 3 |
| EM M-step: pi_1, Sigma_1 | Chen et al. (2016), Appendix A, Equations A.32-A.33 |
| EM M-step: a_eta | Chen et al. (2016), Appendix A, Equation A.34 |
| EM M-step: a_mu | Chen et al. (2016), Appendix A, Equation A.35 |
| EM M-step: sigma_eta_sq | Chen et al. (2016), Appendix A, Equation A.36 |
| EM M-step: sigma_mu_sq | Chen et al. (2016), Appendix A, Equation A.37 |
| EM M-step: r | Chen et al. (2016), Appendix A, Equation A.38 |
| EM M-step: phi | Chen et al. (2016), Appendix A, Equation A.39 (also Eq 24 in main text) |
| EM convergence robustness | Chen et al. (2016), Section 2.3.3, Figure 4 |
| Robust observation model (z_tau) | Chen et al. (2016), Section 3.1, Equations 25-27 |
| Lasso in Kalman correction | Chen et al. (2016), Section 3.1, Equations 28-30 |
| Soft-thresholding solution | Chen et al. (2016), Section 3.1, Equations 33-34 |
| Robust EM: modified r | Chen et al. (2016), Section 3.2, Equation 35 |
| Robust EM: modified phi | Chen et al. (2016), Section 3.2, Equation 36 |
| MAPE definition | Chen et al. (2016), Section 3.3, Equation 37 |
| VWAP definition | Chen et al. (2016), Section 4.3, Equation 39 |
| Static VWAP weights | Chen et al. (2016), Section 4.3, Equation 40 |
| Dynamic VWAP weights | Chen et al. (2016), Section 4.3, Equation 41 |
| VWAP tracking error definition | Chen et al. (2016), Section 4.3, Equation 42 |
| Cross-validation for N, lambda | Chen et al. (2016), Section 4.1, paragraph on cross-validation |
| Out-of-sample MAPE results | Chen et al. (2016), Section 4.2, Table 3 |
| VWAP tracking error results | Chen et al. (2016), Section 4.3, Table 4 |
| Robustness to outliers | Chen et al. (2016), Section 3.3, Table 1 |
| Data description (30 securities) | Chen et al. (2016), Section 4.1, Table 2 |
| EM initialization procedure | Researcher inference (see Initialization section) |
| Joseph form covariance update | Researcher inference (standard Kalman filter practice) |
| Zero-volume bin handling options | Researcher inference, with paper noting exclusion in Section 4.1 |
