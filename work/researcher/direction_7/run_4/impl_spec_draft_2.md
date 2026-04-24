# Implementation Specification: Kalman Filter State-Space Model for Intraday Volume

## Overview

This model forecasts intraday trading volume by decomposing log-volume into three
additive components -- a daily average, an intraday seasonal pattern, and an intraday
dynamic deviation -- and tracking them via a linear Gaussian state-space model. The
Kalman filter provides optimal one-step-ahead and multi-step-ahead predictions, the
Rauch-Tung-Striebel (RTS) smoother provides full-sample state estimates for calibration,
and the EM algorithm estimates all model parameters in closed form. A robust extension
adds Lasso-penalized sparse noise detection for automatic outlier handling. The approach
converts the multiplicative volume decomposition of Brownlees et al. (2011) into a
tractable linear additive model via logarithmic transformation, enabling exact Kalman
recursions with a minimal 2-dimensional state.

Source: Chen, Feng, Palomar (2016), "Forecasting Intraday Trading Volume: A Kalman
Filter Approach." All algorithmic details, equations, and results are from this single
paper unless otherwise noted.

## Algorithm

### Model Description

The model operates on log-transformed intraday volume. Raw volume in each bin is first
normalized by daily shares outstanding (to correct for splits and enable cross-asset
comparison), then the natural logarithm is taken. This log transformation is central:
it eliminates positiveness constraints, reduces right-skewness, and makes Gaussian noise
assumptions defensible (Paper, Section 2, Figure 1).

The log-volume observation y_{t,i} for day t, bin i (where t = 1..D days, i = 1..I bins
per day) is decomposed as:

    y_{t,i} = eta_t + phi_i + mu_{t,i} + v_{t,i}

where:
- eta_t is the log daily average component (changes only at day boundaries)
- phi_i is the log intraday periodic component (one value per bin, constant across days)
- mu_{t,i} is the log intraday dynamic component (captures short-term deviations)
- v_{t,i} ~ N(0, r) is observation noise

Assumptions:
- Volume in every bin is strictly positive (log(0) is undefined; zero-volume bins must
  be excluded or imputed before model application).
- All noise terms are Gaussian.
- eta_t is piecewise constant within each day (changes only at day transitions).
- phi_i is stationary across the training window (same seasonal shape every day).
- mu_{t,i} follows a first-order autoregressive process within and across days.

Input: time series of normalized log-volume observations {y_{t,i}} for t=1..D, i=1..I.
Output: one-step-ahead or multi-step-ahead log-volume forecasts, convertible to volume
via exponentiation.

(Paper, Section 2, Equation 3)

### State-Space Formulation

To apply the Kalman filter, the model is rewritten with a single time index tau that
runs sequentially through all bins across all days. The mapping is:

    tau = (t - 1) * I + i

so tau = 1, 2, ..., N where N = D * I is the total number of bins in the dataset.

The state-space model is:

    State transition:   x_{tau+1} = A_tau * x_tau + w_tau
    Observation:        y_tau = C * x_tau + phi_tau + v_tau

where:

- x_tau = [eta_tau, mu_tau]^T is the 2x1 hidden state vector.

- C = [1, 1] is the 1x2 observation matrix (both components contribute additively
  to the observation).

- phi_tau is the seasonal component for the bin corresponding to time tau. If tau
  corresponds to bin i of day t, then phi_tau = phi_i.

- A_tau is the 2x2 state transition matrix:
      A_tau = [[a_tau^eta, 0], [0, a^mu]]
  where:
      a_tau^eta = a^eta    if tau = k*I (day boundary, i.e., transitioning to a new day)
      a_tau^eta = 1         otherwise (within-day, eta is constant)

- w_tau ~ N(0, Q_tau) is state noise with diagonal covariance:
      Q_tau = [[sigma_tau_eta^2, 0], [0, (sigma^mu)^2]]
  where:
      sigma_tau_eta^2 = (sigma^eta)^2   if tau = k*I (day boundary)
      sigma_tau_eta^2 = 0                otherwise (no eta noise within a day)

- v_tau ~ N(0, r) is scalar observation noise.

(Paper, Section 2, Equations 4-5, and the bullet definitions on page 3)

### Pseudocode

The implementation consists of four main algorithms plus two VWAP execution strategies.

#### Algorithm 1: Kalman Filter (Prediction)

Purpose: Given known parameters theta, produce one-step-ahead or multi-step-ahead
log-volume forecasts from sequential observations. Also stores intermediate quantities
needed by the smoother (Algorithm 2) and EM (Algorithm 3).

```
Input:
  y[1..N]         -- observed log-volume series (N = D * I bins total)
  theta            -- model parameters: a_eta, a_mu, sigma_eta_sq, sigma_mu_sq, r,
                      phi[1..I], pi_1 (2x1), Sigma_1 (2x2)
  mode             -- "dynamic" or "static"

Output:
  y_hat[1..N]      -- one-step-ahead (dynamic) or multi-step-ahead (static) forecasts
  x_hat[1..N]      -- filtered state estimates (for EM)
  Sigma[1..N]      -- filtered state covariances (for EM)
  x_pred[1..N]     -- predicted state means before correction (for smoother)
  Sigma_pred[1..N] -- predicted state covariances before correction (for smoother)
  A_store[1..N]    -- transition matrices at each step (for smoother)
  K_store[1..N]    -- Kalman gain at each step (for smoother cross-covariance init)
  S_store[1..N]    -- innovation variance at each step (for log-likelihood)

Initialization:
  x_hat[0] = pi_1                    # initial state mean
  Sigma[0] = Sigma_1                 # initial state covariance

For tau = 1, 2, ..., N:
    # --- Predict step ---
    # Determine A_tau based on whether this is a day boundary
    if tau mod I == 1:                # first bin of a new day
        A = [[a_eta, 0], [0, a_mu]]
        Q = [[sigma_eta_sq, 0], [0, sigma_mu_sq]]
    else:
        A = [[1, 0], [0, a_mu]]
        Q = [[0, 0], [0, sigma_mu_sq]]

    A_store[tau] = A                 # store for smoother
    x_pred[tau] = A @ x_hat[tau-1]   # predicted state mean (2x1)
    Sigma_pred[tau] = A @ Sigma[tau-1] @ A^T + Q   # predicted state covariance (2x2)

    # --- Log-volume forecast ---
    phi_current = phi[((tau-1) mod I) + 1]   # seasonal for this bin
    y_hat[tau] = C @ x_pred[tau] + phi_current    # scalar forecast

    # --- Innovation variance (always compute for log-likelihood) ---
    S_store[tau] = C @ Sigma_pred[tau] @ C^T + r   # scalar

    if mode == "dynamic":
        # --- Correct step (incorporate observation) ---
        # Kalman gain (2x1)
        K = Sigma_pred[tau] @ C^T / S_store[tau]

        # Innovation (scalar)
        e = y[tau] - y_hat[tau]              # prediction error

        # Update state
        x_hat[tau] = x_pred[tau] + K * e    # 2x1
        Sigma[tau] = Sigma_pred[tau] - K @ C @ Sigma_pred[tau]   # 2x2

        K_store[tau] = K                     # store for smoother

    else:  # static mode: no correction, just propagate
        x_hat[tau] = x_pred[tau]
        Sigma[tau] = Sigma_pred[tau]
        K_store[tau] = zeros(2,1)            # no correction applied

Return y_hat, x_hat, Sigma, x_pred, Sigma_pred, A_store, K_store, S_store
```

Notes:
- In dynamic mode, the filter updates after every observed bin (real-time forecasting).
- In static mode, correction steps are skipped; the filter produces multi-step-ahead
  forecasts using only information from the prior day's close.
- Since the state is 2x1 and the observation is scalar, all matrix inversions reduce
  to scalar divisions. The Kalman gain K is a 2x1 vector = Sigma_pred @ C^T / S.
- The day-boundary condition "tau mod I == 1" assumes tau is 1-indexed and the first
  bin of each day is at positions 1, I+1, 2I+1, etc.
- x_pred, Sigma_pred, A_store, and K_store must be stored for all tau because the
  smoother (Algorithm 2) uses them in its backward pass. S_store is needed for the
  innovation-form log-likelihood used in EM convergence checking.

(Paper, Section 2.2, Algorithm 1)

#### Algorithm 2: Kalman Smoother (RTS Backward Pass)

Purpose: Given filtered estimates from Algorithm 1, compute smoothed state estimates
using all observations in the training window. Required for the EM E-step. This
algorithm also computes the cross-covariance sufficient statistics needed by the M-step.

```
Input:
  x_hat[1..N]      -- filtered state means from Kalman filter
  Sigma[1..N]      -- filtered state covariances from Kalman filter
  x_pred[1..N]     -- predicted state means (before correction) from Kalman filter
  Sigma_pred[1..N] -- predicted state covariances (before correction)
  A_store[1..N]    -- transition matrices used at each step
  K_store[N]       -- Kalman gain at last step (for cross-covariance initialization)

Output:
  x_smooth[1..N]   -- smoothed state means
  Sigma_smooth[1..N] -- smoothed state covariances
  P[1..N]          -- smoothed second moments P_tau = E[x_tau x_tau^T | all data]
  P_cross[2..N]    -- smoothed cross-second-moments P_{tau,tau-1}

# ---------------------------------------------------------------
# Part 1: RTS backward smoothing pass
# ---------------------------------------------------------------

Initialization:
  x_smooth[N] = x_hat[N]
  Sigma_smooth[N] = Sigma[N]

For tau = N-1, N-2, ..., 1:
    # Smoother gain (2x2)
    L[tau] = Sigma[tau] @ A_store[tau+1]^T @ inv(Sigma_pred[tau+1])

    # Smoothed state
    x_smooth[tau] = x_hat[tau] + L[tau] @ (x_smooth[tau+1] - x_pred[tau+1])

    # Smoothed covariance
    Sigma_smooth[tau] = Sigma[tau] + L[tau] @ (Sigma_smooth[tau+1] - Sigma_pred[tau+1]) @ L[tau]^T

# ---------------------------------------------------------------
# Part 2: Cross-covariance backward recursion (for EM M-step)
# ---------------------------------------------------------------

# Initialize at tau = N:
#   Sigma_{N,N-1|N} = (I - K_N @ C) @ A_store[N] @ Sigma[N-1]
# where K_N is the Kalman gain at the last step from Algorithm 1.
# (Paper, Appendix A, Equation A.21)

Sigma_cross[N] = (eye(2) - K_store[N] @ C) @ A_store[N] @ Sigma[N-1]

# Backward recursion for tau = N-1, N-2, ..., 2:
For tau = N-1, N-2, ..., 2:
    Sigma_cross[tau] = Sigma[tau] @ L[tau-1]^T
        + L[tau] @ (Sigma_cross[tau+1] - A_store[tau+1] @ Sigma[tau]) @ L[tau-1]^T

# (Paper, Appendix A, Equation A.20)

# ---------------------------------------------------------------
# Part 3: Compute sufficient statistics for EM
# ---------------------------------------------------------------

For tau = 1..N:
    P[tau] = Sigma_smooth[tau] + x_smooth[tau] @ x_smooth[tau]^T

For tau = 2..N:
    P_cross[tau] = Sigma_cross[tau] + x_smooth[tau] @ x_smooth[tau-1]^T

# (Paper, Appendix A, Equations A.18-A.19, A.22)

Return x_smooth, Sigma_smooth, P, P_cross
```

Notes:
- The smoother gain L[tau] involves inverting Sigma_pred[tau+1], which is 2x2. For
  numerical stability, use the analytic 2x2 inverse formula rather than a general solver.
  For matrix [[a,b],[c,d]], inverse = (1/(ad-bc)) * [[d,-b],[-c,a]].
- L[tau] must be stored for all tau because the cross-covariance recursion uses it.
- The cross-covariance initialization at tau=N (Equation A.21) requires K_N, which must
  be passed from Algorithm 1. This is the Kalman gain at the final time step.
- The paper defines the sufficient statistics using superscript notation:
  P_tau^{(1,1)} = first element of P[tau] (corresponding to eta*eta)
  P_tau^{(2,2)} = second element (mu*mu)
  P_{tau,tau-1}^{(1,1)} = (1,1) element of P_cross[tau]
  P_{tau,tau-1}^{(2,2)} = (2,2) element of P_cross[tau]

(Paper, Section 2.3.1, Algorithm 2; cross-covariance from Appendix A, Equations A.20-A.22)

#### Algorithm 3: EM Algorithm (Parameter Estimation)

Purpose: Estimate all model parameters theta from the training data by iterating
E-step (filter + smoother) and M-step (closed-form updates) until convergence.

```
Input:
  y[1..N]          -- training log-volume series
  theta_init       -- initial parameter values (can be arbitrary; EM is robust to init)
  max_iter         -- maximum EM iterations (e.g., 50)
  tol              -- convergence tolerance on relative log-likelihood change (e.g., 1e-6)

Output:
  theta_final      -- estimated parameters

theta = theta_init
log_L_prev = -inf
j = 0

Repeat:
    j = j + 1

    # --- E-step ---
    # Run Kalman filter (Algorithm 1) in dynamic mode with current theta
    # Returns: x_hat, Sigma, x_pred, Sigma_pred, A_store, K_store, S_store
    y_hat, x_hat, Sigma, x_pred, Sigma_pred, A_store, K_store, S_store =
        KalmanFilter(y, theta, mode="dynamic")

    # Compute innovation-form log-likelihood for convergence check:
    # log L = -0.5 * sum_{tau=1}^{N} [log(2*pi*S_store[tau]) + e[tau]^2 / S_store[tau]]
    # where e[tau] = y[tau] - y_hat[tau] is the innovation (prediction error)
    # (Paper, Appendix A.1, Equation A.8; standard Kalman filter innovation form)
    log_L = -0.5 * sum_{tau=1}^{N} (log(2*pi*S_store[tau]) + (y[tau] - y_hat[tau])^2 / S_store[tau])

    # Run Kalman smoother (Algorithm 2)
    # Returns: x_smooth, Sigma_smooth, P, P_cross
    x_smooth, Sigma_smooth, P, P_cross =
        KalmanSmoother(x_hat, Sigma, x_pred, Sigma_pred, A_store, K_store[N])

    # --- M-step (all closed-form updates) ---
    # IMPORTANT: The ordering of updates matters. phi must be computed before r
    # because the r update (Eq A.38) uses the newly updated phi^{(j+1)}.
    # All other updates depend only on E-step quantities and can be computed
    # in any order.

    # Initial state mean (2x1):
    pi_1 = x_smooth[1]                                    # Eq A.32 / Eq 17

    # Initial state covariance (2x2):
    Sigma_1 = P[1] - x_smooth[1] @ x_smooth[1]^T          # Eq A.33 / Eq 18

    # Daily AR coefficient a^eta:
    # Sum only over day-boundary time steps (tau in D where D = {I+1, 2I+1, ...})
    numerator_eta = sum_{tau in D} P_cross[tau]^{(1,1)}
    denominator_eta = sum_{tau in D} P[tau-1]^{(1,1)}
    a_eta = numerator_eta / denominator_eta                # Eq A.34 / Eq 19

    # Enforce stationarity: clamp to (0, 1)
    # (Researcher inference: the EM update is unconstrained; clamping prevents
    # nonstationary estimates. If the unclamped value falls outside (0, 1),
    # log a warning as it may indicate model misspecification or insufficient data.)
    if a_eta <= 0 or a_eta >= 1:
        warn("a_eta = {a_eta} outside (0,1); clamping for stationarity")
        a_eta = clip(a_eta, 0.001, 0.999)

    # Intraday AR coefficient a^mu:
    numerator_mu = sum_{tau=2}^{N} P_cross[tau]^{(2,2)}
    denominator_mu = sum_{tau=2}^{N} P[tau-1]^{(2,2)}
    a_mu = numerator_mu / denominator_mu                   # Eq A.35 / Eq 20

    # Enforce stationarity: clamp to (0, 1)
    if a_mu <= 0 or a_mu >= 1:
        warn("a_mu = {a_mu} outside (0,1); clamping for stationarity")
        a_mu = clip(a_mu, 0.001, 0.999)

    # Daily process noise variance (sigma^eta)^2:
    # D_count = number of day boundaries = T - 1 (where T = number of days)
    (sigma_eta)^2 = (1/(T-1)) * sum_{tau in D} {
        P[tau]^{(1,1)} + (a_eta)^2 * P[tau-1]^{(1,1)}
        - 2 * a_eta * P_cross[tau]^{(1,1)}
    }                                                      # Eq A.36 / Eq 21

    # Intraday process noise variance (sigma^mu)^2:
    (sigma_mu)^2 = (1/(N-1)) * sum_{tau=2}^{N} {
        P[tau]^{(2,2)} + (a_mu)^2 * P[tau-1]^{(2,2)}
        - 2 * a_mu * P_cross[tau]^{(2,2)}
    }                                                      # Eq A.37 / Eq 22

    # Seasonality vector phi_i for each bin i = 1..I:
    # MUST be computed BEFORE r, because the r update uses phi^{(j+1)}.
    # phi depends only on E-step quantities (smoothed states), not on other M-step
    # parameters, so it can be computed independently.
    phi[i] = (1/T) * sum_{t=1}^{T} (y_{t,i} - C @ x_smooth_{t,i})  # Eq A.39 / Eq 24
    # where x_smooth_{t,i} is the smoothed state for day t, bin i.

    # Observation noise variance r:
    # This uses the NEWLY UPDATED phi^{(j+1)} (not the old phi from the previous
    # iteration). The (j+1) superscript on phi in Eq A.38 is explicit about this
    # dependency: the M-step jointly sets partial derivatives to zero (Eqs A.30-A.31),
    # and the closed-form for r depends on the new phi.
    r = (1/N) * sum_{tau=1}^{N} {
        y[tau]^2 + C @ P[tau] @ C^T
        - 2 * y[tau] * C @ x_smooth[tau]
        + (phi[tau])^2
        - 2 * y[tau] * phi[tau]
        + 2 * phi[tau] * C @ x_smooth[tau]
    }                                                      # Eq A.38 / Eq 23

    # Equivalently (more compact, numerically preferable):
    r = (1/N) * sum_{tau=1}^{N} {
        (y[tau] - phi[tau] - C @ x_smooth[tau])^2
        + C @ Sigma_smooth[tau] @ C^T
    }
    # NOTE: phi[tau] here refers to the NEWLY computed phi values from the line above.

    # --- Check convergence ---
    # Use relative change in observed-data log-likelihood (innovation form).
    # This is well-defined and scale-invariant, unlike parameter-change criteria
    # which mix quantities of different scales.
    if j > 1 and abs(log_L - log_L_prev) / abs(log_L_prev) < tol:
        break
    if j >= max_iter:
        break

    log_L_prev = log_L
    theta = {pi_1, Sigma_1, a_eta, a_mu, sigma_eta^2, sigma_mu^2, r, phi[1..I]}

Return theta
```

Notes on EM M-step derivation:
- All updates are derived by setting partial derivatives of the expected complete-data
  log-likelihood Q(theta | theta_old) to zero. Each derivative yields a closed-form
  solution because the model is linear-Gaussian. See Appendix A, Equations A.24-A.39.
- The set D = {kI + 1 : k = 1, 2, ...} indexes all day-boundary transitions. The sum
  for a^eta and (sigma^eta)^2 runs only over these indices because eta transitions only
  at day boundaries.
- The sum for a^mu and (sigma^mu)^2 runs over all tau >= 2 because mu transitions at
  every step.
- The phi update is a simple per-bin average of (observation - state prediction) over
  all training days, computed by grouping time indices by their bin position.
- The r update uses the smoothed covariance (Sigma_smooth, not the outer-product P) in
  the compact form. Both forms are equivalent; the compact form is numerically preferable.
- **M-step ordering constraint:** phi must be updated before r. The paper's Equation A.38
  uses phi^{(j+1)} (with the (j+1) superscript), meaning the r update requires the
  newly estimated phi from the current M-step, not the old phi. Since phi depends only
  on E-step quantities (Eq A.39), it can be computed first without any M-step dependency.
  All other parameter updates (pi_1, Sigma_1, a^eta, a^mu, sigma^eta, sigma^mu) depend
  only on E-step sufficient statistics and can be computed in any order relative to each
  other and to phi/r. (Paper, Appendix A.3, Equations A.38-A.39)

(Paper, Section 2.3.2, Algorithm 3; Appendix A for derivation)

#### Algorithm 4: Robust Kalman Filter (Lasso Extension)

Purpose: Handle outliers in real-time market data by adding a sparse noise term z_tau
to the observation equation and applying Lasso regularization.

The modified observation equation is:

    y_tau = C * x_tau + phi_tau + v_tau + z_tau

where z_tau is a sparse outlier term (zero most of the time, large when an outlier
is present).

```
Input:
  y[1..N]          -- observed log-volume series (may contain outliers)
  theta            -- model parameters including lambda (regularization)
  mode             -- "dynamic" or "static"

Output:
  y_hat[1..N]      -- forecasts
  x_hat[1..N]      -- filtered state estimates (for robust EM)
  Sigma[1..N]      -- filtered state covariances (for robust EM)
  x_pred[1..N]     -- predicted state means (for smoother)
  Sigma_pred[1..N] -- predicted state covariances (for smoother)
  A_store[1..N]    -- transition matrices (for smoother)
  K_store[1..N]    -- Kalman gains (for smoother)
  S_store[1..N]    -- innovation variances (for log-likelihood)
  z_star[1..N]     -- inferred outlier values (mostly zero)

The Kalman filter proceeds as in Algorithm 1, but the correction step is modified.

For each tau in dynamic mode:
    # Predict step: identical to Algorithm 1
    A, Q = construct_AQ(tau, theta)   # same day-boundary logic
    A_store[tau] = A
    x_pred[tau] = A @ x_hat[tau-1]
    Sigma_pred[tau] = A @ Sigma[tau-1] @ A^T + Q

    # Forecast (same as standard)
    phi_current = phi[((tau-1) mod I) + 1]
    y_hat[tau] = C @ x_pred[tau] + phi_current

    # Innovation
    e_tau = y[tau] - phi_current - C @ x_pred[tau]

    # Innovation variance and precision (scalar)
    S_store[tau] = C @ Sigma_pred[tau] @ C^T + r
    W_tau = 1.0 / S_store[tau]

    # Solve Lasso subproblem for z_tau (soft-thresholding):
    threshold = lambda / (2 * W_tau)

    if e_tau > threshold:
        z_star[tau] = e_tau - threshold
    elif e_tau < -threshold:
        z_star[tau] = e_tau + threshold
    else:
        z_star[tau] = 0

    # Modified correction (subtract outlier from innovation):
    e_clean = e_tau - z_star[tau]

    # Kalman gain (same formula as standard)
    K = Sigma_pred[tau] @ C^T * W_tau              # 2x1 vector

    # State update using cleaned innovation
    x_hat[tau] = x_pred[tau] + K * e_clean
    Sigma[tau] = Sigma_pred[tau] - K @ C @ Sigma_pred[tau]

    K_store[tau] = K

Return y_hat, x_hat, Sigma, x_pred, Sigma_pred, A_store, K_store, S_store, z_star
```

Notes:
- The soft-thresholding has an analytic closed-form solution because W_tau is scalar
  (the state is 2D and observation is 1D). This is Equation 33 in the paper.
- The threshold lambda/(2*W_tau) is time-varying because W_tau depends on the current
  predictive variance. When the model is uncertain (large Sigma_pred), W_tau is small,
  so the threshold is large -- the model tolerates larger deviations before flagging
  them as outliers. This is a desirable adaptive property.
- The quantity e_tau - z_star[tau] represents the "cleaned" innovation after removing
  the outlier component. The standard Kalman correction is then applied to this cleaned
  innovation.
- The covariance update Sigma[tau] is the same as the standard Kalman filter -- it does
  not depend on the innovation value, only on the gain and prediction covariance.
- lambda is selected by cross-validation (not estimated by EM).

(Paper, Section 3.1, Equations 29-34)

#### Robust EM Modifications

When estimating parameters for the robust model, the EM algorithm is modified as follows:

**E-step:** The forward pass uses Algorithm 4 (the robust Kalman filter) instead of
Algorithm 1. This produces both the filtered states and the z_star values. The RTS
smoother (Algorithm 2) is then applied to the filtered outputs of Algorithm 4 without
modification -- the smoother operates on the "cleaned" filtered states produced by the
robust filter. The z_star values from the forward pass are stored for use in the M-step.

(Paper, Section 3.2, paragraph before Eq 35: "z_1*...z_N* denote the optimal outlier
values obtained from solving Problem (30)" -- these come from the robust filter's
forward pass and feed into the modified M-step.)

**M-step:** The updates for r and phi are modified to account for the inferred outlier
terms z_star. All other updates (pi_1, Sigma_1, a^eta, a^mu, sigma^eta, sigma^mu)
remain unchanged because the outlier term affects only the observation equation, not
the state transition.

```
# Modified seasonality (computed BEFORE r, same ordering constraint as standard EM):
phi[i] = (1/T) * sum_{t=1}^{T} (y_{t,i} - C @ x_smooth_{t,i} - z_star_{t,i})  # Eq 36

# Modified observation noise variance (uses newly updated phi):
r = (1/N) * sum_{tau=1}^{N} {
    (y[tau] - phi[tau] - C @ x_smooth[tau] - z_star[tau])^2
    + C @ Sigma_smooth[tau] @ C^T
}                                                          # Eq 35
```

(Paper, Section 3.2, Equations 35-36)

#### VWAP Execution Strategies

Two VWAP execution strategies convert volume forecasts into trading schedules:

**Static VWAP** (Paper, Section 4.3, Equation 40):
Compute volume weights before market open using static predictions for all I bins of the
upcoming day. The weight for bin i is:

    w_{t,i}^(s) = volume_hat_{t,i}^(s) / sum_{j=1}^{I} volume_hat_{t,j}^(s)

where volume_hat_{t,i}^(s) = exp(y_hat_{t,i}^(s)) is the static volume forecast
converted from log scale.

**Dynamic VWAP** (Paper, Section 4.3, Equation 41):
Revise weights at each bin using the latest dynamic volume prediction. After observing
bins 1 through i-1, the weight for bin i is:

    w_{t,i}^(d) = (volume_hat_{t,i}^(d) / sum_{j=i}^{I} volume_hat_{t,j}^(d))
                  * (1 - sum_{j=1}^{i-1} w_{t,j}^(d))

For the last bin (i = I):
    w_{t,I}^(d) = 1 - sum_{j=1}^{I-1} w_{t,j}^(d)

In the denominator sum_{j=i}^{I}, the forecast for bin i uses the one-step-ahead
dynamic prediction (conditioned on observations up to bin i-1). The forecasts for bins
i+1 through I use multi-step-ahead predictions from the current filtered state: propagate
the predict step forward without correction for each remaining bin. These are NOT the
pre-market static forecasts -- they are forward projections from the latest posterior
state after observing bins 1 through i-1.

(Paper, Section 4.3, Equation 41 and surrounding text on page 10)

### Data Flow

```
Raw volume per bin (shares traded)
    |
    v
Normalize by daily shares outstanding
    |
    v
volume_{t,i} = shares_traded_{t,i} / shares_outstanding_t
    |
    v
Take natural logarithm
    |
    v
y_{t,i} = ln(volume_{t,i})              [N = D*I scalar observations]
    |
    v
Flatten to single time series: y[tau] for tau = 1..N
    |
    +--> Training window: y[1..N_train]
    |       |
    |       v
    |    EM Algorithm (Algorithm 3):
    |       Loop until convergence:
    |         E-step: Kalman Filter (Alg 1 or 4) -> Kalman Smoother (Alg 2)
    |           Input:  y[1..N_train], theta^(j)
    |           Output: x_smooth[1..N], Sigma_smooth[1..N], P[1..N], P_cross[2..N]
    |                   (also: z_star[1..N] if using robust filter)
    |         M-step: Closed-form parameter updates (phi before r)
    |           Output: theta^(j+1)
    |       |
    |       v
    |    Estimated parameters theta = {a_eta, a_mu, sigma_eta^2, sigma_mu^2, r,
    |                                   phi[1..I], pi_1, Sigma_1}
    |
    +--> Out-of-sample: y[N_train+1..N]
            |
            v
         Kalman Filter (Alg 1 or 4) with estimated theta:
           Dynamic mode: correct after each bin
           Static mode: predict full day without corrections
            |
            v
         y_hat[tau] = C @ x_pred + phi_{bin(tau)}    [log-volume forecast]
            |
            v
         volume_hat[tau] = exp(y_hat[tau])            [volume forecast]
            |
            v
         VWAP weights (static or dynamic strategy)
```

**Log-normal bias note:** The conversion volume_hat = exp(y_hat) produces the
**conditional mode** (most likely value) of the volume distribution, not the
conditional mean. For a Gaussian Y ~ N(mu, sigma^2), E[exp(Y)] = exp(mu + sigma^2/2),
so exp(y_hat) systematically underestimates the expected volume. The bias-corrected
estimator is:

    volume_hat_corrected = exp(y_hat + S_tau / 2)

where S_tau is the predictive variance of y_hat (the innovation variance from the
Kalman filter). However, the paper does not apply this correction, and all MAPE
benchmarks (Table 3) use exp(y_hat) without correction. For VWAP weights (which are
ratios), the bias largely cancels if all bins have similar predictive variance.

**Recommendation:** Use exp(y_hat) without correction for MAPE benchmarking (to match
the paper's reported numbers). For applications requiring unbiased absolute volume
estimates, apply the bias correction.

(Researcher inference: the bias correction is a standard log-normal property not
discussed in the paper. The paper's MAPE results use the uncorrected form.)

Types and shapes at each stage:
- y[tau]: scalar (float64), log-volume for one bin
- x_tau: 2x1 vector [eta, mu]
- A_tau: 2x2 matrix (time-varying)
- Q_tau: 2x2 diagonal matrix (time-varying)
- C: 1x2 vector [1, 1]
- phi: I-vector (one entry per bin)
- Sigma_tau: 2x2 covariance matrix
- K_tau: 2x1 Kalman gain vector
- S_tau (innovation variance): scalar
- W_tau (innovation precision): scalar = 1/S_tau

### Variants

This specification implements the full model from Chen, Feng, Palomar (2016), which
has two variants:

1. **Standard Kalman Filter** -- the base model without outlier handling.
2. **Robust Kalman Filter** -- adds Lasso-penalized sparse noise term for outlier
   detection.

Both variants should be implemented. The robust variant is recommended as the primary
model for production use because:
- It achieves slightly better MAPE (0.46 vs 0.47 average across 30 securities) on clean
  data and degrades much more gracefully under contamination (Paper, Table 1, Section 3.3).
- The CMEM benchmark fails entirely (N/A) under medium and large outlier conditions,
  while the robust Kalman filter continues to produce usable forecasts.
- The computational overhead of the Lasso step is negligible (one soft-thresholding
  operation per bin).

The standard variant is simpler and useful as a debugging baseline: any implementation
bug in the robust path that does not appear in the standard path is isolated to the
Lasso extension.

(Paper, Section 3, Table 1)

## Parameters

| Parameter | Description | Recommended Value | Sensitivity | Range |
|-----------|-------------|-------------------|-------------|-------|
| a^eta | Day-to-day AR(1) coefficient for the daily component eta. Controls persistence of daily volume level across days. | Data-driven (EM). Typically close to 1 for persistent daily volume. Synthetic experiment converges near true value from any initialization (Paper, Figure 4a). | Medium. Controls how quickly the daily level adapts to regime changes. Values too far from 1 would cause the daily component to mean-revert too fast or become nonstationary. | (0, 1) for stationarity. |
| a^mu | Bin-to-bin AR(1) coefficient for the intraday dynamic component mu. Controls persistence of intraday deviations. | Data-driven (EM). Typically moderate (allows short-term dynamics to propagate but not dominate). Converges reliably from any init (Paper, Figure 4b). | Medium-high. Determines how much the last bin's dynamic deviation influences the next bin's forecast. Key parameter for dynamic prediction quality. | (0, 1) for stationarity. |
| (sigma^eta)^2 | Process noise variance for the daily component at day boundaries. Controls how much eta can change day-to-day. | Data-driven (EM). Converges reliably (Paper, Figure 4d). | Low-medium. Larger values allow more day-to-day variation; smaller values make eta more persistent. Indirectly coupled with a^eta. | (0, inf). Typically much smaller than (sigma^mu)^2 since daily changes are smooth. |
| (sigma^mu)^2 | Process noise variance for the intraday dynamic component at every bin. Controls how much mu can change bin-to-bin. | Data-driven (EM). Converges reliably (Paper, Figure 4e). | Low-medium. Similar role to sigma^eta but at the intraday frequency. | (0, inf). |
| r | Observation noise variance. Represents measurement noise not captured by the state components. | Data-driven (EM). Converges reliably (Paper, Figure 4c). | Low. Primarily affects confidence intervals and Kalman gain magnitude. Smaller r gives more weight to observations vs. predictions. | (0, inf). |
| phi[1..I] | Intraday seasonality vector. One value per bin capturing the average U-shaped (or exchange-specific) intraday volume pattern. | Data-driven (EM). Estimated as mean residual per bin over training days. Converges reliably (Paper, Figure 4f). | High. This is the dominant predictable component of intraday volume. An incorrect seasonal pattern would systematically bias all forecasts. | Unrestricted (log scale). Typically exhibits the well-known U-shape: high at open and close, low midday. |
| pi_1 | Initial state mean (2x1 vector). Starting values for [eta_1, mu_1]. | Data-driven (EM). Set to x_smooth[1] after first EM iteration. | Very low. EM rapidly overrides any initialization (Paper, Section 2.3.3). | Unrestricted. |
| Sigma_1 | Initial state covariance (2x2 matrix). Uncertainty about the initial state. | Data-driven (EM). Set to P[1] - x_smooth[1]*x_smooth[1]^T. | Very low. Overridden within a few filter steps. | Positive semi-definite. |
| lambda | Lasso regularization parameter (robust model only). Controls the outlier detection threshold. | Selected by cross-validation on a held-out period. Not estimated by EM. | Medium-high (for robust model). Too small: treats noise as outliers, over-cleaning. Too large: misses real outliers. The effective threshold lambda/(2*W_tau) adapts dynamically. | (0, inf). Practical range depends on the scale of log-volume innovations. |
| N | Training window length (number of trading days). How much history is used for EM estimation. | Selected by cross-validation. Paper uses a rolling window scheme re-estimated periodically. | Medium. Too short: insufficient data for reliable EM estimation. Too long: model cannot adapt to structural changes. | Typically 100-500 trading days. Must be large enough for EM convergence (at least several dozen days). |
| I | Number of intraday bins per trading day. Determined by exchange trading hours and bin width. | 26 for NYSE/NASDAQ (6.5 hours / 15 min). Varies by exchange. | Configuration parameter, not tunable. Affects state-space dimensions indirectly (more bins = more updates per day). | Determined by trading hours / bin width. |

### Initialization

**EM initialization (theta_init):**
The EM algorithm is robust to initial parameter values (Paper, Section 2.3.3, Figure 4).
A reasonable starting point:

1. a^eta_init = 0.99 (high persistence for daily component)
2. a^mu_init = 0.5 (moderate persistence for intraday dynamics)
3. (sigma^eta)^2_init = var(daily mean log-volume) across training days
4. (sigma^mu)^2_init = var(detrended log-volume) / I
5. r_init = 0.1 * var(log-volume)
6. phi_init[i] = mean of y_{t,i} across all training days t, for each bin i
   (Researcher inference: the paper does not specify the initialization of phi for EM
   iteration 0; using the raw per-bin mean is a natural choice since the EM update
   formula converges to the residual mean, and starting from the raw mean places phi
   close to its converged value.)
7. pi_1_init = [mean(daily log-volume), 0]^T
8. Sigma_1_init = diag(1, 1) (or any positive definite matrix)

The paper demonstrates in Figure 4 that the EM converges to the same parameters
regardless of initial values, so the exact initialization is not critical. However,
starting closer to reasonable values reduces the number of iterations needed.

(Paper, Section 2.3.3)

**Kalman filter initialization (for out-of-sample prediction):**
Use the EM-estimated pi_1 and Sigma_1 from the training phase. Alternatively, for
a rolling-window scheme, initialize the filter state from the end-of-window smoothed
state of the previous training run.

(Researcher inference: the paper does not explicitly discuss warm-starting the filter
across rolling windows, but carrying forward the last smoothed state is the standard
practice in state-space models with rolling re-estimation.)

### Calibration

**Step-by-step calibration procedure:**

1. **Data preparation:**
   - Obtain intraday volume data at 15-minute bin resolution.
   - Normalize each bin's volume by daily shares outstanding.
   - Take natural log. Exclude or impute any zero-volume bins.
   - Arrange as a flat time series y[1..N] with N = D * I.

2. **Train/validation/test split:**
   - Training set: first portion of data (e.g., Jan 2013 to Dec 2014).
   - Validation set: held-out period for cross-validating N and lambda
     (e.g., Jan 2015 to May 2015, as in Paper Section 4.1).
   - Test set: final period for out-of-sample evaluation
     (e.g., Jun 2015 to Jun 2016, D=250 days).

3. **Cross-validate N and lambda:**
   - For each candidate N in a grid (e.g., 50, 100, 150, ..., 500 days):
     - For each candidate lambda in a grid (e.g., 0.01, 0.05, 0.1, 0.5, 1.0, ...):
       - Run EM on the training set of length N.
       - Evaluate MAPE on the validation set using the robust Kalman filter.
     - For the standard Kalman filter, only cross-validate N (no lambda).
   - Select the (N, lambda) pair with lowest validation MAPE.

4. **Rolling-window estimation:**
   - For each test day d:
     - Use the most recent N days as training data.
     - Run EM to estimate theta.
     - Run Kalman filter on day d to produce forecasts.
   - Re-estimation frequency: the paper uses a standard rolling window (implying
     daily re-estimation). For computational efficiency, re-estimation can be done
     less frequently (e.g., weekly) if parameters are stable.
     (Researcher inference: the paper does not specify re-estimation frequency
     explicitly, but the rolling-window scheme implies updating when new data enters
     the window.)

5. **EM convergence:**
   - Run EM iterations until the relative change in observed-data log-likelihood is
     below tolerance or a maximum iteration count is reached. The convergence criterion
     is: |log L^{(j+1)} - log L^{(j)}| / |log L^{(j)}| < tol, using the innovation-form
     log-likelihood:

         log L = -0.5 * sum_{tau=1}^{N} [log(2*pi*S_tau) + e_tau^2 / S_tau]

     where S_tau is the innovation variance and e_tau = y_tau - y_hat_tau is the
     innovation at each step. (Paper, Appendix A.1, Equation A.8)

   - The paper shows convergence within "a few iterations" on synthetic data (Paper,
     Section 2.3.3, Figure 4). On real data, convergence speed may differ. A budget
     of 20-50 iterations with tolerance 1e-6 is a reasonable starting point, but the
     developer should monitor actual convergence on real data rather than relying on a
     fixed iteration budget. (Researcher inference: the 20-50 iteration range is not
     from the paper; it is a practical recommendation. The paper only demonstrates
     fast convergence on synthetic data.)

(Paper, Section 4.1)

## Validation

### Expected Behavior

**Volume prediction accuracy (MAPE):**

MAPE (Mean Absolute Percentage Error) is defined as (Paper, Section 3.3, Equation 37):

    MAPE = (1/M) * sum_{tau=1}^{M} |volume_tau - volume_hat_tau| / volume_tau

where M is the total number of out-of-sample bins, volume_tau is the actual volume,
and volume_hat_tau = exp(y_hat_tau) is the predicted volume obtained by exponentiating
the log-volume forecast. Note that MAPE operates in volume space (not log-volume space),
so the exp() conversion is part of the evaluation pipeline.

**Dynamic mode MAPE (out-of-sample):**
- Robust Kalman Filter: average MAPE 0.46 across 30 securities.
- Standard Kalman Filter: average MAPE 0.47.
- CMEM benchmark: average MAPE 0.65.
- Rolling Mean baseline: average MAPE 1.28.

(Paper, Section 4.2, Table 3, bottom row "Average". Dynamic prediction columns.)

**Static mode MAPE (out-of-sample):**
- Robust Kalman Filter: average MAPE 0.61.
- Standard Kalman Filter: average MAPE 0.62.
- CMEM benchmark: average MAPE 0.90.
- Rolling Mean baseline: average MAPE 1.28.

(Paper, Section 4.2, Table 3, bottom row "Average". Static prediction columns.)

**VWAP tracking error (basis points, out-of-sample):**
- Robust Kalman Filter (dynamic VWAP): average 6.38 bps.
- Standard Kalman Filter (dynamic VWAP): average 6.39 bps.
- CMEM (dynamic VWAP): average 7.01 bps.
- Rolling Mean (dynamic VWAP): average 10.68 bps.

(Paper, Section 4.3, Table 4, bottom row "Average". Dynamic VWAP tracking columns.)

**EM convergence:**
- Parameters converge within a few iterations from any starting values.
- Figure 4 shows convergence from 10 different random initializations on synthetic
  data, all converging to the true parameter values.

(Paper, Section 2.3.3, Figure 4.)

**Relative improvements:**
- 64% MAPE improvement over Rolling Mean (dynamic prediction).
- 29% MAPE improvement over dynamic CMEM.
- 15% VWAP tracking error improvement over Rolling Mean.
- 9% VWAP tracking error improvement over CMEM.

(Paper, Section 4.2, page 8; Section 4.3, page 11.)

### Sanity Checks

1. **EM convergence on synthetic data:** Generate synthetic log-volume data from the
   state-space model with known parameters. Run EM and verify that estimated parameters
   converge to the true values within a few iterations. Compare against Figure 4 of the
   paper.
   (Paper, Section 2.3.3)

2. **Seasonality shape:** After EM estimation, plot phi[1..I]. It should exhibit the
   well-known U-shape (or J-shape for markets with high opening volume): high values
   at the first and last bins, low values in the middle of the day. Compare visually
   against Figure 4f.
   (Paper, Section 2.3.3, Figure 4f)

3. **State component behavior:**
   - eta_t should be approximately constant within each day and change smoothly across
     days (reflecting the daily volume trend).
   - mu_{t,i} should fluctuate around zero within each day, with deviations decaying
     at rate a^mu.
   (Paper, Section 2, model definition)

4. **Robust filter sparsity:** When running the robust Kalman filter on clean data,
   z_star[tau] should be zero for the vast majority of bins (>95%). On artificially
   contaminated data (10% of bins perturbed), approximately 10% of z_star values
   should be nonzero.
   (Paper, Section 3.3)

5. **Standard vs. robust equivalence on clean data:** On clean (curated) data, the
   standard and robust Kalman filters should produce very similar forecasts (MAPE
   difference < 0.02). Table 3 shows the difference is 0.01 on average.
   (Paper, Table 3, comparing robust KF vs. standard KF columns)

6. **Dynamic vs. static ordering:** Dynamic predictions should always outperform static
   predictions (lower MAPE) because they incorporate real-time information. Table 3
   confirms this for all 30 securities.
   (Paper, Table 3)

7. **Rolling Mean baseline:** Implement the rolling mean baseline as: for each bin i,
   predict the average of y_{t',i} over the prior N' days. The resulting MAPE should be
   around 1.28 (matching the paper's RM benchmark), validating the data pipeline.
   (Paper, Table 3, RM column)

8. **Filter covariance positive definiteness:** Sigma[tau] must remain positive definite
   at every step. If it becomes negative definite or singular, there is a numerical bug.
   The 2x2 structure makes this easy to check: both diagonal elements must be positive
   and the determinant must be positive.
   (Researcher inference: standard Kalman filter property, not explicitly stated in paper.)

9. **Log-likelihood monotonicity:** The observed-data log-likelihood (innovation form)
   should increase monotonically across EM iterations. A decrease indicates an
   implementation bug in the E-step or M-step.
   (Researcher inference: standard EM property, not explicitly stated in paper.)

### Edge Cases

1. **Zero-volume bins:** The model cannot handle bins with zero traded volume because
   log(0) is undefined. The paper explicitly excludes zero-volume bins (Paper, Section
   4.1). Options for handling:
   - Skip (treat as missing observation: skip the correction step and only run the
     predict step for that bin).
   - Impute with a small value (e.g., 1 share / shares_outstanding) before taking log.
   The skip-correction approach is more principled for the Kalman filter framework.
   (Researcher inference: the paper does not specify a handling strategy, only excludes
   such bins. The missing-observation Kalman filter approach is standard.)

2. **Half-day trading sessions:** The paper excludes half-day sessions (Paper, Section
   4.1). If included, the number of bins I would change for those days, requiring either
   padding or a variable-length observation model. Recommended: exclude half-day sessions
   to match the paper.

3. **Day boundaries:** The transition matrix A_tau and noise Q_tau change at day
   boundaries. The implementation must correctly detect when tau transitions from the
   last bin of one day to the first bin of the next. Off-by-one errors here would
   cause eta to transition at the wrong time.

4. **Numerical precision of 2x2 matrix inverse:** The smoother gain L_tau requires
   inverting Sigma_pred[tau+1]. For a 2x2 matrix [[a,b],[c,d]], the inverse is
   (1/(ad-bc)) * [[d,-b],[-c,a]]. If ad-bc is very small, the inversion is unstable.
   In practice this should not happen because Sigma_pred includes process noise, but
   the determinant should be checked.

5. **Very long training windows:** If N is very large, the sufficient statistics in
   the M-step are sums over N terms. Numerical accumulation errors could affect
   precision. Use compensated (Kahan) summation for the M-step sums if N > 10,000 bins.

6. **Assets with structural breaks:** The model assumes stationarity within the training
   window. If an asset undergoes a structural break (e.g., a stock split that was not
   properly adjusted in the shares-outstanding normalization, or a change in market
   microstructure), the EM-estimated parameters will be biased. The rolling-window
   approach partially mitigates this by limiting the influence of old data.

7. **Multi-exchange bin count differences:** Different exchanges have different trading
   hours, so I varies (e.g., I=26 for NYSE 6.5h, I=34 for an 8.5h session). The model
   must be calibrated separately per exchange (or per I value). The seasonality vector
   phi has I elements, so it is exchange-specific by construction.

### Known Limitations

1. **Cannot handle zero-volume bins.** The log transformation is undefined at zero.
   This limits applicability to liquid securities or requires an imputation strategy.
   (Paper, Section 4.1)

2. **No exogenous covariates.** The model uses only past volume to predict future volume.
   It does not incorporate price, volatility, spread, order flow, or event calendars.
   The paper identifies adding covariates as future work (Paper, Section 5).

3. **Linear dynamics only.** The AR(1) state transition for both eta and mu assumes
   linear dynamics. Nonlinear volume dynamics (e.g., regime switching, threshold effects
   as in the BDF-SETAR model) are not captured.

4. **No uncertainty in volume-space forecasts.** While the Kalman filter provides
   Gaussian prediction intervals in log-space, converting these to volume-space via
   exponentiation yields log-normal prediction intervals that are asymmetric and may
   not accurately reflect the true volume distribution, especially in the tails.
   Additionally, exp(y_hat) is a biased estimator of expected volume (see log-normal
   bias note in Data Flow section).

5. **Gaussian noise assumption.** Although the log transformation makes the Gaussian
   assumption more defensible (Paper, Figure 1), extreme market events may still produce
   non-Gaussian innovations even in log-space. The robust Kalman filter partially
   mitigates this.

6. **Single-asset model.** Each security is modeled independently. There is no
   cross-sectional information sharing (unlike the BDF model which uses PCA across
   stocks). This means the model cannot exploit correlated volume patterns across
   related securities.

7. **Static seasonality within training window.** The phi vector is estimated as a
   fixed pattern over the training window. If the intraday seasonal pattern shifts
   (e.g., due to regulatory changes in market hours or the growth of closing auctions),
   the rolling window approach adapts only gradually.

## Paper References

| Spec Section | Paper Source |
|---|---|
| Model Description (log decomposition) | Chen et al. (2016), Section 2, Equation 3 |
| State-space formulation | Chen et al. (2016), Section 2, Equations 4-5, page 3 definitions |
| Kalman Filter algorithm | Chen et al. (2016), Section 2.2, Algorithm 1 |
| Kalman Filter output quantities for smoother | Chen et al. (2016), Algorithm 2 input requirements |
| Kalman Smoother algorithm | Chen et al. (2016), Section 2.3.1, Algorithm 2 |
| Smoother cross-covariance recursion | Chen et al. (2016), Appendix A, Equations A.20-A.22 |
| Smoother cross-covariance initialization (K_N) | Chen et al. (2016), Appendix A, Equation A.21 |
| EM algorithm structure | Chen et al. (2016), Section 2.3.2, Algorithm 3 |
| EM M-step closed-form updates | Chen et al. (2016), Appendix A, Equations A.32-A.39 (= Eqs 17-24) |
| EM M-step ordering (phi before r) | Chen et al. (2016), Appendix A, Equation A.38 (phi^{(j+1)} superscript) |
| EM convergence (innovation log-likelihood) | Chen et al. (2016), Appendix A.1, Equation A.8 |
| EM convergence properties | Chen et al. (2016), Section 2.3.3, Figure 4 |
| Robust Kalman filter (Lasso) | Chen et al. (2016), Section 3.1, Equations 25-34 |
| Robust EM: E-step uses robust filter | Chen et al. (2016), Section 3.2, paragraph before Eq 35 |
| Robust EM: modified M-step | Chen et al. (2016), Section 3.2, Equations 35-36 |
| Outlier robustness results | Chen et al. (2016), Section 3.3, Table 1 |
| MAPE definition | Chen et al. (2016), Section 3.3, Equation 37 |
| Empirical data setup | Chen et al. (2016), Section 4.1, Table 2 |
| MAPE results | Chen et al. (2016), Section 4.2, Table 3 |
| VWAP tracking error results | Chen et al. (2016), Section 4.3, Table 4 |
| VWAP execution strategies | Chen et al. (2016), Section 4.3, Equations 39-41 |
| Dynamic VWAP remaining-bin forecasts | Chen et al. (2016), Section 4.3, Eq 41, page 10 |
| Log-likelihood derivation | Chen et al. (2016), Appendix A, Equations A.1-A.8 |
| EM E-step derivation | Chen et al. (2016), Appendix A.2, Equations A.9-A.17 |
| EM M-step derivation | Chen et al. (2016), Appendix A.3, Equations A.23-A.39 |
| Multiplicative decomposition origin | Brownlees, Cipollini, Gallo (2011), Equation 2 |
| CMEM benchmark comparison | Chen et al. (2016), Tables 1, 3, 4 |
| Log-normal bias correction | Researcher inference (standard log-normal property; not discussed in paper) |
| EM stationarity clamping | Researcher inference (paper does not discuss constraint enforcement in EM) |
| EM iteration budget (20-50) | Researcher inference (paper shows fast convergence on synthetic data only) |
| Initialization robustness | Researcher inference (standard practice; paper shows EM is insensitive to init) |
| Rolling window warm-start | Researcher inference (standard state-space practice, not explicitly in paper) |
| Zero-volume handling (skip correction) | Researcher inference (standard missing-data Kalman extension) |
| EM phi initialization | Researcher inference (natural starting point; paper does not specify init for phi) |
