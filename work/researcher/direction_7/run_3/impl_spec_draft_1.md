# Implementation Specification: Kalman Filter State-Space Model for Intraday Volume

## Overview

This model forecasts intraday trading volume by decomposing log-volume into three
additive components -- a daily average level, an intraday seasonal pattern, and an
intraday dynamic residual -- within a linear Gaussian state-space framework. The Kalman
filter provides optimal recursive prediction and correction, while the EM algorithm
estimates all parameters in closed form. A robust variant uses Lasso regularization
to detect and suppress outliers automatically. The model operates on 15-minute volume
bins normalized by shares outstanding and log-transformed. It supports both static
forecasting (all bins predicted before market open) and dynamic forecasting (updated
after each observed bin).

Source: Chen, Feng, and Palomar (2016), "Forecasting Intraday Trading Volume: A Kalman
Filter Approach."

## Algorithm

### Model Description

The model takes as input a time series of intraday volume observations, aggregated into
fixed-width bins (e.g., 15 minutes). Each bin's volume is normalized by daily shares
outstanding (to remove scale effects from splits and corporate actions), then
log-transformed. The log transformation converts the multiplicative three-component
volume structure into an additive one, eliminates positiveness constraints, and makes
Gaussian noise assumptions defensible.

The log-volume observation y_{t,i} for day t, bin i is decomposed as:

    y_{t,i} = eta_t + phi_i + mu_{t,i} + v_{t,i}

where:
- eta_t: daily average component (log scale). Slowly evolving, captures day-to-day
  level shifts. Constant within a day, transitions at day boundaries only.
- phi_i: intraday periodic component. Deterministic seasonal pattern (the U-shape).
  Same for all days; one value per bin position. Estimated as a simple average across
  training days, not a Fourier parameterization.
- mu_{t,i}: intraday dynamic component. Fast-moving, captures bin-to-bin deviations
  from the seasonal pattern within a day.
- v_{t,i}: observation noise, i.i.d. Gaussian with variance r.

The model outputs one-step-ahead or multi-step-ahead forecasts of log-volume, which are
exponentiated to produce volume forecasts in the original scale.

Assumptions:
- Non-zero volume in every bin (log(0) is undefined).
- Gaussian noise is adequate for log-volume (supported by Q-Q plots in Paper, Section 2,
  Figure 1).
- The daily component is piecewise constant within each day.
- State transitions are first-order autoregressive.

(Paper: Section 2, Equations 2-5; Figure 1 for Q-Q evidence)

### Pseudocode

The implementation comprises four major algorithms: data preprocessing, Kalman filter,
Kalman smoother (RTS), and EM parameter estimation, plus the robust Lasso extension.

#### Step 0: Data Preprocessing

```
Input:
  raw_volume[t, i]  -- raw volume for day t, bin i (t=1..T, i=1..I)
  shares_out[t]     -- shares outstanding for day t

Output:
  y[tau]            -- log-volume series indexed by global time tau=1..N where N=T*I

Procedure:
  1. For each day t and bin i:
       turnover[t,i] = raw_volume[t,i] / shares_out[t]
  2. Exclude any bins where turnover[t,i] == 0
     (record which tau indices are excluded for gap handling)
  3. y[tau] = ln(turnover[t,i])
     where tau = (t-1)*I + i maps (day, bin) to global index
  4. Exclude half-day trading sessions entirely
```

(Paper: Section 4.1. Volume is "computed as Equation (1)" which defines
volume_{t,i} = shares_traded / shares_outstanding. Half-day sessions excluded.)

#### Step 1: Kalman Filter (Algorithm 1 from Paper)

This is the core prediction engine. It produces one-step-ahead state estimates
(prediction) and incorporates observations (correction) recursively.

```
Input:
  y[1..N]           -- observed log-volume series
  theta             -- parameter set: {a_eta, a_mu, sigma_eta^2, sigma_mu^2, r,
                       phi[1..I], pi_1, Sigma_1}

Output:
  x_hat[tau|tau]    -- filtered state estimates (2x1 vectors)
  Sigma[tau|tau]    -- filtered state covariances (2x2 matrices)
  x_hat[tau+1|tau]  -- predicted state estimates
  Sigma[tau+1|tau]  -- predicted state covariances
  y_hat[tau+1|tau]  -- predicted log-volume (scalar)

Definitions:
  C = [1, 1]        -- observation vector (1x2)
  A[tau] = [[a_eta_tau, 0],    -- state transition matrix (2x2)
            [0,         a_mu]]
  where a_eta_tau = a_eta   if tau is a day boundary (tau = k*I for some k)
                  = 1       otherwise

  Q[tau] = [[sigma_eta_tau^2, 0],    -- process noise covariance (2x2)
            [0,               sigma_mu^2]]
  where sigma_eta_tau^2 = sigma_eta^2   if tau is a day boundary
                        = 0             otherwise

  phi[tau] = phi_i where i = ((tau-1) mod I) + 1  -- seasonal value for this bin position

Procedure:
  Initialize:
    x_hat[1|0] = pi_1          -- initial state mean (2x1)
    Sigma[1|0] = Sigma_1       -- initial state covariance (2x2)

  For tau = 1, 2, ..., N-1:
    -- PREDICT (lines 2-3 of Algorithm 1)
    x_hat[tau+1|tau] = A[tau] * x_hat[tau|tau]
    Sigma[tau+1|tau] = A[tau] * Sigma[tau|tau] * A[tau]^T + Q[tau]

    -- Predicted observation
    y_hat[tau+1|tau] = C * x_hat[tau+1|tau] + phi[tau+1]

    -- CORRECT (lines 4-6 of Algorithm 1)
    -- Compute Kalman gain:
    W = (C * Sigma[tau+1|tau] * C^T + r)^{-1}    -- scalar inverse
    K[tau+1] = Sigma[tau+1|tau] * C^T * W         -- 2x1 vector

    -- Innovation (prediction error):
    e[tau+1] = y[tau+1] - phi[tau+1] - C * x_hat[tau+1|tau]

    -- Correct state estimate:
    x_hat[tau+1|tau+1] = x_hat[tau+1|tau] + K[tau+1] * e[tau+1]

    -- Correct covariance:
    Sigma[tau+1|tau+1] = Sigma[tau+1|tau] - K[tau+1] * C * Sigma[tau+1|tau]
```

Note: Since C = [1, 1] and the state is 2x1, the quantity C * Sigma * C^T is a scalar
(sum of all four elements of Sigma), so W is a scalar division, and K is a 2x1 vector.
No matrix inversion beyond scalar reciprocal is ever needed.

(Paper: Section 2.2, Algorithm 1, Equations 4-5, 7-8)

#### Step 2: Kalman Smoother -- Rauch-Tung-Striebel (Algorithm 2 from Paper)

Used during EM calibration only (not during prediction). Runs backward over the
filtered estimates to produce smoothed state estimates conditioned on ALL observations.

```
Input:
  x_hat[tau|tau], Sigma[tau|tau]       -- from forward Kalman filter
  x_hat[tau+1|tau], Sigma[tau+1|tau]   -- from forward Kalman filter
  A[tau]                               -- transition matrices

Output:
  x_hat[tau|N]    -- smoothed state estimates
  Sigma[tau|N]    -- smoothed state covariances
  Sigma[tau,tau-1|N] -- smoothed cross-covariances (for EM sufficient statistics)

Procedure:
  Initialize:
    x_hat[N|N] = x_hat[N|N]       -- last filtered estimate
    Sigma[N|N] = Sigma[N|N]       -- last filtered covariance

  For tau = N-1, N-2, ..., 1:
    -- Smoother gain:
    L[tau] = Sigma[tau|tau] * A[tau]^T * Sigma[tau+1|tau]^{-1}    -- 2x2

    -- Smoothed state:
    x_hat[tau|N] = x_hat[tau|tau] + L[tau] * (x_hat[tau+1|N] - x_hat[tau+1|tau])

    -- Smoothed covariance:
    Sigma[tau|N] = Sigma[tau|tau] + L[tau] * (Sigma[tau+1|N] - Sigma[tau+1|tau]) * L[tau]^T

  -- Cross-covariance (needed for EM M-step):
  -- Initialize:
  Sigma[N,N-1|N] = (I - K[N]*C) * A[N-1] * Sigma[N-1|N-1]

  For tau = N-1, N-2, ..., 2:
    Sigma[tau,tau-1|N] = Sigma[tau|tau] * L[tau-1]^T
                         + L[tau] * (Sigma[tau+1,tau|N] - A[tau] * Sigma[tau|tau]) * L[tau-1]^T
```

Note: Sigma[tau+1|tau] is 2x2, so its inversion is a direct 2x2 analytic inverse
(ad-bc determinant formula). Since the matrix is symmetric positive definite, this is
always well-conditioned.

(Paper: Section 2.3.1, Algorithm 2, Equations 10-11. Cross-covariance initialization
from Appendix A, Equation A.21)

#### Step 3: EM Algorithm (Algorithm 3 from Paper)

Iteratively estimates all model parameters by alternating between computing sufficient
statistics (E-step via filter + smoother) and closed-form parameter updates (M-step).

```
Input:
  y[1..N]           -- observed log-volume training series
  theta^(0)         -- initial parameter guesses
  max_iter          -- maximum EM iterations
  tol               -- convergence tolerance on log-likelihood change

Output:
  theta_hat         -- estimated parameters

Definitions -- Sufficient Statistics:
  For each tau, compute from the smoother output:
    x_hat[tau] = x_hat[tau|N]                              -- smoothed state mean (2x1)
    P[tau] = Sigma[tau|N] + x_hat[tau|N] * x_hat[tau|N]^T -- E[x * x^T | all data] (2x2)
    P[tau,tau-1] = Sigma[tau,tau-1|N] + x_hat[tau|N] * x_hat[tau-1|N]^T
                                                            -- E[x_tau * x_{tau-1}^T | all data]

  Notation for P sub-elements:
    P^{(1,1)}[tau] = P[tau][1,1]    -- corresponds to eta*eta
    P^{(2,2)}[tau] = P[tau][2,2]    -- corresponds to mu*mu
    P^{(1,1)}[tau,tau-1] = P[tau,tau-1][1,1]  -- cross-time eta*eta
    P^{(2,2)}[tau,tau-1] = P[tau,tau-1][2,2]  -- cross-time mu*mu

  Let D = {tau : tau = k*I+1 for k=1,2,...} be the set of day-boundary indices.

Procedure:
  j = 0
  Repeat:
    j = j + 1

    -- E-STEP:
    Run Kalman filter (Algorithm 1) with current theta^(j-1) on y[1..N]
    Run Kalman smoother (Algorithm 2) on filter output
    Compute sufficient statistics x_hat[tau], P[tau], P[tau,tau-1] for all tau

    -- M-STEP (all updates are closed-form):

    -- Initial state mean:
    pi_1^(j) = x_hat[1]

    -- Initial state covariance:
    Sigma_1^(j) = P[1] - x_hat[1] * x_hat[1]^T

    -- Daily AR coefficient (uses only day-boundary transitions):
    a_eta^(j) = [ sum_{tau in D} P^{(1,1)}[tau,tau-1] ]
                / [ sum_{tau in D} P^{(1,1)}[tau-1] ]

    -- Intraday dynamic AR coefficient (uses all transitions):
    a_mu^(j) = [ sum_{tau=2}^{N} P^{(2,2)}[tau,tau-1] ]
               / [ sum_{tau=2}^{N} P^{(2,2)}[tau-1] ]

    -- Daily process noise variance:
    [sigma_eta^2]^(j) = (1/(T-1)) * sum_{tau in D}
                        { P^{(1,1)}[tau] + (a_eta^(j))^2 * P^{(1,1)}[tau-1]
                          - 2 * a_eta^(j) * P^{(1,1)}[tau,tau-1] }

    -- Intraday dynamic process noise variance:
    [sigma_mu^2]^(j) = (1/(N-1)) * sum_{tau=2}^{N}
                       { P^{(2,2)}[tau] + (a_mu^(j))^2 * P^{(2,2)}[tau-1]
                         - 2 * a_mu^(j) * P^{(2,2)}[tau,tau-1] }

    -- Observation noise variance:
    r^(j) = (1/N) * sum_{tau=1}^{N}
            [ y[tau]^2 + C * P[tau] * C^T - 2*y[tau]*C*x_hat[tau]
              + (phi^(j))^2 - 2*y[tau]*phi^(j) + 2*phi^(j)*C*x_hat[tau] ]

    -- Seasonality vector (one value per bin position):
    For each bin position i = 1..I:
      phi_i^(j) = (1/T) * sum_{t=1}^{T} (y[t,i] - C * x_hat[t,i])

      where y[t,i] and x_hat[t,i] use the (day,bin) indexing mapped from global tau

    -- Compute log-likelihood for convergence check (Equation A.8 in Paper)

  Until |log_lik^(j) - log_lik^(j-1)| < tol  OR  j >= max_iter

  Return theta_hat = theta^(j)
```

(Paper: Section 2.3.2, Algorithm 3, Equations 17-24. Closed-form M-step derivations
in Appendix A, Equations A.24-A.39)

#### Step 4: Robust Kalman Filter with Lasso (Section 3 of Paper)

Modifies the standard Kalman correction step to handle outliers via sparse noise.

The observation model becomes:

    y[tau] = C * x[tau] + phi[tau] + v[tau] + z[tau]

where z[tau] is a sparse outlier term (zero most of the time, large when an outlier
occurs).

```
Input:
  Same as standard Kalman filter plus lambda (Lasso regularization parameter)

Modified Correction Step (replaces lines 4-6 of Algorithm 1):
  At each tau where an observation is available:

    -- Compute innovation:
    e[tau] = y[tau] - phi[tau] - C * x_hat[tau|tau-1]

    -- Compute scalar precision of innovation:
    W[tau] = (C * Sigma[tau|tau-1] * C^T + r)^{-1}

    -- Solve Lasso subproblem for outlier z*[tau]:
    -- The closed-form soft-thresholding solution is:
    threshold = lambda / (2 * W[tau])

    if e[tau] > threshold:
        z*[tau] = e[tau] - threshold
    elif e[tau] < -threshold:
        z*[tau] = e[tau] + threshold
    else:
        z*[tau] = 0

    -- Modified innovation (outlier removed):
    e_clean[tau] = e[tau] - z*[tau]

    -- Standard Kalman correction with cleaned innovation:
    K[tau] = Sigma[tau|tau-1] * C^T * W[tau]
    x_hat[tau|tau] = x_hat[tau|tau-1] + K[tau] * e_clean[tau]
    Sigma[tau|tau] = Sigma[tau|tau-1] - K[tau] * C * Sigma[tau|tau-1]
```

The threshold lambda/(2*W[tau]) is time-varying because W[tau] depends on the current
predictive variance. When the model is uncertain (large Sigma), the threshold widens,
tolerating larger innovations. When the model is confident, the threshold tightens,
rejecting smaller deviations. This provides automatic adaptive outlier detection.

(Paper: Section 3.1, Equations 25-34. Soft-thresholding derived from Equations 30, 33.)

#### Step 4b: Robust EM Modifications

When calibrating the robust model, the M-step updates for r and phi incorporate
the inferred outlier terms z*[tau]:

```
  r^(j) = (1/N) * sum_{tau=1}^{N}
          [ y[tau]^2 + C*P[tau]*C^T - 2*y[tau]*C*x_hat[tau]
            + (phi^(j))^2 - 2*y[tau]*phi^(j) + 2*phi^(j)*C*x_hat[tau]
            + (z*[tau])^2 - 2*z*[tau]*y[tau]
            + 2*z*[tau]*C*x_hat[tau] + 2*z*[tau]*phi^(j) ]

  phi_i^(j) = (1/T) * sum_{t=1}^{T} (y[t,i] - C*x_hat[t,i] - z*[t,i])
```

All other M-step equations remain unchanged.

(Paper: Section 3.2, Equations 35-36)

#### Step 5: Prediction Modes

**Dynamic prediction (one-step-ahead):**
After calibration, run the Kalman filter on out-of-sample data. At each bin tau,
produce the forecast for tau+1:

    y_hat[tau+1|tau] = C * x_hat[tau+1|tau] + phi[tau+1]

Then observe y[tau+1], perform the correction step, and advance. This gives the
best possible forecast because it uses all information up to the current bin.

**Static prediction (multi-step-ahead):**
At the end of day t (after bin I), forecast all I bins of day t+1 without any
corrections during day t+1. The h-step-ahead forecast is:

    x_hat[tau+h|tau] = A^h * x_hat[tau|tau]
    y_hat[tau+h|tau] = C * x_hat[tau+h|tau] + phi[tau+h]

where A^h means applying the transition matrix h times. Since eta is constant within
a day (a_eta_tau = 1 for intraday steps), the eta component does not change. The mu
component decays as (a_mu)^h toward zero.

For static prediction, the state after the last bin of day t propagates through the
day boundary (applying a_eta once) and then through h-1 intraday steps.

**Converting to volume scale:**
The forecast is in log space. To produce a volume forecast:

    volume_hat[t,i] = exp(y_hat[tau]) * shares_out[t]

Note: exp(E[log(V)]) is the geometric mean, not the arithmetic mean. If unbiased
arithmetic-mean forecasts are needed, apply the log-normal bias correction:

    volume_hat_unbiased = exp(y_hat + 0.5 * prediction_variance)

where prediction_variance = C * Sigma[tau+1|tau] * C^T + r. The paper does not
explicitly discuss this correction but uses MAPE on volume (not log-volume) as the
evaluation metric, so the bias is implicitly present.
Researcher inference: the bias correction is standard for log-normal models but is
not mentioned in the paper. Whether to apply it is a design decision.

(Paper: Section 2.2, Equation 9; Section 4.2)

#### Step 6: VWAP Execution

**Static VWAP:**
Before market open, compute volume forecasts for all I bins. Compute VWAP weights:

    w[i] = volume_hat[i] / sum_{j=1}^{I} volume_hat[j]

Execute w[i] fraction of the order in bin i.

(Paper: Section 4.3, Equation 40)

**Dynamic VWAP:**
After each observed bin i, revise the forecast for remaining bins i+1..I using
one-step-ahead dynamic predictions. Redistribute the remaining order proportionally:

    For bin i (current, i = 1..I-1):
      w[i] = volume_hat_dynamic[i] / sum_{j=i}^{I} volume_hat_dynamic[j]
             * (1 - sum_{k=1}^{i-1} w[k])

    For the last bin (i = I):
      w[I] = 1 - sum_{k=1}^{I-1} w[k]

(Paper: Section 4.3, Equation 41)

### Data Flow

```
Raw intraday volume data (T days x I bins)
    |
    v
[Normalize by shares outstanding] --> turnover[t,i]
    |
    v
[Log transform] --> y[t,i] = ln(turnover[t,i])
    |
    v
[Reshape to global time series] --> y[tau], tau = 1..N, N = T*I
    |
    +------> [EM Calibration loop] ------+
    |           |                         |
    |           v                         v
    |        [Kalman Filter forward]   [M-step: closed-form
    |           |                       parameter updates]
    |           v                         |
    |        [RTS Smoother backward]      |
    |           |                         |
    |           +--- sufficient stats --->+
    |                                     |
    |        [Repeat until convergence]   |
    |                                     v
    |                              theta_hat (estimated parameters)
    |
    +------> [Kalman Filter with theta_hat on out-of-sample data]
                |
                v
          y_hat[tau+1|tau] (log-volume forecast)
                |
                v
          [Exponentiate] --> volume_hat = exp(y_hat) * shares_outstanding
                |
                v
          [Compute VWAP weights] --> w[1..I]
```

**Shapes and types at each stage:**

| Stage | Variable | Shape | Type |
|-------|----------|-------|------|
| Input | raw_volume | (T, I) | float, non-negative |
| Input | shares_out | (T,) | float, positive |
| After log | y | (N,) where N=T*I | float |
| State vector | x[tau] | (2,) | float |
| Transition matrix | A[tau] | (2, 2) | float, time-varying |
| Process noise cov | Q[tau] | (2, 2) | float, diagonal, time-varying |
| Observation vector | C | (2,) or (1, 2) | float, constant = [1, 1] |
| Kalman gain | K[tau] | (2,) or (2, 1) | float |
| Innovation precision | W[tau] | scalar | float, positive |
| Seasonality | phi | (I,) | float |
| Filter output | x_hat[tau\|tau] | (N, 2) | float |
| Filter covariance | Sigma[tau\|tau] | (N, 2, 2) | float, symmetric PD |
| Smoother output | x_hat[tau\|N] | (N, 2) | float |
| Predicted log-vol | y_hat | (N,) | float |
| Output volume | volume_hat | (T, I) | float, positive |
| VWAP weights | w | (I,) | float in [0,1], sums to 1 |

### Variants

**Implemented variant: Robust Kalman Filter with Lasso (the full model).**

The paper presents two variants:
1. Standard Kalman Filter (Section 2) -- no outlier handling.
2. Robust Kalman Filter with Lasso (Section 3) -- adds sparse noise term.

The robust variant should be the primary implementation because:
- It subsumes the standard variant (setting lambda = infinity recovers the standard
  filter, since the threshold becomes infinite and z* is always zero).
- It performs equal to or better than the standard variant on clean data (Table 1:
  identical or 0.01 better MAPE on clean data).
- It degrades gracefully under contamination while the standard variant and CMEM
  degrade rapidly or fail entirely (Table 1: at medium outlier levels, CMEM produces
  N/A indicating solver failure).
- The computational overhead is negligible (one soft-thresholding operation per bin).

The standard variant can be obtained by setting lambda to a very large value or by
simply skipping the outlier detection step, which is useful for debugging and
validation against the standard Kalman filter results.

(Paper: Section 3, Tables 1 and 3)

## Parameters

| Parameter | Description | Recommended Value | Sensitivity | Range |
|-----------|-------------|-------------------|-------------|-------|
| a_eta | Daily AR coefficient for eta (daily average level) | Data-driven via EM. Synthetic experiment converges near 0.98-0.99 (Figure 4a). | Medium -- controls persistence of daily level. | (0, 1) strictly. Values near 1 indicate high persistence. |
| a_mu | Intraday AR coefficient for mu (dynamic component) | Data-driven via EM. Synthetic experiment converges near 0.5-0.7 (Figure 4b). | Medium -- controls decay of intraday dynamics. Lower values mean faster mean-reversion. | (0, 1) strictly. |
| sigma_eta^2 | Process noise variance for daily component | Data-driven via EM. | Low -- well-identified by EM. | (0, inf). Typically small (daily level changes slowly). |
| sigma_mu^2 | Process noise variance for intraday dynamic | Data-driven via EM. | Low -- well-identified by EM. | (0, inf). Typically larger than sigma_eta^2. |
| r | Observation noise variance | Data-driven via EM. Synthetic converges near 0.05 (Figure 4c). | Low -- well-identified. | (0, inf). |
| phi[1..I] | Intraday seasonality vector | Data-driven via EM. Captures U-shaped pattern. Values at open and close bins are largest. | Low per element -- many degrees of freedom but each is a simple mean. | Unrestricted (log scale). |
| pi_1 | Initial state mean (2x1) | Set to [mean(y), 0] or estimate via EM. | Very low -- EM converges regardless of initialization (Figure 4). | Unrestricted. |
| Sigma_1 | Initial state covariance (2x2) | Set to diagonal with moderate values, e.g., diag(1, 1) or diag(var(y), var(y)). | Very low -- EM converges regardless. | Symmetric positive definite. |
| lambda | Lasso regularization for robust variant | Selected by cross-validation on a held-out validation set. | High -- determines outlier sensitivity. Too small: real signal treated as outlier. Too large: outliers not detected. | (0, inf). Paper does not report typical values. |
| N | Training window length (number of bins = T_train * I) | Selected by cross-validation. Paper uses January 2013 to variable endpoint as training, with Jan-May 2015 as validation. | High -- too short loses information, too long introduces non-stationarity. | Minimum ~100 days * I bins. Upper bound limited by stationarity. |
| I | Number of intraday bins per day | 26 for NYSE (6.5h / 15min). Exchange-dependent. | Configuration parameter, not tuned. | Determined by exchange hours and bin width. |

### Initialization

**EM initialization (theta^(0)):**
The paper demonstrates (Section 2.3.3, Figure 4) that EM convergence is robust to
initial parameter choice. Recommended initialization:

1. a_eta^(0) = 0.9 (any value in (0.5, 1.0) works)
2. a_mu^(0) = 0.5 (any value in (0.1, 0.9) works)
3. sigma_eta^2^(0) = var(daily_means) where daily_means are the average log-volume per day
4. sigma_mu^2^(0) = var(y) * 0.1 (rough guess)
5. r^(0) = var(y) * 0.5
6. phi^(0) = for each bin i, compute the average of y[t,i] across all days t, then
   subtract the grand mean. This is a reasonable initial seasonal estimate.
7. pi_1^(0) = [mean(y[1..I]), 0]^T (first day's average, zero dynamic component)
8. Sigma_1^(0) = diag(var(y), var(y))

Convergence typically occurs within 5-15 EM iterations (Figure 4 shows convergence
within ~10 iterations from diverse starting points).

(Paper: Section 2.3.3, Figure 4)

**Kalman filter state initialization for out-of-sample:**
Use the EM-estimated pi_1 and Sigma_1, or warm-start from the last filtered state
of the training period.

### Calibration

**Rolling window calibration procedure:**

```
1. Choose validation period (e.g., 100 trading days before out-of-sample start).
2. Define candidate grid:
   - N_candidates: training windows of [100, 150, 200, 300, 500] days
                   (equivalently, multiply by I for number of bins)
   - lambda_candidates: [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
     (Researcher inference: the paper does not specify the grid; these are
     reasonable values spanning orders of magnitude.)
3. For each (N_candidate, lambda_candidate) pair:
   a. Train the model on the N_candidate bins immediately before the validation period.
   b. Run EM to convergence (max_iter=100, tol=1e-6).
   c. Produce dynamic predictions on the validation period.
   d. Compute MAPE on validation period.
4. Select (N*, lambda*) = argmin MAPE over the grid.
5. For out-of-sample evaluation:
   Use a rolling window of N* bins, re-running EM at each window shift.
```

The paper uses data between January 2015 and May 2015 as the cross-validation set
(Section 4.1). The standard rolling window scheme re-estimates parameters as the
window slides, but the paper does not specify the re-estimation frequency. A
practical choice is daily re-estimation (shift the window by I bins each day).

Researcher inference: the re-estimation frequency is not specified in the paper.
Daily re-estimation is the natural choice given the rolling window scheme and is
computationally feasible since EM converges in few iterations and all operations
are O(N) with small constants (2x2 matrices).

(Paper: Section 4.1, paragraph on cross-validation)

## Validation

### Expected Behavior

**Dynamic prediction MAPE (one-step-ahead):**

| Model | Average MAPE across 30 securities |
|-------|-----------------------------------|
| Robust Kalman Filter | 0.46 |
| Standard Kalman Filter | 0.47 |
| CMEM (dynamic) | 0.65 |
| Rolling Mean (RM) | 1.28 |

(Paper: Section 4.2, Table 3 averages. "the robust Kalman filter with dynamic
prediction performs the best in the empirical studies. It gives an average MAPE
of 0.46")

**Static prediction MAPE (all bins forecast before market open):**

| Model | Average MAPE |
|-------|-------------|
| Robust Kalman Filter | 0.61 |
| Standard Kalman Filter | 0.62 |
| CMEM (static) | 0.90 |
| Rolling Mean | 1.28 |

(Paper: Section 4.2, Table 3 averages)

**VWAP tracking error (basis points, dynamic strategy):**

| Model | Average tracking error (bps) |
|-------|------------------------------|
| Robust Kalman Filter | 6.38 |
| Standard Kalman Filter | 6.39 |
| CMEM | 7.01 |
| Rolling Mean | 7.48 |

This represents a 15% improvement over RM and 9% over CMEM.

(Paper: Section 4.3, Table 4 averages, Equation 42 for tracking error definition)

**Per-ticker MAPE ranges (dynamic prediction):**
- Best: AAPL at 0.21 (robust) to 0.21 (standard)
- Worst: 2800HK at 1.94 (robust), a leveraged ETF with extreme variability
- Most U.S. large-cap stocks: 0.21-0.42

(Paper: Table 3)

**EM convergence behavior:**
- Parameters converge within approximately 10 iterations from diverse initializations.
- Convergence is monotonic in log-likelihood (guaranteed by EM theory).
- Final parameter values are insensitive to initialization (Figure 4).

(Paper: Section 2.3.3, Figure 4)

### Sanity Checks

1. **Seasonality phi should show U-shape:** After EM converges, plot phi[1..I]. It
   should show high values at the first and last bins (market open and close) and
   lower values in the middle of the day. This is the well-documented intraday
   volume pattern. If phi is flat or inverted, something is wrong with the
   data or the EM.

2. **AR coefficients in (0,1):** a_eta should be close to 1 (0.95-0.99), reflecting
   high persistence of the daily level. a_mu should be moderate (0.3-0.7), reflecting
   faster mean-reversion of intraday dynamics. If either is negative or greater than 1,
   the model is mis-specified or the data has issues.
   (Paper: Figure 4a shows a_eta converging near 0.98; Figure 4b shows a_mu near 0.5-0.7)

3. **Robust vs standard equivalence on clean data:** On curated (outlier-free) data,
   the robust and standard Kalman filters should produce nearly identical predictions
   (MAPE difference < 0.02). If they diverge significantly on clean data, lambda is
   too small.
   (Paper: Table 1, "No outliers" rows show identical or near-identical MAPE)

4. **Rolling mean baseline:** Implement a simple rolling mean baseline (average of
   the same bin over the past 20-60 days). The Kalman filter should beat this by
   a large margin (60%+ MAPE reduction in dynamic mode). If it does not, there is
   likely a bug in the implementation.
   (Paper: Table 3, RM column)

5. **Innovation sequence whiteness:** The innovation sequence e[tau] = y[tau] -
   y_hat[tau|tau-1] should be approximately white noise if the model is correctly
   specified. Compute the autocorrelation function of the innovations; significant
   autocorrelation at lag 1 indicates model mis-specification (the AR orders may
   need to be higher, but the paper uses AR(1) throughout).
   Researcher inference: this is a standard Kalman filter diagnostic not explicitly
   discussed in the paper.

6. **Log-likelihood monotonically increases:** Track the log-likelihood at each EM
   iteration. It must be non-decreasing. A decrease indicates a bug in the E-step
   or M-step implementation.
   (Paper: Section 2.3.2, standard EM property)

7. **Synthetic data recovery:** Generate synthetic data from known parameters
   (a_eta=0.98, a_mu=0.5, sigma_eta^2=0.01, sigma_mu^2=0.05, r=0.05, I=26,
   T=500). Run EM and verify that recovered parameters match the true values within
   a few percent. This is the validation approach used in Figure 4.
   (Paper: Section 2.3.3)

### Edge Cases

1. **Zero-volume bins:** The model requires log(volume) > -inf. Bins with exactly
   zero volume must be handled before log transformation. Options:
   - Exclude zero bins and skip the Kalman correction step at those times (treat
     as missing observations). The prediction step still runs; only the correction
     is skipped.
   - Replace zeros with a small positive value (e.g., 1 share). This introduces
     bias but is simple.
   The paper explicitly excludes zero-volume bins: "bins with zero volume are
   excluded" (Section 4.1). For liquid stocks this is rarely an issue; for illiquid
   stocks this is a significant limitation.
   (Paper: Section 4.1, Section 5)

2. **Day boundaries:** The transition matrix A[tau] changes at day boundaries. The
   implementation must correctly detect when tau transitions from the last bin of
   day t to the first bin of day t+1. At this point:
   - a_eta_tau switches from 1 to a_eta
   - sigma_eta_tau^2 switches from 0 to sigma_eta^2
   Incorrect boundary handling will corrupt the eta state estimate.
   (Paper: Section 2, definition of A_tau and Q_tau)

3. **Half-day trading sessions:** Days with fewer than I bins (e.g., day before
   holidays in U.S. markets) should be excluded entirely, as the seasonal vector
   phi[1..I] assumes a fixed number of bins per day.
   (Paper: Section 4.1, "excluding half-day sessions")

4. **Covariance matrix positive definiteness:** The Kalman filter covariance update
   Sigma[tau+1|tau+1] = Sigma[tau+1|tau] - K*C*Sigma[tau+1|tau] can lose positive
   definiteness due to floating-point errors in long sequences. Use the
   Joseph form for numerical stability:
   Sigma[tau+1|tau+1] = (I - K*C) * Sigma[tau+1|tau] * (I - K*C)^T + K*r*K^T
   Researcher inference: this is a standard Kalman filter implementation practice
   not discussed in the paper. Given the 2x2 state, this is cheap and strongly
   recommended.

5. **Very large or very small volumes:** After normalization by shares outstanding,
   some securities (especially leveraged ETFs) can have extreme turnover values.
   The log transform helps, but the Gaussian assumption may be strained. The
   robust variant mitigates this. Table 3 shows 2800HK (a leveraged ETF) has
   MAPE of 1.94 versus 0.21-0.42 for normal stocks.
   (Paper: Table 2 shows the range of mean turnover; Table 3 shows MAPE by ticker)

6. **Numerical precision in smoother:** The RTS smoother requires inverting
   Sigma[tau+1|tau] (2x2). If this matrix is near-singular (possible if one
   component has very low process noise), use the analytic 2x2 inverse with an
   explicit check on the determinant. Add epsilon (e.g., 1e-12) to the diagonal
   if needed.
   Researcher inference: standard numerical practice.

7. **Exchange-varying I:** Different exchanges have different trading hours, hence
   different I. The model must be calibrated separately per exchange (or per
   value of I). The seasonality vector phi has I elements, so mixing securities
   with different I in a single model is not possible.
   (Paper: Table 2 lists securities across 9 exchanges)

### Known Limitations

1. **Cannot handle zero-volume bins.** The log transform is undefined at zero.
   This makes the model unsuitable for illiquid securities with frequent
   zero-volume intervals. (Paper: Section 5, explicitly noted)

2. **No exogenous covariates.** The model uses only past volume to predict future
   volume. It does not incorporate price, volatility, spread, order flow, or
   event calendars (earnings, expirations). The paper identifies this as future
   work. (Paper: Section 5)

3. **AR(1) dynamics only.** Both eta and mu follow AR(1) processes. Higher-order
   dynamics are not captured. If volume exhibits longer memory, this will show
   up as autocorrelation in the innovations. (Paper: Equations 4-5, implicit in
   the state-space formulation)

4. **Single-security model.** No cross-sectional information is used. Each
   security is modeled independently. Market-wide volume shocks must be captured
   indirectly through the daily component's AR dynamics. (Paper: throughout;
   contrast with Bialkowski et al. 2008 which uses PCA across stocks)

5. **Gaussian noise assumption.** While the log transform improves normality
   (Figure 1), the tails of log-volume may still be heavier than Gaussian.
   The robust variant partially addresses this but does not model heavy tails
   explicitly. (Paper: Section 2, Figure 1)

6. **Static prediction degrades for later bins.** In static mode, the mu forecast
   decays as (a_mu)^h. For large h (e.g., the last bin of the day, h=I=26), the
   dynamic component contribution is essentially zero, and the forecast reduces
   to eta + phi[i]. Dynamic prediction is strongly preferred when feasible.
   (Paper: Section 4.2, static MAPE is 0.61 vs dynamic 0.46)

7. **No distributional output for position-sizing.** The model produces point
   forecasts. While the Kalman filter naturally produces prediction intervals
   via Sigma[tau+1|tau], the paper does not validate these intervals. The
   prediction variance could be used for confidence bands, but their calibration
   is unverified. (Researcher inference)

## Paper References

| Spec Section | Paper Source |
|-------------|-------------|
| Model decomposition (y = eta + phi + mu + v) | Chen et al. 2016, Section 2, Equation 3 |
| State-space formulation | Chen et al. 2016, Section 2, Equations 4-5 |
| Time-varying A_tau and Q_tau | Chen et al. 2016, Section 2, bullet points after Equation 5 |
| Kalman filter algorithm | Chen et al. 2016, Section 2.2, Algorithm 1 |
| Kalman smoother (RTS) | Chen et al. 2016, Section 2.3.1, Algorithm 2 |
| EM algorithm | Chen et al. 2016, Section 2.3.2, Algorithm 3 |
| M-step closed-form updates (pi_1, Sigma_1) | Chen et al. 2016, Appendix A.3, Equations A.32-A.33 |
| M-step update for a_eta | Chen et al. 2016, Appendix A.3, Equation A.34 |
| M-step update for a_mu | Chen et al. 2016, Appendix A.3, Equation A.35 |
| M-step update for sigma_eta^2 | Chen et al. 2016, Appendix A.3, Equation A.36 |
| M-step update for sigma_mu^2 | Chen et al. 2016, Appendix A.3, Equation A.37 |
| M-step update for r | Chen et al. 2016, Appendix A.3, Equation A.38 |
| M-step update for phi | Chen et al. 2016, Appendix A.3, Equation A.39 |
| Robust Lasso extension | Chen et al. 2016, Section 3.1, Equations 25-34 |
| Robust EM modifications | Chen et al. 2016, Section 3.2, Equations 35-36 |
| Soft-thresholding solution | Chen et al. 2016, Section 3.1, Equations 33-34 |
| EM convergence insensitivity | Chen et al. 2016, Section 2.3.3, Figure 4 |
| MAPE results | Chen et al. 2016, Section 4.2, Table 3 |
| VWAP tracking error results | Chen et al. 2016, Section 4.3, Table 4 |
| Robustness to outliers | Chen et al. 2016, Section 3.3, Table 1 |
| Data description and exchanges | Chen et al. 2016, Section 4.1, Table 2 |
| Static VWAP weights | Chen et al. 2016, Section 4.3, Equation 40 |
| Dynamic VWAP weights | Chen et al. 2016, Section 4.3, Equation 41 |
| Log-volume normality evidence | Chen et al. 2016, Section 2, Figure 1 |
| Smoother cross-covariance init | Chen et al. 2016, Appendix A, Equation A.21 |
| Log-likelihood function | Chen et al. 2016, Appendix A.1, Equation A.8 |
| Joseph form for covariance | Researcher inference (standard Kalman practice) |
| Log-normal bias correction | Researcher inference (standard statistical result) |
| Innovation whiteness check | Researcher inference (standard Kalman diagnostic) |
| Cross-validation grid values | Researcher inference (reasonable defaults) |
| Re-estimation frequency | Researcher inference (daily, not specified in paper) |
