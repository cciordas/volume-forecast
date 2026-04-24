# Implementation Specification: Kalman Filter State-Space Model for Intraday Volume

## Overview

This model forecasts intraday trading volume by decomposing log-volume into three
additive components -- a daily average, an intraday periodic (seasonal) pattern, and an
intraday dynamic component -- within a linear Gaussian state-space framework. The Kalman
filter provides recursive one-step-ahead and multi-step-ahead predictions, and the EM
algorithm estimates all model parameters in closed form. A robust extension adds
Lasso-penalized sparse noise detection for automatic outlier handling. The log
transformation is the key design choice: it converts the multiplicative volume
decomposition into a tractable linear additive model, eliminates positiveness constraints,
reduces right-skewness, and makes Gaussian noise assumptions defensible.

The model is from Chen, Feng, and Palomar (2016), "Forecasting Intraday Trading Volume:
A Kalman Filter Approach."

## Algorithm

### Model Description

The model operates on log-transformed intraday volume at a fixed bin granularity (e.g.,
15-minute bins). Each trading day t has I bins indexed i = 1, ..., I. The number of bins
per day depends on the exchange's trading hours.

Raw volume in each bin is first normalized by daily shares outstanding (to correct for
splits and scale differences), then the natural logarithm is taken. The log-volume
observation for day t, bin i is decomposed as:

    y_{t,i} = eta_t + phi_i + mu_{t,i} + v_{t,i}

where:
- eta_t: log daily average component (slowly evolving, changes only at day boundaries)
- phi_i: log intraday periodic component (the U-shaped seasonal pattern, deterministic
  per bin index, same for all days)
- mu_{t,i}: log intraday dynamic component (fast-moving, captures deviations from the
  seasonal pattern)
- v_{t,i}: observation noise, i.i.d. N(0, r)

The model takes as input a time series of normalized log-volume observations and produces
as output one-step-ahead or multi-step-ahead log-volume forecasts. These are converted
back to linear volume via exponentiation for VWAP execution.

Reference: Chen et al. (2016), Section 2, Equations (1)-(3).

### Pseudocode

The model has three main algorithmic components: the Kalman filter (prediction), the
Kalman smoother (used during calibration), and the EM algorithm (parameter estimation).
A robust extension modifies the filter's correction step.

#### Notation and Indexing

The paper uses a single time index tau = 1, 2, ..., N where N = T * I (T training days,
I bins per day). The mapping between (t, i) and tau is:

    tau = (t - 1) * I + i

Day boundaries occur at tau = k * I for k = 1, 2, ... The state vector is:

    x_tau = [eta_tau, mu_tau]^T    (2x1 vector)

Reference: Chen et al. (2016), Section 2, page 3.

#### State-Space Formulation

State transition:
    x_{tau+1} = A_tau * x_tau + w_tau

Observation:
    y_tau = C * x_tau + phi_tau + v_tau

where:
- C = [1, 1] is the observation vector (1x2)
- A_tau is the 2x2 state transition matrix (time-varying):
      A_tau = [[a_tau^eta, 0], [0, a^mu]]
  where a_tau^eta = a^eta if tau = k*I (day boundary), else a_tau^eta = 1
- w_tau ~ N(0, Q_tau) is process noise with diagonal covariance:
      Q_tau = [[(sigma_tau^eta)^2, 0], [0, (sigma^mu)^2]]
  where (sigma_tau^eta)^2 = (sigma^eta)^2 if tau = k*I (day boundary), else 0
- v_tau ~ N(0, r) is scalar observation noise
- phi_tau = phi_i where i = ((tau - 1) mod I) + 1

The key structural insight: within a day, eta is held constant (a_tau^eta = 1, noise = 0),
so only mu evolves. At day boundaries, eta transitions with AR coefficient a^eta and
receives process noise.

Reference: Chen et al. (2016), Section 2, Equations (4)-(5), page 3.

#### Algorithm 1: Kalman Filter (Prediction)

```
Input: parameters theta = (a^eta, a^mu, (sigma^eta)^2, (sigma^mu)^2, r, phi, pi_1, Sigma_1)
       observations y_1, ..., y_N

Initialize:
    x_hat_{1|0} = pi_1          # initial state mean (2x1)
    Sigma_{1|0} = Sigma_1       # initial state covariance (2x2)

For tau = 1, 2, ..., N:
    # --- Prediction step ---
    x_hat_{tau+1|tau} = A_tau * x_hat_{tau|tau}
    Sigma_{tau+1|tau} = A_tau * Sigma_{tau|tau} * A_tau^T + Q_tau

    # --- Correction step (skip for static prediction) ---
    # Innovation (scalar):
    innovation = y_tau - C * x_hat_{tau|tau-1} - phi_tau

    # Innovation variance (scalar):
    S_tau = C * Sigma_{tau|tau-1} * C^T + r     # this is scalar

    # Kalman gain (2x1):
    K_tau = Sigma_{tau|tau-1} * C^T / S_tau     # 2x1 vector divided by scalar

    # Corrected state estimate:
    x_hat_{tau|tau} = x_hat_{tau|tau-1} + K_tau * innovation

    # Corrected covariance:
    Sigma_{tau|tau} = Sigma_{tau|tau-1} - K_tau * S_tau * K_tau^T
    # equivalently: (I - K_tau * C) * Sigma_{tau|tau-1}

    # --- Forecast output ---
    # One-step-ahead log-volume forecast:
    y_hat_{tau+1|tau} = C * x_hat_{tau+1|tau} + phi_{tau+1}
```

For dynamic prediction: run both prediction and correction at each bin.
For static prediction: run correction steps only on training data, then run prediction
steps only (skip correction) for all I bins of the forecast day.

For h-step-ahead forecast:
    x_hat_{tau+h|tau} is obtained by iterating the prediction step h times
    y_hat_{tau+h|tau} = C * x_hat_{tau+h|tau} + phi_{tau+h}

Reference: Chen et al. (2016), Section 2.2, Algorithm 1, page 4.

#### Algorithm 2: Kalman Smoother (Rauch-Tung-Striebel)

Used during EM calibration to obtain smoothed state estimates over the full training set.

```
Input: filtered estimates x_hat_{tau|tau}, Sigma_{tau|tau} for tau = 1..N
       predicted estimates x_hat_{tau+1|tau}, Sigma_{tau+1|tau} for tau = 1..N-1
       transition matrices A_tau for tau = 1..N-1

# Initialize at last time step:
x_hat_{N|N} and Sigma_{N|N} are already available from the filter

For tau = N-1, N-2, ..., 1:
    # Smoother gain:
    L_tau = Sigma_{tau|tau} * A_tau^T * inv(Sigma_{tau+1|tau})

    # Smoothed state:
    x_hat_{tau|N} = x_hat_{tau|tau} + L_tau * (x_hat_{tau+1|N} - x_hat_{tau+1|tau})

    # Smoothed covariance:
    Sigma_{tau|N} = Sigma_{tau|tau} + L_tau * (Sigma_{tau+1|N} - Sigma_{tau+1|tau}) * L_tau^T
```

Also compute the cross-covariance needed by EM:

```
# Initialize:
Sigma_{N,N-1|N} = (I - K_N * C) * A_{N-1} * Sigma_{N-1|N-1}

For tau = N-1, N-2, ..., 2:
    Sigma_{tau,tau-1|N} = Sigma_{tau|tau} * L_{tau-1}^T
                         + L_tau * (Sigma_{tau+1,tau|N} - A_tau * Sigma_{tau|tau}) * L_{tau-1}^T
```

Reference: Chen et al. (2016), Section 2.3.1, Algorithm 2, page 5.

#### Algorithm 3: EM Parameter Estimation

```
Input: training observations y_1, ..., y_N
       initial parameter guess theta^(0)

Set j = 0

Repeat until convergence:
    # --- E-step ---
    Run Kalman filter (Algorithm 1) with theta^(j) to get:
        x_hat_{tau|tau}, Sigma_{tau|tau} for tau = 1..N
        x_hat_{tau+1|tau}, Sigma_{tau+1|tau} for tau = 1..N

    Run Kalman smoother (Algorithm 2) to get:
        x_hat_{tau|N} for tau = 1..N        (smoothed state means)
        P_tau for tau = 1..N                (smoothed second moments)
        P_{tau,tau-1} for tau = 2..N        (smoothed cross-moments)

    where the sufficient statistics are:
        x_hat_tau = x_hat_{tau|N}                                    # smoothed mean
        P_tau = Sigma_{tau|N} + x_hat_{tau|N} * x_hat_{tau|N}^T     # smoothed second moment
        P_{tau,tau-1} = Sigma_{tau,tau-1|N} + x_hat_{tau|N} * x_hat_{tau-1|N}^T

    # --- M-step (all closed-form) ---

    # Initial state mean:
    pi_1^(j+1) = x_hat_1

    # Initial state covariance:
    Sigma_1^(j+1) = P_1 - x_hat_1 * x_hat_1^T

    # AR coefficient for daily component (using day-boundary indices only):
    # Let D = {tau : tau = kI+1, k = 1,2,...} be the set of day-boundary transitions
    a^eta^(j+1) = [sum_{tau in D} P_{tau,tau-1}^(1,1)] / [sum_{tau in D} P_{tau-1}^(1,1)]

    # AR coefficient for intraday dynamic component (all time steps):
    a^mu^(j+1) = [sum_{tau=2}^{N} P_{tau,tau-1}^(2,2)] / [sum_{tau=2}^{N} P_{tau-1}^(2,2)]

    # Process noise variance for daily component:
    (sigma^eta)^2^(j+1) = (1/(T-1)) * sum_{tau in D} [P_tau^(1,1) + (a^eta^(j+1))^2 * P_{tau-1}^(1,1)
                                                        - 2 * a^eta^(j+1) * P_{tau,tau-1}^(1,1)]

    # Process noise variance for dynamic component:
    (sigma^mu)^2^(j+1) = (1/(N-1)) * sum_{tau=2}^{N} [P_tau^(2,2) + (a^mu^(j+1))^2 * P_{tau-1}^(2,2)
                                                        - 2 * a^mu^(j+1) * P_{tau,tau-1}^(2,2)]

    # Observation noise variance:
    r^(j+1) = (1/N) * sum_{tau=1}^{N} [y_tau^2 + C * P_tau * C^T - 2 * y_tau * C * x_hat_tau
                                        + (phi_tau^(j+1))^2 - 2 * y_tau * phi_tau^(j+1)
                                        + 2 * phi_tau^(j+1) * C * x_hat_tau]

    # Seasonality vector (per-bin average residual):
    phi_i^(j+1) = (1/T) * sum_{t=1}^{T} (y_{t,i} - C * x_hat_{t,i})
    for each bin i = 1, ..., I

    j = j + 1

Until |theta^(j) - theta^(j-1)| < epsilon or j > max_iterations
```

Note on M-step notation: P_tau^(1,1) means the (1,1) element of the 2x2 matrix P_tau,
P_tau^(2,2) means the (2,2) element, etc. P_{tau,tau-1}^(1,1) is the (1,1) element of
the cross-moment matrix.

Note on phi update ordering: The phi update (Equation A.39) uses the smoothed state
estimates x_hat_{t,i} from the current E-step. The r update (Equation A.38) uses the
newly computed phi^(j+1). This ordering matters -- phi must be updated before r within
each M-step.

Reference: Chen et al. (2016), Section 2.3.2, Algorithm 3, page 5-6; Appendix A.3,
Equations (A.32)-(A.39), pages 14-15.

#### Algorithm 4: Robust Kalman Filter (Lasso Extension)

The observation equation is augmented with a sparse noise term z_tau:

    y_tau = C * x_tau + phi_tau + v_tau + z_tau

where z_tau is nonzero only at outlier bins. The correction step in Algorithm 1 is
replaced by solving a Lasso-penalized quadratic minimization:

```
# Replace the standard correction step with:

# Define the innovation residual:
e_tau = y_tau - phi_tau - C * x_hat_{tau|tau-1}

# Compute the weight (scalar, time-varying):
W_tau = (C * Sigma_{tau+1|tau} * C^T + r)^{-1}

# Solve Lasso problem (closed-form soft-thresholding):
threshold = lambda / (2 * W_tau)

if e_tau > threshold:
    z_tau* = e_tau - threshold
elif e_tau < -threshold:
    z_tau* = e_tau + threshold
else:
    z_tau* = 0

# Modified correction:
e_tau_modified = e_tau - z_tau*
x_hat_{tau+1|tau+1} = x_hat_{tau+1|tau} + K_{tau+1} * (e_tau_modified - z_tau*)
```

Wait -- let me be more precise. The paper defines (Equations 31-32):

```
# Corrected innovation after removing outlier:
e_{tau+1} = y_{tau+1} - phi_{tau+1} - C * x_hat_{tau+1|tau}

# Soft-thresholding on e_{tau+1}:
z_{tau+1}* = soft_threshold(e_{tau+1}, lambda / (2 * W_{tau+1}))

# Modified Kalman correction:
x_hat_{tau+1|tau+1} = x_hat_{tau+1|tau} + K_{tau+1} * (e_{tau+1} - z_{tau+1}*)
Sigma_{tau+1|tau+1} = Sigma_{tau+1|tau} - K_{tau+1} * S_{tau+1} * K_{tau+1}^T
```

The soft-thresholding operator:

```
soft_threshold(x, delta):
    if x > delta:   return x - delta
    if x < -delta:  return x + delta
    else:            return 0
```

The residual after removing the outlier, used for EM parameter updates:

```
e_tau - z_tau* :
    if |e_tau| > lambda/(2*W_tau):  equals lambda/(2*W_tau) * sign(e_tau)
    if |e_tau| <= lambda/(2*W_tau): equals e_tau
```

This means the effective observation fed to the filter is clamped -- outliers beyond
the threshold contribute only up to the threshold amount.

Reference: Chen et al. (2016), Section 3.1, Equations (25)-(34), pages 6-7.

#### EM Modifications for Robust Model

In the M-step, the updates for r and phi are modified to account for the inferred
outlier values z_tau*:

```
# Modified observation noise variance:
r^(j+1) = (1/N) * sum_{tau=1}^{N} [y_tau^2 + C * P_tau * C^T - 2 * y_tau * C * x_hat_tau
                                    + (phi_tau^(j+1))^2 - 2 * y_tau * phi_tau^(j+1)
                                    + 2 * phi_tau^(j+1) * C * x_hat_tau
                                    + 2 * z_tau* * C * x_hat_tau
                                    + (z_tau*)^2 - 2 * z_tau* * y_tau
                                    + 2 * z_tau* * phi_tau^(j+1)]

# Modified seasonality:
phi_i^(j+1) = (1/T) * sum_{t=1}^{T} (y_{t,i} - C * x_hat_{t,i} - z_{t,i}*)
```

Reference: Chen et al. (2016), Section 3.2, Equations (35)-(36), page 7.

### Data Flow

```
Raw Input:
    volume_{t,i}  (shares traded in bin i of day t)
    shares_outstanding_t  (daily shares outstanding for normalization)
    I  (number of bins per day, exchange-dependent)

Step 1: Normalize
    normalized_{t,i} = volume_{t,i} / shares_outstanding_t
    Type: float64, shape (T, I)

Step 2: Log-transform
    y_{t,i} = ln(normalized_{t,i})
    Type: float64, shape (T, I)
    Precondition: normalized_{t,i} > 0 (zero-volume bins must be excluded)

Step 3: Flatten to single time index
    y_tau for tau = 1, ..., N where N = T * I
    Type: float64, shape (N,)

Step 4: Split into training/validation/test
    Training: first T_train days (N_train = T_train * I bins)
    Validation: next T_val days (for cross-validation of N and lambda)
    Test: remaining days

Step 5: EM Calibration (on training set)
    Input: y_1, ..., y_{N_train}, initial theta^(0)
    Output: estimated theta = (a^eta, a^mu, (sigma^eta)^2, (sigma^mu)^2, r, phi, pi_1, Sigma_1)
    Internal: runs Kalman filter + smoother + M-step iteratively
    Type: parameter vector of mixed scalars and arrays

Step 6: Kalman Filter Prediction (on test set)
    Input: estimated theta, new observations y_tau arriving one at a time
    Output per bin: y_hat_{tau+1|tau} (one-step-ahead log-volume forecast)
    Type: float64, scalar per bin

Step 7: Convert to volume forecast
    predicted_volume_{t,i} = exp(y_hat_{t,i}) * shares_outstanding_t
    Type: float64, shape per bin

Step 8: Compute VWAP weights
    For static VWAP:
        w_{t,i}^(s) = predicted_volume_{t,i} / sum_j(predicted_volume_{t,j})
    For dynamic VWAP (at bin i, having observed bins 1..i-1):
        w_{t,i}^(d) = predicted_volume_{t,i} / sum_{j=i}^{I}(predicted_volume_{t,j})
                     * (1 - sum_{j=1}^{i-1} w_{t,j}^(d))
    Type: float64, shape (I,), sums to 1.0
```

Reference: Chen et al. (2016), Section 2 (model), Section 4.1 (data setup), Section 4.3
(VWAP), Equations (1), (39)-(41).

### Variants

**Implement:** The Robust Kalman Filter (Section 3 of the paper). This is the most
complete variant, encompassing the standard Kalman filter as the special case lambda = 0.
The standard filter should be available by simply setting lambda = 0 (which disables the
soft-thresholding step, since the threshold becomes zero and z_tau* = 0 always).

**Rationale:** The robust variant achieves comparable or better performance than the
standard filter on clean data, and substantially better performance on data with outliers
(which is the norm for real-time market data). The CMEM benchmark fails entirely under
medium/large outlier contamination, making robustness a critical practical requirement.

Reference: Chen et al. (2016), Section 3, Table 1 (contaminated data results).

## Parameters

| Parameter | Description | Recommended Value | Sensitivity | Range |
|-----------|-------------|-------------------|-------------|-------|
| a^eta | AR(1) coefficient for daily component at day boundaries | Data-driven (EM); typically close to 1 (high persistence) | Medium -- controls day-to-day smoothing of daily level | (0, 1) for stationarity |
| a^mu | AR(1) coefficient for intraday dynamic component | Data-driven (EM); typically < a^eta | Medium -- controls intraday mean-reversion speed | (0, 1) for stationarity |
| (sigma^eta)^2 | Process noise variance for daily component | Data-driven (EM) | Low-medium -- absorbed by EM estimation | (0, inf) |
| (sigma^mu)^2 | Process noise variance for dynamic component | Data-driven (EM) | Low-medium -- absorbed by EM estimation | (0, inf) |
| r | Observation noise variance | Data-driven (EM) | Low-medium -- absorbed by EM estimation | (0, inf) |
| phi (vector) | Intraday seasonality, one value per bin | Data-driven (EM); captures U-shape in log space | High -- the seasonal pattern is the dominant predictable component | (-inf, inf) per bin |
| pi_1 (vector) | Initial state mean [eta_1, mu_1]^T | Data-driven (EM); initialized from first observation | Low -- EM converges regardless of initialization | (-inf, inf) |
| Sigma_1 (matrix) | Initial state covariance (2x2) | Data-driven (EM); initialized as identity or diagonal | Low -- EM converges regardless of initialization | positive definite |
| lambda | Lasso regularization for robust filter | Cross-validated on held-out period | High -- controls outlier sensitivity; too low = no robustness, too high = over-smoothing | [0, inf); 0 recovers standard filter |
| N (or T_train) | Training window length in bins (or days) | Cross-validated; paper uses rolling window | High -- too short = noisy estimates, too long = stale parameters | Typically 60-500 trading days |
| I | Number of intraday bins per day | Exchange-dependent; 15-min bins typical | N/A (structural, not tuned) | Determined by exchange hours / bin size |

### Initialization

**EM initialization (theta^(0)):**
The paper demonstrates (Section 2.3.3, Figure 4) that EM converges to the same parameter
values regardless of initial values. Nevertheless, reasonable defaults accelerate
convergence:

- a^eta^(0) = 0.99 (high persistence for daily component)
- a^mu^(0) = 0.5 (moderate persistence for intraday dynamics)
- (sigma^eta)^2^(0) = sample variance of daily mean log-volume
- (sigma^mu)^2^(0) = sample variance of demeaned, deseasonalized log-volume
- r^(0) = 0.1 * sample variance of log-volume
- phi^(0) = sample mean of y_{t,i} across days for each bin i (crude seasonal estimate)
- pi_1^(0) = [mean(y), 0]^T (daily average at sample mean, dynamic component at zero)
- Sigma_1^(0) = diag(sample_var(y), sample_var(y)) (diffuse initialization)

Reference: Chen et al. (2016), Section 2.3.3. The paper shows convergence from multiple
random initializations in Figure 4 (page 6). Researcher inference: the specific default
values above are not from the paper but are standard reasonable choices for starting EM
on this model structure.

**Kalman filter initialization (for prediction after calibration):**
- x_hat_{1|0} = pi_1 (estimated initial state from EM)
- Sigma_{1|0} = Sigma_1 (estimated initial covariance from EM)

### Calibration

**Rolling window EM calibration procedure:**

1. Select a cross-validation period (the paper uses January 2015 to May 2015).
2. Define a grid of candidate training window lengths N (in days) and, for the robust
   model, Lasso regularization values lambda.
3. For each (N, lambda) pair:
   a. Run EM on the training window of length N ending just before the cross-validation
      period.
   b. Run the (robust) Kalman filter on the cross-validation period using the estimated
      parameters.
   c. Compute MAPE on the cross-validation period.
4. Select the (N*, lambda*) pair with lowest cross-validation MAPE.
5. For out-of-sample forecasting, use a rolling window of length N*:
   a. At the start of each new day in the test period, re-run EM on the most recent N*
      bins to update parameters.
   b. Run the Kalman filter forward using updated parameters.

**EM convergence criterion:**
- Relative change in log-likelihood < epsilon (suggest epsilon = 1e-6)
- OR maximum iterations reached (suggest max_iter = 100; the paper shows convergence
  in "a few iterations")

Reference: Chen et al. (2016), Section 4.1 (cross-validation setup), Section 4.2
(rolling window). Researcher inference: the specific convergence threshold and max
iterations are not stated in the paper; 1e-6 and 100 are standard EM defaults.

## Validation

### Expected Behavior

**Volume prediction MAPE (dynamic prediction, out-of-sample):**
- Robust Kalman Filter: average MAPE = 0.46 across 30 securities
- Standard Kalman Filter: average MAPE = 0.47
- CMEM benchmark: average MAPE = 0.65
- Rolling Mean baseline: average MAPE = 1.28

This represents a 64% improvement over RM and 29% over CMEM.

**Volume prediction MAPE (static prediction, out-of-sample):**
- Robust Kalman Filter: average MAPE = 0.61
- Standard Kalman Filter: average MAPE = 0.62
- CMEM benchmark: average MAPE = 0.90
- Rolling Mean baseline: average MAPE = 1.28

**VWAP tracking error (dynamic strategy, out-of-sample):**
- Robust Kalman Filter: average 6.38 basis points
- Standard Kalman Filter: 6.39 bps
- CMEM: 7.01 bps
- RM: 7.48 bps

**Qualitative behavior:**
- The phi vector should exhibit a U-shape (or J-shape) in log space, with higher values
  at market open and close, lower values mid-day.
- a^eta should be close to 1 (high persistence of daily level).
- a^mu should be notably less than 1 (intraday dynamics mean-revert faster).
- EM should converge within 5-10 iterations from any reasonable initialization.

Reference: Chen et al. (2016), Section 4.2 (Table 3, MAPE results), Section 4.3
(Table 4, VWAP tracking error), Section 2.3.3 (EM convergence, Figure 4).

### Sanity Checks

1. **EM convergence test:** Initialize EM with 5 different random parameter vectors.
   All should converge to the same parameter values (within numerical tolerance).
   Reference: Chen et al. (2016), Section 2.3.3, Figure 4 -- convergence demonstrated
   from multiple initializations.

2. **Seasonality shape:** Plot the estimated phi vector. It should show the characteristic
   U-shape for U.S. equities (high open, low midday, high close). For a 15-minute bin
   on NYSE (I=26), the first and last few bins should have the highest phi values.

3. **State dimension check:** The Kalman gain K_tau should be a 2x1 vector, the innovation
   variance S_tau should be a scalar. If these dimensions are wrong, the state-space
   formulation is incorrectly set up.

4. **Covariance positive definiteness:** Sigma_{tau|tau} must remain positive definite at
   all time steps. If it becomes negative definite or singular, there is a numerical bug
   in the covariance update. Use the Joseph form (Sigma = (I - K*C)*Sigma_pred*(I - K*C)^T
   + K*r*K^T) for better numerical stability if needed.

5. **Robust filter reduces to standard:** With lambda = 0, the robust Kalman filter output
   should be identical (to machine precision) to the standard Kalman filter output,
   because z_tau* = 0 for all tau.

6. **Log-volume distribution:** Q-Q plot of log-volume residuals (y_tau - y_hat_tau) should
   be approximately Gaussian. Heavy tails in the standard model should be reduced by the
   robust model (since outliers are absorbed by z_tau*).

7. **Rolling mean baseline:** A simple rolling mean predictor (average of same bin over
   the past T_train days) should produce MAPE around 1.0-1.5 for U.S. large-cap stocks.
   The Kalman filter should improve on this by 50-70%.

8. **Specific ticker reference (SPY, clean data):** Dynamic MAPE approximately 0.24,
   static MAPE approximately 0.36. Reference: Chen et al. (2016), Table 3, SPY row.

### Edge Cases

1. **Zero-volume bins:** log(0) is undefined. These bins MUST be excluded from the
   observation sequence. When a bin is missing, skip the correction step and only run
   the prediction step (the Kalman filter naturally handles missing observations by
   simply not updating). The bin index tau must still advance to maintain correct
   alignment with the seasonality vector phi.
   Reference: Chen et al. (2016), Section 2, page 3 -- "we assume the observed volumes
   are non-zero for all bins."

2. **Day boundaries:** The transition matrix A_tau switches behavior at tau = k*I. Off-by-one
   errors here will corrupt the entire model. The day boundary occurs BETWEEN the last bin
   of day k and the first bin of day k+1. Specifically, the transition from tau = k*I to
   tau = k*I + 1 uses a^eta and (sigma^eta)^2; all other transitions use 1 and 0 for the
   eta component.

3. **Half-day sessions:** The paper excludes half-day sessions entirely (e.g., day before
   holidays in the U.S.). If included, I would vary by day, breaking the seasonality
   alignment. Exclude these days. Reference: Chen et al. (2016), Section 4.1.

4. **Very short training windows:** If N is too small, EM may produce degenerate parameter
   estimates (e.g., variances near zero). Enforce a minimum training window of at least
   20 trading days (20 * I bins).

5. **AR coefficient at boundary:** If a^eta or a^mu drifts to exactly 1.0 during EM,
   the process becomes a random walk and variance grows unboundedly. In practice this
   is unlikely but could occur with insufficient data. Consider clamping to [0.001, 0.999].
   Researcher inference: the paper does not discuss this edge case; the clamping suggestion
   is standard practice for AR parameter estimation.

6. **Outlier storms (robust model):** If a large fraction of bins are flagged as outliers
   (z_tau* != 0 for many consecutive bins), the filter effectively ignores observations
   and runs open-loop. This can happen if lambda is too small. Monitor the fraction of
   nonzero z_tau* values; if it exceeds, say, 20%, lambda may need to be increased.
   Researcher inference: threshold percentage is not from the paper.

7. **Cross-exchange differences in I:** Different exchanges have different trading hours,
   producing different I values (e.g., NYSE 6.5 hours / 15 min = 26 bins; Xetra 8.5
   hours / 15 min = 34 bins; Tokyo varies). The seasonality vector phi has dimension I,
   so the model must be parameterized per exchange. Reference: Chen et al. (2016),
   Table 2 (listing multiple exchanges).

### Known Limitations

1. **Cannot handle zero-volume bins.** This limits applicability to liquid securities
   where every bin has nonzero volume. Illiquid stocks with frequent zero-volume bins
   cannot be modeled without modification (e.g., imputation or mixed-state extension).
   Reference: Chen et al. (2016), Section 2, assumption of nonzero volumes.

2. **Log-normal bias in exponentiation.** The model forecasts E[log(volume)], but what is
   typically needed is E[volume] = exp(E[log(volume)] + 0.5 * Var[log(volume)]). The
   paper does not discuss this bias correction. For VWAP weight computation (ratios),
   the bias may cancel, but for absolute volume forecasts, a log-normal correction factor
   exp(0.5 * prediction_variance) should be applied.
   Researcher inference: this is a known issue with log-space forecasting models, not
   discussed in the paper.

3. **No exogenous variables.** The model is purely autoregressive with seasonality. It
   does not incorporate market-wide factors, news, earnings events, or cross-sectional
   information. The paper acknowledges this as future work.
   Reference: Chen et al. (2016), Section 5 (Conclusion).

4. **Assumes stationarity within the training window.** The AR structure and fixed
   seasonality assume the data-generating process is stable over the training period.
   Structural breaks (e.g., stock splits not corrected by normalization, changes in
   market microstructure) can degrade performance.

5. **Static VWAP degrades for longer horizons.** Multi-step-ahead predictions compound
   the AR decay, so static forecasts (all bins predicted at once before market open)
   are substantially worse than dynamic forecasts (updated after each bin). The paper
   confirms this: static MAPE ~0.61 vs dynamic MAPE ~0.46.
   Reference: Chen et al. (2016), Table 3.

6. **Single-stock model.** Each stock is modeled independently. There is no mechanism
   to share information across stocks or capture market-wide volume shocks.

## Paper References

| Spec Section | Paper Source |
|---|---|
| Model Description | Chen et al. (2016), Section 2, Equations (1)-(5) |
| State-Space Formulation | Chen et al. (2016), Section 2, Equations (4)-(5), page 3 |
| Kalman Filter Algorithm | Chen et al. (2016), Section 2.2, Algorithm 1, page 4 |
| Multi-step Forecast | Chen et al. (2016), Section 2.2, Equation (9), page 4 |
| Kalman Smoother | Chen et al. (2016), Section 2.3.1, Algorithm 2, page 5 |
| EM Algorithm | Chen et al. (2016), Section 2.3.2, Algorithm 3, pages 5-6 |
| EM Closed-Form M-step | Chen et al. (2016), Appendix A.3, Equations (A.32)-(A.39), pages 14-15 |
| EM Convergence Properties | Chen et al. (2016), Section 2.3.3, Figure 4, page 6 |
| Robust Kalman Filter | Chen et al. (2016), Section 3.1, Equations (25)-(34), pages 6-7 |
| Robust EM Modifications | Chen et al. (2016), Section 3.2, Equations (35)-(36), page 7 |
| Outlier Simulation Results | Chen et al. (2016), Section 3.3, Table 1, pages 7-9 |
| Data Setup and Normalization | Chen et al. (2016), Section 4.1, Equation (1), page 8 |
| Cross-Validation Procedure | Chen et al. (2016), Section 4.1, page 8 |
| MAPE Definition | Chen et al. (2016), Section 3.3, Equation (37), page 7 |
| VWAP Static Weights | Chen et al. (2016), Section 4.3, Equation (40), page 10 |
| VWAP Dynamic Weights | Chen et al. (2016), Section 4.3, Equation (41), page 10 |
| VWAP Tracking Error | Chen et al. (2016), Section 4.3, Equations (39), (42), page 10 |
| Out-of-Sample MAPE Results | Chen et al. (2016), Section 4.2, Table 3, page 10 |
| Out-of-Sample VWAP Results | Chen et al. (2016), Section 4.3, Table 4, page 11 |
| Log-Normal Bias Correction | Researcher inference (standard result, not in paper) |
| EM Initialization Defaults | Researcher inference (standard practice) |
| AR Coefficient Clamping | Researcher inference (standard practice) |
| Outlier Fraction Monitoring | Researcher inference |
