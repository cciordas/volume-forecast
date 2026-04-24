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

The loop is structured as correction-then-prediction. At each tau, the correction step
incorporates observation y_tau to update x_hat_{tau|tau-1} into x_hat_{tau|tau}, and then
the prediction step propagates forward to produce x_hat_{tau+1|tau}. This ordering ensures
all quantities are defined before use: at tau=1, the correction uses x_hat_{1|0} from
initialization.

```
Input: parameters theta = (a^eta, a^mu, (sigma^eta)^2, (sigma^mu)^2, r, phi, pi_1, Sigma_1)
       observations y_1, ..., y_N

Initialize:
    x_hat_{1|0} = pi_1          # initial state mean (2x1)
    Sigma_{1|0} = Sigma_1       # initial state covariance (2x2)

For tau = 1, 2, ..., N:

    # --- Correction step (update with observation y_tau) ---
    if y_tau is observed (bin has nonzero volume):
        # Innovation (scalar):
        innovation_tau = y_tau - C * x_hat_{tau|tau-1} - phi_tau

        # Innovation variance (scalar):
        S_tau = C * Sigma_{tau|tau-1} * C^T + r

        # Kalman gain (2x1):
        K_tau = Sigma_{tau|tau-1} * C^T / S_tau

        # Corrected state estimate:
        x_hat_{tau|tau} = x_hat_{tau|tau-1} + K_tau * innovation_tau

        # Corrected covariance (Joseph form for numerical stability):
        Sigma_{tau|tau} = (I - K_tau * C) * Sigma_{tau|tau-1} * (I - K_tau * C)^T
                          + K_tau * r * K_tau^T
    else:
        # Missing observation: skip correction, pass through prediction
        x_hat_{tau|tau} = x_hat_{tau|tau-1}
        Sigma_{tau|tau} = Sigma_{tau|tau-1}

    # --- Prediction step (propagate to tau+1) ---
    x_hat_{tau+1|tau} = A_tau * x_hat_{tau|tau}
    Sigma_{tau+1|tau} = A_tau * Sigma_{tau|tau} * A_tau^T + Q_tau

    # --- Forecast output ---
    # One-step-ahead log-volume forecast:
    y_hat_{tau+1|tau} = C * x_hat_{tau+1|tau} + phi_{tau+1}

    # Store S_tau and K_tau for smoother and EM (needed later)
```

For dynamic prediction: run both correction and prediction at each bin.
For static prediction: run correction steps only on training data, then run prediction
steps only (skip correction) for all I bins of the forecast day.

For h-step-ahead forecast:
    x_hat_{tau+h|tau} is obtained by iterating the prediction step h times:
        For k = 1, 2, ..., h:
            x_hat_{tau+k|tau} = A_{tau+k-1} * x_hat_{tau+k-1|tau}
            Sigma_{tau+k|tau} = A_{tau+k-1} * Sigma_{tau+k-1|tau} * A_{tau+k-1}^T + Q_{tau+k-1}
        (starting from x_hat_{tau|tau} and Sigma_{tau|tau})
    y_hat_{tau+h|tau} = C * x_hat_{tau+h|tau} + phi_{tau+h}

    The prediction covariance Sigma_{tau+h|tau} grows with each step due to accumulated
    process noise. The corresponding observation variance is:
        S_{tau+h|tau} = C * Sigma_{tau+h|tau} * C^T + r
    This determines the prediction confidence interval width and the log-normal bias
    correction factor exp(0.5 * S_{tau+h|tau}) for multi-step forecasts. For static
    prediction (all I bins predicted at once), later bins have larger Sigma_{tau+h|tau},
    explaining why static MAPE (0.61) exceeds dynamic MAPE (0.46).
    Researcher inference: the recursive covariance formula is a standard Kalman filter
    result, not specific to this paper. The paper gives only the mean formula (Equation 9).

Note on the Joseph form: The standard covariance update
Sigma_{tau|tau} = Sigma_{tau|tau-1} - K_tau * S_tau * K_tau^T is algebraically
equivalent but numerically unstable (can produce non-positive-definite covariances
due to floating-point errors). Since the state dimension is only 2x2, the Joseph form
has negligible computational overhead and should always be used.

Reference: Chen et al. (2016), Section 2.2, Algorithm 1, page 4. The paper's Algorithm 1
uses predict-then-correct with shifted indices (predicting tau+1 then correcting tau+1).
The correction-then-prediction ordering used here is mathematically equivalent and avoids
referencing undefined quantities at tau=1.

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

# Missing-observation bookkeeping:
# Let O = {tau : y_tau is observed} be the set of observed bins (nonzero volume).
# Let N_obs = |O| (number of observed bins).
# Let T_i = number of days where bin i has an observed volume (for i = 1, ..., I).
# The paper assumes all volumes are nonzero (Section 2, page 3), so O = {1..N},
# N_obs = N, and T_i = T for all i. If zero-volume bins are present, Algorithm 1's
# missing-observation branch handles the filter side; the M-step adjustments below
# handle the estimation side. Researcher inference: these M-step adjustments are not
# in the paper (which assumes no missing data) but follow directly from restricting
# the sufficient statistics to observed data points.

Set j = 0

Repeat until convergence:
    # --- E-step ---
    Run Kalman filter (Algorithm 1) with theta^(j) to get:
        x_hat_{tau|tau}, Sigma_{tau|tau} for tau = 1..N
        x_hat_{tau+1|tau}, Sigma_{tau+1|tau} for tau = 1..N
        S_tau, innovation_tau for tau in O  (innovation variances and innovations,
                                             stored only for observed bins)

    Run Kalman smoother (Algorithm 2) to get:
        x_hat_{tau|N} for tau = 1..N        (smoothed state means)
        P_tau for tau = 1..N                (smoothed second moments)
        P_{tau,tau-1} for tau = 2..N        (smoothed cross-moments)

    where the sufficient statistics are:
        x_hat_tau = x_hat_{tau|N}                                    # smoothed mean
        P_tau = Sigma_{tau|N} + x_hat_{tau|N} * x_hat_{tau|N}^T     # smoothed second moment
        P_{tau,tau-1} = Sigma_{tau,tau-1|N} + x_hat_{tau|N} * x_hat_{tau-1|N}^T

    # Compute log-likelihood for convergence check (innovation-based):
    # Sum over OBSERVED bins only (innovation_tau and S_tau are undefined for missing bins):
    LL^(j) = -(N_obs/2) * log(2*pi) - (1/2) * sum_{tau in O} [log(S_tau) + innovation_tau^2 / S_tau]

    # --- M-step (all closed-form) ---
    # IMPORTANT: phi must be computed before r (r uses phi^(j+1) per Equation A.38)

    # Initial state mean:
    pi_1^(j+1) = x_hat_1

    # Initial state covariance:
    Sigma_1^(j+1) = P_1 - x_hat_1 * x_hat_1^T

    # AR coefficient for daily component (using day-boundary indices only):
    # Let D = {tau : tau = kI+1, k = 1, 2, ..., T-1} be the set of day-boundary transitions
    # (T-1 elements, one per day-to-day transition from day 1->2 through day (T-1)->T)
    a^eta^(j+1) = [sum_{tau in D} P_{tau,tau-1}^(1,1)] / [sum_{tau in D} P_{tau-1}^(1,1)]

    # AR coefficient for intraday dynamic component (all time steps):
    a^mu^(j+1) = [sum_{tau=2}^{N} P_{tau,tau-1}^(2,2)] / [sum_{tau=2}^{N} P_{tau-1}^(2,2)]

    # Process noise variance for daily component:
    (sigma^eta)^2^(j+1) = (1/(T-1)) * sum_{tau in D} [P_tau^(1,1) + (a^eta^(j+1))^2 * P_{tau-1}^(1,1)
                                                        - 2 * a^eta^(j+1) * P_{tau,tau-1}^(1,1)]

    # Process noise variance for dynamic component:
    (sigma^mu)^2^(j+1) = (1/(N-1)) * sum_{tau=2}^{N} [P_tau^(2,2) + (a^mu^(j+1))^2 * P_{tau-1}^(2,2)
                                                        - 2 * a^mu^(j+1) * P_{tau,tau-1}^(2,2)]

    # Seasonality vector (per-bin average residual) -- MUST be computed before r:
    # Sum over OBSERVED instances of bin i only; normalize by T_i (not T):
    phi_i^(j+1) = (1/T_i) * sum_{t : y_{t,i} observed} (y_{t,i} - C * x_hat_{t,i})
    for each bin i = 1, ..., I

    # Observation noise variance (uses phi^(j+1) from above):
    # Sum over OBSERVED bins only; normalize by N_obs (not N):
    r^(j+1) = (1/N_obs) * sum_{tau in O} [y_tau^2 + C * P_tau * C^T - 2 * y_tau * C * x_hat_tau
                                           + (phi_tau^(j+1))^2 - 2 * y_tau * phi_tau^(j+1)
                                           + 2 * phi_tau^(j+1) * C * x_hat_tau]

    j = j + 1

    # --- Convergence check ---
    if j >= 2:
        relative_change = |LL^(j-1) - LL^(j-2)| / (|LL^(j-2)| + 1e-16)
        if relative_change < epsilon:
            break
    if j >= max_iterations:
        break
```

Note on M-step notation: P_tau^(1,1) means the (1,1) element of the 2x2 matrix P_tau,
P_tau^(2,2) means the (2,2) element, etc. P_{tau,tau-1}^(1,1) is the (1,1) element of
the cross-moment matrix.

Note on log-likelihood: The innovation-based log-likelihood formula is a standard result
for linear Gaussian state-space models, derived from the prediction error decomposition
of the observed-data likelihood. It uses quantities already computed in the Kalman filter
(innovation_tau and S_tau), so no additional computation is needed. The alternative --
the expected complete-data log-likelihood Q(theta | theta^(j)) from Equation (A.10),
pages 14-15 -- requires computing terms from the sufficient statistics. Either can be
used for convergence monitoring; the innovation-based form is simpler.
Researcher inference: the specific convergence formula is not stated in the paper.

Reference: Chen et al. (2016), Section 2.3.2, Algorithm 3, page 5-6; Appendix A.3,
Equations (A.32)-(A.39), pages 14-15. Equation (A.38) for r uses phi^(j+1), confirming
the ordering constraint. Equation (A.39) for phi does not depend on r.

#### Algorithm 4: Robust Kalman Filter (Lasso Extension)

The observation equation is augmented with a sparse noise term z_tau:

    y_tau = C * x_tau + phi_tau + v_tau + z_tau

where z_tau is nonzero only at outlier bins. The correction step in Algorithm 1 is
replaced by a Lasso-penalized correction that uses soft-thresholding to detect and
remove outliers from the innovation residual.

The robust correction step (replaces the correction step in Algorithm 1):

```
# At each tau where y_tau is observed:

# Innovation residual:
e_tau = y_tau - phi_tau - C * x_hat_{tau|tau-1}

# Innovation variance (same as standard filter):
S_tau = C * Sigma_{tau|tau-1} * C^T + r

# Weight for Lasso (scalar, time-varying):
W_tau = 1 / S_tau

# Kalman gain (same as standard filter):
K_tau = Sigma_{tau|tau-1} * C^T / S_tau

# Soft-thresholding to detect outlier component:
threshold_tau = lambda / (2 * W_tau)

z_tau* = soft_threshold(e_tau, threshold_tau)

# Corrected state estimate (using cleaned innovation):
x_hat_{tau|tau} = x_hat_{tau|tau-1} + K_tau * (e_tau - z_tau*)

# Corrected covariance (Joseph form, same as standard filter):
Sigma_{tau|tau} = (I - K_tau * C) * Sigma_{tau|tau-1} * (I - K_tau * C)^T
                  + K_tau * r * K_tau^T
```

The soft-thresholding operator:

```
soft_threshold(x, delta):
    if x > delta:   return x - delta
    if x < -delta:  return x + delta
    else:            return 0
```

The effect on the innovation residual e_tau - z_tau*:

```
e_tau - z_tau* :
    if |e_tau| > threshold_tau:  equals threshold_tau * sign(e_tau)   [clamped]
    if |e_tau| <= threshold_tau: equals e_tau                          [unchanged]
```

This means the effective innovation fed to the Kalman gain is clamped: observations
whose residual exceeds the threshold contribute only up to the threshold amount, with
the excess attributed to the outlier z_tau*. When lambda = 0, threshold_tau = 0 and
z_tau* = 0 for all tau, recovering the standard Kalman filter exactly.

Reference: Chen et al. (2016), Section 3.1, Equations (25)-(34), pages 6-7.
Specifically: Equation (31) defines e_tau, Equation (32) gives the corrected state
update x_hat = x_hat_pred + K * (e - z*), Equations (33)-(34) give the soft-thresholding
solution and the clamped residual.

#### EM Modifications for Robust Model

In the M-step, the updates for r and phi are modified to account for the inferred
outlier values z_tau*:

```
# Modified seasonality (compute before r):
# Sum over OBSERVED instances of bin i only; normalize by T_i (not T).
# z_{t,i}* is zero for missing bins (robust correction was skipped), so missing bins
# must be excluded to avoid referencing undefined z* values.
phi_i^(j+1) = (1/T_i) * sum_{t : y_{t,i} observed} (y_{t,i} - C * x_hat_{t,i} - z_{t,i}*)

# Modified observation noise variance (uses phi^(j+1) and z_tau* from above):
# Sum over OBSERVED bins only; normalize by N_obs (not N).
r^(j+1) = (1/N_obs) * sum_{tau in O} [y_tau^2 + C * P_tau * C^T - 2 * y_tau * C * x_hat_tau
                                       + (phi_tau^(j+1))^2 - 2 * y_tau * phi_tau^(j+1)
                                       + 2 * phi_tau^(j+1) * C * x_hat_tau
                                       + 2 * z_tau* * C * x_hat_tau
                                       + (z_tau*)^2 - 2 * z_tau* * y_tau
                                       + 2 * z_tau* * phi_tau^(j+1)]
```

All other M-step updates (pi_1, Sigma_1, a^eta, a^mu, (sigma^eta)^2, (sigma^mu)^2)
remain unchanged from Algorithm 3.

Reference: Chen et al. (2016), Section 3.2, Equations (35)-(36), page 7.

### Data Flow

```
Raw Input:
    volume_{t,i}  (shares traded in bin i of day t)
    shares_outstanding_t  (daily shares outstanding for normalization; single value per
                           day, same for all bins i within day t, typically sourced from
                           a reference data provider, not from the volume feed itself)
    price_{t,i}   (last recorded transaction price in bin i of day t; needed only for
                   VWAP evaluation, not for volume prediction)
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

Step 8: Compute VWAP

    VWAP definition (actual):
        VWAP_t = sum_{i=1}^{I} volume_{t,i} * price_{t,i} / sum_{i=1}^{I} volume_{t,i}
    where price_{t,i} is the last recorded transaction price in bin i.
    Reference: Chen et al. (2016), Section 4.3, Equation (39), page 10.

    For static VWAP (weights set before market open):
        w_{t,i}^(s) = predicted_volume_{t,i} / sum_{j=1}^{I}(predicted_volume_{t,j})
    Reference: Chen et al. (2016), Section 4.3, Equation (40), page 10.

    For dynamic VWAP (weights updated after each observed bin):
        w_{t,i}^(d) = predicted_volume_{t,i} / sum_{j=i}^{I}(predicted_volume_{t,j})
                     * (1 - sum_{j=1}^{i-1} w_{t,j}^(d))       for i = 1, ..., I-1
        w_{t,I}^(d) = 1 - sum_{j=1}^{I-1} w_{t,j}^(d)         for i = I (last bin)
    Reference: Chen et al. (2016), Section 4.3, Equation (41), page 10.

    VWAP weights sum to 1.0 for both strategies.
    Type: float64, shape (I,)
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
| epsilon | EM convergence threshold (relative change in log-likelihood) | 1e-6 | Low -- standard EM stopping criterion | (0, 1e-3) |
| max_iterations | Maximum EM iterations | 100 | Low -- EM typically converges in 5-10 iterations | [20, 500] |

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
      bins to update parameters. For rolling-window re-estimation, initialize theta^(0)
      with the previous day's converged parameters rather than the defaults from the
      Initialization section. This warm-start typically reduces EM iterations from 5-10
      down to 1-3, since parameters change slowly day to day (3-5x runtime improvement).
      Researcher inference: warm-starting is not discussed in the paper but is standard
      practice for rolling-window EM and follows from the slow parameter drift assumption.
   b. Run the Kalman filter forward using updated parameters.

**EM convergence criterion:**
- Relative change in innovation-based log-likelihood < epsilon (default epsilon = 1e-6):
      relative_change = |LL^(j) - LL^(j-1)| / (|LL^(j-1)| + 1e-16) < epsilon
  where LL^(j) = -(N_obs/2)*log(2*pi) - (1/2)*sum_{tau in O} [log(S_tau) + innovation_tau^2 / S_tau]
  and the sum is over observed bins only (N_obs = N when all bins are observed)
- OR maximum iterations reached (default max_iter = 100; the paper shows convergence
  in "a few iterations")

Reference: Chen et al. (2016), Section 4.1 (cross-validation setup), Section 4.2
(rolling window). Researcher inference: the specific convergence threshold and max
iterations are not stated in the paper; 1e-6 and 100 are standard EM defaults. The
innovation-based log-likelihood formula is a standard result for linear Gaussian
state-space models.

## Validation

### Expected Behavior

**MAPE definition:**

    MAPE = (1/M) * sum_{tau=1}^{M} |volume_tau - predicted_volume_tau| / volume_tau

where M is the total number of bins in the out-of-sample set and volumes are in the
original linear scale (not log). Reference: Chen et al. (2016), Section 3.3,
Equation (37), page 7.

**VWAP tracking error definition:**

    VWAP_TE = (1/D) * sum_{t=1}^{D} |VWAP_t - replicated_VWAP_t| / VWAP_t

where D is the number of out-of-sample days, VWAP_t is the actual VWAP (Equation 39),
and replicated_VWAP_t = sum_{i=1}^{I} w_{t,i} * price_{t,i} uses model-predicted
weights. The result is expressed in basis points (multiply by 10000).
Reference: Chen et al. (2016), Section 4.3, Equation (42), page 10.

**Volume prediction MAPE (dynamic prediction, out-of-sample):**
- Robust Kalman Filter: average of per-ticker MAPEs = 0.46 across 30 securities
- Standard Kalman Filter: average of per-ticker MAPEs = 0.47
- CMEM benchmark: average of per-ticker MAPEs = 0.65
- Rolling Mean baseline: average of per-ticker MAPEs = 1.28

This represents a 64% improvement over RM and 29% over CMEM.

**Volume prediction MAPE (static prediction, out-of-sample):**
- Robust Kalman Filter: average of per-ticker MAPEs = 0.61
- Standard Kalman Filter: average of per-ticker MAPEs = 0.62
- CMEM benchmark: average of per-ticker MAPEs = 0.90
- Rolling Mean baseline: average of per-ticker MAPEs = 1.28

Note: these are averages of per-ticker MAPEs (each ticker's MAPE is computed separately,
then averaged across tickers), not a single global MAPE across all bins of all tickers.

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

Reference: Chen et al. (2016), Section 4.2 (Table 3, Average row, MAPE results),
Section 4.3 (Table 4, VWAP tracking error; page 11 text for 6.38 bps),
Section 2.3.3 (EM convergence, Figure 4).

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
   in the covariance update. The Joseph form (used in Algorithm 1) should prevent this,
   but verify numerically (e.g., check that both eigenvalues of the 2x2 matrix are
   positive at every step).

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
   observation sequence. When a bin is missing, the correction step is skipped and only
   the prediction step runs (implemented as the else branch in Algorithm 1's correction
   step). The bin index tau must still advance to maintain correct alignment with the
   seasonality vector phi.
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
   exp(0.5 * prediction_variance) should be applied, where prediction_variance is
   S_tau (the innovation variance from the Kalman filter) for one-step-ahead forecasts.
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
| MAPE Definition | Chen et al. (2016), Section 3.3, Equation (37), page 7 |
| Data Setup and Normalization | Chen et al. (2016), Section 4.1, Equation (1), page 8 |
| Cross-Validation Procedure | Chen et al. (2016), Section 4.1, page 8 |
| VWAP Definition (actual) | Chen et al. (2016), Section 4.3, Equation (39), page 10 |
| VWAP Static Weights | Chen et al. (2016), Section 4.3, Equation (40), page 10 |
| VWAP Dynamic Weights | Chen et al. (2016), Section 4.3, Equation (41), page 10 |
| VWAP Tracking Error | Chen et al. (2016), Section 4.3, Equation (42), page 10 |
| Out-of-Sample MAPE Results | Chen et al. (2016), Section 4.2, Table 3, Average row, page 10 |
| Out-of-Sample VWAP Results | Chen et al. (2016), Section 4.3, Table 4, page 11 |
| Log-Likelihood Convergence | Researcher inference (standard innovation-based LL for linear Gaussian SSM) |
| Log-Normal Bias Correction | Researcher inference (standard result, not in paper) |
| EM Initialization Defaults | Researcher inference (standard practice) |
| AR Coefficient Clamping | Researcher inference (standard practice) |
| Outlier Fraction Monitoring | Researcher inference |
| EM M-step Missing-Obs Adjustments | Researcher inference (restricting sums to observed bins; paper assumes all nonzero) |
| Multi-step Prediction Covariance | Researcher inference (standard Kalman result; paper gives only mean in Eq 9) |
| Rolling-Window EM Warm-Start | Researcher inference (standard practice for rolling-window EM) |

## Changes from Draft 1

This section documents the revisions made in response to the critique of draft 1.

**M1 (Major) -- Kalman filter loop ordering:** Reordered Algorithm 1 to
correction-then-prediction. At each tau, the correction step now uses x_hat_{tau|tau-1}
(available from initialization or previous iteration's prediction) to produce
x_hat_{tau|tau}, then the prediction step uses x_hat_{tau|tau} to produce
x_hat_{tau+1|tau}. All quantities are defined before use at every tau including tau=1.

**M2 (Major) -- Robust filter double subtraction:** Removed the double subtraction of
z_tau*. The corrected state update now reads
x_hat_{tau|tau} = x_hat_{tau|tau-1} + K_tau * (e_tau - z_tau*), matching Equation (32)
which subtracts z* exactly once.

**HM1 -- Conflicting robust filter versions:** Consolidated Algorithm 4 into a single
clean presentation. Removed the first incomplete attempt and the "Wait" transition.
The single authoritative version uses consistent tau-based indexing throughout, with
clear references to Equations (31)-(34).

**ME1 -- Missing MAPE formula:** Added the MAPE definition (Equation 37) at the start
of the Validation section, with explicit note that volumes are in linear (not log) scale.

**ME2 -- Missing VWAP tracking error formula:** Added the VWAP tracking error definition
(Equation 42) in the Validation section, including the basis points conversion.

**ME3 -- Missing log-likelihood formula:** Added the innovation-based log-likelihood
formula in the EM convergence criterion, both in Algorithm 3 pseudocode and in the
Calibration section. Noted that this uses quantities already computed in the filter.

**ME4 -- Missing observations not in pseudocode:** Integrated the missing-observation
handling directly into Algorithm 1's correction step as an if/else branch, so a
developer cannot miss it.

**ME5 -- Missing VWAP equation (39):** Added the actual VWAP definition (Equation 39) in
the Data Flow section, including the note that price data (last trade price per bin) is
a required input for VWAP evaluation.

**MI1 -- Joseph form should be primary:** Made the Joseph form the primary covariance
update in Algorithm 1 (and Algorithm 4). Added a note explaining why the standard form
is numerically risky and the Joseph form has negligible overhead for 2x2 state.

**MI2 -- Normalization clarification:** Added a note in the Data Flow section that
shares_outstanding_t is a single value per day (same for all bins), typically from a
reference data provider.

**MI3 -- Table 3 averages clarification:** Changed all MAPE reporting to explicitly
state "average of per-ticker MAPEs" and added a note clarifying this is not a single
global MAPE.

**MI4 -- Phi update ordering:** Moved the phi update BEFORE the r update in the M-step
pseudocode (both standard and robust) and added an inline comment marking this ordering
as important.

## Changes from Draft 2

This section documents the revisions made in response to the critique of draft 2.

**ME1 (Medium) -- EM M-step not adapted for missing observations:** Added
missing-observation bookkeeping to Algorithm 3: defined O (observed set), N_obs, and T_i.
Restricted log-likelihood sum to observed bins with N_obs normalization. Restricted phi
update to observed instances per bin with T_i normalization. Restricted r update to
observed bins with N_obs normalization. Applied the same adjustments to the robust EM
phi and r updates (which also reference z_tau*, undefined for missing bins). Added a
precondition note explaining that under the paper's nonzero-volume assumption, O = {1..N}
and all adjustments reduce to the original formulas. Marked as researcher inference since
the paper assumes no missing data.

**MI1 (Minor) -- Day-boundary set D upper bound:** Changed D definition from
"k = 1, 2, ..." to "k = 1, 2, ..., T-1" and added a parenthetical noting this gives
T-1 elements (one per day-to-day transition).

**MI2 (Minor) -- Rolling window warm-start:** Added a note in the calibration section
(step 5a) recommending warm-start initialization from the previous day's converged
parameters for rolling-window EM re-estimation. Noted the expected 3-5x runtime
improvement. Marked as researcher inference.

**MI3 (Minor) -- Multi-step prediction variance:** Added the recursive prediction
covariance formula Sigma_{tau+k|tau} alongside the existing h-step mean formula.
Added the corresponding observation variance S_{tau+h|tau} and noted its role in
confidence intervals, log-normal bias correction, and the static-vs-dynamic MAPE gap.
Marked as researcher inference (standard Kalman result, paper gives only mean in Eq 9).
