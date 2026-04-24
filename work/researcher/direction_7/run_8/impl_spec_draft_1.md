# Implementation Specification: Kalman Filter State-Space Model for Intraday Volume

## Overview

A linear Gaussian state-space model that forecasts intraday trading volume by decomposing log-volume into three additive components: a daily average level, an intraday periodic (seasonal) pattern, and an intraday dynamic component. The log transformation converts the multiplicative volume structure into a tractable linear additive form, enabling exact Kalman filter recursions for prediction and closed-form EM updates for parameter estimation. A robust variant adds Lasso-penalized sparse noise detection for automatic outlier handling in live market data.

The model achieves 64% MAPE improvement over rolling means and 29% over the Component Multiplicative Error Model (CMEM), with 15% and 9% improvements respectively in VWAP tracking error (Paper: chen_feng_palomar_2016, Section 4, Tables 3-4).

## Algorithm

### Model Description

The model operates on intraday volume data aggregated to fixed-width bins (e.g., 15 minutes). For each bin, raw volume is normalized by daily shares outstanding and log-transformed. The log-volume observation for day t, bin i is decomposed as:

    y_{t,i} = eta_t + phi_i + mu_{t,i} + v_{t,i}

where:
- eta_t: daily average component (log scale) -- slowly evolving, captures day-to-day level shifts. Constant within a day, transitions with AR(1) dynamics at day boundaries.
- phi_i: intraday periodic component -- the U-shaped seasonal pattern. Deterministic, one value per bin index, shared across all days.
- mu_{t,i}: intraday dynamic component -- captures transient deviations from the seasonal pattern. Evolves every bin via AR(1) dynamics.
- v_{t,i}: observation noise, i.i.d. Gaussian with variance r.

**Assumptions:**
- Non-zero volume in every bin (log(0) is undefined). Zero-volume bins must be excluded or imputed before processing.
- Gaussian noise is reasonable in log space (supported by Q-Q plots in Paper, Section 2, Figure 1).
- The daily component is piecewise constant within each day.
- All three components are additive in log space (multiplicative in raw volume space).

**Inputs:** Time series of intraday volume observations, normalized by shares outstanding, at fixed bin granularity. Historical training window of N bins (= T days x I bins/day).

**Outputs:** One-step-ahead or multi-step-ahead log-volume forecasts for each bin, convertible to raw volume via exponentiation. Optionally, VWAP execution weights.

(Paper: chen_feng_palomar_2016, Section 2, Equations 1-5)

### Pseudocode

#### Notation and Index Convention

The paper uses a unified time index tau = 1, 2, ..., N where N = T * I (T training days, I bins per day). The mapping is: for day t and bin i, tau = (t-1)*I + i. Day boundaries occur at tau = k*I for k = 1, 2, ... The state vector is x_tau = [eta_tau, mu_tau]^T (dimension 2). The observation is scalar y_tau.

(Paper: chen_feng_palomar_2016, Section 2, page 3)

#### Step 1: Data Preprocessing

```
INPUT: raw_volume[t][i] for t=1..T_total, i=1..I
INPUT: shares_outstanding[t] for each day t

FOR each day t, bin i:
    normalized_volume[t][i] = raw_volume[t][i] / shares_outstanding[t]
    IF normalized_volume[t][i] <= 0:
        MARK as missing (exclude from training/filtering)
    ELSE:
        y[t][i] = log(normalized_volume[t][i])
```

Researcher inference: The paper states volume is "normalized by daily outstanding shares" (Section 4.1) but does not specify handling of exact zeros beyond excluding them. The mark-as-missing approach is the natural extension of the Kalman filter framework, which can skip correction steps for missing observations.

#### Step 2: EM Parameter Estimation (Model Calibration)

The EM algorithm estimates all model parameters from training data. It alternates between:
- E-step: Run forward Kalman filter + backward RTS smoother to compute sufficient statistics.
- M-step: Closed-form updates for all parameters.

```
INPUT: training observations y[1..N], initial parameter guess theta^(0)
OUTPUT: estimated parameters theta = (pi_1, Sigma_1, a_eta, a_mu, 
         sigma_eta^2, sigma_mu^2, r, phi[1..I])

INITIALIZE theta^(0):
    a_eta = 0.99, a_mu = 0.5
    sigma_eta^2 = 0.01, sigma_mu^2 = 0.1
    r = 0.5
    phi[i] = mean of y[tau] over all tau where bin(tau) == i, for i=1..I
    pi_1 = [mean(y), 0]^T
    Sigma_1 = diag(1.0, 1.0)

REPEAT until convergence (or max_iterations):
    j = j + 1
    
    // === E-STEP ===
    // Forward pass: Kalman filter (Algorithm 1 applied to training data)
    // Produces: x_hat[tau|tau], Sigma[tau|tau] for tau=1..N
    //           x_hat[tau+1|tau], Sigma[tau+1|tau] for tau=1..N
    RUN kalman_filter(y[1..N], theta^(j-1))
    
    // Backward pass: RTS smoother (Algorithm 2)
    // Produces: x_hat[tau|N], P[tau], P[tau,tau-1] for tau=1..N
    RUN rts_smoother(filter_output, theta^(j-1))
    
    // Compute sufficient statistics:
    // x_hat_tau = E[x_tau | y_1..y_N, theta^(j)]           (Eq A.15)
    // P_tau = E[x_tau * x_tau^T | y_1..y_N, theta^(j)]     (Eq A.16)
    // P_{tau,tau-1} = E[x_tau * x_{tau-1}^T | y_1..y_N, theta^(j)]  (Eq A.17)
    //
    // From smoother output:
    //   x_hat_tau = x_hat[tau|N]
    //   P_tau = Sigma[tau|N] + x_hat[tau|N] * x_hat[tau|N]^T
    //   P_{tau,tau-1} = Sigma[tau,tau-1|N] + x_hat[tau|N] * x_hat[tau-1|N]^T
    
    // === M-STEP === (all closed-form, Paper Appendix A.3, Eqs A.32-A.39)
    
    // Initial state mean and covariance:
    pi_1^(j+1) = x_hat_1                                      // Eq A.32
    Sigma_1^(j+1) = P_1 - x_hat_1 * x_hat_1^T                // Eq A.33
    
    // Daily AR coefficient (using only day-boundary transitions):
    // Let D = {tau : tau = k*I+1, k=1,2,...} (set of day-boundary indices)
    a_eta^(j+1) = [sum_{tau in D} P_{tau,tau-1}^(1,1)] / 
                  [sum_{tau in D} P_{tau-1}^(1,1)]             // Eq A.34
    
    // Intraday AR coefficient (using all transitions):
    a_mu^(j+1) = [sum_{tau=2}^{N} P_{tau,tau-1}^(2,2)] / 
                 [sum_{tau=2}^{N} P_{tau-1}^(2,2)]            // Eq A.35
    
    // Process noise variance for daily component:
    [sigma_eta^2]^(j+1) = (1/(T-1)) * sum_{tau in D} {
        P_tau^(1,1) + (a_eta^(j+1))^2 * P_{tau-1}^(1,1) 
        - 2 * a_eta^(j+1) * P_{tau,tau-1}^(1,1)
    }                                                          // Eq A.36
    
    // Process noise variance for dynamic component:
    [sigma_mu^2]^(j+1) = (1/(N-1)) * sum_{tau=2}^{N} {
        P_tau^(2,2) + (a_mu^(j+1))^2 * P_{tau-1}^(2,2) 
        - 2 * a_mu^(j+1) * P_{tau,tau-1}^(2,2)
    }                                                          // Eq A.37
    
    // Observation noise variance:
    r^(j+1) = (1/N) * sum_{tau=1}^{N} [
        y_tau^2 + C * P_tau * C^T - 2*y_tau * C * x_hat_tau
        + (phi_tau^(j+1))^2 - 2*y_tau*phi_tau^(j+1) 
        + 2*phi_tau^(j+1) * C * x_hat_tau
    ]                                                          // Eq A.38
    
    // Seasonality vector (simple average residual per bin):
    FOR i = 1..I:
        phi_i^(j+1) = (1/T) * sum_{t=1}^{T} (y_{t,i} - C * x_hat_{t,i})
                                                               // Eq A.39
    
    // Check convergence:
    IF |log_likelihood^(j+1) - log_likelihood^(j)| < epsilon:
        BREAK

OUTPUT theta^(j+1)
```

(Paper: chen_feng_palomar_2016, Section 2.3, Algorithms 2-3, Appendix A, Equations A.32-A.39)

#### Step 3: Kalman Filter (Prediction)

```
INPUT: parameters theta, observations y[tau] arriving sequentially
OUTPUT: one-step-ahead predictions y_hat[tau+1|tau]

DEFINE:
    C = [1, 1]  (observation matrix, 1x2)
    
    A_tau = [[a_tau_eta, 0],    (state transition matrix, 2x2)
             [0,         a_mu]]
    where a_tau_eta = a_eta if tau is a day boundary (tau = k*I),
                      1     otherwise
    
    Q_tau = [[sigma_tau_eta^2, 0          ],   (process noise covariance)
             [0,               sigma_mu^2 ]]
    where sigma_tau_eta^2 = sigma_eta^2 if tau is a day boundary,
                            0           otherwise

INITIALIZE:
    x_hat[1|0] = pi_1
    Sigma[1|0] = Sigma_1

FOR tau = 1, 2, ...:
    // --- PREDICTION STEP ---
    // Predict state for next time step:
    x_hat[tau+1|tau] = A_tau * x_hat[tau|tau]                  // Alg 1, line 2
    Sigma[tau+1|tau] = A_tau * Sigma[tau|tau] * A_tau^T + Q_tau // Alg 1, line 3
    
    // Predict observation:
    y_hat[tau+1|tau] = C * x_hat[tau+1|tau] + phi_{tau+1}     // Eq 9
    
    // --- CORRECTION STEP (if observation y_{tau+1} is available) ---
    // Kalman gain (scalar inversion since observation is 1-D):
    S = C * Sigma[tau+1|tau] * C^T + r          // innovation variance (scalar)
    K[tau+1] = Sigma[tau+1|tau] * C^T / S       // Kalman gain (2x1 vector)
                                                               // Alg 1, line 4
    
    // Innovation (prediction error):
    e[tau+1] = y[tau+1] - y_hat[tau+1|tau]
    
    // Update state estimate:
    x_hat[tau+1|tau+1] = x_hat[tau+1|tau] + K[tau+1] * e[tau+1]  // Alg 1, line 5
    
    // Update covariance:
    Sigma[tau+1|tau+1] = Sigma[tau+1|tau] - K[tau+1] * S * K[tau+1]^T
                       = (I_2 - K[tau+1] * C) * Sigma[tau+1|tau]  // Alg 1, line 6
    
    // Convert forecast to linear scale if needed:
    // volume_hat[tau+1] = exp(y_hat[tau+1|tau])
```

**Dynamic prediction mode:** Execute both prediction and correction steps at each bin as observations arrive. This produces one-bin-ahead forecasts.

**Static prediction mode:** At the end of day t (after processing all I bins), skip correction steps and perform h successive prediction steps for h = 1..I to forecast all bins of day t+1. The multi-step-ahead forecast is:

```
FOR h = 1, 2, ..., I:
    x_hat[tau+h|tau] = A^h * x_hat[tau|tau]    // where A^h means h applications
    y_hat[tau+h|tau] = C * x_hat[tau+h|tau] + phi_{tau+h}     // Eq 9
```

Note: since eta is constant within a day (a_tau_eta = 1 for intraday steps), the static multi-step prediction only decays the mu component: mu_hat decays as (a_mu)^h while eta_hat remains constant.

(Paper: chen_feng_palomar_2016, Section 2.2, Algorithm 1, Equation 9)

#### Step 4: RTS Smoother (for EM calibration only)

```
INPUT: Kalman filter output {x_hat[tau|tau], Sigma[tau|tau], 
        x_hat[tau+1|tau], Sigma[tau+1|tau]} for tau=1..N
       Parameters theta
OUTPUT: Smoothed estimates {x_hat[tau|N], Sigma[tau|N]} for tau=1..N
        Cross-covariance {Sigma[tau,tau-1|N]} for tau=2..N

// Initialize from last filter step:
x_hat[N|N] = x_hat[N|N]  (from filter)
Sigma[N|N] = Sigma[N|N]  (from filter)

// Backward pass:
FOR tau = N-1, N-2, ..., 1:
    // Smoother gain:
    L_tau = Sigma[tau|tau] * A_tau^T * inv(Sigma[tau+1|tau])   // Alg 2, line 2
    
    // Smoothed state:
    x_hat[tau|N] = x_hat[tau|tau] + L_tau * (x_hat[tau+1|N] - x_hat[tau+1|tau])
                                                               // Alg 2, line 3
    
    // Smoothed covariance:
    Sigma[tau|N] = Sigma[tau|tau] + L_tau * (Sigma[tau+1|N] - Sigma[tau+1|tau]) * L_tau^T
                                                               // Alg 2, line 4

// Cross-covariance computation (needed for M-step):
// Initialize:
Sigma[N,N-1|N] = (I_2 - K[N] * C) * A_{N-1} * Sigma[N-1|N-1]  // Eq A.21

FOR tau = N-1, N-2, ..., 2:
    Sigma[tau,tau-1|N] = Sigma[tau|tau] * L_{tau-1}^T 
        + L_tau * (Sigma[tau+1,tau|N] - A_tau * Sigma[tau|tau]) * L_{tau-1}^T
                                                               // Eq A.20
```

(Paper: chen_feng_palomar_2016, Section 2.3.1, Algorithm 2, Appendix A Equations A.18-A.22)

#### Step 5: Robust Kalman Filter (Lasso Extension)

The robust variant modifies the observation equation to include a sparse outlier term z_tau:

    y_tau = C * x_tau + phi_tau + v_tau + z_tau

where z_tau is nonzero only at outlier bins. The correction step is modified to solve a Lasso-penalized problem.

```
// Modified correction step (replaces Step 3 correction):

// Define innovation residual:
e_tau = y_tau - phi_tau - C * x_hat[tau|tau-1]                 // Eq 31

// Innovation variance (scalar):
W_tau = (C * Sigma[tau|tau-1] * C^T + r)^{-1}                 // scalar

// Solve Lasso subproblem (Eq 30):
// min_{z_tau} (e_tau - z_tau)^T * W_tau * (e_tau - z_tau) + lambda * |z_tau|
//
// Since W_tau and e_tau are scalar, the solution is soft-thresholding:

threshold = lambda / (2 * W_tau)

IF e_tau > threshold:
    z_tau_star = e_tau - threshold                             // Eq 33
ELIF e_tau < -threshold:
    z_tau_star = e_tau + threshold                             // Eq 33
ELSE:
    z_tau_star = 0                                             // Eq 33

// The effective innovation after outlier removal:
e_tau_clean = e_tau - z_tau_star                               // Eq 34
// Note: e_tau_clean is soft-thresholded version of e_tau:
//   IF |e_tau| > threshold: e_tau_clean = sign(e_tau) * threshold
//   ELSE: e_tau_clean = e_tau

// Corrected state estimate:
x_hat[tau|tau] = x_hat[tau|tau-1] + K_tau * (e_tau - z_tau_star)  // Eq 32
// where K_tau is the standard Kalman gain as before

// Covariance update same as standard Kalman filter.
```

**Robust EM modification** (for calibration with outlier-contaminated training data):

```
// In the M-step, r and phi are updated using the inferred z* values:

r^(j+1) = (1/N) * sum_{tau=1}^{N} [
    y_tau^2 + C*P_tau*C^T - 2*y_tau*C*x_hat_tau + (phi_tau^(j+1))^2
    - 2*y_tau*phi_tau^(j+1) + 2*phi_tau^(j+1)*C*x_hat_tau
    + (z_tau_star)^2 - 2*z_tau_star*y_tau 
    + 2*z_tau_star*phi_tau^(j+1) + 2*z_tau_star*C*x_hat_tau   // Eq 35 (partial)
]

phi_i^(j+1) = (1/T) * sum_{t=1}^{T} (y_{t,i} - C*x_hat_{t,i} - z_{t,i}_star)
                                                               // Eq 36
```

(Paper: chen_feng_palomar_2016, Section 3, Equations 25-36)

#### Step 6: VWAP Execution

**Static VWAP** (weights set before market open):

```
// At end of day t, forecast all bins of day t+1 using static prediction:
FOR i = 1..I:
    y_hat[t+1,i] = static_predict(i steps ahead from end of day t)
    volume_hat[t+1,i] = exp(y_hat[t+1,i])

// Compute weights:
total = sum_{i=1}^{I} volume_hat[t+1,i]
FOR i = 1..I:
    w[t+1,i] = volume_hat[t+1,i] / total                      // Eq 40

// Execute: in bin i, trade w[t+1,i] fraction of total order
```

**Dynamic VWAP** (weights updated after each bin):

```
// After observing bins 1..j of day t+1:
FOR i = 1..j:
    // Already executed with weight w[t+1,i]

// Re-forecast remaining bins j+1..I using dynamic prediction:
remaining_proportion = 1 - sum_{i=1}^{j} w[t+1,i]

FOR i = j+1..I:
    y_hat[t+1,i] = dynamic_predict(1 step ahead, using obs through bin j)
    volume_hat[t+1,i] = exp(y_hat[t+1,i])

total_remaining = sum_{i=j+1}^{I} volume_hat[t+1,i]
FOR i = j+1..I:
    w[t+1,i] = remaining_proportion * volume_hat[t+1,i] / total_remaining
                                                               // Eq 41
```

(Paper: chen_feng_palomar_2016, Section 4.3, Equations 39-42)

### Data Flow

```
Raw volume data (T days x I bins/day)
    |
    v
[Preprocessing] -- normalize by shares outstanding, take log
    |
    v
Log-volume series y[1..N] where N = T*I
    |
    +---> [EM Calibration] (on training window of N_train bins)
    |         |
    |         +---> Forward Kalman Filter --> {x_hat[tau|tau], Sigma[tau|tau]}
    |         |
    |         +---> Backward RTS Smoother --> {x_hat[tau|N], P_tau, P_{tau,tau-1}}
    |         |
    |         +---> M-step closed-form updates --> theta^(j+1)
    |         |
    |         +---> (repeat until convergence)
    |         |
    |         v
    |     Estimated parameters theta = {a_eta, a_mu, sigma_eta^2, sigma_mu^2, 
    |                                    r, phi[1..I], pi_1, Sigma_1}
    |
    +---> [Online Prediction] (using estimated theta)
              |
              +---> Kalman Filter predict/correct at each bin
              |     State: x_tau = [eta_tau, mu_tau]^T  (2x1)
              |     Covariance: Sigma_tau (2x2)
              |     Observation: y_tau (scalar)
              |
              +---> Log-volume forecast: y_hat = C*x_hat + phi_i (scalar)
              |
              +---> exp(y_hat) --> volume forecast (scalar per bin)
              |
              +---> [VWAP Weight Computation] --> execution schedule

Types/shapes at each stage:
- x_tau: float[2] (state vector)
- Sigma_tau: float[2][2] (state covariance)
- A_tau: float[2][2] (transition matrix, time-varying)
- Q_tau: float[2][2] (process noise covariance, time-varying)
- C: float[2] = [1, 1] (observation vector, constant)
- K_tau: float[2] (Kalman gain)
- y_tau: float (scalar observation)
- phi: float[I] (seasonality vector)
- r: float (observation noise variance, scalar)
- S_tau: float (innovation variance, scalar)
```

### Variants

**Implement both variants:**

1. **Standard Kalman Filter** -- the base model without outlier handling. Simpler, slightly faster, and performs comparably to the robust version on clean (curated) data. Implement first as the foundation.

2. **Robust Kalman Filter** (Lasso extension) -- adds sparse noise detection via soft-thresholding in the correction step. Requires one additional hyperparameter (lambda). Implement as a subclass or configuration option on top of the standard model.

The robust variant is recommended for production use on live market data where outliers are expected. On curated historical data, both variants give very similar results (Paper, Table 3: average MAPE 0.46 vs 0.47).

(Paper: chen_feng_palomar_2016, Sections 2 and 3)

## Parameters

| Parameter | Description | Recommended Value | Sensitivity | Range |
|-----------|-------------|-------------------|-------------|-------|
| a_eta | Day-to-day AR(1) coefficient for daily component | EM-estimated; typically 0.95-0.999 | Medium -- controls daily persistence. Values near 1 mean slow adaptation to level shifts. | (0, 1) for stationarity |
| a_mu | Bin-to-bin AR(1) coefficient for dynamic component | EM-estimated; typically 0.3-0.7 | Medium -- controls intraday mean-reversion speed. Higher values = more persistent intraday deviations. | (0, 1) for stationarity |
| sigma_eta^2 | Process noise variance for daily component (at day boundaries) | EM-estimated | Low -- determined by data; interacts with a_eta | (0, inf) |
| sigma_mu^2 | Process noise variance for dynamic component | EM-estimated | Low -- determined by data; interacts with a_mu | (0, inf) |
| r | Observation noise variance | EM-estimated | Low-medium -- affects Kalman gain magnitude | (0, inf) |
| phi[1..I] | Intraday seasonality vector (I values) | EM-estimated as mean residual per bin (Eq A.39) | Low -- captures stable U-shaped pattern | (-inf, inf) in log space |
| pi_1 | Initial state mean [eta_0, mu_0]^T | EM-estimated (Eq A.32); initialize as [mean(y), 0]^T | Low -- EM converges regardless of initialization (Paper Section 2.3.3, Figure 4) | any real 2-vector |
| Sigma_1 | Initial state covariance (2x2) | EM-estimated (Eq A.33); initialize as diag(1,1) | Low -- same as above | positive definite 2x2 |
| lambda | Lasso regularization (robust model only) | Cross-validated; not reported in paper | High -- controls outlier sensitivity. Too small = no robustness; too large = over-smoothing. | (0, inf) |
| N | Training window length (in bins) | Cross-validated; paper uses Jan 2013 to validation start (~500 days, varies by exchange) | Medium -- too short = underfitting; too long = stale parameters | Typically 100-1000 days worth of bins |
| I | Number of bins per trading day | 26 for NYSE (6.5 hr session / 15 min); varies by exchange | N/A (structural) | Determined by exchange hours and bin width |
| max_em_iterations | Maximum EM iterations | 50-100 | Low -- EM typically converges in < 20 iterations (Paper Figure 4) | 10-200 |
| em_convergence_tol | EM convergence tolerance (log-likelihood change) | 1e-6 | Low | 1e-8 to 1e-4 |

### Initialization

**EM parameter initialization** (Paper Section 2.3.3 shows robustness to initialization):

1. Set a_eta = 0.99 (near unit root for daily persistence).
2. Set a_mu = 0.5 (moderate intraday mean-reversion).
3. Set sigma_eta^2 = sample variance of daily mean log-volumes / 10.
4. Set sigma_mu^2 = sample variance of demeaned log-volumes / 10.
5. Set r = sample variance of demeaned log-volumes / 2.
6. Set phi[i] = sample mean of y[t,i] across all training days t, for each bin i.
7. Set pi_1 = [mean of all y values, 0]^T.
8. Set Sigma_1 = diag(var(y), var(y)).

Researcher inference: The paper states EM is "robust to the choice of initial parameters" (Section 2.3.3) and shows convergence from widely varying initializations in Figure 4. The specific initial values above are reasonable defaults; any values in the correct order of magnitude should work. The key insight is that initialization sensitivity is NOT a concern for this model (unlike CMEM).

**Kalman filter state initialization** (for online prediction after EM):

After EM converges, pi_1 and Sigma_1 are the estimated initial state parameters. For online prediction starting at the beginning of a new out-of-sample period, initialize the filter state with:
- x_hat[1|0] = pi_1 (from EM)
- Sigma[1|0] = Sigma_1 (from EM)

Alternatively, run the Kalman filter through the last portion of the training data to "warm up" the state before entering the out-of-sample period. This avoids the transient from the prior and is recommended for production use.

### Calibration

**Rolling window re-estimation procedure:**

```
1. Select training window length N (number of days) and, for the robust
   model, lambda via cross-validation:
   
   a. Define a validation period (e.g., 5 months of data).
   b. For each candidate N in a grid (e.g., 100, 200, 300, ..., 1000 days):
      - For each candidate lambda in a grid (for robust model):
        - Train on the N days preceding the validation period.
        - Compute MAPE on the validation period.
   c. Select (N, lambda) pair with lowest validation MAPE.
   
   (Paper Section 4.1: validation period is Jan-May 2015)

2. For out-of-sample forecasting, use a standard rolling window:
   
   a. At the start of each out-of-sample day d:
      - Use the most recent N days as training data.
      - Run EM to estimate theta.
      - Use theta for Kalman filter predictions on day d.
   
   b. Optionally, re-estimate less frequently (e.g., weekly) if 
      computational budget is limited. The EM is fast enough for 
      daily re-estimation given the small state dimension.

3. EM convergence check:
   - Monitor the log-likelihood at each iteration.
   - Stop when |LL^(j+1) - LL^(j)| < epsilon (e.g., 1e-6).
   - Or after max_iterations (e.g., 50).
   - EM typically converges in fewer than 20 iterations.
```

Researcher inference: The paper does not specify the re-estimation frequency. Daily re-estimation is feasible given the O(N) per-iteration cost of the EM (forward filter + backward smoother, each O(N) with 2x2 matrices). Weekly re-estimation is a reasonable economy if needed.

## Validation

### Expected Behavior

**Volume prediction accuracy (MAPE, dynamic prediction, out-of-sample):**

| Model | Average MAPE |
|-------|-------------|
| Robust Kalman Filter | 0.46 |
| Standard Kalman Filter | 0.47 |
| CMEM | 0.65 |
| Rolling Means | 1.28 |

(Paper: Table 3, average over 30 securities, Section 4.2)

**Volume prediction accuracy (MAPE, static prediction, out-of-sample):**

| Model | Average MAPE |
|-------|-------------|
| Robust Kalman Filter | 0.61 |
| Standard Kalman Filter | 0.62 |
| CMEM | 0.90 |
| Rolling Means | 1.28 |

(Paper: Table 3, Section 4.2)

**VWAP tracking error (basis points, dynamic strategy):**

| Model | Average TE (bps) |
|-------|-----------------|
| Robust Kalman Filter | 6.38 |
| Standard Kalman Filter | 6.39 |
| CMEM | 7.01 |
| Rolling Means | 7.48 |

(Paper: Table 4, Section 4.3)

**Per-ticker MAPE examples (dynamic prediction):**

| Ticker | Robust KF | Standard KF | CMEM | RM |
|--------|-----------|-------------|------|-----|
| AAPL | 0.21 | 0.21 | 0.23 | 0.44 |
| SPY | 0.24 | 0.24 | 0.26 | 0.35 |
| IBM | 0.24 | 0.24 | 0.27 | 0.42 |
| DIA | 0.38 | 0.38 | 0.45 | 0.73 |

(Paper: Table 3, selected rows)

**EM convergence:** Parameters should converge within ~10-20 iterations from any reasonable initialization. Figure 4 in the paper demonstrates convergence from 8 different initializations for each parameter, all converging to the same values within ~5 iterations.

### Sanity Checks

1. **Seasonality shape:** After EM estimation, phi[1..I] should exhibit the well-known U-shape (or J-shape): higher values at the open (phi[1]) and close (phi[I]), lower values mid-day. Plot phi and verify visually.

2. **AR coefficient ranges:** a_eta should be close to 1 (high daily persistence, e.g., 0.95-0.999). a_mu should be moderate (e.g., 0.3-0.7), reflecting faster mean-reversion of intraday dynamics.

3. **EM convergence monotonicity:** The log-likelihood must be non-decreasing at every EM iteration. A decrease indicates a bug in the E-step or M-step implementation.

4. **Kalman filter innovation check:** The standardized innovations e_tau / sqrt(S_tau) should be approximately N(0,1). Compute their mean (should be ~0) and variance (should be ~1). If the variance is significantly > 1, the model is underestimating uncertainty.

5. **Synthetic data test:** Generate synthetic data from the model with known parameters. Run EM and verify that estimated parameters converge to the true values. This is the test performed in Paper Section 2.3.3 / Figure 4.

6. **Rolling means baseline:** Implement a simple rolling means predictor (average of same bin across last N_rm days) and verify that the Kalman filter substantially outperforms it (expect ~50-60% MAPE reduction).

7. **State covariance positive-definiteness:** Sigma[tau|tau] and Sigma[tau+1|tau] must remain positive definite throughout the filter run. If they become negative or singular, there is a numerical issue.

8. **Robust filter outlier detection:** On clean data, z_tau_star should be zero for the vast majority of bins (>95%). On artificially contaminated data (add large values to 10% of bins), z_tau_star should be nonzero at approximately the contaminated bins.

### Edge Cases

1. **Zero-volume bins:** Log(0) is undefined. These must be excluded before processing. For the Kalman filter, skip the correction step when a bin is missing (propagate the state with prediction only). This preserves the filter state without incorporating bad data.

   Researcher inference: The paper explicitly excludes zero-volume bins (Section 4.1) but does not describe the Kalman filter handling of missing observations. The standard approach in state-space models is to skip the correction step, which is well-established in the Kalman filter literature.

2. **Half-day sessions:** The paper excludes half-day sessions (Section 4.1, "excluding half-day sessions"). If included, the day would have fewer bins, creating a mismatch with the seasonality vector phi. Either exclude these days or handle by truncating phi to match the available bins.

3. **Day boundary transitions:** The transition matrix A_tau changes at day boundaries (tau = k*I). The implementation must correctly detect when a step crosses from one day to the next and switch from (a_eta=1, sigma_eta^2=0) to (a_eta=estimated, sigma_eta^2=estimated).

4. **Very small training windows:** With too few training days, the EM may not reliably estimate all parameters (especially sigma_eta^2 which only receives one data point per day boundary). Minimum recommended: ~50 trading days (roughly 2.5 months).

5. **Numerical stability of covariance updates:** The Joseph form of the covariance update Sigma[tau+1|tau+1] = (I - K*C) * Sigma[tau+1|tau] * (I - K*C)^T + K*r*K^T is more numerically stable than the simple form. Consider using it if positive-definiteness issues arise, although with 2x2 matrices this is unlikely.

6. **Lambda = 0 in robust model:** Degenerates to the standard Kalman filter (z_tau_star = e_tau always, removing all innovation signal). Lambda must be strictly positive.

7. **Lambda too large in robust model:** The threshold lambda/(2*W_tau) exceeds all innovations, so z_tau_star = 0 always, again reducing to the standard Kalman filter. This is harmless but wastes the robust capability.

8. **Stocks changing exchange or shares outstanding:** Abrupt changes in shares outstanding (splits, offerings) cause level shifts in normalized volume. The daily component eta should absorb gradual shifts, but sudden large changes may trigger the robust filter's outlier detection or require re-calibration.

### Known Limitations

1. **Cannot handle zero-volume bins.** The log transformation is undefined for zero volume. This limits applicability to sufficiently liquid securities. The paper acknowledges this (Section 5 discussion, also noted in the summary).

2. **Linear Gaussian assumption.** The model assumes Gaussian noise in log space. While Q-Q plots support this for liquid securities (Figure 1), heavy-tailed distributions or regime changes may violate this assumption.

3. **No exogenous covariates.** The model uses only past volume observations. It does not incorporate price, volatility, spread, or event information. The paper identifies this as future work (Section 5).

4. **Static seasonality within training window.** The seasonality vector phi is estimated as a simple average and does not evolve over time. If the intraday pattern changes (e.g., due to market structure changes), the model adapts only through the rolling window re-estimation.

5. **No direct comparison with BDF or GAS-Dirichlet models.** The paper benchmarks against CMEM and rolling means only (Section 4). Performance relative to other model families is unknown.

6. **Single-stock model.** No cross-sectional information is used. Each security is modeled independently. This is in contrast to the BDF approach (Direction 2) which exploits cross-sectional commonality.

7. **Bias from log-space prediction.** Forecasting in log space and exponentiating introduces a Jensen's inequality bias: E[exp(y)] > exp(E[y]). The paper does not apply a bias correction. For volume share computation (ratios), this bias partially cancels, but for absolute volume forecasts, a correction factor of exp(S_tau/2) could be applied.

   Researcher inference: This is a well-known issue in log-normal forecasting but is not discussed in the paper. The bias is small when prediction variance S_tau is small.

## Paper References

| Spec Section | Paper Source |
|-------------|-------------|
| Model decomposition (y = eta + phi + mu + v) | chen_feng_palomar_2016, Section 2, Equations 1-5 |
| State-space formulation (A_tau, C, Q_tau) | chen_feng_palomar_2016, Section 2, page 3-4 |
| Kalman filter algorithm | chen_feng_palomar_2016, Section 2.2, Algorithm 1 |
| Dynamic vs static prediction | chen_feng_palomar_2016, Section 2.2, Equation 9 |
| RTS smoother | chen_feng_palomar_2016, Section 2.3.1, Algorithm 2 |
| EM algorithm structure | chen_feng_palomar_2016, Section 2.3.2, Algorithm 3 |
| Sufficient statistics definitions | chen_feng_palomar_2016, Appendix A.2, Equations A.15-A.22 |
| M-step closed-form updates | chen_feng_palomar_2016, Appendix A.3, Equations A.32-A.39 |
| Cross-covariance initialization | chen_feng_palomar_2016, Appendix A, Equation A.21 |
| EM convergence demonstration | chen_feng_palomar_2016, Section 2.3.3, Figure 4 |
| Robust observation model | chen_feng_palomar_2016, Section 3.1, Equations 25-30 |
| Soft-thresholding solution | chen_feng_palomar_2016, Section 3.1, Equations 33-34 |
| Robust EM modifications | chen_feng_palomar_2016, Section 3.2, Equations 35-36 |
| Robustness simulations | chen_feng_palomar_2016, Section 3.3, Table 1 |
| Data setup and normalization | chen_feng_palomar_2016, Section 4.1 |
| MAPE definition | chen_feng_palomar_2016, Section 3.3, Equation 37 |
| Volume prediction results | chen_feng_palomar_2016, Section 4.2, Table 3 |
| VWAP formulas and tracking error | chen_feng_palomar_2016, Section 4.3, Equations 39-42 |
| VWAP results | chen_feng_palomar_2016, Section 4.3, Table 4 |
| Log-normality justification | chen_feng_palomar_2016, Section 2, Figure 1 |
| Ticker universe | chen_feng_palomar_2016, Section 4.1, Table 2 |
| EM initialization robustness | Researcher inference (reasonable defaults; paper shows convergence from arbitrary starting points) |
| Missing observation handling | Researcher inference (standard Kalman filter technique, not in paper) |
| Jensen's inequality bias | Researcher inference (well-known log-normal property, not discussed in paper) |
