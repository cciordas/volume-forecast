# Implementation Specification: Kalman Filter State-Space Model for Intraday Volume

## Overview

A linear Gaussian state-space model that forecasts intraday trading volume by decomposing log-volume into three additive components: a daily average level, an intraday periodic (seasonal) pattern, and an intraday dynamic component. The log transformation converts the multiplicative volume structure into a tractable linear additive form, enabling exact Kalman filter recursions for prediction and closed-form EM updates for parameter estimation. A robust variant adds Lasso-penalized sparse noise detection for automatic outlier handling in live market data.

The model achieves 64% MAPE improvement over rolling means and 29% over the Component Multiplicative Error Model (CMEM), with 15% and 9% improvements respectively in VWAP tracking error (Paper: chen_feng_palomar_2016, Section 5 / Conclusion, page 11; raw numbers in Section 4.2, Table 3 and Section 4.3, Table 4).

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

(Paper: chen_feng_palomar_2016, Section 2, Equations 3-5)

### Pseudocode

#### Notation and Index Convention

The paper uses a unified time index tau = 1, 2, ..., N where N = T * I (T training days, I bins per day). The mapping is: for day t and bin i, tau = (t-1)*I + i. The bin index within a day is bin(tau) = ((tau-1) mod I) + 1. Day boundaries occur at tau = k*I for k = 1, 2, ...

The state vector is x_tau = [eta_tau, mu_tau]^T (dimension 2). The observation is scalar y_tau. The observation matrix is C = [1, 1] (1x2 row vector), so C * x_tau = eta_tau + mu_tau.

**Notation for 2x2 matrix elements:** Throughout this document, P^(i,j) denotes the (i,j) element of the 2x2 matrix P. For example, P_tau^(1,1) is the (1,1) element (top-left) of P_tau, corresponding to the eta-eta second moment. P_tau^(2,2) is the (2,2) element (bottom-right), corresponding to the mu-mu second moment. P_{tau,tau-1}^(i,j) denotes the (i,j) element of the cross-moment matrix P_{tau,tau-1}. These are NOT EM iteration superscripts.

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

j = 0
REPEAT until convergence (or max_iterations):
    j = j + 1
    
    // === E-STEP ===
    // Forward pass: Kalman filter (Step 3 applied to training data)
    // Produces: x_hat[tau|tau], Sigma[tau|tau] for tau=1..N
    //           x_hat[tau+1|tau], Sigma[tau+1|tau] for tau=1..N
    //           Innovation e[tau] and innovation variance S[tau] for tau=1..N
    RUN kalman_filter(y[1..N], theta^(j-1))
    
    // Backward pass: RTS smoother (Step 4)
    // Produces: x_hat[tau|N], Sigma[tau|N] for tau=1..N
    //           Sigma[tau,tau-1|N] for tau=2..N
    RUN rts_smoother(filter_output, theta^(j-1))
    
    // Compute sufficient statistics from smoother output:
    // x_hat_tau = x_hat[tau|N]                                    (Eq A.15/A.18)
    //
    // P_tau is the SECOND MOMENT (not the covariance):
    // P_tau = E[x_tau * x_tau^T | y_1..y_N, theta^(j)]
    //       = Sigma[tau|N] + x_hat[tau|N] * x_hat[tau|N]^T       (Eq A.16/A.19)
    // (P_tau = Cov + mean*mean^T; do NOT confuse with Sigma[tau|N])
    //
    // P_{tau,tau-1} is the cross second moment:
    // P_{tau,tau-1} = E[x_tau * x_{tau-1}^T | y_1..y_N, theta^(j)]
    //              = Sigma[tau,tau-1|N] + x_hat[tau|N] * x_hat[tau-1|N]^T
    //                                                              (Eq A.17/A.22)
    
    FOR tau = 1..N:
        P[tau] = Sigma[tau|N] + x_hat[tau|N] * x_hat[tau|N]^T
    FOR tau = 2..N:
        P[tau,tau-1] = Sigma[tau,tau-1|N] + x_hat[tau|N] * x_hat[tau-1|N]^T
    
    // === M-STEP === (all closed-form, Paper Appendix A.3, Eqs A.32-A.39)
    
    // Initial state mean and covariance:
    pi_1^(j+1) = x_hat[1|N]                                       // Eq A.32
    Sigma_1^(j+1) = P[1] - x_hat[1|N] * x_hat[1|N]^T             // Eq A.33
    
    // Daily AR coefficient (using only day-boundary transitions):
    // D = set of day-boundary indices = {I+1, 2I+1, 3I+1, ..., (T-1)*I+1}
    // These are the first bins of days 2, 3, ..., T.
    // |D| = T-1 (one transition per consecutive day pair).
    // k ranges from 1 to T-1, so tau = kI+1 ranges from I+1 to (T-1)*I+1.
    LET D = {tau : tau = k*I+1, for k = 1, 2, ..., T-1}
    
    a_eta^(j+1) = [sum_{tau in D} P[tau,tau-1]^(1,1)] / 
                  [sum_{tau in D} P[tau-1]^(1,1)]                  // Eq A.34
    
    // Intraday AR coefficient (using all consecutive transitions):
    a_mu^(j+1) = [sum_{tau=2}^{N} P[tau,tau-1]^(2,2)] / 
                 [sum_{tau=2}^{N} P[tau-1]^(2,2)]                 // Eq A.35
    
    // Process noise variance for daily component:
    // Denominator is T-1 = |D| (number of day-boundary transitions)
    [sigma_eta^2]^(j+1) = (1/(T-1)) * sum_{tau in D} {
        P[tau]^(1,1) + (a_eta^(j+1))^2 * P[tau-1]^(1,1) 
        - 2 * a_eta^(j+1) * P[tau,tau-1]^(1,1)
    }                                                              // Eq A.36
    
    // Process noise variance for dynamic component:
    // Denominator is N-1 (number of consecutive-bin transitions)
    [sigma_mu^2]^(j+1) = (1/(N-1)) * sum_{tau=2}^{N} {
        P[tau]^(2,2) + (a_mu^(j+1))^2 * P[tau-1]^(2,2) 
        - 2 * a_mu^(j+1) * P[tau,tau-1]^(2,2)
    }                                                              // Eq A.37
    
    // Observation noise variance:
    r^(j+1) = (1/N) * sum_{tau=1}^{N} [
        y_tau^2 + C * P[tau] * C^T - 2*y_tau * C * x_hat[tau|N]
        + (phi_{bin(tau)}^(j+1))^2 - 2*y_tau*phi_{bin(tau)}^(j+1) 
        + 2*phi_{bin(tau)}^(j+1) * C * x_hat[tau|N]
    ]                                                              // Eq A.38
    
    // Seasonality vector (simple average residual per bin):
    FOR i = 1..I:
        phi_i^(j+1) = (1/T) * sum_{t=1}^{T} (y_{t,i} - C * x_hat[(t-1)*I+i|N])
                                                                   // Eq A.39
    
    // NOTE on phi/r ordering: Eq A.38 references phi^(j+1), meaning phi
    // should be updated (Eq A.39) BEFORE r (Eq A.38) within each M-step.
    // The paper's derivation sets the gradient to zero simultaneously, but
    // in practice, compute phi first, then use it in the r update.
    // Researcher inference: this ordering follows from the simultaneous
    // optimality conditions in Appendix A.3; either order gives the same
    // fixed point, but computing phi first avoids needing a second pass.
    
    // Check convergence using observed-data log-likelihood:
    // Computed from Kalman filter innovations during the E-step forward pass.
    // Standard result for linear Gaussian state-space models (see
    // Shumway & Stoffer 1982, or any state-space textbook):
    LL^(j) = -(N/2) * log(2*pi) 
             - (1/2) * sum_{tau=1}^{N} [log(S[tau]) + e[tau]^2 / S[tau]]
    // where S[tau] = C * Sigma[tau|tau-1] * C^T + r is the innovation
    // variance and e[tau] = y[tau] - phi_{bin(tau)} - C * x_hat[tau|tau-1]
    // is the innovation, both computed during the Kalman filter forward pass.
    
    IF j > 1 AND |LL^(j) - LL^(j-1)| < epsilon:
        BREAK

OUTPUT theta^(j)
```

(Paper: chen_feng_palomar_2016, Section 2.3, Algorithms 2-3, Appendix A, Equations A.32-A.39)

#### Step 3: Kalman Filter (Prediction and Correction)

The filter iterates over the observation index tau. At each tau, we already have the predicted state x_hat[tau|tau-1] (from initialization at tau=1, or from the previous iteration's prediction step). We then (a) compute the forecast and innovation, (b) correct the state with the observation, and (c) predict the state for the next time step.

```
INPUT: parameters theta, observations y[tau] arriving sequentially
OUTPUT: filtered states x_hat[tau|tau], predicted states x_hat[tau+1|tau],
        one-step-ahead forecasts y_hat[tau], innovations e[tau],
        innovation variances S[tau]

DEFINE:
    C = [1, 1]  (observation matrix, 1x2 row vector)
    
    A_tau = [[a_tau_eta, 0],    (state transition matrix, 2x2)
             [0,         a_mu]]
    where a_tau_eta = a_eta if tau is a day boundary (tau mod I == 0),
                      1     otherwise
    
    Q_tau = [[sigma_tau_eta^2, 0          ],   (process noise covariance)
             [0,               sigma_mu^2 ]]
    where sigma_tau_eta^2 = sigma_eta^2 if tau is a day boundary (tau mod I == 0),
                            0           otherwise

INITIALIZE:
    x_hat[1|0] = pi_1                    // prior mean
    Sigma[1|0] = Sigma_1                 // prior covariance

FOR tau = 1, 2, ..., N:
    // --- FORECAST AND INNOVATION ---
    // One-step-ahead observation forecast (made BEFORE observing y_tau):
    y_hat[tau] = C * x_hat[tau|tau-1] + phi_{bin(tau)}             // Eq 9
    
    // Innovation (prediction error):
    e[tau] = y[tau] - y_hat[tau]
           = y[tau] - phi_{bin(tau)} - C * x_hat[tau|tau-1]
    
    // Innovation variance (scalar since observation is 1-D):
    S[tau] = C * Sigma[tau|tau-1] * C^T + r                       // scalar
    
    // --- CORRECTION STEP (incorporate observation y_tau) ---
    // Kalman gain (2x1 vector):
    K[tau] = Sigma[tau|tau-1] * C^T / S[tau]                      // Alg 1, line 4
    
    // Corrected state estimate (posterior):
    x_hat[tau|tau] = x_hat[tau|tau-1] + K[tau] * e[tau]           // Alg 1, line 5
    
    // Corrected covariance (posterior):
    Sigma[tau|tau] = Sigma[tau|tau-1] - K[tau] * S[tau] * K[tau]^T
                   = (I_2 - K[tau] * C) * Sigma[tau|tau-1]        // Alg 1, line 6
    
    // --- PREDICTION STEP (project state forward to tau+1) ---
    x_hat[tau+1|tau] = A_tau * x_hat[tau|tau]                     // Alg 1, line 2
    Sigma[tau+1|tau] = A_tau * Sigma[tau|tau] * A_tau^T + Q_tau   // Alg 1, line 3

    // Convert forecast to linear scale if needed:
    // volume_hat[tau] = exp(y_hat[tau])
```

**Missing observations:** If y[tau] is missing (zero-volume bin), skip the correction step entirely: set x_hat[tau|tau] = x_hat[tau|tau-1] and Sigma[tau|tau] = Sigma[tau|tau-1], then proceed to the prediction step. This propagates the state without incorporating the bad observation. (Researcher inference: standard Kalman filter technique for missing data, not discussed in the paper.)

**Static multi-step prediction** (forecasting all bins of day t+1 from end of day t):

At the end of day t (after processing tau = t*I), predict all I bins of day t+1. Since all h steps are intraday, the transition matrix is:

    A_intraday = [[1, 0], [0, a_mu]]

(Because eta is constant within a day: a_tau_eta = 1 for intraday steps.)

```
// Starting from x_hat[t*I | t*I] after processing last bin of day t:
// (Exception: the first step crosses the day boundary, so use A with a_eta)

// Step 1: cross day boundary (tau = t*I is a day boundary)
x_hat[t*I+1 | t*I] = A_boundary * x_hat[t*I | t*I]
    where A_boundary = [[a_eta, 0], [0, a_mu]]
Sigma[t*I+1 | t*I] = A_boundary * Sigma[t*I | t*I] * A_boundary^T + Q_boundary
    where Q_boundary = [[sigma_eta^2, 0], [0, sigma_mu^2]]

// Steps 2..I: intraday (no eta noise, eta unchanged)
FOR h = 2, 3, ..., I:
    x_hat[t*I+h | t*I] = A_intraday * x_hat[t*I+h-1 | t*I]
    Sigma[t*I+h | t*I] = A_intraday * Sigma[t*I+h-1 | t*I] * A_intraday^T + Q_intraday
        where Q_intraday = [[0, 0], [0, sigma_mu^2]]

// Simplification for the state mean (since A_intraday is diagonal):
// x_hat[t*I+h | t*I] = [a_eta * eta_hat, (a_mu)^h * mu_hat]^T
//   for h = 1, ..., I
// where eta_hat = x_hat[t*I | t*I]^(1) and mu_hat = x_hat[t*I | t*I]^(2)
//
// So the eta component stays at a_eta * eta_hat (one jump at boundary),
// and the mu component decays geometrically as (a_mu)^h.

// Log-volume forecast for each bin:
FOR h = 1, ..., I:
    y_hat[t*I+h | t*I] = C * x_hat[t*I+h | t*I] + phi_{h}       // Eq 9
    volume_hat[t*I+h] = exp(y_hat[t*I+h | t*I])
```

**Multi-step prediction covariance** (needed for forecast uncertainty and for the innovation variance at the first correction step when resuming online filtering):

```
// The multi-step covariance is computed recursively alongside the mean:
// Sigma[t*I+h | t*I] = A * Sigma[t*I+h-1 | t*I] * A^T + Q
// (already shown above in the multi-step loop)
//
// The observation forecast variance at step h is:
// S[t*I+h] = C * Sigma[t*I+h | t*I] * C^T + r
```

(Paper: chen_feng_palomar_2016, Section 2.2, Algorithm 1, Equation 9)

#### Step 4: RTS Smoother (for EM calibration only)

```
INPUT: Kalman filter output {x_hat[tau|tau], Sigma[tau|tau], 
        x_hat[tau+1|tau], Sigma[tau+1|tau], K[tau]} for tau=1..N
       Parameters theta
OUTPUT: Smoothed estimates {x_hat[tau|N], Sigma[tau|N]} for tau=1..N
        Cross-covariance {Sigma[tau,tau-1|N]} for tau=2..N

// Initialize from last filter step:
x_hat[N|N] = x_hat[N|N]  (from filter)
Sigma[N|N] = Sigma[N|N]  (from filter)

// Backward pass:
FOR tau = N-1, N-2, ..., 1:
    // Smoother gain:
    L[tau] = Sigma[tau|tau] * A_tau^T * inv(Sigma[tau+1|tau])      // Alg 2, line 2
    
    // Smoothed state:
    x_hat[tau|N] = x_hat[tau|tau] + L[tau] * (x_hat[tau+1|N] - x_hat[tau+1|tau])
                                                                   // Alg 2, line 3
    
    // Smoothed covariance:
    Sigma[tau|N] = Sigma[tau|tau] + L[tau] * (Sigma[tau+1|N] - Sigma[tau+1|tau]) * L[tau]^T
                                                                   // Alg 2, line 4

// Cross-covariance computation (needed for M-step):
// Initialize at tau = N:
Sigma[N,N-1|N] = (I_2 - K[N] * C) * A_{N-1} * Sigma[N-1|N-1]    // Eq A.21

FOR tau = N-1, N-2, ..., 2:
    Sigma[tau,tau-1|N] = Sigma[tau|tau] * L[tau-1]^T 
        + L[tau] * (Sigma[tau+1,tau|N] - A_tau * Sigma[tau|tau]) * L[tau-1]^T
                                                                   // Eq A.20
```

(Paper: chen_feng_palomar_2016, Section 2.3.1, Algorithm 2, Appendix A Equations A.18-A.22)

#### Step 5: Robust Kalman Filter (Lasso Extension)

The robust variant modifies the observation equation to include a sparse outlier term z_tau:

    y_tau = C * x_tau + phi_tau + v_tau + z_tau

where z_tau is nonzero only at outlier bins. The correction step is modified to solve a Lasso-penalized problem.

This step replaces the correction portion of Step 3. The prediction step, initialization, indexing convention, and loop structure are identical to Step 3. Only the correction step changes.

```
// Modified correction step (replaces the correction portion of Step 3):
// Uses the same index convention: at each tau, we have x_hat[tau|tau-1].

FOR tau = 1, 2, ..., N:
    // --- FORECAST AND INNOVATION (same as Step 3) ---
    y_hat[tau] = C * x_hat[tau|tau-1] + phi_{bin(tau)}
    e[tau] = y[tau] - y_hat[tau]
           = y[tau] - phi_{bin(tau)} - C * x_hat[tau|tau-1]        // Eq 31
    S[tau] = C * Sigma[tau|tau-1] * C^T + r
    K[tau] = Sigma[tau|tau-1] * C^T / S[tau]

    // --- ROBUST CORRECTION (replaces standard correction) ---
    // Innovation variance weight (scalar):
    W[tau] = 1 / (C * Sigma[tau|tau-1] * C^T + r)                 // = 1/S[tau]

    // Solve Lasso subproblem (Eq 30):
    // min_{z} (e[tau] - z)^2 * W[tau] + lambda * |z|
    // Solution is soft-thresholding (since all quantities are scalar):
    
    threshold = lambda / (2 * W[tau])
    //        = lambda * S[tau] / 2
    
    IF e[tau] > threshold:
        z_star[tau] = e[tau] - threshold                           // Eq 33
    ELIF e[tau] < -threshold:
        z_star[tau] = e[tau] + threshold                           // Eq 33
    ELSE:
        z_star[tau] = 0                                            // Eq 33
    
    // Cleaned innovation after outlier removal:
    e_clean[tau] = e[tau] - z_star[tau]                            // Eq 34
    // Equivalently: e_clean[tau] = sign(e[tau]) * min(|e[tau]|, threshold)
    //   if |e[tau]| > threshold: e_clean[tau] = sign(e[tau]) * threshold
    //   if |e[tau]| <= threshold: e_clean[tau] = e[tau]
    
    // Corrected state estimate using cleaned innovation:
    x_hat[tau|tau] = x_hat[tau|tau-1] + K[tau] * e_clean[tau]     // Eq 32
    
    // Covariance update (same as standard Kalman filter):
    Sigma[tau|tau] = (I_2 - K[tau] * C) * Sigma[tau|tau-1]
    
    // --- PREDICTION STEP (same as Step 3) ---
    x_hat[tau+1|tau] = A_tau * x_hat[tau|tau]
    Sigma[tau+1|tau] = A_tau * Sigma[tau|tau] * A_tau^T + Q_tau
```

**Robust EM modification** (for calibration with outlier-contaminated training data):

In the M-step, r and phi are updated using the inferred z_star values from the robust E-step:

```
// Observation noise variance (replaces Eq A.38):
r^(j+1) = (1/N) * sum_{tau=1}^{N} [
    y_tau^2 + C*P[tau]*C^T - 2*y_tau*C*x_hat[tau|N]
    + (phi_{bin(tau)}^(j+1))^2
    - 2*y_tau*phi_{bin(tau)}^(j+1) + 2*phi_{bin(tau)}^(j+1)*C*x_hat[tau|N]
    + (z_star[tau])^2 - 2*z_star[tau]*y_tau 
    + 2*z_star[tau]*phi_{bin(tau)}^(j+1) + 2*z_star[tau]*C*x_hat[tau|N]
]                                                                  // Eq 35

// Seasonality vector (replaces Eq A.39):
FOR i = 1..I:
    phi_i^(j+1) = (1/T) * sum_{t=1}^{T} (y_{t,i} - C*x_hat[(t-1)*I+i|N] - z_star[(t-1)*I+i])
                                                                   // Eq 36
```

All other M-step updates (pi_1, Sigma_1, a_eta, a_mu, sigma_eta^2, sigma_mu^2) remain the same as in the standard EM (Eqs A.32-A.37).

(Paper: chen_feng_palomar_2016, Section 3, Equations 25-36)

#### Step 6: VWAP Execution

**MAPE (volume prediction error metric):**

    MAPE = (1/M) * sum_{tau=1}^{M} |volume_tau - predicted_volume_tau| / volume_tau

where M is the total number of out-of-sample bins. The predicted volume is in linear scale: predicted_volume_tau = exp(y_hat_tau). volume_tau is the actual observed volume (also linear scale, after normalizing by shares outstanding).

(Paper: chen_feng_palomar_2016, Section 3.3, Equation 37)

**VWAP tracking error:**

    VWAP_TE = (1/D) * sum_{t=1}^{D} |VWAP_t - replicated_VWAP_t| / VWAP_t

where D is the number of out-of-sample days (250 in the paper), VWAP_t is the true volume-weighted average price on day t, and replicated_VWAP_t is the VWAP achieved using the model's predicted weights:

    VWAP_t = sum_{i=1}^{I} (volume_{t,i} * price_{t,i}) / sum_{i=1}^{I} volume_{t,i}
    replicated_VWAP_t = sum_{i=1}^{I} w_{t,i} * price_{t,i}

where price_{t,i} is the last recorded transaction price in bin i of day t. Tracking error is expressed in basis points (1 bps = 0.01%).

(Paper: chen_feng_palomar_2016, Section 4.3, Equations 39, 42)

**Static VWAP** (weights set before market open):

```
// At end of day t, forecast all bins of day t+1 using static prediction:
FOR i = 1..I:
    y_hat[t+1,i] = static_predict(i steps ahead from end of day t)
    // (using the multi-step prediction from Step 3)
    volume_hat[t+1,i] = exp(y_hat[t+1,i])

// Compute weights:
total = sum_{i=1}^{I} volume_hat[t+1,i]
FOR i = 1..I:
    w[t+1,i] = volume_hat[t+1,i] / total                          // Eq 40

// Execute: in bin i, trade w[t+1,i] fraction of total order
```

(Paper: chen_feng_palomar_2016, Section 4.3, Equation 40)

**Dynamic VWAP** (weights updated online after each bin):

The dynamic strategy is an online procedure. After each bin is observed, the Kalman filter state is updated, and the weight for the next bin is computed using the most recent forecasts. The key formula (Eq 41) allocates the remaining order proportion to the next bin based on its predicted volume share of all remaining predicted volume.

```
// Online procedure during day t+1:
// Before market open: initialize filter state from end of day t.
// Compute initial multi-step forecasts for all I bins from state at end of day t.

// Pre-compute initial forecasts (these serve as the starting denominator):
FOR i = 1..I:
    volume_forecast[i] = exp(static_predict(i steps ahead from end of day t))

// Process bins sequentially:
cumulative_weight = 0.0

FOR i = 1, 2, ..., I-1:
    // At this point, we have forecasts for bins i..I.
    // Compute weight for bin i:
    remaining_volume = sum_{j=i}^{I} volume_forecast[j]
    w[t+1,i] = (volume_forecast[i] / remaining_volume) * (1 - cumulative_weight)
                                                                   // Eq 41
    cumulative_weight = cumulative_weight + w[t+1,i]
    
    // Execute: trade w[t+1,i] fraction of total order in bin i
    
    // After bin i is observed:
    // (a) Update Kalman filter with observation y[t+1,i]
    //     (run one correction step + prediction step from Step 3)
    // (b) Re-forecast remaining bins i+1..I using multi-step prediction
    //     from the updated state:
    FOR j = i+1, ..., I:
        // (j-i)-step-ahead prediction from current state
        volume_forecast[j] = exp(multi_step_predict(j-i steps ahead))

// Last bin:
w[t+1,I] = 1 - cumulative_weight
// Execute: trade remaining fraction in last bin
```

**Note on the dynamic VWAP denominator:** At each bin i, the denominator sum_{j=i}^{I} volume_forecast[j] uses the most recently available forecasts. For bin i itself, this is a 1-step-ahead prediction made after observing bin i-1. For bins i+1..I, these are multi-step-ahead predictions (2-step, 3-step, ...) from the state after bin i-1. These multi-step forecasts are re-computed after each bin observation, so the denominator becomes more accurate as more data arrives. The paper describes volume_{t,i}^{(d)} as "dynamic volume predictions" (Section 4.3, page 10) without fully specifying whether the denominator uses re-computed multi-step forecasts or only 1-step forecasts; the formulation above is the natural interpretation that makes the formula well-defined at each step.

Researcher inference: The paper's description of dynamic prediction as "one-bin-ahead forecasting" (Section 4.2, page 8) refers to the fact that each bin's own prediction is a 1-step-ahead forecast. However, Eq 41's denominator requires volume predictions for all remaining bins, which necessarily involves multi-step forecasts for bins beyond i+1. The above formulation resolves this by re-forecasting all remaining bins at each step.

(Paper: chen_feng_palomar_2016, Section 4.3, Equations 40-42)

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
    |         +---> Forward Kalman Filter --> {x_hat[tau|tau], Sigma[tau|tau],
    |         |                                e[tau], S[tau]}
    |         |
    |         +---> Observed-data log-likelihood from innovations
    |         |
    |         +---> Backward RTS Smoother --> {x_hat[tau|N], Sigma[tau|N],
    |         |                                Sigma[tau,tau-1|N]}
    |         |
    |         +---> Sufficient statistics: P[tau], P[tau,tau-1]
    |         |
    |         +---> M-step closed-form updates --> theta^(j+1)
    |         |
    |         +---> (repeat until LL converges)
    |         |
    |         v
    |     Estimated parameters theta = {a_eta, a_mu, sigma_eta^2, sigma_mu^2, 
    |                                    r, phi[1..I], pi_1, Sigma_1}
    |
    +---> [Online Prediction] (using estimated theta)
              |
              +---> Kalman Filter: at each tau:
              |       Have x_hat[tau|tau-1], Sigma[tau|tau-1]     (2x1, 2x2)
              |       Forecast: y_hat[tau] = C*x_hat + phi       (scalar)
              |       Innovate: e[tau] = y[tau] - y_hat[tau]     (scalar)
              |       Correct:  x_hat[tau|tau]                   (2x1)
              |       Predict:  x_hat[tau+1|tau]                 (2x1)
              |
              +---> exp(y_hat) --> volume forecast (scalar per bin)
              |
              +---> [VWAP Weight Computation] --> execution schedule

Types/shapes at each stage:
- x_tau: float[2] (state vector)
- Sigma_tau: float[2][2] (state covariance)
- A_tau: float[2][2] (transition matrix, time-varying at day boundaries)
- Q_tau: float[2][2] (process noise covariance, time-varying at day boundaries)
- C: float[2] = [1, 1] (observation vector, constant)
- K_tau: float[2] (Kalman gain)
- y_tau: float (scalar observation)
- e_tau: float (scalar innovation)
- S_tau: float (scalar innovation variance)
- phi: float[I] (seasonality vector)
- r: float (observation noise variance, scalar)
- z_star_tau: float (scalar, robust model only; 0 for standard model)
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
| sigma_eta^2 | Process noise variance for daily component (at day boundaries only) | EM-estimated | Low -- determined by data; interacts with a_eta | (0, inf) |
| sigma_mu^2 | Process noise variance for dynamic component (every bin) | EM-estimated | Low -- determined by data; interacts with a_mu | (0, inf) |
| r | Observation noise variance | EM-estimated | Low-medium -- affects Kalman gain magnitude | (0, inf) |
| phi[1..I] | Intraday seasonality vector (I values) | EM-estimated as mean residual per bin (Eq A.39) | Low -- captures stable U-shaped pattern | (-inf, inf) in log space |
| pi_1 | Initial state mean [eta_0, mu_0]^T | EM-estimated (Eq A.32); initialize as [mean(y), 0]^T | Low -- EM converges regardless of initialization (Paper Section 2.3.3, Figure 4) | any real 2-vector |
| Sigma_1 | Initial state covariance (2x2) | EM-estimated (Eq A.33); initialize as diag(1,1) | Low -- same as above | positive definite 2x2 |
| lambda | Lasso regularization (robust model only) | Cross-validated; not reported in paper | High -- controls outlier sensitivity. Too small = no robustness; too large = over-smoothing. | (0, inf); must be strictly > 0 |
| N | Training window length (in days) | Cross-validated; paper uses Jan 2013 to validation start (~500 days, varies by exchange) | Medium -- too short = underfitting; too long = stale parameters | Typically 100-1000 days |
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
        - Warm-start: initialize EM with previous day's converged
          parameters (this typically reduces convergence to 2-5 iterations
          instead of 10-20).
        - Cold-start fallback: if warm-start fails to converge (LL
          decreases), restart from default initialization.
      - Use theta for Kalman filter predictions on day d.
   
   b. Optionally, re-estimate less frequently (e.g., weekly) if 
      computational budget is limited. The EM is fast enough for 
      daily re-estimation given the small state dimension.

3. EM convergence check:
   - Compute the observed-data log-likelihood at each iteration using
     the innovation-based formula:
     LL = -(N_bins/2)*log(2*pi) - (1/2)*sum[log(S_tau) + e_tau^2/S_tau]
   - Stop when |LL^(j+1) - LL^(j)| < epsilon (e.g., 1e-6).
   - Or after max_iterations (e.g., 50).
   - EM typically converges in fewer than 20 iterations.
   - LL must be non-decreasing. A decrease indicates a bug.
```

Researcher inference: The paper does not specify the re-estimation frequency or warm-start strategy. Daily re-estimation is feasible given the O(N) per-iteration cost of the EM (forward filter + backward smoother, each O(N) with 2x2 matrices). Warm-starting is a standard EM practice that the paper does not discuss. Weekly re-estimation is a reasonable economy if needed.

## Validation

### Expected Behavior

**Volume prediction accuracy (MAPE, dynamic prediction, out-of-sample):**

| Model | Average MAPE |
|-------|-------------|
| Robust Kalman Filter | 0.46 |
| Standard Kalman Filter | 0.47 |
| CMEM | 0.65 |
| Rolling Means | 1.28 |

(Paper: Table 3, average over 30 securities, Section 4.2, page 9)

**Volume prediction accuracy (MAPE, static prediction, out-of-sample):**

| Model | Average MAPE |
|-------|-------------|
| Robust Kalman Filter | 0.61 |
| Standard Kalman Filter | 0.62 |
| CMEM | 0.90 |
| Rolling Means | 1.28 |

(Paper: Table 3, Section 4.2, page 9)

**VWAP tracking error (basis points, dynamic strategy):**

| Model | Average TE (bps) |
|-------|-----------------|
| Robust Kalman Filter | 6.38 |
| Standard Kalman Filter | 6.39 |
| CMEM | 7.01 |
| Rolling Means | 7.48 |

(Paper: Table 4, Section 4.3, page 10)

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

3. **EM convergence monotonicity:** The observed-data log-likelihood (computed from innovations) must be non-decreasing at every EM iteration. A decrease indicates a bug in the E-step or M-step implementation.

4. **Kalman filter innovation check:** The standardized innovations e[tau] / sqrt(S[tau]) should be approximately N(0,1). Compute their mean (should be ~0) and variance (should be ~1). If the variance is significantly > 1, the model is underestimating uncertainty.

5. **Synthetic data test:** Generate synthetic data from the model with known parameters. Run EM and verify that estimated parameters converge to the true values. This is the test performed in Paper Section 2.3.3 / Figure 4.

6. **Rolling means baseline:** Implement a simple rolling means predictor (average of same bin across last N_rm days) and verify that the Kalman filter substantially outperforms it (expect ~50-60% MAPE reduction).

7. **State covariance positive-definiteness:** Sigma[tau|tau] and Sigma[tau+1|tau] must remain positive definite throughout the filter run. If they become negative or singular, there is a numerical issue.

8. **Robust filter outlier detection:** On clean data, z_star[tau] should be zero for the vast majority of bins (>95%). On artificially contaminated data (add large values to 10% of bins), z_star[tau] should be nonzero at approximately the contaminated bins.

### Edge Cases

1. **Zero-volume bins:** Log(0) is undefined. These must be excluded before processing. For the Kalman filter, skip the correction step when a bin is missing (set x_hat[tau|tau] = x_hat[tau|tau-1], Sigma[tau|tau] = Sigma[tau|tau-1], then proceed to prediction). This preserves the filter state without incorporating bad data.

   Researcher inference: The paper explicitly excludes zero-volume bins (Section 4.1) but does not describe the Kalman filter handling of missing observations. The standard approach in state-space models is to skip the correction step, which is well-established in the Kalman filter literature.

2. **Half-day sessions:** The paper excludes half-day sessions (Section 4.1, "excluding half-day sessions"). If included, the day would have fewer bins, creating a mismatch with the seasonality vector phi. Either exclude these days or handle by truncating phi to match the available bins.

3. **Day boundary transitions:** The transition matrix A_tau changes at day boundaries (tau mod I == 0). The implementation must correctly detect when a step crosses from one day to the next and switch from (a_tau_eta=1, sigma_tau_eta^2=0) to (a_tau_eta=a_eta, sigma_tau_eta^2=sigma_eta^2).

4. **Very small training windows:** With too few training days, the EM may not reliably estimate all parameters (especially sigma_eta^2 which only receives one data point per day boundary). Minimum recommended: ~50 trading days (roughly 2.5 months).

5. **Numerical stability of covariance updates:** The Joseph form of the covariance update Sigma[tau|tau] = (I - K*C) * Sigma[tau|tau-1] * (I - K*C)^T + K*r*K^T is more numerically stable than the simple form. Consider using it if positive-definiteness issues arise, although with 2x2 matrices this is unlikely.

6. **Lambda = 0 in robust model:** z_star[tau] = e[tau] for all bins (threshold is zero, so all innovations are fully attributed to outliers). The cleaned innovation e_clean[tau] = e[tau] - z_star[tau] = 0 always, so the filter ignores all observations and runs in pure prediction mode. This is pathological -- not equivalent to the standard Kalman filter. Lambda must be strictly positive.

7. **Lambda too large in robust model:** The threshold lambda/(2*W[tau]) = lambda*S[tau]/2 exceeds all innovations, so z_star[tau] = 0 always. The cleaned innovation equals the raw innovation, reducing to the standard Kalman filter. This is harmless but wastes the robust capability.

8. **Stocks changing exchange or shares outstanding:** Abrupt changes in shares outstanding (splits, offerings) cause level shifts in normalized volume. The daily component eta should absorb gradual shifts, but sudden large changes may trigger the robust filter's outlier detection or require re-calibration.

### Known Limitations

1. **Cannot handle zero-volume bins.** The log transformation is undefined for zero volume. This limits applicability to sufficiently liquid securities. The paper acknowledges this (Section 5 discussion, also noted in the summary).

2. **Linear Gaussian assumption.** The model assumes Gaussian noise in log space. While Q-Q plots support this for liquid securities (Figure 1), heavy-tailed distributions or regime changes may violate this assumption.

3. **No exogenous covariates.** The model uses only past volume observations. It does not incorporate price, volatility, spread, or event information. The paper identifies this as future work (Section 5).

4. **Static seasonality within training window.** The seasonality vector phi is estimated as a simple average and does not evolve over time. If the intraday pattern changes (e.g., due to market structure changes), the model adapts only through the rolling window re-estimation.

5. **No direct comparison with BDF or GAS-Dirichlet models.** The paper benchmarks against CMEM and rolling means only (Section 4). Performance relative to other model families is unknown.

6. **Single-stock model.** No cross-sectional information is used. Each security is modeled independently. This is in contrast to the BDF approach (Direction 2) which exploits cross-sectional commonality.

7. **Bias from log-space prediction.** Forecasting in log space and exponentiating introduces a Jensen's inequality bias: E[exp(y)] > exp(E[y]). The paper does not apply a bias correction. For volume share computation (ratios), this bias partially cancels, but for absolute volume forecasts, a correction factor of exp(S[tau]/2) could be applied, where S[tau] is the innovation variance (prediction uncertainty).

   Researcher inference: This is a well-known issue in log-normal forecasting but is not discussed in the paper. The bias is small when prediction variance S[tau] is small.

## Paper References

| Spec Section | Paper Source |
|-------------|-------------|
| Model decomposition (y = eta + phi + mu + v) | chen_feng_palomar_2016, Section 2, Equations 3-5 |
| State-space formulation (A_tau, C, Q_tau) | chen_feng_palomar_2016, Section 2, pages 3-4 |
| Kalman filter algorithm | chen_feng_palomar_2016, Section 2.2, Algorithm 1, page 4 |
| Dynamic vs static prediction | chen_feng_palomar_2016, Section 2.2, Equation 9, page 4 |
| RTS smoother | chen_feng_palomar_2016, Section 2.3.1, Algorithm 2, page 5 |
| EM algorithm structure | chen_feng_palomar_2016, Section 2.3.2, Algorithm 3, page 5 |
| Sufficient statistics definitions | chen_feng_palomar_2016, Appendix A.2, Equations A.15-A.22, page 14 |
| M-step: initial state (pi_1, Sigma_1) | chen_feng_palomar_2016, Appendix A.3, Equations A.32-A.33, page 15 |
| M-step: AR coefficients (a_eta, a_mu) | chen_feng_palomar_2016, Appendix A.3, Equations A.34-A.35, page 15 |
| M-step: process noise (sigma_eta^2, sigma_mu^2) | chen_feng_palomar_2016, Appendix A.3, Equations A.36-A.37, page 15 |
| M-step: observation noise (r) | chen_feng_palomar_2016, Appendix A.3, Equation A.38, page 15 |
| M-step: seasonality (phi) | chen_feng_palomar_2016, Appendix A.3, Equation A.39, page 15 |
| Cross-covariance initialization | chen_feng_palomar_2016, Appendix A, Equations A.20-A.21, page 14 |
| EM convergence demonstration | chen_feng_palomar_2016, Section 2.3.3, Figure 4, page 6 |
| Robust observation model | chen_feng_palomar_2016, Section 3.1, Equations 25-30, page 7 |
| Soft-thresholding solution | chen_feng_palomar_2016, Section 3.1, Equations 33-34, page 7 |
| Robust EM modifications | chen_feng_palomar_2016, Section 3.2, Equations 35-36, page 7 |
| Robustness simulations | chen_feng_palomar_2016, Section 3.3, Table 1, page 7 |
| MAPE definition | chen_feng_palomar_2016, Section 3.3, Equation 37, page 7 |
| Data setup and normalization | chen_feng_palomar_2016, Section 4.1, page 8 |
| Improvement percentages (64%, 29%, 15%, 9%) | chen_feng_palomar_2016, Section 5, page 11 (also abstract) |
| Volume prediction results | chen_feng_palomar_2016, Section 4.2, Table 3, page 9 |
| VWAP formulas (weights, tracking error) | chen_feng_palomar_2016, Section 4.3, Equations 39-42, pages 8-10 |
| VWAP results | chen_feng_palomar_2016, Section 4.3, Table 4, page 10 |
| Log-normality justification | chen_feng_palomar_2016, Section 2, Figure 1, page 3 |
| Ticker universe | chen_feng_palomar_2016, Section 4.1, Table 2, page 8 |
| Observed-data log-likelihood from innovations | Standard Kalman filter result (Shumway & Stoffer 1982); not in paper |
| EM initialization robustness | Researcher inference (reasonable defaults; paper shows convergence from arbitrary starting points in Figure 4) |
| Missing observation handling | Researcher inference (standard Kalman filter technique, not in paper) |
| Jensen's inequality bias | Researcher inference (well-known log-normal property, not discussed in paper) |
| EM warm-start strategy | Researcher inference (standard EM practice, not discussed in paper) |
| Dynamic VWAP denominator re-forecasting | Researcher inference (natural interpretation of Eq 41; paper does not fully specify whether denominator uses re-computed multi-step forecasts) |
