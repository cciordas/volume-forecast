# Implementation Specification: Kalman Filter State-Space Model for Intraday Volume

## Overview

This model forecasts intraday trading volume by decomposing log-volume into three
additive components -- a daily average level, an intraday seasonal pattern, and an
intraday dynamic residual -- within a linear Gaussian state-space framework. The Kalman
filter provides optimal recursive prediction and correction, while the EM algorithm
estimates all parameters in closed form. A robust variant adds Lasso-penalized sparse
noise detection for automatic outlier handling. The approach converts the multiplicative
volume decomposition of Brownlees et al. (2011) into a tractable linear model via
logarithmic transformation.

Source: Chen, Feng, and Palomar (2016), "Forecasting Intraday Trading Volume: A Kalman
Filter Approach."

## Algorithm

### Model Description

The model operates on log-transformed intraday volume. Raw volume in each bin is first
normalized by shares outstanding (to handle splits and scale changes), then the natural
logarithm is taken. This converts the inherently multiplicative, positive-valued volume
process into an additive, real-valued one where Gaussian assumptions are defensible.

The log-volume observation y_{t,i} for day t, bin i decomposes as:

    y_{t,i} = eta_t + phi_i + mu_{t,i} + v_{t,i}

where:
- eta_t: daily average component (log scale). Captures slow day-to-day level shifts.
  Constant within a day, evolves as AR(1) across day boundaries.
- phi_i: intraday periodic component. The U-shaped seasonal pattern. One value per bin
  index i, shared across all days. Estimated as simple averages, not Fourier series.
- mu_{t,i}: intraday dynamic component. Captures short-lived departures from the seasonal
  pattern. Evolves as AR(1) bin-to-bin, continuously within and across days.
- v_{t,i}: observation noise, i.i.d. N(0, r).

The model is linear-Gaussian, so the Kalman filter gives statistically optimal state
estimates. The EM algorithm provides closed-form parameter updates, avoiding the
numerical difficulties of GMM-based alternatives.

Reference: Paper Section 2, Equations (3)-(5), Figure 2.

### Pseudocode

The implementation has three main components: Kalman filter (prediction), Kalman smoother
(calibration support), and EM algorithm (parameter estimation). Plus an optional robust
extension.

#### Notation and Index Conventions

The paper uses a unified time index tau = 1, 2, ..., N where N = T * I (T training days,
I bins per day). The mapping from (day t, bin i) to tau is:

    tau = (t - 1) * I + i

Day boundaries occur at tau = k*I for k = 1, 2, ... A day boundary is detected when
tau mod I == 0 (end of day) or equivalently tau mod I == 1 (start of day). The transition
matrix changes at these boundaries.

Reference: Paper Section 2, paragraph below Equation (5).

#### Step 1: Data Preprocessing

```
Input: raw_volume[t, i] for t = 1..T_total, i = 1..I
Input: shares_outstanding[t] for t = 1..T_total

1. Compute normalized volume:
   vol_norm[t, i] = raw_volume[t, i] / shares_outstanding[t]

2. Check for zero-volume bins:
   If vol_norm[t, i] == 0, mark bin as missing (cannot take log of zero).
   Options: (a) exclude bin from estimation, (b) impute with small positive value.
   The paper assumes all volumes are non-zero and excludes zero bins.

3. Take natural logarithm:
   y[t, i] = ln(vol_norm[t, i])

4. Reshape into unified time series:
   y[tau] for tau = 1..N, where tau = (t-1)*I + i
```

Reference: Paper Section 2, Equation (1) for normalization; Section 4.1 for data setup.
Zero-volume exclusion: Paper Section 2 assumption (implicit -- log(0) undefined).

#### Step 2: Kalman Filter (Algorithm 1 -- Prediction)

The hidden state is x_tau = [eta_tau, mu_tau]^T (dimension 2).

```
Input: parameters theta = {a_eta, a_mu, sigma_eta^2, sigma_mu^2, r, phi[1..I], pi_1, Sigma_1}
Input: observations y[1..N]

Initialize:
  x_hat[1|0] = pi_1          # 2x1 vector: [eta_0, mu_0]^T
  Sigma[1|0] = Sigma_1       # 2x2 matrix

For tau = 1, 2, ..., N:

  # --- Prediction step ---
  # Build time-varying transition matrix A_tau
  If tau corresponds to first bin of a day (tau mod I == 1, or tau == 1):
    # Day boundary transition (from previous day's last bin to this day's first bin)
    A_tau_prev = [[a_eta, 0], [0, a_mu]]
    Q_tau_prev = [[sigma_eta^2, 0], [0, sigma_mu^2]]
  Else:
    # Within-day transition (eta is constant, only mu evolves)
    A_tau_prev = [[1, 0], [0, a_mu]]
    Q_tau_prev = [[0, 0], [0, sigma_mu^2]]

  # Note: A_tau_prev and Q_tau_prev govern the transition FROM tau-1 TO tau.
  # For tau=1, use pi_1 and Sigma_1 directly (no transition needed).
  # For tau >= 2:
  x_hat[tau|tau-1] = A_tau_prev * x_hat[tau-1|tau-1]
  Sigma[tau|tau-1] = A_tau_prev * Sigma[tau-1|tau-1] * A_tau_prev^T + Q_tau_prev

  # --- Observation model ---
  C = [1, 1]                  # 1x2 row vector
  phi_tau = phi[((tau-1) mod I) + 1]   # seasonal component for this bin index

  # --- Correction step ---
  # Innovation
  e_tau = y[tau] - C * x_hat[tau|tau-1] - phi_tau

  # Innovation variance (scalar)
  S_tau = C * Sigma[tau|tau-1] * C^T + r

  # Kalman gain (2x1 vector)
  K_tau = Sigma[tau|tau-1] * C^T / S_tau

  # Corrected state estimate
  x_hat[tau|tau] = x_hat[tau|tau-1] + K_tau * e_tau

  # Corrected covariance
  Sigma[tau|tau] = Sigma[tau|tau-1] - K_tau * S_tau * K_tau^T
  # Equivalently: Sigma[tau|tau] = (I_2 - K_tau * C) * Sigma[tau|tau-1]

  # --- Forecast output ---
  y_hat[tau] = C * x_hat[tau|tau-1] + phi_tau   # one-step-ahead forecast
```

For **dynamic prediction** (one-bin-ahead): run both predict and correct steps at each
tau, producing y_hat[tau] before observing y[tau].

For **static prediction** (all bins of next day forecast at once): after the last bin of
day t, skip correction steps and only run prediction steps for all I bins of day t+1.
Multi-step prediction:

```
For h = 1, 2, ..., I:
  x_hat[tau+h|tau] = A_{tau+h} * x_hat[tau+h-1|tau]
  y_hat[tau+h|tau] = C * x_hat[tau+h|tau] + phi_{tau+h}
```

where the transition matrices alternate between day-boundary and within-day forms
as appropriate.

Reference: Paper Section 2.2, Algorithm 1 (page 4), Equation (9) for multi-step.

#### Step 3: Kalman Smoother (Algorithm 2 -- for EM Calibration)

The Rauch-Tung-Striebel backward smoother, run after the forward filter pass over the
training data.

```
Input: filtered estimates x_hat[tau|tau], Sigma[tau|tau] for tau = 1..N
Input: predicted estimates x_hat[tau+1|tau], Sigma[tau+1|tau] for tau = 1..N-1
Input: transition matrices A_{tau+1} for tau = 1..N-1

# Initialize with final filtered estimates
x_hat[N|N] and Sigma[N|N] are already computed from the forward pass.

For tau = N-1, N-2, ..., 1:

  # Smoother gain
  L_tau = Sigma[tau|tau] * A_{tau+1}^T * inv(Sigma[tau+1|tau])

  # Smoothed state estimate
  x_hat[tau|N] = x_hat[tau|tau] + L_tau * (x_hat[tau+1|N] - x_hat[tau+1|tau])

  # Smoothed covariance
  Sigma[tau|N] = Sigma[tau|tau] + L_tau * (Sigma[tau+1|N] - Sigma[tau+1|tau]) * L_tau^T

# Also compute the cross-covariance needed for EM:
# Initialize:
Sigma[N,N-1|N] = (I_2 - K_N * C) * A_N * Sigma[N-1|N-1]

For tau = N-1, N-2, ..., 2:
  Sigma[tau,tau-1|N] = Sigma[tau|tau] * L_{tau-1}^T
                       + L_tau * (Sigma[tau+1,tau|N] - A_{tau+1} * Sigma[tau|tau]) * L_{tau-1}^T
```

Reference: Paper Section 2.3.1, Algorithm 2 (page 5), Equations (10)-(11).
Cross-covariance initialization: Paper Appendix A.2, Equation (A.21).

#### Step 4: EM Algorithm (Algorithm 3 -- Parameter Estimation)

```
Input: training observations y[1..N] (N = T * I)
Input: initial parameter guesses theta^(0)

Set j = 0
Repeat until convergence:

  # === E-step ===
  Run Kalman filter (Algorithm 1) with theta^(j) on y[1..N]
    -> produces x_hat[tau|tau], Sigma[tau|tau], x_hat[tau+1|tau], Sigma[tau+1|tau]
  Run Kalman smoother (Algorithm 2)
    -> produces x_hat[tau|N], Sigma[tau|N], Sigma[tau,tau-1|N] for all tau

  # Compute sufficient statistics:
  # Define P matrices (these are the key quantities):
  P_tau = Sigma[tau|N] + x_hat[tau|N] * x_hat[tau|N]^T         # 2x2
  P_{tau,tau-1} = Sigma[tau,tau-1|N] + x_hat[tau|N] * x_hat[tau-1|N]^T  # 2x2

  # Notation for elements:
  # P_tau^{(k,l)} = element (k,l) of P_tau, where k,l in {1,2}
  # P_tau^{(1,1)} = E[eta_tau^2 | data]
  # P_tau^{(2,2)} = E[mu_tau^2 | data]
  # P_{tau,tau-1}^{(1,1)} = E[eta_tau * eta_{tau-1} | data]
  # P_{tau,tau-1}^{(2,2)} = E[mu_tau * mu_{tau-1} | data]

  # === M-step (all closed-form updates) ===

  # Initial state mean (Eq A.32):
  pi_1^{(j+1)} = x_hat[1|N]

  # Initial state covariance (Eq A.33):
  Sigma_1^{(j+1)} = P_1 - x_hat[1|N] * x_hat[1|N]^T

  # Daily AR coefficient (Eq A.34):
  # Sum only over day-boundary transitions: D = {tau : tau = k*I+1, k=1,2,...}
  a_eta^{(j+1)} = [sum_{tau in D} P_{tau,tau-1}^{(1,1)}]
                 / [sum_{tau in D} P_{tau-1}^{(1,1)}]

  # Intraday AR coefficient (Eq A.35):
  # Sum over ALL transitions tau = 2..N
  a_mu^{(j+1)} = [sum_{tau=2}^{N} P_{tau,tau-1}^{(2,2)}]
                / [sum_{tau=2}^{N} P_{tau-1}^{(2,2)}]

  # Daily process noise variance (Eq A.36):
  # |D| = T-1 (number of day-boundary transitions)
  [sigma_eta^2]^{(j+1)} = (1/(T-1)) * sum_{tau in D} {
      P_tau^{(1,1)} + (a_eta^{(j+1)})^2 * P_{tau-1}^{(1,1)}
      - 2 * a_eta^{(j+1)} * P_{tau,tau-1}^{(1,1)}
  }

  # Intraday process noise variance (Eq A.37):
  [sigma_mu^2]^{(j+1)} = (1/(N-1)) * sum_{tau=2}^{N} {
      P_tau^{(2,2)} + (a_mu^{(j+1)})^2 * P_{tau-1}^{(2,2)}
      - 2 * a_mu^{(j+1)} * P_{tau,tau-1}^{(2,2)}
  }

  # Observation noise variance (Eq A.38):
  r^{(j+1)} = (1/N) * sum_{tau=1}^{N} {
      y[tau]^2 + C * P_tau * C^T - 2 * y[tau] * C * x_hat[tau|N]
      + (phi_tau^{(j+1)})^2
      - 2 * y[tau] * phi_tau^{(j+1)} + 2 * phi_tau^{(j+1)} * C * x_hat[tau|N]
  }

  # Seasonality vector (Eq A.39 / Eq 24):
  For i = 1, 2, ..., I:
    phi_i^{(j+1)} = (1/T) * sum_{t=1}^{T} (y[t,i] - C * x_hat[(t-1)*I+i | N])

  j = j + 1

Check convergence:
  If |log_likelihood^{(j)} - log_likelihood^{(j-1)}| / |log_likelihood^{(j-1)}| < epsilon:
    Stop
```

Note on the M-step update order: phi depends on x_hat (from E-step only), and r depends
on the newly updated phi^{(j+1)}. So within each M-step iteration, update phi first,
then r. All other updates (a_eta, a_mu, sigma_eta^2, sigma_mu^2, pi_1, Sigma_1) depend
only on E-step quantities and can be computed in any order.

Reference: Paper Section 2.3.2, Algorithm 3 (page 6), Appendix A.3 Equations (A.32)-(A.39).
Note on update ordering: Researcher inference -- the paper lists updates without specifying
order, but r depends on phi^{(j+1)} per Eq (A.38)/(23), so phi must be computed first.

#### Step 5: Robust Kalman Filter Extension (Optional)

The robust variant adds a sparse outlier term z_tau to the observation equation:

    y_tau = C * x_tau + phi_tau + v_tau + z_tau

where z_tau is zero most of the time but can take large values when outliers occur.

```
Modification to Step 2 (Kalman filter correction step):

  # Compute innovation residual
  e_tau = y[tau] - phi_tau - C * x_hat[tau|tau-1]

  # Compute innovation precision (scalar)
  W_tau = 1 / (C * Sigma[tau|tau-1] * C^T + r)

  # Solve Lasso-penalized problem (Eq 30):
  # min_{z_tau} (e_tau - z_tau)^2 * W_tau + lambda * |z_tau|
  #
  # Closed-form soft-thresholding solution (Eq 33):
  threshold = lambda / (2 * W_tau)

  If e_tau > threshold:
    z_tau_star = e_tau - threshold
  Elif e_tau < -threshold:
    z_tau_star = e_tau + threshold
  Else:
    z_tau_star = 0

  # Modified correction (Eq 31-32):
  e_tau_clean = e_tau - z_tau_star    # innovation with outlier removed
  x_hat[tau|tau] = x_hat[tau|tau-1] + K_tau * e_tau_clean

  # Covariance update unchanged (same K_tau as standard filter)
```

Reference: Paper Section 3.1, Equations (29)-(34).

#### Step 5b: Robust EM Modifications

When the robust filter is used, the EM M-step updates for r and phi are modified to
account for the inferred outlier values z_tau*:

```
  # Modified observation noise variance (Eq 35):
  r^{(j+1)} = (1/N) * sum_{tau=1}^{N} {
      y[tau]^2 + C * P_tau * C^T - 2 * y[tau] * C * x_hat[tau|N]
      + (phi_tau^{(j+1)})^2 - 2 * (z_tau*)^2 * y[tau]
      + 2 * z_tau* * C * x_hat[tau|N]
      + 2 * z_tau* * phi_tau^{(j+1)}
  }

  # Modified seasonality (Eq 36):
  For i = 1, 2, ..., I:
    phi_i^{(j+1)} = (1/T) * sum_{t=1}^{T} (y[t,i] - C * x_hat[(t-1)*I+i | N] - z_{(t-1)*I+i}*)
```

Reference: Paper Section 3.2, Equations (35)-(36).

#### Step 6: VWAP Execution Strategies

Two VWAP replication strategies use the volume forecasts:

```
# Static VWAP (all weights set before market open):
For day t+1, using static volume predictions vol_hat[t+1, i] for i = 1..I:
  Convert from log: vol_hat_linear[t+1, i] = exp(y_hat[t+1, i])
  w_static[i] = vol_hat_linear[t+1, i] / sum_j(vol_hat_linear[t+1, j])

# Dynamic VWAP (weights revised at each bin):
For bin i = 1, 2, ..., I-1:
  # Remaining weight to distribute = 1 - sum_{j=1}^{i-1} w[j] (already executed)
  # Use dynamic one-bin-ahead forecast for remaining bins
  remaining_weight = 1 - sum_{j=1}^{i-1} w[j]
  vol_hat_linear[t, j] = exp(y_hat_dynamic[t, j]) for j = i..I
  w_dynamic[i] = remaining_weight * vol_hat_linear[t, i] / sum_{j=i}^{I} vol_hat_linear[t, j]

For bin I (last bin):
  w_dynamic[I] = 1 - sum_{j=1}^{I-1} w_dynamic[j]
```

Reference: Paper Section 4.3, Equations (39)-(41).

### Data Flow

```
Raw volume data [T_total days x I bins, positive integers]
  |
  v
Normalize by shares outstanding [T_total x I, positive reals]
  |
  v
Log transform [T_total x I, reals]  (zero-volume bins excluded)
  |
  v
Split into training (T days) and out-of-sample (D days)
  |
  +---> Training data [N = T*I observations]
  |       |
  |       v
  |     EM Algorithm (iterate until convergence):
  |       |
  |       +---> Forward Kalman Filter [N steps]
  |       |       State: x_hat[tau|tau] (2x1), Sigma[tau|tau] (2x2)
  |       |       Output: filtered states, predicted states
  |       |
  |       +---> Backward RTS Smoother [N steps, reverse]
  |       |       Output: smoothed states x_hat[tau|N], Sigma[tau|N]
  |       |       Output: cross-covariances Sigma[tau,tau-1|N]
  |       |
  |       +---> M-step: closed-form parameter updates
  |       |       Output: theta^{(j+1)} = {a_eta, a_mu, sigma_eta^2, sigma_mu^2,
  |       |                                 r, phi[1..I], pi_1, Sigma_1}
  |       |
  |       +---> Convergence check on log-likelihood
  |
  +---> Out-of-sample prediction [D days]:
          |
          v
        Kalman Filter with learned theta:
          - Dynamic mode: predict + correct each bin
          - Static mode: predict all bins of next day, no correction
          |
          v
        Log-volume forecasts y_hat[tau]
          |
          v
        Exponentiate: vol_hat[tau] = exp(y_hat[tau]) * shares_outstanding
          |
          v
        VWAP weights (static or dynamic strategy)
```

Types and shapes at each step:
- x_tau: float64[2] (state vector)
- Sigma_tau: float64[2, 2] (state covariance)
- A_tau: float64[2, 2] (transition matrix -- two variants, cached)
- Q_tau: float64[2, 2] (process noise covariance -- two variants, cached)
- C: float64[1, 2] = [[1, 1]] (observation matrix, constant)
- K_tau: float64[2, 1] (Kalman gain)
- S_tau: float64 scalar (innovation variance)
- phi: float64[I] (seasonality vector)
- y[tau]: float64 scalar (observation)

### Variants

This specification implements the **robust Kalman filter with EM calibration** as described
in Chen, Feng, and Palomar (2016). This is the most complete variant from the paper,
encompassing:

1. **Standard Kalman filter** (Section 2): the base model without outlier handling.
2. **Robust Kalman filter** (Section 3): adds Lasso-penalized sparse noise detection.

The robust variant is chosen because:
- It subsumes the standard model (setting lambda = infinity recovers the standard filter).
- It is the paper's recommended variant for production use with real-time (non-curated) data.
- It achieves the best empirical results (MAPE 0.46 vs 0.47 standard, both far better
  than CMEM at 0.65).
- The computational overhead of soft-thresholding is negligible.

Both variants should be implemented, with the robust extension as an optional layer
controlled by the lambda parameter. Setting lambda to a very large value (or infinity)
effectively disables the robust correction.

Reference: Paper Section 3, Section 4.2 (Table 3), Section 5 (Conclusion).

## Parameters

| Parameter | Description | Recommended Value | Sensitivity | Range |
|-----------|-------------|-------------------|-------------|-------|
| a_eta | AR(1) coefficient for daily component across day boundaries | Data-driven via EM. Typically close to 1 (high persistence). Synthetic experiment convergence shown in Fig 4(a). | Medium -- controls how quickly the daily level adapts. Values near 1 mean slow adaptation; much below 0.9 means rapid forgetting. | (0, 1) for stationarity |
| a_mu | AR(1) coefficient for intraday dynamic component | Data-driven via EM. Synthetic experiment convergence shown in Fig 4(b). | Medium -- governs mean-reversion speed of intraday dynamics. Higher values mean more persistent deviations. | (0, 1) for stationarity |
| sigma_eta^2 | Process noise variance for daily component (at day boundaries only) | Data-driven via EM. Convergence shown in Fig 4(d). | Low-medium -- controls day-to-day volatility of the daily level. | (0, infinity) |
| sigma_mu^2 | Process noise variance for intraday dynamic component | Data-driven via EM. Convergence shown in Fig 4(e). | Low-medium -- controls bin-to-bin volatility of the dynamic component. | (0, infinity) |
| r | Observation noise variance | Data-driven via EM. Convergence shown in Fig 4(c). | Medium -- balances trust in observations vs state predictions. Too small: filter overfits to noise. Too large: filter ignores observations. | (0, infinity) |
| phi[1..I] | Intraday seasonality vector (one value per bin) | Data-driven via EM. Captures the U-shaped intraday pattern. Convergence shown in Fig 4(f). | High -- the seasonality is the dominant predictable component. Errors here propagate to all forecasts. | (-infinity, infinity) in log space |
| pi_1 | Initial state mean [eta_0, mu_0]^T | Data-driven via EM (Eq A.32). Can initialize to [mean(y), 0]^T for first EM iteration. | Low -- EM converges regardless of initialization (Fig 4). | R^2 |
| Sigma_1 | Initial state covariance (2x2 matrix) | Data-driven via EM (Eq A.33). Can initialize to identity or diag([var(y), var(y)]). | Low -- EM converges regardless of initialization. | Positive definite 2x2 |
| lambda | Lasso regularization parameter (robust model only) | Selected by cross-validation. Not reported in the paper. | High for robust model -- controls the outlier detection threshold. Too small: normal observations treated as outliers. Too large: outliers not detected (reverts to standard KF). | (0, infinity) |
| N | Training window length (number of bins = T_train * I) | Selected by cross-validation on held-out period (Jan-May 2015 in the paper). Not reported. | High -- too short: insufficient data for parameter estimation. Too long: model cannot adapt to regime changes. | Typically 6-24 months of data |
| I | Number of intraday bins per day | Exchange-dependent: 26 for 6.5-hour sessions at 15-min bins (NYSE), varies for other exchanges. | N/A (data characteristic, not tunable) | Determined by exchange hours and bin width |
| epsilon | EM convergence threshold | 1e-6 (Researcher inference -- standard EM practice) | Low -- the EM converges quickly; any reasonable small value works. | (0, 1e-3) |

### Initialization

EM initialization (for the first EM iteration, before any E-step has run):

```
1. Compute mean and variance of all training log-volumes:
   y_bar = mean(y[1..N])
   y_var = var(y[1..N])

2. Initialize parameters:
   a_eta^(0) = 0.95        # reasonable AR persistence
   a_mu^(0) = 0.5          # moderate intraday persistence
   sigma_eta^2^(0) = 0.1 * y_var
   sigma_mu^2^(0) = 0.1 * y_var
   r^(0) = 0.5 * y_var
   phi^(0)[i] = (1/T) * sum_{t=1}^{T} y[t,i] - y_bar   for each i
   pi_1^(0) = [y_bar, 0]^T
   Sigma_1^(0) = diag([y_var, y_var])
```

The paper demonstrates (Section 2.3.3, Figure 4) that EM converges to the same parameter
values regardless of initial values, so exact initialization is not critical. The values
above are Researcher inference based on standard EM practice.

Reference: Paper Section 2.3.3 for convergence insensitivity. Initialization values:
Researcher inference.

### Calibration

The calibration procedure uses the EM algorithm with a rolling training window:

```
1. Select training window length N (= T_train * I bins) and regularization
   parameter lambda via cross-validation:
   - Define a validation period (e.g., 5 months before out-of-sample start)
   - Grid search over candidate N values (e.g., 6, 9, 12, 18, 24 months)
   - For each N, for each validation day:
     a. Train EM on the preceding N bins
     b. Predict next day (dynamic mode)
     c. Compute MAPE on validation day
   - For the robust model, also grid search over lambda values
   - Select (N, lambda) pair minimizing average validation MAPE

2. Rolling window re-estimation:
   For each out-of-sample day d:
   a. Take the N most recent bins as training data
   b. Run EM to convergence (typically 5-20 iterations)
   c. Use converged parameters for Kalman filter prediction on day d
   d. Slide window forward by I bins (one day)

3. Cross-validation period:
   The paper uses January 2015 to May 2015 as cross-validation set,
   with June 2015 to June 2016 as out-of-sample (250 days).
```

Reference: Paper Section 4.1 for cross-validation setup; Section 4.2 for rolling window
scheme ("standard rolling window forecasting scheme").

## Validation

### Expected Behavior

On typical liquid equity or ETF data at 15-minute resolution:

- **Dynamic prediction MAPE** should be approximately 0.20-0.50 across securities,
  with an average around 0.46 (robust) or 0.47 (standard). This means the average
  absolute percentage error between predicted and actual normalized volume is about 46%.
  Reference: Paper Table 3, "Average" row.

- **Static prediction MAPE** should be approximately 0.35-0.60, averaging around 0.61
  (robust). Static is always worse than dynamic because it cannot incorporate intraday
  information. Reference: Paper Table 3.

- **VWAP tracking error** (dynamic strategy) should average approximately 6.38 basis
  points, with individual securities ranging from about 2.6 to 18 bps.
  Reference: Paper Table 4, "Average" row.

- The **U-shaped seasonality** (phi vector) should show high values at the first and
  last bins of the day and lower values in the middle, reflecting the well-known
  intraday volume pattern.

- **EM convergence** should occur within 5-20 iterations from any reasonable initialization.
  Reference: Paper Section 2.3.3, Figure 4.

- The daily component eta should be highly persistent (a_eta close to 1), while the
  intraday dynamic mu should be less persistent (a_mu varies more widely).

### Sanity Checks

1. **Synthetic data test**: Generate synthetic data from the model with known parameters
   (e.g., a_eta=0.98, a_mu=0.6, sigma_eta=0.05, sigma_mu=0.1, r=0.2, phi = smooth
   U-shape). Run EM on the synthetic data. Verify that recovered parameters match the
   generating parameters within reasonable tolerance (e.g., <5% relative error for
   AR coefficients after convergence). Reference: Paper Section 2.3.3, Figure 4 shows
   this exact experiment.

2. **Seasonality recovery**: After EM estimation, plot phi[1..I]. It should resemble a
   U-shape (high at open and close, low mid-day) for standard equity markets.
   Reference: Paper Figure 4(f) shows converged phi values.

3. **Filter consistency**: The innovation sequence e_tau = y[tau] - y_hat[tau] should be
   approximately white noise with variance approximately equal to the innovation variance
   S_tau. Compute autocorrelation of innovations -- it should be insignificant at all lags.
   Researcher inference based on standard Kalman filter diagnostics.

4. **Robust filter outlier detection**: On clean data, the robust filter should produce
   z_tau* = 0 for nearly all bins. Artificially inject a few large outliers (e.g., add
   +3 to log-volume for 5% of bins); the robust filter should correctly identify and
   neutralize most of them. Reference: Paper Section 3.3, Table 1 (contamination
   experiment on SPY, DIA, IBM).

5. **Improvement over rolling mean**: The Kalman filter MAPE should be substantially
   lower than a simple rolling mean baseline (average of same bin over the last T_train
   days). The paper reports 64% improvement. Even a modest implementation should show
   >30% improvement. Reference: Paper Section 4.2.

6. **Log-likelihood monotonicity**: The EM log-likelihood should be non-decreasing across
   iterations. If it decreases, there is a bug in the E-step or M-step implementation.
   Researcher inference based on EM theory (Jensen's inequality guarantee).

### Edge Cases

1. **Zero-volume bins**: Log(0) is undefined. These bins must be detected and handled
   before the log transform. Options:
   - Skip the bin entirely (treat as missing observation in the Kalman filter by running
     only the prediction step without correction).
   - Replace with a small positive value (e.g., 1 share / shares_outstanding).
   The paper does not address this -- it assumes all volumes are positive.
   Reference: Paper Section 2 assumption. Handling: Researcher inference.

2. **Half-day sessions**: Days with fewer than I bins (e.g., early market closures).
   The paper excludes these entirely. Alternatively, handle by adjusting the transition
   matrix to account for the shortened day.
   Reference: Paper Section 4.1 ("excluding half-day sessions").

3. **First day of training**: At tau=1, there is no prior state to transition from.
   Use pi_1 and Sigma_1 directly. The EM will learn appropriate values.
   Reference: Paper Equation (6), initial state definition.

4. **Near-singular covariance**: If Sigma[tau|tau] becomes near-singular (e.g., due to
   very small r relative to process noise), the Kalman gain saturates and the smoother
   gain L_tau may be numerically unstable. Add a small epsilon (e.g., 1e-10) to the
   diagonal of covariance matrices if needed.
   Researcher inference based on standard Kalman filter numerical practice.

5. **Non-stationary parameters**: If a_eta or a_mu is estimated >= 1 by EM, the process
   is non-stationary. Clamp to (0, 0.9999) or investigate whether the training window
   contains a structural break.
   Researcher inference -- the paper assumes stationarity but does not discuss clamping.

6. **Negative variance estimates**: EM should produce non-negative variance estimates by
   construction (they are sums of squares). If numerical issues produce negative values,
   clamp to a small positive floor (e.g., 1e-10).
   Researcher inference.

7. **Cross-exchange differences**: The number of bins I varies by exchange (e.g., 26 for
   NYSE 6.5-hour sessions, different for European/Asian exchanges with different hours).
   The model handles this naturally as I is a configuration parameter, but phi must be
   re-estimated for each exchange.
   Reference: Paper Section 4.1, Table 2.

### Known Limitations

1. **Cannot handle zero-volume bins**: The log transform is undefined at zero. This limits
   applicability to liquid securities. For illiquid securities with frequent zero-volume
   bins, a different approach (e.g., the zero-augmented model of Naimoli and Storti 2019)
   is needed. Reference: Paper Section 2 (implicit assumption).

2. **Linear Gaussian assumption**: The model assumes Gaussian noise in log-space. While
   the log transform improves normality substantially (Figure 1), residual non-Gaussianity
   (e.g., heavy tails from news events) is not modeled. Reference: Paper Section 2.

3. **No exogenous variables**: The model does not incorporate external information (e.g.,
   scheduled events, market-wide volume surges, volatility). All prediction comes from
   the volume series itself. Reference: Paper Section 5 (listed as future work).

4. **Piecewise constant daily component**: eta_t is constant within a day, which means the
   daily level cannot adjust intraday even if early bins reveal that today is an unusually
   high/low volume day. Only mu_{t,i} can adapt, but it has limited memory (AR(1)).
   The Bayesian model of Markov et al. (2019) handles this better via conjugate updating.
   Researcher inference from model structure.

5. **Single-security model**: Unlike the BDF approach (Bialkowski et al. 2008), this model
   is fitted independently per security. It cannot exploit cross-sectional information.
   Reference: Paper Section 4 (each security estimated independently).

6. **No comparison with BDF or GAS-Dirichlet**: The paper compares only against CMEM and
   rolling means, not against other approaches in the literature.
   Reference: Paper Section 4.

7. **Log-to-linear bias**: Exponentiating log-volume forecasts produces median forecasts,
   not mean forecasts (due to Jensen's inequality). For unbiased mean volume forecasts,
   a bias correction exp(y_hat + 0.5 * forecast_variance) should be applied. The paper
   does not discuss this correction.
   Researcher inference from log-normal theory.

## Paper References

| Spec Section | Paper Source |
|---|---|
| Overview, Model Description | Chen et al. (2016), Section 2, Equations (2)-(5) |
| Kalman Filter pseudocode | Chen et al. (2016), Section 2.2, Algorithm 1 |
| Multi-step prediction | Chen et al. (2016), Section 2.2, Equation (9) |
| Kalman Smoother pseudocode | Chen et al. (2016), Section 2.3.1, Algorithm 2 |
| Cross-covariance computation | Chen et al. (2016), Appendix A.2, Equations (A.20)-(A.22) |
| EM Algorithm pseudocode | Chen et al. (2016), Section 2.3.2, Algorithm 3 |
| EM M-step closed-form updates | Chen et al. (2016), Appendix A.3, Equations (A.32)-(A.39) |
| EM convergence properties | Chen et al. (2016), Section 2.3.3, Figure 4 |
| Robust filter formulation | Chen et al. (2016), Section 3.1, Equations (25)-(34) |
| Robust EM modifications | Chen et al. (2016), Section 3.2, Equations (35)-(36) |
| Robustness simulation | Chen et al. (2016), Section 3.3, Table 1 |
| VWAP strategies | Chen et al. (2016), Section 4.3, Equations (39)-(42) |
| Empirical MAPE results | Chen et al. (2016), Section 4.2, Table 3 |
| VWAP tracking results | Chen et al. (2016), Section 4.3, Table 4 |
| Data setup and securities | Chen et al. (2016), Section 4.1, Table 2 |
| EM initialization insensitivity | Researcher inference (initialization values); convergence property from Paper Section 2.3.3 |
| Zero-volume handling strategies | Researcher inference (paper excludes zero bins without discussion) |
| Log-to-linear bias correction | Researcher inference from log-normal distribution theory |
| M-step update ordering | Researcher inference (phi before r, based on Eq A.38 dependency) |
| Near-singular covariance handling | Researcher inference (standard numerical Kalman filter practice) |
| Stationarity clamping | Researcher inference (paper assumes stationarity without explicit enforcement) |
