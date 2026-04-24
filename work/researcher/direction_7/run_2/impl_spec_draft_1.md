# Implementation Specification: Kalman Filter State-Space Model for Intraday Volume

## Overview

This specification describes a linear Gaussian state-space model that forecasts
intraday trading volume by decomposing log-volume into three additive components:
a daily average level, an intraday periodic (seasonal) pattern, and an intraday
dynamic component. The model uses a Kalman filter for prediction and an
Expectation-Maximization (EM) algorithm with closed-form updates for parameter
estimation. A robust extension adds Lasso-penalized sparse noise detection to
handle outliers automatically. The approach converts the well-known multiplicative
volume decomposition of Brownlees et al. (2011) into a tractable linear system by
working in log space.

Source: Chen, Feng, and Palomar (2016), "Forecasting Intraday Trading Volume:
A Kalman Filter Approach."

## Algorithm

### Model Description

The model operates on log-transformed intraday volume. Raw volume in each
15-minute bin is first normalized by shares outstanding (to handle splits and
scale differences), then log-transformed. The log transformation converts the
multiplicative three-component decomposition into an additive one, eliminates
positivity constraints, and makes the Gaussian noise assumption more defensible
(Paper, Section 2, Figure 1).

The observation equation is:

    y_{t,i} = eta_t + phi_i + mu_{t,i} + v_{t,i}

where:
- y_{t,i}: observed log-volume for day t, bin i
- eta_t: log daily average component (same for all bins within day t)
- phi_i: log intraday periodic component (same value for bin i across all days)
- mu_{t,i}: log intraday dynamic component (varies by day and bin)
- v_{t,i}: observation noise ~ N(0, r)

Input: historical time series of intraday bin volumes (normalized by shares
outstanding) for a single security, with bins at regular intervals (e.g.,
15-minute). Zero-volume bins must be excluded.

Output: one-step-ahead or multi-step-ahead forecasts of log-volume for each
bin, which are exponentiated to produce volume forecasts and then converted
to VWAP execution weights.

Assumptions:
- Log-volume is approximately Gaussian (Paper, Section 2, Figure 1).
- The daily component is constant within each trading day and follows an
  AR(1) process across days.
- The intraday dynamic component follows an AR(1) process across bins.
- All noise terms are Gaussian and mutually independent.
- No zero-volume bins (log(0) is undefined).

### Pseudocode

The model has three main algorithmic components: (1) the Kalman filter for
prediction, (2) the Kalman smoother for calibration, and (3) the EM algorithm
that orchestrates calibration. Plus the robust extension.

#### Notation Conventions

The paper uses a unified time index tau = 1, 2, ..., N that linearizes the
(day, bin) pairs. Day t spans bins tau = (t-1)*I + 1 through tau = t*I, where
I is the number of bins per day. The set of day-boundary indices D contains
tau = kI for k = 1, 2, ..., (i.e., tau values that are the last bin of each day;
the transition FROM these indices TO tau+1 applies the day-boundary dynamics).

Throughout this spec:
- "tau in D" means tau is a day-boundary index (tau mod I == 0), meaning
  the NEXT step (tau -> tau+1) crosses a day boundary.
- Subscript notation: x_{tau|tau} means the filtered estimate at tau given
  observations up to tau; x_{tau|N} means the smoothed estimate at tau given
  all N observations.

(Paper, Section 2, Equations 4-5; Section 2.2, Algorithm 1)

#### Algorithm 1: Kalman Filter (Prediction)

```
FUNCTION kalman_filter(y[1..N], theta):
    # theta = {a_eta, a_mu, sigma_eta_sq, sigma_mu_sq, r, phi[1..I], pi_1, Sigma_1}
    # N = total number of bins in training/prediction window
    # I = number of bins per trading day

    # Initialize
    x_hat = pi_1                          # 2x1 state mean [eta, mu]^T
    Sigma = Sigma_1                       # 2x2 state covariance
    C = [1, 1]                            # 1x2 observation matrix (row vector)

    FOR tau = 1 TO N:
        # --- Prediction step ---
        # Build transition matrix A_tau based on whether we cross a day boundary
        IF (tau - 1) is a day boundary (i.e., (tau - 1) mod I == 0 AND tau > 1):
            # Day boundary: eta transitions with AR coefficient
            A = [[a_eta, 0],
                 [0,      a_mu]]
            Q = [[sigma_eta_sq, 0],
                 [0,            sigma_mu_sq]]
        ELSE:
            # Within day: eta is constant (coefficient = 1, no noise)
            A = [[1,     0],
                 [0, a_mu]]
            Q = [[0,            0],
                 [0, sigma_mu_sq]]
        END IF

        x_pred = A @ x_hat               # Predicted state mean (2x1)
        Sigma_pred = A @ Sigma @ A^T + Q  # Predicted state covariance (2x2)

        # --- Determine phi for this bin ---
        bin_index = ((tau - 1) mod I) + 1  # 1-based bin index within day
        phi_tau = phi[bin_index]

        # --- Correction step (if observation available) ---
        IF y[tau] is observed:
            # Innovation
            innovation = y[tau] - C @ x_pred - phi_tau    # scalar

            # Kalman gain
            W = C @ Sigma_pred @ C^T + r                  # scalar (innovation variance)
            K = Sigma_pred @ C^T / W                      # 2x1 Kalman gain

            # Update
            x_hat = x_pred + K * innovation               # 2x1
            Sigma = Sigma_pred - K @ C @ Sigma_pred       # 2x2
                  = (I_2 - K @ C) @ Sigma_pred            # equivalent form
        ELSE:
            # Static prediction: skip correction, propagate prediction
            x_hat = x_pred
            Sigma = Sigma_pred
        END IF

        # Store for output
        x_filtered[tau] = x_hat
        Sigma_filtered[tau] = Sigma

        # Log-volume forecast (one-step-ahead, before correction):
        y_hat[tau] = C @ x_pred + phi_tau
    END FOR

    RETURN x_filtered, Sigma_filtered, y_hat
END FUNCTION
```

(Paper, Section 2.2, Algorithm 1, Equations 7-9)

**Multi-step-ahead forecast:** For h-step-ahead prediction from time tau, apply
the prediction step h times without correction. The forecast is:

    y_hat_{tau+h|tau} = C @ x_{tau+h|tau} + phi_{tau+h}

where x_{tau+h|tau} is obtained by recursively applying the transition A
(with appropriate day-boundary handling) h times starting from x_{tau|tau}.

(Paper, Section 2.2, Equation 9)

**Static vs. dynamic prediction:**
- Dynamic: correct after each observed bin (use the full filter loop above).
- Static: run corrections through the last bin of the previous day, then
  propagate predictions for all I bins of the next day without corrections.

(Paper, Section 2.2, paragraph after Algorithm 1)

#### Algorithm 2: Kalman Smoother (RTS Backward Pass)

```
FUNCTION kalman_smoother(x_filtered[1..N], Sigma_filtered[1..N],
                          A_matrices[1..N], theta):
    # Backward pass: Rauch-Tung-Striebel smoother
    # Requires the filtered estimates from a forward Kalman filter pass

    x_smooth[N] = x_filtered[N]
    Sigma_smooth[N] = Sigma_filtered[N]

    FOR tau = N-1 DOWN TO 1:
        # Build A_{tau+1} (transition from tau to tau+1)
        IF tau is a day boundary (tau mod I == 0):
            A_next = [[a_eta, 0], [0, a_mu]]
            Q_next = [[sigma_eta_sq, 0], [0, sigma_mu_sq]]
        ELSE:
            A_next = [[1, 0], [0, a_mu]]
            Q_next = [[0, 0], [0, sigma_mu_sq]]
        END IF

        # Predicted covariance at tau+1 (from tau)
        Sigma_pred_next = A_next @ Sigma_filtered[tau] @ A_next^T + Q_next

        # Smoother gain
        L_tau = Sigma_filtered[tau] @ A_next^T @ inv(Sigma_pred_next)  # 2x2

        # Smoothed estimates
        x_smooth[tau] = x_filtered[tau]
            + L_tau @ (x_smooth[tau+1] - A_next @ x_filtered[tau])
        Sigma_smooth[tau] = Sigma_filtered[tau]
            + L_tau @ (Sigma_smooth[tau+1] - Sigma_pred_next) @ L_tau^T

        # Store smoother gain for cross-covariance computation
        L[tau] = L_tau
    END FOR

    # Cross-covariance P_{tau, tau-1|N} needed for EM M-step
    # Initialize: Sigma_{N, N-1|N} = (I - K_N C) A_N Sigma_{N-1|N-1}
    #   but more cleanly computed via the recursion below
    Sigma_cross[N] = (I_2 - K_N @ C) @ A_N @ Sigma_filtered[N-1]

    FOR tau = N-1 DOWN TO 2:
        Sigma_cross[tau] = Sigma_filtered[tau] @ L[tau-1]^T
            + L[tau] @ (Sigma_cross[tau+1] - A_{tau+1} @ Sigma_filtered[tau]) @ L[tau-1]^T
    END FOR

    RETURN x_smooth, Sigma_smooth, Sigma_cross
END FUNCTION
```

(Paper, Section 2.3.1, Algorithm 2, Equations 10-11; Appendix A.2, Equations A.18-A.22)

**Note on Sigma_cross computation:** The paper defines three sufficient statistics
for the EM M-step (Equations 14-16):

    x_hat_tau = E[x_tau | y_{1:N}]                  (smoothed state mean)
    P_tau     = E[x_tau x_tau^T | y_{1:N}]          (smoothed second moment)
    P_{tau,tau-1} = E[x_tau x_{tau-1}^T | y_{1:N}]  (smoothed cross-moment)

These relate to the smoother outputs as:

    P_tau = Sigma_smooth[tau] + x_smooth[tau] @ x_smooth[tau]^T
    P_{tau,tau-1} = Sigma_cross[tau] + x_smooth[tau] @ x_smooth[tau-1]^T

The cross-covariance recursion (Appendix A, Equation A.20-A.22) is:

    Sigma_{tau,tau-1|N} = Sigma_{tau|N} @ L_{tau-1}^T  (simplified form)

initialized at:

    Sigma_{N,N-1|N} = (I_2 - K_N C) @ A_N @ Sigma_{N-1|N-1}

(Paper, Appendix A.2, Equations A.18-A.22)

#### Algorithm 3: EM Algorithm (Parameter Estimation)

```
FUNCTION em_calibrate(y[1..N], I, max_iter, tol):
    # N = T * I total bins (T training days, I bins per day)
    # Initialize parameters theta^(0)
    theta = initialize_parameters(y, I)

    FOR j = 0, 1, 2, ... UNTIL convergence:
        # === E-step ===
        # Run forward Kalman filter with current theta
        x_filt, Sigma_filt, _ = kalman_filter(y, theta)

        # Run backward RTS smoother
        x_smooth, Sigma_smooth, Sigma_cross = kalman_smoother(x_filt, Sigma_filt, theta)

        # Compute sufficient statistics
        FOR tau = 1 TO N:
            x_hat[tau] = x_smooth[tau]                    # smoothed mean (2x1)
            P[tau] = Sigma_smooth[tau] + x_hat[tau] @ x_hat[tau]^T  # second moment (2x2)
        END FOR
        FOR tau = 2 TO N:
            P_cross[tau] = Sigma_cross[tau] + x_hat[tau] @ x_hat[tau-1]^T  # cross moment (2x2)
        END FOR

        # === M-step (closed-form updates) ===

        # Notation: P^(k,l)_{tau} denotes element (k,l) of matrix P[tau]
        # D = set of day-boundary indices {I, 2I, 3I, ...}

        # --- Initial state mean and covariance ---
        pi_1 = x_hat[1]                                            # (Eq A.32)
        Sigma_1 = P[1] - x_hat[1] @ x_hat[1]^T                   # (Eq A.33)

        # --- AR coefficient for daily component (eta) ---
        # Only uses day-boundary transitions (tau in D, transition to tau+1)
        numerator_eta = SUM over tau in D: P_cross[tau+1]^(1,1)    # (Eq A.34, but see note)
        denominator_eta = SUM over tau in D: P[tau]^(1,1)
        a_eta = numerator_eta / denominator_eta                    # (Eq 19 / A.34)

        # --- AR coefficient for dynamic component (mu) ---
        numerator_mu = SUM over tau = 2 TO N: P_cross[tau]^(2,2)   # (Eq A.35)
        denominator_mu = SUM over tau = 2 TO N: P[tau-1]^(2,2)
        a_mu = numerator_mu / denominator_mu                       # (Eq 20 / A.35)

        # --- Process noise variance for eta ---
        sigma_eta_sq = (1 / (T - 1)) *
            SUM over tau in D:
                [ P[tau+1]^(1,1) + a_eta^2 * P[tau]^(1,1)
                  - 2 * a_eta * P_cross[tau+1]^(1,1) ]            # (Eq 21 / A.36)
        # Note: T-1 because there are T-1 day transitions in T days

        # --- Process noise variance for mu ---
        sigma_mu_sq = (1 / (N - 1)) *
            SUM over tau = 2 TO N:
                [ P[tau]^(2,2) + a_mu^2 * P[tau-1]^(2,2)
                  - 2 * a_mu * P_cross[tau]^(2,2) ]               # (Eq 22 / A.37)

        # --- Observation noise variance ---
        r = (1 / N) * SUM over tau = 1 TO N:
            [ y[tau]^2 + C @ P[tau] @ C^T
              - 2 * y[tau] * C @ x_hat[tau]
              + phi[bin(tau)]^2
              - 2 * y[tau] * phi[bin(tau)]
              + 2 * phi[bin(tau)] * C @ x_hat[tau] ]               # (Eq 23 / A.38)

        # Simplification: r = (1/N) * SUM[ (y[tau] - phi[bin(tau)] - C @ x_hat[tau])^2
        #                                   + C @ Sigma_smooth[tau] @ C^T ]

        # --- Seasonality vector ---
        FOR i = 1 TO I:
            phi[i] = (1 / T) * SUM over t = 1 TO T:
                ( y_{t,i} - C @ x_hat_{t,i} )                     # (Eq 24 / A.39)
            # i.e., average residual for bin i across all training days
        END FOR

        # Pack updated parameters
        theta = {a_eta, a_mu, sigma_eta_sq, sigma_mu_sq, r, phi, pi_1, Sigma_1}

        # Check convergence
        IF relative change in log-likelihood < tol:
            BREAK
        END IF
    END FOR

    RETURN theta
END FUNCTION
```

(Paper, Section 2.3.2, Algorithm 3, Equations 17-24; Appendix A.3, Equations A.32-A.39)

**Clarification on day-boundary indexing for a_eta update:** The paper's Equation
19 (and A.34) sums P_cross^(1,1) over "tau = kI+1" (the first bin of each new
day) for the numerator, and P^(1,1) at "tau = kI+1" to "N" for a related sum.
The key insight: a_eta only governs transitions at day boundaries. The numerator
is the sum of cross-moments E[eta_{tau} * eta_{tau-1}] specifically at those
transitions where the AR(1) applies (tau such that tau-1 is in D). The
denominator is the sum of E[eta_{tau-1}^2] at the same transitions. Within-day
transitions have a_eta_effective = 1, so they do not contribute to the a_eta
estimate.

(Paper, Appendix A.3, Equation A.26 derivation and Equation A.34)

#### Algorithm 4: Robust Kalman Filter (Lasso Extension)

```
FUNCTION robust_kalman_filter(y[1..N], theta, lambda):
    # Same as Algorithm 1, but the correction step is modified
    # Observation model: y[tau] = C @ x[tau] + phi[tau] + v[tau] + z[tau]
    # where z[tau] is a sparse outlier term

    # Initialize same as standard filter
    x_hat = pi_1
    Sigma = Sigma_1
    C = [1, 1]

    FOR tau = 1 TO N:
        # --- Prediction step (identical to standard) ---
        [same as Algorithm 1]

        x_pred = A @ x_hat
        Sigma_pred = A @ Sigma @ A^T + Q

        # --- Correction step with Lasso ---
        IF y[tau] is observed:
            # Innovation
            e_tau = y[tau] - phi[bin(tau)] - C @ x_pred    # scalar

            # Innovation variance (scalar since observation is 1D)
            W_tau = (C @ Sigma_pred @ C^T + r)^{-1}       # scalar: precision

            # Solve Lasso: min_{z} (e - z)^T W (e - z) + lambda |z|
            # Closed-form soft-thresholding solution:
            threshold = lambda / (2 * W_tau)
            IF e_tau > threshold:
                z_star = e_tau - threshold
            ELIF e_tau < -threshold:
                z_star = e_tau + threshold
            ELSE:
                z_star = 0
            END IF

            # Modified innovation (outlier removed)
            e_clean = e_tau - z_star

            # Standard Kalman update with cleaned innovation
            K = Sigma_pred @ C^T * W_tau                   # 2x1 Kalman gain (note: W is precision here)
            # Equivalently: K = Sigma_pred @ C^T / (C @ Sigma_pred @ C^T + r)
            x_hat = x_pred + K * e_clean
            Sigma = Sigma_pred - K @ C @ Sigma_pred

            z_detected[tau] = z_star
        ELSE:
            x_hat = x_pred
            Sigma = Sigma_pred
            z_detected[tau] = 0
        END IF

        x_filtered[tau] = x_hat
        Sigma_filtered[tau] = Sigma
        y_hat[tau] = C @ x_pred + phi[bin(tau)]
    END FOR

    RETURN x_filtered, Sigma_filtered, y_hat, z_detected
END FUNCTION
```

(Paper, Section 3.1, Equations 25-34)

**Key insight on W_tau:** In the paper, W_{tau+1} is defined as the inverse of
the innovation variance: W_{tau+1} = (C Sigma_{tau+1|tau} C^T + r)^{-1}. Since
the observation is scalar, this is just a scalar precision. The Lasso threshold
is lambda / (2 * W_{tau+1}), which equals lambda * (C Sigma_{tau+1|tau} C^T + r) / 2.
Higher prediction uncertainty -> wider threshold -> fewer outliers detected.

(Paper, Section 3.1, Equations 30, 33-34)

#### Algorithm 5: Robust EM Calibration

The EM for the robust model is identical to Algorithm 3 except:

1. In the E-step, use the robust Kalman filter (Algorithm 4) instead of the
   standard filter. Store the detected outlier values z_star[tau].

2. The M-step updates for r and phi are modified to account for the outliers:

```
    # Modified observation noise variance
    r = (1 / N) * SUM over tau = 1 TO N:
        [ y[tau]^2 + C @ P[tau] @ C^T
          - 2 * y[tau] * C @ x_hat[tau]
          + (phi[bin(tau)])^2 + (z_star[tau])^2
          - 2 * y[tau] * phi[bin(tau)] - 2 * z_star[tau] * y[tau]
          + 2 * z_star[tau] * phi[bin(tau)]
          + 2 * z_star[tau] * C @ x_hat[tau]
          + 2 * phi[bin(tau)] * C @ x_hat[tau] ]              # (Eq 35)

    # Modified seasonality
    FOR i = 1 TO I:
        phi[i] = (1 / T) * SUM over t = 1 TO T:
            ( y_{t,i} - C @ x_hat_{t,i} - z_star_{t,i} )      # (Eq 36)
    END FOR
```

3. All other M-step updates (a_eta, a_mu, sigma_eta_sq, sigma_mu_sq, pi_1,
   Sigma_1) remain unchanged because they depend only on the state sufficient
   statistics, not on the observation model.

(Paper, Section 3.2, Equations 35-36)

#### Algorithm 6: VWAP Execution Weights

```
FUNCTION compute_vwap_weights_static(y_hat_static[1..I]):
    # Convert log-volume forecasts to volume proportions
    # y_hat_static[i] = predicted log-volume for bin i (all predicted before market open)

    vol_hat[i] = exp(y_hat_static[i])  FOR i = 1..I
    total = SUM(vol_hat)
    w[i] = vol_hat[i] / total          FOR i = 1..I
    RETURN w[1..I]
END FUNCTION

FUNCTION compute_vwap_weights_dynamic(y_hat_dynamic, volumes_observed, current_bin):
    # At bin current_bin, we have observed actual volumes for bins 1..current_bin-1
    # and one-step-ahead forecast for bin current_bin
    # Redistribute remaining execution proportionally among remaining bins

    # For bins 1..current_bin-1: actual weights used (already executed)
    # For bins current_bin..I: use predicted volumes, normalized

    # Remaining fraction to execute
    actual_cumulative = SUM(volumes_observed[1..current_bin-1])
    # Predicted remaining volume
    FOR j = current_bin TO I:
        vol_hat[j] = exp(y_hat_dynamic[j])   # dynamic forecast for bin j
    END FOR
    remaining_total = SUM(vol_hat[current_bin..I])

    # Weight for current bin
    w[current_bin] = vol_hat[current_bin] / remaining_total
                     * (1 - actual_cumulative / predicted_total_daily)

    RETURN w[current_bin]
END FUNCTION
```

(Paper, Section 4.3, Equations 39-41)

**Static VWAP weight** (Equation 40):

    w_{t,i}^(s) = volume_hat_{t,i}^(s) / SUM_{j=1}^{I} volume_hat_{t,j}^(s)

**Dynamic VWAP weight** (Equation 41):

    w_{t,i}^(d) = (volume_hat_{t,i}^(d) / SUM_{j=i}^{I} volume_hat_{t,j}^(d))
                  * (1 - SUM_{j=1}^{i-1} w_{t,j}^(d))      for i = 1, ..., I-1
    w_{t,I}^(d) = 1 - SUM_{j=1}^{I-1} w_{t,j}^(d)          for the last bin

### Data Flow

```
Raw volume per bin (shares traded)
    |
    v
Normalize by shares outstanding  -->  volume_{t,i} = shares_traded / shares_outstanding
    |
    v
Log transform  -->  y_{t,i} = log(volume_{t,i})
    |                (Input: scalar per bin; must be > 0)
    v
+---------------------------------------------------+
| EM Calibration (offline, on training window)       |
|                                                    |
|   Forward Kalman Filter  -->  x_filt (2x1 per bin) |
|         |                     Sigma_filt (2x2)     |
|         v                                          |
|   RTS Backward Smoother  -->  x_smooth, P, P_cross|
|         |                                          |
|         v                                          |
|   M-step: closed-form updates for                  |
|     a_eta, a_mu, sigma_eta_sq, sigma_mu_sq,        |
|     r, phi[1..I], pi_1, Sigma_1                    |
|                                                    |
|   Repeat until convergence                         |
+---------------------------------------------------+
    |
    | Calibrated theta
    v
+---------------------------------------------------+
| Online Prediction (Kalman Filter or Robust KF)     |
|                                                    |
|   For each new bin:                                |
|     Predict: x_pred (2x1), Sigma_pred (2x2)       |
|     Forecast: y_hat = C @ x_pred + phi (scalar)   |
|     Correct: x_hat = x_pred + K * innovation      |
|       (skip correction for static mode)            |
+---------------------------------------------------+
    |
    v
y_hat (scalar log-volume forecast per bin)
    |
    v
exp(y_hat)  -->  volume forecast (shares / shares_outstanding)
    |
    v
Normalize to weights  -->  w_i = vol_hat_i / SUM(vol_hat)
    |
    v
VWAP execution schedule
```

**Shapes at each step:**
- State x_tau: 2x1 vector [eta, mu]^T
- State covariance Sigma_tau: 2x2 matrix
- Transition matrix A_tau: 2x2 (time-varying)
- Process noise covariance Q_tau: 2x2 diagonal (time-varying)
- Observation matrix C: 1x2 = [1, 1]
- Observation noise r: scalar
- Kalman gain K: 2x1
- Innovation variance W: scalar
- Seasonality phi: vector of length I (number of bins per day)
- y_hat: scalar per bin

### Variants

**Implemented variant: Robust Kalman Filter with EM calibration.** This is the
most complete model described in the paper (Section 3), which subsumes the
standard Kalman filter as a special case (lambda -> infinity or equivalently
z_star = 0 for all bins). The robust variant is chosen because:

1. It achieves the best empirical performance (Paper, Table 1, Table 3).
2. The standard KF is recovered by setting lambda to a very large value, so
   implementing the robust version provides both variants.
3. Production market data contains outliers that the standard KF cannot handle
   gracefully (Paper, Section 3, Table 1: CMEM fails entirely at medium/large
   outlier levels, standard KF degrades significantly).

**Not implemented:**
- The CMEM benchmark model (Brownlees et al. 2011) -- this is a separate
  direction (Direction 1).
- The rolling mean (RM) baseline -- trivial, not a modeling direction.

(Paper, Section 3; Section 4.2, Table 3)

## Parameters

| Parameter | Description | Recommended Value | Sensitivity | Range |
|-----------|-------------|-------------------|-------------|-------|
| a_eta | AR(1) coefficient for daily component (day-to-day persistence) | Data-driven via EM. Paper's synthetic experiment converges to ~0.98 (Figure 4a) | High -- controls how much yesterday's daily level carries forward. Values near 1 mean high persistence | (0, 1) for stationarity |
| a_mu | AR(1) coefficient for intraday dynamic component (bin-to-bin persistence) | Data-driven via EM. Paper's synthetic experiment converges to ~0.5 (Figure 4b) | Medium -- controls intraday mean reversion speed | (0, 1) for stationarity |
| sigma_eta_sq | Process noise variance for daily component | Data-driven via EM. Paper's synthetic: ~0.01 (Figure 4d) | Medium -- larger values let eta adapt faster but increase prediction variance | (0, inf) |
| sigma_mu_sq | Process noise variance for intraday dynamic component | Data-driven via EM. Paper's synthetic: ~0.015 (Figure 4e) | Medium -- controls intraday innovation magnitude | (0, inf) |
| r | Observation noise variance | Data-driven via EM. Paper's synthetic: ~0.05 (Figure 4c) | Medium -- larger r means filter trusts observations less | (0, inf) |
| phi[1..I] | Intraday seasonality vector (one value per bin) | Data-driven via EM (simple mean of residuals). Exhibits U-shape (higher at open/close) | Low once estimated -- stable across days | (-inf, inf) in log space |
| pi_1 | Initial state mean [eta_1, mu_1]^T | Set to [mean(y), 0]^T or estimate via EM. Paper: EM converges regardless of initialization (Figure 4) | Very low -- EM is robust to initialization | Any 2x1 vector |
| Sigma_1 | Initial state covariance (2x2) | Set to diagonal with moderate variances (e.g., [[var(y), 0], [0, var(y)]]) or estimate via EM | Very low -- EM is robust to initialization | Any positive definite 2x2 |
| N | Training window length (number of total bins = T_train * I) | Selected by cross-validation. Not specified in paper; try T_train in {60, 90, 120, 180, 250} days | High -- too short = noisy estimates, too long = stale model | Must have T_train >= 20 days minimum |
| lambda | Lasso regularization parameter (robust model only) | Selected by cross-validation jointly with N. Not specified in paper | High -- too small = over-detects outliers, too large = no outlier detection (reduces to standard KF) | (0, inf); lambda -> inf recovers standard KF |
| I | Number of bins per trading day | Determined by exchange hours and bin width. 26 for NYSE 15-min bins (6.5 hours / 15 min) | N/A -- fixed by data design | Typically 13-52 for 30-min to 10-min bins |

### Initialization

**EM initialization (theta^(0)):**

The paper demonstrates that EM convergence is robust to initialization choice
(Section 2.3.3, Figure 4). Nevertheless, reasonable defaults accelerate convergence:

1. Compute the overall mean of y[1..N]: call it y_bar.
2. For each bin i, compute phi^(0)[i] = mean of y_{t,i} across all training
   days t, minus y_bar.
3. Set pi_1 = [y_bar, 0]^T (daily level at overall mean, zero dynamic component).
4. Set Sigma_1 = diag(var(y), var(y)) (generous initial uncertainty).
5. Set a_eta^(0) = 0.95 (high persistence for daily level).
6. Set a_mu^(0) = 0.5 (moderate intraday persistence).
7. Set sigma_eta_sq^(0) = 0.01, sigma_mu_sq^(0) = 0.01 (small process noise).
8. Set r^(0) = var(y) * 0.1 (observation noise is fraction of total variance).

(Researcher inference: specific initialization values are not given in the paper.
The paper states EM converges regardless of initialization (Section 2.3.3,
Figure 4), so these are reasonable defaults based on the model structure and
typical log-volume statistics. The synthetic experiment in Figure 4 uses "different
choices of initial parameters" without specifying them.)

**Kalman filter state initialization at prediction time:**

Use the final smoothed state from calibration, or if starting a new rolling
window, use pi_1 and Sigma_1 from the most recent EM fit.

### Calibration

**Calibration procedure (step by step):**

1. **Data preparation:**
   a. Collect intraday volume data for T_train days at I bins per day.
   b. Normalize each bin's volume by shares outstanding for that day.
   c. Take the natural log: y_{t,i} = ln(normalized_volume_{t,i}).
   d. Exclude any bins with zero volume (or impute, but paper excludes them).
   e. Exclude half-day trading sessions (Paper, Section 4.1).
   f. Linearize into a single time series y[1..N] where N = T_train * I.

2. **EM iteration:**
   a. Initialize theta^(0) as described above.
   b. E-step: run forward Kalman filter (Algorithm 1), then backward RTS
      smoother (Algorithm 2), and compute sufficient statistics P[tau] and
      P_cross[tau].
   c. M-step: apply closed-form updates (Equations 17-24 / A.32-A.39).
   d. Repeat until relative change in expected log-likelihood < tol (e.g.,
      tol = 1e-6) or max_iter reached (e.g., 50 iterations; convergence is
      typically within 5-10 iterations per Figure 4).

3. **Cross-validation for N and lambda:**
   a. Define a validation period (Paper uses January 2015 - May 2015).
   b. For each candidate N in a grid (e.g., {60, 90, 120, 180, 250} days *I):
      - For each candidate lambda in a grid (e.g., log-spaced from 0.01 to 100):
        - Train on the N bins preceding each validation day.
        - Predict the validation day (dynamic mode).
        - Compute MAPE on validation set.
   c. Select (N, lambda) pair with minimum validation MAPE.
   d. For the standard KF: only cross-validate N (no lambda).

4. **Rolling re-estimation:**
   a. In production, re-run EM on a rolling window of the most recent N bins.
   b. The paper uses a standard rolling window: for each out-of-sample day,
      train on the preceding N bins (Paper, Section 4.2).
   c. Researcher inference: re-estimation frequency is not specified. Daily
      re-estimation is the natural choice given the rolling window scheme,
      and computational cost is trivial (2D state, convergence in ~10 EM
      iterations).

(Paper, Section 2.3.2, Algorithm 3; Section 4.1-4.2)

## Validation

### Expected Behavior

**Volume prediction MAPE (dynamic mode, out-of-sample):**
- Robust Kalman Filter: average MAPE = 0.46 across 30 securities
- Standard Kalman Filter: average MAPE = 0.47
- CMEM benchmark: average MAPE = 0.65
- Rolling Mean baseline: average MAPE = 1.28

(Paper, Section 4.2, Table 3)

**Volume prediction MAPE (static mode, out-of-sample):**
- Robust Kalman Filter: average MAPE = 0.61
- Standard Kalman Filter: average MAPE = 0.62
- CMEM benchmark: average MAPE = 0.90
- Rolling Mean: average MAPE = 1.28

(Paper, Section 4.2, Table 3)

**VWAP tracking error (dynamic strategy, out-of-sample):**
- Robust Kalman Filter: average 6.38 basis points
- Standard Kalman Filter: average 6.39 bps
- CMEM: 7.01 bps
- Rolling Mean: 7.48 bps

(Paper, Section 4.3, Table 4)

**Per-security examples** (dynamic MAPE, mean +/- std from Table 3):
- SPY: Robust KF 0.24 +/- 0.19, Standard KF 0.24 +/- 0.19, CMEM 0.26 +/- 0.22
- AAPL: Robust KF 0.21 +/- 0.17, Standard KF 0.21 +/- 0.17, CMEM 0.23 +/- 0.20
- IBM: Robust KF 0.24 +/- 0.20, Standard KF 0.24 +/- 0.21, CMEM 0.27 +/- 0.28
- QQQ: Robust KF 0.30 +/- 0.26, Standard KF 0.30 +/- 0.27, CMEM 0.33 +/- 0.31

(Paper, Table 3)

**EM convergence behavior:**
- Parameters converge within approximately 5-10 iterations from arbitrary
  initial values (Paper, Section 2.3.3, Figure 4).
- Convergence is monotonic (log-likelihood increases at each step).
- Insensitive to initialization (unlike CMEM's GMM, which is brittle).

(Paper, Section 2.3.3, Figure 4)

### Sanity Checks

1. **EM convergence:** Run EM from 5 different random initializations. All
   should converge to the same parameter values (within numerical tolerance).
   This verifies the EM implementation is correct and the likelihood surface
   is well-behaved. (Paper, Section 2.3.3, Figure 4)

2. **Seasonality shape:** The estimated phi vector should exhibit the well-known
   U-shape: higher values at market open (bin 1) and close (bin I), lower
   values mid-day. Verify visually. (Paper, Section 2, description of phi_i
   as "U-shaped intraday seasonal pattern")

3. **AR coefficient ranges:** a_eta should be close to 1 (high daily
   persistence, typically 0.9-0.99). a_mu should be moderate (0.3-0.7,
   indicating partial mean reversion within the day). (Paper, Figure 4:
   synthetic experiments converge to a_eta ~0.98, a_mu ~0.5)

4. **Prediction variance growth:** In static multi-step prediction, forecast
   uncertainty (Sigma_{tau+h|tau}) should grow with horizon h. The dynamic
   mode should have lower prediction variance than static mode because it
   incorporates more recent information. (Standard Kalman filter property)

5. **Robust filter outlier detection:** On clean data, the robust KF with
   reasonable lambda should detect very few outliers (z_star = 0 for most
   bins). On artificially contaminated data (e.g., adding +3 sigma shocks
   to 10% of bins), z_star should be nonzero primarily at the contaminated
   bins. (Paper, Section 3.3, Table 1)

6. **Log-likelihood monotonicity:** The expected log-likelihood Q(theta) should
   increase (or remain constant) at every EM iteration. A decrease indicates
   an implementation bug. (Standard EM property)

7. **Baseline comparison:** The model should outperform a simple rolling mean
   baseline (mean of same-bin log-volumes over prior 20 days). If it does not,
   something is wrong with the state-space formulation or calibration. (Paper,
   Table 3: KF achieves 64% MAPE improvement over RM)

8. **VWAP weights sum to 1:** After converting log-volume forecasts to weights,
   verify that SUM(w_i) = 1 for both static and dynamic strategies. (Paper,
   Section 4.3, Equations 40-41)

### Edge Cases

1. **Zero-volume bins:** log(0) is undefined. The model cannot handle bins with
   zero traded volume. Options:
   - Exclude zero-volume bins and treat them as missing observations (skip
     the Kalman correction step for those bins).
   - Impute with a small positive value (e.g., 1 share / shares_outstanding)
     before taking log. This introduces bias.
   - The paper explicitly excludes zero-volume bins (Paper, Section 4.1:
     "the model assumes non-zero volumes").

2. **Half-day trading sessions:** Sessions with fewer than I bins (e.g., day
   before holidays in the U.S.) should be excluded from training data. If
   encountered during prediction, either skip the day or use a reduced I
   with appropriately matched phi values. (Paper, Section 4.1: "excluding
   half-day sessions")

3. **Day-boundary transition:** The Kalman filter transition matrix changes at
   day boundaries. Implementation must correctly detect when tau crosses from
   one day to the next. Off-by-one errors here will corrupt the eta dynamics.
   Test: run filter on a 2-day sequence and verify that eta transitions with
   a_eta only between the last bin of day 1 and the first bin of day 2.

4. **Numerical stability of covariance matrices:** Sigma and Sigma_pred must
   remain positive semi-definite. In practice, use the Joseph form for the
   covariance update: Sigma = (I - K C) Sigma_pred (I - K C)^T + K r K^T,
   which is numerically more stable than the standard form Sigma = Sigma_pred
   - K C Sigma_pred. (Researcher inference: standard numerical Kalman filter
   best practice, not discussed in the paper but essential for production
   implementations)

5. **Very large or very small volumes:** After normalization by shares
   outstanding, volumes can span several orders of magnitude. The log
   transform handles this, but extreme values in log space can still affect
   EM convergence. Consider winsorizing log-volumes at +/- 5 standard
   deviations before calibration.

6. **Degenerate EM solutions:** If sigma_eta_sq or sigma_mu_sq converge to
   zero, the corresponding state component becomes deterministic, and the
   filter may become ill-conditioned. Add a small floor (e.g., 1e-10) to
   variance estimates in the M-step. (Researcher inference: standard
   numerical precaution for EM, not discussed in paper)

7. **Lambda cross-validation:** If the optimal lambda is at the boundary of
   the search grid, the grid should be extended. If lambda -> 0 is optimal,
   the data may have severe outlier issues requiring investigation. If
   lambda -> inf is optimal, the standard KF suffices.

### Known Limitations

1. **Cannot handle zero-volume bins.** This limits applicability to illiquid
   securities or very fine time granularities. (Paper, Section 2:
   "log(0) is undefined")

2. **Single-security model.** No cross-sectional information is used. Each
   security is modeled independently. This means the model cannot exploit
   market-wide volume patterns shared across stocks. (Paper: entire model
   is per-security; no cross-sectional dimension, unlike the BDF model of
   Direction 2)

3. **Linear dynamics only.** The AR(1) assumptions for both eta and mu are
   restrictive. Real volume dynamics may be nonlinear (e.g., regime-switching,
   threshold effects). The log transform partially addresses this but does
   not capture true nonlinearity. (Paper: the model is explicitly linear
   Gaussian)

4. **No exogenous variables.** The model has no mechanism to incorporate
   external information (e.g., scheduled events, earnings, FOMC, options
   expiry). Such events are known to cause predictable volume surges.
   (Paper, Section 5: identifies "incorporating additional covariates"
   as future work)

5. **Gaussian noise assumption.** While the log transform makes the Gaussian
   assumption more defensible, Figure 1 in the paper still shows some
   departure from normality (thin left tail). The robust extension mitigates
   this for outliers but does not address systematic non-Gaussianity.
   (Paper, Section 2, Figure 1)

6. **Seasonality is static.** The phi vector is estimated as a fixed average
   over the training window and does not adapt to changing intraday patterns
   (e.g., the shift from U-shape to inverted-J-shape on high-gap days noted
   by Markov et al. 2019). (Paper, Equation 24: phi is a time-invariant
   average)

7. **No benchmark against BDF or GAS-Dirichlet.** The paper compares only
   against CMEM and rolling means. Performance relative to PCA-based
   (Direction 2) or Dirichlet (Direction 3) approaches is unknown.
   (Paper, Section 4)

## Paper References

| Spec Section | Paper Source |
|---|---|
| Model Description (observation equation) | Paper, Section 2, Equation 3 |
| State-space formulation | Paper, Section 2, Equations 4-5 |
| State transition matrix (time-varying A_tau) | Paper, Section 2, definition after Equation 5 |
| Process noise structure (Q_tau) | Paper, Section 2, Q_tau definition |
| Kalman filter algorithm | Paper, Section 2.2, Algorithm 1 |
| Multi-step prediction | Paper, Section 2.2, Equation 9 |
| Static vs dynamic prediction | Paper, Section 2.2, paragraphs after Algorithm 1 |
| Kalman smoother (RTS) | Paper, Section 2.3.1, Algorithm 2 |
| Sufficient statistics for EM | Paper, Equations 14-16 |
| EM algorithm | Paper, Section 2.3.2, Algorithm 3 |
| M-step closed-form updates | Paper, Equations 17-24; Appendix A.3, Equations A.32-A.39 |
| Cross-covariance recursion | Paper, Appendix A.2, Equations A.20-A.22 |
| EM convergence properties | Paper, Section 2.3.3, Figure 4 |
| Robust observation model | Paper, Section 3, Equation 25 |
| Robust state-space formulation | Paper, Section 3.1, Equations 26-27 |
| Lasso-in-Kalman derivation | Paper, Section 3.1, Equations 28-30 |
| Soft-thresholding solution | Paper, Section 3.1, Equations 33-34 |
| Robust EM modifications (r, phi) | Paper, Section 3.2, Equations 35-36 |
| MAPE definition | Paper, Section 3.3, Equation 37 |
| Empirical data description | Paper, Section 4.1, Table 2 |
| Volume prediction results | Paper, Section 4.2, Table 3 |
| VWAP tracking error results | Paper, Section 4.3, Table 4 |
| Static VWAP weights | Paper, Section 4.3, Equation 40 |
| Dynamic VWAP weights | Paper, Section 4.3, Equation 41 |
| VWAP tracking error metric | Paper, Section 4.3, Equation 42 |
| Log-likelihood function | Paper, Appendix A.1, Equation A.8 |
| EM initialization robustness | Researcher inference (specific values not in paper; Figure 4 confirms robustness) |
| Joseph form covariance update | Researcher inference (standard Kalman filter best practice) |
| Variance floor in M-step | Researcher inference (standard EM numerical precaution) |
| Rolling re-estimation frequency | Researcher inference (paper uses rolling window but does not specify re-estimation cadence) |
