# Implementation Specification: Kalman Filter State-Space Model for Intraday Volume

## Overview

This specification describes a linear Gaussian state-space model that forecasts
intraday trading volume by decomposing log-volume into three additive components:
a daily average level, an intraday periodic (seasonal) pattern, and an intraday
dynamic component. The model uses a Kalman filter for online prediction and a
Rauch-Tung-Striebel smoother plus Expectation-Maximization (EM) algorithm with
fully closed-form parameter updates for calibration. A robust extension adds
Lasso-penalized sparse noise detection to handle outliers automatically via
soft-thresholding in the Kalman correction step.

The approach converts the multiplicative three-component volume decomposition of
Brownlees, Cipollini, and Gallo (2011) into a tractable linear system by working
in log space. The state dimension is only 2, making all Kalman operations
reducible to 2x2 matrix algebra and scalar divisions.

Source: Chen, Feng, and Palomar (2016), "Forecasting Intraday Trading Volume:
A Kalman Filter Approach."

## Algorithm

### Model Description

The model operates on log-transformed intraday volume. For each trading day
divided into I equally-spaced bins (e.g., 15-minute intervals), raw volume in
each bin is first normalized by shares outstanding (to handle splits, buybacks,
and scale differences across securities), then log-transformed. The log
transformation converts the multiplicative decomposition into an additive one,
eliminates positivity constraints on components, and makes the Gaussian noise
assumption more defensible empirically (Paper, Section 2, Figure 1 -- Q-Q plots
show log-volume is approximately Gaussian while raw volume is heavily
right-skewed).

The observation equation decomposes log-volume as:

    y_{t,i} = eta_t + phi_i + mu_{t,i} + v_{t,i}

where:
- y_{t,i}: observed log-volume for day t, bin i (i = 1, ..., I)
- eta_t: log daily average component (constant within day t, evolves across days)
- phi_i: log intraday periodic component (the U-shaped seasonal pattern; same
  value for bin i on every day)
- mu_{t,i}: log intraday dynamic component (varies by day and bin, captures
  short-term deviations from the seasonal pattern)
- v_{t,i}: observation noise, i.i.d. N(0, r)

(Paper, Section 2, Equation 3)

**Input:** Historical time series of intraday bin volumes (shares traded per bin,
normalized by shares outstanding) for a single security, organized as T training
days with I bins per day.

**Output:** One-step-ahead (dynamic) or multi-step-ahead (static) forecasts of
log-volume for each future bin, convertible to volume forecasts by exponentiation.

**Assumptions:**
- Log-volume is approximately Gaussian (Paper, Section 2, Figure 1).
- eta_t is constant within each trading day and follows an AR(1) process across
  days (Paper, Section 2, Equations 4-5).
- mu_{t,i} follows an AR(1) process across consecutive bins, including across
  day boundaries (Paper, Section 2, Equations 4-5).
- All noise terms (process noise for eta, process noise for mu, observation
  noise v) are Gaussian and mutually independent (Paper, Section 2).
- No zero-volume bins exist in the data; log(0) is undefined (Paper, Section 4.1).
- Half-day trading sessions are excluded (Paper, Section 4.1).

### Pseudocode

#### Notation and Indexing Conventions

The paper uses a unified linear time index tau = 1, 2, ..., N that flattens the
(day, bin) pairs into a single sequence. The mapping is:

    tau = (t - 1) * I + i

where t is the 1-based day index and i is the 1-based bin index within the day.
Total observations: N = T * I.

The inverse mapping:
    day(tau) = floor((tau - 1) / I) + 1
    bin(tau) = ((tau - 1) mod I) + 1

The set of day-boundary indices D identifies positions where the NEXT step
crosses a day boundary:
    D = {tau : tau mod I == 0 AND tau < N}
    D = {I, 2I, 3I, ..., (T-1)*I}

D has T-1 elements (one for each day-to-day transition). When tau is in D, the
transition from tau to tau+1 applies the day-boundary dynamics (eta transitions
with AR coefficient a^eta and receives process noise; when tau is NOT in D, eta
is held constant).

Notation conventions:
- x_{tau|tau}: filtered state estimate at tau given observations y_1, ..., y_tau
- x_{tau|tau-1}: predicted state estimate at tau given observations up to tau-1
- x_{tau|N}: smoothed state estimate at tau given ALL N observations
- Sigma_{tau|tau}: filtered state covariance
- Sigma_{tau|tau-1}: predicted state covariance
- Sigma_{tau|N}: smoothed state covariance

(Paper, Section 2, Equations 4-5; Section 2.2, Algorithm 1)

#### Algorithm 1: Kalman Filter (Prediction and Filtering)

This algorithm performs the forward pass: at each time step tau, it predicts
the next state, then corrects using the observed volume. It produces both
one-step-ahead forecasts and filtered state estimates.

```
FUNCTION kalman_filter(y[1..N], theta):
    # theta = {a_eta, a_mu, sigma_eta_sq, sigma_mu_sq, r, phi[1..I], pi_1, Sigma_1}
    # N = T * I = total number of bins in training window
    # I = number of bins per trading day

    # Observation matrix (constant, 1x2 row vector)
    C = [1, 1]

    # Storage for filtered estimates (needed by smoother)
    x_filt = array of N 2x1 vectors     # x_{tau|tau}
    Sigma_filt = array of N 2x2 matrices # Sigma_{tau|tau}
    x_pred = array of N 2x1 vectors     # x_{tau+1|tau} (predicted)
    Sigma_pred = array of N 2x2 matrices # Sigma_{tau+1|tau} (predicted)
    K_gains = array of N 2x1 vectors     # Kalman gains (needed by smoother)

    # Initialize: set x_{1|0} = pi_1, Sigma_{1|0} = Sigma_1
    x_pred[1] = pi_1       # 2x1 vector
    Sigma_pred[1] = Sigma_1 # 2x2 matrix

    FOR tau = 1 TO N:
        # --- Determine time-varying system matrices ---
        b = bin(tau)                     # 1-based bin index: ((tau-1) mod I) + 1
        phi_tau = phi[b]                 # seasonal component for this bin

        # --- Correction step (incorporate observation y_tau) ---
        # Innovation (scalar):
        innovation = y[tau] - C * x_pred[tau] - phi_tau
        #            = y[tau] - x_pred[tau][1] - x_pred[tau][2] - phi_tau

        # Innovation variance (scalar):
        # Paper, Algorithm 1, line 4:
        S_tau = C * Sigma_pred[tau] * C^T + r
        #     = Sigma_pred[tau][1,1] + Sigma_pred[tau][1,2]
        #       + Sigma_pred[tau][2,1] + Sigma_pred[tau][2,2] + r

        # Kalman gain (2x1 vector):
        # Paper, Algorithm 1, line 4:
        K_tau = Sigma_pred[tau] * C^T / S_tau
        #     = [Sigma_pred[tau][1,1] + Sigma_pred[tau][1,2],
        #        Sigma_pred[tau][2,1] + Sigma_pred[tau][2,2]]^T / S_tau

        # Filtered state estimate (2x1 vector):
        # Paper, Algorithm 1, line 5:
        x_filt[tau] = x_pred[tau] + K_tau * innovation

        # Filtered covariance (2x2 matrix):
        # Paper, Algorithm 1, line 6:
        Sigma_filt[tau] = Sigma_pred[tau] - K_tau * S_tau * K_tau^T
        # Equivalent: Sigma_filt[tau] = (I_2 - K_tau * C) * Sigma_pred[tau]

        # Store Kalman gain for smoother
        K_gains[tau] = K_tau

        # --- Prediction step (propagate to tau+1) ---
        IF tau < N:
            # Build A_tau (transition matrix for step tau -> tau+1)
            # Paper, Section 2, Equation 4:
            IF tau in D:   # day boundary: tau mod I == 0
                A_tau = [[a_eta, 0],
                         [0,     a_mu]]
                Q_tau = [[sigma_eta_sq, 0],
                         [0,            sigma_mu_sq]]
            ELSE:          # within-day: eta held constant
                A_tau = [[1,     0],
                         [0, a_mu]]
                Q_tau = [[0,            0],
                         [0, sigma_mu_sq]]

            # Predicted state for tau+1:
            # Paper, Algorithm 1, line 2:
            x_pred[tau+1] = A_tau * x_filt[tau]

            # Predicted covariance for tau+1:
            # Paper, Algorithm 1, line 3:
            Sigma_pred[tau+1] = A_tau * Sigma_filt[tau] * A_tau^T + Q_tau

    # --- Forecast generation ---
    # The one-step-ahead log-volume forecast at time tau+1, given data up to tau:
    # y_hat_{tau+1|tau} = C * x_pred[tau+1] + phi_{bin(tau+1)}
    #                   = x_pred[tau+1][1] + x_pred[tau+1][2] + phi[bin(tau+1)]

    RETURN x_filt, Sigma_filt, x_pred, Sigma_pred, K_gains
```

(Paper, Section 2.2, Algorithm 1, lines 1-7)

**Dynamic prediction mode:** For each bin tau in the out-of-sample period, run
both the prediction step (to produce the forecast y_hat_{tau|tau-1}) and the
correction step (to update the state with the observed y_tau). This yields
one-bin-ahead forecasts that incorporate the most recent observation.

**Static prediction mode:** At the end of training day T (i.e., after processing
tau = N = T*I), use the filtered state x_{N|N} to produce multi-step-ahead
forecasts for all I bins of day T+1 WITHOUT performing any correction steps.
For h = 1, 2, ..., I:

```
FUNCTION static_forecast(x_filtered_last, Sigma_filtered_last, theta, I):
    # x_filtered_last = x_{N|N}, the filtered state at end of training
    # Produce forecasts for bins 1..I of next day

    x_curr = x_filtered_last
    Sigma_curr = Sigma_filtered_last
    forecasts = array of I scalars

    FOR h = 1 TO I:
        # Determine transition type
        IF h == 1:
            # First prediction crosses day boundary (from last bin of day T)
            A = [[a_eta, 0], [0, a_mu]]
            Q = [[sigma_eta_sq, 0], [0, sigma_mu_sq]]
        ELSE:
            # Within-day prediction
            A = [[1, 0], [0, a_mu]]
            Q = [[0, 0], [0, sigma_mu_sq]]

        x_next = A * x_curr
        Sigma_next = A * Sigma_curr * A^T + Q

        forecasts[h] = C * x_next + phi[h]
        #            = x_next[1] + x_next[2] + phi[h]

        # For multi-step, no correction: carry forward predicted state
        x_curr = x_next
        Sigma_curr = Sigma_next

    RETURN forecasts
```

(Paper, Section 2.2, Equation 9)

#### Algorithm 2: Rauch-Tung-Striebel Smoother (Calibration E-step)

The backward smoother computes smoothed state estimates x_{tau|N} and
covariances Sigma_{tau|N} given all N observations. It runs backward from
tau = N to tau = 1, using the forward filter outputs.

```
FUNCTION kalman_smoother(x_filt, Sigma_filt, x_pred, Sigma_pred, theta, N, I):
    # Inputs: forward filter outputs from Algorithm 1
    # Outputs: smoothed estimates and cross-covariances needed by EM

    x_smooth = array of N 2x1 vectors      # x_{tau|N}
    Sigma_smooth = array of N 2x2 matrices  # Sigma_{tau|N}
    Sigma_cross = array of N-1 2x2 matrices # Sigma_{tau,tau-1|N}

    # Initialize: last time step is already smoothed
    # Paper, Algorithm 2, line (implied initialization):
    x_smooth[N] = x_filt[N]
    Sigma_smooth[N] = Sigma_filt[N]

    # Backward pass
    FOR tau = N-1 DOWNTO 1:
        # Build A_tau for step tau -> tau+1
        IF tau in D:   # tau mod I == 0
            A_tau = [[a_eta, 0], [0, a_mu]]
        ELSE:
            A_tau = [[1, 0], [0, a_mu]]

        # Smoother gain (2x2 matrix):
        # Paper, Algorithm 2, line 2:
        L_tau = Sigma_filt[tau] * A_tau^T * inverse(Sigma_pred[tau+1])

        # Smoothed state estimate:
        # Paper, Algorithm 2, line 3:
        x_smooth[tau] = x_filt[tau]
            + L_tau * (x_smooth[tau+1] - x_pred[tau+1])

        # Smoothed covariance:
        # Paper, Algorithm 2, line 4:
        Sigma_smooth[tau] = Sigma_filt[tau]
            + L_tau * (Sigma_smooth[tau+1] - Sigma_pred[tau+1]) * L_tau^T

    # Cross-covariance computation (needed for EM M-step)
    # Paper, Appendix A, Equations A.20-A.22:

    # Initialize at tau = N:
    # Paper, Equation A.21:
    # K_N is the Kalman gain at step N from the filter
    Sigma_cross[N-1] = (I_2 - K_gains[N] * C) * A_{N-1} * Sigma_filt[N-1]
    # where A_{N-1} is the transition matrix for step N-1 -> N

    # Backward recursion for cross-covariances:
    FOR tau = N-2 DOWNTO 1:
        # Paper, Equation A.20:
        # Build A_tau for step tau -> tau+1
        IF tau in D:
            A_tau = [[a_eta, 0], [0, a_mu]]
        ELSE:
            A_tau = [[1, 0], [0, a_mu]]

        # L_{tau+1} is the smoother gain computed above for tau+1
        L_tau_plus_1 = ... # (stored from smoothing loop above)

        Sigma_cross[tau] = Sigma_filt[tau+1] * L_tau^T  # Researcher note: see detailed formula below
            # Actually: Sigma_cross uses the recursive formula from Eq A.20

    # DETAILED CROSS-COVARIANCE FORMULA (Paper, Appendix A, Eq A.20):
    # Sigma_{tau,tau-1|N} = Sigma_{tau|tau} * L_{tau-1}^T
    #     + L_tau * (Sigma_{tau+1,tau|N} - A_tau * Sigma_{tau|tau}) * L_{tau-1}^T
    #
    # This is computed backward from tau = N-1 downto 2.
    # Initialize: Sigma_{N,N-1|N} = (I_2 - K_N * C) * A_{N-1} * Sigma_{N-1|N-1}
    #   (Paper, Eq A.21)

    RETURN x_smooth, Sigma_smooth, Sigma_cross, L_gains
```

(Paper, Section 2.3.1, Algorithm 2; Appendix A.2, Equations A.18-A.22)

**Implementation note on cross-covariances:** The paper defines the cross-covariance
Sigma_{tau,tau-1|N} as E[x_tau * x_{tau-1}^T | {y}_{1..N}] and provides:

    Initialization (tau = N):
        Sigma_{N,N-1|N} = (I_2 - K_N * C) * A_{N-1} * Sigma_{N-1|N-1}

    Backward recursion (tau = N-1 downto 2):
        Sigma_{tau,tau-1|N} = Sigma_{tau|tau} * L_{tau-1}^T
            + L_tau * (Sigma_{tau+1,tau|N} - A_tau * Sigma_{tau|tau}) * L_{tau-1}^T

where L_tau is the smoother gain for step tau. This recursive formula (Eq A.20)
builds the full set of cross-covariances needed by the EM M-step.

#### Algorithm 3: EM Algorithm (Calibration)

The EM algorithm iterates between running the forward filter + backward smoother
(E-step) and updating all parameters in closed form (M-step). It produces the
maximum-likelihood parameter estimates.

```
FUNCTION em_algorithm(y[1..N], theta_init, I, max_iter, tol):
    # theta_init: initial parameter values
    # I: bins per day, N = T * I
    # max_iter: maximum EM iterations (e.g., 50)
    # tol: convergence tolerance on log-likelihood change

    theta = theta_init
    T = N / I              # number of training days

    FOR j = 1 TO max_iter:
        # === E-step ===
        # Run forward Kalman filter (Algorithm 1):
        x_filt, Sigma_filt, x_pred, Sigma_pred, K_gains =
            kalman_filter(y, theta)

        # Run backward smoother (Algorithm 2):
        x_smooth, Sigma_smooth, Sigma_cross, L_gains =
            kalman_smoother(x_filt, Sigma_filt, x_pred, Sigma_pred, theta, N, I)

        # Compute sufficient statistics:
        # Paper, Appendix A, Equations A.15-A.17:
        # x_hat_tau = x_smooth[tau]                   (smoothed state mean)
        # P_tau = Sigma_smooth[tau] + x_smooth[tau] * x_smooth[tau]^T
        #       = E[x_tau * x_tau^T | y_{1..N}]
        # P_{tau,tau-1} = Sigma_cross[tau-1] + x_smooth[tau] * x_smooth[tau-1]^T
        #               = E[x_tau * x_{tau-1}^T | y_{1..N}]

        # Define sufficient statistics as 2x2 matrices:
        FOR tau = 1 TO N:
            P[tau] = Sigma_smooth[tau] + x_smooth[tau] * x_smooth[tau]^T

        FOR tau = 2 TO N:
            P_cross[tau] = Sigma_cross[tau-1] + x_smooth[tau] * x_smooth[tau-1]^T

        # Notation for component access:
        # P[tau]^{(1,1)} = P[tau][1,1] (eta-eta block)
        # P[tau]^{(2,2)} = P[tau][2,2] (mu-mu block)
        # P[tau]^{(1,2)} = P[tau][1,2] (eta-mu cross block, scalar here)
        # P_cross[tau]^{(1,1)} = P_cross[tau][1,1]
        # P_cross[tau]^{(2,2)} = P_cross[tau][2,2]
        # etc.

        # === M-step (all closed-form updates) ===

        # Initial state mean (Paper, Eq A.32 / Eq 17):
        pi_1_new = x_smooth[1]

        # Initial state covariance (Paper, Eq A.33 / Eq 18):
        Sigma_1_new = P[1] - x_smooth[1] * x_smooth[1]^T
        #           = Sigma_smooth[1]

        # AR coefficient for daily component (Paper, Eq A.34 / Eq 19):
        # Sum over day-boundary indices only (tau in D, i.e., tau = kI+1 means
        # the step FROM tau-1=kI TO tau=kI+1 is a day boundary).
        # D_plus = {tau : tau-1 in D} = {kI+1 : k=1,...,T-1}
        # These are the tau values where eta actually transitions.
        num_a_eta = sum over tau in D_plus of P_cross[tau]^{(1,1)}
        den_a_eta = sum over tau in D_plus of P[tau-1]^{(1,1)}
        a_eta_new = num_a_eta / den_a_eta

        # AR coefficient for intraday dynamic (Paper, Eq A.35 / Eq 20):
        num_a_mu = sum_{tau=2}^{N} P_cross[tau]^{(2,2)}
        den_a_mu = sum_{tau=2}^{N} P[tau-1]^{(2,2)}
        a_mu_new = num_a_mu / den_a_mu

        # Process noise variance for daily component (Paper, Eq A.36 / Eq 21):
        # |D| = T - 1 (number of day transitions)
        sigma_eta_sq_new = (1 / (T - 1)) * sum over tau in D_plus of
            { P[tau]^{(1,1)} + a_eta_new^2 * P[tau-1]^{(1,1)}
              - 2 * a_eta_new * P_cross[tau]^{(1,1)} }

        # Process noise variance for dynamic component (Paper, Eq A.37 / Eq 22):
        sigma_mu_sq_new = (1 / (N - 1)) * sum_{tau=2}^{N} of
            { P[tau]^{(2,2)} + a_mu_new^2 * P[tau-1]^{(2,2)}
              - 2 * a_mu_new * P_cross[tau]^{(2,2)} }

        # Observation noise variance (Paper, Eq A.38 / Eq 23):
        r_new = (1 / N) * sum_{tau=1}^{N} of
            { y[tau]^2 + C * P[tau] * C^T - 2 * y[tau] * C * x_smooth[tau]
              + (phi_new[bin(tau)])^2
              - 2 * y[tau] * phi_new[bin(tau)]
              + 2 * phi_new[bin(tau)] * C * x_smooth[tau] }

        # Note: phi_new must be computed before r_new, or they can be
        # computed jointly. The paper computes phi first.

        # Seasonality vector (Paper, Eq A.39 / Eq 24):
        FOR i = 1 TO I:
            phi_new[i] = (1 / T) * sum_{t=1}^{T} of
                ( y_{t,i} - C * x_smooth[tau(t,i)] )
            # where tau(t,i) = (t-1)*I + i is the linear index for day t, bin i
            # C * x_smooth[tau] = x_smooth[tau][1] + x_smooth[tau][2]

        # Update theta:
        theta = {a_eta_new, a_mu_new, sigma_eta_sq_new, sigma_mu_sq_new,
                 r_new, phi_new, pi_1_new, Sigma_1_new}

        # Check convergence:
        # Compute log-likelihood (Paper, Eq A.8) or monitor parameter changes.
        # Terminate if relative change in log-likelihood < tol.

    RETURN theta
```

(Paper, Section 2.3.2, Algorithm 3; Appendix A.3, Equations A.32-A.39)

**Critical implementation detail on the M-step summation indices:** The M-step
updates for a^eta and sigma_eta_sq sum only over day-boundary transitions. In the
paper's notation, these correspond to tau values in {kI+1 : k = 1, ..., T-1},
i.e., the first bin of each day starting from day 2. The a^mu and sigma_mu_sq
updates sum over ALL consecutive pairs tau = 2, ..., N. The r and phi updates
sum over all N observations. Getting the summation ranges wrong is the most
common implementation error.

(Paper, Algorithm 3, lines 12-15 vs lines 13-15; Appendix A, Equations A.26 vs A.27)

#### Algorithm 4: Robust Kalman Filter (Lasso Extension)

The robust variant adds a sparse noise term z_tau to the observation equation to
handle outliers:

    y_tau = C * x_tau + phi_tau + v_tau + z_tau

where z_tau is zero most of the time and nonzero only when an outlier occurs.
The Lasso penalty encourages sparsity in z_tau.

The modification affects ONLY the correction step of the Kalman filter. The
prediction step is unchanged.

```
FUNCTION robust_kalman_correction(y_tau, x_pred_tau, Sigma_pred_tau, phi_tau,
                                   r, C, lambda):
    # Modified correction step for the robust Kalman filter
    # Paper, Section 3.1, Equations 29-34

    # Innovation (before accounting for sparse noise):
    e_tau = y_tau - phi_tau - C * x_pred_tau
    # e_tau is a scalar

    # Innovation precision (scalar):
    # Paper, Section 3.1, Equation 30:
    W_tau = (C * Sigma_pred_tau * C^T + r)^{-1}
    # W_tau = 1 / (Sigma_pred_tau[1,1] + Sigma_pred_tau[1,2]
    #              + Sigma_pred_tau[2,1] + Sigma_pred_tau[2,2] + r)

    # Solve Lasso-penalized correction:
    # Paper, Equation 30:
    # min_{z_tau} (e_tau - z_tau)^T * W_tau * (e_tau - z_tau)
    #             + v_tau^T * r^{-1} * v_tau + lambda * |z_tau|
    #
    # For scalar z_tau, this has closed-form soft-thresholding solution:

    # Threshold:
    threshold = lambda / (2 * W_tau)

    # Optimal sparse noise (soft-thresholding):
    # Paper, Equation 33:
    IF e_tau > threshold:
        z_star = e_tau - threshold
    ELIF e_tau < -threshold:
        z_star = e_tau + threshold
    ELSE:
        z_star = 0

    # The residual after removing the outlier:
    # Paper, Equation 34:
    e_clean = e_tau - z_star
    # Equivalently:
    # IF |e_tau| > threshold:  e_clean = sign(e_tau) * threshold
    # ELSE:                    e_clean = e_tau

    # Standard Kalman correction using cleaned innovation:
    # Paper, Equation 31-32:
    S_tau = 1 / W_tau   # = C * Sigma_pred_tau * C^T + r
    K_tau = Sigma_pred_tau * C^T / S_tau

    x_filt_tau = x_pred_tau + K_tau * e_clean
    Sigma_filt_tau = Sigma_pred_tau - K_tau * S_tau * K_tau^T

    RETURN x_filt_tau, Sigma_filt_tau, K_tau, z_star
```

(Paper, Section 3.1, Equations 29-34)

**Interpretation:** The threshold lambda/(2*W_tau) adapts dynamically because
W_tau depends on the current predictive variance. When the model is uncertain
(large predictive variance, small W_tau), the threshold is larger and the filter
is more tolerant of large innovations. When the model is confident (small
predictive variance, large W_tau), the threshold is smaller and outliers are
detected more aggressively. This is a natural adaptive mechanism.

(Paper, Section 3.1, paragraph after Equation 34)

#### Algorithm 5: Robust EM Calibration

The EM algorithm for the robust model is identical to Algorithm 3 except:

1. The E-step uses the robust Kalman filter (Algorithm 4) instead of the
   standard filter, storing z_star[tau] for each time step.

2. The M-step updates for r and phi are modified to account for the inferred
   outlier values:

```
# Modified observation noise variance (Paper, Equation 35):
r_new = (1 / N) * sum_{tau=1}^{N} of
    { y[tau]^2 + C * P[tau] * C^T - 2 * y[tau] * C * x_smooth[tau]
      + (phi_new[bin(tau)])^2 - 2 * (z_star[tau])^2
      - 2 * y[tau] * phi_new[bin(tau)]
      + 2 * z_star[tau] * C * x_smooth[tau]
      + 2 * z_star[tau] * phi_new[bin(tau)]^{(j+1)}
      - 2 * z_star[tau] * y[tau] }

# Simplified from Paper Eq 35:
# r_new = (1/N) * sum_{tau=1}^{N} [
#     y_tau^2 + C*P_tau*C^T - 2*y_tau*C*x_hat_tau
#     + (phi_new)^2 + (z_star)^2
#     + 2*z_star*C*x_hat_tau + 2*z_star*phi_new
#     - 2*y_tau*phi_new - 2*z_star*y_tau
# ]

# Modified seasonality (Paper, Equation 36):
FOR i = 1 TO I:
    phi_new[i] = (1 / T) * sum_{t=1}^{T} of
        ( y_{t,i} - C * x_smooth[tau(t,i)] - z_star[tau(t,i)] )
```

(Paper, Section 3.2, Equations 35-36)

All other M-step updates (a_eta, a_mu, sigma_eta_sq, sigma_mu_sq, pi_1,
Sigma_1) remain unchanged because the sparse noise z_tau only appears in the
observation equation, not the state transition.

**Smoother interaction with robust filter:** When running the smoother in the
robust EM, the smoother still uses the standard RTS equations (Algorithm 2),
but operates on the filtered estimates produced by the robust Kalman filter.
The z_star values from the filter forward pass are stored and used only in the
M-step updates for r and phi. (Researcher inference: the paper does not
explicitly address whether the smoother needs modification for the robust case.
Since the smoother operates on the state-space model {A, C, Q, r} and the robust
filter produces valid filtered state estimates and covariances, the standard
smoother equations apply. The z_star values are treated as fixed known
quantities in the M-step, not as additional hidden states.)

#### Algorithm 6: VWAP Execution Strategies

Two VWAP execution strategies convert volume forecasts into execution weights.

```
FUNCTION static_vwap_weights(volume_forecasts_static[1..I]):
    # Paper, Equation 40
    # volume_forecasts_static are in log space; exponentiate first
    vol_linear = array of I values
    FOR i = 1 TO I:
        vol_linear[i] = exp(volume_forecasts_static[i])

    total = sum(vol_linear)
    weights = array of I values
    FOR i = 1 TO I:
        weights[i] = vol_linear[i] / total

    RETURN weights


FUNCTION dynamic_vwap_weights(volume_forecasts_dynamic, observed_volumes,
                               current_bin):
    # Paper, Equation 41
    # At bin i (1-indexed), we have observed bins 1..i-1 and forecast bins i..I
    # volume_forecasts_dynamic[i] is the one-step-ahead forecast for bin i
    # observed_volumes[1..i-1] are actual volumes (linear scale) for completed bins

    # For bins i = 1, ..., I-1:
    #   w_{t,i} = forecast_volume_{t,i} / sum_{j=i}^{I} forecast_volume_{t,j}
    #             * (1 - sum_{j=1}^{i-1} w_{t,j})
    #
    # For the last bin i = I:
    #   w_{t,I} = 1 - sum_{j=1}^{I-1} w_{t,j}

    # Implementation: at each bin, the weight is the proportion of predicted
    # remaining volume that this bin represents, scaled by the remaining
    # execution fraction.

    i = current_bin
    IF i == I:
        weight = 1.0 - sum of weights assigned to bins 1..I-1
    ELSE:
        # Predicted volume for bin i (linear scale):
        pred_i = exp(volume_forecasts_dynamic[i])
        # Sum of predicted volumes for remaining bins i..I:
        remaining_pred = sum_{j=i}^{I} exp(volume_forecasts_dynamic[j])
        # Fraction already executed:
        already_executed = sum of weights assigned to bins 1..i-1
        # Weight for this bin:
        weight = (pred_i / remaining_pred) * (1.0 - already_executed)

    RETURN weight
```

(Paper, Section 4.3, Equations 40-41)

### Data Flow

```
Input: raw_volume[t, i] for t = 1..T, i = 1..I
       shares_outstanding[t] for t = 1..T

Step 1: Normalize
    normalized[t, i] = raw_volume[t, i] / shares_outstanding[t]

Step 2: Log-transform
    y[t, i] = log(normalized[t, i])
    Flatten to y[tau] for tau = 1..N where tau = (t-1)*I + i

Step 3: Calibrate (EM Algorithm, Algorithm 3 or 5)
    Input:  y[1..N_train] (training window, N_train = T_train * I)
    Output: theta = {a_eta, a_mu, sigma_eta_sq, sigma_mu_sq, r,
                     phi[1..I], pi_1, Sigma_1, [lambda if robust]}

Step 4a: Predict (Dynamic mode)
    FOR each out-of-sample bin tau:
        Run one prediction step -> y_hat_{tau|tau-1}
        Observe y_tau
        Run one correction step -> x_{tau|tau}
    Output: log-volume forecasts y_hat[tau] for each out-of-sample bin

Step 4b: Predict (Static mode)
    At end of training, run multi-step prediction (Algorithm static_forecast)
    Output: log-volume forecasts y_hat[1..I] for next full day

Step 5: Convert to linear scale
    volume_forecast[tau] = exp(y_hat[tau]) * shares_outstanding[day(tau)]

Step 6: Compute VWAP weights (Algorithm 6)
    Static: weights from full-day static forecasts
    Dynamic: weights updated after each bin observation

Data shapes at each step:
    raw_volume:       T x I matrix of positive reals (shares traded)
    shares_outstanding: T-length vector of positive integers
    normalized:       T x I matrix of small positive reals (typically 0.001 to 1.0)
    y (log-volume):   N-length vector of reals (typically -8 to 2)
    theta:            parameter struct (see Parameters section)
    x_filt:           N-length array of 2x1 vectors
    Sigma_filt:       N-length array of 2x2 symmetric PD matrices
    y_hat:            forecast-length vector of reals
    volume_forecast:  forecast-length vector of positive reals
    weights:          I-length vector on the simplex (sum = 1)
```

### Variants

**Implemented variant:** The robust Kalman filter with Lasso regularization
(Section 3 of the paper) is the primary implementation target. This subsumes the
standard Kalman filter as the special case lambda = infinity (no outliers ever
detected) or equivalently lambda = 0 with z_star always zero when the data is
clean.

**Rationale:** The robust variant achieves the best empirical performance (Table 3:
average MAPE 0.46 dynamic, 0.61 static) and, critically, maintains performance
under data contamination while the standard Kalman filter and CMEM degrade
significantly (Table 1). For production use with non-curated real-time market
data, robustness to outliers is essential.

The standard (non-robust) Kalman filter should also be available as a
configuration option (set lambda to a very large value or skip the
soft-thresholding step).

(Paper, Section 3.3, Table 1; Section 4.2, Table 3)

## Parameters

| Parameter | Description | Recommended Value | Sensitivity | Range |
|-----------|-------------|-------------------|-------------|-------|
| a_eta | AR(1) coefficient for daily component across days | Data-driven via EM (typically close to 1.0 based on Figure 4a convergence plot) | Medium -- controls daily mean-reversion speed | (0, 1) for stationarity; empirically 0.8-1.0 |
| a_mu | AR(1) coefficient for intraday dynamic across bins | Data-driven via EM (typically 0.3-0.7 based on Figure 4b convergence plot) | Medium -- controls intraday dynamic persistence | (0, 1) for stationarity; empirically 0.2-0.8 |
| sigma_eta_sq | Process noise variance for daily component | Data-driven via EM | Low -- adapts to data scale | (0, inf); typically small relative to r |
| sigma_mu_sq | Process noise variance for intraday dynamic | Data-driven via EM | Low -- adapts to data scale | (0, inf); typically comparable to r |
| r | Observation noise variance | Data-driven via EM | Medium -- affects filter responsiveness | (0, inf); typically 0.01-1.0 for log-volume |
| phi[1..I] | Intraday seasonality vector (one per bin) | Data-driven via EM; exhibits characteristic U-shape | High -- captures the dominant intraday pattern | Unrestricted reals; typically -3 to 0 for normalized log-volume |
| pi_1 | Initial state mean (2x1 vector) | Data-driven via EM; initialized to [mean(y), 0]^T | Low -- EM converges regardless of initialization (Figure 4) | Unrestricted reals |
| Sigma_1 | Initial state covariance (2x2 PSD matrix) | Data-driven via EM; initialized to diagonal with var(y) | Low -- EM converges regardless of initialization | Symmetric positive definite |
| lambda | Lasso regularization for robust filter | Selected by cross-validation | High for robust mode -- controls outlier sensitivity | (0, inf); larger = fewer outliers detected |
| N_train | Training window length (number of bins) | Selected by cross-validation; paper uses Jan 2013 to May 2015 (~600 days, varying by ticker) | Medium -- too short = poor estimation, too long = stale parameters | 100*I to 1000*I (100 to 1000 trading days) |
| I | Number of intraday bins per day | Exchange-dependent: 26 for NYSE 6.5-hr session at 15-min bins | Fixed by data -- not tunable | Determined by exchange hours / bin size |
| max_iter | Maximum EM iterations | 50 | Low -- EM typically converges in 5-15 iterations (Figure 4) | 10-100 |
| tol | EM convergence tolerance | 1e-6 (relative log-likelihood change) | Low | 1e-8 to 1e-4 |

(Paper, Section 2.3.3, Figure 4 for convergence behavior; Section 4.1 for data setup;
Section 4.2 for cross-validation description)

### Initialization

**EM initialization (theta_init):** The EM algorithm is robust to initialization
choice (Paper, Section 2.3.3, Figure 4). Recommended initialization:

1. **phi_init[i]:** For each bin i, compute the mean of y_{t,i} across all
   training days T. This gives a reasonable starting point for the seasonal
   pattern.

2. **pi_1:** Set to [mean(y - phi_init), 0]^T. The first component approximates
   the initial daily level; the second component (mu) starts at zero (no
   initial dynamic deviation).

3. **Sigma_1:** Set to diag(var(y), var(y)). A diffuse prior reflecting initial
   uncertainty.

4. **a_eta, a_mu:** Initialize to 0.5 (mid-range; EM will find the correct
   values regardless).

5. **sigma_eta_sq, sigma_mu_sq:** Initialize to var(y) / 3 (roughly partition
   total variance among three noise sources).

6. **r:** Initialize to var(y) / 3.

(Paper, Section 2.3.3 -- demonstrates convergence from diverse initial values)

**Kalman filter initialization (for prediction):** The filter starts with the
EM-estimated pi_1 and Sigma_1. For rolling-window operation, when recalibrating
on a new window, the EM re-estimates these from data.

### Calibration

The calibration procedure is:

1. **Select training window:** Choose N_train bins (T_train days) of historical
   data ending before the forecast period.

2. **Preprocess:** Normalize by shares outstanding, take logs, flatten to a
   single vector y[1..N_train]. Verify no zero-volume bins exist.

3. **Initialize theta** as described above.

4. **Run EM (Algorithm 3 or 5)** until convergence (relative log-likelihood
   change < tol or max_iter reached).

5. **Cross-validate N_train and lambda:**
   - Hold out a validation period (e.g., 5 months before the forecast period;
     Paper uses January 2015 to May 2015).
   - For each candidate (N_train, lambda) pair:
     a. Calibrate on the N_train bins ending before the validation period.
     b. Produce dynamic one-step-ahead forecasts on the validation period.
     c. Compute MAPE on the validation period.
   - Select the (N_train, lambda) pair with minimum validation MAPE.

6. **Recalibrate** with the optimal (N_train, lambda) on the final training
   window and produce out-of-sample forecasts.

**Rolling recalibration:** For production use, recalibrate periodically (e.g.,
weekly or monthly) by shifting the training window forward. The paper uses a
fixed training window for the entire out-of-sample period (Section 4.1), but
rolling recalibration is standard practice and recommended.

(Paper, Section 4.1 for data setup; Section 4.2 for cross-validation)

## Validation

### Expected Behavior

**Volume prediction accuracy (MAPE):**
- Dynamic prediction (one-bin-ahead): average MAPE of 0.46 for robust Kalman
  filter, 0.47 for standard Kalman filter across 30 securities on 8 exchanges
  (Paper, Section 4.2, Table 3, "Average" row).
- Static prediction (full-day-ahead): average MAPE of 0.61 for robust Kalman
  filter (Paper, Section 4.2, Table 3).
- For comparison: CMEM achieves 0.65 dynamic / 0.90 static; rolling mean
  achieves 1.28 for both (Paper, Table 3).
- Individual ticker variation is substantial: MAPE ranges from 0.21 (AAPL
  dynamic) to 1.94 (2800HK static) (Paper, Table 3).

**VWAP tracking error:**
- Dynamic VWAP: average 6.38 basis points for robust Kalman filter
  (Paper, Section 4.3, Table 4, "Average" row).
- Static VWAP: average 6.85 basis points (Paper, Table 4).
- For comparison: CMEM achieves 8.97 dynamic / 10.91 static; rolling mean
  achieves 7.48 / 11.16 (Paper, Table 4).

**EM convergence:**
- Parameters converge within 5-15 iterations from diverse initial values
  (Paper, Section 2.3.3, Figure 4).
- The converged parameter values are insensitive to initialization -- different
  starting points yield the same final estimates (Paper, Figure 4).

(Paper, Section 4.2, Tables 1 and 3; Section 4.3, Table 4; Section 2.3.3, Figure 4)

### Sanity Checks

1. **Seasonality shape:** After calibration, phi[1..I] should exhibit the
   characteristic U-shape (high values at market open and close, lower values
   mid-day). Plot phi and verify visually. For US equities on NYSE with I=26
   bins, expect phi to be highest at bins 1-2 and 25-26.
   (Paper, Section 2 -- describes the U-shaped intraday pattern)

2. **AR coefficient ranges:** After EM convergence, a_eta should be close to 1.0
   (daily levels are persistent) and a_mu should be between 0.2 and 0.8
   (intraday dynamics are moderately persistent but mean-reverting).
   (Paper, Section 2.3.3, Figure 4a-b)

3. **EM convergence:** Log-likelihood should increase monotonically at each EM
   iteration. If it decreases, there is a bug in the E-step or M-step.
   (Standard EM property)

4. **Filter innovation statistics:** The innovations (y_tau - C * x_{tau|tau-1}
   - phi_{bin(tau)}) should be approximately zero-mean and uncorrelated if the
   model is correctly specified. Compute the sample mean and ACF of innovations.
   (Standard Kalman filter diagnostic)

5. **Robust filter outlier rate:** With the robust filter on clean (curated) data,
   z_star should be nonzero for a small fraction of bins (< 5%). On
   artificially contaminated data with 10% outlier rate, the detected fraction
   should be close to 10%.
   (Paper, Section 3.3 -- 10% contamination rate used in simulations)

6. **Forecast bias:** The mean forecast error (y_tau - y_hat_tau) should be
   approximately zero over a large out-of-sample period. Systematic bias
   indicates a calibration problem.

7. **Covariance positive definiteness:** Sigma_pred and Sigma_filt must remain
   positive definite throughout the filter. If eigenvalues approach zero or go
   negative, the implementation has a numerical stability issue.

### Edge Cases

1. **Zero-volume bins:** log(0) is undefined. The model cannot handle zero-volume
   bins. Options:
   - Exclude zero-volume bins and treat them as missing observations. In the
     Kalman filter, skip the correction step for missing bins (only run the
     prediction step). This preserves the state estimate without incorporating
     the missing observation.
   - Impute a small positive volume (e.g., 1 share) before log-transforming.
     This is a hack and may introduce bias.
   - Restrict the model to liquid securities where zero-volume bins are rare.
   (Paper, Section 4.1 -- explicitly excludes zero-volume bins)

2. **Day boundaries:** The transition matrix A_tau changes at day boundaries.
   The implementation must correctly identify when tau is the last bin of a day
   (tau mod I == 0) and apply the day-boundary dynamics for the transition to
   the next bin. Off-by-one errors here are critical and will cause incorrect
   daily component evolution.

3. **Half-day sessions:** Trading days with fewer than I bins (e.g., day before
   holidays in US markets) should be excluded entirely. Including them would
   misalign the bin indexing and corrupt the seasonal estimates.
   (Paper, Section 4.1)

4. **Numerical stability of 2x2 inverse:** The smoother gain L_tau requires
   inverting Sigma_pred[tau+1], which is a 2x2 matrix. Use the explicit 2x2
   inverse formula (ad-bc determinant) rather than a general matrix inverse.
   If the determinant is near zero, the filter has diverged.

5. **EM degenerate solutions:** If sigma_eta_sq or sigma_mu_sq converge to zero,
   the model collapses a component. This is not necessarily wrong (it means the
   data does not support that component) but should be flagged.
   Clamp variances to a small positive minimum (e.g., 1e-10) to avoid
   numerical issues.

6. **Very large lambda:** As lambda -> infinity, the robust filter never detects
   outliers (z_star = 0 always) and reduces to the standard Kalman filter.
   This is expected behavior, not an error.

7. **Cross-exchange differences:** The number of bins per day I varies by
   exchange (e.g., NYSE = 26, TYO = 30 for 4.5-hour session at 15-min bins).
   The implementation must accept I as a parameter, not hard-code it.
   (Paper, Table 2 -- multiple exchanges with different I values)

### Known Limitations

1. **No zero-volume support:** The model is fundamentally incompatible with
   zero-volume bins due to the log transformation. This limits applicability
   to liquid securities. (Paper, Section 4.1)

2. **Single-security model:** Unlike the BDF model (Direction 2), this model
   operates on a single security at a time. Cross-sectional information is
   not exploited. (Paper, throughout)

3. **No exogenous variables:** The model does not incorporate external information
   such as volatility, spreads, order flow, or calendar effects. Future work
   identified by the authors includes adding covariates.
   (Paper, Section 5, Conclusion)

4. **Gaussian assumption:** The model assumes Gaussian noise. While the log
   transformation makes this more reasonable, heavy-tailed distributions or
   time-varying volatility in the residuals could violate this assumption.
   The robust extension partially mitigates this for outliers but does not
   address systematic non-Gaussianity.

5. **Fixed bin size:** The model assumes a fixed, regular bin size throughout.
   Variable-length bins (e.g., auctions, irregular intervals) are not
   supported. (Paper, Section 2)

6. **No intraday updating of daily component:** Within a trading day, eta is
   held strictly constant. The model cannot revise its estimate of the daily
   level based on intraday evidence until the next day boundary. This means
   if the first few bins reveal that today's volume is much higher or lower
   than expected, the model adapts only through mu (intraday dynamic), not
   through eta. (Paper, Section 2, Equations 4-5)

7. **Linear state-space structure:** The AR(1) dynamics for both eta and mu are
   restrictive. Higher-order AR processes or nonlinear dynamics (such as the
   SETAR model in Direction 2) might capture more complex patterns but would
   break the linear Gaussian framework and the closed-form EM.

## Paper References

| Spec Section | Paper Source |
|---|---|
| Model Description (observation equation) | Chen et al. (2016), Section 2, Equation 3 |
| State-space formulation | Chen et al. (2016), Section 2, Equations 4-5 |
| Log-volume motivation | Chen et al. (2016), Section 2, Figure 1; references Ajinkya and Jain (1989) |
| Multiplicative decomposition origin | Brownlees, Cipollini, Gallo (2011), Section 2 |
| Algorithm 1 (Kalman filter) | Chen et al. (2016), Section 2.2, Algorithm 1 |
| Multi-step prediction | Chen et al. (2016), Section 2.2, Equation 9 |
| Algorithm 2 (Smoother) | Chen et al. (2016), Section 2.3.1, Algorithm 2 |
| Cross-covariance recursion | Chen et al. (2016), Appendix A.2, Equations A.20-A.22 |
| Algorithm 3 (EM) | Chen et al. (2016), Section 2.3.2, Algorithm 3 |
| EM closed-form M-step derivations | Chen et al. (2016), Appendix A.3, Equations A.24-A.39 |
| Sufficient statistics definitions | Chen et al. (2016), Appendix A.2, Equations A.15-A.17 |
| Joint log-likelihood | Chen et al. (2016), Appendix A.1, Equation A.8 |
| EM convergence demonstration | Chen et al. (2016), Section 2.3.3, Figure 4 |
| Robust Kalman filter (Lasso) | Chen et al. (2016), Section 3.1, Equations 25-34 |
| Robust EM modifications | Chen et al. (2016), Section 3.2, Equations 35-36 |
| Robustness simulations | Chen et al. (2016), Section 3.3, Table 1 |
| Data description and setup | Chen et al. (2016), Section 4.1, Table 2 |
| MAPE definition | Chen et al. (2016), Section 3.3, Equation 37 |
| Cross-validation procedure | Chen et al. (2016), Section 4.2 (paragraph on N and lambda selection) |
| Volume prediction results | Chen et al. (2016), Section 4.2, Table 3 |
| VWAP tracking error definition | Chen et al. (2016), Section 4.3, Equation 42 |
| VWAP static weights | Chen et al. (2016), Section 4.3, Equation 40 |
| VWAP dynamic weights | Chen et al. (2016), Section 4.3, Equation 41 |
| VWAP tracking results | Chen et al. (2016), Section 4.3, Table 4 |
| Lasso reference | Tibshirani (1996) via Chen et al. (2016), Section 3.1 |
| EM framework reference | Shumway and Stoffer (1982) via Chen et al. (2016), Section 2.3.2 |
| Efficient Lasso-in-Kalman | Mattingley and Boyd (2010) via Chen et al. (2016), Section 3.1 |
