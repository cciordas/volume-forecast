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

Data requirements:
- Intraday volume per bin (shares traded) for T training days at I bins per day.
- Daily shares outstanding per security (needed for normalization; changes due
  to buybacks, issuances, and splits must be tracked).
- Half-day trading sessions should be excluded (Paper, Section 4.1).

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
tau = kI for k = 1, 2, ..., T-1 (i.e., tau values that are the last bin of
each day EXCEPT the last day; the transition FROM these indices TO tau+1
applies the day-boundary dynamics). Note: D has T-1 elements because there
are T-1 day-to-day transitions in T days.

Throughout this spec:
- "tau in D" means tau is a day-boundary index (tau mod I == 0 AND tau < N),
  meaning the NEXT step (tau -> tau+1) crosses a day boundary.
- Subscript notation: x_{tau|tau} means the filtered estimate at tau given
  observations up to tau; x_{tau|N} means the smoothed estimate at tau given
  all N observations.
- bin(tau) = ((tau - 1) mod I) + 1: the 1-based bin index within the day.

(Paper, Section 2, Equations 4-5; Section 2.2, Algorithm 1)

#### Algorithm 1: Kalman Filter (Prediction)

```
FUNCTION kalman_filter(y[1..N], theta):
    # theta = {a_eta, a_mu, sigma_eta_sq, sigma_mu_sq, r, phi[1..I], pi_1, Sigma_1}
    # N = total number of bins in training/prediction window
    # I = number of bins per trading day

    # Initialize
    C = [1, 1]                            # 1x2 observation matrix (row vector)

    FOR tau = 1 TO N:
        # --- Prediction step ---
        IF tau == 1:
            # First bin: use initial state directly, no transition applied.
            # pi_1 is the prior mean for x_1 (EM sets pi_1 = x_hat_1, the
            # smoothed estimate at tau=1; see Eq A.32). No A is applied.
            x_pred = pi_1                     # 2x1
            Sigma_pred = Sigma_1              # 2x2
        ELSE:
            # Build transition matrix A based on whether tau-1 is in D
            # (i.e., whether the step from tau-1 to tau crosses a day boundary)
            IF (tau - 1) mod I == 0:
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
        END IF

        # --- Determine phi for this bin ---
        phi_tau = phi[bin(tau)]

        # --- Correction step (if observation available) ---
        IF y[tau] is observed:
            # Innovation
            innovation = y[tau] - C @ x_pred - phi_tau    # scalar

            # Innovation variance (scalar since observation is 1D)
            S_tau = C @ Sigma_pred @ C^T + r              # scalar > 0

            # Kalman gain
            K = Sigma_pred @ C^T / S_tau                  # 2x1

            # Update state
            x_hat = x_pred + K * innovation               # 2x1
            Sigma = (I_2 - K @ C) @ Sigma_pred @ (I_2 - K @ C)^T + K * r * K^T
                                                          # 2x2 (Joseph form)
        ELSE:
            # Static prediction: skip correction, propagate prediction
            x_hat = x_pred
            Sigma = Sigma_pred
            K = zeros(2, 1)
        END IF

        # Store for output (all needed by smoother)
        x_filtered[tau] = x_hat
        Sigma_filtered[tau] = Sigma
        Sigma_predicted[tau] = Sigma_pred    # needed by smoother for L_tau
        K_stored[tau] = K                    # needed by cross-covariance init
        A_stored[tau] = A if tau > 1 else I_2  # transition used at this step

        # Log-volume forecast (one-step-ahead, before correction):
        y_hat[tau] = C @ x_pred + phi_tau
    END FOR

    RETURN x_filtered, Sigma_filtered, Sigma_predicted, K_stored, A_stored, y_hat
END FUNCTION
```

(Paper, Section 2.2, Algorithm 1, Equations 7-9)

**Note on initialization:** pi_1 represents the prior mean for x_1, not a
predicted value from a previous state. The EM update pi_1 = x_hat_1
(Eq A.32) sets it to the smoothed estimate x_{1|N}. On the first iteration
of the filter, x_pred = pi_1 and Sigma_pred = Sigma_1 serve as the prior,
and the first observation y[1] is used in the correction step. No transition
matrix is applied at tau=1. (Paper, Algorithm 1; Appendix A.3, Eq A.32)

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
                          Sigma_predicted[1..N], K_stored[1..N],
                          A_stored[1..N], theta):
    # Backward pass: Rauch-Tung-Striebel smoother
    # Requires filtered estimates AND predicted covariances from forward pass

    x_smooth[N] = x_filtered[N]
    Sigma_smooth[N] = Sigma_filtered[N]

    FOR tau = N-1 DOWN TO 1:
        # Smoother gain: L_tau = Sigma_{tau|tau} A_{tau+1}^T Sigma_{tau+1|tau}^{-1}
        # where A_{tau+1} is the transition from tau to tau+1
        # and Sigma_{tau+1|tau} = Sigma_predicted[tau+1]

        A_next = A_stored[tau+1]              # transition from tau to tau+1
        L_tau = Sigma_filtered[tau] @ A_next^T @ inv(Sigma_predicted[tau+1])  # 2x2

        # Smoothed estimates
        x_smooth[tau] = x_filtered[tau]
            + L_tau @ (x_smooth[tau+1] - A_next @ x_filtered[tau])
        Sigma_smooth[tau] = Sigma_filtered[tau]
            + L_tau @ (Sigma_smooth[tau+1] - Sigma_predicted[tau+1]) @ L_tau^T

        # Store smoother gain for cross-covariance computation
        L[tau] = L_tau
    END FOR

    # Cross-covariance Sigma_{tau, tau-1|N} needed for EM M-step
    # Initialize at tau=N using Eq A.21:
    #   Sigma_{N, N-1|N} = (I_2 - K_N @ C) @ A_N @ Sigma_{N-1|N-1}
    # where K_N is the Kalman gain at the last time step and A_N = A_stored[N]
    Sigma_cross[N] = (I_2 - K_stored[N] @ C) @ A_stored[N] @ Sigma_filtered[N-1]

    # Backward recursion for cross-covariance (Eq A.20):
    #   Sigma_{tau, tau-1|N} = Sigma_{tau|N} L_{tau-1}^T
    #       + L_tau (Sigma_{tau+1, tau|N} - A_{tau+1} Sigma_{tau|N}) L_{tau-1}^T
    # Note: uses SMOOTHED covariance Sigma_{tau|N}, not filtered.
    FOR tau = N-1 DOWN TO 2:
        Sigma_cross[tau] = Sigma_smooth[tau] @ L[tau-1]^T
            + L[tau] @ (Sigma_cross[tau+1] - A_stored[tau+1] @ Sigma_smooth[tau]) @ L[tau-1]^T
    END FOR

    RETURN x_smooth, Sigma_smooth, Sigma_cross
END FUNCTION
```

(Paper, Section 2.3.1, Algorithm 2, Equations 10-11; Appendix A.2, Equations A.18-A.22)

**Note on cross-covariance recursion:** The recursion at Equation A.20 uses
Sigma_{tau|N} (the SMOOTHED covariance), not Sigma_{tau|tau} (filtered).
Draft 1 incorrectly used Sigma_filtered; this is now corrected. The A matrix
index in the second term (A_{tau+1}) refers to the transition from tau to
tau+1, which is consistent with A_stored[tau+1].
(Paper, Appendix A.2, Equation A.20)

**Sufficient statistics** for the EM M-step (Equations 14-16):

    x_hat_tau = E[x_tau | y_{1:N}]                  (smoothed state mean)
    P_tau     = E[x_tau x_tau^T | y_{1:N}]          (smoothed second moment)
    P_{tau,tau-1} = E[x_tau x_{tau-1}^T | y_{1:N}]  (smoothed cross-moment)

These relate to the smoother outputs as:

    P_tau = Sigma_smooth[tau] + x_smooth[tau] @ x_smooth[tau]^T
    P_{tau,tau-1} = Sigma_cross[tau] + x_smooth[tau] @ x_smooth[tau-1]^T

(Paper, Equations 14-16; Appendix A, Equations A.15-A.17)

#### Algorithm 3: EM Algorithm (Parameter Estimation)

```
FUNCTION em_calibrate(y[1..N], I, max_iter, tol):
    # N = T * I total bins (T training days, I bins per day)
    # Initialize parameters theta^(0)
    theta = initialize_parameters(y, I)

    Q_prev = -inf    # previous expected log-likelihood

    FOR j = 0, 1, 2, ... UNTIL convergence:
        # === E-step ===
        # Run forward Kalman filter with current theta
        x_filt, Sigma_filt, Sigma_pred, K_stored, A_stored, _ = kalman_filter(y, theta)

        # Run backward RTS smoother
        x_smooth, Sigma_smooth, Sigma_cross = kalman_smoother(
            x_filt, Sigma_filt, Sigma_pred, K_stored, A_stored, theta)

        # Compute sufficient statistics
        FOR tau = 1 TO N:
            x_hat[tau] = x_smooth[tau]                    # smoothed mean (2x1)
            P[tau] = Sigma_smooth[tau] + x_hat[tau] @ x_hat[tau]^T  # second moment (2x2)
        END FOR
        FOR tau = 2 TO N:
            P_cross[tau] = Sigma_cross[tau] + x_hat[tau] @ x_hat[tau-1]^T  # cross moment (2x2)
        END FOR

        # === M-step (closed-form updates) ===

        # Notation: P^(k,l)[tau] denotes element (k,l) of matrix P[tau]
        # D' = day-boundary indices used in eta sums: the first bin of each
        #      new day, i.e., tau = kI+1 for k = 1, ..., T-1.
        # This gives exactly T-1 terms per sum.

        # --- Initial state mean and covariance ---
        pi_1 = x_hat[1]                                            # (Eq A.32)
        Sigma_1 = P[1] - x_hat[1] @ x_hat[1]^T                   # (Eq A.33)

        # --- AR coefficient for daily component (eta) ---
        # Sum over day-boundary transitions: tau = kI+1 for k = 1, ..., T-1
        # Numerator: P_cross at the first bin of each new day (tau)
        # Denominator: P at the last bin of the previous day (tau-1 = kI)
        numerator_eta = SUM for k = 1 to T-1: P_cross[k*I + 1]^(1,1)  # (Eq A.34 num)
        denominator_eta = SUM for k = 1 to T-1: P[k*I]^(1,1)          # (Eq A.34 den)
        a_eta = numerator_eta / denominator_eta                        # (Eq 19 / A.34)

        # --- AR coefficient for dynamic component (mu) ---
        numerator_mu = SUM for tau = 2 TO N: P_cross[tau]^(2,2)       # (Eq A.35)
        denominator_mu = SUM for tau = 2 TO N: P[tau-1]^(2,2)
        a_mu = numerator_mu / denominator_mu                          # (Eq 20 / A.35)

        # --- Process noise variance for eta ---
        # T-1 day-boundary transitions, T-1 terms in sum
        sigma_eta_sq = (1 / (T - 1)) *
            SUM for k = 1 to T-1:
                [ P[k*I + 1]^(1,1)
                  + a_eta^2 * P[k*I]^(1,1)
                  - 2 * a_eta * P_cross[k*I + 1]^(1,1) ]             # (Eq 21 / A.36)

        # --- Process noise variance for mu ---
        sigma_mu_sq = (1 / (N - 1)) *
            SUM for tau = 2 TO N:
                [ P[tau]^(2,2)
                  + a_mu^2 * P[tau-1]^(2,2)
                  - 2 * a_mu * P_cross[tau]^(2,2) ]                  # (Eq 22 / A.37)

        # --- Observation noise variance ---
        # Simplified form (equivalent to expanding Eq A.38):
        r = (1 / N) * SUM for tau = 1 TO N:
            [ (y[tau] - phi[bin(tau)] - C @ x_hat[tau])^2
              + C @ Sigma_smooth[tau] @ C^T ]                         # (Eq 23 / A.38)

        # --- Seasonality vector ---
        FOR i = 1 TO I:
            phi[i] = (1 / T) * SUM for t = 1 TO T:
                ( y_{t,i} - C @ x_hat_{t,i} )                        # (Eq 24 / A.39)
            # i.e., average residual for bin i across all training days
            # where y_{t,i} = y[(t-1)*I + i] and x_hat_{t,i} = x_hat[(t-1)*I + i]
        END FOR

        # Pack updated parameters
        theta = {a_eta, a_mu, sigma_eta_sq, sigma_mu_sq, r, phi, pi_1, Sigma_1}

        # Check convergence using expected log-likelihood Q (Eq A.10)
        Q_curr = compute_Q(theta, x_hat, P, P_cross, y, phi)
        IF j > 0 AND abs(Q_curr - Q_prev) / abs(Q_prev) < tol:
            BREAK
        END IF
        Q_prev = Q_curr
    END FOR

    RETURN theta
END FUNCTION
```

(Paper, Section 2.3.2, Algorithm 3, Equations 17-24; Appendix A.3, Equations A.32-A.39)

**Expected log-likelihood Q for convergence monitoring:**

The EM algorithm monotonically increases Q(theta | theta^(j)). The paper
gives Q in Equation A.10:

    Q(theta | theta^(j)) = -E_1 - E_2 - E_3 - E_4
        - (N/2) log(r) - ((N-1)/2) log(sigma_mu_sq)
        - ((T-1)/2) log(sigma_eta_sq) - (1/2) log|Sigma_1|
        - ((2N+T)/2) log(2*pi)

where E_1 through E_4 are defined in Equations A.11-A.14 using the sufficient
statistics P, P_cross, and x_hat. In practice, computing Q requires
evaluating these sums, which are already computed as part of the M-step
updates. The convergence check should monitor the relative change in Q
between successive iterations. A decrease in Q indicates an implementation
bug. (Paper, Appendix A.1, Equation A.10)

**Clarification on day-boundary indexing for a_eta and sigma_eta_sq updates:**
The paper's Equation A.34 sums over tau = kI+1 for k = 1, ..., T-1. This
gives exactly T-1 terms. In the numerator, P_cross[kI+1]^(1,1) is the
cross-moment E[eta_{kI+1} * eta_{kI}] -- the cross-moment spanning the day
boundary from day k to day k+1. In the denominator, P[kI]^(1,1) is
E[eta_{kI}^2] -- the second moment at the last bin of day k. The last day
boundary at tau = TI = N is NOT included because there is no transition out of
the last day. (Paper, Appendix A.3, Equations A.26, A.34, A.36)

**Note on phi identifiability:** Both eta and phi contribute to the mean level
of y. If you add a constant c to all phi[i] and subtract c from eta, the
observation equation is unchanged. The EM updates for eta and phi jointly
determine the split: phi[i] absorbs the average residual per bin (Eq A.39)
and eta absorbs the daily level. No explicit sum-to-zero or other constraint
on phi is needed -- the EM updates are self-consistent. However, the absolute
level of phi and eta individually is not identifiable from the observation
equation alone; only the sum eta + phi[i] is identifiable. Do NOT impose a
constraint like SUM(phi) = 0, as this would conflict with the EM updates.
(Paper, Equation 24; identifiability is implicit in the model structure --
Researcher inference on the constraint discussion)

#### Algorithm 4: Robust Kalman Filter (Lasso Extension)

```
FUNCTION robust_kalman_filter(y[1..N], theta, lambda):
    # Same as Algorithm 1, but the correction step is modified
    # Observation model: y[tau] = C @ x[tau] + phi[tau] + v[tau] + z[tau]
    # where z[tau] is a sparse outlier term

    # Initialize same as standard filter
    C = [1, 1]

    FOR tau = 1 TO N:
        # --- Prediction step (identical to standard) ---
        IF tau == 1:
            x_pred = pi_1
            Sigma_pred = Sigma_1
        ELSE:
            [build A, Q same as Algorithm 1]
            x_pred = A @ x_hat
            Sigma_pred = A @ Sigma @ A^T + Q
        END IF

        # --- Correction step with Lasso ---
        IF y[tau] is observed:
            # Innovation
            e_tau = y[tau] - phi[bin(tau)] - C @ x_pred    # scalar

            # Innovation variance (scalar since observation is 1D)
            S_tau = C @ Sigma_pred @ C^T + r               # scalar > 0

            # Solve Lasso: min_{z} (e - z)^2 / S + lambda |z|
            # Closed-form soft-thresholding solution (Eq 33-34):
            # The threshold is lambda * S / 2
            # (because W = 1/S is the precision, and threshold = lambda/(2W) = lambda*S/2)
            threshold = lambda * S_tau / 2

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
            K = Sigma_pred @ C^T / S_tau                   # 2x1 Kalman gain
            x_hat = x_pred + K * e_clean
            Sigma = (I_2 - K @ C) @ Sigma_pred @ (I_2 - K @ C)^T + K * r * K^T
                                                           # Joseph form

            z_detected[tau] = z_star
        ELSE:
            x_hat = x_pred
            Sigma = Sigma_pred
            K = zeros(2, 1)
            z_detected[tau] = 0
        END IF

        x_filtered[tau] = x_hat
        Sigma_filtered[tau] = Sigma
        Sigma_predicted[tau] = Sigma_pred
        K_stored[tau] = K
        A_stored[tau] = A if tau > 1 else I_2
        y_hat[tau] = C @ x_pred + phi[bin(tau)]
    END FOR

    RETURN x_filtered, Sigma_filtered, Sigma_predicted, K_stored, A_stored, y_hat, z_detected
END FUNCTION
```

(Paper, Section 3.1, Equations 25-34)

**Key insight on the Lasso threshold:** The paper defines W_{tau+1} =
(C Sigma_{tau+1|tau} C^T + r)^{-1} as the precision (Eq 30). The threshold
in the soft-thresholding operator is lambda / (2 * W) = lambda * S / 2 where
S = 1/W is the innovation variance. Higher prediction uncertainty (larger S)
-> wider threshold -> fewer outliers detected. The spec uses S_tau
(innovation variance) throughout to avoid precision/variance confusion.
(Paper, Section 3.1, Equations 30, 33-34)

#### Algorithm 5: Robust EM Calibration

The EM for the robust model is identical to Algorithm 3 except:

1. In the E-step, use the robust Kalman filter (Algorithm 4) instead of the
   standard filter. Store the detected outlier values z_detected[tau].

2. **The RTS smoother (Algorithm 2) is applied WITHOUT modification** to the
   filtered estimates from the robust Kalman filter. The smoother operates on
   state estimates, not on observations directly, so the outlier detection does
   not affect the smoothing recursion. The z_detected values from the forward
   pass are stored and used ONLY in the M-step updates for r and phi.
   (Paper, Section 3.2: only modifies Equations 35-36, implying smoother is
   unchanged)

3. The M-step updates for r and phi are modified to account for the outliers:

```
    # Modified observation noise variance (Eq 35 / A.38 modified)
    # Simplified form: subtract z_star from the observation residual
    r = (1 / N) * SUM for tau = 1 TO N:
        [ (y[tau] - phi[bin(tau)] - z_detected[tau] - C @ x_hat[tau])^2
          + C @ Sigma_smooth[tau] @ C^T ]

    # Modified seasonality (Eq 36)
    FOR i = 1 TO I:
        phi[i] = (1 / T) * SUM for t = 1 TO T:
            ( y_{t,i} - C @ x_hat_{t,i} - z_detected_{t,i} )
    END FOR
```

4. All other M-step updates (a_eta, a_mu, sigma_eta_sq, sigma_mu_sq, pi_1,
   Sigma_1) remain unchanged because they depend only on the state sufficient
   statistics, not on the observation model.

(Paper, Section 3.2, Equations 35-36)

#### Algorithm 6: VWAP Execution Weights

```
FUNCTION compute_vwap_weights_static(y_hat_static[1..I]):
    # Convert log-volume forecasts to volume proportions (Eq 40)
    # y_hat_static[i] = predicted log-volume for bin i (all predicted before market open)

    vol_hat[i] = exp(y_hat_static[i])  FOR i = 1..I
    total = SUM(vol_hat)
    w[i] = vol_hat[i] / total          FOR i = 1..I
    RETURN w[1..I]
END FUNCTION

FUNCTION compute_vwap_weights_dynamic(vol_hat_dynamic[1..I]):
    # Compute VWAP weights from dynamically-updated volume forecasts (Eq 41)
    # vol_hat_dynamic[i] = exp(y_hat_dynamic[i]) for each bin i
    # where y_hat_dynamic[i] is the one-step-ahead forecast made at bin i-1
    # (i.e., re-forecast using all information up to bin i-1 via Kalman correction)
    #
    # The weight formula is recursive: each weight accounts for the cumulative
    # weight already assigned to previous bins.

    cumulative_weight = 0
    FOR i = 1 TO I-1:
        remaining_predicted = SUM(vol_hat_dynamic[i..I])
        w[i] = (vol_hat_dynamic[i] / remaining_predicted) * (1 - cumulative_weight)
        cumulative_weight = cumulative_weight + w[i]
    END FOR
    w[I] = 1 - cumulative_weight
    RETURN w[1..I]
END FUNCTION
```

(Paper, Section 4.3, Equations 39-41)

**Static VWAP weight** (Equation 40):

    w_{t,i}^(s) = volume_hat_{t,i}^(s) / SUM_{j=1}^{I} volume_hat_{t,j}^(s)

**Dynamic VWAP weight** (Equation 41):

    w_{t,i}^(d) = (volume_hat_{t,i}^(d) / SUM_{j=i}^{I} volume_hat_{t,j}^(d))
                  * (1 - SUM_{j=1}^{i-1} w_{t,j}^(d))      for i = 1, ..., I-1
    w_{t,I}^(d) = 1 - SUM_{j=1}^{I-1} w_{t,j}^(d)          for the last bin

**Note on dynamic forecasts:** The "dynamic" aspect is that vol_hat_dynamic[i]
is re-forecast at each bin i using all observations up to bin i-1 (via Kalman
filter correction at bin i-1 followed by a one-step-ahead prediction to
bin i). The weight formula itself is deterministic given the forecasts.
In production, at each bin i during the trading day: (a) observe y[i-1],
(b) run Kalman correction for bin i-1, (c) predict forward for bins i through I,
(d) compute vol_hat_dynamic[i..I] = exp(y_hat[i..I]), (e) compute w[i] using
the formula above.

### Data Flow

```
Raw volume per bin (shares traded)
    |
    v
Normalize by shares outstanding  -->  volume_{t,i} = shares_traded / shares_outstanding
    |                                 (requires daily shares outstanding per security)
    v
Log transform  -->  y_{t,i} = log(volume_{t,i})
    |                (Input: scalar per bin; must be > 0)
    v
+---------------------------------------------------+
| EM Calibration (offline, on training window)       |
|                                                    |
|   Forward Kalman Filter  -->  x_filt (2x1 per bin) |
|         |                     Sigma_filt (2x2)     |
|         |                     Sigma_pred (2x2)     |
|         |                     K (2x1), A (2x2)     |
|         v                                          |
|   RTS Backward Smoother  -->  x_smooth, P, P_cross|
|         |                                          |
|         v                                          |
|   M-step: closed-form updates for                  |
|     a_eta, a_mu, sigma_eta_sq, sigma_mu_sq,        |
|     r, phi[1..I], pi_1, Sigma_1                    |
|                                                    |
|   Repeat until convergence (monitor Q function)    |
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
    |                        (static: Eq 40; dynamic: Eq 41 recursive)
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
- Innovation variance S: scalar
- Seasonality phi: vector of length I (number of bins per day)
- y_hat: scalar per bin

**Forward pass stored quantities (needed by smoother):**
- x_filtered[1..N]: filtered state means (2x1 each)
- Sigma_filtered[1..N]: filtered state covariances (2x2 each)
- Sigma_predicted[1..N]: predicted state covariances (2x2 each)
- K_stored[1..N]: Kalman gains (2x1 each) -- needed for cross-covariance init
- A_stored[1..N]: transition matrices (2x2 each) -- needed for smoother gain

**Computational complexity:** The state is 2D, so all matrix operations are
O(1). The Kalman filter is O(N) per pass, the smoother is O(N), and the
M-step sums are O(N). Each EM iteration is therefore O(N), and convergence
typically requires 5-10 iterations (Paper, Figure 4). Total calibration cost:
O(N * num_iterations). With N = 250 * 26 = 6500 bins and 10 iterations, this
is ~65,000 scalar operations -- trivial.

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
| I | Number of bins per trading day | Determined by exchange hours and bin width. Researcher inference: 26 for NYSE with 15-min bins (9:30 AM to 4:00 PM = 6.5 hours = 390 min / 15 = 26). Bins are [9:30, 9:45), [9:45, 10:00), ..., [3:45, 4:00). Paper does not specify I for NYSE but states "I varies from market to market" (Section 4.1) | N/A -- fixed by data design | Typically 13-52 for 30-min to 10-min bins |

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
   b. Obtain daily shares outstanding for each security and each day.
   c. Normalize each bin's volume: volume_{t,i} = shares_traded_{t,i} / shares_outstanding_t.
   d. Take the natural log: y_{t,i} = ln(volume_{t,i}).
   e. Exclude any bins with zero volume (log(0) is undefined; the paper
      explicitly excludes zero-volume bins -- Section 4.1).
   f. Exclude half-day trading sessions (Paper, Section 4.1).
   g. Linearize into a single time series y[1..N] where N = T_train * I.

2. **EM iteration:**
   a. Initialize theta^(0) as described above.
   b. E-step: run forward Kalman filter (Algorithm 1), then backward RTS
      smoother (Algorithm 2), and compute sufficient statistics P[tau] and
      P_cross[tau].
   c. M-step: apply closed-form updates (Equations 17-24 / A.32-A.39).
   d. Repeat until relative change in expected log-likelihood Q < tol (e.g.,
      tol = 1e-6) or max_iter reached (e.g., 50 iterations; convergence is
      typically within 5-10 iterations per Figure 4). Monitor Q using
      Equation A.10; a decrease in Q indicates an implementation bug.

3. **Cross-validation for N and lambda:**
   a. Define a validation period (Paper uses January 2015 - May 2015).
   b. For each candidate N in a grid (e.g., {60, 90, 120, 180, 250} days * I):
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

**MAPE definition** (Paper, Section 3.3, Equation 37):

    MAPE = (1/M) * SUM_{tau=1}^{M} |volume_tau - predicted_volume_tau| / volume_tau

where M is the total number of bins in the out-of-sample set. IMPORTANT:
volumes are in the ORIGINAL (non-log) scale. To compute MAPE from log-volume
predictions: predicted_volume_tau = exp(y_hat_tau). The shares-outstanding
normalization cancels in the ratio (both numerator and denominator are
normalized), so MAPE can equivalently be computed on normalized volumes.

**EM convergence behavior:**
- Parameters converge within approximately 5-10 iterations from arbitrary
  initial values (Paper, Section 2.3.3, Figure 4).
- Convergence is monotonic (Q function increases at each step).
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

6. **Q function monotonicity:** The expected log-likelihood Q(theta) should
   increase (or remain constant) at every EM iteration. A decrease indicates
   an implementation bug. (Standard EM property; Paper, Appendix A.1)

7. **Baseline comparison:** The model should outperform a simple rolling mean
   baseline (mean of same-bin log-volumes over prior 20 days). If it does not,
   something is wrong with the state-space formulation or calibration. (Paper,
   Table 3: KF achieves 64% MAPE improvement over RM)

8. **VWAP weights sum to 1:** After converting log-volume forecasts to weights,
   verify that SUM(w_i) = 1 for both static and dynamic strategies. For the
   dynamic formula (Eq 41), this holds by construction since the last weight
   is 1 - cumulative_weight. (Paper, Section 4.3, Equations 40-41)

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
   remain positive semi-definite. Use the Joseph form for the covariance
   update:
       Sigma = (I_2 - K @ C) @ Sigma_pred @ (I_2 - K @ C)^T + K * r * K^T
   which is numerically more stable than the standard form
       Sigma = Sigma_pred - K @ C @ Sigma_pred
   (Researcher inference: standard numerical Kalman filter best practice,
   not discussed in the paper but essential for production implementations)

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
| Data requirements (shares outstanding) | Researcher inference (paper normalizes by shares outstanding in Section 4.1 but does not list as explicit data requirement) |
| State-space formulation | Paper, Section 2, Equations 4-5 |
| State transition matrix (time-varying A_tau) | Paper, Section 2, definition after Equation 5 |
| Process noise structure (Q_tau) | Paper, Section 2, Q_tau definition |
| Day-boundary set D (T-1 elements) | Paper, Section 2, D = {kI : k=1,...,T-1}; Appendix A.3, Eq A.34 summation bounds |
| Kalman filter algorithm | Paper, Section 2.2, Algorithm 1 |
| Filter initialization (pi_1 as prior, no transition at tau=1) | Paper, Algorithm 1; Appendix A.3, Eq A.32 |
| Multi-step prediction | Paper, Section 2.2, Equation 9 |
| Static vs dynamic prediction | Paper, Section 2.2, paragraphs after Algorithm 1 |
| Kalman smoother (RTS) | Paper, Section 2.3.1, Algorithm 2 |
| Cross-covariance recursion (uses smoothed covariance) | Paper, Appendix A.2, Equations A.20-A.22 |
| Cross-covariance initialization | Paper, Appendix A.2, Equation A.21 |
| Sufficient statistics for EM | Paper, Equations 14-16; Appendix A, Equations A.15-A.17 |
| EM algorithm | Paper, Section 2.3.2, Algorithm 3 |
| M-step: pi_1 and Sigma_1 | Paper, Appendix A.3, Equations A.32-A.33 |
| M-step: a_eta (T-1 terms) | Paper, Appendix A.3, Equation A.34 |
| M-step: a_mu | Paper, Appendix A.3, Equation A.35 |
| M-step: sigma_eta_sq (T-1 terms) | Paper, Appendix A.3, Equation A.36 |
| M-step: sigma_mu_sq | Paper, Appendix A.3, Equation A.37 |
| M-step: r | Paper, Appendix A.3, Equation A.38 |
| M-step: phi | Paper, Appendix A.3, Equation A.39 |
| Expected log-likelihood Q | Paper, Appendix A.1, Equation A.10 |
| EM convergence properties | Paper, Section 2.3.3, Figure 4 |
| Phi identifiability (no constraint needed) | Researcher inference (implicit in model structure; Eq 24 and joint EM updates) |
| Robust observation model | Paper, Section 3, Equation 25 |
| Robust state-space formulation | Paper, Section 3.1, Equations 26-27 |
| Lasso-in-Kalman derivation | Paper, Section 3.1, Equations 28-30 |
| Soft-thresholding solution (threshold = lambda * S / 2) | Paper, Section 3.1, Equations 33-34 |
| Smoother unchanged in robust mode | Paper, Section 3.2 (only Eqs 35-36 modified, implying smoother is unchanged) |
| Robust EM modifications (r, phi) | Paper, Section 3.2, Equations 35-36 |
| MAPE definition | Paper, Section 3.3, Equation 37 |
| Empirical data description | Paper, Section 4.1, Table 2 |
| Volume prediction results | Paper, Section 4.2, Table 3 |
| VWAP tracking error results | Paper, Section 4.3, Table 4 |
| Static VWAP weights | Paper, Section 4.3, Equation 40 |
| Dynamic VWAP weights (recursive formula) | Paper, Section 4.3, Equation 41 |
| VWAP tracking error metric | Paper, Section 4.3, Equation 42 |
| I = 26 for NYSE 15-min bins | Researcher inference (derived from NYSE hours 9:30-16:00; paper says "I varies from market to market" in Section 4.1) |
| Bin boundary convention [9:30, 9:45) | Researcher inference (standard convention, not specified in paper) |
| EM initialization values | Researcher inference (specific values not in paper; Figure 4 confirms robustness) |
| Joseph form covariance update | Researcher inference (standard Kalman filter best practice) |
| Variance floor in M-step | Researcher inference (standard EM numerical precaution) |
| Rolling re-estimation frequency | Researcher inference (paper uses rolling window but does not specify re-estimation cadence) |
| Computational complexity O(N) | Researcher inference (follows from 2D state; not stated in paper) |

## Revision History

### Draft 2 changes (addressing Critique 1)

**Major issues addressed:**
- M1: Restructured Algorithm 1 to explicitly handle tau=1 initialization separately (no transition applied). Added explanatory note on pi_1 semantics.
- M2: Fixed a_eta summation to explicitly use k=1..T-1 (T-1 terms), replacing ambiguous "SUM over tau in D". Clarified D has T-1 elements (excludes last day).
- M3: Replaced W_tau (precision) notation with S_tau (innovation variance) throughout Algorithms 1 and 4. Threshold is now lambda * S_tau / 2. Dropped the dual notation.
- M4: Added explicit statement that the RTS smoother is unchanged in the robust case, with explanation of why. Noted z_detected storage for M-step.
- M5: Rewrote compute_vwap_weights_dynamic to directly implement Eq 41's recursive formula. Removed incorrect mixing of actual/predicted volumes.

**Minor issues addressed:**
- m1: Fixed cross-covariance recursion in Algorithm 2 to use Sigma_smooth (smoothed covariance) instead of Sigma_filtered. Added explanatory note.
- m2: Added Q function formula reference (Eq A.10) and convergence monitoring guidance.
- m3: Fixed sigma_eta_sq summation to use k=1..T-1 (same fix as M2).
- m4: Added MAPE definition from Eq 37 with clarification that it operates on original-scale volumes.
- m5: Marked I=26 for NYSE as "Researcher inference" with bin boundary convention note.
- m6: Added note on phi identifiability and warning against imposing sum-to-zero constraint.
- m7: Added "daily shares outstanding per security" to data requirements in Model Description and Calibration sections.
- m8: Added "Forward pass stored quantities" section documenting what Algorithms 1/4 must store for the smoother. Updated Algorithm 1 to store Sigma_predicted, K_stored, A_stored. Updated Algorithm 2 inputs to include these.
