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
  observed[tau]     -- boolean flag: true if observation is valid, false if missing

Procedure:
  1. For each day t and bin i:
       turnover[t,i] = raw_volume[t,i] / shares_out[t]
  2. For each (t, i) pair:
       tau = (t-1)*I + i
       if turnover[t,i] > 0:
           y[tau] = ln(turnover[t,i])
           observed[tau] = true
       else:
           y[tau] = NaN (or sentinel value; not used)
           observed[tau] = false
  3. Exclude half-day trading sessions entirely
     (days with fewer than I bins are dropped; remaining days re-indexed)
```

(Paper: Section 4.1. Volume is "computed as Equation (1)" which defines
volume_{t,i} = shares_traded / shares_outstanding. Half-day sessions excluded.)

#### Step 1: Kalman Filter (Algorithm 1 from Paper)

This is the core prediction engine. It produces one-step-ahead state estimates
(prediction) and incorporates observations (correction) recursively.

```
Input:
  y[1..N]           -- observed log-volume series
  observed[1..N]    -- boolean flags for valid observations
  theta             -- parameter set: {a_eta, a_mu, sigma_eta^2, sigma_mu^2, r,
                       phi[1..I], pi_1, Sigma_1}

Output:
  x_hat[tau|tau]    -- filtered state estimates (2x1 vectors)
  Sigma[tau|tau]    -- filtered state covariances (2x2 matrices)
  x_hat[tau+1|tau]  -- predicted state estimates
  Sigma[tau+1|tau]  -- predicted state covariances
  y_hat[tau+1|tau]  -- predicted log-volume (scalar)
  K[tau]            -- Kalman gains (stored for smoother cross-covariance init)

Definitions:
  C = [1, 1]        -- observation vector (1x2)
  A[tau] = [[a_eta_tau, 0],    -- state transition matrix (2x2)
            [0,         a_mu]]
  where a_eta_tau = a_eta   if tau is a day boundary (tau = k*I for some k >= 1)
                  = 1       otherwise

  Q[tau] = [[sigma_eta_tau^2, 0],    -- process noise covariance (2x2)
            [0,               sigma_mu^2]]
  where sigma_eta_tau^2 = sigma_eta^2   if tau is a day boundary (tau = k*I)
                        = 0             otherwise

  phi[tau] = phi_i where i = ((tau-1) mod I) + 1  -- seasonal value for this bin position

Procedure:
  -- Initialize predicted state at time 1:
  x_hat[1|0] = pi_1          -- initial state mean (2x1)
  Sigma[1|0] = Sigma_1       -- initial state covariance (2x2)

  -- Initial correction: process first observation to produce x_hat[1|1]
  if observed[1]:
      W_init = (C * Sigma[1|0] * C^T + r)^{-1}        -- scalar inverse
      K[1] = Sigma[1|0] * C^T * W_init                 -- 2x1 vector
      e_init = y[1] - phi[1] - C * x_hat[1|0]
      x_hat[1|1] = x_hat[1|0] + K[1] * e_init
      Sigma[1|1] = (I_2 - K[1] * C) * Sigma[1|0] * (I_2 - K[1] * C)^T + K[1] * r * K[1]^T
      -- Store for log-likelihood: e[1] = e_init, S[1] = C * Sigma[1|0] * C^T + r = 1/W_init
  else:
      x_hat[1|1] = x_hat[1|0]
      Sigma[1|1] = Sigma[1|0]
      K[1] = [0, 0]^T

  -- Main loop:
  For tau = 1, 2, ..., N-1:
    -- PREDICT (lines 2-3 of Algorithm 1)
    x_hat[tau+1|tau] = A[tau] * x_hat[tau|tau]
    Sigma[tau+1|tau] = A[tau] * Sigma[tau|tau] * A[tau]^T + Q[tau]

    -- Predicted observation
    y_hat[tau+1|tau] = C * x_hat[tau+1|tau] + phi[tau+1]

    -- CORRECT (lines 4-6 of Algorithm 1)
    if observed[tau+1]:
        -- Compute Kalman gain:
        W[tau+1] = (C * Sigma[tau+1|tau] * C^T + r)^{-1}    -- scalar inverse
        K[tau+1] = Sigma[tau+1|tau] * C^T * W[tau+1]         -- 2x1 vector

        -- Innovation (prediction error):
        e[tau+1] = y[tau+1] - phi[tau+1] - C * x_hat[tau+1|tau]

        -- Correct state estimate (Joseph form for numerical stability):
        x_hat[tau+1|tau+1] = x_hat[tau+1|tau] + K[tau+1] * e[tau+1]
        Sigma[tau+1|tau+1] = (I_2 - K[tau+1]*C) * Sigma[tau+1|tau] * (I_2 - K[tau+1]*C)^T
                             + K[tau+1] * r * K[tau+1]^T
    else:
        -- Missing observation: skip correction
        x_hat[tau+1|tau+1] = x_hat[tau+1|tau]
        Sigma[tau+1|tau+1] = Sigma[tau+1|tau]
        K[tau+1] = [0, 0]^T
```

Note on Joseph form: The standard covariance update
Sigma[tau+1|tau+1] = Sigma[tau+1|tau] - K*C*Sigma[tau+1|tau] is algebraically
equivalent but can lose positive definiteness due to floating-point errors in long
sequences. The Joseph form (I - K*C) * Sigma * (I - K*C)^T + K*r*K^T guarantees
symmetry and positive semi-definiteness. Given the 2x2 state, the computational
overhead is negligible.
Researcher inference: Joseph form is standard Kalman filter practice, not discussed
in the paper.

Note on scalar operations: Since C = [1, 1] and the state is 2x1, C * Sigma * C^T
is a scalar (sum of all four elements of Sigma), so W is a scalar division, and K is
a 2x1 vector. No matrix inversion beyond scalar reciprocal is ever needed.

(Paper: Section 2.2, Algorithm 1, Equations 4-5, 7-8. The initial correction step is
implied by Algorithm 1's structure which assumes x_hat[tau|tau] is available at the
start of each iteration; the spec makes it explicit.)

#### Step 2: Kalman Smoother -- Rauch-Tung-Striebel (Algorithm 2 from Paper)

Used during EM calibration only (not during prediction). Runs backward over the
filtered estimates to produce smoothed state estimates conditioned on ALL observations.

The RTS smoother is unchanged in the robust variant. It operates on the filtered
states produced by the robust Kalman filter, which already incorporate outlier
cleaning via the soft-thresholding step. The z* values are only used in the forward
filter correction and in the M-step updates for r and phi.

```
Input:
  x_hat[tau|tau], Sigma[tau|tau]       -- from forward Kalman filter
  x_hat[tau+1|tau], Sigma[tau+1|tau]   -- from forward Kalman filter
  A[tau]                               -- transition matrices
  K[N]                                 -- Kalman gain at last time step (for cross-cov init)

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
  Sigma[N,N-1|N] = (I_2 - K[N]*C) * A[N-1] * Sigma[N-1|N-1]

  For tau = N-1, N-2, ..., 2:
    Sigma[tau,tau-1|N] = Sigma[tau|tau] * L[tau-1]^T
                         + L[tau] * (Sigma[tau+1,tau|N] - A[tau] * Sigma[tau|tau]) * L[tau-1]^T
```

Note: Sigma[tau+1|tau] is 2x2, so its inversion is a direct 2x2 analytic inverse
(ad-bc determinant formula). Since the matrix is symmetric positive definite, this is
always well-conditioned. If the determinant is below a threshold (e.g., 1e-12), add
epsilon to the diagonal before inverting.
Researcher inference: the epsilon regularization is standard numerical practice.

(Paper: Section 2.3.1, Algorithm 2, Equations 10-11. Cross-covariance initialization
from Appendix A, Equation A.21)

#### Step 3: EM Algorithm (Algorithm 3 from Paper)

Iteratively estimates all model parameters by alternating between computing sufficient
statistics (E-step via filter + smoother) and closed-form parameter updates (M-step).

```
Input:
  y[1..N]           -- observed log-volume training series
  observed[1..N]    -- boolean flags for valid observations
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
    P^{(1,1)}[tau,tau-1] = P[tau,tau-1][1,1]  -- cross-time eta_tau * eta_{tau-1}
    P^{(2,2)}[tau,tau-1] = P[tau,tau-1][2,2]  -- cross-time mu_tau * mu_{tau-1}
    (These are the (1,1) and (2,2) elements of the 2x2 cross-time matrix P[tau,tau-1],
     analogous to the notation for P[tau] above.)

  Let D_start = {tau : tau = k*I+1 for k=1,2,...,T-1} be the set of first-bin-of-day indices
  (exactly T-1 elements, one for each day transition).
  These are the DESTINATION indices of day transitions: the transition matrix A[tau-1]
  applies at tau-1 = kI (last bin of day k), producing the state at tau = kI+1 (first
  bin of day k+1). The M-step sums over these destination indices.

  Bin-position mapping: for any global index tau, the bin position is
    i(tau) = ((tau-1) mod I) + 1
  so phi[tau] means phi_{i(tau)}, the seasonality value for the bin position of tau.

Procedure:
  j = 0
  Repeat:
    j = j + 1

    -- E-STEP:
    Run Kalman filter (Step 1) with current theta^(j-1) on y[1..N]
    Run Kalman smoother (Step 2) on filter output
    Compute sufficient statistics x_hat[tau], P[tau], P[tau,tau-1] for all tau

    -- M-STEP (all updates are closed-form):
    -- IMPORTANT: The computation order matters. Parameters computed earlier in the
    -- M-step must be used (with superscript j+1) in formulas that depend on them.
    -- The paper's Equations A.32-A.39 use (j+1) superscripts for already-updated
    -- parameters within the same M-step.

    -- Group 1: Parameters that depend only on sufficient statistics (no dependencies)

    -- Initial state mean (Equation A.32):
    pi_1^(j+1) = x_hat[1]

    -- Initial state covariance (Equation A.33):
    Sigma_1^(j+1) = P[1] - x_hat[1] * x_hat[1]^T

    -- Daily AR coefficient (Equation A.34, uses only sufficient statistics):
    a_eta^(j+1) = [ sum_{tau in D_start} P^{(1,1)}[tau,tau-1] ]
                  / [ sum_{tau in D_start} P^{(1,1)}[tau-1] ]

    -- Intraday dynamic AR coefficient (Equation A.35, uses only sufficient statistics):
    a_mu^(j+1) = [ sum_{tau=2}^{N} P^{(2,2)}[tau,tau-1] ]
                 / [ sum_{tau=2}^{N} P^{(2,2)}[tau-1] ]

    -- Seasonality vector (Equation A.39, uses only sufficient statistics):
    For each bin position i = 1..I:
      phi_i^(j+1) = (1/T) * sum_{t=1}^{T} (y[t,i] - C * x_hat[t,i])

      where y[t,i] and x_hat[t,i] use the (day,bin) indexing mapped from global tau

    -- Group 2: Parameters that depend on Group 1 results (use j+1 superscripts)

    -- Daily process noise variance (Equation A.36, uses a_eta^(j+1)):
    [sigma_eta^2]^(j+1) = (1/(T-1)) * sum_{tau in D_start}
                          { P^{(1,1)}[tau] + (a_eta^(j+1))^2 * P^{(1,1)}[tau-1]
                            - 2 * a_eta^(j+1) * P^{(1,1)}[tau,tau-1] }

    -- Intraday dynamic process noise variance (Equation A.37, uses a_mu^(j+1)):
    [sigma_mu^2]^(j+1) = (1/(N-1)) * sum_{tau=2}^{N}
                         { P^{(2,2)}[tau] + (a_mu^(j+1))^2 * P^{(2,2)}[tau-1]
                           - 2 * a_mu^(j+1) * P^{(2,2)}[tau,tau-1] }

    -- Observation noise variance (Equation A.38, uses phi^(j+1)):
    r^(j+1) = (1/N) * sum_{tau=1}^{N}
              [ y[tau]^2 + C * P[tau] * C^T - 2*y[tau]*C*x_hat[tau]
                + (phi[tau]^(j+1))^2 - 2*y[tau]*phi[tau]^(j+1)
                + 2*phi[tau]^(j+1)*C*x_hat[tau] ]

              where phi[tau]^(j+1) = phi_{i(tau)}^(j+1) is the just-computed
              seasonality value for the bin position corresponding to global index tau.

    -- Parameter clamping (Researcher inference: standard numerical safeguard):
    epsilon = 1e-8
    a_eta^(j+1)       = clamp(a_eta^(j+1), epsilon, 1 - epsilon)
    a_mu^(j+1)        = clamp(a_mu^(j+1), epsilon, 1 - epsilon)
    [sigma_eta^2]^(j+1) = max([sigma_eta^2]^(j+1), epsilon)
    [sigma_mu^2]^(j+1)  = max([sigma_mu^2]^(j+1), epsilon)
    r^(j+1)           = max(r^(j+1), epsilon)

    -- Compute observed-data log-likelihood for convergence check
    -- (prediction error decomposition, computed from Kalman filter innovations):
    N_obs = count of tau where observed[tau] = true

    log_lik^(j) = -0.5 * sum_{tau: observed[tau]} [ ln(S[tau]) + e[tau]^2 / S[tau] ]
                  - (N_obs/2) * ln(2*pi)

    where:
      S[tau] = C * Sigma[tau|tau-1] * C^T + r   -- innovation variance (scalar)
      e[tau] = y[tau] - phi[tau] - C * x_hat[tau|tau-1]   -- innovation (scalar)
    and S[tau], e[tau] are already computed during the Kalman filter forward pass.
    Both the summation and the constant term use N_obs (the count of observed bins),
    not N. Missing observations contribute no innovation and must be excluded
    consistently from both terms. For liquid stocks N_obs = N, but for securities
    with zero-volume bins the distinction matters for correct absolute log-likelihood
    values (AIC/BIC model comparison, cross-validation across datasets with different
    missingness rates, debugging against independent implementations).

    This is the standard Kalman filter log-likelihood via prediction error
    decomposition, which gives the observed-data log-likelihood log P({y_tau}).
    It requires only filter outputs (no smoother needed) and is the correct
    quantity for model comparison (e.g., AIC/BIC).

    Note: The paper's Equation A.8 (Appendix A.1) gives the joint log-likelihood
    log P({x_tau}, {y_tau}), whose expected value under the posterior defines the
    Q function maximized in the M-step. The Q function also increases monotonically
    under correct EM updates and could alternatively be used for convergence
    monitoring. However, the innovation-based form above is preferred because:
    (a) it is simpler and does not require smoother output,
    (b) it gives the actual model log-likelihood usable for model comparison,
    (c) it is the standard convergence diagnostic for Kalman filter EM.
    Researcher inference: the choice of innovation-based log-likelihood over Eq A.8
    for convergence monitoring is standard practice, not discussed in the paper.

    In practice, tracking the relative change
    |log_lik^(j) - log_lik^(j-1)| / |log_lik^(j-1)| is sufficient for convergence.
    The relative change formula works correctly despite missing observations because
    N_obs is constant across iterations (the same observations are missing in every
    E-step), so any constant-term error would cancel in the difference.
    Alternatively, monitoring the parameter change norm
    ||theta^(j+1) - theta^(j)|| / ||theta^(j)|| works as a convergence criterion.

    (Paper: Appendix A.1, Equation A.8 for the joint log-likelihood context.
     Innovation-based form is Researcher inference: standard Kalman filter result.)

  -- On the first iteration (j=1), skip the convergence check (log_lik^(0) has
  -- not been computed). Equivalently, initialize log_lik^(0) = -infinity so the
  -- relative change is always above tol on the first iteration.
  Until |log_lik^(j) - log_lik^(j-1)| / |log_lik^(j-1)| < tol  OR  j >= max_iter

  Return theta_hat = theta^(j+1)
```

(Paper: Section 2.3.2, Algorithm 3, Equations 17-24. Closed-form M-step derivations
in Appendix A, Equations A.24-A.39. M-step computation order follows the dependency
structure: Equations A.34, A.35, A.39 are independent of other parameters; Equations
A.36, A.37, A.38 depend on the just-updated values from A.34, A.35, A.39 respectively,
as indicated by their (j+1) superscripts.)

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
  At each tau where observed[tau] is true:

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
    Sigma[tau|tau] = (I_2 - K[tau]*C) * Sigma[tau|tau-1] * (I_2 - K[tau]*C)^T
                     + K[tau] * r * K[tau]^T

  At each tau where observed[tau] is false:
    -- Missing observation: skip correction
    x_hat[tau|tau] = x_hat[tau|tau-1]
    Sigma[tau|tau] = Sigma[tau|tau-1]
    z*[tau] = 0
```

The threshold lambda/(2*W[tau]) is time-varying because W[tau] depends on the current
predictive variance. When the model is uncertain (large Sigma), the threshold widens,
tolerating larger innovations. When the model is confident, the threshold tightens,
rejecting smaller deviations. This provides automatic adaptive outlier detection.

Note: The initial correction step at tau=1 (described in Step 1) also uses the robust
modification when the robust variant is active.

(Paper: Section 3.1, Equations 25-34. Soft-thresholding derived from Equations 30, 33.)

#### Step 4b: Robust EM Modifications

When calibrating the robust model, the EM uses the robust Kalman filter (Step 4) in
the E-step instead of the standard filter. The RTS smoother (Step 2) is unchanged --
it operates on the filtered states produced by the robust Kalman filter, which already
incorporate outlier cleaning via the soft-thresholding step. The z* values are only
used in the forward filter correction and in the M-step updates for r and phi below.

The M-step updates for r and phi incorporate the inferred outlier terms z*[tau]:

```
  -- Observation noise variance (robust version of Equation A.38, uses phi[tau]^(j+1)):
  r^(j+1) = (1/N) * sum_{tau=1}^{N}
            [ y[tau]^2 + C*P[tau]*C^T - 2*y[tau]*C*x_hat[tau]
              + (phi[tau]^(j+1))^2 - 2*y[tau]*phi[tau]^(j+1)
              + 2*phi[tau]^(j+1)*C*x_hat[tau]
              + (z*[tau])^2 - 2*z*[tau]*y[tau]
              + 2*z*[tau]*C*x_hat[tau] + 2*z*[tau]*phi[tau]^(j+1) ]

            where phi[tau]^(j+1) = phi_{i(tau)}^(j+1) is the just-computed seasonality
            value for the bin position corresponding to global index tau.

  -- Seasonality vector (robust version of Equation A.39):
  phi_i^(j+1) = (1/T) * sum_{t=1}^{T} (y[t,i] - C*x_hat[t,i] - z*[t,i])

            where z*[t,i] is the outlier term at global index tau = (t-1)*I + i.
```

All other M-step equations (pi_1, Sigma_1, a_eta, a_mu, sigma_eta^2, sigma_mu^2)
remain unchanged. The M-step computation order is the same as the standard case,
with the robust phi computed before the robust r (since r depends on phi^(j+1)).

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
corrections during day t+1. Since the transition matrix A[tau] is time-varying
(different at day boundaries vs intraday), the multi-step prediction must be
computed explicitly rather than as a matrix power:

```
For static prediction of day t+1 from the end of day t:
  Let tau_0 = t*I  (last bin of day t)

  -- Step 1: Day boundary transition (tau_0 -> tau_0+1):
  x_hat[tau_0+1|tau_0] = [[a_eta, 0], [0, a_mu]] * x_hat[tau_0|tau_0]

  -- Step 2: Intraday steps (tau_0+1 -> tau_0+h, for h = 2..I):
  For h = 2, ..., I:
    x_hat[tau_0+h|tau_0] = [[1, 0], [0, a_mu]] * x_hat[tau_0+h-1|tau_0]

  -- Equivalently, in closed form:
  eta_hat = a_eta * x_hat[tau_0|tau_0][1]          (constant for all bins)
  mu_hat[h] = (a_mu)^h * x_hat[tau_0|tau_0][2]     (decays geometrically)

  -- Predicted log-volume for each bin h = 1..I:
  y_hat[tau_0+h|tau_0] = eta_hat + mu_hat[h] + phi[h]
```

The first transition applies a_eta (day-boundary AR decay on the daily component),
and all subsequent intraday transitions use a_eta_tau = 1 (eta is constant within
a day). The mu component decays as (a_mu)^h across all h steps (both the day
boundary and intraday transitions apply a_mu).

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
After each observed bin, revise forecasts for remaining bins and redistribute the
remaining order. The key is the interleaving of observation, Kalman filter update,
and weight computation. The step-by-step procedure is:

```
Input:
  Calibrated Kalman filter state at end of previous day: x_hat[tI|tI], Sigma[tI|tI]
  Total order quantity Q to execute across I bins on day t+1

Procedure:
  -- Before market open: produce initial forecasts for all bins
  Propagate state through day boundary:
    x_hat[tI+1|tI] = A[tI] * x_hat[tI|tI]   (uses a_eta)
    Sigma[tI+1|tI] = A[tI] * Sigma[tI|tI] * A[tI]^T + Q[tI]

  For h = 1, ..., I:
    if h > 1:
      x_hat[tI+h|tI] = A_intraday * x_hat[tI+h-1|tI]   -- propagate state
      where A_intraday = [[1,0],[0,a_mu]]
    y_hat[tI+h|tI] = C * x_hat[tI+h|tI] + phi[h]
    volume_hat[h] = exp(y_hat[tI+h|tI]) * shares_out[t+1]

  remaining = 1.0

  For i = 1, ..., I-1:
    -- BEFORE bin i is observed:
    -- Compute weight using current best forecasts
    w[i] = remaining * volume_hat[i] / sum_{j=i}^{I} volume_hat[j]
    remaining = remaining - w[i]

    -- Execute w[i] * Q shares in bin i

    -- AFTER bin i is observed (y[tI+i] now available):
    -- Run Kalman filter correction with the actual observation
    e = y[tI+i] - phi[i] - C * x_hat[tI+i|tI+i-1]
    S = C * Sigma[tI+i|tI+i-1] * C^T + r
    K = Sigma[tI+i|tI+i-1] * C^T / S
    x_hat[tI+i|tI+i] = x_hat[tI+i|tI+i-1] + K * e
    Sigma[tI+i|tI+i] = (I_2 - K*C) * Sigma[tI+i|tI+i-1] * (I_2 - K*C)^T + K*r*K^T

    -- Update forecasts for remaining bins i+1, ..., I using new filtered state
    For h = i+1, ..., I:
      steps = h - i
      eta_forecast = x_hat[tI+i|tI+i][1]        (eta unchanged within day)
      mu_forecast = (a_mu)^steps * x_hat[tI+i|tI+i][2]
      y_hat_new = eta_forecast + mu_forecast + phi[h]
      volume_hat[h] = exp(y_hat_new) * shares_out[t+1]

    -- Predict next bin's state for the next iteration's correction
    x_hat[tI+i+1|tI+i] = A_intraday * x_hat[tI+i|tI+i]
    Sigma[tI+i+1|tI+i] = A_intraday * Sigma[tI+i|tI+i] * A_intraday^T + Q_intraday

  -- Last bin: execute remainder
  w[I] = remaining   (execute remaining * Q shares in last bin)
```

Note: volume_hat[i] in the weight computation for bin i is the forecast made BEFORE
bin i is observed (the one-step-ahead prediction). The actual volume at bin i is NOT
used in computing w[i] for bin i. After bin i is observed, the filter is corrected and
forecasts for bins i+1..I are updated, which affects w[i+1] and subsequent weights.
The paper's Equation 41 uses volume_{t,i}^{(d)} to denote this dynamic prediction,
and the text below Eq 41 states "order slicing is revised at each new bin as a new
intraday volume is gradually observed."

(Paper: Section 4.3, Equation 41, and surrounding text on page 10)

### Data Flow

```
Raw intraday volume data (T days x I bins)
    |
    v
[Normalize by shares outstanding] --> turnover[t,i]
    |
    v
[Log transform, flag zeros as missing] --> y[tau], observed[tau]
    |
    v
[Reshape to global time series] --> y[tau], tau = 1..N, N = T*I
    |
    +------> [EM Calibration loop] ------+
    |           |                         |
    |           v                         v
    |        [Kalman Filter forward]   [M-step: closed-form
    |        (with initial correction   parameter updates,
    |         at tau=1)                  ordered by dependency]
    |           |                         |
    |           v                         v
    |        [RTS Smoother backward]   [Parameter clamping]
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
| Missing flags | observed | (N,) | boolean |
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
| Outlier terms | z*[tau] | (N,) | float, mostly zero (robust only) |
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

6. **Log-likelihood monotonically increases:** Track the observed-data log-likelihood
   (innovation-based, from Step 3) at each EM iteration. It must be non-decreasing.
   A decrease indicates a bug in the E-step or M-step implementation. Note: the
   innovation-based log-likelihood is computed from the filter's forward pass and
   does not require the smoother.
   (Paper: Section 2.3.2, standard EM property)

7. **Synthetic data recovery:** Generate synthetic data from known parameters
   (a_eta=0.98, a_mu=0.5, sigma_eta^2=0.01, sigma_mu^2=0.05, r=0.05, I=26,
   T=500). Run EM and verify that recovered parameters match the true values within
   a few percent. This is the validation approach used in Figure 4.
   (Paper: Section 2.3.3)

### Edge Cases

1. **Zero-volume bins:** The model requires log(volume) > -inf. Bins with exactly
   zero volume are flagged as missing observations (observed[tau] = false) during
   preprocessing. At these time steps, the Kalman correction step is skipped:
   the prediction step still runs, but x_hat[tau|tau] = x_hat[tau|tau-1] and
   Sigma[tau|tau] = Sigma[tau|tau-1]. This is integrated directly into the Kalman
   filter pseudocode (Step 1, conditional on observed[tau]).
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
   Note on terminology: "day boundary" in the transition matrix refers to tau = kI
   (the last bin of a day), where A[tau] uses a_eta. The M-step set D_start =
   {kI+1} refers to the first-bin-of-day destination indices of these transitions.
   (Paper: Section 2, definition of A_tau and Q_tau)

3. **Half-day trading sessions:** Days with fewer than I bins (e.g., day before
   holidays in U.S. markets) should be excluded entirely, as the seasonal vector
   phi[1..I] assumes a fixed number of bins per day.
   (Paper: Section 4.1, "excluding half-day sessions")

4. **Covariance matrix positive definiteness:** The Joseph form for the covariance
   update (specified in Step 1) guarantees symmetry and positive semi-definiteness
   by construction. This avoids the numerical drift that can occur with the standard
   form Sigma - K*C*Sigma over long sequences.
   Researcher inference: this is a standard Kalman filter implementation practice
   not discussed in the paper.

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
| Initial correction step (explicit) | Implied by Algorithm 1 structure; spec makes explicit |
| Missing observation handling | Researcher inference (standard Kalman practice); paper excludes zeros in Section 4.1 |
| Joseph form for covariance | Researcher inference (standard Kalman practice) |
| Kalman smoother (RTS) | Chen et al. 2016, Section 2.3.1, Algorithm 2 |
| Smoother unchanged in robust variant | Chen et al. 2016, Section 3.2 (implicit: z* used only in E-step filter and M-step) |
| EM algorithm | Chen et al. 2016, Section 2.3.2, Algorithm 3 |
| M-step closed-form updates (pi_1, Sigma_1) | Chen et al. 2016, Appendix A.3, Equations A.32-A.33 |
| M-step update for a_eta | Chen et al. 2016, Appendix A.3, Equation A.34 |
| M-step update for a_mu | Chen et al. 2016, Appendix A.3, Equation A.35 |
| M-step update for sigma_eta^2 (uses a_eta^(j+1)) | Chen et al. 2016, Appendix A.3, Equation A.36 |
| M-step update for sigma_mu^2 (uses a_mu^(j+1)) | Chen et al. 2016, Appendix A.3, Equation A.37 |
| M-step update for r (uses phi[tau]^(j+1)) | Chen et al. 2016, Appendix A.3, Equation A.38 |
| M-step update for phi | Chen et al. 2016, Appendix A.3, Equation A.39 |
| M-step computation order | Chen et al. 2016, Appendix A.3, dependency structure of Equations A.34-A.39 |
| Innovation-based log-likelihood (convergence) | Researcher inference (standard Kalman filter prediction error decomposition); Eq A.8 context |
| Joint log-likelihood (Q function context) | Chen et al. 2016, Appendix A.1, Equation A.8 |
| Robust Lasso extension | Chen et al. 2016, Section 3.1, Equations 25-34 |
| Robust EM modifications (r, phi) | Chen et al. 2016, Section 3.2, Equations 35-36 |
| Soft-thresholding solution | Chen et al. 2016, Section 3.1, Equations 33-34 |
| EM convergence insensitivity | Chen et al. 2016, Section 2.3.3, Figure 4 |
| MAPE results | Chen et al. 2016, Section 4.2, Table 3 |
| VWAP tracking error results | Chen et al. 2016, Section 4.3, Table 4 |
| Robustness to outliers | Chen et al. 2016, Section 3.3, Table 1 |
| Data description and exchanges | Chen et al. 2016, Section 4.1, Table 2 |
| Static VWAP weights | Chen et al. 2016, Section 4.3, Equation 40 |
| Dynamic VWAP weights and interleaving procedure | Chen et al. 2016, Section 4.3, Equation 41 and surrounding text |
| Static prediction day-boundary decomposition | Chen et al. 2016, Section 2, time-varying A_tau definition; explicit form is Researcher inference |
| Log-volume normality evidence | Chen et al. 2016, Section 2, Figure 1 |
| Smoother cross-covariance init | Chen et al. 2016, Appendix A, Equation A.21 |
| Parameter clamping | Researcher inference (standard numerical safeguard) |
| Log-normal bias correction | Researcher inference (standard statistical result) |
| Innovation whiteness check | Researcher inference (standard Kalman diagnostic) |
| Cross-validation grid values | Researcher inference (reasonable defaults) |
| Re-estimation frequency | Researcher inference (daily, not specified in paper) |
| N_obs in log-likelihood constant term | Researcher inference (correct PED form excludes missing obs from constant) |
| First-iteration convergence skip | Researcher inference (standard EM implementation practice) |
| Initial correction e[1]/S[1] mapping to log-likelihood | Researcher inference (connects initial correction naming to PED naming) |
