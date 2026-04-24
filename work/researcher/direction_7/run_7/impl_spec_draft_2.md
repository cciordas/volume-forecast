# Implementation Specification: Kalman Filter State-Space Model for Intraday Volume

## Overview

This specification describes a state-space model for forecasting intraday trading volume
using the Kalman filter. The model decomposes log-volume into three additive components
(daily level, intraday seasonality, intraday dynamic) and estimates all parameters via
the Expectation-Maximization (EM) algorithm with closed-form M-step updates. A robust
extension handles outliers through Lasso-penalized Kalman correction with
soft-thresholding. The model supports two prediction modes (static and dynamic) and two
VWAP execution strategies.

Source: Chen, Feng, and Palomar (2016), "Forecasting Intraday Trading Volume: A Kalman
Filter Approach."

## Algorithm

### Model Description

The model operates on log-volume time series constructed from intraday volume bins
(typically 15-minute intervals). Raw volume is normalized by shares outstanding to
produce turnover, then log-transformed:

    turnover[t,i] = volume[t,i] / shares_outstanding[t]
    y[t,i] = ln(turnover[t,i])

where t = 1,...,T indexes trading days and i = 1,...,I indexes intraday bins. The
log-volume is decomposed additively:

    y[t,i] = eta_t + phi_i + mu[t,i] + v[t,i]

where:
- eta_t: daily average level (log scale). Constant within a day; AR(1) across days.
- phi_i: intraday seasonal pattern (the U-shape). Static across days; one value per bin.
- mu[t,i]: intraday dynamic component. AR(1) at every bin transition.
- v[t,i]: observation noise, i.i.d. N(0, r).

Source: Paper Section 2, Equation 3.

**Assumptions:**
- Constant number of bins I per day (half-day sessions excluded).
- All bins have positive volume (log(0) undefined; zero-volume bins treated as missing).
- Gaussian noise in both state and observation equations.
- eta and phi are not separately identifiable up to a constant offset; the EM resolves
  this split without explicit constraints (phi absorbs per-bin residual, eta absorbs
  daily level). Do NOT impose sum(phi)=0 -- this would conflict with EM updates.
- Researcher inference: the identifiability discussion is implicit in the paper's model
  structure; the EM M-step equations (A.32-A.39) are self-consistent without constraints.

### Pseudocode

#### Step 0: Data Preprocessing

```
Input: volume[t,i] for t=1..T, i=1..I; shares_outstanding[t]
Output: y[1..N], observed[1..N], phi_position[1..N]

N = T * I

for t = 1 to T:
    for i = 1 to I:
        tau = (t - 1) * I + i
        phi_position[tau] = i        # bin position within day (1-based)
        turnover = volume[t,i] / shares_outstanding[t]
        if turnover > 0:
            y[tau] = ln(turnover)
            observed[tau] = true
        else:
            y[tau] = 0.0              # placeholder (not used)
            observed[tau] = false
```

Source: Paper Section 4.1, Equation 1. The global indexing tau = (t-1)*I + i is from
Section 2, below Equation 3.

Helper functions:

```
function is_day_boundary(tau, I):
    # Returns true if the transition from tau to tau+1 crosses a day boundary.
    # Equivalently: tau is the last bin of some day.
    return (tau mod I) == 0

function index_to_day_bin(tau, I):
    t = (tau - 1) / I + 1    # integer division, 1-based day
    i = ((tau - 1) mod I) + 1  # 1-based bin
    return (t, i)
```

Source: Researcher inference (standard index arithmetic from paper's tau convention).

#### Step 1: EM Initialization

```
Input: y[1..N], observed[1..N], phi_position[1..N], I, T
Output: theta = {a_eta, a_mu, sig2_eta, sig2_mu, r, phi[1..I], pi_1[2], Sigma_1[2x2]}

# Compute per-bin means for phi initialization
for i = 1 to I:
    bins_i = {tau : phi_position[tau] == i AND observed[tau]}
    phi[i] = mean(y[tau] for tau in bins_i)

# Compute residuals and daily averages for eta initialization
for tau = 1 to N:
    if observed[tau]:
        resid[tau] = y[tau] - phi[phi_position[tau]]

for t = 1 to T:
    day_bins = {tau : tau in [(t-1)*I+1 .. t*I] AND observed[tau]}
    if day_bins is non-empty:
        daily_avg[t] = mean(resid[tau] for tau in day_bins)
    else:
        daily_avg[t] = 0.0

# AR coefficient initialization
a_eta = clamp(autocorrelation(daily_avg, lag=1), 0.01, 0.99)
a_mu = 0.5

# Variance initialization using stationary AR(1) formula
sig2_eta = max(var(daily_avg) * (1.0 - a_eta^2), 0.01 * var(daily_avg))
sig2_mu = 0.1 * var(resid[observed])   # heuristic fraction
r = 0.1 * var(resid[observed])

# Initial state
pi_1 = [daily_avg[1], 0.0]^T
Sigma_1 = diag(var(daily_avg), var(resid[observed]))
```

Source: Researcher inference. Paper Section 2.3.3 and Figure 4 demonstrate that EM
converges from diverse initial values within a few iterations, so initialization is
not critical. The decomposition-based heuristic above provides a reasonable starting
point that typically reduces the number of EM iterations needed. The floor on sig2_eta
(0.01 * var(daily_avg)) prevents near-singular Sigma_1 when a_eta is close to 1;
researcher inference.

#### Step 2: Kalman Filter (Forward Pass)

**Important convention:** The paper's Algorithm 1 defines pi_1 as the *filtered* state
at tau=1, i.e., x_filt[1] = pi_1, Sigma_filt[1] = Sigma_1. The filter loop predicts
tau+1 from filtered tau and corrects with y[tau+1]. Observation y[1] is NOT processed
by a correction step; it enters only through the M-step (Eqs A.38-A.39 sum over all
tau including tau=1, using smoothed estimates) and indirectly through the smoother's
backward pass. The EM update pi_1 = x_smooth[1] (Eq A.32) propagates information from
all observations back to the initial state.

```
Input: y[1..N], observed[1..N], phi_position[1..N], theta, I, lambda
Output: x_filt[1..N][2], Sigma_filt[1..N][2x2],
        x_pred[2..N][2], Sigma_pred[2..N][2x2],
        A_used[2..N][2x2], e[2..N], S[2..N], z_star[1..N]

# Constants
z_star[1..N] = 0.0    # standard mode: stays zero throughout
C = [1, 1]             # 1x2 observation vector

# tau = 1: initial state IS the filtered state (no correction with y[1])
x_filt[1] = pi_1
Sigma_filt[1] = Sigma_1

for tau = 2 to N:
    # --- Prediction step ---
    if is_day_boundary(tau - 1, I):
        A = [[a_eta, 0], [0, a_mu]]
        Q = [[sig2_eta, 0], [0, sig2_mu]]
    else:
        A = [[1, 0], [0, a_mu]]
        Q = [[0, 0], [0, sig2_mu]]

    A_used[tau] = A
    x_pred[tau] = A @ x_filt[tau - 1]
    Sigma_pred[tau] = A @ Sigma_filt[tau - 1] @ A^T + Q

    # --- Correction step ---
    if observed[tau]:
        # Innovation
        phi_tau = phi[phi_position[tau]]
        y_hat = C @ x_pred[tau] + phi_tau
        e[tau] = y[tau] - y_hat
        S[tau] = C @ Sigma_pred[tau] @ C^T + r    # scalar

        # Robust outlier detection (soft-thresholding)
        # If lambda is very large (e.g. 1e10), threshold is huge and z_star stays 0
        threshold = lambda * S[tau] / 2.0
        if e[tau] > threshold:
            z_star[tau] = e[tau] - threshold
        elif e[tau] < -threshold:
            z_star[tau] = e[tau] + threshold
        else:
            z_star[tau] = 0.0

        e_clean = e[tau] - z_star[tau]

        # Kalman gain (2x1 vector)
        K = Sigma_pred[tau] @ C^T / S[tau]

        # State correction
        x_filt[tau] = x_pred[tau] + K * e_clean

        # Covariance correction (Joseph form for numerical stability)
        I_KC = I_2 - K @ C    # 2x2 matrix
        Sigma_filt[tau] = I_KC @ Sigma_pred[tau] @ I_KC^T + (K @ K^T) * r

    else:
        # Missing observation: skip correction
        x_filt[tau] = x_pred[tau]
        Sigma_filt[tau] = Sigma_pred[tau]
        e[tau] = 0.0     # sentinel, not used in log-likelihood
        S[tau] = 0.0     # sentinel
        z_star[tau] = 0.0
```

Source: Paper Algorithm 1 (page 4). The loop "for tau = 1, 2, ... do" predicts
x_{tau+1|tau} and corrects with y_{tau+1}, so the first correction uses y_2, not y_1.
The implementation above matches this exactly: x_filt[1] = pi_1 is given, and the loop
processes tau=2..N. Robust modification from Section 3.1, Equations 29-34.
Soft-thresholding from Equations 33-34; the equivalence threshold = lambda*S/2 =
lambda/(2*W) holds because W = 1/S (Equation 30). Joseph form covariance update is
researcher inference (standard Kalman filter practice for numerical stability; not
discussed in paper). Missing observation handling is researcher inference (paper excludes
zero-volume bins per Section 4.1).

Note: z_star[1] is always 0 (no correction at tau=1, so no outlier detection).

#### Step 3: RTS Smoother (Backward Pass)

```
Input: x_filt[1..N], Sigma_filt[1..N], x_pred[2..N], Sigma_pred[2..N], A_used[2..N]
Output: x_smooth[1..N][2], Sigma_smooth[1..N][2x2], L_stored[1..N-1][2x2]

# Initialize at final time step
x_smooth[N] = x_filt[N]
Sigma_smooth[N] = Sigma_filt[N]

for tau = N-1 downto 1:
    # Smoother gain (2x2 matrix)
    # L[tau] = Sigma_filt[tau] @ A_used[tau+1]^T @ inv(Sigma_pred[tau+1])
    # Sigma_pred[tau+1] is 2x2; use analytic inverse with epsilon guard

    Sp = Sigma_pred[tau + 1]
    det = Sp[0,0] * Sp[1,1] - Sp[0,1] * Sp[1,0]
    if abs(det) < 1e-12:
        Sp_reg = Sp + 1e-10 * I_2
        det = Sp_reg[0,0] * Sp_reg[1,1] - Sp_reg[0,1] * Sp_reg[1,0]
        Sp = Sp_reg
    Sp_inv = (1.0 / det) * [[Sp[1,1], -Sp[0,1]], [-Sp[1,0], Sp[0,0]]]

    L = Sigma_filt[tau] @ A_used[tau + 1]^T @ Sp_inv
    L_stored[tau] = L

    # Smoothed state and covariance
    x_smooth[tau] = x_filt[tau] + L @ (x_smooth[tau + 1] - x_pred[tau + 1])
    Sigma_smooth[tau] = Sigma_filt[tau] + L @ (Sigma_smooth[tau + 1] - Sigma_pred[tau + 1]) @ L^T
```

Source: Paper Algorithm 2, Equations 10-11. The smoother uses A_used[tau+1] which is
the transition matrix from tau to tau+1. At tau=1, the smoother refines pi_1 using
information from all future observations; this is how y_1's effect propagates to the
initial state through the EM update pi_1 = x_smooth[1]. The analytic 2x2 inverse with
epsilon regularization is researcher inference (standard numerical practice).

#### Step 4: Sufficient Statistics

```
Input: x_smooth[1..N], Sigma_smooth[1..N], L_stored[1..N-1]
Output: P[1..N][2x2], P_cross[2..N][2x2]

# Second-moment matrices (E[x x^T | all data])
for tau = 1 to N:
    P[tau] = Sigma_smooth[tau] + x_smooth[tau] @ x_smooth[tau]^T

# Cross-covariance (non-recursive formula)
# P_cross[tau] = E[x_tau x_{tau-1}^T | all data]
#              = Sigma_cross[tau] + x_smooth[tau] @ x_smooth[tau-1]^T
# where Sigma_cross[tau] = Sigma_smooth[tau] @ L_stored[tau-1]^T
# (Shumway & Stoffer, Property 6.3)
for tau = 2 to N:
    Sigma_cross = Sigma_smooth[tau] @ L_stored[tau - 1]^T
    P_cross[tau] = Sigma_cross + x_smooth[tau] @ x_smooth[tau - 1]^T
```

Source: Paper Appendix A, Equations A.19, A.22. The non-recursive cross-covariance
formula Sigma_cross[tau] = Sigma_smooth[tau] @ L[tau-1]^T is from Shumway and Stoffer
(1982), Property 6.3. This avoids the backward recursion in Equations A.20-A.21
and does not require storing K[N]. Researcher inference on formula choice (the paper
uses the recursive form in Equations A.20-A.21; the non-recursive form is algebraically
equivalent).

#### Step 5: M-Step (Parameter Updates)

All M-step equations have closed-form solutions. Notation: P^(k,l)[tau] denotes the
(k,l) element of the 2x2 matrix P[tau], using 1-based element indices matching the
paper. In 0-based code: P^(1,1) = P[0,0], P^(2,2) = P[1,1], P^(1,2) = P[0,1].

The set D_start = {kI + 1 for k = 1, ..., T-1} contains indices of each day's first
bin (except day 1). |D_start| = T - 1.

```
Input: x_smooth[1..N], Sigma_smooth[1..N], P[1..N], P_cross[2..N],
       y[1..N], observed[1..N], phi_position[1..N], z_star[1..N], I, T, N
Output: theta_new = {a_eta, a_mu, sig2_eta, sig2_mu, r, phi[1..I], pi_1, Sigma_1}

C = [1, 1]

# --- Initial state ---
pi_1 = x_smooth[1]                     # Eq A.32 (= Eq 17)
Sigma_1 = Sigma_smooth[1]              # Equivalent to Eq A.33: P[1] - x_smooth[1] @ x_smooth[1]^T
                                        # Uses Sigma_smooth directly to avoid catastrophic
                                        # cancellation when subtracting two similar PSD matrices.

# --- Daily AR coefficient a_eta ---
# Sum over day transitions: tau in D_start
numer_eta = 0.0
denom_eta = 0.0
for k = 1 to T - 1:
    tau = k * I + 1                     # first bin of day k+1
    numer_eta += P_cross[tau][0, 0]     # P^(1,1)_{tau, tau-1}
    denom_eta += P[tau - 1][0, 0]       # P^(1,1)_{tau-1}
a_eta = numer_eta / denom_eta           # Eq A.34 (= Eq 19)

# --- Intraday AR coefficient a_mu ---
numer_mu = 0.0
denom_mu = 0.0
for tau = 2 to N:
    numer_mu += P_cross[tau][1, 1]      # P^(2,2)_{tau, tau-1}
    denom_mu += P[tau - 1][1, 1]        # P^(2,2)_{tau-1}
a_mu = numer_mu / denom_mu             # Eq A.35 (= Eq 20)

# --- Daily process noise sigma_eta^2 ---
sig2_eta = 0.0
for k = 1 to T - 1:
    tau = k * I + 1
    sig2_eta += P[tau][0, 0] + a_eta^2 * P[tau - 1][0, 0] - 2 * a_eta * P_cross[tau][0, 0]
sig2_eta /= (T - 1)                    # Eq A.36 (= Eq 21)

# --- Intraday process noise sigma_mu^2 ---
sig2_mu = 0.0
for tau = 2 to N:
    sig2_mu += P[tau][1, 1] + a_mu^2 * P[tau - 1][1, 1] - 2 * a_mu * P_cross[tau][1, 1]
sig2_mu /= (N - 1)                     # Eq A.37 (= Eq 22)

# --- Seasonality phi (MUST be computed BEFORE r) ---
# For robust variant, subtract z_star; for standard, z_star=0 so this is a no-op.
for i = 1 to I:
    bins_i = {tau : phi_position[tau] == i AND observed[tau]}
    count_i = |bins_i|
    if count_i > 0:
        phi[i] = (1.0 / count_i) * sum(y[tau] - C @ x_smooth[tau] - z_star[tau]
                                         for tau in bins_i)
    else:
        phi[i] = 0.0  # no observed data for this bin position
    # Eq A.39 (standard) / Eq 36 (robust)

# --- Observation noise r (uses phi_new from above) ---
N_obs = count of tau where observed[tau]
r = 0.0
for tau = 1 to N:
    if observed[tau]:
        phi_tau = phi[phi_position[tau]]
        residual = y[tau] - phi_tau - z_star[tau] - C @ x_smooth[tau]
        r += residual^2 + C @ Sigma_smooth[tau] @ C^T
r /= N_obs                             # Eq A.38 (standard) / Eq 35 (robust)
                                        # N_obs instead of N for missing data correctness

# --- Parameter clamping ---
EPSILON = 1e-8
a_eta = clamp(a_eta, EPSILON, 1.0 - EPSILON)
a_mu = clamp(a_mu, EPSILON, 1.0 - EPSILON)
sig2_eta = max(sig2_eta, EPSILON)
sig2_mu = max(sig2_mu, EPSILON)
r = max(r, EPSILON)
```

Source: Paper Appendix A, Equations A.32-A.39 (standard model). Robust M-step
modifications from Equations 35-36 (only phi and r change; all other equations are
identical because z_star affects only the observation equation). The compact r formula
with C @ Sigma_smooth @ C^T is derived by expanding P = Sigma_smooth + x_smooth @
x_smooth^T and collecting terms; this is algebraically equivalent to Equation A.38.

**M-step ordering constraints** (3 dependencies):
1. phi MUST be computed before r, because r uses phi^(j+1) (Eq A.38 depends on A.39).
2. a_eta MUST be computed before sig2_eta (Eq A.36 uses a_eta^(j+1)).
3. a_mu MUST be computed before sig2_mu (Eq A.37 uses a_mu^(j+1)).

Source: Paper Appendix A, Equations A.34-A.39. The ordering constraint for phi/r follows
from the derivation in Equations A.30-A.31 where the partial derivative for phi_tau uses
the condition that r has absorbed the new phi.

**Note on y[1] in M-step:** The r and phi updates (Eqs A.38-A.39) sum over all tau
from 1 to N, including tau=1. Even though y[1] is not processed by a correction step
in the Kalman filter (Step 2), y[1] contributes to the M-step through x_smooth[1] and
Sigma_smooth[1], which are computed by the smoother. This is consistent with the paper's
convention where pi_1 = x_smooth[1] absorbs information from all observations.

#### Step 6: Convergence Monitoring

```
Input: e[2..N], S[2..N], observed[1..N]
Output: log_likelihood (scalar)

# Innovation-form log-likelihood (only over corrected observations: tau=2..N)
N_corrected = count of tau in {2..N} where observed[tau]
log_lik = -0.5 * sum(ln(S[tau]) + e[tau]^2 / S[tau] for tau in {2..N} where observed[tau])
         - (N_corrected / 2.0) * ln(2 * pi)
```

Source: Researcher inference (standard Kalman filter innovation log-likelihood; Shumway
and Stoffer, 1982). The sum runs over tau=2..N because tau=1 has no correction step and
therefore no innovation. The paper does not specify the convergence criterion. For the
robust variant, e[tau] in this formula is the raw innovation (before subtracting z_star),
and the log-likelihood serves as a convergence heuristic rather than the exact
observed-data log-likelihood (the Lasso penalty on z makes the robust EM not a standard
EM, so monotonic increase is not strictly guaranteed). A parameter-change fallback
criterion max(|theta_new - theta_old| / (|theta_old| + 1e-10)) < tol can be used as
backup.

#### Step 7: EM Main Loop

```
Input: y[1..N], observed[1..N], phi_position[1..N], I, T, lambda,
       theta_init, max_iter, tol
Output: theta_final, x_filt_final[1..N], Sigma_filt_final[1..N]

theta = theta_init
log_lik_prev = -infinity

for j = 1 to max_iter:
    # E-step
    x_filt, Sigma_filt, x_pred, Sigma_pred, A_used, e, S, z_star
        = KalmanFilter(y, observed, phi_position, theta, I, lambda)     # Step 2

    x_smooth, Sigma_smooth, L_stored
        = RTSSmoother(x_filt, Sigma_filt, x_pred, Sigma_pred, A_used)  # Step 3

    P, P_cross
        = SufficientStatistics(x_smooth, Sigma_smooth, L_stored)       # Step 4

    # Convergence check (placed AFTER E-step, BEFORE M-step)
    log_lik = InnovationLogLikelihood(e, S, observed)                   # Step 6

    relative_change = abs(log_lik - log_lik_prev) / max(abs(log_lik_prev), 1.0)
    if relative_change < tol AND j > 1:
        break    # theta from previous M-step is final; consistent with current E-step

    log_lik_prev = log_lik

    # M-step
    theta = MStep(x_smooth, Sigma_smooth, P, P_cross,
                  y, observed, phi_position, z_star, I, T, N)           # Step 5

theta_final = theta
x_filt_final = x_filt
Sigma_filt_final = Sigma_filt
```

Source: Paper Algorithm 3. The convergence check placement (after E-step, before M-step)
is researcher inference (standard EM practice). When the loop exits due to convergence
at iteration j, theta_final holds M-step parameters from iteration j-1, and the E-step
outputs are from iteration j run with those parameters -- these are consistent. The
filtered state at the last bin (x_filt[N], Sigma_filt[N]) is the starting point for
prediction.

**Robust/standard unification:** The same code path handles both modes. For the standard
Kalman filter, set lambda = 1e10 (or any very large value). The soft-thresholding
threshold in Step 2 becomes lambda * S / 2, which for lambda = 1e10 is approximately
5e9 * S -- far larger than any plausible innovation. All z_star values remain zero,
making the robust code path identical to the standard filter. The M-step phi and r
updates (Eqs 35-36) reduce to the standard updates (Eqs A.38-A.39) when z_star = 0.
Source: Researcher inference; Paper Sections 3.1-3.2 (Eqs 33-36).

#### Step 8: Static Prediction (Multi-Step Ahead)

```
Input: x_filt_last[2] (= x_filt[N]), theta, I
Output: y_hat_static[1..I], volume_hat_static[1..I]

# Predict all I bins of the next day from end-of-day state
x_curr = x_filt_last

for h = 1 to I:
    # First bin crosses day boundary; subsequent bins are within-day
    if h == 1:
        A = [[a_eta, 0], [0, a_mu]]
    else:
        A = [[1, 0], [0, a_mu]]

    x_curr = A @ x_curr
    y_hat_static[h] = C @ x_curr + phi[h]
    volume_hat_static[h] = exp(y_hat_static[h])
```

Source: Paper Section 2.2, Equation 9. The day-boundary transition at h=1 follows from
Equations 4-5 (A_tau uses a_eta when transitioning across days). Note that
exp(y_hat_static) produces the geometric-mean volume estimate (turnover units), not the
arithmetic mean. For paper MAPE benchmarks, use exp(y_hat) directly.

**Log-normal bias correction** (optional, for production use):

    volume_hat_corrected[h] = exp(y_hat_static[h] + 0.5 * prediction_variance[h])

where prediction_variance[h] = C @ Sigma_curr @ C^T + r, with Sigma_curr propagated
through the same A/Q matrices as x_curr.

Source: Researcher inference (standard log-normal property; E[V] = exp(E[log V] +
Var(log V)/2) > exp(E[log V]) by Jensen's inequality). The paper does not discuss or
apply this correction.

#### Step 9: Dynamic Prediction (One-Step Ahead)

```
Input: x_filt_prev[2], Sigma_filt_prev[2x2], y_new (scalar), theta, bin_index, I, lambda
Output: x_filt_new[2], Sigma_filt_new[2x2], y_hat (scalar), volume_hat (scalar)

# Determine transition matrices
if bin_index == 1:    # first bin of day: day boundary crossing
    A = [[a_eta, 0], [0, a_mu]]
    Q = [[sig2_eta, 0], [0, sig2_mu]]
else:
    A = [[1, 0], [0, a_mu]]
    Q = [[0, 0], [0, sig2_mu]]

# Predict
x_pred = A @ x_filt_prev
Sigma_pred = A @ Sigma_filt_prev @ A^T + Q
y_hat = C @ x_pred + phi[bin_index]
volume_hat = exp(y_hat)

# Correct (same as Step 2 correction)
e = y_new - y_hat
S = C @ Sigma_pred @ C^T + r

threshold = lambda * S / 2.0
if e > threshold:
    z_star = e - threshold
elif e < -threshold:
    z_star = e + threshold
else:
    z_star = 0.0

e_clean = e - z_star
K = Sigma_pred @ C^T / S
x_filt_new = x_pred + K * e_clean
I_KC = I_2 - K @ C
Sigma_filt_new = I_KC @ Sigma_pred @ I_KC^T + (K @ K^T) * r
```

Source: Paper Algorithm 1 (same as Step 2, applied one bin at a time). The day-boundary
check at bin_index==1 is consistent with Equations 4-5 and the dynamic VWAP procedure.

**Production note:** As written, this function processes a single observation. In a
production system where dynamic predictions are made sequentially through a trading day,
maintain the running filter state (x_filt, Sigma_filt) externally and call this function
once per new observation, passing in the previous state. This avoids re-running the
filter from scratch. Source: Researcher inference.

#### Step 10: Static Forecast of Remaining Bins

```
Input: x_state[2], bin_start, I, theta
Output: volume_hat_remaining[bin_start..I]

# Forecast bins bin_start through I from current state (no correction steps)
x_curr = x_state
for h = bin_start to I:
    if h == bin_start:
        # No additional transition needed; x_state is already the predicted state
        # for bin_start (either from initial day-boundary transition or from
        # within-day prediction)
        pass
    else:
        A = [[1, 0], [0, a_mu]]    # within-day transition
        x_curr = A @ x_curr

    y_hat = C @ x_curr + phi[h]
    volume_hat_remaining[h] = exp(y_hat)
```

Source: Paper Section 2.2, Equation 9 (static prediction applied to remaining bins
within a day). Researcher inference on the integration with dynamic VWAP.

#### Step 11: Dynamic VWAP Execution

```
Input: x_filt_end_of_prev_day[2], Sigma_filt_end_of_prev_day[2x2],
       y_live[1..I] (observed one at a time), theta, I, lambda
Output: weights[1..I]

# State from previous day's last bin
x_filt_prev = x_filt_end_of_prev_day
Sigma_filt_prev = Sigma_filt_end_of_prev_day
cumulative_weight = 0.0

for i = 1 to I:
    if i < I:
        # Predict this bin and all remaining bins
        # Day boundary at i==1, within-day otherwise
        if i == 1:
            A = [[a_eta, 0], [0, a_mu]]
            Q = [[sig2_eta, 0], [0, sig2_mu]]
        else:
            A = [[1, 0], [0, a_mu]]
            Q = [[0, 0], [0, sig2_mu]]

        x_pred_i = A @ x_filt_prev
        # Volume forecasts for bin i and all remaining bins i..I
        volume_hat = StaticForecastRemaining(x_pred_i, i, I, theta)   # Step 10

        # Weight for bin i: proportion of remaining volume
        vol_remaining = sum(volume_hat[i..I])
        weight_i = (volume_hat[i] / vol_remaining) * (1.0 - cumulative_weight)
        weights[i] = weight_i
        cumulative_weight += weight_i

        # Observe y_live[i] and correct state
        x_filt_new, Sigma_filt_new, _, _
            = DynamicPredict(x_filt_prev, Sigma_filt_prev,
                             y_live[i], theta, i, I, lambda)           # Step 9
        x_filt_prev = x_filt_new
        Sigma_filt_prev = Sigma_filt_new

    else:
        # Last bin: absorb all remaining weight
        weights[I] = 1.0 - cumulative_weight
```

Source: Paper Section 4.3, Equations 40-41. The dynamic VWAP weight formula (Eq 41)
computes w[i] as the proportion of predicted remaining volume, scaled by the unexecuted
fraction. The volume forecasts for the weight computation use the pre-correction state
(x_pred_i, conditioned on information up to bin i-1), not the post-correction state.
After computing the weight, the observation is used to correct the state for the next
iteration. The day-boundary transition at i==1 is from Equations 4-5.

**Efficiency note:** As written, StaticForecastRemaining is called I-1 times, each
producing a multi-step forecast. Since x_pred_i is already the predicted state for bin i,
and StaticForecastRemaining propagates forward from there, the total cost is O(I^2) in
the number of bins, which is acceptable for typical values of I (e.g., 26). The dominant
cost in the EM calibration loop is the Kalman filter at O(N) per iteration. Source:
Researcher inference.

#### Step 12: Static VWAP Weights

```
Input: volume_hat_static[1..I]     # from Step 8
Output: weights_static[1..I]

total = sum(volume_hat_static[1..I])
for i = 1 to I:
    weights_static[i] = volume_hat_static[i] / total
```

Source: Paper Section 4.3, Equation 40.

#### Step 13: Evaluation Metrics

```
Input: y_actual[1..I] (log-volume), y_pred[1..I] (log-volume predictions)
Output: mape (scalar), tracking_error (scalar, basis points)

# --- MAPE (Mean Absolute Percentage Error) ---
# Computed in LINEAR scale, not log scale. No Jensen's bias correction.
vol_actual = exp(y_actual[1..I])     # linear-scale volumes
vol_pred = exp(y_pred[1..I])         # linear-scale predictions
mape = mean(|vol_actual[i] - vol_pred[i]| / vol_actual[i] for i = 1 to I)

# --- VWAP Tracking Error ---
# Difference between execution VWAP (using model weights) and market VWAP,
# measured in basis points.
# Market VWAP uses actual volume proportions as weights.
market_weights[i] = vol_actual[i] / sum(vol_actual)   # actual volume fractions
# model_weights[i] from Step 11 (dynamic) or Step 12 (static)

# Assuming price[i] is the VWAP price for bin i:
vwap_market = sum(market_weights[i] * price[i] for i = 1 to I)
vwap_model = sum(model_weights[i] * price[i] for i = 1 to I)
tracking_error = abs(vwap_model - vwap_market) / vwap_market * 10000   # basis points
```

Source: Paper Section 3.3, Equation 37 (MAPE); Section 4.3, Equation 42 (tracking
error). The MAPE formula uses exp(y) to convert back to linear scale; verified against
Eq 37 which operates on volume (linear), not log-volume. The paper does not apply
Jensen's bias correction to the predictions before computing MAPE.

#### Step 14: Rolling-Window Calibration with Cross-Validation

```
Input: y_all (full time series), shares_out_all, I,
       N_candidates (list of training window lengths in days),
       lambda_candidates (list of lambda values),
       validation_days (number of days for validation, e.g. 100)
Output: best_N_days, best_lambda

# Validation period: last validation_days days before out-of-sample start
# Training periods: preceding N_days days for each candidate

best_mape = +infinity
for N_days in N_candidates:
    for lambda in lambda_candidates:
        mapes = []
        for each validation day d:
            # Train: EM on N_days days ending just before d
            T_train = N_days
            N_train = T_train * I
            y_train = extract_window(y_all, d, T_train, I)

            theta = EMCalibrate(y_train, ..., lambda, max_iter, tol)

            # Predict day d using dynamic mode
            y_pred = DynamicPredictDay(theta, y_actual_day_d, I, lambda)

            # Compute MAPE for day d (using Step 13)
            mape_d = compute_mape(y_actual_day_d, y_pred)
            mapes.append(mape_d)

        avg_mape = mean(mapes)
        if avg_mape < best_mape:
            best_mape = avg_mape
            best_N_days = N_days
            best_lambda = lambda
```

Source: Paper Section 4.1. The paper uses January-May 2015 as validation, June 2015 to
June 2016 as out-of-sample. The two-level structure (EM for fixed parameters, CV over
hyperparameters) is explicit in the paper. The rolling-window evaluation within
validation is researcher inference (paper does not fully specify the CV inner loop).

**EM warm start:** When re-estimating parameters daily in the rolling window, initialize
EM with the previous day's calibrated parameters. This typically reduces iterations
from 10-20 (cold start) to 2-5 (warm start). Researcher inference; paper does not
discuss.

### Data Flow

```
Raw volume [T x I]
    |
    v  (normalize by shares_outstanding, log transform)
Log-volume y [N] where N = T * I
    |
    v  (EM calibration loop)
    +----> Kalman Filter (forward) --> x_filt [N x 2], Sigma_filt [N x 2 x 2]
    |          |                       (x_filt[1]=pi_1; corrections for tau=2..N)
    |          v
    |      RTS Smoother (backward) --> x_smooth [N x 2], Sigma_smooth [N x 2 x 2]
    |          |
    |          v
    |      Sufficient Statistics --> P [N x 2 x 2], P_cross [N-1 x 2 x 2]
    |          |
    |          v
    |      M-Step --> theta_new
    |          |
    |          v (iterate until convergence)
    +<---------+
    |
    v  (prediction with calibrated theta)
    +----> Static Prediction (Step 8) --> y_hat_static [I], volume_hat_static [I]
    |          |
    |          v
    |      Static VWAP Weights (Step 12) --> weights_static [I]
    |
    +----> Dynamic Prediction (Step 9, one bin at a time)
               |
               v
           Dynamic VWAP (Step 11) --> weights_dynamic [I]
```

**Types and shapes:**

| Quantity | Type | Shape | Description |
|----------|------|-------|-------------|
| y | float64 | (N,) | Log-volume observations |
| observed | bool | (N,) | Whether each bin has valid data |
| phi_position | int | (N,) | Bin index within day (1 to I) |
| x_filt, x_smooth | float64 | (N, 2) | State vectors (x_filt[0] = pi_1) |
| x_pred | float64 | (N-1, 2) | Predicted states (indexed 2..N) |
| Sigma_filt, Sigma_smooth | float64 | (N, 2, 2) | State covariance matrices |
| Sigma_pred | float64 | (N-1, 2, 2) | Predicted covariances (indexed 2..N) |
| A_used | float64 | (N-1, 2, 2) | Transition matrices (indexed 2..N) |
| L_stored | float64 | (N-1, 2, 2) | Smoother gains (indexed 1..N-1) |
| P | float64 | (N, 2, 2) | Second-moment matrices |
| P_cross | float64 | (N-1, 2, 2) | Cross-moment matrices (indexed 2..N) |
| e | float64 | (N-1,) | Innovations (indexed 2..N; no innovation at tau=1) |
| S | float64 | (N-1,) | Innovation variances (indexed 2..N) |
| z_star | float64 | (N,) | Outlier corrections (z_star[1]=0 always) |
| phi | float64 | (I,) | Seasonality vector |
| pi_1 | float64 | (2,) | Initial state mean |
| Sigma_1 | float64 | (2, 2) | Initial state covariance |
| C | float64 | (2,) | Observation vector [1, 1] |
| a_eta, a_mu | float64 | scalar | AR coefficients |
| sig2_eta, sig2_mu, r | float64 | scalar | Noise variances |
| lambda | float64 | scalar | Lasso penalty (robust only) |

### Variants

**Implemented variant:** Robust Kalman Filter (Section 3). This subsumes the standard
Kalman filter as a special case when lambda is set to a very large value (e.g., 1e10),
which makes the soft-thresholding threshold so large that z_star is always zero.

**Justification:** The robust variant adds negligible computational overhead (one
soft-thresholding operation per bin) and performs equal or better than the standard
variant on both clean and contaminated data (Table 1: MAPE difference <= 0.01 on clean
data; large improvements under outlier contamination). Implementing a single robust code
path avoids maintaining two separate implementations.

Source: Paper Section 3, Tables 1 and 3.

## Parameters

| Parameter | Description | Recommended Value | Sensitivity | Range |
|-----------|-------------|-------------------|-------------|-------|
| a_eta | Day-to-day AR coefficient for daily level | Data-driven (EM); typically 0.95-0.99 | Medium | (0, 1) |
| a_mu | Bin-to-bin AR coefficient for intraday dynamic | Data-driven (EM); typically 0.4-0.6 | Medium | (0, 1) |
| sig2_eta | Process noise variance for daily component | Data-driven (EM); typically 0.01-0.05 | Medium | (0, inf) |
| sig2_mu | Process noise variance for intraday dynamic | Data-driven (EM); typically 0.01-0.1 | Medium | (0, inf) |
| r | Observation noise variance | Data-driven (EM); typically 0.05-0.1 | Medium | (0, inf) |
| phi[1..I] | Intraday seasonality vector | Data-driven (EM); U-shaped | High (structural) | (-inf, inf) |
| pi_1 | Initial state mean | Data-driven (EM) | Very low | R^2 |
| Sigma_1 | Initial state covariance | Data-driven (EM) | Very low | PSD 2x2 |
| lambda | Lasso regularization (robust only) | Cross-validation; grid 0.01-100 | High | (0, inf); 1e10 for standard |
| N_days | Training window length (days) | Cross-validation; candidates 60-500 | Medium-High | >= 20 days |
| I | Number of intraday bins per day | Exchange-dependent; 26 for NYSE (6.5h/15min) | Configuration | 1-52 |
| max_iter | Maximum EM iterations | 100 | Very low | 20-200 |
| tol | EM convergence tolerance (relative) | 1e-8 | Very low | 1e-6 to 1e-10 |

Typical AR coefficient values from Figure 4 (synthetic convergence experiments):
a_eta converges near 0.98, a_mu near 0.5, r near 0.05.

Source: Paper Section 2 (model parameters), Section 4.1 (cross-validation), Figure 4
(synthetic convergence). The "typical" ranges are from Figure 4; actual values are
always data-driven via EM. Lambda is never estimated by EM; it requires cross-validation
(Section 4.1).

**Lambda grid construction guidance:** Since the effective soft-thresholding cutoff is
lambda * S[tau] / 2, and S[tau] varies across bins and EM iterations, a useful anchoring
strategy is: (1) fit the standard KF first (lambda = 1e10), (2) collect the innovation
variances S[tau] from the final E-step, (3) set lambda_candidates as multiples of
1/median(sqrt(S)), e.g., [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
times 1/median(sqrt(S)). This scales the grid to the data's natural innovation scale.
Researcher inference; paper does not specify lambda grid values.

### Initialization

See Step 1 (EM Initialization) in the pseudocode above. Key points:

1. **phi:** Per-bin mean of observed log-volumes.
2. **eta (daily level):** Mean of residuals (y - phi) per day.
3. **a_eta:** Sample autocorrelation of daily averages at lag 1, clamped to [0.01, 0.99].
4. **a_mu:** Fixed at 0.5 (moderate starting guess).
5. **Variance terms:** Derived from sample statistics using stationary AR(1) formulas.
   sig2_eta has a floor of 0.01 * var(daily_avg) to prevent near-singular Sigma_1.
6. **pi_1:** [daily_avg[1], 0]^T (first day's level, zero dynamic component).
7. **Sigma_1:** Diagonal with variances of daily averages and residuals.

The paper (Section 2.3.3, Figure 4) demonstrates that EM converges from wildly different
initial values to the same parameter estimates within a few iterations. Initialization
quality affects only the number of iterations, not the final result.

**EM warm start for rolling windows:** Initialize with previous day's calibrated theta.
Reduces iterations from ~10-20 to ~2-5. Source: Researcher inference.

### Calibration

**Level 1: EM parameter estimation** (for fixed N_days, lambda):
Run the EM algorithm (Step 7) on the training window. All parameters except lambda are
estimated. Typical convergence in 5-20 iterations from cold start.

**Level 2: Cross-validation over hyperparameters** (N_days, lambda):
Grid search over candidate pairs. For each pair, evaluate dynamic-mode MAPE on a
validation period (e.g., 100 trading days before out-of-sample start). Select the pair
with lowest average validation MAPE. See Step 14 for full procedure.

**Candidate grids:**
- N_days: [60, 100, 150, 200, 300, 500] trading days.
- lambda: log-spaced from 0.01 to 100 (15-20 points), plus 1e10 as an "infinity"
  candidate (allows CV to automatically select standard KF if robustness is unnecessary).
  See lambda grid construction guidance in Parameters section for data-adaptive scaling.

Source: Paper Section 4.1 (cross-validation on January-May 2015 data). The specific
grid values are researcher inference; the paper does not report them.

**Re-estimation frequency:** Daily (shift training window by I bins each day).
Computationally feasible because the state dimension is 2, so all Kalman operations
are O(1) per bin, making a full EM pass O(N) where N = T*I. For a 250-day window
with I=26, N=6500, and a single EM iteration involves ~65k scalar operations.
Source: Researcher inference.

## Validation

### Expected Behavior

**Volume prediction MAPE** (out-of-sample, 30 securities, D=250 days):

| Model | Dynamic MAPE | Static MAPE |
|-------|-------------|-------------|
| Robust Kalman Filter | 0.46 | 0.61 |
| Standard Kalman Filter | 0.47 | 0.62 |
| CMEM (dynamic/static) | 0.65 | 0.90 |
| Rolling Mean | N/A | 1.28 |

Source: Paper Table 3. Robust KF improves over RM by 64% and over CMEM by 29% (dynamic).
Rolling Mean has no dynamic variant (it does not update intraday), so only static MAPE
is reported.

**VWAP tracking error** (out-of-sample, basis points):

| Model | Dynamic TE | Static TE |
|-------|-----------|----------|
| Robust Kalman Filter | 6.38 | 6.85 |
| Standard Kalman Filter | 6.39 | 6.89 |
| CMEM | 7.01 | 7.71 |
| Rolling Mean | N/A | 7.48 |

Source: Paper Table 4, Equation 42. Robust KF improves over RM by ~15% (static) and
over CMEM by ~9% (dynamic). Rolling Mean has no dynamic variant, so only static TE is
reported.

**Per-ticker MAPE range** (dynamic, robust KF):
- Best: AAPL at 0.21 (mean), std 0.17.
- Worst: 2800HK (leveraged ETF) at 1.94.
- Most U.S. large-cap stocks: 0.21-0.42.
- Source: Paper Table 3.

### Sanity Checks

1. **Synthetic parameter recovery:** Generate data from known parameters (a_eta=0.98,
   a_mu=0.5, sig2_eta=0.01, sig2_mu=0.05, r=0.05, I=26, T=500). Run EM. Recovered
   parameters should match within 5% of true values.
   Source: Paper Section 2.3.3, Figure 4.

2. **Log-likelihood monotonicity:** For standard model (lambda=1e10), innovation
   log-likelihood must be non-decreasing across EM iterations. A decrease signals an
   implementation bug. For robust model, monotonicity is not strictly guaranteed (see
   Step 6 notes).
   Source: Researcher inference (standard EM property).

3. **Robust = standard on clean data:** With lambda=1e10 (standard mode), all z_star
   should be identically zero. MAPE should match standard KF implementation exactly.
   Source: Paper Table 1 (comparable performance on clean data).

4. **AR coefficients in (0, 1):** After EM, both a_eta and a_mu should be strictly
   between 0 and 1. Values outside suggest model misspecification or data issues.
   Source: Paper model assumptions (stationary AR(1)).

5. **Phi U-shape:** The estimated phi vector should exhibit the characteristic U-shape
   (higher values at market open and close, lower in midday). This is the well-documented
   intraday volume seasonality pattern.
   Source: Paper Section 2 (intraday periodic component).

6. **Beat rolling mean:** On any reasonable dataset, the Kalman filter (both standard
   and robust) should achieve lower MAPE than the rolling mean baseline.
   Source: Paper Table 3 (KF outperforms RM on all 30 securities).

7. **Multi-initialization convergence:** Run EM from 5 different random initializations.
   All should converge to the same parameter values (within tolerance).
   Source: Paper Section 2.3.3, Figure 4.

8. **Contaminated data robustness:** Inject outliers (10x/0.1x volume spikes) into 10%
   of bins. Robust KF MAPE should degrade much less than standard KF.
   Source: Paper Section 3.3, Table 1.

9. **Prediction variance grows with horizon:** In static prediction, the prediction
   variance C @ Sigma_curr @ C^T + r should increase with forecast horizon h.
   Source: Researcher inference (uncertainty accumulates).

10. **VWAP weights sum to 1:** Both static and dynamic VWAP weights must sum to
    exactly 1.0.
    Source: Paper Equations 40-41 (weights are proportions of total).

### Edge Cases

1. **Zero-volume bins:** Log(0) is undefined. Flag as missing (observed[tau] = false).
   The Kalman filter skips correction for missing bins (prediction passes through
   unchanged). The M-step sums for phi and r exclude unobserved bins and use N_obs
   denominators.
   Source: Paper Section 4.1 (excludes zero-volume bins). Missing data handling is
   researcher inference.

2. **Half-day sessions:** Days with fewer than I bins. Exclude entirely from training
   and prediction. The seasonality vector phi assumes exactly I bins per day.
   Source: Paper Section 4.1 ("excluding half-day sessions").

3. **Sigma_pred near-singular in smoother:** The 2x2 Sigma_pred[tau+1] must be inverted
   in the RTS smoother. If the determinant is below 1e-12, add 1e-10 to the diagonal
   before inverting. This can occur when observation noise r is very small relative to
   process noise.
   Source: Researcher inference (numerical safeguard).

4. **Lambda at CV grid boundary:** If the optimal lambda falls at the boundary of the
   search grid, extend the grid. lambda -> 0 suggests severe outlier contamination
   requiring investigation. lambda -> inf (1e10 selected) means standard KF suffices.
   Source: Researcher inference.

5. **Lambda = 0:** Degenerate case. Every innovation is classified as an outlier
   (z_star = e always), so no information from observations reaches the state estimate.
   Lambda must be strictly positive. Enforce lambda > 0 in the CV grid.
   Source: Researcher inference (follows from Equations 33-34).

6. **Extreme log-volumes:** After normalization, log-volumes can have extreme values.
   Consider winsorizing at +/- 5 standard deviations before calibration to prevent
   EM convergence issues.
   Source: Researcher inference.

7. **Minimum training window:** The M-step requires at least T >= 2 days (T-1 >= 1
   day transitions) to estimate a_eta and sig2_eta. Practically, at least 20 trading
   days are needed for stable estimates. If N_days < 2, the D_start set is empty and
   the M-step formulas for a_eta (Eq A.34) and sig2_eta (Eq A.36) have zero denominators.
   Source: Researcher inference (follows from M-step equations).

8. **Float64 precision:** Use float64 throughout. Long EM iteration chains and small
   normalized volumes (amplified by log transform) can accumulate rounding errors with
   lower precision.
   Source: Researcher inference.

9. **Covariance positive-definiteness:** Verify that Sigma_filt and Sigma_pred have
   positive diagonal elements and positive determinant at every tau. The Joseph form
   covariance update (Step 2) provides this guarantee in exact arithmetic; floating-point
   accumulation over very long sequences may still require monitoring.
   Source: Researcher inference.

### Known Limitations

1. **No zero-volume bins:** The model assumes all bins have positive volume. Zero-volume
   bins must be excluded or imputed. The paper does not address this.
   Source: Paper Section 4.1.

2. **No exogenous covariates:** The model uses only volume history. It cannot incorporate
   volatility, spread, order flow, or other predictive signals.
   Source: Paper Section 5 (future work mentions additional covariates).

3. **Single-security model:** Each security is modeled independently. No cross-sectional
   information sharing.
   Source: Paper Section 4 (all experiments are per-security).

4. **AR(1) dynamics only:** Both eta and mu follow first-order autoregressive processes.
   Higher-order dependencies, regime switching, or nonlinear dynamics are not captured.
   Source: Paper Section 2, Equations 4-5.

5. **Static seasonality:** The phi vector is estimated once during calibration and does
   not adapt to intraday pattern changes (e.g., inverted-J on gap days, event-driven
   pattern shifts).
   Source: Paper Equation A.39 (phi is a simple average over all training days).

6. **Static prediction degrades with horizon:** In static prediction, the mu component
   decays geometrically as (a_mu)^h. With a_mu ~ 0.5, by mid-day (h=13): 0.5^13 ~
   0.0001, effectively zero. Static predictions rely almost entirely on eta + phi for
   later bins.
   Source: Paper Section 2.2, Equation 9.

7. **No intraday updating of daily component:** Within a trading day, eta is held
   strictly constant (A_tau has 1 in position (1,1) for within-day transitions,
   Q_tau has 0). If early bins reveal unexpectedly high/low volume, the model adapts
   only through mu, not eta. Eta is revised only at the next day boundary.
   Source: Paper Section 2, Equations 4-5 (structural consequence of the state-space
   formulation).

8. **Gaussian noise assumption:** The model assumes Gaussian noise for both state
   transitions and observations. The robust variant partially mitigates heavy-tailed
   observation noise but does not address non-Gaussian state dynamics.
   Source: Paper Section 2 (model assumptions).

## Paper References

| Spec Section | Paper Source | Notes |
|-------------|-------------|-------|
| Volume decomposition (y = eta + phi + mu + v) | Section 2, Equation 3 | |
| Turnover normalization | Section 4.1, Equation 1 | |
| State-space form (x, A, C, Q) | Section 2, Equations 4-5 | |
| State vector x = [eta, mu]^T | Section 2, below Eq 4 | |
| Observation vector C = [1, 1] | Section 2, below Eq 4 | |
| Time-varying A_tau (day boundary) | Section 2, below Eq 4 | |
| Time-varying Q_tau (day boundary) | Section 2, below Eq 4 | |
| Kalman filter: x_filt[1]=pi_1, loop tau=2..N | Algorithm 1, page 4 | Loop starts at tau=1, predicts tau+1, corrects with y_{tau+1} |
| Kalman gain formula | Algorithm 1, line 4 | |
| Kalman correction | Algorithm 1, lines 5-6 | |
| Joseph form covariance update | N/A | Researcher inference (standard practice) |
| Missing observation handling | N/A | Researcher inference; paper excludes zero-volume bins |
| RTS smoother algorithm | Algorithm 2, Equations 10-11 | |
| Smoother gain L_tau | Algorithm 2, line 2 | |
| Non-recursive cross-covariance | Shumway & Stoffer (1982) | Researcher inference; paper uses recursive A.20-A.21 |
| Sufficient statistics P, P_cross | Appendix A, Equations A.19, A.22 | |
| M-step: pi_1 | Appendix A, Equation A.32 (= Eq 17) | |
| M-step: Sigma_1 = Sigma_smooth[1] | Appendix A, Equation A.33 (= Eq 18) | Researcher inference: equivalent form avoiding cancellation |
| M-step: a_eta | Appendix A, Equation A.34 (= Eq 19) | |
| M-step: a_mu | Appendix A, Equation A.35 (= Eq 20) | |
| M-step: sig2_eta | Appendix A, Equation A.36 (= Eq 21) | Uses a_eta^(j+1) |
| M-step: sig2_mu | Appendix A, Equation A.37 (= Eq 22) | Uses a_mu^(j+1) |
| M-step: r | Appendix A, Equation A.38 (= Eq 23) | Uses phi^(j+1); sums over all tau including tau=1 |
| M-step: phi | Appendix A, Equation A.39 (= Eq 24) | Sums over all tau including tau=1 |
| M-step ordering (phi before r) | Appendix A, Eqs A.30-A.31 derivation | |
| y[1] in M-step but not filter | Algorithm 1 + Eqs A.38-A.39 | y[1] enters M-step via x_smooth[1], not via filter correction |
| Robust observation model (z_tau) | Section 3.1, Equations 25-27 | |
| Robust Lasso formulation | Section 3.1, Equations 28-30 | |
| Soft-thresholding solution | Section 3.1, Equations 33-34 | |
| Robust M-step: r modification | Section 3.2, Equation 35 | |
| Robust M-step: phi modification | Section 3.2, Equation 36 | |
| Static prediction | Section 2.2, Equation 9 | |
| Dynamic prediction | Section 2.2 (one-bin-ahead) | |
| MAPE formula | Section 3.3, Equation 37 | Linear scale; no bias correction |
| VWAP formula | Section 4.3, Equation 39 | |
| Static VWAP weights | Section 4.3, Equation 40 | |
| Dynamic VWAP weights | Section 4.3, Equation 41 | |
| VWAP tracking error | Section 4.3, Equation 42 | |
| EM convergence (innovation LL) | N/A | Researcher inference; Shumway & Stoffer (1982) |
| Parameter clamping | N/A | Researcher inference (numerical safeguard) |
| Sigma_1 initialization floor | N/A | Researcher inference (prevents near-singular initial covariance) |
| Log-normal bias correction | N/A | Researcher inference (standard log-normal property) |
| EM warm start in rolling windows | N/A | Researcher inference |
| EM initialization heuristics | N/A | Researcher inference; paper shows robustness (Section 2.3.3, Figure 4) |
| Cross-validation procedure | Section 4.1 | Grid values and lambda scaling are researcher inference |
| Dynamic MAPE benchmarks | Table 3 | Average over 30 securities |
| Static MAPE benchmarks | Table 3 | Average over 30 securities |
| Dynamic VWAP TE benchmarks | Table 4 | Average over 30 securities |
| Static VWAP TE benchmarks | Table 4 | Average over 30 securities; RM has no dynamic variant |
| Contaminated data results | Section 3.3, Table 1 | SPY, DIA, IBM with small/medium/large outliers |
| Synthetic convergence experiments | Section 2.3.3, Figure 4 | Multiple initial conditions |
