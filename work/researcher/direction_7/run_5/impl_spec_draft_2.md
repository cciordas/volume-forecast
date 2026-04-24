# Implementation Specification: Kalman Filter State-Space Model for Intraday Volume

## Overview

This specification describes a linear Gaussian state-space model for forecasting
intraday trading volume, based on Chen, Feng, and Palomar (2016). The model
decomposes log-volume into three additive components -- a daily average level
(eta), an intraday seasonal pattern (phi), and an intraday dynamic residual (mu)
-- and estimates them using Kalman filtering, RTS smoothing, and the EM algorithm.
A robust variant adds Lasso-penalized sparse noise detection for automatic outlier
handling. The model produces both static (next-day) and dynamic (one-bin-ahead)
volume forecasts, which feed into VWAP execution strategies.

The state dimension is 2 (eta, mu), and the observation is scalar. All Kalman
operations reduce to 2x2 matrix algebra and scalar divisions, making the model
computationally lightweight and suitable for real-time intraday use.

## Algorithm

### Model Description

The model operates on intraday volume data divided into fixed-length bins
(typically 15 minutes). For a trading day with I bins and a training window of
T days, the total number of bins is N = T * I.

**Input:** A matrix of raw intraday volumes, volume[t, i] for day t = 1..T and
bin i = 1..I, plus shares outstanding per day, shares_out[t].

**Output:** Forecasts of volume for each bin of the next trading day (static mode)
or one-bin-ahead forecasts updated after each observation (dynamic mode).

**Assumptions:**
- All trading days have exactly I bins (half-day sessions excluded).
- Volume is strictly positive in all bins (zero-volume bins treated as missing).
- Log-volume residuals are approximately Gaussian.
- The daily level eta is constant within a day, AR(1) across days.
- The intraday dynamic mu is AR(1) at every bin transition.
- Observation noise variance r is constant across all bins.
- The seasonal pattern phi is static (same across all training days).

**Decomposition** (Paper, Section 2, Equation 3):

    y_{t,i} = eta_t + phi_i + mu_{t,i} + v_{t,i}

where y_{t,i} = ln(volume[t,i] / shares_out[t]) is the log turnover, eta_t is
the daily average component, phi_i is the intraday periodic component, mu_{t,i}
is the intraday dynamic component, and v_{t,i} ~ N(0, r) is observation noise.

**Identifiability of phi and eta:** The decomposition y = eta + phi + mu + v has
an identifiability degree of freedom: shifting phi by a constant c and adjusting
eta by -c produces identical observations. This is resolved implicitly by the EM
algorithm: eta captures the day-level mean via its AR(1) dynamics and initial
state pi_1, while phi captures the per-bin residual pattern after removing the
state contribution C @ x_smooth. No explicit constraint (e.g., sum(phi) = 0) is
needed or desirable, as it would conflict with the closed-form M-step update for
phi (Paper, Eq A.39 / Eq 24). Researcher inference -- the paper does not discuss
this identifiability issue explicitly, but the EM structure resolves it.

### Pseudocode

#### Step 0: Data Preprocessing

```
INPUT: volume[t, i] for t=1..T, i=1..I; shares_out[t] for t=1..T
OUTPUT: y[1..N], observed[1..N], phi_position[1..N]

for t = 1 to T:
    for i = 1 to I:
        tau = (t - 1) * I + i
        phi_position[tau] = i                    # bin position for seasonality lookup
        if volume[t, i] > 0:
            y[tau] = ln(volume[t, i] / shares_out[t])
            observed[tau] = true
        else:
            y[tau] = 0.0                         # placeholder, not used
            observed[tau] = false
```

**Source:** Paper, Section 4.1, Equation 1 (turnover normalization) and
Equation 3 (log transform). The global indexing convention tau = (t-1)*I + i
follows from the paper's notation tau = 1, 2, ..., N below Equation 4.

#### Step 1: EM Initialization

```
INPUT: y[1..N], observed[1..N], phi_position[1..N], I, T
OUTPUT: theta^(0) = {a_eta, a_mu, sigma_eta_sq, sigma_mu_sq, r, phi[1..I], pi_1, Sigma_1}

# Compute per-bin means across days
for i = 1 to I:
    bin_values = [y[(t-1)*I + i] for t = 1..T if observed[(t-1)*I + i]]
    phi[i] = mean(bin_values)

# Compute grand mean
y_bar = mean(y[tau] for tau = 1..N if observed[tau])

# Adjust phi to represent seasonal deviation
for i = 1 to I:
    phi[i] = phi[i] - y_bar

# Compute per-day means for eta initialization
for t = 1 to T:
    day_values = [y[(t-1)*I + i] for i = 1..I if observed[(t-1)*I + i]]
    eta_init[t] = mean(day_values)

# Compute residuals for mu initialization
for tau = 1 to N:
    if observed[tau]:
        t = (tau - 1) // I + 1
        i = phi_position[tau]
        mu_init[tau] = y[tau] - eta_init[t] - phi[i]

# AR coefficient initialization from sample autocorrelation
a_eta = clamp(sample_autocorrelation(eta_init[1..T]), 0.01, 0.99)
a_mu  = clamp(sample_autocorrelation(mu_init[observed]), 0.01, 0.99)

# Variance initialization
sigma_eta_sq = var(eta_init[1..T]) * (1 - a_eta^2)   # stationary AR(1) innovation variance
sigma_mu_sq  = var(mu_init[observed]) * (1 - a_mu^2)
r = var(y[observed]) * 0.5                             # observation noise as fraction of total

# Initial state
pi_1 = [eta_init[1], 0.0]^T                           # first day's mean, mu starts at 0
Sigma_1 = diag(var(eta_init), var(mu_init[observed]))  # diagonal initial covariance
```

**Source:** Paper, Section 2.3.3 notes that EM is robust to initial parameter
choice (Figure 4 shows convergence from diverse initializations). The specific
initialization heuristic above is Researcher inference -- the paper does not
prescribe a particular initialization strategy. The decomposition-based approach
(estimating each component from its corresponding data feature) provides more
informed starting values than arbitrary constants.

#### Step 2: Kalman Filter (E-step, Forward Pass)

```
INPUT: y[1..N], observed[1..N], phi_position[1..N], theta
OUTPUT: x_filt[1..N], Sigma_filt[1..N], x_pred[1..N], Sigma_pred[1..N],
        A_used[2..N], e[1..N], S[1..N], z_star[1..N]

# Note: A_used is indexed from 2 to N (NOT 1 to N). A_used[tau] stores the
# transition matrix that produced the prediction at time tau from state at tau-1.
# At tau = 1, the state is initialized directly from pi_1 and Sigma_1, with no
# transition matrix involved. The smoother's backward loop accesses A_used[tau+1]
# for tau = N-1 down to 1, so the minimum index accessed is A_used[2], which exists.

# z_star[1..N]: outlier terms for robust variant. Initialize all to 0.0.
# For standard (non-robust) mode, z_star remains all zeros throughout.
# For robust mode, z_star[tau] is updated during the correction step below.
for tau = 1 to N:
    z_star[tau] = 0.0

for tau = 1 to N:

    # --- Prediction ---
    if tau == 1:
        x_pred[1] = pi_1
        Sigma_pred[1] = Sigma_1
    else:
        # Determine transition matrix A and process noise Q
        # Day boundary: tau-1 is the last bin of the previous day
        #   i.e., phi_position[tau-1] == I, equivalently (tau-1) mod I == 0
        if (tau - 1) mod I == 0:
            # Day boundary transition
            A = [[a_eta, 0],
                 [0,     a_mu]]
            Q = [[sigma_eta_sq, 0],
                 [0,            sigma_mu_sq]]
        else:
            # Within-day transition: eta held constant
            A = [[1,    0],
                 [0, a_mu]]
            Q = [[0,           0],
                 [0, sigma_mu_sq]]

        A_used[tau] = A
        x_pred[tau] = A @ x_filt[tau - 1]
        Sigma_pred[tau] = A @ Sigma_filt[tau - 1] @ A^T + Q

    # --- Correction ---
    i = phi_position[tau]
    C = [1, 1]                                    # 1x2 row vector

    if observed[tau]:
        S[tau] = C @ Sigma_pred[tau] @ C^T + r    # scalar: sum of all 4 Sigma elements + r
        K = Sigma_pred[tau] @ C^T / S[tau]         # 2x1 Kalman gain (scalar division)
        e[tau] = y[tau] - C @ x_pred[tau] - phi[i] # scalar innovation

        # Robust outlier detection (if lambda is finite)
        # For standard mode (lambda = infinity), threshold is infinite,
        # so z_star remains 0 and e_clean = e.
        threshold = lambda * S[tau] / 2
        if e[tau] > threshold:
            z_star[tau] = e[tau] - threshold
        elif e[tau] < -threshold:
            z_star[tau] = e[tau] + threshold
        else:
            z_star[tau] = 0.0

        e_clean = e[tau] - z_star[tau]

        x_filt[tau] = x_pred[tau] + K * e_clean

        # Joseph form for numerical stability
        I_2 = [[1, 0], [0, 1]]
        temp = I_2 - K @ C                        # 2x2, where K is 2x1 and C is 1x2
        Sigma_filt[tau] = temp @ Sigma_pred[tau] @ temp^T + (K * r) @ K^T
    else:
        # Missing observation: skip correction
        x_filt[tau] = x_pred[tau]
        Sigma_filt[tau] = Sigma_pred[tau]
        K = [0, 0]^T
        e[tau] = 0.0
        S[tau] = 0.0                              # sentinel: not used in log-likelihood
        z_star[tau] = 0.0                         # no observation, no outlier
```

**Source:** Paper, Algorithm 1 (page 4). Prediction: lines 2-3 (Equations 7-8
in the paper's notation). Correction: lines 4-6 (Kalman gain, state update,
covariance update). The time-varying A and Q matrices follow from the model
definition on page 3, specifically the piecewise definition of a_eta_tau and
(sigma_eta_tau)^2. The Joseph form covariance update is Researcher inference
(standard Kalman filter practice for numerical stability; the paper uses the
simpler form Sigma_{tau+1|tau+1} = Sigma_{tau+1|tau} - K * C * Sigma_{tau+1|tau}
in Algorithm 1 line 6, which is algebraically equivalent). Missing observation
handling follows from Section 4.1 (zero-volume bins excluded from the model).

The robust soft-thresholding follows from Paper, Section 3.1, Equation 33. The
threshold lambda / (2 * W_{tau+1}) equals lambda * S[tau] / 2 because
W = 1/S (Eq 30 defines W as the inverse of the predictive variance scalar).

**Note on indexing convention:** The paper's Algorithm 1 uses the notation where
the prediction step at time tau produces x_{tau+1|tau} and Sigma_{tau+1|tau}.
This pseudocode reindexes so that x_pred[tau] is the prediction *for* time tau
(produced from the state at tau-1), which is more natural for implementation.
The correspondence is: this code's x_pred[tau] = paper's x_{tau|tau-1}.

#### Step 3: RTS Smoother (E-step, Backward Pass)

```
INPUT: x_filt[1..N], Sigma_filt[1..N], x_pred[1..N], Sigma_pred[1..N], A_used[2..N]
OUTPUT: x_smooth[1..N], Sigma_smooth[1..N], Sigma_cross[2..N], L_stored[1..N-1]

# Initialize at the last time step
x_smooth[N] = x_filt[N]
Sigma_smooth[N] = Sigma_filt[N]

# Allocate smoother gain storage
# L_stored[tau] for tau = 1..N-1 stores the gain computed at position tau
# in the backward pass (used later for cross-covariance computation)
L_stored = array of (N-1) 2x2 matrices

# Backward recursion
for tau = N-1 downto 1:
    # Smoother gain L (2x2)
    # Need inverse of Sigma_pred[tau+1] (2x2)
    # Use analytic formula: inv([[a,b],[c,d]]) = (1/(ad-bc)) * [[d,-b],[-c,a]]
    det = Sigma_pred[tau+1][0,0] * Sigma_pred[tau+1][1,1]
        - Sigma_pred[tau+1][0,1] * Sigma_pred[tau+1][1,0]
    if abs(det) < 1e-12:
        # Regularize: add epsilon to diagonal
        Sigma_pred_reg = Sigma_pred[tau+1] + 1e-10 * I_2
        det = Sigma_pred_reg[0,0] * Sigma_pred_reg[1,1]
            - Sigma_pred_reg[0,1] * Sigma_pred_reg[1,0]
        Sigma_pred_inv = (1/det) * [[Sigma_pred_reg[1,1], -Sigma_pred_reg[0,1]],
                                     [-Sigma_pred_reg[1,0], Sigma_pred_reg[0,0]]]
    else:
        Sigma_pred_inv = (1/det) * [[Sigma_pred[tau+1][1,1], -Sigma_pred[tau+1][0,1]],
                                     [-Sigma_pred[tau+1][1,0], Sigma_pred[tau+1][0,0]]]

    L = Sigma_filt[tau] @ A_used[tau+1]^T @ Sigma_pred_inv
    L_stored[tau] = L                             # store for cross-covariance computation

    x_smooth[tau] = x_filt[tau] + L @ (x_smooth[tau+1] - x_pred[tau+1])
    Sigma_smooth[tau] = Sigma_filt[tau] + L @ (Sigma_smooth[tau+1] - Sigma_pred[tau+1]) @ L^T

# Cross-covariance for M-step (non-recursive formula)
for tau = 2 to N:
    Sigma_cross[tau] = Sigma_smooth[tau] @ L_stored[tau-1]^T
```

**Source:** Paper, Algorithm 2 (page 5), Equations 10-11. The smoother gain L_tau
corresponds to the paper's L_tau = Sigma_{tau|tau} * A_tau^T * Sigma_{tau+1|tau}^{-1}.
The backward recursion for x and Sigma is standard RTS.

**Cross-covariance derivation:** The paper provides a recursive form in Equations
A.20-A.21 (Appendix A). This spec uses the equivalent non-recursive formula:

    Sigma_cross[tau] = Cov(x_tau, x_{tau-1} | all data) = Sigma_smooth[tau] @ L[tau-1]^T

**Proof sketch:** By definition, the RTS smoother gives:
  x_smooth[tau] = x_filt[tau] + L[tau] @ (x_smooth[tau+1] - x_pred[tau+1])

The cross-covariance between smoothed states at consecutive times can be derived
by substituting the smoother update and applying the identity:
  Cov(x_tau, x_{tau-1}) = Cov(L[tau-1] @ x_smooth[tau], x_{tau-1})
                         = L[tau-1] @ Cov(x_smooth[tau], x_{tau-1})

After working through the algebra (substituting the RTS update for x_smooth[tau]
and noting that the filtering error x_filt[tau] - x_true[tau] is uncorrelated
with the smoother correction term), we obtain:

  Sigma_cross[tau] = Sigma_smooth[tau] @ L[tau-1]^T

This result appears in Shumway and Stoffer, Time Series Analysis and Its
Applications, 4th edition, Chapter 6, Property 6.3 (the exact numbering varies
by edition; look for the "lag-one covariance smoother" result).

**Alternative implementation:** If the developer prefers the paper's recursive
form (Eq A.20-A.21), note that it requires storing the Kalman gain K[N] from the
forward pass for initialization (Eq A.21 sets the starting value using
K_N = Sigma_{N|N-1} @ A_N^T @ Sigma_{N|N}^{-1}, which is exactly the Kalman
gain at the last step). The non-recursive form avoids this dependency.
Researcher inference on choice of non-recursive form.

#### Step 4: Sufficient Statistics (E-step Completion)

```
INPUT: x_smooth[1..N], Sigma_smooth[1..N], Sigma_cross[2..N]
OUTPUT: P[1..N], P_cross[2..N]

for tau = 1 to N:
    # P[tau] = E[x_tau * x_tau^T | all data]
    P[tau] = Sigma_smooth[tau] + x_smooth[tau] @ x_smooth[tau]^T    # 2x2

for tau = 2 to N:
    # P_cross[tau] = E[x_tau * x_{tau-1}^T | all data]
    P_cross[tau] = Sigma_cross[tau] + x_smooth[tau] @ x_smooth[tau-1]^T   # 2x2
```

**Source:** Paper, Appendix A, Equations A.18-A.19 (P_tau), A.22 (P_{tau,tau-1}).
The notation P^{(k,l)} used in the M-step equations refers to the (k,l) element
of these 2x2 matrices: P^{(1,1)} is the eta-eta element, P^{(2,2)} is the
mu-mu element, P^{(1,2)} = P^{(2,1)} is the cross-element.

#### Step 5: M-step (Parameter Updates)

The M-step updates must be computed in a specific order because some updates
depend on others from the same iteration. The required order is:

1. pi_1, Sigma_1 (independent)
2. a_eta, a_mu (independent of each other)
3. sigma_eta_sq, sigma_mu_sq (depend on a_eta, a_mu respectively)
4. phi (depends on x_smooth only)
5. r (depends on phi from this iteration)

**Missing data principle for denominators:** The E-step provides smoothed state
estimates x_smooth[tau] and Sigma_smooth[tau] for ALL bins, including missing
ones, because the Kalman filter's prediction step fills in unobserved positions.
Therefore, state-dynamics parameters (a_eta, a_mu, sigma_eta_sq, sigma_mu_sq)
sum over all bins -- their sufficient statistics P[tau] and P_cross[tau] are
defined for every tau regardless of whether y[tau] was observed. In contrast,
observation-equation parameters (phi, r) involve the actual observation y[tau],
which is undefined for missing bins. These parameters sum only over observed
bins, using N_obs (or per-bin count) as the denominator instead of N (or T).
Researcher inference on the explanation; the formulas follow from standard
EM-with-missing-data theory and are consistent with the paper's Appendix A
which assumes complete data.

```
INPUT: x_smooth[1..N], Sigma_smooth[1..N], P[1..N], P_cross[2..N],
       y[1..N], observed[1..N], phi_position[1..N], I, T,
       z_star[1..N]           # all zeros for standard variant
OUTPUT: theta^(j+1)

# --- Initial state (Paper, Eq A.32-A.33) ---
pi_1 = x_smooth[1]
Sigma_1 = Sigma_smooth[1]
    # Equivalent to P[1] - x_smooth[1] @ x_smooth[1]^T (Eq A.33)
    # but avoids subtracting two similar PSD matrices (numerically stabler)

# --- AR coefficients (Paper, Eq A.34-A.35) ---
# Define day-boundary set: D_start = {kI + 1 for k = 1..T-1}
# These are the first bins of days 2, 3, ..., T
# At these indices, the transition from tau-1 to tau used a_eta.

# a_eta: sum over day-boundary transitions
num_eta = 0.0
den_eta = 0.0
for k = 1 to T-1:
    tau = k * I + 1                              # first bin of day k+1
    num_eta += P_cross[tau][0, 0]                # P^{(1,1)}_{tau, tau-1}
    den_eta += P[tau - 1][0, 0]                  # P^{(1,1)}_{tau-1}
a_eta = num_eta / den_eta                        # Eq A.34

# a_mu: sum over all consecutive transitions
num_mu = 0.0
den_mu = 0.0
for tau = 2 to N:
    num_mu += P_cross[tau][1, 1]                 # P^{(2,2)}_{tau, tau-1}
    den_mu += P[tau - 1][1, 1]                   # P^{(2,2)}_{tau-1}
a_mu = num_mu / den_mu                           # Eq A.35

# --- Process noise variances (Paper, Eq A.36-A.37) ---
# sigma_eta_sq: over day-boundary transitions, using a_eta^(j+1)
sigma_eta_sq = 0.0
for k = 1 to T-1:
    tau = k * I + 1
    sigma_eta_sq += P[tau][0, 0]
                  + a_eta^2 * P[tau - 1][0, 0]
                  - 2 * a_eta * P_cross[tau][0, 0]
sigma_eta_sq /= (T - 1)                          # Eq A.36, denominator is |D| = T-1

# sigma_mu_sq: over all consecutive transitions, using a_mu^(j+1)
sigma_mu_sq = 0.0
for tau = 2 to N:
    sigma_mu_sq += P[tau][1, 1]
                 + a_mu^2 * P[tau - 1][1, 1]
                 - 2 * a_mu * P_cross[tau][1, 1]
sigma_mu_sq /= (N - 1)                           # Eq A.37, denominator is N-1

# --- Seasonality (Paper, Eq A.39 / Eq 24; robust: Eq 36) ---
C = [1, 1]
for i = 1 to I:
    phi[i] = 0.0
    count = 0
    for t = 1 to T:
        tau = (t - 1) * I + i
        if observed[tau]:
            phi[i] += y[tau] - C @ x_smooth[tau] - z_star[tau]
            count += 1
    phi[i] /= count                              # count = observed days for bin i

# --- Observation noise (Paper, Eq A.38; robust: Eq 35) ---
# IMPORTANT: uses phi^(j+1) computed above
r = 0.0
N_obs = 0
for tau = 1 to N:
    if observed[tau]:
        i = phi_position[tau]
        residual = y[tau] - phi[i] - z_star[tau] - C @ x_smooth[tau]
        r += residual^2 + C @ Sigma_smooth[tau] @ C^T
        N_obs += 1
r /= N_obs                                       # N_obs = total observed bins

# --- Parameter clamping (Researcher inference) ---
EPSILON = 1e-8
a_eta = clamp(a_eta, EPSILON, 1.0 - EPSILON)
a_mu  = clamp(a_mu, EPSILON, 1.0 - EPSILON)
sigma_eta_sq = max(sigma_eta_sq, EPSILON)
sigma_mu_sq  = max(sigma_mu_sq, EPSILON)
r = max(r, EPSILON)
```

**Source:** Paper, Appendix A.3, Equations A.32-A.39. The observation noise formula
(Eq A.38) in the paper is written in fully expanded form using P_tau notation.
The compact form used here -- r = (1/N_obs) * sum[(y - phi - z_star - C @ x_smooth)^2 +
C @ Sigma_smooth @ C^T] -- is equivalent. This can be verified by substituting
P_tau = Sigma_smooth[tau] + x_smooth[tau] @ x_smooth[tau]^T (Eq A.19) into
Eq A.38 and collecting terms. The Sigma_1 = Sigma_smooth[1] form (instead of
Eq A.33's P_1 - x_hat_1 @ x_hat_1^T) avoids catastrophic cancellation --
Researcher inference.

The phi-before-r ordering is required because Eq A.38 uses phi^{(j+1)}.
The paper's Eq A.38 explicitly shows phi_tau^{(j+1)} terms. Parameter clamping
is Researcher inference (standard practice to prevent degenerate parameters
during EM iteration; the paper does not discuss it).

**Unified M-step:** The formulas above handle both standard and robust variants.
For the standard variant, z_star[tau] = 0 for all tau, so the z_star terms
vanish and the formulas reduce to Eqs A.38-A.39. For the robust variant,
z_star[tau] is subtracted from the observation residual per Eqs 35-36 (Paper,
Section 3.2). All other M-step equations (a_eta, a_mu, sigma_eta_sq,
sigma_mu_sq, pi_1, Sigma_1) are unchanged because z_star affects only the
observation equation, not the state dynamics. The smoother (Step 3) operates
on the filtered states which already incorporate the outlier cleaning, so it
is also unchanged.

#### Step 6: Convergence Check

```
INPUT: e[1..N], S[1..N], observed[1..N] from current E-step
OUTPUT: log_likelihood (scalar), converged (boolean)

# Innovation-form log-likelihood
log_lik = 0.0
N_obs = 0
for tau = 1 to N:
    if observed[tau]:
        log_lik += -0.5 * (ln(S[tau]) + e[tau]^2 / S[tau])
        N_obs += 1
log_lik -= (N_obs / 2) * ln(2 * pi)

# Convergence test
if iteration > 1:
    relative_change = abs(log_lik - log_lik_prev) / max(abs(log_lik_prev), 1.0)
    converged = (relative_change < tol)
else:
    converged = false

log_lik_prev = log_lik
```

**Source:** The innovation-form log-likelihood is standard Kalman filter theory
(not explicitly written in the paper). The paper's Appendix A provides the
Q-function (Eq A.10) which is the theoretically correct EM convergence criterion.
In practice, the innovation-form log-likelihood is simpler to compute and serves
as an adequate convergence monitor. Researcher inference: using
max(abs(log_lik_prev), 1.0) in the denominator prevents division by zero
when log-likelihood is near zero.

**Correctness check:** The log-likelihood (or Q-function) should be monotonically
non-decreasing across EM iterations. A decrease indicates an implementation bug.

#### Step 7: EM Main Loop

```
INPUT: y[1..N], observed[1..N], phi_position[1..N], I, T, max_iter, tol, lambda
OUTPUT: theta_final, x_filt[1..N], Sigma_filt[1..N]

theta = initialize_parameters(y, observed, phi_position, I, T)    # Step 1
log_lik_prev = -infinity

for j = 1 to max_iter:
    # E-step
    x_filt, Sigma_filt, x_pred, Sigma_pred, A_used, e, S, z_star =
        kalman_filter(y, observed, phi_position, theta, lambda)   # Step 2
    x_smooth, Sigma_smooth, Sigma_cross, L_stored =
        rts_smoother(x_filt, Sigma_filt, x_pred, Sigma_pred, A_used)  # Step 3
    P, P_cross =
        sufficient_statistics(x_smooth, Sigma_smooth, Sigma_cross)     # Step 4

    # Convergence check (using innovations from E-step)
    log_lik, converged = check_convergence(e, S, observed, log_lik_prev, tol, j)
    log_lik_prev = log_lik

    if converged:
        break

    # M-step
    theta = m_step(x_smooth, Sigma_smooth, P, P_cross,
                   y, observed, phi_position, I, T, z_star)       # Step 5

theta_final = theta
# Note on final state: When the loop exits due to convergence at iteration j,
# theta_final holds the M-step parameters from iteration j-1, and
# x_filt[N], Sigma_filt[N] are from the E-step of iteration j (run with those
# same parameters). These are consistent: the filtered states were produced by
# theta_final. Skipping the final M-step is standard EM practice -- the
# parameters that produced a converged E-step are the final parameters.
# The filtered state at the last bin (x_filt[N], Sigma_filt[N]) is the starting
# point for prediction (Steps 10-12).
```

**Source:** Paper, Algorithm 3 (page 6). The E-step/M-step alternation is
described in Section 2.3.2. Figure 4 demonstrates convergence within a few
iterations from diverse initializations.

#### Step 8: Static Prediction

```
INPUT: theta, x_filt[N] (filtered state at end of last training day),
       Sigma_filt[N], I
OUTPUT: y_hat_static[1..I], volume_hat_static[1..I]

# At the end of day T (tau = N = T*I), produce forecasts for day T+1.
# First bin of day T+1 requires a day-boundary transition.

# Day boundary transition (tau = N to tau = N+1)
x_next = [[a_eta, 0], [0, a_mu]] @ x_filt[N]
Sigma_next = [[a_eta, 0], [0, a_mu]] @ Sigma_filt[N] @ [[a_eta, 0], [0, a_mu]]^T
            + [[sigma_eta_sq, 0], [0, sigma_mu_sq]]

# First bin forecast
y_hat_static[1] = C @ x_next + phi[1]
x_curr = x_next
Sigma_curr = Sigma_next

# Remaining bins (within-day transitions: eta constant)
for h = 2 to I:
    A_within = [[1, 0], [0, a_mu]]
    Q_within = [[0, 0], [0, sigma_mu_sq]]
    x_curr = A_within @ x_curr
    Sigma_curr = A_within @ Sigma_curr @ A_within^T + Q_within
    y_hat_static[h] = C @ x_curr + phi[h]

# Convert to volume space
# IMPORTANT: exp(y_hat) gives predicted TURNOVER (volume / shares_out), NOT raw
# volume. This is because y = ln(volume / shares_out) in preprocessing.
# For VWAP weights, this suffices -- shares_out cancels in the ratio
#   w[i] = turnover_hat[i] / sum(turnover_hat) = volume_hat[i] / sum(volume_hat).
# For MAPE evaluation (Paper, Eq 37), shares_out also cancels in the ratio.
# For raw volume forecasts (order sizing, market impact, reporting):
#   raw_volume_hat[h] = exp(y_hat_static[h]) * shares_out[T+1]
for h = 1 to I:
    volume_hat_static[h] = exp(y_hat_static[h])
```

**Source:** Paper, Section 2.2, Equation 9. The multi-step-ahead prediction
formula x_{tau+h|tau} = A_{tau+h-1} * ... * A_tau * x_{tau|tau} simplifies
because within a day, eta is constant (A[0,0] = 1, Q[0,0] = 0). Thus
eta_hat = a_eta * x_filt[N][0] for all bins, while mu_hat[h] decays as
(a_mu)^h * x_filt[N][1]. The code above propagates both components through
the transition matrices explicitly.

**Note on log-normal bias:** exp(E[log(V)]) gives the geometric mean, not the
arithmetic mean. An unbiased point estimate would be
exp(y_hat + 0.5 * (C @ Sigma_curr @ C^T + r)). However, the paper evaluates
MAPE using exp(y_hat) directly (the bias correction is not discussed). For
reproducing paper benchmarks, use exp(y_hat) without correction. For production
use, the bias correction may improve forecast accuracy. Researcher inference.

#### Step 9: Dynamic Prediction (One-Bin-Ahead)

```
INPUT: theta, x_filt_prev (filtered state at bin tau-1),
       Sigma_filt_prev, phi_position of current bin, y_observed (if available),
       lambda
OUTPUT: y_hat, volume_hat, x_filt_updated, Sigma_filt_updated

# Predict
if is_day_boundary(tau - 1):
    A = [[a_eta, 0], [0, a_mu]]
    Q = [[sigma_eta_sq, 0], [0, sigma_mu_sq]]
else:
    A = [[1, 0], [0, a_mu]]
    Q = [[0, 0], [0, sigma_mu_sq]]

x_pred = A @ x_filt_prev
Sigma_pred = A @ Sigma_filt_prev @ A^T + Q

i = phi_position of current bin
y_hat = C @ x_pred + phi[i]
volume_hat = exp(y_hat)

# Correct (if observation available -- dynamic mode)
if y_observed is not None:
    S = C @ Sigma_pred @ C^T + r
    K = Sigma_pred @ C^T / S
    e = y_observed - C @ x_pred - phi[i]

    # Robust correction (if lambda is finite)
    threshold = lambda * S / 2
    z_star = sign(e) * max(abs(e) - threshold, 0)
    e_clean = e - z_star

    x_filt_updated = x_pred + K * e_clean
    temp = I_2 - K @ C
    Sigma_filt_updated = temp @ Sigma_pred @ temp^T + (K * r) @ K^T
else:
    x_filt_updated = x_pred
    Sigma_filt_updated = Sigma_pred
```

**Source:** Paper, Section 2.2 (dynamic prediction) and Algorithm 1. This is
the same Kalman filter loop but applied one step at a time during live trading.

#### Step 10: Multi-Step-Ahead Prediction from Mid-Day

```
INPUT: theta, x_filt_current (filtered state at bin i after correction),
       Sigma_filt_current, current_bin (i), I
OUTPUT: y_hat_remaining[i+1..I], volume_hat_remaining[i+1..I]

# After correcting through bin i during dynamic mode, forecast all remaining
# bins i+1 through I. This is needed at each step of the dynamic VWAP strategy
# to compute vol_remaining.

# The logic is identical to Step 8 (static prediction) but:
#   - Starting from x_filt_current at bin i instead of x_filt[N]
#   - Using within-day transitions only (no day boundary can occur mid-day)

x_curr = x_filt_current
Sigma_curr = Sigma_filt_current

for h = i+1 to I:
    A_within = [[1, 0], [0, a_mu]]
    Q_within = [[0, 0], [0, sigma_mu_sq]]
    x_curr = A_within @ x_curr
    Sigma_curr = A_within @ Sigma_curr @ A_within^T + Q_within
    y_hat_remaining[h] = C @ x_curr + phi[h]
    volume_hat_remaining[h] = exp(y_hat_remaining[h])
```

**Source:** Paper, Section 2.2, Equation 9. The within-day multi-step-ahead
formula uses only within-day transitions (A = [[1,0],[0,a_mu]],
Q = [[0,0],[0,sigma_mu_sq]]) because eta is held constant within a trading day.
Researcher inference on extracting this as a separate step; the paper describes
it implicitly in the dynamic VWAP procedure (Section 4.3).

#### Step 11: VWAP Strategies

**Static VWAP** (Paper, Equation 40):

```
INPUT: volume_hat_static[1..I]
OUTPUT: w_static[1..I]

total = sum(volume_hat_static[1..I])
for i = 1 to I:
    w_static[i] = volume_hat_static[i] / total
```

**Dynamic VWAP** (Paper, Equation 41):

```
INPUT: theta, x_filt[N], Sigma_filt[N], I, total order quantity Q_total,
       live volume observations y_live[1..I]
OUTPUT: shares executed per bin

cumulative_executed = 0.0

# Day boundary: transition from end of training to first bin of new day
x_filt_current = predict through day boundary (Step 9 with is_day_boundary=true)

for i = 1 to I - 1:
    # Forecast current bin
    y_hat_i, volume_hat_i, x_filt_updated, Sigma_filt_updated =
        dynamic_predict(theta, x_filt_current, Sigma_filt_current,
                        phi_position=i, y_observed=y_live[i], lambda)  # Step 9

    # Forecast remaining bins i+1..I from updated state
    volume_hat_remaining[i+1..I] =
        multi_step_ahead(theta, x_filt_updated, Sigma_filt_updated, i, I)  # Step 10

    # Total forecasted volume for bins i..I
    vol_remaining = volume_hat_i + sum(volume_hat_remaining[i+1..I])

    # Weight for bin i: proportion of remaining to execute
    w[i] = volume_hat_i / vol_remaining

    # Execute
    remaining_shares = Q_total - cumulative_executed
    execute[i] = w[i] * remaining_shares
    cumulative_executed += execute[i]

    # Update state for next iteration
    x_filt_current = x_filt_updated
    Sigma_filt_current = Sigma_filt_updated

# Last bin: execute whatever remains
execute[I] = Q_total - cumulative_executed
```

**Source:** Paper, Section 4.3, Equations 39-42. The static weights (Eq 40) are
fixed before market open. The dynamic weights (Eq 41) adapt at each bin using
one-bin-ahead forecasts, redistributing the remaining order proportionally
among remaining bins.

**Backtesting vs production mode** (Researcher inference): In backtesting, the
full day of dynamic one-step-ahead forecasts can be collected by running the
Kalman filter through the entire day. In production, at each bin the filter
is corrected through the previous bin and multi-step-ahead forecasts are
generated for all remaining bins.

### Data Flow

```
Raw volume[T, I] + shares_out[T]
    |
    v
[Preprocessing: normalize, log-transform]
    |
    v
y[N], observed[N], phi_position[N]     (N = T * I)
    |
    v
[EM Initialization] --> theta^(0)
    |
    v
+--[EM Loop]-----------------------------------------------+
|                                                           |
|  [Kalman Filter: forward pass]                            |
|      y[N], theta, lambda --> x_filt[N] (2x1 each),       |
|                      Sigma_filt[N] (2x2 each),            |
|                      x_pred[N], Sigma_pred[N],            |
|                      A_used[N-1] (indexed 2..N),          |
|                      e[N], S[N], z_star[N]                |
|      |                                                    |
|      v                                                    |
|  [RTS Smoother: backward pass]                            |
|      --> x_smooth[N] (2x1), Sigma_smooth[N] (2x2),       |
|          L_stored[N-1] (2x2), Sigma_cross[N-1] (2x2)     |
|      |                                                    |
|      v                                                    |
|  [Sufficient Statistics]                                  |
|      --> P[N] (2x2), P_cross[N-1] (2x2)                  |
|      |                                                    |
|      v                                                    |
|  [Convergence Check] --> log_lik (scalar)                 |
|      |                                                    |
|      v                                                    |
|  [M-step] --> theta^(j+1)                                 |
|      |                                                    |
+------+--- if not converged, loop back to Kalman Filter ---+
    |
    v (converged)
theta_final = {a_eta, a_mu, sigma_eta_sq, sigma_mu_sq, r,
               phi[I], pi_1 (2x1), Sigma_1 (2x2)}
    |
    +---> [Static Prediction] --> volume_hat[I] --> VWAP weights[I]
    |
    +---> [Dynamic Prediction] --> per-bin forecast + correction loop
    |         uses Step 9 (one-bin-ahead) + Step 10 (multi-step remaining)
```

**Types and shapes:**

| Quantity        | Type    | Shape     | Description                              |
|-----------------|---------|-----------|------------------------------------------|
| y               | float64 | (N,)     | Log-turnover observations                |
| observed        | bool    | (N,)     | Observation mask                         |
| phi_position    | int     | (N,)     | Bin index 1..I for each tau              |
| x_filt, x_pred  | float64 | (N, 2)   | Filtered/predicted state vectors         |
| x_smooth        | float64 | (N, 2)   | Smoothed state vectors                   |
| Sigma_filt, etc.| float64 | (N, 2, 2)| State covariance matrices                |
| A_used          | float64 | (N-1, 2, 2) | Transition matrices (indexed 2..N)    |
| L_stored        | float64 | (N-1, 2, 2) | Smoother gains (indexed 1..N-1)       |
| P               | float64 | (N, 2, 2)| Second moment E[x x^T | all data]       |
| P_cross         | float64 | (N-1, 2, 2) | Cross second moment (indexed 2..N)   |
| e               | float64 | (N,)     | Innovations                              |
| S               | float64 | (N,)     | Innovation variances                     |
| K               | float64 | (N, 2)   | Kalman gains                             |
| z_star          | float64 | (N,)     | Detected outliers (robust variant)       |
| phi             | float64 | (I,)     | Seasonal pattern                         |
| pi_1            | float64 | (2,)     | Initial state mean                       |
| Sigma_1         | float64 | (2, 2)   | Initial state covariance                 |
| a_eta, a_mu     | float64 | scalar   | AR coefficients                          |
| sigma_eta_sq, sigma_mu_sq | float64 | scalar | Process noise variances        |
| r               | float64 | scalar   | Observation noise variance               |

### Variants

This specification implements the **robust Kalman filter variant** as the primary
model, with the standard Kalman filter recovered by setting lambda = infinity
(or equivalently, a very large value like 1e10). This follows the paper's
recommendation (Section 3) and empirical evidence (Tables 1 and 3) showing that
the robust variant performs equal to or better than the standard variant in all
conditions, including clean data where the MAPE difference is at most 0.01.

The paper describes only one model architecture. There are no alternative
structural variants to choose from.

**Note on lambda = infinity:** Setting lambda = infinity (or a large value like
1e10) recovers the standard filter exactly (threshold becomes infinite, z_star is
always zero). Paper, Section 3 and Table 1 show that robust performs equal or
better even on clean data (MAPE difference <= 0.01).

## Parameters

| Parameter      | Description                                | Recommended Value     | Sensitivity | Range            |
|----------------|--------------------------------------------|-----------------------|-------------|------------------|
| a_eta          | Daily AR(1) coefficient                    | EM-estimated; ~0.98-0.99 converged | Low (EM-estimated) | (0, 1)    |
| a_mu           | Intraday AR(1) coefficient                 | EM-estimated; ~0.5-0.7 converged | Low (EM-estimated) | (0, 1)     |
| sigma_eta_sq   | Daily process noise variance               | EM-estimated          | Low         | (0, inf)         |
| sigma_mu_sq    | Intraday process noise variance            | EM-estimated          | Low         | (0, inf)         |
| r              | Observation noise variance                 | EM-estimated; ~0.05   | Low         | (0, inf)         |
| phi[1..I]      | Intraday seasonal pattern                  | EM-estimated          | N/A         | (-inf, inf)      |
| pi_1           | Initial state mean                         | EM-estimated          | Very low    | (-inf, inf) each |
| Sigma_1        | Initial state covariance                   | EM-estimated          | Very low    | PSD 2x2          |
| I              | Number of intraday bins                    | 26 (NYSE, 15-min)     | N/A (fixed) | 13-52 typical    |
| lambda         | Lasso regularization parameter             | CV-selected           | **High**    | (0, inf)         |
| N_days         | Training window length (days)              | CV-selected; 100-500  | Medium      | 60-500           |
| max_iter       | Maximum EM iterations                      | 100                   | Very low    | 50-200           |
| tol            | EM convergence tolerance (relative)        | 1e-8                  | Very low    | 1e-6 to 1e-10   |
| EPSILON        | Parameter clamping floor                   | 1e-8                  | Very low    | 1e-12 to 1e-6   |

**Source for converged parameter values:** Paper, Figure 4 (synthetic experiments
showing a_eta converging near 0.98-0.99, a_mu near 0.5-0.7, r near 0.05).
Lambda and N_days sensitivity: Paper, Section 4.1 (selected by cross-validation;
the paper does not report typical lambda values).

### Initialization

See Step 1 (EM Initialization) in the pseudocode above. Summary:

1. phi initialized from per-bin mean minus grand mean.
2. AR coefficients from sample autocorrelation of decomposed components.
3. Variance parameters from component-level sample variances, scaled by
   stationary AR(1) innovation variance formula.
4. pi_1 = [first day's mean level, 0]^T.
5. Sigma_1 = diagonal matrix of component variances.

For rolling-window re-estimation (Researcher inference), use the previous day's
calibrated parameters as the starting point for EM. This typically reduces
convergence to 2-5 iterations vs 10-20 from cold start, since consecutive
windows share most of their data.

### Calibration

The calibration procedure has two levels:

**Level 1: EM parameter estimation** (for fixed N_days and lambda)
- Run the EM loop (Step 7) on the training window of N_days * I bins.
- All model parameters (a_eta, a_mu, sigma_eta_sq, sigma_mu_sq, r, phi,
  pi_1, Sigma_1) are estimated.

**Level 2: Cross-validation for N_days and lambda** (Paper, Section 4.1)
1. Define a validation period (e.g., 100 trading days before the out-of-sample
   period).
2. Define candidate grids:
   - N_days_candidates = [60, 100, 150, 200, 300, 500]
   - lambda_candidates = log-spaced from 0.01 to 100 (e.g., 15-20 points),
     PLUS lambda = 1e10 (representing infinity / standard KF). Including the
     standard KF as a candidate allows the CV procedure to automatically select
     it if no outlier robustness is needed. Researcher inference on the grid
     range and the inclusion of the infinity candidate; the paper specifies
     cross-validation (Section 4.1) but does not report the lambda grid.
3. For each (N_days, lambda) pair:
   a. Train: run EM on the N_days training window ending before the
      validation period.
   b. Evaluate: compute dynamic MAPE on the validation period using the
      rolling window procedure (re-estimate daily, shift window by I bins).
4. Select the (N_days, lambda) pair that minimizes validation MAPE.
5. Use these for out-of-sample evaluation.

**Source:** Paper, Section 4.1 ("The optimal training data length N and the
optimal Lasso regularization parameter lambda are determined by the
cross-validation method. Specifically, data between January 2015 and May 2015
are considered as a cross-validation set.")

## Validation

### Expected Behavior

**MAPE formula** (Paper, Section 3.3, Equation 37):

    MAPE = (1/M) * sum_{tau=1}^{M} |volume_tau - predicted_volume_tau| / volume_tau

where M is the total number of bins in the out-of-sample period, volume_tau is
the actual volume (or turnover) in bin tau, and predicted_volume_tau is the
model's forecast. Since both actual and predicted are in turnover space
(volume / shares_out), shares_out cancels and the MAPE is the same whether
computed in turnover or raw volume.

**Volume prediction MAPE** (Paper, Table 3, averaged across 30 securities):

| Model                  | Dynamic MAPE | Static MAPE |
|------------------------|-------------|-------------|
| Robust Kalman Filter   | 0.46        | 0.61        |
| Standard Kalman Filter | 0.47        | 0.62        |
| CMEM (benchmark)       | 0.65        | 0.90        |
| Rolling Mean           | 1.28        | 1.28        |

**VWAP tracking error** (Paper, Table 4, basis points):

| Model                  | Dynamic TE  | Static TE   |
|------------------------|-------------|-------------|
| Robust Kalman Filter   | 6.38 bps    | 6.85 bps    |
| Standard Kalman Filter | 6.39 bps    | 6.89 bps    |
| CMEM                   | 7.01 bps    | 7.71 bps    |
| Rolling Mean           | 7.48 bps    | 7.48 bps    |

**Per-ticker range** (Paper, Table 3): Best MAPE is AAPL at 0.21 (dynamic).
Worst is 2800HK at 1.94 (leveraged ETF). Most U.S. large-cap equities: 0.21-0.42.

**Seasonal pattern phi:** Should exhibit a U-shape (high volume at open, lower
mid-day, higher at close), matching the well-documented intraday volume pattern.
Paper, Figure 4f shows converged phi values.

**EM convergence:** Parameters should converge within 5-20 iterations from
reasonable initializations. Paper, Figure 4 and Section 2.3.3.

### Sanity Checks

1. **Synthetic parameter recovery:** Generate synthetic data with known
   parameters (a_eta=0.98, a_mu=0.5, sigma_eta_sq=0.01, sigma_mu_sq=0.05,
   r=0.05, I=26, T=500). Run EM and verify estimated parameters converge
   to within 5% of true values. Paper, Section 2.3.3 and Figure 4.

2. **Log-likelihood monotonicity:** The innovation-form log-likelihood (or
   Q-function) must be non-decreasing across EM iterations. Any decrease
   signals an implementation bug.

3. **Robust equals standard on clean data:** With lambda = 1e10 (infinity),
   the robust filter should produce identical results to the standard
   filter. With finite lambda on clean data, z_star should be zero or near-zero
   for the vast majority of bins.

4. **AR coefficients in (0, 1):** Both a_eta and a_mu must converge to values
   strictly between 0 and 1 (stationarity requirement). a_eta should be close
   to 1 (high persistence), a_mu moderate (0.5-0.7).

5. **Phi U-shape:** The estimated seasonal vector phi should show high values
   at bins 1-2 (market open) and bins I-1, I (market close), with lower values
   mid-day.

6. **Beat rolling mean:** Dynamic MAPE should be substantially lower than the
   rolling mean baseline (paper shows ~64% improvement).

7. **Multi-initialization convergence:** Run EM from 5 different random
   initializations. All should converge to the same parameter values (within
   numerical precision). Paper, Figure 4 demonstrates this.

8. **Contaminated data resilience:** Inject outliers (10x volume spikes) into
   10% of bins. The robust filter should degrade much less than the standard
   filter. Paper, Section 3.3 and Table 1.

9. **Prediction variance grows with horizon:** In static prediction, the
   predictive variance C @ Sigma_curr @ C^T + r should increase monotonically
   with forecast horizon h. This follows from the additive process noise at
   each step.

10. **VWAP weights sum to 1:** Both static and dynamic VWAP weights must sum
    to exactly 1.0 (by construction in the formulas).

### Edge Cases

1. **Zero-volume bins:** Must be flagged as missing (observed = false) during
   preprocessing. ln(0) is undefined. The Kalman filter skips the correction
   step for missing bins, carrying forward the predicted state. Paper,
   Section 4.1.

2. **Half-day sessions:** Days with fewer than I bins must be excluded entirely
   from the training data. The model assumes a fixed number of bins per day.
   Paper, Section 4.1 ("excluding half-day trading sessions").

3. **Lambda = 0:** Degenerate case where every innovation is treated as an
   outlier (z_star = e always, no state correction occurs). Lambda must be
   strictly positive. Researcher inference.

4. **Lambda at CV grid boundary:** If the optimal lambda is at the edge of the
   search grid, extend the grid. lambda -> 0 suggests severe data quality
   issues. lambda -> infinity means standard KF suffices. Researcher inference.

5. **Near-singular predicted covariance:** During the RTS smoother, inv(Sigma_pred)
   may be ill-conditioned. Apply epsilon regularization (add 1e-10 to diagonal)
   when the 2x2 determinant is below 1e-12. Researcher inference.

6. **Extreme log-volumes:** After log transformation, extreme values can affect
   EM convergence. Consider winsorizing at +/- 5 standard deviations.
   Researcher inference.

7. **Different exchanges:** The number of bins I varies by exchange (e.g., 26 for
   NYSE 6.5h/15min, different for Tokyo, Hong Kong, etc.). Model must be
   calibrated separately per I value. Paper, Table 2.

8. **Float64 precision:** Use double-precision floating point throughout. Small
   normalized volumes are amplified by the log transform, and long EM iteration
   chains accumulate rounding errors. Researcher inference.

### Known Limitations

1. **No zero-volume handling:** The log transform requires strictly positive
   volumes. Illiquid instruments with frequent zero-volume bins cannot be
   modeled without imputation or modification.

2. **No exogenous covariates:** The model uses only past volume. Volatility,
   spread, order flow, and other variables are not incorporated. Paper,
   Section 5 identifies this as future work.

3. **Single-security model:** Each security is modeled independently. No
   cross-sectional information is used.

4. **AR(1) dynamics only:** Both eta and mu follow AR(1) processes. Higher-order
   autoregressive structure is not captured.

5. **Static seasonality:** The seasonal pattern phi is estimated as a fixed vector
   over the training window. It does not adapt to regime changes (e.g., inverted
   patterns on gap days, earnings announcements). Researcher inference.

6. **Static prediction degrades for later bins:** In static mode, mu decays as
   (a_mu)^h. For a_mu = 0.5, by bin 13 (mid-day) the mu contribution is
   0.5^13 ~ 0.0001, effectively zero. Static prediction for later bins relies
   almost entirely on eta + phi. Paper, Section 2.2 (implicit in Equation 9).

7. **Fixed bin granularity:** The model assumes a fixed I. Adaptive or irregular
   time grids are not supported.

8. **Gaussian noise assumption:** Despite the log transform making normality more
   defensible (Paper, Figure 1), the true distribution still has heavier tails.
   The robust variant mitigates this partially.

## Paper References

| Spec Section                  | Paper Source                                         |
|-------------------------------|------------------------------------------------------|
| Model decomposition           | Section 2, Equation 3                                |
| Phi/eta identifiability       | Researcher inference (resolved by EM structure)      |
| State-space formulation       | Section 2, Equations 4-5, page 3                     |
| Time-varying A, Q             | Section 2, page 3 (piecewise definitions)            |
| Kalman filter algorithm       | Algorithm 1, page 4                                  |
| Multi-step prediction         | Section 2.2, Equation 9                              |
| RTS smoother                  | Algorithm 2, page 5                                  |
| Cross-covariance (non-recursive) | Shumway & Stoffer, Ch. 6, Property 6.3 + proof sketch above |
| Sufficient statistics         | Appendix A, Equations A.18-A.19, A.22                |
| EM algorithm                  | Algorithm 3, page 6                                  |
| M-step: pi_1                  | Appendix A, Equation A.32                            |
| M-step: Sigma_1               | Appendix A, Equation A.33                            |
| M-step: a_eta                 | Appendix A, Equation A.34                            |
| M-step: a_mu                  | Appendix A, Equation A.35                            |
| M-step: sigma_eta_sq          | Appendix A, Equation A.36                            |
| M-step: sigma_mu_sq           | Appendix A, Equation A.37                            |
| M-step: r                     | Appendix A, Equation A.38                            |
| M-step: phi                   | Appendix A, Equation A.39 / Equation 24              |
| Missing data denominators     | Researcher inference (standard EM-with-missing-data) |
| Q-function decomposition      | Appendix A, Equations A.10-A.14                      |
| Joint log-likelihood          | Appendix A, Equation A.8                             |
| Robust observation model      | Section 3.1, Equations 25-27                         |
| Robust Lasso formulation      | Section 3.1, Equations 29-30                         |
| Soft-thresholding solution    | Section 3.1, Equations 33-34                         |
| Robust M-step: r              | Section 3.2, Equation 35                             |
| Robust M-step: phi            | Section 3.2, Equation 36                             |
| MAPE definition               | Section 3.3, Equation 37                             |
| Static VWAP weights           | Section 4.3, Equation 40                             |
| Dynamic VWAP weights          | Section 4.3, Equation 41                             |
| VWAP tracking error           | Section 4.3, Equation 42                             |
| Cross-validation              | Section 4.1                                          |
| Lambda CV grid                | Researcher inference (grid range + infinity candidate)|
| Empirical MAPE results        | Table 3                                              |
| Contamination results         | Table 1                                              |
| VWAP tracking results         | Table 4                                              |
| EM convergence behavior       | Section 2.3.3, Figure 4                              |
| Ticker/exchange summary       | Table 2                                              |
| Joseph form covariance        | Researcher inference (standard KF practice)          |
| Sigma_1 = Sigma_smooth[1]     | Researcher inference (equivalent to Eq A.33)         |
| Parameter clamping            | Researcher inference                                 |
| EM warm-start in rolling window | Researcher inference                               |
| Log-normal bias correction    | Researcher inference                                 |
| Turnover vs raw volume        | Researcher inference (follows from preprocessing)    |
| Multi-step from mid-day       | Researcher inference (implicit in Section 4.3)       |
| Winsorization                 | Researcher inference                                 |
