# Critique of Implementation Specification Draft 1: Kalman Filter State-Space Model for Intraday Volume

**Direction:** 7, **Run:** 3
**Role:** Critic
**Date:** 2026-04-10

## Summary

The draft is well-structured, covers the full algorithmic pipeline, and provides
generally accurate citations. It would allow a knowledgeable developer to produce a
mostly-correct implementation. However, I found 3 major issues that would cause
incorrect numerical results if implemented as written, 4 medium issues that could
cause bugs or confusion, and 3 minor issues affecting clarity. The major issues all
relate to the EM M-step specification.

---

## Major Issues

### M1. Kalman filter pseudocode has a missing initial correction step

**Spec section:** Step 1, Kalman Filter pseudocode, lines starting at "Initialize"

The pseudocode initializes x_hat[1|0] = pi_1 and Sigma[1|0] = Sigma_1, then enters
a loop `For tau = 1, 2, ..., N-1` where the first operation is PREDICT:

    x_hat[tau+1|tau] = A[tau] * x_hat[tau|tau]

At tau=1, this requires x_hat[1|1] -- the *corrected* state at time 1. But only
x_hat[1|0] (the *predicted* state) has been defined. The correction of x_hat[1|0]
using observation y[1] to produce x_hat[1|1] is never performed.

The paper's Algorithm 1 (Section 2.2, page 4) has the same structural ambiguity --
it assumes x_hat[tau|tau] is available at the start of each iteration. The text
explains: "the Kalman filter uses the state estimate (x_hat[tau|tau], Sigma[tau|tau])
to predict..." but never shows the initial correction explicitly.

**Impact:** The first observation y[1] is effectively skipped, and the predict step
at tau=1 uses the prior mean pi_1 directly. For long training windows this is a small
numerical effect (the filter "catches up"), but for short windows or synthetic data
recovery tests it will produce measurably wrong results and confuse the developer
during validation.

**Fix:** Add an explicit initial correction step before the loop:

```
-- Initial correction (process first observation):
W_init = (C * Sigma[1|0] * C^T + r)^{-1}
K_init = Sigma[1|0] * C^T * W_init
e_init = y[1] - phi[1] - C * x_hat[1|0]
x_hat[1|1] = x_hat[1|0] + K_init * e_init
Sigma[1|1] = Sigma[1|0] - K_init * C * Sigma[1|0]
```

Then the loop runs from tau=1 to N-1 as written.

### M2. EM M-step uses stale parameter values instead of current-iteration updates

**Spec section:** Step 3, EM Algorithm, M-STEP block

The paper's M-step equations use the *just-updated* parameter values within the same
M-step. Specifically:

1. Equation A.36 for sigma_eta^2 uses `(a^eta)^{(j+1)}` (the updated a_eta from the
   same iteration). The spec writes `(a_eta^(j))^2`, using the *previous* iteration's
   value.

2. Equation A.37 for sigma_mu^2 uses `(a^mu)^{(j+1)}`. The spec writes `(a_mu^(j))^2`.

3. Equation A.38 for r uses `phi_tau^{(j+1)}` (the updated phi from the same
   iteration). The spec writes `(phi^(j))^2`, using the previous iteration's phi.

This is visible directly in the paper: Equations A.34-A.39 (pages 15) consistently
use the (j+1) superscript for parameters that are computed earlier in the same M-step.

**Impact:** Using stale values will cause the EM to converge to slightly different
parameter values or converge more slowly. In some cases, it may not converge at all
(the EM guarantee of monotonic likelihood increase depends on correctly maximizing the
Q function, which requires using the jointly optimal values).

**Fix:** Specify the M-step computation order explicitly:

```
1. Compute pi_1, Sigma_1 (independent of other parameters)
2. Compute a_eta  (Equation A.34)
3. Compute a_mu   (Equation A.35)
4. Compute phi    (Equation A.39)
5. Compute sigma_eta^2 using the just-computed a_eta^(j+1)  (Equation A.36)
6. Compute sigma_mu^2  using the just-computed a_mu^(j+1)   (Equation A.37)
7. Compute r using the just-computed phi^(j+1)               (Equation A.38)
```

Update all formulas to use the (j+1) superscript for parameters already computed in
the same M-step.

### M3. The r update formula in the spec is ambiguous about phi indexing

**Spec section:** Step 3, M-step, observation noise variance formula

The spec writes:

    r^(j) = (1/N) * sum_{tau=1}^{N} [ ... + (phi^(j))^2 - 2*y[tau]*phi^(j)
             + 2*phi^(j)*C*x_hat[tau] ]

The `phi^(j)` here is written as if it were a single scalar, but phi is a vector of I
elements, and the correct term is `phi[tau]` (i.e., `phi_{((tau-1) mod I) + 1}`) --
the seasonality value for the bin position corresponding to global index tau.

The paper's Equation A.38 writes `phi_tau^{(j+1)}`, making the tau-dependence explicit.

**Impact:** A developer could implement this as a single scalar phi rather than
looking up the correct bin-position-specific value. This would produce incorrect
parameter estimates and destroy the seasonal pattern.

**Fix:** Replace every instance of `phi^(j)` in the r formula with `phi[tau]^(j+1)`
(or equivalently `phi_{((tau-1) mod I) + 1}^(j+1)`), making the per-bin indexing
explicit.

---

## Medium Issues

### MD1. Log-likelihood formula omitted from EM convergence check

**Spec section:** Step 3, EM Algorithm, end of procedure

The spec says "Compute log-likelihood for convergence check (Equation A.8 in Paper)"
but does not include the formula. This forces the developer to read the paper's
Appendix A.1 to implement convergence checking.

The log-likelihood (Equation A.8) has multiple terms involving log(r), log(sigma_mu^2),
log(sigma_eta^2), log|Sigma_1|, and sums over the observations and states. It is
nontrivial and should be written out in full in the spec, consistent with the level of
detail given for all other formulas.

**Fix:** Include the full log-likelihood formula from Equation A.8 in the spec. Note
that in practice, since the EM guarantees monotonic increase, tracking the *change* in
log-likelihood is sufficient, and one can also monitor the parameter change norm as
an alternative convergence criterion.

### MD2. Smoother behavior under the robust model is not specified

**Spec section:** Step 4b, Robust EM Modifications

The spec says the robust variant modifies the correction step and shows modified
M-step updates for r and phi, then states "All other M-step equations remain
unchanged." But it does not address whether the Kalman smoother (Algorithm 2) needs
modification for the robust case.

The paper's Section 3.2 says "let z_1*...z_N* denote the solutions of problem (30)
calculated in the E-step." This implies the z* values are computed during the forward
filter pass (using the modified correction in Equation 32), and the smoother runs
unchanged on the filtered output -- which already incorporates outlier cleaning. The
smoother itself is not modified; only its inputs differ.

**Impact:** A developer might wonder whether the smoother needs modification, or
might incorrectly try to incorporate z* into the smoother equations.

**Fix:** Add an explicit statement: "The RTS smoother (Algorithm 2) is unchanged in
the robust variant. It operates on the filtered states produced by the robust Kalman
filter, which already incorporate outlier cleaning via the soft-thresholding step. The
z* values are only used in the forward filter correction and in the M-step updates
for r and phi."

### MD3. Missing observation handling not integrated into main pseudocode

**Spec section:** Step 0 (preprocessing) and Edge Cases item 1

The spec mentions excluding zero-volume bins in preprocessing and says in the edge
cases section that the Kalman correction step should be skipped for missing
observations. But this guidance is separated from the main Kalman filter pseudocode
(Step 1), where a developer would actually need it.

For the Kalman filter, handling missing observations is straightforward: skip the
correction step (lines 4-6 of Algorithm 1) and only run the prediction step. The
filtered state becomes x_hat[tau+1|tau+1] = x_hat[tau+1|tau] and
Sigma[tau+1|tau+1] = Sigma[tau+1|tau]. This should be shown as a conditional branch
in the main pseudocode.

**Fix:** Add a conditional to the Kalman filter loop:

```
if y[tau+1] is observed:
    -- CORRECT (standard correction step)
    ...
else:
    -- SKIP correction (missing observation)
    x_hat[tau+1|tau+1] = x_hat[tau+1|tau]
    Sigma[tau+1|tau+1] = Sigma[tau+1|tau]
```

### MD4. Day-boundary set D is confusingly labeled and potentially inconsistent

**Spec section:** Step 3, EM Algorithm, Definitions

The spec defines: "Let D = {tau : tau = k*I+1 for k=1,2,...} be the set of
day-boundary indices."

This is numerically correct for the M-step sums (matching Equations A.34, A.36 which
sum over tau=kI+1), but calling these "day-boundary indices" is confusing because:

1. In the Kalman filter section (Step 1), the day boundary is defined as tau = k*I
   (i.e., the *last* bin of a day), where A[tau] switches to using a_eta.
2. In the M-step, D = {kI+1} refers to the *first* bin of each new day -- the
   *destination* of the day-boundary transition.

A developer seeing "day-boundary" in both places could reasonably assume they refer
to the same set of indices.

**Fix:** Rename D to something like "day-start indices" or "first-bin-of-day indices"
and add a note clarifying the relationship: "These are the destination indices of day
transitions. The transition matrix A[tau] applies at tau = kI (last bin of day k),
producing the state at tau = kI+1 (first bin of day k+1). The M-step sums over the
destination indices."

---

## Minor Issues

### MN1. Dynamic VWAP does not clarify what "volume_hat_dynamic[i]" means for the current bin

**Spec section:** Step 6, Dynamic VWAP

The formula uses `volume_hat_dynamic[i]` for all bins i=1..I. For bin i, is this the
one-step-ahead forecast made *before* bin i is observed, or the actual observed volume?
The paper's Equation 41 uses `volume_{t,i}^(d)` which denotes the dynamic prediction.
At bin i, the weight uses the forecast (not the observation) for that bin. This is
subtle and should be stated explicitly.

### MN2. Robust EM: the r formula in Step 4b has the same phi-indexing issue as the standard r formula

**Spec section:** Step 4b, robust r update

The formula writes `(phi^(j))^2` and `2*z*[tau]*phi^(j)` where phi should be
phi[tau] (bin-position-specific). The same fix as M3 applies here. Additionally,
the z* terms should use the updated phi^(j+1) per the paper's Equation 35.

### MN3. The spec does not mention parameter constraints or clamping

**Spec section:** Parameters, EM M-step

The EM M-step can occasionally produce parameter values outside their valid range
due to numerical issues (e.g., negative variance estimates, or AR coefficients
outside (0,1)). The spec notes that a_eta and a_mu should be in (0,1) and variances
should be positive, but does not say what to do if the M-step produces invalid values.

Standard practice is to clamp: if sigma_eta^2 < epsilon, set it to epsilon; if
a_eta >= 1 or a_eta <= 0, clamp to (epsilon, 1-epsilon). This is Researcher inference
but important for robustness.

---

## Correctness Verification of Citations

I verified the following spec claims against the paper:

| Spec Claim | Paper Source | Verified? |
|-----------|-------------|-----------|
| Model decomposition y = eta + phi + mu + v | Section 2, Eq. 3 | Yes |
| A_tau definition with day-boundary switching | Section 2, after Eq. 5 | Yes |
| Q_tau definition | Section 2, after Eq. 5 | Yes |
| Kalman filter (Algorithm 1) | Section 2.2, page 4 | Yes (but see M1) |
| Smoother (Algorithm 2) | Section 2.3.1, page 5 | Yes |
| EM (Algorithm 3) | Section 2.3.2, page 6 | Yes (but see M2, M3) |
| M-step equations A.32-A.39 | Appendix A.3, pages 15 | Structure correct, iteration indices wrong (M2) |
| Cross-covariance init (A.21) | Appendix A, page 14 | Yes |
| Soft-thresholding (Eqs. 33-34) | Section 3.1, page 7 | Yes |
| Robust EM (Eqs. 35-36) | Section 3.2, page 7 | Yes (but see MN2) |
| MAPE results Table 3 averages | Section 4.2, page 10 | Yes -- average row matches |
| VWAP tracking Table 4 averages | Section 4.3, page 12 | Yes -- average row matches |
| Convergence insensitivity | Section 2.3.3, Figure 4 | Yes |
| Static VWAP Eq. 40 | Section 4.3, page 10 | Yes |
| Dynamic VWAP Eq. 41 | Section 4.3, page 10 | Yes |
| MAPE definition Eq. 37 | Section 3.3, page 7 | Yes |
| VWAP tracking error Eq. 42 | Section 4.3, page 10 | Yes |
| Log-likelihood Eq. A.8 | Appendix A.1, page 13 | Cited but not included (MD1) |

All "Researcher inference" items are correctly labeled as such.

---

## Overall Assessment

The spec demonstrates strong understanding of the paper and provides a comprehensive
blueprint. The pseudocode structure is clear and the data flow diagram is helpful. The
parameter table, validation benchmarks, and edge cases sections are thorough.

The 3 major issues (M1-M3) would each independently cause incorrect numerical results.
M2 is the most consequential -- using stale parameters in the M-step breaks the EM
convergence guarantee and could cause the developer to spend significant debugging
time on what appears to be a correct implementation. These must be fixed before the
spec is usable.

The medium issues (MD1-MD4) would cause implementation friction or subtle bugs in
edge cases. MD1 (missing log-likelihood formula) is the most impactful since every
other formula is fully specified.

Issue count: 3 major, 4 medium, 3 minor.
