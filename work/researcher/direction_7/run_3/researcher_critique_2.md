# Critique of Implementation Specification Draft 2: Kalman Filter State-Space Model for Intraday Volume

**Direction:** 7, **Run:** 3
**Role:** Critic
**Date:** 2026-04-10

## Summary

Draft 2 is a substantial improvement over draft 1. All 3 major issues, all 4
medium issues, and all 3 minor issues from critique 1 have been addressed. The
spec is now close to implementation-ready. This review found 0 major issues, 2
medium issues, and 3 minor issues. The medium issues would not produce incorrect
results but could cause confusion or wasted debugging time. The spec is usable
as-is for a competent developer, but the fixes below would improve clarity.

---

## Resolution of Previous Issues

All 10 issues from critique 1 are resolved:

| ID | Issue | Resolution | Verified |
|----|-------|------------|----------|
| M1 | Missing initial correction step | Added at lines 129-138 with missing obs handling | Yes |
| M2 | Stale M-step parameters | Grouped into dependency order with (j+1) superscripts | Yes, matches A.34-A.39 |
| M3 | phi indexing in r formula | Uses phi[tau]^(j+1) with explicit bin-mapping note | Yes, matches A.38 |
| MD1 | Missing log-likelihood formula | Full formula added at lines 347-356 | Yes (but see MD-NEW-1 below) |
| MD2 | Smoother in robust variant | Explicit statement added at lines 193-195 and 440-443 | Yes |
| MD3 | Missing obs in main pseudocode | Conditional integrated into Kalman filter loop | Yes |
| MD4 | D set labeling | Renamed to D_start with clarifying note | Yes |
| MN1 | Dynamic VWAP semantics | Explicit clarification added | Yes |
| MN2 | Robust r phi indexing | Uses phi[tau]^(j+1) consistently | Yes |
| MN3 | Parameter clamping | Clamping block added with epsilon = 1e-8 | Yes |

---

## Medium Issues

### MD-NEW-1. Log-likelihood formula uses smoothed states but should use filtered innovations

**Spec section:** Step 3, EM Algorithm, log-likelihood computation (lines 347-359)

The spec writes the log-likelihood (Equation A.8) as a function of the smoothed
states (x_hat[tau], mu_hat, eta_hat), and the description says "where
x_hat_filt[tau] are the smoothed states." This is conceptually confused and
could lead to a subtle implementation bug.

The log-likelihood in Equation A.8 (paper, page 13) is the *joint*
log-likelihood of the observed data and hidden states:

    log P({x_tau}, {y_tau})

This is the quantity whose *expected value* under the posterior (E-step) defines
the Q function that the M-step maximizes. The EM convergence check, however,
should monitor the *observed-data* log-likelihood log P({y_tau}), not the joint
log-likelihood. The observed-data log-likelihood can be computed efficiently from
the Kalman filter's innovations:

    log P({y_tau}) = -0.5 * sum_{tau=1}^{N} [ log(S[tau]) + e[tau]^2 / S[tau] ]
                     - (N/2) * log(2*pi)

where S[tau] = C * Sigma[tau|tau-1] * C^T + r is the innovation variance and
e[tau] = y[tau] - phi[tau] - C * x_hat[tau|tau-1] is the innovation. This is
the standard Kalman filter log-likelihood (prediction error decomposition).

The spec's current formula (from Equation A.8) is the joint log-likelihood, and
plugging in the smoothed states does not produce the correct observed-data
log-likelihood. In practice, the EM still works because the Q function increases
monotonically, and monitoring the Q function or parameter changes is sufficient
for convergence detection. But the spec labels this "log-likelihood for
convergence check" which is misleading.

**Impact:** A developer who implements the formula as written and uses smoothed
states will compute a quantity that is not the observed-data log-likelihood,
though it will still work for convergence monitoring (since the Q function is
also monotonically increasing under correct EM updates). The confusion arises if
the developer tries to use this value for model comparison (e.g., AIC/BIC) or
debugging — it will not match standard Kalman filter log-likelihood computations.

**Fix:** Either:
(a) Replace the formula with the prediction error decomposition (innovation-based)
    log-likelihood, which is the standard and computationally simpler form; or
(b) Keep the current formula but relabel it as "Q function value" (the expected
    complete-data log-likelihood) and note that it is used for convergence
    monitoring, not for model comparison. Add a note that the observed-data
    log-likelihood can be computed from the innovation sequence if needed.

Option (a) is preferred because the innovation-based formula is simpler, requires
no smoother output, and produces the actual model log-likelihood.

### MD-NEW-2. Dynamic VWAP formula does not match the paper's Equation 41

**Spec section:** Step 6, Dynamic VWAP (lines 529-534)

The spec writes:

    w[i] = volume_hat_dynamic[i] / sum_{j=i}^{I} volume_hat_dynamic[j]
           * (1 - sum_{k=1}^{i-1} w[k])

The paper's Equation 41 defines:

    w_{t,i}^{(d)} = volume_{t,i}^{(d)} / sum_{j=i}^{I} volume_{t,j}^{(d)}
                    * (1 - sum_{k=1}^{i-1} w_{t,k}^{(d)})          for i = 1, ..., I-1

These are algebraically equivalent only if the recursion is unrolled. However,
there is a simpler and less error-prone way to express this. The formula
simplifies to:

    w[i] = volume_hat_dynamic[i] / sum_{j=i}^{I} volume_hat_dynamic[j]
           * remaining_fraction

where remaining_fraction = 1 - sum_{k<i} w[k]. But note that remaining_fraction
can be computed incrementally (remaining = 1 initially, then remaining -= w[i]
after each bin), and if volume_hat_dynamic is updated between bins (because new
observations arrive), the denominator changes too.

The issue is that the spec does not make clear *when* volume_hat_dynamic[j] for
j > i is updated. The paper says "order slicing is revised at each new bin as a
new intraday volume is gradually observed" (page 10, below Equation 41). This
means:

1. At bin i, we have just observed volume at bin i.
2. We update the Kalman filter state using this observation.
3. We produce *new* forecasts for bins i+1, ..., I using the updated state.
4. We compute w[i+1] using these updated forecasts.

But the spec's formula computes w[i] using volume_hat_dynamic[i] — the forecast
for bin i. This forecast was made *before* bin i was observed. The actual
observed volume at bin i is not used in the weight for bin i. The spec's
clarifying text (lines 537-541) correctly states this, but the procedural flow
of "observe bin i → update filter → reforecast remaining → compute next weight"
is not spelled out step by step.

**Impact:** A developer could implement the dynamic VWAP loop incorrectly by
computing all weights at once using stale forecasts, rather than updating
forecasts after each observation.

**Fix:** Add a step-by-step procedure for dynamic VWAP:

```
remaining = 1.0
For i = 1, ..., I-1:
  -- Before bin i is observed:
  volume_hat_dynamic[i] = exp(y_hat[tau_i | tau_{i-1}]) * shares_out[t]
  w[i] = remaining * volume_hat_dynamic[i] / sum_{j=i}^{I} volume_hat_dynamic[j]
  remaining -= w[i]

  -- Execute w[i] fraction of the order in bin i

  -- After bin i is observed:
  Run Kalman filter correction with y[tau_i]
  Update forecasts for bins i+1, ..., I using the new filtered state

w[I] = remaining   -- execute remainder in last bin
```

This makes the interleaving of observation, filtering, and weight computation
explicit.

---

## Minor Issues

### MN-NEW-1. The M-step denominator for sigma_eta^2 uses (T-1) but the sum has T-1 terms only if the first day is excluded

**Spec section:** Step 3, M-step, sigma_eta^2 formula (line 320-322)

The spec writes:

    [sigma_eta^2]^(j+1) = (1/(T-1)) * sum_{tau in D_start} { ... }

D_start = {kI+1 for k=1,2,...}. For T training days, the day-start indices are
tau = I+1, 2I+1, ..., (T-1)*I+1, which gives T-1 terms. So the denominator
(T-1) is correct.

However, the paper's Equation A.36 (page 15) and the derivation in Equation
A.13 sum over tau = kI+1 for k = 1, ..., with the sum indexed by D (using the
paper's notation). The paper writes:

    [(sigma^eta)^2]^(j+1) = 1/(T-1) * sum_{tau=kI+1} { P^{(1,1)}_tau + ... }

This matches the spec. **No error here**, but the spec should explicitly state
that D_start has exactly T-1 elements (one for each day transition) so the
developer can verify the denominator. Currently the set definition says
"k=1,2,..." without an upper bound.

**Fix:** Change the D_start definition to include the upper bound:
"D_start = {tau : tau = k*I+1 for k=1,2,...,T-1}" to make the cardinality
explicit.

### MN-NEW-2. Sufficient statistics use notation inconsistent with the paper

**Spec section:** Step 3, EM Algorithm, Definitions (lines 258-268)

The spec defines:

    P[tau] = Sigma[tau|N] + x_hat[tau|N] * x_hat[tau|N]^T

This matches the paper's Equation A.19: P_tau = Sigma_{tau|N} + x_hat_{tau|N} *
x_hat_{tau|N}^T. Good.

However, for the cross-time statistic, the spec writes:

    P[tau,tau-1] = Sigma[tau,tau-1|N] + x_hat[tau|N] * x_hat[tau-1|N]^T

The paper's Equation A.16 defines P_{tau,tau-1} = E[x_tau * x_{tau-1}^T | ...],
and the smoother-based computation yields exactly this expression. So the formula
is correct.

The minor issue is that the spec uses `P^{(1,1)}[tau,tau-1]` and
`P^{(2,2)}[tau,tau-1]` to denote diagonal elements of the cross-time matrix, but
never explicitly states that these are the (1,1) and (2,2) elements of the 2x2
matrix P[tau,tau-1]. The notation is introduced for P[tau] (lines 265-266) but
not repeated for P[tau,tau-1] (lines 267-268), where it says "cross-time
eta*eta" and "cross-time mu*mu" but does not give the matrix element definition.

**Fix:** Add after line 268:

    P^{(1,1)}[tau,tau-1] = P[tau,tau-1][1,1]  -- cross-time eta*eta
    P^{(2,2)}[tau,tau-1] = P[tau,tau-1][2,2]  -- cross-time mu*mu

This mirrors the notation already given for P[tau] at lines 265-266.

### MN-NEW-3. The static prediction formula A^h does not specify the day boundary within the h steps

**Spec section:** Step 5, Static prediction (lines 486-494)

The spec writes:

    x_hat[tau+h|tau] = A^h * x_hat[tau|tau]

and then explains "Since eta is constant within a day (a_eta_tau = 1 for
intraday steps), the eta component does not change. The mu component decays as
(a_mu)^h toward zero."

This is correct for intraday steps (h = 1, ..., I within the same day), but for
static prediction the starting point is the end of day t (tau = t*I, the last
bin), and the first step crosses a day boundary (A[t*I] uses a_eta). So A^h is
not simply [[1,0],[0,a_mu^h]] — the first application of A uses a_eta, and
subsequent applications use 1.

The spec partially addresses this: "the state after the last bin of day t
propagates through the day boundary (applying a_eta once) and then through h-1
intraday steps." This is correct in prose, but the formula `A^h` is misleading
because A is time-varying — there is no single matrix to exponentiate.

**Fix:** Replace `A^h * x_hat[tau|tau]` with the explicit computation:

```
For static prediction of day t+1:
  -- Day boundary transition:
  x_hat[tI+1|tI] = [[a_eta, 0], [0, a_mu]] * x_hat[tI|tI]

  -- Intraday steps (no eta process noise, a_eta_tau = 1):
  For h = 2, ..., I:
    x_hat[tI+h|tI] = [[1, 0], [0, a_mu]] * x_hat[tI+h-1|tI]

  -- Equivalently:
  eta_hat = a_eta * eta_hat[tI|tI]            (constant for all bins)
  mu_hat[h] = a_mu^h * mu_hat[tI|tI]          (decays geometrically)
  y_hat[tI+h|tI] = eta_hat + mu_hat[h] + phi[h]
```

This makes the day-boundary transition explicit and avoids the misleading `A^h`
notation.

---

## Correctness Verification of New Material in Draft 2

| Spec Claim | Paper Source | Verified? |
|-----------|-------------|-----------|
| Initial correction step structure | Algorithm 1, implied | Yes — consistent with Algorithm 1's assumption that x_hat[tau\|tau] is available |
| Joseph form covariance update | Not in paper (Researcher inference) | Correctly labeled as Researcher inference; algebraically equivalent to standard form |
| M-step Group 1/Group 2 dependency order | Equations A.34-A.39 superscripts | Yes — A.34, A.35, A.39 use only sufficient statistics; A.36 uses a^eta(j+1), A.37 uses a^mu(j+1), A.38 uses phi(j+1) |
| Log-likelihood formula | Equation A.8 | Structurally matches but see MD-NEW-1 regarding usage |
| Parameter clamping epsilon = 1e-8 | Researcher inference | Correctly labeled; reasonable value |
| D_start = {kI+1} with clarifying note | Equations A.34, A.36 sums | Yes — matches paper's summation ranges |
| Robust smoother unchanged statement | Section 3.2 | Yes — z* only appears in Equations 35-36 (M-step); smoother not modified |
| Robust phi formula with z* subtracted | Equation 36 (=A.39 robust) | Yes — phi_i = (1/T) sum (y_{t,i} - C*x_hat_{t,i} - z*_{t,i}) matches Eq. 36 |
| Robust r formula with z* terms | Equation 35 | Yes — all z* cross-terms present and correctly signed |

---

## Overall Assessment

Draft 2 is a high-quality implementation specification. All major and medium
issues from critique 1 have been resolved correctly and thoroughly. The M-step
dependency ordering, phi indexing, initial correction step, missing observation
handling, and parameter clamping are all now correct and clearly presented.

The two new medium issues (MD-NEW-1 and MD-NEW-2) are matters of clarity rather
than correctness — the EM will converge correctly as implemented, and the VWAP
weights are mathematically correct. The fixes would prevent confusion during
implementation and testing.

The three minor issues are purely presentational and would take minimal effort
to address.

Issue count: 0 major, 2 medium, 3 minor.

The spec is ready for implementation after addressing the medium issues, or
usable as-is with the understanding that the log-likelihood formula is the Q
function (not the marginal likelihood) and the dynamic VWAP procedure requires
careful interleaving of observation and forecasting.
