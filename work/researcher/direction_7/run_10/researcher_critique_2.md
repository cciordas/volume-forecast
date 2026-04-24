# Critique of Implementation Specification Draft 2: Kalman Filter State-Space Model

**Direction:** 7 — Kalman Filter State-Space Model
**Run:** 10
**Role:** Critic
**Draft reviewed:** impl_spec_draft_2.md

## Summary

Draft 2 is a substantial improvement over draft 1. All 10 issues from critique 1
(2 major, 5 moderate, 3 minor) have been addressed correctly. The critical robust r
formula error is fixed, the log-likelihood formula is now integrated into the forward
pass, the Joseph form is the primary covariance update, bias correction is explicit in
Step 6, and missing-observation handling is thorough throughout.

Re-verification of all key formulas against the paper confirms correctness: the standard
r update (Eq A.38), robust r update (Eq 35), all EM M-step updates (Eqs A.32-A.39),
the soft-thresholding solution (Eqs 33-34), and the cross-covariance recursion
(Eqs A.20-A.21) all match their paper sources.

The remaining issues are moderate to minor. I count 0 major issues, 2 moderate issues,
and 2 minor issues.

---

## Moderate Issues

### Mo1. Dynamic VWAP uses wrong forecast variance for multi-step bins (Implementability)

The dynamic VWAP section (Step 6, lines 513-524) computes bias-corrected volume
forecasts for all remaining bins j = i..I:

```
V_i = S_tau = C * Sigma[tau|tau-1] * C^T + r
vol_hat_linear[t, j] = exp(y_hat_dynamic[t, j] + 0.5 * V_j) for j = i..I
```

The comment says "Use dynamic one-bin-ahead forecast for remaining bins," but this is
incorrect for j > i. At bin i, the forecast for bin i itself is one-step-ahead, but
forecasts for bins i+1, i+2, ..., I are multi-step-ahead (h = 2, 3, ..., I-i+1 steps).
The forecast variance V_j must increase with forecast horizon h:

```
V_{i+h} = C * Sigma[tau+h | tau] * C^T + r    for h = 0, 1, ..., I-i
```

where Sigma[tau+h|tau] is the multi-step prediction covariance computed via the
recursion already described in Step 2's multi-step prediction section (line 184-188).
Using the one-step variance S_tau for all remaining bins underestimates the variance
for distant bins, causing the bias correction to be too small for those bins. This
systematically underweights high-uncertainty bins relative to near-term bins.

The fix is straightforward: at each bin i, run the multi-step prediction recursion for
h = 1..I-i to obtain Sigma[tau+h|tau] and V_{i+h} for each remaining bin. This
machinery is already defined in the spec (Step 2 multi-step prediction) but not
referenced here.

Additionally, the comment "Use dynamic one-bin-ahead forecast for remaining bins"
should be changed to "Use multi-step forecasts from current state for remaining bins."

Evidence: The multi-step prediction variance formula V[tau+h|tau] = C * Sigma[tau+h|tau]
* C^T + r is already in Step 2 (line 187). This variance grows with h because each
additional prediction step adds process noise Q. The static VWAP section (lines 507-511)
correctly uses Sigma[tau_last + i | tau_last] for each bin, but the dynamic section does
not mirror this pattern.

### Mo2. EM convergence check has off-by-one indexing error (Correctness)

The EM pseudocode (Step 4, lines 259-337) has a subtle indexing error in the
convergence check. The flow is:

```
Set j = 0
Repeat:
  E-step with theta^(j) -> produces log_likelihood^(j)
  M-step -> produces theta^(j+1)
  j = j + 1
  If j >= 2:
    rel_change = |log_likelihood^{(j)} - log_likelihood^{(j-1)}| / |log_likelihood^{(j-1)}|
```

Tracing the execution:
- Iteration 1: j=0. E-step stores LL indexed as LL^(0). M-step. j becomes 1. Check j>=2? No.
- Iteration 2: j=1. E-step stores LL indexed as LL^(1). M-step. j becomes 2. Check j>=2? Yes. References LL^(2) and LL^(1). But LL^(2) has not been computed -- the most recently computed LL is LL^(1) from the just-completed E-step.

The log-likelihood is computed during the E-step and indexed by the value of j at that
time (before increment). After j = j + 1, the check references LL^(j) = LL^(new j),
which is one ahead of the last computed value.

Two possible fixes:

**Option A** (preferred -- also more efficient): Move the convergence check to immediately
after the E-step, before the M-step:

```
Set j = 0
Repeat:
  E-step with theta^(j) -> produces log_likelihood^(j)
  If j >= 1:
    rel_change = |log_likelihood^{(j)} - log_likelihood^{(j-1)}| / |log_likelihood^{(j-1)}|
    If rel_change < epsilon: Stop (converged)
  M-step -> produces theta^(j+1)
  j = j + 1
```

This avoids the off-by-one and also saves an unnecessary M-step computation when
convergence is detected.

**Option B**: Keep the current placement but fix the indices:

```
  j = j + 1
  If j >= 2:
    rel_change = |log_likelihood^{(j-1)} - log_likelihood^{(j-2)}| / |log_likelihood^{(j-2)}|
```

Option A is preferred because it is clearer and avoids an unnecessary M-step on the
final iteration.

---

## Minor Issues

### mi1. Prediction step not explicitly guarded for tau=1 (Clarity)

The Kalman filter pseudocode (Step 2, lines 101-153) initializes x_hat[1|0] = pi_1 and
Sigma[1|0] = Sigma_1 (lines 97-98), then enters the loop "For tau = 1, 2, ..., N"
where the prediction step computes:

```
x_hat[tau|tau-1] = A_to_tau * x_hat[tau-1|tau-1]
Sigma[tau|tau-1] = A_to_tau * Sigma[tau-1|tau-1] * A_to_tau^T + Q_to_tau
```

At tau=1, this would require x_hat[0|0] and Sigma[0|0], which do not exist. The comment
on line 115 says "For tau=1, use pi_1 and Sigma_1 directly (no transition needed)," but
this is not enforced in the pseudocode structure. A developer reading the loop might
implement the prediction step for all tau including tau=1 and encounter an index-out-of-
bounds error.

Suggested fix: add an explicit guard:

```
For tau = 1, 2, ..., N:
  If tau == 1:
    # Use initialization directly
    x_hat[1|0] = pi_1
    Sigma[1|0] = Sigma_1
  Else:
    # Prediction step
    x_hat[tau|tau-1] = A_to_tau * x_hat[tau-1|tau-1]
    Sigma[tau|tau-1] = A_to_tau * Sigma[tau-1|tau-1] * A_to_tau^T + Q_to_tau

  # Correction step (unchanged)
  [...]
```

### mi2. EM monotonicity check not integrated into pseudocode (Completeness)

The spec mentions (line 357-359) that the log-likelihood must be non-decreasing across
EM iterations and that a decrease indicates a bug. This is an excellent diagnostic, but
it is described only in a note below the pseudocode, not as an explicit check within the
loop. For a developer implementing this, an explicit assertion would catch bugs early:

```
If j >= 1 and log_likelihood^(j) < log_likelihood^(j-1) - 1e-10:
    Raise error: "EM log-likelihood decreased -- implementation bug"
```

The small tolerance (1e-10) accounts for floating-point rounding. This is a defensive
programming suggestion, not a theoretical issue.

---

## Verification of Citations (Draft 2)

| Spec Claim | Cited Source | Verified? |
|---|---|---|
| Decomposition y = eta + phi + mu + v | Paper Section 2, Eq 3 | Yes |
| State-space form Eqs 4-5 | Paper Section 2, Eqs 4-5 | Yes |
| Unified time index tau = (t-1)*I + i | Paper Section 2, below Eq 5 | Yes |
| A_to_tau time-varying at day boundaries | Paper Section 2, Eq 4 definition | Yes |
| Kalman filter pseudocode | Paper Algorithm 1, page 4 | Yes |
| Joseph form covariance update | Researcher inference (Grewal & Andrews) | Appropriate |
| Log-likelihood (prediction error decomposition) | Paper Appendix A.1, Eq A.8 | Yes |
| Multi-step prediction | Paper Eq 9 | Yes |
| Smoother pseudocode | Paper Algorithm 2, page 5 | Yes |
| Cross-covariance init Eq A.21 | Paper Appendix A.2, Eq A.21 | Yes |
| Cross-covariance recursion Eq A.20 | Paper Appendix A.2, Eq A.20 | Yes |
| EM M-step: pi_1 (Eq A.32) | Paper Appendix A.3, Eq A.32 | Yes |
| EM M-step: Sigma_1 (Eq A.33) | Paper Appendix A.3, Eq A.33 | Yes |
| EM M-step: a_eta (Eq A.34) | Paper Appendix A.3, Eq A.34 | Yes |
| EM M-step: a_mu (Eq A.35) | Paper Appendix A.3, Eq A.35 | Yes |
| EM M-step: sigma_eta^2 (Eq A.36) | Paper Appendix A.3, Eq A.36 | Yes |
| EM M-step: sigma_mu^2 (Eq A.37) | Paper Appendix A.3, Eq A.37 | Yes |
| EM M-step: r standard (Eq A.38) | Paper Appendix A.3, Eq A.38 | Yes |
| EM M-step: phi standard (Eq A.39/24) | Paper Appendix A.3, Eq A.39 / Eq 24 | Yes |
| Robust filter soft-thresholding (Eqs 29-34) | Paper Section 3.1, Eqs 29-34 | Yes |
| Robust r update (Eq 35) | Paper Section 3.2, Eq 35 | Yes (fixed from draft 1) |
| Robust phi update (Eq 36) | Paper Section 3.2, Eq 36 | Yes |
| VWAP formulas (Eqs 39-41) | Paper Section 4.3, Eqs 39-41 | Yes |
| Average dynamic MAPE 0.46 | Paper Table 3, "Average" row | Yes |
| VWAP tracking error 6.38 bps | Paper Table 4, "Average" row | Yes |
| EM convergence in 5-10 iterations | Paper Section 2.3.3, Figure 4 | Yes |
| D = 250 out-of-sample days | Paper Section 4.1 | Yes |
| Bias correction exp(y_hat + 0.5*V) | Researcher inference (log-normal theory) | Appropriate |
| Lambda guidance grid | Researcher inference | Appropriate |
| Missing obs handling | Researcher inference | Appropriate, well-reasoned |
| Smoother unchanged for robust case | Researcher inference | Appropriate, correct reasoning |

All citations verified. No mismatches found.

---

## Summary of Required Changes

| ID | Severity | Section | Action |
|----|----------|---------|--------|
| Mo1 | Moderate | Step 6 (Dynamic VWAP) | Use multi-step prediction variance V_{i+h} for bins j > i, reference Step 2 multi-step recursion |
| Mo2 | Moderate | Step 4 (EM loop) | Fix convergence check indexing; move check before M-step (Option A) |
| mi1 | Minor | Step 2 (KF loop) | Add explicit tau==1 guard in prediction step |
| mi2 | Minor | Step 4 (EM loop) | Add explicit LL monotonicity assertion inside loop |

---

## Overall Assessment

Draft 2 is a high-quality implementation specification. All critical and major issues
from the first round have been resolved correctly. The core algorithm (Kalman filter,
smoother, EM, robust extension) is now accurate and well-cited. The remaining 4 issues
are moderate-to-minor and primarily concern pseudocode clarity rather than algorithmic
correctness. A developer with Kalman filter experience could implement correctly from
this spec, though fixing Mo1 and Mo2 would prevent likely bugs in the VWAP strategy and
EM convergence check respectively.
