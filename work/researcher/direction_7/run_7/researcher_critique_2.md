# Critique of Implementation Specification Draft 2: Kalman Filter State-Space Model for Intraday Volume

## Summary

Draft 2 is a substantial improvement over draft 1. All 5 major issues from
critique 1 have been resolved, most importantly M1 (Kalman filter initialization
convention now correctly matches Algorithm 1). The unified robust/standard code
path is clean and well-documented. The pseudocode is detailed enough for direct
implementation.

I found no major issues. The 7 minor issues below are documentation
clarifications and missing function definitions that would reduce implementation
ambiguity. The specification is ready for implementation with these small
improvements.

---

## Resolution of Critique 1 Issues

| Issue | Status | Notes |
|-------|--------|-------|
| M1 (KF init convention) | FIXED | x_filt[1]=pi_1, loop tau=2..N. Matches Algorithm 1 exactly. Convention note at lines 138-143 is clear and correct. |
| M2 (conflicting cross-covariance) | FIXED (was N/A in this draft lineage) | Single non-recursive formula presented. |
| M3 (L_stored not returned) | FIXED (was N/A in this draft lineage) | L_stored in smoother output (line 228) and stored in loop (line 248). |
| M4 (drafting artifacts) | FIXED (was N/A in this draft lineage) | No artifacts. Tables are clean. |
| M5 (no unified robust EM) | FIXED | Unified code path with z_star=0 default. Explicit unification paragraph in Step 7 (lines 462-468). |
| m1 (VWAP efficiency) | FIXED | Efficiency note at lines 632-637 confirms O(I^2) is acceptable. |
| m2 (lambda grid) | FIXED | Data-adaptive grid guidance at lines 831-837. |
| m3 (MAPE/TE pseudocode) | FIXED | Step 13 added with both formulas. |
| m4 (missing obs pseudocode) | FIXED | Integrated into Step 2 correction (lines 203-209). |
| m5 (RM dynamic N/A) | FIXED | Both tables now show N/A for RM dynamic. |
| m6 (N_candidates range) | FIXED | Candidate grid at line 871. |
| m7 (predict_dynamic caching) | FIXED | Production note at lines 549-553. |
| m8 (Sigma_1 floor) | FIXED | Floor at line 120: max(..., 0.01 * var(daily_avg)). |

---

## Minor Issues

### m1. DynamicPredictDay in Step 14 is used but never defined

**Location:** Step 14, line 707.

**Problem:** The rolling-window calibration calls `DynamicPredictDay(theta,
y_actual_day_d, I, lambda)` but this function is not defined anywhere in the
pseudocode. A developer must infer that this means: (1) take the end-of-training
filtered state from the EM output, (2) run Step 9 (DynamicPredict) sequentially
for each bin i=1..I of day d, using y_actual_day_d[i] as the observation, and
(3) collect the one-step-ahead predictions y_hat[i] before each correction.

**Impact:** Medium. A developer who implements this incorrectly (e.g., using
static prediction instead of dynamic, or forgetting to carry the filtered state
forward) would get wrong cross-validation MAPE values and select suboptimal
hyperparameters.

**Fix:** Add a brief pseudocode block:

```
function DynamicPredictDay(theta, x_filt_last, Sigma_filt_last, y_day, I, lambda):
    # One-step-ahead dynamic predictions for a full day
    y_pred = array(I)
    x_prev = x_filt_last
    Sigma_prev = Sigma_filt_last
    for i = 1 to I:
        x_new, Sigma_new, y_hat, _ = DynamicPredict(
            x_prev, Sigma_prev, y_day[i], theta, i, I, lambda)  # Step 9
        y_pred[i] = y_hat
        x_prev = x_new
        Sigma_prev = Sigma_new
    return y_pred
```

Note: this also makes explicit that the function needs the end-of-training
filtered state and covariance as inputs, which are not mentioned in Step 14's
current call signature.

### m2. extract_window in Step 14 is used but never defined

**Location:** Step 14, line 702.

**Problem:** `extract_window(y_all, d, T_train, I)` is called without defining
what it returns or how it indexes. A developer needs to know:
- Does the window end at day d-1 (exclusive of validation day) or day d?
- Does the function also extract the corresponding `observed`, `phi_position`,
  and `shares_outstanding` arrays, or just `y`?
- Is d a day index or a bin index?

**Impact:** Low-medium. Incorrect window boundaries would cause look-ahead bias
(if validation day data leaks into training) or wasted data (if the window ends
too early).

**Fix:** Add a brief definition or clarify in prose: "extract_window returns
y[1..N_train], observed[1..N_train], phi_position[1..N_train] for the N_train =
T_train * I bins ending at the last bin of day d-1 (the day immediately before
validation day d). The training window must not overlap the validation day."

### m3. Dynamic VWAP does not produce end-of-day filtered state

**Location:** Step 11, lines 592-623.

**Problem:** The dynamic VWAP loop processes observations y_live[1] through
y_live[I-1] via DynamicPredict, but the last observation y_live[I] is never
passed through DynamicPredict (the loop just assigns the remaining weight). This
means the filtered state after the complete trading day is not available from
Step 11's outputs.

A developer who uses dynamic VWAP for live execution and also needs the
end-of-day filtered state for the next day's prediction (or for the rolling
EM warm start) would need to run an additional DynamicPredict call after the
VWAP loop completes.

**Impact:** Low. This is an interface design issue, not a correctness issue. The
VWAP weights themselves are correct.

**Fix:** Either:
(a) Add a note after Step 11 stating that the developer should run one final
    DynamicPredict call with y_live[I] to obtain the end-of-day state; or
(b) Move the DynamicPredict call inside the `else` branch at i=I (after
    assigning the weight) and add x_filt_final, Sigma_filt_final to the output.

Option (a) is simpler and avoids complicating the VWAP pseudocode.

### m4. Robust EM convergence fallback lacks trigger specification

**Location:** Step 6, lines 409-414.

**Problem:** The spec mentions a parameter-change fallback criterion
`max(|theta_new - theta_old| / (|theta_old| + 1e-10)) < tol` for the robust
model but does not specify when to use it. Should both criteria be checked every
iteration? Should the fallback activate only when the log-likelihood decreases?
Should it use the same `tol` value as the log-likelihood criterion?

**Impact:** Low. The robust EM typically converges even with only the
log-likelihood criterion (Lasso penalty violations are small in practice). But
for edge cases with highly contaminated data, the lack of specification could
cause the EM to oscillate without terminating.

**Fix:** Add a concrete recommendation:

```
# Convergence check (robust model):
# Primary: relative change in innovation log-likelihood
# Fallback: relative change in parameter vector (triggers if LL decreases)
if relative_change < tol AND j > 1:
    break
if log_lik < log_lik_prev AND j > 1:
    param_change = max(|theta_new - theta_old| / (|theta_old| + 1e-10))
    if param_change < tol:
        break
```

Mark as researcher inference.

### m5. Cross-covariance formula citation is imprecise

**Location:** Step 4, lines 282-286.

**Problem:** The spec cites "Shumway and Stoffer (1982), Property 6.3" for the
non-recursive cross-covariance formula. Property 6.3 is from the Shumway and
Stoffer textbook (Time Series Analysis and Its Applications, multiple editions:
2000, 2006, 2011, 2017), not the 1982 paper. The 1982 paper ("An Approach to
Time Series Smoothing and Forecasting Using the EM Algorithm") introduced EM for
state-space models but does not contain this specific numbered property.

**Impact:** Very low. A developer looking up the 1982 paper to verify the formula
would not find it, but the formula is easily verified algebraically.

**Fix:** Change citation to "Shumway and Stoffer (2006, Property 6.3)" or
"Shumway and Stoffer (2017, Property 6.3)". The non-recursive form
Sigma_{tau,tau-1|N} = Sigma_smooth[tau] @ L[tau-1]^T is also derivable directly
from the RTS smoother equations.

### m6. A_used subscript convention differs from paper without explicit mapping

**Location:** Steps 2-4 (Kalman filter, smoother, sufficient statistics).

**Problem:** The paper subscripts the transition matrix with the SOURCE time
step: A_tau transitions from tau to tau+1 (Algorithm 1, line 2: x_{tau+1|tau} =
A_tau x_{tau|tau}). The spec subscripts with the DESTINATION time step:
A_used[tau] transitions from tau-1 to tau (Step 2, line 170: x_pred[tau] =
A_used[tau] @ x_filt[tau-1]).

This means paper's A_tau = spec's A_used[tau+1]. The smoother correctly uses
A_used[tau+1] (line 247), but a developer cross-referencing the spec with the
paper's Algorithm 2 (which uses A_tau in L_tau = Sigma_{tau|tau} A_tau^T
Sigma_{tau+1|tau}^{-1}) might be confused by the index shift.

**Impact:** Low. The pseudocode is internally consistent, so a developer who
follows only the spec will implement correctly. The risk is only for developers
who verify against the paper.

**Fix:** Add a one-line mapping note to the Paper References table or to Step 3:
"Note: the paper's A_tau (source-indexed) corresponds to this spec's
A_used[tau+1] (destination-indexed)."

### m7. N_obs denominators not explicitly marked as researcher inference

**Location:** Step 5, lines 354 and 361.

**Problem:** The r update uses N_obs (count of observed bins) and the phi update
uses count_i (count of observed bins at position i) instead of the paper's N and
T respectively. While this is the correct MLE adjustment for missing data, the
spec only marks it with a comment ("N_obs instead of N for missing data
correctness") without the "researcher inference" label used elsewhere in the
document for deviations from the paper.

**Evidence:** Paper Eq A.38 uses N; Eq A.39 uses T. The paper excludes
zero-volume bins entirely (Section 4.1) rather than treating them as missing
observations, so the paper's formulas implicitly assume all bins are observed.

**Impact:** Very low. The deviation is correct and well-motivated; it just needs
consistent labeling for traceability.

**Fix:** Add "(researcher inference)" after the N_obs comment on line 361 and
after the count_i logic on line 346.

---

## Verified Correct

The following elements were checked against the paper and confirmed accurate in
draft 2:

- **KF initialization convention** (Algorithm 1): x_filt[1] = pi_1, loop
  tau=2..N, first correction at tau=2 using y[2]. Matches paper exactly.
  y[1] contributes only through M-step sums and smoother backward pass.
- **KF prediction/correction formulas**: x_pred, Sigma_pred, K, e, S, x_filt,
  Sigma_filt all match Algorithm 1 lines 2-6. Joseph form is a numerically
  superior equivalent of line 6.
- **Soft-thresholding** (Eqs 33-34): threshold = lambda * S / 2. Matches
  paper's lambda/(2W) since W = 1/S (Eq 30).
- **RTS smoother** (Algorithm 2): L, x_smooth, Sigma_smooth match lines 2-4.
  Correct use of A_used[tau+1] for paper's A_tau.
- **Non-recursive cross-covariance**: Sigma_cross[tau] = Sigma_smooth[tau] @
  L[tau-1]^T. Verified algebraic equivalence with A.20-A.21 at tau=N:
  reduces to (I - K_N C) A_{N-1} Sigma_{N-1|N-1}, matching A.21 exactly.
  General case follows from Shumway & Stoffer textbook Property 6.3.
- **M-step pi_1 = x_smooth[1]** (Eq A.32): Correct.
- **M-step Sigma_1 = Sigma_smooth[1]**: Algebraically equivalent to A.33
  (P_1 - x_hat_1 x_hat_1^T) since P_1 = Sigma_smooth[1] + x_smooth[1]
  x_smooth[1]^T. Avoids catastrophic cancellation.
- **M-step a_eta** (Eq A.34): Sum over D_start = {kI+1, k=1..T-1}. Correct
  indices and P_cross element [0,0].
- **M-step a_mu** (Eq A.35): Sum over tau=2..N. Correct indices and P_cross
  element [1,1].
- **M-step sig2_eta** (Eq A.36): Uses updated a_eta^(j+1). Denominator T-1.
  Correct.
- **M-step sig2_mu** (Eq A.37): Uses updated a_mu^(j+1). Denominator N-1.
  Correct.
- **M-step phi** (Eq A.39 / Eq 36): Includes z_star subtraction for robust
  case. Correct.
- **M-step r** (Eq A.38 / Eq 35): Residual = y - phi - z_star - C x_smooth.
  Verified: expanding (y-phi-z*-Cx)^2 + C Sigma C^T reproduces the full A.38
  expression using P = Sigma + x x^T. Correct.
- **M-step ordering**: phi before r (Eqs A.30-A.31), a_eta before sig2_eta,
  a_mu before sig2_mu. All correctly documented.
- **y[1] in M-step**: r and phi sums include tau=1, consistent with Eqs A.38-
  A.39. y[1] enters through x_smooth[1] and the direct residual, not through
  a filter correction step. Correctly documented.
- **Innovation log-likelihood sum range**: tau=2..N (no innovation at tau=1).
  Correct.
- **Unified robust/standard path**: lambda=1e10 makes threshold >> any
  plausible innovation, so z_star=0 everywhere. M-step Eqs 35-36 reduce to
  A.38-A.39 when z_star=0. Correct.
- **Static prediction** (Eq 9): Day boundary at h=1, within-day transitions
  for h>1. Correct.
- **Dynamic prediction** (Algorithm 1, single-step): Same correction logic
  as Step 2. Correct.
- **VWAP weights** (Eqs 40-41): Static (proportional to predicted volume) and
  dynamic (proportional to predicted remaining volume, scaled by unexecuted
  fraction). Correct.
- **MAPE** (Eq 37): Linear scale, exp(y_pred) vs exp(y_actual). Correct. No
  Jensen's bias correction, matching paper.
- **VWAP TE** (Eq 42): Basis points. Correct.
- **MAPE benchmarks** (Table 3): Dynamic (Robust KF: 0.46, KF: 0.47, CMEM:
  0.65, RM: 1.28), Static (Robust KF: 0.61, KF: 0.62, CMEM: 0.90, RM:
  1.28). All correct.
- **VWAP TE benchmarks** (Table 4): Dynamic (Robust KF: 6.38, KF: 6.39,
  CMEM: 7.01), Static (Robust KF: 6.85, KF: 6.89, CMEM: 7.71, RM: 7.48).
  RM dynamic correctly marked N/A. All correct.
- **Parameter table**: All 13 parameters listed with reasonable ranges.
- **Initialization heuristics**: Consistent with Figure 4 (EM robust to
  init). sig2_eta floor prevents near-singular Sigma_1.
- **Data flow diagram and types table**: Shapes and indexing consistent with
  pseudocode.

---

## Issue Count Summary

| Severity | Count |
|----------|-------|
| Major    | 0     |
| Minor    | 7     |

All minor issues are documentation clarifications or missing function
definitions. No algorithmic errors were found. The specification accurately
reflects the paper's model and algorithms, with all deviations from the paper
(Joseph form, non-recursive cross-covariance, N_obs denominators, missing
observation handling) clearly identified and correctly motivated.

The specification is implementation-ready. The minor issues above would improve
developer confidence when cross-referencing with the paper but are unlikely to
cause implementation errors for a developer who follows the pseudocode as
written.
