# Critique of Implementation Specification Draft 3: Kalman Filter State-Space Model

**Direction:** 7 — Kalman Filter State-Space Model
**Run:** 10
**Role:** Critic
**Draft reviewed:** impl_spec_draft_3.md

## Summary

Draft 3 is a high-quality implementation specification. All 4 issues from critique 2
(2 moderate, 2 minor) have been addressed correctly:

- Mo1 (dynamic VWAP multi-step variance): Fixed. The dynamic VWAP section (lines 532-540)
  now uses multi-step prediction variance V_{i+h} with an explicit h-step loop, correctly
  accounting for growing forecast variance at longer horizons.
- Mo2 (EM convergence off-by-one): Fixed. The convergence and monotonicity checks
  (lines 276-285) are now placed immediately after the E-step and before the M-step,
  adopting Option A from the critique. Indexing is correct: at j>=1, log_likelihood^(j)
  and log_likelihood^(j-1) are both available from completed E-steps.
- mi1 (tau=1 guard): Fixed. Lines 117-123 add an explicit If/Else guard separating the
  tau==1 initialization from the tau>=2 prediction step.
- mi2 (EM monotonicity assertion): Fixed. Lines 278-279 add an explicit assertion with
  1e-10 tolerance, integrated into the main loop rather than described only in a note.

Re-verification of all key formulas against the paper confirms continued correctness.
No regressions introduced.

The remaining issues are 1 moderate and 3 minor. I count 0 major issues, 1 moderate
issue, and 3 minor issues.

---

## Moderate Issues

### Mo1. Robust model log-likelihood for EM convergence monitoring is unspecified (Completeness/Implementability)

The EM pseudocode (Step 4, lines 267-348) accumulates the log-likelihood during the
forward pass using the prediction error decomposition (Step 2, line 148):

```
log_likelihood += -0.5 * (e_tau^2 / S_tau + ln(S_tau) + ln(2*pi))
```

where e_tau = y[tau] - C * x_hat[tau|tau-1] - phi_tau is the raw innovation.

For the standard Kalman filter, this is correct. However, when the robust filter
(Step 5) is active, the state update uses the cleaned innovation e_tau_clean =
e_tau - z_star_tau, while the log-likelihood still accumulates using the raw e_tau.
This creates an inconsistency: the model's effective observation equation is
y_tau = C*x_tau + phi_tau + v_tau + z_tau (Eq 25), meaning the "noise" from the
model's perspective is v_tau = y_tau - C*x_tau - phi_tau - z_star_tau, not
y_tau - C*x_tau - phi_tau.

The practical consequence is that the EM monotonicity assertion (line 278-279) may
fire spuriously in the robust case. The standard prediction error decomposition
log-likelihood is not the correct objective for the robust EM, because the robust
model has an additional z_tau term that absorbs part of the innovation. The cleaned
innovation e_tau_clean better represents the model residual.

Two options:

**Option A** (recommended): In the robust case, accumulate the log-likelihood using
the cleaned innovation:

```
# Robust log-likelihood accumulation (replaces standard line 148 when robust filter active):
log_likelihood += -0.5 * (e_tau_clean^2 / S_tau + ln(S_tau) + ln(2*pi))
```

This treats the z_star-cleaned observation as the effective data point, consistent
with the M-step updates (Eqs 35-36) which are derived from E[(y - z_star - phi - Cx)^2].

**Option B**: Monitor parameter convergence instead of log-likelihood convergence for
the robust case:

```
# Parameter-based convergence (robust alternative):
param_change = max(|a_eta^(j+1) - a_eta^(j)| / |a_eta^(j)|,
                   |a_mu^(j+1) - a_mu^(j)| / |a_mu^(j)|,
                   |r^(j+1) - r^(j)| / |r^(j)|, ...)
If param_change < epsilon: Stop
```

Option A is preferred because it preserves the structure of the existing convergence
check and is simpler to implement.

The paper does not discuss convergence monitoring for the robust EM (Algorithm 3 says
only "until convergence" without specifying the criterion). This is therefore a gap
in the paper that the spec must fill, and the current spec fills it only for the
standard case.

---

## Minor Issues

### mi1. Dynamic VWAP multi-step conditioning notation is confusing (Clarity)

The dynamic VWAP section (lines 534-539) uses the notation:

```
Sigma[tau_i + h | tau_i] for h = 0, 1, ..., I-i
where Sigma[tau_i + 0 | tau_i] = Sigma[tau_i | tau_i - 1] (the prediction covariance).
```

In standard Kalman filter notation, "Sigma[a | b]" means the covariance of the state
at time a conditioned on observations through time b. Under this convention,
"Sigma[tau_i | tau_i]" is the *filtered* covariance (after incorporating the
observation at tau_i), which is different from "Sigma[tau_i | tau_i - 1]" (the
*prediction* covariance, before incorporating tau_i's observation).

The spec redefines the base case to be the prediction covariance, which breaks the
standard convention. A developer familiar with Kalman filter notation might implement
the recursion starting from the filtered covariance Sigma[tau_i | tau_i] (after
correction), producing variances that are too small (because they assume bin i has
already been observed when it has not).

The computation is correct in substance -- the multi-step recursion starting from
Sigma[tau_i | tau_i - 1] and propagating forward without corrections produces the
correct forecast variances conditioned on tau_{i-1}. The fix is purely notational:

```
# At bin i, information available: observations 1..i-1 (filtered state at tau_{i-1}).
# Multi-step predictions from tau_{i-1} for remaining bins:
For h = 1, 2, ..., I-i+1:
  # h=1 -> bin i (one-step-ahead), h=2 -> bin i+1 (two-step-ahead), etc.
  V_{i+h-1} = C * Sigma[tau_{i-1} + h | tau_{i-1}] * C^T + r
```

where Sigma[tau_{i-1} + h | tau_{i-1}] uses the standard multi-step prediction
recursion from Step 2 with starting point Sigma[tau_{i-1} | tau_{i-1}] (filtered
covariance after bin i-1). This is notationally consistent and produces the same
numerical values.

### mi2. y_hat_dynamic referenced but not defined in VWAP section (Clarity)

Line 540 references "y_hat_dynamic[t, i+h]" in the volume exponentiation:

```
vol_hat_linear[t, i+h] = exp(y_hat_dynamic[t, i+h] + 0.5 * V_{i+h})
```

This variable is not defined within the VWAP section. A developer must infer that:
- For h=0: y_hat_dynamic[t, i] = C * x_hat[tau_i | tau_{i-1}] + phi_i (the standard
  one-step-ahead forecast from Step 2, line 151).
- For h>0: y_hat_dynamic[t, i+h] = C * x_hat[tau_i + h | tau_{i-1}] + phi_{i+h}
  (the multi-step forecast from Step 2, line 190).

Adding an explicit definition would prevent ambiguity:

```
# Forecast mean for each remaining bin:
y_hat_dynamic[t, i+h] = C * x_hat[tau_{i-1} + h+1 | tau_{i-1}] + phi_{i+h}
                         for h = 0, 1, ..., I-i
```

### mi3. Internal inconsistency in EM iteration count ranges (Consistency)

The spec states two different ranges for typical EM convergence:
- Line 265 (Step 4 pseudocode comment): "typical convergence in 5-20 iterations"
- Line 764 (Validation section): "EM convergence should occur within 5-20 iterations"
- Line 653 (Parameters table, max_iterations): "Paper Fig 4 shows convergence in ~5-10
  iterations"

The narrower "5-10" range appears in the Parameters table and is closer to what
Figure 4 shows (most parameters converge by iteration 5-8). The broader "5-20" range
in the pseudocode and validation sections provides more margin. These are not
contradictory but the inconsistency could confuse a developer setting expectations.

Suggested fix: use "5-10" as the typical range (matching the paper) and note that
"up to 20 iterations may be needed for difficult datasets" as a separate qualifier.

---

## Verification of Citations (Draft 3)

| Spec Claim | Cited Source | Verified? |
|---|---|---|
| Decomposition y = eta + phi + mu + v | Paper Section 2, Eq 3 | Yes |
| State-space form Eqs 4-5 | Paper Section 2, Eqs 4-5 | Yes |
| Unified time index tau = (t-1)*I + i | Paper Section 2, below Eq 5 | Yes |
| A_to_tau time-varying at day boundaries | Paper Section 2, Eq 4 definition | Yes |
| Kalman filter pseudocode (with tau=1 guard) | Paper Algorithm 1, page 4 | Yes |
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
| Robust r update (Eq 35) | Paper Section 3.2, Eq 35 | Yes |
| Robust phi update (Eq 36) | Paper Section 3.2, Eq 36 | Yes |
| EM convergence check (before M-step) | Researcher inference | Appropriate, correct placement |
| EM monotonicity assertion | Researcher inference | Appropriate |
| VWAP formulas (Eqs 39-41) | Paper Section 4.3, Eqs 39-41 | Yes |
| Dynamic VWAP multi-step variance | Paper Eq 9 + Step 2 multi-step | Yes, correctly integrated |
| Average dynamic MAPE 0.46 | Paper Table 3, "Average" row | Yes |
| VWAP tracking error 6.38 bps | Paper Table 4, "Average" row | Yes |
| EM convergence in 5-10 iterations | Paper Section 2.3.3, Figure 4 | Yes |
| D = 250 out-of-sample days | Paper Section 4.1 | Yes |
| Bias correction exp(y_hat + 0.5*V) | Researcher inference (log-normal theory) | Appropriate |
| Lambda guidance grid | Researcher inference | Appropriate |
| Missing obs handling | Researcher inference | Appropriate, well-reasoned |
| Smoother unchanged for robust case | Researcher inference | Appropriate, correct reasoning |
| M-step update order (phi before r) | Researcher inference | Correct, follows from Eq A.38 |

All citations verified. No mismatches found.

---

## Summary of Required Changes

| ID | Severity | Section | Action |
|----|----------|---------|--------|
| Mo1 | Moderate | Steps 2/4/5 (Robust LL) | Specify that robust EM uses cleaned innovation e_tau_clean for log-likelihood accumulation |
| mi1 | Minor | Step 6 (Dynamic VWAP) | Rewrite multi-step conditioning notation to use tau_{i-1} as the conditioning point |
| mi2 | Minor | Step 6 (Dynamic VWAP) | Define y_hat_dynamic explicitly in terms of Step 2 forecast formulas |
| mi3 | Minor | Steps 4/Parameters/Validation | Harmonize EM iteration count: "5-10 typical, up to 20 for difficult cases" |

---

## Overall Assessment

Draft 3 is an excellent implementation specification that is very close to final quality.
All critical, major, and moderate issues from previous rounds have been resolved
correctly. The core algorithm (Kalman filter, smoother, EM, robust extension, VWAP
strategies) is accurate, well-cited, and directly translatable to code. The remaining
moderate issue (robust log-likelihood) is a completeness gap in an area where the paper
itself is silent, and the 3 minor issues are clarity improvements. A competent developer
with Kalman filter experience could implement the full model correctly from this spec,
with the caveat that the robust EM convergence monitoring needs the clarification
described in Mo1 to avoid a spurious monotonicity assertion failure.
