# Critique of Implementation Specification Draft 5: Kalman Filter State-Space Model

**Direction:** 7 — Kalman Filter State-Space Model
**Run:** 10
**Role:** Critic
**Draft reviewed:** impl_spec_draft_5.md

## Summary

Draft 5 resolves the single remaining issue from critique 4. The spec is now
complete and ready for finalization.

- Mo1 (Step 5b robust phi/r ordering): Fixed. Lines 478-484 now present robust
  phi first, with an explicit ordering note at lines 474-475:
  "As in Step 4, phi must be updated BEFORE r because the r formula depends on
  phi^{(j+1)}". Lines 486-503 follow with robust r, including a comment
  "uses phi^{(j+1)} from above". This matches the ordering in Step 4 (lines
  349-364) and eliminates the risk of a developer computing r before phi.

Re-verification of the robust formulas against paper Equations 35-36 (page 7)
confirms continued correctness:

- **Robust phi (Eq 36):** The spec's formula (lines 482-484) correctly subtracts
  z_star from each term: `y[t,i] - C * x_hat[...] - z_star_{...}`, matching
  the paper's `(1/T) * sum (y_{t,i} - C x_hat_{t,i} - z*_{t,i})`.

- **Robust r (Eq 35):** The spec's formula (lines 492-503) includes all 10
  terms: the 6 standard terms from Eq A.38 plus the 4 additional z_star terms
  (`+(z_star)^2`, `-2*z_star*y`, `+2*z_star*C*x_hat`, `+2*z_star*phi^{(j+1)}`).
  All signs and dependencies verified correct.

No regressions were introduced. The file size increase (48391 -> 48618 bytes) is
consistent with the reordering and added comments.

I count 0 major issues, 0 moderate issues, and 0 minor issues.

---

## Verification of Citations (Draft 5)

Spot-check of all key citations from prior rounds — no changes or regressions:

| Spec Claim | Cited Source | Verified? |
|---|---|---|
| Decomposition y = eta + phi + mu + v | Paper Section 2, Eq 3 | Yes |
| State-space form Eqs 4-5 | Paper Section 2, Eqs 4-5 | Yes |
| Kalman filter pseudocode | Paper Algorithm 1, page 4 | Yes |
| Joseph form covariance update | Researcher inference (Grewal & Andrews) | Appropriate |
| Log-likelihood (prediction error decomposition) | Paper Appendix A.1, Eq A.8 | Yes |
| Robust log-likelihood (cleaned innovation) | Researcher inference | Appropriate |
| Smoother pseudocode | Paper Algorithm 2, page 5 | Yes |
| Cross-covariance (Eqs A.20-A.21) | Paper Appendix A.2 | Yes |
| EM M-step updates (Eqs A.32-A.39) | Paper Appendix A.3 | Yes |
| Robust filter soft-thresholding (Eqs 29-34) | Paper Section 3.1 | Yes |
| **Robust phi update (Eq 36) — ordering fixed** | **Paper Section 3.2, Eq 36** | **Yes** |
| **Robust r update (Eq 35) — ordering fixed** | **Paper Section 3.2, Eq 35** | **Yes** |
| M-step update order (phi before r) | Researcher inference | Correct |
| VWAP formulas (Eqs 39-41) | Paper Section 4.3 | Yes |
| Average dynamic MAPE 0.46 | Paper Table 3 | Yes |
| VWAP tracking error 6.38 bps | Paper Table 4 | Yes |

All citations verified. No mismatches found.

---

## Overall Assessment

Draft 5 is a finalized implementation specification. All issues raised across
five critique rounds have been resolved:

| Round | Issues Found | Severity |
|-------|-------------|----------|
| 1 | 10 issues | 2 major, 5 moderate, 3 minor |
| 2 | 4 issues | 0 major, 2 moderate, 2 minor |
| 3 | 4 issues | 0 major, 1 moderate, 3 minor |
| 4 | 1 issue | 0 major, 1 moderate, 0 minor |
| 5 | 0 issues | — |

The specification comprehensively covers:
- Standard and robust Kalman filter with correct pseudocode
- Kalman smoother with cross-covariance computation
- EM algorithm with all closed-form M-step updates
- Log-likelihood computation (standard and robust variants)
- Missing observation handling throughout all components
- VWAP execution strategies (static and dynamic) with bias correction
- Parameter initialization, calibration procedure, and lambda selection guidance
- Edge cases, sanity checks, and known limitations

Every algorithmic step traces to a specific paper equation or is explicitly
marked as researcher inference with clear reasoning. A developer with basic
Kalman filter knowledge can implement the full model directly from this spec.

**Recommendation:** Finalize draft 5 as the implementation specification.
