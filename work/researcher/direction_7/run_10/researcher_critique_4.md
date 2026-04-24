# Critique of Implementation Specification Draft 4: Kalman Filter State-Space Model

**Direction:** 7 — Kalman Filter State-Space Model
**Run:** 10
**Role:** Critic
**Draft reviewed:** impl_spec_draft_4.md

## Summary

Draft 4 is an excellent implementation specification. All 4 issues from critique 3
(1 moderate, 3 minor) have been addressed correctly:

- Mo1 (robust EM log-likelihood): Fixed. Lines 148-155 add an explicit If/Else
  branching on robust_mode, using e_tau_clean for log-likelihood accumulation in the
  robust case. Lines 186-196 provide a thorough note explaining why e_tau_clean is the
  correct residual (consistent with M-step derivation from Eqs 35-36). Lines 370-393
  reiterate both formulas (standard and robust) in a standalone summary, making the
  distinction unmistakable.
- mi1 (dynamic VWAP notation): Fixed. Lines 559-573 rewrite the multi-step predictions
  using tau_{i-1} as the conditioning point with h=1,...,I-i+1, consistent with standard
  Kalman filter conventions. No ambiguity remains about whether the base case is the
  filtered or prediction covariance.
- mi2 (y_hat_dynamic undefined): Fixed. Lines 568-569 explicitly define
  y_hat_dynamic[t, i+h-1] = C * x_hat[tau_{i-1} + h | tau_{i-1}] + phi_{i+h-1},
  with a clear mapping to the Step 2 multi-step prediction formulas.
- mi3 (EM iteration count inconsistency): Fixed. All references (lines 284, 691,
  766, 801-802) now consistently say "5-10 iterations typical, up to 20 for difficult
  cases."

Re-verification of all key formulas against the paper confirms continued correctness.
No regressions introduced.

The remaining issues are 1 moderate and 0 minor. I count 0 major issues, 1 moderate
issue, and 0 minor issues.

---

## Moderate Issues

### Mo1. Robust M-step (Step 5b) presents r and phi updates in wrong order (Implementability)

Step 4 (standard EM, lines 349-364) correctly orders the M-step updates: phi first
(lines 349-354), then r (lines 356-364), with an explicit note at line 350:

```
# Note: update phi BEFORE r, because r depends on phi^{(j+1)}.
```

This ordering is necessary because the r formula (Eq A.38) references phi^{(j+1)} --
the *updated* phi from the current M-step iteration, not the previous iteration's phi.

Step 5b (robust EM modifications, lines 471-518) presents the replacement formulas in
the **opposite order**: robust r first (lines 477-494), then robust phi (lines 496-501).
The robust r formula also depends on phi^{(j+1)} (visible at line 488:
`+ (phi_tau^{(j+1)})^2` and line 493: `+ 2 * z_star_tau * phi_tau^{(j+1)}`).

A developer implementing Step 5b would naturally read the code block top-to-bottom and
compute r before phi, producing incorrect r estimates. The ordering note from Step 4
is not repeated in Step 5b, and the reversed presentation order actively contradicts it.

**Fix**: Reorder the Step 5b code block to present robust phi first, then robust r
(matching Step 4's ordering). Add a cross-reference to the ordering note:

```
  # Modified seasonality (Eq 36) -- computed BEFORE r (same ordering as Step 4):
  For i = 1, 2, ..., I:
    T_obs_i = count of days t where is_observed[(t-1)*I + i] = True
    phi_i^{(j+1)} = (1/T_obs_i) * sum_{t : is_observed[(t-1)*I+i]} {
        y[t,i] - C * x_hat[(t-1)*I+i | N] - z_star_{(t-1)*I+i}
    }

  # Modified observation noise variance (Eq 35) -- uses phi^{(j+1)} from above:
  N_obs = count of tau where is_observed[tau] = True
  r^{(j+1)} = (1/N_obs) * sum_{tau : is_observed[tau]} {
      [standard + z_star terms as currently written]
  }
```

---

## Verification of Citations (Draft 4)

| Spec Claim | Cited Source | Verified? |
|---|---|---|
| Decomposition y = eta + phi + mu + v | Paper Section 2, Eq 3 | Yes |
| State-space form Eqs 4-5 | Paper Section 2, Eqs 4-5 | Yes |
| Unified time index tau = (t-1)*I + i | Paper Section 2, below Eq 5 | Yes |
| A_to_tau time-varying at day boundaries | Paper Section 2, Eq 4 definition | Yes |
| Kalman filter pseudocode (with tau=1 guard) | Paper Algorithm 1, page 4 | Yes |
| Joseph form covariance update | Researcher inference (Grewal & Andrews) | Appropriate |
| Log-likelihood (prediction error decomposition) | Paper Appendix A.1, Eq A.8 | Yes |
| Robust log-likelihood (cleaned innovation) | Researcher inference | Appropriate, consistent with M-step derivation |
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
| Dynamic VWAP y_hat_dynamic definition | Step 2 multi-step + notation | Yes, now explicit |
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
| Mo1 | Moderate | Step 5b (Robust EM) | Reorder robust phi and r updates so phi comes first (matching Step 4 ordering), and add ordering note |

---

## Overall Assessment

Draft 4 is an outstanding implementation specification that is ready for finalization
after one simple reordering fix. All issues from the previous three critique rounds
(2 critical/major in round 1, 2 moderate in round 2, 1 moderate + 3 minor in round 3)
have been resolved correctly across four drafts. The core algorithm (Kalman filter,
smoother, EM, robust extension, VWAP strategies) is accurate, comprehensively cited,
and directly translatable to code.

The single remaining issue (Mo1) is a presentation ordering problem in Step 5b: the
robust r and phi formulas are listed in the wrong order relative to their data
dependency. The fix is mechanical -- swap the two code blocks and add a note. No
algorithmic, mathematical, or conceptual issues remain.

A developer with basic Kalman filter knowledge could implement the full model correctly
from this spec after the Mo1 fix. The spec is thorough in its treatment of edge cases,
missing data, numerical stability, and the gap between the paper's assumptions and
real-world data.
