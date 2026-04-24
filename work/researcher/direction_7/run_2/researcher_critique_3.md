# Critique of Implementation Specification Draft 3: Kalman Filter State-Space Model for Intraday Volume

## Summary

Draft 3 correctly addresses all 4 issues from critique 2. The cross-covariance
regression (M1) is fixed by replacing the recursive Eq A.20 with the simpler
non-recursive Shumway & Stoffer (1982) formula. The E_1-E_4 formulas are now
explicit with correct post-M-step simplification. Zero-volume bin handling is
clear and consistent throughout. Sigma_1 uses the numerically stable direct
assignment.

I verified all corrections against the paper (pages 1-10, 14-16) and found no
new major issues. The specification is now self-contained, internally
consistent, and implementable. I identify 2 minor issues that a developer
should be aware of but that do not block implementation.

---

## Major Issues

None.

---

## Minor Issues

### m1. Dynamic VWAP function signature implies batch computation but production requires incremental recomputation

**Spec location:** Lines 613-631 (compute_vwap_weights_dynamic function).

**Problem:** The function takes a single vector vol_hat_dynamic[1..I] as input
and computes all weights in one pass. The comment (lines 615-617) says
vol_hat_dynamic[i] is "the one-step-ahead forecast made at bin i-1 (i.e.,
re-forecast using all information up to bin i-1 via Kalman correction)."

However, the production description (lines 649-652) says: at each bin i,
"predict forward for bins i through I" and "compute vol_hat_dynamic[i..I]."
This means the denominator SUM(vol_hat_dynamic[i..I]) should use multi-step
forecasts from state at bin i-1 -- not one-step-ahead forecasts from
different states.

For **backtesting** (the paper's evaluation in Section 4.3), this distinction
does not matter: vol_hat_dynamic[i] is the one-step-ahead forecast, and
the formula in Eq 41 uses these collected forecasts to compute weights
retrospectively. The function as written correctly implements this.

For **production** (computing w[i] in real-time before observing y[i]), the
denominator must use multi-step forecasts from the current state, because
future one-step-ahead forecasts are not yet available.

**Recommendation:** Add a clarifying note distinguishing the two usage modes:

```
# Backtesting mode: vol_hat_dynamic[i] = one-step-ahead forecast for bin i
#   (collected after running the filter through the full day).
#   This matches the paper's evaluation (Section 4.3, Table 4).
#
# Production mode: at each bin i, recompute multi-step-ahead forecasts
#   for bins i..I from the current state, then call this function with
#   only the remaining bins [i..I]. The cumulative_weight must be tracked
#   externally across calls.
```

This does not affect the mathematical formula (Eq 41 is correctly stated)
but helps a developer avoid a subtle production bug.

(Paper, Section 4.3, Equation 41)

### m2. A_stored[1] is set to I_2 but never used

**Spec location:** Algorithm 1, line 157.

**Problem:** At tau=1, A_stored[tau] = I_2 (2x2 identity). This value is
never accessed: the smoother loops from N-1 down to 1, and at tau=1 it
uses A_stored[tau+1] = A_stored[2], not A_stored[1]. The cross-covariance
loop also starts at tau=2, using L[tau-1] = L[1] (not A_stored[1]).

**Recommendation:** This is harmless but slightly misleading. Either:
(a) Add a comment: "A_stored[1] = I_2 (placeholder, not used by smoother)."
(b) Or simply don't store A_stored[1] at all.

Not a correctness issue.

---

## Assessment of Critique 2 Corrections

| Critique 2 Issue | Status in Draft 3 | Verdict |
|---|---|---|
| M1: Cross-covariance regression (smoothed -> filtered) | Fixed: replaced recursive A.20 entirely with non-recursive S&S formula Σ_{τ,τ-1\|N} = Σ_{τ\|N} L_{τ-1}^T | **Correct** -- verified algebraically and against A.21 initialization |
| m1: Q function E_1-E_4 not defined | Fixed: explicit formulas from A.11-A.14, with post-M-step simplification | **Correct** -- verified E_1=N/2, E_2=(N-1)/2, E_3=(T-1)/2, E_4=1 |
| m2: Zero-volume bin handling ambiguous | Fixed: consistent "mark as unobserved" throughout (data prep, data flow, edge case 1) | **Correct** -- tau index preservation clearly explained |
| m3: Sigma_1 subtraction vs direct assignment | Fixed: Sigma_1 = Sigma_smooth[1] with equivalence comment | **Correct** -- numerically stable and mathematically identical |

---

## Detailed Verification of Key Algorithms

### Cross-covariance (non-recursive formula)

The replacement formula Σ_{τ,τ-1|N} = Σ_{τ|N} L_{τ-1}^T is correct. Verified:

1. **At τ = N (base case):** Σ_{N|N} = Σ_{N|N} (smoothed = filtered at last
   time step). The non-recursive formula gives Σ_{N,N-1|N} = Σ_{N|N} L_{N-1}^T.
   Expanding L_{N-1} = Σ_{N-1|N-1} A_{N-1}^T Σ_{N|N-1}^{-1}, this equals
   Σ_{N|N} Σ_{N|N-1}^{-1} A_{N-1} Σ_{N-1|N-1} = (I - K_N C) A_{N-1} Σ_{N-1|N-1},
   which matches the paper's Eq A.21. ✓

2. **Equivalence derivation** (lines 238-248): The algebraic proof is correct.
   The key steps use: (a) substituting the S&S formula at τ+1 into A.20,
   (b) the identity A_τ Σ_{τ|τ} = Σ_{τ+1|τ} L_τ^T from the smoother gain
   definition, and (c) the smoother covariance update. ✓

3. **Loop bounds:** τ = N down to 2, which produces Sigma_cross[2..N].
   This covers all P_cross[tau] needed by the M-step (tau=2..N for a_mu and
   sigma_mu_sq; tau=kI+1 for a_eta and sigma_eta_sq). ✓

### E_1-E_4 formulas

Verified each against the paper:

- **E_1** (lines 398-410): Matches Eq A.11. The simplified form correctly
  factors (y - φ - Cx̂)^2 + CΣC^T = y^2 + CPC^T - 2yCx̂ + φ^2 - 2yφ + 2φCx̂
  using P = Σ_smooth + x̂x̂^T. ✓

- **E_2** (lines 412-416): Matches Eq A.12. Sum over τ=2..N, element (2,2). ✓

- **E_3** (lines 418-423): Matches Eq A.13. D' = {kI+1 : k=1,...,T-1} gives
  T-1 terms covering day-boundary transitions for eta. ✓

- **E_4** (lines 425-428): Matches Eq A.14. Two-term structure: quadratic
  form on mean deviation plus trace of covariance ratio. ✓

- **Post-M-step simplification** (lines 430-451): Correct. Each E_k reduces
  because the M-step zeroes the corresponding gradient. The claim E_4 = 1
  follows from pi_1 = x̂_1 (mean term vanishes) and Σ_1 = Σ_smooth[1]
  (trace term = tr(I_2)/2 = 1). ✓

### M-step updates

All M-step updates verified against Appendix A.3:

| Update | Paper Eq | Spec Location | Verified |
|---|---|---|---|
| pi_1 = x̂_1 | A.32 | Line 319 | ✓ |
| Σ_1 = Σ_smooth[1] | A.33 (simplified) | Line 320 | ✓ |
| a_eta | A.34 | Lines 331-333 | ✓ (T-1 terms) |
| a_mu | A.35 | Lines 336-338 | ✓ |
| σ_η^2 | A.36 | Lines 342-346 | ✓ (T-1 terms) |
| σ_μ^2 | A.37 | Lines 349-353 | ✓ |
| r | A.38 | Lines 357-359 | ✓ (simplified form) |
| φ_i | A.39 | Lines 362-367 | ✓ |

### Robust KF

- Soft-thresholding threshold = λ S_τ / 2, consistent with Eq 33-34 and
  W = 1/S. ✓
- Modified M-step updates for r (Eq 35) and φ (Eq 36) correctly subtract
  z_detected. ✓
- Smoother unchanged in robust mode -- correctly stated with reasoning. ✓

### VWAP

- Static formula (Eq 40): simple normalization. ✓
- Dynamic formula (Eq 41): recursive structure correctly implemented. ✓
  (See m1 above for production vs. backtesting clarification.)

---

## Completeness Assessment

The specification is now complete for implementation:

1. **Algorithmic clarity:** All 6 algorithms (Kalman filter, RTS smoother, EM
   calibration, robust KF, robust EM, VWAP weights) have unambiguous
   pseudocode directly translatable to code. ✓

2. **Parameters:** All 11 parameters documented with recommended values,
   sensitivity, and ranges. Initialization procedure is concrete with
   specific default values. ✓

3. **Calibration:** Full step-by-step procedure including data preparation,
   EM iteration, cross-validation for N and lambda, and rolling
   re-estimation. ✓

4. **Validation:** Quantitative benchmarks from Tables 1, 3, 4 with specific
   per-security examples. 8 sanity checks, 7 edge cases, 7 known
   limitations. ✓

5. **Traceability:** 40+ entries in the paper references table, with
   researcher inferences explicitly marked. ✓

6. **Self-containment:** All formulas are explicit (no "see paper" references
   for anything a developer would need to code). The E_1-E_4 formulas
   (previously external) are now included. ✓

7. **Internal consistency:** Cross-covariance, zero-volume handling, and
   Sigma_1 are now consistent throughout. No contradictions detected. ✓

---

## Verdict

Draft 3 is ready for implementation. All major issues from critiques 1 and
2 have been resolved. The two minor issues identified here (m1: dynamic VWAP
clarification, m2: A_stored[1] placeholder) are cosmetic and do not affect
correctness or implementability. A developer can proceed directly from this
specification.

The adversarial refinement has converged. No further rounds are needed.
