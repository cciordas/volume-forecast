# Critique of Implementation Specification Draft 4: Dual-Mode Volume Forecast (Raw + Percentage)

**Direction:** 4, **Run:** 5
**Date:** 2026-04-12

## Summary

Draft 4 addressed all 3 issues from Critique 3 (0 major, 1 moderate, 2 minor). All three are resolved correctly and completely. One trivial residual text inconsistency remains from the MAPE variant cleanup.

Remaining issues: **0 major, 0 moderate, 1 minor**. The spec is implementation-ready.

---

## Resolution Verification

All 3 issues from Critique 3 were claimed resolved. Verification:

| Issue | Resolved? | Notes |
|-------|-----------|-------|
| Mo1: predict_interday MA buffer reconstruction circularity | Yes | Lines 675-711: Sequential burn-in processing replaces the undefined `model.predict_from_params()`. MA buffer initialized to zeros, then processed forward through `burn_in + q` days using full AR+MA prediction at each step. `burn_in = max(2*q, 10)` is reasonable for invertible MA processes (exponential decay of initialization error). The inner loop correctly computes one-step-ahead predictions using both AR history (fetched from volume_data) and the evolving MA buffer, then computes residuals and shifts the buffer. For q=0 (pure AR), the MA loop is a no-op and the function degenerates correctly to AR-only prediction. The final forecast at lines 713-721 uses the reconstructed buffers. Clean, implementable solution. |
| m1: Calibration section MAPE variant still active | Yes | Line 1554 now reads "Variant (future enhancement): MAPE minimization..." with conditional language "If implemented, switch to MAPE variant only if...". Consistent with Function 5 pseudocode (lines 496-498) which also describes MAPE as a "future enhancement." |
| m2: Day-index type ambiguity | Yes | Lines 75-76 add a clear convention: "All date arithmetic in this specification uses trading-day indices. The expression `day_d - N` refers to N trading days before day_d. Non-trading days (weekends, holidays) are not included in the index space. The `is_full_trading_day()` predicate filters half-day trading sessions (which have only 13 bins instead of 26) from full-day sessions; it does not filter weekends or holidays, which are already absent from the trading-day index." This resolves the ambiguity globally for all range expressions throughout the spec. |

---

## Minor Issues

### m1. Residual "both MSE and MAPE" language in Function 5 docstring

**Location:** Function 5 (OptimizeRegimeWeights), line 426.

**Problem:** The function docstring contains:
```
[Researcher inference: the paper does not specify the loss function,
 constraint set, or optimizer for weight optimization. We implement
 both MSE and MAPE objectives (see design decision below).]
```

The phrase "We implement both MSE and MAPE objectives" is inconsistent with the rest of the spec, which now consistently describes MAPE as a future enhancement (Function 5 pseudocode lines 496-498, Calibration section line 1554). The docstring implies both are implemented; the pseudocode implements only MSE.

**Impact:** Trivial. A developer reading the pseudocode would see only MSE is implemented. The docstring contradiction is cosmetic.

**Recommendation:** Change to: "We implement MSE as the primary objective; a MAPE-based alternative is noted as a future enhancement."

---

## Overall Assessment

Draft 4 has resolved all issues accumulated across four rounds of critique. The specification is now:

- **Algorithmically complete:** All 10 functions and 6 helpers have full pseudocode. No undefined functions, no placeholder steps, no circular dependencies.
- **Internally consistent:** Train/predict symmetry is maintained throughout (regime classification uses causal bins, surprise baselines are unconditional, ARMA buffers initialize to zeros, MAPE variant is consistently a future enhancement except for the trivial m1 above).
- **Well-cited:** Every algorithmic step traces to a specific paper section or is explicitly marked as "Researcher inference" with reasoning.
- **Implementation-ready:** A developer can translate each function directly to code. Edge cases are enumerated (13 cases), sanity checks are specific and testable (18 checks), and parameter sensitivity guidance is provided.

The single remaining issue (m1) is a one-line text edit in a docstring comment. It does not affect implementability. **The spec is ready for finalization.**
