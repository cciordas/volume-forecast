# Critique of Implementation Specification Draft 3: Dual-Mode Volume Forecast (Raw + Percentage)

**Direction:** 4, **Run:** 5
**Date:** 2026-04-12

## Summary

Draft 3 has resolved all 6 issues from Critique 2. Five are resolved cleanly; one (m3, MAPE variant) is partially resolved -- the pseudocode is correct but a contradictory description remains in the Calibration section. One new moderate issue emerged from the predict_interday helper added in this draft (MA buffer reconstruction circularity). Two minor issues remain.

Remaining issues: **0 major, 1 moderate, 2 minor**. The spec is implementation-ready after addressing the moderate issue with a small clarification. The minor issues are developer-guidance quality improvements.

---

## Resolution Verification

All 6 issues from Critique 2 were claimed resolved. Verification:

| Issue | Resolved? | Notes |
|-------|-----------|-------|
| Mo1: ClassifyRegime training/prediction mismatch | Yes | Function 5 (lines 452-461) now uses `observed_before_j = {k: ... for k in 1..j-1}` and passes `j - 1` to ClassifyRegime. EvaluateWeights (lines 594-600) identically uses `observed_before_j` and `j - 1`. Both match Function 6's prediction-time call (line 863: `current_bin - 1`). Internally consistent. |
| Mo2: Dead H_array parameter | Yes | `H_array` removed from EvaluateWeights signature (line 572-575) and from calling code (lines 513-516). Clean removal. |
| m1: ARMA conditioning zeros | Yes | Line 831-832 now reads "initialize AR/MA buffers to zeros (matching the per-segment likelihood training assumption)." "Post-training state" language fully removed. Consistent with Function 3's per-segment training (line 256: "Initialize AR buffer to zeros, MA buffer to zeros"). |
| m2: Exhibit 9 footnote | Yes | Line 1540 includes parenthetical: "note: the arithmetic from Exhibit 9's own HVWAP=0.0143 and DVWAP=0.01381 yields ~3.43%, suggesting a rounding or transcription error in the original paper." Correctly cites the paper's stated 2.95%. |
| m3: MAPE variant removed from pseudocode | Partial | Function 5 pseudocode (lines 486-489) correctly relegates MAPE to "future enhancement." However, Calibration section (line 1515) still contains active switching language (see m1 below). |
| m4: predict_interday state management | Yes | New pseudocode (lines 656-683) provides complete buffer reconstruction. However, the MA buffer computation has a circularity issue (see Mo1 below). |

---

## Moderate Issues

### Mo1. predict_interday MA buffer reconstruction uses undefined function and has circular dependency

**Location:** Helper predict_interday, lines 666-672.

**Problem:** The MA buffer reconstruction computes residuals as:
```
fitted_prev = model.predict_from_params(ar_history_asof=d_prev)
ma_history.append(actual_prev - fitted_prev)
```

Two issues:

1. **Undefined function:** `model.predict_from_params(ar_history_asof=d_prev)` is not defined anywhere in the spec. Its semantics are ambiguous: does it compute the one-step-ahead prediction using only AR terms (ignoring MA), or does it include MA terms?

2. **Circular dependency (if MA terms are included):** To compute the fitted value at day d_prev for an ARMA(p,q) model, the function needs both the AR buffer (observations before d_prev) and the MA buffer (residuals before d_prev). But the MA buffer at d_prev depends on fitted values at even earlier days, which themselves require MA buffers, creating a recursive dependency that extends back to the start of the observation history. The code iterates over only q days `(day_d - q)..(day_d - 1)`, which is insufficient to bootstrap the MA state.

The AR buffer reconstruction (lines 660-663) is correct because AR terms depend only on observed volumes, not on past residuals.

**Impact:** A developer implementing this function would encounter either an undefined function call or, if they implement it with MA terms, incorrect residual estimates due to the missing MA state initialization. For pure AR models (q=0), this is a non-issue. For ARMA models with q > 0 (which are common -- the spec allows q up to 5), the MA buffer will be incorrect.

**Recommendation:** Replace the MA buffer reconstruction with sequential processing from a burn-in window. Specifically:

```
# Reconstruct MA buffers by sequential processing from a burn-in point.
# Start from (day_d - q - burn_in) with MA buffers initialized to zeros,
# and process observations forward. After burn_in steps, the MA state
# converges to the true state (burn-in artifact decays exponentially
# for invertible MA processes). burn_in = max(2*q, 10) is sufficient.
#
# For each day d_prev in (day_d - q - burn_in)..(day_d - 1):
#     ar_vals = [volume for recent p trading days before d_prev]
#     pred = constant + sum(phi_k * ar_vals[-(k+1)]) + sum(theta_k * ma_buf[-(k+1)])
#     resid = actual[d_prev] - pred
#     shift ma_buf: append resid, drop oldest if > q
#
# Use the final ma_buf state for the one-step-ahead forecast.
```

Alternatively, for simplicity, compute fitted values using AR terms only (ignoring MA contributions). This is a reasonable approximation and avoids the circularity entirely: `fitted_prev = constant + sum(phi_k * volume[bin_i, d_prev - k] for k in 1..p)`. Document this as an AR-only approximation for residual computation during training.

---

## Minor Issues

### m1. Calibration section still describes MAPE variant as active

**Location:** Calibration section, line 1515.

**Problem:** The Calibration section (line 1515) reads: "Variant: MAPE minimization with variable transformation (w_raw = exp(w_log)) and Nelder-Mead. Switch to MAPE variant if it produces lower out-of-sample MAPE."

This describes the MAPE variant as an active component of the system with an explicit switching mechanism. However, Function 5's pseudocode (lines 486-489) relegates it to a "future enhancement": "A MAPE-based alternative... could be explored as a future enhancement -- see Calibration section."

These two descriptions contradict each other. A developer reading Function 5 would skip the MAPE variant; a developer reading the Calibration section would implement it with switching logic.

**Recommendation:** Align the Calibration section with Function 5. Replace the active description with a future-enhancement note:

"Variant (future enhancement): MAPE minimization with variable transformation (w_raw = exp(w_log)) and Nelder-Mead could be explored as an alternative. If implemented, switch to MAPE variant only if it produces lower out-of-sample MAPE."

### m2. Day-index type ambiguity in range expressions

**Location:** Throughout the spec, including predict_interday (lines 660-663), compute_H_asof (lines 628-630), ComputeSeasonalFactors (lines 86-88), and Function 5 training loops.

**Problem:** Range expressions like `(day_d - p)..(day_d - 1)` with `is_full_trading_day()` filtering are used throughout. The semantic interpretation depends on whether day variables represent calendar dates or trading-day indices:

- **If calendar dates:** `(day_d - 5)..(day_d - 1)` is 5 calendar days. After filtering weekends and holidays, this yields ~3 trading days (e.g., from a Monday, looking back 5 calendar days includes Saturday and Sunday). For p=5, the AR buffer would consistently have fewer than p entries.

- **If trading-day indices:** `(day_d - 5)..(day_d - 1)` is 5 trading days. `is_full_trading_day()` only filters half-days. The AR buffer would have close to p entries (minus occasional half-days).

The spec never clarifies which convention is used. The `is_full_trading_day()` predicate suggests calendar dates (since trading-day indices would already exclude non-trading days), but the parameter values (N_hist=21 meaning "Prior 21 days" per Exhibit 1) suggest trading days (~1 month of trading).

**Impact:** If calendar dates are intended, every range expression that expects N trading days must use a wider calendar window (e.g., `N * 7/5` with margin). If trading-day indices are intended, the `is_full_trading_day()` filter should be documented as filtering only half-days, not weekends/holidays. A single clarifying statement would resolve this for the entire spec.

**Recommendation:** Add a brief note to the Data Flow or Overview section clarifying the convention. For example: "All date arithmetic in this specification uses trading-day indices. The expression `day_d - N` refers to N trading days before day_d. Non-trading days (weekends, holidays) are not included in the index space. The `is_full_trading_day()` predicate filters half-day trading sessions (13 bins) from full-day sessions (26 bins)."

---

## Overall Assessment

Draft 3 has resolved all issues from Critiques 1 and 2 effectively, with one partial resolution (MAPE variant text inconsistency) and one new issue introduced by the predict_interday buffer reconstruction added in this draft.

The spec is now comprehensive and internally consistent across its core algorithms:
- **Model A** (Functions 1-6): Fully specified with all helpers, consistent train/predict regime classification, correct ARMA conditioning, and complete weight optimization.
- **Model B** (Functions 7-8): Fully specified with consistent unconditional surprise baselines, complete CV loop, and clear provenance flags.
- **Orchestration** (Functions 9-10): Complete daily workflow and re-estimation schedule.
- **Parameters, Validation, Edge Cases**: Thorough and well-cited with 18 sanity checks and 13 edge cases.

The remaining issues are:
- **Mo1** (predict_interday MA circularity): A real implementability gap in a helper function. Easy to fix with either sequential burn-in processing or an AR-only approximation for residual computation. This is the only issue that could cause a developer to get stuck.
- **m1** (Calibration MAPE text): A residual inconsistency from the m3 resolution. Text-only fix.
- **m2** (day-index convention): A clarification that applies globally. One sentence resolves it.

**The spec is implementation-ready.** A skilled developer could resolve all three remaining issues independently, but documenting them eliminates ambiguity.
