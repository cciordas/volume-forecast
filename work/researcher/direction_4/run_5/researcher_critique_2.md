# Critique of Implementation Specification Draft 2: Dual-Mode Volume Forecast (Raw + Percentage)

**Direction:** 4, **Run:** 5
**Date:** 2026-04-12

## Summary

Draft 2 is a substantial improvement over Draft 1. All 11 issues from Critique 1 have been addressed, most of them well. The M1 surprise baseline mismatch is resolved cleanly (unified to unconditional), the M2 CV placeholder is now fully specified, and the three missing helper functions (Mo1) have complete pseudocode. The Humphery-Jenner provenance flags (Mo4) are thorough and genuinely useful.

Remaining issues are **0 major, 2 moderate, and 4 minor**. The spec is approaching implementation readiness; these are refinements, not blockers.

---

## Resolution Verification

All 11 issues from Critique 1 were claimed resolved. Verification:

| Issue | Resolved? | Notes |
|-------|-----------|-------|
| M1: Surprise baseline mismatch | Yes | Unified to unconditional in both Function 7 (line 904-905) and Function 8 (line 1028-1029). Internally consistent. |
| M2: CV placeholder | Yes | Full expanding-window CV loop in Function 8 (lines 1079-1118). Specifies day-level folds, min 21 training days, MAE metric. |
| Mo1: Missing helpers | Yes | BuildCumVolDistribution (lines 523-557), EvaluateWeights (lines 562-607), compute_hist_pct (lines 673-715) all have complete pseudocode. |
| Mo2: 11-term constraint | Yes | Hard constraint with fallback in Function 3 (lines 284-338). Logic is clear and complete. |
| Mo3: N_hist shared | Yes | Explicit reasoning in compute_hist_pct (lines 677-689) and Parameters section (line 1426). |
| Mo4: Humphery-Jenner provenance | Yes | Provenance notes in Function 7 (lines 848-852), Function 7 Steps 2 and 6, and Known Limitations (line 1589). |
| m1: Inter-day leakage | Yes | Note in Function 5 (lines 412-417). |
| m2: ClassifyRegime_Training | Yes | Direct call to ClassifyRegime with training distribution (Function 5, lines 446-449). |
| m3: Exhibit 6 description | Yes | Revised description (line 1483) accurately reflects non-monotonic median. |
| m4: Day-boundary handling | Yes | Per-segment likelihood recommended as correct approach (lines 188-204), concatenation as pragmatic alternative. |
| m5: No-intercept citation | Yes | Marked as Researcher inference (lines 1121-1130) with self-justifying reasoning. |

---

## Moderate Issues

### Mo1. ClassifyRegime receives current_bin - 1 but cumvol_distribution is indexed by current_bin

**Location:** Function 6 (ForecastRawVolume), line 813 vs. Function 4 (ClassifyRegime), line 373.

**Problem:** In Function 6, ClassifyRegime is called with `current_bin - 1` as the second argument:
```
regime = ClassifyRegime(
    observed_volumes, current_bin - 1,  # bins observed so far
    ...)
```

But inside ClassifyRegime (Function 4), the `current_bin` parameter is used both as a guard check (`IF current_bin < min_regime_bins`) and as the index into the cumulative volume distribution (`cumvol_distribution[current_bin]`). When called with `current_bin - 1`, the cumulative volume computed as `sum(observed_volumes[j] for j in 1..current_bin)` sums bins 1 through current_bin-1, which is correct (those are the observed bins). However, the distribution lookup `cumvol_distribution[current_bin]` retrieves the distribution at bin index current_bin-1, which represents "historical cumulative volumes through bin current_bin-1." This is internally consistent.

BUT: in Function 5 (OptimizeRegimeWeights, line 448), ClassifyRegime is called with `j` (the current bin index) as the second argument, and `observed_up_to_j` contains bins 1 through j:
```
observed_up_to_j = {k: volume_data[stock, bin=k, day=d]
                    for k in 1..j}
regime = ClassifyRegime(observed_up_to_j, j, ...)
```

Here, cumulative volume sums bins 1..j (including the target bin j), and the distribution lookup is `cumvol_distribution[j]`. During training in Function 5, the model is classifying the regime AFTER observing bin j (including j's own volume), which means the regime for bin j's weight lookup reflects information that includes bin j's observation. In contrast, during prediction (Function 6), the regime is classified based on bins observed BEFORE the current forecast target, which is the correct causal direction.

**Impact:** The training/prediction mismatch means the regime assignment in training uses one more bin of information than at prediction time. The regime for predicting bin 10 is based on cumvol through bin 9 (prediction) vs. cumvol through bin 10 (training). This could introduce a small systematic bias in weight optimization: training assigns regimes using "future" information (the actual volume of the bin being forecast).

**Recommendation:** In Function 5, change the ClassifyRegime call to use bins 1..j-1 (observed before bin j) to match prediction-time behavior:
```
observed_up_to_j_minus_1 = {k: volume_data[stock, bin=k, day=d]
                            for k in 1..j-1}
regime = ClassifyRegime(observed_up_to_j_minus_1, j-1, ...)
```
For bin j=1, this produces an empty observed_volumes and current_bin=0, which will hit the `min_regime_bins` guard and return the default regime -- matching prediction behavior.

### Mo2. EvaluateWeights uses model_a.H but should use day-appropriate H

**Location:** Function 5 (OptimizeRegimeWeights), lines 503-506 and helper EvaluateWeights, line 594.

**Problem:** The EvaluateWeights signature includes `H_array` as a parameter (line 565: `H_array`), and the function uses `compute_H_asof()` internally (line 594). This is correct -- it computes H as of each validation day d. However, in the calling code (Function 5, line 503-506), the function is invoked with `H` as one of the arguments:
```
val_mape = EvaluateWeights(
    volume_data, stock, val_start, val_end,
    weights, cumvol_dist, percentiles, n_reg,
    H, interday_models, intraday_model, seasonal_factor)
```

But `H` is a parameter of OptimizeRegimeWeights (line 398), which is the current-date H array. Inside EvaluateWeights, the function actually calls `compute_H_asof()` per bin per day (line 594), ignoring the `H_array` parameter entirely. So `H_array` is a dead parameter -- it is passed but never used.

This is not a correctness bug (the function does the right thing internally), but a dead parameter in the API creates confusion. A developer implementing from this spec might assume `H_array` is used somewhere and wire it incorrectly.

**Recommendation:** Remove `H_array` from EvaluateWeights' signature and from the calling code in Function 5. The function already computes H correctly via `compute_H_asof()`. Alternatively, if the intent was to use the pre-computed H for computational efficiency (avoiding recomputation), document this and actually use it -- but note that for validation days different from reference_date, the pre-computed H would be stale.

---

## Minor Issues

### m1. Intraday ARMA conditioning state initialization ambiguity

**Location:** Function 6 (ForecastRawVolume), lines 783-793.

**Problem:** The conditioning description says "reset AR/MA buffers to post-training state" (line 789). But what is the "post-training state"? After fitting an ARMA model on N_intraday_fit days of data, the AR/MA buffers contain the residuals and observations from the last p/q observations of the last training day. When we reset-and-reprocess for a new day's observed bins, we start from this saved state and then process today's bins.

However, the post-training state reflects the end of the training window's last day, which is (reference_date - 1) at initial fit time but gets staler as days pass (until the next daily re-fit). Between re-fits, the intraday model is re-fit daily (Function 10, line 1267), so this staleness is at most 1 day.

The actual ambiguity is subtler: when conditioning on today's observed bins, should the AR/MA buffers start from zeros (treating each day as a fresh start, matching the per-segment likelihood training assumption), or from the post-training state (carrying overnight context)? The per-segment likelihood training approach (Function 3) explicitly initializes each day's segment to zeros, meaning training assumes no overnight carryover. But if conditioning at prediction time uses the post-training state (which includes the last training day's context), there is a train/predict mismatch in initial conditions.

**Recommendation:** Clarify that conditioning should initialize AR/MA buffers to zeros (matching the per-segment likelihood training assumption). The "post-training state" language is misleading. Replace line 789 with: "reset AR/MA buffers to zeros (matching the per-segment day-boundary treatment used in training)."

### m2. Exhibit 9 bottom-95% for 30-minute bins appears incorrect

**Location:** Validation section, line 1490.

**Problem:** The spec states "30-minute bins: 2.95% median reduction" -- this correctly matches the median column of Exhibit 9. However, looking at the Exhibit 9 values for 30-minute bins: HVWAP = 0.0143, DVWAP = 0.01381. The percentage reduction is (0.0143 - 0.01381) / 0.0143 = 3.43%, not 2.95%. Let me re-check: the paper says 2.95% for 30-min median. Looking at Exhibit 9: the median row shows 0.0143 vs 0.01381 but the paper states 2.95% as the percent reduction. The computation (0.0143 - 0.01381)/0.0143 = 0.0049/0.0143 = 3.43%. But the paper explicitly says "2.95%*" in the table.

Actually, reading more carefully at Exhibit 9, the 30-minute row shows: HVWAP = 0.0143, DVWAP = 0.01381, Percent Reduction = 2.95%. The arithmetic: (0.01430 - 0.01381) / 0.01430 = 0.00049 / 0.01430 = 3.43%. The paper's stated 2.95% does not match its own numbers. This is likely a rounding or transcription error in the original paper.

**Recommendation:** Note the apparent arithmetic inconsistency in Exhibit 9's 30-minute row in a footnote. The spec should cite the paper's stated 2.95% (which it does), but flag that the arithmetic from the table's own HVWAP/DVWAP values yields ~3.43%. This discrepancy is minor and does not affect implementation.

### m3. Regime weight optimization: MSE vs MAPE selection criterion not fully specified

**Location:** Function 5 (OptimizeRegimeWeights), lines 468-479.

**Problem:** The spec describes implementing both MSE and MAPE optimization as a design variant (lines 475-479): "We implement MSE as the primary and MAPE as a variant. If MAPE variant produces lower out-of-sample MAPE, switch." But the pseudocode only shows the MSE optimization (lines 487-500) without any code for the MAPE variant or the switching logic. A developer reading the pseudocode would implement MSE only.

The text mentions the MAPE approach requires "variable transformation (w_raw = exp(w_log)) and Nelder-Mead" (Calibration section, line 1465), but this appears only in the Calibration section, not in the pseudocode. The switching logic ("If MAPE variant produces lower out-of-sample MAPE, switch") has no pseudocode.

**Recommendation:** Either (a) add pseudocode for the MAPE variant and switching logic after the MSE optimization block in Function 5, or (b) simplify by removing the MAPE variant reference from Function 5 and keeping it only as a "future enhancement" note in the Calibration section. Option (b) is recommended to avoid bloating the pseudocode with an untested variant.

### m4. predict_interday helper state management for training-time calls

**Location:** Function 5 helper predict_interday, lines 626-636.

**Problem:** The helper calls `model.forecast(steps=1)[0]` (line 635). During the training loop in Function 5, this is called repeatedly for different days d. But `model.forecast()` depends on the model's internal AR/MA buffer state. For day d, the model should reflect observations through d-1.

The problem is: the Function 5 loop iterates over training days, but the interday model's buffers are not explicitly updated between iterations. The model was fitted on the full `N_interday_fit` window, and `model.forecast(steps=1)` produces one prediction from whatever state the model currently holds. Calling it multiple times without updating state would return the same value each time.

The spec acknowledges the leakage issue (Function 5, lines 412-417) but does not address the mechanical problem of HOW to produce a correct interday forecast for each training day d. The developer needs to either: (a) save and restore model state for each training day, or (b) re-initialize the model's buffers to reflect observations through d-1 before each forecast.

**Recommendation:** Add a note to predict_interday clarifying that for training-time usage, the model's AR/MA buffers must be set to reflect the observation history through day d-1. A simple approach: the helper should accept an explicit observation history and reset the model's buffers accordingly (similar to the intraday ARMA's condition() mechanism). Alternatively, note that during training, the interday forecast for day d can be approximated by: (a) using the model's parameters (phi, theta, constant) with the actual volume observations at bin i for days d-p through d-1 as the AR inputs, and the residuals for those days as the MA inputs.

---

## Verification of Key Changes in Draft 2

| Changed Section | Change | Correct? |
|---|---|---|
| Function 7, Step 3: unconditional baseline | Uses `ForecastRawVolume(model_a, stock, date, current_bin=1, observed_volumes={})` | Yes, consistent with Function 8 |
| Function 8: unconditional baseline in training | Uses `ForecastRawVolume(model_a, stock, d, current_bin=1, observed_volumes={})` | Yes, matches Function 7 |
| Function 8: CV loop | Expanding window, day-level folds, min 21 train days, MAE metric | Yes, fully specified |
| BuildCumVolDistribution | Per-bin CDFs of historical cumulative volumes | Yes, matches ClassifyRegime usage |
| EvaluateWeights | MAPE on validation set with per-day per-bin evaluation | Yes, though H_array is dead (see Mo2) |
| compute_hist_pct | Per-bin average fraction of daily volume, shared N_hist | Yes, with clear rationale |
| 11-term hard constraint | Re-fit with reduced order if combined >= 11 | Yes, correct fallback logic |
| Provenance flags | Throughout Function 7, Function 8, Parameters, Known Limitations | Yes, thorough and accurate |
| No-intercept marking | Researcher inference with self-justifying reasoning | Yes, appropriate |
| Exhibit 6 description | "generally increasing... with significant bin-to-bin variation" | Yes, accurately reflects chart |
| Day-boundary handling | Per-segment likelihood recommended, concatenation as alternative | Yes, correct guidance |

## Overall Assessment

Draft 2 has resolved all major and moderate issues from Critique 1 effectively. The spec is now comprehensive, internally consistent, and implementation-ready for the core algorithms. The remaining issues are refinements:

- **Mo1** (ClassifyRegime training/prediction mismatch) is a subtle consistency issue that could introduce a small regime assignment bias during training. It is easy to fix (change observed_up_to_j to observed_up_to_j-1 in Function 5).
- **Mo2** (dead H_array parameter) is a spec cleanliness issue, not a correctness bug.
- The minor issues (m1-m4) are clarifications and edge-case specifications that a skilled developer could resolve independently, though documenting them would reduce ambiguity.

The spec is ready for implementation after addressing Mo1. The remaining issues can be noted as developer guidance.
