# Critique of Implementation Specification Draft 1: Dual-Mode Volume Forecast (Raw + Percentage)

**Direction:** 4, **Run:** 5
**Date:** 2026-04-12

## Summary

This is a strong, detailed draft with clear pseudocode, comprehensive parameter documentation, and thorough edge-case coverage. The 10-function decomposition is well-structured and the data flow diagram is excellent. However, I identified **2 major issues, 4 moderate issues, and 5 minor issues** that need attention before a developer can implement confidently.

---

## Major Issues

### M1. Surprise baseline mismatch between training (Function 8) and prediction (Function 7)

**Location:** Function 7 (ForecastVolumePercentage), lines 634-636 vs. Function 8 (TrainPercentageRegression), lines 757-774.

**Problem:** During **prediction** (Function 7, Step 3), surprises are computed using Model A's **unconditional** forecasts:
```
raw_fcst_unconditional = ForecastRawVolume(
    model_a, stock, date, current_bin=1, observed_volumes={})
```
But during **training** (Function 8), surprises are computed using Model A's **dynamic/conditioned** forecasts:
```
raw_fcst = ForecastRawVolume(model_a, stock, d, j, observed_so_far)
```

The comment in Function 8 (line 738-743) explicitly states: "training should use DYNAMIC forecasts (conditioned on bins 1..j-1 for target bin j) to match prediction-time behavior." But prediction-time behavior (Function 7) actually uses **unconditional** forecasts. The stated rationale contradicts the implementation.

**Impact:** Conditioned forecasts are systematically more accurate than unconditional ones (they incorporate observed information), so training-time surprises will be systematically smaller in magnitude than prediction-time surprises. The OLS regression, calibrated on smaller training surprises, may produce coefficients that underfit the larger prediction-time surprises. Alternatively, the deviation constraint masks this, but it undermines the regression's value.

**Recommendation:** Choose one approach and use it consistently:
- **(a) Both unconditional (simpler, internally consistent):** Training and prediction both use market-open (unconditional) forecasts as the surprise baseline. This is computationally cheaper and avoids mismatch.
- **(b) Both conditioned (more informative, paper-aligned):** Training and prediction both use conditioned forecasts. This means Function 7 Step 3 should use `ForecastRawVolume(model_a, stock, date, current_bin=j, observed_volumes_up_to_j-1)` for each observed bin j, not a single unconditional call.

Option (a) is simpler and aligns with the paper's language ("departures from a naive volume forecast model," p.18). Option (b) is more sophisticated but requires matching Function 7 to Function 8's dynamic approach.

### M2. Cross-validation implementation in Function 8 is a placeholder

**Location:** Function 8 (TrainPercentageRegression), lines 806-813.

**Problem:** The time-series cross-validation for selecting the optimal lag count L is described as:
```
cv_errors = []
# ... (standard expanding-window CV implementation)
cv_error = mean(cv_errors)
```

This is not implementable. A developer needs to know:
1. How are day boundaries respected in the expanding-window split? (Pool all bins of a day into the same fold? Or split at the bin level?)
2. What is the minimum number of training days before the first validation fold?
3. What error metric is used for CV (MAE? MSE? MAPE of the percentage deviation?)
4. How many folds? (Expanding window with step size 1 day? 5 days? Fixed number of folds?)
5. Is the OLS re-fit on each training window, or are coefficients carried forward?

**Recommendation:** Provide complete pseudocode for the CV loop. Suggested design:
- Folds: expanding window with 1-day step. Minimum training period: 21 days.
- Each fold trains OLS on all (X, y) pairs from days 1..k, evaluates MAE on day k+1's (X, y) pairs.
- Error metric: MAE of percentage deviation predictions (matches the final evaluation metric).
- Final L selection: L with lowest average MAE across all validation folds.

---

## Moderate Issues

### Mo1. Helper functions called but never defined

**Location:** Function 5 (OptimizeRegimeWeights), lines 333-334 and 395-398.

**Problem:** Two functions are called but have no pseudocode:
- `BuildCumVolDistribution(volume_data, stock, train_start, train_end)` -- builds the historical cumulative volume distribution for regime assignment.
- `EvaluateWeights(volume_data, stock, val_start, val_end, weights, cumvol_dist, percentiles, H, interday_models, intraday_model, seasonal_factor)` -- evaluates weight configuration on validation set.

Additionally, `compute_hist_pct()` is called in Function 10 (ReEstimationSchedule, line 957) but never defined.

**Impact:** A developer cannot implement the regime optimization or re-estimation pipeline without these functions. BuildCumVolDistribution in particular has design decisions: is the distribution built per-bin (a separate CDF for each bin index) or aggregated across all bins? The ClassifyRegime function (Function 4, line 283) uses `cumvol_distribution[current_bin]`, suggesting per-bin distributions, but this needs explicit confirmation and pseudocode.

**Recommendation:** Add pseudocode for all three functions. For BuildCumVolDistribution, clarify:
- It builds per-bin CDFs (sorted arrays of historical cumulative volumes at each bin index).
- Whether cumulative volume is computed from bin 1 through bin i, or only for bin i.
- How many days go into the distribution.

### Mo2. The "fewer than 11 terms" constraint interpretation

**Location:** Function 3 (FitIntraDayARMA), lines 236-248; Parameters section, line 1275.

**Problem:** The paper states (p.18): "As a result, we fit each symbol with a dual ARMA model having fewer than 11 terms." The spec interprets "As a result" as indicating this is an observed outcome of AICc selection, not an imposed constraint, and treats it as a warning threshold.

However, re-reading the paper, "As a result" follows the sentence about using AR lags less than five. The full passage is: "we found that the autoregressive (AR) coefficients quickly decayed, so that we used AR lags with a value less than five. As a result, we fit each symbol with a dual ARMA model having fewer than 11 terms." This reads as: because AR < 5, the total terms (inter-day + intraday) are < 11. The arithmetic supports this: if inter-day is ARMA(4,5) = 10 terms and intraday is ARMA(4,5) = 10 terms, the combined count could exceed 11 easily. But the paper says "fewer than 11 terms" for the **combined** dual model. This implies a binding constraint, not just an observation.

**Impact:** If treated as a soft guardrail, some stocks may end up with > 11 combined terms, potentially overfitting. If treated as a hard constraint, the AICc model selection must be jointly constrained across inter-day and intraday models.

**Recommendation:** Acknowledge the ambiguity more explicitly. Implement as a hard constraint with fallback: after independently selecting inter-day and intraday models via AICc, check if combined terms < 11. If not, reduce the higher-order model (prefer reducing the intraday model, since it's a single model affecting all bins, whereas inter-day models are per-bin). Log the constraint activation.

### Mo3. N_hist window for H and hist_pct: same or different?

**Location:** Function 10 (ReEstimationSchedule), lines 951-952 and 957.

**Problem:** The spec uses `N_hist` for computing both H[i] (Model A's historical average) and hist_pct[i] (Model B's historical percentage curve), but never clarifies whether these should share the same window length. The paper mentions N as "a variable that we shall call N" (p.16) for the historical window, but it's ambiguous whether the same N applies to both raw volume and percentage computations.

The distinction matters: H[i] is the per-bin average raw volume, while hist_pct[i] is the per-bin fraction of daily volume. These serve different purposes (H is a forecast component; hist_pct is a baseline for the percentage model) and may benefit from different window lengths. A longer window for hist_pct could produce a more stable seasonal shape, while a shorter window for H captures recent volume trends.

**Recommendation:** Either (a) explicitly state that N_hist is shared between H and hist_pct, with the reasoning, or (b) introduce a separate parameter `N_hist_pct` for Model B's historical percentage computation, with a recommended value and sensitivity assessment.

### Mo4. Humphery-Jenner (2011) is not in the assigned papers

**Location:** Throughout the Volume Percentage Forecast methodology (Function 7, Function 8); Parameters section (max_deviation, pct_switchoff).

**Problem:** The spec relies heavily on Humphery-Jenner (2011) for the dynamic VWAP framework, deviation bounds, and switch-off mechanism. However, this paper is not among the assigned papers in `papers/`. All information about Humphery-Jenner's approach comes second-hand through Satish et al. (2014, pp.18-19, 24).

Key design decisions attributed to Humphery-Jenner that cannot be verified:
- The rolling regression structure for surprise-based adjustments.
- The 10% deviation limit mechanism.
- The 80% switch-off threshold.
- Whether the regression uses an intercept or not.

**Impact:** A developer implementing Model B has no way to verify whether the spec accurately represents Humphery-Jenner's approach, or whether the spec's interpolations (e.g., the surprise formula, the regression structure) are consistent with the original. Satish et al. explicitly state they "extended his work" (p.19), meaning some details differ from the original.

**Recommendation:** Flag this as a known limitation explicitly in the spec. Where possible, note which Model B details come from Satish et al.'s description of Humphery-Jenner vs. from Satish et al.'s own extensions. The developer should understand which parts are well-grounded in a primary source and which are second-hand interpretations.

---

## Minor Issues

### m1. Inter-day ARMA predict_interday function needs state clarification

**Location:** Function 5 helper `predict_interday()`, lines 423-432.

**Problem:** The function calls `model.forecast(steps=1)[0]`, but during training (Function 5, line 346), it's called for each historical day d. For a correct training-time forecast on day d, the model's internal state (AR/MA buffers) must reflect observations through day d-1 only. But the model was fitted on the entire `N_interday_fit` window. If training days overlap with the fitting window, the model has already "seen" the observation for day d during fitting.

This is an information leakage concern: the inter-day ARMA parameters were estimated using data that includes day d, so the "forecast" for day d is partly in-sample. The spec acknowledges a similar issue for Model B (line 1233, edge case 10) but doesn't address it for Model A's weight optimization in Function 5.

**Recommendation:** Add a note acknowledging this leakage in Function 5. Quantify the expected impact: for a 126-day fitting window, each day contributes ~0.8% of the training data, so parameter sensitivity to any single day is small. Alternatively, note that a fully rigorous approach would use expanding-window ARMA re-estimation during training (at much higher computational cost).

### m2. ClassifyRegime_Training vs ClassifyRegime distinction

**Location:** Function 5 (OptimizeRegimeWeights), line 341-342.

**Problem:** The function calls `ClassifyRegime_Training()` but this is never defined. It's presumably similar to `ClassifyRegime()` (Function 4) but adapted for training (using the training-period cumulative volume distribution). A developer would need to know: is this identical to ClassifyRegime but with a different cumvol_distribution argument? If so, just call ClassifyRegime with the training-period distribution. If there are differences (e.g., no min_regime_bins guard for training), they should be specified.

**Recommendation:** Either (a) replace `ClassifyRegime_Training()` with a direct call to `ClassifyRegime()` using the training cumvol_dist and appropriate arguments, or (b) define ClassifyRegime_Training explicitly.

### m3. Exhibit 6 error reduction profile description is approximate

**Location:** Validation section, line 1157.

**Problem:** The spec states: "Error reduction varies by time of day: ~10% at 9:30 (bin 1) increasing to ~33% at 15:30 (bin 25)." Looking at Exhibit 6, the median line starts around 10-12% at 9:30 but fluctuates significantly through the day (not monotonically increasing). The description "increasing to" implies a monotonic trend, which is not accurate. The median curve has notable dips around 11:00-11:30 and is quite noisy.

**Recommendation:** Revise to: "Error reduction varies by time of day: lowest at market open (~10-12% at 9:30) and generally increasing toward close (~30-33% at 15:30), though with significant bin-to-bin variation. Bottom-95% profile shows a smoother increasing trend from ~13% to ~38%."

### m4. Intraday ARMA day-boundary implementation hint could be more specific

**Location:** Function 3 (FitIntraDayARMA), lines 219-223.

**Problem:** The spec suggests two implementation approaches for day-boundary handling:
1. "use statsmodels SARIMAX with missing='drop' and insert NaN at boundaries"
2. "compute per-segment likelihoods and sum them"

For approach 1, inserting NaN between segments with `missing='drop'` does not correctly reset the ARMA state in statsmodels -- NaN handling in SARIMAX uses the Kalman filter's missing observation logic (which maintains state uncertainty rather than resetting it). This would not achieve the intended "independent segment" behavior.

**Recommendation:** Clarify that approach 2 (per-segment likelihood summation) is the correct implementation for truly independent segments. For approach 1, note that the correct statsmodels mechanism would be to fit separate models and aggregate, or to use a custom likelihood function. Alternatively, fitting a single ARMA model on the concatenated series (ignoring boundaries) is a pragmatic approximation whose impact diminishes as N_intraday_fit grows, and could be noted as an acceptable simplification.

### m5. No-intercept citation application

**Location:** Function 8 (TrainPercentageRegression), lines 818-824.

**Problem:** The spec cites Satish et al. p.19 ("we perform both regressions without the inclusion of a constant term") and applies this to the surprise regression "by analogy." However, re-reading p.19, this quote refers specifically to the VWAP tracking error vs. volume percentage error regressions (Exhibits 2-5), NOT to the surprise regression in Model B. The paper's reasoning is: "a perfect volume percentage forecast should evaluate to zero VWAP tracking error" -- this logic applies to the validation regressions, not necessarily to the surprise regression.

The spec's analogy (zero surprise → zero adjustment) is reasonable and likely correct, but the citation is misleading. A developer might look up p.19 and find it's about a different regression entirely.

**Recommendation:** Mark the no-intercept decision for the surprise regression as "Researcher inference" rather than a paper citation. The reasoning (when surprises are zero, delta should be zero) is sound and self-justifying; it doesn't need to lean on the p.19 citation.

---

## Verification of Key Citations

| Spec Claim | Paper Source | Verified? |
|---|---|---|
| 26 bins per trading day | p.16: "26 such bins in a trading day" | Yes |
| N_hist = 21 from Exhibit 1 | Exhibit 1 (p.18): "Next Bin (Prior 21 days)" | Yes, but annotation only |
| AICc with p,q through 5 | p.17: "all values of p and q lags through five... corrected AIC... AIC_c" | Yes |
| Seasonal factor: trailing 6 months | p.17: "dividing by the intraday amount of volume traded in that bin over the trailing six months" | Yes |
| Intraday ARMA: rolling 1 month | p.18: "rolling basis over the most recent month" | Yes |
| AR lags < 5 for intraday | p.18: "AR lags with a value less than five" | Yes |
| Fewer than 11 terms | p.18: "we fit each symbol with a dual ARMA model having fewer than 11 terms" | Yes (interpretation debatable -- see Mo2) |
| Re-seasonalization | p.18: "we re-seasonalize these forecasts via multiplication" | Yes |
| Regime switching with percentile cutoffs | p.18: "regime switching by training several weight models for different historical volume percentile cutoffs" | Yes |
| Median MAPE reduction 24% | p.20: "we reduce the median volume error by 24%" | Yes |
| Bottom-95% reduction 29% | p.20: "the bottom 95% of the distribution by 29%" | Yes |
| Exhibit 9 percentage results | Exhibit 9 (p.23): HVWAP 0.00874, DVWAP 0.00808, 7.55% reduction (15-min) | Yes |
| Exhibit 10 VWAP results | Exhibit 10 (p.23): 9.62 → 8.74 bps, 9.1% reduction, t=2.34, p<0.01 | Yes |
| 10% deviation limit | p.24: "depart no more than 10% away from a historical VWAP curve" | Yes (via Humphery-Jenner) |
| 80% switch-off | p.24: "once 80% of the day's volume is reached, return to a historical approach" | Yes (via Humphery-Jenner) |
| No-intercept regressions | p.19: "we perform both regressions without the inclusion of a constant term" | Verified but applies to VWAP-error regressions, not surprise regression (see m5) |
| Custom curves for special days | p.18: "custom curves for special calendar days" | Yes |
| ARMAX not recommended | p.18: "scant historical data represented by the special cases" | Yes |

## Overall Assessment

The draft is comprehensive and well-organized. The 10-function decomposition provides a clear implementation roadmap, and the data flow diagram is production-quality. Citation accuracy is high -- I verified all major claims against the paper and found them correct or well-reasoned.

The two major issues (M1: surprise baseline mismatch, M2: CV placeholder) are the primary blockers. M1 introduces an internal inconsistency that could produce subtly wrong regression coefficients, while M2 leaves a critical model selection step unimplementable. The moderate issues (Mo1-Mo4) would force a developer to make design decisions without guidance.

After resolving these issues, this spec would be ready for implementation.
