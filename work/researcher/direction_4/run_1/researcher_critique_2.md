# Critique of Implementation Specification Draft 2: Dual-Mode Volume Forecast (Raw + Percentage)

## Summary Assessment

Draft 2 is a substantial improvement over draft 1. All 4 major and 8 minor issues from critique 1 have been addressed thoroughly, with detailed discussions, multiple interpretations where the paper is ambiguous, and clear Researcher Inference annotations. The document is now at a quality level where a developer could implement the model correctly. I identify 0 major issues and 5 minor issues. The draft is ready for implementation with minor clarifications.

---

## Resolution of Critique 1 Issues

All 12 issues from critique 1 have been resolved:

| Issue | Status | Assessment |
|-------|--------|------------|
| M1 (Overview conflates results) | Resolved | Overview now cleanly separates Model A (24% MAPE), Model B (7.55% pct error), and VWAP simulation (9.1%). Lines 13-16. |
| M2 (N_interday unspecified) | Resolved | Added N_interday=252 with excellent discussion of Exhibit 1 "Prior 5 days" ambiguity (lines 59-79). Two interpretations given with well-reasoned justification for choosing (b). |
| M3 (Day-boundary handling) | Resolved | Comprehensive discussion (lines 123-164) with three mitigation options, quantification of affected data (~20%), and reconciliation of Exhibit 1 vs. paper text. |
| M4 (V_total_est undefined) | Resolved | Explicitly defined (lines 464-486) with update ordering (Model A first, then Model B). No circular dependency. |
| m1 (MAPE definition) | Resolved | Precise formula in Calibration section (lines 788-792). |
| m2 (AICc incomplete) | Resolved | Full AIC + AICc formulas in pseudocode (lines 93-101). |
| m3 ("fewer than 11 terms") | Resolved | Thoughtful reinterpretation as descriptive rather than prescriptive, based on "As a result" phrasing (lines 172-186). |
| m4 (Regime classification timing) | Resolved | min_regime_bins parameter added (lines 288-296). |
| m5 (Deseasonalization floor) | Resolved | epsilon integrated directly into pseudocode (line 48-51). |
| m6 (Regression spec too vague) | Resolved | Clarified as single/pooled regression with no intercept (lines 373-419). |
| m7 (Renormalization conflicts) | Resolved | Scales baseline rather than post-multiplying adjusted forecast (lines 536-563). |
| m8 (Chen et al. context) | Resolved | Added that Chen et al. claim to outperform Satish et al. (line 810). |

---

## Minor Issues

### m1. V_total_est uses unconditional raw forecasts for remaining bins

**Location:** ForecastVolumePercentage, lines 480-486.

The code calls `ForecastRawVolume(symbol, day_t, current_bin=1, raw_volume_model)`, producing unconditional forecasts (no intraday ARMA conditioning, default "medium" regime). These are used for two purposes:
1. `raw_total` = sum of all raw forecasts (denominator for expected_pct, line 501).
2. `V_total_est` = sum(observed) + sum(raw_forecasts for remaining bins) (denominator for actual_pct, line 502).

Purpose (1) is defensible: expected_pct should reflect the model's pre-day expectation, so unconditional forecasts are appropriate.

Purpose (2) is suboptimal: V_total_est is described as "the best available estimate" of daily total volume (line 467), but it ignores all intraday information for the remaining bins. By the time we're at bin 15, the raw model conditioned on 14 observed bins would produce substantially better forecasts for bins 15-26 than the unconditional pre-day forecast.

**Recommendation:** Use two separate calls:
```
# For expected_pct baseline (unconditional):
unconditional_forecasts = ForecastRawVolume(symbol, day_t, current_bin=1, raw_model)
raw_total = sum(unconditional_forecasts)
expected_pct[j] = unconditional_forecasts[j] / raw_total

# For V_total_est (conditioned on today's data):
conditioned_forecasts = ForecastRawVolume(symbol, day_t, current_bin, raw_model)
V_total_est = sum(observed) + sum(conditioned_forecasts for remaining bins)
```

This adds one extra ForecastRawVolume call per bin but produces a more accurate V_total_est, especially in the second half of the day. The impact is small early in the day (few observed bins, little conditioning advantage) and grows later.

### m2. Deviation constraint is applied relative to unscaled p_hist but forecast uses scaled baseline

**Location:** ForecastVolumePercentage, lines 516-563.

The deviation constraint (line 516-517) clips delta relative to the original `p_hist[next_bin]`:
```
max_delta = max_deviation * p_hist[next_bin]
delta = clip(delta, -max_delta, +max_delta)
```

Then the final forecast (line 563) uses the *scaled* baseline:
```
p_hat = scale * p_hist[next_bin] + delta
```

If `scale` differs substantially from 1.0, the effective deviation relative to the scaled base is `delta / (scale * p_hist[next_bin])`, which can exceed `max_deviation`. For example, if scale = 0.7 (volume running high, less remaining), the effective deviation is `max_deviation / 0.7 = 14.3%` instead of 10%.

**Problem:** The safety constraint intended to limit deviation to 10% of the baseline can be violated when the remaining fraction differs from the remaining historical percentage.

**Recommendation:** Apply the deviation constraint after scaling:
```
scaled_base = scale * p_hist[next_bin]
max_delta = max_deviation * scaled_base
delta = clip(delta, -max_delta, +max_delta)
p_hat = scaled_base + delta
```

This maintains the 10% deviation limit relative to the actual baseline being used. Alternatively, document the current behavior as intentional (the deviation limit is relative to the *historical* base, not the scaled base) and note the effective deviation can differ from max_deviation.

### m3. Regression training window not specified in TrainVolumePercentageModel

**Location:** TrainVolumePercentageModel, line 427.

The loop says `FOR each day d in training_days` but `training_days` is never defined for the percentage model. The DailyUpdate procedure (line 618-620) uses a 252-day (1-year) window for re-estimation. But the initial training in TrainVolumePercentageModel does not specify the window.

Additionally, the training uses the same `raw_volume_model` for all days (line 430: `ForecastRawVolume(symbol, d, bin=1, raw_volume_model)`). The comment at line 429 says "mimicking out-of-sample usage," but this is not implemented -- the raw_volume_model was trained on all training data including future days relative to d. This is a mild lookahead bias: the ARMA parameters and seasonal factors used for raw forecasts on day d were estimated using data through the end of training, not just through day d-1.

**Recommendation:**
1. Specify `training_days = [train_end_date - N_reg_train .. train_end_date - 1]` with `N_reg_train` as a parameter (recommended: 252 days, matching DailyUpdate).
2. Note the lookahead bias explicitly. For practical purposes, the bias is negligible because: (a) ARMA parameters are estimated on 252+ days, so dropping one day has minimal impact; (b) seasonal factors are 6-month averages that barely change day-to-day. A developer wishing to eliminate it entirely could use expanding-window re-estimation, but this multiplies the computational cost by the number of training days.

### m4. Exhibit 6 time-of-day description slightly imprecise

**Location:** Validation > Expected Behavior, line 800.

The draft states: "Error reduction should increase through the day, from roughly 10-15% at 9:30-10:00 to 25-35% at 15:00-15:30."

Looking at Exhibit 6 (p.22), the median reduction at 9:30 starts at approximately 10-12%, not 10-15%. The upper end reaches approximately 30-33% by 15:30, with a notable dip around 10:00-10:30. The bottom-95% curve shows a steadier increase from ~15% to ~35-40%.

**Recommendation:** Refine to: "Median error reduction increases from approximately 10-12% at 9:30 to 30-33% by 15:30, with some non-monotonicity (a dip around 10:00-10:30). The bottom-95% average reduction increases more smoothly from approximately 15% to 35-40%."

### m5. Hysteresis threshold for regime boundary instability

**Location:** Edge Cases, line 856.

The draft suggests: "implement hysteresis: once a regime is selected, require the percentile to cross the boundary by a margin (e.g., 5 percentile points) before switching."

This is good advice, but the hysteresis implementation is not included in the pseudocode. A developer might skip it since it's only in the Edge Cases section. Additionally, the 5-percentile-point margin is arbitrary and undiscussed.

**Recommendation:** Either add hysteresis logic to the ForecastRawVolume pseudocode (in the regime classification block, lines 292-296) or note explicitly that this is an optional enhancement for robustness, not required for the baseline implementation.

---

## Citation Verification

I verified the following new or modified citations in draft 2 against the paper:

| Claim | Cited Source | Verified? | Notes |
|-------|-------------|-----------|-------|
| "Prior 5 days" for ARMA Daily in Exhibit 1 | Exhibit 1, p.18 | Yes | Exact match of diagram labels |
| "4 Bins Prior to Current Bin" for ARMA Intraday | Exhibit 1, p.18 | Yes | Exact match |
| "As a result, we fit each symbol with a dual ARMA model having fewer than 11 terms" | p.18 | Yes | Exact quote; "As a result" phrasing confirmed |
| "AR lags with a value less than five" | p.18 | Yes | Exact match |
| "We compute this model on a rolling basis over the most recent month" | p.17-18 | Yes | Exact match |
| "we could apply our more extensive volume forecasting model" | p.19 | Yes | "could" language confirmed -- aspirational |
| "we perform both regressions without the inclusion of a constant term" | p.19 | Yes | Exact match; refers to validation regressions, not the surprise regression itself |
| "we developed a separate method for computing deviation bounds" | p.19 | Yes | Exact match |
| "e.g., depart no more than 10% away from a historical VWAP curve" | p.24 | Yes | Exact match with "e.g." qualifier |
| Exhibit 9 bottom-95% for 15-min: 0.00986 vs 0.00924, 6.29% | Exhibit 9, p.23 | Yes | Exact match |
| Exhibit 10 paired t-test 2.34, p < 0.01 | Exhibit 10 footnote, p.23 | Yes | Exact match |
| Chen et al. claim to outperform Satish et al. | Chen et al. 2016 summary | Yes | Per paper summary |

All citations check out. The new citations are accurate and well-sourced.

---

## Completeness Assessment

### Present and adequate:
- Model A pseudocode with all four components (historical, inter-day ARMA, intraday ARMA, regime weights)
- Model B pseudocode with surprise regression, constraints, and renormalization
- V_total_est explicitly defined with update ordering
- Day-boundary handling discussion with mitigation options
- Inter-day ARMA training window (N_interday) parameterized with ambiguity discussion
- AICc formula complete (AIC + correction term)
- "Fewer than 11 terms" interpretation with reasoning
- min_regime_bins for early-day regime stability
- Forecast update as reconditioning vs re-estimation (clarified)
- ARMA convergence failure detection criteria
- DailyUpdate procedure with re-estimation frequencies
- Data flow diagram with shapes and types
- Parameter table with 19 parameters, all with sources, ranges, and sensitivity
- Initialization and calibration procedures (5 and 5 steps respectively)
- MAPE computation formula with paper reference
- Validation section with 7 expected behaviors, 11 sanity checks, 10 edge cases
- 8 known limitations
- Paper references table with 39 entries, including Researcher Inference annotations
- Variants section with justification for primary configuration

### Missing or incomplete (minor):
1. **Regression training window** (m3 above) -- not parameterized in TrainVolumePercentageModel.
2. **V_total_est conditioning** (m1 above) -- uses unconditional forecasts for remaining bins.
3. **Deviation constraint vs. scaling interaction** (m2 above) -- constraint applied relative to unscaled base.

---

## Overall Assessment

Draft 2 is a thorough, well-documented implementation specification that faithfully represents the Satish et al. (2014) model. The proposer addressed all 12 issues from critique 1, often going beyond the minimum fix to provide detailed multi-interpretation analyses (e.g., the N_interday discussion, the "fewer than 11 terms" reinterpretation, the day-boundary mitigation options). The Researcher Inference annotations are honest and well-justified.

The 5 minor issues identified in this critique are refinements rather than corrections. None would block implementation. Issues m1-m3 improve accuracy of the inference pipeline; m4-m5 are documentation precision and optional robustness enhancements.

**Recommendation:** This draft is ready for implementation. The minor issues can be addressed in a quick revision or left as developer notes. No further adversarial rounds are needed.
