# Critique of Implementation Specification Draft 2: Direction 4, Run 6

## Summary

Draft 2 is a substantial improvement. All 14 issues from critique 1 (5 major,
9 minor) have been addressed:

- M1 (joint term constraint): Correctly implemented via max(interday_term_counts)
  budget in Function 4. The design tension between per-bin inter-day models and
  shared intraday model is resolved and well-documented.
- M2 (no-intercept citation): Corrected. Now explicitly marked as Researcher
  inference with proper reasoning.
- M3 (weight normalization): Resolved via softmax parameterization. The
  minimize_mape helper is well-specified.
- M4 (self-updating limits): Function 9a provides an adaptive calibration
  procedure with clear rationale.
- M5 (Model A-based surprise variant): Both naive and sophisticated variants
  are now specified in Functions 8, 9, 10, and 11.
- All 9 minor issues (m1-m9) resolved cleanly: four-component framing,
  Exhibit 1 label explanations, split sanity checks, percentile_rank
  definition, ARMA state initialization, look-ahead acknowledgement,
  Humphery-Jenner citation correction, bin 1 initialization.

The remaining issues are fewer and less severe: **1 major** and **5 minor**.
The spec is approaching implementation-readiness.

---

## Major Issues

### M1. Sophisticated variant: training/prediction baseline mismatch for observed bins

**Spec location:** Function 11 (lines 1220-1235) vs. Function 10 helper
generate_model_a_training_forecasts (lines 1134-1156).

**Problem:** During training, `generate_model_a_training_forecasts` computes
Model A forecasts for ALL bins (including bins that would have been "observed"
in a live setting). The surprise for each bin is then:

    surprise[d, i] = actual_pct[d, i] - model_a_forecast_pct[d, i]

This captures genuine bin-level prediction error — the difference between what
actually happened and what Model A predicted.

During live prediction (Function 11, lines 1220-1235), the Model A-based
baseline for observed bins uses ACTUAL volumes, not Model A forecasts:

    FOR j = 1 TO I:
        IF j <= current_bin:
            model_a_baseline_pct.append(observed_volumes[j] / total_forecast)
        ELSE:
            model_a_baseline_pct.append(raw_forecasts[j] / total_forecast)

In Function 9, the surprise for observed bin j is then:

    surprise[j] = actual_pct[j] - base[j]
                = observed_volumes[j] / estimated_total - observed_volumes[j] / total_forecast

This reduces to:

    surprise[j] = observed_volumes[j] * (1/estimated_total - 1/total_forecast)

The surprise no longer captures bin-level deviations — it only captures the
difference between two estimates of total daily volume. For observed bins, this
signal is qualitatively different from what the regression was trained on.

If estimated_total and total_forecast are similar (which they often will be),
surprises for observed bins will be near-zero regardless of how much the actual
bin percentage deviated from Model A's prediction. The regression was trained
on non-trivial bin-level surprises but receives near-zero inputs at prediction
time, producing near-zero adjustments — effectively negating Model B's
advantage.

**Impact:** The sophisticated variant would underperform the naive variant in
practice despite using a better baseline, because the live-prediction surprise
signal is degraded.

**Fix required:** For the sophisticated variant during live prediction, the
baseline for observed bins should use Model A's FORECASTED percentages (what
Model A would have predicted before observing the actual volume), not the
actual observed volumes:

```
IF model_state.params.use_model_a_baseline:
    # Build baseline entirely from Model A forecasts (pre-observation)
    total_forecast = sum(raw_forecasts[1 : I+1])
    IF total_forecast > 0:
        model_a_baseline_pct = [raw_forecasts[j] / total_forecast FOR j = 1 TO I]
    ELSE:
        model_a_baseline_pct = None
```

This requires that raw_forecasts for already-observed bins reflect what Model A
WOULD HAVE predicted (before seeing the actual data). Function 11 currently
overwrites raw_forecasts at each iteration. The fix would need to store the
pre-observation Model A forecasts computed at the start of the day (when
current_bin = 0), and use those as the baseline throughout the day.

This also means `estimated_total` in Function 9 should use the Model A-based
remaining percentages (from baseline_pct) rather than hist_pct when in
sophisticated mode. See minor issue m4 below.

---

## Minor Issues

### m1. predict_next() idempotency not explicitly stated

**Spec location:** ARMA Model Interface (lines 219-257), Function 6 (line 640),
Function 11 (lines 1206-1214).

**Problem:** In Function 11, the inner loop calls `predict_raw_volume` for every
remaining target_bin at each iteration of the outer loop. Each call to
predict_raw_volume invokes `interday_models[target_bin].predict_next()`.
For any given target_bin, this means `predict_next()` is called once per
iteration of the outer loop until the target_bin is reached.

If `predict_next()` modifies internal state (e.g., advances a time counter or
pushes a forecast into the observation buffer), this would corrupt the model.
The current interface description implies predict_next() is a pure query (it
says "based on... the observation history stored in the model"), but this is
not stated explicitly. By contrast, `append_observation()` clearly modifies
state ("Extends the model's observation buffer and updates the AR/MA recursion
state").

**Fix:** Add an explicit note to predict_next(): "This is a pure query that does
NOT modify the model's internal state. It can be called multiple times and will
return the same value until append_observation() is called."

---

### m2. Function 9a switch-off calibration produces earlier-than-base thresholds

**Spec location:** Function 9a, lines 987-1004.

**Problem:** The calibration logic finds the bin where median cumulative fraction
first crosses `base_switchoff_threshold` (e.g., 0.80), then sets the actual
switchoff_threshold to the median cumulative fraction at the PREVIOUS bin
(switchoff_bin - 1). Since the previous bin's median cumulative fraction is
necessarily BELOW 0.80 (otherwise that bin would have been selected as the
crossover point), this produces a threshold below the base value.

For example, if median cumulative fractions are [... 0.73, 0.76, 0.82, ...]
at bins [19, 20, 21, ...], the crossover is at bin 21, and the threshold is
set to 0.76 (bin 20's median). This means the switch-off fires at 76%
cumulative volume instead of the intended 80%, making the model revert to
historical percentages earlier and for more bins than intended.

The intent seems to be "switch off approximately when the stock typically
reaches the base threshold," but the logic systematically biases downward.

**Fix:** Either:
(a) Use the crossover bin's median directly:
    `switchoff_threshold = median(cum_fractions_by_bin[switchoff_bin])`, or
(b) Interpolate between the two bins to find the actual threshold that
    corresponds to base_switchoff_threshold for this stock, or
(c) Simply use the base_switchoff_threshold directly (0.80) as the adaptive
    mechanism primarily benefits the deviation limit, not the switch-off.

The current clamp `[0.70, 0.95]` does not prevent this issue since 0.76 is
within range.

---

### m3. generate_model_a_training_forecasts uses median regime for all bins

**Spec location:** Function 10 helper, lines 1150-1153.

**Problem:** The comment says "Use median regime for simplicity (we don't have
live cumulative volume during historical reconstruction)." However, during
training, historical cumulative volumes ARE available (they're in
volume_history). Function 5 already reconstructs historical cumulative volume
percentiles for regime assignment (lines 456-462). The same approach could be
used here.

Using the median regime for all bins means the Model A forecasts used as the
surprise baseline do not reflect the regime-dependent weighting that would
have been applied in live prediction. This creates a second training/prediction
mismatch (in addition to M1) that could reduce the sophisticated variant's
effectiveness.

**Fix:** Reconstruct the regime for each (day, bin) using historical cumulative
volume percentiles, mirroring Function 5's approach:

```
IF i == 1:
    regime = assign_regime(0.5, regime_thresholds)
ELSE:
    cum_vol = sum(volume_history[d, 1:i])
    hist_cum_vols = [sum(volume_history[dd, 1:i])
                     FOR dd IN trailing N_hist days before d]
    pctile = percentile_rank(cum_vol, hist_cum_vols)
    regime = assign_regime(pctile, regime_thresholds)
```

This adds computation but eliminates the systematic bias. Alternatively,
acknowledge this simplification's impact in the Known Limitations section.

---

### m4. remaining_pct in Function 9 hardcoded to hist_pct in sophisticated mode

**Spec location:** Function 9, lines 851-855.

**Problem:** The `remaining_pct` and `estimated_total` calculations always use
`hist_pct`:

```
remaining_pct = sum(hist_pct[current_bin+1 : I+1])
estimated_total = cum_vol_today / (1.0 - remaining_pct)
```

In sophisticated mode, the baseline percentages come from Model A, not
hist_pct. Using hist_pct for estimated_total while using Model A-based
percentages for surprises is inconsistent. If Model A predicts a different
volume distribution than hist_pct, the estimated total will be off, and the
actual_pct values will be distorted.

**Fix:** When `baseline_pct IS NOT None`, compute remaining_pct from
baseline_pct instead:

```
base = baseline_pct IF baseline_pct IS NOT None ELSE hist_pct
remaining_pct = sum(base[current_bin+1 : I+1])
```

This ensures the total-volume estimate is consistent with the baseline used
for surprise computation.

---

### m5. Sanity check 15 notation is confusing

**Spec location:** Sanity Check 15, lines 1569-1572.

**Problem:** The check says: "interday_term_counts[i] + intraday_model.term_count
< 11 (= max_dual_arma_terms + 1)." The parenthetical "(= max_dual_arma_terms
+ 1)" is meant to show that 11 = 10 + 1, but reads as defining
max_dual_arma_terms + 1 = 11. Since max_dual_arma_terms is defined as 10 in
the parameter table (the maximum allowed, i.e., "fewer than 11"), the
constraint is equivalently `<= max_dual_arma_terms`. The current notation
mixes the paper's phrasing ("fewer than 11") with the parameter naming in a
way that could confuse a developer.

**Fix:** Rewrite as: "For every bin i, verify that
interday_term_counts[i] + intraday_model.term_count <= max_dual_arma_terms.
This enforces the paper's 'fewer than 11 terms' constraint."

---

## Positive Observations

1. **Comprehensive citation cleanup:** The no-intercept citation correction
   (M2) is handled thoroughly — the false attribution is removed, the correct
   reference for what p.19 actually describes is noted, and the Researcher
   inference justification is clear and well-reasoned.

2. **Strong adaptive calibration:** Function 9a's approach to self-updating
   limits is creative and well-documented. The 95th percentile approach for
   deviation limits and the clamp to [0.5x, 2.0x] of the base value is a
   sensible design. The explicit acknowledgement that Humphery-Jenner's
   original procedure is unknown (Known Limitation 9) is honest and helpful.

3. **Joint term budget well-resolved:** The design decision to use
   max(interday_term_counts) for the intraday budget is cleanly implemented,
   with clear handling of the edge case where all inter-day models are FALLBACK
   (term count 0) and the case where the budget is exhausted (budget < 2).

4. **Sophisticated variant fully threaded:** The Model A-based surprise
   variant is now specified end-to-end through Functions 8, 9, 10, 11, and
   the generate_model_a_training_forecasts helper. The live-prediction
   integration (M1 above notwithstanding) shows thoughtful engineering.

5. **Excellent Researcher inference tracking:** The consolidated list of 28
   inference items is comprehensive and each is marked inline. This is a
   significant improvement over draft 1's 17 items, reflecting the additional
   adaptive mechanisms.

6. **Look-ahead bias handled maturely:** The Function 5 comment block
   (lines 409-423) provides a well-reasoned justification for the in-sample
   ARMA approximation, with three supporting arguments. This is far better
   than simply ignoring the issue.

---

## Summary of Required Changes

| # | Severity | Section | Issue |
|---|----------|---------|-------|
| M1 | Major | Function 11 + 9 | Sophisticated variant baseline uses actual volumes for observed bins, creating near-zero surprises — train/predict mismatch |
| m1 | Minor | ARMA Interface | predict_next() idempotency not explicitly stated |
| m2 | Minor | Function 9a | Switch-off calibration logic systematically produces below-base thresholds |
| m3 | Minor | Function 10 helper | generate_model_a_training_forecasts uses median regime, could reconstruct actual |
| m4 | Minor | Function 9 | remaining_pct hardcoded to hist_pct even in sophisticated mode |
| m5 | Minor | Sanity Check 15 | Confusing notation mixing paper phrasing with parameter naming |
