# Critique of Implementation Specification Draft 3: Dual-Mode Intraday Volume Forecast

## Summary

Draft 3 is a strong revision that resolves all 6 issues from Critique 2
convincingly. The percentage-space surprise reformulation (M1) is the most
significant improvement — it brings Model B into full domain consistency with
Humphery-Jenner's original framework and eliminates the deviation-clamp signal
loss that would have rendered the surprise regression useless. The rolling H_d
fix (M2) and the separate validation hist_avg (m1) are clean and well-reasoned.

The spec is now at a high level of maturity. I identified 1 medium issue and 3
minor issues. The medium issue is a concrete implementability gap in the
pseudocode where a function receives insufficient data for its internal
computation. The minor issues are a subtle domain-mixing question in the
surprise application, a missed re-optimization opportunity in the grid search,
and a precision improvement for the no-intercept validity discussion.

None of these issues would prevent a competent developer from implementing the
model correctly — the medium issue would likely be caught during implementation.
But fixing them would improve the spec's reliability as a standalone document.

---

## Medium Issues

### M1. volume_history slicing in Function 9 is too narrow for Function 5's rolling H_d

**Location:** Function 9, lines 1059-1062 (call to optimize_regime_weights) vs.
Function 5, line 489 (rolling H_d computation).

**Problem:** Function 9 calls optimize_regime_weights with
`volume_history[train_days]` as the first argument:

```
weights = optimize_regime_weights(
    volume_history[train_days], params.N_hist, seasonal_factors,
    interday_models, intraday_model, regime_classifier,
    len(train_days), params.min_volume_floor)
```

Inside Function 5, the rolling H_d computation calls:

```
FOR d IN trailing N_weight_train days:
    H_d = compute_historical_average(volume_history[:d], N_hist)
```

`compute_historical_average(volume_history[:d], N_hist)` needs the N_hist (21)
trading days ending at day d. For the earliest training day d_0, this requires
21 days of data BEFORE d_0. But `volume_history[train_days]` contains only the
N_weight_train (63) training days — it does not include the 21 days preceding
the first training day.

**Consequence:** A developer following the pseudocode literally would either:
- Get an index error when trying to access data before the first training day.
- Silently compute H_d from fewer than 21 days for the first ~21 training days,
  producing progressively less accurate historical averages.

The same issue applies to the regime grid search context (lines 1054-1062),
where `train_days` is an even smaller subset (N_weight_train minus the 21-day
validation split), making the problem worse — up to 21 + 21 = 42 days of
pre-context are needed.

**Impact:** Medium. The spec's computational cost note (lines 575-581) correctly
identifies the data requirement as "N_hist + N_weight_train days" (= 84 days),
and Edge Case 10 (lines 1544-1550) mentions this minimum. But the pseudocode
in Function 9 contradicts these notes by passing a narrower slice.

**Recommendation:** Pass the full volume history (or at least a slice including
N_hist days before the first training day) to optimize_regime_weights:

```
# Option A: pass full history, let Function 5 slice internally
weights = optimize_regime_weights(
    volume_history[:train_end_date], params.N_hist, ...)

# Option B: pass explicit extended slice
extended_start = first_train_day - N_hist
weights = optimize_regime_weights(
    volume_history[extended_start:train_end_date], params.N_hist, ...)
```

Also update the Function 5 signature documentation to note that volume_history
must extend at least N_hist days before the first training day.

---

## Minor Issues

### m1. Delta added to scaled_base without matching the scale adjustment

**Location:** Function 8, lines 937-939.

**Problem:** The surprise application computes:

```
scaled_base = scale * pct_model.hist_pct[next_bin]
adjusted = scaled_base + delta
```

Here, `scaled_base` is a scaled quantity (hist_pct adjusted for the remaining
volume fraction), but `delta` is an unscaled quantity (a predicted
percentage-point departure from unscaled hist_pct, since the regression was
trained on unscaled participation-rate surprises).

When scale != 1.0, this mixes two domains:
- scaled_base is in "conditional remaining-fraction space"
- delta is in "full-day participation-rate space"

**Quantitative impact:** For scale = 0.85 (a day running 15% below historical
volume through the current bin), hist_pct = 0.038, delta = 0.004:
- Current formulation: adjusted = 0.85 * 0.038 + 0.004 = 0.0363
- Scaled formulation: adjusted = 0.85 * (0.038 + 0.004) = 0.0357
- Difference: 0.0006, which is ~15% of the delta value

The difference is (scale - 1) * delta, bounded by |scale - 1| * max_delta. For
typical scale values (0.8-1.2) and max_delta ~ 0.004, the error is up to
~0.0008 — small in absolute terms but comparable to the regression signal.

**Recommendation:** Consider whether the formulation should be:

```
adjusted = scale * (hist_pct[next_bin] + delta)
max_delta = max_deviation * hist_pct[next_bin]    # clamp before scaling
delta = clip(delta, -max_delta, max_delta)
adjusted = scale * (hist_pct[next_bin] + delta)
```

This treats the surprise regression as predicting the departure from the
full-day participation rate, and then scales the entire prediction to match the
remaining volume fraction. If the current (unscaled delta) formulation is
intentional, document the reasoning — e.g., "delta is applied after scaling
because the surprise captures intraday momentum that should not be compressed
by the volume-level adjustment."

---

### m2. Grid search does not re-optimize weights on full window after regime selection

**Location:** Function 9, lines 1078-1091.

**Problem:** The regime grid search evaluates candidate regime counts by:
1. Splitting the training window: train_days (N_weight_train - 21 days) and
   val_days (last 21 days).
2. Optimizing weights on train_days for each candidate.
3. Selecting the candidate with lowest validation MAPE.

After selection, the final model uses the weights from the grid search (line
1080: `n_reg, regime_classifier, weights = best_config`). These weights were
trained on a reduced window — N_weight_train minus 21 validation days (i.e.,
~42 days instead of 63).

Standard practice in model selection is to use the validation split only for
hyperparameter selection (here, regime count), then re-fit the model on the
full training data with the selected hyperparameter. The current approach
wastes 21 days of training data for the final weight optimization.

**Impact:** Minor. The practical effect is modest — 42 vs. 63 training days for
Nelder-Mead weight optimization. But for stocks with trending volume, more
recent data is more relevant, and the validation days (which are the most
recent 21 days) are excluded from weight training.

**Recommendation:** After selecting the best regime count, re-optimize weights
on the full N_weight_train window:

```
# After grid search:
n_reg = best_config.n_reg
# Rebuild classifier and re-optimize on full window
final_classifier = build_regime_classifier(
    volume_history[:train_end_date], params.N_regime_window, n_reg)
final_weights = optimize_regime_weights(
    volume_history[:train_end_date], params.N_hist, seasonal_factors,
    interday_models, intraday_model, final_classifier,
    params.N_weight_train, params.min_volume_floor)
```

---

### m3. No-intercept validity note overstates the risk of nonzero mean surprise

**Location:** Lines 853-862 (Function 7 no-intercept validity requirement) and
Sanity Check 9 (lines 1464-1473).

**Problem:** The no-intercept validity note warns: "If hist_pct were computed from
a substantially different period... mean surprise could be nonzero, biasing
the slope coefficients." And Sanity Check 9 says to check if "|mean| > 0.001."

However, in the current design, hist_pct is computed from the exact same
N_regression_fit window used for surprise computation (Function 7, Step 1 and
Step 2 both iterate over "trailing N_regression_fit days"). Since hist_pct[i]
is by definition the mean of actual_pct[d, i] over those days, the mean
surprise for each bin is exactly zero by construction:

```
mean(surprise[d, i] for d in training_days)
  = mean(actual_pct[d, i] - hist_pct[i] for d in training_days)
  = mean(actual_pct[d, i]) - hist_pct[i]
  = hist_pct[i] - hist_pct[i]
  = 0
```

The warning is only relevant in production between re-estimations, when the
model is applied to new days where actual_pct may drift from the frozen
hist_pct. This distinction is worth stating explicitly.

**Recommendation:** Revise the note to:

"The no-intercept assumption requires that mean surprise is approximately zero.
At training time, this holds exactly by construction (hist_pct is the mean of
actual_pct over the training window). Between re-estimations, hist_pct is
frozen while actual participation rates may drift, potentially introducing a
nonzero mean in prediction-time surprises. The 21-day re-estimation interval
limits this drift. If mean prediction-time surprise deviates significantly from
zero (e.g., |mean| > 0.001 computed over the last N days of live predictions),
consider triggering early re-estimation."

---

## Citation Verification

I re-verified the key citations that changed or were added in Draft 3:

| Claim | Cited Source | Verified? | Notes |
|-------|-------------|-----------|-------|
| Percentage-space surprise = Humphery-Jenner's formulation | p.24 | Yes | "departures from a historical average approach" in VWAP context confirmed |
| "volume surprises — deviations from a naive historical forecast" | p.24 (Humphery-Jenner description) | Yes | Paper describes HJ's approach; "naive historical forecast" is the VWAP curve |
| "both regressions without inclusion of a constant term" | p.19 | Yes | Confirmed: refers specifically to VWAP-error validation regressions, not surprise regression. Spec correctly notes the analogy is Researcher inference. |
| Rolling H = "rolling historical average" | p.17 para 1 | Yes | "a rolling historical average for the volume trading in a given 15-minute bin" — "rolling" supports time-varying computation |
| Deviation clamp 10% | p.24 | Yes | "depart no more than 10% away from a historical VWAP curve" confirmed |

No citation errors found. All new claims accurately reflect the paper.

---

## Critique 2 Issue Resolution Assessment

All 6 issues from Critique 2 were addressed:

| Issue | Resolution Quality | Notes |
|-------|-------------------|-------|
| M1: Domain mismatch (surprise regression) | Excellent | Clean percentage-space reformulation throughout Functions 7 and 8. Domain consistency now verified end-to-end. Three alternative formulations documented for completeness. |
| M2: Static hist_avg across training | Excellent | Rolling H_d per training day in Function 5. Computational cost analysis and edge case documentation thorough. Introduced M1 (slicing gap) in this critique. |
| m1: Validation hist_avg from val period | Good | Separate val_hist_avg computed before validation period. Clean fix. |
| m2: Sanity check 9 std dev | Resolved | Updated to ~0.005-0.015 for percentage-space surprises. |
| m3: No-intercept justification | Good | Verification note added. Slightly overstates risk (see m3 in this critique). |
| m4: Zero-default for missing surprises | Resolved | Documented assumption with clear guidance for illiquid stocks. |

---

## Positive Observations

1. **Percentage-space surprise reformulation:** The Draft 3 M1 fix is exemplary.
   The rationale section (lines 805-843) clearly explains why percentage-space
   surprises are correct, documents all four alternatives (including the
   deprecated Draft 2 approach), and shows the quantitative domain consistency:
   surprise std dev ~0.005-0.015, delta ~0.001-0.005, scaled_base ~0.038, all
   in the same order of magnitude. The deviation clamp now operates on
   comparable quantities.

2. **Train/predict consistency note:** Lines 951-965 transparently document the
   inherent training vs. prediction denominator mismatch (exact total_vol_d vs.
   estimated V_total_est). The analysis of how this diminishes through the day
   and the note about the deviation clamp as a safety bound show mature
   understanding of the operational constraints.

3. **Known Limitation 9 (new):** The addition of the train/predict surprise
   denominator mismatch as an explicit known limitation (lines 1607-1612) is
   good practice — it prevents a future developer from treating this as a bug.

4. **Edge Case 10 update:** The rolling H_d data requirement is correctly stated
   (N_hist + N_weight_train = 84 days), even though the pseudocode has the
   slicing inconsistency noted in M1.

5. **Draft maturity:** The spec is now at ~1670 lines with 11 well-structured
   functions, comprehensive parameter documentation, 14 sanity checks, 12 edge
   cases, and 9 known limitations. The paper reference table is thorough, and
   every Researcher inference is explicitly marked. This is a high-quality
   implementation specification.

---

## Summary of Recommended Changes

| Issue | Severity | Action |
|-------|----------|--------|
| M1: volume_history slicing too narrow for rolling H_d | Medium | Pass extended slice or full history to optimize_regime_weights |
| m1: Delta scaling mismatch with scaled_base | Minor | Scale delta with scale factor, or document the reasoning for not doing so |
| m2: Grid search weights not re-optimized on full window | Minor | Re-fit weights on full N_weight_train after regime count selection |
| m3: No-intercept validity note overstates risk | Minor | Clarify that mean surprise is exactly zero at training time; risk is only during production drift |
