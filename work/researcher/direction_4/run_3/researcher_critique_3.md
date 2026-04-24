# Critique of Implementation Specification Draft 3: Dual-Mode Intraday Volume Forecast

## Summary

Draft 3 is a mature, implementation-ready specification. All 2 medium and 3 minor issues from critique 2 have been addressed comprehensively. The additions -- Function 6b (ConditionIntraDayARMA) with the reset-and-reprocess design, the dynamic forecast caching strategy in Function 8, the multi-step ARMA degradation note, the early-bin padding discussion, and the expected-percentage denominator discussion -- are all well-reasoned, clearly written, and properly labeled as Researcher inference where appropriate.

The remaining issues are minor and concern implementability details that could cause a developer to make suboptimal (but not incorrect) implementation choices. None require structural changes.

**Issue count: 0 major, 0 medium, 3 minor.**

---

## Assessment of Critique 2 Revisions

All 5 issues from critique 2 are resolved:

| Issue | Resolution Quality | Notes |
|-------|-------------------|-------|
| Med-1 (Intraday ARMA state conditioning) | Excellent | Function 6b with explicit reset-and-reprocess pseudocode. Both design alternatives (stateless vs. incremental) discussed. Training state preservation requirement noted. |
| Med-2 (Training forecast ambiguity) | Excellent | Dynamic forecast caching strategy in Function 8 with computational cost analysis. Clear justification for matching prediction-time behavior. |
| m-new-1 (Multi-step ARMA degradation) | Excellent | Note added in Function 6 (lines 399-414) and new sanity check #13 with specific convergence test. |
| m-new-2 (Early-bin padding) | Good | Design choice discussion in Function 8 (lines 711-729). Both alternatives noted with recommendation. |
| m-new-3 (Expected percentage denominator) | Good | Discussion added in Function 8 (lines 694-708). Correctly identified as second-order effect. |

---

## Minor Issues

### m-new-1. `model.forecast(steps=N)` purity requirement not specified

**Spec section:** Function 6 (ForecastRawVolume), lines 394-395; Function 6b (ConditionIntraDayARMA).

**Problem:** Function 6 calls `intraday_model.forecast(steps=steps_ahead)` after conditioning. Standard ARMA multi-step forecasting works by iteratively using each intermediate forecast as the next "observation" in the AR buffer and setting future residuals to zero in the MA buffer. This internal recursion could either:

(a) Modify the model's `ar_buffer` and `ma_buffer` in place (stateful), or
(b) Work on internal copies of the buffers, leaving the model's state unchanged (pure/stateless).

The spec does not state which behavior is required. In practice, Function 6b's reset-and-reprocess design makes the choice self-healing -- the next call to ConditionIntraDayARMA resets the buffers regardless. However, the developer must know that `forecast()` should be a pure function that does not mutate model state. If implemented as stateful, then calling `ForecastRawVolume(s, t, 5, 3)` (2 steps ahead) followed by `ForecastRawVolume(s, t, 10, 3)` (7 steps ahead) would produce an incorrect second forecast if the reset-and-reprocess step were ever removed or optimized away (e.g., by hoisting conditioning outside the loop -- see m-new-2).

**Recommendation:** Add a single sentence in Function 6 or Function 6b specifying that `model.forecast(steps=N)` must be a pure function that does not modify the model's persistent AR/MA buffers. It should operate on internal copies.

---

### m-new-2. Redundant intraday ARMA conditioning in DailyOrchestration loop

**Spec section:** Function 6 (ForecastRawVolume) lines 388-392, Function 9 (DailyOrchestration) lines 812-815.

**Problem:** In DailyOrchestration, the inner loop calls ForecastRawVolume for every remaining bin:

```
for i in (current_bin + 1)..I:
    V_hat = ForecastRawVolume(s, t, i, current_bin)
```

Each call to ForecastRawVolume internally calls `ConditionIntraDayARMA(intraday_model, deseas_observed)` with the same `deseas_observed` array (since `current_bin` is constant across the inner loop). With the reset-and-reprocess design, this re-conditions the model identically (I - current_bin) times per outer loop iteration, for a total of sum_{cb=1}^{I-1} (I - cb) * cb = O(I^3) work across the day (trivial for I=26, but conceptually wasteful).

More importantly, this pattern obscures the actual data flow: the conditioning is a per-current_bin operation, not a per-target-bin operation. A developer reading the pseudocode might not realize the redundancy and implement it literally, then later try to "optimize" by removing the redundant calls -- which would require understanding the forecast() purity requirement from m-new-1.

Additionally, Function 7 (ForecastVolumePercentage) also calls ForecastRawVolume internally (lines 527 and 586), potentially at different `current_bin` values within a single call. This means the intraday model's state is overwritten multiple times within ForecastVolumePercentage. The final state after ForecastVolumePercentage returns is conditioned on `lag_bin - 1` observations (from the last surprise computation), not on `current_bin` observations. This does not cause a bug (the next conditioning call resets state), but it makes the execution order of model state changes non-obvious.

**Recommendation:** Add a note in Function 9 suggesting that the developer should hoist the conditioning step outside the inner loop:

```
# Condition intraday model ONCE for this current_bin
deseas_observed = [volume[s, t, j] / seasonal_factor[j] for j in 1..current_bin]
ConditionIntraDayARMA(intraday_model, deseas_observed)

# Then forecast all remaining bins (conditioning is already done)
for i in (current_bin + 1)..I:
    V_hat = ForecastRawVolume_no_condition(s, t, i, current_bin)
```

Alternatively, keep the current pseudocode (correct by construction due to reset-and-reprocess) and add a comment noting the redundancy for the developer's awareness.

---

### m-new-3. Structural mismatch in surprise denominator between training and prediction

**Spec section:** Function 7 (ForecastVolumePercentage), lines 579-587; Function 8 (TrainPercentageRegression), lines 750-766.

**Problem:** The train/predict denominator discussions in draft 3 (Functions 7 and 8) address two sources of mismatch: (1) the actual-percentage denominator (daily_total vs. V_total_est), and (2) the expected-percentage denominator (sum-of-all-forecasts vs. observed+remaining). However, there is a third structural difference that is not discussed: whether the actual and expected percentages within a single surprise use the **same** or **different** denominators.

**In prediction (Function 7, lines 585-587):**
```
actual_pct = observed_volumes[lag_bin] / V_total_est
expected_pct = ForecastRawVolume(s, t, lag_bin, lag_bin - 1) / V_total_est
surprise = actual_pct - expected_pct
```
Both actual and expected use the **same** denominator (V_total_est at current_bin). The surprise simplifies to `(actual - forecast) / V_total_est`, which is proportional to the raw volume error.

**In training (Function 8, lines 756-766):**
```
actual_pct_lag = volume[s, d, lag_bin] / daily_total
expected_pct_lag = raw_forecast_cache[d][lag_cb][lag_bin] / lag_forecast_total
surprise = actual_pct_lag - expected_pct_lag
```
Actual uses `daily_total` while expected uses `lag_forecast_total` (computed at lag_cb). These are **different** denominators. The surprise does NOT simplify to a ratio of (actual - forecast) over a common denominator.

The same structural asymmetry exists for the target variable (lines 736-747): `actual_pct_d_i` uses `daily_total` while `expected_pct_d_i` uses `forecast_total_d`.

This means the training regression learns coefficients from surprise values computed as differences of fractions with non-matching denominators, while at prediction time the coefficients are applied to surprises computed as differences of fractions with a common denominator. The impact is small (daily_total and lag_forecast_total converge for later bins, and the regression coefficients are small), but it is a distinct source of train/predict inconsistency not covered by the existing denominator discussions.

**Recommendation:** Add a brief note to the denominator discussion in Function 8 acknowledging this structural asymmetry. Two options:
- (a) Accept it (current approach). Note that the mismatch is bounded because lag_forecast_total approximates daily_total when the raw model is well-calibrated.
- (b) Use the same denominator for both terms within each surprise. In training, either use daily_total for both, or use lag_forecast_total for both. Option (b) is trivial to implement and would make the surprise formula structurally identical between training and prediction.

---

## Citation Verification (Spot Checks on New Content)

I spot-checked the following new content in draft 3:

| Claim | Spec Citation | Verified? | Notes |
|-------|-------------|-----------|-------|
| Function 6b: "Current Bin / 4 Bins Prior to Current Bin" | Exhibit 1 | Yes | Caption matches |
| Reset-and-reprocess is mathematically equivalent to incremental | Researcher inference | Yes | Correct: sequential processing of the same observations in the same order produces identical buffer states |
| O(current_bin^2) total work across a day | Researcher inference | Yes | Sum of 1+2+...+26 = 351 total observations processed |
| Dynamic training cost: 252 * 26 * 26 = ~170K calls | Researcher inference | Yes | Arithmetic checks out |
| Training state preservation in FitIntraDayARMA | Researcher inference | N/A | Not verifiable from paper; logically necessary for the reset-and-reprocess design |
| Multi-step ARMA convergence to unconditional mean | Researcher inference | Yes | Standard ARMA property; MA contribution is zero after q steps, AR contribution converges geometrically |
| Sanity check #13: within 5% of 1.0 for 25 steps ahead | Researcher inference | Reasonable | For ARMA(2,2) with stationary coefficients, 25 steps is far beyond the convergence horizon |
| Sanity check #14: buffer consistency test | Researcher inference | Yes | Follows directly from the deterministic reset-and-reprocess logic |

All new content verified. No misrepresentations found.

---

## Overall Assessment

Draft 3 is an excellent implementation specification that has matured through three rounds of refinement. All 17 issues raised across critiques 1 and 2 (5 major + 2 medium + 10 minor) have been thoroughly addressed. The document demonstrates careful attention to implementability, with explicit pseudocode for all key functions, clear design decision blocks with alternatives, proper citation labeling (paper-sourced vs. Researcher inference), and comprehensive validation criteria.

The 3 remaining minor issues are refinements that improve developer experience but do not risk incorrect implementations. The forecast() purity requirement (m-new-1) is the most actionable -- a single sentence addition. The other two (redundant conditioning and surprise denominator structure) are informational notes.

**This draft is ready for implementation.** The remaining issues can be addressed with brief annotations (a few sentences total) and do not require any structural or algorithmic changes.
