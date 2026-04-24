# Critique of Implementation Specification Draft 1: Dual-Mode Intraday Volume Forecast

## Summary

The draft is well-structured, thoroughly referenced, and demonstrates careful
reading of Satish et al. (2014). Researcher inference items are clearly marked,
the pseudocode is detailed, and the validation section is comprehensive. However,
I identified 5 major issues and 8 minor issues. The major issues involve a
parameter that is defined but never used (creating ambiguity about Component 1),
a train/predict asymmetry in regime assignment, underspecified day-boundary
handling for intraday ARMA, a questionable choice of unconditional surprise
baselines in Model B, and a missing discussion of how the percentage forecasts
maintain sum-to-1 coherence across the full day.

---

## Major Issues

### M1. N_hist parameter defined but never used — Component 1 ambiguity

**Location:** Parameter table (line 880) vs. Function 1 (lines 56-77) and
Function 6 (line 376).

**Problem:** The parameter table defines `N_hist = 21` with the description
"Rolling window for Component 1 historical average (trading days)" and
sensitivity "Medium-high." However, this parameter never appears in any
pseudocode. Function 1 (`compute_seasonal_factors`) uses `N_seasonal = 126`
to compute the values that are subsequently used as Component 1 (H) in
Functions 5 and 6.

This matters because the paper distinguishes two different averaging windows:
- **Deseasonalization:** "dividing by the intraday amount of volume traded in
  that bin over the trailing six months" (p.17 para 5) — this is the 126-day
  window.
- **Historical Window Average:** "a rolling historical average for the volume
  trading in a given 15-minute bin" (p.17 para 3) over "the prior N days"
  (p.16) — this is the tunable N, and Exhibit 1 labels it "Prior 21 days."

These are potentially two different quantities: a 21-day rolling average
(Component 1, H) and a 126-day rolling average (deseasonalization divisor for
the intraday ARMA). The spec conflates them by using `seasonal_factors` for
both purposes.

**Evidence:** Exhibit 1 (p.18) shows three separate inputs: "Next Bin (Prior 21
days)" feeds into "Historical Window," while "Current Bin / Today / 4 Bins
Prior to Current Bin" feeds into "ARMA Intraday." The 21-day window is specific
to Component 1; the 6-month window is for deseasonalization.

**Impact:** If Component 1 should use a 21-day average while deseasonalization
uses a 126-day average, the current pseudocode computes the wrong H. A 126-day
average is much smoother and less responsive to recent volume changes than a
21-day average, which would change the model's behavior significantly.

**Recommendation:** Either (a) add a separate `compute_historical_average`
function using `N_hist` for Component 1, keeping `compute_seasonal_factors`
(N_seasonal=126) only for deseasonalization; or (b) explicitly document that
the same 126-day average is used for both and justify why N_hist is not needed.
The paper is ambiguous, but Exhibit 1 suggests option (a).

---

### M2. Train/predict asymmetry in regime assignment

**Location:** Function 5 (lines 315-316) vs. Function 6 (lines 400-401).

**Problem:** The regime classifier is called with inconsistent semantics
between training (Function 5) and prediction (Function 6):

- **Function 5 (training):** For bin `i` being forecast, calls
  `assign_regime(regime_classifier, i, cumvol)` where `cumvol` is the sum of
  bins 1..(i-1). The first argument `i` is the **target bin**.
- **Function 6 (prediction):** Calls
  `assign_regime(model_a.regime_classifier, current_bin, cumvol)` where
  `current_bin = max(observed_volumes.keys())` and `cumvol` includes bins
  1..current_bin. The first argument is the **last observed bin**.

This creates two asymmetries:
1. **Threshold lookup bin:** Training looks up thresholds for bin `i` (target),
   prediction looks up thresholds for `current_bin` (last observed, which is
   < target_bin). Different bins have different threshold vectors since
   cumulative volume distributions differ by bin.
2. **Cumulative volume inclusion:** Training uses cumvol through (target-1),
   prediction uses cumvol through current_bin. When target_bin = current_bin+1
   (next-bin forecast), these are the same. But for multi-step forecasts
   (target_bin > current_bin+1), prediction's cumvol is strictly less than
   what training used.

**Impact:** The regime weights are optimized under one assignment rule and
applied under a different one. This could cause systematic misassignment,
particularly affecting distant-bin forecasts.

**Recommendation:** Standardize to one convention. The natural choice is to
use the last observed bin and cumulative volume through that bin (consistent
with what is known at prediction time). Update Function 5 to match:
`assign_regime(regime_classifier, i-1, cumvol)` where cumvol is bins 1..(i-1).

---

### M3. Day-boundary handling for intraday ARMA is underspecified

**Location:** Function 3 (lines 146-192), specifically the `day_breaks`
parameter at line 177.

**Problem:** The pseudocode passes `day_breaks=day_boundaries` to `fit_ARMA`,
but `fit_ARMA` is a black-box call to an unspecified ARMA fitting library.
Standard ARMA implementations (e.g., `statsmodels.tsa.arima.model.ARIMA`) do
not support a `day_breaks` parameter. The spec does not explain:

1. How the ARMA likelihood is modified at day boundaries. Does the model treat
   each day as an independent segment with separate initial conditions? Or does
   it zero out AR/MA terms that cross overnight gaps?
2. How the effective sample size `n_eff = N_intraday_fit * (I - max(p, q))`
   (line 170) is derived. If each day is an independent segment of length I,
   the usable observations per day are `I - max(p, q)` (dropping the first
   max(p,q) bins where lags are incomplete). But this formula assumes all days
   contribute equally, which is only true if day boundaries reset the ARMA
   state.
3. How prediction works at the start of each day. If the model state resets at
   day boundaries during fitting, what is the initial state for today's first
   prediction?

**Impact:** A developer who ignores `day_breaks` will fit a single long ARMA
series with spurious overnight lag-1 connections (close-of-day-1 -> open-of-
day-2). A developer who tries to implement it will be forced to make design
decisions not covered by the spec.

**Recommendation:** Specify one of these concrete approaches:
- (a) **Concatenate-and-mask:** Fit on the concatenated series but set AR/MA
  contributions to zero for lags that cross day boundaries. Describe the
  modified likelihood.
- (b) **Independent segments:** Fit each day as a separate short time series
  sharing ARMA coefficients (panel ARMA). State the fitting approach (e.g.,
  pooled MLE across segments).
- (c) **Simple concatenation:** Ignore day boundaries entirely and acknowledge
  the approximation. The overnight gap is a single discontinuity per day in a
  26-point series, and the model may still work acceptably.

Also specify the initial prediction state for bin 1 of a new day (e.g.,
unconditional mean = 1.0 in deseasonalized space).

---

### M4. Unconditional surprise baselines in Model B may be wrong

**Location:** Function 7 (lines 452-456) and Function 8 (lines 566-576).

**Problem:** Both training and prediction compute volume surprises using
**unconditional** (pre-market) raw forecasts:
```
raw_fcst = forecast_raw_volume(model_a, stock, d, target_bin=i,
                                observed_volumes={})
surprise = (actual - raw_fcst) / raw_fcst
```

The spec marks this as Researcher inference and justifies it by saying it
ensures "consistency with prediction time." But this reasoning is circular: at
prediction time (Function 8), the model has access to observed volumes for bins
1..current_bin. The unconditional forecast for early bins (which don't use
intraday ARMA conditioning) will be identical whether observed_volumes is empty
or not. But for later bins, the unconditional forecast ignores valuable
information.

More importantly, the paper describes the Humphery-Jenner (2011) approach as:
"volume surprises — deviations from a naive historical forecast" (summary) and
"training a model on decomposed volume, or departures from a historical average
approach" (p.18-19). The "naive historical forecast" / "historical average"
suggests the surprise baseline is Component 1 (H) alone, not the full Model A
forecast. Using the full Model A unconditional forecast as the baseline means
surprises are smaller (Model A is more accurate), which could weaken the
regression signal.

**Impact:** If the paper intends surprises relative to a simple historical
average (as Humphery-Jenner originally proposed), using the full Model A
unconditional forecast produces a different and potentially weaker signal.

**Recommendation:** Consider two alternatives and document the choice:
- (a) Surprise = (actual - H) / H, where H is the historical average
  (Component 1). This matches Humphery-Jenner's original formulation.
- (b) Surprise = (actual - Model_A_conditioned) / Model_A_conditioned, using
  conditioned forecasts. This is the most informative signal at prediction time.
- (c) Keep the current unconditional approach but strengthen the justification.

The paper says "we could apply our more sophisticated raw volume forecasting
model described previously as the base model from which to compute volume
surprises" (p.19). This supports using Model A as the base, but does not
clarify conditioned vs. unconditional.

---

### M5. Percentage forecasts do not maintain sum-to-1 coherence

**Location:** Function 8 (lines 539-617).

**Problem:** The spec produces percentage forecasts one bin at a time, but does
not ensure they sum to 1.0 over the full day. The only mechanism is the
last-bin special case (line 614: assign all remaining fraction). Several issues
undermine coherence:

1. **Scaled_base recomputation:** Each call recomputes `V_total_est`,
   `observed_hist_frac`, and `scale` from scratch. If `V_total_est` changes
   significantly between calls (as more bins are observed), the scale factors
   are inconsistent across bins. The cumulative sum of pct_hat values may
   overshoot or undershoot 1.0 by the time the last bin is reached.

2. **Deviation clamping:** The `max_deviation` constraint clips the adjusted
   percentage symmetrically around `scaled_base`. After clipping, the
   percentages are no longer guaranteed to be consistent with the scaled
   baseline's sum properties.

3. **No renormalization step:** The spec does not renormalize the forecasted
   percentages for remaining bins to sum to 1.0 minus observed fraction.
   Each bin's forecast is independent, potentially leaving large residuals
   for the last bin.

**Impact:** The last-bin percentage forecast (Step 7) absorbs all cumulative
errors. If earlier bins systematically overpredict, the last bin gets a very
small or even negative fraction. The paper's deviation constraint (10%) and
switch-off rule (80%) partially mitigate this, but the spec does not discuss
whether coherence was actually achieved in practice.

**Recommendation:** Add a discussion of whether percentage coherence matters
for the downstream VWAP algorithm (it may not if the algorithm only uses
next-bin forecasts). If it does matter, add an optional renormalization step
that distributes the remaining fraction proportionally across unforcasted bins
after each prediction.

---

## Minor Issues

### m1. N_hist sensitivity labeled "Medium-high" but parameter is unused

**Location:** Parameter table, line 880.

**Problem:** N_hist has sensitivity "Medium-high" but is never used in any
pseudocode. If it's meant for Component 1, implement it. If not, remove it
from the table to avoid confusion with N_seasonal.

---

### m2. Exhibit 1 "4 Bins Prior to Current Bin" not reflected in spec

**Location:** Function 3 and Function 6.

**Problem:** Exhibit 1 (p.18) shows "4 Bins Prior to Current Bin" feeding into
"ARMA Intraday." This suggests the intraday ARMA prediction conditions on the
4 most recent bins, not all bins 1..current_bin. The spec uses all observed
bins (Function 6, line 389: `FOR j = 1 TO current_bin`). This may be a
diagram simplification (showing p_max_intra=4), but the spec should
acknowledge the discrepancy and explain why using all bins is preferred.

---

### m3. Surprise std dev range "0.005-0.015" lacks source

**Location:** Sanity Check 8, line 1025.

**Problem:** The claim "Std dev should be ~0.005-0.015 for liquid stocks" has
no citation and is not marked as Researcher inference. This appears to be an
unsourced heuristic. Either cite a source or mark as Researcher inference.

---

### m4. compute_evaluation_mape uses conditioned forecasts but training MAPE uses in-sample

**Location:** Function 9, lines 703-722 vs. Function 5.

**Problem:** `compute_evaluation_mape` uses conditioned forecasts (observed
bins 1..(i-1)) for each bin, which is the correct out-of-sample setup. But
Function 5 (`optimize_regime_weights`) also uses conditioned forecasts during
weight optimization. This means the weight optimizer sees the same conditioning
setup as the evaluator — which is fine for consistency but means the training
MAPE in Function 5 is NOT truly in-sample (the intraday ARMA conditions on
data from the same day it's forecasting). The spec should clarify whether this
conditioning constitutes data leakage or is intentional (since conditioning on
earlier-same-day bins is the operational setup).

---

### m5. Nelder-Mead convergence is fragile for 3-parameter optimization

**Location:** Function 5, lines 328-345.

**Problem:** Nelder-Mead with only 3 parameters is reasonable, but the
exp-transformation creates a non-convex landscape where the optimizer can get
stuck. The spec specifies `maxiter: 1000, xatol: 1e-4, fatol: 1e-6` but
does not specify:
- What to do if the optimizer does not converge (use initial equal weights?).
- Whether to use multiple random restarts.
- Whether to check that the result improves over equal weights.

**Recommendation:** Add a post-optimization check: if the optimized MAPE is
worse than equal-weights MAPE, fall back to equal weights. Add 2-3 random
restarts from different initializations (e.g., [1, 0, 0], [0, 1, 0],
[0, 0, 1] in real space).

---

### m6. Blocked CV in Function 7 may have insufficient test blocks

**Location:** Function 7, lines 463-489.

**Problem:** With `N_regression_fit = 63` days and `K = 5` folds,
`block_size = 12` days. Each test block has 12 days * (26 - L_max) bins
of test observations. For small L_max, this is adequate. But the spec hard-
codes K=5 without justification and does not handle the case where
`len(days) % K != 0` (the last fold may have a different size or some days
may be dropped).

**Recommendation:** Specify how remainder days are handled (e.g., assign to
the last fold, or drop them).

---

### m7. VWAP tracking error benchmark numbers should specify conditions

**Location:** Validation section, lines 982-992.

**Problem:** The VWAP tracking error benchmarks (8.74 bps vs 9.62 bps) are
from FlexTRADER simulations with specific conditions: 10% of 30-day ADV order
size, day-long orders, May 2011 data. The spec mentions some of these (lines
985-988) but does not clarify that these are simulation-specific numbers that
a developer should NOT expect to match exactly. The developer's validation
should focus on relative improvement (percentage reduction) rather than
absolute bps levels.

**Recommendation:** Add a note that absolute bps values are simulation-
dependent and only the relative improvement (~9%) is the meaningful benchmark.

---

### m8. Missing specification of inter-day ARMA predict_at and predict_next semantics

**Location:** Function 5 (line 303) and Function 6 (line 382).

**Problem:** Function 5 calls `interday_models[i].predict_at(d)` while
Function 6 calls `interday_models[i].predict_next()`. These are different
interfaces to the ARMA model but their semantics are not defined:
- `predict_at(d)`: Predict the volume for a specific past date d. Is this
  the one-step-ahead forecast made at d-1, or the fitted value at d?
- `predict_next()`: Predict the next unobserved value. What day does this
  correspond to?
- How does `append_observation` (Function 10, line 775) interact with
  `predict_next`?

**Recommendation:** Define a concrete ARMA model interface specifying:
(a) what `predict_at(d)` returns (one-step-ahead forecast using information
through d-1); (b) what `predict_next()` returns (one-step-ahead forecast
for the next day after the last appended observation); (c) whether
`append_observation` updates the model's internal state (Kalman filter
style) or just extends the observation buffer.

---

## Citation Verification

I verified the following citations against the paper:

| Claim | Cited Source | Verified? | Notes |
|-------|-------------|-----------|-------|
| 26 bins per day | p.16 | Yes | "26 such bins in a trading day" |
| ARMA p,q through 5 | p.17 | Yes | "all values of p and q lags through five" |
| AICc from Hurvich & Tsai | p.17 | Yes | "corrected AIC, symbolized by AICc" |
| Deseasonalization 6 months | p.17 | Yes | "trailing six months" |
| Intraday ARMA 1-month fit | p.18 | Yes | "most recent month" |
| AR lags < 5 | p.18 | Yes | "AR lags with a value less than five" |
| < 11 terms | p.18 | Yes | "fewer than 11 terms" |
| 10% deviation | p.24 | Yes | "depart no more than 10%" |
| 80% switch-off | p.24 | Yes | "80% of the day's volume is reached" |
| No-intercept regressions | p.19 | Partial | Spec applies by analogy; paper says this about VWAP-error regressions, not surprise regressions |
| 24% median MAPE reduction | p.20 | Yes | Confirmed on p.20 |
| 29% bottom-95% reduction | p.20 | Yes | Confirmed on p.20 |
| MAD 0.00808 vs 0.00874 | Exhibit 9 | Yes | Confirmed in Exhibit 9 |
| VWAP 8.74 vs 9.62 bps | Exhibit 10 | Yes | Confirmed in Exhibit 10 |
| R^2 = 0.5146, coeff = 220.9 | Exhibit 3 | Yes | Confirmed in Exhibit 3 |
| R^2 = 0.5886, coeff = 454.3 | Exhibit 5 | Yes | Confirmed in Exhibit 5 |
| Paired t-test = 2.34 | Exhibit 10 | Yes | Confirmed in Exhibit 10 |
| "separate method for deviation bounds" | p.19 | Yes | Confirmed |

One citation issue: The no-intercept choice for the surprise regression (line
529) is labeled as applied "by analogy" from the VWAP-error regressions. This
is honest but the analogy is weak — the two regressions serve different purposes
(surprise prediction vs. VWAP error attribution). The spec should note that
omitting the intercept in the surprise regression forces zero adjustment when
all recent surprises are zero, which is the stronger justification.

---

## Positive Observations

1. **Researcher inference transparency:** Every non-paper-sourced decision is
   clearly marked with reasoning. This is exemplary.
2. **FALLBACK sentinel pattern:** A clean approach for handling convergence
   failures that keeps the pseudocode simple.
3. **Purity requirement for intraday ARMA state** (lines 420-426): Critical
   insight that prevents subtle state-mutation bugs. Well-motivated.
4. **Data flow diagram:** The ASCII art training/prediction flow is clear and
   matches the pseudocode structure.
5. **Edge case coverage:** Thorough treatment of half-days, splits, special
   calendar days, and zero-volume bins.
6. **Validation section:** Comprehensive with specific quantitative targets
   from the paper. The sanity checks (1-13) are practical and actionable.

---

## Summary of Recommended Changes

| Issue | Severity | Action |
|-------|----------|--------|
| M1: N_hist unused / Component 1 conflation | Major | Separate historical average (N_hist) from deseasonalization (N_seasonal) |
| M2: Regime assignment train/predict asymmetry | Major | Standardize bin/cumvol convention across Functions 5 and 6 |
| M3: Day-boundary ARMA underspecified | Major | Specify concrete approach (segment, mask, or ignore) |
| M4: Unconditional surprise baselines | Major | Justify or revise; consider H-only or conditioned baselines |
| M5: Percentage sum-to-1 coherence | Major | Document whether coherence matters; add renormalization if needed |
| m1: N_hist in table but unused | Minor | Remove or implement |
| m2: Exhibit 1 "4 bins" discrepancy | Minor | Acknowledge and explain |
| m3: Surprise std dev unsourced | Minor | Mark as Researcher inference |
| m4: Conditioned training MAPE | Minor | Clarify data leakage implications |
| m5: Nelder-Mead fragility | Minor | Add convergence fallback and restarts |
| m6: Blocked CV remainder handling | Minor | Specify remainder-day policy |
| m7: VWAP bps benchmark conditions | Minor | Clarify benchmarks are relative |
| m8: ARMA predict interface | Minor | Define predict_at / predict_next / append_observation semantics |
