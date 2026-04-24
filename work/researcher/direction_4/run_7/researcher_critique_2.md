# Critique of Implementation Specification Draft 2: Dual-Mode Intraday Volume Forecast

## Summary

Draft 2 is a substantial improvement over Draft 1. All 5 major and 8 minor
issues from Critique 1 have been addressed, most of them convincingly. The
separation of hist_avg (N_hist=21) from seasonal_factors (N_seasonal=126),
the regime assignment standardization, the independent-segment intraday ARMA
specification, and the multi-restart Nelder-Mead are all well-executed. The
revision notes are thorough and transparent about what changed and why.

However, the revision introduced a new significant issue (M1 below) and
exposed an existing one that was latent in Draft 1 (M2 below). I identified 2
major issues and 4 minor issues total. The major issues involve a domain
mismatch between the surprise regression and its application in percentage
forecasting, and a static hist_avg applied across all training days.

---

## Major Issues

### M1. Units/domain mismatch between surprise regression and percentage application

**Location:** Function 7 (lines 681-689) vs. Function 8 (lines 847-849, 864).

**Problem:** The surprise regression is trained and applied in raw-volume-
surprise space, but its output (delta) is added as a level adjustment to a
volume percentage. These quantities are in different domains:

- **Surprise definition** (Function 7, line 689):
  `surprise = (actual - H) / H`
  where actual and H are raw share counts. This produces a dimensionless
  ratio in the range ~[-0.5, +2.0] for typical stocks, with std dev ~0.3-0.6
  (per Sanity Check 9).

- **Application** (Function 8, lines 864):
  `adjusted = scaled_base + delta`
  where scaled_base is a volume percentage (~0.015-0.08 for a typical bin).
  delta = dot(beta, lag_vector) is a predicted raw-volume surprise (same
  domain as the training targets: magnitude ~0.05-0.3).

Adding a raw-volume surprise (magnitude ~0.1) to a volume percentage
(magnitude ~0.04) produces a nonsensical result (0.14 when the correct
answer should be ~0.04). The deviation clamp (max_delta = 0.10 * 0.04 =
0.004) absorbs the entire regression signal, rendering the surprise
regression effectively useless.

**Quantitative demonstration:** For an average bin with hist_pct ≈ 1/26 ≈
0.038, scale ≈ 1.0:
- scaled_base ≈ 0.038
- max_delta = 0.10 * 0.038 = 0.0038
- Typical delta from regression ≈ beta_1 * surprise_lag_1 ≈ 0.3 * 0.3 = 0.09
- After clamp: delta is clipped to 0.0038, losing 96% of the signal
- The regression coefficients essentially don't matter

**Evidence:** The relationship between raw-volume surprise and percentage
surprise is approximately: surprise_pct ≈ hist_pct * surprise_raw. Since
hist_pct ≈ 1/26, percentage surprises are ~25x smaller than raw-volume
surprises. The Humphery-Jenner (2011) dynamic VWAP framework works with
volume percentages directly ("volume surprises — deviations from a naive
historical forecast" where the forecast IS a percentage/VWAP curve), not
raw volumes.

**Impact:** This effectively disables Model B's dynamic adjustment for all
but the most extreme cases. The paper reports a 7.55% MAD reduction from the
dynamic approach (Exhibit 9), which cannot be achieved if the regression
signal is clamped away.

**Recommendation:** Redefine surprises in percentage space to maintain domain
consistency. Three options:

- **(a) Percentage-space surprise (recommended):**
  ```
  actual_pct = volume_history[d, i] / total_volume[d]
  surprise = actual_pct - hist_pct[i]
  ```
  Then delta (a predicted percentage-point departure) can be added directly
  to scaled_base. This matches Humphery-Jenner's original formulation where
  surprises are departures of actual participation rates from historical
  participation rates.

- **(b) Multiplicative application:**
  ```
  adjusted = scaled_base * (1 + delta)
  ```
  Keep raw-volume surprises but apply multiplicatively. The deviation clamp
  becomes |delta| <= max_deviation, which is 0.10 — allowing the regression
  signal through. This preserves the current surprise definition and baseline
  choice.

- **(c) Scaled additive application:**
  ```
  adjusted = scaled_base + delta * hist_pct[next_bin]
  ```
  Scale the raw-volume surprise by hist_pct to convert to percentage space
  before adding. Mathematically equivalent to (a) if the regression
  coefficients are the same.

Option (a) is cleanest and most consistent with Humphery-Jenner. Option (b)
requires the least change to the current spec. The choice should be
documented with justification.

**Note:** This issue also existed in Draft 1 (which used unconditional Model A
forecasts as the surprise base, but still in raw-volume space). Critique 1's
M4 focused on the choice of baseline, not the domain mismatch. The M4
resolution (switching to H) did not introduce this issue — it was latent.

---

### M2. Static hist_avg applied across entire training window introduces systematic bias

**Location:** Function 9 (lines 938-939) vs. Functions 5 (line 490) and 7
(line 684).

**Problem:** hist_avg is computed once at train_end_date (the 21-day average
ending at train_end_date) and then used as the H component for ALL training
days in the weight optimization (N_weight_train = 63 days) and surprise
regression (N_regression_fit = 63 days). But the paper describes H as a
"rolling historical average" (p.17 para 1), meaning it should be recomputed
for each training day.

For a training day d that is 63 days before train_end_date, the correct H
would be the 21-day average ending at d. The spec instead uses the 21-day
average ending at train_end_date, which is a fundamentally different quantity.
For stocks with trending volume (common in practice — volume can trend up
30-50% over a quarter due to earnings, news flow, or market regime changes),
this introduces systematic bias:

- **In weight optimization (Function 5):** H is too high for early training
  days of uptrending stocks (and too low for downtrending). This biases the
  optimizer toward lower w_H for uptrending stocks and higher w_H for
  downtrending stocks, misallocating weight from a component that would
  actually be useful with the correct time-varying H.

- **In surprise regression (Function 7):** Surprises computed with the wrong H
  are systematically biased. For uptrending stocks, early-day surprises are
  systematically negative (actual < future_H), distorting the regression
  coefficients.

**Edge case 10 assessment:** The spec acknowledges a related issue: "The bias
is negligible because the 21-day rolling average changes minimally when
dropping one day." But the issue is not dropping one day — it is using an H
from 63 days in the future. Over 63 trading days (~3 calendar months), a
21-day rolling average can change substantially. The claim of negligible
bias is not supported for the full training window span.

**Impact:** The weight optimization and surprise regression are trained on
biased component values and surprise signals. The magnitude depends on
volume trend strength — minimal for stable-volume stocks, potentially large
for trending stocks. Since the paper's universe is the top 500 by dollar
volume (which includes high-growth and event-driven names), this affects a
meaningful fraction of the universe.

**Recommendation:** Recompute hist_avg as a rolling quantity for each training
day. This requires changing Functions 5 and 7 to compute H on the fly:

```
# In Function 5, inside the day loop:
FOR d IN trailing N_weight_train days:
    # Compute time-varying H for day d
    H_d = compute_historical_average(volume_history[:d], N_hist)
    ...
    H = H_d[i]
```

The computational cost is moderate: one 21-day average computation per
training day (63 total), each over 26 bins. This is O(63 * 21 * 26) ≈ 34K
operations — negligible relative to the ARMA fitting.

Alternatively, if computational simplicity is preferred, acknowledge the
bias explicitly and document it as a design trade-off (static H for
simplicity, at the cost of accuracy for trending stocks).

---

## Minor Issues

### m1. Regime grid search validation uses hist_avg computed from validation period

**Location:** Function 9 (lines 938-939, 962-963).

**Problem:** The regime grid search splits training data into train_days and
val_days (last 21 days). hist_avg is computed from the N_hist=21 days before
train_end_date, which is exactly val_days. This means the validation MAPE
evaluation uses an H component computed from the validation data itself,
making H artificially accurate for validation days.

This biases the regime selection toward configurations that give more weight
to H (since H is unfairly good during validation). In production, H is
computed from the 21 days before the prediction day — it never includes the
prediction day's data.

**Recommendation:** Compute a separate hist_avg for the validation evaluation
using data ending before val_days start:
```
val_hist_avg = compute_historical_average(volume_history[:val_start], N_hist)
```
Or use a larger validation gap (e.g., 5-day buffer between hist_avg window
and validation period).

---

### m2. Sanity Check 9 surprise std dev estimate inconsistent with units mismatch

**Location:** Sanity Check 9 (line 1361).

**Problem:** The revised surprise std dev estimate of "~0.3-0.6 for liquid
stocks" (up from 0.005-0.015 in Draft 1) is now correctly sized for
raw-volume surprises with the H baseline. However, this range itself
confirms the M1 domain mismatch: typical delta values of ~0.1-0.3 are 3-8x
larger than typical scaled_base values of ~0.038, meaning the deviation
clamp would be saturated.

If M1 is resolved by switching to percentage-space surprises (option a),
the expected std dev would return to approximately 0.005-0.015 (the Draft 1
estimate was accidentally correct for percentage-space surprises).

**Recommendation:** Update the sanity check estimate after resolving M1.
Document both values: ~0.3-0.6 if using raw-volume surprises, ~0.005-0.015
if using percentage-space surprises.

---

### m3. No-intercept regression justification could be strengthened

**Location:** Lines 787-793.

**Problem:** The spec applies no-intercept to the surprise regression "by
analogy" from the VWAP-error regressions. The Draft 2 revision added a
functional justification: "omitting the intercept forces zero adjustment
when all recent surprises are zero." This is better, but there is a
subtlety: OLS without an intercept is only unbiased when the true
data-generating process has no intercept. If surprises have a nonzero mean
(e.g., due to systematic H bias from M2), the no-intercept regression will
project the mean onto the slope coefficients, producing biased betas.

**Recommendation:** Add a note: "The no-intercept assumption requires that
mean surprise is approximately zero. Verify via Sanity Check 9. If mean
surprise deviates significantly from zero (e.g., > 0.02 in absolute value),
consider adding an intercept or debiasing H."

---

### m4. build_surprise_regression uses 0.0 for missing surprises

**Location:** Function 7, line 747.

**Problem:** `surprises_today.get(i - lag, 0.0)` returns 0.0 when a surprise
value is missing for a given (day, bin) pair. In practice, surprise_data
should contain entries for all (day, bin) combinations in the training
window, so this default should never trigger. But if any (day, bin) pair is
excluded (e.g., due to min_volume_floor filtering in the surprise computation
at line 686-688, where H <= min_volume_floor sets surprise = 0.0), the
regression treats "zero surprise" and "missing data" identically.

The consequence depends on whether the zero-surprise bins were genuinely
zero-surprise or just missing. For illiquid bins where H < min_volume_floor
(meaning the stock's 21-day average volume for that bin is < 100 shares),
surprise is set to 0.0. If this bin then appears as a lag for a subsequent
bin's regression, the regression treats the illiquid bin as "no surprise"
rather than "no information." This could slightly bias betas downward (toward
zero) for stocks with many illiquid bins.

**Recommendation:** This is minor for the target universe (top 500 by dollar
volume, where illiquid bins are rare). Document the assumption: "Missing or
illiquid-bin surprises are treated as zero (no surprise). For illiquid
stocks, consider excluding bins with H < min_volume_floor from the lag
vector entirely, or using NaN-aware regression."

---

## Citation Verification

I re-verified the key citations that changed between drafts:

| Claim | Cited Source | Verified? | Notes |
|-------|-------------|-----------|-------|
| Surprise baseline = "departures from historical average" | p.18-19 | Yes | "departures from a historical average approach" confirmed |
| N_hist = 21 from Exhibit 1 | Exhibit 1, p.18 | Yes | "Next Bin (Prior 21 days)" confirmed |
| Deseasonalization = 6-month window | p.17 para 5 | Yes | "trailing six months" confirmed |
| "we could apply our more sophisticated raw volume forecasting model" | p.19 | Yes | Quote accurate; context is listing extensions to Humphery-Jenner |
| "depart no more than 10%" | p.24 | Yes | Confirmed, referencing Humphery-Jenner |
| "separate method for computing the deviation bounds" | p.19 | Yes | Confirmed; implies production used adaptive bounds |
| p_max_intra = 4 ("less than five") | p.18 | Yes | "AR lags with a value less than five" confirmed |
| "fewer than 11 terms" as outcome | p.18 | Yes | "As a result, we fit each symbol with a dual ARMA model having fewer than 11 terms" |

**Paper re-read on surprise methodology:** The paper (p.19) says: "we could
apply our more sophisticated raw volume forecasting model described
previously as the base model from which to compute volume surprises." The
draft's choice to use H instead is defensible as discussed, but I note the
paper's actual production system likely used the full Model A as the surprise
base. This is not a new issue — it was addressed in the M4 resolution — but
should be noted as a known divergence from the paper's likely implementation.

**Humphery-Jenner context:** The paper describes Humphery-Jenner's approach as
working with "volume surprises — deviations from a naive historical forecast"
and "training a model on decomposed volume." In Humphery-Jenner's original
work, the "naive historical forecast" is a percentage forecast (VWAP curve),
and "decomposed volume" refers to actual percentage minus historical
percentage. This supports defining surprises in percentage space (M1
recommendation a).

---

## Draft 1 Issue Resolution Assessment

All 13 issues from Critique 1 were addressed:

| Issue | Resolution Quality | Notes |
|-------|-------------------|-------|
| M1: N_hist / Component 1 | Excellent | Clean separation via Function 1a |
| M2: Regime assignment asymmetry | Excellent | Standardized convention, well-documented |
| M3: Day-boundary handling | Excellent | Independent-segment approach clearly specified |
| M4: Unconditional surprise baseline | Good | H-baseline well-justified; but introduced M1 domain mismatch (latent issue) |
| M5: Percentage coherence | Good | Discussion and optional renormalization added |
| m1: N_hist unused | Resolved | Now used in Function 1a |
| m2: Exhibit 1 "4 bins" | Resolved | Acknowledged with explanation |
| m3: Surprise std dev unsourced | Resolved | Marked as Researcher inference; value updated |
| m4: Conditioned training MAPE | Resolved | Clarified as intentional |
| m5: Nelder-Mead fragility | Resolved | Multi-restart + convergence fallback added |
| m6: CV remainder handling | Resolved | Remainder to last fold specified |
| m7: VWAP bps benchmarks | Resolved | Clarified as relative benchmarks |
| m8: ARMA interface | Excellent | Concrete interface specified with 5 methods |

---

## Positive Observations

1. **Revision notes quality:** The Draft 2 revision notes at the top are
   exemplary — each critique point is listed with what changed, making it
   easy to verify resolutions.

2. **Function 1a separation:** Clean, well-documented, with explicit
   justification for why N_hist and N_seasonal should differ. The note about
   responsiveness vs. stability is valuable.

3. **Independent-segment ARMA:** The day-boundary handling specification
   (lines 278-380) is thorough, covering fitting, initial-day prediction,
   and mid-day prediction with clear references to standard ARMA practice.

4. **ARMA model interface:** The 5-method interface (predict_at, predict_next,
   append_observation, make_state, predict) is well-specified with clear
   semantics for each method. The distinction between inter-day and intraday
   methods (make_state/predict are intraday-only) is clean.

5. **Surprise baseline alternatives documentation:** The three-option
   discussion (lines 769-797) is honest about the paper's ambiguity and
   provides clear reasoning for the choice.

6. **Percentage coherence discussion:** The argument that VWAP algorithms
   use only next-bin forecasts (making full-day coherence non-critical) is
   well-reasoned and practical.

---

## Summary of Recommended Changes

| Issue | Severity | Action |
|-------|----------|--------|
| M1: Domain mismatch in surprise regression | Major | Redefine surprises in percentage space, or apply delta multiplicatively |
| M2: Static hist_avg across training window | Major | Compute rolling hist_avg per training day, or document as trade-off |
| m1: Validation uses hist_avg from validation period | Minor | Use separate hist_avg for validation evaluation |
| m2: Sanity check 9 std dev inconsistent with M1 fix | Minor | Update after M1 resolution |
| m3: No-intercept assumption requires mean-zero surprises | Minor | Add verification note |
| m4: Zero-default for missing surprises | Minor | Document assumption for illiquid stocks |
