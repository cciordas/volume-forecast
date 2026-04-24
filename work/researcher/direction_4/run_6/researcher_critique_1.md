# Critique of Implementation Specification Draft 1: Direction 4, Run 6

## Summary

The draft is well-structured, detailed, and clearly written. It provides a
thorough decomposition of the Satish et al. (2014) dual-model system into 12
implementable functions with clear pseudocode. However, I identified several
issues ranging from citation errors to algorithmic ambiguities that could lead
a developer to produce an incorrect implementation. I found **5 major issues**
and **9 minor issues**.

---

## Major Issues

### M1. "Fewer than 11 terms" constraint misapplied (Function 4)

**Spec location:** Function 4, lines 262-268.

**Problem:** The paper states (p.18 para 1): "we fit each symbol with a dual
ARMA model having fewer than 11 terms." The word "dual" refers to the
combined inter-day + intraday ARMA system — i.e., the total number of
parameters across BOTH models should be fewer than 11. The spec applies this
constraint only to the intraday model as a "soft constraint on each individual
model" (line 265-266).

This is incorrect. The constraint is a joint constraint: if the inter-day
ARMA for a given bin uses, say, ARMA(3,2) = 6 terms (including constant),
then the intraday model is limited to at most 4 terms. The spec's current
approach of independently capping each model at 10 terms allows a combined
total of up to 20 terms, violating the paper's specification.

**Fix required:** The constraint must be enforced jointly. One approach: fit the
inter-day model first (Function 3), record its term count per bin, then pass
the remaining term budget to the intraday model fitting (Function 4). Since
the intraday model is shared across all bins but inter-day models are per-bin,
this creates a complication — the intraday model must satisfy the constraint
for ALL bins simultaneously. The spec should explicitly address this design
tension and propose a resolution (e.g., use the maximum inter-day term count
across all bins to determine the intraday budget, or use a global cap like 5
for each model).

**Paper evidence:** p.18 para 1: "we fit each symbol with a dual ARMA model
having fewer than 11 terms." The phrase "dual ARMA model" refers to the pair
of ARMA components described in the preceding paragraphs.

---

### M2. No-intercept citation is wrong (Function 8)

**Spec location:** Function 8, lines 653-654.

**Problem:** The spec cites Satish et al. 2014, p.19 ("Note that we perform both
regressions without the inclusion of a constant term") as justification for
the no-intercept surprise regression. However, reading p.19 in context, this
sentence appears in the "Results and Discussion" section under "Validating
Volume Percentage Prediction Error," and refers to the two VWAP tracking error
vs. volume percentage error regressions shown in Exhibits 3 and 5 — NOT the
surprise regression in Model B. The full quote continues: "This means that our
model does not assume that there is a positive amount of VWAP error if our
volume predictions are 100% accurate." This clearly describes the
VWAP-error-vs-percentage-error validation regressions, not the surprise model.

**Impact:** The surprise regression may or may not use an intercept — the paper
does not explicitly specify this for the Model B surprise regression. The
spec's argument that "mean surprise is zero by construction" (line 639-641) is
a reasonable Researcher inference that supports the no-intercept choice, but
it should be marked as such rather than attributed to p.19.

**Fix required:** Remove the false citation. Mark the no-intercept choice as
Researcher inference, supported by the mean-zero-surprise argument. Consider
also noting that Humphery-Jenner (2011) may contain the actual specification
for this regression.

---

### M3. Weight normalization is unspecified (Function 5)

**Spec location:** Function 5, minimize_mape helper (lines 441-473).

**Problem:** The weights w_H, w_D, w_A are constrained to be non-negative (via
exp-transform) but are NOT constrained to sum to 1 or any fixed value. This
means the optimizer could produce arbitrarily scaled weights (e.g.,
[100, 200, 300]). The MAPE objective would still be minimized, but:

1. The resulting weights are not interpretable as proportions.
2. Numerical stability could suffer — large weights amplify noise.
3. The "default" fallback weight of [1/3, 1/3, 1/3] (line 463) produces a
   fundamentally different scale than unconstrained optimized weights, so the
   fallback is inconsistent with optimized regimes.

The paper says "a dynamic weight overlay on top of these three components...
that minimizes the error on in-sample data" (p.18 para 1) but does not
specify whether weights are normalized.

**Fix required:** Either:
(a) Add a sum-to-1 constraint (most natural interpretation of "weight overlay"),
    using softmax instead of exp-transform, or
(b) Explicitly state that weights are unnormalized and explain why this is
    acceptable (the MAPE objective finds the right scale), or
(c) Add a normalization step after optimization: w_j = w_j / sum(w).

Option (a) is recommended as it produces interpretable weights and matches the
natural meaning of "weighting" three components.

---

### M4. "Self-updating" deviation limits ignored

**Spec location:** Function 9, deviation_limit parameter.

**Problem:** The paper on p.24 describes Humphery-Jenner's model as having
"self-updating deviation limits and switch-off parameters" — both attributes
are listed as strengths that the authors preserved. The spec implements
deviation_limit and switchoff_threshold as fixed constants (0.10 and 0.80
respectively). The "self-updating" mechanism is completely absent.

This is significant because a fixed 10% deviation limit may be appropriate for
some stocks but not others (e.g., illiquid stocks with higher variance in
volume patterns may need wider limits, while stable large-caps may need
tighter ones). Humphery-Jenner's original model presumably adapts these limits
to each stock's characteristics.

**Fix required:** Acknowledge this gap explicitly. Either:
(a) Describe a self-updating mechanism (e.g., set deviation_limit as a
    percentile of historical surprise magnitudes for each stock), or
(b) Note that the Humphery-Jenner (2011) paper should be consulted for the
    self-updating procedure, and mark the fixed-value approach as a
    simplification with known limitations, or
(c) Add a per-stock calibration step for deviation_limit and
    switchoff_threshold to the calibration procedure.

---

### M5. Model B surprise baseline ambiguity unresolved

**Spec location:** Functions 8 and 9, and Overview (lines 15-17).

**Problem:** The paper explicitly describes two options for computing volume
surprises in Model B:
1. Naive: surprises relative to historical average percentages.
2. Sophisticated: surprises relative to the raw volume model (Model A) output.

The paper states (p.19): "we could apply our more extensive volume forecasting
model described previously as the base model from which to compute volume
surprises." This implies the sophisticated approach was tested and potentially
preferred.

The Overview correctly notes both options (lines 15-17: "Model A's raw volume
forecasts can optionally provide the surprise signal"), but Functions 8 and 9
implement ONLY the naive approach. The "optional" Model A integration is never
specified in pseudocode.

**Fix required:** Either:
(a) Add a variant of Function 9 that computes surprises relative to Model A
    forecasts instead of hist_pct, with pseudocode for how to convert Model A
    raw volume forecasts into percentage baselines, or
(b) Provide a clear architectural note explaining how to swap the surprise
    baseline, specifying the exact interface point where Model A output would
    replace hist_pct in the computation.

---

## Minor Issues

### m1. Paper says "four components," spec says three

**Spec location:** Overview, line 8 ("three signal components").

**Problem:** The paper on p.17 states: "The raw volume forecast model consists of
four components, see Exhibit 1." Exhibit 1 shows: Historical Window, ARMA
Daily, ARMA Intraday, and Dynamic Weights Engine. The spec treats the dynamic
weight overlay as a combination mechanism rather than a component, which is a
reasonable interpretation, but contradicts the paper's explicit count.

**Fix:** Acknowledge the paper's "four component" framing and note that the spec
treats the weight engine as the combination mechanism rather than a signal
source. This is a minor framing issue but could confuse a developer
cross-referencing the paper.

---

### m2. Exhibit 1 "Prior 5 days" label not discussed

**Spec location:** Function 3 (inter-day ARMA), Parameter table (N_interday_fit).

**Problem:** Exhibit 1 labels the ARMA Daily input as "Next Bin (Prior 5 days)."
The spec sets N_interday_fit = 252 (1 year) as the fitting window and p_max = 5
for lag order. The "Prior 5 days" label in Exhibit 1 most likely refers to the
AR lag structure (p up to 5) rather than a 5-day fitting window, but this is
never discussed. A developer looking at Exhibit 1 might reasonably interpret
this as "only use 5 days of data" and be confused by the 252-day fitting
window.

**Fix:** Add a note explaining the Exhibit 1 label: "Prior 5 days" refers to the
maximum AR lag order (p_max = 5), not the fitting window length.

---

### m3. Exhibit 1 "4 Bins Prior to Current Bin" not discussed

**Spec location:** Function 4 (intraday ARMA), Function 6 (predict_raw_volume).

**Problem:** Exhibit 1 labels the ARMA Intraday input as "4 Bins Prior to Current
Bin." This suggests a fixed 4-lag lookback, but the spec uses AICc model
selection with p_max_intra = 4, which could select fewer lags. The exhibit
label should be discussed to clarify whether "4 bins" is fixed or a maximum.

**Fix:** Add a note explaining the Exhibit 1 label and reconciling it with the
AICc-based selection approach.

---

### m4. Sanity check #1 mixes up Model A and Model B metrics

**Spec location:** Sanity Check #1, lines 1106-1109.

**Problem:** The check says: "With all regime weights set to [1,0,0] (H only),
Model A should reproduce the simple rolling average baseline. MAPE should
match the 'Historical' column in Exhibit 9 (~0.00874 for 15-minute percentage
error)."

But Exhibit 9 reports volume PERCENTAGE errors (Model B metric: mean absolute
deviation), not raw volume MAPE (Model A metric). With weights [1,0,0], Model A
produces H as raw volume — its MAPE would be the "Historical Window" baseline
for raw volume, which is NOT reported in Exhibit 9. Exhibit 9's 0.00874 is the
median absolute percentage error for the historical approach.

**Fix:** Split into two checks:
- Model A with [1,0,0] weights: verify it reproduces H exactly. No direct
  benchmark number is available from the paper (Exhibit 6 shows reduction
  percentages, not absolute MAPE).
- Model B with zero regression coefficients: verify it reproduces hist_pct.
  The 0.00874 figure from Exhibit 9 is the correct benchmark for this check.

---

### m5. Model B cannot forecast bin 1 — not handled in orchestration

**Spec location:** Function 11, lines 866-872.

**Problem:** Edge case #2 (line 1159-1162) correctly notes that Model B cannot
produce a forecast for bin 1 (needs at least one observed bin). However,
Function 11's orchestration loop (lines 855-887) starts with current_bin = 0
and the Model B prediction guard is `IF current_bin >= 1 AND current_bin < I`,
which correctly skips bin 1. But the spec doesn't specify what to output for
bin 1's percentage forecast — pct_forecasts[1] is never set. The array is
initialized but its first element remains uninitialized.

**Fix:** Explicitly set pct_forecasts[1] = hist_pct[1] before the loop, matching
edge case #2's recommendation.

---

### m6. Percentile rank function undefined

**Spec location:** Functions 5 and 6.

**Problem:** The `percentile_rank(value, reference_values)` function is used in
both Function 5 (line 374) and Function 6 (line 541) but is never defined. A
developer must decide: is this the fraction of reference values less than
`value`? Less than or equal? Does it use linear interpolation? The choice
affects regime assignment at boundary values.

**Fix:** Provide a brief definition, e.g.: "percentile_rank(x, ref) = count(ref
< x) / len(ref)". Or specify whether to use scipy.stats.percentileofscore
with kind='rank', 'strict', 'weak', or 'mean'.

---

### m7. Intraday ARMA state reset semantics unclear

**Spec location:** Function 6, lines 526-532.

**Problem:** The spec calls `intraday_model.make_state(deseas_today)` to create
a fresh state from the deseasonalized observations seen so far today. But the
ARMA model was fitted on independent daily segments (Function 4). The
`make_state` function initializes "a fresh prediction state from a sequence of
deseasonalized observations WITHOUT mutating the model's stored state" — but
how does it handle the initial conditions? Does it run a Kalman filter forward
from the unconditional distribution? Does it use exact diffuse initialization?
This affects the first few predictions of each day significantly.

**Fix:** Specify the state initialization: e.g., "The state is initialized with
the unconditional mean for AR lags and zero for MA residuals. Observations are
then processed sequentially through the Kalman filter to build up the
conditional state."

---

### m8. Volume history slicing inconsistency in train_full_model

**Spec location:** Function 10, lines 766-801.

**Problem:** The function creates separate slices of volume_history for each
component (vol_hist_seasonal, vol_hist_short, vol_hist_interday, etc.), but
the slice for weight optimization (line 787-788) uses
`train_end_date - N_weight_train - N_hist + 1` which assumes the weight
training procedure needs N_hist extra days of lookback. However, Function 5
also needs the interday_models and intraday_model to produce predictions for
each training day. These models need their own lookback to have been fitted,
but the slice doesn't account for whether the ARMA models were fitted using
data that overlaps with the weight training window.

This creates a potential look-ahead bias: the ARMA models are fitted on data
up to train_end_date (Step 3-4), but Function 5 uses them to produce
"predictions" for training days within that same window. The inter-day ARMA
predictions via `predict_at(d)` are conditional on the model fitted on
[train_end_date - N_interday_fit + 1, train_end_date], which includes data
after day d — producing optimistic in-sample "predictions."

**Fix:** Either:
(a) Note this as an approximation (using in-sample fitted values rather than
    true out-of-sample predictions for weight training) and justify it, or
(b) Use a walk-forward approach: for each training day d, fit ARMA models
    using only data up to d-1. This is computationally expensive but avoids
    look-ahead bias.

---

### m9. Deviation limit reference page number

**Spec location:** Paper References table, line 1261.

**Problem:** The spec cites "Satish et al. 2014, p.24" for the deviation limit
(10%) and switch-off (80%). On p.24, the paper is describing Humphery-Jenner's
model properties: "self-limiting attributes (e.g., once 80% of the day's
volume is reached, return to a historical approach) and deviation limits (e.g.,
depart no more than 10% away from a historical VWAP curve)." These values are
attributed to Humphery-Jenner, not independently specified by Satish et al.

**Fix:** The citation should read "Satish et al. 2014, p.24, describing
Humphery-Jenner (2011) model attributes." This clarifies that the primary
source for these parameters is Humphery-Jenner, and a developer wanting to
understand the full rationale should consult that paper.

---

## Positive Observations

1. **Excellent traceability:** Every function has explicit paper references,
   and Researcher inference items are consistently marked. The consolidated
   list of 17 inference items (lines 1279-1298) is a valuable developer
   reference.

2. **Strong fallback handling:** The FALLBACK sentinel pattern for failed ARMA
   fitting is well-designed and consistently applied across Functions 3, 4, 5,
   and 6.

3. **Thorough edge cases:** The 11 edge cases cover the important failure modes
   comprehensively, including half-trading days, stock splits, and empty
   regime buckets.

4. **Clear data flow diagram:** The training and prediction phase data flow
   (lines 944-962) gives a good architectural overview.

5. **Reasonable parameter defaults:** The parameter table provides sensible
   defaults with ranges that are well-justified by the paper.

---

## Summary of Required Changes

| # | Severity | Section | Issue |
|---|----------|---------|-------|
| M1 | Major | Function 4 | "Fewer than 11 terms" must be a joint constraint |
| M2 | Major | Function 8 | No-intercept citation is wrong (p.19 refers to VWAP regressions) |
| M3 | Major | Function 5 | Weight normalization unspecified — could produce arbitrary scales |
| M4 | Major | Function 9 | Self-updating deviation limits from Humphery-Jenner not addressed |
| M5 | Major | Functions 8-9 | Model A-based surprise variant never specified in pseudocode |
| m1 | Minor | Overview | "Three components" vs. paper's "four components" |
| m2 | Minor | Function 3 | Exhibit 1 "Prior 5 days" label not discussed |
| m3 | Minor | Function 4 | Exhibit 1 "4 Bins Prior" label not discussed |
| m4 | Minor | Sanity Check 1 | Mixes up Model A MAPE with Model B Exhibit 9 metric |
| m5 | Minor | Function 11 | pct_forecasts[1] left uninitialized |
| m6 | Minor | Functions 5-6 | percentile_rank function undefined |
| m7 | Minor | Function 6 | ARMA state initialization semantics unclear |
| m8 | Minor | Function 10 | Look-ahead bias in weight training procedure |
| m9 | Minor | References | Deviation limit citation should reference Humphery-Jenner |
