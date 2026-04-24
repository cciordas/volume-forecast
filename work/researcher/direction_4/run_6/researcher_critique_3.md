# Critique of Implementation Specification Draft 3: Direction 4, Run 6

## Summary

Draft 3 resolves all 6 issues from critique 2 (1 major, 5 minor) cleanly.
The spec has converged substantially across three rounds:

- Round 1: 5 major + 9 minor issues
- Round 2: 1 major + 5 minor issues
- Round 3: 0 major + 2 minor issues

The spec is implementation-ready. The two remaining minor issues are
design refinements that a developer could address during implementation
without risk of a fundamentally incorrect model.

### Resolution of Critique 2 Issues

- **M1 (Sophisticated variant baseline mismatch):** Resolved. Function 11
  (lines 1229-1259) now computes Model A's pre-observation forecasts once
  at start of day (current_bin=0, no observations) and holds them as a
  fixed baseline throughout the day. The comment block (lines 1230-1238)
  clearly explains why actual volumes must not replace Model A forecasts
  for observed bins, and how the fixed baseline maintains consistency with
  generate_model_a_training_forecasts. Researcher inference item 27
  correctly documents this design choice.

- **m1 (predict_next idempotency):** Resolved. Lines 226-228 explicitly
  state: "This is a pure query that does NOT modify the model's internal
  state. It can be called multiple times and will return the same value
  until append_observation() is called." Added as Researcher inference
  item 29.

- **m2 (Switch-off calibration bias):** Resolved. Lines 1001-1007 use the
  crossover bin's median directly, with a clear explanation of why draft
  2's previous-bin approach systematically biased below the base
  threshold. Added as Researcher inference item 30.

- **m3 (Median regime in training helper):** Resolved. Lines 1136-1184
  reconstruct per-(day, bin) regime from historical cumulative volume
  percentiles, mirroring Function 5's approach. The comment block
  (lines 1166-1172) explains the rationale. Added as Researcher inference
  item 28.

- **m4 (remaining_pct hardcoded to hist_pct):** Resolved. Lines 854-856
  now use `base` (which is `baseline_pct` if provided, else `hist_pct`)
  for the remaining_pct computation, ensuring consistency between the
  total-volume estimate and the surprise baseline.

- **m5 (Confusing sanity check notation):** Resolved. Sanity Check 15
  (lines 1616-1619) now reads: "verify that interday_term_counts[i] +
  intraday_model.term_count <= max_dual_arma_terms. This enforces the
  paper's 'fewer than 11 terms' constraint."

---

## Minor Issues

### m1. Pre-observation baseline introduces a secondary training/prediction context mismatch

**Spec location:** Function 11 (lines 1242-1259) vs.
generate_model_a_training_forecasts (lines 1155-1184).

**Problem:** The M1 fix from critique 2 was correctly applied — the live
baseline no longer uses actual volumes for observed bins. However, the
chosen approach (computing ALL Model A forecasts at current_bin=0) creates
a secondary mismatch in the amount of intraday context available.

During training (generate_model_a_training_forecasts), the Model A forecast
for bin i uses:
- Intraday ARMA context from bins 1..i-1 (via `make_state(deseas_today[1:i])`)
- Regime assignment based on cumulative volume through bins 1..i-1

During live prediction, the pre-observation baseline computes ALL forecasts
with current_bin=0:
- Intraday ARMA uses `make_state([])` — unconditional forecast for ALL bins
- Regime assignment defaults to median (0.5 percentile) for ALL bins

For early bins (i=1,2), this difference is negligible since training also has
minimal context. For later bins (i=20+), the training forecast benefits from
19+ bins of intraday ARMA conditioning and an accurate regime assignment,
while the live baseline uses no conditioning and median regime. This means:

- Training surprises for later bins reflect prediction error of a
  well-conditioned Model A forecast
- Live surprises for later bins reflect prediction error of an
  unconditioned Model A forecast, which will systematically have larger
  magnitude (the unconditioned forecast is less accurate)

The regression, trained on moderate-magnitude surprises, receives
larger-magnitude surprises during live prediction for later bins. This
could cause the delta adjustments to be larger than intended, though the
deviation clamp provides a safety bound.

**Severity:** Low. The deviation clamp limits the practical impact, and the
mismatch direction (larger live surprises) is far less harmful than the
original M1 (near-zero live surprises that nullified Model B entirely).
The current approach is pragmatically sound.

**Potential fix (for future refinement):** Build the sophisticated baseline
iteratively during the day rather than pre-computing it:

```
# At each iteration of the main loop, BEFORE observing the actual volume:
IF model_state.params.use_model_a_baseline AND current_bin >= 1:
    # Baseline for bins 1..current_bin uses Model A forecasts computed
    # with bins 1..j-1 as context for each bin j (matching training)
    baseline_pct_today[current_bin] = predict_raw_volume(
        current_bin, observed_volumes[1:current_bin], current_bin - 1, ...)
        / total_model_a_forecast
```

This would require storing per-bin Model A forecasts as they are produced
(before each bin's actual volume is observed), matching the training
procedure exactly. The total for normalization would also need to be
updated iteratively (observed bins' Model A forecasts + remaining bins'
current forecasts). This adds complexity but eliminates the context
mismatch.

Alternatively, modify generate_model_a_training_forecasts to also use
context-free (current_bin=0) forecasts for all bins, matching the live
approach. This is simpler but produces a lower-quality baseline.

Either way, the current approach is functional and the deviation clamp
provides adequate safety.

---

### m2. N_surprise_lags cross-validation described but not implemented in pseudocode

**Spec location:** Calibration section, step 4 (lines 1512-1515) vs.
Function 10 (lines 1079-1095).

**Problem:** The Calibration section describes cross-validation for
N_surprise_lags selection:

> "For each candidate N_surprise_lags in {1,...,10}, fit the regression
> on the first portion and evaluate MAE on the held-out portion. Select
> the lag count with lowest validation MAE."

However, Function 10 (train_full_model) takes params.N_surprise_lags as a
fixed parameter and passes it directly to fit_surprise_regression (line
1094). There is no function that performs the cross-validation search over
lag counts. The regime count selection (N_regimes) has a full implementation
in Function 5 (lines 468-503), but the analogous procedure for surprise
lags is only described in prose.

A developer implementing the calibration pipeline would need to write this
CV loop from scratch, interpreting the prose description.

**Fix:** Either:
(a) Add a helper function (e.g., `select_surprise_lags`) that implements
    the CV search and is called by train_full_model before
    fit_surprise_regression, or
(b) Simplify the Calibration section to say that N_surprise_lags is a
    fixed hyperparameter (recommended value: 5), selected via offline
    experimentation rather than per-stock CV. This is arguably more
    practical since the paper says the optimal number was identified
    globally for U.S. equities, not per stock.

---

## Positive Observations

1. **Clean M1 resolution:** The pre-observation baseline approach is
   well-engineered. The comment block in Function 11 (lines 1230-1238)
   provides a clear, self-contained explanation of why actual volumes
   must not be used, directly referencing the training procedure. A
   developer reading this section would understand the design constraint
   without needing to cross-reference the critique.

2. **Researcher inference tracking matured:** The consolidated list has
   grown to 30 items (from 17 in draft 1, 28 in draft 2), with each new
   item (29: predict_next purity, 30: switch-off crossover bin) properly
   documented inline and in the summary list.

3. **Training helper regime reconstruction:** The
   generate_model_a_training_forecasts helper (lines 1166-1184) now
   mirrors Function 5's regime reconstruction approach, with clear
   comments explaining why the median-regime shortcut was removed and
   how the per-(day,bin) regime assignment avoids training/prediction
   mismatch. This is thorough engineering.

4. **Switch-off calibration fix well-documented:** Lines 1001-1007 include
   a parenthetical explaining what draft 2 did wrong and why the current
   approach is correct. This kind of revision history aids developer
   understanding.

5. **Function 9 baseline consistency:** The remaining_pct fix (using `base`
   instead of hardcoded `hist_pct`) is cleanly integrated, with a comment
   (lines 853-855) explaining that the total-volume estimate must be
   consistent with the surprise baseline.

6. **Overall spec quality:** At 1818 lines with 30 documented Researcher
   inference items, comprehensive paper references, 16 sanity checks, 12
   edge cases, and 10 known limitations, this spec provides a developer
   with everything needed to implement the model correctly. The
   pseudocode is directly translatable to code.

---

## Summary of Required Changes

| # | Severity | Section | Issue |
|---|----------|---------|-------|
| m1 | Minor | Function 11 + training helper | Pre-obs baseline creates secondary context mismatch (less harmful than original M1, bounded by deviation clamp) |
| m2 | Minor | Calibration section vs. Function 10 | N_surprise_lags CV described in prose but missing from pseudocode |

---

## Convergence Assessment

The spec has converged. The two remaining issues are:
- m1: A design refinement for the sophisticated variant. The current
  approach is functional and safe (deviation clamp bounds the impact).
  The fix is complex (iterative baseline construction) and could be
  deferred to implementation-time optimization.
- m2: A documentation/completeness gap. Either adding a small helper
  function or simplifying the prose resolves it.

Neither issue risks a fundamentally incorrect implementation. A developer
following this spec will produce a working model that matches the paper's
methodology. I recommend accepting draft 3 (with optional m1/m2 fixes)
as the final implementation specification.
