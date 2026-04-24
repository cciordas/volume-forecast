# Critique 1: Implementation Specification Draft 1 — Dual-Mode Volume Forecast (Direction 4, Run 4)

## Summary

Draft 1 is a thorough and well-structured specification that covers both Model A (Raw Volume) and Model B (Volume Percentage) from Satish, Saxena, and Palmer (2014). The pseudocode is detailed, citations are generally accurate, and researcher inferences are clearly marked. However, there are 3 major issues, 2 medium issues, and 6 minor issues that need to be addressed before a developer can implement confidently.

---

## Major Issues

### M1. Nelder-Mead does not support bound constraints

**Location:** `train_raw_volume_model`, Step 5c, lines 242-244.

**Problem:** The pseudocode specifies:
```
result = minimize(mape_loss, x0=[1/3, 1/3, 1/3],
                  method='Nelder-Mead',
                  bounds=[(0, None), (0, None), (0, None)])
```
Nelder-Mead is an unconstrained simplex method. In scipy's `minimize`, passing `bounds` with `method='Nelder-Mead'` is silently ignored (prior to scipy 1.7) or raises a warning (1.7+). The non-negativity constraint on weights (`w1, w2, w3 >= 0`) will not be enforced.

**Impact:** A developer following this pseudocode literally will get unconstrained weights that may be negative, contradicting the stated constraint and producing nonsensical forecasts (e.g., negative weight on historical average).

**Recommendation:** Either:
- (a) Switch to a bounded optimizer: `method='L-BFGS-B'` supports bounds natively, or `method='trust-constr'`.
- (b) Use a variable transformation: optimize over `w_raw = exp(w_log)` to enforce positivity, keeping Nelder-Mead.
- (c) Use scipy's `minimize` with `method='Nelder-Mead'` and add a penalty term for negative weights to mape_loss. This is the simplest but least clean approach.

Option (b) is recommended because MAPE is non-differentiable, making gradient-based methods (L-BFGS-B) problematic. Nelder-Mead is appropriate for MAPE, but needs the transformation.

### M2. Intraday ARMA conditioning: purity and state management unspecified

**Location:** `forecast_raw_volume`, line 311.

**Problem:** The pseudocode calls:
```
conditioned_state = model_a.intraday_model.condition(observed_deseas)
```
This appears to return a new state object, but the spec never clarifies whether `condition()` is pure (returns a fresh conditioned copy) or mutates `model_a.intraday_model` in place. This distinction is critical because:

1. **During live prediction:** `forecast_raw_volume` is called at each new bin observation (bin 1, bin 2, ..., bin 25). If `condition()` mutates, calling it at bin 5 with observations [1..4] and then at bin 6 with observations [1..5] could double-condition on bins 1-4.

2. **During percentage model training:** `forecast_raw_volume` is called in a loop over historical days (line 388 in `train_percentage_model`). If `condition()` mutates the shared model, the intraday ARMA state leaks across days.

3. **During regime weight optimization:** The inner loop of Step 5c calls helper functions `D_for_day` and `A_for_day` (lines 221-222) that presumably also invoke ARMA prediction. These are undefined — are they re-conditioning per day?

**Recommendation:** Explicitly specify:
- `condition()` is pure: it returns a new state object without modifying the underlying model.
- Add a note: "The intraday ARMA model parameters (AR/MA coefficients) are fixed after training. `condition()` only initializes the state vector (recent observations and residuals) for forecasting; it does not re-estimate parameters."
- Define the `H_for_day`, `D_for_day`, and `A_for_day` helper functions used in weight optimization, or replace them with inline logic showing how each component is computed for a historical day.

### M3. Missing orchestration / daily workflow function

**Location:** Overall spec structure.

**Problem:** The spec provides four functions (`train_raw_volume_model`, `forecast_raw_volume`, `train_percentage_model`, `forecast_volume_percentage`) and two helpers, but no top-level orchestration showing:

1. **Daily initialization:** What happens at 9:30 ET each day? Which functions are called, in what order?
2. **Intraday update loop:** At each new bin observation, what is the sequence of calls? Is Model A's forecast updated first, then Model B's surprise computed from the updated forecast, or from the market-open unconditional forecast?
3. **Re-estimation schedule:** The Calibration section (line 699) says "Re-estimate all component models daily" and "Re-run regime grid search monthly." How does a developer implement this? Is daily re-estimation done before market open using data through the prior day?
4. **State carry-over between days:** After a trading day ends, what state needs to be preserved for the next day? The inter-day ARMA state (new observation added), the cumulative volume distribution (new day added)?

**Impact:** Without an orchestration function, a developer must infer the interaction protocol between Models A and B from scattered comments. The coupling between models (Model A's output feeds Model B's surprise signal) makes this non-trivial and error-prone.

**Recommendation:** Add a `FUNCTION run_daily(stock, date, model_a, model_b)` pseudocode function that shows the complete intraday loop: market-open initialization, per-bin update, forecast generation, and end-of-day state update.

---

## Medium Issues

### Med1. "Fewer than 11 terms" constraint formulation is ambiguous

**Location:** Lines 126-134, Component 4 (Intraday ARMA).

**Problem:** The spec interprets the paper's "a dual ARMA model having fewer than 11 terms" (p.18) as a joint constraint:
```
max_i(interday_terms[i]) + intraday_terms < 11
```
This uses the *maximum* inter-day complexity across all 26 bins to constrain the single intraday model. If one bin (say, a high-volume opening bin) selects ARMA(5,2) with 8 terms, the intraday model is limited to 2 terms (constant + one lag), even though the other 25 bins may have much simpler inter-day models.

**Alternative interpretation:** The constraint could apply per-bin: for each bin i, `interday_terms[i] + intraday_terms < 11`. This would allow the intraday model more freedom when most bins have parsimonious inter-day models.

**Paper evidence:** The paper says "we fit each symbol with a dual ARMA model having fewer than 11 terms" (p.18). The phrase "each symbol" (not "each bin") suggests a symbol-level constraint, supporting the spec's interpretation. However, the "fewer than 11 terms" is presented as an observed outcome of AICc selection, not as an imposed constraint: "As a result, we fit each symbol with a dual ARMA model having fewer than 11 terms." The sentence describes what AICc naturally selects, not a hard bound.

**Recommendation:** Clarify that the 11-term figure is an empirical observation, not a constraint. Keep it as a soft guardrail (warn if exceeded) rather than a hard budget that restricts the AICc search space. This avoids the potentially over-restrictive max-across-bins formulation.

### Med2. Cross-validation in surprise regression ignores temporal structure

**Location:** Lines 416-420, `train_percentage_model` Step 3.

**Problem:** The spec prescribes "K-fold cross-validation (K=5)" for selecting the optimal number of surprise lag terms. The training data has strong temporal structure: observations from the same day are correlated (bins share the same day's volume regime), and observations from adjacent days may share serial correlation in surprises.

Standard K-fold randomly shuffles observations into folds, breaking temporal ordering. This creates leakage: a fold's test set may contain bin 15 of day d while the training set contains bins 14 and 16 of the same day, artificially inflating the model's apparent performance.

**Recommendation:** Use time-series cross-validation: either expanding-window (train on days 1..d, validate on days d+1..d+k) or blocked K-fold (split at day boundaries so entire days are in the same fold). Specify explicitly which approach to use.

---

## Minor Issues

### m1. N_hist = 21 sourced from diagram annotation, not methodology text

**Location:** Parameter table, line 658.

**Problem:** The spec cites "Satish et al. 2014, Exhibit 1 caption, 'Prior 21 days'" for N_hist = 21. However, the methodology text on p.16-17 introduces N as "a variable that we shall call N" without disclosing its value. The "Prior 21 days" annotation appears in Exhibit 1's diagram, which could be illustrative rather than the recommended operational value. The paper may have used a different N in their actual experiments.

**Recommendation:** Note that 21 is inferred from the diagram and may be illustrative. Mark sensitivity as "Medium-High" rather than "Medium" to encourage the developer to test alternative values (e.g., 10, 42, 63).

### m2. n_eff formula for day-boundary AICc is a rough approximation

**Location:** Line 166-167.

**Problem:** The effective sample size is computed as:
```
n_eff = len(deseasonalized_series) - len(day_boundary_indices) * p
```
This subtracts p observations per day boundary, approximating the loss of initial conditions at each segment start. For AICc, which is `AIC + 2k(k+1)/(n-k-1)`, an inaccurate n can substantially affect the penalty. With 21 days and p=4, n_eff = 546 - 84 = 462 vs. n = 546 — a 15% reduction that noticeably changes AICc rankings.

**Recommendation:** Note explicitly that this is an approximation. A more precise approach: sum the log-likelihoods per day-segment independently (each segment has I - p effective observations), then compute AICc using the total effective n = 21 * (26 - p). Alternatively, note that for the purpose of model selection the approximation is adequate since it affects all candidates similarly.

### m3. Missing explicit MAPE evaluation formula

**Location:** Validation section, lines 706-709.

**Problem:** The Validation section cites MAPE reduction percentages but never defines the exact MAPE formula used for evaluation. The paper (p.17) defines:
```
MAPE = 100% × (1/N) × Σ_i |Predicted_Volume_i - Raw_Volume_i| / Raw_Volume_i
```
The training loss `mape_loss` (lines 234-240) computes a related quantity but uses `min_volume_floor` filtering and operates on individual regime samples rather than the full evaluation set. A developer needs to know: (a) the exact evaluation MAPE formula, (b) whether `min_volume_floor` filtering also applies to evaluation, and (c) whether MAPE is computed per-bin then averaged across bins, or pooled across all (day, bin) pairs.

**Recommendation:** Add an explicit `FUNCTION compute_evaluation_mape(...)` showing the exact formula from the paper. Clarify the aggregation: the paper's MAPE formula sums over bins (index i runs over all bins) and then takes the mean, suggesting per-day MAPE averaged across days, or a grand pooled average.

### m4. V_total_est accuracy concern in early-day switchoff check

**Location:** Lines 450-456, `forecast_volume_percentage`.

**Problem:** The 80% switchoff condition uses `V_total_est = observed + forecasted_remaining`. Early in the day (e.g., bin 3), the forecasted remaining volume dominates `V_total_est`, making the ratio `cumulative_observed / V_total_est` sensitive to forecast error. A poor early forecast could trigger or suppress the switchoff inappropriately.

**Impact:** Low — the 80% threshold naturally prevents early-day activation (cumulative volume at bin 3 is ~15-20% of daily total, well below 80%). But a developer might worry about edge cases.

**Recommendation:** Add a brief note: "In practice, the switchoff condition only activates in the last 2-4 bins of the day. V_total_est estimation error is minimal at that point because most volume has been observed."

### m5. H_for_day, D_for_day, A_for_day helper functions undefined

**Location:** Lines 220-222, `train_raw_volume_model` Step 5c.

**Problem:** The weight optimization loop references `H_for_day(stock, j, d)`, `D_for_day(stock, j, d)`, and `A_for_day(stock, j, d)` to get each component's forecast for a historical (day, bin) pair. These are never defined. The developer must infer:
- `H_for_day`: rolling mean as of day d (using data up to d-1? or including d?). If including d, there's lookahead bias.
- `D_for_day`: inter-day ARMA forecast for bin j on day d — using the model fitted at train_end_date, not at d. This introduces lookahead.
- `A_for_day`: intraday ARMA forecast for bin j on day d — what is the conditioning? Is it conditioned on bins 1..j-1 of day d (sequential), or unconditional?

**Recommendation:** Define these helpers explicitly, or inline the logic. At minimum, specify: (a) H_for_day uses a trailing window ending at day d-1 (no lookahead); (b) D_for_day uses the fitted inter-day model to produce a one-step-ahead forecast for day d; (c) A_for_day uses the intraday model conditioned on bins 1..j-1 of day d (to match the live prediction scenario).

### m6. Renormalization scale factor computed but not stored or propagated

**Location:** Lines 521-528, `forecast_volume_percentage` Step 6.

**Problem:** The code computes a scale factor for bins target_bin+1..I:
```
scale = remaining_after_target / remaining_hist_after_target
```
But this factor is only computed, never returned or stored. The comment says "This scale factor would be applied to subsequent bin forecasts," but the function only returns `pct_forecast` for the current target bin. On the next call (for the next bin), a fresh `scaled_base` is computed from scratch (lines 496-502), which may not be consistent with the previously computed scale factor.

**Recommendation:** Clarify that renormalization is implicit: each call to `forecast_volume_percentage` independently computes `scaled_base` using the current remaining fraction and remaining historical percentages. The explicit scale factor computation in Step 6 is informational and can be removed from the pseudocode to avoid confusion. Alternatively, if the scale factor is meant to be propagated as state, add it to the return value and show how subsequent calls consume it.

---

## Citation Verification

| Spec Claim | Cited Source | Verified? |
|-----------|-------------|-----------|
| I = 26 bins | Satish et al. p.16, "26 such bins" | Yes |
| N_hist = 21 | Exhibit 1, "Prior 21 days" | Partial — diagram annotation, not methodology text (see m1) |
| N_seasonal = 126 | p.17, "trailing six months" | Yes |
| N_intraday_fit = 21 (one month) | p.18, "rolling basis over the most recent month" | Yes |
| p, q through 5 | p.17, "all values of p and q lags through five" | Yes |
| AR < 5 for intraday | p.18, "AR lags with a value less than five" | Yes |
| Fewer than 11 terms | p.18, "a dual ARMA model having fewer than 11 terms" | Yes, but see Med1 on interpretation |
| AICc | p.17, "corrected AIC... Hurvich and Tsai" | Yes |
| 10% deviation limit | p.24, referencing Humphery-Jenner | Yes |
| 80% switchoff | p.24, referencing Humphery-Jenner | Yes |
| No-intercept regression | p.19, "without the inclusion of a constant term" | Yes |
| 24% median MAPE reduction | p.20, "reduce the median volume error by 24%" | Yes |
| 29% bottom-95% reduction | p.20 | Yes |
| Exhibit 9 results | p.23, Exhibit 9 table | Yes — values match exactly |
| Exhibit 10 VWAP results | p.23, Exhibit 10 table | Yes — 9.62 vs 8.74 bps, 9.1% reduction |
| Custom curves recommendation | p.18, "scant historical data" | Yes |
| Chen et al. benchmark | Chen et al. 2016 summary | Yes (cross-checked against summary) |

All major citations verified. No misrepresentations found.

---

## Overall Assessment

The draft is solid and well-organized. The 3 major issues (optimizer bounds, ARMA state purity, missing orchestration) are all implementability concerns — a developer following the spec literally would hit concrete problems. The medium issues (11-term constraint interpretation, temporal cross-validation) affect model quality but have reasonable defaults. The minor issues are clarifications that reduce ambiguity.

**Estimated severity:** 3 major, 2 medium, 6 minor. Recommend one revision round to address majors and mediums, which should bring the spec to implementation-ready status.
