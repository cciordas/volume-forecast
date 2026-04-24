# Critique of Implementation Specification Draft 2: Dual-Mode Intraday Volume Forecast

## Summary

Draft 2 is a substantial improvement over draft 1. All 5 major and 7 minor issues from critique 1 have been addressed thoroughly. The additions -- Function 2b (UpdateInterDayState), the Exhibit 1 "4 Bins" interpretation discussion, the MSE/MAPE design decision block, the train/predict denominator discussion, and the explicit Metrics subsection -- are all well-reasoned, clearly written, and properly cited with Researcher inference labels where appropriate.

The remaining issues are narrower in scope: 2 medium issues related to implementability gaps and 3 minor issues. None require fundamental restructuring. This draft is close to implementation-ready.

**Issue count: 0 major, 2 medium, 3 minor.**

---

## Assessment of Critique 1 Revisions

All 12 issues from critique 1 are resolved:

| Issue | Resolution Quality | Notes |
|-------|-------------------|-------|
| M1 (Exhibit 1 "4 Bins") | Excellent | Both interpretations presented with clear justification for choice (2). Developer guidance for alternative. |
| M2 (MSE vs MAPE) | Excellent | Full design decision block with pros/cons and recommendation to implement both. |
| M3 (Train/predict denominator) | Good | Mismatch acknowledged, two approaches described, simpler one recommended. See m-new-3 for a remaining subtlety. |
| M4 (Missing metric formulas) | Excellent | Full Metrics subsection with exact formulas and aggregation hierarchy. |
| M5 (ARMA state update) | Excellent | Function 2b with explicit pseudocode. Well-integrated into orchestration. |
| m1 (N_hist sensitivity) | Good | Sensitivity raised to Medium-High with tuning note. |
| m2 (Deviation bounds) | Good | Strong warning about proprietary bounds in both Function 7 and parameter table. |
| m3 ("fewer than 11 terms") | Good | Soft constraint with warning implemented. |
| m4 (N_intraday ambiguity) | Good | Clarified as 21 trading days. |
| m5 (DEFAULT_REGIME) | Good | Specified as 1 (middle tercile) with rationale. |
| m6 (Pre-market regime) | Good | Explicit note added in Function 4. |
| m7 (Exhibit 10 phrasing) | Good | Tightened to distinguish per-category vs. aggregate. |

---

## Medium Issues

### Med-1. Missing intraday ARMA state conditioning function

**Spec section:** Function 6 (ForecastRawVolume), lines 388-392.

**Problem:** Draft 2 added Function 2b (UpdateInterDayState) for the inter-day ARMA state update, which is well-specified with explicit buffer management. However, there is no corresponding function for the intraday ARMA state conditioning. Function 6 calls:

```
intraday_model.update_state(deseas_observed)
```

where `deseas_observed` is an array of today's deseasonalized bin observations. This differs from the inter-day case in two ways:

1. **Input type:** UpdateInterDayState takes a single scalar observation. The intraday update takes an array of observations. How should the array be processed? Sequentially (filtering each observation through the model, computing a residual at each step)? Or as a batch (resetting the AR buffer wholesale)?

2. **Repeated conditioning:** At bin 5, `deseas_observed` contains bins [1,2,3,4,5]. At bin 6, it contains [1,2,3,4,5,6]. Does `update_state` reprocess from scratch each time (wasteful but stateless), or does it incrementally add only the new observation (requires maintaining state across bins within the day)? The pseudocode re-creates the full `deseas_observed` array each time, suggesting re-processing from scratch, but this is ambiguous.

The inter-day case is simple (one new observation per day). The intraday case is more complex and equally critical for correctness. A developer would need to choose between two approaches, and the wrong choice would produce different residual histories and thus different MA states.

**Recommendation:** Add a `ConditionIntraDayARMA(model, deseas_observed)` function analogous to Function 2b. Show explicitly that it processes observations sequentially to build up the residual buffer, starting from the model's initial state (fitted on the training window). Clarify whether the function resets to the training-window state and re-processes all observations each bin (simpler, stateless), or maintains state incrementally (more efficient).

---

### Med-2. Ambiguous `current_bin` for raw forecasts used in percentage regression training

**Spec section:** Function 8 (TrainPercentageRegression), lines 567-578.

**Problem:** The training loop uses `raw_forecast[s, d, i]` and `raw_forecast[s, d, lag_bin]` to compute expected percentages and surprises, but does not specify what `current_bin` value was used to generate these raw forecasts. There are two possibilities:

**(a) Static forecasts (current_bin = 0):** All raw forecasts for training day d are generated once at the start of the day, using only pre-market information. Expected percentages are computed from this static set.

**(b) Dynamic forecasts (current_bin = i - 1):** At each bin i, the raw forecast is generated using all observations through bin i-1, as would happen in live prediction.

The prediction function (Function 7) uses the dynamic approach: `ForecastRawVolume(s, t, lag_bin, lag_bin - 1)`. If training uses static forecasts but prediction uses dynamic forecasts, the surprise signal has different statistical properties at train vs. predict time. This is a separate issue from the M3 denominator mismatch (which was about actual percentages, not expected percentages).

If training uses dynamic forecasts (option b), the training procedure must run the full Model A pipeline at each bin for each training day, which multiplies computational cost by a factor of I (26x). This is feasible but expensive for 500 symbols over 252 days.

**Recommendation:** Clarify which approach is used. The natural choice is (b) -- dynamic forecasts -- to match prediction behavior. Note the computational cost and suggest caching: run the Model A pipeline once per training day, storing forecasts at each current_bin, then use these cached forecasts for the regression. If (a) is chosen for computational reasons, note the train/predict mismatch and its expected impact (early-day surprises will differ most).

---

## Minor Issues

### m-new-1. Multi-step ARMA forecast degradation not discussed

**Spec section:** Function 6 (ForecastRawVolume), lines 394-395.

**Problem:** The function computes `steps_ahead = i - current_bin` and calls `intraday_model.forecast(steps=steps_ahead)`. For ARMA(p,q) models, multi-step forecasts degrade rapidly:
- After p steps, the AR component uses only its own past forecasts (not observations), converging toward the unconditional mean.
- After q steps, the MA component contributes nothing (past residuals are zero in the forecast horizon).

For an ARMA(2,2) model with current_bin=1 forecasting bin 26, `steps_ahead=25`. The forecast is essentially the unconditional mean of the deseasonalized series (approximately 1.0), which after re-seasonalization equals the seasonal factor -- i.e., the historical average.

This means the intraday ARMA component A provides diminishing incremental value for distant bins, and the model's performance for distant bins relies primarily on the historical average H and inter-day ARMA D. This is not a bug -- it is expected behavior and the regime weights should adapt accordingly -- but a developer unaware of this property might be surprised that A converges to H for distant bins and wonder if their implementation is broken.

**Recommendation:** Add a brief note in Function 6 or Sanity Checks explaining that multi-step ARMA forecasts converge to the unconditional mean, and that the intraday ARMA's primary value is for near-term bins. The developer should verify this convergence (add as sanity check: for steps_ahead > 10, the intraday ARMA forecast should approximately equal the seasonal factor for that bin).

---

### m-new-2. Early-bin padding in regression training

**Spec section:** Function 8 (TrainPercentageRegression), lines 571-575.

**Problem:** The training loop starts at `i in 2..I` and pads missing lags with zeros. For K_reg = 3, bins 2 and 3 have 1 and 2 "real" surprise lags respectively, with the remainder padded as zero. This means the regression is trained on a mixture of real and synthetic (zero-padded) data points, which could bias the coefficient estimates.

Two approaches exist:
1. **Include padded observations (current):** More training data, but zero-padded early-bin observations may bias beta coefficients toward zero.
2. **Exclude early bins (start at i = K_reg + 1):** Cleaner coefficient estimates, fewer training samples. For K_reg = 3, this loses 2 out of 25 bins per day (8% of data) -- a modest cost.

**Recommendation:** Note this as a design choice. The current approach (include padded) is reasonable and consistent with Function 7's prediction-time padding. Optionally note that starting at `i = K_reg + 1` is a cleaner alternative the developer could test.

---

### m-new-3. Expected percentage denominator also differs between train and predict

**Spec section:** Functions 7 and 8.

**Problem:** The M3 discussion from critique 1 (well-addressed in draft 2) focused on the actual percentage denominator. However, the expected percentage denominator also differs:

- **Training (Function 8, line 567-568):** `expected_pct_d_i = raw_forecast[s, d, i] / sum(raw_forecast[s, d, j] for j in 1..I)` -- denominator is the sum of ALL raw forecasts for the day.
- **Prediction (Function 7, line 495):** `expected_pct = ForecastRawVolume(s, t, lag_bin, lag_bin - 1) / V_total_est` -- denominator is observed_total + remaining_forecasts.

In training, the sum of raw forecasts for all 26 bins is a fixed quantity for day d. In prediction, V_total_est evolves as bins are observed and is a mix of observed actuals and forecast remainders. Even if the same raw forecast model is used, these two denominators will generally differ because V_total_est includes actual observed volumes for early bins.

This is a smaller effect than the M3 actual-percentage mismatch (since the expected percentages are model-derived and less noisy), but it adds a second source of train/predict inconsistency. The discussion in draft 2 could mention this for completeness.

**Recommendation:** Add one sentence to the train/predict denominator discussion noting that the expected percentage computation also uses a different denominator structure, and that both mismatches diminish as more bins are observed.

---

## Citation Verification (Spot Checks on New Content)

I spot-checked the following new citations and claims added in draft 2:

| Claim | Spec Citation | Verified? | Notes |
|-------|-------------|-----------|-------|
| MAPE formula: 100% * (1/N) * sum(\|Pred - Actual\| / Actual) | p.17 | Yes | Matches paper's formula exactly |
| Percentage error: (1/N) * sum(\|Pred_Pct - Actual_Pct\|) | p.17 | Yes | "leave off the normalization factor" confirmed |
| "500-name universe over 250 trading days" | p.20 | Yes | Exact text |
| "26 forecasts for each of the 15-minute intervals" | p.20 | Yes | Exact text |
| "we developed a separate method for computing the deviation bounds" | p.19 | Yes | Exact quote |
| Exhibit 9: 0.00874 vs 0.00808, 7.55%* | p.23, Exhibit 9 | Yes | Matches table |
| "7%-10% reduction within each category" vs "9.1% across all" | p.23 | Yes | Correct distinction now made |
| Exhibit 3: 220.9, SE 29.46, t-stat 7.496, R^2 0.5146 | p.20, Exhibit 3 | Yes | All values match |
| Exhibit 5: 454.3, SE 71.78, t-stat 6.329, R^2 0.5886 | p.21, Exhibit 5 | Yes | All values match |
| "without the inclusion of a constant term" applies to validation regressions | p.19 | Yes | Correct: the quote is in the "Results and Discussion" section, referring to VWAP-error-vs-percentage-error regressions. Applying it to the surprise regression is correctly labeled as Researcher inference. |

All new citations verified. No misrepresentations found.

---

## Overall Assessment

Draft 2 is a high-quality implementation specification. The proposer addressed all 12 issues from critique 1 thoroughly, with detailed discussions, proper citation labeling, and actionable developer guidance. The document is well-structured with clear separation between paper-sourced claims and Researcher inferences.

The 2 medium issues (intraday ARMA state conditioning and training forecast ambiguity) are genuine implementability gaps that could lead a developer to make incorrect implementation choices, but they are narrow in scope and straightforward to address. The 3 minor issues are refinements that improve developer understanding but do not risk incorrect implementations.

A revised draft addressing the medium issues would be ready for implementation. The minor issues are optional improvements.
