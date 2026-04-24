# Critique of Implementation Specification Draft 1: Dual-Mode Intraday Volume Forecast

## Summary

The draft is well-structured, thorough, and demonstrates careful reading of Satish et al. (2014). The dual-model architecture is clearly described, pseudocode is detailed, and the paper references are precise. Researcher inferences are properly labeled. However, I identify 5 major issues and 7 minor issues that should be addressed before a developer can implement confidently.

**Issue count: 5 major, 7 minor.**

---

## Major Issues

### M1. Intraday ARMA conditioning contradicts Exhibit 1 annotation

**Spec section:** Function 6 (ForecastRawVolume), lines 261-268.

**Problem:** The spec conditions the intraday ARMA on ALL observed bins today:
```
deseas_observed = [volume[s, t, j] / seasonal_factor[j] for j in 1..current_bin]
```

However, Exhibit 1 (Satish et al. 2014, p.18) explicitly labels the input to "ARMA Intraday" as "Current Bin" and "4 Bins Prior to Current Bin" -- a total of 5 bins, not all observed bins. This annotation is specific and deliberate.

There are two defensible interpretations:
1. Only the 5 most recent bins are used to condition/update the ARMA state at prediction time (i.e., the Kalman/state update uses only the last 5 deseasonalized observations).
2. The AR order is at most 4 (consistent with "AR lags with a value less than five," p.18), so regardless of how many observations are fed to the state updater, only the last ~4 matter for the AR prediction. In this case, feeding all observed bins is harmless but the Exhibit's annotation describes the effective memory.

The spec adopts interpretation (2) implicitly but does not discuss the Exhibit's "4 Bins" annotation at all. This is a significant omission because a developer reading the Exhibit would conclude that only 5 bins should be passed. The spec should explicitly address this discrepancy and justify its choice.

**Recommendation:** Add a paragraph in Function 6 discussing the "4 Bins Prior to Current Bin" annotation from Exhibit 1, explain both interpretations, and state which is implemented and why. If interpretation (2) is chosen, note that the implementation is equivalent to passing all bins when p <= 4.

---

### M2. Weight optimization uses MSE but the paper's metric is MAPE

**Spec section:** Function 5 (OptimizeWeights), lines 213-228.

**Problem:** The spec uses MSE as the weight optimization loss function (marked as Researcher inference). However, the paper's primary error metric for raw volume is MAPE (p.17), and the paper says the weight overlay "minimizes the error on in-sample data" (p.18, para 3). If "error" refers to MAPE, then MSE-optimal weights will differ from MAPE-optimal weights because:

- MSE penalizes large absolute errors quadratically, biasing weights toward reducing errors on high-volume bins/days.
- MAPE penalizes relative errors equally across volume scales, which is the stated evaluation criterion.

The choice of loss function directly affects the learned weights and therefore forecast quality. A model trained on MSE but evaluated on MAPE may produce suboptimal MAPE because the weights are skewed toward fitting high-volume observations.

The spec notes this alternative in passing ("An alternative is unconstrained weights with MAPE minimization") but buries it as a parenthetical rather than treating it as a primary design decision.

**Recommendation:** Either (a) change the primary recommendation to MAPE minimization (which requires gradient-free optimization since MAPE is non-smooth), or (b) keep MSE but add a strong justification for why MSE is preferred despite the metric mismatch (e.g., MSE is convex and has a unique solution for simplex-constrained problems, making it more robust). Present MAPE minimization as a variant the developer should test.

---

### M3. Inconsistent denominator in surprise computation between training and prediction

**Spec section:** Function 7 (ForecastVolumePercentage) vs. Function 8 (TrainPercentageRegression).

**Problem:** During **training** (Function 8, lines 391-392), surprises are computed using the actual total daily volume:
```
actual_pct_d_i = volume[s, d, i] / sum(volume[s, d, j] for j in 1..I)
```

During **prediction** (Function 7, lines 335-336), surprises are computed using the estimated total daily volume:
```
actual_pct = observed_volumes[lag_bin] / V_total_est
```

where `V_total_est = observed_total + sum(remaining_forecasts)`.

This creates a train/predict mismatch. In training, the denominator is exact; in prediction, it is a noisy estimate that evolves as more bins are observed. Early in the day, `V_total_est` can be significantly off from the true daily total, which means the surprise signal has different statistical properties at training vs. prediction time.

**Recommendation:** The spec should:
1. Acknowledge this mismatch explicitly.
2. Discuss whether the training phase should mimic the prediction phase by using "leave-future-out" estimated totals (i.e., for training day d at bin i, use the sum of actuals through bin i plus forecasts for bins i+1..I as the denominator, to match what happens in production).
3. If the simpler approach (actual daily total) is kept, justify why the mismatch is acceptable (e.g., late in the day V_total_est converges to the actual total, and early-day surprises are small and padded with zeros anyway).

---

### M4. Missing explicit MAPE and percentage error formulas in the Validation section

**Spec section:** Validation > Expected Behavior (lines 600-630).

**Problem:** The spec quotes specific numerical results (24% MAPE reduction, 7.55% percentage error reduction, etc.) but never defines the exact formulas the developer should implement to compute these metrics. The paper defines two distinct metrics (p.17):

1. **MAPE** (raw volume): `100% * (1/N) * sum_i(|Predicted_Volume_i - Raw_Volume_i| / Raw_Volume_i)` where i runs over all bins. Note the per-bin normalization by actual volume.

2. **Percentage error** (volume percentage): `(1/N) * sum_i(|Predicted_Percentage_i - Actual_Percentage_i|)` -- this is a mean absolute error WITHOUT normalization (since percentages are already normalized).

Without these formulas, a developer could easily implement the wrong metric (e.g., use percentage error for raw volume, or add normalization to the percentage metric) and then wonder why their numbers don't match the paper's benchmarks.

**Recommendation:** Add a "Metrics" subsection in Validation that provides the exact formulas for both MAPE and percentage prediction error, citing p.17 of the paper. Also clarify the aggregation: is MAPE computed per-symbol and then the median taken across symbols? Or is it computed across all (symbol, day, bin) triples? The paper aggregates "across the 500-name universe over 250 trading days" with "26 forecasts for each of the 15-minute intervals" (p.20), suggesting per-symbol-day aggregation followed by cross-symbol median.

---

### M5. No discussion of how the inter-day ARMA is updated within the out-of-sample period

**Spec section:** Functions 2 and 6, Calibration section (lines 593-596).

**Problem:** The spec describes fitting the inter-day ARMA on a training window and then forecasting one step ahead. But it does not clearly describe what happens on day t+1 after the actual volume for day t becomes available:

- Does the ARMA model get re-estimated (full MLE) daily? The Calibration section says "Weekly (every 5 trading days): inter-day ARMA models (full re-estimation with AICc selection)" and "Between inter-day re-estimations: update ARMA state by conditioning on new daily observations without re-running MLE."
- The "conditioning" / state update step is critical but not described in the pseudocode. For an ARMA(p,q) model, "conditioning on new observations" means computing the new residual e[t] = V[t] - predicted[t], updating the MA state, and shifting the AR observation buffer. This is not trivial and is absent from the pseudocode.
- The distinction between "re-estimation" (new MLE fit, possibly new order selection) and "state update" (fixed parameters, update recursion state) needs to be made explicit in the pseudocode, not just in the calibration text.

**Recommendation:** Add a `UpdateInterDayState(model, new_observation)` function to the pseudocode that shows exactly how the ARMA state is updated when a new daily observation arrives. Clarify that the model parameters (AR/MA coefficients) remain fixed between re-estimation cycles, but the internal state (recent residuals, observation buffer) is updated.

---

## Minor Issues

### m1. N_hist = 21 is stated as "recommended" but the paper treats N as tunable

**Spec section:** Parameters table, line 528.

The paper says "One chooses the number of days of historical data to use, a variable that we shall call N" (p.16). Exhibit 1 shows "Prior 21 days" which the spec takes as N_hist = 21. This is a reasonable default, but the paper explicitly treats N as a tunable parameter without stating 21 is optimal. The sensitivity should be noted as "Medium-High" since the historical average is one of three forecast components and its accuracy directly depends on this window length.

---

### m2. Deviation bounds -- "a separate method" is understated

**Spec section:** Parameters table (max_deviation), Function 7 step 6.

The paper states: "we developed a separate method for computing the deviation bounds" (p.19), indicating the actual implementation uses adaptive/proprietary bounds, not the fixed 10% from Humphery-Jenner (2011). The spec correctly notes the 10% comes from Humphery-Jenner but should more strongly flag that the fixed 10% is a simplification of what the paper actually used. The developer should be aware that this parameter is a likely candidate for tuning and that the paper's published results may have used tighter or adaptive bounds.

---

### m3. The "fewer than 11 terms" constraint should be implementable

**Spec section:** Function 3 (FitIntraDayARMA), line 148.

The spec interprets "fewer than 11 terms" as an empirical observation rather than a hard constraint. While this interpretation is defensible, making it an optional hard constraint is trivial (check combined parameter count after both ARMA models are selected) and would provide a useful safety valve against overfitting. The spec should recommend implementing it as a soft constraint: if the combined parameter count exceeds 10, log a warning rather than silently accepting.

---

### m4. Intraday ARMA fitting window inconsistency

**Spec section:** Function 3 (FitIntraDayARMA), parameter N_intraday.

The parameter table says N_intraday = 21 trading days ("1 month"), and the paper says "rolling basis over the most recent month" (p.18). However, one trading month is approximately 21 days, while one calendar month varies (20-23 trading days). The spec should clarify that "21 trading days" is the implementation choice and note that the paper's "month" is ambiguous between calendar and trading time.

---

### m5. Missing fallback for regime classification at beginning of day

**Spec section:** Function 4 (ClassifyRegime), line 163.

The spec uses `DEFAULT_REGIME` when `i < min_regime_bins` but does not define what `DEFAULT_REGIME` is. This should be specified -- the most natural choice is the middle regime (regime 1 for n_regimes=3), since it represents "typical" volume, but the spec leaves this to the developer's discretion.

---

### m6. Pre-market static forecast does not use regime weights

**Spec section:** Function 9 (DailyOrchestration), lines 435-438.

The pre-market forecast loop (STEP 1) calls `ForecastRawVolume(s, t, i, current_bin=0)`, which triggers `ClassifyRegime(s, t, 0, ...)`. With `current_bin=0 < min_regime_bins`, the regime is `DEFAULT_REGIME`. This means all pre-market forecasts use the same default regime weights. Is this intentional? The spec should explicitly note that pre-market forecasts cannot be regime-adapted (since no intraday volume has been observed yet) and that the default regime should be chosen to minimize pre-market forecast error. This is a design point that affects Model B's baseline expectations.

---

### m7. Exhibit 10 reports "7%-10% reduction within each category" but the spec says "7-10% across" categories

**Spec section:** Validation > Expected Behavior, line 618.

The paper says "we achieved a 7%-10% reduction within each category and a 9.1% reduction across all simulated orders" (p.23). The spec says "Per-category reductions: 7-10% across Dow 30, midcap, and high-variance stock groups." The meaning is the same but the phrasing could be clearer: the 7-10% is the range of reductions when computed separately for each stock category, while 9.1% is the aggregate. The current phrasing is not wrong but could be tightened.

---

## Citation Verification

I verified the following citations against the paper:

| Claim | Spec Citation | Verified? | Notes |
|-------|-------------|-----------|-------|
| 24% median MAPE reduction | p.20 | Yes | "reduce the median volume error by 24%" |
| 29% bottom-95% MAPE reduction | p.20 | Yes | "the bottom 95% of the distribution by 29%" |
| AICc from Hurvich & Tsai | p.17 | Yes | Correctly cited |
| "AR lags with a value less than five" | p.18 | Yes | Exact quote |
| "fewer than 11 terms" | p.18 | Yes | "a dual ARMA model having fewer than 11 terms" |
| "trailing six months" deseasonalization | p.17 | Yes | "dividing by the intraday amount of volume traded in that bin over the trailing six months" |
| "rolling basis over the most recent month" | p.18 | Yes | Exact quote |
| Exhibit 9 values (0.00874 vs 0.00808) | p.23, Exhibit 9 | Yes | Matches table exactly |
| Exhibit 10 values (9.62 vs 8.74 bps) | p.23, Exhibit 10 | Yes | Matches table exactly |
| Paired t-test 2.34 | p.23, Exhibit 10 | Yes | Matches |
| Exhibit 3 (220.9, R^2=0.5146) | p.20 | Yes | Matches table |
| Exhibit 5 (454.3, R^2=0.5886) | p.21 | Yes | Matches table |
| "depart no more than 10%" | p.24 | Yes | Humphery-Jenner reference |
| "80% of the day's volume" | p.24 | Yes | Humphery-Jenner reference |
| "Prior 21 days" | Exhibit 1, p.18 | Yes | Exhibit caption |
| "Prior 5 days" | Exhibit 1, p.18 | Yes | Exhibit caption |
| "we could apply" (aspirational) | p.19 | Yes | Exact word is "could" |
| No-intercept regressions | p.19 | Yes | "without the inclusion of a constant term" |
| 600+ VWAP orders, 10% of 30-day ADV | p.23 | Yes | Matches text |
| Custom curves for special days | p.18 | Yes | Correct |

All major citations verified. No misrepresentations found.

---

## Overall Assessment

The draft is strong -- it captures the paper's methodology accurately, flags proprietary gaps appropriately, and provides well-reasoned Researcher inferences. The major issues are primarily about implementability gaps (M1, M3, M4, M5) and one substantive design question (M2). None require fundamental restructuring. A revised draft addressing these issues would be ready for implementation.
