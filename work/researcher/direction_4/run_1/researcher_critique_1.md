# Critique of Implementation Specification Draft 1: Dual-Mode Volume Forecast (Raw + Percentage)

## Summary Assessment

The draft is well-structured and generally faithful to the Satish et al. (2014) paper. The pseudocode is reasonably detailed, and the parameter table is comprehensive. However, there are several issues ranging from a significant misattribution of a performance figure, through algorithmic ambiguities that would cause a developer to make incorrect implementation choices, to missing details that the paper does provide. I identify 4 major issues and 8 minor issues.

---

## Major Issues

### M1. Misattributed "24% median MAPE reduction" headline figure

**Location:** Overview (line 13), Validation > Expected Behavior (line 422), Paper References table (line 524).

The draft states: "24% median MAPE reduction for raw volume" and cites "Satish et al. 2014, Exhibits 6 and 10" and "p.20, Exhibit 6."

**Problem:** The paper (p.20) says "we reduce the median volume error by 24% and the average of the bottom 95% of the distribution by 29%." Exhibit 6 shows these reductions bin-by-bin. This is correct. However, the Overview (line 13) bundles this with "9.1% VWAP tracking error reduction" as if both come from the same model. In fact, the 24% MAPE reduction comes from the **raw volume model (Model A, the "dual ARMA" approach)**, while the 9.1% VWAP tracking error reduction comes from the **volume percentage model (Model B, the "dynamic VWAP curve")** applied in the FlexTRADER simulator (Exhibit 10). The draft's Overview conflates these, which could mislead a developer into thinking Model A alone produces the VWAP improvement. The Validation section (line 433) correctly separates them, but the Overview should be fixed for consistency.

**Recommendation:** Rewrite the Overview to clearly separate: Model A achieves 24% median MAPE reduction for raw volume; Model B achieves 7.55% median reduction in absolute volume percentage error; and Model B's dynamic curve reduces VWAP tracking error by 9.1% in simulation.

### M2. Inter-day ARMA training window is unspecified -- draft silently assumes a long window without flagging the gap

**Location:** Pseudocode, Model A Component 2 (lines 53-68).

The pseudocode says `daily_series_i = [V[s, d, i] for d in training_window]` and comments "e.g., 250 days or more for model selection." The paper (p.17) says the inter-day ARMA is "a per-symbol, per-bin ARMA(p, q) model" and discusses AICc selection, but **does not specify the training window length** for this component. Exhibit 1 shows "Prior 5 days" feeding into "ARMA Daily," which is much shorter than 250 days and likely refers to the forecast horizon or recent conditioning data, not the full estimation window.

**Problem:** The draft uses "250 days or more" as an example value without any paper citation, and does not include this as a parameter in the Parameters table. A developer would not know what window to use. The "Prior 5 days" label in Exhibit 1 is ambiguous -- it could mean the ARMA is fit on only 5 days of data (which would be too short for reliable estimation of ARMA(5,5)), or it could mean something else entirely (e.g., the number of recent days used for forecasting state initialization).

**Recommendation:** 
1. Add an explicit `N_arma_interday` parameter to the Parameters table with a note that the paper does not disclose this value.
2. Flag the Exhibit 1 "Prior 5 days" ambiguity explicitly in the pseudocode.
3. Provide a researcher inference for a reasonable default (e.g., 60-250 days) with justification.

### M3. Intraday ARMA series construction -- concatenation across days creates artificial discontinuities

**Location:** Pseudocode, Model A Component 3 (lines 86-101).

The pseudocode concatenates deseasonalized intraday observations from multiple days into a single long series, then fits ARMA to it:
```
for d in (t - N_arma_intraday) .. (t - 1):
    for j in 1..I:
        intraday_series.append(V[s, d, j] / S[j])
for j in 1..current_bin:
    intraday_series.append(V_deseas[s, t, j])
```

**Problem:** This concatenation treats the last bin of day d and the first bin of day d+1 as adjacent observations in the ARMA series. But overnight gaps create structural breaks -- the volume in bin 26 (15:45-16:00) and bin 1 of the next day (9:30-9:45) are separated by 17.5 hours. Fitting ARMA to this concatenated series without any day-boundary treatment will introduce spurious autocorrelation at the day-boundary lags and could corrupt the AR coefficient estimates.

The paper (p.17-18) says "We compute this model on a rolling basis over the most recent month" but does not explicitly describe how day boundaries are handled. The Exhibit 1 diagram shows "Current Bin" and "4 Bins Prior to Current Bin" feeding the intraday ARMA, suggesting the model may use only the current day's observed bins (plus perhaps the end of the prior day) rather than a long concatenated series.

**Recommendation:**
1. Address day-boundary handling explicitly. Options include: (a) fitting ARMA only on the current day's observed bins plus a few lags from the prior day's close; (b) treating each day as a separate segment and using the 1-month window only for parameter estimation (not for continuous series construction); (c) inserting missing-value markers at day boundaries.
2. Reconcile the Exhibit 1 diagram (which shows only "4 Bins Prior to Current Bin") with the "rolling one-month window" description. These may refer to different aspects: the 1-month window for parameter estimation, and the 4-bin lookback for the actual forecast conditioning set.

### M4. Volume percentage model: V_total_est is undefined and creates circular dependency

**Location:** Pseudocode, Model B (lines 170-174).

The pseudocode uses `V_total_est` to compute actual percentages:
```
actual_pct[j] = V[s, t, j] / V_total_est
```
But `V_total_est` is never defined or computed. The draft acknowledges this as a "Known Limitation" (line 491-492) but still uses it in the pseudocode without providing a concrete computation.

**Problem:** A developer cannot implement this step. There are at least three possible definitions: (a) sum of Model A's full-day forecasts, (b) sum of observed bins so far plus Model A forecasts for remaining bins, (c) historical average daily volume. The choice significantly affects the surprise calculation and therefore the model's behavior.

**Recommendation:** 
1. Define V_total_est explicitly in the pseudocode. The most natural choice is: `V_total_est = SUM(V[s,t,j] for observed j) + SUM(V_hat_raw[s,t,j] for remaining j)`, which updates as more bins are observed. This is consistent with the coupling described in the Overview.
2. Address the circular dependency: Model B uses V_total_est from Model A, but if Model A is also being updated intraday, specify the update ordering (Model A first, then Model B).

---

## Minor Issues

### m1. MAPE definition mismatch

**Location:** Validation > Expected Behavior (line 422-424), Paper References (line 521).

The draft cites "24% median MAPE reduction" but does not define MAPE precisely. The paper (p.17) defines MAPE as:

    MAPE = 100% x (1/N) x SUM(|Predicted_Volume - Raw_Volume| / Raw_Volume)

This is an average across bins, not across days. The draft's reference to "across all intraday intervals" (line 423) is ambiguous -- it could mean the MAPE is computed across all bins pooled, or it's the average of per-bin MAPEs. From Exhibit 6, the 24% is a summary across the per-bin results. The spec should define MAPE precisely so the developer implements the correct evaluation metric.

### m2. AICc formula is incomplete

**Location:** Pseudocode (line 62).

The draft gives: `AICc = AIC + 2*k*(k+1) / (n-k-1), where k = p+q+1 (including constant)`.

This is correct but incomplete. The AIC itself is not defined. The developer needs: `AIC = -2*log_likelihood + 2*k`, and then the correction term. Also, the definition of `n` (sample size) should be explicit -- is it the number of observations in the training window, or the effective sample size after differencing (if differencing is used)?

### m3. The "fewer than 11 terms" constraint is ambiguously implemented

**Location:** Pseudocode (lines 97-99).

The draft implements the constraint as:
```
subject to: p + q + 1 + best_p + best_q + 1 < 11  # total terms < 11
```

**Problem:** The paper (p.18) says "we fit each symbol with a dual ARMA model having fewer than 11 terms." It is unclear whether "terms" means parameters (p+q+constant for each model) or just AR+MA order terms (p+q). If "terms" means coefficients, then the constant terms should be counted; if "terms" means lag orders, they should not. The constraint as written counts constants (the +1 terms), which may over-restrict the search space.

**Recommendation:** Note the ambiguity and recommend testing both interpretations.

### m4. Regime classification timing ambiguity

**Location:** Pseudocode, Component 4 (lines 117-127).

The pseudocode computes `cum_vol_today = SUM(V[s, t, j] for j in 1..current_bin)` and then compares to historical cumulative volumes. But for the first bin (current_bin=1), the cumulative volume is just one observation, which will have high variance in percentile rank. Early-day regime classification is unreliable.

**Recommendation:** Specify a minimum number of observed bins before regime switching activates (e.g., only switch after bin 3 or 4). Before that, use the "medium" regime (or equal weights) as default.

### m5. Deseasonalization denominator edge case

**Location:** Pseudocode (line 80).

`V_deseas[s, t, j] = V[s, t, j] / S[j]` -- if S[j] is zero (or near-zero), this creates division-by-zero or numerical explosion. The Edge Cases section (line 461) mentions a floor of 1 share, but this should be integrated into the pseudocode at the point of computation rather than left to the Edge Cases section.

### m6. Rolling regression specification for Model B is too vague

**Location:** Pseudocode, Model B Step 3 (lines 183-193).

The regression is described as OLS of surprise on K lagged surprises, but critical details are missing:
- How many days of history form the regression training set? The paper says "rolling regression" but the window length is not specified.
- Is the regression fit per-bin (separate regression for each bin position i) or pooled across all bins?
- The Humphery-Jenner (2011) framework has "self-updating deviation bounds" -- the draft's fixed max_deviation=0.10 may not capture this. The paper (p.18-19) says they "developed a separate method for computing deviation bounds." This suggests the bounds may be adaptive, not fixed at 10%.

**Recommendation:** Flag these as researcher inferences that need in-sample optimization.

### m7. Model B renormalization (Step 6) may conflict with safety constraints

**Location:** Pseudocode, Model B Step 6 (lines 216-219).

After applying the deviation constraint and switch-off threshold, the code renormalizes:
```
pct_hat[s, t, i] = pct_raw * (remaining_pct / remaining_hist)
```

**Problem:** This renormalization can undo the safety constraints. If the switch-off threshold triggered (pct_raw = pct_hist[i]), the renormalization still scales it by `remaining_pct / remaining_hist`, which may differ from 1.0. The intent of the switch-off is to "return to historical curve," but renormalization alters the historical curve proportions.

**Recommendation:** Clarify the interaction between safety constraints and renormalization. If the switch-off has triggered, the renormalization should distribute `remaining_pct` proportionally to the *remaining* historical percentages for all future bins, not just adjust the current bin.

### m8. Missing comparison context from Chen et al. 2016

**Location:** Validation > Comparison benchmark (lines 437-439).

The draft cites Chen et al. MAPE of 0.46 and VWAP of 6.38 bps, but these are averages across 30 securities on different markets and time periods. The draft correctly notes the comparison is imperfect (line 439) but does not mention that Chen et al. (2016) specifically claim to outperform the Satish et al. approach (Chen et al. summary, line 149: "Satish et al. (2014) used a similar decomposition with ARMA models; this paper claims to outperform that approach as well"). This context is important for the developer to understand that the Kalman filter model may in fact be superior on the same data.

---

## Citation Verification

I verified the following citations against the paper:

| Claim | Cited Source | Verified? | Notes |
|-------|-------------|-----------|-------|
| 26 bins per day, 15-min | p.16 | Yes | Exact match |
| AICc from Hurvich and Tsai | p.17 | Yes | Exact match |
| Deseasonalization over trailing 6 months | p.17 | Yes | Exact match |
| Intraday ARMA on rolling 1-month window | p.18 | Yes | Exact match |
| AR lags < 5, total terms < 11 | p.18 | Yes | Exact match |
| Dynamic weight overlay with regime switching | p.18 | Yes | Exact match |
| Custom curves for special days | p.18 | Yes | Exact match |
| 10% deviation limit | p.19 (via Humphery-Jenner) | Yes | Exact match |
| 80% switch-off threshold | p.19 (via Humphery-Jenner) | Yes | Exact match, though paper says "p.24" in the Humphery-Jenner discussion |
| N_hist = 21 from Exhibit 1 | Exhibit 1 | Yes | Exhibit 1 shows "Prior 21 days" |
| 24% median MAPE reduction | p.20, Exhibit 6 | Yes | Exact match |
| 29% bottom 95% MAPE reduction | p.20 | Yes | Exact match |
| Exhibit 9 percentage results | p.23, Exhibit 9 | Yes | Values match exactly |
| Exhibit 10 VWAP results | p.23, Exhibit 10 | Yes | 9.62 vs 8.74 bps, 9.1% reduction |
| R^2=0.51, coef=220.9 (Dow 30) | p.20, Exhibit 3 | Yes | Exact match (R^2=0.5146) |
| R^2=0.59, coef=454.3 (high-var) | p.21, Exhibit 5 | Yes | Exact match (R^2=0.5886) |
| Chen et al. MAPE 0.46 | Chen et al. Table 3 | Yes | Per summary |
| Chen et al. VWAP 6.38 bps | Chen et al. Table 4 | Yes | Per summary |

All citations check out. The paper references are accurate and well-sourced.

---

## Completeness Assessment

### Present and adequate:
- Model A pseudocode (with caveats noted above)
- Model B pseudocode (with caveats noted above)
- Weight calibration procedure
- Data flow description
- Parameter table with sources
- Initialization procedure
- Edge cases (thorough)
- Known limitations (thorough and honest about proprietary gaps)
- Paper references table (comprehensive)

### Missing or incomplete:
1. **V_total_est computation** (Major -- M4 above)
2. **Day-boundary handling in intraday ARMA** (Major -- M3 above)
3. **Inter-day ARMA training window parameter** (Major -- M2 above)
4. **Per-bin vs pooled regression for Model B** (Minor -- m6 above)
5. **ARMA estimation failure rates**: The paper does not discuss convergence failures, but for 500 stocks x 26 bins x 36 order combinations = 468,000 fits, some will fail. The Edge Cases section (line 476) mentions a fallback but does not specify how to detect failure (convergence tolerance, maximum iterations, Hessian check).
6. **Forecast update sequencing**: When Model A is updated intraday (after each new bin), the pseudocode does not specify whether ARMA models are re-estimated or only the forecast is updated by conditioning on new observations. Re-estimation at every bin would be expensive; reconditioning (updating the ARMA state with new data without re-estimating parameters) is the likely intent but is not stated.

---

## Overall Assessment

The draft is a solid first attempt that captures the core structure of the Satish et al. (2014) system. The main weaknesses are: (1) the intraday ARMA series construction needs day-boundary handling, (2) the V_total_est dependency in Model B needs an explicit definition, (3) the inter-day ARMA training window needs to be parameterized, and (4) the Overview conflates which model produces which performance gain. The citation work is excellent -- every claim I checked matched the source material. The Researcher Inference markings are appropriate and honest about proprietary gaps.

A revised draft addressing the 4 major issues would be ready for implementation. The minor issues are real but unlikely to block a competent developer.
