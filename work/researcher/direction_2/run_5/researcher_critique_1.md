# Critique of Implementation Specification: PCA Factor Decomposition for Intraday Volume (BDF Model)

## Critique of impl_spec_draft_1.md

**Overall Assessment:** This is a high-quality implementation specification. The algorithm
is well-decomposed into functions, the pseudocode is detailed and directly translatable to
code, citations are thorough and mostly accurate, and researcher inferences are clearly
labeled. The SVD-based formulation for factor extraction is a significant improvement over
eigendecomposition-based approaches, avoiding unnecessary dimension branching. The spec
correctly identifies the ARMA(1,1) mislabel in BDF 2008.

I identified 3 major issues, 4 medium issues, and 5 minor issues.

---

## Major Issues

### M1. `run_dynamic_execution` conflates batch and event-driven patterns (Function 9, lines 638-696)

**Problem:** The function signature accepts `observed_turnover` as a list parameter, and
the docstring says it "grows as bins complete; initially empty." But the function contains
a `for j in range(k)` loop that iterates over all bins in a single call. Inside the loop,
`observed_turnover[j]` is accessed conditionally (line 685), but the list was passed in at
call time and does not grow during iteration.

A developer would face a fundamental design question: should this function be called once
at market open (in which case `observed_turnover` is empty and never updates), or should
it be called repeatedly as each bin completes (in which case the for-loop is wrong)?

**Evidence:** BDF 2008, Section 4.2.2 describes an event-driven process: "we include the
information on intraday volume after each time interval" and re-forecast remaining bins.
This is inherently a per-bin callback, not a batch loop.

**Recommendation:** Either:
(a) Restructure as an event-driven function `execute_one_bin(model, stock_idx, bin_idx,
shares_remaining, last_specific)` that is called once per bin by an external event loop,
returning `(shares_to_trade, updated_last_specific)`. This is the clean design that
matches BDF's description. Or:
(b) Keep the batch loop but pass a callback/generator that yields actual turnover values
as they become available, and clearly document the simulation semantics.

Option (a) is strongly preferred for implementability.

### M2. Validation metrics do not distinguish evaluation protocols (Validation section, lines 930-996)

**Problem:** The spec presents validation benchmarks from both Szucs 2017 Table 2a
(per-stock MSE/MAPE) and BDF 2008 Table 2 (portfolio MAPE) without clearly specifying the
evaluation protocol used to generate these numbers. The critical detail: Szucs's MSE/MAPE
numbers are computed using **one-step-ahead intraday updating** -- the forecast for bin t
uses actual observed volumes for bins 0..t-1. This is stated in Szucs 2017, Section 3:
"While parameters are updated daily, the information base for the forecast is updated
every 15 minutes. This is because 26 data points are to be forecasted each day. Although
the parameters of the models are unchanged during the day, it makes sense to take
advantage of the actuals that unfold during the day. This approach is often called
one-step-ahead forecasting."

A developer who implements the model and evaluates using static next-day forecasts (all 26
bins predicted at market open without intraday updates) would get much worse numbers and
incorrectly conclude the implementation is wrong.

**Recommendation:** Add an explicit "Evaluation Protocol" subsection that specifies:
1. For each forecast day, estimate the model on the prior L-day window.
2. For bin j=0: forecast using `last_specific = model["specific"][-1, stock_idx]`.
3. For bin j>0: forecast using `last_specific = actual_turnover[j-1] - common_forecast[j-1, stock_idx]`.
4. Compute MSE and MAPE over all (stock, day, bin) triplets.
5. The Szucs Table 2a numbers use this protocol. The BDF Table 2 VWAP execution numbers
   use the dynamic execution strategy with an additional VWAP weight computation step.

### M3. Incorrect citation for BDF_SETAR vs U-method pairwise MAPE (Sanity Check 9, line 1073)

**Problem:** The spec states: "BDF_SETAR beats U-method 32/1 by MAPE." This is
incorrect. Szucs 2017, Table 2c shows BDF_SETAR row, U column as **33/0** -- BDF_SETAR
has lower MAPE than the U-method on all 33 stocks, with 0 losses.

The 32/1 figure is the BDF_AR vs U-method comparison (correct for BDF_AR). The spec
appears to have copied the BDF_AR result for BDF_SETAR.

**Evidence:** Szucs 2017, Table 2c (MAPE pairwise comparison):
- BDF_AR row, U column: 32/1 (BDF_AR wins 32, loses 1)
- BDF_SETAR row, U column: 33/0 (BDF_SETAR wins all 33)

**Fix:** Change line 1073 to: "BDF_SETAR beats U-method 33/0 by MAPE."

---

## Medium Issues

### N1. Missing specification for first-bin forecast in evaluation context (Function 6-7)

**Problem:** The spec clearly describes how `last_specific` is initialized for dynamic
execution (line 662: `model["specific"][-1, stock_idx]`). However, it does not explicitly
state how the forecast for bin j=0 of the next day is produced for validation purposes.

The question: for evaluating forecast accuracy (MSE/MAPE as in Szucs Table 2a), what is
the forecast for the first bin of the day? There are two possibilities:
(a) Use the common component forecast only: `x_hat[0] = c_forecast[0, i]` (since no
intraday data is available yet for the specific component).
(b) Use common + AR/SETAR specific forecast initialized from the last in-sample residual:
`x_hat[0] = c_forecast[0, i] + forecast_specific(ts_params[i], e_hat[-1, i])`.

The spec implies (b) based on the dynamic execution function, but this should be stated
explicitly in the validation protocol since the first-bin forecast quality significantly
affects overall metrics.

**Recommendation:** Add a note in the Validation section specifying that option (b) is
used, consistent with BDF 2008 Section 2.3's conditioning on end-of-estimation-window
information.

### N2. Overnight gap treatment in AR/SETAR is acknowledged but implications not quantified (lines 377-383)

**Problem:** The spec correctly notes that treating the series as contiguous across day
boundaries introduces periodic ACF spikes at multiples of k (BDF 2008, Fig. 2). However,
it does not address a practical implementation question: should the AR(1)/SETAR model be
fit on the contiguous series (last bin of day d followed by first bin of day d+1), or
should it be fit only on within-day bins (treating each day as an independent short
series)?

BDF and Szucs both treat the series as contiguous (implicit in their formulation of a
single AR process on the full P-length series). But a developer might reasonably wonder
whether the overnight gap degrades AR coefficient estimation, especially since the
first-bin-of-day to last-bin-of-day transition has different dynamics than consecutive
intraday bins.

**Recommendation:** Add an explicit statement: "The AR(1)/SETAR is fit on the full
contiguous series of length P = L*k, treating overnight gaps as ordinary lag-1 transitions.
This is the approach used by both BDF 2008 and Szucs 2017. Do not segment the series by
day." This removes ambiguity for the developer.

### N3. SETAR grid search duplicates computation in Phase 1 vs Phase 2 (Function 5, lines 439-506)

**Problem:** The two-phase SETAR estimation (grid search for SSR in Phase 1, then
re-estimation in Phase 2) is a good efficiency optimization but introduces unnecessary
code duplication. Phase 2 repeats the exact same OLS computation as Phase 1 for the
winning tau. A developer might introduce subtle bugs by having two independent OLS
implementations that must agree.

**Recommendation:** Refactor the pseudocode to have Phase 1 store the OLS coefficients
(beta1, beta2) alongside the SSR for each candidate tau, or at minimum note that Phase 2
is a re-computation of Phase 1's winning case. A simpler approach: store the winning
beta1, beta2, resid1, resid2 during Phase 1's sweep, eliminating Phase 2 entirely. This
is a minor efficiency tradeoff (storing a few extra arrays) but reduces code paths.

### N4. Missing practical guidance on total_ss computation (Function 2, line 199)

**Problem:** The spec computes `total_ss = sum(X ** 2)` which requires materializing X^2
in memory. For large P and N (e.g., P=13000, N=500 for a 500-stock universe with k=26,
L=500 days), this is a (13000, 500) float64 array = ~50 MB, which is manageable. But the
comment doesn't note that `total_ss` can be computed incrementally or via
`np.sum(X * X)` without creating a temporary, or via `np.linalg.norm(X, 'fro')**2`.

More importantly, if the developer uses `scipy.sparse.linalg.svds`, the full matrix X
must still be in memory for the total_ss computation. The spec should note that the
truncated SVD does NOT avoid loading X into memory -- it only avoids forming XX' or X'X
explicitly.

**Recommendation:** Add a brief note on memory: "The full turnover matrix X must reside
in memory for both the SVD and total_ss computation. For practical cross-sections (N <
500, L < 250), this is under 100 MB and not a concern. Use
`np.linalg.norm(X, 'fro')**2` for total_ss."

---

## Minor Issues

### P1. Szucs Table 2b MSE pairwise for BDF_AR vs U is 33/0, not stated in the spec's Sanity Check 8 (line 1069)

The spec says "BDF should produce lower MSE than U-method for all or nearly all stocks.
Szucs 2017, Table 2b: BDF_AR beats U-method 33/0 by MSE; BDF_SETAR beats U-method 33/0
by MSE." This is correct per Table 2b. No issue here -- just confirming this citation
checks out.

### P2. Parameter table missing sigma2 output from AR(1) and SETAR (Parameter table, lines 863-876)

The parameter table lists model inputs (k, L, r_max, etc.) but does not list the
estimated outputs like sigma2 (AR(1) innovation variance) or sigma2_1/sigma2_2 (SETAR
per-regime variances). These are computed by fit_ar1 and fit_setar but never used in the
forecasting functions -- `forecast_specific` uses only the point forecast, not the
variance. The spec should either:
(a) Note that sigma2 values are computed for diagnostic purposes only (e.g., testing
residual normality, comparing regime volatilities) and are not used in point forecasting.
(b) Remove sigma2 from the return values if they serve no purpose.

Option (a) is better since sigma2 is useful for confidence intervals and model diagnostics.

### P3. Turnover range estimate may be too narrow (Type summary, line 833)

The spec states "Turnover values: float64 (typical range 1e-5 to 1e-1 for individual
15-20 min bins)." For highly liquid large-cap stocks (e.g., AAPL), daily turnover can
reach 1-2% of shares outstanding, so a single 15-minute bin at market open could be
0.002-0.005 (2e-3 to 5e-3). For less liquid stocks, a quiet midday bin might be 1e-5.
The upper range of 1e-1 (10% turnover in a single 15-minute bin) would be extremely
unusual -- this might occur only during a trading halt release or extreme event. A more
realistic upper bound for normal conditions would be 1e-2.

This is not a correctness issue (the code handles any float64), but realistic range
estimates help developers set appropriate sanity-check thresholds.

### P4. Common component can have negative values despite volume non-negativity (Data Flow, line 838)

The spec states "Common component C_hat: float64, shape (P, N), non-negative." However,
the common component is computed as C_hat = F_hat @ Lambda_hat.T, which is a product of
real-valued matrices. There is no guarantee that C_hat is non-negative. PCA does not
enforce non-negativity constraints. In practice, for well-behaved turnover data, C_hat
will typically be positive (since turnover is positive and the first factor captures the
level), but it is not mathematically guaranteed.

**Fix:** Change "non-negative" to "typically positive but not guaranteed non-negative."
This affects the negative forecast handling in `compute_vwap_weights`: if C_hat itself is
negative for some bin, the negative forecast issue arises even without a large negative
specific component.

### P5. `u_method_benchmark` uses nested loops instead of vectorized reshape (Function 11, lines 758-781)

The U-method benchmark is implemented with explicit nested loops over bins and days. The
spec's own `forecast_common` function (Function 3) uses a vectorized reshape+mean
approach. For consistency and clarity, the U-method should use the same pattern:

```
stock_data = all_turnover_data[start_row:end_row, stock_idx]  # (L*k,)
stock_3d = reshape(stock_data, (L, k))
forecast = mean(stock_3d, axis=0)  # (k,)
```

This also eliminates potential off-by-one errors in the manual loop indexing.

---

## Citation Verification Summary

All major citations were verified against the original papers. Results:

| Spec Claim | Source | Verified? |
|-----------|--------|-----------|
| BDF Eq. 10 is AR(1) not ARMA(1,1) | BDF 2008 Eq. 10 text | Yes -- confirmed no MA term |
| Szucs Eq. 5 confirms AR(1) | Szucs 2017 Eq. 5 | Yes |
| BDF Eq. 11 SETAR model | BDF 2008 Eq. 11 | Yes |
| Szucs Eq. 6-7 SETAR + indicator | Szucs 2017 Eq. 6-7 | Yes |
| BDF Eq. 9 common forecast | BDF 2008 Eq. 9 | Yes |
| BDF Eq. 8 combined forecast | BDF 2008 Eq. 8 | Yes |
| No centering in BDF | BDF 2008 Eq. 4-6, Bai 2003 | Yes -- no demeaning step |
| Bai-Ng IC for factor selection | BDF 2008 Section 2.2 | Yes -- "criteria of Bai and Ng (2002)" |
| IC_p2 variant choice | Not specified in BDF | Correct -- BDF does not name IC variant |
| BDF k=25, 20-min bins | BDF 2008 Section 3.1, Fig. 1 | Yes |
| Szucs k=26, 15-min bins | Szucs 2017 Section 2 | Yes |
| Szucs turnover = V/TSO | Szucs 2017 Section 2 | Yes -- "x_t = V_t/TSO_t" |
| BDF turnover = shares/float | BDF 2008 Section 2.2 | Yes |
| BDF half-days excluded | BDF 2008 Section 3.1 | Yes -- Dec 24 and 31, 2003 |
| Szucs non-zero volume filter | Szucs 2017 Section 2 | Yes |
| Szucs Table 2a MSE/MAPE | Szucs 2017 Table 2a | Yes -- values match |
| Szucs Table 2b MSE pairwise | Szucs 2017 Table 2b | Yes -- BDF_AR 33/0, BDF_SETAR 33/0 vs U |
| Szucs Table 2c MAPE pairwise | Szucs 2017 Table 2c | **MISMATCH** -- BDF_SETAR vs U is 33/0 not 32/1 |
| BDF Table 2 portfolio MAPE | BDF 2008 Table 2 | Yes -- values match |
| BDF Table 2 VWAP execution MAPE | BDF 2008 Table 2 | Yes -- values match |
| SETAR better 36/39 stocks | BDF 2008 Section 3.2 | Yes |
| ~10% TE reduction | BDF 2008 Section 4.3.3 | Yes |
| Static execution dominated | BDF 2008 Section 4.2 | Yes -- page 1717 quote matches |
| Szucs 2648 forecast days | Szucs 2017 Section 3 | Yes -- 2668 total - 20 window = 2648 |
| Szucs MSE* formula | Szucs 2017 Eq. 14 | Yes |
| Computational cost ~2 hours | Szucs 2017 Section 5 | Yes -- "about two hours to run" |
| BDF footnote 5: U-method = Eq. 9 on x | BDF 2008 footnote 5 | Yes |

---

## Summary of Required Changes

**Must fix (Major):**
1. Restructure `run_dynamic_execution` to be event-driven or clearly document simulation semantics (M1).
2. Add explicit "Evaluation Protocol" subsection defining the one-step-ahead intraday-updating methodology used for Szucs benchmarks (M2).
3. Fix BDF_SETAR vs U-method MAPE pairwise from 32/1 to 33/0 (M3).

**Should fix (Medium):**
4. Clarify first-bin forecast protocol for validation (N1).
5. Add explicit statement about contiguous series treatment for AR/SETAR (N2).
6. Simplify SETAR two-phase estimation to avoid code duplication (N3).
7. Add memory/computation note for total_ss (N4).

**Nice to fix (Minor):**
8. Clarify sigma2 is diagnostic-only (P2).
9. Adjust turnover range upper bound from 1e-1 to 1e-2 (P3).
10. Change C_hat from "non-negative" to "typically positive" (P4).
11. Vectorize U-method benchmark for consistency (P5).
