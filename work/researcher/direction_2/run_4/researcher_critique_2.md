# Critique of Implementation Specification Draft 2: PCA Factor Decomposition (BDF)

**Direction 2, Run 4 — Critic Review, Round 2**

## Summary Assessment

Draft 2 thoroughly addresses all 3 major and 7 minor issues raised in Critique 1.
The revision quality is excellent: each fix goes beyond the minimum requested, adding
context and cross-references that strengthen the document overall. The spec is now
implementation-ready.

I identify **0 major issues** and **3 minor issues** below. None are blocking.

---

## Resolution of Critique 1 Issues

| Issue | Severity | Resolved? | Assessment |
|-------|----------|-----------|------------|
| M1: TSO vs. float | Major | **Yes** | Comprehensive. Lines 104-129 document the discrepancy, justify TSO, and explicitly warn about benchmark comparability. Validation section (lines 1018-1021, 1045-1047) correctly distinguishes which benchmarks are comparable. Parameter sensitivity raised to "Medium." |
| M2: ARMA vs. AR justification | Major | **Yes** | Three independent arguments (lines 306-327) are well-structured: (a) equation authority, (b) Szucs independent confirmation, (c) OLS/MLE consistency check. The OLS/MLE argument is particularly valuable as it provides a formal consistency proof. |
| M3: Dynamic execution explanation | Major | **Yes** | Dedicated subsection "Dynamic Execution Weight Normalization" (lines 736-772) clearly explains the normalization role. The concrete examples of over/under-trading (lines 758-765) are especially helpful for a developer. Also updated in Model Description (lines 67-74) and Data Flow (lines 900-903). |
| m1: IC_p2 SVD unification | Minor | **Yes** | Cleanly unified via singular values (lines 170-179), eliminating the P>=N branch. |
| m2: Variance DoF | Minor | **Yes** | Changed to MLE-consistent T-1 denominator (line 348) with explicit justification (lines 341-346). Also applied consistently to SETAR (lines 441, 448). |
| m3: SETAR notation | Minor | **Yes** | Parentheticals now read "(regime 1 AR coeff)" etc. (lines 459-462). Clear and unambiguous. |
| m4: MAPE computation | Minor | **Mostly** | Added `compute_mape` (lines 777-817) and `compute_mse_star` (lines 820-853). Per-stock MAPE matches Szucs Eq. (2). However, portfolio-level MAPE (BDF Table 2) function is not provided. See m2-new below. |
| m5: Overnight transition | Minor | **Yes** | Comment at lines 630-639 clearly explains the overnight gap and its implications, with BDF Fig. 2 reference. |
| m6: Stationarity check | Minor | **Yes** | Implemented as explicit code (lines 605-615) with U-method fallback defined (lines 617-621, 653-655). The fallback is now unambiguous: raw turnover time-of-day average. |
| m7: day_index precondition | Minor | **Yes** | Assert with informative error message (lines 702-705). |

---

## New Minor Issues

### m1-new. `forecasts_history` documented as return value but never constructed

**Spec section:** `run_dynamic_execution` return documentation (lines 572-576) and function body (line 681)

**Problem:** The function signature documents two return values:
```
Returns:
    weights_history: list of length k, ...
    forecasts_history: list of forecasts at each step
```

But the function body only constructs and returns `weights_history` (line 681):
```
return weights_history
```

The `forecasts_history` list is never initialized, populated, or returned. A developer will either implement it (adding code not in the spec) or wonder if it was intentionally omitted.

**Action:** Either (a) add `forecasts_history` construction in the loop body (accumulate `x_forecasts` for each stock at each step) and return it alongside `weights_history`, or (b) remove it from the return documentation. Option (a) is preferred since forecasts are needed for validation (computing MAPE against actuals).

### m2-new. Portfolio-level MAPE function not provided

**Spec section:** Validation utilities (Group 5) and Expected Behavior (lines 1035-1054)

**Problem:** The spec lists portfolio-level MAPE benchmarks from BDF 2008 Table 2 (e.g., PCA-SETAR Mean = 0.0752), but the validation utilities only include per-stock MAPE (`compute_mape`) and MSE* (`compute_mse_star`). There is no function for computing portfolio-level MAPE.

BDF 2008's portfolio MAPE is computed differently from per-stock MAPE. BDF computes the MAPE of the portfolio-level forecast error — i.e., aggregate the forecast and actual across stocks first (using index weights), then compute MAPE on the aggregate. This differs from averaging per-stock MAPEs (Szucs approach) and produces much lower values (~0.08 vs ~0.40) due to diversification of idiosyncratic errors.

Without a `compute_portfolio_mape` function, a developer cannot reproduce the BDF Table 2 benchmarks, even if using float-denominated turnover. BDF 2008 Section 4.3.3 describes computing "the VWAP for the whole index based on the average of VWAP over equities" using "the same weights as used for construction of the index."

**Action:** Add a `compute_portfolio_mape` function that: (a) accepts per-stock forecasts, actuals, and portfolio weights, (b) computes weighted-average turnover forecast and actual at each bin, and (c) computes MAPE on the aggregated series. Note that portfolio weights may be equal-weight (1/N) or market-cap-weight (matching the index). BDF uses CAC40 index weights but does not provide them explicitly; equal-weight is a reasonable approximation.

### m3-new. VWAP execution cost benchmarks cannot be validated without price data

**Spec section:** Expected Behavior, VWAP execution cost table (lines 1056-1069)

**Problem:** The spec lists VWAP execution cost benchmarks from BDF 2008 Table 2 (e.g., dynamic PCA-SETAR Mean = 0.0898), but the model does not take price data as input and provides no function for computing VWAP execution cost. The VWAP execution cost is defined by BDF as the MAPE of (realized_execution_price - VWAP) / VWAP, which requires trade-level price data for each bin.

A developer implementing this spec will be able to validate volume forecast accuracy (using `compute_mape` and `compute_mse_star`) but will not be able to reproduce the VWAP execution cost benchmarks without additional price data and a VWAP cost computation function.

**Action:** Either (a) add a note in the Validation section explicitly stating that VWAP execution cost validation requires price data beyond the scope of this model, and that these benchmarks are provided for context only — the primary validation target is volume forecast MAPE/MSE, or (b) add a `compute_vwap_execution_cost` function with price data as an additional input. Option (a) is simpler and sufficient.

---

## Citation Verification (Round 2)

I re-verified key citations that were modified in draft 2:

| Spec Claim | Citation | Verified? | Notes |
|---|---|---|---|
| BDF uses float for turnover | BDF 2008 Sec 2.1, p. 1711 | Yes | "the number of traded shares S_i divided by the number of floated shares X_i" |
| Szucs uses TSO for turnover | Szucs 2017 Sec 2, p. 4 | Yes | "x_t = V_t / TSO_t" |
| AR(1) justification: Eq. (10) has no MA | BDF 2008 Eq. (10) | Yes | e_{i,t} = psi_1 * e_{i,t-1} + psi_2 + epsilon_{i,t} — no MA term |
| AR(1) justification: Szucs labels AR(1) | Szucs 2017 Eq. (5) | Yes | e_p = c + theta_1 * e_{p-1} + eps_p |
| AR(1) justification: OLS/MLE equivalence | BDF 2008 Sec 2.3 | Yes | "estimate by maximum likelihood" — MLE=OLS only for AR |
| MLE variance denominator T-1 | BDF 2008 Sec 2.3 | Yes | Consistent with MLE specification |
| SVD approach for IC_p2 | Bai & Ng (2002) | Yes | Algebraically equivalent to eigenvalue approach |
| Dynamic weight normalization | BDF 2008 Sec 4.2.2 | Yes | "the proportion is only applied on the remaining volume to trade after interval t" |
| Overnight transition ACF spikes | BDF 2008 Fig. 2 | Yes | Periodic spikes at multiples of k=25 visible in specific component ACF |
| U-method fallback | Szucs 2017 Eq. (3) | Yes | Simple time-of-day average |
| SETAR notation: Szucs intercept-first | Szucs 2017 Eq. (6) | Yes | (c_{1,1} + theta_{1,2} * e_{p-1}) — intercept before AR coeff |
| Portfolio MAPE 0.0752 | BDF 2008 Table 2 | Yes | PCA-SETAR volume prediction Mean MAPE |
| SETAR 30/33 DJIA by MAPE | Szucs 2017 Table 2c | **Partial** | Summary says "26-30/33" — the 30 is the upper bound of the range across different pairwise comparisons |

---

## Overall Assessment

Draft 2 is excellent. All major issues are resolved with thorough, well-cited revisions. The three new minor issues are non-blocking quality improvements:

- m1-new (forecasts_history) is a pseudocode consistency fix — trivial to address.
- m2-new (portfolio MAPE) and m3-new (VWAP cost) are about validation completeness, not algorithmic correctness. A developer can implement and validate the core model without them.

**Recommendation:** The spec is ready for implementation. The three minor items can be addressed as comments to the developer rather than requiring another revision round. If the proposer wants to do a final polish, these are quick fixes, but they do not justify a full revision cycle.
