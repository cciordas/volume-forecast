# Critique of Implementation Specification Draft 1: PCA Factor Decomposition (BDF)

**Direction 2, Run 4 — Critic Review**

## Summary Assessment

This is a strong draft. The overall structure is comprehensive, the pseudocode is detailed enough to implement directly, and citations are unusually thorough with explicit paper-section-equation references throughout. The spec correctly identifies the BDF-SETAR dynamic execution variant as the target and provides a clear data flow diagram.

I identify **3 major issues** and **7 minor issues** below.

---

## Major Issues

### M1. Turnover denominator: TSO vs. float — papers disagree, spec sidesteps

**Spec section:** `prepare_turnover_matrix` (line 71), Parameter table (`shares_measure`)

**Problem:** The spec defines turnover as `volume / TSO` and cites BDF 2008 Section 2.1. However, BDF 2008 explicitly defines turnover as "traded shares divided by **float**" (paper summary confirms: "volume measured as turnover (traded shares / float shares)"). The BDF 2008 paper text on p. 1711 states: "the turnover for stock i at date t, i.e. the number of traded shares S_i divided by the **panel structure** of volume. As shown in Darolles and Le Fol (2003), the **market turnover**..." and the summary further clarifies "adjusted for splits and dividends."

Meanwhile, Szucs 2017 Section 2 explicitly uses TSO: "x_t = V_t / TSO_t" with TSO downloaded from Bloomberg.

The spec acknowledges this discrepancy only via a parameter `shares_measure` with default "TSO" but does not discuss the implications. This is a **substantive modeling choice**, not a minor parameter. Using TSO vs. float can produce significantly different turnover magnitudes, especially for stocks with large insider/institutional holdings. The PCA decomposition, threshold tau, and AR coefficients are all fitted to these values, so the choice affects model calibration.

**Action required:** Explicitly document that BDF 2008 uses float and Szucs 2017 uses TSO. State which is preferred for the implementation and why. If TSO is chosen (for data availability), note that benchmark numbers from BDF 2008 Table 2 may not be directly comparable since they used float-adjusted turnover.

### M2. ARMA(1,1) vs. AR(1) — the spec asserts equivalence without sufficient justification

**Spec section:** `fit_ar1` docstring (lines 251-254)

**Problem:** The spec states: "BDF labels this 'ARMA(1,1)' but Eq. (10) contains no MA term." This is a critical claim that needs more careful treatment.

Looking at BDF 2008 Eq. (10): `e_{i,t} = psi_1 * e_{i,t-1} + psi_2 + epsilon_{i,t}`. The spec is correct that this equation, as written, has no MA term — it is indeed an AR(1) with intercept. However, the paper text on p. 1712 explicitly calls it "ARMA(1,1) with white noise." There are two possible explanations:

1. The paper's text label is simply wrong/misleading, and the equation is authoritative (the spec's interpretation).
2. The paper intended an MA(1) term but dropped it from the equation display for brevity, or the MA coefficient was found to be negligible in practice.

Szucs 2017 Eq. (5) writes it as `e_p = c + theta_1 * e_{p-1} + epsilon_p` and labels it AR(1), supporting interpretation (1).

The spec's choice to implement AR(1) is likely correct, but the justification should be strengthened. The current note reads like a dismissal of BDF's own labeling. A developer reading this might wonder if they should add an MA term.

**Action required:** Strengthen the justification. Note that: (a) the equation as written in BDF 2008 has no MA term, (b) Szucs 2017 independently labels it AR(1) and reproduces good results, and (c) if an MA(1) term were present, conditional MLE would NOT be equivalent to OLS (the spec currently claims OLS equivalence, which is only valid for pure AR models). This last point is an important consistency check that confirms AR(1) is the correct interpretation.

### M3. Dynamic execution: the spec implements one-step-ahead updating but describes it as multi-step forecasting

**Spec section:** `run_dynamic_execution` (lines 540-574), `forecast_specific` (lines 397-438)

**Problem:** There is a subtle but important inconsistency in the dynamic execution logic. The spec's `forecast_specific` function produces multi-step-ahead forecasts by iterating the AR/SETAR recursion forward. However, in the dynamic execution loop (lines 543-573), at each step j, the code:

1. Forecasts `k - j` steps ahead for remaining bins.
2. Computes weights from these forecasts.
3. Observes actual turnover for bin j.
4. Updates `e_state[i]` from the observed actual.

The issue is that **only the first weight** (for the immediate next bin) from each multi-step forecast is ever acted upon, because the state is updated with the actual observation before the next iteration. This means the multi-step forecast is computed but only the first element matters for the weights that get executed. The remaining elements affect the normalization of the first weight (since weights are proportional to forecasts), so they are not wasted, but this subtlety should be made explicit.

BDF 2008 Section 4.2.2 describes this correctly: "we can only update our strategy for the rest of the trading day" and "the proportion is only applied on the remaining volume to trade after interval t." The spec's pseudocode is functionally correct, but the comment on line 665 ("Only weights_history[j][0, i] is acted upon at step j") should be elevated to the main algorithm description with an explanation of why the full multi-step forecast still matters (for normalization).

**Action required:** Add a clear explanation in the algorithm description (not just a parenthetical in the data flow) that: (a) at each step, only the first weight is executed, (b) the remaining multi-step forecasts serve as the denominator for normalizing that first weight, and (c) this is why multi-step forecast quality matters even in dynamic mode — poor multi-step forecasts will distort the normalization and hence the first weight.

---

## Minor Issues

### m1. IC_p2 residual variance formula — potential off-by-one in eigenvalue scaling

**Spec section:** `select_factor_count` (lines 112-128)

The V(r) computation uses `total_ss - scaling * sum(eigenvalues[0:r])` where `scaling` is N or P depending on which matrix was eigendecomposed. This is algebraically correct for the relationship between eigenvalues of X'X/P and the singular values of X, but the code branches on `P >= N` vs `P < N` and uses different scaling factors. A developer might find this confusing.

The cleaner approach: always compute via SVD (which the spec already uses in `extract_factors`), and compute V(r) = (1/(NP)) * sum(s[r:]^2) where s are the singular values of X. This avoids the branching entirely and is numerically identical.

**Action:** Consider unifying the factor-count selection to also use SVD, or at minimum add a comment explaining why the eigenvalue approach is used here instead of SVD.

### m2. Innovation variance degrees of freedom

**Spec section:** `fit_ar1` (line 272)

The spec computes `sigma2 = sum(residuals ** 2) / (T - 1 - 2)` with the comment "T-1 obs, 2 params." This gives the unbiased estimator. However, for consistency with MLE (which BDF 2008 specifies), the denominator should be `T - 1` (the biased MLE estimator). For large T (= L*k = 520), the difference is negligible, but the spec should be consistent with its claimed estimation method.

**Action:** Either use `T - 1` for MLE consistency, or note that the unbiased estimator is a deliberate deviation. The same applies to sigma2 in `fit_setar` (lines 365, 372).

### m3. SETAR notation mapping table has swapped subscript semantics

**Spec section:** Notation mapping table (lines 379-392)

The table maps `phi_12` to Szucs's `c_{1,1}` (1st regime intercept) and `phi_11` to Szucs's `theta_{1,2}` (1st regime AR coefficient). The parenthetical "(2nd arg)" and "(1st arg)" labels are confusing because they seem to refer to argument position in the equation, but Szucs's notation uses (regime, parameter_index) convention. The mapping is correct but the explanatory notes are misleading.

**Action:** Rewrite the parenthetical notes to say "(regime 1 intercept)" and "(regime 1 AR coeff)" instead of "(1st arg)" and "(2nd arg)."

### m4. Missing MAPE computation specification

**Spec section:** Validation (lines 750-786)

The spec provides benchmark MAPE values from both papers but does not specify how to compute MAPE for the implementation's own validation. The Szucs 2017 MAPE (Eq. 2) is a per-observation average: MAPE = (1/N) * sum |Y_t - Y_t^f| / Y_t. BDF 2008's portfolio MAPE is computed differently (MAPE of the portfolio-level error). The spec should include a `compute_mape` function in the pseudocode to avoid ambiguity.

**Action:** Add a validation utility function that computes per-stock per-bin MAPE (matching Szucs 2017 Eq. 2) and portfolio-level MAPE (matching BDF 2008 Table 2). Include the MSE* formula (Szucs Eq. 14) as well, since it's referenced in the benchmarks.

### m5. No guidance on what happens at the first bin (j=0) in dynamic execution

**Spec section:** `run_dynamic_execution` (lines 543-573)

At j=0, the code forecasts all k bins using the last specific component value from the estimation window (E_hat[-1, :]). This is the overnight transition — the last bin of the previous day's estimation window to the first bin of the new trading day. The spec correctly initializes `e_state` from `E_hat[-1, :]` but does not discuss whether this overnight transition is handled differently from intraday transitions.

BDF 2008 Fig. 2 shows periodic ACF spikes at multiples of k, indicating overnight transitions are structurally different. The spec mentions this in Edge Case 6 but does not address it in the algorithm. Since BDF 2008 itself does not treat overnight transitions differently, this is not a bug, but a note in the pseudocode would help a developer understand why the first bin's forecast may be less accurate.

**Action:** Add a brief comment in `run_dynamic_execution` at the initialization step noting that the overnight transition (E_hat[-1,:] to bin 0 of the forecast day) may introduce additional forecast error due to overnight dynamics not captured by the AR/SETAR model.

### m6. Stationarity check is mentioned but not specified

**Spec section:** `run_dynamic_execution` (lines 527-529)

The spec mentions: "Stationarity check: if |psi_1| >= 1 (AR) or |phi_11| >= 1 or |phi_21| >= 1 (SETAR), fall back to U-method for this stock (Researcher inference)." This is mentioned as a comment but not implemented in the pseudocode. The fallback to U-method is not defined either — does "U-method" mean using only c_forecast (the common component average), or using the raw X average from `u_method_benchmark`?

**Action:** Either implement the stationarity check as an explicit code block with a defined fallback (specify which U-method variant to use), or remove the comment and address it only in Edge Cases.

### m7. The `daily_rolling_update` function uses 0-based day_index but the boundary condition is unclear

**Spec section:** `daily_rolling_update` (lines 600-621)

The function computes `start_bin = (day_index - L_days) * k`. If `day_index < L_days`, this produces a negative start index. The spec states in Initialization (line 727) that the first forecast day is `L_days + 1`, but it does not add a guard in the function. A developer might call this function with an invalid day_index.

**Action:** Add a precondition check: `assert day_index >= L_days` with an informative error message.

---

## Citation Verification

I verified the following citations against the source papers:

| Spec Claim | Citation | Verified? | Notes |
|---|---|---|---|
| Turnover = volume / TSO | BDF 2008 Sec 2.1 | **Partial** | BDF uses float, not TSO (see M1) |
| No centering in PCA | BDF 2008 Sec 2.2 | Yes | Eq. (6) minimizes without centering; Bai (2003) framework absorbs intercept |
| F'F/T = I_r normalization | BDF 2008 Eq. (6) | Yes | "Concentrating out A and using the normalization F'F/T = I" |
| Common component forecast Eq. (9) | BDF 2008 Sec 2.3 | Yes | Exact match |
| AR(1) Eq. (10) | BDF 2008 Sec 2.3 | Yes | Equation matches, but text says "ARMA(1,1)" (see M2) |
| SETAR Eq. (11) | BDF 2008 Sec 2.3 | Yes | Exact match including indicator function |
| I(x) = 1 if x <= tau | BDF 2008 Eq. (11) | Yes | "I(x) equals to 1 when x <= tau and 0 elsewhere" |
| Szucs AR(1) Eq. (5) | Szucs 2017 Sec 4.1 | Yes | e_p = c + theta_1 * e_{p-1} + epsilon_p |
| Szucs SETAR Eq. (6)-(7) | Szucs 2017 Sec 4.1 | Yes | Intercept-first ordering confirmed |
| SETAR outperforms AR: 36/39 CAC40 | BDF 2008 Sec 3.2 | Yes | "ARMA beats SETAR for only 3 out of 39 stocks" |
| SETAR outperforms AR: 30/33 DJIA by MAPE | Szucs 2017 Table 2c | Yes | "BDF_SETAR is best overall and beats all other models on 26-30/33 individual shares" |
| Per-stock MAPE ~0.40 | Szucs 2017 Table 2a | Yes | BDF_SETAR MAPE = 0.399 |
| Portfolio MAPE ~0.075 | BDF 2008 Table 2 | Yes | PCA-SETAR Mean = 0.0752 |
| Dynamic VWAP execution cost | BDF 2008 Table 2 | Yes | PC-SETAR dynamic Mean = 0.0898 |
| Static execution worse than classical | BDF 2008 Sec 4.2 p. 1717 | Yes | "Hence, such a strategy is just impossible to implement" — static requires knowing total daily volume |
| k = 25 for BDF, k = 26 for Szucs | Both papers | Yes | BDF: 20-min bins 9:20-17:20; Szucs: 15-min bins 9:30-16:00 |
| L = 20 days | Both papers | Yes | Consistent |
| MLE = OLS for AR(1) | BDF 2008 Sec 2.3 | **Conditional** | Valid only for AR(1), not ARMA(1,1) (see M2) |
| Estimation time comparison | Szucs 2017 Sec 5 | Yes | "BDF model runs in ~2 hours. BCG_0... ~60 machine-days" |

---

## Completeness Check

| Required Element | Present? | Quality |
|---|---|---|
| Model description | Yes | Good |
| Pseudocode | Yes | Excellent — 11 functions, well-organized |
| Data flow | Yes | Good diagram |
| Variants discussion | Yes | Well-justified |
| Parameter table | Yes | Complete |
| Initialization | Yes | Good |
| Calibration | Yes | Good |
| Expected behavior | Yes | Excellent — both paper benchmarks included |
| Sanity checks | Yes | Excellent — 11 checks |
| Edge cases | Yes | Good — 7 cases |
| Known limitations | Yes | Comprehensive — 10 items |
| Paper references | Yes | Excellent traceability table |
| Researcher inference marking | Yes | Consistently marked |

---

## Overall Assessment

The draft is well above average quality. The pseudocode is implementable as-is for the core algorithm. The three major issues (TSO vs. float, ARMA vs. AR labeling justification, and dynamic execution explanation) are all resolvable with text clarifications rather than algorithmic changes. The minor issues are quality improvements that would reduce developer confusion but are not blocking.

**Recommendation:** One revision round should suffice to address all issues.
