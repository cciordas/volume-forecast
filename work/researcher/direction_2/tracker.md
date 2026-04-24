# Findings Tracker: Direction 2 — PCA Factor Decomposition (BDF) for Intraday Volume

## Summary
- Total findings: 87
- Runs processed: [1, 2, 3, 4, 5, 6, 7]
- Last update: run 7

## Findings

### F1. Model decomposition: additive common + specific
**Category:** algorithm
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): Intraday turnover decomposed as x = c + e, where c is a market-wide common component extracted by PCA and e is a stock-specific residual. Reference: BDF 2008, Section 2.1-2.2, Eq. (2)-(5).
**Best version so far:** Run 3 — clear overview (lines 5-15) connecting decomposition to VWAP execution, with explicit inputs/outputs specification and key assumption statement.

### F2. PCA factor extraction method
**Category:** algorithm
**Approach A** (runs 1, 2): Eigendecomposition of X'X/T (N x N matrix), take r eigenvectors. More efficient when N < T. SVD noted as numerically stable alternative. Reference: BDF 2008, Section 2.2, Eq. (6); Bai (2003).
**Approach B** (runs 3, 4, 5, 6, 7): Truncated SVD directly on X as the primary method, avoiding P<=N vs P>N branching entirely. F_hat = sqrt(P) * U[:,:r], Lambda_hat = V[:,:r] @ diag(s[:r]) / sqrt(P). Uses scipy.sparse.linalg.svds or sklearn randomized_svd. SVD internally chooses the smaller matrix. Reference: BDF 2008, Section 2.2; Bai (2003).
**Best version so far:** Run 5 — combined extract_and_select_factors function (lines 167-267) performs factor count selection and extraction in a single truncated SVD call, with complete Eckart-Young derivation for V(r), inline verification of F'F/P = I_r, and notes on IC_p1 alternative.

### F3. Factor normalization: F'F/P = I_r
**Category:** algorithm
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): Factors normalized so F_hat' @ F_hat / P = I_r (P = L*k total observations). Runs 1-2 achieve via eigenvalue scaling: F_hat = X @ V_r @ diag(1/sqrt(lambda)). Runs 3-6 achieve via SVD: F_hat = sqrt(P) * U[:,:r]. Both yield identical results. Reference: BDF 2008, Eq. (6); Bai (2003).
**Best version so far:** Run 3 — SVD-based derivation (lines 292-298) with explicit verification: F_hat.T @ F_hat / P = P * U.T @ U / P = I_r. Cleaner than eigenvalue scaling.

### F4. Loading recovery: Lambda_hat from F_hat and X
**Category:** algorithm
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): After computing normalized F_hat, loadings recovered. Runs 1-2: A_hat = F_hat.T @ X / T, shape (r, N). Runs 3-6: Lambda_hat = (Vt.T * s) / sqrt(P), shape (N, r). Algebraically equivalent. Reference: BDF 2008, Section 2.2.
**Best version so far:** Run 3 — SVD-based formula avoids the matrix multiplication F_hat.T @ X, directly using singular values.

### F5. Column-mean centering before PCA
**Category:** algorithm
**Approach A** (runs 1, 2, 3, 4, 5, 7): Do NOT center (subtract column means) before PCA. In the Bai (2003) large-dimensional factor framework, column means are absorbed into factor loadings. BDF 2008 does not mention demeaning and Eq. (6) does not include it. Reference: Bai 2003; BDF 2008, Section 2.2.
**Approach B** (run 6): Center by column means before PCA. col_means = mean(X, axis=0); X_centered = X - col_means. Add col_means back when computing common component forecast. Justification: standard PCA practice; ensures extracted factors capture intraday shape variation rather than the mean turnover level. Notes "Neither BDF 2008 nor Szucs 2017 explicitly prescribes centering before eigendecomposition" but argues centering is beneficial. Reference: Researcher inference; standard PCA practice.
**Best version so far:** Run 4 (Approach A) — strongest emphasis with "Critical: do NOT subtract column means" inline comment in pseudocode (line 136), complete rationale ("centering would strip level information and produce a biased decomposition of the U-shaped seasonal pattern"), and explicit Bai 2003 reference. However, run 6 (Approach B) presents a coherent alternative with explicit col_means add-back in the forecast step.

### F6. Factor count selection via Bai & Ng (2002) information criterion
**Category:** algorithm
**Approach A** (runs 1, 2, 3, 4, 5, 7): IC_p2 criterion: IC_p2(r) = ln(V(r)) + r * ((N+P)/(N*P)) * ln(min(N,P)). V(r) = residual variance from rank-r approximation. Reference: BDF 2008, Section 2.2; Bai & Ng (2002).
**Approach B** (run 6): IC_p1 criterion: IC_p1(r) = ln(V(r)) + r * ((N+P)/(N*P)) * ln((N*P)/(N+P)). Same V(r), different penalty term. Reference: BDF 2008, Section 2.2; Bai & Ng (2002).
**Note:** BDF 2008 does not specify which IC variant was used (see also F65). Both typically select r = 1-3 for volume data.
**Best version so far:** Run 4 (Approach A) — complete pseudocode using SVD singular values for V(r) computation: V_r = (1/(N*P)) * (total_ss - sum(s[0:r]^2)), with V(r)<=0 floating-point guard. Run 6 (Approach B) provides complete pseudocode for IC_p1 with explicit SVD reuse across truncation levels.

### F7. Factor count stability heuristic
**Category:** algorithm
**Single source** (run 1): If selected r varies by more than 1 across consecutive days, fix r at the mode of recent selections to avoid day-to-day instability. (Researcher inference.)
**Best version so far:** Run 1

### F8. Common component forecast: time-of-day bin average across L days
**Category:** algorithm
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): c_forecast[j, i] = mean of C_hat values at bin j across all L days in the estimation window. Run 6 adds col_means back to the time-of-day average (consistent with its centering approach in F5). Reference: BDF 2008, Section 2.3, Eq. (9).
**Best version so far:** Run 4 — complete pseudocode with 3D reshape (L, k, N) and mean across axis=0; adds explicit BDF equation reference (Eq. 9), "computed BEFORE the trading day begins" note, and "fixed for entire day, NOT updated intraday" clarification.

### F9. AR(1) specific component model
**Category:** algorithm
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): e_{i,t} = psi_1 * e_{i,t-1} + psi_2 + epsilon_{i,t}. BDF 2008 labels this "ARMA(1,1)" but Eq. (10) contains no MA term -- it is AR(1) with intercept. Szucs 2017 correctly labels it "AR(1)". Run 6 uses notation e_t = c + theta_1 * e_{t-1} + epsilon_t (same model, different variable names). Reference: BDF 2008, Section 2.3, Eq. (10); Szucs 2017, Eq. (5).
**Best version so far:** Run 4 — provides three independent lines of evidence for AR(1) over ARMA(1,1): (a) equation as written has no MA term, (b) Szucs 2017 Eq. 5 independently labels it AR(1) with strong results, (c) OLS/MLE equivalence holds only for AR(1) not ARMA. Most rigorous justification.

### F10. SETAR specific component model
**Category:** algorithm
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): e_{i,t} = (phi_11 * e_{i,t-1} + phi_12) * I(e_{i,t-1} <= tau) + (phi_21 * e_{i,t-1} + phi_22) * (1 - I(e_{i,t-1} <= tau)) + epsilon_{i,t}. Two-regime threshold autoregression. Reference: BDF 2008, Section 2.3, Eq. (11); Szucs 2017, Eq. (6).
**Best version so far:** Run 4 — complete pseudocode with two-phase estimation (grid search then final fit), full notation mapping table across papers, and Szucs intercept-ordering warning.

### F11. SETAR estimation: profile MLE via grid search over tau
**Category:** algorithm
**Approach A** (runs 1, 2, 6): Grid search over 71 tau candidates from 15th to 85th percentile of lagged values in 1-percentile steps. Run 6: `percentiles(x_lag, range(15, 86))`. Min regime size not explicitly specified in run 6. Reference: BDF 2008, Section 2.3; Hansen 1997.
**Approach B** (runs 3, 4, 5, 7): Grid search over 100 tau candidates (n_grid=100) from 15th to 85th percentile via linspace. Min regime size: max(0.10*n, 20). Reference: BDF 2008, Section 2.3; Hansen 1997. (See F69 for single-pass vs two-phase implementation variants.)
**Best version so far:** Run 5 — single-pass estimation stores winning coefficients and residuals during grid search, eliminating redundant re-computation. Includes "if best_tau is None: return None" guard with clear fallback-to-AR(1) contract.

### F12. AR(1)/SETAR estimation equivalence: MLE = OLS under Gaussian errors
**Category:** algorithm
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): BDF 2008 Section 2.3 specifies "estimate by maximum likelihood." For AR(1), conditional MLE is equivalent to OLS. For SETAR, profile MLE with grid search reduces to conditional OLS grid search. Run 6 provides explicit inline citations of BDF 2008 Section 2.3 in both fit_ar1 and fit_setar pseudocode. Reference: BDF 2008, Section 2.3.
**Best version so far:** Run 4 — additionally provides the converse argument: if an MA(1) term were present, OLS/MLE equivalence would NOT hold (MA models require iterative optimization), strengthening the AR(1) identification.

### F13. Dynamic intraday forecasting: update after each observed bin
**Category:** algorithm
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): At each completed bin j, observe actual turnover, extract specific component as e_observed = x_observed - c_forecast, then produce one-step-ahead forecast via AR(1)/SETAR recursion. Run 6 provides complete execute_dynamic_vwap function with remaining_shares tracking and fraction-based allocation. Reference: BDF 2008, Section 4.2.2.
**Best version so far:** Run 4 — complete run_dynamic_execution pseudocode (lines 547-686) with stationarity checks, U-method fallback per stock, explicit overnight transition note, and both weights_history and forecasts_history outputs.

### F14. Day-start initialization of specific component
**Category:** algorithm
**Approach A** (runs 1, 2): At start of each trading day (j_current=0), set specific component forecast to zero for all stocks. Justification: (a) unconditional mean ~0 by PCA construction, (b) consistent with BDF Section 4.2.1, (c) avoids carrying overnight info. (Researcher inference.)
**Approach B** (runs 3, 4, 5, 6, 7): Initialize last_specific = E_hat[-1, :], i.e., the last in-sample residual from the final bin of the final day in the estimation window. Run 6: e_last = e_hat[-1, stock_i] with explicit note "the natural choice for a dynamic model, consistent with the one-step-ahead forecasting framework." Reference: BDF 2008 Section 2.3.
**Best version so far:** Run 4 — adds explicit overnight transition note (lines 631-640): "this overnight gap introduces dynamics not captured by the AR/SETAR model" and "the first bin's forecast may therefore be less accurate than subsequent intraday bins." References BDF 2008 Fig. 2 ACF spikes.

### F15. Multi-step SETAR forecasts via deterministic iteration
**Category:** algorithm
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): For bins beyond j+1, iterate the AR(1)/SETAR recursion forward using point forecasts as inputs to threshold comparison. Run 6: explicit chaining loop (e_last = e_forecast after each step) with detailed inline explanation of why chaining is needed (denominator for participation fraction). Reference: BDF 2008 does not specify method. (Researcher inference.)
**Best version so far:** Run 4 — dedicated forecast_specific function (lines 470-511) with clear loop structure, explicit threshold comparison per step, and inline note about geometric decay.

### F16. Multi-step AR(1) forecasts: iterate recursion forward
**Category:** algorithm
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): e_hat[j+1] = psi_1 * e_hat[j] + psi_2. Decays geometrically toward psi_2/(1-psi_1) (unconditional mean). Run 6 adds explicit decay formula in sanity check: at horizon h, forecast ~ c/(1-theta_1) + theta_1^h * (e_last - c/(1-theta_1)). Reference: BDF 2008, Section 4.2.2.
**Best version so far:** Run 6 — provides the explicit closed-form decay formula (lines 766-773), connecting it directly to why static execution fails (BDF 2008 Section 4.2.1: "long-horizon ARMA forecasts collapse to zero").

### F17. Negative forecast handling with fallback
**Category:** edge case
**Approach A** (runs 1, 2): Floor negative turnover forecasts to 1e-8 and re-normalize. If >50% of remaining bins produce negative raw forecasts, fall back to common component only (set e_forecast=0). (Researcher inference.)
**Approach B** (runs 3, 4, 5, 6, 7): Clip forecasts at 0.0 (not 1e-8). Run 6: `x_forecast = max(x_forecast, 0.0)` per bin. Fallback to uniform distribution (1/k_remaining or 1/k) when total forecast <= 0. Reference: Researcher inference.
**Best version so far:** Run 4 — compute_vwap_weights function (lines 514-544) with clean implementation: maximum(forecasts, 0.0), total <= 0 check, equal-weight fallback with inline explanation.

### F18. Static execution excluded
**Category:** algorithm
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): Static (all-bins-at-once) execution explicitly excluded. Multi-step AR/SETAR forecasts decay toward unconditional mean, neutralizing specific component. Run 6: "Using the model's full-day forecast at market open (without intraday updating) is worse than the simple historical average." Reference: BDF 2008, Section 4.2.
**Best version so far:** Run 4 — provides specific page reference (BDF 2008, Section 4.2, penultimate paragraph of p. 1717) and connects to Variants section explaining why dynamic is preferred.

### F19. SETAR preferred uniformly over AR(1)
**Category:** algorithm
**Approach A** (run 1): Use SETAR for all stocks, not per-stock selection. BDF 2008: only 3 of 31 stocks favor AR. Szucs 2017 confirms BDF-SETAR best MAPE.
**Approach B** (runs 2, 3, 4, 5, 6, 7): SETAR outperforms AR(1) for 36 of 39 CAC40 stocks (BDF 2008, Section 3.2). Szucs 2017 confirms on 33 DJIA stocks: BDF-SETAR MAPE 0.399 vs BDF-AR 0.403. AR(1) retained as fallback only when SETAR estimation fails. Run 6 additionally notes AR(1) wins on MSE (6.49e-4 vs 6.60e-4) and has fewer extreme squared errors. Reference: BDF 2008, Section 3.2; Szucs 2017, Section 4-5.
**Best version so far:** Run 3 — correctly cites "36 of 39" stocks, adds detail that no improvement test is applied (SETAR is default whenever estimation converges), and provides the Szucs Table 2c pairwise detail (30/3 stocks favor SETAR by MAPE).

### F20. Cross-day boundary: fit concatenated series (known limitation)
**Category:** algorithm
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): AR(1)/SETAR fit to full concatenated multi-day specific component series including cross-day transitions. Overnight jumps may inflate innovation variance and attenuate AR coefficient. Run 6 adds: "The overnight gap between these observations is fundamentally different from the 15-20 minute gap within a day, but the time-series model treats them identically." BDF and Szucs do NOT address this. Reference: BDF 2008.
**Best version so far:** Run 4 — adds specific detail in dynamic execution context (lines 631-640): overnight gap between last bin of estimation and first bin of forecast "introduces dynamics not captured by the AR/SETAR model." First bin forecast may be less accurate. References BDF 2008, Fig. 2 periodic ACF spikes.

### F21. Factor sign indeterminacy across re-estimations
**Category:** algorithm
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): Sign of eigenvectors/singular vectors may flip across daily re-estimations. Does not affect common component (C = F@A, sign flips cancel through the full chain: C unchanged -> E unchanged -> AR/SETAR params unchanged). No special handling required. Reference: BDF 2008; Bai 2003.
**Best version so far:** Run 4 — extends to rotation indeterminacy: "any orthogonal transformation cancels through the product" (line 244), not just sign flips. Explicitly notes individual factors "are not interpreted" per BDF.

### F22. Volume share computation for VWAP
**Category:** algorithm
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): w[j+1] = x_hat[j+1] / sum(x_hat[j+1:k]) applied to remaining order volume. Run 6 implements as fraction = remaining_forecasts[0] / total_remaining_forecast, with uniform fallback. Reference: BDF 2008, Section 4.2.2.
**Best version so far:** Run 4 — compute_vwap_weights function (lines 514-544) with explicit clipping, equal-weight fallback, and inline equivalence note: "Since we normalize, this is equivalent to weights = forecasts_remaining / sum(forecasts_remaining)."

### F23. Daily rolling update procedure
**Category:** algorithm
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): At end of each trading day, slide estimation window forward by one day, re-run PCA + model fitting for next day's forecasts. Reference: BDF 2008, Section 3.1; Szucs 2017, Section 3.
**Best version so far:** Run 4 — daily_rolling_update function (lines 689-715) with explicit precondition assertion (day_index >= L_days) and clear bin-index computation.

### F24. Intraday bins: k = 26 (15-min bins, 9:30-16:00)
**Category:** parameter
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): 26 bins of 15 minutes for US equities. 25 bins of 20 minutes also validated (BDF 2008 CAC40). Range: 13-52. Reference: BDF 2008, Section 3.1; Szucs 2017, Section 2.
**Best version so far:** Run 4 — parameter table (line 991) with range 13-52 and explicit source: "Szucs 2017, Section 2 (15-min bins, 9:30 to 16:00 = 6.5 hours = 26 bins)."

### F25. Rolling window: L_days = 20 trading days
**Category:** parameter
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): Medium sensitivity. Range: 10-40. Run 6 specifies range 10-60. Reference: BDF 2008, Section 3.1; Szucs 2017, Section 3.
**Best version so far:** Run 4 — parameter table (line 992) with range 10-40 and dual-source reference: "BDF 2008, Section 3.1; Szucs 2017, Section 3. Both use L = 20."

### F26. Factor count r typically 1-3, r_max upper bound
**Category:** parameter
**Approach A** (runs 1, 2, 3, 4, 5, 7): r selected by Bai-Ng IC, typically 1-3. r_max = 10 (upper bound for search). Reference: BDF 2008, Section 2.2.
**Approach B** (run 6): r selected by Bai-Ng IC, typically 1-3. r_max = min(20, min(N, P) - 1). Justification: "practical choice, generous for volume data where r = 1-3 is typical. Bai & Ng 2002 recommend searching up to a reasonable upper bound; 20 is conservative." Reference: BDF 2008, Section 2.2; Bai & Ng (2002).
**Best version so far:** Run 4 (Approach A) — parameter table notes "BDF 2008 does not specify r_max but reports r is typically small (1-3)." r_max = 10 is more computationally economical and sufficient given expected r values.

### F27. Cross-section size: N >= 30 stocks
**Category:** parameter
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): PCA asymptotics require large N. BDF 2008 uses N=39, Szucs 2017 uses N=33. Minimum practical N is approximately 10-20 for stable factor estimation. Run 6 states N >= 20 as assumption. Reference: BDF 2008; Szucs 2017; Bai 2003.
**Best version so far:** Run 3 — adds lower bound detail: "Minimum practical N is approximately 10-20" (line 1157), below the N >= 30 from papers, providing a practical floor.

### F28. SETAR minimum regime parameters
**Category:** parameter
**Approach A** (runs 1, 2, 6): min_regime_frac = 0.15, min_regime_obs = 15. Each regime must have at least max(15, 0.15*T) observations. Run 6: tau grid restricted to [15th, 85th] percentile to "ensure sufficient obs in each regime" but does not specify explicit min_regime_obs. Reference: researcher inference based on Hansen 1997.
**Approach B** (runs 3, 4, 5): min_regime_frac = 0.10, min_regime_obs = 20. Each regime must have at least max(0.10*n, 20) observations. tau_quantile_range = [0.15, 0.85]. Reference: researcher inference.
**Approach C** (run 7): min_regime_obs = 10 (fixed, no fraction-based scaling). tau_quantile_range = [0.15, 0.85], n_grid = 100. Parameter table specifies range 5-20 for min_regime_obs. Reference: researcher inference.
**Best version so far:** Run 4 — parameter table (lines 998-999) cleanly specifies all four SETAR grid-search parameters with ranges. Run 7's lower min_regime_obs = 10 is more permissive, potentially allowing SETAR to fit in shorter estimation windows.

### F29. Turnover definition: x = V / TSO
**Category:** parameter
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): Turnover is shares traded divided by total shares outstanding. Run 6 adds: "TSO should be the most recent value available before the trading day to avoid look-ahead bias." Runs 3-4 add "shares_measure" parameter allowing "float" or "TSO" as denominator choice. Reference: BDF 2008, Section 3.1; Szucs 2017, Section 2.
**Best version so far:** Run 4 — most detailed treatment: inline comment (lines 96-127) explicitly quotes both BDF 2008 (float) and Szucs 2017 (TSO), explains implications for SETAR threshold and benchmark comparability, and notes "relative ordering of model performance is robust to this choice."

### F30. AR(1) parameter ranges
**Category:** parameter
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): psi_1 typically positive and < 1 (stationary). Runs 1-2: range 0.3-0.8. Runs 3-4: range 0.1-0.6 (sanity check). psi_2 near zero. Run 6: theta_1 in (-1, 1), clamped if |theta_1| >= 1. Reference: BDF 2008, Eq. (10).
**Best version so far:** Run 4 — provides actionable sanity check (line 1163): "If |psi_1| >= 1, the specific component is non-stationary and should be investigated."

### F31. SETAR parameter ranges and count
**Category:** parameter
**Approach A** (runs 1, 2, 6, 7): 6 params per stock: (phi_11, phi_12, phi_21, phi_22, tau, sigma_eps). Single sigma_eps for entire model. Run 6: sigma_eps = sqrt(SSR / (n - 4)). Reference: BDF 2008, Eq. (11).
**Approach B** (runs 3, 4, 5): 7 params per stock: (phi_11, phi_12, phi_21, phi_22, tau, sigma2_1, sigma2_2). Separate innovation variance per regime. Reference: BDF 2008, Eq. (11); researcher inference for per-regime variance.
**Best version so far:** Run 4 — per-regime variance with MLE estimator: sigma2_1 = sum(resid1^2)/n1, sigma2_2 = sum(resid2^2)/n2 (see F66 for DOF discussion).

### F32. Turnover matrix X shape convention: (P, N)
**Category:** index convention
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): X has shape (P, N) where P = L*k rows (time), N = n_stocks columns. Rows ordered chronologically. Run 6 uses consistent P notation with explicit shape annotations on SVD output. Reference: BDF 2008, Section 2.2.
**Best version so far:** Run 4 — consistent P notation throughout all pseudocode, with explicit shape annotations on every function parameter and return value.

### F33. Bin indexing convention
**Category:** index convention
**Approach A** (run 1): 1-based bin indices (j = 1 to k_bins).
**Approach B** (runs 2, 3, 4, 5, 6, 7): 0-based bin indices (j = 0 to k_bins - 1). Consistent with Python array indexing.
**Best version so far:** Run 4 — 0-based indexing used consistently across all 14 functions.

### F34. Type information: all float64
**Category:** other
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): All turnover values float64. Factor matrix: float64 (P, r). Loading matrix: float64 (N, r). VWAP weights: float64 in [0,1], sum to 1. AR(1) params: 3 floats. SETAR params: 7 floats per runs 3-5 (6 per runs 1-2, 6; see F31).
**Best version so far:** Run 3 — includes turnover typical range (1e-5 to 1e-1) and explicitly notes VWAP weights are non-negative.

### F35. MAPE validation target: ~0.40 per-stock per-bin (Szucs 2017)
**Category:** validation
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): For liquid US equities at 15-min: BDF-SETAR MAPE 0.399, BDF-AR 0.403, U-method 0.503 (Szucs 2017). Per-stock per-bin MAPE. Reference: Szucs 2017, Section 5, Table 2a.
**Best version so far:** Run 4 — complete validation table (lines 1060-1067) with MSE, MSE*, and MAPE for all three models, plus interpretation: "40% MAPE is typical for intraday volume" and "~20% MAPE reduction" vs baseline.

### F36. MAPE portfolio-level: ~0.07-0.09 (BDF 2008)
**Category:** validation
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): For CAC40 at 20-min: PCA-SETAR mean MAPE 0.0752, PCA-ARMA 0.0829, Classical 0.0905 (BDF 2008, Table 2). Run 6: BDF-SETAR in-sample ~0.075 MAPE on CAC40. Reference: BDF 2008, Table 2.
**Best version so far:** Run 4 — full table with Mean, Std, Q95 (lines 1091-1098) plus note on denominator difference (float vs TSO) affecting benchmark comparability.

### F37. MAPE scale discrepancy explanation
**Category:** validation
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): ~5x difference between papers explained by aggregation level: BDF computes portfolio-level MAPE, Szucs computes per-stock per-bin MAPE. Also different datasets, bin widths, sample lengths. Both show same relative ordering. Reference: Szucs 2017 Eq. 2.
**Best version so far:** Run 4 — concise explanation (lines 1105-1110): "portfolio-level error... diversifies away idiosyncratic forecast errors."

### F38. MSE validation targets
**Category:** validation
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): For 33 DJIA stocks at 15-min: BDF-AR MSE 6.49e-4, BDF-SETAR MSE 6.60e-4, U-method MSE 1.02e-3. Both BDF variants reduce MSE ~35% vs U-method. Note: by MSE, AR slightly edges SETAR. Reference: Szucs 2017.
**Best version so far:** Run 3 — adds explicit explanation of MSE vs MAPE discrepancy: "MSE is scale-sensitive and penalizes large errors more heavily; MAPE treats all percentage errors equally. For VWAP execution, MAPE is typically more relevant."

### F39. VWAP tracking error validation
**Category:** validation
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): Dynamic BDF-SETAR MAPE 0.0898, BDF-ARMA 0.0922, Classical 0.1006. ~10% relative reduction. Run 6 additionally cites theoretical (static) PCA-SETAR MAPE 0.0975. Reference: BDF 2008, Table 2 panel 3.
**Best version so far:** Run 4 — full table with Mean, Std, Q95 (lines 1112-1119) plus note that improvements can reach 50% for high-volatility stocks (e.g., CAP GEMINI, EADS).

### F40. Common component variance ratio sanity check
**Category:** validation
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): Var(C_hat)/Var(X) should be >0.5 for most stocks. If <0.30, PCA is misconfigured. Typical range 0.6-0.8 for liquid stocks. Reference: BDF 2008, Section 3.1. (Researcher inference.)
**Best version so far:** Run 4 — sanity check #5 (lines 1159-1161) with actionable threshold.

### F41. Common component U-shape
**Category:** validation
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): Common component should exhibit clear U-shaped intraday pattern. Run 6 adds: "high at market open, declining through midday, rising again toward close" and "broadly similar across stocks." Reference: BDF 2008, Section 1, Section 3.1, Fig. 3.
**Best version so far:** Run 4 — sanity check #3 (lines 1148-1151) with specific instruction: "Average c_forecast across stocks. The resulting (k,) vector should exhibit a clear U-shape."

### F42. Specific component mean-zero and low serial correlation
**Category:** validation
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): Specific component approximately mean-zero with much lower variance than raw turnover. ACF should show significant lag-1 correlation decaying quickly. Run 6 adds quantitative sanity check: abs(mean(e_hat[:, i])) < 0.001 * mean(abs(e_hat[:, i])). Reference: BDF 2008, Section 3.1, Fig. 2.
**Best version so far:** Run 4 — sanity check #4 (lines 1153-1157) with quantitative reference: specific component values in [-0.025, 0.010] vs common component in [0, 0.035].

### F43. SETAR should outperform AR(1) on >= 70% of stocks
**Category:** validation
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): BDF 2008: only 3 of 39 favor AR. Szucs 2017: 26-30 of 33 stocks favor SETAR by MAPE. Reference: BDF 2008, Section 3.2; Szucs 2017.
**Best version so far:** Run 4 — sanity check #7 (lines 1168-1171): "BDF 2008 reports 36/39 (92%); Szucs 2017 reports 30/33 (91%) by MAPE."

### F44. Eigenvalue scree plot diagnostic
**Category:** validation
**Consensus** (runs 1, 2, 3, 4, 5): Should show clear gap between first r eigenvalues and rest. If r > 5, suspect data quality issues. (Researcher inference.)
**Best version so far:** Run 1 — all equivalent.

### F45. Toy PCA verification example
**Category:** validation
**Approach A** (run 1): Construct (T=8, N=3) matrix with known factor. Verify recovery.
**Approach B** (runs 2, 3, 4): Construct (P=200, N=5) matrix with U-shape factor. Verify recovery. Common component should correlate > 0.95 with true factor pattern.
**Best version so far:** Run 4 — sanity check #10 (lines 1182-1185) with explicit correlation threshold (>0.95) and realistic dimensions.

### F46. Zero-volume bin handling
**Category:** edge case
**Approach A** (run 1): Exclude stocks with frequent zero-volume bins. Replace isolated zeros with small value or interpolate.
**Approach B** (runs 2, 3, 4, 5, 6, 7): Isolated zero-volume bins are valid data points. Stocks with frequent zeros (>5% of bins) excluded from cross-section. Do NOT replace zeros for PCA. Exclude zero-actual bins from MAPE. Run 6 adds: "zero-volume bins within the estimation window may destabilize the SETAR threshold estimation and will produce division-by-zero in MAPE evaluation." Reference: Szucs 2017, Section 2.
**Best version so far:** Run 4 — edge case #1 (lines 1193-1198) with Szucs reference: "all stocks had non-zero volume in every bin."

### F47. Half-days and early closures: exclude from estimation window
**Category:** edge case
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): Half-days have fewer than k bins. Exclude entirely. Run 6 cites BDF 2008 Section 3.1: "Christmas Eve and New Year's Eve excluded." Reference: BDF 2008, Section 3.1.
**Best version so far:** Run 4 — edge case #2 (lines 1200-1202) with specific BDF reference: "December 24 and 31 excluded."

### F48. Cross-section changes (IPOs/delistings)
**Category:** edge case
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): Maintain stable universe within each estimation window. Remove exiting stocks for entire window. Wait for entering stocks to have full history. If universe changes, re-estimate from scratch. Run 6: "Use the intersection of stocks available for the full L-day estimation window." Reference: researcher inference.
**Best version so far:** Run 4 — edge case #3 (lines 1204-1208): "the specific component series must be re-extracted under the new factor structure because changing N alters the PCA decomposition."

### F49. Extreme volume events
**Category:** edge case
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): No special handling needed -- SETAR captures regime switching. 20-day rolling window limits exposure. Run 6 adds: "The SETAR model is better equipped to handle this via regime switching" with BDF 2008 Section 3.2 reference (outperforms for high-volatility stocks). (Researcher inference.)
**Best version so far:** Run 4 — integrates extreme event handling into the stationarity check: non-stationary AR/SETAR coefficient triggers U-method fallback for that stock.

### F50. Known limitations list
**Category:** other
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): (1) Cross-section required. (2) No daily volume forecast. (3) No price impact. (4) Linear factor model. (5) SETAR threshold static within window. (6) Requires TSO data. (7) Overnight boundary contamination. (8) No intraday events. (9) Additive decomposition can produce negative forecasts. (10) Factor rotation indeterminacy. Run 6 covers all 10 with specific BDF references. Reference: BDF 2008 various sections.
**Best version so far:** Run 4 — 10 detailed limitations (lines 1236-1287) with specific paper references for each.

### F51. Computational cost
**Category:** other
**Consensus** (runs 2, 3, 5, 6, 7): Szucs 2017: ~2 hours for 33 stocks over 2,648 days. Per-day estimation on order of seconds. Run 6 adds: "this timing is consistent with the efficient N x N eigendecomposition, not the P x P approach." Reference: Szucs 2017, Section 5.
**Best version so far:** Run 3 — adds source reference: "Szucs 2017, Section 5, last paragraph."

### F52. U-method baseline explicit implementation
**Category:** algorithm
**Consensus** (runs 2, 3, 4, 5, 6, 7): Explicit pseudocode for U-method: u_forecast[stock, j] = mean of turnover at bin j across L days. Run 6 provides formula: x_hat_U[bin_j, stock_i] = (1/L) * sum_{d=0}^{L-1} x[bin_j + d*k, stock_i]. Reference: Szucs 2017, Section 4, Eq. (3); BDF 2008, footnote 5.
**Best version so far:** Run 4 — u_method_benchmark function (lines 718-738) with complete pseudocode, docstring, and connection to dynamic execution fallback.

### F53. Common forecast = U-method applied to C_hat, not X
**Category:** algorithm
**Consensus** (runs 2, 3, 4, 5, 6, 7): The BDF common component forecast is the U-method applied to C_hat, not X. The predictive improvement over U-method comes entirely from the specific component AR/SETAR forecast. Run 6 quotes BDF 2008 footnote 5: "Replacing c by x in Eq. (9), we get the classical prediction." Reference: BDF 2008, Section 2.3, Eq. (9), footnote 5.
**Best version so far:** Run 4 — u_method_benchmark docstring (lines 720-725): "equivalent to the common component forecast applied to X instead of C_hat. The BDF model's improvement over U-method comes entirely from the specific component AR/SETAR forecast."

### F54. SETAR notation mapping across papers
**Category:** index convention
**Consensus** (runs 2, 3, 4, 5, 6, 7): Explicit mapping between spec, BDF Eq. (11), and Szucs Eq. (6) notation. Warning: Szucs puts intercept first, opposite of spec ordering. Run 6 uses (c_1, theta_12, c_2, theta_22) notation. Reference: BDF 2008, Eq. (11); Szucs 2017, Eq. (6).
**Best version so far:** Run 4 — full notation mapping table (lines 450-464) with all seven parameters and explicit warning about Szucs intercept ordering.

### F55. Large-N computation path: XX'/P eigendecomposition
**Category:** algorithm
**Consensus** (runs 2, 3, 4, 5, 6, 7): When N > P, eigendecompose XX'/P (or use SVD which handles both cases automatically). Both paths yield identical common components. Reference: BDF 2008, Section 2.2; researcher inference.
**Best version so far:** Run 4 — SVD approach makes explicit branching unnecessary; truncated SVD implementations choose the smaller matrix automatically.

### F56. One-step-ahead vs multi-step forecast distinction
**Category:** algorithm
**Consensus** (runs 2, 3, 4, 5, 6, 7): (1) One-step-ahead: after observing bin j, forecast bin j+1 only. This is what published accuracy numbers measure. (2) Multi-step: forecasts for all remaining bins by iteration. No published benchmark. Accuracy degrades with horizon. Only weights[0] is acted upon. Run 6 adds explicit detail: "only remaining_forecasts[0] is used for the current bin's participation fraction. The remaining elements are used only for the denominator." Reference: BDF 2008, Section 4.2.2; researcher inference.
**Best version so far:** Run 4 — dedicated "Dynamic Execution Weight Normalization" subsection (lines 741-777) explains WHY multi-step forecasts matter: they serve as the normalizing denominator that determines the magnitude of the first weight. Poor multi-step forecasts distort normalization. Most detailed mechanistic explanation.

### F57. Loop order: bin outer, stock inner for temporal consistency
**Category:** algorithm
**Consensus** (runs 2, 4, 5, 6, 7): In dynamic execution, bin loop must be OUTER and stock loop INNER. All stocks' bin j weights computed simultaneously, then all observe actuals. Run 6 execute_dynamic_vwap: "for bin_j in range(k)" outer loop. Reference: researcher inference.
**Best version so far:** Run 4 — run_dynamic_execution (lines 648-685) implements this with explicit "for j in 0..k-1" outer loop and "for i in 0..N-1" inner loop.

### F58. Example fitted parameter ranges for debugging
**Category:** parameter
**Consensus** (runs 2, 3, 4, 5, 6, 7): Typical values: r = 1-2 factors, psi_1 ~ 0.1-0.6, psi_2 ~ near zero, SETAR tau ~ near median, Var(C)/Var(X) ~ 0.6-0.8. Reference: researcher inference; BDF 2008, Fig. 2.
**Best version so far:** Run 4 — sanity checks #5 and #6 (lines 1159-1166) with actionable thresholds: "|psi_1| >= 1" and "Var(C)/Var(X) < 0.30" trigger investigation.

### F59. Eigenvalue shortcut for IC residual variance
**Category:** algorithm
**Consensus** (runs 2, 3, 4, 5, 7): V(r) computed from singular values / eigenvalues: avoids O(r*N*T) matrix reconstructions. Reference: researcher inference; standard PCA property.
**Best version so far:** Run 4 — uses SVD singular values directly: V_r = (1/(N*P)) * (total_ss - sum(s[0:r]^2)) (lines 174-178). Simplest formulation, no P<=N vs P>N branching needed.

### F60. Forecast positivity sanity check
**Category:** validation
**Consensus** (runs 2, 3, 4, 5, 6, 7): Vast majority of bin forecasts should be positive without needing floor/clip. Run 6: "Check what fraction of forecasts require clamping -- it should be very small (<1%) for liquid stocks." Reference: researcher inference.
**Best version so far:** Run 4 — sanity check #9 (lines 1177-1180): "A high rate of negative forecasts indicates model failure."

### F61. No-centering verification test
**Category:** validation
**Approach A** (runs 2, 4, 5, 7): Verify X is NOT mean-centered by running PCA with and without centering. Uncentered should produce lower MAPE. Reference: researcher inference.
**Approach B** (run 6): Run 6 centers by column means and expects this to be the correct approach (see F5 Approach B). This verification test, if run, would determine which approach is empirically better.
**Best version so far:** Run 4 — sanity check #11 (lines 1187-1189). The test itself is valuable regardless of which outcome is expected; the empirical result should settle F5.

### F62. Singular or near-singular X'X edge case
**Category:** edge case
**Consensus** (runs 2, 3, 4, 5, 6, 7): If N is very large relative to T, X'X may be ill-conditioned. Use SVD for numerical stability. Reference: researcher inference.
**Best version so far:** Run 4 — edge case #7 (lines 1231-1234): "Using SVD (as specified) rather than eigendecomposition of X'X avoids this issue by design."

### F63. Volume participation limits
**Category:** edge case
**Consensus** (runs 2, 4, 5, 6, 7): VWAP weight is a target, capped by participation rate limits (e.g., 20% of bin volume). Unexecuted volume carried to subsequent bins. Run 6: "the trader's order is small relative to market volume (no price impact). For large orders, participation-rate limits or multi-day execution is needed." Reference: researcher inference.
**Best version so far:** Run 4 — limitation #4 (lines 1254-1257): "participation rate limits should be imposed externally."

### F64. Multi-step forecast degradation as known limitation
**Category:** other
**Consensus** (runs 2, 3, 4, 5, 6, 7): Multi-step iterated forecasts degrade with horizon. No published benchmark. Mitigated by dynamic execution replacing them with fresh one-step-ahead forecasts. Reference: researcher inference.
**Best version so far:** Run 4 — "Dynamic Execution Weight Normalization" subsection (lines 762-774) explains the concrete mechanism: inflated/deflated denominators cause under/over-trading.

### F65. IC variant unspecified in BDF
**Category:** algorithm
**Consensus** (runs 2, 3, 4, 5, 6, 7): BDF 2008 does not specify which Bai-Ng IC variant used. Runs 1-5 default to IC_p2. Run 6 uses IC_p1. Run 3 provides both IC_p2 and IC_p1 formulas for robustness testing. Reference: BDF 2008, Section 2.2; Bai & Ng (2002).
**Best version so far:** Run 3 — provides both IC_p2 and IC_p1 formulas for robustness testing. See also F6 for the primary competing approaches.

### F66. SETAR degrees-of-freedom for sigma_eps
**Category:** algorithm
**Approach A** (run 2): DOF = T-1-5 (5 estimated parameters: 4 AR coefficients + 1 tau). Single sigma_eps for entire model.
**Approach B** (run 3): Per-regime DOF with n1-2 and n2-2 (2 params per regime). Unbiased estimator.
**Approach C** (run 4): Per-regime MLE (biased) with n1 and n2 denominators. sigma2_1 = sum(resid1^2)/n1, sigma2_2 = sum(resid2^2)/n2. Justified by BDF 2008 Section 2.3: "estimate by maximum likelihood." For T-1 = L*k - 1 = 519, the difference from unbiased is ~0.4%.
**Approach D** (run 6): Single sigma_eps with DOF = n-4 (4 estimated parameters: c_1, theta_12, c_2, theta_22; tau NOT counted as a degree of freedom since it is selected by grid search). sigma_eps = sqrt(best_ssr / (len(y) - 4)).
**Approach E** (run 7): Single sigma_eps, biased MLE with denominator T (full sample size, no DOF correction). sigma2 = best_ssr / T. Simplest formulation; consistent with pure MLE philosophy (no finite-sample correction).
**Best version so far:** Run 4 (Approach C) — MLE is consistent with BDF's stated estimation method, and the practical difference is negligible. Per-regime variance (shared with run 3) is more informative than single sigma_eps. However, run 6's point about tau not consuming DOF in the final OLS regression is technically valid. Run 7's Approach E is the simplest but least informative (single sigma, no DOF correction).

### F67. Forecast days derivation: 2648 = 2668 - 20
**Category:** validation
**Consensus** (runs 2, 3, 4, 5, 6): Szucs 2017: 2668 total trading days minus 20 initial estimation = 2648 forecast days. Run 6: "2,668 total days (2,648 forecast days after the initial 20-day window)." Reference: Szucs 2017, Table 1.
**Best version so far:** Run 4 — initialization section (lines 1033-1036): "The first forecast is produced for day L_days + 1."

### F68. V(r) floating-point guard in IC computation
**Category:** edge case
**Consensus** (runs 3, 4, 5, 7): When top r components capture nearly all variance (e.g., synthetic low-rank data), V(r) can become <= 0 due to floating-point arithmetic. Guard: if V_r <= 0, set V_r = 1e-15. Reference: researcher inference.
**Best version so far:** Run 4 — inline in select_factor_count (lines 179-181) with note "This r likely overfits."

### F69. SETAR estimation: single-pass vs two-phase
**Category:** algorithm
**Approach A** (runs 3, 4): Two-phase estimation. Phase 1: grid search computes only SSR for each tau candidate. Phase 2: re-fit OLS at winning tau to obtain coefficients and variance. Avoids computing variance for all n_grid candidates. Reference: researcher inference; efficiency optimization.
**Approach B** (runs 5, 6, 7): Single-pass estimation. Grid search stores winning coefficients, residuals, and regime sizes alongside SSR during the loop. No second pass needed -- variance computed once from stored residuals of winning tau. Run 6: stores best_beta alongside best_ssr and best_tau during grid search. Reference: researcher inference; efficiency optimization.
**Best version so far:** Run 5 — single-pass is strictly more efficient (same number of OLS calls during search, zero additional calls after), and the stored-state approach avoids code duplication between grid search and final fit.

### F70. Common component forecast fixed for entire trading day
**Category:** algorithm
**Consensus** (runs 3, 4, 5, 6, 7): The common component forecast is computed once before the trading day begins and is NOT updated intraday as actual volumes arrive. Only the specific component updates intraday via observed actuals. Run 6: "Model parameters remain fixed during the day. Only the conditioning information (e_last) is updated." Reference: BDF 2008, Section 2.3, paragraph below Eq. 9.
**Best version so far:** Run 4 — stated in both forecast_common (line 278: "NOT updated intraday") and run_dynamic_execution step 2 comment (line 585).

### F71. BDF "8 bp" tracking error clarification
**Category:** validation
**Single source** (run 3): BDF 2008 Section 4.3.3 states tracking error "is lower (8 bp)." This refers to a portfolio-level execution cost under a different aggregation than Table 2 MAPE. For validation, use Table 2 MAPE values directly. Reference: BDF 2008, Section 4.3.3.
**Best version so far:** Run 3

### F72. Scale-adjusted MSE* metric
**Category:** validation
**Consensus** (runs 3, 4, 5): MSE* = (1/N) * sum_{i=1}^{N} MSE_i / (a_i / a_min)^2, where a_i is average turnover of stock i and a_min is smallest average turnover. Normalizes MSE by turnover scale. Reference: Szucs 2017, Section 5, Eq. 14.
**Best version so far:** Run 4 — complete compute_mse_star function (lines 825-857) with docstring, pseudocode, and Szucs Eq. 14 reference. Directly implementable.

### F73. Pairwise stock-level validation benchmarks (Szucs Tables 2b/2c)
**Category:** validation
**Consensus** (runs 3, 4, 5): From Szucs 2017: (a) MSE pairwise: BDF_AR beats U-method 33/0, BDF_SETAR beats U-method 33/0, BDF_SETAR vs BDF_AR = 6/27 (AR wins by MSE). (b) MAPE pairwise: BDF_AR beats U-method 32/1, BDF_SETAR beats U-method 32/1, BDF_SETAR vs BDF_AR = 30/3 (SETAR wins by MAPE). Reference: Szucs 2017, Section 5, Tables 2b-2c.
**Best version so far:** Run 3 — most detailed with full table breakdown.

### F74. Non-stationary specific component fallback
**Category:** edge case
**Approach A** (runs 3, 4, 5): If AR(1) coefficient |psi_1| >= 1 or SETAR |phi_11| >= 1 or |phi_21| >= 1, specific component is non-stationary. Fall back to U-method directly. Reference: researcher inference.
**Approach B** (run 6): Clamping approach: sign(theta) * 0.99 and logs a warning; if frequent, exclude stock or fall back to U-method. Reference: researcher inference.
**Approach C** (run 7): Three-level fallback cascade: SETAR -> AR(1) -> common-component-only (specific forecast = 0). If fit_setar returns None, use AR(1); if AR(1) also non-stationary (|psi_1| >= 1), set specific forecast to 0 and use common component only. This ensures the pipeline never crashes. Reference: researcher inference.
**Best version so far:** Run 7 (Approach C) — cleanest fallback design: three-level cascade with explicit "pipeline never crashes" guarantee. Run 4's stationarity check is the most detailed individual check implementation, but run 7's cascade architecture is the most robust overall design.

### F75. No intraday events limitation
**Category:** other
**Consensus** (runs 3, 4, 5, 6, 7): Model does not handle scheduled intraday events (FOMC announcements, earnings releases) that can dramatically alter volume profile on specific days. Run 6 adds circuit breakers and trading halts to the list. Reference: researcher inference.
**Best version so far:** Run 4 — limitation #8 (lines 1274-1277).

### F76. Factor rotation indeterminacy (beyond sign)
**Category:** algorithm
**Consensus** (runs 3, 4, 5, 6, 7): PCA factors identified only up to rotation. Individual factors F_hat and loadings Lambda_hat are not uniquely interpretable, though common component C_hat = F_hat @ Lambda_hat.T is unique. Distinct from sign indeterminacy (F21): rotation indeterminacy means even the direction of factors is not meaningful, only the subspace they span. Reference: Bai 2003; BDF 2008 does not interpret individual factors.
**Best version so far:** Run 4 — note after extract_factors (lines 242-248) integrates sign and rotation indeterminacy into a single explanation: "any orthogonal transformation cancels through the product."

### F77. Variance estimator choice: MLE (biased) vs unbiased
**Category:** algorithm
**Approach A** (runs 3, 5, 6, 7): Unbiased variance estimator with DOF correction. AR(1): sigma2 = sum(residuals^2) / (n - 2). SETAR per-regime: sigma2_k = sum(resid_k^2) / (n_k - 2). Run 6: AR(1) uses DOF = n-2 (line 221); SETAR uses DOF = n-4 for single sigma_eps (see F66 Approach D). Run 5 adds explicit note that sigma2 is "diagnostic only -- not used in point forecasting."
**Approach B** (run 4): MLE (biased) variance estimator. AR(1): sigma2 = sum(residuals^2) / (T - 1). SETAR per-regime: sigma2_k = sum(resid_k^2) / n_k. Justified by BDF 2008 Section 2.3: "estimate by maximum likelihood." For typical T-1 = 519, difference from unbiased is ~0.4%. Reference: BDF 2008, Section 2.3; consistency with stated MLE estimation.
**Best version so far:** Run 4 — MLE is consistent with BDF's stated estimation method. The practical difference is negligible for typical sample sizes. The explicit justification (lines 338-345) referencing BDF's "estimate by maximum likelihood" is well-motivated.

### F78. compute_portfolio_mape validation function
**Category:** validation
**Single source** (run 4): Dedicated function computing portfolio-level MAPE matching BDF 2008 Table 2 methodology. Aggregates per-stock turnover forecasts/actuals into portfolio-level series using portfolio weights, then computes MAPE on the aggregate. Default: equal weights (1/N). Notes: BDF uses CAC40 index weights; for cap-weighted indices, market-cap weights should be used if available. Portfolio-level MAPE (~0.08) is much lower than per-stock MAPE (~0.40) because idiosyncratic errors diversify. Reference: BDF 2008, Section 4.3.3; researcher inference for equal-weight default. (Lines 860-904.)
**Best version so far:** Run 4

### F79. VWAP execution cost validation scope note
**Category:** validation
**Single source** (run 4): Explicit clarification that VWAP execution cost benchmarks (BDF 2008 Table 2 panel 3) require trade-level price data for each intraday bin, which is beyond the scope of this volume model. Primary validation targets are volume forecast MAPE and MSE. To reproduce VWAP cost benchmarks, developer would need: (a) intraday VWAP prices per bin, (b) simulated trade execution using model weights, (c) execution price vs daily VWAP spread. Reference: researcher inference for implementability scoping. (Lines 1127-1138.)
**Best version so far:** Run 4

### F80. Paper References cross-reference table
**Category:** other
**Consensus** (runs 4, 5, 6, 7): Comprehensive table mapping every spec section to its paper source. Run 6 provides 30+ entries covering: model decomposition, PCA estimation, factor count selection, common/specific component forecasting, AR(1)/SETAR models, dynamic execution, validation metrics, and researcher inferences. Reference: traceability aid.
**Best version so far:** Run 5 — 65+ entries (lines 1312-1379) with finer granularity, including separate entries for contiguous series treatment, ARMA mislabel, overnight ACF, sigma2 diagnostic note, single-pass SETAR, and C_hat non-negativity caveat.

### F81. forecasts_history output for validation
**Category:** algorithm
**Single source** (run 4): run_dynamic_execution returns both weights_history and forecasts_history. forecasts_history is a list of k arrays, each of shape (k-j, N), accumulating per-stock turnover forecasts at each step. Enables downstream per-bin per-stock MAPE validation: compare forecasts_history[j][0, i] against x_actual_today[j, i]. Without this output, validation would require re-running the forecast loop. Reference: researcher inference for validation convenience. (Lines 573-576, 644-645, 673.)
**Best version so far:** Run 4

### F82. Event-driven execute_one_bin architecture
**Category:** algorithm
**Single source** (run 5): Decompose dynamic execution into a single-bin function (execute_one_bin) called by an external event loop, instead of a monolithic run_dynamic_execution function that loops over all bins internally. The function takes model, stock_idx, bin_idx, shares_remaining, and last_specific as inputs and returns shares_to_trade, updated_last_specific, and shares_after. State update (last_specific, shares_remaining) done inline by the caller. Includes example usage pattern showing the external event loop with inline state update (lines 738-765). Reference: researcher inference; architectural choice for event-driven execution systems.
**Best version so far:** Run 5

### F83. Last-bin guarantee: full order completion at market close
**Category:** algorithm
**Consensus** (runs 5, 6, 7): At the last bin (j = k-1), only one forecast remains, so compute_vwap_weights returns [1.0] and shares_to_trade = shares_remaining. This guarantees the full order is completed by market close without special-case logic. Follows directly from the weight normalization. Run 6's execute_dynamic_vwap loop achieves this naturally: when bin_j = k-1, remaining_forecasts has one element, fraction = 1.0. Reference: researcher inference; BDF 2008 Section 4.2.2.
**Best version so far:** Run 5

### F84. Evaluation protocol: one-step-ahead emphasis and static-evaluation warning
**Category:** validation
**Consensus** (runs 5, 6, 7): Detailed evaluation protocol with explicit warning: a developer who evaluates using static next-day forecasts (all k bins predicted at market open without intraday updates) will get substantially worse numbers and incorrectly conclude the implementation is wrong. Run 6: "Static (multi-step) forecasts will produce substantially higher error rates and should not be compared against the benchmarks below." Run 5 includes step-by-step protocol and Szucs 2017 Section 3 quote. Reference: Szucs 2017, Section 3; BDF 2008, Section 2.3.
**Best version so far:** Run 5 — includes the Szucs 2017 quote ("While parameters are updated daily, the information base for the forecast is updated every 15 minutes"), step-by-step protocol, and 1-based to 0-based indexing mapping.

### F85. BIC-based model selection alternative for AR(1) vs SETAR
**Category:** algorithm
**Single source** (run 7): Explicit BIC formulas for model selection as an alternative to residual variance comparison: BIC_ar = T * ln(sigma2_ar) + 2 * ln(T), BIC_setar = T * ln(sigma2_setar) + 5 * ln(T). Select model with lower BIC. Accompanied by detailed analysis: because SETAR has 5 free parameters vs AR(1)'s 2, residual variance comparison is effectively "always use SETAR unless it catastrophically fails." BIC provides proper penalization for the extra parameters. Neither BDF 2008 nor Szucs 2017 provides a model selection procedure. Reference: researcher inference.
**Best version so far:** Run 7

### F86. Closing auction limitation
**Category:** other
**Single source** (run 7): The model treats all bins equally and does not separately handle the closing auction, which can represent 10-15% of daily volume in modern markets. BDF 2008 data predates the rise of closing auctions. Reference: researcher inference — BDF 2008 predates modern closing auctions.
**Best version so far:** Run 7

### F87. Event-day augmentation reference: Markov et al. (2019)
**Category:** other
**Single source** (run 7): For production deployment, run 7 suggests augmenting the BDF model with event-day adjustments as in Markov et al. (2019) to handle extreme volume days (earnings, index rebalancing, major news). No other run references this paper. Reference: run 7, Edge Case #5; Markov et al. 2019.
**Best version so far:** Run 7

**Run 1 audit:** VERDICT: [51 new, 0 competing, 0 reinforcing] — first run, baseline established
**Run 2 audit:** VERDICT: [16 new, 4 competing, 46 reinforcing] — new findings still emerging, continue
**Run 3 audit:** VERDICT: [10 new, 6 competing, 56 reinforcing] — new findings still emerging, continue
**Run 4 audit:** VERDICT: [4 new, 2 competing, 71 reinforcing] — diminishing returns, consider stopping
**Run 5 audit:** VERDICT: [3 new, 1 competing, 77 reinforcing] — diminishing returns, consider stopping
**Run 6 audit:** VERDICT: [0 new, 4 competing, 68 reinforcing] — no new findings, recommend stopping
**Run 7 audit:** VERDICT: [3 new, 3 competing, 72 reinforcing] — minor new findings only (BIC model selection, closing auction limitation, Markov 2019 reference), diminishing returns, recommend stopping

**Run 7 audit:** VERDICT: [3 new, 3 competing, 72 reinforcing] — minor new findings only (BIC model selection, closing auction limitation, Markov 2019 reference), diminishing returns, recommend stopping
