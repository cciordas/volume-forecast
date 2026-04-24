## Status: done
## Current Step: Draft 3 complete, addressing all critique 2 issues

### Log

[2026-04-13 21:51] Started proposer session for direction 2 run 6.
[2026-04-13 21:51] No prior drafts found in work/researcher/direction_2/run_6/. Producing initial spec.
[2026-04-13 21:51] Read paper summaries: bialkowski_darolles_lefol_2008.md, szucs_2017.md.
[2026-04-13 21:52] Read full PDFs for both papers.

Key insights from papers:
- BDF 2008 (foundational): Additive factor decomposition x_{i,t} = c_{i,t} + e_{i,t}. PCA on cross-section of turnovers extracts common U-shape. Specific component modeled by ARMA(1,1) or SETAR. Dynamic VWAP execution updates forecasts intraday. 39 CAC40 stocks, 20-min bins (k=25), L=20 day rolling window.
- Szucs 2017 (comparison): Replicates BDF on 33 DJIA stocks, 15-min bins (m=26), 11 years. BDF dominates CMEM on both MSE and MAPE. BDF_AR uses AR(1) not ARMA(1,1). Estimation ~2 hours vs 60 machine-days for CMEM. Szucs uses simpler AR(1) for specific component (Eq 5).

Ambiguities resolved:
- BDF paper says ARMA(1,1) but the equation (Eq 10) shows e_{i,t} = psi_1 * e_{i,t-1} + psi_2 + epsilon, which is actually AR(1) with intercept, not ARMA(1,1). Szucs also implements it as AR(1). Will implement as AR(1) with intercept.
- SETAR model: two-regime AR(1) with threshold on lagged residual. Threshold tau estimated per stock.
- Number of factors r: selected by Bai & Ng (2002) information criteria. Paper uses IC_p1 criterion.
- PCA estimation: Bai (2003) large-dimensional factor analysis via eigendecomposition.

[2026-04-13 21:53] Writing implementation specification draft 1.
[2026-04-13 21:55] Draft 1 complete. Covers:
- Full algorithm with pseudocode for PCA extraction, AR(1) and SETAR fitting, dynamic intraday forecasting, and VWAP scheduling.
- Data flow diagram with shapes/types at each stage.
- 11 parameters with recommended values, sensitivity, and ranges.
- Initialization and calibration procedures.
- Validation section with 5 expected behaviors (quantitative benchmarks from both papers), 7 sanity checks, 6 edge cases, 7 known limitations.
- Paper reference traceability table with 16 entries, including 4 explicitly marked as Researcher inference.
- Resolved ARMA(1,1) vs AR(1) ambiguity in BDF 2008 by cross-referencing Szucs 2017.

[2026-04-13 22:03] Read critique 1. Assessment: 4 major, 8 minor issues.
- M1 (centering ordering bug): Valid. select_num_factors receives uncentered X but PCA uses centered X. Will fix by centering before factor selection.
- M2 (Table 5 citation): Valid. Dynamic VWAP ~0.0898 is in Table 2 third panel, not Table 5. Table 5 has per-stock values only.
- M3 (PCA on P x P vs N x N): Valid. BDF Sec 2.2 explicitly says "eigenvectors of X'X matrix" which is (N x N). Will switch to SVD-based approach.
- M4 (centering as Researcher inference): Valid. Neither BDF nor Szucs prescribes centering. Will label as Researcher inference.
- m1 (OLS vs MLE equivalence): Valid. Will add note.
- m2 (2,648 vs 2,668): Partially valid. 2,668 is total sample days (Szucs Table 1); 2,648 = 2668-20 is forecast days. Will clarify.
- m3 (MAPE formula missing): Valid. Will add Szucs Eqs. 1-2 with zero-division note.
- m4 (multi-step chaining documentation): Valid. Will add clarifying comment.
- m5 (r_max justification): Valid. Will note as Researcher inference.
- m6 (overnight discontinuity): Valid. Will add to Known Limitations.
- m7 (in-sample vs out-of-sample labels): Valid. Will label each benchmark clearly.
- m8 (theta_1 >= 1 handling): Valid. Will add fallback guidance.

All issues will be addressed in draft 2.

[2026-04-13 22:04] Re-read BDF 2008 Sections 2.2-2.3, 3.1-3.2, 4.2, Tables 2-5. Re-read Szucs 2017 Sections 2-3, Table 1, Eqs. 1-2. Verified:
- BDF Sec 2.2: "The estimated factors matrix F_hat is proportional (up to T^{1/2}) to the eigenvectors corresponding to the r-largest eigenvalues of the X'X matrix" confirms N x N approach.
- BDF Sec 2.2: No centering prescribed. Paper goes straight from X to eigendecomposition.
- BDF Sec 2.3: "estimate Eqs. (10) or (11) by maximum likelihood"
- BDF Table 2: Three panels. Third panel "Result of out-of-sample estimation for VWAP execution": PC/SETAR with dynamical adjustment = 0.0898 Mean MAPE.
- Szucs Table 1: 2,668 days total. 2,648 = 2668-20 is the number of forecast/re-estimation days.
- Szucs Eq 1: MSE = sum((Y_t - Y_t^f)^2) / N. Eq 2: MAPE = sum(|Y_t - Y_t^f|/Y_t) / N.

[2026-04-13 22:05] Writing draft 2.

[2026-04-13 22:10] Draft 2 complete. Changes from draft 1:

Major issues addressed:
- M1: Moved centering step before factor selection. X_centered is now computed at Step 2 of Phase 1 and passed to select_num_factors. Factor selection and PCA extraction both operate on the same centered data.
- M2: Corrected VWAP citation from "Table 5, column 1 mean over all stocks" to "Table 2, third panel, PC/SETAR with dynamical adjustment row, Mean = 0.0898." Added Table 4 (theoretical) and Table 5 (dynamic) as per-stock references.
- M3: Replaced (P x P) eigendecomposition with thin SVD approach. X_centered = U @ diag(S) @ V^T; F_hat = U[:,:r] * sqrt(P); Lambda_hat = V[:,:r] * S[:r] / sqrt(P). Added implementation note in select_num_factors that SVD need only be computed once. Updated Data Flow to show SVD pipeline.
- M4: Explicitly labeled column-mean centering as "Researcher inference" with justification (standard PCA practice, ensures factors capture shape variation, means added back for common forecast). Added a dedicated [Column-Mean Centering] step in Data Flow.

Minor issues addressed:
- m1: Added OLS-MLE equivalence notes to Phase 1 (Step 6), fit_ar1, fit_setar, and Data Flow. Cited BDF 2008 Section 2.3: "estimate ... by maximum likelihood."
- m2: Clarified to "2,668 total days (2,648 forecast days after the initial 20-day window)" in Validation. Cited Szucs 2017 Table 1.
- m3: Added new "Evaluation Metrics" subsection with MSE (Eq. 1) and MAPE (Eq. 2) formulas from Szucs 2017. Added zero-division note and handling guidance.
- m4: Added detailed comment block at top of forecast_intraday explaining multi-step chaining, why it exists, and how execute_dynamic_vwap uses only remaining_forecasts[0]. Added corresponding note in Data Flow.
- m5: Labeled r_max = 20 as Researcher inference with justification (generous for volume data, Bai & Ng recommend "reasonable upper bound").
- m6: Added Known Limitation #8: overnight discontinuity in concatenated specific component series.
- m7: Labeled each MAPE benchmark as "in-sample" (BDF Table 2 first panel) or "rolling out-of-sample" (Szucs full sample) or "out-of-sample" (BDF Table 2 third panel VWAP).
- m8: Added stationarity check with clamping to fit_ar1 and fit_setar. Added Edge Case #7 for non-stationary coefficients. Updated parameter table ranges to note clamping.

Additional improvements:
- Added OLS-MLE equivalence citation to Paper References table.
- Added MSE/MAPE metric citations to Paper References table.
- Added per-stock VWAP table references (Tables 4 and 5) to Paper References.
- Paper References table expanded from 16 to 28 entries.
- Added implementation note after select_num_factors: SVD computed once, reused for all r_candidate values.

[2026-04-13 22:17] Read critique 2. Assessment: 0 major, 4 minor issues.
- m1 (SVD inside loop): Valid. Pseudocode had SVD call inside the for loop despite implementation note saying to compute once. Will move SVD above the loop and remove redundant note.
- m2 (U-method formula missing): Valid. Validation references U-method benchmark but never defines it. Will add formula from Szucs 2017 Eq. 3 / BDF 2008 footnote 5.
- m3 (Evaluation protocol unstated): Valid. Benchmarks assume one-step-ahead dynamic forecasts but this is not explicitly stated in validation section. Will add protocol description.
- m4 (Data layout comment): Valid. Step 5 indexing arithmetic relies on implicit data layout. Will add clarifying comment.

All 4 issues are minor and will be addressed in draft 3.

[2026-04-13 22:18] Writing draft 3. Changes from draft 2:

Minor issues addressed:
- m1: Moved SVD call above the for loop in select_num_factors. Removed the redundant "Implementation note" paragraph that previously compensated for the in-loop placement. The pseudocode now directly computes SVD once and iterates over truncation levels.
- m2: Added new "U-Method Benchmark" subsection to Validation, defining the formula x_hat_U[bin_j, stock_i] = (1/L) * sum_{d=0}^{L-1} x[bin_j + d*k, stock_i]. Cited Szucs 2017 Eq. 3 and BDF 2008 footnote 5.
- m3: Added new "Evaluation Protocol" subsection to Validation, explicitly stating that all benchmarks use one-step-ahead dynamic forecasts. Includes a 3-step procedure for reproducing benchmark numbers. Cited BDF 2008 Section 4.2.2 and Szucs 2017 Section 3.
- m4: Added data layout comment to Step 5 of Phase 1 pseudocode, documenting that X is organized as [day_0_bin_0, ..., day_0_bin_{k-1}, day_1_bin_0, ...] so bin_j of day d is at row index bin_j + d * k.

Additional changes:
- Added 2 new entries to Paper References table: U-method formula and evaluation protocol citations.
- Paper References table now has 30 entries.
