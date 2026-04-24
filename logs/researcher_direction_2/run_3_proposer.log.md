## Status: done
## Current Step: Draft 3 complete (revision of draft 2 based on critique 2)

### [2026-04-13 07:24] Session started — Proposer, Direction 2, Run 3
- Role: Proposer (initial draft)
- Direction: PCA Factor Decomposition (BDF 2008)
- Assigned papers: BDF 2008 (foundational), Szucs 2017 (comparison)
- No prior drafts exist in run_3 — producing initial spec

### [2026-04-13 07:24] Papers read
- Read BDF 2008 summary (.md) and full PDF (14 pages)
- Read Szucs 2017 summary (.md) and full PDF (10 pages)

Key insights from BDF 2008:
- Additive decomposition: x_{i,t} = c_{i,t} + e_{i,t}
- PCA via Bai (2003) on turnover matrix X (P x N), factors via eigendecomposition
- Common component forecast: historical average at same time-of-day over L prior days
- Specific component: AR(1) or SETAR(2-regime) per stock
- SETAR threshold tau estimated per stock via MLE
- Dynamic VWAP execution updates forecasts intraday bin-by-bin
- k=25 bins (20-min intervals, 9:20-17:20 Paris), L=20 days
- Rolling 1-month PCA window (~500 obs), 20-day time series window
- Portfolio MAPE: PCA-SETAR 0.0752, PCA-ARMA 0.0829, classical 0.0905
- Dynamic VWAP TE reduction ~10% vs classical

Key insights from Szucs 2017:
- Replicates BDF on US data (33 DJIA stocks, 11 years)
- k=26 bins (15-min intervals, 9:30-16:00 US), L=20 days
- Turnover = volume / TSO (total shares outstanding, not float)
- Uses AR(1) for specific component (not ARMA(1,1) — Szucs Eq. 5: e_p = c + theta_1 * e_{p-1} + eps)
- BDF_SETAR MAPE 0.399, BDF_AR MAPE 0.403, U-method 0.503
- BDF runs in ~2 hours vs 60+ machine-days for BCG
- 2648 forecast days (2668 total days - 20 estimation window)
- One-step-ahead forecasting: parameters fixed daily, information base updated every 15 min

### [2026-04-13 07:25] Ambiguities identified and resolved
1. k=25 vs k=26: BDF uses 25 (20-min, Paris market), Szucs uses 26 (15-min, US market). Resolution: parameterize k as configurable; document both settings.
2. Float vs TSO: BDF uses shares/float, Szucs uses volume/TSO. Resolution: document both; TSO is more readily available.
3. AR(1) vs ARMA(1,1): BDF Eq. 10 includes intercept psi_2 (AR(1) with drift), not a true ARMA(1,1). Szucs Eq. 5 confirms AR(1) with constant. Resolution: implement AR(1) with constant term.
4. Bai-Ng IC for factor count: BDF references Bai & Ng (2002) but does not specify which of the 6 IC criteria. Resolution: use IC_p2 (most common in practice), note alternatives.
5. SETAR estimation: BDF does not detail the MLE procedure. Resolution: describe grid search over tau + conditional OLS, which is standard for SETAR.
6. Demeaning in PCA: BDF does not explicitly state whether the turnover matrix is demeaned before PCA. Bai (2003) works with raw data (no demeaning required for consistency). Resolution: do NOT demean — the common component should capture both level and shape.

### [2026-04-13 07:26] Drafting implementation specification

### [2026-04-13 07:28] Draft 1 complete
- Wrote impl_spec_draft_1.md (full implementation specification)
- 11 functions with complete pseudocode: estimate_model, select_num_factors, extract_factors,
  forecast_common, fit_ar1, fit_setar, forecast_next_bin, compute_vwap_weights,
  run_dynamic_execution, daily_update, u_method_benchmark
- 10 parameters documented with sensitivity and ranges
- Data preprocessing section with explicit no-centering warning
- Data flow diagram with shapes and types
- Validation: 3 benchmark tables (Szucs per-stock, BDF portfolio, BDF VWAP execution)
- 10 sanity checks, 7 edge cases, 7 known limitations
- 7 researcher inferences explicitly marked
- All algorithmic steps traced to specific paper sections/equations
- Key ambiguities resolved: k=25 vs k=26, float vs TSO, AR(1) vs ARMA(1,1), IC variant,
  SETAR estimation method, centering question
- Awaiting critic review

### [2026-04-13 07:42] Critique 1 received — assessment
- 5 major issues, 9 minor issues
- Severity: M1 (correctness), M2 (performance), M3 (completeness), M4 (clarity), M5 (clarity)
- All major issues are valid and well-evidenced
- Minor issues are mostly precision/guidance improvements
- All will be addressed in draft 2

Major changes planned:
1. M1: Replace dual-case eigendecomposition with unified SVD approach
2. M2: Single eigendecomposition in select_num_factors, V(r) from eigenvalues
3. M3: Add Validation Metrics subsection with MSE/MAPE/MSE* formulas and aggregation
4. M4: Clarify V(r)-to-eigenvalue relationship with explicit formulas
5. M5: Add explicit bootstrapping confirmation note

Minor changes:
- m1: Fix citation 4.3.2 -> 4.3.3
- m2: Specify variance as sum(resid^2)/(n-2) for AR(1), (n_regime-2) for SETAR
- m3: Use neutral regime labels instead of "calm"/"turbulent"
- m4: Correct pairwise numbers to match Szucs Table 2 exactly
- m5: Define fallback: always use SETAR when estimation succeeds
- m6: Add TSO stale/missing data handling guidance
- m7: Add explicit note about periodic ACF at lag k
- m8: Add note about common component NOT updating intraday
- m9: Clarify 2648 out-of-sample vs 2668 total trading days

### [2026-04-13 07:47] Draft 2 complete
- All 5 major issues addressed
- All 9 minor issues addressed
- extract_factors: unified SVD approach (M1)
- select_num_factors: single eigendecomposition, V(r) from eigenvalues with explicit formula (M2, M4)
- New Validation Metrics subsection with MSE, MAPE, MSE* formulas and aggregation procedure (M3)
- Dynamic execution bootstrapping note expanded (M5)
- Pairwise comparison numbers corrected from Szucs Table 2 (m4)
- Sanity checks expanded from 10 to 11, split U-method comparison into MSE and MAPE (m4)
- Citation 4.3.2 corrected to 4.3.3 (m1)
- Variance estimators use n-2 denominator (m2)
- Neutral regime labels (m3)
- SETAR fallback: always use when estimation succeeds (m5)
- TSO data handling guidance added to preprocessing (m6)
- Overnight gap periodic ACF note added (m7)
- Common component fixed intraday note added (m8)
- 2648 out-of-sample days clarified (m9)
- Awaiting critic review or human approval

### [2026-04-13 07:55] Critique 2 received — assessment
- 0 major issues, 5 minor issues
- Critic declares draft 2 "implementation-ready"
- All 5 minor issues are valid polish/optimization suggestions
- All will be addressed in draft 3

Changes planned:
1. m1: Add implementation note about sharing single SVD between select_num_factors and extract_factors
2. m2: Fix "8 bp" ambiguity — clarify as portfolio-level metric, direct validation to Table 2 MAPEs
3. m3: Restructure fit_setar into two phases (grid search, then final estimation for winner)
4. m4: Add guidance on maintaining fixed cross-section for PCA stability
5. m5: Add numerical guard for V(r) <= 0 in IC computation

### [2026-04-13 07:55] Draft 3 complete
- All 5 minor issues from critique 2 addressed
- m1: Added implementation optimization note to estimate_model with explicit instructions for shared SVD
- m2: Removed ambiguous "8 bp" from main text, added clarification note explaining it as portfolio-level metric
- m3: Restructured fit_setar into Phase 1 (grid search tracking only best_tau/best_ssr) and Phase 2 (final estimation with variance for winning tau only)
- m4: Expanded edge case 2 with explicit guidance: maintain fixed cross-section, re-estimate from scratch on reconstitution
- m5: Added V_r <= 0 guard with 1e-15 floor and explanatory comment in select_num_factors
- Updated Paper References table with 5 new entries
- Awaiting critic review or human approval
