## Status: done
## Current Step: Revised draft 3 complete

### Log

[2026-04-13 18:54] Started proposer for direction 2 run 4.
[2026-04-13 18:54] No prior drafts exist in run_4/ directory. Producing initial spec.
[2026-04-13 18:54] Read paper summaries: BDF 2008, Szucs 2017.
[2026-04-13 18:54] Read full PDFs: BDF 2008 (all 14 pages), Szucs 2017 (all 10 pages).
[2026-04-13 18:54] Key insights from BDF 2008:
  - Additive decomposition x = c + e via PCA on cross-section of turnovers
  - Common component forecast via time-of-day averaging (Eq. 9)
  - Specific component modeled by AR(1) (Eq. 10) or SETAR (Eq. 11)
  - Dynamic VWAP execution updates forecasts after each observed bin
  - SETAR outperforms AR(1) for 36/39 stocks
  - PCA uses Bai (2003) framework, factor count via Bai & Ng (2002) IC
[2026-04-13 18:54] Key insights from Szucs 2017:
  - Controlled comparison on 33 DJIA stocks, 11 years
  - BDF-SETAR MAPE 0.399, BDF-AR 0.403, U-method 0.503
  - BDF orders of magnitude faster than BCG
  - Correctly labels BDF Eq. 10 as AR(1) (not ARMA(1,1))
  - Szucs notation: Eq. 5 = AR(1), Eq. 6 = SETAR, Eq. 7 = indicator
[2026-04-13 18:55] Writing implementation specification draft 1.
[2026-04-13 19:00] Completed impl_spec_draft_1.md.
  - 11 functions in 4 groups: data prep, PCA decomposition, time-series modeling, dynamic execution
  - Unified SVD-based factor extraction (avoids P<=N vs P>N branching)
  - Complete SETAR two-phase estimation (grid search then final fit)
  - Dynamic execution loop with bin-by-bin updating
  - U-method baseline included
  - Full parameter table with sources
  - 11 sanity checks, 7 edge cases, 10 known limitations
  - Paper reference traceability table
  - All researcher inferences explicitly marked
[2026-04-13 19:04] Resuming: critique 1 exists. 3 major, 7 minor issues. All text/documentation fixes, no algorithmic changes needed.
[2026-04-13 19:04] Re-read BDF 2008 pp.1-6 (turnover definition, ARMA/AR labeling, dynamic execution).
[2026-04-13 19:04] Re-read Szucs 2017 pp.3-13 (TSO definition, MAPE/MSE formulas, Eq.14 MSE*, results tables).
[2026-04-13 19:05] Verified M1: BDF 2008 p.1711 says "floated shares", Szucs 2017 p.4 says "TSO". Critic is correct.
[2026-04-13 19:05] Verified M2: BDF 2008 p.1712 labels Eq.10 "ARMA(1,1)" but equation has no MA term. Szucs Eq.5 labels it AR(1).
[2026-04-13 19:05] Verified M3: BDF 2008 Sec 4.2.2 confirms dynamic updating; multi-step normalization explanation added.
[2026-04-13 19:13] Completed impl_spec_draft_2.md. All 10 critique issues addressed:
  M1: Expanded turnover definition with explicit paper disagreement, choice rationale, benchmark comparability notes
  M2: Three-point AR(1) justification (equation, Szucs confirmation, OLS/MLE consistency)
  M3: New "Dynamic Execution Weight Normalization" subsection explaining multi-step role
  m1: Unified select_factor_count to use SVD singular values, eliminating P>=N branch
  m2: Changed variance denominator to T-1 (MLE) with explicit justification
  m3: Rewrote notation table parentheticals to "(regime N intercept/AR coeff)"
  m4: Added compute_mape and compute_mse_star functions in Group 5
  m5: Added overnight transition comment at E_hat[-1,:] initialization
  m6: Implemented stationarity check as explicit code with use_u_method flag and fallback
  m7: Added assert day_index >= L_days precondition
[2026-04-13 19:19] Resuming: critique 2 exists. 0 major, 3 minor issues. All non-blocking quality improvements.
[2026-04-13 19:19] Assessment of critique 2 issues:
  m1-new (minor): forecasts_history in return doc but never constructed — pseudocode consistency fix
  m2-new (minor): portfolio-level MAPE function missing — add compute_portfolio_mape
  m3-new (minor): VWAP execution cost benchmarks require price data — add scope note
[2026-04-13 19:20] Producing impl_spec_draft_3.md. Changes:
  m1-new: Added forecasts_history list init, per-step population (all_forecasts array),
          return alongside weights_history. Updated return docs with shape/usage notes.
  m2-new: Added compute_portfolio_mape function (14th function). Weighted-average
          aggregation across stocks per bin, then MAPE on aggregate. Default equal-weight
          with optional portfolio_weights param. Cites BDF 2008 Section 4.3.3.
  m3-new: Added "NOTE ON VWAP EXECUTION COST VALIDATION" paragraph after execution cost
          table. Explains price data requirement, scopes primary validation to MAPE/MSE*/
          portfolio MAPE, and lists steps to reproduce VWAP cost if price data available.
  Also: updated function count (13 -> 14), Data Flow output section, Paper References table.
[2026-04-13 19:20] Completed impl_spec_draft_3.md.
