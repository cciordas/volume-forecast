## Status: done
## Current Step: Draft 3 complete

### [2026-04-12 22:12] Session started
- Role: Proposer, Direction 2 (PCA Factor Decomposition / BDF), Run 2
- No existing drafts in run_2 directory; producing initial spec
- Papers assigned: BDF 2008 (foundational), Szucs 2017 (comparison)

### [2026-04-12 22:12] Paper reading: BDF 2008
- Read full 14-page paper. Key insights:
  - Additive decomposition: x = c + e, common + specific
  - PCA via eigendecomposition of X'X matrix (Eq 6), Bai (2003) framework
  - Normalization: F'F/T = I_r
  - Common component forecast: time-of-day average over L days (Eq 9)
  - Specific component: ARMA(1,1) (Eq 10) or SETAR (Eq 11)
  - Note: Eq 10 labeled "ARMA(1,1)" but has no MA term -- actually AR(1) with intercept
  - Three VWAP strategies: theoretical (upper bound), static (poor), dynamic (recommended)
  - Dynamic VWAP updates forecasts after each observed bin
  - Results: PCA-SETAR MAPE 0.0752 portfolio-level, reduces VWAP TE by ~10%
  - 39 CAC40 stocks, 25 bins of 20 min, 20-day rolling window

### [2026-04-12 22:12] Paper reading: Szucs 2017
- Read full 7-page paper. Key insights:
  - Head-to-head comparison of BDF vs BCG (CMEM) on 33 DJIA stocks, 11 years
  - 26 bins of 15 min, 20-day rolling window
  - BDF-SETAR best by MAPE (0.399), BDF-AR best by MSE (6.49e-4)
  - Both BDF variants dominate BCG and U-method
  - Turnover = V/TSO (total shares outstanding)
  - Computation: BDF ~2 hours vs BCG ~60 machine-days
  - MSE and MAPE formulas: Eqs 1-2
  - U-method benchmark: Eq 3
  - Per-stock per-bin MAPE (not portfolio-level as in BDF 2008)

### [2026-04-12 22:13] Producing initial implementation spec draft 1
- Full spec written to work/researcher/direction_2/run_2/impl_spec_draft_1.md
- Algorithm: 2-phase pipeline (offline PCA + online dynamic forecasting)
  - Phase 1: Construct X matrix, eigendecompose X'X/T, extract r factors via Bai-Ng IC,
    compute common/specific components, forecast common via time-of-day average,
    fit AR(1) and SETAR per stock
  - Phase 2: Dynamic forecasting -- observe actual, extract specific, forecast ahead
    via AR/SETAR, combine with common, compute VWAP weights
- Parameters: 15 parameters documented with sensitivity and ranges
- Validation: 4 benchmark targets (per-stock MAPE, MSE, portfolio MAPE, VWAP TE),
  10 sanity checks, 7 edge cases, 7 known limitations
- Researcher inferences marked: day-start e=0 initialization, SVD alternative,
  SETAR grid resolution/min regime, multi-step SETAR deterministic iteration,
  negative forecast floor, factor sign convention, cross-day boundary issue
- All algorithmic steps traced to paper sections with equation numbers

### [2026-04-12 22:25] Critique assessment and draft 2 revision
- Received critique with 5 major, 8 minor, 3 suggestion issues
- All issues are well-founded and actionable; no structural rewrite needed

**Major issues addressed:**
- M1 (Demeaning/centering): Added explicit "Do NOT mean-center" warning in pseudocode
  (estimate_model Step 1), extract_factors docstring, select_num_factors docstring,
  Initialization section, Data Flow section, and new sanity check 11. This was the
  highest-priority fix -- silent bug risk.
- M2 (One-step vs multi-step validation): Restructured Validation section into subsections:
  "One-Step-Ahead Forecast Accuracy (Szucs benchmarks)", "Portfolio-Level MAPE (BDF
  benchmarks)", "VWAP Tracking Error (BDF benchmarks)", and "Note on Multi-Step Forecast
  Accuracy". Each subsection annotated with k value, dataset, and forecast type.
- M3 (Bin count discrepancy): Updated Parameters table to show both k=25 (BDF/Euronext)
  and k=26 (Szucs/NYSE). Added "Note on k_bins" below table. Annotated each validation
  subsection with the corresponding k and dataset.
- M4 (forecast_dynamic conflates two use cases): Split into two functions:
  forecast_next_bin (high-accuracy one-step) and compute_vwap_weights (multi-step
  approximation). Added run_dynamic_execution showing the full execution loop. Added
  clear documentation distinguishing the two use cases.
- M5 (PCA dimension T > N): Added assertion r <= min(N, T) in extract_factors. Added
  dual-path implementation: N <= T uses X'X/T, N > T uses XX'/T. Marked large-N path
  as researcher inference.

**Minor issues addressed:**
- m1 (SETAR DOF): Added note in fit_setar that tau DOF treatment is an approximation,
  citing Hansen (1997).
- m2 (Bai & Ng IC variant): Added note in select_num_factors that BDF does not specify
  which IC variant; IC_p2 is our default; robustness check via multiple criteria suggested.
- m3 (SETAR fallback): Added AR(1) fallback when no valid tau found in fit_setar.
- m4 (C_hat forecast not truly OOS): Added NOTE in estimate_model Step 5 explaining
  this is a design choice from BDF, not a bug.
- m5 (Missing U-method): Added forecast_u_method function with pseudocode and reference
  to Szucs Eq. 3. Updated sanity check 10 to reference this function.
- m6 (Factor sign convention incomplete): Extended Initialization section to trace the
  full chain: signs cancel in C = F @ A, E = X - C is sign-invariant, therefore AR/SETAR
  params are unaffected.
- m7 (VWAP rebalancing): Added run_dynamic_execution showing the full execution loop.
  Added edge case 9 for volume participation limits.
- m8 (Turnover float vs TSO): Added explicit discussion in preprocess_turnover, Parameters
  note, Known Limitations item 6, and Paper References table.

**Suggestions addressed:**
- S1 (Data preprocessing): Added full preprocess_turnover function covering turnover
  computation, half-day exclusion, split handling, zero-volume handling, optional
  Winsorization.
- S2 (Computational complexity): Added new "Computational Complexity" section with
  analysis for PCA, AR/SETAR, Phase 2, and reference benchmark from Szucs 2017.
- S3 (BDF vs U-method relationship): Added explanation in estimate_model Step 5 and
  forecast_u_method docstring noting that BDF common forecast = U-method on C_hat,
  and improvement comes from specific component forecast.

### [2026-04-12 22:42] Critique 2 assessment and draft 3 revision
- Received critique 2 with 0 major, 4 minor, 2 suggestion issues
- All issues are minor refinements; spec is ready for implementation per critic
- Producing draft 3 with targeted fixes

**Minor issues addressed:**
- m1 (SETAR notation mapping): Added detailed notation mapping comment in fit_setar
  showing correspondence between spec phi_11/phi_12 and BDF Eq. (11) phi_1/phi_2 and
  Szucs Eq. (6) theta_{1,2}/c_{1,1}. Added warning about coefficient ordering difference.
- m2 (Eigenvalue shortcut): Added optimization comment in select_num_factors noting
  that sigma_sq_r can be computed from eigenvalues directly, avoiding matrix
  reconstructions. Retained explicit approach for pedagogical clarity.
- m3 (Loop order): Swapped run_dynamic_execution to bin-outer/stock-inner loop order.
  Added e_last dict for per-stock state tracking. Added comment explaining temporal
  consistency requirement for live execution.
- m4 (50% negative fallback): Added explicit check in compute_vwap_weights counting
  floored forecasts; if >50% floored, recomputes using common-component-only. Updated
  Edge Cases item 3 to cross-reference both pseudocode mechanisms.

**Suggestions addressed:**
- S1 (Example parameter values): Added paragraph in Calibration section with typical
  ranges for a DJIA component: r=1-2, psi_1~0.3-0.7, psi_2~0.001-0.005, tau~0,
  Var(C)/Var(X)~0.6-0.8. Sourced from BDF Fig. 2 ACF/PACF; marked as researcher inference.
- S2 (2648 derivation): Added inline arithmetic "= 2668 total trading days per Szucs
  Table 1 minus 20 initial estimation days" to the validation section.

Updated Paper References table with 6 new entries for all additions.
