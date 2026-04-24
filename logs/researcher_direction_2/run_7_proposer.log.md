## Status: done
## Current Step: Draft 3 complete

### Log

#### [2026-04-13 22:17] Session started
- Role: Proposer, Direction 2, Run 7
- No prior drafts exist — producing initial implementation spec

#### [2026-04-13 22:18] Papers read
- Read all 8 paper summaries for orientation
- Read BDF 2008 (full PDF, 14 pages) — foundational paper for direction 2
  - Key: additive decomposition x = c + e, PCA via eigendecomposition of X'X, Bai (2003) framework
  - Volume measured as turnover (shares / float), NOT shares / TSO
  - Eq (6): V(r) minimization, F'F/T normalization (note: paper uses T, not P)
  - Eq (9): common component forecast as time-of-day average across L days
  - Eq (10): "ARMA(1,1)" — but equation has NO MA term, just AR(1) with intercept
  - Eq (11): SETAR two-regime threshold model
- Read Szucs 2017 (full PDF, 7 pages) — independent validation on DJIA
  - Confirms BDF dominance over BCG on 11-year US data
  - Uses turnover = V / TSO (different denominator than BDF's float)
  - Eq (5): correctly labels the specific component model as AR(1), not ARMA(1,1)
  - Eq (6): SETAR with clear notation
  - 26 bins (15-min), 20-day window, 33 stocks

#### [2026-04-13 22:20] Key insights for implementation
- BDF uses float as denominator; Szucs uses TSO. Both work. TSO is more practical.
- BDF's "ARMA(1,1)" label is misleading — Eq (10) is AR(1) with intercept. Szucs confirms.
- Factor count selection via Bai & Ng (2002) IC_p2 criterion.
- No demeaning before PCA — Bai (2003) framework absorbs means into loadings.
- Dynamic VWAP execution: forecast at open, revise each bin with actuals.
- SETAR threshold estimated via grid search on e_{i,t-1} quantiles.

#### [2026-04-13 22:25] Draft 1 written
- Produced impl_spec_draft_1.md covering all required sections:
  - Algorithm: 4-phase pseudocode (data prep, PCA extraction, forecasting, dynamic VWAP)
  - SVD-based factor extraction with IC_p2 factor count selection
  - Both AR(1) and SETAR specific component models with complete pseudocode
  - Dynamic VWAP execution with intraday updating
  - Data flow diagram with shapes/types
  - Parameters table with 8 parameters, recommended values, sensitivity, ranges
  - Validation: 6 expected behaviors, 6 sanity checks, 6 edge cases, 6 limitations
  - Paper references table: 25 entries mapping spec sections to sources
  - All researcher inferences explicitly marked

#### [2026-04-13 22:25] Session complete
- Status set to done
- Draft 1 ready for critic review

#### [2026-04-13 22:35] Revision session started — reading critique 1
- Critique contains 4 major issues, 9 minor issues
- Major issues: M1 (SETAR notation error), M2 (missing SETAR fallback), M3 (conflated metrics), M4 (missing model selection)
- All issues verified against papers — all are valid and will be addressed

**M1 assessment (major):** Confirmed. BDF Eq (11): phi_11*e + phi_12 (AR coeff first). Szucs Eq (6): c_{1,1} + theta_{1,2}*e (intercept first). Opposite ordering. Will fix the NOTE.

**M2 assessment (major):** Valid. Will add explicit fallback guard returning failure indicator.

**M3 assessment (major):** Valid. Szucs MAPE (~0.40) = turnover forecast accuracy. BDF MAPE (~0.09) = VWAP execution cost. Will add explanatory note.

**M4 assessment (major):** Valid. Will add explicit model selection procedure (Researcher inference).

**m1 assessment (minor):** Confirmed. "36 of 39" is from Section 3.2 (in-sample volume prediction), not Section 4.3.2 (out-of-sample VWAP). Table 6 shows 30/39 for dynamic vs classical VWAP. Will fix citation.

**m2-m9:** All valid, will address each.

#### [2026-04-13 22:52] Draft 2 written
Addressed all 4 major and 9 minor issues from critique 1:

**Major fixes:**
- M1: Fixed SETAR notation NOTE — now states ordering DIFFERS between papers, added explicit parameter mapping (BDF phi_11 = Szucs theta_{1,2}, etc.)
- M2: Added explicit fallback when all SETAR threshold candidates fail — fit_setar returns None, caller falls back to AR(1). Added edge case #7 for full fallback cascade.
- M3: Added explanatory note at top of Expected Behavior distinguishing turnover forecast MAPE (~0.40, Szucs) from VWAP execution cost MAPE (~0.09, BDF). Restructured into clearly labeled subsections.
- M4: Added explicit model selection procedure in daily_pipeline: always fit AR(1), attempt SETAR, use SETAR unless it fails or overfits (sigma2_setar > sigma2_ar). Marked as Researcher inference.

**Minor fixes:**
- m1: Corrected 36/39 citation to BDF Section 3.2 (in-sample volume prediction). Added 30/39 from Table 6 for out-of-sample VWAP separately.
- m2: Added MSE benchmarks from Szucs Table 2a. Noted BDF_AR beats BDF_SETAR on MSE while SETAR wins on MAPE.
- m3: Marked factor count "typically 1-3" as Researcher inference throughout.
- m4: Narrowed turnover range to "typically 1e-3 to 5e-2", cited BDF Table 1 (mean 0.0116, Q95 0.0380).
- m5: Added explicit note that L-day window is used for both PCA estimation and common component averaging, citing BDF Section 3.1.
- m6: Labeled BDF volume prediction MAPE values as "in-sample" with period dates. Added note that BDF out-of-sample evaluates VWAP cost, not volume prediction.
- m7: Added Szucs 2017 N=33 alongside BDF N=39 in Parameters table.
- m8: Added remaining_quantity tracking: initialized to total order size, decremented after each bin's trade. Added to Data Flow and Initialization sections.
- m9: Added explicit MAPE formula (Szucs Eq. (2)) in Expected Behavior and Sanity Check #7. Cross-referenced zero-bin exclusion.

#### [2026-04-13 22:52] Session complete
- Status set to done
- Draft 2 ready for critic review

#### [2026-04-13 22:51] Revision session started — reading critique 2
- Critique contains 0 major issues, 5 minor issues
- Critic states: "This spec is ready for implementation" — all prior issues resolved
- All 5 are citation/polish issues, none affect correctness or implementability

**m1 assessment (minor):** Valid. Out-of-sample VWAP tracking error portfolio numbers come from Table 2 third panel (Section 4.3.3), not Table 5 (Section 4.3.2). Will fix citation.

**m2 assessment (minor):** Valid. "26-30" conflates two comparisons. For SETAR vs AR(1), the correct number is 30/33 (Szucs Table 2c). Will fix.

**m3 assessment (minor):** Valid and insightful. Residual variance comparison is effectively "always SETAR" since SETAR has more parameters. Will acknowledge this explicitly and offer BIC as an alternative, marked as Researcher inference.

**m4 assessment (minor):** Valid. Volume prediction MAPE (Section 3.2) and VWAP tracking error (Section 4.3.1) come from different Table 2 panels. Will split citation.

**m5 assessment (minor):** Valid. AR(1) notation mapping should mirror the SETAR mapping for consistency. Will add.

#### [2026-04-13 22:51] Draft 3 written
Addressed all 5 minor issues from critique 2:

**m1 fix:** Changed out-of-sample VWAP tracking error citation from "Section 4.3.2, Table 5" to "Section 4.3.3, Table 2 (third panel, out-of-sample VWAP execution cost, portfolio of 39 CAC40 stocks)." Updated Paper References table to distinguish portfolio-level (Table 2 third panel) from per-stock (Tables 4-7) results.

**m2 fix:** Changed "26-30 of 33 stocks" to "30 of 33 stocks (Table 2c, MAPE pairwise comparison: BDF_SETAR vs BDF_AR)." Removed the conflated BCG comparison range.

**m3 fix:** Added detailed NOTE in the pseudocode (Phase 3, model selection) acknowledging that the residual variance rule is effectively "always use SETAR" due to parameter count asymmetry (5 vs 2). Offered BIC-based alternative with formulas. Updated Variants section to cross-reference this NOTE. All marked as Researcher inference.

**m4 fix:** Split the Variants section citation: volume prediction MAPE now references "Table 2 first panel, Section 3.2" and VWAP tracking error now references "Table 2 second panel, Section 4.3.1." Added corresponding rows to Paper References table.

**m5 fix:** Added AR(1) notation mapping NOTE to fit_ar1 docstring, mirroring the existing SETAR mapping. Maps BDF psi_1 = Szucs theta_1 (AR coefficient) and BDF psi_2 = Szucs c (intercept). Added row to Paper References table.
