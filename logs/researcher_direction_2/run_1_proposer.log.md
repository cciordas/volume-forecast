## Status: done
## Current Step: Draft 3 complete

### Log

[2026-04-12 21:20] Started proposer for direction 2 (PCA Factor Decomposition / BDF), run 1.
[2026-04-12 21:20] Reading paper summaries for orientation.
[2026-04-12 21:21] Read paper summaries: bialkowski_darolles_lefol_2008.md and szucs_2017.md.
  - BDF 2008 (foundational): Introduces PCA decomposition of intraday turnover into common
    (market-wide U-shape via factor analysis) and specific (stock-level, ARMA or SETAR) components.
    Validated on 39 CAC40 stocks with 20-min bins. SETAR outperforms ARMA for 36/39 stocks.
    Dynamic VWAP execution reduces tracking error by ~10% vs classical.
  - Szucs 2017 (comparison): Head-to-head of BDF vs CMEM (BCG) on 33 DJIA stocks over 11 years.
    BDF wins on both MSE (BDF-AR best) and MAPE (BDF-SETAR best). BDF is orders of magnitude
    faster (~2 hours vs ~60 machine-days for CMEM).

[2026-04-12 21:22] Read full PDFs for implementation details.
  - BDF 2008: Confirmed factor model equations (2)-(6), PCA via eigendecomposition with
    F'F/T = I normalization. Common forecast is simple time-of-day average (Eq. 9).
    SETAR model (Eq. 11) uses threshold indicator I(e <= tau) with two AR(1) regimes.
    Key implementation detail: factor loadings refreshed daily on rolling 1-month window.
    Dynamic VWAP updates forecasts after each 20-min bin.
  - Szucs 2017: Confirmed 15-min bins (k=26), turnover = V/TSO, 20-day rolling window,
    one-step-ahead forecasting. Error metrics: MSE, MSE*, MAPE. BDF-AR best on MSE (31/33 stocks),
    BDF-SETAR best on MAPE (26-30/33 stocks).

[2026-04-12 21:25] Key ambiguities identified and resolved:
  - ARMA(1,1) vs AR(1): BDF 2008 Eq. 10 shows e_{i,t} = psi_1 * e_{i,t-1} + psi_2 + epsilon,
    which is AR(1) with intercept, not ARMA(1,1) with MA component. The paper labels it "ARMA"
    but the equation is AR(1). Implemented as written (AR(1) with intercept).
  - Factor count selection: BDF 2008 cites Bai & Ng (2002) but does not specify which IC variant.
    Recommended IC_p2 as most commonly used in practice.
  - SETAR threshold estimation: Not detailed in BDF 2008. Used standard grid search over
    candidate thresholds (Hansen 1997 / Tong 1990 methodology).
  - MAPE scale discrepancy: BDF 2008 reports ~0.07-0.09; Szucs 2017 reports ~0.40-0.50.
    Likely different definitions or aggregation levels. Both show same relative ordering.

[2026-04-12 21:30] Wrote impl_spec_draft_1.md covering:
  - Full algorithm with overnight estimation + intraday dynamic forecasting pseudocode
  - Data flow from raw ticks through PCA to one-step-ahead forecasts
  - Parameter table with 11 parameters including sensitivity and ranges
  - Initialization and calibration procedures
  - Bai & Ng factor count selection procedure
  - Validation section with expected MAPE/MSE benchmarks from both papers
  - 7 sanity checks, 6 edge cases, 7 known limitations
  - Complete paper reference traceability table

[2026-04-12 21:29] Received critique of draft 1. Assessment:
  - 6 major issues, 8 minor issues raised.
  - Severity breakdown:
    * M1 (eigendecomposition matrix): MAJOR -- paper says X'X (NxN), spec used XX' (TxT).
      Verified in BDF 2008 p.1712: "eigenvectors corresponding to the r-largest eigenvalues
      of the X'X matrix". Will fix to X'X formulation.
    * M2 (AR vs ARMA labeling): MAJOR -- agree, will definitively label as AR(1) throughout
      with footnote explaining BDF 2008's misnomer.
    * M3 (OLS vs MLE estimation): MAJOR -- paper explicitly says MLE. Will note discrepancy
      and justify OLS equivalence for AR(1), add MLE note for SETAR.
    * M4 (dynamic VWAP multi-step forecast gap): MAJOR -- pseudocode only computes one-step
      ahead but VWAP execution needs all remaining bins. Will add explicit multi-step logic.
    * M5 (">50% for high-volatility stocks"): MAJOR -- BDF 2008 conclusion (p.1722) says
      "can even reach 50% for some stocks" but individual stock data in Tables 6-7 shows max
      ~36% (CAP GEMINI). Will cite conclusion text exactly, note data shows up to ~36%.
    * M6 (31 vs 39 stocks): MAJOR -- internal inconsistency in BDF 2008 (Table 1 has 39
      stocks but Section 3.2 text says "31"). Will flag discrepancy.
  - Minor issues (m1-m8): all valid, will address each.
  - Overall: critique is thorough and accurate. All corrections improve the spec.

[2026-04-12 21:30] Re-read BDF 2008 PDF to verify corrections:
  - Confirmed X'X formulation on p.1712.
  - Confirmed Eq. 10 is AR(1) with intercept (no MA term).
  - Confirmed "by maximum likelihood" for estimation (p.1712, Section 2.3).
  - Confirmed Table 1 has 39 stocks; Section 3.2 text says "31".
  - Confirmed conclusion mentions "50% for some stocks" but Table 6 max is ~36%.
  - Confirmed Table 1 mean turnover 0.0166 with stocks up to 0.04+.

[2026-04-12 21:31] Wrote impl_spec_draft_2.md addressing all 6 major and 8 minor issues.

[2026-04-12 21:46] Received critique of draft 2. Assessment:
  - 0 major issues, 5 minor issues raised.
  - Critic states draft 2 is "ready for implementation" and a third draft is
    "not strictly necessary." All 5 issues are polish/clarification items.
  - Severity: all minor. No correctness or implementability concerns.
  - m1: MAPE scale discrepancy explanation too vague -- will add specific
    aggregation-level explanation (portfolio vs per-stock).
  - m2: Per-stock model selection not specified -- will clarify uniform SETAR.
  - m3: SETAR grid technically over lagged values -- pedantic fix.
  - m4: Negative forecast handling lacks volume share fallback -- will add.
  - m5: PCA sign indeterminacy not mentioned -- will add note.

[2026-04-12 21:46] Wrote impl_spec_draft_3.md addressing all 5 minor issues.
  Changes summary:
  - M1: Rewrote PCA to use X'X/T (NxN) with full normalization procedure and SVD alternative.
  - M2: Replaced all "ARMA(1,1)" with "AR(1)" throughout; added explanatory note in Overview.
  - M3: Added MLE citations and OLS-equivalence justification for both AR(1) and SETAR.
  - M4: Added complete multi-step forecast logic (Steps 4-5 in Phase 2) with iterated
    AR/SETAR recursion and deterministic iteration for SETAR.
  - M5: Replaced unverified ">50%" claim with precise Table 6-7 data; cited conclusion text.
  - M6: Flagged 31 vs 39 discrepancy explicitly in Variants section.
  - m1-m8: All addressed (turnover range, demeaning, day-start init, V(r) notation, r_max,
    SETAR grid range, cross-day boundary, worked example).
  - Added Revision History section documenting all changes.
