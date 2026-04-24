## Status: done
## Current Step: Draft 3 complete

### Log

[2026-04-13 19:05] Started proposer for direction 2, run 5.
[2026-04-13 19:05] Confirmed no prior drafts exist in work/researcher/direction_2/run_5/.
[2026-04-13 19:05] Read paper summaries: bialkowski_darolles_lefol_2008.md, szucs_2017.md.
[2026-04-13 19:06] Read full PDF of BDF 2008 (all 14 pages). Key insights:
  - Additive decomposition: x = c + e (Eq. 2-5)
  - PCA via Bai (2003) large-dimensional factor model (Eq. 6)
  - Common forecast via historical average at same time-of-day (Eq. 9)
  - Specific component: AR(1) with intercept (Eq. 10, mislabeled "ARMA(1,1)") or SETAR (Eq. 11)
  - Estimation by MLE (= OLS for AR, conditional OLS + grid search for SETAR)
  - Three VWAP strategies: theoretical (upper bound), static (dominated), dynamic (operational)
  - Dynamic execution: re-forecast after each observed bin (Section 4.2.2)
  - SETAR outperforms for 36/39 stocks (Section 3.2)
  - Portfolio tracking error reduction ~10% average, up to 50% (Section 4.3.3)
  - Table 2: portfolio MAPE for volume prediction and VWAP execution
  - Tables 4-7: per-stock VWAP execution cost
[2026-04-13 19:07] Read full PDF of Szucs 2017 (all 14 pages). Key insights:
  - Independent validation on 33 DJIA stocks, 11 years
  - BDF clearly outperforms BCG and U-method
  - BDF_AR best by MSE, BDF_SETAR best by MAPE
  - Per-stock pairwise comparisons in Tables 2b, 2c
  - MSE* scale-adjusted metric (Eq. 14)
  - BDF estimation ~2 hours vs BCG ~60 machine-days
  - Turnover = volume / TSO (not float)
  - k=26 bins of 15 minutes, L=20 days
  - Szucs Eq. 5 confirms AR(1) (not ARMA(1,1))
  - Szucs Eq. 6 confirms SETAR specification
[2026-04-13 19:08] Writing implementation specification draft 1.
[2026-04-13 19:10] Draft 1 complete. Key design decisions in this run:
  - Combined extract_and_select_factors function using a single truncated SVD for both
    factor-count selection and extraction. V(r) computed directly from singular values
    (total_ss - sum(s[:r]^2))/(N*P), avoiding P<=N vs P>N branching entirely.
  - Explicit Eckart-Young theorem citation for the V(r) formula derivation.
  - Separated forecast_specific as its own function for clarity and reuse.
  - SETAR notation mapping table between spec, BDF Eq. 11, and Szucs Eq. 6.
  - Two-phase SETAR estimation: grid search on SSR only, variance deferred to winner.
  - 11 functions total, all with complete pseudocode and shape annotations.
  - 13 sanity checks, 9 edge cases, 8 known limitations.
  - Full paper reference table mapping every spec section to source.

[2026-04-13 19:20] Received critique 1. Assessment:
  - 3 major issues, 4 medium, 5 minor (1 confirmed correct = P1).
  - M1 (MAJOR): run_dynamic_execution conflates batch and event-driven patterns. Will
    restructure to event-driven design (option a) with execute_one_bin function.
  - M2 (MAJOR): Validation metrics missing evaluation protocol (one-step-ahead intraday
    updating). Will add explicit Evaluation Protocol subsection.
  - M3 (MAJOR): Critic claims BDF_SETAR vs U MAPE pairwise should be 33/0, not 32/1.
    REJECTED after re-verification: Table 2c clearly shows BDF_SETAR vs U = 32/1.
    The 33/0 figure is BCG_3 vs U (in the same table). The critic confused the rows.
    Original draft was correct; no change made.
  - N1 (MEDIUM): First-bin forecast not specified for validation. Added to Evaluation
    Protocol subsection: bin 0 uses last in-sample specific component.
  - N2 (MEDIUM): Overnight gap treatment needs explicit statement. Added clear statement
    to fit_ar1 that series is contiguous across day boundaries, with citations.
  - N3 (MEDIUM): SETAR two-phase duplicates OLS. Refactored to store winning coefficients
    during grid search, eliminating Phase 2 entirely.
  - N4 (MEDIUM): Missing memory guidance for total_ss. Added note to use
    np.linalg.norm(X, 'fro')**2 and memory context.
  - P2 (MINOR): sigma2 is diagnostic-only. Added note to fit_ar1 and fit_setar.
  - P3 (MINOR): Turnover range upper bound 1e-1 -> 1e-2. Fixed in Type summary.
  - P4 (MINOR): C_hat "non-negative" -> "typically positive". Fixed in Type summary
    with explanation that PCA does not enforce non-negativity.
  - P5 (MINOR): Vectorize U-method. Refactored to use reshape+mean pattern.
  All valid issues addressed (except M3 which was rejected based on re-verification).

[2026-04-13 19:26] Draft 2 complete. Changes from draft 1:
  1. NEW: Evaluation Protocol subsection with step-by-step one-step-ahead procedure,
     including explicit first-bin initialization (N1 + M2).
  2. RESTRUCTURED: run_dynamic_execution -> execute_one_bin, event-driven design with
     example usage showing external event loop (M1).
  3. REFACTORED: SETAR fit_setar stores winning coefficients during grid search,
     eliminating redundant Phase 2 re-computation (N3).
  4. ADDED: Explicit contiguous series statement in fit_ar1 with citations (N2).
  5. ADDED: Memory/computation note for total_ss using Frobenius norm (N4).
  6. ADDED: sigma2 diagnostic-only notes in fit_ar1 and fit_setar (P2).
  7. FIXED: Turnover range upper bound 1e-1 -> 1e-2 in Type summary (P3).
  8. FIXED: C_hat "non-negative" -> "typically positive" in Type summary (P4).
  9. REFACTORED: U-method benchmark to use vectorized reshape+mean (P5).
  10. RETAINED: BDF_SETAR vs U MAPE = 32/1 (critic M3 was incorrect).
  11. ADDED: Note about C_hat potential negativity in compute_vwap_weights comments.
  12. EXPANDED: Paper References table with new entries for evaluation protocol,
      contiguous series, sigma2 diagnostic, total_ss Frobenius norm.

[2026-04-13 19:37] Received critique 2. Assessment:
  - 1 medium issue, 3 minor issues. All usability/clarity improvements, no correctness
    problems. Critic confirms draft 2 is ready for implementation as-is.
  - Critic also corrected their own error from critique 1 (M3 re BDF_SETAR vs U = 32/1).
  - M1 (MEDIUM): Example usage shows confusing two-call-per-bin pattern. ACCEPTED.
    Replaced with single-call pattern: inline state update after observing actual volume.
  - P1 (MINOR): Evaluation Protocol uses 1-based indexing, daily_update uses 0-based.
    ACCEPTED. Added indexing convention mapping note.
  - P2 (MINOR): daily_update missing day_index precondition. ACCEPTED. Added assert and
    docstring precondition.
  - P3 (MINOR): Last-bin execution guarantee not documented. ACCEPTED. Added explicit
    note after example usage.

[2026-04-13 19:37] Draft 3 complete. Changes from draft 2:
  1. SIMPLIFIED: execute_one_bin example to single-call pattern with inline state update (M1).
  2. ADDED: Last-bin guarantee note explaining weights=[1.0] property (P3).
  3. ADDED: Indexing convention mapping in Evaluation Protocol (P1).
  4. ADDED: day_index precondition assertion and docstring note in daily_update (P2).
