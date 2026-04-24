# Critique of Implementation Specification Draft 2: PCA Factor Decomposition (BDF)

## Summary

Draft 2 is a substantial improvement over draft 1. All 4 major issues and all 9
minor issues from critique 1 have been addressed effectively:

- **M1 (SETAR notation):** Fixed with a detailed NOTE and explicit parameter name
  mapping between BDF and Szucs. Verified correct against BDF Eq. (11) and
  Szucs Eq. (6).
- **M2 (SETAR fallback):** Fixed with Step 3.5 returning None and the daily
  pipeline handling the fallback to AR(1).
- **M3 (Metric conflation):** Fixed with a clear "Important note on metrics"
  section distinguishing turnover forecast MAPE from VWAP execution cost MAPE.
- **M4 (Model selection):** Fixed with a residual-variance-based selection rule
  in the daily pipeline, clearly marked as Researcher inference.
- **m1-m9:** All addressed — correct section references, MSE benchmarks added,
  factor count marked as Researcher inference, turnover range narrowed, L/h
  clarified, in-sample labels added, Szucs N=33 added, remaining_quantity
  tracked, MAPE formula included.

No major issues remain. I identified 5 minor issues, mostly citation
imprecisions and one theoretical weakness in the model selection rule. The spec
is now implementable as-is; these issues would improve precision but are not
blocking.

---

## Minor Issues

### m1. Out-of-sample VWAP tracking error cited from wrong section and table

**Location:** Validation, Expected Behavior (line 668)

**Problem:** The spec cites "BDF 2008, Section 4.3.2, Table 5" for the
out-of-sample VWAP tracking error portfolio numbers (Dynamic PCA-SETAR 0.0898,
Dynamic PCA-ARMA 0.0922, Classical 0.1006). These aggregate portfolio-level
numbers come from the **third panel of Table 2**, described in **Section 4.3.3**
("Portfolio in and out-sample results"). Table 5 contains per-stock
out-of-sample results for the dynamic PCA-SETAR model, not the portfolio
aggregate. Section 4.3.2 discusses stock-level results, not portfolio results.

**Evidence:** BDF 2008, p.1716, Table 2 — the third panel is titled "By
out-of-sample estimation for VWAP order execution" and contains the aggregate
Mean/Std/Q95 rows. Tables 4 and 5 (pp.1719) contain per-stock results.

**Fix:** Change citation to "BDF 2008, Section 4.3.3, Table 2 (third panel,
out-of-sample VWAP execution cost, portfolio of 39 CAC40 stocks)."

### m2. "26-30 of 33 stocks" conflates multiple Szucs comparisons

**Location:** Daily pipeline comment (line 443)

**Problem:** The spec says "SETAR outperforms AR(1) in ... 26-30 of 33 stocks in
Szucs 2017." The range "26-30" conflates SETAR's pairwise win count against
different opponents:
- BDF_SETAR vs BDF_AR: **30/3** (SETAR wins on 30 of 33 stocks by MAPE).
- BDF_SETAR vs BCG_3: **26/7** (SETAR wins on 26 of 33 stocks by MAPE).

Since the context is specifically about SETAR vs AR(1), the relevant number is
30/33, not the range 26-30. The "26" figure comes from the SETAR vs BCG_3
comparison, which is irrelevant here.

**Evidence:** Szucs 2017, Section 5, Table 2c (MAPE pairwise comparison).

**Fix:** Change to "30 of 33 stocks in Szucs 2017 (Table 2c, MAPE pairwise
comparison: BDF_SETAR vs BDF_AR)."

### m3. Model selection by residual variance is effectively a null test

**Location:** Daily pipeline Step 3 (lines 462-467) and Variants section
(lines 531-542)

**Problem:** The model selection rule is: "Use SETAR unless sigma2_setar >
sigma2_ar." However, SETAR has 5 free parameters (phi_11, phi_12, phi_21,
phi_22, tau) versus AR(1) with 2 (psi_1, psi_2). On the same training data,
fitting two separate OLS regressions on regime subsets will almost always
produce lower total sum of squared residuals than a single OLS on the full set,
because SETAR can fit different slopes and intercepts to each regime subset.
The only case where sigma2_setar > sigma2_ar would occur is if the threshold
is so poor that the regime-specific fits are numerically worse than the pooled
fit — an edge case that the grid search already guards against.

In practice, this rule is equivalent to "always use SETAR unless it
catastrophically fails," which is not truly model selection. A more principled
approach would use a penalized criterion (e.g., BIC) that accounts for the
extra parameters:

    BIC_ar  = T * ln(sigma2_ar)  + 2 * ln(T)
    BIC_setar = T * ln(sigma2_setar) + 5 * ln(T)

This is not a blocking issue because the papers suggest SETAR generally does
outperform AR(1), so "always SETAR" is a defensible default. But the spec
should either:
(a) Acknowledge that the residual-variance rule is effectively "always SETAR,"
    or
(b) Replace it with BIC comparison if genuine model selection is desired.

Mark this note as Researcher inference since neither paper provides a selection
procedure.

### m4. Variants section conflates two Table 2 panels under one reference

**Location:** Variants section (lines 545-548)

**Problem:** The spec states: "BDF 2008 Table 2: SETAR achieves lower in-sample
volume prediction MAPE (0.0752 vs 0.0829 for ARMA) and lower VWAP tracking
error (0.0706 vs 0.0772). Reference: BDF 2008, Section 3.2, Table 2."

These two numbers come from different panels of Table 2 and different sections:
- Volume prediction MAPE (0.0752): Table 2 first panel, Section 3.2.
- VWAP tracking error (0.0706): Table 2 second panel, Section 4.3.1.

Citing both under "Section 3.2" is imprecise. A developer cross-referencing
the paper would find volume prediction in Section 3.2 but would not find VWAP
tracking error there.

**Fix:** Split the citation: "SETAR achieves lower in-sample volume prediction
MAPE (0.0752 vs 0.0829; Table 2 first panel, Section 3.2) and lower in-sample
VWAP tracking error (0.0706 vs 0.0772; Table 2 second panel, Section 4.3.1)."

### m5. Missing AR(1) notation mapping between BDF and Szucs

**Location:** Pseudocode Phase 3, fit_ar1 (lines 177-205)

**Problem:** The spec provides a detailed notation mapping between BDF and Szucs
for SETAR (fit_setar docstring, lines 220-235) but does not provide the
equivalent mapping for AR(1). Szucs 2017 Eq. (5) writes the AR(1) as:

    e_p = c + theta_1 * e_{p-1} + epsilon_p

while the spec (following BDF Eq. (10)) uses:

    e_t = psi_1 * e_{t-1} + psi_2 + epsilon_t

The mapping is: Szucs' c = spec's psi_2 (intercept), Szucs' theta_1 = spec's
psi_1 (AR coefficient). Note that Szucs writes the intercept FIRST (same
ordering as in the SETAR case), while the spec writes the AR coefficient FIRST
(same ordering as BDF).

For consistency with the SETAR documentation, the AR(1) docstring should
include a similar notation mapping note.

**Fix:** Add to fit_ar1 docstring: "NOTE on notation: BDF 2008 Eq. (10) writes
psi_1 * e_{t-1} + psi_2 (AR coefficient first). Szucs 2017 Eq. (5) writes
c + theta_1 * e_{p-1} (intercept first). Mapping: psi_1 = Szucs' theta_1,
psi_2 = Szucs' c."

---

## Assessment Summary

### Resolution of Prior Issues

All 13 issues from critique 1 have been resolved satisfactorily. The most
impactful fixes were:
- The SETAR notation mapping (M1) eliminates a source of developer confusion.
- The SETAR fallback and model selection logic (M2, M4) prevent runtime crashes
  and provide a clear decision rule.
- The metric distinction (M3) with the "Important note on metrics" section is
  well-written and prevents apples-to-oranges validation comparisons.
- The remaining_quantity tracking (m8) makes the VWAP execution section
  self-contained.

### Algorithmic Clarity

The pseudocode remains well-structured and directly implementable. The model
selection logic integrates cleanly into the daily pipeline. The SETAR fallback
cascade (SETAR -> AR(1) -> common-only) provides robust error handling.

### Completeness

The spec is comprehensive. The Paper References table at the end provides
excellent traceability, with Researcher inference items clearly marked. MSE
benchmarks now complement MAPE benchmarks for validation.

### Correctness

All algorithmic steps correctly reflect the source papers. The remaining
citation imprecisions (m1, m4) are minor and do not affect the mathematical
content. The 5 new issues are all minor — none would lead to incorrect
implementation.

### Implementability

A developer can implement the model from this spec without consulting the
original papers. The main area where judgment is still required is the model
selection rule (m3), but the spec's default ("effectively always SETAR") is
defensible and functional.

### Overall Assessment

**This spec is ready for implementation.** The 5 minor issues identified are
polish items that would improve citation precision and theoretical rigor but
do not affect the correctness or implementability of the model. A third
proposer-critic round would yield diminishing returns.
