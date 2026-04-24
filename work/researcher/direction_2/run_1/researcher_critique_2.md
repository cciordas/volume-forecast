# Critique of Implementation Specification Draft 2: PCA Factor Decomposition (BDF)

## Summary

Draft 2 is a substantial improvement over draft 1. All 6 major and 8 minor issues
from critique 1 have been addressed thoroughly and correctly. The eigendecomposition
now operates on the correct matrix (X'X/T) with a properly derived normalization
procedure. The AR(1) labeling is consistent throughout. MLE is correctly cited with
the OLS equivalence noted. The multi-step forecast gap is filled with explicit
pseudocode. VWAP tracking claims are now properly sourced with per-stock data from
Tables 6-7. The 31-vs-39 discrepancy is flagged. All minor issues (demeaning,
day-start initialization, notation, parameter table, grid search, cross-day boundary,
toy example) are incorporated.

The spec is now close to implementation-ready. I found 0 major issues and 5 minor
issues. None of them would prevent a competent developer from producing a correct
implementation; they are clarifications that would improve precision and prevent
potential misinterpretation.

---

## Minor Issues

### m1. MAPE scale discrepancy explanation is too vague

**Location:** Expected Behavior, point 3 (around line 603-607).

**Problem:** The spec notes that BDF 2008 reports MAPE values of 0.07-0.09 for
intraday volume prediction while Szucs 2017 reports 0.40-0.50 for the same models,
and attributes this to "different MAPE definitions or data characteristics." This is
a 5x difference that deserves a more precise explanation. Leaving it vague may
cause a developer to doubt their implementation is correct when they see results in
one range vs the other.

The most likely explanation is the level of aggregation:
- BDF 2008 appears to compute MAPE at the portfolio level (averaging turnover
  predictions across all N stocks first, then computing percentage errors on the
  smoother portfolio-level series). Portfolio averaging diversifies away stock-level
  noise, dramatically reducing MAPE.
- Szucs 2017 computes per-stock per-bin MAPE and then averages across stocks (Eq.
  2, p.6). Individual stock turnover in 15-minute bins is much noisier than
  portfolio-level turnover, yielding higher MAPE.

Additionally, the datasets differ: CAC40 European stocks (2003-2004) vs DJIA US
stocks (2001-2012), different bin widths (20-min vs 15-min), and different sample
lengths (1 year vs 11 years).

**Recommendation:** Replace the vague note with a specific explanation distinguishing
portfolio-level MAPE (BDF 2008) from per-stock MAPE (Szucs 2017). Add guidance for
the developer: "When validating, compute per-stock per-bin MAPE following Szucs 2017
Eq. (2); expect values around 0.40 for liquid US equities. The BDF 2008 values
around 0.07-0.09 use a different aggregation level and should not be used as targets
for per-stock validation."

### m2. Model selection between AR(1) and SETAR is not specified per stock

**Location:** Variants section (line 392-414); Phase 1, Step 6 (lines 147-197).

**Problem:** The spec recommends SETAR as the primary variant and AR(1) as a fallback,
but does not specify whether to use the same model type for all stocks or to select
per stock. BDF 2008 Section 3.2 shows that SETAR outperforms AR(1) for the majority
of stocks but not all (3 of 31 stocks favor AR). The Phase 2 function takes a global
`model_type` parameter, implying one model for the entire cross-section.

A developer might reasonably ask: should I run SETAR for all stocks uniformly (as
the spec implies), or should I fit both models per stock and select the better one
(e.g., by AIC or out-of-sample MAPE)?

**Recommendation:** Clarify that the primary implementation should use SETAR
uniformly for all stocks. BDF 2008's results show it wins for the large majority,
and the marginal cost of fitting SETAR vs AR(1) is small (only the threshold grid
search is extra). If per-stock model selection is desired, suggest comparing in-sample
AIC between AR(1) and SETAR for each stock, but note this adds complexity with
minimal expected benefit.

### m3. SETAR threshold grid is defined over e_series values but should technically use lagged values

**Location:** Phase 1, Step 6 Option B (lines 180-191); Initialization point 4
(lines 468-476).

**Problem:** The SETAR grid search description says to use "the 15th to 85th
percentile of e_series values" as candidate thresholds. But the threshold tau
operates on the lagged value e_{i,t-1} (the regressor), not e_{i,t} (the dependent
variable). Technically, the grid should be defined over the set of lagged values
{e_{i,1}, e_{i,2}, ..., e_{i,T-1}}, not over all values {e_{i,1}, ..., e_{i,T}}.

In practice this distinction is negligible (the two sets differ by one observation
at each end and have nearly identical percentiles for T >= 500), so this is not a
correctness issue. But stating it precisely prevents a developer from wondering
whether there is a subtle bug.

**Recommendation:** Change "percentile of e_series values" to "percentile of the
lagged values {e_{i,1}, ..., e_{i,T-1}}" or simply note that the grid is computed
from the regressor values (which differ from the full series by one observation at
each end).

### m4. Negative forecast handling does not discuss impact on volume shares

**Location:** Phase 2, Step 5 (lines 294-301); Edge Cases, point 2 (lines 712-718).

**Problem:** The spec correctly floors negative turnover forecasts to 1e-8 and notes
this as an edge case. However, it does not discuss the downstream effect on volume
share computation (Step 6). If multiple future bins are floored to 1e-8, the
remaining bins with positive forecasts will receive disproportionately large volume
shares. In the extreme case where all remaining bins except the next one are floored,
the model would recommend concentrating all remaining volume in the next bin, which
is clearly wrong for VWAP execution.

**Recommendation:** Add a note that if more than a specified fraction of remaining
bins (e.g., more than 50%) produce negative forecasts before flooring, the model
should fall back to the common component forecast alone (setting e_forecast = 0 for
all remaining bins) rather than using the floored values. This prevents pathological
volume concentration. Mark as Researcher inference.

### m5. The spec does not mention sign indeterminacy of PCA factors

**Location:** Phase 1, Step 3 (lines 94-131); Initialization point 2 (lines 448-455).

**Problem:** The spec notes that eigenvectors are "unique up to sign/rotation" in
Initialization point 2, and correctly states this "does not affect the common
component C = F A." However, it does not mention that across consecutive daily
re-estimations, the sign of the factors and loadings may flip. This does not affect
the common component (since C = F @ A, a simultaneous sign flip in F and A cancels
out), but it can confuse a developer who visualizes or logs the factor values and
sees them flip sign from day to day.

**Recommendation:** Add a brief note in the Calibration section (daily rolling update)
that factor sign may flip across estimation windows and this is expected behavior.
If consistent factor sign is desired for visualization, apply a sign convention
(e.g., require the first loading to be positive) after each estimation.

---

## Citation Verification Summary

| Spec Claim | Cited Source | Verified? | Notes |
|---|---|---|---|
| PCA eigendecomposition on X'X/T with normalization | BDF 2008, Eq. 6, text below | Correct | Now matches paper's stated formulation. |
| Normalization: F_hat = X @ V_r @ diag(1/sqrt(lambda)) | Researcher derivation | Correct | Derivation is mathematically sound: V_r' (X'X/T) V_r = diag(lambda), so F'F/T = diag(lambda), and scaling by 1/sqrt(lambda) gives I_r. |
| AR(1) labeling (not ARMA) | BDF 2008, Eq. 10; Szucs 2017, Eq. 5 | Correct | Properly resolved: BDF Eq. 10 has no MA term; Szucs calls it AR(1). |
| MLE estimation, OLS equivalence | BDF 2008, Section 2.3 | Correct | Paper says MLE; spec correctly notes OLS equivalence under Gaussian errors. |
| SETAR estimation via profile MLE/grid search | Hansen 1997, Tong 1990 | Correct | Standard methodology, properly cited. |
| SETAR grid: 15th-85th percentile | Researcher inference | Appropriate | Grid narrowed from 10th-90th per critique 1; min_regime_frac added. |
| Common component forecast Eq. (9) | BDF 2008, Eq. 9 | Correct | Time-of-day bin averaging across L_days. |
| Multi-step forecast via iterated recursion | Researcher inference | Appropriate | Standard for AR models; deterministic SETAR iteration clearly marked as approximation. |
| Day-start e_forecast = 0 | Researcher inference | Appropriate | Consistent with static forecast baseline (BDF Section 4.2.1) and unconditional mean of specific component. |
| k=25 for CAC40, k=26 for DJIA | BDF 2008, Section 3.1; Szucs 2017, Section 2 | Correct | 25 20-min intervals; 26 15-min intervals. |
| L=20 trading days | BDF 2008, Section 3.1; Szucs 2017, Section 3 | Correct | Both papers use 20-day windows. |
| BDF-SETAR MAPE 0.399, BDF-AR 0.403 | Szucs 2017, Table 2a | Correct | Values match exactly. |
| BDF-AR MSE 6.49e-4, SETAR 6.60e-4 | Szucs 2017, Table 2a | Correct | Values match exactly. |
| U-method MSE 1.02e-3, MAPE 0.503 | Szucs 2017, Table 2a | Correct | Values match exactly. |
| VWAP tracking: 10.06% classical, 8.98% dynamic SETAR | BDF 2008, Table 2 (out-of-sample) | Correct | Values 0.1006 and 0.0898 correctly converted to percentages. |
| CAP GEMINI: 23.23% to 14.91%, ~36% relative reduction | BDF 2008, Tables 6-7 | Correct | Classical = 0.2323, Dynamic SETAR = 0.1491; (0.2323-0.1491)/0.2323 = 35.8%. |
| BDF 2008 conclusion "can even reach 50%" | BDF 2008, Section 5, p.1722 | Correct | Direct quote from conclusion. |
| 9 stocks show deterioration, 7 by less than 1 bp | BDF 2008, Tables 6-7 | Correct | Verified from Tables 6-7. |
| 2 hours for 33 stocks, 2648 days | Szucs 2017, Section 4 | Correct | ~2 hrs BDF vs ~60 machine-days BCG_0. 2668 total days - 20 window = 2648 forecast days. |
| Demeaning not required (Bai 2003) | Researcher inference | Appropriate | Standard in large-dimensional factor models. |
| Cross-day boundary contamination | Researcher inference | Appropriate | Correctly flagged as limitation. |
| 31 vs 39 stock count discrepancy | BDF 2008, Table 1 vs Section 3.2 | Correct | Flagged without resolution, which is appropriate. |
| BDF 2008 MAPE: SETAR 0.0752, AR 0.0829, Classical 0.0905 | BDF 2008, Table 2, panel 1 | Correct | Per-interval MAPE values from volume prediction section. |

---

## Overall Assessment

Draft 2 is a high-quality implementation specification that would enable a developer
to implement the BDF model correctly. All major issues from critique 1 have been
resolved. The pseudocode is complete and directly translatable to code. The parameter
table is comprehensive. Validation benchmarks are properly sourced with specific
paper references. Edge cases and limitations are thoroughly documented.

The 5 minor issues identified above are polish items, not blockers:
- **m1 (MAPE scale explanation)** would help a developer calibrate expectations but
  does not affect implementation correctness.
- **m2 (per-stock model selection)** is a design decision that a developer can handle.
- **m3 (SETAR grid on lagged values)** is pedantic for T >= 500.
- **m4 (negative forecast fallback)** is a robustness enhancement for edge cases.
- **m5 (sign indeterminacy)** is a debugging/visualization convenience.

**Recommendation:** This spec is ready for implementation. The proposer may optionally
address m1 and m4 for robustness, but a third draft is not strictly necessary.
