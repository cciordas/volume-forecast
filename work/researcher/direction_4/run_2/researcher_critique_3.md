# Critique of Implementation Specification Draft 3: Dual-Mode Volume Forecast

## Summary

Draft 3 has addressed all 5 minor issues from Critique 2. The rolling
regression specification (formerly the most significant gap) is now fully
specified with OLS form, training window, feature construction, and pooling
strategy. The "Today" Exhibit 1 input, joint constraint semantics, inter-day
budget cap justification, and multi-step degradation are all now explicitly
documented with Researcher inference markings. Citation discipline remains
strong throughout.

The specification is now at implementation quality. The two remaining items
below are cosmetic-level observations -- neither would cause a developer to
make incorrect implementation choices, but addressing them would marginally
improve robustness.

**Issue count:** 0 major, 2 minor (cosmetic).

---

## Assessment of Critique 2 Resolutions

All 5 issues from Critique 2 have been resolved satisfactorily:

| Issue | Resolution | Adequate? |
|-------|-----------|-----------|
| m1. Rolling regression specification | Fully specified: OLS without intercept, N_regression_fit=63 days, pooled training across days, L lagged surprise terms, daily re-estimation. Added to parameter table. | Yes |
| m2. "Today" input in Exhibit 1 | Researcher inference note at lines 96-101 explains "Today" as data source qualifier, not a separate feature | Yes |
| m3. Joint constraint per-bin vs. per-symbol semantics | Researcher inference note at lines 140-149 explains max-across-bins conservative approach with alternatives discussed | Yes |
| m4. Inter-day ARMA cap at 8 terms justification | Researcher inference note at lines 69-74 explains the rationale for reserving 2 terms for intraday | Yes |
| m5. Multi-step intraday prediction error accumulation | Researcher inference note at lines 179-190 discusses degradation with horizon, potential mitigation strategies | Yes |

---

## Minor Issues (Cosmetic)

### m1. Renormalization Division-by-Zero When Predicting the Last Bin

**Spec section:** Pseudocode Part B, Step 5 (lines 395-402)

**Problem:** The renormalization step computes:

```
scale_factor = (remaining - adjusted_pct) / (remaining_hist - hist_pct[j + 1])
```

When `j + 1 = I` (forecasting the last bin of the day), `remaining_hist =
hist_pct[I]`, so the denominator becomes `hist_pct[I] - hist_pct[I] = 0`.
Although the subsequent loop `FOR each bin i in (j+2)..I` would be empty
(no bins to scale), in most programming languages the `scale_factor`
expression is eagerly evaluated and would raise a `ZeroDivisionError` before
the loop is reached.

**Recommended fix:** Guard the computation:

```
IF j + 1 < I:
    scale_factor = (remaining - adjusted_pct) / (remaining_hist - hist_pct[j + 1])
    FOR each bin i in (j+2)..I:
        pct_forecast[i] = hist_pct[i] * scale_factor
# When j + 1 == I, no redistribution is needed; pct_forecast[I] = adjusted_pct
```

This is a single-line guard and does not affect the algorithm's logic.

### m2. N_interday_predict Range Column Is Confusing

**Spec section:** Parameters table, N_interday_predict row (line 590)

**Problem:** The Range column says `max(p, 5)`, implying the parameter should
be at least 5. But for an ARMA with AR order p=2, only 2 lagged daily
observations are needed for prediction. The "Prior 5 days" label in Exhibit 1
(p. 18) is the display label for the data input arrow, not a minimum
requirement. The recommended value of 5 days is appropriate as a default
(it accommodates any AR order up to 5), but the Range column should clarify
that the true minimum is the selected AR order p.

**Recommended fix:** Change the Range to `p (minimum; 5 is a safe default that
covers all candidate AR orders)` or simply `1-5`.

---

## Verification of Key Citations (Draft 3 Additions)

| Spec Claim | Paper Location | Verified? |
|-----------|---------------|-----------|
| OLS without intercept for surprise regression | p. 19, "we perform both regressions without the inclusion of a constant term" | Yes (spec correctly notes this directly refers to VWAP-error regressions and extends by inference) |
| No-intercept philosophy: zero surprise = zero adjustment | p. 19, "our model does not assume that there is a positive amount of VWAP error if our volume predictions are 100% accurate" | Yes |
| "Today" as data source qualifier (Exhibit 1) | Exhibit 1, p. 18: "Current Bin" and "4 Bins Prior to Current Bin" labeled under "Today" | Yes (inference is reasonable) |
| Max inter-day budget of 8 terms | Researcher inference | Correctly marked; paper does not specify budget split |
| Conservative max-across-bins joint constraint | Researcher inference | Correctly marked; paper says "per symbol" but inter-day is per-bin |
| Multi-step degradation | p. 18, "techniques that predict only the next interval will perform better" | Yes (correctly noted as indirect acknowledgment) |
| N_regression_fit = 63 days | Researcher inference | Correctly marked; paper does not specify |

All citations verified as accurate.

---

## Overall Assessment

Draft 3 is ready for implementation. All issues from Critiques 1 and 2 (2
major, 12 minor total) have been resolved. The specification is comprehensive,
well-cited, and provides sufficient detail for a developer to implement the
model without consulting the original papers. Researcher inference is
consistently marked throughout, allowing the developer to identify which
design choices are from the paper vs. the researcher's interpolation.

The two remaining cosmetic items (division-by-zero guard in renormalization,
parameter table range clarification) are trivial to fix and do not affect
the overall implementability of the specification.

**Recommendation:** This draft is suitable for finalization. No further
revision rounds are needed.
