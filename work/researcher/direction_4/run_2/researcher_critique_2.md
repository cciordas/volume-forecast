# Critique of Implementation Specification Draft 2: Dual-Mode Volume Forecast

## Summary

Draft 2 has substantially improved. All 9 issues from Critique 1 (2 major, 7
minor) have been addressed. The intraday ARMA now correctly operates on
within-day sequences with day-boundary-aware fitting. The 11-term constraint
is properly applied as a joint constraint across both ARMA models. Researcher
inference is marked throughout. The volume percentage error metric is now
explicitly defined. Citation discipline remains strong.

The remaining issues are minor and relate primarily to implementability gaps
in the volume percentage model's rolling regression, a few ambiguities in the
joint constraint implementation, and missing discussion of a specific input
from Exhibit 1.

**Issue count:** 0 major, 5 minor.

---

## Assessment of Critique 1 Resolutions

All issues from Critique 1 have been resolved satisfactorily:

| Issue | Resolution | Adequate? |
|-------|-----------|-----------|
| M1. Intraday ARMA series construction | Rewritten with within-day sequences, day-boundary breaks, AR context from recent bins | Yes |
| M2. Joint 11-term constraint | Now applied across both ARMA models; inter-day capped at 8 to reserve room for intraday | Yes |
| m1. Weight non-negativity inference | Marked as Researcher inference with rationale | Yes |
| m2. "4 Bins Prior" detail | Incorporated into pseudocode and data flow | Yes |
| m3. Renormalization inference | Marked with explanation | Yes |
| m4. MAPE optimization | Nelder-Mead specified, volume floor added | Yes |
| m5. Volume pct error metric | Part D added with MAD formula and citation | Yes |
| m6. Inter-day training window | N_interday_fit vs N_interday_predict distinguished | Yes |
| m7. Weight sum-to-1 | Trade-off discussed, unconstrained default with rationale | Yes |

---

## Minor Issues

### m1. Rolling Regression Specification Remains Insufficiently Detailed for Implementation

**Spec section:** Pseudocode Part B, Step 3 (lines 275-289)

**Problem:** The rolling regression for volume surprises is the least specified
part of the entire document. The pseudocode says:

```
predicted_adjustment = rolling_regression_predict(
    surprise_history=surprise_pct[1..j],
    num_terms=L_optimal
)
```

A developer encountering this would need to make multiple design decisions
that are not addressed:

1. **Regression type:** Is this ordinary least squares (OLS)? The paper (p. 19)
   says "a rolling regression model that adjusts market participation." OLS is
   the most natural interpretation, but it should be stated.

2. **Training window:** How many days of historical surprise data are used to
   estimate the regression coefficients? The pseudocode shows only today's
   surprises as the predictor input, but the coefficients must be estimated
   somewhere. Is it the same 1-month rolling window used for the intraday ARMA,
   or a different window?

3. **Feature specification:** The pseudocode implies the features are
   `surprise_pct[k-1], ..., surprise_pct[k-L]` (lagged surprise percentages
   from earlier bins today). But are there additional features? For example,
   the magnitude of cumulative surprise, the bin index, or an intercept term?
   The paper (p. 19) says "Note that we perform both regressions without the
   inclusion of a constant term (indicating a non-zero y intercept)" -- this
   no-intercept detail appears in the VWAP-error regression discussion, but
   it is unclear whether it also applies to the surprise regression.

4. **Update frequency:** Is the regression re-estimated each day (using
   yesterday's complete surprise sequence), or updated within the day as new
   bins are observed?

**Recommended fix:** Add a concrete regression specification, for example:

```
# [Researcher inference: the paper does not specify the exact regression form.
#  We use OLS without intercept, consistent with p. 19's note about regressions
#  without constant terms. The regression is re-estimated daily using the prior
#  N_regression_fit days of complete intraday surprise sequences.]

# Training: for each historical day d, compute surprise_pct[k] for k=1..I.
# Regress surprise_pct[k] on (surprise_pct[k-1], ..., surprise_pct[k-L])
# for k = L+1..I, pooling across all days in the training window.
# This gives L regression coefficients (no intercept).

# Prediction: apply coefficients to today's most recent L surprise_pct values.
beta = fit_ols_no_intercept(
    X = [surprise_pct[k-1:k-L:-1] for k in L+1..I, for d in training_days],
    y = [surprise_pct[k] for k in L+1..I, for d in training_days]
)
predicted_adjustment = dot(beta, surprise_pct[j:j-L:-1])
```

Also add `N_regression_fit` (training window for surprise regression) to the
parameter table with a Researcher inference note.

### m2. "Today" Input in Exhibit 1 Not Discussed

**Spec section:** Pseudocode Part A, Component 3

**Problem:** Exhibit 1 (p. 18) shows three distinct inputs to the ARMA Intraday
component: "Current Bin," "4 Bins Prior to Current Bin," and "Today." The
draft correctly incorporates "Current Bin" and "4 Bins Prior to Current Bin"
as the AR lag context. However, "Today" appears as a separate labeled input
arrow but is never discussed in the spec.

The most likely interpretation is that "Today" refers to today's intraday
data as the source for the other two inputs (i.e., "Current Bin" and "4 Bins
Prior" are from today's data, as opposed to historical data). Under this
reading, "Today" is not a separate feature but a qualifier on the data source.

**Recommended fix:** Add a brief note in the pseudocode or a comment
acknowledging this input label and stating the interpretation explicitly:

```
# Exhibit 1 shows "Today" as a separate input label alongside "Current Bin"
# and "4 Bins Prior to Current Bin." We interpret "Today" as the data source
# qualifier: the AR lag inputs come from today's observed deseasonalized bins,
# not from historical days. [Researcher inference]
```

This prevents a developer from wondering whether "Today" is a separate
feature (e.g., today's total volume so far, or today's date characteristics).

### m3. Joint Constraint Per-Bin vs. Per-Symbol Semantics Could Be Clearer

**Spec section:** Pseudocode Part A, lines 129-137

**Problem:** The inter-day ARMA is per-bin (26 separate models per stock), while
the intraday ARMA is a single model per stock. The paper says "we fit each
symbol with a dual ARMA model having fewer than 11 terms" (p. 18). This is a
per-symbol constraint, but the inter-day ARMA has different term counts across
bins.

The draft handles this by taking `max(interday_terms_i for i in 1..I)` as
the binding constraint on the intraday budget. This is the most conservative
interpretation: it ensures the joint constraint is satisfied for the
worst-case bin. However, this means the constraint is effectively determined
by whichever bin has the most complex inter-day model, potentially
over-restricting the intraday model unnecessarily.

**Recommended fix:** Briefly acknowledge this design choice and the alternative.
A comment in the pseudocode would suffice:

```
# [Researcher inference: The paper's "fewer than 11 terms" is per-symbol,
#  but the inter-day ARMA is per-bin. We use the maximum inter-day term
#  count across all bins as the binding constraint, which is conservative.
#  An alternative is to use the median or to apply the constraint per-bin
#  (allowing a more complex intraday model when most bins have simple
#  inter-day models). The conservative approach is safer for a first
#  implementation.]
```

### m4. Inter-day ARMA Cap at 8 Terms Needs Brief Justification

**Spec section:** Pseudocode Part A, line 71

**Problem:** The pseudocode says:

```
IF p + q + 1 > 8: CONTINUE
```

This caps the inter-day ARMA at 8 terms, reserving at least 2 terms for the
intraday model (p + q + 1 = 2 means p=1,q=0 or p=0,q=1 with constant). The
code comment on line 69-70 says "Reserve at least 2 terms for intraday
(p=1,q=0 + constant) so inter-day can use at most 8 terms."

This is a reasonable design choice, but it is not motivated by the paper. The
minimum meaningful intraday ARMA could be argued to be 1 term (constant only,
which is just the unconditional mean) or 2 terms (one AR lag + constant).
Different choices here change the inter-day cap (9 vs 8).

**Recommended fix:** Mark this as Researcher inference and briefly explain:

```
# [Researcher inference: cap of 8 terms reserves at least 2 for the intraday
#  ARMA (e.g., AR(1) + constant). A cap of 9 would allow a constant-only
#  intraday model, which provides no dynamic information. We prefer to
#  guarantee at least one AR lag for intraday adaptation.]
```

### m5. Multi-Step Intraday Prediction Error Accumulation Not Discussed

**Spec section:** Pseudocode Part A, lines 160-168

**Problem:** The intraday ARMA uses recursive multi-step prediction: predicted
values are appended to `today_deseasonal` and used as inputs for subsequent
predictions. This is standard practice, but prediction variance grows with
each recursive step. For early bins (e.g., bin 2 predicting bins 3-26), the
ARMA is making 24-step-ahead recursive forecasts, which will be highly
uncertain.

The paper acknowledges this issue implicitly: "techniques that predict only
the next interval will perform better than those attempting to predict volume
percentages for the remainder of the trading day" (p. 18). But this statement
is about the percentage model's advantage, not about how to handle the raw
model's multi-step degradation.

**Recommended fix:** Add a note in the pseudocode or edge cases section:

```
# [Researcher inference: recursive multi-step ARMA predictions degrade with
#  forecast horizon. For bins far from the last observation, the intraday
#  ARMA forecast converges to the unconditional mean and adds little value
#  beyond the historical average and inter-day components. The regime
#  weights may naturally down-weight the intraday component for distant bins
#  if trained on data that reflects this degradation. A developer may also
#  consider capping the intraday forecast horizon (e.g., only use intraday
#  ARMA for the next 4-5 bins and fall back to historical + inter-day for
#  more distant bins).]
```

---

## Verification of Key Citations (Draft 2 Additions)

| Spec Claim | Paper Location | Verified? |
|-----------|---------------|-----------|
| Joint term constraint: dual ARMA < 11 terms | p. 18, "fit each symbol with a dual ARMA model having fewer than 11 terms" | Yes |
| Intraday ARMA within-day: "Current Bin," "4 Bins Prior," "Today" | Exhibit 1, p. 18 | Yes |
| Day-boundary awareness needed | Exhibit 1 inputs are all within-day | Yes (implied) |
| AR lags < 5 for intraday | p. 18, "AR lags with a value less than five" | Yes |
| 1-month rolling for intraday ARMA | p. 18, "compute this model on a rolling basis over the most recent month" | Yes |
| AICc from Hurvich and Tsai | p. 17, "corrected AIC, symbolized by AIC_c" | Yes |
| MAPE formula | p. 17, "Measuring Raw Volume Predictions -- MAPE" | Yes |
| MAD formula for vol pct error | p. 17, "Measuring Percentage Volume Predictions -- Absolute Deviation" | Yes |
| Deviation limit 10% | p. 24, "depart no more than 10% away from a historical VWAP curve" | Yes |
| Switch-off 80% | p. 24, "once 80% of the day's volume is reached, return to a historical approach" | Yes |
| Order size 10% of 30-day ADV | p. 23, "Order size was set to 10% of 30-day average daily volume (ADV)" | Yes |
| No-intercept regressions | p. 19, "we perform both regressions without the inclusion of a constant term" | Yes (but applied to VWAP-error regression; applicability to surprise regression is unclear) |

All newly added citations verified as accurate.

---

## Overall Assessment

Draft 2 is a substantial improvement over Draft 1. All major issues have been
resolved, and the spec is now largely implementable. The five remaining minor
issues are primarily about adding implementability details and marking
inference points that were introduced in the revision. The most important
remaining gap is the rolling regression specification (m1), which a developer
would struggle to implement without additional detail. The other four issues
are clarifications that would improve the document but are not blocking.

The document is approaching final quality. One more revision addressing m1
(the rolling regression) would make it ready for a developer.
