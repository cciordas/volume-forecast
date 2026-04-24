# Critique of Implementation Specification Draft 1: Dual-Mode Volume Forecast

## Summary

The draft is well-structured, generally accurate, and demonstrates careful reading
of the Satish et al. (2014) paper. Citation traceability is strong throughout. The
validation section faithfully reproduces the paper's quantitative benchmarks.
However, there are several issues — two major and several minor — that could lead
a developer to implement an incorrect or suboptimal model. The most significant
concern is the construction of the intraday ARMA series, which the draft likely
gets wrong.

**Issue count:** 2 major, 7 minor.

---

## Major Issues

### M1. Intraday ARMA Series Construction Is Likely Wrong

**Spec section:** Pseudocode Part A, Component 3, Steps 3b-3c (lines 76-89)

**Problem:** The draft constructs the intraday ARMA input by concatenating
deseasonalized volumes across all days in a 1-month rolling window:

```
FOR each day d in rolling_1_month_window:
    FOR each bin i in 1..I:
        deseasonal_vol = volume[stock, bin=i, day=d] / seasonal_factor[i]
        intraday_series.append(deseasonal_vol)
```

This creates a single long time series of ~546 observations (21 days x 26 bins)
and fits one ARMA model to it. This interpretation has a critical structural
problem: lag-1 in this concatenated series connects bin 26 of day d to bin 1 of
day d+1, treating the overnight gap as a continuous transition. This is physically
meaningless — the overnight return-to-high-volume pattern at the open is not a
continuation of the prior day's close dynamics.

**Paper evidence:** Exhibit 1 (p. 18) shows three distinct inputs to the "ARMA
Intraday" component: "Current Bin," "4 Bins Prior to Current Bin," and "Today."
All three inputs are intraday, within-day observations. There is no cross-day
arrow feeding into the intraday ARMA. The text on p. 18 says: "When we examined
intraday deseasonalized data, we found that the autoregressive (AR) coefficients
quickly decayed, so that we used AR lags with a value less than five." The phrase
"AR lags less than five" combined with "4 Bins Prior to Current Bin" in Exhibit 1
strongly suggests the ARMA operates on the within-day bin sequence only.

**Recommended fix:** The intraday ARMA should be fit to the within-day
deseasonalized bin sequence. The 1-month rolling window likely refers to the
estimation sample: ARMA parameters are estimated using the past month's worth
of complete intraday sequences (each day treated as one realization of the
within-day process), and then applied to the current day's partial sequence for
forecasting. Concretely:

```
# Collect training data: each day provides one intraday sequence
training_sequences = []
FOR each day d in rolling_1_month_window:
    day_seq = [volume[stock, bin=i, day=d] / seasonal_factor[i] for i in 1..I]
    training_sequences.append(day_seq)

# Fit ARMA to pooled within-day sequences (or equivalently, stack them
# with breaks at day boundaries so lag-1 does NOT cross days)
intraday_model = fit_ARMA(training_sequences, order=(p, q),
                          handle_day_boundaries=True)
```

The developer needs clear guidance on how to handle day boundaries during ARMA
fitting. Options include: (a) fitting on each day's sequence independently and
averaging parameters, (b) fitting on the concatenated series but marking day
boundaries as missing/break points so the likelihood computation resets at each
boundary, or (c) treating each day as a panel observation. The draft should
specify which approach to use and mark it as Researcher inference if no paper
guidance exists.

### M2. "Fewer Than 11 Terms" Constraint May Be Misattributed

**Spec section:** Pseudocode Part A, Step 3c (line 87); Parameters table, row
`max_total_terms`

**Problem:** The draft applies the "fewer than 11 terms" constraint solely to the
intraday ARMA model:

```
IF p + q + 1 > 10: CONTINUE  # keep total terms < 11
```

But the paper says (p. 18): "we fit each symbol with a dual ARMA model having
fewer than 11 terms." The phrase "dual ARMA model" refers to the combined
inter-day + intraday ARMA system. The total term count likely spans both models:
(p_interday + q_interday + 1) + (p_intraday + q_intraday + 1) < 11.

**Paper evidence:** The sentence immediately follows discussion of both ARMA
components and uses the word "dual," which the paper consistently uses to
describe the two-ARMA system (inter-day + intraday).

**Impact:** If the constraint is joint, the intraday ARMA can use at most
~5-6 parameters (given the inter-day ARMA also needs parameters), which is
significantly more restrictive than the current spec allows (up to 10 intraday
parameters alone).

**Recommended fix:** Clarify that the 11-term constraint is on the combined
(inter-day + intraday) ARMA model. Add a joint constraint:

```
(p_interday + q_interday + 1) + (p_intraday + q_intraday + 1) < 11
```

If the proposer believes the constraint applies only to the intraday model,
provide explicit reasoning for why "dual" doesn't mean "combined."

---

## Minor Issues

### m1. Weight Non-Negativity Constraint Not Marked as Researcher Inference

**Spec section:** Pseudocode Part A, `train_regime_weights`, line 149

**Problem:** The pseudocode includes:

```
constraints = [w_hist >= 0, w_interday >= 0, w_intraday >= 0]
```

The paper (p. 18) says only "a dynamic weight overlay on top of these three
components... that minimizes the error on in-sample data." It does not specify
non-negativity constraints on the weights. While non-negativity is sensible (a
negative weight would invert a forecast component), this is the proposer's
design choice and should be marked as "Researcher inference" in the pseudocode,
as is done elsewhere in the document.

### m2. Exhibit 1 "4 Bins Prior to Current Bin" Detail Missing

**Spec section:** Pseudocode Part A, Component 3; Data Flow diagram

**Problem:** Exhibit 1 (p. 18) explicitly shows "4 Bins Prior to Current Bin" as
a distinct input arrow into the ARMA Intraday component. This detail is absent
from both the pseudocode and the data flow diagram. It suggests the intraday
ARMA specifically uses the most recent 4-5 bins (not the entire day's history)
as its active context, which is consistent with "AR lags with a value less than
five."

**Recommended fix:** Add this detail to the pseudocode. The intraday ARMA
prediction for bin j+1 should explicitly use bins max(1, j-3)..j as its recent
observation window, consistent with Exhibit 1.

### m3. Renormalization Step in Percentage Model Is Unmarked Inference

**Spec section:** Pseudocode Part B, Step 5 (lines 218-228)

**Problem:** The renormalization procedure that ensures remaining percentages sum
to (1 - cumulative_pct) is a substantial algorithmic step that is not described
in the paper. The paper mentions the constraint that volume percentages must
total to 100% (p. 15: "the day's forecasts must total to 100% to be
meaningful"), but does not describe how to redistribute after applying the
surprise-based adjustment. The draft's proportional scaling approach is
reasonable but should be explicitly marked as "Researcher inference."

### m4. MAPE as Optimization Objective May Have Numerical Issues

**Spec section:** Calibration Phase 2, Step 3 (lines 428-433)

**Problem:** The draft uses MAPE directly as the optimization objective for
weight calibration. MAPE is non-differentiable (due to the absolute value) and
can have a degenerate landscape when predicted values are near zero (the
denominator approaches zero for low-volume bins). The draft acknowledges
scipy.optimize.minimize with SLSQP as an option but does not address these
numerical concerns.

**Recommended fix:** Add guidance for the developer:
- Use a smooth approximation (e.g., Huber-like loss) or a derivative-free
  optimizer (Nelder-Mead, Powell) instead of gradient-based SLSQP.
- Alternatively, note that grid search over a discretized weight simplex is
  viable given only 3 weights per regime.
- Warn about numerical instability when bin volumes are near zero and suggest
  excluding or flooring such bins in the MAPE computation.

### m5. Volume Percentage Error Metric Definition Missing from Spec

**Spec section:** Validation, Expected Behavior (lines 459-461)

**Problem:** The draft cites volume percentage forecast errors (e.g., median
absolute error of 0.00874) without defining the error metric anywhere in the
pseudocode. The paper (p. 17) defines the metric as:

```
Error = (1/N) * sum_i |Predicted_Percentage_i - Actual_Percentage_i|
```

This is the mean absolute deviation (not percentage error — no normalization by
actual). This is important because MAPE and MAD produce very different numbers.
The developer needs to know which metric to compute for validation comparison.

**Recommended fix:** Add the volume percentage error formula explicitly, either
in the pseudocode or in the validation section, with a citation to p. 17 of
the paper.

### m6. Inter-day ARMA Training Window Ambiguity Insufficiently Resolved

**Spec section:** Parameters table, `N_interday`; Initialization item 3
(lines 397-398)

**Problem:** The draft notes the ambiguity around "Prior 5 days" in Exhibit 1
but then uses N_interday = 5 as the recommended value in the parameter table
while the initialization section (line 398) suggests the fitting window is
"plausibly longer, perhaps matching N_hist=21 days." This is contradictory.

Fitting an ARMA(p, q) with p, q up to 5 on only 5 data points is
statistically infeasible — a model with 11 parameters cannot be estimated from
5 observations. The "Prior 5 days" label in Exhibit 1 almost certainly refers
to the input data fed into an already-fitted model for prediction (i.e., the
most recent 5 daily observations are used as the ARMA's lagged inputs), not the
training window.

**Recommended fix:** Change the parameter table entry for `N_interday` to
distinguish between:
- **Fitting window:** The number of days used to estimate ARMA parameters
  (likely 63-126 days, i.e., 3-6 months; Researcher inference).
- **Prediction input:** The most recent 5 days of observed volumes used as
  lagged values for one-step-ahead prediction (from Exhibit 1).

### m7. Weight Constraint: Should Weights Sum to 1?

**Spec section:** Pseudocode Part A, `train_regime_weights` (lines 142-153)

**Problem:** The optimization constraints include only non-negativity. The paper
does not specify whether the weights must sum to 1 (convex combination) or are
unconstrained in scale. The draft's pseudocode allows arbitrary positive weights,
meaning the combination is not necessarily a weighted average — it could
amplify or dampen the overall forecast level.

This is a design choice with significant implications: unconstrained weights
can compensate for systematic bias in individual components (e.g., if all three
components underpredict, weights > 1/3 each would help), but they can also
overfit. A sum-to-1 constraint produces a true weighted average but assumes
components are unbiased.

**Recommended fix:** Discuss the trade-off explicitly and recommend a default
(Researcher inference). Given that the paper optimizes weights to minimize MAPE
on in-sample data, unconstrained-scale weights are plausible (MAPE penalizes
both over- and under-prediction). State this choice and the rationale.

---

## Verification of Key Citations

| Spec Claim | Paper Location | Verified? |
|-----------|---------------|-----------|
| 24% median MAPE reduction | p. 20, "Validating Volume Prediction Error" | Yes |
| 29% bottom-95% MAPE reduction | p. 20 | Yes |
| 7.55% vol pct error reduction (15-min) | Exhibit 9, p. 23 | Yes: HVWAP=0.00874, DVWAP=0.00808 |
| 6.29% bottom-95% vol pct reduction | Exhibit 9, p. 23 | Yes: 0.00986 vs 0.00924 |
| 9.62 vs 8.74 bps VWAP tracking | Exhibit 10, p. 23 | Yes |
| 9.1% VWAP reduction | Exhibit 10, p. 23 | Yes |
| R^2 = 0.51 (Dow 30) | Exhibit 3, p. 20 | Yes: 0.5146 |
| R^2 = 0.59 (high-variance) | Exhibit 5, p. 21 | Yes: 0.5886 |
| Deviation limit = 10% | p. 24 (Humphery-Jenner discussion) | Yes |
| Switch-off = 80% | p. 24 (Humphery-Jenner discussion) | Yes |
| Seasonal window = 6 months | p. 17 | Yes |
| Intraday ARMA fit = 1 month rolling | p. 18 | Yes |
| AR lags < 5 | p. 18 | Yes |
| Dual ARMA < 11 terms | p. 18 | Yes (but see M2 on scope) |
| N_hist = 21 from Exhibit 1 | Exhibit 1, p. 18 | Yes: "Prior 21 days" label |
| Chen et al. MAPE = 0.46 robust Kalman | Chen et al. 2016, Table 3 | Consistent with summary |
| Chen et al. VWAP = 6.38 bps | Chen et al. 2016, Table 4 | Consistent with summary |

All quantitative citations verified as accurate.

---

## Overall Assessment

The draft is a solid first attempt with strong citation discipline. The two major
issues (intraday ARMA series construction and the 11-term constraint scope) would
lead to a materially different model if implemented as written. The minor issues
are mostly about marking inferences and adding implementability details. With
these corrections, the spec should be ready for a developer.
