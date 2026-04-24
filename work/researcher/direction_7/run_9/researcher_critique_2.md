# Critique of Implementation Specification Draft 2: Kalman Filter State-Space Model

## Summary

Draft 2 is a major improvement over draft 1. All 12 issues from critique 1 have been
correctly addressed: the two algorithmic bugs (loop ordering, double subtraction) are
fixed, the conflicting robust filter versions are consolidated, and all missing formulas
(MAPE, VWAP TE, log-likelihood, VWAP definition) have been added. The spec is now
accurate, well-cited, and largely implementable as written.

The remaining issues are minor. I count 0 major issues, 1 medium issue, and 3 minor
issues. The medium issue is a consistency gap: the spec adds missing-observation handling
to the filter but does not carry this through to the EM M-step. The minor issues are
clarifications that would help a developer but are not blocking.

---

## Medium Issues

### ME1. EM M-Step Not Adapted for Missing Observations

**Problem:** The spec correctly integrates missing-observation handling into Algorithm 1
(lines 109-128): when y_tau is not observed, the correction step is skipped and the
filter passes through the prediction. However, the EM M-step (Algorithm 3, lines 226-267)
does not account for missing bins:

1. **r update (line 254):** Sums over tau=1..N. For missing bins, there is no observation
   y_tau to include in the squared residual. Including a missing bin in this sum would
   corrupt the r estimate. The sum should be restricted to observed bins, and the
   normalization should be 1/N_obs (number of observed bins) rather than 1/N.

2. **phi update (line 250):** Averages over T days for each bin i. If some days have
   missing volume for bin i, the average should use only the days where bin i was
   observed, with the denominator adjusted accordingly (T_i instead of T, where T_i
   is the number of days with nonzero volume for bin i).

3. **Log-likelihood (line 222):** Sums innovation_tau^2/S_tau over tau=1..N. For missing
   bins, no innovation was computed (correction was skipped). The sum should be over
   observed bins only, and N in the leading -(N/2)*log(2*pi) term should be N_obs.

4. **Robust EM phi and r updates (lines 359-368):** Same issue as above, compounded by
   the z_tau* terms (which are also undefined for missing bins, since the robust
   correction step was skipped).

**Paper context:** The paper explicitly assumes all volumes are nonzero (Section 2, page 3),
so the paper's equations do not handle this case. This is a spec-level extension that was
partially implemented (filter side) but not carried through (EM side).

**Impact:** If a developer implements Algorithm 1 with the missing-observation branch but
uses the EM M-step as written, they will either (a) get an error when referencing y_tau
for a missing bin, or (b) if they default y_tau to some value (e.g., 0), get silently
wrong parameter estimates. The inconsistency between the filter and the EM could cause
subtle bugs.

**Fix:** Add a note to Algorithm 3 and the robust EM section stating that all sums over
tau in the M-step (and the log-likelihood) should be restricted to observed bins only,
with normalizations adjusted to use N_obs (count of observed bins) instead of N where
applicable. For the phi update, normalize by T_i (observed count for bin i) instead of
T. Alternatively, if the paper's assumption of nonzero volumes will be enforced strictly
(dropping any bin with zero volume before model input), state this as a precondition
at the top of Algorithm 3 so the developer knows the assumption is in effect.

---

## Minor Issues

### MI1. Day-Boundary Set D Upper Bound Not Specified

**Problem:** Line 235 defines "D = {tau : tau = kI+1, k = 1,2,...}" without specifying the
upper bound of k. Since there are T training days, the day boundary transitions run from
day 1->2 through day (T-1)->T, giving k = 1, 2, ..., T-1. The set has T-1 elements,
which is consistent with the 1/(T-1) normalization in the sigma^eta update (line 242).
But a developer reading D literally as "k = 1, 2, ..." might not know where to stop.

**Fix:** Write "D = {tau : tau = kI+1, k = 1, 2, ..., T-1}" explicitly.

### MI2. Rolling Window Warm-Start Not Mentioned

**Problem:** The calibration section (lines 504-519) says "At the start of each new day in
the test period, re-run EM on the most recent N* bins to update parameters." It does not
mention whether to initialize EM from scratch (using the defaults in the Initialization
section) or warm-start from the previous day's converged parameters. Warm-starting would
typically reduce EM iterations from 5-10 down to 1-3 on a rolling window, since
parameters change slowly day to day. This is a practical optimization that affects
runtime by 3-5x.

**Fix:** Add a note suggesting warm-start initialization as a practical optimization:
"For rolling-window re-estimation, initialize theta^(0) with the previous day's
converged parameters rather than the defaults. This typically reduces iterations to
1-3 since parameters change slowly." Mark as researcher inference.

### MI3. Multi-Step Prediction Variance Not Shown

**Problem:** The spec describes h-step-ahead forecasting (lines 145-147) as "iterate the
prediction step h times" and gives y_hat_{tau+h|tau} = C * x_hat_{tau+h|tau} + phi_{tau+h}.
However, it does not show the corresponding prediction covariance Sigma_{tau+h|tau}, which
grows with each step due to accumulated process noise:

    Sigma_{tau+h|tau} = A_{tau+h-1} * Sigma_{tau+h-1|tau} * A_{tau+h-1}^T + Q_{tau+h-1}

applied recursively h times. This covariance determines:
- The prediction confidence interval width.
- The S_{tau+h} used in the log-normal bias correction (known limitation 2).
- The effective threshold in the robust filter if applied to multi-step forecasts.

For static prediction (all I bins predicted at once), the growing variance explains why
static MAPE (0.61) is worse than dynamic MAPE (0.46): later bins in the day have more
accumulated uncertainty.

**Fix:** Add the recursive prediction covariance formula after the h-step mean formula.
This is a standard Kalman filter result (not specific to the paper), so mark as researcher
inference.

---

## Verification of Critique 1 Fixes

I verified each fix in draft 2 against the paper:

| Critique 1 Issue | Fix Applied | Verified Correct? |
|---|---|---|
| M1: Loop ordering bug | Correction-then-prediction, tau=1 uses x_hat_{1\|0} from init | Yes -- traced through tau=1,2 |
| M2: Double subtraction | x_hat = x_hat_pred + K * (e - z*), single subtraction | Yes -- matches Eq (32) |
| HM1: Conflicting versions | Single Algorithm 4 with consistent tau indexing | Yes |
| ME1: Missing MAPE formula | Added Eq (37) definition in Validation section | Yes -- matches paper |
| ME2: Missing VWAP TE | Added Eq (42) with bps conversion | Yes -- matches paper |
| ME3: Missing log-likelihood | Innovation-based LL in Algorithm 3 and Calibration | Yes -- standard result, properly flagged as researcher inference |
| ME4: Missing obs not in code | if/else branch in Algorithm 1 correction step | Yes |
| ME5: Missing VWAP Eq (39) | Added in Data Flow Step 8 | Yes -- matches paper |
| MI1: Joseph form | Primary covariance update in Algorithms 1 and 4 | Yes |
| MI2: Normalization note | shares_outstanding_t clarified as per-day, external source | Yes |
| MI3: Table 3 averages | "average of per-ticker MAPEs" throughout | Yes |
| MI4: Phi ordering | phi before r with inline comment in M-step | Yes |

---

## Additional Citation Verification (New Content in Draft 2)

| Spec Claim | Paper Source | Verified? |
|---|---|---|
| MAPE formula Eq (37) | Section 3.3, Eq (37), page 7 | Yes |
| VWAP TE formula Eq (42) | Section 4.3, Eq (42), page 10 | Yes |
| VWAP definition Eq (39) | Section 4.3, Eq (39), page 8 | Yes (formula and price_{t,i} definition match) |
| Robust correction x_hat = x_hat_pred + K*(e-z*) | Eq (32), page 7 | Yes -- single subtraction confirmed |
| Clamped residual e-z* formula | Eq (34), page 7 | Yes -- all three cases verified |
| Robust r update with z* terms | Eq (35), page 7 | Yes -- all additional terms match |
| Robust phi update with z* | Eq (36), page 7 | Yes |

---

## Positive Notes

The draft is now at a high level of quality. Specific strengths:

1. **Algorithm 1 is now clean and unambiguous.** The correction-then-prediction ordering
   with the missing-observation branch makes the loop body self-contained and directly
   translatable to code.

2. **The robust filter consolidation (Algorithm 4) is excellent.** Single presentation,
   consistent tau indexing, clear soft-thresholding definition, and the clamped residual
   interpretation provides good intuition.

3. **The "Changes from Draft 1" section** provides excellent traceability for the review
   process. (This should be removed from the final version since it is not part of the
   implementation specification.)

4. **All researcher inferences are properly marked** (log-likelihood formula, initialization
   defaults, AR clamping, outlier fraction monitoring).

5. **The log-likelihood convergence formula** is well-motivated (uses quantities already
   in the filter) and the alternative (Q function from Appendix A.10) is mentioned for
   completeness.

6. **The Data Flow section** now provides a complete pipeline from raw volume through to
   VWAP weights, with all required inputs (including price data) clearly identified.

7. **Paper references table** has been expanded to cover all new additions and properly
   distinguishes paper citations from researcher inferences.
