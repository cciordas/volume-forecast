# Critique of Implementation Specification Draft 1: PCA Factor Decomposition (BDF)

## Summary

The draft is a solid foundation -- well-structured, with clear pseudocode, good data flow
documentation, and thorough treatment of edge cases and limitations. The paper references
are generally accurate and traceable. However, I identified several issues that range from
a significant algorithmic error (eigendecomposition matrix choice) to model labeling
confusion, estimation method misspecification, and gaps in the dynamic VWAP execution
logic. Below I list 6 major issues and 8 minor issues.

---

## Major Issues

### M1. Eigendecomposition operates on the wrong matrix

**Location:** Pseudocode Phase 1, Step 3 (lines 87-91); Paper References table.

**Problem:** The spec computes the eigendecomposition of `X @ X.T / (N * T)`, which is a
(T x T) matrix. However, BDF 2008 Section 2.2 (text below Eq. 6) states:

> "The estimated factors matrix F_hat is proportional (up to T^{1/2}) to the eigenvectors
> corresponding to the r-largest eigenvalues of the **X'X** matrix"

where X is (T x N). So X'X is an (N x N) matrix. The spec uses XX' which is (T x T).

Both approaches yield mathematically equivalent decompositions (the non-zero eigenvalues
of XX' and X'X are the same up to a scalar, and factors/loadings can be recovered from
either). However, the practical difference is significant:

1. **Computational cost:** T = L_days * k_bins = 20 * 26 = 520, while N is typically 30-500.
   Computing eigendecomposition of a (520 x 520) matrix is more expensive than (N x N) when
   N < T, which is the typical case in BDF's setup. BDF 2008 explicitly notes that "the
   cross-section dimension N is small compared to the time series dimension T" (Section 2.2,
   paragraph before Eq. 3).

2. **Fidelity to paper:** The paper explicitly states the X'X formulation with the F'F/T = I_r
   normalization. The spec should match this.

**Recommendation:** Rewrite Step 3 to compute eigendecomposition of X.T @ X / T (an N x N
matrix). The factor estimates are then recovered as F_hat = X @ eigenvectors[:, :r] (with
appropriate normalization). Alternatively, if the spec wants to use the XX' approach for
generality, it must explain the equivalence and note that the paper uses X'X.

### M2. ARMA(1,1) vs AR(1) model labeling confusion

**Location:** Pseudocode Phase 1, Step 6, Option A (lines 112-119); Parameter table
(psi_1, psi_2 rows).

**Problem:** The spec labels the specific component model as "ARMA(1,1)" following BDF
2008's label, but then correctly notes in a comment (line 116-118) that "This is actually
an AR(1) with intercept as written in the paper." This is confusing and needs to be
resolved definitively, not left as an ambiguous comment.

**Paper evidence:**
- BDF 2008 Eq. (10): e_{i,t} = psi_1 * e_{i,t-1} + psi_2 + epsilon_{i,t}. This is
  indeed an AR(1) with intercept -- there is no moving average term.
- BDF 2008's text calls it "ARMA(1,1) with white noise" which is a misnomer in the paper
  itself. The equation has no MA component (no theta * epsilon_{t-1} term).
- Szucs 2017 Eq. (5): e_p = c + theta_1 * e_{p-1} + epsilon_p. Szucs explicitly calls
  this an "AR(1) model", confirming it is not ARMA.

**Recommendation:** The spec should definitively state this is an AR(1) with intercept
(not ARMA(1,1)). Use the label "AR(1)" throughout (as Szucs does), with a footnote
explaining that BDF 2008 mislabels it as ARMA(1,1). The current approach of using "ARMA"
in the model type flag, parameter names, and variant discussion while noting the
discrepancy in a code comment is a recipe for developer confusion.

### M3. Estimation method for AR(1) and SETAR is misspecified

**Location:** Initialization Section, points 3-4 (lines 315-329).

**Problem:** The spec recommends OLS for estimating the AR(1) specific component model
(point 3) and conditional OLS for SETAR (point 4). However, BDF 2008 Section 2.3 (last
paragraph, after Eq. 11) explicitly states:

> "substitute e_{i,t} for e_hat_{i,t} in Eqs. (10) or (11) and estimate Eqs. (10) or
> (11) **by maximum likelihood**."

While for the simple AR(1) case, OLS and MLE yield identical point estimates (assuming
Gaussian errors), the SETAR case is different -- the threshold parameter tau is typically
estimated by profile likelihood or conditional least squares with grid search, and the
paper says MLE. The spec's SETAR estimation procedure (grid search + conditional OLS) is
a valid and standard approach (Hansen 1997), but it should note the discrepancy with the
paper's stated method and justify the choice.

**Recommendation:** (a) Note that BDF 2008 specifies MLE, not OLS. (b) For AR(1), state
that OLS and MLE are equivalent under Gaussian errors. (c) For SETAR, justify the
conditional OLS grid search approach as a standard equivalent (cite Hansen 1997 or Tong
1990), while noting the paper says MLE.

### M4. Dynamic VWAP execution logic has a gap in remaining volume computation

**Location:** Pseudocode Phase 2, Step 4 (lines 174-184).

**Problem:** The pseudocode for computing volume shares in dynamic execution is incomplete.
The variable `x_forecast_remaining[i, j_current+1 : k_bins]` on line 181 is used but
never computed. The spec only computes the one-step-ahead forecast `x_forecast[i,
j_current + 1]` (Step 3), but the volume share calculation requires forecasts for ALL
remaining bins (j_current+2 through k_bins) to compute the denominator (estimated daily
total).

BDF 2008 Section 4.2.3 describes the dynamic execution as: at each step t, produce the
one-step-ahead forecast and also use the model to get multi-step forecasts for the rest of
the day. The proportion to trade is x_hat_{t+1} / sum_{l=t+1}^{K} x_hat_{l}, applied to
the remaining volume.

**Recommendation:** Add explicit pseudocode for computing the multi-step forecasts for
bins j_current+2 through k_bins. For the AR(1) model, this is straightforward (iterate
the AR recursion). For SETAR, multi-step forecasts are more complex (the regime at future
steps is unknown). The spec should specify how to handle this -- the simplest approach is
to use the common component forecast alone for bins beyond j_current+1 (since multi-step
SETAR forecasts decay to zero for the specific component, as the spec itself notes in the
Known Limitations section about static execution).

### M5. VWAP tracking error numbers need correction

**Location:** Validation Section, point 5 (lines 435-439).

**Problem:** The spec claims "Dynamic PCA-SETAR execution reduces out-of-sample VWAP
tracking error (MAPE) from 10.06% (classical) to 8.98% (dynamic BDF-SETAR)." These
numbers are cited as from "BDF 2008, Table 2, panel 3."

Checking BDF 2008 Table 2 (the summary table I read), the values for out-of-sample VWAP
execution are:
- Classical approach: Mean MAPE = 0.1006
- PC-SETAR with dynamical adjustment: Mean MAPE = 0.0898

These are portfolio-level MAPE values expressed as fractions, not percentages. The spec
converts them to percentages (10.06% and 8.98%), which is fine but should be explicitly
noted. More importantly, the claim of "approximately a 10% relative reduction" is correct
((10.06 - 8.98) / 10.06 = 10.7%).

However, the spec then claims "For high-volatility stocks, reductions can exceed 50%."
I could not find this specific claim in BDF 2008. Some individual stocks show large
improvements (e.g., CAP GEMINI goes from 23.23 to a lower value) but characterizing
this as "high-volatility stocks" with ">50% reduction" needs a specific citation or
should be marked as Researcher inference.

**Recommendation:** Either provide the exact stock and table cell supporting the ">50%
for high-volatility stocks" claim, or mark it as Researcher inference. The portfolio-level
numbers are correctly cited.

### M6. BDF 2008 uses 39 stocks, not 31

**Location:** Validation Section, Sanity Check 5 (line 477), and various places.

**Problem:** The spec says "SETAR wins for 36/39 CAC40 stocks" (Variants section, line
265) and BDF 2008 Section 3.1 says "all the securities included in the CAC40 index."
Table 1 in BDF 2008 lists exactly 39 stocks. But the paper text at the top of Section 3.2
results says "there are only three, of the **31**, companies for which the ARMA slightly
surpasses the SETAR model."

This "31" vs "39" discrepancy exists within BDF 2008 itself (39 stocks in Table 1, but
results reported for 31 or 39 depending on the section -- likely the VWAP results used
fewer stocks or some were excluded). The spec should note this inconsistency rather than
using 39 throughout.

Actually, re-reading more carefully: the BDF paper's Section 3.2 text says "there are
only three, of the 31, companies" -- but Table 1 has 39 companies. This likely means the
MAPE comparison was done on a subset. The spec should flag this discrepancy.

**Recommendation:** Note the 31 vs 39 discrepancy in BDF 2008 and clarify which count
applies to which result.

---

## Minor Issues

### m1. Turnover values in "Type information" may be misleading

**Location:** Data Flow section, Type information (line 252).

**Problem:** The spec states "All turnover values: float64 (typically in range 1e-5 to
1e-2)." Looking at BDF 2008 Table 1, the mean turnover across CAC40 stocks is 0.0166
(~1.7e-2) with Q95 at 0.0445 (~4.5e-2). Several stocks exceed 0.04 routinely. The range
1e-5 to 1e-2 may understate the upper end.

**Recommendation:** Widen the stated typical range to approximately 1e-5 to 5e-2 based
on BDF 2008 Table 1 data.

### m2. The spec does not address demeaning before PCA

**Location:** Pseudocode Phase 1, Steps 1-3.

**Problem:** Standard PCA typically requires demeaning the data (subtracting column means)
before computing eigenvectors. BDF 2008 does not explicitly mention demeaning, and the PCA
formulation in Eq. 6 does not show it. The Bai (2003) framework allows for non-zero means
absorbed into the factor structure. However, a developer might reasonably ask whether to
demean X before eigendecomposition.

**Recommendation:** Add a note clarifying that demeaning is NOT required in the Bai (2003)
PCA framework (the mean is absorbed into the factor loadings), but that doing so would not
materially change results and is acceptable. This prevents a common PCA implementation
pitfall.

### m3. First bin of the day has no specific component forecast

**Location:** Pseudocode Phase 2 (lines 131-186).

**Problem:** The intraday forecast loop starts at j_current = 1 (the most recently
completed bin). But for the very first bin of the day (j = 1), there is no prior observed
specific component to condition on. How should e_forecast be initialized for the first
forecast of the day (predicting bin 1 or bin 2)?

The spec does not address this. BDF 2008 Section 4.2.3 describes starting with the static
forecast at the beginning of the day and then dynamically updating. This implies that for
the first bin, the forecast is just the common component (e_forecast = 0 or uses the last
e value from the previous day).

**Recommendation:** Specify explicitly how to initialize the specific component forecast
at the start of each trading day. Options: (a) set e_forecast = 0 for bin 1 (use common
component only), (b) use the last e value from the previous trading day's final bin, or
(c) use the unconditional mean of the specific component (which is ~0 by construction).
State which option is chosen and why.

### m4. The Bai & Ng IC section uses inconsistent notation for residual variance

**Location:** Factor Count Selection section (lines 363-391).

**Problem:** The formula for V(r) in the IC section (line 374) uses the same V(r) notation
as BDF 2008 Eq. 6, but in the IC context it represents the minimized residual variance
(a scalar), while in Eq. 6 it represents the objective function being minimized. A
developer might confuse the two.

**Recommendation:** Use a distinct symbol for the IC residual variance (e.g., sigma^2(r)
or V_r) to avoid confusion with the PCA objective function V(r).

### m5. Parameter table does not include r_max

**Location:** Parameters table (lines 287-299).

**Problem:** The Bai & Ng IC procedure requires an r_max hyperparameter (the maximum
number of factors to consider). The spec mentions r_max = 10 as an example in the IC
section (line 371) but does not include it in the parameter table.

**Recommendation:** Add r_max to the parameter table with recommended value 10, low
sensitivity, and range 5-20.

### m6. SETAR estimation grid search percentile range is underspecified

**Location:** Initialization Section, point 4 (lines 324-329).

**Problem:** The spec suggests "10th to 90th percentile in 1% steps" as the grid for
threshold candidates. This produces 81 candidate values. The spec does not discuss:
- Whether to use the specific component values from the current window only, or all
  historical data.
- What to do if a regime has too few observations (e.g., < 10) for reliable OLS estimation
  of regime-specific parameters.
- Minimum observations per regime constraint.

**Recommendation:** Add a minimum regime size constraint (e.g., each regime must contain
at least 15% of observations, i.e., trim the grid to the 15th-85th percentile range).
This prevents degenerate regime estimates with very few observations.

### m7. The spec does not specify how to handle the cross-day boundary in the specific component series

**Location:** Pseudocode Phase 1, Step 6 (lines 107-128).

**Problem:** The specific component series E_hat[:,i] spans multiple days (L_days * k_bins
observations). The AR(1)/SETAR model is fit to this concatenated series as if it were
continuous. But at day boundaries (between bin k of day d and bin 1 of day d+1), there is
typically a large overnight jump in the specific component (overnight information arrival).

Fitting AR(1) to this concatenated series without accounting for overnight breaks could
contaminate the parameter estimates. BDF 2008 does not discuss this issue explicitly.

**Recommendation:** Either (a) note this as a known limitation and state that the AR(1)
is fit to the concatenated series following the paper, or (b) recommend fitting the AR(1)
only to within-day observations (treating each day as an independent short series,
estimating parameters by pooling the within-day OLS residuals). Option (a) is simpler and
faithful to the paper; option (b) may produce better estimates.

### m8. The spec could benefit from a concrete worked example

**Location:** General.

**Problem:** The spec is detailed in its pseudocode but lacks a concrete numerical example
that a developer could use to verify their implementation step by step. For instance:
"Given a 3-stock, 2-day, 4-bins-per-day toy example with these specific turnover values,
the PCA should produce these factor values, these loadings, and these specific component
values."

**Recommendation:** Add a small worked example (e.g., N=3, L_days=2, k=4) with explicit
numerical values at each step. This is the single most effective way to prevent
implementation errors.

---

## Citation Verification Summary

| Spec Claim | Cited Source | Verified? | Notes |
|---|---|---|---|
| PCA eigendecomposition on r-largest eigenvalues | BDF 2008, Eq. 6 | Partially | Paper says X'X, spec uses XX'. See M1. |
| Eq. 10 is "ARMA(1,1)" | BDF 2008, Eq. 10 | Incorrect label | Equation is AR(1) with intercept. See M2. |
| Estimation by OLS | BDF 2008, Section 2.3 | Incorrect | Paper says MLE. See M3. |
| SETAR Eq. 11 | BDF 2008, Eq. 11 | Correct | Matches paper exactly. |
| Common component forecast Eq. 9 | BDF 2008, Eq. 9 | Correct | Matches paper. |
| k=25, 20-min bins for CAC40 | BDF 2008, Section 3.1 | Correct | Paper says 25 intervals. |
| k=26, 15-min bins for DJIA | Szucs 2017, Section 2 | Correct | 26 bins, 9:30-16:00. |
| L=20 trading days | BDF 2008, Section 3.1 | Correct | "20-day window." |
| SETAR outperforms ARMA for 36/39 stocks | BDF 2008, Section 3.2 | Needs clarification | Paper says "three of the 31" -- see M6. |
| BDF-SETAR MAPE 0.399, BDF-AR 0.403 | Szucs 2017, Table 2a | Correct | Values match. |
| BDF-AR MSE 6.49e-4, SETAR 6.60e-4 | Szucs 2017, Table 2a | Correct | Values match. |
| VWAP tracking: 10.06% classical, 8.98% dynamic SETAR | BDF 2008, Table 2, panel 3 | Correct | Values match (0.1006, 0.0898). |
| ">50% reduction for high-volatility stocks" | BDF 2008, Section 4.3 | Unverified | No specific citation. See M5. |
| 2 hours runtime for 33 stocks | Szucs 2017, Section 4 | Correct | Szucs reports ~2 hours vs ~60 machine-days. |
| BDF outperforms CMEM | Szucs 2017, Section 5 (Results) | Correct | BDF wins on MAPE; BCG_3 wins on MSE for some periods. |
| Bai & Ng IC_p2 recommendation | Researcher inference | N/A | Correctly marked as inference. |

---

## Overall Assessment

The draft captures the core BDF algorithm correctly and provides good coverage of
validation benchmarks, edge cases, and limitations. The major issues are primarily about
precision rather than fundamental misunderstanding:

- **M1 (eigendecomposition matrix)** is the most impactful for implementation correctness
  -- a developer following the spec literally would compute a larger matrix than necessary
  and would deviate from the paper's stated formulation.
- **M2 (AR vs ARMA labeling)** is a clarity issue that will cause confusion.
- **M3 (OLS vs MLE)** is minor for AR(1) but matters for documenting fidelity to the paper.
- **M4 (dynamic VWAP multi-step forecast gap)** is a significant pseudocode gap -- a
  developer cannot implement the VWAP execution strategy without resolving this.

After addressing these issues, the spec should be implementation-ready.
