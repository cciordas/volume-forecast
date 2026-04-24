# Critique of Implementation Specification: PCA Factor Decomposition (BDF) — Draft 1

## Summary

The draft is well-structured, thorough, and demonstrates strong understanding
of both the BDF 2008 and Szucs 2017 papers. The pseudocode is largely
translatable to code, the parameter table is comprehensive, and the validation
section provides useful sanity checks. However, I identified 4 major issues and
8 minor issues. The major issues involve an ordering bug in the pseudocode, an
incorrect table citation, a computational concern, and a missing Researcher
inference label. The minor issues are mostly about clarity, precision, and
missing detail that could trip up a developer unfamiliar with the papers.

---

## Major Issues

### M1. Centering inconsistency between factor selection and PCA (Pseudocode, Phase 1, lines 82-96)

**Problem:** In `estimate_model`, the call to `select_num_factors(X, N, P)` at
line 85 passes the raw (uncentered) turnover matrix X. Centering does not
happen until line 90: `X_centered = X - mean(X, axis=0)`. The actual PCA
eigendecomposition (lines 95-100) operates on `X_centered`. However,
`select_num_factors` internally calls `estimate_factors(X, r_candidate)` on the
uncentered data to compute residuals and the IC criterion.

This means the number of factors r is selected on uncentered data, but the
decomposition itself uses centered data. These will generally produce different
factor counts because the first eigenvector of uncentered data captures the
overall mean level, whereas centering removes this.

**Impact:** A developer following the pseudocode literally would have an
inconsistent pipeline: factor count selected under one preprocessing regime,
factor extraction performed under another. This could lead to over- or
under-estimation of r by 1 factor.

**Fix:** Either (a) center X before passing it to `select_num_factors`, or (b)
have `select_num_factors` center internally before computing factors and
residuals. The centering step should come before both factor selection and
factor extraction.

**Evidence:** BDF 2008, Section 2.2, Eq. 6 defines V(r) as the minimization
objective. The Bai (2003) procedure that both papers reference does not mandate
centering (it can include an intercept as an additional factor), but if
centering is applied for PCA, it must also be applied for factor count
selection. This is not an issue the papers address explicitly, making it a
practical implementation detail that needs to be handled consistently.

### M2. Incorrect table citation for aggregate dynamic VWAP results (Validation, Section "VWAP tracking error")

**Problem:** The spec states: "Dynamic PCA-SETAR: ~0.090 MAPE on CAC40
out-of-sample (BDF 2008, Table 5, column 1 mean over all stocks)."

Table 5 of BDF 2008 contains per-stock MAPE values for the dynamic PCA-SETAR
model (I verified this directly: Table 5 lists individual company names like
ACCOR, AGF-ASS.GEN.FRANCE, etc., with per-stock Mean/Std/Q95 columns). It does
not contain an aggregate "mean over all stocks" row. The aggregate
out-of-sample VWAP comparison (including the ~0.0898 figure for dynamic
PCA-SETAR) appears in the third panel of **Table 2** (labeled "Result of
out-of-sample estimation for VWAP execution"), not Table 5.

Additionally, Table 3 provides the per-stock out-of-sample VWAP results for
the theoretical PCA-SETAR model, while Table 5 provides the per-stock
out-of-sample results for the dynamic PCA-SETAR model. The overall comparison
table is Table 2.

**Impact:** A developer trying to verify the ~0.090 figure by reading Table 5
would find per-stock values and no aggregate, causing confusion. They would
need to compute the mean themselves from Table 5, or find it in Table 2.

**Fix:** Change the citation to "BDF 2008, Table 2, third panel (out-of-sample
VWAP execution), PC-SETAR dynamic row."

### M3. PCA eigendecomposition operates on (P x P) matrix instead of (N x N) (Pseudocode, Phase 1, lines 95-96)

**Problem:** The spec computes `cov_matrix = (1.0 / N) * X_centered @
X_centered.T` of shape (P, P), where P = k * L. With the paper's parameters
(k=25, L=20), P = 500, yielding a 500 x 500 eigendecomposition. With Szucs's
parameters (k=26, L=20), P = 520.

Meanwhile, N = 39 (CAC40) or 33 (DJIA). Computing the eigendecomposition of
the (N x N) matrix `(1/P) * X_centered.T @ X_centered` (at most 39 x 39)
would be orders of magnitude faster and yield the same eigenvalues. Factors can
be recovered as F_hat = X_centered @ eigenvectors_of_XtX * (1/sqrt(eigenvalues)).

**Impact:** This is not a correctness issue — both approaches produce identical
results. But for a rolling daily re-estimation over thousands of days, the
computational difference is significant. A 500 x 500 eigendecomposition is
roughly (500/39)^3 ≈ 2100x slower than a 39 x 39 one.

**Evidence:** BDF 2008, Section 2.2 states: "the estimated factors F̂ is
proportional (up to T^{1/2}) to the eigenvectors corresponding to the
r-largest eigenvalues of the X'X matrix." With the paper's notation where X is
(T x N), X'X is (N x N). The paper actually references the smaller matrix.
Szucs 2017, Section 4.1 further notes the BDF model runs in about 2 hours for
33 stocks over 2,668 days — this speed is consistent with the (N x N) approach,
not the (P x P) approach.

**Fix:** The pseudocode should compute the eigendecomposition of the (N x N)
matrix and recover factors via the relationship F_hat = X_centered @
V[:, :r] / sqrt(eigenvalues[:r]) * sqrt(P), where V are eigenvectors of
X_centered^T @ X_centered. Alternatively, use SVD of X_centered directly:
X_centered = U @ S @ V^T; then F_hat = U[:, :r] * S[:r] relates to the factors
and Lambda_hat = V[:, :r] relates to loadings (with appropriate normalization).

### M4. Column-mean centering before PCA not marked as Researcher inference (Pseudocode, Phase 1, line 90; Data Flow step 1)

**Problem:** The spec introduces `X_centered = X - mean(X, axis=0)` as the
first PCA preprocessing step and cites "BDF 2008, Section 2.2, Eq. 6." However,
Eq. 6 in BDF 2008 defines only the objective V(r) and the normalization
F'F/T = I_r. Neither BDF 2008 nor Szucs 2017 explicitly prescribes centering
by column means before eigendecomposition. The Bai (2003) procedure that both
papers reference can handle an intercept as an additional estimated factor
rather than pre-subtracting means.

**Impact:** A developer who reads the original paper would not find a centering
instruction and might question whether it is correct. Centering is a reasonable
and standard choice for PCA, but it changes the interpretation of the first
factor (with centering, the first factor captures shape variation; without it,
the first factor may capture the overall level). The spec should explicitly mark
this as a Researcher inference and explain the reasoning, as it does for other
interpolation decisions (e.g., e_last initialization at market open).

**Fix:** Mark the centering step as "Researcher inference" with justification:
centering ensures the PCA factors capture intraday shape variation rather than
the mean turnover level, and the common component forecast adds back the column
means.

---

## Minor Issues

### m1. OLS vs. MLE estimation method discrepancy (Phase 1b, fit_ar1 and fit_setar)

**Problem:** BDF 2008, Section 2.3 explicitly says: "estimate Eqs. (10) or (11)
by maximum likelihood." The spec uses OLS throughout and does not explain the
equivalence.

**Impact:** A developer reading the paper alongside the spec would wonder why
the estimation method differs. For AR(1) with Gaussian errors, conditional MLE
and OLS are algebraically equivalent. For SETAR, conditional MLE under Gaussian
errors reduces to grid search over tau + OLS within each regime (conditional
least squares). Both are correct, but the equivalence should be stated.

**Fix:** Add a note after the AR(1) and SETAR fitting functions explaining that
OLS is used because it is equivalent to conditional MLE under Gaussian errors,
which is what BDF 2008 prescribes.

### m2. Trading days count inaccuracy (Validation, "Estimation speed")

**Problem:** The spec states "~2,648 days" for the Szucs 2017 sample. Szucs
2017, Table 1 reports "Number of days: 2 668" (2,668 trading days).

**Fix:** Change to 2,668.

### m3. MAPE formula not included (Validation section)

**Problem:** The spec references MAPE as the primary evaluation metric but does
not define it. Szucs 2017, Section 3, Eq. 2 defines:

    MAPE = (1/N) * sum_{t=1}^{N} |Y_t - Y_t^f| / Y_t

where Y_t is actual and Y_t^f is forecast. The formula has a division by Y_t,
which means zero-volume bins would produce division-by-zero errors.

**Impact:** A developer implementing the evaluation would need to look up the
formula elsewhere and would miss the zero-division edge case.

**Fix:** Add the MAPE and MSE formulas to the Validation section, citing Szucs
2017 Eqs. 1-2. Note the zero-division edge case and its connection to the
liquidity requirement (all bins must have non-zero volume).

### m4. Multi-step chaining in intraday forecast not clearly documented (Phase 2, forecast_intraday)

**Problem:** The forecast loop at lines 260-276 sets `e_last = e_forecast` at
each step, chaining one-step-ahead predictions. This is the correct approach
for multi-step-ahead forecasting, and it is the reason static execution fails
(BDF 2008, Section 4.2.1: "long-horizon ARMA forecasts collapse to zero").
However, the pseudocode does not clearly distinguish between two usage modes:

1. **Dynamic mode** (observing actual turnover after each bin, then forecasting
   only one bin ahead): truly one-step-ahead, no chaining needed.
2. **Static/initial mode** (forecasting all remaining bins from the current
   observation): uses chained multi-step forecasting.

The loop in `forecast_intraday` always chains, which is correct for static
forecasting but would be wrong if called after each observed bin for a single
next-bin forecast.

**Impact:** A developer might use `forecast_intraday` in a loop (calling it
after each observation to get all remaining forecasts) and accidentally chain
multi-step predictions when only the first forecast is needed for execution.

**Fix:** Add a comment clarifying that `forecast_intraday` produces multi-step
chained forecasts for all remaining bins, and that `execute_dynamic_vwap`
correctly uses only `remaining_forecasts[0]` from each call, effectively
performing one-step-ahead forecasting. The chaining within `forecast_intraday`
is needed for the denominator (total remaining forecast) in the participation
fraction calculation.

### m5. select_num_factors upper bound r_max not well justified (Phase 1a)

**Problem:** The spec sets `r_max = min(20, min(N, P) - 1)` and labels this
"Low — just an upper bound for search." The value 20 appears to be arbitrary.
BDF 2008 does not specify r_max. For the typical case (N=33-39, P=500-520),
this gives r_max = min(20, 32) = 20, which is very conservative given that the
expected r is 1-3.

**Impact:** Searching over 20 candidates is not expensive, so this is primarily
a documentation issue. But a developer might wonder where 20 comes from.

**Fix:** Either cite a source for the upper bound or note it as a practical
choice (Researcher inference). Consider adding that Bai & Ng (2002) recommend
searching up to a "reasonable upper bound" and that 20 is generous for volume
data where 1-3 factors are typical.

### m6. Incomplete specification of PCA on the full estimation window (Algorithm, Phase 1)

**Problem:** The pseudocode treats the specific component e_hat as a single
time series of length P = k * L for each stock, concatenating all L days'
intraday bins into one long series. This means the AR(1)/SETAR model is fit to
a series where bin k of day d is followed by bin 1 of day d+1. The overnight
gap between these two observations is very different from the 15-20 minute gap
within a day.

Neither BDF 2008 nor Szucs 2017 discusses this overnight discontinuity
explicitly, but it is a practical concern. The AR coefficient theta_1 estimated
on this concatenated series reflects a mix of intraday persistence and
overnight dynamics.

**Impact:** The estimated AR coefficient may be biased by overnight jumps in the
specific component. A developer should be aware this is a known limitation.

**Fix:** Add a note (either in Edge Cases or Known Limitations) that the time
series model treats overnight transitions identically to intraday transitions.
This is consistent with both BDF 2008 and Szucs 2017's implementations, but
the developer should be aware of it.

### m7. Validation MAPE benchmarks mix in-sample and out-of-sample without clear labels (Validation, "Forecast accuracy (MAPE)")

**Problem:** Item 4 lists BDF-SETAR MAPE as ~0.075 on CAC40 and ~0.399 on DJIA.
The CAC40 figure (0.0752) is from BDF 2008 Table 2 (labeled "Predictions of
model for intraday volume"), which covers the in-sample estimation period
(September-December 2003). The DJIA figures from Szucs 2017 are full-sample
results (not truly out-of-sample either, since each day uses a rolling window).
The spec says "the higher MAPE in Szucs reflects the 11-year sample including
the 2008 financial crisis," which is helpful context, but does not clearly
label whether these are in-sample, rolling out-of-sample, or full-sample.

Item 5 then cites out-of-sample VWAP results from BDF 2008.

**Impact:** A developer might expect their implementation to match the ~0.075
MAPE on a completely different dataset and be confused when results differ.

**Fix:** Clearly label each benchmark as "in-sample" (BDF 2008 Table 2,
volume prediction), "rolling out-of-sample" (Szucs 2017, full sample with
20-day rolling window), or "out-of-sample" (BDF 2008 Table 2, VWAP execution).

### m8. Missing edge case: what happens when the estimated theta_1 is exactly 1 or very close (Edge Cases)

**Problem:** Sanity check 4 correctly notes that |theta_1| < 1 is required for
stationarity. But the edge case section does not specify what to do if the
estimated theta_1 violates this condition. Should the developer fall back to a
simpler model (e.g., historical average only)? Constrain theta_1 to [-0.99,
0.99]? Skip the stock?

**Impact:** For a stock with highly persistent specific component deviations,
OLS may produce |theta_1| >= 1. Without guidance, a developer might proceed
with a non-stationary model, producing divergent forecasts.

**Fix:** Add handling guidance: if |theta_1| >= 1, either (a) constrain to
theta_1 = sign(theta_1) * 0.99 and log a warning, or (b) fall back to the
U-method for that stock. Note this as Researcher inference.

---

## Citation Verification Summary

| Spec Claim | Cited Source | Verified? | Notes |
|---|---|---|---|
| Decomposition x = c + e | BDF 2008, Sec 2.2, Eq. 4 | Yes | Eq. 4: X_i = FA_i + e_i matches |
| PCA via eigendecomposition | BDF 2008, Sec 2.2, Eq. 5-6 | Yes | Eq. 5-6 describe the factor model and PCA objective |
| Factor count IC_p1 | BDF 2008, Sec 2.2, ref. Bai & Ng 2002 | Yes | Paper references Bai & Ng for factor count selection |
| Common forecast = time-of-day avg | BDF 2008, Sec 2.3, Eq. 9 | Yes | Eq. 9 matches |
| AR(1) specific model | BDF 2008 Eq. 10; Szucs Eq. 5 | Yes | Both confirmed: e_t = c + theta * e_{t-1} + eps |
| SETAR specific model | BDF 2008 Eq. 11; Szucs Eqs. 6-7 | Yes | Both confirmed with matching indicator function |
| "ARMA(1,1)" is actually AR(1) | Researcher inference | Yes | BDF 2008 text says ARMA(1,1) but Eq. 10 has no MA term; Szucs explicitly uses AR(1) in Eq. 5 |
| Dynamic VWAP execution | BDF 2008, Sec 4.2.2 | Yes | Section 4.2.2 describes the dynamic approach |
| ~0.090 MAPE dynamic SETAR | BDF 2008, "Table 5" | **No** | Should be Table 2 third panel or computed from Table 5 per-stock values |
| Estimation window = 20 days | BDF 2008 Sec 3.2; Szucs Sec 3 | Yes | Both confirm L=20 |
| Turnover = V / TSO | BDF 2008 Sec 3.1; Szucs Sec 2 | Yes | Both confirm |
| SETAR tau grid [15th, 85th] | Researcher inference | N/A | Correctly labeled; papers do not specify grid |
| e_last init at market open | Researcher inference | N/A | Correctly labeled; reasonable choice |
| Centering before PCA | BDF 2008, Sec 2.2, Eq. 6 | **No** | Paper does not prescribe centering; should be Researcher inference |

---

## Overall Assessment

The draft is strong — the algorithmic structure is correct, the pseudocode is
mostly implementable, and the Researcher correctly identified the ARMA(1,1)
misnomer. The main areas for improvement are:

1. Fix the ordering bug where factor selection operates on different data than
   the PCA decomposition (M1).
2. Correct the Table 5 citation (M2).
3. Address the computational approach for PCA by using the smaller (N x N)
   matrix (M3).
4. Label the centering decision as Researcher inference (M4).

With these fixes, the spec would be ready for implementation.
