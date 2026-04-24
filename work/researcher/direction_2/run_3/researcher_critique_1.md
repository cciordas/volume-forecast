# Critique of Implementation Specification: PCA Factor Decomposition for Intraday Volume (BDF Model) — Draft 1

## Summary

The draft is a high-quality, well-structured implementation specification. It covers the
full pipeline from data preprocessing through PCA factor extraction, AR(1)/SETAR
modeling, dynamic VWAP execution, and validation. Citations are thorough and mostly
accurate. The document could be implemented by a competent developer.

I identified 5 major issues and 9 minor issues. The major issues concern algorithmic
correctness (PCA dual-case normalization), computational efficiency (redundant
eigendecompositions in factor selection), missing evaluation metric definitions, an
ambiguity in the eigendecomposition matrix formula, and an incomplete treatment of how
the very first forecast day is bootstrapped.

---

## Major Issues

### M1. PCA Dual Case (P > N) Normalization Is Incorrect

**Location:** Function 3: extract_factors, lines 226-231.

**Issue:** In the dual case (P > N), the spec writes:

```
Lambda_hat = sqrt(N) * eigenvectors[:, :r]     # (N, r)
F_hat = X @ Lambda_hat / N                     # (P, r)
```

The spec claims the Bai (2003) normalization F_hat.T @ F_hat / P = I_r holds. But with
this formula, it does not. Let u_j be the unit eigenvectors of X.T @ X / P with
eigenvalues mu_j. Then Lambda_hat = sqrt(N) * u_j, and:

    F_hat = X @ (sqrt(N) * U) / N = X @ U / sqrt(N)

    F_hat.T @ F_hat / P = U.T @ X.T @ X @ U / (N * P)
                        = U.T @ (X.T @ X / P) @ U / N
                        = diag(mu_j) / N

This equals I_r only if mu_j = N for all j, which is not generally true.

**Impact:** While C_hat = F_hat @ Lambda_hat.T is invariant to normalization (it is always
the rank-r SVD approximation), incorrect normalization of F_hat will cause confusion and
may produce wrong results if any downstream code relies on the normalization (e.g., if
the developer checks F'F/P = I as a sanity test, it will fail).

**Recommendation:** Use the SVD relationship explicitly. For the dual case:
1. Eigendecompose (X.T @ X) / P to get eigenvalues mu_j and eigenvectors u_j.
2. Set Lambda_hat = u_j * sqrt(mu_j) (absorb the eigenvalue into loadings).
3. Compute F_hat = X @ Lambda_hat @ diag(1/mu_j) and then re-normalize so F'F/P = I_r.

Alternatively, simply state that for implementation, always compute the SVD of X and
extract F and Lambda from the left/right singular vectors with appropriate scaling. This
avoids the P < N vs P > N branching entirely and is more robust. The spec already
acknowledges (line 248) that C_hat is invariant to the rotation, so the normalization
convention matters less than the correctness of C_hat.

**Paper reference:** Bai (2003), Theorem 1 (convergence requires the specific
normalization). BDF 2008, Section 2.2, text after Eq. 6: "Concentrating out A and using
the normalization F'F/T = I_r..."

---

### M2. Redundant Eigendecompositions in select_num_factors

**Location:** Function 2: select_num_factors, lines 155-171.

**Issue:** The pseudocode calls `extract_factors(turnover_matrix, r)` inside a loop for
r = 1..r_max. Each call performs a separate eigendecomposition. This is wasteful: you need
only ONE eigendecomposition of X @ X.T / N (or its dual), producing all eigenvalues and
eigenvectors sorted by magnitude. Then for each candidate r, the residual variance V(r)
can be computed from the eigenvalues alone (no need to reconstruct the full factor
matrices and subtract):

    V(r) = (1/(N*P)) * (total_sum_of_squares - sum of top r eigenvalues * N * P / ...)

Or more precisely, compute F_hat and Lambda_hat once for r_max, and for each r < r_max,
V(r) can be computed from the projection using only the top r columns.

**Impact:** For r_max = 10, this performs 10 separate eigendecompositions instead of 1. For
large N or P, this is 10x slower than necessary. A developer unfamiliar with the
algorithm might implement it exactly as written and suffer poor performance.

**Recommendation:** Restructure the pseudocode:
1. Perform one eigendecomposition, extracting the top r_max eigenvalues and eigenvectors.
2. For each r from 1 to r_max, compute V(r) using the first r eigenvectors/eigenvalues.
3. Compute IC_r = ln(V(r)) + penalty(r) for each r.
4. Return argmin.

This is a standard optimization for information-criterion-based factor selection.

---

### M3. Missing Evaluation Metric Definitions

**Location:** Validation section (lines 806-905).

**Issue:** The spec presents validation targets (MSE, MAPE values from Szucs Table 2 and
BDF Table 2) but never defines how MSE and MAPE should be computed. Szucs 2017 provides
explicit formulas (Eqs. 1-2):

    MSE = (1/N) * sum_{t=1}^{N} (Y_t - Y_t^f)^2
    MAPE = (1/N) * sum_{t=1}^{N} |Y_t - Y_t^f| / Y_t

Critical ambiguities a developer would face:
1. What is the averaging unit? Is N the total number of bin-level forecasts (all stocks,
   all days, all bins)? Or is MSE/MAPE computed per-stock first and then averaged across
   stocks?
2. Szucs Table 2 reports the "average" across 33 stocks. So the procedure is: compute
   per-stock MSE/MAPE (averaging over all bins and days for that stock), then average
   across stocks.
3. MAPE is undefined when Y_t = 0 (division by zero). The spec notes this in Edge Case 1
   (lines 909-914) but only as a passing comment. Since Szucs filtered for non-zero volume
   stocks, this matters for replication.
4. The spec omits Szucs's MSE* metric (Eq. 14 in Szucs), which is a scale-adjusted MSE
   that normalizes by average turnover. BDF_AR is best by MSE, but BDF_SETAR is best by
   MAPE. Since these lead to different "best model" conclusions, the developer should
   know about both.

**Recommendation:** Add a Validation Metrics subsection before Expected Behavior that
defines MSE, MAPE, and their aggregation procedure. Mention MSE* as an optional
scale-adjusted variant. Specify that MAPE excludes zero-actual observations.

---

### M4. Ambiguity in Eigendecomposition Matrix

**Location:** Function 3: extract_factors, line 221.

**Issue:** The spec writes:

```
M = (X @ X.T) / N          # (P, P) matrix
```

But this is not the matrix whose eigenvectors give F_hat under the Bai (2003)
normalization. The correct derivation from BDF Eq. 6 is:

    Maximize tr(F' X X' F) / (N*T) subject to F'F/T = I_r

Substituting G = F/sqrt(T), this becomes maximize tr(G' (X X'/(N*T)) G) subject to
G'G = I. But equivalently, since the optimization is scale-invariant, the eigenvectors of
X X' / N or X X' / (N*P) or even just X X' are identical — only the eigenvalues change.

So the spec's M = XX'/N is correct for finding eigenvectors, but the eigenvalues of this
matrix are NOT the eigenvalues referenced in V(r). This matters for the IC computation
in select_num_factors: V(r) = (1/(NP)) * ||X - C_hat_r||_F^2. If the developer tries to
compute V(r) from the eigenvalues of M = XX'/N, they need to know the correct scaling.

**Recommendation:** Clarify the relationship between the eigenvalues of M and V(r).
Specifically, if mu_1 >= mu_2 >= ... are the eigenvalues of XX'/N, then:

    V(r) = (1/(NP)) * (||X||_F^2 - P * sum_{j=1}^{r} mu_j)

Or state that V(r) should be computed directly from the residual matrix e_hat = X - C_hat
rather than from eigenvalues, avoiding the scaling confusion entirely.

---

### M5. First Forecast Day Bootstrapping Is Underspecified

**Location:** Function 9: run_dynamic_execution, line 561; Initialization section, point 5
(line 773).

**Issue:** At market open on the first forecast day, last_specific is set to
model["specific"][-1, stock_idx] — the last specific component value from the estimation
window. But what about the first-ever forecast day when the model is first estimated?
The spec says "Requires at least L trading days of historical turnover data" (line 759)
but doesn't address the bootstrapping sequence:

1. On the first estimation day, the model is estimated on days 1..L.
2. We forecast day L+1. At market open, last_specific = e_hat at the last bin of day L.
3. But this e_hat was estimated using data that includes day L itself.

There is no look-ahead issue here (BDF Section 2.3 confirms this), but a developer might
worry about it. The spec should explicitly confirm that using the in-sample residual
from the last bin of the estimation window as the initial e_{t-1} for forecasting is
correct and intended. Additionally, the spec should clarify: does day L's actual observed
data contribute to both the PCA estimation AND the first dynamic execution forecast
initialization? (Yes, it does, and this is by design.)

**Recommendation:** Add a brief note to Initialization point 5 confirming that the
in-sample residual from the last observation of the estimation window is the correct
initialization, citing BDF 2008 Section 2.3 explicitly.

---

## Minor Issues

### m1. Internal Citation Inconsistency for "10% Reduction"

**Location:** Line 856 vs. line 1017.

**Issue:** In the Validation section (line 856), the spec cites "BDF 2008, Section 4.3.2"
for the claim that dynamic PCA-SETAR reduces tracking error by ~10%. In the Paper
References table (line 1017), the same claim is cited as "Section 4.3.3, last paragraph."
Only one of these can be correct. From the BDF paper structure: Section 4.3.2 covers
"Stock by stock out-sample results" and Section 4.3.3 covers "Portfolio in and
out-sample results." The ~10% and ~8 bp figures appear on BDF p.1721 in the
portfolio-level discussion, which is Section 4.3.3.

**Recommendation:** Change line 856 to cite "Section 4.3.3" for consistency with the
reference table.

---

### m2. Variance Formula Unspecified (Population vs. Sample)

**Location:** Functions 5 and 6: fit_ar1 (line 328) and fit_setar (lines 423-424).

**Issue:** The spec writes `sigma2 = var(residuals)` and `sigma2_1 = var(resid1)` without
specifying whether this is population variance (1/n) or sample variance (1/(n-1)).
For OLS residual variance, the standard unbiased estimator uses 1/(n-p) where p is the
number of estimated parameters (2 for AR(1), 2 per regime for SETAR). While this
distinction is minor for large P, it matters for correctness and for matching reported
results.

**Recommendation:** Specify `sigma2 = sum(residuals^2) / (n - 2)` for AR(1) (n
observations, 2 parameters) and similarly for SETAR regimes. Or at minimum, state which
convention to use.

---

### m3. SETAR Regime Labeling Convention

**Location:** Function 6: fit_setar, lines 370-371.

**Issue:** The spec labels regime 1 as "calm" (e_{t-1} <= tau) and regime 2 as "turbulent"
(e_{t-1} > tau). BDF 2008 Eq. 11 uses the same indicator convention (I(x) = 1 when
x <= tau). However, the regime interpretation depends on the sign of the specific
component: negative e values indicate below-average volume (could be "calm"), while
positive e values indicate above-average volume (could be "turbulent" or "high activity").
Whether tau is positive or negative determines which regime is actually "calm."

The labels "calm" and "turbulent" could mislead a developer into expecting specific
regime behaviors. BDF 2008 calls them "regime 1" and "regime 2" without interpretive
labels.

**Recommendation:** Use neutral labels ("regime 1: e_{t-1} <= tau" and "regime 2:
e_{t-1} > tau") or add a note that "calm" and "turbulent" are interpretive shortcuts
that depend on the sign of tau.

---

### m4. Szucs Table 2 Pairwise Comparison Numbers Need Context

**Location:** Validation, Sanity Checks 8 and 9 (lines 893-901).

**Issue:** The spec cites "Szucs 2017, Section 5: BDF beats U-method on 31/33 stocks by
MSE, 26-30/33 by MAPE" and "SETAR wins on MAPE for 26-30/33 stocks." Looking at Szucs
Table 2b (MSE pairwise), BDF_AR beats U on 33/0 and BDF_SETAR beats U on 33/0 — i.e.,
all 33 stocks, not 31. For MAPE (Table 2c), BDF_AR beats U on 32/1 and BDF_SETAR beats
U on 32/1.

The "31/33" figure appears in the Szucs summary text discussing BCG_3 vs U (BCG beats U
on 28/5 by MSE). The spec may be conflating different pairwise comparisons.

For SETAR vs AR: Table 2b shows BDF_SETAR beats BDF_AR on 6/27 by MSE (AR wins). Table
2c shows BDF_SETAR beats BDF_AR on 30/3 by MAPE (SETAR wins on 30 stocks). So "26-30"
is not precise — it should be 30/3 for the full sample.

**Recommendation:** Correct the pairwise comparison numbers to match Szucs Table 2:
- BDF_AR beats U-method: 33/0 by MSE, 32/1 by MAPE
- BDF_SETAR beats U-method: 33/0 by MSE, 32/1 by MAPE
- BDF_SETAR beats BDF_AR: 6/27 by MSE (AR is better), 30/3 by MAPE (SETAR is better)

---

### m5. Missing Fallback Criterion for SETAR vs AR(1)

**Location:** Function 6: fit_setar, lines 440-443.

**Issue:** The spec says "If the SETAR estimation fails...or the SETAR does not improve
over AR(1) by a meaningful margin, fall back to AR(1)." This is marked as researcher
inference, which is fine. But "meaningful margin" is undefined. The developer needs a
concrete criterion.

**Recommendation:** Define the fallback criterion explicitly. Options:
- Use a likelihood ratio test or F-test comparing SETAR SSR to AR(1) SSR, with a
  significance level (e.g., p < 0.05).
- Use a simple relative improvement threshold: fall back if
  (SSR_AR - SSR_SETAR) / SSR_AR < 0.01 (less than 1% improvement).
- Always use SETAR when estimation succeeds (no improvement test). This is simpler and
  consistent with BDF's approach where SETAR is the default.

State the chosen criterion and mark it as researcher inference.

---

### m6. No Guidance on Handling Stale or Missing TSO Data

**Location:** Data Preprocessing, step 3 (line 50-53).

**Issue:** The spec says turnover = volume / total_shares_outstanding, with TSO from an
external data provider. But TSO can change intraday (stock splits take effect at open,
secondary offerings settle mid-day) and may be stale or missing for some dates.

**Recommendation:** Add a note: "Use the most recent available TSO value for each trading
day. If TSO changes intraday (e.g., split), use the post-event TSO for all bins on that
day. If TSO is missing for a date, carry forward the last known value. Flag any stock-day
where TSO changes by more than 10% from the previous day for manual review." (Researcher
inference: practical data handling.)

---

### m7. Overnight Gap Treatment Needs Explicit Implementation Guidance

**Location:** Edge Case 7 (lines 941-946); fit_ar1 (lines 297-331).

**Issue:** The spec correctly notes that the AR/SETAR series is treated as contiguous across
day boundaries, meaning bin k of day d is followed by bin 1 of day d+1. The spec says
"This is consistent with how BDF and Szucs implement the model." However, the spec does
not address the practical implication: the overnight gap may induce autocorrelation
structure at lag k (every k-th observation is a day boundary). This pattern is visible in
BDF's Fig. 2 (ACF of specific component shows periodic spikes at multiples of 25).

For a developer, the question is: should we do anything about this (e.g., add a lag-k
seasonal AR term)? The answer per BDF is no — the simple AR(1)/SETAR is used despite the
periodic structure.

**Recommendation:** Add an explicit note: "The periodic autocorrelation at lag k (visible
in BDF Fig. 2) is a known artifact of treating overnight gaps as continuous. BDF and Szucs
do NOT address this — the simple AR(1)/SETAR captures only the lag-1 dependence. Do not
add seasonal AR terms unless extending the model beyond what the papers describe."

---

### m8. Dynamic Execution Does Not Update the Common Component Intraday

**Location:** Function 9: run_dynamic_execution (lines 535-592).

**Issue:** In the dynamic execution loop, when actual volume is observed for bin j, the spec
updates only the specific component (line 587-588):

```
actual_specific = actual_turnover - model["common_forecast"][j, stock_idx]
```

The common forecast is NOT updated intraday — it remains the pre-computed historical
average. This is correct per BDF (the common component is a stable seasonal shape that
doesn't need intraday updating). However, the spec doesn't explicitly state this design
choice or explain why.

A developer might wonder: "Should I also re-estimate the common component using today's
observed volumes?" The answer is no, and it would be helpful to state this explicitly.

**Recommendation:** Add a brief note after the dynamic execution pseudocode: "The common
component forecast is fixed for the entire day (computed the night before). Only the
specific component updates intraday via observed actuals. This is by design — the common
component captures the stable seasonal shape, which does not change within a day.
(BDF 2008, Section 2.3, paragraph below Eq. 9.)"

---

### m9. Szucs Number of Days Discrepancy in Summary Table

**Location:** Validation, Expected Behavior (line 811).

**Issue:** The spec says "2648 forecast days" for the Szucs results. Szucs Table 1 reports
"Number of days: 2668." Szucs Section 3 says "2648 different parameter estimations and
forecasts for 2648 days." The difference (2668 - 2648 = 20) is the initial estimation
window. The spec should clarify that 2648 is the number of out-of-sample forecast days,
not the total number of trading days in the dataset.

**Recommendation:** Change to "2648 out-of-sample forecast days (from 2668 total trading
days, minus 20-day initial estimation window)."

---

## Verification of Key Citations

I verified the following citations against the original papers:

| Spec Claim | Paper Source | Verified? | Notes |
|-----------|------------|-----------|-------|
| BDF Eq. 9: common forecast formula | BDF p.1712 | Yes | Formula matches |
| BDF Eq. 10: AR(1) is actually AR(1) not ARMA(1,1) | BDF p.1712 | Yes | Equation has no MA term despite text saying "ARMA(1,1)" |
| BDF Eq. 11: SETAR two-regime model | BDF p.1712 | Yes | Formula matches |
| BDF Eq. 8: combined forecast x = c + e | BDF p.1712 | Yes | Formula matches |
| BDF Table 2: portfolio MAPE values | BDF p.1716 | Yes | Values match |
| BDF Section 3.1: 25 bins, 20-min, Dec 24/31 excluded | BDF p.1712 | Yes | Confirmed |
| Szucs Eq. 3: U-method formula | Szucs p.6 | Yes | Formula matches |
| Szucs Eq. 5: AR(1) model | Szucs p.7 | Yes | Matches BDF's AR(1) |
| Szucs Eq. 6-7: SETAR model and indicator | Szucs p.7 | Yes | Matches BDF |
| Szucs Table 2a: MSE and MAPE values | Szucs p.12 | Yes | Values match exactly |
| Szucs Section 2: turnover = V/TSO | Szucs p.4 | Yes | Confirmed |
| Szucs Section 2: non-zero volume filter | Szucs p.4 | Yes | "every stock had trades..." |
| BDF uses 39 stocks | BDF p.1712-1713 | Yes | CAC40 minus partial days |
| SETAR wins 36/39 stocks (BDF) | BDF p.1714 | Yes | Confirmed in estimation results |
| BDF estimation window: 20 days | BDF p.1712 | Yes | L=20 in Eq. 9 |
| No centering before PCA (Researcher inference) | Bai 2003 model definition | Reasonable | Model has no intercept term |

---

## Overall Assessment

The spec is strong and mostly implementation-ready. The major issues are:
1. **M1 (PCA dual case)** — correctness risk, should be fixed before implementation.
2. **M2 (redundant eigendecompositions)** — performance issue, easy to fix.
3. **M3 (missing metric definitions)** — validation gap, needed for correctness checking.
4. **M4 (eigenvalue scaling)** — could cause subtle bugs in IC computation.
5. **M5 (bootstrapping)** — minor confusion risk but important for first-run correctness.

The minor issues are mostly about precision and developer guidance. None would prevent
a competent developer from implementing the model, but fixing them would reduce the risk
of misinterpretation.

I recommend one additional revision round to address M1-M4 (highest priority) and the
factual corrections in m1 and m4.
