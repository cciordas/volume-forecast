# Critique of Implementation Specification Draft 1: PCA Factor Decomposition (BDF)

## Summary

The draft is a strong, well-structured specification that covers the BDF model
comprehensively. The pseudocode is detailed enough to implement directly, the
data flow is clear, and the paper references table is thorough. The proposer
correctly identifies the AR(1)/ARMA(1,1) labeling discrepancy in BDF 2008 and
makes sensible engineering choices (SVD over eigendecomposition, dynamic-only
execution).

I identified 4 major issues (factual errors or gaps that could lead to incorrect
implementation) and 9 minor issues (imprecise citations, missing context, or
opportunities for improvement). The most critical issue is a factual error in
the SETAR notation claim about Szucs 2017.

---

## Major Issues

### M1. Incorrect claim about Szucs 2017 SETAR notation ordering

**Location:** Pseudocode Phase 3, `fit_setar` docstring (lines 217-219)

**Problem:** The spec states: "Szucs 2017 Eq. (6) uses the SAME ordering" as BDF
2008 for the SETAR coefficients. This is factually incorrect.

- BDF 2008 Eq. (11) writes the below-threshold regime as
  `phi_11 * e_{t-1} + phi_12` (AR coefficient first, intercept second).
- Szucs 2017 Eq. (6) writes the below-threshold regime as
  `c_{1,1} + theta_{1,2} * e_{p-1}` (intercept first, AR coefficient second).

The two papers use OPPOSITE ordering of the intercept and slope within each
regime. While this does not affect the mathematical model (addition is
commutative), a developer reading the spec alongside Szucs 2017 could be
confused if they try to cross-reference parameter names. The spec should
acknowledge the notation difference and map Szucs' parameter names to the
implementation's names explicitly.

**Evidence:** BDF 2008, p.1712, Eq. (11); Szucs 2017, p.7, Eq. (6).

**Fix:** Change the NOTE to state that the notation ordering DIFFERS between the
two papers, and provide a mapping: BDF's (phi_11, phi_12) corresponds to
Szucs' (theta_{1,2}, c_{1,1}) for regime 1.

### M2. Missing fallback when all SETAR threshold candidates fail

**Location:** Pseudocode Phase 3, `fit_setar` (lines 237-270)

**Problem:** The SETAR grid search skips candidates where either regime has fewer
than `min_regime_obs` observations. However, the spec does not handle the case
where ALL candidates fail this check (i.e., `best_tau` remains `None` after the
loop). This can happen for stocks with very short estimation windows or
concentrated residual distributions.

The code as written would crash when trying to unpack `best_params[0]` at
line 265 if `best_params` is still `None`.

**Fix:** Add an explicit fallback: if no valid threshold is found, return a
failure indicator and fall back to AR(1) in the calling code. This is
consistent with the spec's own Sanity Check 5 (line 584) which mentions
falling back to AR(1), but the pseudocode does not implement this guard.

### M3. Conflation of Szucs one-step-ahead evaluation with BDF VWAP execution

**Location:** Validation, Expected Behavior (lines 532-544)

**Problem:** The spec cites Szucs 2017 MAPE results (~0.40) alongside BDF 2008
VWAP tracking error results (~0.09) as though they measure the same thing.
They do not:

- Szucs 2017 MAPE values (0.399-0.503) measure **turnover prediction accuracy**
  using one-step-ahead forecasting with actuals fed back each bin. This is a
  forecast accuracy metric, not a VWAP execution metric.
- BDF 2008 MAPE values (0.0706-0.1006) measure **VWAP execution cost** — the
  percentage deviation between the trader's achieved price and the true VWAP.

These are fundamentally different metrics operating at different scales. The
spec correctly labels them in separate bullet points but does not explain the
difference clearly enough for a developer implementing validation tests. A
developer seeing both "~0.40 MAPE" and "~0.09 MAPE" for the same model might
think something is wrong.

**Fix:** Add a clear note at the start of Expected Behavior explaining that two
distinct metrics are used: (1) turnover forecast MAPE (Szucs, higher values
because it measures per-bin percentage error), and (2) VWAP execution cost
MAPE (BDF, lower values because it measures aggregate price deviation). These
cannot be directly compared.

### M4. Missing model selection procedure between AR(1) and SETAR

**Location:** Variants section (lines 455-469) and Phase 3

**Problem:** The spec says "SETAR model is preferred" and "AR(1) variant should
also be implemented for comparison and as a fallback." However, the spec
provides no formal model selection procedure. Questions a developer would face:

1. Should SETAR always be used unless it fails (the fallback approach)?
2. Should AR(1) vs SETAR be selected per-stock using an information criterion
   (e.g., BIC comparison)?
3. Should the choice be made once on the full sample and fixed, or re-evaluated
   daily?

BDF 2008 does not provide explicit model selection guidance — they report both
side by side. For a production implementation, the spec needs a clear rule.

**Fix:** Specify a default strategy, e.g.: "Use SETAR as the primary model for
each stock. Fall back to AR(1) if: (a) no valid SETAR threshold is found,
(b) either SETAR regime has fewer than min_regime_obs observations, or
(c) SETAR residual variance exceeds AR(1) residual variance (indicating
overfitting). Mark this as Researcher inference."

---

## Minor Issues

### m1. Section reference for "36 of 39 stocks" claim may be imprecise

**Location:** Variants section (line 462)

**Problem:** The spec states "BDF 2008 out-of-sample: SETAR outperforms ARMA for
36 of 39 stocks. Reference: BDF 2008, Section 4.3.2." The paper summary
states "ARMA beats SETAR for only 3 out of 39 stocks" but this appears to be
from the in-sample volume prediction analysis (Section 3.2), not the
out-of-sample VWAP execution (Section 4.3.2). For VWAP execution, BDF Table 6
shows 30 of 39 stocks benefit from dynamic SETAR vs. classical, which is a
different comparison.

**Fix:** Verify and correct: if the 36/39 figure is from Section 3.2 (in-sample
volume prediction), cite it as such. If it refers to the VWAP comparison,
the number should be 30/39 (dynamic vs. classical) per BDF Table 6.

### m2. Missing MSE validation benchmarks from Szucs 2017

**Location:** Validation, Expected Behavior (lines 532-544)

**Problem:** The spec only cites MAPE benchmarks from Szucs 2017. However, Szucs
Table 2a also provides MSE values: U-method 1.02E-03, BDF_AR 6.49E-04,
BDF_SETAR 6.60E-04, BCG_3 6.77E-04. Notably, BDF_AR beats BDF_SETAR on MSE
(despite SETAR winning on MAPE). This is relevant for validation because a
developer might use MSE as their primary metric and be surprised that SETAR
is not always best.

**Fix:** Add MSE benchmarks from Szucs 2017 Table 2a. Note that BDF_AR wins on
MSE while BDF_SETAR wins on MAPE.

### m3. Factor count "typically 1-3" lacks citation

**Location:** Pseudocode Phase 2 comment (line 128) and Parameters table (line 483)

**Problem:** The spec says r is "typically 1-3 for intraday volume data" but does
not cite a specific source. BDF 2008 does not report the IC_p2-selected r
values for their data. The summary paper mentions r is "estimated from data."

**Fix:** Either find the source for this claim or mark it as "Researcher inference
based on the low-dimensional structure of intraday volume patterns."

### m4. Turnover value range may be too broad

**Location:** Data Flow, Type details (line 444)

**Problem:** The spec says turnover values are "typically 1e-4 to 1e-1." BDF 2008
Table 1 shows the overall mean turnover across CAC40 stocks is 0.0116 with
Std 0.0146 and Q95 of 0.0380. This suggests the practical range is closer to
0.001 to 0.05 for most observations. The upper bound of 0.1 (10% of float
traded in a single 15-20 min bin) would be extremely unusual even for the
most liquid stocks.

**Fix:** Narrow the range to "typically 1e-3 to 5e-2" and cite BDF Table 1.

### m5. Estimation window terminology: L vs h

**Location:** Parameters table (lines 478-488) and throughout

**Problem:** BDF 2008 uses two potentially confusing window parameters: L = 20
(days for the common component average in Eq. (9)) and h (the historical
window for PCA estimation, described as "1 month" in the Key Parameters table
of the paper summary). The spec uses L for both purposes, implicitly assuming
they are the same. The paper text on p.1712 says "we chose a 20-day window
to construct the common component and L = 20 in Eq. (9)," suggesting L refers
to the common component averaging, while the PCA estimation window is
described separately.

In practice both are 20 days (yielding P = 20 * 25 = 500 observations for
PCA), but the spec should explicitly state this assumption.

**Fix:** Add a note: "In this implementation, the same L-day window is used for
both PCA estimation and common component averaging, following BDF 2008
Section 3.1."

### m6. BDF 2008 in-sample vs out-of-sample distinction for volume prediction

**Location:** Validation, Expected Behavior (lines 532-546)

**Problem:** The BDF 2008 volume prediction MAPE values (0.0752, 0.0829, 0.0905)
cited at the Variants section (line 458-460) are from the in-sample period
(September 2 to December 16, 2003, Section 3.2). The spec does not explicitly
label these as in-sample. The out-of-sample volume prediction MAPE from BDF
is not separately reported (BDF's out-of-sample analysis focuses on VWAP
execution cost, not raw volume prediction accuracy).

**Fix:** Label the BDF MAPE values as "in-sample volume prediction" and note
that BDF's out-of-sample analysis evaluates VWAP execution cost, not raw
volume prediction accuracy.

### m7. Szucs 2017 uses 36 tickers initially, 33 after filtering

**Location:** Parameters table, N description (line 482)

**Problem:** The spec says N = "30+" and cites BDF (39 stocks) and Bai (2003).
The Szucs 2017 data section clarifies: the DJIA database contains 36 tickers,
but 3 were excluded for short history, leaving N = 33. The spec should
reference this as an additional data point for the recommended N range.

**Fix:** Minor — add Szucs 2017 (N=33) alongside BDF (N=39) as examples.

### m8. Dynamic execution: incomplete handling of already-traded quantity

**Location:** Pseudocode Phase 4 (lines 337-368)

**Problem:** The proportions computation at Step 4.2 divides each forecast by
the total full-day forecast to get initial proportions. The subsequent
intraday revision at Step 4.3 recomputes proportions for remaining bins.
However, the spec does not clearly explain how `remaining_quantity` is tracked.

After bin j_obs, the trader has already traded some shares. The remaining
quantity to trade should be: `total_order_quantity - shares_already_traded`.
The spec says "Execute: trade (revised_proportions[j_obs+1, i] *
remaining_quantity[i]) shares in the next bin" but never defines how
`remaining_quantity` is initialized or updated.

**Fix:** Add a variable tracking: `remaining_quantity[i]` is initialized to the
total order size for stock i, and decremented after each bin's trade. The
revised proportions should be applied to this remaining quantity, not to the
original total.

### m9. Missing: explicit definition of MAPE used for validation

**Location:** Validation, Expected Behavior (lines 532-544)

**Problem:** The spec references MAPE values from both BDF and Szucs but does not
define MAPE. Szucs 2017 Eq. (2) defines it as
`MAPE = (1/N) * sum |Y_t - Y_t^f| / Y_t`. BDF 2008 uses a similar definition
but for VWAP execution cost (percentage deviation of achieved vs. true VWAP).
The developer should know the exact formula to implement for validation
comparison.

**Fix:** Add the MAPE formula in the Validation section, specifying it matches
Szucs Eq. (2). Note that zero-actual bins must be excluded (already mentioned
in Edge Cases but should be cross-referenced).

---

## Algorithmic Clarity Assessment

The pseudocode is generally well-structured and translatable to code. The four
phases are logically ordered, the data shapes are documented, and the
top-level daily pipeline ties everything together. Specific strengths:

- The SVD-based extraction is clearly specified with shapes at each step.
- The normalization identity F_hat.T @ F_hat / P = I_r is derivable from the
  pseudocode (verified: sqrt(P)*U.T @ sqrt(P)*U / P = I_r since U has
  orthonormal columns).
- The Lambda_hat formula correctly reconstructs X: F_hat @ Lambda_hat.T =
  sqrt(P)*U * (s*V/sqrt(P)).T = U*diag(s)*Vt = X (rank-r approximation).
- The "no demeaning" instruction is prominently placed and well-justified.

## Completeness Assessment

The spec covers the core algorithm thoroughly. The main gaps are:
- No formal model selection between AR(1) and SETAR (M4).
- Missing SETAR fallback guard in pseudocode (M2).
- Incomplete remaining-quantity tracking in VWAP execution (m8).
- Missing MSE validation benchmarks (m2).

## Correctness Assessment

Citations are generally accurate and well-traced. The main factual error is the
Szucs SETAR notation ordering claim (M1). The section reference for the 36/39
stocks claim (m1) should be verified. All other algorithmic steps correctly
reflect the source papers.

## Implementability Assessment

A developer could implement the model from this spec with moderate effort.
The main risk areas for misimplementation are:
- The turnover/VWAP MAPE confusion (M3) could lead to incorrect validation
  thresholds.
- The missing SETAR fallback (M2) could cause runtime crashes.
- The model selection gap (M4) would force the developer to make an
  undocumented design decision.
