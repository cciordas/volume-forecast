# Critique of Implementation Specification: PCA Factor Decomposition (BDF) — Draft 1

## Summary

The draft is well-structured, thorough, and largely faithful to the source papers.
The pseudocode is concrete and would be translatable to working code by a competent
developer. Paper references are precise and verifiable. The proposer correctly
identifies the ARMA(1,1) mislabel in BDF Eq. (10) and provides detailed SETAR
estimation logic. Researcher inferences are clearly marked throughout.

I identify 5 major issues, 8 minor issues, and 3 suggestions. Most issues are
addressable with targeted revisions rather than structural changes.

---

## Major Issues

### M1. Demeaning/centering not addressed in pseudocode

**Section:** Pseudocode, Step 3 (extract_factors) and Step 2 (select_num_factors)

The pseudocode computes PCA via eigendecomposition of X'X/T without any mention of
whether the columns of X should be mean-centered beforehand. This is a critical
implementation detail because:

- Standard PCA libraries (numpy, sklearn) center columns by default.
- BDF 2008 Section 2.2 does not mention demeaning, and the paper references table
  (line 815) correctly notes "Demeaning not required" citing Bai (2003).
- However, this note appears only in the reference table at the end of the document,
  not in the pseudocode or algorithm description where a developer would look.
- Bai (2003) shows that PCA estimation is consistent without demeaning when factors
  capture the mean structure, but a developer using sklearn's PCA (which centers by
  default) would get different results than the eigendecomposition-of-X'X/T approach
  in the pseudocode.

**Action required:** Add an explicit statement in the pseudocode (before Step 2 or at
the start of extract_factors) that X should NOT be column-centered. The raw turnover
matrix is used directly. If a developer uses a library PCA implementation, they must
disable mean centering. This is the single most likely source of a silent
implementation bug.

### M2. One-step-ahead vs multi-step forecast validation mismatch

**Section:** Validation > Expected Behavior, and Pseudocode > Phase 2

The Szucs (2017) MAPE and MSE benchmarks (MAPE 0.399 for BDF-SETAR, MSE 6.49e-4 for
BDF-AR) are computed using **one-step-ahead** forecasts, where the actual value of the
previous bin is known before forecasting the next bin. Szucs Section 3 states: "the
information base for the forecast is updated every 15 minutes... This approach is
often called one-step-ahead forecasting."

However, the forecast_dynamic function (Phase 2, lines 373-397) also produces
**multi-step iterated** forecasts for all remaining bins by feeding point forecasts
forward as pseudo-observations (e_last = e_forecast). These multi-step forecasts are
used for VWAP weight computation. Multi-step iterated AR/SETAR forecasts will have
strictly worse accuracy than one-step-ahead forecasts because forecast errors
accumulate.

The spec does not clearly distinguish:
1. Which validation benchmarks apply to one-step-ahead forecasts (Szucs MAPE/MSE).
2. Which apply to the full VWAP execution strategy (BDF Table 2 tracking error).
3. That there are no benchmarks for multi-step iterated forecast accuracy per se.

**Action required:** Add a subsection to Validation clarifying that:
- Szucs MAPE/MSE numbers validate only the one-step-ahead forecast (bin j+1 given
  actual bin j).
- BDF Table 2 VWAP tracking error validates the full dynamic execution strategy
  (which uses one-step-ahead forecasts sequentially, NOT multi-step iteration).
- The multi-step iteration in forecast_dynamic (for computing VWAP weights of bins
  beyond j+1) is an approximation with no direct benchmark; its accuracy degrades
  with horizon.

### M3. Bin count discrepancy between BDF 2008 and spec parameters

**Section:** Parameters table, Pseudocode

The spec recommends k_bins = 26 (15-min bins, 9:30-16:00) throughout, citing Szucs
2017. However, BDF 2008 uses k = 25 (20-min bins, 9:20-17:20 on Euronext Paris).
The BDF validation results (Table 2, Tables 4-7) were all obtained with k=25, while
the Szucs validation results were obtained with k=26.

This matters because:
- A developer comparing their implementation output to BDF Table 2 numbers would
  need k=25 and 20-min bins on CAC40 data.
- A developer comparing to Szucs numbers would need k=26 and 15-min bins on DJIA
  data.
- The algorithm is parameterized by k, so any value works, but the validation
  targets are tied to specific k values and datasets.

**Action required:** In the Parameters table, note both values: k=25 (BDF 2008,
20-min, Euronext) and k=26 (Szucs 2017, 15-min, NYSE). In Validation > Expected
Behavior, annotate each benchmark with the corresponding k and dataset. The
recommended default for US equities should remain k=26.

### M4. forecast_dynamic conflates two distinct use cases

**Section:** Pseudocode > Phase 2 (forecast_dynamic)

The forecast_dynamic function serves two purposes that should be separated:

1. **One-step-ahead update:** After observing bin j, forecast bin j+1 only. This is
   what BDF Section 4.2.2 describes and what produces the validated accuracy numbers.
   The trader executes a fraction of the order in bin j+1, then observes the actual
   volume, then re-forecasts bin j+2, etc.

2. **VWAP weight computation:** To compute execution weights for ALL remaining bins
   (not just j+1), the function iterates the AR/SETAR forward using point forecasts
   as pseudo-observations. This is needed to decide what fraction of remaining order
   to place now vs. later.

The current function does both simultaneously without explaining the distinction.
A developer might think the iterated multi-step forecasts (bins j+2, j+3, ...) are
the primary output, when in practice the trader only acts on the weight for bin j+1
and then re-forecasts after observing the actual.

**Action required:** Either split into two functions (forecast_next_bin and
compute_vwap_weights_remaining) or add clear inline documentation explaining that:
- The one-step-ahead forecast (bin j+1) is the high-accuracy forecast.
- The multi-step forecasts (bins j+2 onward) are approximations used only for weight
  allocation and are replaced by fresh one-step-ahead forecasts as actuals arrive.

### M5. PCA dimension handling: T > N assumed but not enforced

**Section:** Pseudocode > extract_factors

The pseudocode computes Sigma = X'X/T of shape (N, N) and eigendecomposes it. This
is efficient when N < T (typical: N=30-50, T=520), but:

1. No guard or assertion checks that T > N (or at least T >= r).
2. If N > T (e.g., a universe of 500 stocks with a 10-day window giving T=260),
   X'X/T is rank-deficient with at most T non-zero eigenvalues. The algorithm still
   works (the top r eigenvectors are well-defined as long as r <= min(N, T)), but
   this should be stated explicitly.
3. The Bai & Ng IC penalty term uses min(N, T), which correctly handles both cases,
   but the factor extraction step doesn't discuss the N > T scenario.

**Action required:** Add a note in extract_factors that the algorithm works for any
N, T relationship as long as r <= min(N, T). Add an assertion: r <= min(N, T). For
very large N, mention that computing XX'/T (shape T x T) and extracting eigenvectors
of that matrix is more efficient, then recovering F_hat directly.

---

## Minor Issues

### m1. AR(1) residual degrees of freedom

**Section:** Pseudocode > fit_ar1, line 233

The formula sigma_eps = sqrt(sum(residuals^2) / (T - 1 - 2)) uses denominator
T - 1 - 2 = T - 3, representing T-1 observations minus 2 estimated parameters
(psi_1 and psi_2). This is correct for an unbiased variance estimator.

However, for SETAR (line 319), the denominator is T - 1 - 5 (5 parameters). But tau
is selected by grid search, not by OLS -- it is not a continuously estimated
parameter in the usual regression sense. The effective degrees of freedom for SETAR
is debated in the literature. Hansen (1997) discusses this issue.

This is unlikely to materially affect results (sigma_eps is informational, not used
in forecasting), but should be noted as an approximation.

### m2. Bai & Ng IC: which criterion variant?

**Section:** Pseudocode > select_num_factors, line 155

The spec implements IC_p2 with penalty = r * ((N + T) / (N * T)) * ln(min(N, T)).
Bai & Ng (2002) define three IC criteria (IC_p1, IC_p2, IC_p3) and three PC criteria.
BDF 2008 Section 2.2 cites Bai & Ng (2002) for factor selection but does not specify
which criterion is used.

IC_p2 is a reasonable default (it is the most commonly used in practice), but:
- The spec should acknowledge that BDF does not specify which IC variant they used.
- Consider implementing multiple criteria and selecting by majority vote or the most
  conservative (largest r), as a robustness check.

### m3. SETAR: no fallback when all tau candidates violate regime size

**Section:** Pseudocode > fit_setar, line 310

If ALL tau candidates violate the minimum regime size constraint (both regimes must
have >= min_obs observations), best_params will be None, and line 310 will crash.
This is unlikely but possible for very short windows or extreme distributions.

**Action required:** Add a fallback: if no valid tau is found, return the AR(1)
parameters as a degenerate SETAR (phi_11=psi_1, phi_12=psi_2, phi_21=psi_1,
phi_22=psi_2, tau=median(e_lag)).

### m4. Common component forecast: C_hat vs raw X averaging

**Section:** Pseudocode > Phase 1, Step 5 (lines 90-96)

The spec averages C_hat (the estimated common component) at each bin position across
L_days. BDF 2008 Eq. (9) averages c_{i,t+1-k*l}. These are equivalent since C_hat
already contains the estimated common component values at each time point.

However, the common component forecast for the next day uses the SAME C_hat that was
estimated from the current window. This means the forecast is not truly
out-of-sample within the estimation window -- the same data used to estimate factors
also provides the averaged common component. This is how BDF describes it, but it
should be explicitly noted as a design choice, not a bug.

### m5. Missing U-method benchmark implementation

**Section:** Validation > Sanity Checks

Sanity check 10 says "Compute MAPE for both the BDF dynamic forecast and the
U-method (time-of-day average)." But the U-method is not defined in the pseudocode.
The U-method is trivially defined (Szucs Eq. 3, BDF Eq. 9 with X instead of C): for
each stock and bin j, the forecast is the average of turnover at bin j over the prior
L days.

**Action required:** Add a brief pseudocode block for the U-method baseline, or at
minimum define it explicitly in the validation section so the developer can implement
the comparison.

### m6. Factor sign convention note is incomplete

**Section:** Initialization (lines 594-597)

The spec correctly notes that eigenvector signs are indeterminate and cancel in C = F
@ A. However, it omits that sign flips between daily re-estimations can cause the
INTERPRETABILITY of individual factors to change (e.g., factor 1 may flip from
"positive = high volume" to "negative = high volume"). Since the model uses C_hat
directly (not individual factors), this is not a functional issue, but it affects
debugging and visualization.

More importantly, sign flips do NOT affect the specific component E = X - C, so the
AR/SETAR parameters are also unaffected. This chain of reasoning should be made
explicit for a developer who might worry about sign consistency.

### m7. VWAP weight rebalancing procedure not fully specified

**Section:** Pseudocode > compute_vwap_weights

The function computes weights for remaining bins, but the spec doesn't fully specify
the execution loop:

1. At bin j, how much of the TOTAL order has already been executed? The spec
   implicitly assumes the remaining order is tracked externally.
2. The weight applies to the REMAINING quantity, not the original order. This is
   stated in the comments but should be more prominent.
3. What if observed actual volume in a bin is much lower than forecast? The trader
   may not be able to execute the planned fraction without exceeding participation
   rate limits. This practical constraint is not discussed.

### m8. Turnover definition needs explicit formula

**Section:** Data Flow (line 481)

The spec says turnover = shares_traded / total_shares_outstanding, with typical
values 1e-5 to 5e-2. This matches Szucs Section 2 (x_t = V_t / TSO_t).

However, BDF 2008 Section 3.1 says turnover = "traded shares / float shares" and
mentions adjustment for "stock's splits and dividends." Float shares is not the same
as total shares outstanding (float excludes insider and restricted shares). The spec
should explicitly state which denominator to use and note the difference. For US
equities, float-adjusted shares are available from standard data providers (CRSP,
Bloomberg).

---

## Suggestions

### S1. Add explicit data preprocessing section

The spec jumps from "turnover_data: dict" input directly to PCA. A short data
preprocessing section would help the developer handle real-world data:
- How to compute turnover from raw volume + shares outstanding data.
- How to handle stock splits, dividends, and corporate actions within the window.
- How to identify and exclude half-days.
- Whether to Winsorize or clip extreme values before PCA.
- How to handle missing data (stocks with trading halts during a bin).

### S2. Add computational complexity analysis

For a developer scaling to large universes (N=500+, L=60), it would help to note:
- PCA: O(N^2 * T) for X'X, O(N^3) for eigendecomposition. Or O(T^2 * N + T^3) if
  using XX' when N > T.
- AR/SETAR fitting: O(T * N) for AR, O(T * N * n_tau) for SETAR where n_tau is
  the number of grid candidates (~71).
- Total per day: dominated by PCA for large N.
- Reference: Szucs 2017 Section 4 notes BDF runs in ~2 hours for 33 stocks over
  2648 days (~2 hours total, not per day).

### S3. Consider documenting the relationship between BDF and U-method

The common component forecast (Step 5 in Phase 1) averages C_hat at each bin
position -- this is the U-method applied to the common component rather than raw
turnover. The full BDF forecast is thus: U-method(common) + AR/SETAR(specific).
Making this relationship explicit would help the developer understand what value
the decomposition adds over the naive baseline. The improvement comes entirely from
the specific component forecast; the common component forecast IS a U-method.

---

## Citation Verification Summary

| Spec Claim | Source | Verified? |
|---|---|---|
| BDF Table 2 panel 1: PCA-SETAR mean MAPE 0.0752 | BDF 2008, Table 2 | Yes |
| BDF Table 2 panel 1: PCA-ARMA mean MAPE 0.0829 | BDF 2008, Table 2 | Yes |
| BDF Table 2 panel 1: Classical mean MAPE 0.0905 | BDF 2008, Table 2 | Yes |
| BDF Table 2 panel 3: Dynamic PCA-SETAR 0.0898 | BDF 2008, Table 2 | Yes |
| BDF Table 2 panel 3: Dynamic PCA-ARMA 0.0922 | BDF 2008, Table 2 | Yes |
| BDF Table 2 panel 3: Classical 0.1006 | BDF 2008, Table 2 | Yes |
| CAP GEMINI: classical 0.2323, dynamic SETAR 0.1491 | BDF 2008, Tables 5-6 | Yes (Table 5 for dynamic, Table 6 for classical) |
| Szucs: BDF-SETAR MAPE 0.399, BDF-AR 0.403, U-method 0.503 | Szucs 2017, Section 4 | Yes |
| Szucs: BDF-AR MSE 6.49e-4, BDF-SETAR 6.60e-4, U-method 1.02e-3 | Szucs 2017, Section 4 | Yes |
| BDF Eq (10) labeled ARMA(1,1) but is AR(1) | BDF 2008, Eq. (10); Szucs 2017, Eq. (5) | Yes -- Szucs Eq. (5) writes e_p = c + theta_1 * e_{p-1} + eps_p, confirming AR(1) with intercept |
| SETAR outperforms AR for 36/39 stocks | BDF 2008, Section 3.2 | Yes -- paper states "only 3 of the 39 stocks" favor ARMA |
| k=25 bins, 20-min intervals | BDF 2008, Section 3.1 | Yes |
| k=26 bins, 15-min intervals | Szucs 2017, Section 2 | Yes |
| Bai & Ng IC for factor count | BDF 2008, Section 2.2 | Yes, cited but BDF does not specify which IC variant |
| Estimation window L=20 days | BDF 2008, Section 3.1; Szucs 2017, Section 3 | Yes |
| Rolling daily re-estimation | BDF 2008, Section 3.1; Szucs 2017, Section 3 | Yes |
| BDF describes SETAR threshold estimation by MLE | BDF 2008, Section 2.3 | Yes, but BDF says "maximum likelihood" generically |
| SETAR grid: 15th-85th percentile, 1% steps | Spec marks as researcher inference | Correct to mark as inference; not from papers |

---

## Issue Severity Assessment

| ID | Severity | Category |
|----|----------|----------|
| M1 | Major | Implementability -- silent bug risk |
| M2 | Major | Validation -- misleading benchmarks |
| M3 | Major | Correctness -- parameter/validation mismatch |
| M4 | Major | Clarity -- two use cases conflated |
| M5 | Medium | Completeness -- edge case not handled |
| m1 | Minor | Correctness -- approximation not noted |
| m2 | Minor | Completeness -- ambiguity in source paper |
| m3 | Minor | Implementability -- crash on edge case |
| m4 | Minor | Clarity -- design choice not flagged |
| m5 | Minor | Completeness -- missing baseline definition |
| m6 | Minor | Clarity -- incomplete reasoning chain |
| m7 | Minor | Completeness -- execution details omitted |
| m8 | Minor | Correctness -- float vs TSO ambiguity |
