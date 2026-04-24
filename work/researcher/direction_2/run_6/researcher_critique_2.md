# Critique of Implementation Specification: PCA Factor Decomposition (BDF) — Draft 2

## Summary

Draft 2 is a substantial improvement over draft 1. All 4 major and 8 minor
issues from critique 1 have been thoroughly addressed:

- The centering/factor-selection ordering bug (M1) is fixed: X is now centered
  before passing to select_num_factors.
- The Table 5 citation error (M2) is corrected to Table 2 third panel.
- The PCA computational approach (M3) now uses SVD with clear explanation of
  the N << P efficiency advantage.
- Centering (M4) is explicitly marked as Researcher inference with rationale.
- All 8 minor issues (OLS/MLE equivalence, trading days count, MAPE formula,
  multi-step chaining documentation, r_max justification, overnight
  discontinuity, benchmark labels, non-stationary AR handling) are resolved.

The draft is now close to implementation-ready. I identified 0 major issues
and 4 minor issues remaining.

---

## Major Issues

None.

---

## Minor Issues

### m1. SVD computed inside the loop in select_num_factors pseudocode (Phase 1a, line 173)

**Problem:** The pseudocode in `select_num_factors` places the SVD call
`U, S, Vt = svd(X_centered, full_matrices=False)` inside the `for
r_candidate` loop (line 173). The implementation note below the function
correctly states: "The SVD in `select_num_factors` only needs to be computed
once; the same decomposition can be reused for all r_candidate values."
However, a developer following the pseudocode literally would compute SVD
r_max times (up to 20 times), since the SVD call appears inside the loop body.

**Impact:** Low. The implementation note catches this, and any competent
developer would likely optimize it. But pseudocode should be directly
translatable to code, and this contradicts the spec's own stated best practice.

**Fix:** Move the SVD call above the for loop:

```
FUNCTION select_num_factors(X_centered, N, P):
    r_max = min(20, min(N, P) - 1)
    best_r = 1
    best_ic = infinity

    # Compute SVD once; reuse for all truncation levels
    U, S, Vt = svd(X_centered, full_matrices=False)

    for r_candidate in range(1, r_max + 1):
        F_r = U[:, :r_candidate] * sqrt(P)
        Lambda_r = Vt[:r_candidate, :].T * S[:r_candidate] / sqrt(P)
        K_r = F_r @ Lambda_r.T
        e_r = X_centered - K_r
        ...
```

Then remove the separate implementation note, since the pseudocode itself
would be correct.

### m2. U-method benchmark formula not provided (Validation section)

**Problem:** The validation section references the U-method (historical
time-of-day average) as the primary benchmark for comparison (items 4 and 5
in "Expected Behavior"), and sanity check 2 says "Compare c_forecast[j, i]
with the U-method benchmark." However, the spec never provides the U-method
formula explicitly. Szucs 2017, Eq. 3 defines it as:

    y_hat_{p+1} = (1/L) * sum_{l=1}^{L} y_{p+1-m*l}

where L is the window length and m is the number of intraday bins. This is
simply the average of raw turnover x_{i,t} at the same time-of-day across
the L prior days — no PCA decomposition involved.

**Impact:** A developer implementing the validation suite needs to compute the
U-method as a baseline. Without the formula, they must look it up in the
paper or guess that it is the simple historical time-of-day average of raw
turnover (not of the common component).

**Fix:** Add the U-method formula to the Validation section, either as a
separate subsection or as part of the benchmark descriptions:

```
U-method benchmark (Szucs 2017, Eq. 3; BDF 2008, Section 3.2 "classical
approach"):
    x_hat_U[bin_j, stock_i] = (1/L) * sum_{d=0}^{L-1} x[bin_j + d*k, stock_i]
This is the simple average of raw turnover at the same time-of-day bin
across the L prior days, with no PCA decomposition.
```

### m3. Evaluation protocol not explicitly stated (Validation section)

**Problem:** The validation benchmarks cite MAPE values from both BDF 2008 and
Szucs 2017, but the spec does not explicitly describe the evaluation protocol
used to compute these metrics. Specifically:

- Szucs 2017, Section 3 states: "While parameters are updated daily, the
  information base for the forecast is updated every 15 minutes. [...] This
  approach is often called one-step-ahead forecasting." This means each bin's
  forecast is conditioned on the *actual* observed turnover of the previous
  bin, not on a chained multi-step forecast.

- BDF 2008, Section 3.2 uses a similar rolling procedure for the in-sample
  evaluation (Table 2, first panel), and Section 4.3 uses dynamic updating for
  the out-of-sample VWAP evaluation.

The spec describes the dynamic execution procedure in detail (Phase 2 and
Phase 3), but the validation section does not explicitly state that the MAPE
benchmarks are computed using one-step-ahead (dynamic) forecasts. A developer
might compute MAPE on static (multi-step) forecasts and get very different
numbers.

**Impact:** Medium-low. The dynamic execution code is correct, and a careful
developer would use it for evaluation. But the validation section should be
self-contained regarding how to reproduce the benchmark numbers.

**Fix:** Add a note to the validation section:

```
Evaluation protocol: All MAPE and MSE benchmarks are computed using
one-step-ahead (dynamic) forecasts: at each bin, the forecast is
conditioned on the actual observed turnover of the immediately preceding
bin, not on chained multi-step predictions. This matches the "dynamic
adjustment" procedure described in BDF 2008 Section 4.2.2 and the
one-step-ahead protocol in Szucs 2017 Section 3.
```

### m4. Common component forecast indexing could be more explicit about day boundaries (Phase 1, Step 5)

**Problem:** The common component forecast loop (lines 131-134) uses:

```python
indices = [bin_j + d * k for d in range(L)]
c_forecast[bin_j, :] = mean(K_hat[indices, :], axis=0) + col_means
```

This gathers K_hat values at positions bin_j, bin_j + k, bin_j + 2k, ...,
which correspond to the same time-of-day bin across L estimation days. While
correct, the indexing implicitly assumes that the estimation window X is
organized as [day_0_bin_0, day_0_bin_1, ..., day_0_bin_{k-1}, day_1_bin_0,
...], i.e., days are concatenated in chronological order with bins sequential
within each day.

This data layout is established in Step 1 (`X = turnover_data[:,
start_bin:end_bin].T`) but is not explicitly documented as a layout
assumption for the indexing arithmetic in Step 5.

**Impact:** Low. The layout is implied by the construction in Step 1 and is
the natural ordering. But making it explicit would help a developer verify
correctness.

**Fix:** Add a brief comment to Step 5:

```
# X is organized as [day_0_bin_0, ..., day_0_bin_{k-1}, day_1_bin_0, ...],
# so bin_j of day d is at row index bin_j + d * k.
```

---

## Citation Verification Summary

| Spec Claim | Cited Source | Verified? | Notes |
|---|---|---|---|
| Decomposition x = c + e | BDF 2008, Sec 2.2, Eq. 4 | Yes | Matches |
| SVD-based PCA extraction | BDF 2008, Sec 2.2, Eq. 5-6; Bai 2003 | Yes | Paper references eigenvectors of X'X; SVD is equivalent |
| Factor count IC_p1 | BDF 2008, Sec 2.2; Bai & Ng 2002 | Yes | Matches |
| F_hat normalization F'F/P = I_r | BDF 2008, Sec 2.2: "F'F/T = I_r" | Yes | P in spec = T in paper |
| Common forecast = TOD average + col_means | BDF 2008, Sec 2.3, Eq. 9 | Yes | Eq. 9 averages common component at same TOD; adding col_means corrects for centering |
| AR(1) with intercept | BDF 2008 Eq. 10; Szucs Eq. 5 | Yes | Both confirmed |
| SETAR with indicator I(e <= tau) | BDF 2008 Eq. 11; Szucs Eqs. 6-7 | Yes | Both confirmed |
| OLS = conditional MLE under Gaussian errors | BDF 2008, Sec 2.3 | Yes | Paper says "maximum likelihood"; OLS equivalence correct |
| Dynamic VWAP execution | BDF 2008, Sec 4.2.2 | Yes | Matches |
| In-sample MAPE ~0.075 | BDF 2008, Table 2, first panel, PC-SETAR | Yes | Table 2 first panel confirmed |
| Out-of-sample VWAP MAPE ~0.090 | BDF 2008, Table 2, third panel, dynamic | Yes | Corrected from critique 1; now cites correct source |
| DJIA MAPE ~0.399 (SETAR) | Szucs 2017, Section 5 | Yes | Consistent with full-sample results |
| DJIA MAPE ~0.403 (AR) | Szucs 2017, Section 5 | Yes | Consistent with full-sample results |
| Sample: 2,668 days, 33 stocks | Szucs 2017, Table 1 | Yes | Corrected from critique 1 |
| MSE formula | Szucs 2017, Sec 3, Eq. 1 | Yes | Matches |
| MAPE formula | Szucs 2017, Sec 3, Eq. 2 | Yes | Matches |
| Centering before PCA | Researcher inference | Yes | Correctly labeled |
| ARMA(1,1) misnomer | Researcher inference | Yes | Correctly identified and justified |
| e_last init at market open | Researcher inference | Yes | Correctly labeled |
| SETAR tau grid [15th, 85th] | Researcher inference | Yes | Correctly labeled |
| r_max = 20 | Researcher inference | Yes | Correctly labeled with justification |
| Stationarity clamping | Researcher inference | Yes | Correctly labeled |
| Overnight discontinuity | Researcher inference | Yes | Correctly labeled as known limitation |

---

## Overall Assessment

Draft 2 is a high-quality implementation specification. All 12 issues from
critique 1 have been addressed thoroughly and correctly. The remaining 4
issues are all minor:

1. SVD inside the loop is a pseudocode inconsistency with the implementation
   note (m1) — easy fix.
2. U-method formula omission (m2) — a developer needs this for validation.
3. Evaluation protocol not stated (m3) — important for reproducing benchmarks
   but inferable from the execution code.
4. Data layout assumption implicit (m4) — cosmetic clarity improvement.

None of these would prevent a competent developer from implementing the model
correctly. The spec is ready for implementation with these minor polish items.
The algorithmic description is clear, citations are accurate, Researcher
inferences are properly labeled, and the validation section provides concrete
benchmarks with correct source attribution.
