# Critique of Implementation Specification: PCA Factor Decomposition (BDF) — Draft 2

## Summary

Draft 2 is a major improvement over draft 1. All 5 major issues, all 8 minor issues, and all
3 suggestions from critique 1 have been addressed, most of them thoroughly. The spec is now
1170 lines and covers preprocessing, algorithm, validation, edge cases, computational
complexity, and the U-method baseline. The pseudocode is concrete and directly translatable
to code. Citations are precise and researcher inferences are clearly marked throughout.

I identify 0 major issues, 4 minor issues, and 2 suggestions. The spec is ready for
implementation with these minor clarifications.

---

## Resolution of Critique 1 Issues

### Major Issues — All Resolved

| ID | Issue | Resolution | Quality |
|----|-------|-----------|---------|
| M1 | Demeaning not in pseudocode | Added explicit "Do NOT mean-center" comments in 3 locations (Step 1, extract_factors docstring, Initialization section). Clear and prominent. | Fully resolved |
| M2 | One-step-ahead vs multi-step validation mismatch | Validation section reorganized into three subsections (Szucs one-step-ahead, BDF portfolio, VWAP tracking error) with explicit annotations. Added "Note on Multi-Step Forecast Accuracy" subsection. | Fully resolved |
| M3 | k=25 vs k=26 not disambiguated | Parameters table now notes both values with source and dataset. Validation benchmarks annotated with corresponding k. | Fully resolved |
| M4 | forecast_dynamic conflates two use cases | Split into `forecast_next_bin` (one-step-ahead) and `compute_vwap_weights` (multi-step). Added `run_dynamic_execution` showing the full loop. Clear inline comments distinguish the two forecast types. | Fully resolved |
| M5 | PCA T > N not enforced | Added assertion `r <= min(N, T)`, dual computation paths (N <= T and N > T), and SVD note. | Fully resolved |

### Minor Issues — All Addressed

| ID | Issue | Resolution |
|----|-------|-----------|
| m1 | SETAR DOF approximation | Noted in SETAR code comment (lines 448-453). |
| m2 | IC variant unspecified | Acknowledged in select_num_factors comment (lines 243-247) and Paper References table. |
| m3 | SETAR fallback | Implemented with AR(1) degenerate fallback (lines 431-436). |
| m4 | C_hat averaging design choice | Noted in Phase 1 Step 5 comment (lines 176-179). |
| m5 | U-method not defined | Full pseudocode added as `forecast_u_method` function. |
| m6 | Factor sign chain incomplete | Full chain reasoning in Initialization section (lines 840-847). |
| m7 | VWAP rebalancing not specified | `run_dynamic_execution` loop clarifies execution flow. Volume participation limits added as Edge Case 9. |
| m8 | Turnover float vs TSO | Addressed in Data Preprocessing Step 1 with both definitions. |

### Suggestions — All Incorporated

| ID | Suggestion | Resolution |
|----|-----------|-----------|
| S1 | Data preprocessing section | Full `preprocess_turnover` function with 5 steps covering turnover computation, half-days, corporate actions, zero-volume bins, and optional Winsorization. |
| S2 | Computational complexity | New "Computational Complexity" section with detailed operation counts and Szucs timing benchmark. |
| S3 | BDF/U-method relationship | Explained in Phase 1 Step 5 comment (lines 173-176) and Paper References table entry for "Common forecast = U-method on C_hat." |

---

## Minor Issues

### m1. Szucs SETAR notation mismatch with spec

**Section:** Step 6b (fit_setar), and Szucs 2017 Eq. (6)

The spec parameterizes SETAR as:
```
e_t = (phi_11 * e_{t-1} + phi_12) * I(e_{t-1} <= tau)
    + (phi_21 * e_{t-1} + phi_22) * (1 - I(e_{t-1} <= tau))
```

Szucs 2017 Eq. (6) uses different notation:
```
e_p = (c_{1,1} + theta_{1,2} * e_{p-1}) * I(e_{p-1})
    + (c_{2,1} + theta_{2,2} * e_{p-1}) * (1 - I(e_{p-1}))
```

Note that in Szucs, the intercept comes FIRST (c_{1,1}) and the AR coefficient SECOND
(theta_{1,2}), whereas the spec puts the AR coefficient first (phi_11) and intercept second
(phi_12). This is functionally equivalent but creates a mapping confusion:
- Szucs c_{1,1} = spec phi_12 (intercept)
- Szucs theta_{1,2} = spec phi_11 (AR coefficient)

BDF 2008 Eq. (11) also puts intercept-like terms (phi_1) first and AR terms second.

**Action:** Add a brief note in the SETAR pseudocode or Paper References mapping:
"phi_11 corresponds to BDF's phi_1 (regime 1 AR coefficient) and Szucs's theta_{1,2};
phi_12 corresponds to BDF's phi_2 (regime 1 intercept) and Szucs's c_{1,1}."
This prevents a developer from accidentally swapping intercept and AR coefficient when
cross-checking against the papers.

### m2. select_num_factors recomputes PCA inside the loop

**Section:** Pseudocode > select_num_factors (lines 232-256)

The `select_num_factors` function computes the full eigendecomposition once (line 225),
but then for each candidate r, it reconstructs F_r, A_r, and C_r from scratch (lines 234-239)
to compute the residual variance. This is correct but wasteful.

Since the eigendecomposition is already done, the residual variance for r factors can be
computed directly from the eigenvalues:
```
sigma_sq_r = (sum(eigenvalues_all) - sum(eigenvalues_all[:r])) / (N * T)
```
This avoids O(r * N * T) matrix multiplications per candidate r.

However, this shortcut only works if F_hat'F_hat/T = I_r normalization holds exactly
(which it does by construction). The spec's approach is more pedagogically clear but
computationally wasteful (O(r_max * N * T) total vs O(N^2) for the eigenvalue shortcut).

**Action:** Not critical -- the current approach is correct. But consider adding a comment
noting the eigenvalue shortcut for production implementations: "Optimization: sigma_sq_r
can be computed as (trace(Sigma) - sum(lambda_1..r)) / N, avoiding matrix reconstructions."

### m3. run_dynamic_execution loops over all stocks sequentially

**Section:** Pseudocode > run_dynamic_execution (lines 524-574)

The execution loop iterates over stocks in the outer loop and bins in the inner loop:
```
for stock_id in stock_ids:
    for j in range(k_bins):
        ...
```

This means ALL bins for stock 1 are processed before any bin for stock 2. In reality,
the dynamic execution runs all stocks IN PARALLEL within each bin -- all stocks'
bin j weights are computed simultaneously, then all observe bin j actuals, then all
forecast bin j+1.

The correct loop structure should be:
```
for j in range(k_bins):
    for stock_id in stock_ids:
        ...
```

With the current (incorrect) ordering, stock 2's bin 0 weight would be computed AFTER
stock 1's bin 25 has already been observed, which is temporally impossible during live
execution.

**Action:** Swap the loop order. The bin loop should be outer, the stock loop inner. This
better reflects the temporal reality of live execution and ensures the pseudocode is
correct for a developer implementing a real-time system. Alternatively, note that the
function is meant for backtesting where temporal ordering within a day doesn't matter,
and add a comment about the correct loop structure for live execution.

### m4. Negative forecast edge case threshold

**Section:** Edge Cases, item 3 (line 1016)

The spec says: "If more than 50% of remaining bins produce negative raw forecasts, fall
back to common-component-only forecasts (set e_forecast = 0)."

This 50% threshold is reasonable but the fallback is only described for the "many
negatives" case. The single-negative case (floor at 1e-8) is handled in `forecast_next_bin`,
but the connection between these two mechanisms could be clearer.

Specifically:
1. `forecast_next_bin` floors individual forecasts at 1e-8 (line 516-517).
2. `compute_vwap_weights` has a fallback to uniform weights if total_remaining <= 0
   (lines 630-632), but this only triggers if ALL forecasts sum to zero or negative.
3. The 50% threshold from Edge Cases is described in prose but not implemented in any
   pseudocode function.

**Action:** Either add the 50% fallback check to `compute_vwap_weights` or note that the
1e-8 floor in `forecast_next_bin` combined with the total <= 0 fallback in
`compute_vwap_weights` already handles the practical cases, and the 50% threshold is an
additional safeguard for implementation. The current state has a gap between prose
(Edge Cases) and pseudocode (functions).

---

## Suggestions

### S1. Add example parameter values for a concrete stock

The spec provides general ranges and BDF/Szucs aggregate benchmarks but no example of
what fitted parameter values look like for a single stock. A developer would benefit from
seeing something like:

"For a typical liquid stock (e.g., a DJIA component), fitted parameters after 20-day
estimation might look like: r = 1-2 factors, psi_1 ~ 0.3-0.7, psi_2 ~ 0.001-0.005,
SETAR tau ~ 0 (near median of specific component), variance ratio Var(C)/Var(X) ~ 0.6-0.8."

This gives the developer concrete numbers to check against during debugging. The BDF
paper's ACF plots (Fig. 2, TOTAL stock) show the specific component has strong AR(1)
structure with PACF dropping at lag 1, suggesting psi_1 is substantial (likely 0.4-0.7).

This is not critical but would reduce debugging time.

### S2. Note on the Szucs "2648 forecast days" computation

The spec states "2648 forecast days" (Validation section, line 880). This is correct:
Szucs Table 1 reports 2,668 total trading days in the sample, minus the 20-day initial
estimation window = 2,648 forecastable days. However, since this arithmetic is not
shown in Szucs 2017 (the paper reports 2,668 days in Table 1 but does not explicitly
state "2,648 forecast days"), it would help to add a brief note:
"2648 = 2668 total days - 20 initial estimation days" to make the derivation transparent.

---

## Citation Verification Summary

All citations verified in critique 1 remain correct in draft 2. New citations added
in draft 2 were verified:

| Spec Claim | Source | Verified? |
|---|---|---|
| Demeaning not required, Bai (2003) | BDF 2008, Section 2.2 cites Bai (2003) | Yes -- BDF says "using the above model" with Bai's estimation, no demeaning step |
| Half-day exclusion: "24th and 31st of December 2003" | BDF 2008, Section 3.1 | Yes -- exact quote matches |
| BDF "traded shares / float shares" + split/dividend adjustment | BDF 2008, Section 3.1 | Yes |
| Szucs V_t / TSO_t with TSO from Bloomberg | Szucs 2017, Section 2 | Yes |
| Szucs "every stock had trades... in every 15-minute interval" | Szucs 2017, Section 2 | Yes |
| BDF footnote 5: "Replacing c by x in Eq. (9)" | BDF 2008, footnote 5 | Yes -- confirms U-method = classical prediction |
| Szucs U-method Eq. (3) | Szucs 2017, Section 4 | Yes -- y_hat = (1/L) sum y_{p+1-m*l} |
| Szucs SETAR Eq. (6) | Szucs 2017, Section 4.1 | Yes -- matches spec's SETAR formulation |
| Szucs one-step-ahead: "information base updated every 15 minutes" | Szucs 2017, Section 3 | Yes |
| Szucs timing: ~2 hours for BDF over full sample | Szucs 2017, Section 4 summary | Yes (from paper summary; not directly verified in PDF results page) |
| 2648 forecast days | Szucs 2017, Table 1 (2668 total - 20 estimation) | Correct derivation, see S2 |
| Large-N path XX'/T | Researcher inference | Correctly marked |
| SVD alternative | Researcher inference | Correctly marked |

---

## Issue Severity Assessment

| ID | Severity | Category |
|----|----------|----------|
| m1 | Minor | Clarity -- notation mapping between spec and papers |
| m2 | Minor | Efficiency -- correct but suboptimal computation |
| m3 | Minor | Correctness -- loop order wrong for live execution |
| m4 | Minor | Completeness -- prose/pseudocode gap for edge case |

---

## Overall Assessment

Draft 2 is a high-quality implementation specification. The pseudocode is concrete,
well-documented, and directly translatable to code. All major issues from critique 1
have been fully resolved. The remaining issues are minor clarifications that a
competent developer could work around, though fixing m3 (loop order) would prevent
a temporal ordering bug in any live execution implementation.

The spec is ready for implementation. A third revision round would yield diminishing
returns -- the issues identified here are refinements, not structural problems.
