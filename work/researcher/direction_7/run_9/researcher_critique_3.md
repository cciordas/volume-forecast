# Critique of Implementation Specification Draft 3: Kalman Filter State-Space Model

## Summary

Draft 3 addresses all 4 issues from critique 2 correctly and completely. The spec is now
at a high level of quality with no algorithmic errors, no medium or major issues, and
only 2 minor clarification opportunities. The document is implementation-ready: a
developer unfamiliar with the paper could implement the full model (standard and robust
Kalman filter, EM calibration, VWAP strategies) from this spec alone.

Issue count: 0 major, 0 medium, 2 minor.

---

## Verification of Critique 2 Fixes

| Critique 2 Issue | Fix Applied | Verified Correct? |
|---|---|---|
| ME1 (Medium): EM M-step not adapted for missing observations | Added O, N_obs, T_i bookkeeping (lines 117-125). LL sum restricted to O with N_obs normalization (lines 148-149). phi sum restricted per bin with T_i normalization (lines 177-178). r sum restricted to O with N_obs normalization (lines 182-185). Robust phi and r also adjusted (lines 289-301). Precondition note explaining paper assumes all nonzero (lines 120-125). | Yes -- all four sub-issues (r, phi, LL, robust variants) addressed. Sums, normalizations, and exclusion of undefined z_tau* for missing bins are all correct. |
| MI1: D upper bound | Changed to "k = 1, 2, ..., T-1" with parenthetical "T-1 elements, one per day-to-day transition" (lines 161-162). | Yes |
| MI2: Warm-start | Added in calibration step 5a (lines 452-457). Warm-start from previous day's converged parameters, expected 3-5x runtime improvement, marked as researcher inference. | Yes |
| MI3: Multi-step prediction covariance | Added recursive Sigma_{tau+k|tau} formula and corresponding S_{tau+h|tau} (lines 47-61). Noted role in confidence intervals, log-normal bias correction, and static-vs-dynamic MAPE gap. Marked as researcher inference. | Yes |

---

## Citation Verification (Draft 3 New Content)

| Spec Claim | Paper Source | Verified? |
|---|---|---|
| Missing-obs M-step adjustments | Researcher inference (paper assumes nonzero, Sec 2 p3) | Yes -- correctly marked; adjustments follow from standard EM restricted-sum logic |
| D = {tau=kI+1, k=1,...,T-1} has T-1 elements | Eq (A.34), sum over tau=kI+1 for k day boundaries; T days yield T-1 transitions | Yes |
| Multi-step covariance Sigma_{tau+k\|tau} | Standard Kalman result; paper gives only mean in Eq (9) | Yes -- correctly marked as researcher inference |
| Warm-start EM initialization | Researcher inference | Yes -- correctly marked |

---

## Minor Issues

### MI1. M-Step Sums That Do Not Require Missing-Obs Restriction Could Use a Clarifying Note

**Observation:** Algorithm 3 now clearly restricts phi, r, and LL sums to observed bins (O).
However, the sums for a^eta, a^mu, (sigma^eta)^2, and (sigma^mu)^2 (lines 163-174) run
over D or {2..N} without restriction to O. This is correct -- these sums use the smoothed
sufficient statistics P_tau, P_{tau,tau-1} which are defined for all tau (the smoother
always runs, regardless of whether a bin was observed, because the prediction step
propagates state at every tau). A developer might wonder why some M-step sums are
restricted and others are not.

**Suggestion:** Add a one-line note after line 174, e.g.: "Note: the a^eta, a^mu,
(sigma^eta)^2, and (sigma^mu)^2 updates use smoothed sufficient statistics (P_tau,
P_{tau,tau-1}), which are defined for all tau regardless of whether y_tau was observed.
Only sums involving raw observations y_tau (phi, r, and log-likelihood) require
restriction to O." This would make the asymmetry self-documenting.

**Impact:** Purely clarificatory. No correctness issue.

### MI2. "Changes from Draft" Sections Should Be Removed for Final Version

**Observation:** The spec retains "Changes from Draft 1" (lines 675-727) and "Changes from
Draft 2" (lines 729-756) sections. These are valuable for the review process but should
not appear in the final implementation specification delivered to the developer, as they
add ~80 lines of revision history that is irrelevant to implementation.

**Suggestion:** When copying to `artifacts/direction_7/impl_spec.md`, remove both "Changes
from Draft" sections.

**Impact:** Cosmetic. No correctness or implementability issue.

---

## Full Citation Audit

I re-verified every algorithmic step, parameter value, and benchmark number in the spec
against the paper source material. Results:

| Spec Section | Paper Reference | Accurate? |
|---|---|---|
| Model decomposition y = eta + phi + mu + v | Sec 2, Eqs (1)-(3) | Yes |
| State-space formulation (A_tau, Q_tau, C) | Sec 2, Eqs (4)-(5), p3 | Yes |
| Day-boundary switching (a_tau^eta, sigma_tau^eta) | Sec 2, p3 | Yes |
| Algorithm 1 (Kalman filter) | Sec 2.2, Algorithm 1, p4 | Yes (equivalent reordering) |
| Algorithm 2 (smoother + cross-covariance) | Sec 2.3.1, Algorithm 2, p5; Eqs (A.20)-(A.22) | Yes |
| Algorithm 3 (EM, all M-step updates) | Sec 2.3.2, Algorithm 3, pp5-6; Eqs (A.32)-(A.39) | Yes |
| phi-before-r ordering | Eq (A.38) uses phi^(j+1); Eq (A.39) independent of r | Yes |
| EM convergence from any init | Sec 2.3.3, Figure 4, p6 | Yes |
| Robust observation model y = Cx + phi + v + z | Sec 3.1, Eq (25), p6 | Yes |
| Robust correction: e_tau, W_tau, z_tau*, clamped residual | Sec 3.1, Eqs (29)-(34), pp6-7 | Yes |
| Robust EM: modified r update | Sec 3.2, Eq (35), p7 | Yes |
| Robust EM: modified phi update | Sec 3.2, Eq (36), p7 | Yes |
| MAPE formula | Sec 3.3, Eq (37), p7 | Yes |
| Dynamic MAPE: Robust 0.46, Std 0.47, CMEM 0.65 | Sec 4.2, Table 3, Average row | Yes |
| Static MAPE: Robust 0.61, Std 0.62, CMEM 0.90, RM 1.28 | Table 3, Average row | Yes |
| SPY dynamic 0.24, static 0.36 | Table 3, SPY row | Yes |
| 64% improvement over RM, 29% over CMEM | Sec 4.2, p8, text | Yes (0.46 vs 1.28 = 64%; 0.46 vs 0.65 = 29%) |
| VWAP definition Eq (39) | Sec 4.3, Eq (39), p10 | Yes |
| Static VWAP weights Eq (40) | Sec 4.3, Eq (40), p10 | Yes |
| Dynamic VWAP weights Eq (41) | Sec 4.3, Eq (41), p10 | Yes |
| VWAP tracking error Eq (42) | Sec 4.3, Eq (42), p10 | Yes |
| VWAP TE: Robust 6.38 bps, Std 6.39, CMEM 7.01, RM 7.48 | Sec 4.3, p11, text | Yes |
| 15-minute bins, exchange-dependent I | Sec 4.1, p8 | Yes |
| Half-day sessions excluded | Sec 4.1, p8 | Yes |
| Cross-validation period Jan-May 2015 | Sec 4.1, p8 | Yes |
| D = 250 out-of-sample trading days | Sec 4.1, p8 | Yes (not cited in spec but consistent) |
| 30 securities across U.S., European, Asian markets | Sec 4.1, Table 2 | Yes |
| All researcher inferences marked | Throughout | Yes -- 8 items properly attributed |

No citation errors, omissions, or misrepresentations found.

---

## Positive Assessment

The spec has reached convergence after three rounds. Specific strengths of the final draft:

1. **Algorithmic completeness.** All four core algorithms (filter, smoother, EM, robust
   filter) are expressed as unambiguous pseudocode with correct loop ordering, index
   conventions, and matrix dimensions. A developer can translate each block directly
   to code.

2. **Missing-observation handling is now consistent end-to-end.** The filter skips
   correction, the smoother propagates through, and the EM M-step restricts observation-
   dependent sums to O with appropriate normalizations. The precondition note (lines
   120-125) explains that under the paper's assumption this reduces to the standard
   formulas, making both the general and special cases clear.

3. **Multi-step prediction is now fully specified.** Both the mean and covariance
   recursions are given, along with the observation variance and its role in log-normal
   bias correction and the static-vs-dynamic performance gap.

4. **Traceability is excellent.** Every algorithmic step and parameter value cites a
   specific paper section, equation, and page number. All researcher inferences (8 items)
   are explicitly marked with reasoning. The paper references table provides a complete
   map.

5. **Validation section is actionable.** Specific MAPE and VWAP TE numbers with exact
   source citations, 8 concrete sanity checks, 7 edge cases with handling procedures,
   and 6 known limitations give the developer a comprehensive test plan.

6. **The robust filter cleanly extends the standard filter.** Setting lambda=0 recovers
   the standard filter exactly (sanity check 5), so a single codebase handles both
   variants.

## Recommendation

This spec is ready for finalization. The two minor issues are optional polish. Copy
draft 3 to `artifacts/direction_7/impl_spec.md` after removing the "Changes from Draft"
sections (lines 675-756).
