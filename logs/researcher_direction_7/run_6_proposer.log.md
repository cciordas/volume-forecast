## Status: done
## Current Step: Draft 4 complete (revision from critique 3)

### Log

[2026-04-10 11:56] Started proposer for direction 7, run 6.
- No prior drafts or critiques exist in run_6/. Producing initial draft.
- Read paper summary: papers/chen_feng_palomar_2016.md
- Read full PDF: papers/chen_feng_palomar_2016.pdf (all 16 pages)
- Read direction 7 description from work/analyst/research_directions_draft_2.md
- Read Brownlees et al. (2011) summary for context on the multiplicative decomposition origin.

Key insights from paper reading:
- State dimension is 2 (eta, mu); observation is scalar. All Kalman operations reduce to 2x2 matrices / scalar divisions.
- Transition matrix A_tau is time-varying: at day boundaries eta transitions with a^eta and process noise sigma_eta^2; within day eta is held constant (a=1, noise=0).
- EM algorithm has fully closed-form M-step updates (Appendix A, Eqs A.32-A.39).
- Smoother uses Rauch-Tung-Striebel backward recursion (Algorithm 2).
- Robust extension adds sparse noise z_tau with Lasso penalty; soft-thresholding solution (Eqs 33-34).
- Robust EM modifies r and phi updates to account for inferred z_tau* (Eqs 35-36).
- Seasonality phi estimated as simple mean residual per bin (Eq 24/A.39), not Fourier.
- Cross-validation needed for N (training window) and lambda (robust regularization).
- VWAP: static (Eq 40) and dynamic (Eq 41) strategies defined.

[2026-04-10 11:56] Beginning draft of impl_spec_draft_1.md.

[2026-04-10 12:02] Completed impl_spec_draft_1.md. Coverage:
- Full pseudocode for all 6 algorithms (Kalman filter, smoother, EM, robust filter, robust EM, VWAP strategies)
- All M-step closed-form updates with precise equation citations to Appendix A
- Data flow with shapes/types at each step
- 14 parameters documented with sensitivity and ranges
- Initialization and calibration procedures including cross-validation
- Quantitative validation benchmarks from Tables 1, 3, 4
- 7 sanity checks, 7 edge cases, 7 known limitations
- Full paper reference map (28 entries)

Key design decisions:
- Robust Kalman filter as primary implementation (subsumes standard as special case)
- Detailed cross-covariance recursion for smoother (Eqs A.20-A.22) which is the most implementation-critical detail
- Explicit treatment of day-boundary indexing (the D set) and summation ranges in M-step
- Flagged smoother interaction with robust filter as Researcher inference (paper is silent)

[2026-04-10 12:09] Received critique 1. Assessment:
- 5 major issues, 8 minor issues raised.
- M1 (cross-covariance pseudocode): MAJOR. Pseudocode is broken/self-contradictory. Will rewrite entirely with clean backward loop. Verified against Eqs A.20-A.21.
- M2 (robust r update sign error): MAJOR. First formula has wrong sign on z_star^2 term (-2 vs +1). Verified against Eq A.38 extended for robust case. Will remove wrong version, keep only correct one.
- M3 (lambda=0 behavior): MAJOR. Claim that lambda=0 means "no outliers" is backwards. lambda->0 means threshold->0, so z_star=e_tau always. Will correct.
- M4 (VWAP benchmarks): MAJOR. CMEM dynamic reported as 8.97 bps, but paper text + Table 4 confirm CMEM dynamic = 7.01 bps. The 8.97 is the std column for KF dynamic. Will correct to: CMEM dynamic=7.01, RM dynamic=7.48. Verified from Table 4 Average row and paper Section 4.3 text.
- M5 (missing log-likelihood formula): MAJOR. EM convergence check has no implementable formula. Will add innovations log-likelihood computed in forward pass.
- m1-m8: All valid minor issues. Will address all.

All critique points verified against paper. Proceeding with draft 2.

[2026-04-10 12:09] Beginning revision for impl_spec_draft_2.md.

[2026-04-10 12:17] Completed impl_spec_draft_2.md. All 13 issues addressed:
- M1: Rewrote cross-covariance section entirely. Clean backward loop with explicit L_gains storage, Eq A.21 initialization, Eq A.20 recursion, and index convention documented.
- M2: Removed incorrect r formula. Single correct version with explicit derivation from E[(y-Cx-phi-z)^2].
- M3: Corrected lambda limiting behavior. lambda->inf = standard KF; lambda->0 = all innovations absorbed as outlier noise.
- M4: Fixed CMEM dynamic VWAP from 8.97 to 7.01 bps. Added cross-check against paper text.
- M5: Added innovations log-likelihood formula to Algorithm 1; added convergence check to Algorithm 3.
- m1: Added L_gains storage in smoother loop.
- m2: Added MAPE formula (Eq 37) to Data Flow Step 7.
- m3: Added VWAP tracking error formula (Eq 42) and VWAP definition (Eq 39) to Data Flow Step 7.
- m4: Added Jensen's inequality bias note to Data Flow Step 5.
- m5: Rewrote dynamic_vwap_weights to produce multi-step forecasts from current filtered state.
- m6: Added explicit Sigma_cross[k] = Sigma_{k+1,k|N} index mapping throughout.
- m7: Clarified shares_outstanding as most recently reported value.
- m8: Added note about paper combining smoother and sufficient statistics in one pass.

Draft 2 also includes a Revision History section documenting all changes.

[2026-04-10 12:26] Received critique 2. Assessment:
- 0 major issues, 4 minor issues raised. Spec is close to implementation-ready.
- m1 (VWAP benchmark misattributions): RM 7.48 is STATIC, not dynamic; CMEM static = 7.71 IS reported. Verified against Table 4 layout. Will correct.
- m2 (Robust EM convergence undefined): Algorithm 5 doesn't specify how to compute log-likelihood for robust case. Will add cleaned-innovation LL formula + parameter-change fallback. Marked as Researcher inference.
- m3 (dynamic VWAP input semantics): x_filt_current ambiguous -- could be x_{i|i} or x_{i-1|i-1}. Will rename to x_filt_prev and clarify.
- m4 (redundant IF/ELSE): h==0 and h>0 branches identical. Will collapse to single loop body.

All 4 issues are minor and straightforward. Proceeding with draft 3.

[2026-04-10 12:26] Beginning revision for impl_spec_draft_3.md.

[2026-04-10 12:30] Completed impl_spec_draft_3.md. All 4 minor issues addressed:
- m1: Corrected VWAP benchmark table. CMEM static = 7.71 now included. RM labeled as static (not dynamic). Clarified cross-strategy nature of 15% improvement claim.
- m2: Added robust EM convergence monitoring section with cleaned-innovation LL formula and parameter-change fallback. Marked as Researcher inference.
- m3: Renamed x_filt_current -> x_filt_prev with explicit semantics (x_{(i-1)|(i-1)}). Removed unused Sigma_filt_prev parameter from signature.
- m4: Collapsed redundant IF/ELSE into single loop body with uniform within-day transition.

Draft 3 is a minor polish over draft 2. No algorithmic changes, only clarity improvements and benchmark corrections.

[2026-04-10 12:36] Received critique 3. Assessment:
- 0 major issues, 3 minor issues raised. Spec is implementation-ready per critic.
- m1 (day-boundary in dynamic_vwap_weights for i=1): Valid. When i=1, x_filt_prev is from last bin of previous day, so first prediction step crosses a day boundary. Must use A=[[a_eta,0],[0,a_mu]] instead of within-day matrix. Small numerical impact (a_eta ~ 1.0) but a correctness defect.
- m2 (RM MAPE "for both" phrasing): Valid. Table 3 reports RM only in static section. Will make consistent with VWAP section's treatment.
- m3 (Algorithm 1 RETURN missing innovations/S_values): Valid. Arrays computed but not returned. Needed for diagnostics and optional bias correction.

All 3 issues straightforward. Proceeding with draft 4.

[2026-04-10 12:36] Beginning revision for impl_spec_draft_4.md.

[2026-04-10 12:36] Completed impl_spec_draft_4.md. All 3 minor issues addressed:
- m1: Added conditional in dynamic_vwap_weights: when h==0 AND i==1, use day-boundary transition A=[[a_eta,0],[0,a_mu]]. Consistent with static_forecast and Algorithm 1.
- m2: Changed "achieves 1.28 for both" to "achieves 1.28 static (RM is inherently static; Paper, Table 3 reports it only in the static section)."
- m3: Added innovations and S_values to Algorithm 1 RETURN. Updated E-step call in Algorithm 3 to accept expanded return tuple.

Draft 4 is a minor polish over draft 3. Three small correctness/consistency fixes, no algorithmic changes.
