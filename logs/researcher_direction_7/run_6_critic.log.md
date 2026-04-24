## Status: done
## Current Step: Critique 3 delivered

### Log

[2026-04-10 12:03] Started critic for direction 7, run 6.
- Found impl_spec_draft_1.md (973 lines). No existing critique. Producing researcher_critique_1.md.
- Read full paper PDF (all 16 pages) to verify citations.
- Read paper summary (chen_feng_palomar_2016.md) for cross-checking.

Key verification findings from paper reading:
- Algorithm 1 (Kalman filter): Spec's predict-then-correct ordering inverted from paper's correct-then-predict, but logically equivalent. Verified.
- Algorithm 2 (Smoother): Cross-covariance recursion (Eq A.20-A.22) pseudocode is incomplete in the spec. Lines 304-306 have placeholder code that contradicts the "detailed formula" given later.
- Algorithm 3 (EM): M-step summation indices verified against Eqs A.32-A.39. Correct.
- Robust filter (Eqs 29-34): Soft-thresholding verified. Correct.
- Robust EM (Eqs 35-36): Found inconsistency in the r update formula between two versions in the spec. First version has -2*z^2 (wrong), second has +z^2 (correct per Eq 35).
- VWAP (Eqs 40-41): Formulas match paper.
- Table 3 (MAPE): Spec numbers verified correct.
- Table 4 (VWAP tracking): Spec cites CMEM dynamic = 8.97 bps. Paper text says 9% improvement over CMEM, which implies CMEM = 7.01 bps. 8.97 appears to be std, not mean.
- Lambda interpretation: Spec claims lambda=0 gives z_star=0. Paper Eq 33-34 show threshold = lambda/(2*W), so lambda=0 gives threshold=0, meaning z_star=e_tau (all outliers). Error in spec.

[2026-04-10 12:10] Completed researcher_critique_1.md.
- 5 major issues, 8 minor issues identified.
- Most critical: Algorithm 2 cross-covariance pseudocode is broken (M1).
- Other major: Robust r formula sign error (M2), lambda=0 behavior wrong (M3), VWAP benchmark numbers wrong (M4), missing log-likelihood formula (M5).
- All citations verified against paper; found 2 citation-level errors (M2 formula, M4 table reading).

[2026-04-10 12:19] Started critique 2 review of impl_spec_draft_2.md (1154 lines).
- Revision history (lines 1092-1154) confirms all 5 major and 8 minor issues addressed.
- Re-read paper Eqs A.20-A.21, Eq 35, Eqs 33-34, Table 4, Section 4.3 to verify fixes.

Verification results:
- M1 (cross-covariance): FIXED. Verified index substitution tau=k+1 in Eq A.20 is correct. Initialization matches Eq A.21.
- M2 (r sign error): FIXED. Single formula with correct +(z*)^2. Derivation transparent.
- M3 (lambda behavior): FIXED. Correctly describes both limits.
- M4 (VWAP benchmarks): PARTIALLY FIXED. RKF and CMEM dynamic numbers correct. But RM 7.48 labeled "dynamic" when it's static (Table 4 has no RM dynamic column). CMEM static = 7.71 available in Table 4 but spec says "not separately reported."
- M5 (log-likelihood): FIXED. Innovations form correctly integrated into Algorithm 1.

New issues found:
- m2: Robust EM convergence monitoring doesn't specify which innovations (raw vs cleaned) to use.
- m3: dynamic_vwap_weights input x_filt_current semantics ambiguous (bin i-1 vs bin i).
- m4: Redundant identical IF/ELSE branches in dynamic_vwap_weights loop.

[2026-04-10 12:19] Completed researcher_critique_2.md.
- 0 major issues, 4 minor issues identified.
- Most important: VWAP benchmark RM/CMEM static misattribution (m1).
- Spec is near implementation-ready. Remaining issues are documentation/clarity fixes.

[2026-04-10 12:31] Started critique 3 review of impl_spec_draft_3.md (1196 lines).
- Revision history (lines 1168-1196) confirms all 4 minor issues from critique 2 addressed.
- Re-read paper Table 3 (Average row), Table 4 (Average row), Eq 41, Section 4.3 to verify fixes.

Verification results:
- m1 (VWAP benchmark misattributions): FIXED. CMEM = 7.01 dynamic / 7.71 static. RM = 7.48 static with note that RM is inherently static. Cross-strategy comparison correctly described.
- m2 (robust EM convergence): FIXED. Cleaned-innovation LL specified, parameter-change fallback included, marked as Researcher inference.
- m3 (dynamic VWAP input semantics): FIXED. Renamed to x_filt_prev with explicit documentation of state identity.
- m4 (redundant IF/ELSE): FIXED. Single loop body, no branching.

New issues found:
- m1: dynamic_vwap_weights missing day-boundary transition when current_bin=1. At i=1, x_filt_prev is previous day's last filtered state; first prediction should use A=[[a_eta,0],[0,a_mu]] not [[1,0],[0,a_mu]].
- m2: MAPE benchmark says RM = 1.28 "for both" but Table 3 only reports RM in the static section.
- m3: Algorithm 1 RETURN doesn't include innovations/S_values arrays, needed for sanity check #4.

[2026-04-10 12:31] Completed researcher_critique_3.md.
- 0 major issues, 3 minor issues identified.
- Most important: day-boundary transition in dynamic_vwap_weights for i=1 (m1).
- Other: RM MAPE "for both" phrasing (m2), Algorithm 1 return signature (m3).
- Spec is implementation-ready. All algorithms directly translatable to code.
