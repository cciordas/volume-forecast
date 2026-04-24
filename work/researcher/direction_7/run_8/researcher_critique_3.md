# Critique of Implementation Specification Draft 3: Kalman Filter State-Space Model for Intraday Volume

## Summary

Draft 3 resolves all 5 issues from critique 2 (1 major, 4 minor). The specification is now
implementation-ready. I found no major issues. I identified 2 minor polish items, neither
of which would cause an incorrect implementation. The spec is at final quality.

---

## Resolution of Critique 2 Issues

| Issue | Status | Notes |
|-------|--------|-------|
| M1 (Algorithm 1 line references shifted) | Resolved | Lines 249-256: added NOTE explaining the reordering. Citations now use equation numbers (Eq 7, 8, 9) instead of Algorithm 1 line numbers, eliminating the cross-referencing confusion. |
| m1 (day boundary convention ambiguity) | Resolved | Lines 211-216: added NOTE connecting the two conventions: "tau mod I == 0" (last bin of old day) is equivalent to "tau+1 in D" (first bin of new day). Explicitly states D marks the destination bin while A_tau/Q_tau use the source bin. |
| m2 (robust EM z_star data flow) | Resolved | Lines 432-433: added full "Robust EM data flow clarification" paragraph explicitly stating (a) robust filter stores z_star, (b) standard RTS smoother runs unchanged, (c) M-step uses both smoother outputs and stored z_star. |
| m3 (VWAP TE units conversion) | Resolved | Line 448: formula now reads "VWAP_TE (bps) = (10000/D) * sum...". Lines 454-455 add explicit note: "multiply by 10,000 to convert to basis points." |
| m4 (r positivity constraint) | Resolved | Lines 769: added to edge cases: "clip r to a small positive floor (e.g., 1e-10)" with note that this should not occur with reasonable data. |

All 5 issues from critique 2 are adequately addressed.

---

## Major Issues

None.

---

## Minor Issues

### m1. Sigma_1 Update Simplification Not Noted (Step 2, line 120)

The M-step computes:

    Sigma_1^(j+1) = P[1] - x_hat[1|N] * x_hat[1|N]^T              // Eq A.33

Since P[1] = Sigma[1|N] + x_hat[1|N] * x_hat[1|N]^T (line 112), this simplifies to:

    Sigma_1^(j+1) = Sigma[1|N]

This is mathematically obvious but could save a developer a few minutes of confusion when
they realize the outer product terms cancel. A one-line comment noting the simplification
would be helpful but is not required -- the formula as stated is correct and directly
implements Eq A.33.

**Recommendation:** Optionally add: "// Note: this simplifies to Sigma[1|N]." No action
required if the proposer prefers to keep the formula in its general form for traceability.

### m2. Jensen's Inequality Bias Correction Factor Location (Line 785-787)

The Jensen's inequality bias note in Known Limitations is well-placed and correctly
identifies the issue: E[exp(y)] > exp(E[y]). The suggested correction factor
exp(S[tau]/2) is mentioned but the spec does not clarify WHERE this S[tau] comes from
in the prediction context.

For static prediction, S[tau] at step h would be the observation forecast variance:
S[t*I+h] = C * Sigma[t*I+h | t*I] * C^T + r (from the multi-step covariance computation
at lines 305-311). For dynamic prediction (1-step-ahead), S[tau] is the innovation
variance computed during the filter.

This is a known limitation note, not a core algorithm step, so precision here is less
critical. A developer who wants to apply the correction would know to use the prediction
variance.

**Recommendation:** No action required. The note is sufficient as-is for a known
limitation. A developer implementing the correction would naturally use the appropriate
prediction variance.

---

## Citation Verification (Draft 3)

I verified all citations that changed between drafts 2 and 3:

| Spec Claim | Paper Source | Verified? |
|------------|-------------|-----------|
| Kalman filter equations now cite Eq 7, 8, 9 (lines 236-247) | Section 2.2, Eqs 7-8 (page 4), Eq 9 (page 4) | Yes |
| Day boundary note references D = {I+1, 2I+1, ...} (line 214) | Section 2, state-space definition, page 3-4 | Yes, consistent with A_tau definition |
| Robust EM clarification cites Algorithm 2 (line 432) | Algorithm 2, page 5 | Yes, smoother is unchanged in robust extension |
| VWAP TE with 10000 multiplier (line 448) | Section 4.3, Eq 42, Table 4 (page 10) | Yes, Table 4 values are in basis points |

All other citations from drafts 1-2 remain accurate and were previously verified.

I also spot-checked the following equations against the paper's Appendix A:

| Spec Equation | Paper Equation | Match? |
|---------------|----------------|--------|
| a_eta M-step (lines 129-130) | Eq A.34, page 15 | Yes |
| sigma_eta^2 M-step (lines 138-141) | Eq A.36, page 15 | Yes |
| r M-step (lines 151-155) | Eq A.38, page 15 | Yes |
| phi M-step (line 159) | Eq A.39, page 15 | Yes |
| Robust r update (lines 416-422) | Eq 35, page 7 | Yes |
| Robust phi update (line 426) | Eq 36, page 7 | Yes |
| Cross-covariance init (line 343) | Eq A.21, page 14 | Yes |
| Cross-covariance recursion (lines 346-347) | Eq A.20, page 14 | Yes |
| Soft-thresholding (lines 383-391) | Eq 33, page 7 | Yes |

No citation errors found.

---

## Overall Assessment

Draft 3 is a complete, correct, and implementation-ready specification. All 18 issues
raised across critiques 1 and 2 (6 major, 12 minor) have been resolved. The two minor
items identified above are purely cosmetic and do not affect correctness or
implementability.

The spec accurately represents the paper's algorithms, with all equations verified against
the source material. Researcher inferences (missing observation handling, warm-start,
Jensen's bias, dynamic VWAP denominator) are clearly marked and well-reasoned.

**Recommendation:** This draft is ready for finalization. No further revision rounds are
needed.
