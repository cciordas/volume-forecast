# Critique of Implementation Specification Draft 4: Dual-Mode Intraday Volume Forecast

## Summary

Draft 4 resolves all 4 issues from Critique 3 cleanly and correctly. The fixes
are well-documented, the revision notes are precise, and no new issues of
substance have been introduced. The spec is now at final quality.

The four resolutions:
- **M1 (volume_history slicing):** Properly fixed. Function 9 now computes
  extended_start = first_train_day - N_hist and passes the extended slice
  to optimize_regime_weights. Function 5's signature documentation explicitly
  states the data requirement. Both the grid search and re-optimization calls
  are handled correctly.
- **m1 (delta scaling):** Cleanly resolved. The formulation is now
  `adjusted = scale * (base_pct + clamped_delta)`, with the clamp operating
  on unscaled quantities before uniform scaling. This eliminates the
  (scale - 1) * delta error and maintains full domain consistency.
- **m2 (grid search re-optimization):** Correctly implemented. After regime
  count selection, the classifier is rebuilt and weights re-optimized on the
  full N_weight_train window. This follows standard model-selection practice.
- **m3 (no-intercept validity):** Well revised. The note now correctly states
  that training-time mean surprise is exactly zero by construction, with the
  warning applying only to production drift between re-estimations. Sanity
  Check 9 updated to match.

I found no major, medium, or minor issues that would warrant another revision
round. The spec is ready for finalization.

---

## Critique 3 Issue Resolution Assessment

All 4 issues from Critique 3 were addressed:

| Issue | Resolution Quality | Notes |
|-------|-------------------|-------|
| M1: volume_history slicing too narrow for rolling H_d | Excellent | Extended slice computation is correct. Both grid search call (lines 1068-1076) and re-optimization call (lines 1101-1110) use extended_start = first_train_day - N_hist. Function 5 signature documentation (lines 467-483) clearly notes the pre-context requirement. Edge Case 10 updated to reflect the extended data requirement (N_hist + N_weight_train = 84 days). |
| m1: Delta scaling mismatch | Excellent | Clean redesign: clamp on unscaled quantities, then scale uniformly. The comment block (lines 937-944) explains the rationale clearly — the regression predicts departures from unscaled hist_pct, so clamping and combining happen in unscaled space before scaling. |
| m2: Grid search weights not re-optimized | Excellent | Re-optimization step added at lines 1096-1111. Rebuilds both the classifier and weights on the full window. The extended_start computation for the re-optimization correctly uses full_train_start - N_hist. |
| m3: No-intercept validity overstated | Good | Revised note (lines 856-868) is accurate and well-structured. Clear distinction between training time (exact zero by construction) and production time (drift risk). One very minor wording suggestion: the phrase "This is a tautology, not something to verify" in Sanity Check 9 (line 1501) is slightly dismissive — it is a mathematical identity, but "tautology" has a pejorative connotation in some contexts. This is purely stylistic and does not affect implementability. |

---

## Citation Verification

I re-verified all citations that were added or modified in Draft 4:

| Claim | Cited Source | Verified? | Notes |
|-------|-------------|-----------|-------|
| Extended volume_history requirement (N_hist + N_weight_train = 84) | Edge Case 10 | Yes | Arithmetic correct: 21 + 63 = 84 |
| Grid search extended requirement (N_hist + N_weight_train - 21 = 63) | Edge Case 10, lines 1585-1586 | Yes | 21 + (63 - 21) = 63, correctly stated |
| Re-optimization as standard model-selection practice | Researcher inference (line 1171) | Yes | Correctly marked as Researcher inference |
| Scaled delta as Researcher inference | Lines 1712-1713 | Yes | Correctly added to Researcher inference list |
| Training-time mean surprise = 0 as tautological | Lines 856-861, 1714 | Yes | Mathematical derivation is correct |

No citation errors found. All new Researcher inference items are properly listed
in the inference manifest (lines 1690-1714).

---

## Comprehensive Quality Assessment

Having reviewed all four drafts across four critique rounds, I assess the spec's
final state across the key quality dimensions:

### Algorithmic Clarity

The spec contains 11 well-structured functions with clear pseudocode that can be
directly translated to code. Each function has explicit input/output
specifications, type annotations, and guard conditions. The ARMA model interface
(lines 216-247) provides concrete method signatures. The data flow diagram
(lines 1246-1290) accurately reflects the dependency structure. No gaps in the
algorithmic description remain.

### Parameter Documentation

All 19 parameters are documented with recommended values, sensitivity ratings,
ranges, and precise paper sources (or explicit Researcher inference markers).
The distinction between N_hist (21 days, Component 1) and N_seasonal (126 days,
deseasonalization) — which was conflated in Draft 1 — is now crystal clear
throughout the document.

### Domain Consistency

The percentage-space surprise reformulation (Draft 3) and the scaled delta fix
(Draft 4) together ensure complete domain consistency in Model B. All quantities
in the surprise pipeline — hist_pct, surprise, delta, scaled_base, adjusted —
operate in the same percentage-point space. The deviation clamp operates on
comparable magnitudes.

### Train/Predict Consistency

Functions 5 and 6 use identical regime assignment conventions (fixed in Draft 2).
Rolling H_d in training (Draft 3) eliminates lookahead bias. Separate validation
hist_avg (Draft 3) prevents validation data leakage. Re-optimization on full
window after grid search (Draft 4) follows standard practice.

### Completeness

- 14 sanity checks with specific thresholds
- 12 edge cases with handling procedures
- 9 known limitations
- Complete paper reference table with 30+ entries
- 25 Researcher inference items explicitly listed
- Comprehensive revision notes tracing the evolution across 4 drafts

### Implementability

A developer unfamiliar with the paper could implement this model correctly from
the spec alone. Every function signature is explicit, every edge case has a
handling procedure, and every design decision is justified with paper citations
or explicit Researcher inference reasoning.

---

## Recommendation

**The spec is ready for finalization.** No further critique rounds are needed.
The progression from Draft 1 (5 major + 8 minor issues) through Draft 4 (zero
issues) demonstrates thorough refinement. The remaining Researcher inference
items are well-reasoned and clearly marked, allowing a developer to evaluate
and potentially revise them based on empirical results.
