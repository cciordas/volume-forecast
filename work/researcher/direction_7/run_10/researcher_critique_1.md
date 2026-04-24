# Critique of Implementation Specification Draft 1: Kalman Filter State-Space Model

**Direction:** 7 — Kalman Filter State-Space Model
**Run:** 10
**Role:** Critic
**Draft reviewed:** impl_spec_draft_1.md

## Summary

The draft is a strong, detailed specification covering the full Kalman filter /
EM pipeline from Chen, Feng, and Palomar (2016). The pseudocode for the standard
Kalman filter, smoother, and base EM algorithm is largely correct and well-cited.
However, there is one critical correctness error in the robust EM update for the
observation noise variance r (Step 5b), and several moderate gaps that would
impede a developer producing a correct implementation. I count 2 major issues,
5 moderate issues, and 3 minor issues.

---

## Major Issues

### M1. Robust r formula in Step 5b is incorrect (Correctness)

The robust EM update for the observation noise variance r (Step 5b, lines
328-334) contains multiple errors relative to Paper Eq 35 / Appendix Eq A.38
modified for z*.

The spec writes:

```
r^{(j+1)} = (1/N) * sum_{tau=1}^{N} {
    y[tau]^2 + C * P_tau * C^T - 2 * y[tau] * C * x_hat[tau|N]
    + (phi_tau^{(j+1)})^2 - 2 * (z_tau*)^2 * y[tau]
    + 2 * z_tau* * C * x_hat[tau|N]
    + 2 * z_tau* * phi_tau^{(j+1)}
}
```

Errors found:

1. **The term `- 2 * (z_tau*)^2 * y[tau]` is wrong.** This appears to multiply
   z-star-squared by y. The correct terms are `+ (z_tau*)^2 - 2 * z_tau* * y[tau]`
   (a standalone squared term plus a separate linear-in-z cross term with y). The
   notation `(z_tau*)^2` where `*` denotes "optimal" is ambiguous and compounds
   the problem.

2. **Two standard terms are missing from the robust formula.** The standard r
   update (Step 4, lines 257-261) includes `-2 * y[tau] * phi_tau^{(j+1)}` and
   `+ 2 * phi_tau^{(j+1)} * C * x_hat[tau|N]`. These terms must still be present
   in the robust version -- replacing y with (y - z*) in the observation equation
   does not remove the phi-y and phi-Cx cross terms; it modifies them. The robust
   formula should be the standard formula with additional z*-related terms.

The correct robust r formula (derived from the expected log-likelihood with
effective observation y - z*) is:

```
r^{(j+1)} = (1/N) * sum_{tau=1}^{N} {
    y[tau]^2 + C * P_tau * C^T - 2 * y[tau] * C * x_hat[tau|N]
    + (phi_tau^{(j+1)})^2
    - 2 * y[tau] * phi_tau^{(j+1)}
    + 2 * phi_tau^{(j+1)} * C * x_hat[tau|N]
    + (z_tau_star)^2
    - 2 * z_tau_star * y[tau]
    + 2 * z_tau_star * C * x_hat[tau|N]
    + 2 * z_tau_star * phi_tau^{(j+1)}
}
```

i.e., the standard formula PLUS four additional terms:
`+ (z*)^2 - 2*z**y + 2*z**Cx_hat + 2*z**phi`.

Evidence: This follows from E[(y - z* - phi - Cx)^2] = E[(y - phi - Cx)^2] -
2*z**(y - phi - Cx_hat) + (z*)^2, which adds (z*)^2 - 2z*y + 2z*phi + 2z*Cx_hat
to the standard formula. Paper Section 3.2, Eq 35 (page 7) should show these
terms, consistent with the derivation from the expected log-likelihood E_1 in
Appendix A.3.

**Severity: Critical.** This formula is used in every EM iteration of the robust
model. Getting it wrong means the robust model's parameter estimates will be
incorrect, likely producing poor forecasts.

### M2. Missing log-likelihood formula for EM convergence (Completeness)

The EM convergence check (Step 4, lines 269-272) requires computing the
log-likelihood at each iteration, but the spec provides no formula for it. A
developer cannot implement the convergence check without this.

The paper provides the joint log-likelihood in Appendix A.1, Eq A.8:

```
log L = -sum_{tau=1}^{N} (y_tau - phi_tau - C*x_hat[tau|tau-1])^2 / (2*S_tau)
        - (1/2) * sum_{tau=1}^{N} log(S_tau)
        - (N/2) * log(2*pi)
```

where S_tau = C * Sigma[tau|tau-1] * C^T + r is the innovation variance (already
computed in the filter). Alternatively, the simpler form using innovations:

```
log L = -(1/2) * sum_{tau=1}^{N} [e_tau^2 / S_tau + log(S_tau) + log(2*pi)]
```

where e_tau = y[tau] - C * x_hat[tau|tau-1] - phi_tau.

This is the prediction error decomposition of the log-likelihood, standard for
state-space models (Shumway and Stoffer, Section 6.3). Since e_tau and S_tau are
already computed during the E-step forward filter pass, computing the
log-likelihood adds negligible cost -- it just requires accumulating the sum
during the forward pass.

Reference: Paper Appendix A.1, Eq A.8 (the full form) and standard Kalman filter
theory for the prediction error decomposition form.

**Severity: Major.** Without this formula, the developer must either guess it, look
it up elsewhere, or use an ad hoc convergence criterion (e.g., parameter change),
which is less reliable.

---

## Moderate Issues

### Mo1. Log-to-linear bias correction buried in limitations (Algorithmic clarity)

The spec correctly identifies (Known Limitations #7, line 657-661) that
exponentiating log-volume forecasts produces median forecasts, not mean forecasts
(Jensen's inequality). However, this is only mentioned as a limitation, not
integrated into the prediction algorithm.

For VWAP execution, the bias matters: volume shares computed from median
forecasts will systematically underweight high-variance bins. The bias correction
is:

```
vol_hat_mean[tau] = exp(y_hat[tau] + 0.5 * V_tau)
```

where V_tau is the forecast variance (available from Sigma[tau|tau-1] as
C * Sigma[tau|tau-1] * C^T + r for one-step-ahead, or the corresponding
multi-step prediction variance).

This should be added as an explicit optional step in Step 6 (VWAP Execution
Strategies) and in the Data Flow section's "Exponentiate" step. The paper does
not discuss this correction (Researcher inference), but it follows directly from
log-normal theory and is standard practice.

### Mo2. No practical guidance for lambda cross-validation (Parameters)

The spec says lambda is "selected by cross-validation. Not reported in the
paper" (line 462) and provides no grid or heuristic range. This leaves the
developer with no starting point. A practical implementation needs:

- A suggested grid range. Since the threshold is lambda / (2 * W_tau) and W_tau
  is the innovation precision (roughly 1/S_tau where S_tau is on the order of the
  observation variance), a reasonable lambda range would be proportional to
  sqrt(r). For example: lambda in {0.1, 0.5, 1.0, 2.0, 5.0, 10.0} * sqrt(r_init).
- The relationship between lambda and the effective outlier threshold
  (lambda / (2*W_tau)) should be made explicit so the developer can reason about
  reasonable values. When lambda -> 0, every observation is treated as an outlier.
  When lambda -> infinity, no observations are treated as outliers.

This is Researcher inference, but the spec should provide it since the paper
omits it entirely.

### Mo3. Missing observation handling incomplete (Completeness)

Edge Case #1 (lines 587-594) describes zero-volume bins and mentions skipping the
correction step, but does not explain how this interacts with:

1. **The smoother**: If a bin was skipped in the forward pass, the smoother must
   also skip it. Specifically, at a missing observation tau, the smoothed state
   x_hat[tau|N] should equal x_hat[tau|tau] = x_hat[tau|tau-1] (since no
   correction occurred), and the smoother gain L_tau should be computed using
   Sigma[tau|tau] = Sigma[tau|tau-1].

2. **The EM sufficient statistics**: The observation noise update (r) should
   exclude missing bins from the sum (divide by number of observed bins, not N).
   The phi update should also adjust the denominator per bin (if bin i has
   T_observed < T days, divide by T_observed, not T).

3. **The cross-covariance initialization**: Sigma[N,N-1|N] uses K_N, which is
   undefined if bin N was missing.

Without these details, a developer who encounters zero-volume data will not know
how to handle it correctly within the EM framework.

### Mo4. Joseph form should be recommended for covariance update (Implementability)

The spec provides two forms for the corrected covariance Sigma[tau|tau] (lines
136-138):
- `Sigma[tau|tau] = Sigma[tau|tau-1] - K_tau * S_tau * K_tau^T`
- `Sigma[tau|tau] = (I_2 - K_tau * C) * Sigma[tau|tau-1]`

Both are algebraically equivalent but numerically fragile. The Joseph
(stabilized) form should be recommended as the primary implementation:

```
Sigma[tau|tau] = (I_2 - K_tau * C) * Sigma[tau|tau-1] * (I_2 - K_tau * C)^T
                 + K_tau * r * K_tau^T
```

This form guarantees symmetry and positive semi-definiteness even with finite
precision arithmetic. Given that the state dimension is only 2, the extra
computation is negligible. This is standard Kalman filter practice (see e.g.,
Grewal and Andrews, "Kalman Filtering: Theory and Practice", Section 6.2).

Researcher inference, but important for a robust implementation.

### Mo5. Transition matrix convention potentially confusing (Algorithmic clarity)

The spec defines A_tau_prev as "governing the transition FROM tau-1 TO tau" (line
113), while the paper uses A_tau to mean "transition FROM tau TO tau+1" (Eq 4:
x_{tau+1} = A_tau * x_tau + w_tau). This means the spec's "A_tau_prev" at time
tau corresponds to the paper's A_{tau-1}.

This is stated in the spec but the naming is confusing: "A_tau_prev" suggests
"the previous A matrix", not "the A matrix for the transition to tau". A
developer cross-referencing the paper will be confused.

Suggestion: Either (a) rename to A_to_tau or A_transition_tau to make the
semantics clear, or (b) use the paper's convention consistently and note that
A_tau governs x_{tau+1} = A_tau * x_tau.

---

## Minor Issues

### mi1. Notation for z* is ambiguous (Clarity)

Throughout Step 5 and Step 5b, the spec uses `z_tau*` where `*` denotes the
optimal value. In the robust r formula, `(z_tau*)^2` reads as "z-tau-star
squared" but the formatting makes it look like "z-tau times star-squared times
something." Use a consistent notation like `z_star_tau` or `z_hat_tau` to avoid
ambiguity.

### mi2. Robust smoother not discussed (Completeness)

Step 5 modifies the Kalman filter correction step for outlier handling, but the
spec does not discuss whether the smoother (Step 3) also needs modification when
the robust filter is used. Since the smoother uses the filtered estimates from
the forward pass, and the robust filter produces different filtered estimates
(via the cleaned innovation e_tau_clean), the smoother operates on the robust
filtered output automatically. However, the cross-covariance initialization
(Sigma[N,N-1|N]) uses K_N, which in the robust case still uses the same gain
(only the innovation is modified, not K). This should be stated explicitly to
prevent the developer from wondering whether to modify the smoother.

### mi3. EM iteration count not bounded (Implementability)

The convergence check is relative change in log-likelihood, with epsilon = 1e-6.
The spec should also recommend a maximum iteration count (e.g., 100) as a safety
bound, in case the log-likelihood oscillates due to numerical issues or very flat
regions of the likelihood surface. The paper's Figure 4 shows convergence in
about 5-10 iterations, so a bound of 50-100 is safe.

---

## Verification of Citations

| Spec Claim | Cited Source | Verified? |
|---|---|---|
| Decomposition y = eta + phi + mu + v | Paper Section 2, Eq 3 | Yes ✓ |
| State-space form Eqs 4-5 | Paper Section 2, Eqs 4-5 | Yes ✓ |
| Unified time index tau = (t-1)*I + i | Paper Section 2, below Eq 5 | Yes ✓ |
| A_tau time-varying at day boundaries | Paper Section 2, Eq 4 definition | Yes ✓ |
| Kalman filter pseudocode | Paper Algorithm 1, page 4 | Yes, re-indexed but correct ✓ |
| Multi-step prediction | Paper Eq 9 | Yes ✓ |
| Smoother pseudocode | Paper Algorithm 2, page 5 | Yes ✓ |
| Cross-covariance init Eq A.21 | Paper Appendix A.2, Eq A.21 | Yes ✓ |
| Cross-covariance recursion Eq A.20 | Paper Appendix A.2, Eq A.20 | Yes ✓ |
| EM M-step: pi_1 update Eq A.32 | Paper Appendix A.3, Eq A.32 | Yes ✓ |
| EM M-step: Sigma_1 update Eq A.33 | Paper Appendix A.3, Eq A.33 | Yes ✓ |
| EM M-step: a_eta update Eq A.34 | Paper Appendix A.3, Eq A.34 | Yes ✓ |
| EM M-step: a_mu update Eq A.35 | Paper Appendix A.3, Eq A.35 | Yes ✓ |
| EM M-step: sigma_eta^2 update Eq A.36 | Paper Appendix A.3, Eq A.36 | Yes ✓ |
| EM M-step: sigma_mu^2 update Eq A.37 | Paper Appendix A.3, Eq A.37 | Yes ✓ |
| EM M-step: r update Eq A.38 | Paper Appendix A.3, Eq A.38 | Yes ✓ (standard) |
| EM M-step: phi update Eq A.39 | Paper Appendix A.3, Eq A.39 | Yes ✓ |
| Robust filter Eqs 29-34 | Paper Section 3.1, Eqs 29-34 | Yes ✓ |
| Robust r update Eq 35 | Paper Section 3.2, Eq 35 | **No** — spec has errors (see M1) |
| Robust phi update Eq 36 | Paper Section 3.2, Eq 36 | Yes ✓ |
| VWAP formulas Eqs 39-41 | Paper Section 4.3, Eqs 39-41 | Yes ✓ |
| Average dynamic MAPE 0.46 | Paper Table 3, "Average" row | Yes ✓ |
| VWAP tracking error 6.38 bps | Paper Table 4, "Average" row | Yes ✓ |
| EM convergence insensitivity | Paper Section 2.3.3, Figure 4 | Yes ✓ |
| D = 250 out-of-sample days | Paper Section 4.1 | Yes ✓ |

---

## Summary of Required Changes

| ID | Severity | Section | Action |
|----|----------|---------|--------|
| M1 | Critical | Step 5b | Fix robust r formula: restore missing standard terms, correct z* terms |
| M2 | Major | Step 4 | Add log-likelihood formula (prediction error decomposition) |
| Mo1 | Moderate | Step 6, Data Flow | Add explicit bias correction step for log-to-linear conversion |
| Mo2 | Moderate | Parameters | Add practical lambda grid range and heuristic |
| Mo3 | Moderate | Edge Cases | Detail missing-observation handling for smoother and EM |
| Mo4 | Moderate | Step 2 | Recommend Joseph form for covariance update |
| Mo5 | Moderate | Step 2 | Clarify transition matrix naming convention |
| mi1 | Minor | Steps 5, 5b | Adopt unambiguous z-star notation |
| mi2 | Minor | Step 5 | Clarify that smoother needs no modification for robust case |
| mi3 | Minor | Step 4 | Add max iteration bound for EM |
