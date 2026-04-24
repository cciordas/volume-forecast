# Critique of Implementation Specification Draft 1: Kalman Filter State-Space Model for Intraday Volume

## Overall Assessment

This is a high-quality implementation specification. The pseudocode is detailed,
the M-step equations are correctly derived from the paper's Appendix A, and the
citation trail is thorough. A competent developer could implement approximately
90% of this model from the spec alone. The issues below are mostly about
closing gaps that would cause confusion or subtle bugs during implementation.

**Issues: 3 medium, 8 minor. No major issues.**

---

## Algorithmic Clarity

### MEDIUM-1: Cross-covariance pseudocode has an undefined variable (Step 3, lines 236-243)

The backward pass pseudocode (lines 213-233) computes the smoother gain `L` as
a local variable inside the loop but never stores it in an array. Then the
cross-covariance computation at line 243 references `L_stored[tau-1]`, which
was never defined.

The implementation note at line 257 says "Store L[tau] for tau = 1..N-1 during
the backward pass," but this instruction appears *after* the pseudocode block
and is easy to miss. A developer copying the pseudocode directly would get an
undefined variable error.

**Fix:** Inside the backward loop (lines 213-233), add an explicit storage line
after computing L:

```
L_stored[tau] = L    # store for cross-covariance computation
```

And declare `L_stored[1..N-1]` in the OUTPUT line of Step 3 (line 206).

### MEDIUM-2: Non-recursive cross-covariance formula needs justification (Step 3, lines 246-255)

The spec claims that `Sigma_cross[tau] = Sigma_smooth[tau] @ L[tau-1]^T` is a
non-recursive equivalent to the paper's recursive Equations A.20-A.21. This
claim is correct (I verified it by induction, substituting A.20 with the
inductive hypothesis and using the identity `L[tau] @ Sigma_pred[tau+1] =
Sigma_filt[tau] @ A_used[tau+1]^T`), but the spec does not provide a proof
sketch or a precise external reference.

The citation "Shumway and Stoffer (1982)" is vague -- there is no equation
number. A developer trying to verify the equivalence would need to work through
the induction themselves. More importantly, a developer who distrusts the claim
might fall back to implementing the paper's recursive form (Eq A.20-A.21),
which requires storing the Kalman gain K[N] for initialization (Eq A.21) -- a
dependency not documented anywhere in the spec.

**Fix:** Either (a) add a 3-4 line proof sketch of the induction step, or (b) at
minimum cite the specific result (e.g., De Jong 1988, or Shumway & Stoffer
textbook edition, Chapter 6, Property 6.2). Also add a note that if the
developer chooses the recursive form from Eq A.20-A.21 instead, K[N] must be
stored from the forward pass.

### MINOR-1: z_star storage not shown in pseudocode (Steps 8-9)

The robust Kalman filter (Step 8) computes `z_star[tau]` for each observed bin,
and the robust M-step (Step 9) reads `z_star[tau]` for all bins. But the
Kalman filter pseudocode in Step 8 only shows `z_star[tau]` being assigned
locally. For the EM loop to work, z_star must be stored as an N-length array
and passed from Step 8 to Step 9.

**Fix:** Add `z_star[1..N]` to the OUTPUT of Step 2/8. Initialize all entries to
0.0 (so unobserved bins have z_star = 0, consistent with Eq 33 where no
observation means no outlier).

### MINOR-2: Convergence check order and final parameter state (Step 7, lines 443-454)

The EM loop checks convergence after the E-step but before the M-step. If
convergence is detected, the loop breaks without running the final M-step. This
means:
- `theta_final` holds parameters from iteration j-1's M-step.
- `x_filt[N]` and `Sigma_filt[N]` are from iteration j's E-step (run with those same parameters).

These are consistent, so predictions will be correct. But a developer might
wonder whether the "last" M-step parameters or the E-step states should be used
for prediction. Add a brief note confirming that theta and the filtered states
from the last E-step are consistent, and that skipping the final M-step is
intentional (standard EM practice: the parameters that produced a converged
E-step are the final parameters).

### MINOR-3: First prediction step has no A_used entry (Step 2, lines 135-140)

At tau = 1, x_pred[1] = pi_1 and Sigma_pred[1] = Sigma_1 are set directly,
with no transition matrix involved. A_used is therefore undefined for index 1.
The smoother's backward loop runs from N-1 to 1, accessing A_used[tau+1] for
tau = N-1 down to 1. At tau = 1, it accesses A_used[2], which exists. So
there is no out-of-bounds access. But the spec should note that A_used is
indexed from 2 to N (not 1 to N) to prevent a developer from allocating the
wrong array size or being confused by the off-by-one.

---

## Ambiguities

### MEDIUM-3: Turnover vs raw volume in output conversion (Step 10, lines 569-572)

The preprocessing transforms raw volume to log-turnover:
`y = ln(volume / shares_out)`. The static prediction step converts back with
`volume_hat = exp(y_hat)`, which gives turnover (volume / shares_out), NOT raw
volume.

For VWAP weight computation, shares_out cancels in the ratio, so this is
correct. For MAPE computation, if the paper's MAPE formula (Eq 37) uses raw
volumes in both numerator and denominator, shares_out also cancels. So for
reproducing paper benchmarks, exp(y_hat) suffices.

However, a developer who wants to produce actual share volume forecasts
(e.g., for order sizing, market impact estimation, or reporting) would need to
multiply by shares_out[T+1]. The spec never states this.

**Fix:** After the `exp(y_hat)` conversion, add a note:

```
# volume_hat_static gives predicted turnover (volume / shares_out).
# For VWAP weights, this suffices (shares_out cancels in the ratio).
# For raw volume forecasts: raw_volume_hat[h] = volume_hat_static[h] * shares_out[T+1]
```

### MINOR-4: Phi identifiability is listed but never explained

The Paper References table (line 1022) lists "Phi identifiability -- Researcher
inference (implicit in model structure)" but nowhere in the spec is the
identifiability issue explained. The model y = eta + phi + mu + v has a
well-known identifiability problem: shifting phi by a constant c and adjusting
eta by -c produces identical observations. A developer encountering this
would wonder whether to add a constraint (e.g., sum(phi) = 0).

The EM algorithm resolves this implicitly: eta absorbs the mean level via its
AR(1) dynamics and initial state, while phi captures per-bin deviations
computed as residuals after removing C @ x_smooth. But this should be stated
explicitly so a developer doesn't add an unnecessary centering constraint that
could conflict with the EM updates.

**Fix:** Add a brief note to the Model Description or Initialization section
explaining that phi and eta are identified through the EM structure: eta
captures the day-level mean via AR(1) dynamics, and phi captures the per-bin
residual pattern. No explicit constraint (e.g., sum(phi) = 0) is needed or
desirable, as it would conflict with the closed-form M-step in Eq A.39.

### MINOR-5: Dynamic prediction multi-step-ahead not specified (Step 11)

Step 11 describes one-bin-ahead dynamic prediction, and Step 12's dynamic VWAP
says "Re-forecast bins i+1..I using updated state." But the spec never provides
pseudocode for multi-step-ahead prediction from an intermediate bin during
dynamic mode. This is needed at each bin of the dynamic VWAP strategy: after
correcting through bin i, the algorithm must forecast bins i+1 through I to
compute vol_remaining.

The logic is identical to Step 10 (static prediction) but starting from
x_filt at the current bin instead of at end-of-day. A developer would likely
figure this out, but given the spec's otherwise comprehensive pseudocode, this
gap is noticeable.

**Fix:** Add a brief "multi-step-ahead from mid-day" pseudocode snippet, or add
a note to Step 12 stating: "For bins i+1..I, apply Step 10's prediction loop
starting from x_filt_updated and Sigma_filt_updated at bin i, using within-day
transitions only (A = [[1,0],[0,a_mu]], Q = [[0,0],[0,sigma_mu_sq]])."

---

## Completeness

### MINOR-6: Missing data interaction with M-step denominators is underdocumented

The spec correctly uses `count` instead of T for phi (line 352) and `N_obs`
instead of N for r (line 364), while keeping T-1 and N-1 for a_eta, a_mu,
sigma_eta_sq, sigma_mu_sq. This is the correct EM-with-missing-data approach:
observation-equation parameters (phi, r) use only observed bins, while
state-dynamics parameters use all smoothed statistics (which the E-step provides
for all bins, including missing ones).

However, this reasoning is not stated anywhere. The spec only has inline
comments like "Eq A.39 uses T; use count for missing data." A developer might
question why phi uses count but a_eta uses T-1.

**Fix:** Add a brief note to the M-step (Step 5) explaining the principle: the
E-step provides smoothed state estimates for all bins (including missing ones),
so state-dynamics parameters sum over all bins. Observation-equation parameters
(phi, r) sum only over observed bins because the observation y[tau] is
undefined for missing bins.

### MINOR-7: Lambda cross-validation grid not specified in enough detail

The calibration section (lines 823-824) suggests "lambda_candidates =
log-spaced from 0.01 to 100 (e.g., 15-20 points)." The paper (Section 4.1)
says cross-validation is used but does not report the lambda grid or typical
optimal values. The spec's grid suggestion is reasonable but entirely
Researcher inference. This should be explicitly marked as such.

Also, the spec doesn't mention that lambda = infinity (standard KF) should be
included as a candidate in the grid, which would allow the CV procedure to
automatically select the standard filter if no outlier robustness is needed.

**Fix:** Mark the grid range as Researcher inference. Add lambda = infinity
(implemented as a large value like 1e10) as a grid candidate.

---

## Correctness

### Citation Verification Results

I verified every M-step equation against the paper's Appendix A:

| Spec Equation | Paper Source | Verification |
|---------------|-------------|--------------|
| pi_1 = x_smooth[1] | Eq A.32 | Correct |
| Sigma_1 = Sigma_smooth[1] | Eq A.33 (equivalent) | Correct -- algebraically equivalent to P_1 - x_hat_1 x_hat_1^T |
| a_eta summation | Eq A.34 | Correct -- D = {kI+1 : k=1..T-1}, indices match |
| a_mu summation | Eq A.35 | Correct -- tau=2..N, (2,2) elements |
| sigma_eta_sq | Eq A.36 | Correct -- uses updated a_eta, denominator T-1 |
| sigma_mu_sq | Eq A.37 | Correct -- uses updated a_mu, denominator N-1 |
| r compact form | Eq A.38 | Correct -- verified by expanding P_tau = Sigma + x_hat x_hat^T |
| phi | Eq A.39 / Eq 24 | Correct |
| Robust r | Eq 35 | Correct -- (y - phi - z* - Cx_hat)^2 + C Sigma C^T is equivalent |
| Robust phi | Eq 36 | Correct |
| Soft-thresholding | Eq 33 | Correct -- lambda/(2W) = lambda*S/2 since W = 1/S |

The Kalman filter day-boundary condition `(tau-1) mod I == 0` correctly
identifies transitions from the last bin of day k to the first bin of day k+1.

The reindexing convention (spec's x_pred[tau] = paper's x_{tau|tau-1}) is
correctly applied throughout.

Benchmark numbers from Tables 1, 3, and 4 are all correctly transcribed.

### MINOR-8: MAPE formula not provided

The spec references the paper's MAPE definition (Eq 37) and reproduces Table 3
values, but doesn't include the actual MAPE formula in the Validation section.
A developer implementing the evaluation would need to look up Eq 37 in the
paper. Since MAPE is the primary evaluation metric, its formula should be in
the spec.

Paper Eq 37: MAPE = (1/M) * sum_tau |volume_tau - predicted_volume_tau| / volume_tau

where M is the total number of bins in the out-of-sample period.

**Fix:** Add the MAPE formula to the Validation / Expected Behavior section.

---

## Implementability

The spec is highly implementable. The 2x2 state dimension keeps all matrix
operations trivial. The pseudocode maps directly to numpy/scipy operations.
The explicit Joseph form covariance update and 2x2 analytic inverse are
practical choices that avoid unnecessary library dependencies.

No issues found beyond those listed above.

---

## Summary of Recommended Changes

| ID | Severity | Section | Description |
|----|----------|---------|-------------|
| MEDIUM-1 | Medium | Step 3 | L_stored undefined in cross-covariance pseudocode |
| MEDIUM-2 | Medium | Step 3 | Non-recursive cross-covariance formula needs proof sketch |
| MEDIUM-3 | Medium | Step 10 | Turnover vs raw volume conversion not clarified |
| MINOR-1 | Minor | Steps 8-9 | z_star array storage not shown in pseudocode |
| MINOR-2 | Minor | Step 7 | Convergence check / final parameter state unclear |
| MINOR-3 | Minor | Step 2 | A_used indexing range not documented |
| MINOR-4 | Minor | Model Description | Phi identifiability unexplained |
| MINOR-5 | Minor | Steps 11-12 | Multi-step-ahead from mid-day not specified |
| MINOR-6 | Minor | Step 5 | Missing data M-step denominator logic underdocumented |
| MINOR-7 | Minor | Calibration | Lambda grid is Researcher inference; missing infinity candidate |
| MINOR-8 | Minor | Validation | MAPE formula not included |
