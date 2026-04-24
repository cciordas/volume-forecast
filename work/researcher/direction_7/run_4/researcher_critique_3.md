# Critique of Implementation Specification Draft 3: Kalman Filter State-Space Model

## Summary Assessment

Draft 3 is excellent. All 1 major issue and all 5 minor issues from critique 2 have been
thoroughly and accurately addressed. The RM VWAP tracking error is now correctly reported
as 7.48 bps (mean, not std), the VWAP tracking error formula (Eq 42) and VWAP definition
(Eq 39) are included, all three M-step ordering constraints are documented with paper
citations, the static-to-dynamic evaluation mode transition is clearly described, the
dynamic VWAP multi-step prediction loop has proper pseudocode, and static VWAP tracking
error results are included with appropriate RM context.

I verified all benchmark numbers against Tables 3 and 4: every MAPE and VWAP tracking
error value is correct. I verified all M-step equations (A.32-A.39) against the paper's
Appendix A: formulas, summation ranges, and ordering constraints are all accurate. The
cross-covariance recursion (A.20-A.22), soft-thresholding (Eqs 33-34), robust M-step
(Eqs 35-36), and VWAP strategies (Eqs 39-42) all match the paper.

I identify **0 major issues** and **2 minor issues** remaining. Neither would cause an
incorrect implementation by a careful developer, but both represent small gaps in
algorithmic precision that are worth documenting.

---

## Major Issues

None.

---

## Minor Issues

### m1. Kalman Filter Initialization at tau=1 Applies Spurious Day-Boundary Transition

**Location:** Algorithm 1 pseudocode, lines ~129-134 (day-boundary detection and A/Q
construction).

The predict step at tau=1 uses the condition "tau mod I == 1" (line 129) to detect a
day boundary. Since tau=1 satisfies this condition, the filter constructs the
day-boundary transition matrix A = [[a^eta, 0], [0, a^mu]] and noise Q =
[[sigma_eta_sq, 0], [0, sigma_mu_sq]], then computes:

    x_pred[1] = A @ x_hat[0] = [[a^eta, 0], [0, a^mu]] @ pi_1

This applies the AR dynamics to the initial state prior, producing x_pred[1] =
[a^eta * pi_1[1], a^mu * pi_1[2]] instead of x_pred[1] = pi_1.

The paper's Algorithm 1 (page 4) does not apply any transition to obtain x_{1|0}.
The initialization is x_{1|0} = pi_1, Sigma_{1|0} = Sigma_1 (the prior for the first
state), and the first correction step refines this into x_{1|1} using y_1. The paper
states "x_1 is the initial state, and it is assumed to follow N(pi_1, Sigma_1)
distribution" (page 4, below Eq 5) -- this defines pi_1 as E[x_1] directly, not as a
state at "time 0" that must be transitioned forward.

The day-boundary condition correctly triggers at tau = I+1, 2I+1, 3I+1, ... (transitions
from the last bin of one day to the first bin of the next). It should NOT trigger at
tau=1, where the "transition" is from the initial prior to the first observation.

The EM M-step confirms this: Eq A.32 sets pi_1 = x_smooth[1] (the smoothed state at
time 1, not a pre-transition state). If the filter applied A @ pi_1 at tau=1, then after
EM convergence, x_pred[1] = A @ x_smooth[1] rather than x_pred[1] = x_smooth[1], which
is inconsistent with the M-step's intent.

**Practical impact:** Minimal for long time series. The single mishandled step at tau=1
has negligible effect on parameter estimation (one step out of N ~ 6500+ bins) and decays
exponentially in the filter's forward pass. However, it introduces a systematic (though
small) bias in the first bin's forecast y_hat[1], and could confuse a developer debugging
initial conditions.

**Fix:** Either:
1. Add a special case for tau=1: "if tau == 1: A = eye(2); Q = zeros(2,2)" before the
   day-boundary check, so that x_pred[1] = pi_1 and Sigma_pred[1] = Sigma_1. Or,
2. Change the day-boundary condition to "tau > 1 and tau mod I == 1", which correctly
   triggers only at I+1, 2I+1, etc.

Both fixes are one-line changes. Option 2 is simpler.

**Source:** Paper, page 4, Algorithm 1 (loop starts at tau=1 with x_{1|1} already
available from correction of x_{1|0} = pi_1); page 4, below Eq 5 (pi_1 is the prior
mean of x_1 directly).

---

### m2. Robust EM Convergence Criterion Not Explicitly Specified

**Location:** Algorithm 3 pseudocode, convergence check (lines ~386-394); Robust EM
Modifications (lines ~520-548).

The standard EM section clearly specifies the convergence criterion: relative change in
the innovation-form log-likelihood (line 389). However, the Robust EM Modifications
section does not state whether this same criterion applies to the robust model.

The robust model uses Algorithm 4 (Lasso-penalized filter) in the E-step forward pass.
This introduces two subtleties:

1. **The Lasso penalty means this is not pure EM.** The standard EM algorithm guarantees
   monotonic increase of the observed-data log-likelihood at each iteration. The robust
   variant alternates between solving a penalized optimization for z* (Lasso subproblem)
   and updating theta in closed form. This is closer to coordinate descent on a penalized
   objective than to classical EM. The monotonicity guarantee may not hold for the
   standard (unpenalized) log-likelihood.

2. **Innovation computation with the robust filter.** When Algorithm 4 is used in the
   E-step, the innovations e_tau = y[tau] - y_hat[tau] are the same as the standard
   filter (the prediction step is identical; only the correction step differs). So the
   innovation-form log-likelihood formula is well-defined and computable. The question is
   whether it's the right quantity to monitor for convergence.

The paper does not discuss convergence criteria for the robust EM (Section 3.2 describes
only the modified M-step updates, not convergence). Using the standard innovation-form
log-likelihood as a convergence heuristic is reasonable and is what most practitioners
would do, but this should be stated explicitly.

**Fix:** Add a note in the Robust EM Modifications section stating:
- The convergence check uses the same innovation-form log-likelihood formula as the
  standard EM.
- Note that the monotonic log-likelihood increase property of standard EM may not strictly
  hold for the robust variant due to the Lasso penalty.
- As a safeguard, also monitor the maximum absolute change in key parameters (a^eta,
  a^mu, r) between iterations, and treat convergence as achieved when both the
  log-likelihood change and the maximum parameter change are below their respective
  tolerances.

Mark the parameter-change safeguard as Researcher inference since the paper does not
discuss robust EM convergence.

**Source:** Paper, Section 3.2 (does not address convergence); the observation about
penalized EM and monotonicity is Researcher inference based on standard optimization
theory.

---

## Positive Observations

Draft 3 is implementation-ready. The following aspects are particularly well executed:

1. **All critique 2 issues resolved with precision.** The RM VWAP fix includes a clear
   explanation of why RM is a static method and should be compared against static KF
   results (lines 907-910). The VWAP^TE formula includes both the tracking error (Eq 42)
   and the true VWAP definition (Eq 39), with the basis-point conversion noted. The
   three M-step ordering constraints are documented in a single, easy-to-find block
   (lines 413-426) with paper equation references for each dependency.

2. **Static-to-dynamic evaluation mode transition** (lines 808-830) is now one of the
   clearest sections in the document. The two modes are described step-by-step with the
   key insight that "the initial state for day d's evaluation comes from the filter's
   posterior at the end of day d-1" explicitly stated (lines 824-826).

3. **Dynamic VWAP multi-step prediction pseudocode** (lines 582-604) is concrete and
   unambiguous. The note about within-day transitions always using A = [[1,0],[0,a^mu]]
   (line 606) is a helpful implementation detail.

4. **Benchmark number accuracy.** Every MAPE and VWAP tracking error value I checked
   against the paper's Tables 3 and 4 is correct, including:
   - Dynamic MAPE: 0.46 / 0.47 / 0.65 / 1.28 (Robust KF / KF / CMEM / RM)
   - Static MAPE: 0.61 / 0.62 / 0.90 / 1.28
   - Dynamic VWAP: 6.38 / 6.39 / 7.01
   - Static VWAP: 6.85 / 6.89 / 7.71
   - RM VWAP: 7.48
   - Relative improvements: 64% over RM, 29% over CMEM, 15% VWAP over RM, 9% VWAP over
     CMEM

5. **Equation fidelity.** All M-step equations (A.32-A.39) match the paper precisely,
   including summation index sets (D for eta-related parameters, 2..N for mu-related),
   denominators (T-1 for sigma^eta, N-1 for sigma^mu), and the correct use of newly
   updated parameters (phi^{(j+1)} in A.38, a^eta^{(j+1)} in A.36, a^mu^{(j+1)} in
   A.37).

6. **Comprehensive Paper References table** (lines 1064-1108) now includes all additions
   from draft 2 and 3, with Researcher inference items clearly marked.

7. **Robust EM E-step specification** clearly states that Algorithm 4 handles the forward
   pass, the standard smoother handles the backward pass, and z_star values are stored
   for the M-step (lines 525-532). The citation to Section 3.2's specific text about
   z_1*...z_N* is a nice touch.

8. **Consistent Researcher inference labeling** throughout: phi initialization, rolling
   window warm-start, iteration budget, zero-volume handling, stationarity clamping, and
   log-normal bias correction are all clearly marked.

## Overall Assessment

The specification has converged to a high-quality, implementation-ready state. The
progression from draft 1 (3 major, 8 minor issues) to draft 2 (1 major, 5 minor) to
draft 3 (0 major, 2 minor) shows systematic improvement with no regressions. The two
remaining minor issues are edge cases that would not prevent a correct implementation
and can be fixed with minimal changes. The document is suitable for finalization.
