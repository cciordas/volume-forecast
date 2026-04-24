# Findings Tracker: Direction 7 — Kalman Filter State-Space Model for Intraday Volume

## Summary
- Total findings: 84
- Runs processed: [3, 1, 2, 5, 4, 6, 7, 8, 9, 10]
- Last update: run 10

## Findings

### F1. Model decomposition
**Category:** algorithm
**Consensus** (runs 3, 1, 2, 5, 4, 6, 7, 8, 9, 10): y_{t,i} = eta_t + phi_i + mu_{t,i} + v_{t,i}. Log-volume decomposed into daily average level (eta), intraday seasonal pattern (phi), intraday dynamic residual (mu), and observation noise (v). Additive in log space; multiplicative in volume space.
**Best version so far:** Run 5 — lists all model assumptions explicitly (eta constant within day AR(1) across days, phi static across days, mu AR(1) every bin, r constant), includes identifiability discussion (phi/eta degree of freedom resolved by EM without explicit constraint)

### F2. Volume normalization procedure
**Category:** algorithm
**Consensus** (runs 3, 1, 2, 5, 4, 6, 7, 8, 9, 10): turnover[t,i] = raw_volume[t,i] / shares_out[t], then y[tau] = ln(turnover[t,i]). Log transform converts multiplicative structure to additive, eliminates positiveness constraints, makes Gaussian noise defensible. Source: Section 4.1, Equation (1)/(3). Run 6 adds that shares_outstanding is "most recently reported value, typically from a corporate actions database or previous day's closing record." Run 7 notes the paper's verbal description of Eq(1) appears inverted ("daily outstanding shares / shares traded") but the formula is turnover = traded/outstanding.
**Best version so far:** Run 3 — includes source equations

### F3. Global time indexing
**Category:** index convention
**Consensus** (runs 3, 1, 2, 5, 4, 6, 7, 8, 9, 10): tau = (t-1)*I + i maps (day t, bin i) to global index. N = T*I total bins. Bin-position recovery: i(tau) = ((tau-1) mod I) + 1. 1-based indexing throughout. Run 7 provides explicit index_to_day_bin and is_day_boundary helper functions with pseudocode.
**Best version so far:** Run 5 — uses phi_position[tau] array computed during preprocessing for consistent bin lookup, plus explicit note on reindexing from paper's x_{tau+1|tau} notation to code's x_pred[tau] (prediction FOR time tau)

### F4. Observation vector C
**Category:** algorithm
**Consensus** (runs 3, 1, 2, 5, 4, 6, 7, 8, 9, 10): C = [1, 1] (1x2). Since state is 2x1, C * Sigma * C^T is scalar (sum of all four Sigma elements), K is 2x1 vector. No matrix inversion beyond scalar reciprocal ever needed.
**Best version so far:** Run 3 — explicitly notes W is scalar reciprocal

### F5. State vector definition
**Category:** algorithm
**Consensus** (runs 3, 1, 2, 5, 4, 6, 7, 8, 9, 10): x[tau] = [eta_t, mu_{t,i}]^T (2x1). eta_t is daily average level (log scale), constant within a day. mu_{t,i} is intraday dynamic component, varies bin-to-bin.
**Best version so far:** Run 3

### F6. Time-varying transition matrix A[tau]
**Category:** algorithm
**Consensus** (runs 3, 1, 2, 5, 4, 6, 7, 8, 9, 10): A[tau] = [[a_eta_tau, 0], [0, a_mu]] where a_eta_tau = a_eta if tau is a day boundary, = 1 otherwise. The eta component is constant within a day and AR(1) across days; mu is AR(1) at every step.
**Best version so far:** Run 5 — integrates boundary detection via (tau-1) mod I == 0, constructs A and Q together in IF/ELSE, and explicitly documents A_used[tau] storage indexed from 2..N with rationale for why the smoother's backward loop (accessing A_used[tau+1] from N-1 down to 1) never underflows

### F7. Time-varying process noise Q[tau]
**Category:** algorithm
**Consensus** (runs 3, 1, 2, 5, 4, 6, 7, 8, 9, 10): Q[tau] = [[sigma_eta_tau^2, 0], [0, sigma_mu^2]] where sigma_eta_tau^2 = sigma_eta^2 if tau = kI (day boundary), = 0 otherwise. eta has no process noise within a day; sigma_mu^2 active at every step.
**Best version so far:** Run 1 — provides explicit matrix notation in pseudocode

### F8. Day boundary index convention
**Category:** index convention
**Consensus** (runs 3, 1, 2, 5, 4, 6, 7, 8, 9, 10): Day boundary in transition matrix: tau = kI (last bin of day k) triggers a_eta and sigma_eta^2. M-step set D_start = {kI+1 for k=1,...,T-1} contains destination indices (first bin of next day). T-1 elements. Run 8 adds explicit clarification of the source/destination duality: "The transition from tau to tau+1 crosses a day boundary when tau is the LAST bin of a day (tau mod I == 0), equivalently when tau+1 is the FIRST bin of a new day (tau+1 in D). Both conditions identify the same transition."
**Best version so far:** Run 5 — uses (tau-1) mod I == 0 check in filter, explicit k=1..T-1 loops in M-step with P_cross[kI+1] and P[kI], and separately handles dynamic VWAP context where is_day_boundary is passed as explicit parameter (i==1 for first bin of new day)

### F9. Kalman filter initialization at tau=1
**Category:** algorithm
**Approach A** (run 3): Explicit initial correction step before the main loop. Initialize x_hat[1|0] = pi_1, Sigma[1|0] = Sigma_1. If observed[1]: compute K[1], correct to get x_hat[1|1]. Store e[1], S[1] for log-likelihood.
**Approach B** (runs 1, 2, 5, 4, 6, 7, 8, 9, 10): Handles tau=1 inside the main loop with a conditional: IF tau == 1, set x_pred = pi_1, Sigma_pred = Sigma_1 (no transition applied). Then the standard correction step processes y[1] if observed. No separate pre-loop code. Run 6 initializes x_pred[1] and Sigma_pred[1] before the loop, then enters the main loop at tau=1 with correction first, prediction second. Run 7 sets x_prev = pi_1, Sigma_prev = Sigma_1 before the loop and enters at tau=1 with the standard predict step (A applied to x_prev); since is_day_boundary(1,I) returns False, A = [[1,0],[0,a_mu]], so x_pred[1] = [pi_1[0], a_mu*pi_1[1]] — equivalent to pi_1 when pi_1[1]=0 (the standard initialization). Run 8 initializes x_hat[1|0] = pi_1, Sigma[1|0] = Sigma_1 before the loop, then enters at tau=1 with forecast->correct->predict — no conditionals needed since the prior IS the predicted state at tau=1.
**Best version so far:** Run 5 — cleaner code structure, integrates tau=1 into unified loop, explicitly notes A_used is not set at tau=1 (no transition involved)

### F10. Joseph form for covariance update
**Category:** algorithm
**Approach A** (runs 3, 1, 2, 5, 6, 9, 10): Sigma[tau|tau] = (I_2 - K*C) * Sigma[tau|tau-1] * (I_2 - K*C)^T + K*r*K^T (Joseph form). Guarantees symmetry and positive semi-definiteness. Algebraically equivalent to standard form but numerically stable for long sequences. Researcher inference (standard practice, not in paper). Run 6 mentions both forms: standard in pseudocode (line 174), Joseph equivalent noted (line 175).
**Approach B** (runs 4, 7, 8): Sigma[tau] = Sigma_pred[tau] - K @ C @ Sigma_pred[tau] (standard form). Used directly in pseudocode without Joseph form. Simpler but can lose positive-definiteness over long sequences due to floating-point accumulation. Run 7 uses standard form in pseudocode but recommends Joseph form for production in edge cases section. Run 8 shows both algebraic forms side by side in Step 3 (Sigma - K*S*K^T = (I-K*C)*Sigma), uses standard form in Step 5 (robust), and recommends Joseph form for numerical issues in edge cases section.
**Best version so far:** Run 2 — shows both Joseph and standard forms side by side, explains why Joseph is preferred for production. Run 4's standard form is simpler but less robust numerically.

### F11. Missing observation handling in Kalman filter
**Category:** edge case
**Consensus** (runs 3, 1, 2, 5, 4, 9, 10): When observed[tau] = false: x_hat[tau|tau] = x_hat[tau|tau-1], Sigma[tau|tau] = Sigma[tau|tau-1], K[tau] = [0,0]^T. Prediction proceeds normally; correction is skipped. Zero-volume bins flagged as missing during preprocessing. Paper: Section 4.1.
**Approach A** (runs 6, 7, 8): Lists as assumption ("No zero-volume bins exist in the data; log(0) is undefined") and as edge case with options: (1) exclude and treat as missing (modify filter to skip correction), (2) impute small volume (e.g., 1 share). Does not integrate missing-observation handling into the filter pseudocode itself; filter assumes all bins observed. Run 8 gives the skip-correction formula explicitly in a paragraph after the filter pseudocode: "set x_hat[tau|tau] = x_hat[tau|tau-1] and Sigma[tau|tau] = Sigma[tau|tau-1], then proceed to the prediction step."
**Best version so far:** Run 5 — formalizes the skip in filter pseudocode, sets e[tau]=0, S[tau]=0 as sentinels (not used in log-likelihood), z_star[tau]=0

### F12. Kalman filter prediction step
**Category:** algorithm
**Consensus** (runs 3, 1, 2, 5, 4, 6, 7, 8, 9, 10): x_hat[tau+1|tau] = A[tau]*x_hat[tau|tau]; Sigma[tau+1|tau] = A[tau]*Sigma[tau|tau]*A[tau]^T + Q[tau]; y_hat[tau+1|tau] = C*x_hat[tau+1|tau] + phi[tau+1]. Paper: Algorithm 1, lines 2-3. Run 8 adds explicit note on loop body ordering: forecast->correct->predict at each tau (vs paper Algorithm 1 which orders predict->correct iterating over tau+1), noting the equations are identical — only loop structure differs.
**Best version so far:** Run 5 — full pseudocode with explicit loop structure, A_used[tau] stored indexed 2..N, all output arrays enumerated including z_star initialization

### F13. RTS smoother backward pass
**Category:** algorithm
**Consensus** (runs 3, 1, 2, 5, 4, 6, 7, 8, 9, 10): L[tau] = Sigma[tau|tau]*A[tau]^T*Sigma[tau+1|tau]^{-1} (2x2 gain). x_hat[tau|N] = x_hat[tau|tau] + L[tau]*(x_hat[tau+1|N] - x_hat[tau+1|tau]). Sigma[tau|N] = Sigma[tau|tau] + L[tau]*(Sigma[tau+1|N] - Sigma[tau+1|tau])*L[tau]^T. Runs tau=N-1 down to 1. Paper: Algorithm 2, Equations 10-11.
**Best version so far:** Run 5 — uses A_used[tau+1] notation, stores L_stored[tau] for tau=1..N-1 for later cross-covariance use, includes epsilon-regularized 2x2 inversion inline

### F14. Smoother cross-covariance computation
**Category:** algorithm
**Approach A** (runs 3, 1, 4, 6, 7, 8, 9, 10): Recursive formula from paper's Eq A.20/A.21. Sigma[N,N-1|N] = (I_2 - K[N]*C) * A[N-1] * Sigma[N-1|N-1]. Then backward recursion from k=N-2 downto 1 using Eq A.20. Run 6 provides full pseudocode with explicit index mapping: Sigma_cross[k] = Sigma_{k+1,k|N}, initialization at k=N-1, backward loop with A_{k+1} construction at each step, and inline comment documenting the index convention. Run 8 provides full pseudocode matching the paper's Eq A.20/A.21 directly.
**Approach B** (runs 2, 5): Non-recursive Shumway & Stoffer (1982) formula: Sigma_cross[tau] = Sigma_smooth[tau] @ L[tau-1]^T. No recursion, no initialization step, each tau independent. Includes algebraic derivation proving equivalence with A.20. K_stored not needed by smoother.
**Best version so far:** Run 5 — provides detailed proof sketch (x_filt[tau-1] conditionally independent of smoother correction, so Cov reduces to Sigma_smooth[tau] @ L[tau-1]^T), cites Shumway & Stoffer Ch. 6 Property 6.3, and notes the recursive alternative requires K[N] storage

### F15. Smoother 2x2 inverse with epsilon regularization
**Category:** algorithm
**Consensus** (runs 3, 1, 5, 4, 6, 7, 8, 9, 10): Sigma[tau+1|tau] inverted via analytic 2x2 formula: inv([[a,b],[c,d]]) = (1/(ad-bc))*[[d,-b],[-c,a]]. Run 8 uses inv() without specifying the method; consistent with analytic formula for 2x2.
**Approach A** (runs 3, 5, 4): If determinant < 1e-12, add epsilon (1e-10) to diagonal before inverting.
**Approach B** (run 1): Uses analytical formula directly; mentions enforcing minimum noise variances (1e-10) upstream to prevent singularity.
**Best version so far:** Run 5 — epsilon guard at inversion point with explicit Sigma_pred_reg temporary variable, includes full pseudocode for both regular and regularized paths

### F16. EM sufficient statistics
**Category:** algorithm
**Consensus** (runs 3, 1, 2, 5, 4, 6, 7, 8, 9, 10): P[tau] = Sigma[tau|N] + x_hat[tau|N]*x_hat[tau|N]^T (= E[x*x^T|all data]). P[tau,tau-1] = Sigma[tau,tau-1|N] + x_hat[tau|N]*x_hat[tau-1|N]^T. Sub-elements: P^{(1,1)} for eta*eta, P^{(2,2)} for mu*mu. Paper: Appendix A, Eq A.19, A.22. Run 8 adds explicit notation note: "P_tau^(i,j) denotes the (i,j) element of the 2x2 matrix P. These are NOT EM iteration superscripts."
**Best version so far:** Run 5 — separate computation loops for P (tau=1..N) and P_cross (tau=2..N), explicit note that P^(k,l) notation in M-step equations refers to (k,l) element of 2x2 matrices

### F17. M-step computation order
**Category:** algorithm
**Approach A** (runs 3, 1, 5, 4, 6, 8, 9, 10): phi must be computed BEFORE r (r uses phi^{j+1}). sigma_eta^2 uses a_eta^{j+1}, sigma_mu^2 uses a_mu^{j+1}. pi_1, Sigma_1, a_eta, a_mu can be computed in any order. Run 4 explicitly enumerates all THREE ordering dependencies as a complete set with paper citations: (1) phi before r (Eq A.38), (2) a_eta before sigma_eta^2 (Eq A.36), (3) a_mu before sigma_mu^2 (Eq A.37). Run 6 includes inline comment "NOTE: phi must be computed BEFORE r, since r depends on phi_new." Run 8 adds researcher inference: "this ordering follows from the simultaneous optimality conditions in Appendix A.3; either order gives the same fixed point, but computing phi first avoids needing a second pass."
**Approach B** (runs 2, 7): Computes r BEFORE phi in sequential pseudocode. r formula uses current phi (from previous iteration), not phi^{j+1}. phi computed last. The compact r formula r = (1/N)*sum[(y - phi - C@x_hat)^2 + C@Sigma_smooth@C^T] uses the old phi.
**Best version so far:** Run 4 — most explicit enumeration of all three M-step ordering constraints with per-equation citations and developer warning. Run 5 has the best pseudocode structure.

### F18. M-step: initial state mean pi_1
**Category:** algorithm
**Consensus** (runs 3, 1, 2, 5, 4, 6, 7, 8, 9, 10): pi_1^(j+1) = x_hat[1] (smoothed state at tau=1). Paper: Equation A.32.
**Best version so far:** Run 3

### F19. M-step: initial state covariance Sigma_1
**Category:** algorithm
**Approach A** (runs 3, 1, 4, 6, 7, 8, 9, 10): Sigma_1^(j+1) = P[1] - x_hat[1]*x_hat[1]^T. Paper: Equation A.33. Run 6 uses this form as the primary formula but adds inline comment "= Sigma_smooth[1]" noting equivalence. Run 7 uses same form and also notes "Equivalently: Sigma_1 = Sigma_smooth[1]". Run 8 uses P[1] - x_hat^T form, adds inline derivation: "since P[1] = Sigma[1|N] + x_hat*x_hat^T, this simplifies to Sigma[1|N]."
**Approach B** (runs 2, 5): Sigma_1^(j+1) = Sigma_smooth[1]. Mathematically equivalent via P[1] = Sigma_smooth[1] + x_hat[1]*x_hat[1]^T (Eq A.19). Avoids numerically unstable subtraction of two positive semi-definite matrices of similar magnitude.
**Best version so far:** Run 5 — uses Sigma_smooth[1] with inline comment showing equivalence to Eq A.33 and explaining why the subtraction form risks catastrophic cancellation

### F20. M-step: daily AR coefficient a_eta
**Category:** algorithm
**Consensus** (runs 3, 1, 2, 5, 4, 6, 7, 8, 9, 10): a_eta^(j+1) = [sum_{tau in D_start} P^{(1,1)}[tau,tau-1]] / [sum_{tau in D_start} P^{(1,1)}[tau-1]]. Sums over D_start = {kI+1} (destination indices of day transitions). Paper: Equation A.34.
**Best version so far:** Run 5 — explicit k=1..T-1 loop with tau=k*I+1, P_cross[tau][0,0] numerator / P[tau-1][0,0] denominator, with inline element-index notation

### F21. M-step: intraday AR coefficient a_mu
**Category:** algorithm
**Consensus** (runs 3, 1, 2, 5, 4, 6, 7, 8, 9, 10): a_mu^(j+1) = [sum_{tau=2}^{N} P^{(2,2)}[tau,tau-1]] / [sum_{tau=2}^{N} P^{(2,2)}[tau-1]]. Sums over all consecutive pairs. Paper: Equation A.35.
**Best version so far:** Run 5 — uses P_cross[tau][1,1] and P[tau-1][1,1] with 0-based element indices matching implementation arrays

### F22. M-step: seasonality phi
**Category:** algorithm
**Approach A** (runs 3, 1, 2, 4, 6, 7, 8): phi_i^(j+1) = (1/T) * sum_{t=1}^{T} (y[t,i] - C*x_hat[t,i]). Simple average across ALL training days for each bin position, denominator is T. Paper: Equation A.39. Run 6 uses tau(t,i) = (t-1)*I + i to convert to linear index.
**Approach B** (runs 5, 10): phi_i^(j+1) = (1/count_i) * sum_{observed} (y[tau] - C*x_smooth[tau] - z_star[tau]), where count_i is the number of OBSERVED days for bin i. Skips missing bins and uses observed count as denominator. Handles robust variant by subtracting z_star (per Eq 36). Not a Fourier parameterization.
**Best version so far:** Run 5 — correctly handles missing data (y[tau] undefined for unobserved bins, so summing over all T would require a placeholder value), integrates robust z_star subtraction in a unified formula

### F23. M-step: daily process noise sigma_eta^2
**Category:** algorithm
**Consensus** (runs 3, 1, 2, 5, 4, 6, 7, 8, 9, 10): [sigma_eta^2]^(j+1) = (1/(T-1)) * sum_{tau in D_start} {P^{(1,1)}[tau] + (a_eta^(j+1))^2 * P^{(1,1)}[tau-1] - 2*a_eta^(j+1)*P^{(1,1)}[tau,tau-1]}. Denominator is T-1 (number of day transitions). Uses a_eta^(j+1). Paper: Equation A.36.
**Best version so far:** Run 5 — explicit k=1..T-1 loop with tau=k*I+1, uses P[tau][0,0] and P_cross[tau][0,0] element notation, inline comment "Eq A.36, denominator is |D| = T-1"

### F24. M-step: intraday process noise sigma_mu^2
**Category:** algorithm
**Consensus** (runs 3, 1, 2, 5, 4, 6, 7, 8, 9, 10): [sigma_mu^2]^(j+1) = (1/(N-1)) * sum_{tau=2}^{N} {P^{(2,2)}[tau] + (a_mu^(j+1))^2 * P^{(2,2)}[tau-1] - 2*a_mu^(j+1)*P^{(2,2)}[tau,tau-1]}. Denominator is N-1. Uses a_mu^(j+1). Paper: Equation A.37.
**Best version so far:** Run 5 — inline in full pseudocode with P[tau][1,1] and P_cross[tau][1,1] element notation

### F25. M-step: observation noise r
**Category:** algorithm
**Approach A** (runs 3, 4, 6, 7, 8, 9, 10): Fully expanded form: r^(j+1) = (1/N) * sum [y^2 + C*P*C^T - 2y*C*x_s + phi^2 - 2y*phi + 2*phi*C*x_s]. Uses phi^{(j+1)}. Paper: Equation A.38. Run 6 provides the complete expansion with all 6 terms explicitly. Run 7 provides both expanded form and compact simplification. Run 8 provides the same expanded form with equation comments. Run 9 uses expanded form but with N_obs denominator (sum over observed bins only).
**Approach B** (runs 1, 2, 5, 4): Compact form: r = (1/N_obs)*sum_{observed}[(y - phi - z_star - C@x_smooth)^2 + C@Sigma_smooth@C^T]. Run 1 notes expansion matches Eq A.38. Run 2 derives this via P = Sigma_smooth + x_hat*x_hat^T. Run 5 uses N_obs denominator (observed bins only) and includes z_star subtraction for unified standard/robust handling.
**Best version so far:** Run 5 — compact form with N_obs denominator (correct for missing data), unified standard/robust via z_star term, explicit derivation via P decomposition noted in source comments

### F26. Parameter clamping
**Category:** algorithm
**Approach A** (runs 3, 5): EPSILON = 1e-8. a_eta, a_mu clamped to (EPSILON, 1-EPSILON). sigma_eta^2, sigma_mu^2, r clamped to >= EPSILON.
**Approach B** (runs 1, 2, 10): Variance floor (1e-10). Run 2 mentions stationarity constraint (0,1) for AR coefficients but no explicit clamping epsilon.
**Approach C** (runs 4): clip(a_eta, 0.001, 0.999) and clip(a_mu, 0.001, 0.999) for AR coefficients. Includes warning log when unclamped value falls outside (0,1) as potential model misspecification signal. Variance clamping not explicitly specified.
**Best version so far:** Run 5 — most complete, covers both AR coefficients and variances with specific EPSILON=1e-8 constant, placed at end of M-step as a single clamping block. Run 4 adds the useful diagnostic of logging a warning when clamping is triggered.

### F27. Convergence monitoring function
**Category:** algorithm
**Approach A** (runs 3, 1, 5, 4, 6, 7, 8, 9, 10): Innovation-based log-likelihood: log_lik = -0.5 * sum_{observed} [ln(S[tau]) + e[tau]^2/S[tau]] - (N_obs/2)*ln(2*pi). Runs 3, 5, 9 use N_obs (correct for missing data); runs 1, 6, 7, 8 use N. Run 8 cites Shumway & Stoffer 1982.
**Approach B** (run 2): Q-function from Eq A.10: Q(theta|theta^(j)) = -E_1 - E_2 - E_3 - E_4 - (N/2)log(r) - ((N-1)/2)log(sigma_mu^2) - ((T-1)/2)log(sigma_eta^2) - (1/2)log|Sigma_1| - ((2N+T)/2)log(2*pi). EM guarantees Q monotonicity; a decrease signals implementation bug.
**Best version so far:** Run 2 — Q-function is the theoretically correct EM convergence criterion (EM guarantees Q monotonicity, not data log-likelihood monotonicity); provides a built-in implementation correctness check. However, run 5's innovation LL with N_obs is simpler to compute and more standard in practice.

### F28. Convergence criterion
**Category:** algorithm
**Consensus** (runs 3, 1, 2, 6): Relative tolerance: |f^(j) - f^(j-1)| / denominator < tol, where f is the monitoring function. All suggest parameter change norm as alternative.
**Approach A** (runs 3, 4, 10): Denominator is |f^(j-1)|; initialize f^(0) = -infinity.
**Approach B** (runs 1, 5, 6): Denominator uses a guard: runs 1, 5 use max(|f_prev|, 1.0); run 6 uses abs(prev_log_lik) + 1e-10. Same intent (prevent division by zero), slightly different guard values.
**Approach C** (run 2): Same relative form. Also notes that monitoring log-determinant terms alone suffices after M-step (see F67).
**Approach D** (runs 7, 8): Absolute tolerance: abs(ll - prev_ll) < tol. No denominator normalization. Simpler but scale-dependent (tolerance must be chosen relative to log-likelihood magnitude, which depends on N).
**Best version so far:** Run 5 — max(abs(log_lik_prev), 1.0) guard in full pseudocode, tol=1e-8 default, initializes log_lik_prev = -infinity, includes correctness check note (log-likelihood must be non-decreasing). Run 7's absolute tolerance is simpler but less portable across different dataset sizes.

### F29. Robust Lasso observation model
**Category:** algorithm
**Consensus** (runs 3, 1, 2, 5, 4, 6, 7, 8, 9, 10): y[tau] = C*x[tau] + phi[tau] + v[tau] + z[tau], where z[tau] is sparse outlier term. Soft-thresholding: threshold = lambda*S_tau/2 (equivalently lambda/(2*W[tau])). z*[tau] = sign(e)*max(|e| - threshold, 0). Standard Kalman correction with e_clean = e - z*. Paper: Section 3.1, Equations 25-34. Run 8 uses threshold = lambda/(2*W[tau]) with full IF/ELIF/ELSE soft-thresholding, derives threshold equivalence lambda/(2W) = lambda*S/2 inline, and adds equivalent formulation for e_clean: "sign(e)*min(|e|, threshold)".
**Best version so far:** Run 5 — IF/ELIF/ELSE soft-thresholding inline in filter correction step, z_star array initialized to 0 at loop start (standard mode keeps all zeros), derives threshold equivalence lambda/(2W) = lambda*S/2 because W=1/S (Eq 30)

### F30. Adaptive threshold in robust filter
**Category:** algorithm
**Consensus** (runs 3, 1, 2, 5, 4, 6, 7, 8, 9, 10): threshold = lambda*S[tau]/2 is time-varying because S[tau] depends on current predictive variance. High uncertainty -> wider threshold. Paper: implicit in Section 3.1.
**Best version so far:** Run 5 — derives lambda/(2W) = lambda*S/2 from W=1/S (Eq 30), S_tau = C@Sigma_pred@C^T + r explicit in filter pseudocode

### F31. RTS smoother unchanged in robust variant
**Category:** algorithm
**Consensus** (runs 3, 1, 2, 5, 4, 6, 7, 8, 9, 10): Smoother operates on filtered states from robust Kalman filter, which already incorporate outlier cleaning. z* values used only in forward filter correction and M-step updates for r and phi. Paper: Section 3.2 (implicit). Run 8 adds detailed data flow clarification: "E-step consists of two sub-steps: (a) robust Kalman filter forward pass which computes and STORES z_star[tau]; (b) standard RTS smoother (unchanged). M-step uses both smoother outputs AND stored z_star."
**Best version so far:** Run 5 — explicitly notes in M-step source comments that z_star affects only observation equation, not state dynamics, and that smoother operates on filtered states which already incorporate outlier cleaning

### F32. Robust EM M-step modifications
**Category:** algorithm
**Consensus** (runs 3, 1, 2, 6, 7, 8, 9, 10): Only r and phi updates change. phi_i^(j+1) subtracts z*. r update uses (y-phi-z*). All other M-step equations unchanged. Paper: Equations 35-36.
**Approach A** (runs 3, 6, 8, 9, 10): Lists individual terms added to r. Run 6 provides full algebraic expansion of E[(y - Cx - phi - z)^2] showing all 10 cross-terms explicitly, including (z*)^2, -2y*z*, +2*z**C*x_hat, +2*z**phi. Run 8 provides complete expanded Eq 35 with all z_star cross-terms in pseudocode.
**Approach B** (runs 1, 2, 5, 4): Compact form: r = (1/N_obs)*sum_{observed}[(y - phi - z* - C@x_smooth)^2 + C@Sigma_smooth@C^T]. Run 5 uses N_obs denominator.
**Best version so far:** Run 5 — unified M-step pseudocode handles both standard and robust: z_star=0 for standard mode makes z_star terms vanish, N_obs denominator correct for missing data

### F33. Robust variant subsumes standard
**Category:** algorithm
**Approach A** (runs 3, 1, 2, 5, 4, 6, 7, 8, 10): Setting lambda = infinity recovers standard filter (threshold infinite, z* always zero). Robust performs equal or better on clean data (Table 1: MAPE difference <= 0.01). Computational overhead negligible. Paper: Section 3, Tables 1, 3. Run 8 recommends implementing robust as primary with standard as subcase, noting "On curated historical data, both variants give very similar results (Table 3: average MAPE 0.46 vs 0.47)."
**Approach B** (run 9): Claims lambda = 0 recovers standard filter ("threshold becomes zero and z_tau* = 0 always"). This is mathematically INCORRECT: when threshold = 0, soft_threshold(e, 0) = e (not 0), so z* = e and e - z* = 0, meaning no state correction occurs (the degenerate case described in F57). Run 9 makes this error in both the Variants section and in sanity check #5. The correct recovery is lambda -> infinity (Approach A).
**Best version so far:** Run 5 — implements robust as primary model with lambda=1e10 for standard recovery. Run 9's Approach B is incorrect and should not be used.

### F34. Static prediction formulas
**Category:** algorithm
**Consensus** (runs 3, 1, 2, 5, 4, 6, 7, 8, 9, 10): From end of day t: eta_hat = a_eta * x_hat[tau_0|tau_0][1] (constant for all bins). mu_hat[h] = (a_mu)^h * x_hat[tau_0|tau_0][2] (decays geometrically). y_hat = eta_hat + mu_hat[h] + phi[h]. First transition uses a_eta (day boundary), subsequent use identity (eta constant within day). Paper: Section 2.2, Equation 9. Run 7 provides standalone predict_static function with explicit day-boundary transition at first bin, then within-day A_intra loop for bins 2..I.
**Best version so far:** Run 5 — full pseudocode propagating x_curr/Sigma_curr through A_within/Q_within matrices at each bin, explicit day-boundary transition first, includes note that exp(y_hat) gives turnover not raw volume (shares_out cancels in VWAP weights and MAPE)

### F35. Dynamic VWAP interleaving procedure
**Category:** algorithm
**Consensus** (runs 3, 1, 2, 5, 4, 6, 7, 8, 10): For each bin i=1..I-1: (1) compute weight w[i] from current forecasts of remaining bins, (2) execute w[i]*Q shares, (3) observe y[tI+i], (4) Kalman correct, (5) re-forecast bins i+1..I. Last bin: w[I] = remaining. Paper: Equation 41. Run 8 provides full dynamic VWAP pseudocode with cumulative_weight tracking, explicit re-forecasting of remaining bins after each observation, and a researcher inference note clarifying that "one-bin-ahead forecasting" (Section 4.2) refers to each bin's own prediction while Eq 41's denominator requires multi-step forecasts for remaining bins.
**Best version so far:** Run 5 — full pseudocode with cumulative_executed tracking, explicit day boundary handling (is_boundary = i==1), calls Step 9 (one-bin-ahead) and Step 10 (multi-step-ahead), distinguishes backtesting vs production modes

### F36. Static VWAP weights
**Category:** algorithm
**Consensus** (runs 3, 1, 2, 5, 4, 6, 7, 8, 10): w[i] = volume_hat[i] / sum_{j=1}^{I} volume_hat[j]. Paper: Equation 40.
**Best version so far:** Run 5 — standalone pseudocode block with total = sum(volume_hat_static[1..I])

### F37. Log-normal bias correction
**Category:** algorithm
**Consensus** (runs 3, 1, 5, 4, 6, 7, 8, 10): exp(E[log(V)]) is geometric mean, not arithmetic. Unbiased: volume_hat = exp(y_hat + 0.5*prediction_variance) where prediction_variance = C*Sigma[tau+1|tau]*C^T + r. Paper does not discuss; researcher inference. Run 7 derives E[V] = exp(E[log V] + Var(log V)/2) > exp(E[log V]) via Jensen's inequality, notes paper does not correct for this bias.
**Best version so far:** Run 5 — explicit formula exp(y_hat + 0.5*(C@Sigma_curr@C^T + r)) with clear recommendation: use exp(y_hat) without correction for paper MAPE benchmarks, bias correction for production use

### F38. Volume conversion from log space
**Category:** algorithm
**Consensus** (runs 3, 1, 2, 5, 4, 6, 7, 8, 10): volume_hat[t,i] = exp(y_hat[tau]) * shares_out[t]. Paper: Section 2.2.
**Best version so far:** Run 5 — explicit note that exp(y_hat) gives predicted TURNOVER not raw volume (y = ln(volume/shares_out)), shares_out cancels in VWAP weight ratios and MAPE ratios

### F39. Number of intraday bins I
**Category:** parameter
**Consensus** (runs 3, 1, 2, 5, 4, 6, 7, 8, 10): I = 26 for NYSE (6.5 hours / 15 minutes). Exchange-dependent. Configuration parameter, not tuned.
**Best version so far:** Run 2 — adds bin boundary convention [9:30:45), ..., [3:45, 4:00) and typical range 13-52 for 30-min to 10-min bins

### F40. Expected a_eta range
**Category:** parameter
**Consensus** (runs 3, 1, 2, 4, 6, 7, 8, 10): Should be close to 1, reflecting high persistence of daily level. Paper: Figure 4a. Run 8 reports range 0.95-0.999 in parameters table.
**Approach A** (runs 3, 2, 5, 4): Converges near 0.98-0.99 in synthetic experiments.
**Approach B** (run 1): Uses 0.95 in synthetic test parameters (sanity check).
**Approach C** (run 6): Reports empirical range 0.8-1.0 in Parameters table.
**Best version so far:** Run 5 — cites Figure 4 for converged values, uses 0.98 in synthetic sanity check (a_eta=0.98)

### F41. Expected a_mu range
**Category:** parameter
**Consensus** (runs 3, 1, 2, 5, 4, 6, 7, 8): Converges near 0.5-0.7 in synthetic experiments. Reflects faster mean-reversion of intraday dynamics. Paper: Figure 4b. Run 8 reports range 0.3-0.7 in parameters table.
**Best version so far:** Run 3

### F42. Expected r range
**Category:** parameter
**Consensus** (runs 3, 1, 2, 4, 6): Converges near 0.05-0.1 in synthetic experiments.
**Approach A** (runs 3, 2, 5): ~0.05. Paper: Figure 4c.
**Approach B** (run 1): Uses 0.1 in synthetic test parameters.
**Approach C** (run 6): Reports range 0.01-1.0 for log-volume in Parameters table (wider range covering full empirical spread).
**Best version so far:** Run 3 — directly cites Figure 4c

### F43. EM initialization values
**Category:** parameter
**Approach A** (run 3): Simple defaults: a_eta^(0)=0.9, a_mu^(0)=0.5, sigma_eta^2^(0)=var(daily_means), sigma_mu^2^(0)=var(y)*0.1, r^(0)=var(y)*0.5, phi^(0)=per-bin average minus grand mean, pi_1^(0)=[mean(y[1..I]),0]^T, Sigma_1^(0)=diag(var(y),var(y)).
**Approach B** (runs 1, 5, 7, 9): Decomposition-based heuristic: phi_init from per-bin mean minus grand mean, eta_init from per-day mean, mu_init as residual y - eta - phi, AR coefficients from sample autocorrelation clamped to [0.01, 0.99], innovation variances via stationary AR(1) formula var*(1-a^2). pi_1 = [eta_init[1], 0], Sigma_1 = diag(var(eta_init), var(mu_init)). Run 7 provides full initialize_theta pseudocode: phi_init from per-bin mean of y, residuals = y - phi_init, daily_avg from residuals, a_eta from autocorr(daily_avg, lag=1), a_mu=0.5, sig2_eta = var(daily_avg)*(1-a_eta^2), r_init = 0.1*var(resid).
**Approach C** (runs 2, 10): a_eta^(0)=0.95, a_mu^(0)=0.5, sigma_eta_sq^(0)=0.01, sigma_mu_sq^(0)=0.01, r^(0)=var(y)*0.1, phi^(0)=per-bin mean minus overall mean, pi_1=[y_bar, 0]^T, Sigma_1=diag(var(y), var(y)).
**Approach D** (runs 4, 8): a_eta^(0)=0.99, a_mu^(0)=0.5, sigma_eta^2=var(daily mean log-volume), sigma_mu^2=var(detrended)/I, r=0.1*var(y), phi=mean of y_{t,i} across training days (raw per-bin mean, not residual), pi_1=[mean(daily log-vol), 0]^T, Sigma_1=diag(1,1). Notes paper does not specify phi initialization and that raw per-bin mean is researcher inference. Run 8 uses a_eta=0.99, a_mu=0.5, sigma_eta^2=var(daily means)/10, sigma_mu^2=var(demeaned)/10, r=var(demeaned)/2, phi=per-bin mean (raw, not residual), pi_1=[mean(y),0], Sigma_1=diag(var(y),var(y)). Slightly differs in sigma_mu^2 and Sigma_1 from run 4 but shares the same approach structure (a_eta=0.99, raw per-bin phi).
**Approach E** (run 6): Equal variance partition: a_eta^(0)=a_mu^(0)=0.5, sigma_eta^2=sigma_mu^2=r=var(y)/3, phi_init from per-bin mean of y, pi_1=[mean(y-phi_init), 0]^T, Sigma_1=diag(var(y), var(y)). Simpler than A-D; relies on EM robustness to initialization (Paper, Section 2.3.3, Figure 4).
**Best version so far:** Run 5 — most detailed pseudocode: explicit loops computing bin_values, day_values, mu_init residuals, sample autocorrelation for AR coefficients, stationary variance formula for innovation variances, with source note that paper (Section 2.3.3, Figure 4) shows robustness to initialization

### F44. EM convergence speed
**Category:** parameter
**Consensus** (runs 3, 1, 2, 5, 4, 6, 7, 8, 10): Converges within 5-20 iterations from reasonable initializations. Paper: Section 2.3.3, Figure 4.
**Approach A** (runs 3, 5): Recommended max_iter=100, tol=1e-8. Run 5 combines run 3's max_iter with run 1's tighter tolerance.
**Approach B** (run 1): Recommended max_iter=50-100, tol=1e-8 (relative). Notes log-likelihood scale depends on N.
**Approach C** (runs 2, 6, 8): Recommended max_iter=50, tol=1e-6. Run 2 notes convergence typically 5-10 iterations. Run 8 uses max_iter=50-100 with tol=1e-6.
**Approach D** (run 4): Recommended max_iter=50 with tol=1e-6. Explicitly marks the 20-50 iteration range as researcher inference (not from paper; paper only demonstrates fast convergence on synthetic data). Notes developer should monitor actual convergence rather than relying on fixed budget.
**Best version so far:** Run 5 — max_iter=100 with tol=1e-8, parameters table shows tol range 1e-6 to 1e-10, sensitivity rated "Very low". Run 4 adds useful caveat that iteration budget recommendations are researcher inference.

### F45. Lambda selection via cross-validation
**Category:** parameter
**Consensus** (runs 3, 1, 2, 4, 6, 7, 8, 10): Lambda selected by CV on held-out validation set. Paper does not report typical values. High sensitivity. Paper: Section 4.1. Run 7 implements full calibrate_rolling function with nested grid search over (N_cand, lambda) pairs, validation MAPE as selection criterion.
**Approach A** (run 3): Candidate grid: [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0].
**Approach B** (runs 2, 5): Log-spaced from 0.01 to 100 (15-20 points). Run 5 adds lambda=1e10 as explicit infinity candidate in the grid, allowing CV to automatically select standard KF if no robustness is needed.
**Best version so far:** Run 5 — log-spaced grid 0.01 to 100 plus lambda=1e10 infinity candidate, with explicit rationale for including the standard KF in the search space

### F46. Training window candidates
**Category:** parameter
**Consensus** (runs 3, 1, 2, 4, 6, 9): N candidates for training window length. Paper uses Jan 2013 to variable endpoint. Selected jointly with lambda by minimizing MAPE on validation period. Run 6 gives range 100-1000 days (100*I to 1000*I bins).
**Approach A** (run 3): [100, 150, 200, 300, 500] days.
**Approach B** (run 2): [60, 90, 120, 180, 250] days.
**Approach C** (run 5): [60, 100, 150, 200, 300, 500] days. Merges run 2's lower bound (60) with run 3's upper range (up to 500).
**Best version so far:** Run 5 — widest range covering both short (60 days) and long (500 days) windows, matching the paper's training period (Jan 2013 to variable endpoint, ~500+ trading days) while also accommodating shorter-history scenarios

### F47. Re-estimation frequency
**Category:** parameter
**Approach A** (runs 3, 1, 2, 5, 4, 8, 10): Daily re-estimation (shift window by I bins each day). Researcher inference; paper does not specify. Computationally feasible given O(N) operations with 2x2 matrices.
**Approach B** (run 6): Weekly or monthly recalibration. Paper uses a fixed training window for the entire out-of-sample period (Section 4.1); rolling recalibration is researcher inference. Run 6 recommends periodic recalibration but at lower frequency than daily.
**Best version so far:** Run 2 — includes explicit computational cost argument (N=6500, ~65k scalar ops) supporting daily feasibility. Daily is more conservative (parameters stay current) and computationally cheap.

### F48. Dynamic prediction MAPE benchmarks
**Category:** validation
**Consensus** (runs 3, 1, 2, 5, 4, 6, 7, 8, 10): Robust KF: 0.46, Standard KF: 0.47, CMEM dynamic: 0.65, Rolling Mean: 1.28. Averages across 30 securities. Paper: Table 3.
**Best version so far:** Run 5 — includes both dynamic and static MAPE in a formatted table

### F49. Static prediction MAPE benchmarks
**Category:** validation
**Consensus** (runs 3, 1, 2, 5, 4, 6, 7, 8, 10): Robust KF: 0.61, Standard KF: 0.62, CMEM static: 0.90, Rolling Mean: 1.28. Paper: Table 3.
**Best version so far:** Run 5 — formatted table alongside dynamic MAPE for side-by-side comparison

### F50. VWAP tracking error benchmarks (dynamic)
**Category:** validation
**Consensus** (runs 3, 1, 2, 5, 4, 6, 7, 8, 10): Robust KF: 6.38 bps, Standard KF: 6.39, CMEM: 7.01, Rolling Mean: 7.48. 15% improvement over RM, 9% over CMEM. Paper: Table 4, Equation 42. Run 7 initially misread Table 4 (confusing mean with std columns) but self-corrected to the correct values.
**Best version so far:** Run 1 — includes cross-check calculation: (7.48-6.38)/7.48 = 14.7% ~ 15%, (7.01-6.38)/7.01 = 9.0%

### F51. Per-ticker MAPE range
**Category:** validation
**Consensus** (runs 3, 1, 2, 5, 6, 9): Best: AAPL at 0.21. Worst: 2800HK at 1.94 (leveraged ETF). Most U.S. large-cap: 0.21-0.42. Paper: Table 3.
**Best version so far:** Run 2 — includes per-security mean +/- std (SPY 0.24+/-0.19, AAPL 0.21+/-0.17, IBM 0.24+/-0.20, QQQ 0.30+/-0.26) and CMEM comparison per ticker

### F52. Sanity checks suite
**Category:** validation
**Core shared checks** (runs 3, 1, 2, 4, 6, 8, 10): phi U-shape, AR coefficients in (0,1), robust vs standard equivalence on clean data, log-likelihood/Q monotonically non-decreasing, synthetic data parameter recovery, beat rolling mean.
**Approach A** (run 3): Seven checks. Unique: innovation whiteness (ACF). Synthetic params: a_eta=0.98, a_mu=0.5, sigma_eta^2=0.01, sigma_mu^2=0.05, r=0.05, I=26, T=500.
**Approach B** (run 1): Eight checks. Unique: dynamic < static MAPE monotonicity, contaminated data test (inject 10x/0.1x outliers in 10% of bins), per-ticker MAPE reference values.
**Approach C** (run 2): Eight checks. Unique: EM from 5 random initializations converges to same values, prediction variance grows with horizon h, VWAP weights sum to 1.
**Approach D** (run 5): Ten checks. Combines all unique checks from runs 3, 1, and 2 into a single suite: synthetic recovery (a_eta=0.98, a_mu=0.5, sigma_eta_sq=0.01, sigma_mu_sq=0.05, r=0.05, I=26, T=500), log-lik monotonicity, robust=standard on clean data, AR in (0,1), phi U-shape, beat rolling mean, multi-init convergence, contaminated data (10x spikes in 10% bins), prediction variance grows with h, VWAP weights sum to 1.
**Approach E** (run 4): Nine checks. Core overlap with prior runs. Unique additions not in run 5's list: (1) state component behavior check — verify eta approximately constant within day with smooth cross-day changes, mu fluctuates around zero decaying at rate a_mu; (2) filter covariance positive-definiteness check — both diagonal elements positive and determinant positive at every tau (easy for 2x2). Also provides dynamic vs static ordering check (dynamic MAPE < static) matching run 1.
**Approach F** (run 6): Seven checks. Adds two not covered by prior approach labels: (1) forecast bias (mean forecast error ~ 0 over large out-of-sample period — systematic bias indicates calibration problem), (2) covariance positive definiteness (eigenvalues of Sigma_pred and Sigma_filt remain positive). The PD check overlaps with run 4's approach E but is independently discovered. Does not include contaminated data test or multi-init convergence.
**Approach G** (run 9): Eight checks. Core overlap with prior runs. Unique addition: (1) log-volume residual Q-Q plot for Gaussianity check -- verify residuals (y_tau - y_hat_tau) are approximately Gaussian; heavy tails in standard model should be reduced by robust model. This is distinct from run 3's innovation whiteness (ACF), which checks serial correlation, not distributional shape. Also includes SPY-specific dynamic MAPE (0.24) and static MAPE (0.36) reference values.
**Best version so far:** Run 5 — most comprehensive (10 checks), but runs 4 and 6 add the forecast bias check (unique to run 6) and state component behavior check (unique to run 4), and run 9 adds Q-Q residual normality check

### F53. Half-day session exclusion
**Category:** edge case
**Consensus** (runs 3, 1, 2, 5, 4, 6, 7, 8, 9, 10): Days with fewer than I bins excluded entirely. Seasonal vector phi[1..I] assumes fixed bins per day. Paper: Section 4.1.
**Best version so far:** Run 2 — addresses both training (exclude) and prediction scenarios (skip day or use reduced I with matched phi)

### F54. Exchange-varying I constraint
**Category:** edge case
**Consensus** (runs 3, 1, 2, 5, 4, 6, 7, 9, 10): Different exchanges have different I. Model calibrated separately per exchange/I value. Cannot mix securities with different I in a single model. Paper: Table 2.
**Best version so far:** Run 3

### F55. Known limitations
**Category:** other
**Core shared** (runs 3, 1, 2, 4, 6, 8): (1) no zero-volume bins, (2) no exogenous covariates, (3) single-security, (4) Gaussian noise assumption.
**Approach A** (run 3): Adds: AR(1) only, static prediction degrades for later bins (mu decays as (a_mu)^h), prediction intervals unvalidated.
**Approach B** (run 1): Adds: fixed bin granularity, no cross-direction comparison.
**Approach C** (run 2): Adds: linear dynamics only (= AR(1) restated), static seasonality (phi doesn't adapt to changing patterns, e.g. inverted-J on gap days), no BDF/GAS-Dirichlet benchmark.
**Approach D** (runs 5, 10): Eight limitations covering all prior entries: no zero-volume, no exogenous covariates, single-security, AR(1) only, static seasonality, static prediction degrades for later bins, fixed bin granularity, Gaussian noise assumption. Adds: robust variant partially mitigates heavy tails.
**Approach E** (run 4): Seven limitations. Core overlap with prior runs. Adds unique: (1) no uncertainty in volume-space forecasts — Gaussian log-space prediction intervals become asymmetric log-normal in volume space, and exp(y_hat) is biased estimator of expected volume (distinct from F37 bias correction; this is about the interpretation of prediction intervals, not point forecasts); (2) linear dynamics only (AR(1), no regime switching or threshold effects like BDF-SETAR).
**Approach F** (run 6): Seven limitations. Adds unique: (1) no intraday updating of daily component — eta held constant within day, model cannot revise daily level estimate from early intraday evidence until next day boundary, adapts only through mu (see F79); (2) linear state-space structure — higher-order AR or nonlinear dynamics would break closed-form EM.
**Approach G** (run 7): Seven limitations. Core overlap with prior runs. Adds unique framing: (1) rolling re-estimation scaling cost — single-security EM is fast (state dim=2) but scaling to large universe requires parallelization; paper does not report computation time; (2) no explicit regime-switching mechanism — rolling window adapts only as fast as old data exits.
**Best version so far:** Run 5 — most comprehensive list (8 items), includes all unique items from prior runs, adds detail on static prediction degradation (a_mu=0.5 gives 0.5^13 ~ 0.0001 by mid-day) and that robust variant partially mitigates heavy tails. Run 6's "no intraday updating of daily component" is a valuable new addition (see F79).

### F56. EM warm-start in rolling window
**Category:** algorithm
**Consensus** (runs 1, 5, 4, 8): In rolling window re-estimation, initialize EM with previous day's calibrated parameters (theta_prev). Dramatically reduces iterations needed (typically 2-5 vs 10-20 from cold start) because consecutive windows share most data. Run 8 adds cold-start fallback strategy: if warm-start fails to converge (LL decreases), restart from default initialization. Researcher inference; paper does not discuss.
**Best version so far:** Run 5 — included in Parameters/Initialization section with specific iteration counts (2-5 vs 10-20)

### F57. Lambda = 0 degenerate case
**Category:** edge case
**Consensus** (runs 1, 5, 6, 7, 8, 10): Setting lambda = 0 means every innovation is treated as an outlier (z* = e always, no correction to state). This is degenerate and should be prevented. Lambda must be strictly positive. Run 7 notes "lambda=0 removes all signal" in parameters table. NOTE: Run 9 incorrectly claims lambda = 0 recovers the standard KF (see F33 Approach B for analysis of the error).
**Best version so far:** Run 5 — listed in edge cases section with clear explanation

### F58. MAPE definition formula
**Category:** validation
**Consensus** (runs 1, 2, 5, 6, 7, 8, 10): MAPE = (1/M) * sum_{tau=1}^{M} |volume_tau - predicted_volume_tau| / volume_tau. M is total out-of-sample bins. Computed in volume space (not log-volume). predicted_volume_tau = exp(y_hat_tau). Paper: Section 3.3, Equation 37. Run 7 initially thought MAPE might be in log space but re-read Eq 37 to confirm linear-scale computation.
**Best version so far:** Run 5 — includes full formula in validation section, notes shares_out cancels in ratio so MAPE is same whether computed in turnover or raw volume

### F59. VWAP tracking error definition formula
**Category:** validation
**Consensus** (runs 1, 5, 4, 6, 8): VWAP_TE = (1/D) * sum_{t=1}^{D} |VWAP_t - replicated_VWAP_t| / VWAP_t. D is number of out-of-sample days. Expressed in basis points (multiply by 10000). Paper: Section 4.3, Equation 42. Run 6 includes the full VWAP and replicated_VWAP formulas (Eq 39) and notes price_{t,i} is the last recorded transaction price in bin i.
**Best version so far:** Run 5 — includes VWAP_t formula (Eq 39: price-weighted volume average), replicated_VWAP_t formula (sum of w*price), notes backtesting assumption that strategy executes at bin closing price

### F60. VWAP tracking error benchmarks (static)
**Category:** validation
**Consensus** (runs 1, 5, 4, 6, 7, 8, 10): Robust KF: 6.85 bps, Standard KF: 6.89, CMEM: 7.71, RM: 7.48. For AAPL static TE: Robust KF = 4.99, KF = 4.97, CMEM = 5.88, RM = 5.84. Paper: Table 4.
**Best version so far:** Run 5 — formatted table with both dynamic and static TE benchmarks in the same validation section

### F61. Robust EM log-likelihood is convergence heuristic
**Category:** algorithm
**Consensus** (runs 1, 4, 6, 10): The innovation-form log-likelihood in robust EM is a practical convergence monitor, not the exact observed-data log-likelihood. Because z* is a deterministic function of e (soft-thresholding), e_corrected has a truncated distribution, not standard Gaussian. EM convergence is guaranteed by the E-step/M-step structure regardless. Paper does not discuss; researcher inference. Run 4 adds: the robust EM is not a standard EM (it alternates Lasso subproblem for z* with closed-form theta updates), so monotonic LL increase may not strictly hold due to the Lasso penalty. Run 6 adds: cleaned-innovation LL formula LL_robust = -0.5 * sum[log(S_tau) + e_clean^2/S_tau + log(2*pi)], and a parameter-change fallback criterion max(|theta_new - theta_old| / (|theta_old| + epsilon)) < tol.
**Best version so far:** Run 4 — most precise characterization of why LL is heuristic (Lasso penalty breaks standard EM guarantee). Run 6 adds the specific cleaned-innovation LL formula and parameter-change fallback, making the robust EM convergence fully specified.

### F62. Data types and shapes table
**Category:** other
**Consensus** (runs 1, 2, 5, 6, 7, 8, 10): Explicit type/shape specification for all model quantities. Run 7 provides complete types/shapes list with float64 annotations and notes that K is computed online (not stored), z_star is robust-only.
**Best version so far:** Run 5 — comprehensive markdown table with Type/Shape/Description columns for all quantities: y (N,), observed (N, bool), phi_position (N, int), x_filt/x_pred/x_smooth (N,2), Sigma variants (N,2,2), A_used/L_stored (N-1,2,2), P/P_cross (N/N-1,2,2), e/S/z_star (N,), K (N,2), phi (I,), pi_1 (2,), Sigma_1 (2,2), scalars

### F63. Float64 precision recommendation
**Category:** edge case
**Consensus** (runs 1, 5, 9): After normalization by shares outstanding, small normalized volumes are amplified by log transform. Use float64 throughout to ensure adequate precision. Run 5 adds that long EM iteration chains accumulate rounding errors.
**Best version so far:** Run 5 — listed in edge cases with additional rationale about EM iteration accumulation

### F64. Fixed bin granularity limitation
**Category:** other
**Consensus** (runs 1, 5, 6): The model assumes a fixed number of bins I per day. Adaptive or irregular time grids are not supported. Listed as a known limitation. Run 6 extends: "variable-length bins (e.g., auctions, irregular intervals) are not supported."
**Best version so far:** Run 6 — explicitly names auctions and irregular intervals as unsupported use cases

### F65. No cross-direction benchmarking
**Category:** other
**Consensus** (runs 1, 2, 6, 10): Paper benchmarks against CMEM and rolling means only. Relative performance against BDF (Direction 2) or GAS-Dirichlet (Direction 3) is unknown. Paper: Section 5. Run 6 mentions "Unlike the BDF model (Direction 2), this model operates on a single security at a time."
**Best version so far:** Run 2 — also mentions this in context of known limitations (Section 4)

### F66. Q-function explicit E-term formulas
**Category:** algorithm
**Single source** (run 2): Full Q-function from Eq A.10 decomposed into four E terms: E_1 = observation fit (Eq A.11), E_2 = mu dynamics (A.12), E_3 = eta dynamics (A.13), E_4 = initial state (A.14). Includes expanded forms: C@P@C^T = P^(1,1) + 2*P^(1,2) + P^(2,2), and equivalent Sigma_smooth-based form E_1 = (1/(2r))*sum[(y-phi-C@x_hat)^2 + C@Sigma_smooth@C^T]. Full log-constant terms: -(N/2)log(r) - ((N-1)/2)log(sigma_mu^2) - ((T-1)/2)log(sigma_eta^2) - (1/2)log|Sigma_1| - ((2N+T)/2)log(2*pi).
**Best version so far:** Run 2

### F67. Post-M-step Q simplification
**Category:** algorithm
**Single source** (run 2): After M-step updates, E terms simplify: E_1 = N/2 (since r_new zeroes the derivative), E_2 = (N-1)/2, E_3 = (T-1)/2, E_4 = 1 (since Sigma_1 = Sigma_smooth[1] and pi_1 = x_hat[1], giving tr(I_2)/2 = 1). Post-M-step Q reduces to constant terms plus log-determinant terms only. Convergence can therefore be monitored by checking log(r), log(sigma_mu^2), log(sigma_eta^2), log|Sigma_1| alone. Researcher inference (follows from M-step optimality conditions; not stated in paper).
**Best version so far:** Run 2

### F68. Dynamic VWAP: backtesting vs production mode
**Category:** algorithm
**Consensus** (runs 2, 5, 4, 6, 9, 10): Backtesting mode: vol_hat_dynamic[i] = one-step-ahead forecast for bin i, collected after running the filter through the full day. All I forecasts available before calling the weight function. This matches the paper's evaluation (Section 4.3, Table 4). Production mode: at each bin i, filter corrected through bin i-1, then multi-step-ahead forecasts for bins i..I. Cumulative_weight for bins 1..i-1 tracked externally across calls, since actual execution weights (not forecast-based) are already committed. The mathematical formula (Eq 41) is identical in both modes; only the contents of vol_hat_dynamic differ. Run 6's dynamic_vwap_weights function implements the production-mode pattern with weights_so_far as input.
**Best version so far:** Run 5 — production mode fully implemented in dynamic VWAP pseudocode (Step 11) with explicit step separation between one-bin-ahead correction (Step 9) and multi-step-ahead remaining forecast (Step 10)

### F69. Phi identifiability — no sum-to-zero constraint
**Category:** algorithm
**Consensus** (runs 2, 5): Both eta and phi contribute to the mean level of y. Adding constant c to all phi[i] and subtracting from eta leaves observation equation unchanged. The EM updates for eta and phi jointly determine the split: phi absorbs per-bin average residual (Eq A.39), eta absorbs daily level. No explicit sum-to-zero or other constraint on phi is needed — EM updates are self-consistent. WARNING: do NOT impose sum(phi)=0, as this would conflict with the EM updates. Researcher inference on the constraint discussion; implicit in paper's model structure.
**Best version so far:** Run 5 — includes the identifiability discussion in the model description section with explicit note that phi captures per-bin residual after removing C@x_smooth, and that no explicit constraint is needed or desirable

### F70. Winsorization of extreme log-volumes
**Category:** edge case
**Consensus** (runs 2, 5): After normalization by shares outstanding, volumes can span several orders of magnitude. Log transform handles this, but extreme values in log space can still affect EM convergence. Consider winsorizing log-volumes at +/- 5 standard deviations before calibration. Researcher inference.
**Best version so far:** Run 5 — listed in edge cases section (#6)

### F71. Lambda cross-validation boundary check
**Category:** edge case
**Consensus** (runs 2, 5): If the optimal lambda is at the boundary of the search grid, the grid should be extended. lambda -> 0 optimal suggests severe outlier issues requiring investigation. lambda -> inf optimal means standard KF suffices. Researcher inference.
**Best version so far:** Run 5 — listed in edge cases section (#4)

### F72. Missing data denominator principle
**Category:** algorithm
**Consensus** (runs 5, 10): State-dynamics parameters (a_eta, a_mu, sigma_eta_sq, sigma_mu_sq) sum over ALL bins including missing ones, because the Kalman filter prediction step fills in unobserved positions and provides smoothed estimates P[tau]/P_cross[tau] for every tau. Observation-equation parameters (phi, r) involve actual y[tau] which is undefined for missing bins, so they sum only over observed bins with N_obs (or per-bin observed count) as denominator instead of N (or T). Formulas follow from standard EM-with-missing-data theory; consistent with paper's Appendix A which assumes complete data.
**Best version so far:** Run 5

### F73. EM convergence check before M-step / skip final M-step
**Category:** algorithm
**Consensus** (runs 5, 10): The convergence check (Step 6) is placed AFTER the E-step and BEFORE the M-step in the EM loop. When the loop exits due to convergence at iteration j, theta_final holds M-step parameters from iteration j-1, and x_filt[N]/Sigma_filt[N] are from the E-step of iteration j run with those same parameters. These are consistent. Skipping the final M-step is standard EM practice. The filtered state at the last bin is the starting point for prediction.
**Best version so far:** Run 5

### F74. Unified M-step for standard and robust variants
**Category:** algorithm
**Consensus** (runs 5, 6, 7, 8, 10): The M-step pseudocode handles both standard and robust variants with a single code path. z_star[tau] is initialized to 0 for all tau at the start of the Kalman filter. For standard mode (lambda=infinity), z_star remains zero throughout, so the z_star terms in the phi and r updates vanish and reduce to the standard Eqs A.38-A.39. For robust mode, z_star[tau] is set by soft-thresholding during the filter correction step. All other M-step equations (a_eta, a_mu, sigma_eta_sq, sigma_mu_sq, pi_1, Sigma_1) are unchanged because z_star affects only the observation equation.
**Best version so far:** Run 5

### F75. Dynamic VWAP uses pre-correction state for weight computation
**Category:** algorithm
**Single source** (run 5): In the dynamic VWAP procedure (Step 11), the volume forecasts used in the weight ratio w[i] = volume_hat_i / vol_remaining are all conditioned on the same information set (observations up to bin i-1). Specifically, volume_hat_i and volume_hat_remaining[i+1..I] are both computed from x_pred_i (the predicted state at bin i BEFORE correcting with y_live[i]), not from x_filt_updated (the post-correction state). This matches the paper's Eq 41. The post-correction state x_filt_updated is used only as the starting state for the NEXT iteration (bin i+1).
**Best version so far:** Run 5

### F76. Paper-to-code reindexing convention
**Category:** index convention
**Single source** (run 5): The paper's Algorithm 1 uses notation where the prediction step at time tau produces x_{tau+1|tau} and Sigma_{tau+1|tau}. The implementation reindexes so that x_pred[tau] is the prediction FOR time tau (produced from state at tau-1). Correspondence: code's x_pred[tau] = paper's x_{tau|tau-1}. This is more natural for implementation (x_pred[tau] and x_filt[tau] share the same time index tau).
**Best version so far:** Run 5

### F77. Comprehensive paper references table
**Category:** other
**Consensus** (runs 5, 4, 6, 7, 8, 10): Full mapping from spec sections to paper sources (35+ entries), covering: model decomposition (Section 2, Eq 3), state-space (Section 2, Eqs 4-5), Kalman filter (Algorithm 1), RTS smoother (Algorithm 2), all M-step equations (Appendix A, Eqs A.32-A.39), robust model (Section 3.1, Eqs 25-34), robust M-step (Eqs 35-36), MAPE (Eq 37), VWAP formulas (Eqs 39-42), and researcher inference items. Run 6 provides an explicit table format mapping each spec section to its paper source (30+ rows).
**Best version so far:** Run 5

### F78. Cross-validation procedure with two-level structure
**Category:** algorithm
**Consensus** (runs 5, 4, 6, 7, 8, 10): Calibration has two explicit levels. Level 1: EM parameter estimation for fixed (N_days, lambda). Level 2: Cross-validation over (N_days, lambda) grid. Validation period = 100 trading days before out-of-sample. For each (N_days, lambda) pair: train EM on training window, evaluate dynamic MAPE on validation period using rolling window (daily re-estimation). Select pair minimizing validation MAPE. Paper source: Section 4.1 ("data between January 2015 and May 2015 are considered as a cross-validation set"). Run 6 gives "5 months" as typical validation period length, matching the paper's January-May 2015 window.
**Best version so far:** Run 5

### F79. No intraday updating of daily component
**Category:** edge case
**Consensus** (runs 6, 10): Within a trading day, eta is held strictly constant (A[tau] has 1 in the (1,1) position for within-day transitions, Q[tau] has 0 in the (1,1) position). The model cannot revise its estimate of the daily level based on intraday evidence until the next day boundary. If the first few bins reveal that today's volume is much higher or lower than expected, the model adapts only through mu (intraday dynamic), not through eta. This is a structural consequence of the state-space formulation (Paper, Section 2, Equations 4-5) and a meaningful limitation for days with regime shifts (e.g., surprise earnings, macro announcements).
**Best version so far:** Run 6

### F80. Day-boundary transition in dynamic VWAP at first bin
**Category:** edge case
**Consensus** (runs 6, 10): In the dynamic VWAP function, when current_bin == 1, x_filt_prev is the filtered state from the last bin of the previous day. The first prediction step therefore crosses a day boundary and must use A = [[a_eta, 0], [0, a_mu]] (day-boundary dynamics) instead of the within-day matrix A = [[1, 0], [0, a_mu]]. This is consistent with static_forecast (which uses day-boundary at h==1) and Algorithm 1 (which applies day-boundary dynamics when tau is in D). The practical impact is small (a_eta ~ 1.0), but omitting it is a correctness defect. Identified via adversarial review in run 6's draft 4 revision. (Paper, Section 2, Equations 4-5.)
**Best version so far:** Run 6

**Run 5 audit:** VERDICT: [7 new, 2 competing, 62 reinforcing] — new findings still emerging (missing data handling, VWAP info-set consistency, code-level design patterns), continue
**Run 4 audit:** VERDICT: [0 new, 5 competing, 73 reinforcing] — no new unique findings; 5 competing approaches add value but diminishing returns, consider stopping
**Run 6 audit:** VERDICT: [2 new, 2 competing, 55 reinforcing] — diminishing returns, new findings are minor edge cases, consider stopping

### F81. Minimum training window constraint
**Category:** edge case
**Approach A** (run 7): If N < 2*I (fewer than 2 trading days), there are insufficient day-boundary transitions to estimate a_eta and sig2_eta (the D_start set would be empty). The minimum sensible training window is ~20 trading days. Researcher inference; paper does not state this explicitly but it follows from the M-step formulas for a_eta (Eq A.34) and sig2_eta (Eq A.36) which sum over D_start = {kI+1 for k=1,...,T-1}.
**Approach B** (run 8): Minimum recommended ~50 trading days (roughly 2.5 months). With too few training days, the EM may not reliably estimate all parameters (especially sigma_eta^2 which only receives one data point per day boundary). More conservative practical recommendation vs run 7's mathematical minimum.
**Best version so far:** Run 7 -- provides the mathematical lower bound (T >= 2); run 8's ~50 day recommendation is a practical guideline for reliable estimation

**Run 7 audit:** VERDICT: [1 new, 1 competing, 55 reinforcing] — no meaningful new contributions, stop

### F82. Corporate actions / shares outstanding changes
**Category:** edge case
**Single source** (run 8): Abrupt changes in shares outstanding (stock splits, secondary offerings) cause level shifts in normalized volume (turnover = raw_volume / shares_outstanding). The daily component eta should absorb gradual shifts, but sudden large changes may trigger the robust filter's outlier detection or require re-calibration. Researcher inference; paper does not discuss.
**Best version so far:** Run 8

**Run 8 audit:** VERDICT: [1 new, 1 competing, 57 reinforcing] — no new contributions, stop

### F83. Outlier fraction monitoring diagnostic
**Category:** edge case
**Single source** (run 9): If a large fraction of bins are flagged as outliers (z_tau* != 0 for many consecutive bins), the filter effectively ignores observations and runs open-loop. This can happen if lambda is too small. Monitor the fraction of nonzero z_tau* values; if it exceeds ~20%, lambda may need to be increased. Distinct from F57 (lambda=0 degenerate) and F71 (lambda at CV boundary): this is a runtime diagnostic for production use, not a calibration concern. The 20% threshold is researcher inference.
**Best version so far:** Run 9

**Run 9 audit:** VERDICT: [1 new, 1 competing, 57 reinforcing] — no new contributions, stop

### F84. Lambda grid construction heuristic based on threshold interpretation
**Category:** parameter
**Single source** (run 10): The effective outlier threshold is lambda*S_tau/2, and S_tau approximately equals r when the state is well-known. To detect outliers at k standard deviations, set lambda such that lambda*r/2 = k*sqrt(r), giving lambda = 2*k/sqrt(r). This yields a principled grid: lambda in {0.1, 0.5, 1.0, 2.0, 5.0, 10.0} * sqrt(r_init), where r_init is estimated from the initial EM iteration or sample variance of residuals. Covers thresholds from very aggressive (0.1*sqrt(r), catching innovations beyond ~0.05 SD) to very conservative (10*sqrt(r), catching only beyond ~5 SD). Distinct from F45's fixed or log-spaced grids: this scales the grid to the data's noise level. Also provides fallback coarse grid {0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0} when r is not yet estimated. Researcher inference; paper omits lambda selection details entirely.
**Best version so far:** Run 10

**Run 10 audit:** VERDICT: [1 new, 0 competing, 62 reinforcing] — no new contributions, stop

**Run 10 audit:** VERDICT: [1 new, 0 competing, 62 reinforcing] — no new contributions, stop
