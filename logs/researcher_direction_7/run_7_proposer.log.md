## Status: done
## Current Step: Draft 3 complete

### Log

[2026-04-10 13:09] (Previous session) Started proposer for direction 7 run 7. Read paper summary and full PDF. Produced draft 1.

[2026-04-10 13:14] (Previous session) Draft 1 completed but subsequently cleaned up (file deleted).

[2026-04-10 13:41] Restarted. Directory empty; re-read paper summary, full PDF (16 pages), and tracker.md (81 findings across 7 runs). Producing fresh draft 1 incorporating best practices from all runs, especially run 5 (consistently best version).

[2026-04-10 13:42] Key design decisions for this draft:
- Non-recursive cross-covariance: Sigma_cross[tau] = Sigma_smooth[tau] @ L[tau-1]^T (F14, Approach B)
- Sigma_1 = Sigma_smooth[1] to avoid catastrophic cancellation (F19, Approach B)
- Unified standard/robust code path via z_star=0 initialization (F74)
- Joseph form for covariance update (F10)
- Innovation log-likelihood for convergence monitoring (F27)
- M-step ordering: phi before r, a_eta before sig2_eta, a_mu before sig2_mu (F17)
- Missing observation handling in filter pseudocode (F11)
- N_obs denominators for observation-equation parameters (F72)
- Minimum training window N >= 2*I constraint (F81)

[2026-04-10 13:42] Writing implementation specification.

[2026-04-10 13:48] Draft 1 complete. Specification covers:
- 13 pseudocode steps: preprocessing, EM init, Kalman filter, RTS smoother, sufficient
  statistics, M-step, convergence monitoring, EM main loop, static prediction, dynamic
  prediction, remaining-bin forecast, dynamic VWAP, static VWAP, rolling-window CV
- Unified standard/robust code path via z_star=0 (lambda=1e10 for standard recovery)
- Non-recursive cross-covariance formula (Shumway & Stoffer, avoids A.20-A.21 recursion)
- Sigma_1 = Sigma_smooth[1] (avoids PSD cancellation from Eq A.33)
- Joseph form covariance update for numerical stability
- Missing observation handling integrated into filter and M-step (N_obs denominators)
- All 3 M-step ordering constraints documented (phi before r, a_eta before sig2_eta, a_mu before sig2_mu)
- Complete parameter table (13 parameters)
- 10 sanity checks, 9 edge cases, 8 known limitations
- 42-entry paper reference table with researcher inference items marked
- Data flow diagram with types/shapes table

[2026-04-10 13:53] Received critique 1: 5 major, 8 minor issues.
Assessment:
- M1 (VALID, Major): Filter initialization convention mismatch. Paper's Algorithm 1
  sets x_filt[1]=pi_1 and loops tau=1..N-1 predicting tau+1, correcting with y_{tau+1}.
  My draft treats pi_1 as x_pred[1] and applies correction with y[1]. Verified against
  paper page 4 Algorithm 1: loop starts "for tau=1,2,..." with "predict x_{tau+1|tau}"
  and "correct x_{tau+1|tau+1}" using y_{tau+1}. Will restructure Step 2.
- M2 (NOT APPLICABLE): Critique references two conflicting cross-covariance formulations
  at lines 247-258, but draft 1 has only the non-recursive formula (Shumway & Stoffer).
  Line numbers don't match draft content. No change needed.
- M3 (NOT APPLICABLE): Critique says L_stored not returned, but draft 1 line 216
  explicitly includes L_stored in output signature and line 236 stores it. No change.
- M4 (NOT APPLICABLE): Critique references "Wait" comments and self-corrections at
  lines 888-893 and 968-978. Draft 1 has no such artifacts. No change needed.
- M5 (PARTIALLY VALID): Code IS unified via z_star=0, but will add explicit clarifying
  notes about robust/standard interaction in the EM loop.
- m1 (Valid): Will add efficiency note for dynamic VWAP.
- m2 (Valid): Will add lambda grid scaling guidance.
- m3 (Valid): Will add MAPE and VWAP TE evaluation pseudocode.
- m4 (Already addressed): Step 2 already has missing observation branch.
- m5 (Valid): Will fix RM dynamic TE table entry to N/A.
- m6 (Already addressed): N_candidates in Parameters section.
- m7 (Valid): Will add production caching note.
- m8 (Valid): Will add floor for Sigma_1 initialization.

Summary: 1 real major fix (M1), 3 inapplicable major issues (M2-M4), 1 clarification (M5),
several valid minor improvements. Writing draft 2.

[2026-04-10 13:53] Verified M1 against paper page 4, Algorithm 1. Confirmed: loop
"for tau=1,2,...do" predicts x_{tau+1|tau} and corrects with y_{tau+1}. First correction
is y_2. x_{1|1} = pi_1 is given. y_1 enters only through M-step (Eqs A.38-A.39 sum
over tau=1..N) and indirectly through smoother backward pass.

[2026-04-10 13:54] Draft 2 changes from draft 1:
1. M1 FIX: Restructured Step 2 (Kalman Filter). x_filt[1] = pi_1, Sigma_filt[1] = Sigma_1
   (no correction). Loop now runs tau=2..N. Added explanation box about paper convention.
   Updated output signatures: x_pred, Sigma_pred, e, S are now indexed 2..N. Updated
   Step 6 (convergence LL) to sum over tau=2..N. Updated Data Flow shapes table.
2. M5 CLARIFY: Added explicit "Robust/standard unification" paragraph to Step 7 explaining
   how lambda=1e10 makes robust code path identical to standard.
3. m1: Added efficiency note to Step 11 confirming O(I^2) is acceptable.
4. m2: Added lambda grid construction guidance in Parameters section with data-adaptive
   scaling via median(sqrt(S)).
5. m3: Added Step 13 (Evaluation Metrics) with MAPE and VWAP TE pseudocode. Step 14 is
   now rolling-window CV (was Step 13).
6. m5: Fixed RM entries to N/A in both MAPE and VWAP TE tables (RM has no dynamic variant).
7. m7: Added production caching note to Step 9 (Dynamic Prediction).
8. m8: Added floor for sig2_eta in Step 1: max(..., 0.01 * var(daily_avg)).
9. Added note to Step 5 explaining how y[1] contributes to M-step despite no filter
   correction.
10. Updated Paper References table with initialization convention, M-step equation numbers,
    and RM dynamic variant note.

[2026-04-10 13:54] Draft 2 complete.

[2026-04-10 14:14] Received critique 2: 0 major, 7 minor issues.
Assessment: All issues are documentation clarifications or missing helper function
definitions. No algorithmic errors. Spec is implementation-ready with these small
improvements. All 7 minor:
- m1 (VALID): DynamicPredictDay undefined. Added full pseudocode definition with
  explicit inputs (x_filt_last, Sigma_filt_last) and loop wrapping Step 9.
- m2 (VALID): extract_window undefined. Added definition specifying window ends at
  day d-1, returns y/observed/phi_position arrays.
- m3 (VALID): Dynamic VWAP missing end-of-day state. Added note with extra
  DynamicPredict call after loop.
- m4 (VALID): Robust EM convergence fallback unspecified. Added concrete pseudocode:
  if LL decreases, check parameter-change criterion instead.
- m5 (VALID): Cross-covariance cited as "1982 paper" but Property 6.3 is from the
  textbook. Fixed to "Shumway and Stoffer (2006 textbook, Property 6.3)".
- m6 (VALID): A_used index convention differs from paper. Added mapping note to Step 3:
  paper's A_tau = spec's A_used[tau+1].
- m7 (VALID): N_obs/count_i denominators not labeled as researcher inference. Added
  "(researcher inference)" labels.

[2026-04-10 14:14] Draft 3 changes from draft 2:
1. m1+m2: Added 3 helper functions after Step 14 loop: extract_window, end_of_training_state,
   DynamicPredictDay. Updated Step 14 call signature to pass filtered state explicitly.
2. m3: Added "End-of-day state note" paragraph after Step 11 loop with extra DynamicPredict call.
3. m4: Added fallback convergence check in Step 7 EM loop: if LL decreases, check max
   relative parameter change instead. Marked as researcher inference.
4. m5: Changed all cross-covariance citations from "Shumway and Stoffer (1982)" to
   "Shumway and Stoffer (2006 textbook)". Left innovation LL citation as 1982 (correct).
5. m6: Added index convention mapping note to Step 3 source paragraph.
6. m7: Added "(researcher inference)" labels to count_i and N_obs comments in Step 5.

[2026-04-10 14:14] Draft 3 complete.
