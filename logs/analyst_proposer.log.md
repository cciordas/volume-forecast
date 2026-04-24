## Status: done
## Current Step: Draft 2 complete (revision based on critique 1)

### Log

[2026-04-10 09:28] Starting proposer session. No prior drafts found in work/analyst/. This is an initial clustering round (independent run).

[2026-04-10 09:28] Read all 9 paper summaries from papers/*.md and the paper manifest. All summaries are complete and well-structured. The Satish et al. 2014 summary correctly describes the FlexTrade four-component model (raw volume + volume percentage forecasting).

[2026-04-10 09:28] Paper-by-paper clustering assessment:

1. Brownlees et al. 2011 (CMEM): Foundational model — multiplicative decomposition into daily/periodic/dynamic components, GMM estimation. Benchmark model for the field. Groups with Naimoli & Storti 2019 which directly extends CMEM.

2. Naimoli & Storti 2019 (H-MIDAS-CMEM): Extends CMEM by replacing daily GARCH-type component with heterogeneous MIDAS cascade at daily+hourly frequencies. Same model family as CMEM — a developer implements the base CMEM first, then adds the MIDAS long-run extension. Groups with Direction 1.

3. Bialkowski et al. 2008 (BDF): Additive PCA factor decomposition — cross-sectional common component (market factors) + stock-specific ARMA/SETAR residual. Fundamentally different approach from CMEM: additive vs multiplicative, cross-sectional vs single-stock, PCA vs GMM. Own direction.

4. Szucs 2017: Head-to-head comparison of CMEM vs BDF on 11 years DJIA data. No novel model introduced. Comparison paper — belongs in both CMEM and BDF directions.

5. Satish et al. 2014: FlexTrade practitioner model — four-component weighted combination (rolling average + inter-day ARMA + intraday ARMA + regime-switching weight overlay) plus a separate volume percentage model extending Humphery-Jenner. Distinct algorithm from all others. Own direction.

6. Markov et al. 2019 (Quintet): Five coordinated Bayesian sub-models (daily prior, U-curve FDA, close auction, liquid bin-level Bayesian updater, illiquid cumulative updater). Distinct ensemble architecture with Bayesian updating. Own direction.

7. McCulloch 2007: Doubly stochastic binomial point process — Cox process framework for relative cumulative volume. Unique probabilistic foundation, nonparametric moment estimation. Own direction.

8. Calvori et al. 2014 (GAS-Dirichlet): Score-driven Dirichlet model for volume shares with GAS dynamics. Different model class from CMEM despite shared authors (Dirichlet distribution, score-driven updates, shares not levels). Own direction.

9. Chen et al. 2016 (Kalman Filter): Log-space additive state-space model with Kalman filter recursions and EM calibration, plus Lasso-robust extension. Different estimation framework from CMEM despite similar three-component decomposition idea. Own direction.

[2026-04-10 09:28] Cross-direction paper assignments:
- Szucs 2017 -> Comparison in Direction 1 (CMEM) and Direction 2 (BDF)
- Chen et al. 2016 -> Foundational in Direction 7 (Kalman); also Comparison in Direction 1 (CMEM) since it benchmarks against CMEM extensively across 30 securities
- Markov et al. 2019 -> Foundational in Direction 5 (Quintet); also Supporting in Direction 4 (Satish) since it references Satish's conceptual similarity and provides context for bin-level ARMA approach

[2026-04-10 09:28] Self-check: Each direction maps to a single model that a developer would implement as one codebase. No direction contains papers describing fundamentally different algorithms. Proceeding to write draft.

[2026-04-10 09:33] Draft 1 written to work/analyst/research_directions_draft_1.md. 7 directions identified:
1. CMEM (Brownlees 2011 foundational + Naimoli 2019 refinement + Szucs 2017 comparison + Chen 2016 comparison)
2. PCA Factor Decomposition / BDF (Bialkowski 2008 foundational + Szucs 2017 comparison)
3. GAS-Dirichlet Volume Share Model (Calvori 2014 foundational)
4. Weighted Component Volume Forecast (Satish 2014 foundational + Markov 2019 supporting)
5. Quintet Bayesian Ensemble (Markov 2019 foundational)
6. Doubly Stochastic Binomial Point Process (McCulloch 2007 foundational)
7. Kalman Filter State-Space Model (Chen 2016 foundational)

Key decisions:
- CMEM and H-MIDAS-CMEM grouped together: same model family, H-MIDAS extends the long-run component
- GAS-Dirichlet separate from CMEM despite shared authors: different model class (Dirichlet/score-driven vs MEM/GMM, shares vs levels)
- Kalman filter separate from CMEM despite similar decomposition idea: different estimation framework (linear state-space/EM vs nonlinear/GMM)
- Quintet separate from Satish: different algorithms (Bayesian ensemble of 5 models vs weighted ARMA combination)
- McCulloch as own direction: unique probabilistic framework (Cox process) with no close relatives
- Satish summary now correctly reflects the FlexTrade paper (four-component + volume percentage model)
- Chen 2016 assigned as Comparison in CMEM direction (benchmarks CMEM across 30 securities)
- Markov 2019 assigned as Supporting in Satish direction (references conceptual similarity)

[2026-04-10 09:33] Session complete. Draft 1 ready for critic review.

[2026-04-10 09:38] Resuming for revision round. Critique 1 exists with 0 major, 4 minor issues.

[2026-04-10 09:38] Critique assessment:
- Issue 1 (minor): Remove Markov 2019 from Direction 4. Valid — the overlap is surface-level ("both forecast bin-level volume"), not algorithmic. Will remove.
- Issue 2 (minor): Add Chen 2016 to Direction 4 as Comparison. Valid — Chen explicitly benchmarks against Satish 2014. Will add.
- Issue 3 (minor): Note Satish 2014 PDF mismatch. Valid — flagging for downstream users. Will add note to implementation notes.
- Issue 4 (minor): Direction 4 contains two models, name is misleading. Valid — the raw volume and percentage models are tightly coupled (surprise signal feeds from raw to percentage) but the direction name only described one. Will rename to "Dual-Mode Volume Forecast (Raw + Percentage)" and clarify the two-model scope in the core model description.

[2026-04-10 09:38] Revisions applied to draft 2:
1. Removed markov_vilenskaia_rashkovich_2019 from Direction 4 papers table.
2. Added chen_feng_palomar_2016 as Comparison to Direction 4 papers table.
3. Added PDF data integrity note to Direction 4 implementation notes.
4. Renamed Direction 4 from "Weighted Component Volume Forecast" to "Dual-Mode Volume Forecast (Raw + Percentage)".
5. Rewrote Direction 4 core model description to explicitly describe both models and their coupling.
6. Updated summary table to reflect new Direction 4 name and description.
7. Updated adversarial refinement rounds to 1.

[2026-04-10 09:38] Draft 2 written to work/analyst/research_directions_draft_2.md. All 4 critique issues addressed. No structural changes to direction count or other directions. Session complete.
