# Findings Tracker: Direction 4 — Dual-Mode Volume Forecast (Raw + Percentage)

## Summary
- Total findings: 94
- Runs processed: [1, 2, 3, 4, 5, 6, 7]
- Last update: run 7 (re-audit)

## Findings

### F1. Overall architecture: dual-model system
**Category:** algorithm
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): Two-model system based on Satish, Saxena, and Palmer (2014). Model A produces raw bin-level volume forecasts combining components (historical rolling average H, inter-day ARMA D, intraday ARMA A) with regime-dependent weights. Model B produces next-bin volume percentage forecasts extending Humphery-Jenner (2011) dynamic VWAP framework, using volume surprises from Model A in a single rolling regression to adjust participation rates. Models are tightly coupled: Model A provides the surprise signal driving Model B. Run 2 frames the regime-switching weight overlay as a distinct "Component 4," making four components total; run 1 treats weights as part of the combination formula (three components plus regime selection). Runs 3, 4, 5, 6, and 7 describe three components + dynamic weight overlay, consistent with run 1's framing. Functionally identical across all runs. Run 5 organizes as 10 functions + 6 helpers with complete pseudocode. Run 6 organizes as 12 functions (7 Model A, 2 Model B, 3 orchestration/training) with Function 9a (calibrate_adaptive_limits) and generate_model_a_training_forecasts helper for sophisticated variant. Run 7 organizes as 11 functions (7 Model A, 2 Model B, 2 orchestration) across 4 drafts of adversarial refinement, with explicit ARMA model interface specification.
**Best version so far:** Run 5 — most mature pseudocode (5 drafts of adversarial refinement); explicit function signatures, data flow diagram with shapes/types table, daily orchestration, re-estimation schedule, and 6 fully specified helper functions (BuildCumVolDistribution, EvaluateWeights, compute_H_asof, predict_interday, predict_intraday_for_training, compute_hist_pct)

### F2. Bin structure: 15-minute intervals, I = 26 bins
**Category:** parameter
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): 26 bins per 6.5-hour U.S. equity trading day (9:30-16:00 ET), 15-minute bin width. Paper also tested 5 and 30 minutes but 15 minutes received most thorough treatment. Alternative I values: {13, 26, 78} for {30, 15, 5}-min bins. 15-minute shows largest improvement (Exhibit 9: 7.55% median reduction vs. 2.25% and 2.95%).
**Best version so far:** Run 2 — includes the comparative improvement statistics

### F3. Raw volume forecast formula
**Category:** algorithm
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): V_hat(s, t, i) = w1(r)*H(s,t,i) + w2(r)*D(s,t,i) + w3(r)*A(s,t,i), where r is regime index selected dynamically based on historical percentile of cumulative volume observed so far on day t. Non-negative clamp applied (V_hat = max(V_hat, 0)). (Satish et al. 2014, p.17-18, Exhibit 1.)
**Best version so far:** Run 5 — ForecastRawVolume (Function 6) with explicit component computation, regime lookup, non-negativity clamp, detailed purity documentation, and inline conditioning mechanism description (reset-and-reprocess for intraday ARMA)

### F4. Volume percentage forecast formula
**Category:** algorithm
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): p_hat(t, i) = p_hist(i) + delta(t, i), where p_hist(i) is historical average volume percentage for bin i, delta(t, i) is correction from a rolling regression on recent volume surprises, subject to deviation and switch-off constraints. Runs 3, 4, 5, and 7 apply delta to a scaled baseline incorporating renormalization into the main formula. Run 5 provides 8-step ForecastVolumePercentage (Function 7). Run 7 provides 7-step forecast_volume_percentage (Function 8) with explicit V_total_est, percentage-space surprises, deviation clamping against unscaled base_pct, then uniform scaling of (base + delta). (Satish et al. 2014, p.18-19.)
**Best version so far:** Run 5 — ForecastVolumePercentage (Function 7) with 8 clearly numbered steps, provenance annotations for Humphery-Jenner components, and inline commentary on design decisions

### F5. Seasonal factor computation
**Category:** algorithm
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): seasonal_factor[i] = arithmetic_mean(volume[s, d, i] for d in trailing 126 trading days). Per-bin arithmetic mean over 6-month window. Run 3 includes floor: seasonal_factor[i] = max(mean(volumes), epsilon). Runs 4, 5, and 7 use min-nonzero replacement (see F7). Run 5 adds InsufficientDataError if all bins have zero seasonal factor. Run 7 adds absolute fallback min_nonzero = 1.0 when all values are zero. (Satish et al. 2014, p.17 para 3.)
**Best version so far:** Run 5 — ComputeSeasonalFactors (Function 1) with min-nonzero replacement, explicit half-day exclusion, and error handling for all-zero case

### F6. N_seasonal = 126 trading days (6 months)
**Category:** parameter
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): Trailing 6-month window for seasonal factors. Sensitivity: low. Range: [63, 252]. If fewer than 6 months available, use all available data (minimum 63 trading days). (Satish et al. 2014, p.17.)
**Best version so far:** Run 2 — adds minimum data requirement

### F7. Floor for zero seasonal factors
**Category:** parameter
**Approach A** (runs 1, 3): epsilon = 1.0 share as floor. Replace seasonal_factor[i] with max(seasonal_factor[i], 1.0). Range: [0.1, 10.0]. Researcher inference.
**Approach B** (runs 2, 4, 5, 6, 7): Replace zero seasonal factors with the minimum non-zero seasonal factor across all bins for that stock, or exclude zero-volume bins from ARMA fitting entirely. Runs 4, 5, and 7 provide explicit pseudocode: min_nonzero = min(sf for sf in seasonal_factor if sf > 0); replace zeros with min_nonzero. Run 7 adds identical logic for hist_avg (Function 1a), ensuring both seasonal and historical components share the same zero-guard pattern. Researcher inference.
**Best version so far:** Run 5 — explicit pseudocode with inline rationale for data-adaptive floor and InsufficientDataError for all-zero edge case

### F8. Inter-day ARMA: one model per bin, AICc selection
**Category:** algorithm
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): I = 26 separate ARMA(p,q) models, one per bin, on daily volume series. Select by AICc. Include constant, enforce stationarity and invertibility. Runs 3, 4, 5, and 7 add explicit convergence checks (converged, stationary, invertible) and skip logic for insufficient data (n <= k+1). Run 7 adds detailed ARMA model interface specification (see F85). (Satish et al. 2014, p.17 para 2.)
**Best version so far:** Run 5 — FitInterDayARMA (Function 2) with complete guard conditions, explicit best_model tracking, and skip of constant-only (p=0,q=0) case

### F9. AICc formula: AIC + 2k(k+1)/(n-k-1)
**Category:** algorithm
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): AIC = -2*log_likelihood + 2*k, AICc = AIC + 2*k*(k+1)/(n-k-1), where k = p + q + 1 (AR params + MA params + constant), n = number of observations. Runs 3, 4, 5, and 7 add guard: skip if n <= k+1 (AICc denominator would be zero or negative). Runs 5 and 7 use n_eff (effective sample size accounting for day-segment conditioning) in the intraday AICc computation. Run 7: n_eff = n_segments * (I - max(p, q)), with explicit guard usable_per_segment <= 0. (Satish et al. 2014, p.17; Hurvich & Tsai 1989, 1993.)
**Best version so far:** Run 5 — includes both the n <= k+1 guard and n_eff for segmented intraday fitting

### F10. "Prior 5 days" interpretation for inter-day ARMA
**Category:** algorithm
**Approach A** (runs 1, 3, 4, 5, 6, 7): "Prior 5 days" = effective AR memory (p_max=5), not training window. Training window is separate and longer. Run 1: N_interday=252 (1 year). Run 3: N_interday=126 (6 months). Run 4: N_interday_fit=63 (3 months). Run 5: N_interday_fit=126 (6 months). Run 7: N_interday_fit=63 (3 months). All agree the annotation describes what data the model "looks at" for prediction, not the training window. Run 7 adds: "Fitting an ARMA(5,5) on 5 observations is statistically infeasible. 63 trading days provides sufficient data while adapting to regime changes." Researcher inference.
**Approach B** (run 2): "Prior 5 days" = the most recent daily observations used as lagged inputs for the ARMA's one-step-ahead prediction (N_interday_predict=5), separate from the fitting window (N_interday_fit=63). Fitting on 5 observations is infeasible for models with up to 11 parameters. Researcher inference.
**Best version so far:** Run 5 — concise inline comment in ForecastRawVolume with paper citation; run 3 has the most thorough standalone discussion

### F11. N_interday_fit (fitting window for inter-day ARMA)
**Category:** parameter
**Approach A** (runs 1, 6): N_interday = 252 trading days (1 year). Range: [126, 504]. Run 6 adds: "needs enough for stable ARMA." Researcher inference.
**Approach B** (runs 2, 4, 7): N_interday_fit = 63 trading days (3 months). Range: [42, 126]. Must be substantially longer than max model order. Runs 4 and 7 add: "Fitting an ARMA(5,5) on only 5 observations is statistically infeasible. 63 trading days provides sufficient data while adapting to regime changes." Researcher inference.
**Approach C** (runs 3, 5): N_interday = 126 trading days (6 months). Consistent with N_seasonal window. Researcher inference. Range: [63, 252]. Run 5 adds: "A 126-day window (6 months) provides sufficient data for stable estimation and aligns with the seasonal factor window."
**Best version so far:** Run 5 — 126 days aligns with N_seasonal; run 4's 63-day justification is also strong for adaptiveness

### F12. Intraday ARMA: one model per symbol, deseasonalized, 21-day rolling window
**Category:** algorithm
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): Single ARMA model per symbol on deseasonalized intraday volume series from most recent 21 trading days. Deseasonalize by dividing each bin's volume by seasonal_factor[i]. Re-seasonalize forecasts via multiplication. Run 3 explicitly notes "most recent month" ambiguity (calendar vs trading month), uses 21 trading days. Run 5 provides FitIntraDayARMA (Function 3) with per-segment likelihood as primary approach, concatenation as explicit pragmatic alternative, and minimum 5 day-segments guard. Run 7 provides fit_intraday_arma (Function 3) with panel ARMA fitting: L_total = SUM over segments, conditional MLE, and explicit n_eff computation. (Satish et al. 2014, p.17-18.)
**Best version so far:** Run 5 — FitIntraDayARMA (Function 3) with per-segment likelihood algorithm detail (initialize buffers to zeros per segment, skip first p observations), n_eff computation, and explicit pragmatic fallback

### F13. Intraday AR constraint: p < 5
**Category:** algorithm
**Approach A** (runs 1, 3, 4, 5, 7): AR order p in {0,1,2,3,4}, MA order q in {0,...,5}. Paper only constrains AR ("AR lags with a value less than five"), not MA. Runs 5 and 7: p_max_intra=4 (AR < 5), q_max_intra=5, with explicit paper citation. Run 7 adds: "The MA order range is unconstrained by the paper; we use q_max_intra = 5 to match the inter-day search range." (Satish et al. 2014, p.18.)
**Approach B** (runs 2, 6): AR order p in {0,...,4}, MA order q in {0,...,4}. q_max_intraday = 4 as parameter. Run 6 parameter table: p_max_intra=4, q_max_intra=4, "constrained by joint dual-ARMA budget." (Satish et al. 2014, p.18.)
**Best version so far:** Run 5 — parameter table with explicit distinction: p_max_intraday=4 from paper, q_max_intraday=5 from researcher inference ("Paper constrains only AR, not MA")

### F14. "Fewer than 11 terms" interpretation
**Category:** algorithm
**Approach A** (run 1): Descriptive, not prescriptive. "As a result" phrasing indicates observed outcome of AR coefficient decay. In practice AICc selects low-order models. Log warning if combined parameter count >= 11. Researcher inference.
**Approach B** (run 2): Prescriptive joint constraint on combined inter-day + intraday ARMA system. Total terms (inter-day p+q+1 + intraday p+q+1) must be strictly less than 11. Enforced during model selection with inter-day budget cap. Researcher inference.
**Approach C** (runs 3, 4, 7): Soft constraint — observed empirical outcome of AICc selection and AR coefficient decay, but useful as overfitting guardrail. If combined parameter count exceeds 10, log a warning but do NOT reject the model. Quotes paper's "As a result, we fit..." phrasing as evidence it describes what happened, not what was enforced. Run 4 adds explicit "Removed parameter: max_interday_budget" note with justification. Run 7 implements in train_full_model (Function 9): LOG_WARNING if combined_terms > 10, but does not reject or reduce order. Researcher inference.
**Approach D** (runs 5, 6): Hard constraint with enforcement during model selection. Run 5: after independent model selection, check if (max interday terms across bins) + (intraday terms) >= 11; if violated, re-run AICc search for intraday with tighter p+q bounds (max_intra_terms = 11 - max_interday_terms - 1); minimum floor of 2 terms. Run 6: enforces the budget DURING intraday fitting — computes intraday_budget = max_dual_arma_terms - max(interday_term_counts) upfront, then skips any (p,q) candidate where k > intraday_budget; returns FALLBACK if budget < 2. Explicit max_dual_arma_terms=10 parameter ("fewer than 11"). Both runs quote the paper passage and read it as a hard constraint. Researcher inference.
**Best version so far:** Run 6 — enforces budget during model selection (avoids fitting then discarding), cleaner control flow; run 5 provides the automatic order reduction fallback

### F15. Day-boundary handling for intraday ARMA
**Category:** algorithm
**Approach A** (run 1): Naive concatenation (recommended default). Accept that overnight transitions weaken AR coefficient estimates slightly. Simplest, likely what paper uses.
**Approach B** (runs 2, 3, 4, 5, 6, 7): Likelihood reset at day boundaries. Each day is an independent within-day sequence. Insert break points so lag-1 does not connect last bin of one day to first bin of next. Run 5 provides the most detailed per-segment likelihood algorithm: for each day-segment, initialize AR/MA buffers to zeros, compute log-likelihood over segment observations, skip first p observations (conditioning on initial values), sum across segments. Uses fit_ARMA_segmented() with explicit n_eff = sum(max(len(seg) - p, 0) for seg in day_segments). Run 7 provides fit_ARMA_panel() with identical semantics and adds 3-phase description: (1) fitting: ARMA state reset to unconditional mean at each segment boundary, first max(p,q) bins burned for initialization; (2) prediction start-of-day: make_state([]) initializes to unconditional mean (~1.0 deseasonalized); (3) prediction mid-day: make_state(observed_deseas) runs AR/MA recursion through observed values, fully conditioned after ~4 bins. Researcher inference.
**Best version so far:** Run 5 — FitIntraDayARMA (Function 3) with step-by-step per-segment algorithm, n_eff formula, and explicit concatenation fallback with contamination analysis

### F16. Regime switching: regime classification at (day, bin) level
**Category:** algorithm
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): Regime classification at (day, bin) level — a single day can transition between regimes as cumulative volume grows. Based on percentile cutoffs of historical cumulative volume distribution per bin. Run 5 provides ClassifyRegime (Function 4) with bisect_left percentile lookup. Run 7 provides build_regime_classifier (Function 4) + assign_regime with explicit convention: last_observed_bin index and cumvol through it, consistent between training (i-1) and prediction (current_bin). Default to middle regime when last_observed_bin < 1. (Satish et al. 2014, p.18 para 4.)
**Best version so far:** Run 5 — ClassifyRegime (Function 4) with explicit bisect_left percentile rank, BuildCumVolDistribution helper documenting per-bin CDF rationale, and min_regime_bins guard

### F17. n_regimes
**Category:** parameter
**Approach A** (runs 1, 3): 3 regimes (tercile buckets). "Training several weight models" consistent with 3 as simplest scheme. Sensitivity: medium-high. Range: [2, 5]. Researcher inference.
**Approach B** (runs 2, 4, 5, 7): Grid search over {3, 4, 5} regimes. Run 5 provides OptimizeRegimeWeights (Function 5) with complete grid search. Run 7 implements grid search in train_full_model (Function 9): for each n_reg candidate, build classifier, split training window (last 21 days as validation), optimize weights on training portion, evaluate MAPE on validation, select best n_reg. Then re-optimizes weights on full window after selection (see F87). Sensitivity: high. Range: [2, 6]. Researcher inference.
**Approach C** (run 6): Grid search over {1, 2, 3, 4, 5} regimes — extends range to include N_regimes=1 (no regime switching) as a baseline option. "Single-regime result: If cross-validation selects N_regimes=1, the model degenerates to a single weight set with no regime switching. This is valid and expected for stocks with stable volume patterns." Run 6's Function 5 uses N_regime_valid=63 days for validation, and re-optimizes on full window after selection. Researcher inference.
**Best version so far:** Run 6 — includes N_regimes=1 as a natural baseline (no regime switching), allowing the model to select no regime switching when appropriate; run 5 has the most detailed pseudocode

### F18. regime_percentiles
**Category:** parameter
**Approach A** (runs 1, 3): [33, 67] for 3 regimes (equal-frequency). Researcher inference.
**Approach B** (runs 2, 4, 5, 6, 7): Equally spaced percentiles computed dynamically per n_reg candidate: n_reg=3 -> [33.3, 66.7]; n_reg=4 -> [25, 50, 75]. Runs 5 and 7: cutoffs = [100 * k / n_regimes for k in 1..n_regimes-1]. Grid search recommended. Researcher inference.
**Best version so far:** Run 5 — inline formula in OptimizeRegimeWeights with concrete examples

### F19. regime_threshold_window
**Category:** parameter
**Approach A** (runs 1, 3): N_regime_window = 60 trading days. Too short (21) = noisy thresholds; too long (126) = miss regime shifts. Sensitivity: medium. Range: [21, 126]. Researcher inference.
**Approach B** (runs 2, 4, 5, 6, 7): N_regime_window = 63 trading days (3 months). Runs 5 and 7 use 63 with explicit train/val split (last 21 days as validation). Range: [21, 126]. Researcher inference.
**Best version so far:** Run 5 — explicit train_start/train_end/val_start/val_end computation in Function 5

### F20. Regime assignment at start of day (j=0)
**Category:** algorithm
**Approach A** (runs 1, 3, 4, 5, 6, 7): Default to "medium" regime for first bins. Runs 1, 3 use min_regime_bins = 3 parameter. Run 5 uses min_regime_bins parameter (default 3) with default_regime. Run 7 implements in assign_regime: "IF last_observed_bin < 1: RETURN n_regimes // 2." Researcher inference.
**Approach B** (run 2): Use unconditional (all-data) weight set or median-regime weights. No minimum bin delay. Researcher inference.
**Best version so far:** Run 5 — ClassifyRegime (Function 4) with parameterized min_regime_bins and default_regime, clean guard logic

### F21. Weight optimization objective, method, and constraints
**Category:** algorithm
**Approach A** (runs 1, 5): Minimize MSE(actual, w1*H + w2*D + w3*A) subject to w1+w2+w3=1, all weights >= 0. scipy.optimize.minimize with SLSQP. Initial guess [1/3, 1/3, 1/3]. Run 5 provides explicit pseudocode in OptimizeRegimeWeights (Function 5): X = column_stack(H_vals, D_vals, A_vals), minimize MSE via SLSQP with simplex constraint, min_samples_per_regime=50 guard. Notes MAPE/Nelder-Mead as future enhancement ("MSE is smooth, convex, and has a unique minimum"). Researcher inference.
**Approach B** (runs 2, 4, 6, 7): Minimize MAPE subject to non-negativity. Nelder-Mead (derivative-free, since MAPE is non-differentiable). Runs 4 and 7 use exp-transformation for non-negativity only (no sum-to-1). Run 6 uses softmax parameterization: w_j = exp(theta_j)/sum(exp(theta_k)), ensuring BOTH non-negativity AND sum-to-1 via Nelder-Mead on unconstrained theta; multiple random restarts (N_optimizer_restarts=5). Run 7 provides 4 fixed starting points. Exclude bins with volume below min_volume_floor. Researcher inference.
**Approach C** (run 3): Implement BOTH MSE and MAPE, compare on held-out data. Primary: MSE with simplex constraint (w1+w2+w3=1, non-negative), SLSQP. Variant: MAPE minimization. Switch to MAPE optimization if it produces lower out-of-sample MAPE. Researcher inference.
**Best version so far:** Run 5 — most complete MSE/SLSQP pseudocode with explicit matrix formulation and simplex constraint; run 4 is best for the MAPE/Nelder-Mead alternative

### F22. N_hist = 21 trading days for rolling mean
**Category:** parameter
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): Historical window for Component 1 rolling average. (Satish et al. 2014, Exhibit 1 caption "Prior 21 days".) Sensitivity: medium-high. Range: [10, 60]. Run 7 separates Function 1 (seasonal, 126-day) from Function 1a (historical average, N_hist-day) with explicit distinction: "shorter window for Component 1 makes it more responsive to recent volume changes, while a longer window for deseasonalization provides a more stable reference." Run 7 also provides rolling H_d computation in optimize_regime_weights with extended volume_history slice (N_hist pre-context before first training day).
**Best version so far:** Run 5 — explicit shared-window rationale in compute_hist_pct with paper reference (p.16); run 4 includes the illustrative caveat

### F23. Volume percentage regression: OLS, no intercept
**Category:** algorithm
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): OLS without intercept. When all recent surprises are zero, delta should be zero. Consistent with paper's statement on p.19 about no-constant regressions. Pooled across all bins and days (not per-bin). Run 7 adds explicit ols_no_intercept helper: beta = solve(X^T X, X^T y). Also documents that mean surprise is exactly zero at training time by construction (hist_pct = mean of actual_pct), so no-intercept is automatically satisfied. Between re-estimations, drift may introduce nonzero mean; monitor and trigger early re-estimation if |mean| > 0.001. (Satish et al. 2014, p.18-19; Humphery-Jenner 2011.)
**Best version so far:** Run 5 — combines the self-justifying rationale with the reference chain clarification from run 3

### F24. Number of lagged surprise terms (K_reg / L_optimal)
**Category:** parameter
**Approach A** (runs 1, 3): K_reg = 3 as default. Sensitivity: medium. Range: [1, 5]. Run 3 notes paper refers to "identifying the optimal number of model terms" (p.19) but does not report the value. Researcher inference.
**Approach B** (runs 2, 4, 5, 7): L_optimal determined by cross-validation over L in {1, ..., L_max}. Range: 1-3 typical. Runs 4 and 7 use blocked K-fold time-series CV (see F82). Run 5 uses expanding-window CV. Run 7 provides complete build_surprise_regression helper with explicit L_max-based bin exclusion: training loop starts at bin (L_max + 1) to avoid edge effects. Researcher inference.
**Approach C** (run 6): N_surprise_lags = 5 as a fixed global hyperparameter, not selected per stock. Cites the paper's statement "we were able to identify the optimal number of model terms for U.S. stocks" (p.19) as evidence a single value was chosen globally across the universe. Recommends an offline study across a representative stock panel if tuning is desired, but this is a one-time exercise, not part of the per-stock calibration loop. Range: [1, 10]. Researcher inference.
**Best version so far:** Run 5 — TrainPercentageRegression (Function 8) with complete expanding-window CV pseudocode; run 6's global-fixed interpretation is well-argued from the paper text

### F25. Surprise baseline: raw volume model (primary)
**Category:** algorithm
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): Primary approach uses raw volume model (Model A) forecasts as the baseline from which surprises are computed. Paper says "we could apply our more extensive volume forecasting model" (Satish et al. 2014, p.19). Run 3 adds important caveat: "could" is aspirational, not confirmative. Run 7 computes surprises in percentage space (actual_pct - hist_pct), making Model A independence explicit: train_percentage_model does not take model_a as input (Draft 3 M1 fix).
**Best version so far:** Run 3 — notes the "could" caveat; run 5 adds fallback configuration option

### F26. Surprise definition and computation
**Category:** algorithm
**Approach A** (runs 1, 3, 5, 6, 7): surprise = actual_pct - expected_pct, computed in percentage space. Both terms share the same denominator at prediction time: surprise = (actual - expected) / V_total_est. Run 7 provides full domain consistency analysis: hist_pct ~ 1/26 ~ 0.038, surprise std ~ 0.005-0.015, delta ~ 0.001-0.005, scaled_base ~ 0.038 — all in same domain. Also documents 4 surprise baseline alternatives ((a) pct-space implemented, (b) raw multiplicative, (c) raw scaled-additive, (d) raw additive deprecated) with domain analysis showing option (d) causes 96% signal absorption by deviation clamp.
**Approach B** (runs 2, 4): surprise = (actual_vol - raw_forecast) / raw_forecast, computed in normalized raw volume space. Direct comparison of actual to raw model output. No need for V_total_est in surprise computation itself. Run 4 adds: "We use relative surprise (percentage deviation from raw forecast) because it normalizes across stocks and bins of different volume levels." Run 4 also applies min_volume_floor guard: if raw_fcst[j] <= min_volume_floor, surprise = 0.0.
**Best version so far:** Run 5 — explicit derivation showing percentage-space units match regression output; run 4 is best for the relative-surprise alternative with min_volume_floor guard

### F27. max_deviation = 0.10 (10% relative deviation)
**Category:** parameter
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): Maximum percentage departure from historical VWAP curve. Paper says "e.g., depart no more than 10% away" (Satish et al. 2014, p.24, referencing Humphery-Jenner). Sensitivity: medium-high. Range: [0.05, 0.20]. Run 3 adds critical caveat: paper states "we developed a separate method for computing the deviation bounds" (p.19), meaning the actual production system used adaptive/proprietary bounds, NOT the fixed 10%. Run 7: max_delta = max_deviation * base_pct (unscaled), applied before scaling (see F31).
**Best version so far:** Run 3 — includes the proprietary bounds caveat from p.19

### F28. Switch-off threshold: pct_switchoff = 0.80 (80%)
**Category:** parameter
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): "Once 80% of the day's volume is reached, return to a historical approach" — delta = 0 for this and all future bins. (Satish et al. 2014, p.24; Humphery-Jenner 2011.) Sensitivity: medium. Range: [0.70, 0.90]. Run 7 implements switch-off in forecast_volume_percentage Step 3: IF observed_frac >= pct_switchoff, delta = 0.
**Best version so far:** Run 5 — ForecastVolumePercentage Step 2 with explicit switch-off logic returning scaled_base; run 4 adds practical activation timing note

### F29. V_total_est computation
**Category:** algorithm
**Approach A** (runs 1, 2, 3, 4, 5, 7): V_total_est = sum(observed volumes for bins 1..current_bin) + sum(conditioned raw forecasts for bins current_bin+1..I). Model A must run FIRST to produce V_hat, then Model B uses these. Sequential dependency, no circular dependency. Run 7 provides explicit computation in forecast_volume_percentage Step 1: loop j = next_bin TO I calling forecast_raw_volume with observed_volumes, V_total_est = observed_total + remaining_forecast.
**Approach B** (run 6): estimated_total = cum_vol_today / (1.0 - remaining_pct), where remaining_pct = sum(base[current_bin+1..I]) and base is either hist_pct (naive) or model_a_baseline_pct (sophisticated). Assumes remaining volume follows the baseline percentages. Does NOT require calling Model A per remaining bin. Simpler computation, avoids sequential dependency. When remaining_pct <= 0 (late in day), falls back to estimated_total = cum_vol_today. Researcher inference.
**Best version so far:** Run 5 — Approach A with explicit V_total_est construction using conditioned forecasts; run 6's Approach B is simpler but less accurate (assumes baseline percentages hold exactly)

### F30. Two separate ForecastRawVolume calls for Model B
**Category:** algorithm
**Approach A** (runs 1, 4, 5, 7): (a) Unconditional forecasts (current_bin=1): for surprise baseline. (b) Conditioned forecasts (current_bin=actual): for V_total_est. Run 7 makes both calls explicit in forecast_volume_percentage: conditioned forecasts (Step 1, V_total_est via observed_volumes) and percentage-space surprises (Step 2, actual_pct = observed/V_total_est vs hist_pct). Surprise uses V_total_est as denominator (not unconditional forecasts). Paper does not distinguish these two uses. Researcher inference.
**Approach B** (run 6): For sophisticated variant, computes ALL Model A pre-observation forecasts ONCE at start of day (current_bin=0, empty observations, median regime). Stores as fixed model_a_baseline_pct for the entire day. No per-bin Model A calls needed for surprise computation. V_total_est uses fraction-based scaling (see F29 Approach B). Documents known trade-off: context-free baseline is less accurate for later bins vs. training baseline which benefits from intraday ARMA conditioning (see F88). Researcher inference.
**Best version so far:** Run 5 — both calls in named steps with explicit consistency rationale; run 6's pre-observation approach is simpler but introduces a secondary context mismatch

### F31. Deviation constraint baseline
**Category:** algorithm
**Approach A** (runs 1, 3, 4, 5): Constraint is relative to scaled_base (after renormalization). scale = remaining_fraction / remaining_hist. max_delta = max_deviation * scaled_base. More conservative. Run 5 implements in ForecastVolumePercentage Steps 5-6: compute scaled_base, then max_delta = max_deviation * scaled_base, delta = clip(delta, -max_delta, max_delta). Researcher inference.
**Approach B** (run 2): Constraint is relative to hist_pct[j+1] directly (before scaling). max_deviation = 0.10 * hist_pct[j+1]. Then renormalize after clipping. Researcher inference.
**Approach C** (runs 6, 7): Clamp delta against UNSCALED base_pct, then uniformly scale the combined (base + delta). Run 6 implements in Function 9: adjusted_pct = base_pct + delta (after clamping), then pct_hat = scale * adjusted_pct. max_delta = max_deviation * base_pct (unscaled), clamped_delta = clip(delta, -max_delta, max_delta), adjusted = scale * (base_pct + clamped_delta). Rationale: "delta is a predicted percentage-point departure from the UNSCALED hist_pct (since regression was trained on unscaled participation-rate surprises). We first clamp delta against the unscaled hist_pct, then apply scaling to the combined quantity. This ensures the entire adjusted participation rate (base + departure) is scaled uniformly to the remaining-fraction space, avoiding domain mixing between scaled and unscaled quantities." Researcher inference.
**Best version so far:** Run 7 — Approach C is most principled: deviation constraint operates in the same domain as the regression output (unscaled percentage space), and uniform scaling preserves the constraint's relative meaning

### F32. Renormalization: scale remaining bins to sum constraint
**Category:** algorithm
**Approach A** (runs 1, 2, 3): Explicit renormalization step after applying surprise-based adjustment and deviation constraints. Run 3 integrates via scale = remaining_fraction / remaining_hist approach. Run 2 explicitly handles last-bin edge case (j+1 == I: skip redistribution).
**Approach B** (runs 4, 5, 7): Implicit renormalization via per-call recomputation of scaled_base. Each call independently computes scale = actual_remaining_frac / remaining_hist_frac, producing a self-consistent sequence. Run 7's forecast_volume_percentage Steps 4-5 provide explicit formula: observed_hist_frac = sum(hist_pct[1..current_bin]), remaining_hist_frac = 1 - observed_hist_frac, scale = actual_remaining_frac / remaining_hist_frac. Last-bin: pct_forecast = actual_remaining_frac. Researcher inference.
**Best version so far:** Run 5 — ForecastVolumePercentage Steps 5-8 with complete implicit renormalization, edge cases for remaining_hist=0 and last bin

### F33. Re-estimation schedule
**Category:** algorithm
**Approach A** (runs 1, 3, 5): Tiered schedule. Daily: seasonal factors, H, hist_pct, intraday ARMA (full re-fit). Daily (state only): inter-day ARMA buffer updates. Weekly (every 5 days): inter-day ARMA full re-estimation with AICc (may change order). Monthly (every 21 days): regime weights, percentage regression. Run 5 provides ReEstimationSchedule (Function 10) with explicit conditional logic using trading_days_since(last_refit_dates) and complete function calls for each tier. Researcher inference.
**Approach B** (run 2): Daily rolling: all component models. Periodic (monthly/quarterly): weight optimization and regime grid search. Researcher inference.
**Approach C** (runs 4, 6, 7): Two-tier schedule. Daily: light update (H recomputation + inter-day ARMA state append via append_observation + hist_pct update + seasonal factor use cached). Monthly: full re-estimation of ALL components (train_full_model). Run 6 parameter table: re_estimation_frequency=Monthly, daily_update_components={hist_avg, hist_pct, ARMA state}. Run 7 implements in run_daily (Function 10): end-of-day state updates (append_observation for interday, rolling hist_avg and seasonal_factors), plus days_since_last_full_reestimation check. reestimation_interval = 21 days. Researcher inference.
**Best version so far:** Run 5 — ReEstimationSchedule (Function 10) with complete three-tier pseudocode, explicit last_refit_dates tracking, and clear distinction between daily state update and weekly/monthly re-estimation

### F34. N_regression_fit (training window for surprise regression)
**Category:** parameter
**Approach A** (runs 1, 3, 6): N_reg_train = 252 trading days (1 year). Run 6: N_surprise_train=252 in parameter table. Researcher inference.
**Approach B** (runs 2, 4, 5, 7): N_regression_fit = 63 trading days (3 months). Run 7 uses blocked K-fold CV (K=5) with contiguous day blocks, remainder days assigned to last fold. Range: [21, 126]. Researcher inference.
**Best version so far:** Run 5 — updated sample count (1575 vs 1386), explicit rationale for adaptiveness

### F35. No-intercept philosophy for regression
**Category:** algorithm
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): No intercept in the surprise regression: when all recent surprises are zero, delta should be zero. Consistent with paper's statement: "we perform both regressions without the inclusion of a constant term..." (Satish et al. 2014, p.19). Run 7 adds: the p.19 quote "explicitly refers to the VWAP-error validation regressions. We apply the same principle to the surprise regression." (See F23.)
**Best version so far:** Run 5 — combines self-justifying rationale with reference chain precision (see F23)

### F36. Lookahead bias in percentage model training
**Category:** edge case
**Consensus** (runs 1, 3, 4, 5, 6, 7): raw_volume_model trained on all data through train_end_date, but used to compute forecasts for earlier days d < train_end_date. Bias is negligible because ARMA parameters on 126+ days change minimally when dropping one day, and seasonal factors are 126-day averages. To eliminate: use expanding-window re-estimation (multiplies cost by N_reg_train). Researcher inference.
**Best version so far:** Run 5 — quantifies the bias magnitude (~0.8% contribution per day) in OptimizeRegimeWeights documentation

### F37. ARMA convergence failure handling
**Category:** edge case
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): Fall back to historical average component when ARMA fails to converge. Run 7: interday FALLBACK → D = H; intraday FALLBACK → A = seasonal_factors[target_bin]. In forecast_raw_volume (Function 6): explicit IS FALLBACK checks for both components with documented fallback values. In fit_interday_arma/fit_intraday_arma: return FALLBACK sentinel when no valid model found. Expected failure rate: 1-5%.
**Best version so far:** Run 5 — fallback logic fully integrated into ForecastRawVolume (Function 6) and predict_interday/predict_intraday_for_training helpers

### F38. Forecast update = reconditioning, not re-estimation
**Category:** algorithm
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): At forecast time during the day, ARMA is NOT re-estimated; only internal state (recent residuals and observations) is updated by conditioning on observed bins. Run 7 implements via make_state/predict interface: make_state(observed_deseas) creates fresh state from observations without mutating model; predict(state, steps) produces multi-step forecasts from state. End-of-day: append_observation advances recursion state without re-estimating coefficients.
**Best version so far:** Run 5 — combines purity documentation with inline algorithm for reset-and-reprocess conditioning

### F39. Regime boundary hysteresis (optional enhancement)
**Category:** algorithm
**Consensus** (runs 1, 3, 7): Optional robustness enhancement: once a regime is selected, require percentile to cross boundary by a margin (e.g., 5 percentile points) before switching. regime_hysteresis parameter (default 5, range [2, 10]). NOT required for baseline. Paper does not mention. Researcher inference.
**Best version so far:** Run 1 — equally clear in both

### F40. Handling insufficient surprise lags for early bins
**Category:** edge case
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): At bins 1 through L (lag count), fewer than L lagged surprises available. Runs 3, 4, 5, 7: if current_bin < L, set delta = 0. Functionally equivalent across all runs.
**Best version so far:** Run 5 — ForecastVolumePercentage Step 4 with inline explanation of equivalence

### F41. Half-day trading sessions handling
**Category:** edge case
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): Days before holidays have 13 bins instead of 26. Exclude from training and evaluation. Run 5 adds date arithmetic convention note: "is_full_trading_day() filters half-day trading sessions (which have only 13 bins instead of 26) from full-day sessions; it does not filter weekends or holidays, which are already absent from the trading-day index."
**Best version so far:** Run 5 — clarifies is_full_trading_day predicate semantics

### F42. Special calendar days: custom curves
**Category:** edge case
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): Option expiration, Fed announcements, index rebalancing days have different patterns. Paper recommends "custom curves for special calendar days... rather than ARMAX models, due to insufficient historical occurrences" (Satish et al. 2014, p.18 para 4). Maintain a calendar and substitute pre-computed curves, bypassing the dynamic model.
**Best version so far:** Run 2 — adds actionable implementation guidance

### F43. Zero-volume bins handling
**Category:** edge case
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): Floor prevents division by zero in deseasonalization. MAPE undefined for zero-volume bins; exclude from error computation. Raw forecast clamped at zero. Specific floor approach differs (see F7).
**Best version so far:** Run 2 — more thorough discussion in context of top-500 stock universe

### F44. Validation target: Model A raw volume MAPE
**Category:** validation
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): 24% median MAPE reduction and 29% bottom-95% MAPE reduction over rolling historical average baseline across all intraday intervals. (Satish et al. 2014, p.20, Exhibit 6.) Error reduction by time-of-day: ~10-12% at 9:30 to ~30-33% at 15:30. Run 5 adds: "significant bin-to-bin variation in the median profile" and "bottom-95% profile shows a smoother increasing trend from ~13% to ~38%." Consistent across SIC industry groups (~15-35%) and beta deciles (~20-35%). (Exhibits 7-8, pp. 22-23.)
**Best version so far:** Run 5 — most complete time-of-day profile noting bin-to-bin variation in median vs smoother bottom-95%

### F45. Validation target: Model B volume percentage error
**Category:** validation
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): 7.55% median reduction in absolute volume percentage error for 15-minute bins (0.00874 historical to 0.00808 dynamic). Bottom-95%: 0.00986 to 0.00924, 6.29% reduction. Wilcoxon signed-rank test. (Satish et al. 2014, Exhibit 9, p.23.) Runs 2, 3, 4, 5 explicitly clarify this is MAD (mean absolute deviation), NOT MAPE. Run 5 adds the Exhibit 9 arithmetic inconsistency note for 30-min bins: "the paper's stated 2.95% does not match arithmetic from its own table values (~3.43%), suggesting a rounding or transcription error." 5-min bins 2.25%, 30-min bins 2.95%.
**Best version so far:** Run 5 — includes all bin-size comparisons plus the 30-min arithmetic inconsistency note

### F46. Validation target: VWAP simulation tracking error
**Category:** validation
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): 9.1% VWAP tracking error reduction (mean 8.74 bps vs. 9.62 bps historical) with 10% ADV order size on 600+ simulated orders across Dow 30, midcap, and high-variance stocks. Paired t-test: 2.34, p < 0.01. 7-10% improvement across stock groups. Std dev: 11.18 bps (historical) vs. 10.08 bps (dynamic). Run 7 adds important clarification (m7): the absolute bps values (8.74, 9.62) are specific to FlexTRADER simulation conditions (10% of 30-day ADV, day-long orders, May 2011 data); a developer should NOT expect to match these absolute numbers; the meaningful benchmark is the relative improvement (~9%) and statistical significance of the paired test. (Satish et al. 2014, Exhibit 10, p.23.)
**Best version so far:** Run 7 — adds the m7 clarification that absolute bps values are simulation-specific, not reproducible targets; run 3 includes standard deviation figures

### F47. VWAP-percentage error regression relationship
**Category:** validation
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): R^2 > 0.50: coefficient ~220.9 bps/unit for Dow 30 (R^2=0.5146, t-stat 7.496), ~454.3 bps/unit for high-variance stocks (R^2=0.5886, t-stat 6.329). Both regressions through the origin (no intercept). (Satish et al. 2014, Exhibits 3 and 5, pp.20-21.)
**Best version so far:** Run 3 — includes t-statistics

### F48. Comparison benchmark: Chen et al. (2016) Kalman filter
**Category:** validation
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): Chen et al. average MAPE 0.46 (dynamic/robust Kalman), 0.61-0.65 (static/CMEM), 1.28 (rolling mean) across 30 securities; VWAP tracking 6.38 bps (dynamic robust Kalman) vs 7.48 bps (rolling means) — 15% improvement. Direct comparison imprecise: different datasets (500 vs. 30 securities), different time periods, different exchanges, different MAPE normalization (Satish uses percentage, Chen uses fractional). (Chen et al. 2016, Tables 3-4.)
**Best version so far:** Run 3 — includes Chen VWAP rolling-mean baseline and normalization caveat

### F49. MAPE computation formula
**Category:** algorithm
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): MAPE = 100% * (1/N) * SUM(|Predicted_Volume - Raw_Volume| / Raw_Volume). Bins with zero actual volume excluded (MAPE undefined). Runs 2, 3, 4, 5, 7 explicitly distinguish: raw volume uses MAPE, percentage volume uses MAD. Run 7 provides compute_evaluation_mape function using conditioned forecasts (bins 1..i-1 observed, matching production). Run 5 adds aggregation hierarchy in sanity check 17: "per-symbol-day MAPE → per-symbol MAPE (mean across days) → cross-symbol median and bottom-95% average." (Satish et al. 2014, p.17.)
**Best version so far:** Run 5 — EvaluateWeights helper with min_volume_floor guard and explicit aggregation hierarchy in validation section

### F50. Sanity checks suite
**Category:** validation
**Consensus** (runs 1, 2, 3, 4, 5, 6, 7): Overlapping checks from all runs. Run 1: (1) weights positive sum-to-1; (2) seasonal U-shape; (3) deseasonalized stationarity (ADF); (4) AICc selects low orders; (5) all regime buckets populated; (6) deviation constraint binds <= 10-20%; (7) 80% switch-off for last 2-4 bins; (8) cumulative forecasts monotonic approaching 1.0; (9) regression |beta_k| < 0.5; (10) Component 1 baseline MAPE; (11) surprise mean ~0, std ~0.005-0.015. Run 2: (1) historical-only weights reproduce baseline; (2) seasonal U-shape; (3) ARMA order stability; (4) joint term constraint; (5) weight non-negativity; (6) volume pct sums to 1.0; (7) deviation limit effective; (8) switch-off behavior; (9) monotonic improvement with components; (10) day-boundary correlation check; (11) surprise coefficient sign (positive for L=1). Run 3 adds: (12) inter-day ARMA state consistency; (13) intraday ARMA multi-step convergence; (14) intraday ARMA conditioning consistency. Run 4 adds: (15) MAPE evaluation consistency; (16) empty regime fallback check. Run 5 adds: (17) train/predict surprise baseline consistency (both use unconditional forecasts); (18) combined term constraint activation (expected < 5% of stocks when p_max_intraday=4).
**Best version so far:** Run 5 — most comprehensive (18 numbered checks in Validation section) with specific numerical thresholds; run 1 for specific thresholds, run 2 for coefficient sign, run 3 for state consistency

### F51. N_interday_predict = 5 (separate from fitting window)
**Category:** parameter
**Consensus** (runs 2, 3, 4, 5, 6, 7): Number of recent daily observations used as ARMA lag inputs for one-step-ahead prediction. Separate parameter from N_interday_fit. Value of 5 from Exhibit 1 "Prior 5 days" label. (Satish et al. 2014, Exhibit 1, p.18.)
**Best version so far:** Run 2 — cleanly separates it as an explicit parameter

### F52. max_interday_budget = 8 terms
**Category:** parameter
**Approach A** (run 2): Maximum terms allowed for any single inter-day ARMA model (p + q + 1 <= 8). Reserves at least 2 terms for the intraday ARMA. Range: [7, 9]. Researcher inference.
**Approach B** (runs 3, 4, 7): No per-model budget cap. The combined 11-term observation is a soft constraint applied globally (see F14 Approach C). Runs 4 and 7 explicitly remove the per-model budget concept. Researcher inference.
**Approach C** (run 5): No per-model budget cap, but enforces combined constraint as hard limit via automatic intraday order reduction (see F14 Approach D). The budget concept is replaced by the combined hard constraint with dynamic adjustment. Researcher inference.
**Best version so far:** Run 5 — replaces the budget concept with the more principled combined hard constraint mechanism; run 4 is best for the removal justification

### F53. Joint term constraint: per-bin vs per-symbol semantics
**Category:** algorithm
**Approach A** (runs 2, 4, 5, 6, 7): Uses max(interday_terms_i for i in 1..I) as the binding constraint — most conservative interpretation ensuring the joint constraint holds for the worst-case bin. Run 5 computes max_interday_terms = max(interday_models[i].p + interday_models[i].q + 1 for i in 1..I) and uses it to compute the allowed intraday budget: max_intra_terms = 11 - max_interday_terms - 1. Explicit pseudocode in Function 3.
**Approach B** (run 3): Combined count = (p_inter + q_inter + 1) + (p_intra + q_intra + 1) checked as a soft constraint per symbol. Does not specify which inter-day bin to use for the check. Researcher inference.
**Best version so far:** Run 5 — complete enforcement pseudocode in Function 3 using worst-case bin with automatic intraday order reduction

### F54. "Today" in Exhibit 1 interpreted as data source qualifier
**Category:** algorithm
**Approach A** (run 2): "Today" is a data source qualifier: AR lag inputs come from today's observed deseasonalized bins, not from historical days. NOT a separate feature. Researcher inference.
**Approach B** (runs 3, 4, 5, 6, 7): "Current Bin" and "4 Bins Prior to Current Bin" — the annotation describes AR order constraint (p < 5), so effective memory is ~4-5 bins regardless of input length. Passing all observed bins is harmless and preserves MA state. Run 7 adds: "This likely reflects the diagram notation for the maximum AR lag (p_max_intra = 4), not a strict limitation to using only the 4 most recent bins. The ARMA model uses all observed bins to build its state via the recursion." Researcher inference.
**Best version so far:** Run 5 — inline annotation in ForecastRawVolume Component 3 with paper citation for "4 Bins Prior to Current Bin"

### F55. Multi-step intraday ARMA prediction degradation
**Category:** algorithm
**Consensus** (runs 2, 3, 4, 5, 6, 7): Recursive multi-step ARMA predictions degrade with forecast horizon. After p steps, AR component uses only past forecasts; after q steps, MA contributes nothing. For distant bins, forecast converges to unconditional mean (~1.0 in deseasonalized space), re-seasonalizing to the seasonal factor. Run 5 adds full paper quote: "techniques that predict only the next interval will perform better than those attempting to predict volume percentages for the remainder of the trading day" (p.18). Regime weights should naturally adapt. Researcher inference.
**Best version so far:** Run 3 — explicit numerical example (ARMA(2,2), current_bin=1, forecasting bin 26); run 5 adds the full paper quote

### F56. Volume percentage error metric is MAD, not MAPE
**Category:** algorithm
**Consensus** (runs 2, 3, 4, 5, 6, 7): The volume percentage error in Exhibit 9 is mean absolute deviation (MAD): Error = (1/I) * sum_i |Predicted_Percentage_i - Actual_Percentage_i|. NOT MAPE — percentage predictions are already normalized (sum to ~1), so no division by actual is needed. Run 5 quotes: "Measuring Percentage Volume Predictions -- Absolute Deviation" (p.17). (Satish et al. 2014, p.17.)
**Best version so far:** Run 3 — strongest warning against confusion with MAPE

### F57. min_volume_floor = 100 shares
**Category:** parameter
**Consensus** (runs 2, 4, 5, 6, 7): Minimum volume threshold for including a bin in MAPE computation. Prevents division-by-zero instability in MAPE. Range: [50, 500]. Excluded from MAPE numerator and denominator when actual volume < floor. Runs 4, 5 apply the same floor to surprise computation. Run 5 adds explicit parameter table entry with sensitivity "Low." Researcher inference.
**Best version so far:** Run 5 — extends floor to both MAPE and surprise computation with parameter table documentation

### F58. Stock splits / corporate actions edge case
**Category:** edge case
**Consensus** (runs 2, 4, 5, 6, 7): Volume data must be adjusted for splits. A 2:1 split doubles apparent volume. Use split-adjusted volume or normalize by daily shares outstanding (as Chen et al. 2016, Section 4.1 suggest). Run 5 specifies "Split-adjusted share counts" as an explicit input requirement in both Overview and Initialization sections. Researcher inference.
**Best version so far:** Run 2 — includes Chen et al. normalization alternative

### F59. Joint term constraint infeasibility edge case
**Category:** edge case
**Consensus** (runs 2, 3, 4, 5, 6, 7): If inter-day ARMA for a bin requires many terms (e.g., p=4,q=4,constant=9), intraday would be limited. Run 2: cap inter-day at max_interday_budget=8. Runs 3, 4: no cap, soft constraint logs warning. Run 5: hard constraint with automatic intraday order reduction — re-fit with max_intra_terms = 11 - max_interday_terms - 1, minimum floor of 2 terms. Expected activation: < 5% of stocks.
**Best version so far:** Run 5 — complete resolution mechanism with automatic order reduction and activation frequency expectation

### F60. No distributional framework (limitation)
**Category:** other
**Consensus** (runs 2, 3, 4, 5, 6, 7): Unlike CMEM or Kalman filter models, this approach does not specify a noise distribution, so cannot produce prediction intervals or density forecasts. Only point forecasts. Run 5 adds: "Volume distributions are positively skewed with heavy tails, but the ARMA framework assumes linear dynamics. A log-transformation (as Chen et al. 2016 use) could reduce skewness, but the paper does not describe this step." Researcher inference.
**Best version so far:** Run 5 — combines distributional critique with specific log-transform suggestion referencing Chen et al.

### F61. No outlier robustness mechanism (limitation)
**Category:** other
**Consensus** (runs 2, 4, 5, 6, 7): Unlike Chen et al.'s robust Kalman filter with Lasso regularization, no built-in outlier handling. Outlier bins directly affect historical averages, ARMA estimates, and surprise computations. Researcher inference.
**Best version so far:** Run 2

### F62. Single-stock model (limitation)
**Category:** other
**Consensus** (runs 2, 3, 4, 5, 6, 7): Each stock is modeled independently. Cross-sectional information (e.g., sector-wide volume surges from ETF flows) not captured. Run 2 references BDF factor model approach. Run 3 references Bialkowski et al. (2008) PCA-based factor decomposition. Researcher inference.
**Best version so far:** Run 2 — equally clear in both

### F63. Static seasonal assumption (limitation)
**Category:** other
**Consensus** (runs 2, 4, 5, 6, 7): 6-month trailing average assumes intraday volume shape is constant over that window. Structural changes (e.g., shifts in closing auction participation, changes in electronic trading) captured slowly. Researcher inference.
**Best version so far:** Run 2

### F64. Last-bin edge case in renormalization
**Category:** edge case
**Consensus** (runs 2, 4, 5, 6, 7): When predicting the last bin (target_bin == I), there are no remaining bins to redistribute. Run 2: skip redistribution, pct_forecast[I] = adjusted_pct. Runs 4, 5: pct_forecast = remaining_frac (assign all remaining fraction to last bin). Run 5 implements in ForecastVolumePercentage Step 8: "IF target_bin == I: pct_forecast = remaining_frac." Researcher inference.
**Best version so far:** Run 5 — ForecastVolumePercentage Step 8 with remaining_frac assignment and inline rationale

### F65. Regime grid search over {3, 4, 5} regimes and cutoff positions
**Category:** algorithm
**Consensus** (runs 2, 4, 5, 6, 7): Grid search over number of regimes {3, 4, 5} and cutoff positions (equally spaced percentiles). Select configuration minimizing out-of-sample MAPE on held-out validation period. Run 5 provides OptimizeRegimeWeights (Function 5) with complete pseudocode including causal regime classification during training, per-regime MSE weight optimization, and EvaluateWeights helper for validation MAPE. Required because regime thresholds are proprietary. Researcher inference.
**Best version so far:** Run 5 — OptimizeRegimeWeights (Function 5) with complete grid search, EvaluateWeights helper, and causal regime classification

### F66. Inter-day ARMA state update mechanism (UpdateInterDayState)
**Category:** algorithm
**Consensus** (runs 3, 4, 5, 6, 7): Detailed specification for updating inter-day ARMA state when a new daily observation arrives WITHOUT re-estimating parameters. Run 7 implements in run_daily (Function 10) end-of-day: append_observation(observed_volumes[i]) for each non-FALLBACK interday model. Run 5 provides explicit UpdateInterDayState function (within Function 9): (1) compute one-step-ahead prediction, (2) compute residual, (3) append observation to AR buffer (drop oldest if > p), (4) append residual to MA buffer (drop oldest if > q). "This is a STATE UPDATE, not re-estimation (parameters unchanged). Full re-estimation happens on the weekly schedule." Researcher inference.
**Best version so far:** Run 5 — standalone UpdateInterDayState function with numbered steps and integration into DailyOrchestration

### F67. Intraday ARMA conditioning mechanism (ConditionIntraDayARMA)
**Category:** algorithm
**Consensus** (runs 3, 4, 5, 6, 7): Mechanism for conditioning intraday ARMA on today's observed bins. All runs use reset-and-reprocess (stateless) design. Run 5 provides inline algorithm in ForecastRawVolume: "initialize AR/MA buffers to zeros (matching the per-segment likelihood training assumption), then sequentially process each observed bin: compute prediction, compute residual, shift buffers." Adds: "Calling condition([1,2,3]) then condition([1,2,3,4]) re-processes bins 1-3 — no state corruption risk." Functionally equivalent across all runs. Researcher inference.
**Best version so far:** Run 5 — inline algorithm in ForecastRawVolume with idempotency guarantee; run 3 adds O(current_bin^2) complexity analysis

### F68. Forecast purity requirement
**Category:** algorithm
**Consensus** (runs 3, 4, 5, 6, 7): model.forecast(steps=N) MUST be a pure function that does NOT modify the model's persistent state. Run 7 documents purity in forecast_raw_volume: "make_state() creates a NEW state object... It does NOT mutate the intraday_model's internal state." Run 5 documents purity throughout: ForecastRawVolume is "PURE: it does not modify model_a's persistent state. All conditioning operates on copies." condition() "returns new state" and forecast() "operates on internal copies of buffers, does NOT modify conditioned's persistent state." Run 5's sanity check 16: "Calling condition([1,2,3]) followed by condition([1,2,3,4]) should produce the same result as a single call to condition([1,2,3,4])." Researcher inference.
**Best version so far:** Run 5 — purity documented at every level (function, condition, forecast) with idempotency check in validation; run 4 adds concrete failure scenarios

### F69. Train/predict denominator mismatch in surprise computation
**Category:** edge case
**Consensus** (runs 3, 5, 6, 7): During prediction, surprises use V_total_est (observed + forecasted remaining) as denominator, which is noisy and evolves as bins are observed. During training, surprises use exact daily total (known in hindsight). This creates a statistical mismatch. Run 5 adds: "This creates a small statistical mismatch that diminishes late in the day as V_total_est converges to the actual total. The mismatch is acceptable because regression coefficients are small and the deviation constraint limits impact." Two approaches: (a) accept mismatch (implemented), (b) use leave-future-out estimated totals (eliminates mismatch but multiplies training compute by I). Researcher inference.
**Best version so far:** Run 5 — quantifies the mismatch behavior and justifies accepting it

### F70. Dynamic vs static training forecasts
**Category:** algorithm
**Approach A** (runs 3, 6): Training surprise regression should use DYNAMIC raw forecasts (matching prediction behavior) rather than STATIC pre-market forecasts. At each bin i of training day d, the raw forecast should use observations through bin i-1. Run 6's generate_model_a_training_forecasts computes per-(day, bin) forecasts with intraday ARMA context from bins 1..i-1 and regime reconstruction from historical cumulative volume percentiles. Run 3 estimates cost: ~170K ForecastRawVolume calls per symbol. Researcher inference.
**Approach B** (runs 4, 5, 7): Training uses STATIC (unconditional, market-open) raw forecasts: ForecastRawVolume(model_a, stock, d, current_bin=1, observed_volumes={}). Run 5's TrainPercentageRegression (Function 8) explicitly documents: "Uses unconditional forecasts as the surprise baseline for BOTH training and prediction, ensuring internal consistency." The unconditional forecast represents the "naive" model's prediction at market open. Achieves train/predict consistency while avoiding O(N * I^2) computation. Researcher inference.
**Best version so far:** Run 5 — TrainPercentageRegression (Function 8) with explicit consistency documentation and M1 resolution reference

### F71. Surprise denominator structural asymmetry
**Category:** edge case
**Consensus** (runs 3, 5, 6, 7): A subtle third mismatch in surprise computation: whether actual_pct and expected_pct within a single surprise share the SAME denominator. In prediction: both use V_total_est (common denominator). In training: both use daily_total (common denominator). Run 5 explicitly uses daily_total for both terms in Function 8: day_surprises[j] = (actual - expected) / daily_total. This resolves the within-surprise mismatch for training. The across-phase mismatch (V_total_est in prediction vs daily_total in training) remains (see F69). Researcher inference.
**Best version so far:** Run 5 — explicitly uses common denominator (daily_total) for both terms in training

### F72. Early-bin padding design choice in training
**Category:** algorithm
**Approach A** (runs 3, 5): Include padded data — more training data, slightly biases coefficients toward zero, matches prediction-time padding. Run 5's Function 8 loops "FOR j in 2..I" and zero-pads missing lags. Researcher inference.: "IF lag_bin < 1: x_val.append(0.0)". Recommended. Researcher inference.
**Approach B** (runs 4, 7): Exclude early bins — training loop starts at bin (L_max + 1), skipping bins with insufficient lags entirely. Cleaner estimates, loses L_max bins per day (~8-12% for L_max=3-5). Run 7's build_surprise_regression: "Only use bins > L_max to avoid edge effects (early bins lack sufficient lags)." Researcher inference.
**Best version so far:** Run 5 — explicit zero-padding in Function 8 matches prediction-time behavior; run 4's approach is cleaner statistically

### F73. Redundant conditioning optimization in DailyOrchestration
**Category:** algorithm
**Consensus** (runs 3, 4, 5, 6, 7): In the daily loop, each call to ForecastRawVolume internally conditions the intraday ARMA. With pure/stateless design, this is correct but redundant. Run 5's DailyOrchestration (Function 9) calls ForecastRawVolume and ForecastVolumePercentage independently per bin, relying on purity for correctness. "Conditioning happens inside ForecastRawVolume via the intraday ARMA's condition() method (pure, stateless)." Researcher inference.
**Best version so far:** Run 3 — identifies the optimization opportunity; runs 4, 5 accept the redundancy cost (trivial for 26 bins)

### F74. Pre-market regime handling (current_bin=0)
**Category:** edge case
**Consensus** (runs 3, 4, 5, 6, 7): When no volume observed (current_bin=0 or 1), regime classification is impossible. Default to middle regime. Run 5: ClassifyRegime with min_regime_bins guard (default 3) and default_regime parameter. Pre-market Model A forecasts are therefore not regime-adapted. Researcher inference.
**Best version so far:** Run 5 — parameterized guard in ClassifyRegime (Function 4)

### F75. Deviation bounds are proprietary simplification
**Category:** algorithm
**Consensus** (runs 3, 4, 5, 6, 7): Paper states "we developed a separate method for computing the deviation bounds" (p.19), meaning the production system used adaptive or proprietary bounds, NOT the fixed 10% from Humphery-Jenner (2011). The 10% is a simplification. max_deviation is a primary tuning candidate. Run 5 adds Humphery-Jenner provenance annotation in ForecastVolumePercentage Step 6. Researcher inference.
**Best version so far:** Run 3

### F76. min_samples_per_regime parameter
**Category:** parameter
**Approach A** (runs 3, 5, 6, 7): Default: 50. Range: [20, 100]. If fewer samples available, fall back to equal weights (1/3, 1/3, 1/3). Sensitivity: low. Run 7 implements in optimize_regime_weights (Function 5): "IF len(samples) < min_samples_per_regime: weights[r] = [1/3, 1/3, 1/3]; CONTINUE." Parameter table entry: min_samples_per_regime = 50. Researcher inference.
**Approach B** (run 4): Threshold: 10 samples. Fall back to equal weights (1/3, 1/3, 1/3) and log warning. Mentioned in Edge Cases section. Researcher inference.
**Best version so far:** Run 5 — explicit guard logic in Function 5 with parameter table entry; more conservative threshold (50) better prevents overfitting

### F77. No price information limitation
**Category:** other
**Consensus** (runs 3, 5, 6, 7): The model is purely volume-driven and agnostic to price dynamics. Cannot exploit volume-price correlations or adjust for price impact. (Satish et al. 2014, implicitly; also noted by Bialkowski et al. 2008.) Researcher inference.
**Best version so far:** Run 3

### F78. Small order size assumption
**Category:** other
**Consensus** (runs 3, 5, 6, 7): The VWAP simulation uses 10% of 30-day ADV as order size (Satish et al. 2014, p.23). For larger orders, price impact becomes significant and the model's volume-only framework becomes insufficient. Researcher inference.
**Best version so far:** Run 3

### F79. Metric incomparability across papers
**Category:** other
**Consensus** (runs 3, 4, 5, 6, 7): Different papers use different MAPE normalization conventions, making direct cross-paper comparison difficult. Satish et al. express MAPE as a percentage; Chen et al. (2016) express it as a fraction. Run 5: "Direct comparison is imprecise: different datasets (500 vs. 30 securities), time periods, exchanges, and MAPE normalization conventions." (Satish et al. 2014, p.17; Chen et al. 2016, p.7 Equation 37.) Researcher inference.
**Best version so far:** Run 3

### F80. Dataset specificity limitation
**Category:** other
**Consensus** (runs 3, 5, 6, 7): Results are reported on the top 500 U.S. equities by dollar volume over one specific year. Performance on less liquid stocks, other markets, or different time periods may differ. Researcher inference.
**Best version so far:** Run 3

### F81. Exp-transformation for Nelder-Mead non-negativity in weight optimization
**Category:** algorithm
**Approach A** (runs 4, 7): Optimize weights in log-space: w_log in R^3, actual weights = exp(w_log). This guarantees non-negativity without explicit bounds (which Nelder-Mead does not support). Initialize at log(1/3) ~ -1.099. Nelder-Mead options: maxiter=1000, xatol=1e-4, fatol=1e-6. Used as the PRIMARY optimization approach (MAPE objective). Run 7 adds multi-restart with 4 starting points (equal, H-dominant, D-dominant, A-dominant) and convergence fallback: if best_loss >= equal_loss, use equal weights. Researcher inference.
**Approach B** (run 5): Exp-transformation noted as a FUTURE ENHANCEMENT only, not the primary approach. Run 5 uses MSE/SLSQP with explicit simplex constraint as primary (see F21 Approach A). Researcher inference.
**Approach C** (run 6): Softmax parameterization: w_j = exp(theta_j)/sum(exp(theta_k)). Guarantees BOTH non-negativity AND sum-to-1 via unconstrained Nelder-Mead on theta parameters. Numerically stabilized: max_theta subtracted before exp. Multiple random restarts (N_optimizer_restarts=5, theta_init ~ Normal(0, 0.5)). Functionally similar to Approach A but explicitly enforces the simplex constraint (sum-to-1) that Approach A omits. Researcher inference.
**Best version so far:** Run 6 — softmax enforces both non-negativity and sum-to-1 in a single parameterization; run 7 has more detailed starting point strategy

### F82. Blocked time-series cross-validation for surprise lag selection
**Category:** algorithm
**Approach A** (runs 4, 7): Blocked K-fold time-series CV for selecting optimal number of surprise lag terms (L). Partition N_regression_fit days into K=5 contiguous blocks. For each fold, hold out one block as test, train on remaining 4 blocks. Only use bins (L_max+1)..I within each block to avoid edge effects. Evaluate MAE on held-out data. Run 7 provides complete pseudocode: block_size = len(days) // K, remainder days assigned to last fold (e.g., 63/5 = 12 each, last gets 15). Preserves temporal structure by splitting at day boundaries. Researcher inference.
**Approach B** (run 5): Expanding-window CV with 1-day step. Train on days 0..k, validate on day k+1, stepping k from min_train_days (21) to n_days-2. For each L in 1..L_max, compute mean MAE across all validation folds. Select L minimizing average MAE. Run 5's TrainPercentageRegression (Function 8) provides complete pseudocode. This respects temporal ordering (always train on past, validate on future) and provides more validation folds than blocked K-fold. Researcher inference.
**Best version so far:** Run 5 — expanding-window CV is more principled for time series (never trains on future data); run 4's blocked approach is computationally simpler with fewer folds

### F83. MA buffer burn-in reconstruction for predict_interday during training
**Category:** algorithm
**Single source** (run 5): When computing inter-day ARMA forecasts during training (predict_interday helper in Function 5), MA residual buffers cannot be directly looked up because residuals depend on prior residuals (recursive dependency). Run 5 resolves this via sequential burn-in processing from zeros: burn_in = max(2*q, 10). Process (burn_in + q) trading days ending at day_d - 1, starting with MA buffer initialized to zeros. For each historical day: compute one-step-ahead prediction from current buffers, compute residual, update MA buffer. After burn_in steps, MA state converges to true state (burn-in artifact decays exponentially for invertible MA processes). Complete pseudocode provided in predict_interday helper with explicit buffer manipulation. Researcher inference; paper does not describe MA buffer reconstruction.
**Best version so far:** Run 5

### F84. min_cv_train_days = 21 parameter for expanding-window CV
**Category:** parameter
**Single source** (run 5): Minimum number of training days before the first validation fold in the expanding-window cross-validation for surprise lag selection. 21 days provides ~525 training samples (21 * 25 bins) for the regression, sufficient for stable OLS with up to 5 regressors. Range: [10, 30]. Sensitivity: Low. Explicit parameter table entry in run 5. Researcher inference; paper does not disclose CV methodology.
**Best version so far:** Run 5

### F85. ARMA model interface specification (5-method contract)
**Category:** algorithm
**Consensus** (runs 6, 7): Explicit 5-method interface for fitted ARMA model objects, applicable to both inter-day and intraday models: (1) predict_at(d) -> float: one-step-ahead forecast for day d using info through d-1 (conditional mean E[Y_d | Y_{d-1}, ...]); used during training. (2) predict_next() -> float: forecast for next unobserved time step; used during live prediction. Run 6 explicitly notes this is a "pure query that does NOT modify the model's internal state." (3) append_observation(value): extends observation buffer and updates AR/MA lag values (Kalman-filter-style state update, NO re-estimation); used for end-of-day state advancement. (4) make_state(observed_values) -> state: (intraday only) creates fresh prediction state from deseasonalized observations without mutating model's internal state. Run 6 specifies initialization: unconditional mean for AR lags, zero for MA residuals. (5) predict_from_state(state, steps) -> list[float]: (intraday only) produces multi-step-ahead forecasts from given state. Also includes term_count -> int property. Both runs note: "These semantics are standard for ARMA state-space implementations." Researcher inference — paper does not specify prediction interface.
**Best version so far:** Run 6 — adds purity guarantees for predict_next() and make_state initialization specification (unconditional mean for AR lags, zero for MA residuals)

### F86. Percentage coherence: forecasts do not sum to 1.0 (and why that's acceptable)
**Category:** algorithm
**Single source** (run 7): Detailed analysis of why bin-by-bin percentage forecasts do NOT strictly sum to 1.0 over the full day. Causes: (a) V_total_est changes as more bins are observed, shifting the scale factor between calls; (b) deviation clamp may distort scaled baseline; (c) no explicit renormalization step. Acceptability argument: "VWAP execution algorithms typically use only the next-bin forecast at each step to decide what fraction of the remaining order to execute. They do not require full-day coherent percentage schedules. The paper evaluates Model B using per-bin absolute deviation (MAD), not cumulative deviation." Mitigation: scaled_base recomputation redistributes remaining volume, last-bin absorbs residual, deviation constraint limits distortion, switch-off reverts late-day. Optional explicit renormalization post-processing provided for downstream consumers needing coherence: raw_forecasts normalized to sum to actual_remaining_frac. Researcher inference — paper does not discuss coherence.
**Best version so far:** Run 7

### F87. Re-optimization of weights on full window after regime count selection
**Category:** algorithm
**Consensus** (runs 6, 7): After grid search selects optimal n_regimes using train/validation split, weights are re-optimized on the full N_weight_train window (not just the reduced training-minus-validation subset). The regime classifier is also rebuilt on the full window. Run 6 Function 5 Step 3: "Re-optimize on full training set with chosen N_regimes" — uses all records, not just fit records. Run 7: "standard practice in model selection: use the validation split only for hyperparameter selection, then re-fit on all available data." Researcher inference — paper does not describe validation methodology.
**Best version so far:** Run 6 — explicit Step 3 with full pseudocode showing re-optimization on all records

### F88. Adaptive per-stock deviation limits and switch-off calibration (Function 9a)
**Category:** algorithm
**Single source** (run 6): Complete calibrate_adaptive_limits function with two sub-procedures: (a) Adaptive deviation limit: compute distribution of actual single-bin surprise magnitudes for each stock, set deviation_limit = 95th percentile of |actual_pct - hist_pct|, clamped to [0.5 * base, 2.0 * base] where base=0.10. Ensures illiquid stocks with large surprises get wider limits, while stable large-caps get tighter limits. (b) Adaptive switch-off threshold: compute median cumulative volume fraction at each bin across calibration window; find first bin where median crosses the base threshold (0.80); set switchoff_threshold = median cumulative fraction at that crossover bin, clamped to [0.70, 0.95]. Calibration window: N_calibration=126 days (~6 months). Addresses the paper's "self-updating deviation limits and switch-off parameters" (Satish et al. 2014, p.24, referencing Humphery-Jenner 2011). Researcher inference — the paper does not describe the update mechanism.
**Best version so far:** Run 6

### F89. Sophisticated variant pre-observation baseline with documented context mismatch
**Category:** algorithm
**Single source** (run 6): Detailed treatment of the sophisticated variant's live-prediction mechanism: (a) At start of day, compute Model A's pre-observation forecasts for ALL bins with current_bin=0 (no observations, median regime). (b) Store as fixed model_a_baseline_pct for the entire day. (c) Surprises = actual_pct - model_a_baseline_pct at each bin. Documents a KNOWN TRADE-OFF: the pre-observation baseline uses context-free forecasts for all bins, while the training baseline (generate_model_a_training_forecasts) benefits from intraday ARMA conditioning and accurate regime assignment. For later bins (i=20+), live surprise magnitudes are systematically larger than training surprises because the context-free forecast is less accurate. The deviation clamp bounds practical impact, and this direction (larger live surprises) is "far less harmful than using actual volumes (which would produce near-zero surprises and nullify Model B)." A potential refinement is to compute the live baseline iteratively (forecasting each bin using only earlier-bin observations). Researcher inference — not discussed in paper. See Known Limitation #11.
**Best version so far:** Run 6

### F90. Percentile rank: strict less-than definition
**Category:** algorithm
**Single source** (run 6): Explicit percentile_rank helper with strict-less-than semantics: percentile_rank(x, ref) = count(ref < x) / len(ref). Edge cases: if len(ref)==0, return 0.5 (no information, median). If value < min(ref), returns 0.0 (lowest regime). If value > max(ref), returns 1.0 (highest regime). Equivalent to scipy.stats.percentileofscore(ref, value, kind='strict')/100. Chosen because it produces 0.0 when a stock's volume is at its historical minimum, naturally placing it in the lowest regime. Researcher inference — paper does not specify percentile computation.
**Best version so far:** Run 6

### F91. N_weight_train = 252 and bin 1 percentage initialization
**Category:** parameter
**Single source** (run 6): N_weight_train = 252 trading days (~1 year) for weight optimization training window. Range: [126, 504]. Sensitivity: medium. This is longer than any other run's weight training window (other runs embed weight optimization within shorter regime windows). Also: pct_forecasts[1] = hist_pct[1] explicit initialization — Model B cannot forecast bin 1 (needs at least one observed bin for surprise computation), so the first bin defaults to the historical percentage. Documented in Function 11's pre-market section and Edge Case #2. Researcher inference.
**Best version so far:** Run 6

### F92. VWAP absolute bps values are simulation-specific, not reproducible targets
**Category:** validation
**Single source** (run 7): Explicit clarification (m7) that the absolute VWAP tracking error values (8.74 bps dynamic, 9.62 bps historical) are specific to FlexTRADER simulation conditions: 10% of 30-day ADV order size, day-long orders, May 2011 data (Satish et al. 2014, p.19). A developer should NOT expect to match these absolute numbers. The meaningful benchmark is the relative improvement (~9% reduction) and the statistical significance of the paired test. Absolute VWAP tracking error depends on order size, stock liquidity, market conditions, and execution simulator details. Researcher inference — the paper does not explicitly state this caveat.
**Best version so far:** Run 7

### F93. Separate validation hist_avg in regime grid search to prevent data leakage
**Category:** algorithm
**Single source** (run 7): In train_full_model (Function 9), the regime count grid search computes a separate val_hist_avg = compute_historical_average(volume_history[:val_days_start], N_hist) for the validation period evaluation. This prevents the validation MAPE from unfairly benefiting from H values computed on validation-period data. The temp_model_a used for validation is constructed with val_hist_avg rather than the training hist_avg. This is a distinct anti-leakage measure from the rolling H_d in training (F22) and the percentage training lookahead bias (F36). Researcher inference — the paper does not discuss validation methodology.
**Best version so far:** Run 7

### F94. Near-zero total volume day exclusion in percentage surprise training
**Category:** edge case
**Single source** (run 7): In train_percentage_model (Function 7, Step 2), days where total_vol_d < min_volume_floor are excluded entirely from the surprise training data. This prevents division-by-near-zero in actual_pct = volume[d,i] / total_vol_d computation. Such days are extremely rare for the target universe (top 500 by dollar volume). Distinct from per-bin zero-volume handling (F43) — this is a whole-day filter based on total daily volume. Also: V_total_est approaching zero guard in forecast_volume_percentage (Function 8): all V_total_est divisions are guarded with IF V_total_est > 0. Researcher inference.
**Best version so far:** Run 7

**Run 1 audit:** VERDICT: [50 new, 0 competing, 0 reinforcing] — first run, baseline established
**Run 2 audit:** VERDICT: [15 new, 13 competing, 28 reinforcing] — new findings still emerging, continue
**Run 3 audit:** VERDICT: [15 new, 4 competing, 46 reinforcing] — new findings still emerging, continue
**Run 4 audit:** VERDICT: [2 new, 5 competing, 68 reinforcing] — diminishing returns, consider stopping
**Run 5 audit:** VERDICT: [2 new, 2 competing, 80 reinforcing] — diminishing returns, consider stopping
**Run 7 audit (initial):** VERDICT: [3 new, 1 competing, 83 reinforcing] — diminishing returns, consider stopping
**Run 6 audit:** VERDICT: [4 new, 4 competing, 83 reinforcing] — diminishing returns, consider stopping
**Run 7 re-audit:** VERDICT: [3 new, 0 competing, 91 reinforcing] — all Draft 4 changes reinforcing or new edge cases; saturated, recommend stopping
