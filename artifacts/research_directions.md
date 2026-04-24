# Research Directions Index

## Summary
- Total directions identified: 7
- Total papers across all directions: 9 (4 papers appear in multiple directions)
- Adversarial refinement rounds completed: 1

| Direction | Description |
|-----------|-------------|
| 1. Component Multiplicative Error Model (CMEM) | Multiplicative decomposition of intraday volume into daily, periodic, and dynamic components with MEM/MIDAS estimation |
| 2. PCA Factor Decomposition (BDF) | Cross-sectional PCA extraction of common market factors plus stock-specific ARMA/SETAR residual dynamics |
| 3. GAS-Dirichlet Volume Share Model | Score-driven Dirichlet model for intraday volume shares with GAS dynamics |
| 4. Dual-Mode Volume Forecast (Raw + Percentage) | Four-component weighted combination for raw volume plus a dynamic volume percentage model for step-by-step VWAP execution |
| 5. Quintet Bayesian Ensemble | Five coordinated Bayesian sub-models for daily volume, U-curve, close auction, and intraday updating |
| 6. Doubly Stochastic Binomial Point Process | Probabilistic model treating relative cumulative volume as a Cox-process-derived binomial mixture |
| 7. Kalman Filter State-Space Model | Linear Gaussian state-space model for log-volume with EM calibration and Lasso-robust filtering |

---

## Direction 1: Component Multiplicative Error Model (CMEM)

### Core Model
The Component Multiplicative Error Model (CMEM) decomposes intraday volume into three multiplicative components: a slowly-evolving daily component (eta_t), a deterministic intraday periodic component (phi_i) parameterized via truncated Fourier series, and a fast-moving intraday dynamic component (mu_{t,i}). The observation equation is x_{t,i} = eta_t * phi_i * mu_{t,i} * eps_{t,i}, with all components estimated jointly by GMM. The H-MIDAS-CMEM extends the daily component with a heterogeneous MIDAS cascade operating at daily and sub-daily frequencies.

### Description
The CMEM, introduced by Brownlees, Cipollini, and Gallo (2011), is the benchmark model in the intraday volume forecasting literature. Its key insight is that intraday volume exhibits structure at three distinct time scales -- day-to-day trends, within-day periodicity (the U-shaped pattern), and within-day deviations from that periodicity -- and these are best captured multiplicatively so that each component scales the others. The daily component follows an ARMA-like recursion with optional asymmetric effects tied to market returns. The periodic component uses Fourier harmonics, providing a smooth, parsimonious representation of the U-shape. The dynamic component captures short-lived departures from the seasonal pattern via an intraday ARMA recursion. Estimation uses efficient GMM without distributional assumptions, making the model robust to zero-volume bins. An expectation targeting constraint (E[mu] = 1) reduces the parameter count and ensures identifiability.

Naimoli and Storti (2019) extend the CMEM by replacing the simple GARCH-type daily component with a heterogeneous MIDAS cascade operating at daily and sub-daily (hourly) frequencies. This H-MIDAS-CMEM is motivated by the Heterogeneous Market Hypothesis: different market participants operate at different frequencies, each contributing to volume variation at a different speed. The MIDAS extension uses Beta polynomial weighting to create smooth, decaying lag structures at each frequency. The paper also introduces the Zero-Augmented Generalized F distribution to handle zero-volume intervals, which the original CMEM cannot accommodate through its distributional framework. Empirically, H-MIDAS-CMEM outperforms the original CMEM and all benchmarks on Xetra-traded stocks, with 8-18% MSE reductions.

The CMEM family has known numerical fragility: the GMM objective is not always smooth, and Szucs (2017) documents that naive daily re-estimation required 60 machine-days for an 11-year sample. Grid search with parameter fallback strategies partially mitigates this. Despite these challenges, the model remains the standard reference point against which newer intraday volume models are evaluated.

### Papers
| Paper ID | Title | Role | Year |
|----------|-------|------|------|
| brownlees_cipollini_gallo_2011 | Intra-daily Volume Modeling and Prediction for Algorithmic Trading | Foundational | 2011 |
| naimoli_storti_2019 | Heterogeneous Component Multiplicative Error Models for Forecasting Trading Volumes | Refinement | 2019 |
| szucs_2017 | Forecasting Intraday Volume: Comparison of Two Early Models | Comparison | 2017 |
| chen_feng_palomar_2016 | Forecasting Intraday Trading Volume: A Kalman Filter Approach | Comparison | 2016 |

### Data Requirements
- Tick-by-tick or high-frequency intraday volume data aggregated to regular bins. Brownlees et al. (2011) use 15-minute bins (NYSE, I=26/day); Naimoli and Storti (2019) use 10-minute bins (Xetra, I=51/day with 8.5-hour sessions). The bin-size sensitivity for the MIDAS variant is an open question.
- Shares outstanding for turnover normalization (CRSP, Bloomberg, or equivalent).
- Daily return series for asymmetric effects in the daily component.
- Minimum coverage: several hundred trading days for reliable estimation; Brownlees et al. use 3 years in-sample.
- For H-MIDAS-CMEM: longer history needed (500+ trading days rolling window) due to long-memory MIDAS filters spanning 240-460 daily lags.

### Implementation Notes
- The model involves nested recursions: eta_t updates once per day, mu_{t,i} updates once per bin. Day-boundary initialization requires carrying forward end-of-day state: mu_{t,0} = mu_{t-1,I}.
- Expectation targeting (alpha_0^mu = 1 - beta_1^mu - alpha_1^mu) removes one parameter and enforces E[mu] = 1.
- The periodic component phi_i is estimated jointly with dynamic parameters (not pre-estimated) in the base CMEM.
- For H-MIDAS-CMEM: two-stage estimation (OLS for seasonal Fourier, then ML for remaining parameters); BIC grid search over MIDAS lag length K_d in {200, 220, ..., 500}.
- Recursive weekly re-estimation recommended. The Slicing loss function (cross-entropy between actual and predicted proportions) is the appropriate evaluation metric.
- Known pitfall: GMM initialization sensitivity -- Szucs recommends grid search with fallback to prior-day parameters when optimization fails.
- The Naimoli extension adds ZAF/DZAF distribution handling, Beta polynomial MIDAS weights, and two-stage standard errors via GMM sandwich formula.

### Estimated Complexity
**Medium-high** -- The base CMEM requires joint GMM estimation with nested recursions and Fourier parameterization. The H-MIDAS-CMEM adds MIDAS polynomial filters and the ZAF distribution. Numerical stability requires careful initialization strategies and grid search. Two-stage estimation for the MIDAS variant adds implementation overhead. Szucs documents that estimation time can be prohibitive without fallback strategies.

---

## Direction 2: PCA Factor Decomposition (BDF)

### Core Model
An additive factor model that decomposes intraday turnover into a market-wide common component (extracted via large-dimensional PCA across all stocks) and a stock-specific idiosyncratic component modeled by ARMA or SETAR time-series processes. The decomposition is x_{i,t} = lambda_i' * F_t + e_{i,t}, where F_t are latent common factors and e_{i,t} is the specific residual.

### Description
Bialkowski, Darolles, and Le Fol (2008) introduced this cross-sectional approach to intraday volume forecasting. The key insight is that the well-known U-shaped intraday volume pattern is largely a market-wide phenomenon shared across all stocks, and can be efficiently extracted by principal components analysis applied to the cross-section of stock turnovers. Once this common component is removed, the remaining stock-specific dynamics are simpler and can be modeled by low-order time-series models.

The common component is forecast by a simple historical average at the same time-of-day across prior days, leveraging its stability. The specific component is where the model's predictive power lies: two alternatives are tested -- an ARMA(1,1) model for linear dynamics and a Self-Exciting Threshold Autoregressive (SETAR) model that switches between calm and turbulent volume regimes based on the lagged residual level. The SETAR variant consistently outperforms ARMA, suggesting that stock-specific volume dynamics are nonlinear, with distinct behavior during high-activity vs. low-activity periods.

Szucs (2017) provides a head-to-head comparison of this model against the CMEM on 11 years of DJIA data, finding that the BDF model produces more accurate forecasts under both MSE and MAPE criteria and is orders of magnitude faster to estimate. The computational advantage stems from the two-step structure: PCA is a standard eigendecomposition, and ARMA/SETAR estimation is well-conditioned maximum likelihood. The entire pipeline runs in approximately 2 hours for 33 stocks over 11 years, versus 60+ machine-days for CMEM.

### Papers
| Paper ID | Title | Role | Year |
|----------|-------|------|------|
| bialkowski_darolles_lefol_2008 | Improving VWAP Strategies: A Dynamic Volume Approach | Foundational | 2008 |
| szucs_2017 | Forecasting Intraday Volume: Comparison of Two Early Models | Comparison | 2017 |

### Data Requirements
- Intraday volume data for a cross-section of stocks (the original paper uses 39 CAC40 constituents; Szucs uses 33 DJIA stocks). The cross-sectional dimension is essential -- this model cannot be applied to a single stock.
- Volume measured as turnover (shares traded / float or total shares outstanding), adjusted for splits.
- Aggregated to regular intervals (20-minute in original; 15-minute in Szucs).
- Rolling estimation window: approximately 1 month (500 intraday observations) for PCA; 20 trading days for time-series models.

### Implementation Notes
- PCA estimation follows Bai (2003) large-dimensional factor analysis; the number of factors r is selected via Bai & Ng (2002) information criterion.
- Two-step estimation: (1) extract common and specific components via PCA on rolling window, (2) fit ARMA or SETAR to specific component residuals.
- Factor loadings are refreshed daily on the rolling window; no look-ahead bias since common component forecasts use prior-day estimates.
- The SETAR threshold tau is estimated per stock -- no universal default exists.
- Static VWAP execution (all bins forecast at market open) performs poorly because multi-step ARMA forecasts decay to zero; dynamic execution (updating after each bin) is essential.
- Computational advantage: the entire pipeline (PCA + ARMA/SETAR) runs in approximately 2 hours for 33 stocks over 11 years.

### Estimated Complexity
**Low-medium** -- PCA eigendecomposition and ARMA/SETAR estimation are standard, well-supported by existing libraries. The main implementation effort is managing the rolling window, cross-sectional data alignment, and the dynamic intraday update loop. The SETAR threshold estimation adds modest complexity. Requires maintaining data for a stock universe (not just individual securities).

---

## Direction 3: GAS-Dirichlet Volume Share Model

### Core Model
A Generalized Autoregressive Score (GAS) model that directly models intraday volume shares (proportions) as a Dirichlet-distributed random vector, with time-varying concentration parameters driven by the score of the Dirichlet log-likelihood. The volume share vector w_t ~ Dir(alpha_t), where alpha_{t,i} = exp(pi_i + beta_{t,i}) combines a Fourier periodic component with score-driven dynamics.

### Description
Calvori, Cipollini, and Gallo (2014) propose working directly with volume shares -- the fraction of total daily volume in each bin -- rather than raw or normalized volumes. This sidesteps the local trend problem that plagues models of absolute volume, since shares are by construction bounded in [0,1] and sum to one within a day. The Dirichlet distribution is the natural distributional choice for simplex-constrained data, and the GAS framework provides a principled mechanism for updating the Dirichlet concentration parameters over time using the score of the conditional log-likelihood.

The GAS dynamics capture non-periodic predictable variation in volume shares beyond the deterministic U-shaped seasonality. The model's persistence parameter b consistently exceeds 0.99 across tickers, indicating that departures from the average seasonal pattern are highly persistent from day to day. The paper derives closed-form expressions for the score and information matrix exploiting the Dirichlet structure, which avoids explicit matrix inversion via a Sherman-Morrison decomposition. Four model specifications (M0-M3) range from a static seasonal model to fully bin-specific score coefficients; the simplest dynamic specification M1 (scalar score and persistence parameters) offers the best trade-off between fit and parsimony.

A notable methodological contribution is the demonstration that standard VWAP MSE is misleading for evaluating share forecasts due to intraday algebraic compensations. The paper proposes bin-wise sum-of-squares alternatives (MSE*) that avoid this problem. A limitation is that the model benchmarks only against its own nested M0 specification (static seasonal), not against external models such as CMEM or BDF. Cross-model comparisons would need to be conducted independently.

### Papers
| Paper ID | Title | Role | Year |
|----------|-------|------|------|
| calvori_cipollini_gallo_2014 | Go with the Flow: A GAS Model for Predicting Intra-daily Volume Shares | Foundational | 2014 |

### Data Requirements
- Tick-by-tick transaction data aggregated to regular intraday bins (15-minute intervals, I=26 per day).
- Volume shares computed by dividing each bin's volume by the daily total (observable only at end of day).
- No shares outstanding data needed (shares are self-normalizing).
- Days with missing bins excluded (trading halts, early closures).
- In-sample: multiple years (paper uses 6.5 years, Jan 2006 to Jul 2012); out-of-sample: 206 days.

### Implementation Notes
- The Dirichlet information matrix has a Sherman-Morrison structure: I_t^(alpha) = diag{psi'(alpha_i)} - psi'(alpha_0) * 11', enabling analytic inversion for the S_t = I_t^{-1} scaling (recommended choice). Identity and diagonal-only scalings produced convergence failures for more than half of estimated models.
- GAS recursion: beta_t = A * s_{t-1} + B * beta_{t-1}, where s_t = I_t^{-1} * diag(alpha_t) * [psi(alpha_0) - psi(alpha_i) + ln(w_{t,i})].
- beta_0 initialization: set to zero or estimate as a fixed parameter.
- M1 (scalar a, b) is recommended for finer time resolutions to avoid parameter explosion; M2 (separate coefficients for last 1-2 bins) captures end-of-day effects.
- The model is estimated one day at a time since w_t is only observable at end of day -- this limits real-time intraday updating (identified as an open limitation).
- Use bin-wise MSE* (not standard VWAP MSE) for forecast evaluation.
- Numerical care required for digamma function psi() and trigamma function psi'() at boundary values of alpha.

### Estimated Complexity
**Medium** -- The core GAS recursion is straightforward, and the Dirichlet score/information matrix have closed forms. The main challenges are: (1) correct implementation of the digamma function and its derivatives at numerical boundaries, (2) maximum likelihood optimization over the GAS parameters with score gradient computation via recursive differentiation, and (3) handling the simplex constraint implicitly through the Dirichlet framework. The single-paper direction means all algorithmic details come from one source.

---

## Direction 4: Dual-Mode Volume Forecast (Raw + Percentage)

### Core Model
A two-model system from a single paper: (1) a four-component raw volume model combining a rolling historical average, an inter-day ARMA forecast (per-symbol, per-bin), a deseasonalized intraday ARMA forecast, and a dynamic regime-switching weight overlay; and (2) a companion volume percentage model extending Humphery-Jenner's (2011) dynamic VWAP framework. The raw volume model feeds surprise calculations into the percentage model, making them a tightly coupled pair.

### Description
Satish, Saxena, and Palmer (2014) present a practitioner-oriented volume forecasting system developed at FlexTrade Systems. The direction encompasses two tightly coupled models that together serve the full VWAP execution workflow.

The raw volume model decomposes prediction into four components: (1) a rolling historical average baseline, (2) an inter-day ARMA(p,q) model fitted per symbol and per bin to daily volume series (model orders selected by AICc), (3) a deseasonalized intraday ARMA model where bin volumes are divided by a trailing six-month average before fitting (re-seasonalized after prediction), and (4) a dynamic weight overlay that combines the three component forecasts via in-sample-optimized weights with regime switching based on historical volume percentile cutoffs. The regime switching allows the model to adapt intraday to high-volume vs. low-volume conditions.

The volume percentage model extends Humphery-Jenner (2011) using rolling regression on volume surprises (deviations from a naive historical forecast) to adjust participation rates, with safety constraints (no more than 10% deviation from historical VWAP curve, switch-off once 80% of daily volume is reached). The raw volume model provides the surprise signal that drives the percentage model's adjustments, linking the two implementations. The raw volume model produces full-day forecasts for scheduling tools, while the percentage model produces next-bin forecasts for step-by-step VWAP algorithms.

The paper demonstrates 24% median MAPE reduction for raw volume and 7.6% reduction for volume percentages versus historical baselines on the top 500 U.S. stocks by dollar volume. VWAP tracking error is reduced by 9.1% in simulation. Notably, many specific parameter values and regime thresholds are withheld as proprietary, so a replicator would need to rediscover these through in-sample optimization.

### Papers
| Paper ID | Title | Role | Year |
|----------|-------|------|------|
| satish_saxena_palmer_2014 | Predicting Intraday Trading Volume and Volume Percentages | Foundational | 2014 |
| chen_feng_palomar_2016 | Forecasting Intraday Trading Volume: A Kalman Filter Approach | Comparison | 2016 |

### Data Requirements
- Intraday volume data at regular intervals (15-minute bins validated; 5- and 30-minute bins also tested).
- Universe: top 500 U.S. stocks by dollar volume (NYSE TAQ data).
- Two years of TAQ data; out-of-sample on final year (~250 trading days).
- Trailing six-month average for intraday deseasonalization.
- Rolling one-month window for intraday ARMA fitting.
- For VWAP simulation: 600+ day-long VWAP orders across Dow 30, midcap, and high-variance stocks; order size at 10% of 30-day ADV.

### Implementation Notes
- This direction implements a two-model system from a single paper. The raw volume model is the primary implementation; the volume percentage model is a companion that depends on the raw model's surprise signal. Both should be built as part of a single codebase.
- ARMA order selection uses corrected AIC (AICc, Hurvich and Tsai); all p, q in {0,...,5} considered. Effective intraday lags kept below 5; combined dual ARMA model has fewer than 11 terms.
- Regime-switching weights: separate weight sets trained for different historical volume percentile cutoffs; appropriate set selected intraday based on observed cumulative volume.
- Custom curves for special calendar days (option expiry, Fed events) recommended rather than ARMAX models due to insufficient historical occurrences.
- Specific weighting coefficients, regime bucket counts, and optimal regression terms for the dynamic VWAP extension are proprietary and not disclosed. Replicators must rediscover these via grid search.
- VWAP algorithm simulation (not just forecast error metrics) is recommended as the ultimate validation test.
- **PDF data integrity note:** The local PDF file `papers/satish_saxena_palmer_2014.pdf` may contain a different paper (Hardle, Hautsch & Mihoci on limit order books). The paper summary correctly describes the Satish et al. model and is sufficient for implementation, but developers should verify the PDF or re-download the correct paper before consulting it for fine details.

### Estimated Complexity
**Low-medium** -- The component models (historical averages, ARMA) are standard and well-supported by statistical libraries. The regime-switching weight overlay adds moderate complexity. The main implementation challenge is the grid search over undisclosed hyperparameters (regime thresholds, weighting coefficients) and the two-model architecture (raw volume + volume percentages). The practitioner orientation means the paper prioritizes production deployability.

---

## Direction 5: Quintet Bayesian Ensemble

### Core Model
An ensemble of five coordinated Bayesian sub-models: (1) a historical daily volume prior based on geometric mean with ARMA and special-day adjustments, (2) an intraday U-curve profile estimated via Functional Data Analysis, (3) a close auction volume model, (4) a conjugate-normal Bayesian bin-level updater for liquid securities, and (5) a cumulative-volume Bayesian updater for illiquid securities. All sub-models produce mutually consistent outputs.

### Description
Markov, Vilenskaia, and Rashkovich (2019) approach intraday volume prediction as a Bayesian inference problem. Rather than building a single complex time-series model, they decompose the prediction task into five interpretable sub-problems and solve each with targeted Bayesian methods. The daily volume prior uses a 20-day geometric mean (robust to log-normal outliers) enhanced by ARMA(1,1) for serial correlation and linear regression on event dummies (earnings, option expirations, index rebalancing, overnight gaps). The intraday profile is modeled via Functional Data Analysis, revealing that high-overnight-gap days shift volume from U-shaped to inverted-J-shaped (front-loaded).

The model's distinctive feature is the intraday Bayesian updating mechanism. For liquid securities, per-bin volume observations are combined with the daily volume prior through conjugate normal-gamma updating, transitioning from a Student-t posterior (unknown variance early in the day) to a Gaussian posterior as more observations accumulate. For illiquid securities, where bin-level data is sparse and noisy, the model switches to cumulative volume observations that are inherently smoother. The effective prior sample size kappa_0 controls how quickly intraday evidence overrides the overnight prior -- this is the key tuning parameter that varies by bin size, market cap, and market.

The paper introduces the Asymmetrical Logarithmic Error (ALE) metric, which penalizes volume overestimation at twice the rate of underestimation, reflecting the asymmetric execution risk in algorithmic trading. Close auction volume receives separate treatment with seasonal adjustments for triple witching and quarterly expirations.

### Papers
| Paper ID | Title | Role | Year |
|----------|-------|------|------|
| markov_vilenskaia_rashkovich_2019 | Quintet Volume Projection | Foundational | 2019 |

### Data Requirements
- Daily closing equity volumes (minimum 20 days for prior, 180 days for U-curve history).
- Intraday bin volume time series (10-minute resolution demonstrated).
- Overnight price gaps (open vs. prior close), normalized by 20-day realized volatility.
- Closing auction volume history (separate from continuous session volume).
- Event calendars: earnings dates, option/futures expiration dates, index rebalancing dates.
- Cross-section for functional regression: representative draws from cap-tier indices (S&P 500, S&P Midcap 400, Russell 2000); test sample covers July 2015 to December 2016.

### Implementation Notes
- Modular design: each of the five sub-models can be built and validated independently, then composed at prediction time.
- The liquid/illiquid switch should be triggered by a stability criterion (e.g., fraction of zero-volume bins or variance of per-bin U-curve estimates).
- kappa_0 (effective prior sample size) is the key tuning parameter: 0.5 * N_prior = 10 for 10-minute bins on liquid U.S. names.
- Grubbs filter for outlier removal in log-volume series before fitting the log-normal.
- Transition from unknown-variance (Student-t) to known-variance (Gaussian) posterior needs a heuristic threshold on number of observed bins -- the paper does not specify an exact threshold.
- ARMA(1,1) coefficients are near-universal across S&P 500 names: phi ~0.7, theta ~-0.3.
- Close auction model uses geometric average with seasonal dummy adjustments; ARMA adds marginal improvement (within 5%) and can be omitted.
- The paper does not fully specify integration between close auction Bayesian update and Models 4/5.
- ALE metric: asymmetric L1 norm in log space, weighted 2x for overestimation.

### Estimated Complexity
**Medium** -- Each individual sub-model is relatively simple (geometric means, ARMA, conjugate Bayesian updates, functional regression). The complexity lies in coordinating five sub-models to produce consistent outputs, handling the liquid/illiquid switch, and implementing the Bayesian updating with appropriate variance estimation transitions. The modular design mitigates this by allowing incremental development and testing.

---

## Direction 6: Doubly Stochastic Binomial Point Process

### Core Model
A probabilistic model of relative intraday cumulative volume based on Cox (doubly stochastic Poisson) process theory. If trade arrivals follow a Cox process with stochastic intensity lambda(t), the relative cumulative volume R(t;K) = N(t)/N(T) has a binomial distribution parameterized by the self-normalized integrated intensity p(t) = Lambda(t)/Lambda(T).

### Description
McCulloch (2007) provides a rigorous probabilistic foundation for intraday volume dynamics that differs fundamentally from the time-series decomposition approaches used by all other papers in this collection. Rather than modeling volume as a time series with periodic and dynamic components, the paper treats each trade as a point event in a doubly stochastic Poisson process and derives the distribution of relative cumulative volume analytically. The key result is that the conditional distribution of the fraction of daily trades executed by time t is exactly binomial, with a "success probability" given by the self-normalized integrated intensity of the underlying Cox process.

The unconditional distribution is then a binomial mixture, where the mixing distribution captures day-to-day variation in the shape of trading intensity. By rescaling volume to the unit interval, data from stocks with vastly different daily trade counts can be pooled into a single 2-D histogram (time x relative volume), from which moments of the mixing distribution are extracted via closed-form recursions. The paper demonstrates on NYSE data that stock-specific intensities are best explained by time-scaling (different stocks trade at different speeds through the same baseline intensity) rather than amplitude-scaling, corroborating Ane and Geman's (2000) finding that trade count serves as a stochastic market clock. The variance of the self-normalized intensity scales as 1/sqrt(K), supporting the time-scaling model.

This model provides a theoretical framework and a nonparametric estimator for the expected volume participation curve E[R(t;K)], which is the key input for VWAP scheduling. However, the paper does not specify a parametric form for the mixing distribution or the baseline intensity, leaving these as open extensions. A developer would need to add a parametric forecasting layer on top of the nonparametric foundation.

### Papers
| Paper ID | Title | Role | Year |
|----------|-------|------|------|
| mcculloch_2007 | Relative Volume as a Doubly Stochastic Binomial Point Process | Foundational | 2007 |

### Data Requirements
- Tick-by-tick trade records (individual trades, not aggregated volume).
- NYSE TAQ database or equivalent with timestamps and trade counts.
- Final daily trade count N(T) = K for each stock-day.
- Large sample to build the 2-D histogram: McCulloch uses 203,158 sample paths across all NYSE stocks over 60 trading days.
- Stocks with at least 50 trades/day for reliable histogram estimation.
- A production forecasting extension would require additional data depending on the parametric model chosen for the mixing distribution (e.g., historical time series of mixing-distribution parameters, conditioning variables for regime detection).

### Implementation Notes
- The 2-D histogram (time x relative volume) is the central data structure -- straightforward to build from tick data.
- Use prime numbers of bins (e.g., 253 = 251 interior + 2 boundary) to avoid floating-point rounding artifacts when mapping k/n fractions to bin indices.
- Moments of the mixing distribution are extracted via closed-form recursions (Section 4); numerically stable given sufficient data but require care for small K (denominators involve K-1, K^2-3K+2, etc.).
- V_n constants must be precomputed from the empirical trade-count distribution before extracting moments from pooled data.
- The paper provides a nonparametric framework but does not deliver a parametric forecasting model. A developer would need to add: (a) a parametric model for the mixing distribution (e.g., beta distribution for p(t)), (b) a time-series model for how the mixing distribution evolves day to day, and (c) a mechanism to condition predictions on intraday observations.
- The variance scaling Var[p(t)] ~ 1/sqrt(K) suggests possible connections to self-similar or fractional processes.

### Estimated Complexity
**Medium-high** -- The theoretical framework is well-specified but completing it into a production forecasting model requires substantial extensions beyond what the paper provides. Building the histogram and moment extraction are straightforward; the challenge is designing the parametric forecasting layer on top of the nonparametric foundation. The point-process machinery (Cox processes, filtration enlargement) requires specialized probability theory knowledge for extensions.

---

## Direction 7: Kalman Filter State-Space Model

### Core Model
A linear Gaussian state-space model for log-volume that decomposes log(volume) into three additive components -- daily average (eta_t), intraday periodic (phi_i), and intraday dynamic (mu_{t,i}) -- with Kalman filter recursions for filtering/prediction and EM algorithm for parameter estimation. A robust variant adds Lasso-penalized sparse noise terms for automatic outlier detection and handling.

### Description
Chen, Feng, and Palomar (2016) reformulate the three-component volume decomposition of Brownlees et al. (2011) in log space, converting the multiplicative model into a tractable linear additive one: y_{t,i} = eta_t + phi_i + mu_{t,i} + v_{t,i}. This log transformation is the paper's key insight -- it eliminates positiveness constraints, reduces right-skewness, and makes Gaussian noise assumptions more defensible. The resulting linear state-space model admits exact Kalman filter recursions for one-step-ahead and multi-step-ahead prediction, and closed-form EM updates for all parameters including the seasonality vector.

The state dimension is only 2 (eta and mu), keeping the Kalman filter computationally lightweight -- all matrix inversions reduce to scalar divisions. Within a trading day, eta is held constant (piecewise constant with transitions only at day boundaries), so the filter effectively propagates only mu during intraday steps. The EM algorithm alternates between a forward filter pass, a backward smoother (Rauch-Tung-Striebel), and closed-form parameter updates, converging rapidly and robustly to initialization choice -- a significant practical advantage over the GMM-based CMEM.

The robust extension handles outliers in real-time (non-curated) market data by adding a sparse noise term to the observation equation and solving a Lasso-penalized quadratic problem at each filter step. The threshold adapts dynamically via the innovation variance, providing automatic outlier detection. Empirically, the robust Kalman filter achieves a 64% MAPE improvement over rolling means and 29% over dynamic CMEM across 30 securities on multiple exchanges (NYSE, NASDAQ, NYSEArca, EPA, LON, ETR, Amsterdam, TYO, HKEX). It degrades gracefully under artificial data contamination while CMEM fails entirely at medium/large outlier levels.

### Papers
| Paper ID | Title | Role | Year |
|----------|-------|------|------|
| chen_feng_palomar_2016 | Forecasting Intraday Trading Volume: A Kalman Filter Approach | Foundational | 2016 |

### Data Requirements
- Intraday volume data at 15-minute granularity.
- Volume normalized by daily outstanding shares (for scale correction across splits).
- Non-zero volumes assumed (zero-volume bins must be excluded or imputed; log(0) is undefined).
- Last transaction price per bin for VWAP evaluation.
- Multi-market coverage demonstrated: NYSE, NASDAQ, NYSEArca, EPA (France), LON (UK), ETR (Germany), Amsterdam, TYO (Japan), HKEX (Hong Kong).
- Training window length N selected by cross-validation; separate validation period needed.
- Out-of-sample: 250 days tested (June 2015 to June 2016).

### Implementation Notes
- State dimension = 2 (eta, mu); observation is scalar -- all Kalman operations are 2x2 matrix algebra or scalar.
- The transition matrix A_tau is time-varying: at day boundaries, a^eta governs daily transition with process noise; within a day, a^eta = 1 with zero noise (eta is piecewise constant).
- Seasonality phi is estimated as the average residual over training days for each bin -- a simple mean, not a Fourier parameterization (unlike CMEM).
- EM M-step has closed-form updates for all parameters: a^eta, a^mu, sigma^eta, sigma^mu, r, phi, pi_1, Sigma_1.
- Robust variant: Lasso threshold lambda/(2*W_tau) is time-varying because W_tau depends on current predictive variance; closed-form soft-thresholding solution at each step.
- Cross-validation needed to select N (training window) and lambda (robust regularization).
- Cannot handle zero-volume bins (log(0) is undefined) -- a limitation for illiquid stocks.
- EM convergence is rapid and insensitive to initialization, unlike CMEM's GMM.
- The paper claims superiority over Satish et al. (2014) and CMEM but does not compare against BDF or GAS-Dirichlet.

### Estimated Complexity
**Low-medium** -- The Kalman filter and EM algorithm are well-understood with extensive library support. The state dimension is minimal (2), keeping all operations computationally trivial. The robust Lasso extension adds a soft-thresholding step per bin. The main implementation work is the EM loop with forward filter + backward smoother, day-boundary handling in the transition matrix, and cross-validation for hyperparameters. Among the models in this collection, this is likely the most straightforward to implement correctly.
