# Go with the Flow: A GAS Model for Predicting Intra-daily Volume Shares
**Authors:** Francesco Calvori, Fabrizio Cipollini, Giampiero M. Gallo
**Year:** 2014 (working paper dated November 22, 2013)
**DOI:** N/A (Università di Firenze working paper)

## Core Contribution

This paper proposes a novel Generalized Autoregressive Score (GAS) model for predicting
intra-daily volume shares — the fraction of total daily volume traded in each time bin.
The central motivation is VWAP-based algorithmic trading: under suitable assumptions,
allocating a daily order proportionally to forecasted volume shares replicates the VWAP,
thereby minimizing market impact. Rather than modeling raw or normalized trading volumes
(which suffer from local trends requiring complex daily components), the authors work
directly with volume shares, which are by construction trend-free, bounded in [0,1], and
sum to one across bins within a day.

The model captures two empirical regularities: the well-known intra-daily U-shaped
periodicity in market activity and a significant additional non-periodic predictable
dynamics. By placing the volume share vector in a Dirichlet distributional framework and
letting the Dirichlet parameters evolve according to GAS dynamics, the paper derives a
complete estimation and inference framework, including score-based specification tests.
A companion contribution is a careful analysis of why standard VWAP MSE measures are
misleading for evaluating share forecast quality, and a proposal for better loss functions.

## Model/Algorithm Description

The trading day is divided into I equally spaced bins (the application uses I = 26 bins
of 15 minutes each). The daily vector of volume shares w_t = (w_{t,1}, ..., w_{t,I})
is assumed to follow a Dirichlet distribution conditional on the previous day's
information set F_{t-1}:

    w_t | F_{t-1} ~ Dir(alpha_t),   alpha_t = (alpha_{t,1}, ..., alpha_{t,I})

Each Dirichlet concentration parameter is parameterized as:

    alpha_{t,i} = exp(pi_i + beta_{t,i})

where pi_i is a deterministic periodic component (a Fourier sine/cosine expansion, as
in Brownlees et al. 2011) and beta_t is a time-varying vector driven by GAS dynamics:

    beta_t = A * s_{t-1} + B * beta_{t-1}

Here s_t = S_t * grad_t^(beta) is the scaled score of the Dirichlet log-likelihood with
respect to beta_t, and A and B are (I x I) coefficient matrices. The score with respect
to beta_t is:

    grad_t^(beta) = diag(alpha_t) * grad_t^(alpha)

where grad_t^(alpha) = {psi(alpha_{t,0}) - psi(alpha_{t,i}) + ln(w_{t,i}) : i=1,...,I}
(psi denotes the digamma function, and alpha_{t,0} = sum_i alpha_{t,i}).

Four scaling matrix choices for S_t are considered: identity, diagonal of I_t^{-1/2},
full inverse information matrix I_t^{-1}, and I_t^{-1/2} (Cholesky). The choice
S_t = I_t^{-1} produces a closed-form scaled score that avoids explicit matrix inversion,
exploiting the structure of the Dirichlet information matrix.

Four model specifications for beta_t are compared:
- M0: beta_t = 0 (periodic component only, no GAS dynamics)
- M1: beta_t = a * s_{t-1} + b * beta_{t-1}  (scalar a, b; same across all bins)
- M2: beta_t = diag(a, ..., a, a_{I-1}, a_I) * s_{t-1} + b * beta_{t-1}
      (last two bins get separate score coefficients)
- M3: beta_t = diag(a_1, ..., a_I) * s_{t-1} + b * beta_{t-1}
      (fully bin-specific score coefficients)

Estimation is by maximum likelihood, with the information matrix approximated by the
outer product of gradients. Analytical score computation is derived via recursive
differentiation of the GAS recursion.

Two classes of score-based specification tests are derived:
1. LM_GAS: Lagrange multiplier test for the absence of GAS dynamics (H0: beta_t = 0).
2. ST(1) and ST(p): Score-based autocorrelation tests using moment conditions on lagged
   cross-products and element-wise products of the score, analogous to residual
   autocorrelation diagnostics in standard time-series models.

## Key Parameters

- I: number of intra-daily bins (application uses I = 26, corresponding to 15-minute
  intervals over a 6.5-hour NYSE trading day)
- pi_i: periodic component for bin i, modeled as a Fourier expansion; estimated
  jointly with the GAS parameters
- a (or a_i): score coefficient(s) in matrix A; empirically estimated around 0.02
  for mid-day bins, rising to approximately 0.04 for the last bin of the day
- b: persistence parameter in matrix B; estimated consistently above 0.99 across
  all tickers (e.g., 0.9901 to 0.9959 in M1), implying near-unit-root persistence
  in beta_t
- Scaling matrix: I_t^{-1} is recommended; identity and diagonal-only scalings
  produced convergence failures for more than half of the estimated models

## Data Requirements

- Tick-by-tick transaction data from the NYSE TAQ database (trades only, no quotes
  required)
- Cleaning: standard high-frequency data filters as in Brownlees and Gallo (2006)
- Days with missing bins (trading halts, early closures) are excluded
- Application dataset: six NYSE tickers — ANF (Abercrombie & Fitch), BAC (Bank of
  America), C (Citigroup), F (Ford Motor), GE (General Electric), JNJ (Johnson &
  Johnson)
- In-sample period: January 3, 2006 to July 31, 2012
- Out-of-sample forecast evaluation: August 1, 2012 to May 31, 2013 (206 days)
- Volume shares are computed by aggregating tick-level volumes into 15-minute bins and
  dividing each bin total by the daily total; the daily total is only known at market
  close, so w_t is observable only at end of day t (one-day-ahead forecasting)

## Results

**In-sample specification tests:** LM_GAS and ST(1)/ST(p) tests under M0 reject the
null of no non-periodic dynamics at p < 0.0001 for all six tickers, confirming that GAS
effects are warranted. After fitting M1–M3, the same tests show substantially improved
p-values, with the simplest specification M1 passing most diagnostics; M2 shows further
improvement on GE and JNJ.

**Parameter estimates (M1):** The persistence parameter b exceeds 0.99 for all tickers.
The score coefficient a ranges from 0.0215 (JNJ) to 0.0314 (C). Standard errors are
very small due to the large sample size.

**Model selection:** Likelihood ratio tests uniformly prefer M1 over M0, and M2 over M1.
AIC supports M2 or M3 equally; BIC supports M2. The simplest GAS model M1 provides the
most favorable out-of-sample forecasting profile relative to its parameter count.

**Out-of-sample forecasting (Diebold-Mariano tests):** All GAS specifications (M1–M3)
significantly outperform M0 under all three share-loss functions (log-likelihood, Slicing
loss, Squared Error). M2 improves significantly over M1 only for GE and JNJ. M3 provides
little additional gain over M2 for most tickers.

**VWAP evaluation:** Standard MSE of virtual VWAP values is decomposed into a share
forecast error component (MSE1), a price difference component (MSE2), and a double
product (DP). MSE1 dominates for most tickers, but intra-daily algebraic compensations
can mask improvements: MSE and MSE1 (as squared sums) systematically understate the
benefit of better share forecasts relative to the bin-wise sum-of-squares counterparts
MSE* and MSE1*. Using MSE*, GAS models show substantial reductions in VWAP tracking
error versus both uniform allocation and M0. The volatility-adjusted MSE^(v*) confirms
these findings. On high-volatility days, model improvements are considerably larger in
absolute terms.

## Relationships

This paper directly extends Brownlees, Cipollini, and Gallo (2011), which modeled
intra-daily volume turnover (not shares) via a three-component Multiplicative Error Model
(MEM). The GAS model here addresses the local trend problem that the MEM framework
leaves unresolved by switching from turnover to volume shares. The periodic component
specification (Fourier expansion) is also inherited from Brownlees et al. (2011).

The paper is closely related to Białkowski, Darolles, and Le Fol (2008), who decompose
intra-daily turnover into a market-wide periodic factor and a stock-specific ARMA/SETAR
residual for VWAP trading. The present work simplifies by working with shares and argues
that shares avoid the need for a separate daily-level model.

The GAS framework itself comes from Creal, Koopman, and Lucas (2012) (also called Dynamic
Conditional Score models by Harvey (2013)). The Dirichlet distribution is a natural
exponential family choice for simplex-valued data, and the GAS approach provides a
principled way to update the Dirichlet concentration parameters using the score.

The LM_GAS test is developed in a companion paper by Calvori, Creal, Koopman, and Lucas
(2013). Score-based specification tests follow Newey (1985) and White (1987).

Forecast comparison uses Diebold and Mariano (1995). The VWAP MSE decomposition and the
discussion of misleading loss functions for VWAP evaluation are original to this paper.

## Implementation Notes

**Dirichlet GAS recursion:** The information matrix of the Dirichlet log-likelihood with
respect to alpha_t has a closed-form Sherman-Morrison-type structure
(I_t^{(alpha)} = diag{psi_dot(alpha_i)} - psi_dot(alpha_0) * 11'), which allows
I_t^{-1} to be computed analytically without full matrix inversion. This is critical for
the S_t = I_t^{-1} scaling, which is the recommended choice.

**Score initialization:** The GAS recursion requires an initial condition for beta_0.
The paper does not discuss initialization explicitly; a common approach is to set
beta_0 = 0 or to treat it as a fixed parameter to be estimated.

**Bin granularity and parsimony:** The M1 parameterization (scalar a and b) is
recommended for implementation at finer time resolutions (e.g., 5-minute bins, I = 78)
because M3's bin-specific parameters scale linearly with I, making it impractical.
The end-of-day effect (higher a for last 1–2 bins) observed in M2 and M3 may warrant
a small modification to M1 for finer bins.

**Loss functions for forecast evaluation:** Standard MSE of virtual VWAP should not be
used as the primary criterion for comparing share forecasting models. The bin-wise
sum-of-squares MSE* (Equation 26) is recommended instead, as it avoids intra-daily
algebraic compensations that can reverse model rankings.

**Software:** The paper does not specify implementation software, though the analytic
score and information matrix expressions are provided in closed form and are well-suited
to direct implementation in standard numerical optimization environments (e.g., Python,
MATLAB, R).

**Open questions:** The paper notes that the model is estimated one day at a time
(w_t is not observable until end of day t), which limits real-time application. Extension
to intra-day updating (e.g., as bins are observed within a day) is left as future work.
The effect of bin width on optimal model choice (M1 vs. M2) as I increases is also
identified as an open question.
