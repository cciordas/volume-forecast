# Paper Manifest

**Total papers:** 9
**Papers downloaded:** 9
**Papers unavailable:** 0

## Unavailable Papers -- Action Needed

None.

## Excluded Papers

The following papers were considered but excluded during final review.

- **A Theory of Intraday Patterns: Volume and Price Variability** -- Admati & Pfleiderer, 1988 -- Purely theoretical model of why U-shaped patterns arise from strategic behavior of traders. No implementable forecasting model.
- **Intraday periodicity and volatility persistence in financial markets** -- Andersen & Bollerslev, 1997 -- Primary contribution is volatility periodicity filtering via Flexible Fourier Form. While FFF is adopted by volume models in this collection, the paper itself addresses a different problem (volatility, not volume).
- **Deutsche Mark-Dollar Volatility: Intraday Activity Patterns** -- Andersen & Bollerslev, 1998 -- FX volatility focus; methodology already covered by Andersen & Bollerslev (1997) which is itself excluded as a methodological ancestor.
- **New Frontiers for ARCH Models** -- Engle, 2002 -- Introduced MEM framework but not volume-specific; foundational ancestor to CMEM but does not itself forecast volume.
- **The Dependence between Hourly Prices and Trading Volume** -- Jain & Joh, 1988 -- Empirical documentation of intraday patterns without an implementable forecasting model.
- **Optimal Slice of a VWAP Trade** -- Konishi, 2002 -- Pure VWAP execution optimization, no novel volume forecasting method.
- **Optimal VWAP Trading under Noisy Conditions** -- Humphery-Jenner, 2011 -- Pure VWAP execution paper, no novel volume forecasting method.
- **Spectral Volume Models: Universal High-Frequency Periodicities in Intraday Trading Activities** -- Wu, Zhang & Dai, 2022 -- Ultra-high-frequency focus (10s-5min periodicities), more recent and at the boundary of scope.
- **Point Forecasting of Intraday Volume Using Bayesian ACV Models** -- Huptas, 2019 -- Narrower scope (tested only on Polish stocks), ACV methodology partially subsumed by other models in collection.
- **Wavelet Decomposition for Intra-day Volume Dynamics** -- Manchaldore, Palit & Soloviev, 2010 -- Limited citation impact, methodology well-served by other decomposition models in the collection.

## Papers

### 1. Intra-daily Volume Modeling and Prediction for Algorithmic Trading
- **Authors:** Christian T. Brownlees, Fabrizio Cipollini, Giampiero M. Gallo
- **Year:** 2011
- **DOI:** 10.1093/jjfinec/nbq024
- **Source URL:** https://academic.oup.com/jfec/article-abstract/9/3/489/840918
- **Local path:** papers/brownlees_cipollini_gallo_2011.pdf
- **Key references:** None in this manifest (foundational paper for this collection)
- **Included because:** Introduces the Component Multiplicative Error Model (CMEM), the benchmark model in the intraday volume forecasting literature, with a three-component multiplicative decomposition (daily, periodic, stochastic intraday).

### 2. Improving VWAP Strategies: A Dynamic Volume Approach
- **Authors:** Jedrzej Bialkowski, Serge Darolles, Gaelle Le Fol
- **Year:** 2008
- **DOI:** 10.1016/j.jbankfin.2007.09.023
- **Source URL:** https://www.sciencedirect.com/science/article/abs/pii/S0378426607003226
- **Local path:** papers/bialkowski_darolles_lefol_2008.pdf
- **Key references:** None in this manifest
- **Included because:** Introduces an additive volume decomposition separating market-wide from stock-specific volume dynamics using ARMA and SETAR models. Despite VWAP framing, the volume forecasting model is the primary methodological contribution. Uses cross-sectional and time series data jointly.

### 3. Forecasting Intraday Volume: Comparison of Two Early Models
- **Authors:** Balazs Arpad Szucs
- **Year:** 2017
- **DOI:** 10.1016/j.frl.2016.11.018
- **Source URL:** https://www.sciencedirect.com/science/article/abs/pii/S1544612316301854
- **Local path:** papers/szucs_2017.pdf
- **Key references:** [1] Brownlees et al. 2011, [2] Bialkowski et al. 2008
- **Included because:** Head-to-head comparison of the two foundational volume forecasting models (Brownlees CMEM vs. Bialkowski additive) on 11 years of NYSE/NASDAQ data. Finds Bialkowski's additive model more accurate and computationally faster.

### 4. Predicting Intraday Trading Volume and Volume Percentages
- **Authors:** Venkatesh Satish, Abhay Saxena, Max Palmer
- **Year:** 2014
- **DOI:** 10.3905/jot.2014.9.3.015
- **Source URL:** https://jot.pm-research.com/content/13/4/107
- **Local path:** papers/satish_saxena_palmer_2014.pdf
- **Key references:** [1] Brownlees et al. 2011, [2] Bialkowski et al. 2008
- **Included because:** Practical four-component volume forecast model: rolling historical average, daily ARMA for serial correlation, deseasonalized intraday ARMA, and a dynamic weighted combination. Directly addresses both total volume and volume percentage forecasting.

### 5. Quintet Volume Projection
- **Authors:** Vladimir Markov, Olga Vilenskaia, Vlad Rashkovich
- **Year:** 2017
- **DOI:** 10.3905/jot.2017.12.2.028
- **Source URL:** https://arxiv.org/abs/1904.01412
- **Local path:** papers/markov_vilenskaia_rashkovich_2019.pdf
- **Key references:** [2] Bialkowski et al. 2008, [1] Brownlees et al. 2011
- **Included because:** Bayesian ensemble of five sub-models that jointly predict end-of-day volume, intraday volume U-curve, close auction volume, and special day seasonalities. Emphasizes unified, consistent framework and introduces asymmetric logarithmic error metric.

### 6. Relative Volume as a Doubly Stochastic Binomial Point Process
- **Authors:** James McCulloch
- **Year:** 2007
- **DOI:** 10.1080/14697680600969735
- **Source URL:** https://www.tandfonline.com/doi/full/10.1080/14697680600969735
- **Local path:** papers/mcculloch_2007.pdf
- **Key references:** None in this manifest
- **Included because:** Unique probabilistic model treating cumulative intraday volume as a Cox (doubly stochastic Poisson) point process, showing that relative volume has a binomial distribution. Provides a distinct theoretical and implementable framework from decomposition-based models.

### 7. Go with the Flow: A GAS Model for Predicting Intra-daily Volume Shares
- **Authors:** Francesco Calvori, Fabrizio Cipollini, Giampiero M. Gallo
- **Year:** 2014
- **DOI:** 10.2139/ssrn.2363483
- **Source URL:** https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2363483
- **Local path:** papers/calvori_cipollini_gallo_2014.pdf
- **Key references:** [1] Brownlees et al. 2011, [2] Bialkowski et al. 2008
- **Included because:** Applies Generalized Autoregressive Score (GAS) framework to model volume shares (proportions). Distinct model class from CMEM, with score-driven dynamics that adapt to time-varying patterns. From the same research group as CMEM (Cipollini, Gallo).

### 8. Heterogeneous Component Multiplicative Error Models for Forecasting Trading Volumes
- **Authors:** Antonio Naimoli, Giuseppe Storti
- **Year:** 2019
- **DOI:** 10.1016/j.ijforecast.2019.06.002
- **Source URL:** https://www.sciencedirect.com/science/article/abs/pii/S0169207019301505
- **Local path:** papers/naimoli_storti_2019.pdf
- **Key references:** [1] Brownlees et al. 2011
- **Included because:** Extends CMEM with heterogeneous MIDAS filters operating at multiple frequencies to capture long-run volume dynamics. Provides more flexible long-run component specification. Tested on XETRA stocks.

### 9. Forecasting Intraday Trading Volume: A Kalman Filter Approach
- **Authors:** Ran Chen, Yiyong Feng, Daniel P. Palomar
- **Year:** 2016
- **DOI:** 10.2139/ssrn.3101695
- **Source URL:** https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3101695
- **Local path:** papers/chen_feng_palomar_2016.pdf
- **Key references:** [1] Brownlees et al. 2011, [2] Bialkowski et al. 2008, [4] Satish et al. 2014
- **Included because:** State-space model using Kalman filter with EM algorithm for parameter estimation. Converts multiplicative volume structure to additive via log transform. Uses Lasso for outlier handling. Distinct methodological framework from time-series decomposition models.
