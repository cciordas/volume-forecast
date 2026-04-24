![](_page_0_Picture_0.jpeg)

![](_page_0_Figure_1.jpeg)

Journal of Banking & Finance 32 (2008) 1709–1722

www.elsevier.com/locate/jbf

# Improving VWAP strategies: A dynamic volume approach <sup>q</sup>

Je˛drzej Białkowski <sup>a</sup> , Serge Darolles <sup>b</sup> , Gae¨lle Le Fol c,\*

<sup>a</sup> Department of Finance, School of Business, Auckland University of Technology, New Zealand <sup>b</sup> Socie´te´ Ge´ne´rale Asset Management AI, Center for Research in Economics and Statistics (CREST), France <sup>c</sup> EPEE, University of Evry, and Center for Research in Economics and Statistics (CREST), France

> Received 10 October 2006; accepted 26 September 2007 Available online 4 November 2007

# Abstract

In this paper, we present a new methodology for modelling intraday volume, which allows for a reduction of the execution risk in VWAP (Volume Weighted Average Price) orders. The results are obtained for all the stocks included in the CAC40 index at the beginning of September 2004. The idea of considered models is based on the decomposition of traded volume into two parts: one reflects volume changes due to market evolution; the second describes the stock specific volume pattern. The dynamic of the specific volume part is depicted by ARMA and SETAR models. The implementation of VWAP strategies allows some dynamic adjustments during the day in order to improve tracking of the end-of-day VWAP. -2007 Elsevier B.V. All rights reserved.

JEL classification: C53; G12; G29

Keywords: Intraday volume; Factor models; Volume Weighted Average Price; VWAP strategies

# 1. Introduction

In the existing literature not much attention has been paid to modelling volume observed on stock markets, despite the fact that this is an important market characteristic for practitioners, who aim to lower the market impact of their trades. This impact can be measured by comparing the execution price of an order to a benchmark price. The larger this price difference, the higher the market impact. Volume Weighted Average Price (VWAP)<sup>1</sup> is one such

VWAP execution orders represent around 50% of all the institutional investors' trading.<sup>2</sup> The simplicity of such strategies explains their success. Firstly, investors who ask for VWAP execution accept they will postpone or sequence their trades in order to reduce their trading cost when selling or buying large amounts of shares. By doing so, they reduce their market impact and thus increase the profitability of

benchmark. Informally, the VWAP of a stock over a period of time is the average price paid per share during the given period. The VWAP benchmark is therefore the sum of every transaction price paid, weighted by its volume. The goal of any trader, tracking the VWAP benchmark, is to define ex ante strategies, which ex post leads to an average trading price being as close as possible to the VWAP price. Hence, VWAP strategies are defined as buying or selling a fixed number of shares at an average price that tracks the VWAP benchmark.

<sup>q</sup> This paper was reviewed and accepted while Prof. Giorgio Szego was the Managing Editor of The Journal of Banking and Finance and by the past Editorial Board.

<sup>\*</sup> Corresponding author.

E-mail address: [Gaelle.Le-Fol@ensae.fr](mailto:Gaelle.Le-Fol@ensae.fr) (G. Le Fol). <sup>1</sup> See [Madhavan \(2002\)](#page-13-0) for a detailed presentation of strengths and weaknesses of VWAP strategies, and Ting (2006) on using VWAP prices rather than closing prices in quantitative studies.

<sup>2</sup> See ''Marching up the learning curve: The second buy-side algorithmic trading survey'', Bank of America, 12 February 2007.

their transactions by accepting a risk in time. Likewise, VWAP orders allow foreign investors to avoid the high risk related to the fact that their orders have to be placed before the market opens. Secondly, it is a common practice to evaluate the performance of traders based on their ability to execute the orders at a price better or equal to VWAP. In this case, VWAP can be seen as an optimal benchmark.<sup>3</sup> Finally, VWAP is a better benchmark than any fixed time benchmark that improves both market transparency and efficiency (see [Cushing and Madhavan, 2000\)](#page-13-0).

Successful implementation of VWAP strategies requires a reliable model for an intraday evolution of the volume. Surprisingly, the model for an intraday price evolution is not necessary. In fact, only large enough traders are able influence price and then make the VWAP whatever their strategy is. In practice, investors' trades are much smaller than the market. Thus, if market participants want to trade at a price as close as possible to the VWAP benchmark, they need a model capable of predicting volume evolution.

If volume has been analysed in the financial market literature, it has often been used for a better understanding of other financial variables, such as price ([Easley and O'Hara,](#page-13-0) [1987; Foster and Viswanathan, 1990,](#page-13-0) for example) or volatility [\(Tauchen and Pitts, 1983; Karpoff and Boyd, 1987;](#page-13-0) [Andersen, 1996; Manganelli, 2002](#page-13-0), for example). Moreover most of these studies use daily or even lower data frequency (one exception is [Darrat et al., 2003](#page-13-0) who examined intraday data of stocks from the Dow Jones index, and reported significant lead-lag relations between volume and volatility). The rare studies that solely concentrate on volume are [Kaastra and Boyd \(1995\) and Darolles and Le Fol](#page-13-0) [\(2003\)](#page-13-0). It is now common knowledge that intraday volume moves around a U-shape seasonal pattern (see for example [Biais et al., 1995; Gourie´roux et al., 1999](#page-13-0) for the French stock market.). These seasonal fluctuations have hampered volume modelling. One way to circumvent this problem is to work on a transaction or market time scale instead of a calendar time scale (see [Engle, 2000; Gourie´roux and Le](#page-13-0) [Fol, 1998](#page-13-0), for example). However, this transformation is useless when working on strategies that are defined on a calendar time scale ([Le Fol and Mercier \(1998\)](#page-13-0) suppose that the time transformation is fixed and use this hypothesis to pass from one time scale to the other). Some approaches correct volume on a stock-by-stock time varying average volume ([Engle and Russel, 1998; Easley and](#page-13-0) [O'Hara, 1987\)](#page-13-0), while others take the time varying across stock and average volume (see [McCulloch, 2004\)](#page-13-0). In all these studies, seasonal variation is just a problem that is adequately and empirically disposed of. However, in this paper we do not eliminate seasonality because it is recognized as a factor affecting all stocks traded on examined markets. Therefore, seasonality is used to construct our volume benchmark for VWAP strategies.

The aim is to discriminate between the seasonal part of volume and the dynamic one. The identification of such components of volume comes from the observation that seasonal fluctuations are common across stocks, whereas the dynamic approach is individually stock specific. It makes our approach different from the usual static method of improving VWAP trades (see Konishi, 2002). The main advantages of our approach are efficiency and simplicity in comparison with the static improvement of VWAP proposed in the previous studies.

This paper is in line with the methodology proposed by [Darolles and Le Fol \(2003\)](#page-13-0) for volume decomposition. It has the following contribution to the existing literature. Firstly, it proposes an alternative model for intraday volume. Secondly, it proposes a dynamically updated prediction of volume based on the combination of an unobserved factors model and times series models. Finally, we use VWAP strategies to test the accuracy of the approach.

Basically, volume is decomposed into two components: the first describes the size of volume on ordinary days and is extracted from the stocks included in the CAC40 index. The second component measures the abnormal or unexpected changes of volume specific to each stock. The CAPM is one of the most famous models for returns that are based on such techniques. [Lo and Wang \(2000\)](#page-13-0) were the first ones to apply this idea to volumes (see also [Darol](#page-13-0)[les and Le Fol, 2003](#page-13-0)). This study is a natural extension of their work on high-frequency data relating to the problem of optimal executions of VWAP orders. Furthermore, it is worth highlighting that, by separating the market part from observed volume, two additional goals were obtained. Firstly, the specific component, as a measure of liquidity for a particular company, is a much more reliable indicator of arbitrage activity than the observed volume. Secondly, this decomposition allows us to accurately remove seasonal variations, without imposing any particular form. This result contradicts [Hobson \(2006\)](#page-13-0) who concluded that refinements to the volume profile do not yield significant benefits of VWAP execution strategies.

The paper is organized as follows: Section 2 commences with a simple example illustrating why only volume is an important variable for a trader who is tracking VWAP. It next provides a description of the proposed models for a market component and a specific component of volume. Section [3](#page-3-0) contains a description and summary statistics of the data, as well as estimations and results of in and out samples. Applications of the model to VWAP strategies are presented in Section [4.](#page-5-0) Section [5](#page-12-0) concludes the paper.

# 2. The volume-trading model

In this section we first explain why tracking of VWAP can be achieved without a prediction of prices. Next, we introduce the volume statistical model that includes the method of volume decomposition and the intraday volume dynamics.

<sup>3</sup> [Berkowitz et al. \(1988\)](#page-13-0) show that VWAP is a good proxy for the optimal price attainable by passive traders.

<span id="page-2-0"></span>The major problem of intraday volume is its high intraday seasonal variation. Two approaches have been considered to deal with this problem. The first takes an historical average of volume for any stock as its seasonal pattern or normal volume (Easley and O'Hara, 1987). The second takes the average volume across stocks to get this normal volume (McCulloch, 2004). The volume seasonality comes from market characteristics and habits and is not a stock specific feature. We extend McCulloch's (2004) method to extract the common volume with a factor analysis. Such a method allows us to get a "normal" volume component, which is common across stocks and exhibits intra-daily seasonality, and a specific stationary component. Next, we propose to model the dynamics of the aforementioned components taken separately.

# 2.1. Predicting volume to track the VWAP

As mentioned above, the goal of any trader tracking the VWAP is to define ex ante strategies, which ex post lead to an average trading price as close as possible to the VWAP. In order to achieve the goal, the trader does not need to predict the sequence of prices. In fact, as soon as the trader knows the future sequence of intra day trading volume, he can adapt his trading scheme accordingly. In doing so, if the trader can mimic perfectly the intraday volume sequence of the day, the trader can execute an order at VWAP price whatever the intraday price evolution. As a consequence, the problem is to adequately forecast the intra day volume.

To see this, let us consider a simple financial market where trades can only occur only three times a day. At the end of the day, we can see that 5000 shares have been traded: 2500 at time 1 with price 100, 1000 at time 2 with price 101 and 1500 at time 3 with price 102. The VWAP of the day is then equal to  $\frac{100 \times 2500 + 101 \times 1000 + 102 \times 1500}{5000} = 100.8$ .

We assume that, that day, a trader Y had to trade 1/100th of the daily volume and that he knew that the volume pattern was going to be 2500 (50%), 1000 (20%) and 1500 (30%).

- If we suppose that he is able to trade without impact on prices, he just has to trade 25, 10 and 15 shares at times 1, 2 and 3, respectively, in order to get the VWAP. In fact, his transaction price is 100×25+101×10+102×15/50 and is equal to the VWAP of the day.
- If his trades have an impact on prices and become 101, 101.5 and 103, he will get the unitary price  $\frac{101 \times 25 + 101.5 \times 10 + 103 \times 15}{50} = 101.7$ . This price is greater than the VWAP that would prevail if he had not traded. However, the VWAP of the day include trader *Y*'s trades and is calculated ex post with all the observed prices. The VWAP of the day would then be

This simple example highlights the importance in finding a good volume-trading model, which is the aim of the following sections.

#### 2.2. Intraday volume decomposition

Following Darolles and Le Fol (2003), the volume is measured by the turnover. Let  $x_{it} = \frac{V_{it}}{N_{it}}$ , i = 1, ..., N, t = 1, ..., T denote the turnover series for stock i at date t, i.e. the number of traded shares  $V_{it}$  divided by the number of floated shares  $N_{it}$ .  $x_{i,t}$  is the panel structure of volume. As shown in Darolles and Le Fol (2003), the market turnover  $x_t^T$  can be written as:

$$x_{t}^{I} = \frac{\sum_{i} P_{it} V_{it}}{\sum_{k} P_{kt} N_{kt}} = \frac{\sum_{i} P_{it} N_{it} \frac{V_{in}}{N_{it}}}{\sum_{k} P_{kt} N_{kt}} = \sum_{i} w_{it} x_{it}, \tag{1}$$

where  $P_{it}$  is the transaction price for stock i at date t, and  $w_{it}$  is the stock relative capitalization.

A natural extension of the CAPM (returns decomposition) to volume is to regress  $x_{i,t}$  on  $x_t^I$  imposing a one-factor model. Following Lo and Wang (2000) and Darolles and Le Fol (2003), we use a more general specification given by:

$$x_{i,t} = \lambda_i' F_t + e_{i,t} = c_{i,t} + e_{i,t}, \tag{2}$$

where  $F_t$  is a vector (r,1) of unobservable common factors,  $\lambda_i$  is a vector (r,1) of factor loadings, and  $e_{i,t}$  is the idiosyncratic component of  $x_{i,t}$  (see Bai (2003) for general assumptions on  $e_{i,t}$  to ensure consistency of the estimation). Lo and Wang (2000) use such a model with a factor dimension r which is equal to one. In fact, we extend Lo and Wang's approach in two ways: first we use a less restrictive assumption -r fixed, and we estimate this dimension following Bai and Ng (2002). Second, the model is estimated as in Bai (2003), which provides convergent estimators when the errors are serially correlated.

Usually, the cross-section dimension N is small compared to the time series dimension T. In particular, this allows us to cast the model under a state-space set-up and to estimate the unobserved factors by maximizing the Gaussian Likelihood via Kalman Filter. As N increases, the state-space and the number of parameters to be estimated increases and the estimation problem becomes impossible to solve. However, factor models can also be estimated by the method of principal components. In particular, Bai (2003) provides rates of convergence and limiting distributions in the general setting of large N and large T.

Eq. (2) can be rewritten as an N-dimension time series with T observations:

$$X_t = \Lambda F_t + e_t, \tag{3}$$

 $<sup>\</sup>frac{101 \times 2500 + 101.5 \times 1000 + 103 \times 1500}{5000} = 101.7$  and again, the daily price obtained by mimicking the intraday volume pattern matches the VWAP.

<sup>&</sup>lt;sup>4</sup> Or as soon as he is able to predict the future sequence of volume.

<span id="page-3-0"></span>where  $X_t = (x_{1,t}, \dots, x_{N,t})'$ ,  $\Lambda = (\lambda_1, \dots, \lambda_N)$  and  $e_t = (e_{1,t}, \dots, e_{N,t})'$ , or as a *T*-dimension system with *N* observations:

$$\underline{X}_i = F\lambda_i + \underline{e}_i,\tag{4}$$

where  $\underline{X}_i = (x_{i,1}, \dots, x_{i,T})'$ ,  $F = (F_1, \dots, F_T)'$ , and  $\underline{e}_i = (e_{i,1}, \dots, e_{i,T})'$ . The matrix notation is then:

$$X = F\Lambda' + e, (5)$$

where  $X = (X_1, ..., X_N)$ , and  $e = (e_1, ..., e_N)$ , and the matrices  $\Lambda$  and F are unknown. The method of principal components minimizes the following objective function:

$$V(r) = \min_{A,F} (NT)^{-1} \sum_{i=1}^{N} \sum_{t=1}^{T} (X_{i,t} - \lambda_i' F_t)^2.$$
 (6)

Concentrating out  $\Lambda$  and using the normalization  $F'F/T = I_r - \text{an } (r \times r)$  identity matrix, the problem is identical to maximizing tr[F'(X'X)F]. The estimated factors matrix  $\widehat{F}$  is proportional (up to  $T^{1/2}$ ) to the eigenvectors corresponding to the r-largest eigenvalues of the X'X matrix, and  $\widehat{\Lambda}' = (\widehat{F}'\widehat{F})^{-1}\widehat{F}'X = \widehat{F}'X/T$  are the corresponding factor loadings.

#### 2.3. Intraday volume dynamics

Using the above model, we are left with a filtering-fore-casting problem of the components. The first step consists in using a time-wise filter. We estimate Eq. (2) at date  $t_1$  on past observations  $t_1$ ,  $t_1 - 1$ , ...,  $t_1 - h + 1$ , to get  $\hat{\lambda}$  and  $\hat{F}$ . We get the filtered volume:

$$\hat{x}_{i,t} = \hat{\lambda}_i' \hat{F}_t + \hat{e}_{i,t} = \hat{c}_{i,t} + \hat{e}_{i,t}, \quad t = t_1 - h + 1, \dots, t_1.$$
 (7)

From  $\hat{\lambda}_i$  and  $\hat{F}$ , we can forecast  $\hat{x}_{i,t_1+1}$ :

$$x_{i,t_1+1} = \hat{c}_{i,t_1+1} + \hat{e}_{i,t_1+1}, \tag{8}$$

with  $\hat{c}_{i,t_1+1}$  and  $\hat{e}_{i,t_1+1}$  to be defined. The first part of the right-hand-side of the equation,  $\hat{c}_{i,t_1+1}$ , has a pretty stable seasonal shape across time. Note that we can observe upward or downward shifts but the general form is preserved which has no impact on the VWAP. The second part of the equation allows for dynamic modelling.

In order to incorporate the aforementioned seasonal features into the model for intraday volume, we propose for the common component an historical average of the common components of intraday volume over the last *L*-trading days and equal to:

$$\hat{c}_{i,t_1+1} = \frac{1}{L} \sum_{l=1}^{L} c_{i,t_1+1-k \times l}, \tag{9}$$

with k being the number of lags needed to ensure that we are averaging across days for any date t. In our empirical example, k = 25 and the size of the interval  $\delta t$  is equal to 20 min.<sup>5</sup>

The second term  $e_{i,t}$  represents intraday specific volume for each equity and is modelled by considering two specifi-

cations. The first one is an ARMA(1,1) with white noise, defined as:

$$e_{i,t} = \psi_1 e_{i,t-1} + \psi_2 + \epsilon_{i,t}. \tag{10}$$

The alternative considered specification is a SETAR (self-extracting threshold autoregressive model) that allows for changes in regime in the dynamics. We get:

$$e_{i,t} = (\phi_{11}e_{i,t-1} + \phi_{12})\mathbf{I}(e_{i,t-1}) + (\phi_{21}e_{i,t-1} + \phi_{22})[1 - \mathbf{I}(e_{i,t-1})] + \epsilon_{i,t}.$$
(11)

where I(x) equals to 1 when  $x \le \tau$  and 0 elsewhere. Therefore, we assume that when the specific part of intraday volume exceeds a threshold value of  $\tau$  its dynamics is described by a different set of parameters.

We use a two-step procedure to estimate the parameters in Eqs. (10) and (11). First, we extract the residuals  $\hat{e}_{i,t}$  conducting a factor analysis as described in Section 2.2. Now, substitute  $e_{i,t}$  for  $\hat{e}_{i,t}$  in Eqs. (10) or (11) and estimate Eqs. (10) or (11) by maximum likelihood.

### 3. Empirical analysis

#### 3.1. The data

The empirical results are based on the analysis of all the securities included in the CAC40 index at the beginning of September 2004. Tick-by-tick volume and prices were obtained from the EURONEXT historical database. We consider a one-year sample, ranging from the beginning of September 2003 to the end of August 2004. The data is adjusted for the stock's splits and dividends. The 24th and 31st of December 2003 were excluded from the sample. For any 20-min interval, volume means the sum of the traded volumes and price is the average price for the period.

This study focuses on modelling volume during the day with continuous trading; therefore we consider transactions between 9:20 and 17:20, and exclude pre-opening trades, see Fig. 1. As a result, there are 25 (k = 25) 20-min intervals  $(\delta t = 20mn)$  per day. In addition to high-frequency data from EURONEXT, volume weighted average prices, with a daily horizon for each company, were used. Finally, we chose a 20-day window to construct the common component and L = 20 in Eq. (9).

In Table 1, we present intraday volume summary statistics for securities from the CAC40 index. The comparison

![](_page_3_Figure_31.jpeg)

Fig. 1. Decomposition of the trading day into 20-min intervals.

<sup>&</sup>lt;sup>5</sup> Replacing c by x in Eq. (9), we get the classical prediction of volume that we will call classical approach in the following text, tables and figures.

<span id="page-4-0"></span>Table 1 Summary statistics for the intraday aggregated volume over 20-min intervals, 2 September 2003 to 31 August 2004

| Companies          | Mean   | Std    | Q5     | Q95    | Companies          | Mean   | Std    | Q5     | Q95    |
|--------------------|--------|--------|--------|--------|--------------------|--------|--------|--------|--------|
| ACCOR              | 0.0191 | 0.0273 | 0.0028 | 0.0523 | MICHELIN           | 0.0167 | 0.0238 | 0.0024 | 0.0450 |
| AGF-ASS.GEN.FRANCE | 0.0076 | 0.0087 | 0.001  | 0.0212 | PERNOD-RICARD      | 0.0157 | 0.0303 | 0.0022 | 0.0427 |
| AIR LIQUIDE        | 0.0120 | 0.0182 | 0.0022 | 0.0314 | PEUGEOT            | 0.0205 | 0.0454 | 0.0035 | 0.0515 |
| ALCATEL            | 0.0381 | 0.0383 | 0.0062 | 0.1064 | PINPRINT.REDOUTE   | 0.0149 | 0.0210 | 0.0020 | 0.0426 |
| ARCELOR            | 0.0234 | 0.0241 | 0.0034 | 0.0648 | RENAULT            | 0.0165 | 0.0414 | 0.0024 | 0.0412 |
| AXA                | 0.0166 | 0.0220 | 0.0034 | 0.0404 | SAINT GOBAIN       | 0.0154 | 0.0332 | 0.0030 | 0.0382 |
| BNP PARIBAS        | 0.0147 | 0.0350 | 0.0034 | 0.0338 | SANOFI-AVENTIS     | 0.0151 | 0.0228 | 0.0020 | 0.0444 |
| BOUYGUES           | 0.0129 | 0.0264 | 0.0019 | 0.0344 | SCHNEIDER ELECTRIC | 0.0145 | 0.0264 | 0.0021 | 0.0378 |
| CAP GEMINI         | 0.0438 | 0.0514 | 0.0058 | 0.1241 | SOCIETE GENERALE   | 0.0155 | 0.0205 | 0.0031 | 0.0390 |
| CARREFOUR          | 0.0132 | 0.0232 | 0.0025 | 0.0317 | SODEXHO ALLIANCE   | 0.0172 | 0.0318 | 0.0016 | 0.0518 |
| CASINO GUICHARD    | 0.0106 | 0.0118 | 0.0013 | 0.0312 | STMICROELECTRONICS | 0.0223 | 0.0230 | 0.0030 | 0.0604 |
| CREDIT AGRICOLE    | 0.0083 | 0.0120 | 0.0012 | 0.0233 | SUEZ               | 0.0162 | 0.0182 | 0.0032 | 0.0418 |
| DANONE             | 0.0149 | 0.0310 | 0.0024 | 0.0381 | TF1                | 0.0198 | 0.0449 | 0.0026 | 0.0531 |
| DEXIA              | 0.0055 | 0.0069 | 0.0006 | 0.0164 | THALES             | 0.0120 | 0.0134 | 0.0016 | 0.0336 |
| EADS               | 0.0092 | 0.0092 | 0.0015 | 0.0265 | THOMSON (EX:TMM)   | 0.0270 | 0.0465 | 0.0035 | 0.0776 |
| FRANCE TELECOM     | 0.0123 | 0.0115 | 0.0025 | 0.0312 | TOTAL              | 0.0150 | 0.0277 | 0.0031 | 0.0373 |
| L'OREAL            | 0.0069 | 0.0120 | 0.0014 | 0.0177 | VEOLIA ENVIRON.    | 0.0120 | 0.0158 | 0.0017 | 0.0333 |
| LAFARGE            | 0.0188 | 0.0307 | 0.0035 | 0.0477 | VINCI (EX.SGE)     | 0.0261 | 0.0687 | 0.0034 | 0.0689 |
| LAGARDERE S.C.A.   | 0.0163 | 0.0385 | 0.002  | 0.0423 | VIVENDI UNIVERSAL  | 0.0215 | 0.0203 | 0.0044 | 0.0543 |
| LVMH               | 0.0105 | 0.0185 | 0.0018 | 0.0276 |                    |        |        |        |        |
| Overall            | 0.0166 | 0.0265 | 0.0026 | 0.0445 |                    |        |        |        |        |

of the mean with the 5% and 95% quantiles gives clear indications of the large dispersion of volume stock by stock. For companies like SODEXHO ALLIANCE, SANOFI-AVENTIS and CREDIT AGRICOLE, the mean is around three times lower than the 95%-quantile. On average, the ratio mean to 95%-quantile is equal to 2.70. In turn, 5% quantiles are five to nine times smaller than the mean. This strong dispersion comes from the strong intraday seasonal variation. It is worth noting that the table also shows large dispersion across equities, where the average volume ranges from 0.006 for DEXIA up to 0.0438 for CAP GEMINI. The explanation comes from the equities' particular events, such as earning announcements, dividend payments, changes in management board, etc., which have direct influence on the price and volume of their stock. These observations encourage the application of a model, such as the one we propose, which is based on volume decomposition in the market and its specific components.

# 3.2. Estimation results

The first step of our methodology is to run a factor analysis on the intraday volumes for all companies included in CAC40. Over long periods factor analysis fails to capture the dynamic links that prevail. Therefore, we chose to work on a 1-month period to decompose volume. Next, we calculated the autocorrelation (ACF) and partial autocorrelation functions (PACF) for common and specific parts, which are plotted in [Fig. 2](#page-5-0) for TOTAL equity. The upper graphs in the figure show typical characteristics of the intraday volume, namely seasonal variations. From the middle figures, one recognizes the ability of common components to capture seasonal variations. The last graphs illustrate ACF and PACF for the specific parts of volume. The fast decay of the autocorrelation suggests that the ARMA type model is suitable to depict this time series.<sup>6</sup>

[Fig. 3](#page-6-0) shows the result of our decomposition for two succeeding days, for TOTAL company. The upper graphs give the intraday evolution of volume where we can see a stochastic evolution around a seasonal U-shape pattern. The middle graphs give the intraday evolution of the common component. This part of the volume is the same for any day of the sample. Finally, the lower graph represents the evolution of the specific component. This component is responsible for the stochastic behaviour around the seasonal pattern and changes from day to day.

The final stage to evaluate the accuracy of the models is to use the mean absolute percentage error (MAPE).

The summary<sup>7</sup> for all examined companies is given in the first panel of [Table 2.](#page-7-0) To get summary results, we calculate MAPE of a market portfolio and then calculate the mean, standard error and last 5%-quantile. The outcomes indicate that both models based on principal component decomposition outperform the classical approach to predict the daily U-shape of volume. Moreover, the SETAR model fits the daily volume dynamics better than the ARMA model does. In fact, there are only three, of the 31, companies for which the ARMA slightly surpasses the SETAR model. A further argument in favour of the decomposition concept comes from the fact that the standard deviation and the 95%-quantile for both models are

<sup>6</sup> The inspection of residuals confirmed that stationary time series models are accurate to describe the dynamics of the specific volume. The residuals exhibit the characteristic of white noise. The conclusions drawn from autocorrelation function plots are further confirmed by the results of Portmanteau tests. These results are not shown in the paper but available upon request.

<sup>7</sup> Stock by stock results are available upon request.

<span id="page-5-0"></span>![](_page_5_Figure_2.jpeg)

Fig. 2. Autocorrelation and partial autocorrelation functions of the two components, TOTAL stock.

significantly smaller than the one observed for the classical approach.

To summarize, we have demonstrated that models based on decomposition are better in modelling intraday volume than those assuming the calculation of simple averages from historical data. The importance of this outcome will be discussed in the next section, which focuses on the problem of reducing the cost of VWAP orders.

# 4. Application to VWAP strategies

# 4.1. VWAP strategies: An overview

# 4.1.1. Trends in algorithmic trading

An actual trend observed in financial markets is the increase of usage of algorithmic trading. Measurability is one of the more obvious benefits of benchmarking. Two main factors explain this phenomenon. On the one hand, the computer-trading offer is now easily accessible. Sell-side firms' execution systems have been used internally by traders for years; these systems have become recently available directly to clients via electronic platforms. At the same time, firms are looking for ways to outsource their trading desks, to increase their capacity to execute more volume. Major brokerage houses are then franchising their computer-trading strategies to smaller firms, which in turn are pressured to offer the service. Small and midsize broker–dealers that lack resources and time to invest in developing VWAP engines and other quantitative strategies can then offer algorithmic trading to their buy-side customers.

On the other hand, buy-side customers are asking for the algorithms. The buy-side is being more closely moni-

<span id="page-6-0"></span>![](_page_6_Figure_2.jpeg)

Fig. 3. TOTAL stock daily volume patterns on 9 (left) and 10 (right) September 2003. The first two graphs represent the intraday turnover evolution. The next two give the common component evolution and the final two, the specific component evolution.

tored and scrutinized for its execution quality. Pre-trade analytic tools are readily and easily available in this execution environment. It allows clients to obtain analysis relevant to the context in which they make trades. Moreover, market fragmentation drives traders to use electronic tools to access the market in different ways. Quant fund traders have begun to be a larger part of the market liquidity and they need flexible and easy access to the market.

# 4.1.2. VWAP benchmark

Several benchmarks are proposed in the field of algorithmic trading but the most common and popular one is VWAP. The main reason is obvious: the computation of daily VWAP is straightforward for anyone with access to daily stock transactions records. In general, brokers propose several ways to reach the VWAP benchmark. Agency and guaranteed VWAP execution services are the two main possibilities. In the guaranteed case, the execution is guaranteed at VWAP for a fixed commission per share, and the broker–dealer ensures the entire risk of failing to meet the benchmark. In the agency trading case, the order is sent to a broker–dealer, to trade on an agency basis, with the aim of obtaining the VWAP or better. Obviously, the transaction costs are not the same – they depend on the method chosen – and the larger the client's residual risk, the smaller the cost.

# 4.1.3. Timing dimension

VWAP strategies introduce a time dimension in the order execution process. If the trader loses control of whether the trade will be executed during the day, VWAP strategies allow the trader to dilute the impact of orders

<span id="page-7-0"></span>Table 2 Comparison of intraday volume and VWAP predictions, based on mean absolute percentage error (MAPE), for the period from 2 September to 16 December 2003

| Models                                         | Mean        | Std    | Q95    |
|------------------------------------------------|-------------|--------|--------|
| Performance of model for intraday volume       |             |        |        |
| SETAR                                          | 0.0752      | 0.0869 | 0.2010 |
| ARMA                                           | 0.0829      | 0.0973 | 0.2330 |
| Classical approach                             | 0.0905      | 0.1050 | 0.2490 |
| Result of in-sample estimation for VWAP        | order execu | ıtion  |        |
| PC-SETAR                                       | 0.0706      | 0.0825 | 0.2030 |
| PC-ARMA                                        | 0.0772      | 0.0877 | 0.2173 |
| Classical approach                             | 0.1140      | 0.1358 | 0.3702 |
| Result of out-sample estimation for VWAI       | order exec  | cution |        |
| PC-SETAR theoretical                           | 0.0770      | 0.0942 | 0.2432 |
| PC-ARMA theoretical                            | 0.0833      | 0.0956 | 0.2498 |
| PC-SETAR with dynamical adjustment of forecast | 0.0898      | 0.0954 | 0.2854 |
| PC-ARMA with dynamical adjustment of forecast  | 0.0922      | 0.0994 | 0.2854 |
| Classical approach                             | 0.1006      | 0.1171 | 0.3427 |

*Note:* The cost of the VWAP order execution is calculated for a basket made of all stocks included in the CAC40 index. Therefore, we compute the VWAP for the whole index based on the average of VWAP over equities. We use the same weights as used for construction of the index at the beginning of September 2004. All costs are expressed in percentage of the end of the day volume weighted price.

through the day. To understand the immediacy and good price trade-off, let us take the two examples of action and investor traders. Action traders go where the action is, meaning that they do not care about the firm stock they are trading. Investor traders lack that flexibility. Since their job represents the final task in a sequential decision process, they are expected to trade specific stocks, even if the action is over. Of course, trade information cannot remain proprietary for long and trade delays, resulting in trade process lags, can cause marked variations from the manager's original decision price. VWAP strategies ensure investor traders' good participation during the day, and then trade completion at closing time.

#### 4.1.4. Size effect

Under particular conditions, VWAP evaluation may be misleading and even harmful to portfolio performance. When large numbers of shares must be traded, liquidity concerns are balanced against price goals and trade evaluation becomes more complicated. Action traders watch the market for this reason and try to benefit from those trades. Naive investors could indiscreetly reveal their interest for the market or a particular stock. Action traders can then cut themselves in by capturing available liquidity and reselling it to unskilled traders. Using automatic participation strategies as VWAP may be dangerous in these cases. Since it pays no attention to the full size of the trade, trading costs are biased by the VWAP benchmark since the benchmark itself depends on the trades.

For this reason, some firms offer multi-days VWAP strategies to respond to customers' requests. To further

reduce the market impact of large orders, customers can specify their own volume participation by limiting the volume of their orders on days when a low volume is expected.

#### 4.1.5. Trade motivation

Most trading observed on the market, such as balancing or inflow trading, is not price sensitive and evaluation by a VWAP analysis will not be misleading. However, some trades and hence trading prices reflect objectives that cannot be captured by a VWAP analysis. To see this, we must look deeper into trading motivations to determine whether a particular price represents a good or bad execution. Let us consider two types of traders: value and growth managers. Value managers are looking for under-priced situations. They buy stock and wait until good news raises its price before they sell it. Growth managers react to good news and hope that it portends to more good news. Thus, while growth managers buy on good news value managers sell on good news. Consequently, growth managers have a clear trading disadvantage because they buy when the buying interest dominates the market. They are frequently lower ranked than value traders. If the skilled traders can understand the motivations beyond the decisions, they will try to adjust their strategy accordingly. Automatic participation algorithms cannot take into account such dimensions in trading.

# 4.2. VWAP dynamic implementation

In this section, we propose three different implementations of VWAP strategies. Each of them depends on selection of volume shape forecaster. The first implementation is called theoretical VWAP execution. It is based on one-step ahead predictions of the specific part of the volume. In the second one, the forecast of the specific part of the volume is predicted at once for the entire day (1–25 steps ahead predictions). As the forecast is done only once and never revised during the day, we call this the static execution. The last implementation involves firstly predicting the specific part of the volume for the entire day and then adjusting the forecast as the day goes on and the information content increases. We call this approach a dynamic VWAP execution as predictions are dynamically adjusted during the day.

#### 4.2.1. Theoretical VWAP execution

In this study the specific turnover is measured on a 20 mn basis. For any interval t = 1, ..., 25 during the day, and any stock i, we can easily predict  $\hat{x}_{i,t+1}$  from the observation up to  $x_{i,t}$ . However, in order to complete execution of the order at price close to the VWAP benchmark, we need to know the total trade volume for the examined day. This information allows us to calculate what part of the order should be traded at time t+1, as it is given by

$$\frac{\sum_{k=1}^{K} x_{i,k}}{\sum_{k=1}^{K} x_{i,k}}$$

Hence, such a strategy is just impossible to implement without knowing the *K* turnovers of the day or the equivalent total volume for the day. Obviously, this value is unknown before the market closes.

Therefore, theoretical VWAP execution has a purely theoretical character. Nevertheless it is worth testing this strategy and comparing its performance with other two approaches. Theoretical VWAP execution can be treated as a benchmark for the proposed intraday volume models.

#### 4.2.2. Static VWAP execution

As mentioned above, traders cannot use the theoretical execution since they do not know the daily volume, at the beginning of the day. However, they can use the dynamic model of  $x_{i,t}$  to predict at once  $\hat{x}_{i,1}, \hat{x}_{i,2}, \dots, \hat{x}_{i,25}$  and calculate the proportions to trade at each t interval. The portion to trade for each 20-min interval is given by

$$\frac{x_{i,t}}{\sum_{i=1}^{25} \hat{x}_{i,t}}$$

The simplicity of such a strategy is offset by the poor quality of the long-term estimates given by the ARMA models. Briefly, the specific volume prediction will become zero and the dynamic part of the model will have no effect on VWAP implementation. In such a scheme, we add just one step to the classical approach, where we do a rolling cross-sectional decomposition before taking a historical average. Static VWAP execution will be certainly worse than the classical approach, since the specific volume plays almost no part in the static approach in forecasting intraday volume. The average common volume contains less information on volume than the historical average of itself.

# 4.2.3. Dynamic VWAP execution

Nevertheless, the proposed decomposition together with elements of theoretical and static VWAP execution can help to improve the quality of execution. It can be achieved by taking advantage of the dynamic part of the model. The idea is to incorporate all available information about intraday volume after each step.

The prediction  $\hat{x}_{i,t+1}$ ,  $t=1,\ldots,25$  is still the one-step ahead prediction of the dynamic model as with theoretical execution. We also use the same model to get all the  $\hat{x}_{i,t+l}$ ,  $l \ge 1$  until the end of the day. The proportion  $\frac{\hat{x}_{i,t+1}}{\sum_{l=t+1}^{25} \hat{x}_{i,l}}$  is only applied on the remaining volume to trade after interval t.

As a consequence, at the very beginning of the day, we trade without information what is equivalent to application of static VWAP execution. Then, at each new time interval, we improve our prediction of future volume to trade by including the information on intraday volume already traded. Finally, the last part of volume to trade is determined in a similar manner to the case of theoretical VWAP execution.

#### 4.3. Empirical results

In this section, the question about the usefulness of the above-discussed models for the prediction of volume weight average price (VWAP) is addressed. Obviously, the answer has an important meaning for brokers, who are supposed to execute VWAP orders, and whose trades are evaluated according to benchmarks based on VWAP.

This empirical study focuses on VWAP orders with a one-day horizon. The examination is organized as follows: the proxy of volume weighted price is computed based on 25 time points during a trading day. The first point corresponds to the time 9:20 am and the last to the time 5:20 pm. The time interval between two succeeding time points is 20 min. The equity price for each of the 25 points was computed as an arithmetic average of the price of the transaction that took place in the previous 20 min. The prediction of volume is carried out using three models. Our two models are based on principal component decomposition with an ARMA or a SETAR model for specific part of volume. The third model is the classical approach to describe daily pattern of intraday volume.

We examine VWAP prediction errors in three different ways. Firstly, we make in-sample stock-by-stock and portfolio VWAP predictions for a period between 2 September and 16 December 2003 (75 trading days) and we subtract the true VWAP to get the in-sample prediction errors for each day. Secondly, we examine the out-sample case. Each time, we make a one-day out-sample prediction. For example, estimating from 2 September 2003 to 7 October 2003 (20 trading days), we get the first VWAP prediction for the following day, namely 8 October 2003. Again, the true VWAP is subtracted from the predicted one to show the first out-sample error. Then, we move our estimation window by one day, thus estimating from 3 September 2003 to 8 October 2003 and predicting for 9 October 2003 and so forth. As a result, for out-sample predictions, we obtain VWAP predictions errors for 50 days for all stocks included in the CAC40.

# 4.3.1. Single stock in-sample results

In the case of all stocks, the decomposition models outperform the classical approach. If the PCA-ARMA model is already doing a very good job, the PCA-SETAR model allows for an additional reduction of more than 1 bp on average for 29 stocks. For 8 of the stocks, the ARMA model is better but the improvement is lower than 1 bp and hence negligible. For the last 2 stocks, the ARMA model out-performs the SETAR model by almost 1 bp: LAFARGE and TF1.

From a broker's perspective the 95%-quantile contains important information about the risk of applying a particular model. The 95% -quantile has a much smaller value for

<sup>8</sup> These stock-by-stock results are not presented here but are available upon request.

the decomposition models than for the classical approach. Furthermore, the SETAR model seems to be better than the ARMA to describe the specific part of the intraday volume. This is due to the SETAR's ability to discriminate between turbulent and calm periods in the market. The 95%-quantiles for the classical approach and the model with an ARMA model for specific part range from 19 bp to 78 bp, and from 11 bp to 49 bp, respectively, with an average of 37.02 and 21.73 bp, respectively. In the SETAR case, the 95%-quantiles for all companies range from 8 bp to 39 bp, with an average of 20.3 bp. Here again, the risk associated with the use of the decomposition models is significantly lower.

As a result of in-sample performance comparisons, we show that decomposition models can be successfully used to predict the volume weight of average price (VWAP). Furthermore, a broker who exploits our approach to forecast VWAP, compared to the classical one, is lowering his execution risk.

# 4.3.2. Single stock out-sample results

In this section the above-presented in-sample results are confirmed by out-of-sample ones. This analysis is carried out by applying a 20-days moving window. Thus, the decomposition is performed using the 20 trading days preceding the day where the execution of the VWAP order takes place. The average common part of intraday volume is computed and known during the evening of the day preceding VWAP trades. In turn, the specific part is forecasted with a 20-min delay, on the considered day.

The out-of-sample performance of models under consideration for the period from 2 September to 16 December 2003 is summarized in Tables 3–5.

Before starting to analyse the results, two comments must be made. Firstly, it is fundamental here, unlike the in-sample part, to present the results of the models based on the volume decomposition for static, dynamic and theoretical VWAP execution algorithms (see Section [4.2](#page-7-0) for a description). While this distinction is useless within the in-sample study, you cannot escape it for the out-sample analysis. In fact, all approaches need a prediction of the intra-daily and daily volumes to implement the strategies except the theoretical one, which takes the most recent one. As a consequence, the theoretical approach cannot be implemented by a trader but the results are still interesting, as they give an idea of the upper improvement limit of our approach. As expected, the static method gives very poor results. For succinctness of the exposition, we only comment on the SETAR specification results that out-perform the ARMA ones.

We begin by analysing the results of the theoretical approach; comparing columns 3 to 2 in [Tables 6 and 7](#page-11-0). In the case of all 39 stocks of our sample, the decomposition model outperforms the classical approach. For all companies, use of the classical approach results in a higher risk of execution of VWAP orders. The gains in basis points are greater than 1 bp for 30 of the 39 stocks of the sample (77% of stocks). CAP GEMINI and THOMSON are the stocks with the most important gain with a MAPE falling from 23.23 bp to 14.48 bp (8.75 bp) and from 14.8 bp to 7.84 bp (6.76 bp), respectively. Conversely, for 9 stocks the gain is below 1 bp and can be considered as insignificant. If these results seem promising, remember that these gains are theoretical since they correspond to a non-realistic VWAP execution in practical terms.

Analysis of the dynamic VWAP execution is a version of the theoretical VWAP execution that can be implemented

Table 3 Summary of out-sample estimated costs of execution of VWAP order for the period from 2 September to 16 December 2003 (classical approach)

| Company            | MAPE               |        |        | Company            | MAPE   |        |        |
|--------------------|--------------------|--------|--------|--------------------|--------|--------|--------|
|                    | Mean<br>Std<br>Q95 |        |        |                    | Mean   | Std    | Q95    |
| ACCOR              | 0.1047             | 0.1209 | 0.3640 | MICHELIN           | 0.1541 | 0.2062 | 0.5566 |
| AGF-ASS.GEN.FRANCE | 0.1316             | 0.1434 | 0.4207 | PERNOD-RICARD      | 0.0775 | 0.0747 | 0.2456 |
| AIR LIQUIDE        | 0.0801             | 0.0786 | 0.2678 | PEUGEOT            | 0.0762 | 0.0916 | 0.2929 |
| ALCATEL            | 0.1336             | 0.1212 | 0.4367 | PINPRINT.REDOUTE   | 0.1389 | 0.1280 | 0.4169 |
| ARCELOR            | 0.1171             | 0.1334 | 0.3517 | RENAULT            | 0.1406 | 0.1255 | 0.3647 |
| AXA                | 0.0930             | 0.1345 | 0.3863 | SAINT GOBAIN       | 0.0979 | 0.0858 | 0.2530 |
| BNP PARIBAS        | 0.0782             | 0.0650 | 0.2086 | SANOFI-AVENTIS     | 0.0999 | 0.1084 | 0.3797 |
| BOUYGUES           | 0.1715             | 0.1007 | 0.3081 | SCHNEIDER ELECTRIC | 0.0865 | 0.1316 | 0.2251 |
| CAP GEMINI         | 0.2323             | 0.2953 | 1.1384 | SOCIETE GENERALE   | 0.0699 | 0.0687 | 0.2081 |
| CARREFOUR          | 0.0628             | 0.0598 | 0.1920 | SODEXHO ALLIANCE   | 0.1233 | 0.1340 | 0.4149 |
| CASINO GUICHARD    | 0.1465             | 0.2198 | 0.4399 | STMICROELECTRONICS | 0.0906 | 0.0905 | 0.2589 |
| CREDIT AGRICOLE    | 0.1389             | 0.1972 | 0.5424 | SUEZ               | 0.0968 | 0.0988 | 0.2836 |
| DANONE             | 0.0548             | 0.0490 | 0.1567 | TF1                | 0.1103 | 0.1006 | 0.2909 |
| DEXIA              | 0.1099             | 0.2243 | 0.5361 | THALES             | 0.0959 | 0.1399 | 0.4515 |
| EADS               | 0.1947             | 0.2397 | 0.5939 | THOMSON (EX:TMM)   | 0.1460 | 0.1588 | 0.4243 |
| FRANCE TELECOM     | 0.1398             | 0.2118 | 0.5025 | TOTAL              | 0.0528 | 0.0532 | 0.1632 |
| L'OREAL            | 0.0866             | 0.0922 | 0.2765 | VEOLIA ENVIRON.    | 0.1300 | 0.1624 | 0.4083 |
| LAFARGE            | 0.1076             | 0.1371 | 0.4255 | VINCI (EX.SGE)     | 0.0774 | 0.1088 | 0.2503 |
| LAGARDERE S.C.A.   | 0.1003             | 0.0837 | 0.2752 | VIVENDI UNIVERSAL  | 0.1095 | 0.1013 | 0.2883 |
| LVMH               | 0.1131             | 0.1155 | 0.3259 |                    |        |        |        |

Table 4 Summary of out-sample estimated costs of execution of VWAP order for the period from 2 September to 16 December 2003 (theoretical PCA-SETAR model)

| Company            | MAPE        |        |        | Company            | MAPE   |        |        |
|--------------------|-------------|--------|--------|--------------------|--------|--------|--------|
|                    | Mean<br>Std |        | Q95    |                    | Mean   | Std    | Q95    |
| ACCOR              | 0.0906      | 0.1379 | 0.5364 | MICHELIN           | 0.1380 | 0.1744 | 0.5735 |
| AGF-ASS.GEN.FRANCE | 0.1023      | 0.1336 | 0.2869 | PERNOD-RICARD      | 0.0532 | 0.0558 | 0.1824 |
| AIR LIQUIDE        | 0.0726      | 0.0662 | 0.1764 | PEUGEOT            | 0.0590 | 0.0655 | 0.2257 |
| ALCATEL            | 0.0845      | 0.0904 | 0.2785 | PINPRINT.REDOUTE   | 0.0778 | 0.0775 | 0.2235 |
| ARCELOR            | 0.0665      | 0.0621 | 0.1796 | RENAULT            | 0.1076 | 0.0937 | 0.2666 |
| AXA                | 0.0720      | 0.1060 | 0.2207 | SAINT GOBAIN       | 0.0895 | 0.0642 | 0.2260 |
| BNP PARIBAS        | 0.0710      | 0.0588 | 0.1843 | SANOFI-AVENTIS     | 0.0707 | 0.0813 | 0.2167 |
| BOUYGUES           | 0.1623      | 0.0831 | 0.3104 | SCHNEIDER ELECTRIC | 0.0788 | 0.1284 | 0.2194 |
| CAP GEMINI         | 0.1448      | 0.1955 | 0.5736 | SOCIETE GENERALE   | 0.0653 | 0.0684 | 0.1991 |
| CARREFOUR          | 0.0537      | 0.0487 | 0.1621 | SODEXHO ALLIANCE   | 0.0806 | 0.0857 | 0.3230 |
| CASINO GUICHARD    | 0.1054      | 0.1873 | 0.2054 | STMICROELECTRONICS | 0.0802 | 0.0993 | 0.2468 |
| CREDIT AGRICOLE    | 0.0902      | 0.1329 | 0.1860 | SUEZ               | 0.0725 | 0.0663 | 0.2060 |
| DANONE             | 0.0459      | 0.0409 | 0.1228 | TF1                | 0.0899 | 0.0804 | 0.2605 |
| DEXIA              | 0.0848      | 0.1849 | 0.2075 | THALES             | 0.0782 | 0.0867 | 0.3332 |
| EADS               | 0.1434      | 0.1973 | 0.3531 | THOMSON (EX:TMM)   | 0.0784 | 0.0621 | 0.2189 |
| FRANCE TELECOM     | 0.1006      | 0.1902 | 0.2805 | TOTAL              | 0.0496 | 0.0538 | 0.1789 |
| L'OREAL            | 0.0698      | 0.0844 | 0.2083 | VEOLIA ENVIRON.    | 0.0899 | 0.0921 | 0.3095 |
| LAFARGE            | 0.0964      | 0.1311 | 0.4730 | VINCI (EX.SGE)     | 0.0559 | 0.0706 | 0.1473 |
| LAGARDERE S.C.A.   | 0.0816      | 0.0708 | 0.2544 | VIVENDI UNIVERSAL  | 0.0746 | 0.0670 | 0.2228 |
| LVMH               | 0.0913      | 0.1225 | 0.2859 |                    |        |        |        |

Table 5 Summary of out-sample estimated costs of execution of VWAP order for the period from 2 September to 16 December 2003 (dynamic PCA-SETAR model)

| Company            | MAPE               |        |        | Company            | MAPE   |        |        |  |
|--------------------|--------------------|--------|--------|--------------------|--------|--------|--------|--|
|                    | Mean<br>Std<br>Q95 |        |        | Mean               | Std    | Q95    |        |  |
| ACCOR              | 0.1121             | 0.1244 | 0.3671 | MICHELIN           | 0.1513 | 0.1653 | 0.5016 |  |
| AGF-ASS.GEN.FRANCE | 0.1209             | 0.1413 | 0.3503 | PERNOD-RICARD      | 0.0745 | 0.0706 | 0.2182 |  |
| AIR LIQUIDE        | 0.0818             | 0.0757 | 0.2707 | PEUGEOT            | 0.0801 | 0.0960 | 0.3046 |  |
| ALCATEL            | 0.1079             | 0.0955 | 0.3420 | PINPRINT.REDOUTE   | 0.0998 | 0.1119 | 0.3359 |  |
| ARCELOR            | 0.1062             | 0.1146 | 0.3214 | RENAULT            | 0.1287 | 0.1138 | 0.3845 |  |
| AXA                | 0.0889             | 0.1234 | 0.4045 | SAINT GOBAIN       | 0.0952 | 0.0775 | 0.2713 |  |
| BNP PARIBAS        | 0.0742             | 0.0590 | 0.2068 | SANOFI-AVENTIS     | 0.0897 | 0.0944 | 0.2861 |  |
| BOUYGUES           | 0.1773             | 0.0978 | 0.3608 | SCHNEIDER ELECTRIC | 0.1027 | 0.1417 | 0.3239 |  |
| CAP GEMINI         | 0.1491             | 0.1322 | 0.3913 | SOCIETE GENERALE   | 0.0617 | 0.0600 | 0.1601 |  |
| CARREFOUR          | 0.0638             | 0.0562 | 0.2154 | SODEXHO ALLIANCE   | 0.1182 | 0.1280 | 0.3861 |  |
| CASINO GUICHARD    | 0.1129             | 0.1076 | 0.3595 | STMICROELECTRONICS | 0.0768 | 0.0867 | 0.2882 |  |
| CREDIT AGRICOLE    | 0.1102             | 0.1375 | 0.4637 | SUEZ               | 0.0908 | 0.0970 | 0.3022 |  |
| DANONE             | 0.0531             | 0.0441 | 0.1611 | TF1                | 0.1118 | 0.1040 | 0.3402 |  |
| DEXIA              | 0.0779             | 0.1018 | 0.1759 | THALES             | 0.1027 | 0.1270 | 0.3967 |  |
| EADS               | 0.1404             | 0.1359 | 0.4196 | THOMSON (EX:TMM)   | 0.1398 | 0.1780 | 0.4116 |  |
| FRANCE TELECOM     | 0.1080             | 0.1257 | 0.3492 | TOTAL              | 0.0508 | 0.0515 | 0.1594 |  |
| L'OREAL            | 0.0832             | 0.0888 | 0.2448 | VEOLIA ENVIRON.    | 0.1286 | 0.1511 | 0.4065 |  |
| LAFARGE            | 0.1075             | 0.1358 | 0.4483 | VINCI (EX.SGE)     | 0.0755 | 0.0969 | 0.2544 |  |
| LAGARDERE S.C.A.   | 0.1141             | 0.1003 | 0.3482 | VIVENDI UNIVERSAL  | 0.1020 | 0.1012 | 0.2591 |  |
| LVMH               | 0.0959             | 0.1001 | 0.2879 |                    |        |        |        |  |

and this allows us to check if the above theoretical situation can be reached. The results [\(Tables 6 and 7,](#page-11-0) column 4 compared to column 2) are, of course, more mitigated. Over 39 stocks, 30 stocks show a lower execution error when the classical algorithm is replaced by the dynamic VWAP one. However, over the 9 stocks presenting a deteriorated execution, 7 correspond to deterioration smaller than 1 bp, hence insignificant. Only two stocks, LAGADERE (1.3 bp) and SCHNEIDER (1.6 bp), present significant, although limited, deterioration. Conversely, for the 30 well-performing stocks, improvements can reach high levels: 8 bp for CAP GEMINI, 5 bp for EADS. All in all, 14 stocks show a decrease in the VWAP execution risk larger than 1 bp as shown in [Table 7](#page-12-0).

Comparison of the theoretical and dynamic executions provides some insight concerning the loss we bear due to the fact that we do not have access to the overall information at the very beginning of the day. Of course we cannot

<span id="page-11-0"></span>Table 6 Comparison of execution risk exposure

| Companies          | Classical<br>approach<br>(in %) | Theoretical<br>PCA-SETAR<br>(in %) | Dynamical<br>PCA-SETAR<br>(in %) | Difference                              |                                       |                                      |  |
|--------------------|---------------------------------|------------------------------------|----------------------------------|-----------------------------------------|---------------------------------------|--------------------------------------|--|
|                    |                                 |                                    |                                  | Theoretical SETAR<br>classical approach | Dynamical SETAR<br>classical approach | Theoretical SETAR<br>dynamical SETAR |  |
| ACCOR              | 0.1047                          | 0.0906                             | 0.1121                           | 0.0141                                  | 0.0074                                | 0.0215                               |  |
| AGF-ASS.GEN.FRANCE | 0.1316                          | 0.1023                             | 0.1209                           | 0.0293                                  | 0.0107                                | 0.0186                               |  |
| AIR LIQUIDE        | 0.0801                          | 0.0726                             | 0.0818                           | 0.0075                                  | 0.0017                                | 0.0092                               |  |
| ALCATEL            | 0.1336                          | 0.0845                             | 0.1079                           | 0.0491                                  | 0.0257                                | 0.0234                               |  |
| ARCELOR            | 0.1171                          | 0.0665                             | 0.1062                           | 0.0506                                  | 0.0109                                | 0.0397                               |  |
| AXA                | 0.0930                          | 0.0720                             | 0.0889                           | 0.0210                                  | 0.0041                                | 0.0169                               |  |
| BNP PARIBAS        | 0.0782                          | 0.0710                             | 0.0742                           | 0.0072                                  | 0.0040                                | 0.0032                               |  |
| BOUYGUES           | 0.1715                          | 0.1623                             | 0.1773                           | 0.0092                                  | 0.0058                                | 0.0150                               |  |
| CAP GEMINI         | 0.2323                          | 0.1448                             | 0.1491                           | 0.0875                                  | 0.0832                                | 0.0043                               |  |
| CARREFOUR          | 0.0628                          | 0.0537                             | 0.0638                           | 0.0091                                  | 0.0010                                | 0.0101                               |  |
| CASINO GUICHARD    | 0.1465                          | 0.1054                             | 0.1129                           | 0.0411                                  | 0.0336                                | 0.0075                               |  |
| CREDIT AGRICOLE    | 0.1389                          | 0.0902                             | 0.1102                           | 0.0487                                  | 0.0287                                | 0.0200                               |  |
| DANONE             | 0.0548                          | 0.0459                             | 0.0531                           | 0.0089                                  | 0.0017                                | 0.0072                               |  |
| DEXIA              | 0.1099                          | 0.0848                             | 0.0779                           | 0.0251                                  | 0.0320                                | 0.0069                               |  |
| EADS               | 0.1947                          | 0.1434                             | 0.1404                           | 0.0513                                  | 0.0543                                | 0.0030                               |  |
| FRANCE TELECOM     | 0.1398                          | 0.1006                             | 0.108                            | 0.0392                                  | 0.0318                                | 0.0074                               |  |
| L'OREAL            | 0.0866                          | 0.0698                             | 0.0832                           | 0.0168                                  | 0.0034                                | 0.0134                               |  |
| LAFARGE            | 0.1076                          | 0.0964                             | 0.1075                           | 0.0112                                  | 0.0001                                | 0.0111                               |  |
| LAGARDERE S.C.A.   | 0.1003                          | 0.0816                             | 0.1141                           | 0.0187                                  | 0.0138                                | 0.0325                               |  |
| LVMH               | 0.1131                          | 0.0913                             | 0.0959                           | 0.0218                                  | 0.0172                                | 0.0046                               |  |
| MICHELIN           | 0.1541                          | 0.138                              | 0.1513                           | 0.0161                                  | 0.0028                                | 0.0133                               |  |
| PERNOD-RICARD      | 0.0775                          | 0.0532                             | 0.0745                           | 0.0243                                  | 0.0030                                | 0.0213                               |  |

Note: The first column gives the name of the stock, the following three columns give the means of MAPE. The first column, named Difference, is the difference between the theoretical implementation PCA-SETAR model and the classical approach. A negative value means that the theoretical implementation PCA-SETAR model out-performs the classical approach since it reduces the execution risk to use the first approach instead of the latter one. The second column, is the difference between the dynamic implementation PCA-SETAR and the classical and the last one is the difference between theoretical and dynamic implementation.

erase or modify the past trades with the arrival of new information. In fact, we can only update our strategy for the rest of the trading day. The difference in MAPE between the theoretical and dynamic VWAP execution models represents the loss related to the lack of information on the daily volume.

As we can see in Tables 6 and 7, the loss can vary a lot from one stock to another. It is not significant (lower than 10%) for 13 stocks whereas it can be greater than 50% for two stocks. In fact, the error for ARCELOR increases from 6.6 bp to 10.6 bp (60%) and from 8 bp to 14 bp for THOM-SON (78%). On average, the loss is larger than 1 bp.

Finally, we can conduct one more analysis of our method by studying the link between the improvement seen by using our method and the classical approach error. The idea here is to see if our method is able or unable to correct the largest errors made when applying the classical approach. To do this, [Fig. 4](#page-12-0) shows the scatter plot of the classical approach tracking error on the x-axis against the gain or loss observed by applying our dynamical strategy on the y-axis. Here again, the gain or loss of our strategy is measured in terms of the difference between the mean of MAPE between the dynamic PCA-SETAR model and the classical approach. When this difference is positive, we suffer a loss; when it is negative, we gain by applying our strategy instead of the classical one. When the scatter plot and regression line are examined, we can see that the larger the error, the larger the gain. In fact, when the classical approach is efficient (the tracking error is below 10%), the incorporation of the intraday volume dynamic has a limited impact (or no impact). However, in cases where the classical approach tracks worse than the VWAP (CAP GEMINI and EADS), the improvement is the largest. This result confirms that our dynamic VWAP execution is a real improvement since it is more efficient on average and the more worse the execution provided by the classical approach, the larger the correction allowed by our model.

# 4.3.3. Portfolio in and out-sample results

The results for single stocks advocate the approach based on principal component decomposition. In order to summarize the results, we estimate the cost of the VWAP order execution for a basket made of all stocks included in the CAC40 index. Therefore, we compute the VWAP for the whole index based on the average of VWAP over equities. We use the same weights as used for construction of the index at the beginning of September 2004. The second and third panels in [Table 2](#page-7-0) present the summary of the model's performance in comparison with the VWAP order for the whole index.

Application of the decomposition model with the specific part described by SETAR results in an execution risk

<span id="page-12-0"></span>Table 7 Comparison of execution risk exposure(continued)

| Companies          | Classical<br>approach<br>(in %) | Theoretical<br>PCA-SETAR<br>(in %) | Dynamical<br>PCA-SETAR<br>(in %) | Difference                      |                               |                            |  |
|--------------------|---------------------------------|------------------------------------|----------------------------------|---------------------------------|-------------------------------|----------------------------|--|
|                    |                                 |                                    |                                  | Theor. SETAR<br>Class. approach | Dyn. SETAR<br>Class. approach | Theor. SETAR<br>Dyn. SETAR |  |
| PEUGEOT            | 0.0762                          | 0.059                              | 0.0801                           | 0.0172                          | 0.0039                        | 0.0211                     |  |
| PINPRINT.REDOUTE   | 0.1389                          | 0.0778                             | 0.0998                           | 0.0611                          | 0.0391                        | 0.0220                     |  |
| RENAULT            | 0.1406                          | 0.1076                             | 0.1287                           | 0.0330                          | 0.0119                        | 0.0211                     |  |
| SAINT GOBAIN       | 0.0979                          | 0.0895                             | 0.0952                           | 0.0084                          | 0.0027                        | 0.0057                     |  |
| SANOFI-AVENTIS     | 0.0999                          | 0.0707                             | 0.0897                           | 0.0292                          | 0.0102                        | 0.0190                     |  |
| SCHNEIDER ELECTRIC | 0.0865                          | 0.0788                             | 0.1027                           | 0.0077                          | 0.0162                        | 0.0239                     |  |
| SOCIETE GENERALE   | 0.0699                          | 0.0653                             | 0.0617                           | 0.0046                          | 0.0082                        | 0.0036                     |  |
| SODEXHO ALLIANCE   | 0.1233                          | 0.0806                             | 0.1182                           | 0.0427                          | 0.0051                        | 0.0376                     |  |
| STMICROELECTRONICS | 0.0906                          | 0.0802                             | 0.0768                           | 0.0104                          | 0.0138                        | 0.0034                     |  |
| SUEZ               | 0.0968                          | 0.0725                             | 0.0908                           | 0.0243                          | 0.0060                        | 0.0183                     |  |
| TF1                | 0.1103                          | 0.0899                             | 0.1118                           | 0.0204                          | 0.0015                        | 0.0219                     |  |
| THALES             | 0.0959                          | 0.0782                             | 0.1027                           | 0.0177                          | 0.0068                        | 0.0245                     |  |
| THOMSON (EX:TMM)   | 0.1460                          | 0.0784                             | 0.1398                           | 0.0676                          | 0.0062                        | 0.0614                     |  |
| TOTAL              | 0.0528                          | 0.0496                             | 0.0508                           | 0.0032                          | 0.0020                        | 0.0012                     |  |
| VEOLIA ENVIRON.    | 0.1300                          | 0.0899                             | 0.1286                           | 0.0401                          | 0.0014                        | 0.0387                     |  |
| VINCI (EX.SGE)     | 0.0774                          | 0.0559                             | 0.0755                           | 0.0215                          | 0.0019                        | 0.0196                     |  |
| VIVENDI UNIVERSAL  | 0.1095                          | 0.0746                             | 0.102                            | 0.0349                          | 0.0075                        | 0.0274                     |  |

Note: The first column gives the name of the stock, the following three columns give the means of MAPE. The first column, named Difference, is the difference between the theoretical implementation PCA-SETAR model and the classical approach. A negative value means that the theoretical implementation PCA-SETAR model out-performs the classical approach since it reduces the execution risk to use the first approach instead of the latter one. The second column, is the difference between the dynamic implementation PCA-SETAR and the classical and the last one is the difference between theoretical and dynamic implementation.

![](_page_12_Figure_5.jpeg)

Fig. 4. Dependency between the classical approach tracking error and the gain/loss for dynamical strategy.

for portfolio fall greater than 4 bp (a drop of around 40%) in the in-sample comparison. The out-sample results confirm the superiority of our method. In fact, the trading tracking error of the CAC basket using the classical approach is on average 10 bp, which equates to approximately 8 bp when using the theoretical VWAP execution, diminishing the error by 20%. Bear in mind that this is the upper improvement limit of our method. To compare with a strategy that can be implemented, we need to focus on dynamic VWAP execution results. Here again, the tracking error is lower (8 bp) and use of our method allows for a reduction of the error by 10%. Note that use of our methodology in practical terms implies that we should not use means of MAPE but, rather, calculate the errors in the basket and then calculate the MAPE of the error. However, this remark does not question our conclusions because the results would be even better in that case. In fact, individual stock errors could then compensate which is not possible using the average MAPE.

The above outcomes show that use of the decomposition of volume into market and specific parts reduces the cost of execution of VWAP orders. From the perspective of brokerage houses, which are directly engaged in the process of executing VWAP orders, an additional issue of ''beating the VWAP'' seems crucial. It is clear that the primary aim of a broker is to keep the execution price of orders as close as possible to the VWAP price and, in this manner, to generate profits from commissions paid by clients who asked for execution of VWAP orders. Nevertheless, there is another potential source of profit. An additional gain can be made when brokers manage to execute the sale of a VWAP-order at a higher price; higher, that is, than the observed end-of-day volume weighed average price. The same applies to a buy VWAP order at a lower price than the observed volume weighed average price. However, in order to beat the VWAP, our price-adjusted-volume model is not sufficient and it is essential to derive a bivariate model for volume and price.

# 5. Conclusion

In this paper, we present a new methodology for modelling the dynamics of intraday volume, which allows for a <span id="page-13-0"></span>significant reduction of the execution risk in VWAP (volume weighted average price) orders. The models are based on the decomposition of traded volume into two parts: one reflecting volume changes due to market evolutions, the second describing the stock-specific volume pattern. The first component of volume is taken as a static cross-historical average whereas the dynamics of the specific part of volume are depicted by ARMA and SETAR models.

This methodology allows us to propose an accurate statistical method of volume predictions. These predictions are then used in a benchmark-tracking price framework.

The following results were obtained through our analysis. Not only did we get round the problem of seasonal fluctuations but we use it to propose a new price benchmark. We also show that some simple time-series models give good volume predictions. Also, applications of our methodology to VWAP strategies reduce the VWAP tracking error, and thus the execution risk due to the use of such order type and so the associated cost. On average, and depending on the retained strategy, the reduction is greater than 10% and can even reach 50% for some stocks.

# Acknowledgements

We are grateful to participants at the 17th Asian FMA conference in Auckland, New Zealand, 10th Symposium on Finance, Banking, and Insurance, University of Karlsruhe, Germany and International Conference on Finance, University of Copenhagen Denmark, for valuable comments. The previous version of this paper was entitled Decomposing volume for VWAP strategies.

# References

- Andersen, T., 1996. Return volatility and trading volume: An information flow interpretation of stochastic volatility. Journal of Finance 51, 169–204. Bai, J., 2003. Inferential theory for factor models of large dimensions.
- Econometrica 71 (1), 135–171.
- Bai, J., Ng, S., 2002. Determining the number of factors in approximate factor models. Econometrica 70 (1), 191–221.
- Berkowitz, S., Logue, D., Noser, E., 1988. The total cost of transactions on the NYSE. Journal of Finance 41, 97–112.

- Biais, B., Hillion, P., Spatt, C., 1995. An empirical analysis of the limit order book and the order flow in the Paris bourse. Journal of Finance 50, 1655–1689.
- Cushing, D., Madhavan, A., 2000. Stock returns and institutional trading at the close. Journal of Financial Markets 3, 45–67.
- Darolles, S., Le Fol, G., 2003. Trading volume and arbitrage. Working paper, CREST.
- Darrat, A., Rahman, S., Zhong, M., 2003. Intraday trading volume and return volatility of the DJIA stocks: A note. Journal of Banking and Finance 27, 2035–2043.
- Easley, D., O'Hara, M., 1987. Price, trade size, and information in securities markets. Journal of Financial Economics 19, 69–90.
- Engle, R., Russel, J., 1998. Autoregressive conditional duration: A new model for irregularly spaced transaction data. Econometrica 66, 1127– 1162.
- Engle, R., 2000. The econometrics of ultra high frequency data. Econometrica 68, 1–22.
- Foster, D., Viswanathan, S., 1990. A theory of intraday variations in volume, variance and trading costs in securities market. Review of Financial Studies 3, 593–624.
- Gourie´roux, C., Jasiak, J., Le Fol, G., 1999. Intra-day market activity. Journal of Financial Markets 2, 193–226.
- Gourie´roux, C., Le Fol, G., 1998. Effet des modes de ne´gociation sur les echanges. Revue Economique 49, 795–808.
- Hobson, D., 2006. VWAP and volume profiles. Journal of Trading 1, 38– 42.
- Kaastra, I., Boyd, M., 1995. Forecasting futures trading volume using neural networks. Journal of Futures Markets 15, 953–970.
- Karpoff, J., Boyd, M., 1987. The relationship between price changes and trading volume: A survey. Journal of Financial and Quantitative Analysis 22, 109–126.
- Konishi, H., 2002. Optimal slice of a VWAP trade. Journal of Financial Markets 5 (2), 197–221.
- Le Fol, G., Mercier, L., 1998. Time deformation: Definition and comparisons. Journal of Computational Intelligence in Finance 6 (5), 19–33.
- Lo, A., Wang, J., 2000. Trading volume: Definition, data analysis, and implication of portfolio theory. Review of Financial Studies 13, 257– 300.
- Madhavan, A., 2002. VWAP Strategies Transaction Performance: The Changing Face of Trading Investment Guides Series. Institutional Investor Inc., pp. 32–38.
- Manganelli, S., 2002. Duration, volume, and volatility impact of trades. European Central Bank Working Papers Series No. 125.
- McCulloch, J., 2004. Relative volume as a doubly stochastic binomial point process. Working Papers Series.
- Tauchen, G., Pitts, M., 1983. The price variability-volume relationship on speculative markets. Econometrica 51, 485–505.