"""
BDF PCA Factor Decomposition for Intraday Volume Forecasting.

Implements the Bialkowski, Darolles, Le Fol (2008) model for decomposing
intraday trading volume into common and specific components using PCA,
with AR(1) and SETAR time-series models for the specific component.

Classes
-------
BDFModel
    Main model class implementing the full BDF pipeline.

Functions
---------
compute_u_method
    Compute the naive U-method (time-of-day average) benchmark.
compute_mape
    Compute Mean Absolute Percentage Error per Szucs 2017 Eq. (2).
compute_mse
    Compute Mean Squared Error per Szucs 2017 Eq. (1).
"""

import numpy as np
from numpy.linalg import lstsq as np_lstsq
from scipy.sparse.linalg import svds
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AR1Params:
    """
    AR(1) model parameters for the specific component.

    Attributes
    ----------
    psi_1 : float
        Autoregressive coefficient.
    psi_2 : float
        Intercept.
    sigma2 : float
        Residual variance.
    """

    psi_1: float
    psi_2: float
    sigma2: float


@dataclass
class SETARParams:
    """
    SETAR model parameters for the specific component.

    Attributes
    ----------
    phi_11 : float
        AR coefficient for regime 1 (below threshold).
    phi_12 : float
        Intercept for regime 1.
    phi_21 : float
        AR coefficient for regime 2 (above threshold).
    phi_22 : float
        Intercept for regime 2.
    tau : float
        Threshold value.
    sigma2 : float
        Residual variance.
    """

    phi_11: float
    phi_12: float
    phi_21: float
    phi_22: float
    tau: float
    sigma2: float


@dataclass
class DailyResult:
    """
    Results from a single day's BDF pipeline run.

    Attributes
    ----------
    r : int
        Number of factors selected by IC_p2.
    c_forecast : np.ndarray
        Common component forecast, shape (k, N).
    model_types : list
        Model type per stock ('AR1' or 'SETAR').
    ar_params : list
        AR(1) parameters per stock.
    setar_params : list
        SETAR parameters per stock (None if AR(1) selected).
    e_last : np.ndarray
        Last specific component value per stock, shape (N,).
    full_day_forecast : np.ndarray
        Full-day turnover forecast, shape (k, N).
    F_hat : np.ndarray
        Estimated factors, shape (P, r).
    Lambda_hat : np.ndarray
        Estimated loadings, shape (N, r).
    C_hat : np.ndarray
        Common component, shape (P, N).
    e_hat : np.ndarray
        Specific component, shape (P, N).
    """

    r: int
    c_forecast: np.ndarray
    model_types: list
    ar_params: list
    setar_params: list
    e_last: np.ndarray
    full_day_forecast: np.ndarray
    F_hat: np.ndarray
    Lambda_hat: np.ndarray
    C_hat: np.ndarray
    e_hat: np.ndarray


class BDFModel:
    """
    BDF PCA Factor Decomposition model for intraday volume forecasting.

    Implements the full pipeline: PCA factor extraction with IC_p2 factor
    count selection, common component forecasting via time-of-day averaging,
    specific component modeling with AR(1) and SETAR, and dynamic VWAP
    execution with intraday forecast updates.

    Attributes
    ----------
    k : int
        Number of intraday bins per trading day.
    L : int
        Rolling estimation window length in trading days.
    r_max : int
        Maximum number of factors for IC_p2 search.
    n_grid : int
        Number of threshold candidates for SETAR grid search.
    tau_quantile_range : tuple
        Quantile range for SETAR threshold candidates.
    min_regime_obs : int
        Minimum observations per SETAR regime.

    Methods
    -------
    extract_factors(X)
        Extract common factors from turnover matrix via PCA/SVD.
    forecast_common(C_hat)
        Forecast common component for next day.
    fit_ar1(e_series)
        Fit AR(1) model to specific component series.
    fit_setar(e_series)
        Fit SETAR model to specific component series.
    forecast_specific(model_type, ar_params, setar_params, e_prev)
        One-step-ahead forecast of specific component.
    run_daily_pipeline(X)
        Run the full daily pipeline on an estimation window.
    dynamic_vwap_execution(daily_result, actual_turnover)
        Simulate dynamic VWAP execution for a forecast day.
    """

    def __init__(
        self,
        k: int = 26,
        L: int = 20,
        r_max: int = 10,
        n_grid: int = 100,
        tau_quantile_range: tuple = (0.15, 0.85),
        min_regime_obs: int = 10,
    ) -> None:
        """
        Initialize BDF model with configuration parameters.

        Parameters
        ----------
        k:                Number of intraday bins per trading day.
        L:                Rolling estimation window in trading days.
        r_max:            Maximum candidate factor count for IC_p2.
        n_grid:           Number of SETAR threshold grid candidates.
        tau_quantile_range: Quantile range for threshold search.
        min_regime_obs:   Minimum observations per SETAR regime.
        """
        self.k = k
        self.L = L
        self.r_max = r_max
        self.n_grid = n_grid
        self.tau_quantile_range = tau_quantile_range
        self.min_regime_obs = min_regime_obs

    def extract_factors(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Extract common factors from turnover matrix X using truncated SVD.

        Selects factor count r via Bai & Ng (2002) IC_p2 information criterion.
        No demeaning is applied per Bai (2003) / BDF 2008.

        Parameters
        ----------
        X: Turnover matrix of shape (P, N), P = L * k.

        Returns
        -------
        Tuple of (F_hat, Lambda_hat, C_hat, e_hat, r) where F_hat is the
        factor matrix (P, r), Lambda_hat is loadings (N, r), C_hat is common
        component (P, N), e_hat is specific component (P, N), r is selected
        factor count.
        """
        P, N = X.shape
        r_max = min(self.r_max, min(P, N) - 1)

        total_ss = np.sum(X ** 2)

        # Truncated SVD: svds returns in ascending order, so we reverse
        U, s, Vt = svds(X.astype(np.float64), k=r_max)
        # Sort in descending order of singular values
        idx = np.argsort(s)[::-1]
        U = U[:, idx]
        s = s[idx]
        Vt = Vt[idx, :]

        # Select r via IC_p2
        best_ic = np.inf
        best_r = 1
        cumsum_s2 = np.cumsum(s ** 2)

        for r in range(1, r_max + 1):
            V_r = (total_ss - cumsum_s2[r - 1]) / (N * P)
            if V_r <= 0:
                V_r = 1e-15
            penalty = r * ((N + P) / (N * P)) * np.log(min(N, P))
            ic = np.log(V_r) + penalty
            if ic < best_ic:
                best_ic = ic
                best_r = r

        r = best_r

        # Extract factors and loadings with normalization F'F/P = I_r
        F_hat = np.sqrt(P) * U[:, :r]  # (P, r)
        Lambda_hat = (Vt[:r, :].T * s[:r]) / np.sqrt(P)  # (N, r)

        C_hat = F_hat @ Lambda_hat.T  # (P, N)
        e_hat = X - C_hat  # (P, N)

        return F_hat, Lambda_hat, C_hat, e_hat, r

    def forecast_common(self, C_hat: np.ndarray) -> np.ndarray:
        """
        Forecast common component for the next day by time-of-day averaging.

        Reshapes C_hat into (L, k, N) and averages across days for each bin.

        Parameters
        ----------
        C_hat: Common component matrix, shape (P, N) where P = L * k.

        Returns
        -------
        Common component forecast, shape (k, N).
        """
        P, N = C_hat.shape
        C_3d = C_hat.reshape(self.L, self.k, N)
        return C_3d.mean(axis=0)  # (k, N)

    def fit_ar1(self, e_series: np.ndarray) -> AR1Params:
        """
        Fit AR(1) with intercept to a specific component time series.

        Model: e_t = psi_1 * e_{t-1} + psi_2 + epsilon_t.
        Estimated by OLS.

        Parameters
        ----------
        e_series: Specific component series, shape (P,).

        Returns
        -------
        AR1Params with estimated coefficients and residual variance.
        """
        y = e_series[1:]
        x = e_series[:-1]
        design = np.column_stack([x, np.ones(len(y))])
        coeffs, _, _, _ = np_lstsq(design, y, rcond=None)
        psi_1 = coeffs[0]
        psi_2 = coeffs[1]
        residuals = y - design @ coeffs
        sigma2 = np.var(residuals)
        return AR1Params(psi_1=psi_1, psi_2=psi_2, sigma2=sigma2)

    def fit_setar(self, e_series: np.ndarray) -> Optional[SETARParams]:
        """
        Fit SETAR model to a specific component time series.

        Two-regime threshold AR model with grid search over threshold
        candidates drawn from quantiles of the lagged series.

        Parameters
        ----------
        e_series: Specific component series, shape (P,).

        Returns
        -------
        SETARParams if a valid threshold is found, None otherwise.
        """
        y = e_series[1:]
        x_lag = e_series[:-1]
        T = len(y)

        probs = np.linspace(
            self.tau_quantile_range[0], self.tau_quantile_range[1], self.n_grid
        )
        tau_candidates = np.quantile(x_lag, probs)

        best_ssr = np.inf
        best_tau = None
        best_coeffs_low = None
        best_coeffs_high = None

        for tau in tau_candidates:
            mask_low = x_lag <= tau
            mask_high = ~mask_low
            n_low = mask_low.sum()
            n_high = mask_high.sum()

            if n_low < self.min_regime_obs or n_high < self.min_regime_obs:
                continue

            design_low = np.column_stack([x_lag[mask_low], np.ones(n_low)])
            coeffs_low, _, _, _ = np_lstsq(design_low, y[mask_low], rcond=None)

            design_high = np.column_stack([x_lag[mask_high], np.ones(n_high)])
            coeffs_high, _, _, _ = np_lstsq(design_high, y[mask_high], rcond=None)

            resid_low = y[mask_low] - design_low @ coeffs_low
            resid_high = y[mask_high] - design_high @ coeffs_high
            ssr = np.sum(resid_low ** 2) + np.sum(resid_high ** 2)

            if ssr < best_ssr:
                best_ssr = ssr
                best_tau = tau
                best_coeffs_low = coeffs_low
                best_coeffs_high = coeffs_high

        if best_tau is None:
            return None

        return SETARParams(
            phi_11=best_coeffs_low[0],
            phi_12=best_coeffs_low[1],
            phi_21=best_coeffs_high[0],
            phi_22=best_coeffs_high[1],
            tau=best_tau,
            sigma2=best_ssr / T,
        )

    def forecast_specific(
        self,
        model_type: str,
        ar_params: AR1Params,
        setar_params: Optional[SETARParams],
        e_prev: float,
    ) -> float:
        """
        Compute one-step-ahead forecast of specific component.

        Parameters
        ----------
        model_type:    'AR1' or 'SETAR'.
        ar_params:     AR(1) parameters (used if model_type is 'AR1').
        setar_params:  SETAR parameters (used if model_type is 'SETAR').
        e_prev:        Previous specific component value.

        Returns
        -------
        One-step-ahead specific component forecast.
        """
        if model_type == "AR1":
            return ar_params.psi_1 * e_prev + ar_params.psi_2
        else:
            if e_prev <= setar_params.tau:
                return setar_params.phi_11 * e_prev + setar_params.phi_12
            else:
                return setar_params.phi_21 * e_prev + setar_params.phi_22

    def run_daily_pipeline(self, X: np.ndarray) -> DailyResult:
        """
        Run the full BDF daily pipeline on an estimation window.

        Performs PCA factor extraction, common component forecasting,
        per-stock AR(1)/SETAR fitting with model selection, and produces
        full-day turnover forecasts.

        Parameters
        ----------
        X: Turnover matrix for the estimation window, shape (L*k, N).

        Returns
        -------
        DailyResult containing all model outputs and forecasts.
        """
        P, N = X.shape

        # Step 1: PCA factor extraction
        F_hat, Lambda_hat, C_hat, e_hat, r = self.extract_factors(X)

        # Step 2: Forecast common component
        c_forecast = self.forecast_common(C_hat)  # (k, N)

        # Step 3: Fit specific component models per stock
        model_types = []
        ar_params_list = []
        setar_params_list = []

        for i in range(N):
            e_series_i = e_hat[:, i]

            # Always fit AR(1) as baseline
            ar_p = self.fit_ar1(e_series_i)
            ar_params_list.append(ar_p)

            # Check AR(1) stationarity
            if abs(ar_p.psi_1) >= 1.0:
                # Non-stationary: use common-only (specific = 0)
                ar_params_list[-1] = AR1Params(psi_1=0.0, psi_2=0.0, sigma2=ar_p.sigma2)
                model_types.append("AR1")
                setar_params_list.append(None)
                continue

            # Attempt SETAR fit
            setar_p = self.fit_setar(e_series_i)

            if setar_p is None:
                model_types.append("AR1")
                setar_params_list.append(None)
            elif setar_p.sigma2 > ar_p.sigma2:
                # SETAR worse than AR(1) — revert
                model_types.append("AR1")
                setar_params_list.append(None)
            else:
                model_types.append("SETAR")
                setar_params_list.append(setar_p)

        # Step 4: Last specific component value per stock
        e_last = e_hat[-1, :]  # (N,)

        # Step 5: Produce full-day forecast (multi-step from open)
        full_day_forecast = np.zeros((self.k, N))
        for i in range(N):
            e_prev = e_last[i]
            for j in range(self.k):
                e_fc = self.forecast_specific(
                    model_types[i], ar_params_list[i], setar_params_list[i], e_prev
                )
                full_day_forecast[j, i] = c_forecast[j, i] + e_fc
                e_prev = e_fc

        # Floor negative forecasts at zero
        full_day_forecast = np.maximum(full_day_forecast, 0.0)

        return DailyResult(
            r=r,
            c_forecast=c_forecast,
            model_types=model_types,
            ar_params=ar_params_list,
            setar_params=setar_params_list,
            e_last=e_last,
            full_day_forecast=full_day_forecast,
            F_hat=F_hat,
            Lambda_hat=Lambda_hat,
            C_hat=C_hat,
            e_hat=e_hat,
        )

    def dynamic_vwap_execution(
        self,
        daily_result: DailyResult,
        actual_turnover: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate dynamic VWAP execution for a forecast day.

        Updates forecasts bin-by-bin as actual turnover is observed,
        reallocating remaining order quantity proportionally.

        Parameters
        ----------
        daily_result:    Output of run_daily_pipeline for this day.
        actual_turnover: Actual observed turnover, shape (k, N).

        Returns
        -------
        Tuple of (proportions, revised_forecasts) where proportions is the
        realized allocation per bin (k, N) and revised_forecasts is the
        final forecast used for each bin (k, N).
        """
        k = self.k
        N = actual_turnover.shape[1]
        c_forecast = daily_result.c_forecast
        model_types = daily_result.model_types
        ar_params = daily_result.ar_params
        setar_params = daily_result.setar_params

        proportions = np.zeros((k, N))
        revised_forecasts = np.zeros((k, N))
        remaining_frac = np.ones(N)  # fraction of order remaining

        for j_obs in range(k):
            if j_obs == 0:
                # Use initial full-day forecast
                forecast = daily_result.full_day_forecast.copy()
            else:
                # Observe actual turnover for previous bin, update
                e_actual_prev = actual_turnover[j_obs - 1, :] - c_forecast[j_obs - 1, :]

                forecast = np.zeros((k, N))
                for i in range(N):
                    e_prev = e_actual_prev[i]
                    for j in range(j_obs, k):
                        e_fc = self.forecast_specific(
                            model_types[i], ar_params[i], setar_params[i], e_prev
                        )
                        forecast[j, i] = c_forecast[j, i] + e_fc
                        e_prev = e_fc

                # Floor negatives
                forecast = np.maximum(forecast, 0.0)

            # Compute proportions for remaining bins
            remaining_forecast = forecast[j_obs:, :]
            total_remaining = remaining_forecast.sum(axis=0)
            # Handle zero total (set uniform)
            zero_mask = total_remaining == 0
            if zero_mask.any():
                remaining_bins = k - j_obs
                remaining_forecast[:, zero_mask] = 1.0 / remaining_bins
                total_remaining[zero_mask] = 1.0

            prop_this_bin = remaining_forecast[0, :] / total_remaining
            proportions[j_obs, :] = prop_this_bin * remaining_frac
            remaining_frac -= proportions[j_obs, :]
            revised_forecasts[j_obs, :] = forecast[j_obs, :] if j_obs < k else 0.0

        return proportions, revised_forecasts

    def dynamic_one_step_ahead(
        self,
        daily_result: DailyResult,
        actual_turnover: np.ndarray,
    ) -> np.ndarray:
        """
        Compute one-step-ahead forecasts using dynamic updates.

        For each bin j, uses actual turnover from bin j-1 (or e_last for
        bin 0) to produce the one-step-ahead forecast. This is the
        evaluation mode matching Szucs 2017.

        Parameters
        ----------
        daily_result:    Output of run_daily_pipeline for this day.
        actual_turnover: Actual observed turnover, shape (k, N).

        Returns
        -------
        One-step-ahead forecasts, shape (k, N).
        """
        k = self.k
        N = actual_turnover.shape[1]
        forecasts = np.zeros((k, N))

        for i in range(N):
            for j in range(k):
                if j == 0:
                    e_prev = daily_result.e_last[i]
                else:
                    # Use actual residual from previous bin
                    e_prev = (
                        actual_turnover[j - 1, i]
                        - daily_result.c_forecast[j - 1, i]
                    )

                e_fc = self.forecast_specific(
                    daily_result.model_types[i],
                    daily_result.ar_params[i],
                    daily_result.setar_params[i],
                    e_prev,
                )
                forecasts[j, i] = daily_result.c_forecast[j, i] + e_fc

        return forecasts


def compute_u_method(X: np.ndarray, k: int) -> np.ndarray:
    """
    Compute the U-method (time-of-day average) benchmark forecast.

    For each bin j and stock i, the forecast is the average turnover in
    bin j across all days in the estimation window.

    Parameters
    ----------
    X: Turnover matrix, shape (P, N) where P = L * k.
    k: Number of bins per day.

    Returns
    -------
    U-method forecast, shape (k, N).
    """
    P, N = X.shape
    L = P // k
    X_3d = X.reshape(L, k, N)
    return X_3d.mean(axis=0)


def compute_mape(
    actual: np.ndarray, forecast: np.ndarray, min_actual: float = 1e-10
) -> float:
    """
    Compute Mean Absolute Percentage Error per Szucs 2017 Eq. (2).

    Excludes bins where actual turnover is below min_actual to avoid
    division by zero.

    Parameters
    ----------
    actual:     Actual turnover values.
    forecast:   Forecast turnover values.
    min_actual: Minimum actual value to include in computation.

    Returns
    -------
    MAPE value (not in percent, i.e. 0.40 means 40%).
    """
    mask = actual > min_actual
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs(actual[mask] - forecast[mask]) / actual[mask])


def compute_mse(actual: np.ndarray, forecast: np.ndarray) -> float:
    """
    Compute Mean Squared Error per Szucs 2017 Eq. (1).

    Parameters
    ----------
    actual:   Actual turnover values.
    forecast: Forecast turnover values.

    Returns
    -------
    MSE value.
    """
    return np.mean((actual - forecast) ** 2)
