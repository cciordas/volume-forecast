"""
Kalman filter state-space model for intraday volume forecasting.

Implements the standard and robust Kalman filter with EM calibration
from Chen, Feng, and Palomar (2016). Decomposes log-volume into daily
level, intraday seasonality, and intraday dynamic components.

Classes
-------
KalmanVolumeParams
    Container for all model parameters.
KalmanVolumeModel
    Kalman filter/smoother with EM calibration for volume forecasting.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from dataclasses import dataclass, field
from typing import Optional


@njit(cache=True)
def _kalman_filter_numba(
    y_flat: np.ndarray,
    obs_flat: np.ndarray,
    phi: np.ndarray,
    I: int,
    a_eta: float,
    a_mu: float,
    sigma_eta_sq: float,
    sigma_mu_sq: float,
    r: float,
    pi_1: np.ndarray,
    Sigma_1: np.ndarray,
    robust: bool,
    lam: float,
) -> tuple:
    """
    Numba-optimized forward Kalman filter pass.
    """
    N = len(y_flat)

    x_filt = np.zeros((N, 2))
    Sigma_filt = np.zeros((N, 2, 2))
    x_pred = np.zeros((N, 2))
    Sigma_pred = np.zeros((N, 2, 2))
    K_all = np.zeros((N, 2))
    A_all = np.zeros((N, 2, 2))
    y_hat = np.zeros(N)
    e_all = np.zeros(N)
    S_all = np.zeros(N)
    z_star = np.zeros(N)

    log_likelihood = 0.0

    for tau in range(N):
        phi_tau = phi[tau % I]
        bin_in_day = tau % I

        if tau == 0:
            x_pred[0, 0] = pi_1[0]
            x_pred[0, 1] = pi_1[1]
            Sigma_pred[0] = Sigma_1.copy()
            A_all[0, 0, 0] = 1.0
            A_all[0, 1, 1] = 1.0
        else:
            if bin_in_day == 0:
                a00 = a_eta
                q00 = sigma_eta_sq
            else:
                a00 = 1.0
                q00 = 0.0
            a11 = a_mu
            q11 = sigma_mu_sq

            A_all[tau, 0, 0] = a00
            A_all[tau, 1, 1] = a11

            # x_pred = A @ x_filt[tau-1]
            x_pred[tau, 0] = a00 * x_filt[tau - 1, 0]
            x_pred[tau, 1] = a11 * x_filt[tau - 1, 1]

            # Sigma_pred = A @ Sigma_filt @ A^T + Q (diagonal A and Q)
            sf = Sigma_filt[tau - 1]
            Sigma_pred[tau, 0, 0] = a00 * a00 * sf[0, 0] + q00
            Sigma_pred[tau, 0, 1] = a00 * a11 * sf[0, 1]
            Sigma_pred[tau, 1, 0] = a11 * a00 * sf[1, 0]
            Sigma_pred[tau, 1, 1] = a11 * a11 * sf[1, 1] + q11

        # Forecast: C = [1, 1]
        y_hat[tau] = x_pred[tau, 0] + x_pred[tau, 1] + phi_tau

        if obs_flat[tau]:
            # Innovation
            e_tau = y_flat[tau] - x_pred[tau, 0] - x_pred[tau, 1] - phi_tau
            e_all[tau] = e_tau

            # Innovation variance: C @ Sigma_pred @ C + r
            sp = Sigma_pred[tau]
            S_tau = sp[0, 0] + sp[0, 1] + sp[1, 0] + sp[1, 1] + r
            S_all[tau] = S_tau

            # Kalman gain: Sigma_pred @ C / S
            K0 = (sp[0, 0] + sp[0, 1]) / S_tau
            K1 = (sp[1, 0] + sp[1, 1]) / S_tau
            K_all[tau, 0] = K0
            K_all[tau, 1] = K1

            # Robust outlier detection
            e_clean = e_tau
            if robust:
                threshold = lam * S_tau / 2.0
                if e_tau > threshold:
                    z_star[tau] = e_tau - threshold
                elif e_tau < -threshold:
                    z_star[tau] = e_tau + threshold
                e_clean = e_tau - z_star[tau]

            # Corrected state
            x_filt[tau, 0] = x_pred[tau, 0] + K0 * e_clean
            x_filt[tau, 1] = x_pred[tau, 1] + K1 * e_clean

            # Joseph form covariance: (I - K*C) @ Sigma_pred @ (I - K*C)^T + r*K*K^T
            # IKC = [[1-K0, -K0], [-K1, 1-K1]]
            ikc00 = 1.0 - K0
            ikc01 = -K0
            ikc10 = -K1
            ikc11 = 1.0 - K1

            # temp = IKC @ Sigma_pred
            t00 = ikc00 * sp[0, 0] + ikc01 * sp[1, 0]
            t01 = ikc00 * sp[0, 1] + ikc01 * sp[1, 1]
            t10 = ikc10 * sp[0, 0] + ikc11 * sp[1, 0]
            t11 = ikc10 * sp[0, 1] + ikc11 * sp[1, 1]

            # result = temp @ IKC^T + r * K * K^T
            Sigma_filt[tau, 0, 0] = t00 * ikc00 + t01 * ikc01 + r * K0 * K0
            Sigma_filt[tau, 0, 1] = t00 * ikc10 + t01 * ikc11 + r * K0 * K1
            Sigma_filt[tau, 1, 0] = t10 * ikc00 + t11 * ikc01 + r * K1 * K0
            Sigma_filt[tau, 1, 1] = t10 * ikc10 + t11 * ikc11 + r * K1 * K1

            # Log-likelihood
            log_likelihood += -0.5 * (
                e_clean * e_clean / S_tau + np.log(S_tau) + np.log(2 * np.pi)
            )
        else:
            x_filt[tau, 0] = x_pred[tau, 0]
            x_filt[tau, 1] = x_pred[tau, 1]
            Sigma_filt[tau] = Sigma_pred[tau].copy()

    return (x_filt, Sigma_filt, x_pred, Sigma_pred, K_all, A_all,
            y_hat, e_all, S_all, z_star, log_likelihood)


@njit(cache=True)
def _kalman_smoother_numba(
    x_filt: np.ndarray,
    Sigma_filt: np.ndarray,
    x_pred: np.ndarray,
    Sigma_pred: np.ndarray,
    A_all: np.ndarray,
    K: np.ndarray,
    N: int,
) -> tuple:
    """
    Numba-optimized RTS backward smoother.
    """
    x_smooth = np.zeros((N, 2))
    Sigma_smooth = np.zeros((N, 2, 2))
    L_all = np.zeros((N, 2, 2))
    cross_cov = np.zeros((N, 2, 2))

    x_smooth[N - 1] = x_filt[N - 1].copy()
    Sigma_smooth[N - 1] = Sigma_filt[N - 1].copy()

    for tau in range(N - 2, -1, -1):
        A_next = A_all[tau + 1]
        sp_next = Sigma_pred[tau + 1]

        # Regularized inverse of sp_next (2x2)
        eps = 1e-10
        det = (sp_next[0, 0] + eps) * (sp_next[1, 1] + eps) - sp_next[0, 1] * sp_next[1, 0]
        inv00 = (sp_next[1, 1] + eps) / det
        inv01 = -sp_next[0, 1] / det
        inv10 = -sp_next[1, 0] / det
        inv11 = (sp_next[0, 0] + eps) / det

        # L = Sigma_filt[tau] @ A_next^T @ inv(Sigma_pred[tau+1])
        sf = Sigma_filt[tau]
        # temp = sf @ A_next^T  (A is diagonal)
        t00 = sf[0, 0] * A_next[0, 0]
        t01 = sf[0, 1] * A_next[1, 1]
        t10 = sf[1, 0] * A_next[0, 0]
        t11 = sf[1, 1] * A_next[1, 1]

        # L = temp @ inv
        L_all[tau, 0, 0] = t00 * inv00 + t01 * inv10
        L_all[tau, 0, 1] = t00 * inv01 + t01 * inv11
        L_all[tau, 1, 0] = t10 * inv00 + t11 * inv10
        L_all[tau, 1, 1] = t10 * inv01 + t11 * inv11

        L = L_all[tau]

        # Smoothed state
        dx0 = x_smooth[tau + 1, 0] - x_pred[tau + 1, 0]
        dx1 = x_smooth[tau + 1, 1] - x_pred[tau + 1, 1]
        x_smooth[tau, 0] = x_filt[tau, 0] + L[0, 0] * dx0 + L[0, 1] * dx1
        x_smooth[tau, 1] = x_filt[tau, 1] + L[1, 0] * dx0 + L[1, 1] * dx1

        # Smoothed covariance: sf + L @ (Sigma_smooth[tau+1] - sp_next) @ L^T
        ds = Sigma_smooth[tau + 1] - sp_next
        # temp2 = L @ ds
        u00 = L[0, 0] * ds[0, 0] + L[0, 1] * ds[1, 0]
        u01 = L[0, 0] * ds[0, 1] + L[0, 1] * ds[1, 1]
        u10 = L[1, 0] * ds[0, 0] + L[1, 1] * ds[1, 0]
        u11 = L[1, 0] * ds[0, 1] + L[1, 1] * ds[1, 1]

        Sigma_smooth[tau, 0, 0] = sf[0, 0] + u00 * L[0, 0] + u01 * L[0, 1]
        Sigma_smooth[tau, 0, 1] = sf[0, 1] + u00 * L[1, 0] + u01 * L[1, 1]
        Sigma_smooth[tau, 1, 0] = sf[1, 0] + u10 * L[0, 0] + u11 * L[0, 1]
        Sigma_smooth[tau, 1, 1] = sf[1, 1] + u10 * L[1, 0] + u11 * L[1, 1]

    # Cross-covariance
    # Init: Sigma[N,N-1|N] = (I-K_N*C) @ A_N @ Sigma_filt[N-2]
    # C = [1, 1], so IKC = [[1-K0, -K0], [-K1, 1-K1]]
    K0N = K[N - 1, 0]
    K1N = K[N - 1, 1]
    ikc00 = 1.0 - K0N
    ikc01 = -K0N
    ikc10 = -K1N
    ikc11 = 1.0 - K1N

    A_N = A_all[N - 1]
    sf_nm2 = Sigma_filt[N - 2]

    # temp = A_N @ sf_nm2 (A diagonal)
    t00 = A_N[0, 0] * sf_nm2[0, 0]
    t01 = A_N[0, 0] * sf_nm2[0, 1]
    t10 = A_N[1, 1] * sf_nm2[1, 0]
    t11 = A_N[1, 1] * sf_nm2[1, 1]

    # cross_cov[N-1] = IKC @ temp
    cross_cov[N - 1, 0, 0] = ikc00 * t00 + ikc01 * t10
    cross_cov[N - 1, 0, 1] = ikc00 * t01 + ikc01 * t11
    cross_cov[N - 1, 1, 0] = ikc10 * t00 + ikc11 * t10
    cross_cov[N - 1, 1, 1] = ikc10 * t01 + ikc11 * t11

    for tau in range(N - 2, 0, -1):
        A_next = A_all[tau + 1]
        L_prev = L_all[tau - 1]
        sf_tau = Sigma_filt[tau]
        L_tau = L_all[tau]

        # diff = cross_cov[tau+1] - A_next @ sf_tau  (A diagonal)
        d00 = cross_cov[tau + 1, 0, 0] - A_next[0, 0] * sf_tau[0, 0]
        d01 = cross_cov[tau + 1, 0, 1] - A_next[0, 0] * sf_tau[0, 1]
        d10 = cross_cov[tau + 1, 1, 0] - A_next[1, 1] * sf_tau[1, 0]
        d11 = cross_cov[tau + 1, 1, 1] - A_next[1, 1] * sf_tau[1, 1]

        # temp = L_tau @ diff
        m00 = L_tau[0, 0] * d00 + L_tau[0, 1] * d10
        m01 = L_tau[0, 0] * d01 + L_tau[0, 1] * d11
        m10 = L_tau[1, 0] * d00 + L_tau[1, 1] * d10
        m11 = L_tau[1, 0] * d01 + L_tau[1, 1] * d11

        # cross_cov[tau] = sf_tau @ L_prev^T + temp @ L_prev^T
        for ii in range(2):
            for jj in range(2):
                val = 0.0
                for kk in range(2):
                    val += sf_tau[ii, kk] * L_prev[jj, kk]
                    if ii == 0:
                        if kk == 0:
                            val += m00 * L_prev[jj, 0] if ii == 0 else m10 * L_prev[jj, 0]
                cross_cov[tau, ii, jj] = val

        # Simpler explicit version
        cross_cov[tau, 0, 0] = (sf_tau[0, 0] * L_prev[0, 0] + sf_tau[0, 1] * L_prev[0, 1]
                                + m00 * L_prev[0, 0] + m01 * L_prev[0, 1])
        cross_cov[tau, 0, 1] = (sf_tau[0, 0] * L_prev[1, 0] + sf_tau[0, 1] * L_prev[1, 1]
                                + m00 * L_prev[1, 0] + m01 * L_prev[1, 1])
        cross_cov[tau, 1, 0] = (sf_tau[1, 0] * L_prev[0, 0] + sf_tau[1, 1] * L_prev[0, 1]
                                + m10 * L_prev[0, 0] + m11 * L_prev[0, 1])
        cross_cov[tau, 1, 1] = (sf_tau[1, 0] * L_prev[1, 0] + sf_tau[1, 1] * L_prev[1, 1]
                                + m10 * L_prev[1, 0] + m11 * L_prev[1, 1])

    return x_smooth, Sigma_smooth, L_all, cross_cov


@dataclass
class KalmanVolumeParams:
    """
    Container for all Kalman filter volume model parameters.

    Attributes
    ----------
    a_eta : float
        AR(1) coefficient for daily component.
    a_mu : float
        AR(1) coefficient for intraday dynamic component.
    sigma_eta_sq : float
        Process noise variance for daily component.
    sigma_mu_sq : float
        Process noise variance for intraday dynamic component.
    r : float
        Observation noise variance.
    phi : ndarray
        Intraday seasonality vector, shape (I,).
    pi_1 : ndarray
        Initial state mean, shape (2,).
    Sigma_1 : ndarray
        Initial state covariance, shape (2, 2).
    """

    a_eta: float = 0.95
    a_mu: float = 0.5
    sigma_eta_sq: float = 0.1
    sigma_mu_sq: float = 0.1
    r: float = 0.5
    phi: np.ndarray = field(default_factory=lambda: np.zeros(26))
    pi_1: np.ndarray = field(default_factory=lambda: np.zeros(2))
    Sigma_1: np.ndarray = field(default_factory=lambda: np.eye(2))

    def copy(self) -> KalmanVolumeParams:
        """
        Return a deep copy of the parameters.

        Returns
        -------
        Deep copy of this parameter set.
        """
        return KalmanVolumeParams(
            a_eta=self.a_eta,
            a_mu=self.a_mu,
            sigma_eta_sq=self.sigma_eta_sq,
            sigma_mu_sq=self.sigma_mu_sq,
            r=self.r,
            phi=self.phi.copy(),
            pi_1=self.pi_1.copy(),
            Sigma_1=self.Sigma_1.copy(),
        )


class KalmanVolumeModel:
    """
    Kalman filter/smoother with EM calibration for intraday volume.

    Implements the state-space model from Chen, Feng, Palomar (2016)
    with optional robust Lasso-penalized outlier detection.

    Attributes
    ----------
    I : int
        Number of intraday bins per day.
    params : KalmanVolumeParams
        Current model parameters.
    robust : bool
        Whether to use the robust filter extension.
    lam : float
        Lasso regularization parameter for robust filter.

    Methods
    -------
    initialize_params(y, is_observed)
        Set initial parameter guesses from training data.
    fit(y, is_observed, max_iter, epsilon, verbose)
        Run EM algorithm to estimate parameters.
    kalman_filter(y, is_observed, params)
        Run the forward Kalman filter pass.
    kalman_smoother(filtered, params)
        Run the RTS backward smoother.
    predict_static(x_filtered, Sigma_filtered, params)
        Produce static forecasts for next day.
    predict_dynamic(y, is_observed, params)
        Produce dynamic one-step-ahead forecasts.
    """

    def __init__(
        self,
        bins_per_day: int = 26,
        robust: bool = False,
        lam: float = 1.0,
    ) -> None:
        """
        Initialize the Kalman volume model.

        Parameters
        ----------
        bins_per_day: Number of intraday bins per trading day.
        robust:      Whether to use the robust filter extension.
        lam:         Lasso regularization parameter (robust mode only).
        """
        self.I = bins_per_day
        self.robust = robust
        self.lam = lam
        self.params = KalmanVolumeParams()

    def initialize_params(
        self,
        y: np.ndarray,
        is_observed: np.ndarray,
    ) -> KalmanVolumeParams:
        """
        Set initial parameter guesses from training data statistics.

        Parameters
        ----------
        y:           Log-volume observations, shape (T, I).
        is_observed: Boolean mask, shape (T, I).

        Returns
        -------
        Initialized parameter set.
        """
        T, I = y.shape
        assert I == self.I

        y_flat = y[is_observed]
        y_bar = np.mean(y_flat)
        y_var = np.var(y_flat)

        params = KalmanVolumeParams(
            a_eta=0.95,
            a_mu=0.5,
            sigma_eta_sq=0.1 * y_var,
            sigma_mu_sq=0.1 * y_var,
            r=0.5 * y_var,
            phi=np.zeros(I),
            pi_1=np.array([y_bar, 0.0]),
            Sigma_1=np.diag([y_var, y_var]),
        )

        # Initialize phi as bin means minus grand mean
        for i in range(I):
            obs_mask = is_observed[:, i]
            if np.any(obs_mask):
                params.phi[i] = np.mean(y[obs_mask, i]) - y_bar

        self.params = params
        return params

    def _build_transition(
        self,
        tau: int,
        params: KalmanVolumeParams,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build the transition matrix A and process noise Q for time step tau.

        Parameters
        ----------
        tau:    Unified time index (0-based).
        params: Model parameters.

        Returns
        -------
        Tuple of (A, Q) matrices, each shape (2, 2).
        """
        bin_in_day = tau % self.I
        if bin_in_day == 0:
            # Day boundary (first bin of day)
            A = np.array([
                [params.a_eta, 0.0],
                [0.0, params.a_mu],
            ])
            Q = np.array([
                [params.sigma_eta_sq, 0.0],
                [0.0, params.sigma_mu_sq],
            ])
        else:
            # Within-day (eta constant, mu evolves)
            A = np.array([
                [1.0, 0.0],
                [0.0, params.a_mu],
            ])
            Q = np.array([
                [0.0, 0.0],
                [0.0, params.sigma_mu_sq],
            ])
        return A, Q

    def kalman_filter(
        self,
        y: np.ndarray,
        is_observed: np.ndarray,
        params: Optional[KalmanVolumeParams] = None,
    ) -> dict:
        """
        Run the forward Kalman filter pass.

        Parameters
        ----------
        y:           Log-volume observations, shape (T, I).
        is_observed: Boolean mask, shape (T, I).
        params:      Model parameters. Uses self.params if None.

        Returns
        -------
        Dictionary with filtered states, covariances, predictions, gains,
        innovations, innovation variances, z_star values, and log-likelihood.
        """
        if params is None:
            params = self.params

        y_flat = y.ravel().astype(np.float64)
        obs_flat = is_observed.ravel().astype(np.bool_)

        result = _kalman_filter_numba(
            y_flat, obs_flat, params.phi,
            self.I, params.a_eta, params.a_mu,
            params.sigma_eta_sq, params.sigma_mu_sq, params.r,
            params.pi_1, params.Sigma_1,
            self.robust, self.lam,
        )

        (x_filt, Sigma_filt, x_pred, Sigma_pred, K_all, A_all,
         y_hat, e_all, S_all, z_star, log_likelihood) = result

        return {
            "x_filt": x_filt,
            "Sigma_filt": Sigma_filt,
            "x_pred": x_pred,
            "Sigma_pred": Sigma_pred,
            "K": K_all,
            "A": A_all,
            "y_hat": y_hat,
            "e": e_all,
            "S": S_all,
            "z_star": z_star,
            "log_likelihood": log_likelihood,
        }

    def kalman_smoother(
        self,
        filtered: dict,
        params: Optional[KalmanVolumeParams] = None,
    ) -> dict:
        """
        Run the RTS backward smoother after a forward filter pass.

        Parameters
        ----------
        filtered: Output dictionary from kalman_filter.
        params:   Model parameters. Uses self.params if None.

        Returns
        -------
        Dictionary with smoothed states, covariances, and cross-covariances.
        """
        if params is None:
            params = self.params

        N = filtered["x_filt"].shape[0]

        x_smooth, Sigma_smooth, L_all, cross_cov = _kalman_smoother_numba(
            filtered["x_filt"],
            filtered["Sigma_filt"],
            filtered["x_pred"],
            filtered["Sigma_pred"],
            filtered["A"],
            filtered["K"],
            N,
        )

        return {
            "x_smooth": x_smooth,
            "Sigma_smooth": Sigma_smooth,
            "L": L_all,
            "cross_cov": cross_cov,
        }

    def _em_m_step(
        self,
        y: np.ndarray,
        is_observed: np.ndarray,
        smoothed: dict,
        filtered: dict,
        params: KalmanVolumeParams,
    ) -> KalmanVolumeParams:
        """
        Perform the EM M-step to update parameters (vectorized).

        Parameters
        ----------
        y:           Log-volume observations, shape (T, I).
        is_observed: Boolean mask, shape (T, I).
        smoothed:    Output from kalman_smoother.
        filtered:    Output from kalman_filter.
        params:      Current parameters (for robust z_star).

        Returns
        -------
        Updated parameter set.
        """
        T, I = y.shape
        N = T * I

        x_s = smoothed["x_smooth"]          # (N, 2)
        Sigma_s = smoothed["Sigma_smooth"]   # (N, 2, 2)
        cross_cov = smoothed["cross_cov"]    # (N, 2, 2)
        z_star = filtered["z_star"]          # (N,)

        y_flat = y.ravel()
        obs_flat = is_observed.ravel()

        new_params = params.copy()

        # Initial state
        new_params.pi_1 = x_s[0].copy()
        new_params.Sigma_1 = Sigma_s[0].copy()

        # Day boundary indices (0-based): first bin of each day except day 0
        db = np.arange(I, N, I)  # tau = I, 2I, 3I, ...

        # Precompute per-element quantities
        cc_11 = cross_cov[1:, 0, 0]  # cross_cov[tau][0,0] for tau=1..N-1
        cc_22 = cross_cov[1:, 1, 1]
        xs_0 = x_s[:, 0]  # eta smoothed
        xs_1 = x_s[:, 1]  # mu smoothed
        sig_00 = Sigma_s[:, 0, 0]
        sig_11 = Sigma_s[:, 1, 1]

        # P_{tau,tau-1}^{(k,k)} = cross_cov[tau][k,k] + x_s[tau,k]*x_s[tau-1,k]
        # P_{tau}^{(k,k)} = Sigma_s[tau][k,k] + x_s[tau,k]^2

        # a_eta: over day boundaries
        db_idx = db - 1  # index into 1-based offset arrays
        P_cross_11_db = cross_cov[db, 0, 0] + xs_0[db] * xs_0[db - 1]
        P_prev_11_db = sig_00[db - 1] + xs_0[db - 1] ** 2
        new_params.a_eta = np.clip(
            np.sum(P_cross_11_db) / np.sum(P_prev_11_db), 0.0, 0.9999
        )

        # a_mu: over ALL transitions tau=1..N-1
        P_cross_22_all = cc_22 + xs_1[1:] * xs_1[:-1]
        P_prev_22_all = sig_11[:-1] + xs_1[:-1] ** 2
        new_params.a_mu = np.clip(
            np.sum(P_cross_22_all) / np.sum(P_prev_22_all), 0.0, 0.9999
        )

        # sigma_eta_sq: over day boundaries
        P_tau_11_db = sig_00[db] + xs_0[db] ** 2
        sum_sigma_eta = np.sum(
            P_tau_11_db
            + new_params.a_eta ** 2 * P_prev_11_db
            - 2 * new_params.a_eta * P_cross_11_db
        )
        new_params.sigma_eta_sq = max(sum_sigma_eta / len(db), 1e-10)

        # sigma_mu_sq: over ALL transitions
        P_tau_22_all = sig_11[1:] + xs_1[1:] ** 2
        sum_sigma_mu = np.sum(
            P_tau_22_all
            + new_params.a_mu ** 2 * P_prev_22_all
            - 2 * new_params.a_mu * P_cross_22_all
        )
        new_params.sigma_mu_sq = max(sum_sigma_mu / (N - 1), 1e-10)

        # phi: update BEFORE r (vectorized)
        # C @ x_s[tau] = x_s[tau, 0] + x_s[tau, 1]
        Cx_s = xs_0 + xs_1  # (N,)
        residuals = y_flat - Cx_s
        if self.robust:
            residuals = residuals - z_star

        residuals_mat = residuals.reshape(T, I)
        obs_mat = is_observed
        new_phi = np.zeros(I)
        for i in range(I):
            mask = obs_mat[:, i]
            if np.any(mask):
                new_phi[i] = np.mean(residuals_mat[mask, i])
        new_params.phi = new_phi

        # r: observation noise variance (vectorized)
        # r = (1/N_obs) * sum_obs [y^2 + C*P*C^T - 2*y*C*x_s + phi^2
        #      - 2*y*phi + 2*phi*C*x_s + robust terms]
        # C*P*C^T = sig_00 + sig_11 + 2*Sigma_s[:,0,1] + Cx_s^2
        #   (since P = Sigma_s + x_s*x_s^T, C*P*C^T = C*Sigma_s*C^T + (C*x_s)^2)
        CPCt = (
            sig_00 + sig_11 + 2 * Sigma_s[:, 0, 1] + Cx_s ** 2
        )  # (N,)

        phi_tau = new_params.phi[np.arange(N) % I]  # (N,)

        r_terms = (
            y_flat ** 2
            + CPCt
            - 2 * y_flat * Cx_s
            + phi_tau ** 2
            - 2 * y_flat * phi_tau
            + 2 * phi_tau * Cx_s
        )

        if self.robust:
            r_terms += (
                z_star ** 2
                - 2 * z_star * y_flat
                + 2 * z_star * Cx_s
                + 2 * z_star * phi_tau
            )

        N_obs = int(np.sum(obs_flat))
        new_params.r = max(np.sum(r_terms[obs_flat]) / N_obs, 1e-10)

        return new_params

    def fit(
        self,
        y: np.ndarray,
        is_observed: np.ndarray,
        max_iter: int = 100,
        epsilon: float = 1e-6,
        verbose: bool = False,
        warm_start: bool = False,
    ) -> dict:
        """
        Run the EM algorithm to estimate all model parameters.

        Parameters
        ----------
        y:           Log-volume observations, shape (T, I).
        is_observed: Boolean mask, shape (T, I).
        max_iter:    Maximum number of EM iterations.
        epsilon:     Relative convergence threshold for log-likelihood.
        verbose:     Whether to print iteration progress.
        warm_start:  If True, use current self.params as starting point
                     instead of re-initializing from data statistics.

        Returns
        -------
        Dictionary with converged parameters, log-likelihood history,
        and number of iterations.
        """
        if not warm_start:
            self.initialize_params(y, is_observed)
        params = self.params

        ll_history = []

        for j in range(max_iter):
            # E-step
            filtered = self.kalman_filter(y, is_observed, params)
            smoothed = self.kalman_smoother(filtered, params)
            ll = filtered["log_likelihood"]
            ll_history.append(ll)

            if verbose:
                print(f"EM iter {j:3d}: log-likelihood = {ll:.4f}")

            # Convergence check
            if j >= 1:
                # Monotonicity assertion
                if ll < ll_history[j - 1] - 1e-6:
                    print(
                        f"WARNING: EM log-likelihood decreased at iter {j}: "
                        f"{ll:.6f} < {ll_history[j-1]:.6f}"
                    )

                rel_change = abs(ll - ll_history[j - 1]) / max(
                    abs(ll_history[j - 1]), 1e-10
                )
                if rel_change < epsilon:
                    if verbose:
                        print(f"EM converged at iteration {j} (rel_change={rel_change:.2e})")
                    break

            # M-step
            params = self._em_m_step(y, is_observed, smoothed, filtered, params)

        self.params = params
        return {
            "params": params,
            "log_likelihood_history": ll_history,
            "iterations": len(ll_history),
        }

    def predict_static(
        self,
        x_last: np.ndarray,
        Sigma_last: np.ndarray,
        params: Optional[KalmanVolumeParams] = None,
    ) -> dict:
        """
        Produce static forecasts for the next full day (no intraday updates).

        Parameters
        ----------
        x_last:     Filtered state at end of previous day, shape (2,).
        Sigma_last: Filtered covariance at end of previous day, shape (2, 2).
        params:     Model parameters. Uses self.params if None.

        Returns
        -------
        Dictionary with log-volume forecasts, forecast variances, and
        bias-corrected linear volume forecasts for each bin.
        """
        if params is None:
            params = self.params

        I = self.I
        C = np.array([1.0, 1.0])

        y_hat = np.zeros(I)
        V_hat = np.zeros(I)
        vol_hat = np.zeros(I)

        x_curr = x_last.copy()
        Sigma_curr = Sigma_last.copy()

        for h in range(I):
            # Build transition for this bin
            if h == 0:
                # Day boundary
                A = np.array([
                    [params.a_eta, 0.0],
                    [0.0, params.a_mu],
                ])
                Q = np.array([
                    [params.sigma_eta_sq, 0.0],
                    [0.0, params.sigma_mu_sq],
                ])
            else:
                A = np.array([
                    [1.0, 0.0],
                    [0.0, params.a_mu],
                ])
                Q = np.array([
                    [0.0, 0.0],
                    [0.0, params.sigma_mu_sq],
                ])

            x_curr = A @ x_curr
            Sigma_curr = A @ Sigma_curr @ A.T + Q

            y_hat[h] = C @ x_curr + params.phi[h]
            V_hat[h] = C @ Sigma_curr @ C + params.r
            # Bias-corrected linear forecast
            vol_hat[h] = np.exp(y_hat[h] + 0.5 * V_hat[h])

        # Compute weights
        weights = vol_hat / np.sum(vol_hat)

        return {
            "y_hat": y_hat,
            "V_hat": V_hat,
            "vol_hat": vol_hat,
            "weights": weights,
        }

    def predict_dynamic(
        self,
        y: np.ndarray,
        is_observed: np.ndarray,
        params: Optional[KalmanVolumeParams] = None,
    ) -> dict:
        """
        Run the Kalman filter producing one-step-ahead dynamic forecasts.

        Parameters
        ----------
        y:           Log-volume observations, shape (T, I).
        is_observed: Boolean mask, shape (T, I).
        params:      Model parameters. Uses self.params if None.

        Returns
        -------
        Dictionary with log-volume forecasts, forecast variances, filtered
        states, and covariances.
        """
        filtered = self.kalman_filter(y, is_observed, params)
        T, I = y.shape
        N = T * I

        y_hat = filtered["y_hat"]
        S = filtered["S"]

        # Bias-corrected linear forecasts
        vol_hat = np.exp(y_hat + 0.5 * S)

        return {
            "y_hat": y_hat.reshape(T, I),
            "V_hat": S.reshape(T, I),
            "vol_hat": vol_hat.reshape(T, I),
            "x_filt": filtered["x_filt"],
            "Sigma_filt": filtered["Sigma_filt"],
        }

    def compute_vwap_dynamic(
        self,
        y_day: np.ndarray,
        is_observed_day: np.ndarray,
        x_prev: np.ndarray,
        Sigma_prev: np.ndarray,
        params: Optional[KalmanVolumeParams] = None,
    ) -> np.ndarray:
        """
        Compute dynamic VWAP weights for a single day.

        At each bin, revise remaining weights using multi-step forecasts
        from the current filtered state.

        Parameters
        ----------
        y_day:           Log-volume for the day, shape (I,).
        is_observed_day: Observation mask for the day, shape (I,).
        x_prev:          Filtered state at end of previous day.
        Sigma_prev:      Filtered covariance at end of previous day.
        params:          Model parameters. Uses self.params if None.

        Returns
        -------
        Dynamic VWAP weight vector, shape (I,).
        """
        if params is None:
            params = self.params

        I = self.I
        C = np.array([1.0, 1.0])
        weights = np.zeros(I)

        x_curr = x_prev.copy()
        Sigma_curr = Sigma_prev.copy()

        for i in range(I):
            if i == I - 1:
                weights[i] = 1.0 - np.sum(weights[:i])
                break

            # Multi-step predictions for remaining bins i..I-1
            remaining_vol = np.zeros(I - i)
            x_ms = x_curr.copy()
            Sigma_ms = Sigma_curr.copy()

            for h in range(I - i):
                bin_idx = i + h
                if h == 0 and i == 0:
                    # Day boundary
                    A = np.array([
                        [params.a_eta, 0.0],
                        [0.0, params.a_mu],
                    ])
                    Q = np.array([
                        [params.sigma_eta_sq, 0.0],
                        [0.0, params.sigma_mu_sq],
                    ])
                elif h == 0 and i > 0:
                    # Already within the day, first step from current state
                    # This is within-day transition
                    A = np.array([
                        [1.0, 0.0],
                        [0.0, params.a_mu],
                    ])
                    Q = np.array([
                        [0.0, 0.0],
                        [0.0, params.sigma_mu_sq],
                    ])
                else:
                    A = np.array([
                        [1.0, 0.0],
                        [0.0, params.a_mu],
                    ])
                    Q = np.array([
                        [0.0, 0.0],
                        [0.0, params.sigma_mu_sq],
                    ])

                x_ms = A @ x_ms
                Sigma_ms = A @ Sigma_ms @ A.T + Q
                V_h = C @ Sigma_ms @ C + params.r
                y_h = C @ x_ms + params.phi[bin_idx]
                remaining_vol[h] = np.exp(y_h + 0.5 * V_h)

            remaining_weight = 1.0 - np.sum(weights[:i])
            weights[i] = remaining_weight * remaining_vol[0] / np.sum(remaining_vol)

            # Now update state with actual observation at bin i
            if i == 0:
                A_step = np.array([
                    [params.a_eta, 0.0],
                    [0.0, params.a_mu],
                ])
                Q_step = np.array([
                    [params.sigma_eta_sq, 0.0],
                    [0.0, params.sigma_mu_sq],
                ])
            else:
                A_step = np.array([
                    [1.0, 0.0],
                    [0.0, params.a_mu],
                ])
                Q_step = np.array([
                    [0.0, 0.0],
                    [0.0, params.sigma_mu_sq],
                ])

            x_pred_i = A_step @ x_curr
            Sigma_pred_i = A_step @ Sigma_curr @ A_step.T + Q_step

            if is_observed_day[i]:
                e_i = y_day[i] - C @ x_pred_i - params.phi[i]
                S_i = C @ Sigma_pred_i @ C + params.r
                K_i = Sigma_pred_i @ C / S_i

                e_clean = e_i
                if self.robust:
                    threshold = self.lam * S_i / 2.0
                    if e_i > threshold:
                        z_i = e_i - threshold
                    elif e_i < -threshold:
                        z_i = e_i + threshold
                    else:
                        z_i = 0.0
                    e_clean = e_i - z_i

                x_curr = x_pred_i + K_i * e_clean
                IKC = np.eye(2) - np.outer(K_i, C)
                Sigma_curr = (
                    IKC @ Sigma_pred_i @ IKC.T
                    + params.r * np.outer(K_i, K_i)
                )
            else:
                x_curr = x_pred_i
                Sigma_curr = Sigma_pred_i

        return weights


def compute_mape(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    is_observed: np.ndarray,
) -> float:
    """
    Compute mean absolute percentage error on normalized volume.

    Operates in linear (not log) space per Paper Eq 37. Computes
    MAPE = (1/M) * sum |actual - predicted| / actual, where actual
    and predicted are the normalized volumes (exp of log-volumes).

    Parameters
    ----------
    y_actual: Actual log-volume, shape (T, I) or flat.
    y_pred:   Predicted log-volume, shape matching y_actual.
    is_observed: Boolean mask, same shape.

    Returns
    -------
    MAPE value (0 to inf, where 0.46 means 46% average error).
    """
    actual_flat = y_actual.ravel()
    pred_flat = y_pred.ravel()
    obs_flat = is_observed.ravel()

    # Convert to linear space
    actual_lin = np.exp(actual_flat[obs_flat])
    pred_lin = np.exp(pred_flat[obs_flat])

    # Avoid division by zero
    mask = actual_lin > 1e-12
    ape = np.abs(actual_lin[mask] - pred_lin[mask]) / actual_lin[mask]
    return float(np.mean(ape))


def rolling_mean_baseline(
    y: np.ndarray,
    is_observed: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Compute rolling mean baseline forecasts for each bin.

    For each day t and bin i, forecast is the average of y[t-window:t, i].

    Parameters
    ----------
    y:           Log-volume, shape (T, I).
    is_observed: Boolean mask, shape (T, I).
    window:      Number of trailing days to average.

    Returns
    -------
    Forecast array, shape (T, I). First 'window' days are NaN.
    """
    T, I = y.shape
    forecasts = np.full((T, I), np.nan)

    for t in range(window, T):
        for i in range(I):
            obs_mask = is_observed[t - window:t, i]
            if np.any(obs_mask):
                forecasts[t, i] = np.mean(y[t - window:t, i][obs_mask])

    return forecasts
