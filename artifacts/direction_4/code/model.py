"""
Dual-Mode Intraday Volume Forecast (Satish, Saxena, Palmer 2014).

Model A forecasts raw bin-level trading volume by combining three
components (historical average, inter-day ARMA, intraday ARMA) with
regime-switching dynamic weights. Model B forecasts next-bin volume
percentages via a surprise-regression adjustment on top of a historical
volume percentage curve.

Classes
-------
ModelParams
    All configurable parameters for the dual-model system.
RegimeClassifier
    Classifies current volume regime based on cumulative volume percentiles.
ModelA
    Trained Model A state for raw volume forecasting.
PercentageModel
    Trained Model B state for volume percentage forecasting.

Functions
---------
compute_seasonal_factors
    Compute per-bin rolling average for deseasonalization.
compute_historical_average
    Compute per-bin rolling average for Component 1 (H).
fit_interday_arma
    Fit per-bin inter-day ARMA models with AICc selection.
fit_intraday_arma
    Fit single intraday ARMA on deseasonalized volume segments.
build_regime_classifier
    Build regime classifier from cumulative volume percentiles.
assign_regime
    Assign regime index given observed cumulative volume.
optimize_regime_weights
    Find optimal per-regime component weights via MAPE minimization.
forecast_raw_volume
    Produce raw volume forecast for a target bin.
train_percentage_model
    Train Model B surprise regression.
forecast_volume_percentage
    Forecast next-bin volume fraction.
train_full_model
    Top-level training orchestration.
compute_evaluation_mape
    Compute MAPE over an evaluation period.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA

# Sentinel for failed ARMA fits
FALLBACK = "FALLBACK"

# Number of bins per trading day (15-min bins, 9:30-16:00 ET)
I = 26


@dataclass
class ModelParams:
    """
    All configurable parameters for the dual-model system.

    Attributes
    ----------
    N_seasonal : int
        Trailing window for deseasonalization (trading days).
    N_hist : int
        Rolling window for Component 1 historical average.
    N_interday_fit : int
        Fitting window for inter-day ARMA.
    p_max_inter : int
        Maximum AR order for inter-day ARMA.
    q_max_inter : int
        Maximum MA order for inter-day ARMA.
    N_intraday_fit : int
        Rolling window for intraday ARMA.
    p_max_intra : int
        Maximum AR order for intraday ARMA.
    q_max_intra : int
        Maximum MA order for intraday ARMA.
    N_regime_window : int
        Lookback for cumulative volume distribution.
    regime_candidates : list
        Candidate regime counts for grid search.
    N_weight_train : int
        Training window for weight optimization.
    min_samples_per_regime : int
        Minimum samples per regime bucket.
    min_volume_floor : float
        Minimum volume for MAPE computation.
    N_regression_fit : int
        Training window for surprise regression.
    L_max : int
        Maximum number of lagged surprise terms.
    max_deviation : float
        Maximum relative departure from historical pct curve.
    pct_switchoff : float
        Cumulative volume fraction triggering switch to historical curve.
    reestimation_interval : int
        Days between full model re-estimation.
    """

    N_seasonal: int = 126
    N_hist: int = 21
    N_interday_fit: int = 63
    p_max_inter: int = 5
    q_max_inter: int = 5
    N_intraday_fit: int = 21
    p_max_intra: int = 4
    q_max_intra: int = 5
    N_regime_window: int = 63
    regime_candidates: list = field(default_factory=lambda: [3, 4, 5])
    N_weight_train: int = 63
    min_samples_per_regime: int = 50
    min_volume_floor: float = 100.0
    N_regression_fit: int = 63
    L_max: int = 5
    max_deviation: float = 0.10
    pct_switchoff: float = 0.80
    reestimation_interval: int = 21


@dataclass
class RegimeClassifier:
    """
    Classify current volume regime based on cumulative volume percentiles.

    Attributes
    ----------
    thresholds : dict
        Per-bin list of volume thresholds for regime boundaries.
    n_regimes : int
        Number of regimes.
    """

    thresholds: dict
    n_regimes: int


@dataclass
class ModelA:
    """
    Trained Model A state for raw volume forecasting.

    Attributes
    ----------
    hist_avg : np.ndarray
        Per-bin historical average (Component 1).
    seasonal_factors : np.ndarray
        Per-bin seasonal factors for deseasonalization.
    interday_models : list
        Per-bin fitted inter-day ARMA models (or FALLBACK).
    intraday_model : object
        Single intraday ARMA model (or FALLBACK).
    regime_classifier : RegimeClassifier
        Regime classification system.
    weights : dict
        Per-regime weight vectors [w_H, w_D, w_A].
    """

    hist_avg: np.ndarray
    seasonal_factors: np.ndarray
    interday_models: list
    intraday_model: object
    regime_classifier: RegimeClassifier
    weights: dict


@dataclass
class PercentageModel:
    """
    Trained Model B state for volume percentage forecasting.

    Attributes
    ----------
    beta : np.ndarray
        Surprise regression coefficients.
    L : int
        Number of lag terms.
    hist_pct : np.ndarray
        Historical volume percentage curve.
    """

    beta: np.ndarray
    L: int
    hist_pct: np.ndarray


def compute_seasonal_factors(volume_matrix: np.ndarray, n_seasonal: int) -> np.ndarray:
    """
    Compute per-bin rolling average for deseasonalization (Function 1).

    Parameters
    ----------
    volume_matrix: 2D array (n_days, I) of daily bin volumes.
    n_seasonal:    Number of trailing days to average.

    Returns
    -------
    Array of length I with per-bin seasonal factors.
    """
    data = volume_matrix[-n_seasonal:]
    sf = np.mean(data, axis=0)
    # Guard against zero: replace with min non-zero
    nonzero = sf[sf > 0]
    if len(nonzero) > 0:
        floor = np.min(nonzero)
    else:
        floor = 1.0
    sf[sf == 0] = floor
    return sf


def compute_historical_average(volume_matrix: np.ndarray, n_hist: int) -> np.ndarray:
    """
    Compute per-bin rolling average for Component 1 (Function 1a).

    Parameters
    ----------
    volume_matrix: 2D array (n_days, I) of daily bin volumes.
    n_hist:        Number of trailing days to average.

    Returns
    -------
    Array of length I with per-bin historical averages.
    """
    data = volume_matrix[-n_hist:]
    ha = np.mean(data, axis=0)
    nonzero = ha[ha > 0]
    if len(nonzero) > 0:
        floor = np.min(nonzero)
    else:
        floor = 1.0
    ha[ha == 0] = floor
    return ha


def _fit_arma_with_aicc(
    series: np.ndarray, p: int, q: int
) -> tuple:
    """
    Fit a single ARMA(p,q) model and return (aicc, model) or (inf, None).

    Parameters
    ----------
    series: 1D array of observations.
    p:      AR order.
    q:      MA order.

    Returns
    -------
    Tuple of (AICc value, fitted model result or None).
    """
    n = len(series)
    k = p + q + 1  # +1 for constant
    if n <= k + 1:
        return (np.inf, None)
    try:
        model = ARIMA(series, order=(p, 0, q), trend="c",
                       enforce_stationarity=True,
                       enforce_invertibility=True)
        result = model.fit(method_kwargs={"maxiter": 200, "disp": False})
        if not result.mle_retvals.get("converged", True):
            return (np.inf, None)

        ll = result.llf
        aic = -2 * ll + 2 * k
        aicc = aic + 2 * k * (k + 1) / (n - k - 1)
        return (aicc, result)
    except Exception:
        return (np.inf, None)


def fit_interday_arma(
    volume_matrix: np.ndarray,
    n_interday_fit: int,
    p_max: int,
    q_max: int,
) -> list:
    """
    Fit I independent ARMA(p,q) models on daily volume series (Function 2).

    Parameters
    ----------
    volume_matrix:  2D array (n_days, I) of daily bin volumes.
    n_interday_fit: Number of trailing days for fitting.
    p_max:          Maximum AR order.
    q_max:          Maximum MA order.

    Returns
    -------
    List of I fitted model results (or FALLBACK sentinel).
    """
    data = volume_matrix[-n_interday_fit:]
    n_bins = data.shape[1]
    models = []

    for i in range(n_bins):
        series = data[:, i].copy()
        best_aicc = np.inf
        best_model = None

        for p in range(p_max + 1):
            for q in range(q_max + 1):
                if p == 0 and q == 0:
                    continue  # skip trivial model
                aicc, result = _fit_arma_with_aicc(series, p, q)
                if aicc < best_aicc:
                    best_aicc = aicc
                    best_model = result

        if best_model is None:
            models.append(FALLBACK)
        else:
            models.append(best_model)

    return models


def _predict_interday_at(model_result, volume_series: np.ndarray, day_idx: int) -> float:
    """
    One-step-ahead inter-day ARMA forecast for a specific day.

    Parameters
    ----------
    model_result: Fitted statsmodels ARIMA result.
    volume_series: Full daily volume series for this bin.
    day_idx:       Index of the day to forecast (uses data up to day_idx-1).

    Returns
    -------
    Forecasted volume for day_idx.
    """
    try:
        # Apply the model to the data up to day_idx and forecast one step
        res = model_result.apply(volume_series[:day_idx])
        fc = res.forecast(steps=1)
        return float(fc.iloc[0]) if hasattr(fc, 'iloc') else float(fc[0])
    except Exception:
        return float(np.mean(volume_series[max(0, day_idx - 21):day_idx]))


def _predict_interday_next(model_result, volume_series: np.ndarray) -> float:
    """
    One-step-ahead forecast for the next unobserved day.

    Parameters
    ----------
    model_result: Fitted statsmodels ARIMA result.
    volume_series: Full daily volume series through last observed day.

    Returns
    -------
    Forecasted volume for next day.
    """
    try:
        res = model_result.apply(volume_series)
        fc = res.forecast(steps=1)
        return float(fc.iloc[0]) if hasattr(fc, 'iloc') else float(fc[0])
    except Exception:
        return float(np.mean(volume_series[-21:]))


def fit_intraday_arma(
    volume_matrix: np.ndarray,
    seasonal_factors: np.ndarray,
    n_intraday_fit: int,
    p_max: int,
    q_max: int,
) -> object:
    """
    Fit a single intraday ARMA on deseasonalized volume segments (Function 3).

    Each day is treated as an independent segment. We concatenate all segments
    and fit a single ARMA model. Day boundaries are handled by fitting on
    the full concatenated series (approximation: the conditional MLE approach
    with segment breaks is complex; we use the concatenated approach and rely
    on the short segment length to limit cross-day contamination).

    Parameters
    ----------
    volume_matrix:    2D array (n_days, I) of daily bin volumes.
    seasonal_factors: Per-bin seasonal factors for deseasonalization.
    n_intraday_fit:   Number of trailing days for fitting.
    p_max:            Maximum AR order.
    q_max:            Maximum MA order.

    Returns
    -------
    Fitted ARMA model result or FALLBACK sentinel.
    """
    data = volume_matrix[-n_intraday_fit:]
    n_days = data.shape[0]

    # Build deseasonalized segments
    segments = []
    for d in range(n_days):
        deseas = data[d, :] / seasonal_factors
        segments.append(deseas)

    # Concatenate all segments for fitting
    concat = np.concatenate(segments)

    best_aicc = np.inf
    best_model = None

    for p in range(p_max + 1):
        for q in range(q_max + 1):
            if p == 0 and q == 0:
                continue
            n_eff = n_days * (I - max(p, q))
            if n_eff <= 0:
                continue
            k = p + q + 1
            if n_eff <= k + 1:
                continue
            try:
                model = ARIMA(concat, order=(p, 0, q), trend="c",
                               enforce_stationarity=True,
                               enforce_invertibility=True)
                result = model.fit(method_kwargs={"maxiter": 200, "disp": False})
                if not result.mle_retvals.get("converged", True):
                    continue
                ll = result.llf
                aic = -2 * ll + 2 * k
                aicc = aic + 2 * k * (k + 1) / (n_eff - k - 1)
                if aicc < best_aicc:
                    best_aicc = aicc
                    best_model = result
            except Exception:
                continue

    if best_model is None:
        return FALLBACK
    return best_model


def _predict_intraday(
    intraday_model,
    observed_deseas: np.ndarray,
    seasonal_factors: np.ndarray,
    steps: int,
) -> np.ndarray:
    """
    Multi-step intraday forecast from observed deseasonalized values.

    Creates a fresh prediction state from the observations without mutating
    the model's internal state.

    Parameters
    ----------
    intraday_model:  Fitted ARMA result for intraday.
    observed_deseas: Array of deseasonalized observations so far today.
    seasonal_factors: Per-bin seasonal factors (for context, not used here).
    steps:           Number of steps ahead to forecast.

    Returns
    -------
    Array of length `steps` with deseasonalized forecasts.
    """
    if intraday_model is FALLBACK:
        return np.ones(steps)  # unconditional mean ~1.0

    try:
        if len(observed_deseas) == 0:
            # No observations: forecast from unconditional mean
            # Use the model's unconditional mean
            res = intraday_model
            fc = res.forecast(steps=steps)
            if hasattr(fc, 'values'):
                fc = fc.values
            return np.array(fc, dtype=float)
        else:
            # Apply model to observed data and forecast
            res = intraday_model.apply(observed_deseas)
            fc = res.forecast(steps=steps)
            if hasattr(fc, 'values'):
                fc = fc.values
            return np.array(fc, dtype=float)
    except Exception:
        return np.ones(steps)


def build_regime_classifier(
    volume_matrix: np.ndarray,
    n_regime_window: int,
    n_regimes: int,
) -> RegimeClassifier:
    """
    Build regime classifier from cumulative volume percentiles (Function 4).

    Parameters
    ----------
    volume_matrix:   2D array (n_days, I) of daily bin volumes.
    n_regime_window: Number of trailing days for distribution.
    n_regimes:       Number of regimes.

    Returns
    -------
    RegimeClassifier with per-bin thresholds.
    """
    data = volume_matrix[-n_regime_window:]
    cutoffs = [100 * k / n_regimes for k in range(1, n_regimes)]

    thresholds = {}
    for i in range(I):
        # Cumulative volume through bin i for each day
        cumvol = np.sum(data[:, :i + 1], axis=1)
        thresholds[i + 1] = [float(np.percentile(cumvol, c)) for c in cutoffs]

    return RegimeClassifier(thresholds=thresholds, n_regimes=n_regimes)


def assign_regime(
    classifier: RegimeClassifier,
    last_observed_bin: int,
    cumulative_volume: float,
) -> int:
    """
    Assign regime index given observed cumulative volume.

    Parameters
    ----------
    classifier:        RegimeClassifier with thresholds.
    last_observed_bin: Last bin for which volume has been observed (1-indexed).
    cumulative_volume: Sum of observed volumes through last_observed_bin.

    Returns
    -------
    Regime index in [0, n_regimes - 1].
    """
    if last_observed_bin < 1:
        return classifier.n_regimes // 2

    thresh = classifier.thresholds[last_observed_bin]
    regime = 0
    for t in thresh:
        if cumulative_volume > t:
            regime += 1
        else:
            break
    return regime


def optimize_regime_weights(
    volume_matrix: np.ndarray,
    n_hist: int,
    seasonal_factors: np.ndarray,
    interday_models: list,
    intraday_model: object,
    regime_classifier: RegimeClassifier,
    n_weight_train: int,
    min_volume_floor: float,
    full_volume_matrix: Optional[np.ndarray] = None,
) -> dict:
    """
    Find optimal per-regime component weights via MAPE minimization (Function 5).

    Parameters
    ----------
    volume_matrix:      2D array with enough history (N_hist + n_weight_train days).
                        The last n_weight_train days are the training period.
    n_hist:             Window for rolling historical average.
    seasonal_factors:   Per-bin seasonal factors.
    interday_models:    Per-bin inter-day ARMA models.
    intraday_model:     Intraday ARMA model.
    regime_classifier:  Regime classification system.
    n_weight_train:     Number of training days.
    min_volume_floor:   Minimum volume for MAPE computation.
    full_volume_matrix: Full volume matrix for inter-day ARMA predict_at (optional).

    Returns
    -------
    Dict mapping regime index to weight vector [w_H, w_D, w_A].
    """
    n_regimes = regime_classifier.n_regimes
    # Training days are the last n_weight_train days of volume_matrix
    total_days = volume_matrix.shape[0]
    train_start = total_days - n_weight_train

    samples_by_regime = {r: [] for r in range(n_regimes)}

    for d_idx in range(train_start, total_days):
        # Compute rolling hist_avg for this training day
        # Use data up to (but not including) this day's row for computing H
        # Actually, hist_avg for day d should use volume ending at day d-1
        # The spec says "N_hist-day average ending at d" -- we interpret as
        # including day d itself in the history window (as a rolling avg of
        # the most recent N_hist days up to and including d-1, predicting d).
        # But the spec says "volume_history[:d]" which means up to but not incl d.
        h_end = d_idx
        h_start = max(0, h_end - n_hist)
        if h_end > h_start:
            h_d = np.mean(volume_matrix[h_start:h_end], axis=0)
            nonzero = h_d[h_d > 0]
            if len(nonzero) > 0:
                h_d[h_d == 0] = np.min(nonzero)
            else:
                h_d[:] = 1.0
        else:
            h_d = np.ones(I)

        cumvol = 0.0
        for i in range(I):
            actual = volume_matrix[d_idx, i]
            if actual < min_volume_floor:
                cumvol += actual
                continue

            H = h_d[i]

            # Component 2: Inter-day ARMA
            if interday_models[i] is FALLBACK:
                D = H
            else:
                # Use predict_at: forecast for this day using history up to d-1
                bin_series = volume_matrix[:d_idx, i]
                D = _predict_interday_at(interday_models[i], bin_series, d_idx)
                if np.isnan(D) or np.isinf(D):
                    D = H

            # Component 3: Intraday ARMA
            if intraday_model is FALLBACK:
                A = seasonal_factors[i]
            else:
                # Condition on observed bins 1..(i-1) from same day
                if i > 0:
                    observed = volume_matrix[d_idx, :i]
                    observed_deseas = observed / seasonal_factors[:i]
                else:
                    observed_deseas = np.array([])
                forecasts = _predict_intraday(
                    intraday_model, observed_deseas, seasonal_factors, steps=1
                )
                A = forecasts[0] * seasonal_factors[i]
                if np.isnan(A) or np.isinf(A):
                    A = seasonal_factors[i]

            # Regime assignment
            regime = assign_regime(regime_classifier, i, cumvol)  # i is 0-indexed last-observed bin+1...
            # Wait: bin index is 1-indexed in the spec. i here is 0-indexed.
            # The spec says: regime = assign_regime(classifier, i-1, cumvol) where i is 1-indexed target bin
            # So if target bin (1-indexed) = i+1, then last_observed = (i+1)-1 = i (1-indexed) = i (but i is 0-indexed here)
            # Actually let me be more careful. In the spec:
            # FOR i = 1 TO I: ... regime = assign_regime(classifier, i-1, cumvol)
            # i is 1-indexed target bin. i-1 is the last observed bin (1-indexed).
            # In our code, i is 0-indexed (0 to I-1). Target bin is i+1 (1-indexed).
            # Last observed bin (1-indexed) = (i+1) - 1 = i. But i is 0-indexed here.
            # assign_regime expects 1-indexed bin. So last_observed_bin = i (0-indexed) = i (which as 1-indexed would be i).
            # Hmm, this is confusing. Let's use explicit 1-indexed logic.
            target_bin_1indexed = i + 1
            last_obs_bin_1indexed = target_bin_1indexed - 1  # = i
            regime = assign_regime(regime_classifier, last_obs_bin_1indexed, cumvol)

            samples_by_regime[regime].append((H, D, A, actual))
            cumvol += actual

    # Optimize weights per regime
    weights = {}
    for r in range(n_regimes):
        samples = samples_by_regime[r]
        if len(samples) < 50:  # min_samples_per_regime
            weights[r] = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
            continue

        samples_arr = np.array(samples)  # (n, 4): H, D, A, actual
        H_arr = samples_arr[:, 0]
        D_arr = samples_arr[:, 1]
        A_arr = samples_arr[:, 2]
        actual_arr = samples_arr[:, 3]

        def mape_loss(w_log: np.ndarray) -> float:
            """
            Compute MAPE loss with exp-transformed weights.

            Parameters
            ----------
            w_log: Log-space weight vector (3 elements).

            Returns
            -------
            Mean absolute percentage error.
            """
            w = np.exp(w_log)
            forecast = w[0] * H_arr + w[1] * D_arr + w[2] * A_arr
            forecast = np.maximum(forecast, 0.0)
            ape = np.abs(forecast - actual_arr) / actual_arr
            return float(np.mean(ape))

        starting_points = [
            np.log([1 / 3, 1 / 3, 1 / 3]),
            np.log([0.8, 0.1, 0.1]),
            np.log([0.1, 0.8, 0.1]),
            np.log([0.1, 0.1, 0.8]),
        ]

        best_loss = np.inf
        best_result = None
        for w0 in starting_points:
            try:
                result = minimize(
                    mape_loss, w0, method="Nelder-Mead",
                    options={"maxiter": 1000, "xatol": 1e-4, "fatol": 1e-6}
                )
                if result.fun < best_loss:
                    best_loss = result.fun
                    best_result = result
            except Exception:
                continue

        equal_loss = mape_loss(np.log([1 / 3, 1 / 3, 1 / 3]))
        if best_result is None or best_loss >= equal_loss:
            weights[r] = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        else:
            weights[r] = np.exp(best_result.x)

    return weights


def forecast_raw_volume(
    model_a: ModelA,
    volume_matrix: np.ndarray,
    day_idx: int,
    target_bin: int,
    observed_volumes: dict,
) -> float:
    """
    Produce raw volume forecast for a target bin (Function 6).

    Parameters
    ----------
    model_a:          Trained Model A state.
    volume_matrix:    Full volume matrix (for inter-day ARMA).
    day_idx:          Index of current day in volume_matrix.
    target_bin:       Target bin (1-indexed, 1..I).
    observed_volumes: Dict {bin_index(1-indexed): volume} for bins observed today.

    Returns
    -------
    Forecasted volume (shares) for target_bin.
    """
    current_bin = max(observed_volumes.keys()) if observed_volumes else 0
    tb_idx = target_bin - 1  # 0-indexed

    # Component 1: Historical average
    H = model_a.hist_avg[tb_idx]

    # Component 2: Inter-day ARMA
    if model_a.interday_models[tb_idx] is FALLBACK:
        D = H
    else:
        bin_series = volume_matrix[:day_idx, tb_idx]
        D = _predict_interday_next(model_a.interday_models[tb_idx], bin_series)
        if np.isnan(D) or np.isinf(D):
            D = H

    # Component 3: Intraday ARMA
    if model_a.intraday_model is FALLBACK:
        A = model_a.seasonal_factors[tb_idx]
    else:
        if current_bin > 0:
            observed_deseas = np.array([
                observed_volumes[j] / model_a.seasonal_factors[j - 1]
                for j in range(1, current_bin + 1)
                if j in observed_volumes
            ])
        else:
            observed_deseas = np.array([])

        steps_ahead = target_bin - current_bin
        if steps_ahead <= 0:
            steps_ahead = 1
        forecasts = _predict_intraday(
            model_a.intraday_model, observed_deseas,
            model_a.seasonal_factors, steps=steps_ahead
        )
        A_deseas = forecasts[-1]
        A = A_deseas * model_a.seasonal_factors[tb_idx]
        if np.isnan(A) or np.isinf(A):
            A = model_a.seasonal_factors[tb_idx]

    # Regime classification
    cumvol = sum(observed_volumes.values())
    regime = assign_regime(model_a.regime_classifier, current_bin, cumvol)

    # Weighted combination
    w = model_a.weights[regime]
    v_hat = w[0] * H + w[1] * D + w[2] * A
    return max(v_hat, 0.0)


def train_percentage_model(
    volume_matrix: np.ndarray,
    n_regression_fit: int,
    l_max: int,
    min_volume_floor: float,
) -> PercentageModel:
    """
    Train Model B surprise regression (Function 7).

    Parameters
    ----------
    volume_matrix:    2D array (n_days, I) of daily bin volumes.
    n_regression_fit: Number of trailing days for training.
    l_max:            Maximum number of lag terms.
    min_volume_floor: Minimum total daily volume for inclusion.

    Returns
    -------
    Trained PercentageModel.
    """
    data = volume_matrix[-n_regression_fit:]
    n_days = data.shape[0]

    # Step 1: Historical percentage curve
    daily_totals = np.sum(data, axis=1)
    # Avoid division by zero
    valid_mask = daily_totals > min_volume_floor
    if np.sum(valid_mask) == 0:
        hist_pct = np.ones(I) / I
    else:
        pct_matrix = data[valid_mask] / daily_totals[valid_mask, np.newaxis]
        hist_pct = np.mean(pct_matrix, axis=0)

    # Step 2: Compute surprises
    # surprise[d, i] = actual_pct[d, i] - hist_pct[i]
    surprise_matrix = np.zeros((n_days, I))
    valid_days = []
    for d in range(n_days):
        if daily_totals[d] < min_volume_floor:
            continue
        valid_days.append(d)
        for i in range(I):
            actual_pct = data[d, i] / daily_totals[d]
            surprise_matrix[d, i] = actual_pct - hist_pct[i]

    if len(valid_days) < 10:
        return PercentageModel(beta=np.zeros(1), L=1, hist_pct=hist_pct)

    # Step 3: Select optimal L via blocked time-series CV
    best_L = 1
    best_cv_error = np.inf

    for L_candidate in range(1, l_max + 1):
        K = 5
        block_size = len(valid_days) // K
        if block_size < 2:
            continue
        cv_errors = []

        for fold in range(K):
            if fold < K - 1:
                test_indices = valid_days[fold * block_size:(fold + 1) * block_size]
            else:
                test_indices = valid_days[fold * block_size:]
            train_indices = [d for d in valid_days if d not in test_indices]

            X_train, y_train = _build_surprise_regression(
                surprise_matrix, train_indices, L_candidate, l_max
            )
            if len(y_train) == 0 or X_train.shape[0] == 0:
                continue
            beta = _ols_no_intercept(X_train, y_train)

            X_test, y_test = _build_surprise_regression(
                surprise_matrix, test_indices, L_candidate, l_max
            )
            if len(y_test) == 0:
                continue
            y_pred = X_test @ beta
            cv_errors.append(float(np.mean(np.abs(y_test - y_pred))))

        if cv_errors:
            mean_cv = np.mean(cv_errors)
            if mean_cv < best_cv_error:
                best_cv_error = mean_cv
                best_L = L_candidate

    # Step 4: Fit final model
    X_all, y_all = _build_surprise_regression(
        surprise_matrix, valid_days, best_L, l_max
    )
    if len(y_all) > 0:
        beta_final = _ols_no_intercept(X_all, y_all)
    else:
        beta_final = np.zeros(best_L)

    return PercentageModel(beta=beta_final, L=best_L, hist_pct=hist_pct)


def _build_surprise_regression(
    surprise_matrix: np.ndarray,
    day_indices: list,
    L: int,
    l_max: int,
) -> tuple:
    """
    Build regression matrices for surprise model.

    Parameters
    ----------
    surprise_matrix: 2D array (n_days, I) of surprise values.
    day_indices:     List of day indices to use.
    L:               Number of lag terms.
    l_max:           Maximum lag (for edge-effect avoidance).

    Returns
    -------
    Tuple of (X, y) arrays for regression.
    """
    X_rows = []
    y_values = []
    for d in day_indices:
        for i in range(l_max, I):  # Skip first l_max bins
            lags = [surprise_matrix[d, i - lag - 1] if (i - lag - 1) >= 0 else 0.0
                    for lag in range(L)]
            X_rows.append(lags)
            y_values.append(surprise_matrix[d, i])
    if not X_rows:
        return np.zeros((0, L)), np.zeros(0)
    return np.array(X_rows), np.array(y_values)


def _ols_no_intercept(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    OLS regression without intercept.

    Parameters
    ----------
    X: Design matrix (n, L).
    y: Response vector (n,).

    Returns
    -------
    Coefficient vector (L,).
    """
    try:
        return np.linalg.solve(X.T @ X, X.T @ y)
    except np.linalg.LinAlgError:
        # Fallback: use least squares
        result, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        return result


def forecast_volume_percentage(
    model_a: ModelA,
    pct_model: PercentageModel,
    volume_matrix: np.ndarray,
    day_idx: int,
    current_bin: int,
    observed_volumes: dict,
    max_deviation: float,
    pct_switchoff: float,
    min_volume_floor: float,
) -> float:
    """
    Forecast next-bin volume fraction (Function 8).

    Parameters
    ----------
    model_a:          Trained Model A.
    pct_model:        Trained percentage model.
    volume_matrix:    Full volume matrix.
    day_idx:          Index of current day.
    current_bin:      Last observed bin (1-indexed).
    observed_volumes: Dict {bin(1-indexed): volume} for bins seen today.
    max_deviation:    Maximum relative departure from hist pct.
    pct_switchoff:    Cumulative fraction triggering switch-off.
    min_volume_floor: Minimum volume floor.

    Returns
    -------
    Forecasted volume fraction for next bin.
    """
    next_bin = current_bin + 1
    if next_bin > I:
        return 0.0

    # Step 1: Estimate total daily volume
    observed_total = sum(observed_volumes.values())
    remaining_forecast = 0.0
    for j in range(next_bin, I + 1):
        remaining_forecast += forecast_raw_volume(
            model_a, volume_matrix, day_idx, target_bin=j,
            observed_volumes=observed_volumes
        )
    v_total_est = observed_total + remaining_forecast
    if v_total_est <= 0:
        v_total_est = max(observed_total, 1.0)

    # Step 2: Percentage-space surprises
    surprises = []
    for j in range(1, current_bin + 1):
        actual_pct = observed_volumes.get(j, 0.0) / v_total_est
        surprise = actual_pct - pct_model.hist_pct[j - 1]
        surprises.append(surprise)

    # Step 3: Surprise-based adjustment
    L = pct_model.L
    if current_bin < L:
        delta = 0.0
    else:
        observed_frac = observed_total / v_total_est
        if observed_frac >= pct_switchoff:
            delta = 0.0
        else:
            lag_vector = np.array([
                surprises[current_bin - lag - 1] for lag in range(L)
            ])
            delta = float(np.dot(pct_model.beta, lag_vector))

    # Step 4: Scale baseline
    observed_hist_frac = np.sum(pct_model.hist_pct[:current_bin])
    remaining_hist_frac = 1.0 - observed_hist_frac
    actual_remaining_frac = 1.0 - (observed_total / v_total_est)

    if remaining_hist_frac > 0:
        scale = actual_remaining_frac / remaining_hist_frac
    else:
        scale = 1.0

    # Step 5: Apply with deviation constraint
    base_pct = pct_model.hist_pct[next_bin - 1]
    max_delta = max_deviation * base_pct
    clamped_delta = np.clip(delta, -max_delta, max_delta)
    adjusted = scale * (base_pct + clamped_delta)

    # Step 6: Non-negativity
    pct_forecast = max(adjusted, 0.0)

    # Step 7: Last-bin special case
    if next_bin == I:
        pct_forecast = max(actual_remaining_frac, 0.0)

    return pct_forecast


def compute_evaluation_mape(
    model_a: ModelA,
    volume_matrix: np.ndarray,
    eval_day_indices: list,
    min_volume_floor: float,
) -> float:
    """
    Compute MAPE over an evaluation period using conditioned forecasts.

    Parameters
    ----------
    model_a:          Trained Model A.
    volume_matrix:    Full volume matrix.
    eval_day_indices: List of day indices to evaluate.
    min_volume_floor: Minimum volume for inclusion.

    Returns
    -------
    Mean absolute percentage error.
    """
    total_ape = 0.0
    count = 0
    for d_idx in eval_day_indices:
        for i in range(I):
            actual = volume_matrix[d_idx, i]
            if actual < min_volume_floor:
                continue
            # Build observed volumes for bins 1..(i) -- i.e., bins before target
            observed = {}
            for j in range(i):
                observed[j + 1] = volume_matrix[d_idx, j]
            target_bin = i + 1
            predicted = forecast_raw_volume(
                model_a, volume_matrix, d_idx, target_bin, observed
            )
            total_ape += abs(predicted - actual) / actual
            count += 1
    if count == 0:
        return np.inf
    return total_ape / count


def compute_baseline_mape(
    volume_matrix: np.ndarray,
    eval_day_indices: list,
    n_hist: int,
    min_volume_floor: float,
) -> float:
    """
    Compute baseline MAPE using historical average only (for comparison).

    Parameters
    ----------
    volume_matrix:    Full volume matrix.
    eval_day_indices: Day indices to evaluate.
    n_hist:           Rolling window for historical average.
    min_volume_floor: Minimum volume for inclusion.

    Returns
    -------
    Baseline MAPE using historical average.
    """
    total_ape = 0.0
    count = 0
    for d_idx in eval_day_indices:
        h_start = max(0, d_idx - n_hist)
        if h_start == d_idx:
            continue
        hist_avg = np.mean(volume_matrix[h_start:d_idx], axis=0)
        for i in range(I):
            actual = volume_matrix[d_idx, i]
            if actual < min_volume_floor:
                continue
            predicted = hist_avg[i]
            if predicted <= 0:
                continue
            total_ape += abs(predicted - actual) / actual
            count += 1
    if count == 0:
        return np.inf
    return total_ape / count


def train_full_model(
    volume_matrix: np.ndarray,
    train_end_idx: int,
    params: ModelParams,
) -> tuple:
    """
    Top-level training function (Function 9).

    Parameters
    ----------
    volume_matrix: Full 2D array (n_days, I) of daily bin volumes.
    train_end_idx: Index of the last training day (exclusive upper bound).
    params:        Model parameters.

    Returns
    -------
    Tuple of (ModelA, PercentageModel).
    """
    data = volume_matrix[:train_end_idx]

    # Step 1: Seasonal factors
    sf = compute_seasonal_factors(data, params.N_seasonal)

    # Step 1a: Historical average (at train_end_date, for prediction)
    hist_avg = compute_historical_average(data, params.N_hist)

    # Step 2: Inter-day ARMA
    interday_models = fit_interday_arma(
        data, params.N_interday_fit, params.p_max_inter, params.q_max_inter
    )

    # Step 3: Intraday ARMA
    intraday_model = fit_intraday_arma(
        data, sf, params.N_intraday_fit, params.p_max_intra, params.q_max_intra
    )

    # Step 4: Regime grid search
    best_config = None
    best_oos_mape = np.inf

    for n_reg in params.regime_candidates:
        rc = build_regime_classifier(data, params.N_regime_window, n_reg)

        # Split: last 21 days as validation
        val_size = 21
        n_total = data.shape[0]
        val_start = n_total - val_size
        train_end_for_weights = val_start

        # Training window for weights (before validation)
        wt_size = min(params.N_weight_train, train_end_for_weights)
        wt_start = train_end_for_weights - wt_size

        # Extended slice including N_hist pre-context
        ext_start = max(0, wt_start - params.N_hist)
        wt_data = data[ext_start:train_end_for_weights]
        actual_wt_size = train_end_for_weights - wt_start

        weights = optimize_regime_weights(
            wt_data, params.N_hist, sf, interday_models, intraday_model,
            rc, actual_wt_size, params.min_volume_floor,
        )

        # Evaluate on validation period
        val_hist_avg = compute_historical_average(data[:val_start], params.N_hist)
        temp_model_a = ModelA(
            val_hist_avg, sf, interday_models, intraday_model, rc, weights
        )
        val_days = list(range(val_start, n_total))
        val_mape = compute_evaluation_mape(
            temp_model_a, volume_matrix, val_days, params.min_volume_floor
        )

        if val_mape < best_oos_mape:
            best_oos_mape = val_mape
            best_config = (n_reg, rc, weights)

    n_reg = best_config[0]

    # Re-optimize on full N_weight_train window
    final_rc = build_regime_classifier(data, params.N_regime_window, n_reg)
    n_total = data.shape[0]
    full_wt_size = min(params.N_weight_train, n_total)
    full_wt_start = n_total - full_wt_size
    full_ext_start = max(0, full_wt_start - params.N_hist)
    full_wt_data = data[full_ext_start:]
    weights = optimize_regime_weights(
        full_wt_data, params.N_hist, sf, interday_models, intraday_model,
        final_rc, full_wt_size, params.min_volume_floor,
    )

    # Log combined ARMA terms check
    max_interday_k = 0
    for i in range(I):
        if interday_models[i] is not FALLBACK:
            k = interday_models[i].model.k_ar + interday_models[i].model.k_ma + 1
            max_interday_k = max(max_interday_k, k)
    if intraday_model is not FALLBACK:
        intraday_k = intraday_model.model.k_ar + intraday_model.model.k_ma + 1
    else:
        intraday_k = 0
    combined_terms = max_interday_k + intraday_k
    if combined_terms > 10:
        print(f"WARNING: Combined ARMA terms = {combined_terms} exceeds 10 (soft limit)")

    model_a = ModelA(
        hist_avg, sf, interday_models, intraday_model, final_rc, weights
    )

    # Step 5: Percentage model
    pct_model = train_percentage_model(
        data, params.N_regression_fit, params.L_max, params.min_volume_floor
    )

    return model_a, pct_model


def load_volume_matrix(data_dir: str) -> tuple:
    """
    Load prepared data into a volume matrix.

    Parameters
    ----------
    data_dir: Path to prepared data directory.

    Returns
    -------
    Tuple of (volume_matrix, dates, tickers) where volume_matrix is a dict
    mapping ticker to 2D array (n_days, I).
    """
    import json
    import os

    metadata_path = os.path.join(data_dir, "metadata.json")
    with open(metadata_path) as f:
        metadata = json.load(f)

    tickers = metadata["tickers"]
    volume_data = {}
    dates_data = {}

    for ticker in tickers:
        fpath = os.path.join(data_dir, f"{ticker}_15m_volume.parquet")
        df = pd.read_parquet(fpath)
        cols = [f"bin_{i}" for i in range(1, I + 1)]
        matrix = df[cols].values.astype(np.float64)
        volume_data[ticker] = matrix
        dates_data[ticker] = pd.to_datetime(df.index).date if hasattr(df.index, 'date') else df.index.tolist()

    return volume_data, dates_data, tickers
