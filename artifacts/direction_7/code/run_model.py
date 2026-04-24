"""
Run the Kalman filter volume model on prepared data.

Loads prepared log-volume matrices, fits the model via EM on training data,
evaluates on out-of-sample data using dynamic and static predictions,
and computes MAPE and VWAP tracking error.

Functions
---------
load_data
    Load prepared log-volume and observation mask for a ticker.
run_single_ticker
    Run the full pipeline for one ticker.
main
    Run all tickers and produce summary results.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Allow import from same directory
sys.path.insert(0, str(Path(__file__).parent))
from kalman_volume import (
    KalmanVolumeModel,
    compute_mape,
    rolling_mean_baseline,
)

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "direction_7" / "prepared"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "code"


def load_data(ticker: str) -> dict:
    """
    Load prepared log-volume and observation mask for a ticker.

    Parameters
    ----------
    ticker: Stock ticker symbol (e.g., "SPY").

    Returns
    -------
    Dictionary with log-volume matrix, observation mask, dates, and
    raw volume matrix.
    """
    log_vol = pd.read_parquet(DATA_DIR / f"{ticker}_log_volume.parquet")
    observed = pd.read_parquet(DATA_DIR / f"{ticker}_observed.parquet")
    raw_vol = pd.read_parquet(DATA_DIR / f"{ticker}_raw_volume_15min.parquet")

    return {
        "y": log_vol.values,
        "is_observed": observed.values.astype(bool),
        "dates": log_vol.index,
        "raw_volume": raw_vol.values,
    }


def run_single_ticker(
    ticker: str,
    train_days: int = 252,
    lambda_val: float = 5.0,
    robust: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Run the full pipeline for one ticker.

    Splits data into training and out-of-sample, fits via EM on training,
    then evaluates with rolling-window re-estimation.

    Parameters
    ----------
    ticker:     Stock ticker symbol.
    train_days: Number of training days per rolling window.
    lambda_val: Lasso regularization parameter (robust mode).
    robust:     Whether to use the robust filter.
    verbose:    Whether to print progress.

    Returns
    -------
    Dictionary with MAPE scores, VWAP tracking errors, estimated parameters,
    and forecast arrays.
    """
    data = load_data(ticker)
    y = data["y"]
    is_observed = data["is_observed"]
    dates = data["dates"]
    raw_volume = data["raw_volume"]

    T, I = y.shape

    if verbose:
        print(f"\n{'='*60}")
        print(f"Ticker: {ticker}, T={T}, I={I}, train_days={train_days}")
        print(f"Robust: {robust}, lambda: {lambda_val}")
        print(f"{'='*60}")

    # Out-of-sample days
    oos_start = train_days
    oos_days = T - oos_start
    if oos_days < 10:
        print(f"WARNING: only {oos_days} OOS days for {ticker}")

    # Storage for OOS results
    dynamic_y_hat = np.full((T, I), np.nan)
    static_y_hat = np.full((T, I), np.nan)
    dynamic_weights = np.full((T, I), np.nan)
    static_weights = np.full((T, I), np.nan)

    # Rolling window evaluation
    # Re-estimate every 21 days (monthly) to keep runtime reasonable
    re_estimate_interval = 21
    current_params = None
    last_fitted_model = None

    # Lambda scaling: use an initial standard fit to estimate r, then scale lambda
    effective_lambda = lambda_val
    if robust:
        # Do an initial standard fit to get r estimate for lambda scaling
        init_model = KalmanVolumeModel(bins_per_day=I, robust=False)
        init_result = init_model.fit(
            y[:oos_start], is_observed[:oos_start], max_iter=50, epsilon=1e-6
        )
        r_init = init_model.params.r
        # lambda_val is in units of k (number of std devs for outlier threshold)
        # lambda = 2 * k / sqrt(r)
        effective_lambda = 2 * lambda_val / np.sqrt(max(r_init, 1e-10))
        if verbose:
            print(f"  Lambda scaling: r_init={r_init:.4f}, "
                  f"k={lambda_val}, effective_lambda={effective_lambda:.2f}")

    t0 = time.time()

    for d in range(oos_start, T):
        # Training window
        t_start = d - train_days
        y_train = y[t_start:d]
        obs_train = is_observed[t_start:d]

        # Re-estimate parameters periodically
        if current_params is None or (d - oos_start) % re_estimate_interval == 0:
            model = KalmanVolumeModel(
                bins_per_day=I, robust=robust, lam=effective_lambda
            )
            # Warm-start from previous params if available
            if current_params is not None:
                model.params = current_params.copy()
                fit_result = model.fit(
                    y_train, obs_train, max_iter=30, epsilon=1e-6,
                    warm_start=True,
                )
            else:
                fit_result = model.fit(
                    y_train, obs_train, max_iter=100, epsilon=1e-6
                )
            current_params = model.params
            last_fitted_model = model
            if verbose and (d - oos_start) % (re_estimate_interval * 5) == 0:
                print(
                    f"  Day {d} ({dates[d]}): EM converged in "
                    f"{fit_result['iterations']} iters, "
                    f"LL={fit_result['log_likelihood_history'][-1]:.1f}, "
                    f"a_eta={current_params.a_eta:.4f}, "
                    f"a_mu={current_params.a_mu:.4f}, "
                    f"r={current_params.r:.4f}"
                )

        # Dynamic prediction: run filter on training + day d
        # Use full training window + day d for filtering
        y_ext = y[t_start:d + 1]
        obs_ext = is_observed[t_start:d + 1]
        dyn_result = last_fitted_model.predict_dynamic(y_ext, obs_ext, current_params)

        # The last day's forecasts (dynamic, one-step-ahead)
        dynamic_y_hat[d] = dyn_result["y_hat"][-1]

        # Static prediction: use state at end of day d-1
        # The filtered state at the end of training
        N_train = train_days * I
        x_last = dyn_result["x_filt"][N_train - 1]
        Sigma_last = dyn_result["Sigma_filt"][N_train - 1].reshape(2, 2)

        static_result = last_fitted_model.predict_static(
            x_last, Sigma_last, current_params
        )
        static_y_hat[d] = static_result["y_hat"]
        static_weights[d] = static_result["weights"]

        # Dynamic VWAP weights
        dyn_w = last_fitted_model.compute_vwap_dynamic(
            y[d], is_observed[d], x_last, Sigma_last, current_params
        )
        dynamic_weights[d] = dyn_w

    elapsed = time.time() - t0

    # Compute MAPE (dynamic and static)
    oos_mask = np.zeros((T, I), dtype=bool)
    oos_mask[oos_start:] = is_observed[oos_start:]

    dynamic_mape = compute_mape(y, dynamic_y_hat, oos_mask)
    static_mape = compute_mape(y, static_y_hat, oos_mask)

    # Rolling mean baseline
    rm_forecast = rolling_mean_baseline(y, is_observed, window=train_days)
    rm_mask = oos_mask & ~np.isnan(rm_forecast)
    rm_mape = compute_mape(y, rm_forecast, rm_mask)

    # VWAP tracking error (bps)
    # Actual VWAP weights = actual volume share per bin
    actual_weights = np.zeros((T, I))
    for d in range(oos_start, T):
        day_vol = raw_volume[d]
        total = np.sum(day_vol)
        if total > 0:
            actual_weights[d] = day_vol / total

    # Tracking error = sum of absolute weight differences / 2 (in bps * 10000)
    # Paper uses: TE = sum_i |w_hat_i - w_actual_i| * 10000 / 2
    # Actually, paper Eq 42 defines it differently. Let's use simple weight MAE.
    dynamic_te_per_day = np.zeros(oos_days)
    static_te_per_day = np.zeros(oos_days)
    for d_idx, d in enumerate(range(oos_start, T)):
        dynamic_te_per_day[d_idx] = np.sum(
            np.abs(dynamic_weights[d] - actual_weights[d])
        ) / 2 * 10000  # in bps
        static_te_per_day[d_idx] = np.sum(
            np.abs(static_weights[d] - actual_weights[d])
        ) / 2 * 10000

    dynamic_te = float(np.mean(dynamic_te_per_day))
    static_te = float(np.mean(static_te_per_day))

    if verbose:
        print(f"\n  Results for {ticker}:")
        print(f"    Dynamic MAPE: {dynamic_mape:.4f}")
        print(f"    Static MAPE:  {static_mape:.4f}")
        print(f"    RM MAPE:      {rm_mape:.4f}")
        print(f"    Dynamic TE:   {dynamic_te:.2f} bps")
        print(f"    Static TE:    {static_te:.2f} bps")
        print(f"    Improvement over RM: {(1 - dynamic_mape/rm_mape)*100:.1f}%")
        print(f"    Time: {elapsed:.1f}s")

    return {
        "ticker": ticker,
        "dynamic_mape": dynamic_mape,
        "static_mape": static_mape,
        "rm_mape": rm_mape,
        "improvement_pct": (1 - dynamic_mape / rm_mape) * 100 if rm_mape > 0 else 0,
        "dynamic_te_bps": dynamic_te,
        "static_te_bps": static_te,
        "params": {
            "a_eta": float(current_params.a_eta),
            "a_mu": float(current_params.a_mu),
            "sigma_eta_sq": float(current_params.sigma_eta_sq),
            "sigma_mu_sq": float(current_params.sigma_mu_sq),
            "r": float(current_params.r),
        },
        "phi": current_params.phi.tolist(),
        "elapsed_s": elapsed,
        "oos_days": oos_days,
        "dynamic_y_hat": dynamic_y_hat,
        "static_y_hat": static_y_hat,
        "dynamic_weights": dynamic_weights,
        "static_weights": static_weights,
    }


def main() -> None:
    """
    Run all tickers and print summary results.
    """
    with open(DATA_DIR / "metadata.json") as f:
        meta = json.load(f)
    tickers = meta["tickers"]

    # Run with default lambda, can tune later via cross-validation
    # Use a moderate lambda scaled by sqrt(r_init) -- we'll use a fixed value
    # and then potentially refine
    results = []
    for ticker in tickers:
        r = run_single_ticker(
            ticker,
            train_days=252,
            lambda_val=5.0,
            robust=True,
            verbose=True,
        )
        results.append(r)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Ticker':<8} {'Dyn MAPE':>10} {'Stat MAPE':>10} {'RM MAPE':>10} "
          f"{'Improv%':>8} {'Dyn TE':>8} {'Stat TE':>8}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['ticker']:<8} {r['dynamic_mape']:>10.4f} {r['static_mape']:>10.4f} "
            f"{r['rm_mape']:>10.4f} {r['improvement_pct']:>8.1f} "
            f"{r['dynamic_te_bps']:>8.2f} {r['static_te_bps']:>8.2f}"
        )
    print("-" * 80)
    avg_dyn = np.mean([r["dynamic_mape"] for r in results])
    avg_stat = np.mean([r["static_mape"] for r in results])
    avg_rm = np.mean([r["rm_mape"] for r in results])
    avg_imp = np.mean([r["improvement_pct"] for r in results])
    avg_dte = np.mean([r["dynamic_te_bps"] for r in results])
    avg_ste = np.mean([r["static_te_bps"] for r in results])
    print(
        f"{'Average':<8} {avg_dyn:>10.4f} {avg_stat:>10.4f} {avg_rm:>10.4f} "
        f"{avg_imp:>8.1f} {avg_dte:>8.2f} {avg_ste:>8.2f}"
    )

    # Save results
    results_file = Path(__file__).parent / "results.json"
    save_results = []
    for r in results:
        save_r = {k: v for k, v in r.items() if k not in (
            "dynamic_y_hat", "static_y_hat", "dynamic_weights", "static_weights"
        )}
        save_results.append(save_r)
    with open(results_file, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
