"""
Evaluation script for the Dual-Mode Intraday Volume Forecast model.

Trains the model on a subset of stocks, evaluates on an out-of-sample
period, and produces validation metrics including MAPE comparisons,
sanity checks, and Model B (percentage) evaluation.

Functions
---------
run_evaluation
    Run full evaluation pipeline for all tickers.
evaluate_single_stock
    Train and evaluate a single stock.
"""

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

from model import (
    I,
    FALLBACK,
    ModelParams,
    assign_regime,
    compute_baseline_mape,
    compute_evaluation_mape,
    compute_historical_average,
    compute_seasonal_factors,
    forecast_raw_volume,
    forecast_volume_percentage,
    load_volume_matrix,
    train_full_model,
)

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "data", "direction_4", "prepared"
)


def evaluate_single_stock(
    ticker: str,
    volume_matrix: np.ndarray,
    params: ModelParams,
    train_end_idx: int,
    eval_day_indices: list,
) -> dict:
    """
    Train and evaluate a single stock.

    Parameters
    ----------
    ticker:          Stock ticker symbol.
    volume_matrix:   2D array (n_days, I) of daily bin volumes.
    params:          Model parameters.
    train_end_idx:   Index of last training day (exclusive).
    eval_day_indices: List of day indices for evaluation.

    Returns
    -------
    Dict with evaluation metrics for this stock.
    """
    t0 = time.time()

    # Train
    model_a, pct_model = train_full_model(volume_matrix, train_end_idx, params)
    train_time = time.time() - t0

    # Evaluate Model A: full model MAPE
    model_mape = compute_evaluation_mape(
        model_a, volume_matrix, eval_day_indices, params.min_volume_floor
    )

    # Evaluate baseline: historical average only
    baseline_mape = compute_baseline_mape(
        volume_matrix, eval_day_indices, params.N_hist, params.min_volume_floor
    )

    # MAPE reduction
    if baseline_mape > 0 and not np.isinf(baseline_mape):
        mape_reduction = (baseline_mape - model_mape) / baseline_mape * 100
    else:
        mape_reduction = 0.0

    # Evaluate per-bin MAPE for Model A
    per_bin_model_mape = np.zeros(I)
    per_bin_baseline_mape = np.zeros(I)
    per_bin_count = np.zeros(I)

    for d_idx in eval_day_indices:
        # Pre-compute hist avg for this day (baseline)
        h_start = max(0, d_idx - params.N_hist)
        if h_start < d_idx:
            hist_avg = np.mean(volume_matrix[h_start:d_idx], axis=0)
        else:
            hist_avg = volume_matrix[d_idx]

        for i in range(I):
            actual = volume_matrix[d_idx, i]
            if actual < params.min_volume_floor:
                continue

            observed = {j + 1: volume_matrix[d_idx, j] for j in range(i)}
            predicted = forecast_raw_volume(
                model_a, volume_matrix, d_idx, i + 1, observed
            )
            per_bin_model_mape[i] += abs(predicted - actual) / actual
            per_bin_baseline_mape[i] += abs(hist_avg[i] - actual) / actual
            per_bin_count[i] += 1

    mask = per_bin_count > 0
    per_bin_model_mape[mask] /= per_bin_count[mask]
    per_bin_baseline_mape[mask] /= per_bin_count[mask]

    # Evaluate Model B: Mean Absolute Deviation of percentage forecasts
    pct_mad_model = []
    pct_mad_baseline = []

    for d_idx in eval_day_indices:
        daily_total = np.sum(volume_matrix[d_idx])
        if daily_total < params.min_volume_floor:
            continue

        for current_bin in range(1, I):
            next_bin = current_bin + 1
            actual_pct = volume_matrix[d_idx, next_bin - 1] / daily_total

            observed = {j + 1: volume_matrix[d_idx, j] for j in range(current_bin)}

            pct_hat = forecast_volume_percentage(
                model_a, pct_model, volume_matrix, d_idx, current_bin,
                observed, params.max_deviation, params.pct_switchoff,
                params.min_volume_floor,
            )
            pct_mad_model.append(abs(pct_hat - actual_pct))

            # Baseline: static hist_pct scaled
            observed_total = sum(observed.values())
            observed_hist_frac = np.sum(pct_model.hist_pct[:current_bin])
            remaining_hist_frac = 1.0 - observed_hist_frac
            if daily_total > 0:
                actual_remaining = 1.0 - observed_total / daily_total
            else:
                actual_remaining = 1.0
            if remaining_hist_frac > 0:
                scale = actual_remaining / remaining_hist_frac
            else:
                scale = 1.0
            baseline_pct = scale * pct_model.hist_pct[next_bin - 1]
            pct_mad_baseline.append(abs(baseline_pct - actual_pct))

    mean_mad_model = np.mean(pct_mad_model) if pct_mad_model else np.nan
    mean_mad_baseline = np.mean(pct_mad_baseline) if pct_mad_baseline else np.nan
    if mean_mad_baseline > 0 and not np.isnan(mean_mad_baseline):
        mad_reduction = (mean_mad_baseline - mean_mad_model) / mean_mad_baseline * 100
    else:
        mad_reduction = 0.0

    # Sanity checks
    sf = model_a.seasonal_factors
    ha = model_a.hist_avg

    # Check 1: U-shape
    midday_avg = np.mean(sf[8:18])  # bins 9-18 (midday)
    edge_avg = (sf[0] + sf[-1]) / 2  # first and last bins
    u_shape = edge_avg > midday_avg

    # Check 2: hist_avg / seasonal ratio
    ratio = ha / sf
    ratio_ok = np.all((ratio > 0.3) & (ratio < 3.0))

    # Check 4: ARMA parsimony
    arma_orders = []
    for m in model_a.interday_models:
        if m is not FALLBACK:
            arma_orders.append(m.model.k_ar + m.model.k_ma)
    median_order = np.median(arma_orders) if arma_orders else 0
    fallback_count = sum(1 for m in model_a.interday_models if m is FALLBACK)

    # Check 5: Weight non-negativity
    weights_nonneg = all(
        np.all(w >= 0) for w in model_a.weights.values()
    )

    # Check 10: Surprise regression coefficients
    beta_bounded = np.all(np.abs(pct_model.beta) < 1.0)

    results = {
        "ticker": ticker,
        "model_mape": model_mape,
        "baseline_mape": baseline_mape,
        "mape_reduction_pct": mape_reduction,
        "pct_mad_model": mean_mad_model,
        "pct_mad_baseline": mean_mad_baseline,
        "mad_reduction_pct": mad_reduction,
        "per_bin_model_mape": per_bin_model_mape.tolist(),
        "per_bin_baseline_mape": per_bin_baseline_mape.tolist(),
        "u_shape": u_shape,
        "ratio_ok": ratio_ok,
        "median_arma_order": float(median_order),
        "fallback_count": fallback_count,
        "weights_nonneg": weights_nonneg,
        "beta_bounded": beta_bounded,
        "beta_values": pct_model.beta.tolist(),
        "L_selected": pct_model.L,
        "n_regimes": model_a.regime_classifier.n_regimes,
        "weights": {str(k): v.tolist() for k, v in model_a.weights.items()},
        "train_time_s": train_time,
        "n_eval_days": len(eval_day_indices),
    }
    return results


def run_evaluation(
    data_dir: str = DATA_DIR,
    max_tickers: int = 10,
    output_path: str = None,
) -> dict:
    """
    Run full evaluation pipeline for selected tickers.

    Parameters
    ----------
    data_dir:    Path to prepared data directory.
    max_tickers: Maximum number of tickers to evaluate.
    output_path: Path to save JSON results (optional).

    Returns
    -------
    Dict with per-ticker and aggregate results.
    """
    params = ModelParams()

    volume_data, dates_data, tickers = load_volume_matrix(data_dir)

    # Select tickers: use the most liquid ones
    daily_stats = pd.read_parquet(os.path.join(data_dir, "daily_stats.parquet"))
    avg_dv = daily_stats.groupby("ticker")["dollar_volume"].mean().sort_values(ascending=False)
    selected = [t for t in avg_dv.index if t in volume_data][:max_tickers]

    print(f"Evaluating {len(selected)} tickers: {selected}")

    # Train/test split: use last 63 days as OOS, train on everything before
    all_results = []
    for ticker in selected:
        vm = volume_data[ticker]
        n_days = vm.shape[0]

        # Need at least N_seasonal + N_weight_train + 21 (val) + 63 (OOS)
        min_required = params.N_seasonal + params.N_weight_train + 21 + 63
        if n_days < min_required:
            print(f"  {ticker}: insufficient data ({n_days} < {min_required}), skipping")
            continue

        oos_size = 63
        train_end = n_days - oos_size
        eval_days = list(range(train_end, n_days))

        print(f"  {ticker}: training on {train_end} days, evaluating on {oos_size} days...")
        try:
            result = evaluate_single_stock(
                ticker, vm, params, train_end, eval_days
            )
            all_results.append(result)
            print(f"    Model MAPE: {result['model_mape']:.4f}, "
                  f"Baseline MAPE: {result['baseline_mape']:.4f}, "
                  f"Reduction: {result['mape_reduction_pct']:.1f}%, "
                  f"Time: {result['train_time_s']:.1f}s")
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_results:
        print("No tickers evaluated successfully.")
        return {}

    # Aggregate metrics
    model_mapes = [r["model_mape"] for r in all_results]
    baseline_mapes = [r["baseline_mape"] for r in all_results]
    reductions = [r["mape_reduction_pct"] for r in all_results]
    mad_models = [r["pct_mad_model"] for r in all_results if not np.isnan(r["pct_mad_model"])]
    mad_baselines = [r["pct_mad_baseline"] for r in all_results if not np.isnan(r["pct_mad_baseline"])]
    mad_reductions = [r["mad_reduction_pct"] for r in all_results]

    aggregate = {
        "n_tickers": len(all_results),
        "median_model_mape": float(np.median(model_mapes)),
        "median_baseline_mape": float(np.median(baseline_mapes)),
        "median_mape_reduction_pct": float(np.median(reductions)),
        "mean_mape_reduction_pct": float(np.mean(reductions)),
        "mean_pct_mad_model": float(np.mean(mad_models)) if mad_models else None,
        "mean_pct_mad_baseline": float(np.mean(mad_baselines)) if mad_baselines else None,
        "mean_mad_reduction_pct": float(np.mean(mad_reductions)),
    }

    output = {
        "params": {
            "N_seasonal": params.N_seasonal,
            "N_hist": params.N_hist,
            "N_interday_fit": params.N_interday_fit,
            "N_intraday_fit": params.N_intraday_fit,
            "N_weight_train": params.N_weight_train,
            "N_regression_fit": params.N_regression_fit,
            "L_max": params.L_max,
            "max_deviation": params.max_deviation,
            "pct_switchoff": params.pct_switchoff,
        },
        "aggregate": aggregate,
        "per_ticker": all_results,
    }

    print("\n=== Aggregate Results ===")
    print(f"Tickers evaluated: {aggregate['n_tickers']}")
    print(f"Median Model MAPE: {aggregate['median_model_mape']:.4f}")
    print(f"Median Baseline MAPE: {aggregate['median_baseline_mape']:.4f}")
    print(f"Median MAPE Reduction: {aggregate['median_mape_reduction_pct']:.1f}%")
    print(f"Mean MAPE Reduction: {aggregate['mean_mape_reduction_pct']:.1f}%")
    if aggregate["mean_pct_mad_model"] is not None:
        print(f"Mean Pct MAD (Model): {aggregate['mean_pct_mad_model']:.5f}")
        print(f"Mean Pct MAD (Baseline): {aggregate['mean_pct_mad_baseline']:.5f}")
        print(f"Mean MAD Reduction: {aggregate['mean_mad_reduction_pct']:.1f}%")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")

    return output


if __name__ == "__main__":
    output_path = os.path.join(
        os.path.dirname(__file__), "..", "evaluation_results.json"
    )
    run_evaluation(output_path=output_path, max_tickers=10)
