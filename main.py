"""
main.py
=======
Orchestration script for the Extreme Heat Prediction thesis experiment.

Runs a complete, reproducible pipeline:
  1.  Generate synthetic Davao City time-series data
  2.  DataPipeline → clean, sequence, and split (baseline & impact modes)
  3.  Train Baseline LSTM   (temperature, humidity, wind_speed)
  4.  Train Impact-Centric LSTM (+ UHI, wet-bulb, PM2.5, heat index)
  5.  Evaluate both models (RMSE, MAE, Accuracy, F1, Precision, Recall)
  6.  Compare models side-by-side
  7.  XAI – SHAP feature importance for the impact-centric model
  8.  AlertModule – process test predictions and dispatch alerts

Usage
-----
    python main.py [--days 365] [--epochs 30] [--seq 24] [--no-shap]

Dependencies
------------
    pip install tensorflow>=2.12 scikit-learn pandas numpy shap matplotlib

Author : [Your Name]
Thesis : Enhancing Extreme Heat Prediction Using Impact-Centric Variables
         in Machine Learning Models
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

# Project modules
from data_generator     import generate_davao_dataset, save_datasets
from entities           import HeatEvent, ImpactRecord, Severity, WeatherRecord
from data_pipeline      import DataPipeline
from lstm_model         import LSTMModel
from evaluation_module  import EvaluationModule
from xai_module         import XAIModule
from alert_module import AlertModule, DataFetchModule


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

SEED = 42
np.random.seed(SEED)
try:
    import tensorflow as tf
    tf.random.set_seed(SEED)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(text: str) -> None:
    width = 68
    print("\n" + "▓" * width)
    print(f"  {text}")
    print("▓" * width + "\n")


def _compute_class_weight(y_train: np.ndarray) -> dict:
    """Inverse-frequency class weights to address extreme-event imbalance."""
    pos = y_train.sum()
    neg = len(y_train) - pos
    if pos == 0 or neg == 0:
        return {0: 1.0, 1: 1.0}
    total = len(y_train)
    return {0: total / (2 * neg), 1: total / (2 * pos)}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_experiment(
    n_days         : int  = 365,
    seq_len        : int  = 24,
    epochs         : int  = 30,
    batch_size     : int  = 64,
    run_shap       : bool = True,
    output_dir     : str  = "outputs",
    data_dir       : str  = "data",
) -> None:
    """
    Execute the full thesis experiment.

    Parameters
    ----------
    n_days      : Simulation days for synthetic dataset.
    seq_len     : LSTM lookback window (hours).
    epochs      : Maximum training epochs per model.
    batch_size  : Mini-batch size.
    run_shap    : If False, skip SHAP computation (faster debugging).
    output_dir  : Directory for plots, reports, weights.
    data_dir    : Directory for CSV data files.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # ══════════════════════════════════════════════════════════════════
    # STEP 1 – Synthetic data generation
    # ══════════════════════════════════════════════════════════════════
    _banner("STEP 1 · Synthetic Data Generation (Davao City)")

    weather_df, impact_df = generate_davao_dataset(
        n_days=n_days, start_date="2022-01-01"
    )
    save_datasets(weather_df, impact_df, output_dir=data_dir)

    # Quick sanity check using entity classes
    sample_wr = WeatherRecord(
        timestamp   = weather_df["timestamp"].iloc[0],
        temperature = float(weather_df["temperature"].iloc[0]),
        humidity    = float(weather_df["humidity"].iloc[0]),
        wind_speed  = float(weather_df["wind_speed"].iloc[0]),
        location    = "Davao City",
    )
    assert sample_wr.validate(), "WeatherRecord validation failed!"

    sample_ir = ImpactRecord(
        timestamp     = impact_df["timestamp"].iloc[0],
        uhi_intensity = float(impact_df["uhi_intensity"].iloc[0]),
        wet_bulb_temp = float(impact_df["wet_bulb_temp"].iloc[0]),
        pm25_level    = float(impact_df["pm25_level"].iloc[0]),
        heat_index    = float(impact_df["heat_index"].iloc[0]),
    )
    print(f"Entity checks passed → {sample_wr}")
    print(f"Entity checks passed → {sample_ir}")

    extreme_rate = impact_df["is_extreme"].mean() * 100
    print(f"Class balance: {extreme_rate:.1f}% extreme heat hours")

    # ══════════════════════════════════════════════════════════════════
    # STEP 2 – DataPipeline (Baseline)
    # ══════════════════════════════════════════════════════════════════
    _banner("STEP 2 · DataPipeline — Baseline Features")

    pipeline_base = DataPipeline(sequence_length=seq_len, forecast_horizon=1)
    base_data = pipeline_base.prepare(
        weather_src = weather_df,
        impact_src  = impact_df,
        mode        = "baseline",
    )

    X_tr_b, y_tr_b = base_data["X_train"], base_data["y_train"]
    X_vl_b, y_vl_b = base_data["X_val"],   base_data["y_val"]
    X_ts_b, y_ts_b = base_data["X_test"],  base_data["y_test"]

    cw_base = _compute_class_weight(y_tr_b)
    print(f"Class weights (baseline): {cw_base}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 3 – DataPipeline (Impact-Centric)
    # ══════════════════════════════════════════════════════════════════
    _banner("STEP 3 · DataPipeline — Impact-Centric Features")

    pipeline_imp = DataPipeline(sequence_length=seq_len, forecast_horizon=1)
    imp_data = pipeline_imp.prepare(
        weather_src = weather_df,
        impact_src  = impact_df,
        mode        = "impact",
    )

    X_tr_i, y_tr_i = imp_data["X_train"], imp_data["y_train"]
    X_vl_i, y_vl_i = imp_data["X_val"],   imp_data["y_val"]
    X_ts_i, y_ts_i = imp_data["X_test"],  imp_data["y_test"]

    cw_imp = _compute_class_weight(y_tr_i)
    print(f"Class weights (impact):   {cw_imp}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 4 – Train Baseline LSTM
    # ══════════════════════════════════════════════════════════════════
    _banner("STEP 4 · Training Baseline LSTM")

    baseline_lstm = LSTMModel(
        sequence_length  = seq_len,
        forecast_horizon = 1,
        units            = 64,
        dropout_rate     = 0.3,
        learning_rate    = 1e-3,
    )
    baseline_lstm.build_baseline()
    baseline_lstm.train(
        X            = X_tr_b,
        y            = y_tr_b,
        epochs       = epochs,
        batch_size   = batch_size,
        validation_data = (X_vl_b, y_vl_b.squeeze()),
        class_weight = cw_base,
        checkpoint_path = f"{output_dir}/baseline_best.weights.h5",
    )
    baseline_lstm.save_weights(f"{output_dir}/baseline_final.weights.h5")

    # ══════════════════════════════════════════════════════════════════
    # STEP 5 – Train Impact-Centric LSTM
    # ══════════════════════════════════════════════════════════════════
    _banner("STEP 5 · Training Impact-Centric LSTM")

    impact_lstm = LSTMModel(
        sequence_length  = seq_len,
        forecast_horizon = 1,
        units            = 64,
        dropout_rate     = 0.3,
        learning_rate    = 1e-3,
    )
    impact_lstm.build_impact_centric()
    impact_lstm.train(
        X            = X_tr_i,
        y            = y_tr_i,
        epochs       = epochs,
        batch_size   = batch_size,
        validation_data = (X_vl_i, y_vl_i.squeeze()),
        class_weight = cw_imp,
        checkpoint_path = f"{output_dir}/impact_best.weights.h5",
    )
    impact_lstm.save_weights(f"{output_dir}/impact_final.weights.h5")

    # ══════════════════════════════════════════════════════════════════
    # STEP 6 – Evaluate both models
    # ══════════════════════════════════════════════════════════════════
    _banner("STEP 6 · Evaluation")

    evaluator = EvaluationModule(threshold=0.5)

    y_pred_base = baseline_lstm.predict(X_ts_b)
    y_pred_imp  = impact_lstm.predict(X_ts_i)

    metrics_base = evaluator.evaluate(
        y          = y_ts_b,
        y_hat      = y_pred_base,
        model_name = "Baseline LSTM",
        verbose    = True,
    )
    metrics_imp = evaluator.evaluate(
        y          = y_ts_i,
        y_hat      = y_pred_imp,
        model_name = "Impact-Centric LSTM",
        verbose    = True,
    )

    # ══════════════════════════════════════════════════════════════════
    # STEP 7 – Model comparison
    # ══════════════════════════════════════════════════════════════════
    _banner("STEP 7 · Model Comparison")

    comparison_df = evaluator.compare_models(metrics_base, metrics_imp)
    comparison_path = f"{output_dir}/model_comparison.csv"
    comparison_df.to_csv(comparison_path)
    print(f"Comparison table saved → '{comparison_path}'.")

    # ══════════════════════════════════════════════════════════════════
    # STEP 8 – XAI (SHAP) on impact-centric model
    # ══════════════════════════════════════════════════════════════════
    _banner("STEP 8 · Explainability (SHAP)")

    if run_shap:
        xai = XAIModule(
            model        = impact_lstm.model,
            feature_cols = imp_data["feature_cols"],
        )
        try:
            xai.compute_shap(
                X            = X_ts_i,
                n_background = 30,    # keep small for speed
                n_explain    = 50,
            )
            importance_df = xai.rank_features()
            xai.plot_importance(
                df          = importance_df,
                output_path = f"{output_dir}/shap_importance.png",
                title       = "Impact-Centric LSTM — SHAP Feature Importance",
            )
            xai.generate_report(
                model_name  = "Impact-Centric LSTM",
                output_path = f"{output_dir}/xai_report.txt",
            )
        except Exception as exc:
            print(f"[XAI] SHAP computation skipped due to error: {exc}")
            print("      Install SHAP: pip install shap")
    else:
        print("[XAI] SHAP skipped (--no-shap flag).")

    # ══════════════════════════════════════════════════════════════════
    # STEP 9 – Alert generation
    # ══════════════════════════════════════════════════════════════════
    _banner("STEP 9 · Alert Module")

    alert_module = AlertModule(
        risk_threshold = 0.5,
        log_path       = f"{output_dir}/alert_log.json",
        location       = "Davao City",
    )

    # Use merged test dataframe to get heat index values
    merged_test_slice = imp_data["merged_df"].iloc[
        -(len(y_ts_i) + seq_len) : -seq_len
    ]
    hi_test = merged_test_slice["heat_index"].values[:len(y_pred_imp)]
    ts_test = list(merged_test_slice.index[:len(y_pred_imp)])

    events = alert_module.process_predictions(
        probabilities   = y_pred_imp[:20],    # process first 20 for demo
        heat_indices    = hi_test[:20],
        timestamps      = ts_test[:20],
        notify_agencies = False,
    )

    print(f"[AlertModule] {len(events)} HeatEvent(s) generated during demo.")
    for evt in events[:3]:
        print(f"  {evt}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 10 – DataFetchModule demo
    # ══════════════════════════════════════════════════════════════════
    _banner("STEP 10 · DataFetchModule (Live Data Stub)")

    fetch_module = DataFetchModule(location="Davao City, PH")
    live_df      = fetch_module.fetch_all()
    fetch_module.store_raw_data(live_df, label="live_combined",
                                output_dir=f"{data_dir}/raw")
    print("Live fetch demo complete.")
    print(live_df.to_string(index=False))

    # ══════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════
    elapsed = time.time() - t_start
    _banner(f"EXPERIMENT COMPLETE  ({elapsed:.1f}s)")

    print(f"  Output directory : {Path(output_dir).resolve()}")
    print(f"  Data directory   : {Path(data_dir).resolve()}")
    print()
    print("  Files generated:")
    for f in sorted(Path(output_dir).iterdir()):
        print(f"    ✓  {f.name}")

    print()
    print("  Key Results:")
    print(f"    Baseline   Accuracy : {metrics_base.accuracy:.4f}   "
          f"F1 : {metrics_base.f1:.4f}")
    print(f"    Impact     Accuracy : {metrics_imp.accuracy:.4f}   "
          f"F1 : {metrics_imp.f1:.4f}")
    print()
    delta_f1 = metrics_imp.f1 - metrics_base.f1
    arrow    = "▲ improvement" if delta_f1 > 0 else "▼ regression"
    print(f"    ΔF1 (impact − baseline) : {delta_f1:+.4f}  {arrow}")
    print()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extreme Heat Prediction — Thesis Experiment Runner"
    )
    parser.add_argument("--days",    type=int,  default=365,
                        help="Simulation days (default: 365)")
    parser.add_argument("--epochs",  type=int,  default=30,
                        help="Maximum LSTM training epochs (default: 30)")
    parser.add_argument("--seq",     type=int,  default=24,
                        help="Lookback window in hours (default: 24)")
    parser.add_argument("--batch",   type=int,  default=64,
                        help="Mini-batch size (default: 64)")
    parser.add_argument("--no-shap", action="store_true",
                        help="Skip SHAP computation (faster debugging)")
    parser.add_argument("--outdir",  type=str,  default="outputs",
                        help="Output directory (default: outputs/)")
    parser.add_argument("--datadir", type=str,  default="data",
                        help="Data directory (default: data/)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_experiment(
        n_days     = args.days,
        seq_len    = args.seq,
        epochs     = args.epochs,
        batch_size = args.batch,
        run_shap   = not args.no_shap,
        output_dir = args.outdir,
        data_dir   = args.datadir,
    )