"""
evaluation_module.py
====================
EvaluationModule – computes all performance metrics for the heat-prediction
LSTM models and produces a side-by-side comparison report.

Metrics implemented (matching the 9-block diagram)
----------------------------------------------------
  computeRMSE      – Root Mean Squared Error
  computeMAE       – Mean Absolute Error
  computeAccuracy  – Binary classification accuracy
  computeF1Score   – F1-Score (harmonic mean of precision & recall)
  computePrecision – Positive predictive value
  computeRecall    – True positive rate (sensitivity)
  compareModels    – Full comparison table (baseline vs impact-centric)

Author : [Your Name]
Thesis : Enhancing Extreme Heat Prediction Using Impact-Centric Variables
         in Machine Learning Models
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# ---------------------------------------------------------------------------
# Lightweight result container
# ---------------------------------------------------------------------------

@dataclass
class ModelMetrics:
    """Stores all evaluation metrics for one model run."""
    model_name : str
    rmse       : float = 0.0
    mae        : float = 0.0
    accuracy   : float = 0.0
    f1         : float = 0.0
    precision  : float = 0.0
    recall     : float = 0.0
    auc_roc    : float = 0.0
    threshold  : float = 0.5
    extras     : Dict  = field(default_factory=dict)


# ---------------------------------------------------------------------------
# EvaluationModule
# ---------------------------------------------------------------------------

class EvaluationModule:
    """
    Centralised evaluation service for binary extreme-heat classifiers.

    All methods accept:
      y_true : Ground-truth binary labels  (np.ndarray, float or int).
      y_pred : Predicted probabilities     (np.ndarray, float in [0, 1]).
      threshold : Decision boundary for converting probabilities to labels.

    Parameters
    ----------
    threshold : Default probability threshold for converting model output
                to binary class labels (0 / 1).
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _binarise(self, y_pred: np.ndarray, threshold: Optional[float]) -> np.ndarray:
        """Convert probability array to binary predictions."""
        t = threshold if threshold is not None else self.threshold
        return (y_pred >= t).astype(int)

    # ------------------------------------------------------------------
    # Individual metric methods
    # ------------------------------------------------------------------

    def compute_rmse(
        self,
        y: np.ndarray,
        y_hat: np.ndarray,
    ) -> float:
        """
        Root Mean Squared Error between continuous predictions and labels.

        Useful for regression-style evaluation of probability outputs.

        Parameters
        ----------
        y     : True binary labels.
        y_hat : Predicted probabilities.

        Returns
        -------
        float : RMSE ∈ [0, 1].
        """
        return float(np.sqrt(mean_squared_error(y.ravel(), y_hat.ravel())))

    computeRMSE = compute_rmse   # diagram alias

    def compute_mae(
        self,
        y: np.ndarray,
        y_hat: np.ndarray,
    ) -> float:
        """
        Mean Absolute Error between predictions and true labels.

        Parameters
        ----------
        y     : True binary labels.
        y_hat : Predicted probabilities.

        Returns
        -------
        float : MAE ∈ [0, 1].
        """
        return float(mean_absolute_error(y.ravel(), y_hat.ravel()))

    computeMAE = compute_mae     # diagram alias

    def compute_accuracy(
        self,
        y: np.ndarray,
        y_hat: np.ndarray,
        threshold: Optional[float] = None,
    ) -> float:
        """
        Binary classification accuracy.

        Parameters
        ----------
        y         : True binary labels.
        y_hat     : Predicted probabilities.
        threshold : Override default decision boundary.

        Returns
        -------
        float : Accuracy ∈ [0, 1].
        """
        y_bin = self._binarise(y_hat, threshold)
        return float(accuracy_score(y.ravel().astype(int), y_bin))

    computeAccuracy = compute_accuracy   # diagram alias

    def compute_f1_score(
        self,
        y: np.ndarray,
        y_hat: np.ndarray,
        threshold: Optional[float] = None,
        average: str = "binary",
    ) -> float:
        """
        F1-Score (harmonic mean of precision and recall).

        Parameters
        ----------
        y         : True binary labels.
        y_hat     : Predicted probabilities.
        threshold : Override default decision boundary.
        average   : Sklearn averaging strategy ('binary', 'macro', etc.).

        Returns
        -------
        float : F1 ∈ [0, 1].
        """
        y_bin = self._binarise(y_hat, threshold)
        return float(
            f1_score(
                y.ravel().astype(int), y_bin,
                average=average, zero_division=0
            )
        )

    computeF1Score = compute_f1_score    # diagram alias

    def compute_precision(
        self,
        y: np.ndarray,
        y_hat: np.ndarray,
        threshold: Optional[float] = None,
    ) -> float:
        """
        Precision – fraction of positive predictions that are correct.

        Parameters
        ----------
        y         : True binary labels.
        y_hat     : Predicted probabilities.
        threshold : Override default decision boundary.

        Returns
        -------
        float : Precision ∈ [0, 1].
        """
        y_bin = self._binarise(y_hat, threshold)
        return float(
            precision_score(
                y.ravel().astype(int), y_bin,
                zero_division=0
            )
        )

    computePrecision = compute_precision   # diagram alias

    def compute_recall(
        self,
        y: np.ndarray,
        y_hat: np.ndarray,
        threshold: Optional[float] = None,
    ) -> float:
        """
        Recall (Sensitivity) – fraction of actual positives correctly identified.

        Parameters
        ----------
        y         : True binary labels.
        y_hat     : Predicted probabilities.
        threshold : Override default decision boundary.

        Returns
        -------
        float : Recall ∈ [0, 1].
        """
        y_bin = self._binarise(y_hat, threshold)
        return float(
            recall_score(
                y.ravel().astype(int), y_bin,
                zero_division=0
            )
        )

    computeRecall = compute_recall   # diagram alias

    # ------------------------------------------------------------------
    # Full evaluation bundle
    # ------------------------------------------------------------------

    def evaluate(
        self,
        y: np.ndarray,
        y_hat: np.ndarray,
        model_name: str = "model",
        threshold: Optional[float] = None,
        verbose: bool = True,
    ) -> ModelMetrics:
        """
        Compute all metrics at once and return a ModelMetrics dataclass.

        Parameters
        ----------
        y          : True binary labels.
        y_hat      : Predicted probabilities.
        model_name : Label for logging / comparison tables.
        threshold  : Override decision boundary.
        verbose    : Print a formatted report if True.

        Returns
        -------
        ModelMetrics
        """
        t = threshold if threshold is not None else self.threshold

        rmse      = self.compute_rmse(y, y_hat)
        mae       = self.compute_mae(y, y_hat)
        accuracy  = self.compute_accuracy(y, y_hat, t)
        f1        = self.compute_f1_score(y, y_hat, t)
        precision = self.compute_precision(y, y_hat, t)
        recall    = self.compute_recall(y, y_hat, t)

        # AUC-ROC (needs at least one positive and one negative class)
        try:
            auc = float(roc_auc_score(y.ravel().astype(int), y_hat.ravel()))
        except ValueError:
            auc = float("nan")

        # Confusion matrix
        y_bin = self._binarise(y_hat, t)
        cm    = confusion_matrix(y.ravel().astype(int), y_bin)

        metrics = ModelMetrics(
            model_name=model_name,
            rmse=rmse, mae=mae, accuracy=accuracy,
            f1=f1, precision=precision, recall=recall,
            auc_roc=auc, threshold=t,
            extras={"confusion_matrix": cm},
        )

        if verbose:
            self._print_report(metrics, cm, y, y_hat, t)

        return metrics

    # ------------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------------

    def compare_models(
        self,
        m1: ModelMetrics,
        m2: ModelMetrics,
    ) -> pd.DataFrame:
        """
        Produce a formatted DataFrame comparing two ModelMetrics objects.

        Parameters
        ----------
        m1 : Metrics for model 1 (e.g. baseline LSTM).
        m2 : Metrics for model 2 (e.g. impact-centric LSTM).

        Returns
        -------
        pd.DataFrame
            Side-by-side metric comparison with absolute Δ column.
        """
        compareModels = self.compare_models   # self-alias for diagram

        rows = {
            "RMSE"      : (m1.rmse,      m2.rmse),
            "MAE"       : (m1.mae,       m2.mae),
            "Accuracy"  : (m1.accuracy,  m2.accuracy),
            "F1-Score"  : (m1.f1,        m2.f1),
            "Precision" : (m1.precision, m2.precision),
            "Recall"    : (m1.recall,    m2.recall),
            "AUC-ROC"   : (m1.auc_roc,   m2.auc_roc),
        }

        records = []
        for metric, (v1, v2) in rows.items():
            delta = v2 - v1
            better = (
                "↑ Impact" if (
                    (metric in ("RMSE", "MAE") and delta < 0) or
                    (metric not in ("RMSE", "MAE") and delta > 0)
                )
                else ("↓ Baseline" if delta != 0 else "Tie")
            )
            records.append({
                "Metric"       : metric,
                m1.model_name  : f"{v1:.4f}",
                m2.model_name  : f"{v2:.4f}",
                "Δ (m2 − m1)" : f"{delta:+.4f}",
                "Better"       : better,
            })

        df = pd.DataFrame(records).set_index("Metric")

        print("\n" + "═" * 72)
        print("  MODEL COMPARISON")
        print("═" * 72)
        print(df.to_string())
        print("═" * 72 + "\n")

        return df

    compareModels = compare_models   # diagram alias

    # ------------------------------------------------------------------
    # Private printer
    # ------------------------------------------------------------------

    def _print_report(
        self,
        metrics : ModelMetrics,
        cm      : np.ndarray,
        y       : np.ndarray,
        y_hat   : np.ndarray,
        threshold: float,
    ) -> None:
        y_bin = self._binarise(y_hat, threshold)
        sep = "─" * 55
        print(f"\n{sep}")
        print(f"  Evaluation Report — {metrics.model_name}")
        print(sep)
        print(f"  Threshold  : {threshold:.2f}")
        print(f"  Samples    : {len(y):,}  (pos={int(y.sum()):,}, "
              f"neg={int((1 - y).sum()):,})")
        print(sep)
        print(f"  RMSE       : {metrics.rmse:.4f}")
        print(f"  MAE        : {metrics.mae:.4f}")
        print(f"  Accuracy   : {metrics.accuracy:.4f}")
        print(f"  Precision  : {metrics.precision:.4f}")
        print(f"  Recall     : {metrics.recall:.4f}")
        print(f"  F1-Score   : {metrics.f1:.4f}")
        print(f"  AUC-ROC    : {metrics.auc_roc:.4f}")
        print(sep)
        print("  Confusion Matrix:")
        print(f"    TN={cm[0,0]:>6}  FP={cm[0,1]:>6}")
        print(f"    FN={cm[1,0]:>6}  TP={cm[1,1]:>6}")
        print(sep)
        print("  Classification Report:")
        print(
            classification_report(
                y.ravel().astype(int), y_bin,
                target_names=["Normal", "Extreme"],
                zero_division=0,
            )
        )