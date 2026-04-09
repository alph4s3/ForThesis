"""
xai_module.py
=============
XAIModule – Explainable AI layer using SHAP (SHapley Additive exPlanations)
for the extreme-heat LSTM prediction system.

Provides feature importance, ranked attribution, and explanation reports
for both the baseline and impact-centric models.

Reference
---------
Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting
model predictions. *Advances in Neural Information Processing Systems*, 30.

Author : [Your Name]
Thesis : Enhancing Extreme Heat Prediction Using Impact-Centric Variables
         in Machine Learning Models
"""

from __future__ import annotations

import os
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class XAIModule:
    """
    SHAP-based explainability module for LSTM heat-prediction models.

    Wraps SHAP's KernelExplainer (model-agnostic) which is compatible
    with arbitrary Keras models without requiring modification.

    Parameters
    ----------
    model        : Trained Keras Model whose predict() method returns
                   probabilities of shape (n_samples,).
    feature_cols : Ordered list of feature names matching X's last axis.
    """

    def __init__(
        self,
        model,                      # Keras Model
        feature_cols: List[str],
    ) -> None:
        self.model        = model
        self.feature_cols = feature_cols
        self._explainer   = None
        self._shap_values : Optional[np.ndarray] = None
        self._X_flat      : Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Helper – flatten 3-D LSTM input for SHAP
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        """
        Collapse (n, seq_len, n_features) → (n, seq_len × n_features)
        so that SHAP's tabular explainer can operate on a 2-D matrix.
        """
        return X.reshape(len(X), -1)

    def _predict_flat(self, X_flat: np.ndarray) -> np.ndarray:
        """
        Prediction wrapper that accepts flattened input and returns
        a 1-D probability array (needed by KernelExplainer).
        """
        n, total = X_flat.shape
        seq_len  = total // len(self.feature_cols)
        X_3d     = X_flat.reshape(n, seq_len, len(self.feature_cols))
        preds    = self.model.predict(X_3d, verbose=0).ravel()
        return preds.astype(np.float64)

    # ------------------------------------------------------------------
    # SHAP computation
    # ------------------------------------------------------------------

    def compute_shap(
        self,
        X              : np.ndarray,
        n_background   : int  = 50,
        n_explain      : int  = 100,
        random_state   : int  = 42,
    ) -> np.ndarray:
        """
        Compute SHAP values using KernelExplainer.

        KernelExplainer is model-agnostic and works with any callable
        predict function, making it suitable for LSTM / RNN architectures.

        Parameters
        ----------
        X            : Input sequences (n_samples, seq_len, n_features).
        n_background : Number of background samples for SHAP baseline.
                       Larger → more accurate but slower (50–200 typical).
        n_explain    : Number of test samples to explain.
        random_state : Seed for reproducible background sampling.

        Returns
        -------
        np.ndarray of shape (n_explain, seq_len × n_features) with SHAP
        values for each input dimension.
        """
        try:
            import shap
        except ImportError:
            raise ImportError(
                "SHAP is required. Install with: pip install shap"
            )

        rng = np.random.default_rng(random_state)
        X_flat  = self._flatten(X)

        # Sample background (summary statistics reduce computation time)
        bg_idx  = rng.choice(len(X_flat), size=min(n_background, len(X_flat)),
                             replace=False)
        background = X_flat[bg_idx]

        # Explain a subset of samples
        exp_idx    = rng.choice(len(X_flat), size=min(n_explain, len(X_flat)),
                               replace=False)
        X_explain  = X_flat[exp_idx]

        print(
            f"[XAIModule] Computing SHAP values — "
            f"{len(background)} background, {len(X_explain)} explained …"
        )

        self._explainer   = shap.KernelExplainer(
            self._predict_flat, background, link="logit"
        )
        shap_values       = self._explainer.shap_values(X_explain, silent=True)
        self._shap_values = np.array(shap_values)
        self._X_flat      = X_explain

        print(f"[XAIModule] SHAP values computed. Shape: {self._shap_values.shape}")
        return self._shap_values

    computeSHAP = compute_shap   # diagram alias

    # ------------------------------------------------------------------
    # Feature ranking
    # ------------------------------------------------------------------

    def rank_features(self) -> pd.DataFrame:
        """
        Aggregate per-timestep SHAP values into per-feature importance
        scores and return a ranked DataFrame.

        Each input dimension corresponds to one feature at one time step.
        We sum absolute SHAP values across all time steps for each feature,
        then normalise to [0, 1].

        Returns
        -------
        pd.DataFrame with columns ['Feature', 'Mean |SHAP|', 'Normalised'].
        Rows sorted by importance descending.
        """
        if self._shap_values is None:
            raise RuntimeError("Run compute_shap() first.")

        shap_abs = np.abs(self._shap_values)   # (n_explain, seq × feats)
        n_feats  = len(self.feature_cols)
        seq_len  = shap_abs.shape[1] // n_feats

        # Sum across time steps for each feature
        feat_importance = np.zeros(n_feats)
        for t in range(seq_len):
            feat_importance += shap_abs[:, t * n_feats : (t + 1) * n_feats].mean(axis=0)

        feat_importance /= seq_len           # average over time steps
        norm             = feat_importance / (feat_importance.sum() + 1e-8)

        df = pd.DataFrame({
            "Feature"     : self.feature_cols,
            "Mean |SHAP|" : feat_importance.round(6),
            "Normalised"  : norm.round(4),
        }).sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)
        df.index += 1   # rank starts at 1

        print("\n[XAIModule] Feature Importance Ranking:")
        print(df.to_string())
        return df

    rankFeatures = rank_features   # diagram alias

    # ------------------------------------------------------------------
    # Plot importance (saves figure; does not display interactively)
    # ------------------------------------------------------------------

    def plot_importance(
        self,
        df         : Optional[pd.DataFrame] = None,
        output_path: str = "outputs/shap_importance.png",
        title      : str = "SHAP Feature Importance",
    ) -> None:
        """
        Save a horizontal bar chart of feature importance to disk.

        Parameters
        ----------
        df          : DataFrame from rank_features(); computed if None.
        output_path : Destination PNG path.
        title       : Plot title string.
        """
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend
        import matplotlib.pyplot as plt

        if df is None:
            df = self.rank_features()

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, max(3, len(df) * 0.55)))
        colors = plt.cm.RdYlGn_r(
            np.linspace(0.15, 0.85, len(df))
        )

        bars = ax.barh(
            df["Feature"][::-1],
            df["Mean |SHAP|"][::-1],
            color=colors,
            edgecolor="white",
            linewidth=0.5,
        )

        for bar, val in zip(bars, df["Mean |SHAP|"][::-1]):
            ax.text(
                bar.get_width() + 0.0001,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center", ha="left", fontsize=9,
            )

        ax.set_xlabel("Mean |SHAP value|", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0, df["Mean |SHAP|"].max() * 1.2)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[XAIModule] Importance plot saved → '{output_path}'.")

    plotImportance = plot_importance   # diagram alias

    # ------------------------------------------------------------------
    # Explain a single prediction
    # ------------------------------------------------------------------

    def explain_prediction(self, x: np.ndarray) -> pd.DataFrame:
        """
        Explain one individual prediction by computing SHAP values for
        a single sample and pairing them with feature names.

        Parameters
        ----------
        x : Single input sequence of shape (1, seq_len, n_features)
            OR (seq_len, n_features).

        Returns
        -------
        pd.DataFrame with columns
            ['Feature', 'Average Value', 'SHAP Value', 'Direction'].
        """
        if self._explainer is None:
            raise RuntimeError(
                "Run compute_shap() before explain_prediction()."
            )

        if x.ndim == 2:
            x = x[np.newaxis, ...]    # add batch dimension

        x_flat = self._flatten(x)
        sv     = self._explainer.shap_values(x_flat, silent=True).ravel()

        n_feats = len(self.feature_cols)
        seq_len = len(sv) // n_feats

        # Average SHAP and value per feature over time steps
        feat_shap = np.zeros(n_feats)
        feat_val  = np.zeros(n_feats)
        for t in range(seq_len):
            feat_shap += sv[t * n_feats : (t + 1) * n_feats]
            feat_val  += x_flat[0, t * n_feats : (t + 1) * n_feats]
        feat_shap /= seq_len
        feat_val  /= seq_len

        df = pd.DataFrame({
            "Feature"        : self.feature_cols,
            "Average Value"  : feat_val.round(4),
            "SHAP Value"     : feat_shap.round(6),
            "Direction"      : ["↑ Risk" if s > 0 else "↓ Risk"
                                for s in feat_shap],
        }).sort_values("SHAP Value", key=np.abs, ascending=False)

        return df

    explainPrediction = explain_prediction   # diagram alias

    # ------------------------------------------------------------------
    # Generate text report
    # ------------------------------------------------------------------

    def generate_report(
        self,
        model_name  : str = "LSTM Model",
        output_path : str = "outputs/xai_report.txt",
    ) -> str:
        """
        Write a plain-text XAI summary report to disk.

        Parameters
        ----------
        model_name  : Model label for the report header.
        output_path : Destination text file.

        Returns
        -------
        str : The report content as a string.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        df = self.rank_features()

        lines = [
            "=" * 60,
            f"  XAI REPORT — {model_name}",
            "=" * 60,
            "",
            "Feature Importance (SHAP KernelExplainer)",
            "-" * 60,
            df.to_string(),
            "",
            "Interpretation Guide",
            "-" * 60,
            "• Mean |SHAP| : average magnitude of feature contribution.",
            "  Higher = more influential in the model's predictions.",
            "• Normalised  : fractional share of total importance.",
            "",
            "Key Findings",
            "-" * 60,
        ]

        top3 = df.head(3)
        for rank, row in top3.iterrows():
            lines.append(
                f"  #{rank}  {row['Feature']:20s}  "
                f"SHAP={row['Mean |SHAP|']:.4f}  "
                f"({row['Normalised']*100:.1f}% of total importance)"
            )

        lines += ["", "=" * 60]
        report = "\n".join(lines)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"[XAIModule] Report saved → '{output_path}'.")
        return report

    generateReport = generate_report   # diagram alias