"""
lstm_model.py
=============
LSTMModel class – builds, trains, evaluates, and persists LSTM networks
for extreme heat event prediction.

Two architectures are implemented:
  • buildBaseline()      – 3 conventional meteorological features
  • buildImpactCentric() – 3 + 4 impact-centric features (UHI, wet-bulb,
                           PM2.5, heat index)

Both share the same architectural template; only the input shape differs,
allowing a clean ablation study.

Framework : TensorFlow / Keras (≥ 2.12)

Author : [Your Name]
Thesis : Enhancing Extreme Heat Prediction Using Impact-Centric Variables
         in Machine Learning Models
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np

# TensorFlow / Keras imports – suppress verbose startup messages
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf


class LSTMModel:
    """
    LSTM-based binary classifier for extreme heat event prediction.

    The class wraps a Keras functional model and exposes the interface
    described in the 9-block architecture diagram.

    Parameters
    ----------
    sequence_length  : Lookback window (time steps per sample).
    forecast_horizon : Output steps (1 = next-hour, n = multi-step).
    units            : Number of LSTM hidden units per layer.
    dropout_rate     : Dropout fraction for regularisation.
    learning_rate    : Adam optimiser learning rate.
    use_bidirectional: If True, wraps the first LSTM in Bidirectional().
    """

    # Feature counts for each model variant
    N_BASELINE_FEATURES : int = 3   # temp, humidity, wind_speed
    N_IMPACT_FEATURES   : int = 7   # above + uhi, wet_bulb, pm25, heat_index

    def __init__(
        self,
        sequence_length   : int   = 24,
        forecast_horizon  : int   = 1,
        units             : int   = 64,
        dropout_rate      : float = 0.3,
        learning_rate     : float = 1e-3,
        use_bidirectional : bool  = False,
    ) -> None:
        self.sequence_length   = sequence_length
        self.forecast_horizon  = forecast_horizon
        self.units             = units
        self.dropout_rate      = dropout_rate
        self.learning_rate     = learning_rate
        self.use_bidirectional = use_bidirectional

        self.model    : Optional[tf.keras.Model] = None
        self.history  : Optional[tf.keras.callbacks.History] = None
        self._variant : str = "uninitialised"

    # ------------------------------------------------------------------
    # Private builder
    # ------------------------------------------------------------------

    def _build(self, n_features: int, variant: str) -> tf.keras.Model:
        """
        Construct the Keras LSTM model graph.

        Architecture
        ------------
        Input(seq_len, n_features)
          └─► LSTM(units, return_sequences=True)  ← optionally Bidirectional
              └─► BatchNorm → Dropout
                  └─► LSTM(units // 2)
                      └─► BatchNorm → Dropout
                          └─► Dense(32, relu)
                              └─► Dense(forecast_horizon, sigmoid)

        Parameters
        ----------
        n_features : Number of input features.
        variant    : 'baseline' or 'impact' (for logging / checkpointing).

        Returns
        -------
        Compiled Keras Model.
        """
        inp = tf.keras.Input(shape=(self.sequence_length, n_features),
                             name="input_sequence")

        # ── Layer 1: LSTM (return sequences for stacking) ───────────
        lstm1 = tf.keras.layers.LSTM(
            self.units,
            return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            name="lstm_1",
        )
        if self.use_bidirectional:
            x = tf.keras.layers.Bidirectional(lstm1, name="bilstm_1")(inp)
        else:
            x = lstm1(inp)

        x = tf.keras.layers.BatchNormalization(name="bn_1")(x)
        x = tf.keras.layers.Dropout(self.dropout_rate, name="drop_1")(x)

        # ── Layer 2: LSTM (collapse sequence) ───────────────────────
        x = tf.keras.layers.LSTM(
            self.units // 2,
            return_sequences=False,
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            name="lstm_2",
        )(x)
        x = tf.keras.layers.BatchNormalization(name="bn_2")(x)
        x = tf.keras.layers.Dropout(self.dropout_rate, name="drop_2")(x)

        # ── Dense head ───────────────────────────────────────────────
        x = tf.keras.layers.Dense(32, activation="relu", name="dense_1")(x)

        # Output: sigmoid for binary / multi-step classification
        out = tf.keras.layers.Dense(
            self.forecast_horizon,
            activation="sigmoid",
            name="output",
        )(x)

        model = tf.keras.Model(inputs=inp, outputs=out, name=f"LSTM_{variant}")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                "accuracy",
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )

        print(f"\n[LSTMModel] Built '{variant}' model — "
              f"{model.count_params():,} parameters.")
        model.summary(print_fn=lambda s: print(f"  {s}"))
        return model

    # ------------------------------------------------------------------
    # Public builders
    # ------------------------------------------------------------------

    def build_baseline(self) -> "LSTMModel":
        """
        Initialise the **baseline** LSTM using only conventional
        meteorological features (temperature, humidity, wind_speed).

        Returns
        -------
        self (for method chaining)
        """
        self.model    = self._build(self.N_BASELINE_FEATURES, "baseline")
        self._variant = "baseline"
        return self

    # Alias matching the diagram naming convention
    buildBaseline = build_baseline

    def build_impact_centric(self) -> "LSTMModel":
        """
        Initialise the **impact-centric** LSTM using the extended feature
        set (conventional + UHI intensity, wet-bulb, PM2.5, heat index).

        Returns
        -------
        self (for method chaining)
        """
        self.model    = self._build(self.N_IMPACT_FEATURES, "impact_centric")
        self._variant = "impact_centric"
        return self

    # Alias matching the diagram naming convention
    buildImpactCentric = build_impact_centric

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X            : np.ndarray,
        y            : np.ndarray,
        epochs       : int                    = 50,
        batch_size   : int                    = 64,
        validation_data: Optional[tuple]      = None,
        class_weight : Optional[dict]         = None,
        checkpoint_path: Optional[str]        = None,
    ) -> tf.keras.callbacks.History:
        """
        Train the LSTM model with early stopping and learning-rate decay.

        Parameters
        ----------
        X               : Training sequences (n, seq_len, feats).
        y               : Binary labels (n, forecast_horizon).
        epochs          : Maximum training epochs.
        batch_size      : Mini-batch size.
        validation_data : (X_val, y_val) tuple for monitored callbacks.
        class_weight    : Dict mapping class indices to weights, e.g.
                          {0: 1.0, 1: 5.0} for imbalanced extreme events.
        checkpoint_path : If given, saves best weights to this path.

        Returns
        -------
        Keras History object with per-epoch loss/metric logs.
        """
        if self.model is None:
            raise RuntimeError(
                "Model not built. Call build_baseline() or "
                "build_impact_centric() first."
            )

        callbacks: List = [
            tf.keras.callbacks.EarlyStopping(
                monitor   = "val_loss" if validation_data else "loss",
                patience  = 10,
                restore_best_weights = True,
                verbose   = 1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor  = "val_loss" if validation_data else "loss",
                factor   = 0.5,
                patience = 5,
                min_lr   = 1e-6,
                verbose  = 1,
            ),
        ]

        if checkpoint_path:
            os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath          = checkpoint_path,
                    monitor           = "val_loss" if validation_data else "loss",
                    save_best_only    = True,
                    save_weights_only = True,
                    verbose           = 0,
                )
            )

        # Flatten y for single-step (avoids shape warnings)
        y_fit = y.squeeze() if self.forecast_horizon == 1 else y

        print(
            f"\n[LSTMModel] Training '{self._variant}' model — "
            f"{len(X):,} samples, {epochs} max epochs."
        )

        self.history = self.model.fit(
            X, y_fit,
            epochs          = epochs,
            batch_size      = batch_size,
            validation_data = validation_data,
            class_weight    = class_weight,
            callbacks       = callbacks,
            verbose         = 1,
        )
        return self.history

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate probability predictions for input sequences.

        Parameters
        ----------
        X : Input array of shape (n_samples, seq_len, n_features).

        Returns
        -------
        np.ndarray of shape (n_samples,) with probabilities in [0, 1].
        """
        if self.model is None:
            raise RuntimeError("Model is not built.")
        probs = self.model.predict(X, verbose=0).squeeze()
        return probs.astype(np.float32)

    def forecast_next(self, X: np.ndarray, n: int = 1) -> np.ndarray:
        """
        Auto-regressively forecast *n* future time steps.

        At each step the predicted probability is appended to the
        input window (sliding) and the oldest step is dropped.  This
        is a simplified recursive strategy; for production use a
        sequence-to-sequence decoder is recommended.

        Parameters
        ----------
        X : Seed sequence of shape (1, seq_len, n_features).
        n : Number of future steps to forecast.

        Returns
        -------
        np.ndarray of shape (n,) with predicted probabilities.
        """
        if self.model is None:
            raise RuntimeError("Model is not built.")

        window = X.copy()  # (1, seq_len, feats)
        preds  = []

        for _ in range(n):
            prob = float(self.model.predict(window, verbose=0).squeeze())
            preds.append(prob)

            # Slide window: drop oldest step, append new (replicate last
            # feature row, patching the target-like column with the
            # forecast probability – simplified approach)
            new_step        = window[:, -1:, :].copy()
            new_step[0, 0, 0] = prob          # overwrite first feature
            window          = np.concatenate(
                [window[:, 1:, :], new_step], axis=1
            )

        return np.array(preds, dtype=np.float32)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_weights(self, path: str) -> None:
        """
        Save model weights to an H5 file.

        Parameters
        ----------
        path : Destination file path (e.g. 'models/impact_lstm.weights.h5').
        """
        if self.model is None:
            raise RuntimeError("Nothing to save; model is not built.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.model.save_weights(path)
        print(f"[LSTMModel] Weights saved → '{path}'.")

    def load_weights(self, path: str) -> None:
        """
        Restore previously saved weights into the current model.

        Parameters
        ----------
        path : H5 file path written by save_weights().
        """
        if self.model is None:
            raise RuntimeError(
                "Build the model first before loading weights."
            )
        self.model.load_weights(path)
        print(f"[LSTMModel] Weights loaded ← '{path}'.")

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def variant(self) -> str:
        """Name of the currently built variant ('baseline' / 'impact_centric')."""
        return self._variant

    def __repr__(self) -> str:
        params = (
            f"{self.model.count_params():,}" if self.model else "not built"
        )
        return (
            f"LSTMModel(variant='{self._variant}', "
            f"seq_len={self.sequence_length}, "
            f"units={self.units}, "
            f"params={params})"
        )