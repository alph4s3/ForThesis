"""
data_pipeline.py
================
DataPipeline class – loading, merging, preprocessing, and sequencing
time-series data for the Extreme Heat Prediction LSTM models.

Design follows the 9-block architecture diagram:
  WeatherRecord ──► DataPipeline ──► LSTMModel

Author : [Your Name]
Thesis : Enhancing Extreme Heat Prediction Using Impact-Centric Variables
         in Machine Learning Models
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# DataPipeline
# ---------------------------------------------------------------------------

class DataPipeline:
    """
    End-to-end preprocessing pipeline for the heat-prediction system.

    Responsibilities
    ----------------
    1. Load raw weather and impact CSV / DataFrame sources.
    2. Merge datasets on a shared timestamp index.
    3. Impute or interpolate missing values.
    4. Create overlapping sliding-window sequences for LSTM ingestion.
    5. Perform chronological train / validation / test split.
    6. Fit and apply Min-Max feature scaling (scaler fitted on train only).

    Parameters
    ----------
    sequence_length : Number of consecutive time steps fed into the LSTM
                      as one input sample (lookback window).
    forecast_horizon : Number of future steps to predict (multi-step).
    target_col      : Column name of the prediction target.
    """

    # Baseline feature set (conventional meteorology)
    BASELINE_FEATURES: list[str] = [
        "temperature", "humidity", "wind_speed"
    ]

    # Impact-centric feature set (adds UHI, wet-bulb, PM2.5, heat index)
    IMPACT_FEATURES: list[str] = [
        "temperature", "humidity", "wind_speed",
        "uhi_intensity", "wet_bulb_temp", "pm25_level", "heat_index",
    ]

    def __init__(
        self,
        sequence_length  : int = 24,    # 24-hour lookback
        forecast_horizon : int = 1,     # predict next hour
        target_col       : str = "is_extreme",
    ) -> None:
        self.sequence_length  = sequence_length
        self.forecast_horizon = forecast_horizon
        self.target_col       = target_col

        self._merged_df   : Optional[pd.DataFrame] = None
        self._scaler_base : MinMaxScaler = MinMaxScaler()
        self._scaler_imp  : MinMaxScaler = MinMaxScaler()

    # ------------------------------------------------------------------
    # 1 · Loading
    # ------------------------------------------------------------------

    def load_weather_data(self, src: str | pd.DataFrame) -> pd.DataFrame:
        """
        Load weather records from a CSV file path or an existing DataFrame.

        Parameters
        ----------
        src : File path string or pre-loaded DataFrame.

        Returns
        -------
        pd.DataFrame
            Weather records with 'timestamp' parsed as datetime index.
        """
        if isinstance(src, (str, Path)):
            df = pd.read_csv(src, parse_dates=["timestamp"])
            print(f"[DataPipeline] Loaded weather data from '{src}' "
                  f"({len(df):,} rows).")
        else:
            df = src.copy()
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            print(f"[DataPipeline] Received weather DataFrame ({len(df):,} rows).")

        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def load_impact_data(self, src: str | pd.DataFrame) -> pd.DataFrame:
        """
        Load impact-centric records from a CSV file path or DataFrame.

        Parameters
        ----------
        src : File path string or pre-loaded DataFrame.

        Returns
        -------
        pd.DataFrame
            Impact records with 'timestamp' parsed as datetime.
        """
        if isinstance(src, (str, Path)):
            df = pd.read_csv(src, parse_dates=["timestamp"])
            print(f"[DataPipeline] Loaded impact data from '{src}' "
                  f"({len(df):,} rows).")
        else:
            df = src.copy()
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            print(f"[DataPipeline] Received impact DataFrame ({len(df):,} rows).")

        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # 2 · Merging
    # ------------------------------------------------------------------

    def merge_datasets(
        self,
        weather_df : pd.DataFrame,
        impact_df  : pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Inner-join weather and impact DataFrames on the 'timestamp' column.

        Parameters
        ----------
        weather_df : Output of load_weather_data().
        impact_df  : Output of load_impact_data().

        Returns
        -------
        pd.DataFrame
            Merged DataFrame indexed by timestamp, containing all columns
            from both sources (duplicates resolved by suffixing '_imp').
        """
        merged = pd.merge(
            weather_df, impact_df,
            on="timestamp", how="inner", suffixes=("", "_imp")
        )
        merged = merged.set_index("timestamp").sort_index()

        # Drop duplicate timestamp-based columns produced by the join
        dup_cols = [c for c in merged.columns if c.endswith("_imp")]
        merged = merged.drop(columns=dup_cols, errors="ignore")

        print(
            f"[DataPipeline] Merged dataset: {len(merged):,} rows × "
            f"{len(merged.columns)} columns."
        )
        self._merged_df = merged
        return merged

    # ------------------------------------------------------------------
    # 3 · Missing-value handling
    # ------------------------------------------------------------------

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values using time-series-appropriate strategies:
          - Numeric columns  → linear interpolation (time-aware), then
                               forward-fill any leading NaNs.
          - Categorical cols → forward-fill.

        Parameters
        ----------
        df : Merged DataFrame (may contain NaNs).

        Returns
        -------
        pd.DataFrame
            DataFrame with no remaining NaN values.
        """
        before = df.isna().sum().sum()

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        # Linear interpolation for numeric features
        df[num_cols] = (
            df[num_cols]
            .interpolate(method="linear", limit_direction="both")
        )

        # Forward-fill categorical
        if cat_cols:
            df[cat_cols] = df[cat_cols].ffill()

        # Final safety fill (e.g., leading NaNs after interpolation)
        df = df.ffill().bfill()

        after = df.isna().sum().sum()
        print(
            f"[DataPipeline] Missing values: {before} → {after} "
            f"(after interpolation + forward-fill)."
        )
        return df

    # ------------------------------------------------------------------
    # 4 · Sequence creation
    # ------------------------------------------------------------------

    def create_sequences(
        self,
        df           : pd.DataFrame,
        feature_cols : list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Construct overlapping sliding-window samples for LSTM training.

        For each time step *t*, the input window is
        ``X[t] = features[t - seq_len : t]``  and the label is
        ``y[t] = target[t : t + horizon]``.

        Parameters
        ----------
        df           : Preprocessed, merged DataFrame.
        feature_cols : List of column names to use as LSTM features.

        Returns
        -------
        X : np.ndarray of shape
            (n_samples, sequence_length, n_features).
        y : np.ndarray of shape
            (n_samples, forecast_horizon).
        """
        data   = df[feature_cols].values.astype(np.float32)
        target = df[self.target_col].values.astype(np.float32)

        X_list, y_list = [], []
        end = len(data) - self.forecast_horizon

        for i in range(self.sequence_length, end + 1):
            X_list.append(data[i - self.sequence_length : i])
            y_list.append(target[i : i + self.forecast_horizon])

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        print(
            f"[DataPipeline] Created sequences: "
            f"X={X.shape}, y={y.shape} "
            f"(seq_len={self.sequence_length}, "
            f"horizon={self.forecast_horizon})."
        )
        return X, y

    # ------------------------------------------------------------------
    # 5 · Chronological train / validation / test split
    # ------------------------------------------------------------------

    def train_test_split(
        self,
        X     : np.ndarray,
        y     : np.ndarray,
        ratio : tuple[float, float, float] = (0.7, 0.15, 0.15),
    ) -> tuple[
        np.ndarray, np.ndarray,
        np.ndarray, np.ndarray,
        np.ndarray, np.ndarray,
    ]:
        """
        Chronological (non-random) train / validation / test split to
        prevent look-ahead bias in time-series evaluation.

        Parameters
        ----------
        X     : Feature array (n_samples, seq_len, n_features).
        y     : Label array   (n_samples, horizon).
        ratio : (train_frac, val_frac, test_frac) — must sum to 1.

        Returns
        -------
        X_train, y_train, X_val, y_val, X_test, y_test
        """
        assert abs(sum(ratio) - 1.0) < 1e-6, "Ratios must sum to 1."
        n   = len(X)
        i1  = int(n * ratio[0])
        i2  = int(n * (ratio[0] + ratio[1]))

        X_train, y_train = X[:i1],    y[:i1]
        X_val,   y_val   = X[i1:i2],  y[i1:i2]
        X_test,  y_test  = X[i2:],    y[i2:]

        print(
            f"[DataPipeline] Split → "
            f"train={len(X_train):,}, "
            f"val={len(X_val):,}, "
            f"test={len(X_test):,}."
        )
        return X_train, y_train, X_val, y_val, X_test, y_test

    # ------------------------------------------------------------------
    # 6 · Feature scaling
    # ------------------------------------------------------------------

    def scale_features(
        self,
        X_train : np.ndarray,
        X_val   : np.ndarray,
        X_test  : np.ndarray,
        mode    : str = "baseline",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply Min-Max scaling per feature.

        The scaler is **fitted on the training set only** to prevent
        data leakage into validation and test sets.

        Parameters
        ----------
        X_train : Training features (n, seq, feats).
        X_val   : Validation features.
        X_test  : Test features.
        mode    : 'baseline' or 'impact' — selects which scaler to use.

        Returns
        -------
        Scaled X_train, X_val, X_test.
        """
        scaler = self._scaler_base if mode == "baseline" else self._scaler_imp

        n_train, seq, feats = X_train.shape

        # Flatten to 2-D for sklearn, then reshape back
        def _fit_transform(arr: np.ndarray) -> np.ndarray:
            flat = arr.reshape(-1, feats)
            return scaler.fit_transform(flat).reshape(arr.shape)

        def _transform(arr: np.ndarray) -> np.ndarray:
            flat = arr.reshape(-1, feats)
            return scaler.transform(flat).reshape(arr.shape)

        X_tr_s = _fit_transform(X_train)
        X_vl_s = _transform(X_val)
        X_ts_s = _transform(X_test)

        print(f"[DataPipeline] Features scaled (mode='{mode}').")
        return X_tr_s, X_vl_s, X_ts_s

    # ------------------------------------------------------------------
    # High-level convenience method
    # ------------------------------------------------------------------

    def prepare(
        self,
        weather_src  : str | pd.DataFrame,
        impact_src   : str | pd.DataFrame,
        mode         : str = "baseline",
        split_ratio  : tuple[float, float, float] = (0.70, 0.15, 0.15),
    ) -> dict:
        """
        Full pipeline: load → merge → impute → sequence → split → scale.

        Parameters
        ----------
        weather_src : Path or DataFrame for weather records.
        impact_src  : Path or DataFrame for impact records.
        mode        : 'baseline' uses conventional features only;
                      'impact' adds UHI, wet-bulb, PM2.5, heat index.
        split_ratio : (train, val, test) fractions.

        Returns
        -------
        dict with keys:
            X_train, y_train, X_val, y_val, X_test, y_test,
            feature_cols, merged_df
        """
        feature_cols = (
            self.BASELINE_FEATURES if mode == "baseline"
            else self.IMPACT_FEATURES
        )

        weather_df = self.load_weather_data(weather_src)
        impact_df  = self.load_impact_data(impact_src)
        merged     = self.merge_datasets(weather_df, impact_df)
        merged     = self.handle_missing_values(merged)

        X, y = self.create_sequences(merged, feature_cols)
        X_tr, y_tr, X_vl, y_vl, X_ts, y_ts = self.train_test_split(
            X, y, ratio=split_ratio
        )
        X_tr, X_vl, X_ts = self.scale_features(
            X_tr, X_vl, X_ts, mode=mode
        )

        print(
            f"[DataPipeline] Pipeline complete (mode='{mode}'). "
            f"Features: {feature_cols}"
        )

        return {
            "X_train"      : X_tr,
            "y_train"      : y_tr,
            "X_val"        : X_vl,
            "y_val"        : y_vl,
            "X_test"       : X_ts,
            "y_test"       : y_ts,
            "feature_cols" : feature_cols,
            "merged_df"    : merged,
        }