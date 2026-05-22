"""Scratch (from-scratch) MLP model inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

from src.data import PRESSURE_COL, TARGET_COL, TEXTURAL_COLS


def _histogram_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("feature_")]


def _load_config(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "model_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing model config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


class ScratchPredictor:
    """Molecule-specific scratch MLP trained on histogram + textural + fugacity."""

    def __init__(self, model_dir: Path | str) -> None:
        self.model_dir = Path(model_dir)
        self.config = _load_config(self.model_dir)

        paths = self.config.get("artifact_paths", {})
        model_path = self.model_dir / paths.get("keras_model", "best_model_hpo.keras")
        preprocessor_path = self.model_dir / paths.get("preprocessor", "preprocessor.joblib")

        self._preprocessor = joblib.load(preprocessor_path)
        self._model = keras.models.load_model(model_path)
        self._feature_cols: list[str] = list(self.config.get("feature_columns", []))
        if not self._feature_cols:
            raise ValueError(f"feature_columns is empty in {self.model_dir / 'model_config.json'}")
        self._prediction_column = self.config.get("prediction_column", "prediction_loading")

    def _validate_inputs(self, df: pd.DataFrame) -> None:
        missing = sorted(set(self._feature_cols) - set(df.columns))
        if missing:
            raise ValueError(f"Input is missing columns required for scratch prediction: {missing}")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        self._validate_inputs(df)
        x = self._preprocessor.transform(df[self._feature_cols])
        if hasattr(x, "toarray"):
            x = x.toarray()
        x = np.asarray(x, dtype=np.float32)
        if not np.isfinite(x).all():
            raise ValueError("Non-finite values in scratch feature matrix.")
        return self._model.predict(x, verbose=0).reshape(-1)

    def predict_with_column(
        self, df: pd.DataFrame, column_name: str | None = None
    ) -> pd.DataFrame:
        col = column_name or self._prediction_column
        out = df.copy()
        out[col] = self.predict(df)
        return out

    @staticmethod
    def default_feature_columns(df: pd.DataFrame) -> list[str]:
        hist = _histogram_columns(df)
        if not hist:
            raise ValueError("No feature_* columns found in input.")
        return hist + [PRESSURE_COL] + TEXTURAL_COLS
