"""General (multi-adsorbate expert) model inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

from src.data import ADS_COL, MOLECULAR_COLS, PRESSURE_COL, TEXTURAL_COLS


def _histogram_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("feature_")]


def build_feature_columns(
    df: pd.DataFrame,
    *,
    include_textural: bool,
    include_ads: bool,
    include_histogram: bool,
) -> list[str]:
    cols = (_histogram_columns(df) if include_histogram else []) + MOLECULAR_COLS + [PRESSURE_COL]
    if include_textural:
        cols.extend(TEXTURAL_COLS)
    if include_ads and ADS_COL in df.columns:
        cols.append(ADS_COL)
    return cols


def load_bundle_config(model_dir: Path) -> dict[str, Any]:
    cfg_path = model_dir / "model_config.json"
    if not cfg_path.is_file():
        return {
            "features": {
                "include_textural": True,
                "include_ads": True,
                "include_histogram": True,
            }
        }
    return json.loads(cfg_path.read_text(encoding="utf-8"))


class GeneralPredictor:
    """Pretrained General model (model_all) for expert loading predictions."""

    def __init__(self, model_dir: Path | str) -> None:
        self.model_dir = Path(model_dir)
        manifest = load_bundle_config(self.model_dir)
        feats = manifest.get("features", {})
        self.include_textural = bool(feats.get("include_textural", True))
        self.include_ads = bool(feats.get("include_ads", True))
        self.include_histogram = bool(feats.get("include_histogram", True))

        paths = manifest.get("artifact_paths", {})
        model_path = self.model_dir / paths.get("keras_model", "best_model_hpo.keras")
        preprocessor_path = self.model_dir / paths.get("preprocessor", "preprocessor.joblib")

        self._preprocessor = joblib.load(preprocessor_path)
        self._model = keras.models.load_model(model_path)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        feature_cols = build_feature_columns(
            df,
            include_textural=self.include_textural,
            include_ads=self.include_ads,
            include_histogram=self.include_histogram,
        )
        missing = sorted(set(feature_cols) - set(df.columns))
        if missing:
            raise ValueError(f"Input is missing columns required by the General model: {missing}")

        x = self._preprocessor.transform(df[feature_cols])
        if hasattr(x, "toarray"):
            x = x.toarray()
        x = np.asarray(x, dtype=np.float32)
        return self._model.predict(x, verbose=0).reshape(-1)

    def predict_with_column(
        self, df: pd.DataFrame, column_name: str = "prediction_loading"
    ) -> pd.DataFrame:
        out = df.copy()
        out[column_name] = self.predict(df)
        return out
