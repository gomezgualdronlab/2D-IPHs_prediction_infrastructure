"""Single-Feature Stacking (SFS) downstream model inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

from inference.general import GeneralPredictor
from src.data import ADS_COL, DROP_DESCRIPTOR_COLS, EXPERT_COL, ID_COLS, TARGET_COL


def _load_config(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "model_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing model config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


class SFSPredictor:
    """SFS model: General expert prediction + downstream MLP correction."""

    def __init__(
        self,
        model_dir: Path | str,
        *,
        general_model_dir: Path | str | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.config = _load_config(self.model_dir)

        if general_model_dir is None:
            rel = self.config.get("general_model_dir", "../general")
            general_dir = (self.model_dir / rel).resolve()
        else:
            general_dir = Path(general_model_dir).resolve()
        self._general = GeneralPredictor(general_dir)

        paths = self.config.get("artifact_paths", {})
        model_path = self.model_dir / paths.get("keras_model", "best_model_hpo.keras")
        preprocessor_path = self.model_dir / paths.get("preprocessor", "preprocessor.joblib")

        self._preprocessor = joblib.load(preprocessor_path)
        self._model = keras.models.load_model(model_path)

        policy = self.config.get("feature_policy", {})
        self._feature_cols: list[str] = list(self.config.get("features_after_drop", []))
        if not self._feature_cols:
            raise ValueError(f"features_after_drop is empty in {self.model_dir / 'model_config.json'}")

        self._expert_col = policy.get("added_stacked_feature", EXPERT_COL)
        self._drop_descriptors = list(policy.get("dropped_molecule_descriptors", DROP_DESCRIPTOR_COLS))
        self._drop_ids = list(policy.get("dropped_id_columns", ID_COLS))
        self._drop_ads = bool(policy.get("dropped_ads_column", True))
        self._prediction_column = self.config.get("prediction_column", "prediction_loading")

    def _validate_inputs(self, df: pd.DataFrame) -> None:
        required = set(self._drop_descriptors) | set(self._feature_cols) - {self._expert_col}
        required.discard(TARGET_COL)
        if not self._drop_ads:
            required.add(ADS_COL)
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"Input is missing columns required for SFS prediction: {missing}")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        self._validate_inputs(df)
        stacked = self._general.predict_with_column(df, column_name=self._expert_col)

        x = stacked[self._feature_cols]
        if not np.isfinite(x.to_numpy(dtype=np.float64)).all():
            raise ValueError("Non-finite values in SFS feature matrix.")

        x_scaled = self._preprocessor.transform(x).astype(np.float32)
        return self._model.predict(x_scaled, verbose=0).reshape(-1)

    def predict_with_column(
        self, df: pd.DataFrame, column_name: str | None = None
    ) -> pd.DataFrame:
        col = column_name or self._prediction_column
        out = df.copy()
        out[col] = self.predict(df)
        return out
