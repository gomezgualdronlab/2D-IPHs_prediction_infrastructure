#!/usr/bin/env python3
"""CLI: predict MOF gas loading from a CSV using bundled models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from inference.general import GeneralPredictor  # noqa: E402
from inference.scratch import ScratchPredictor  # noqa: E402
from inference.sfs import SFSPredictor  # noqa: E402

MODELS = {
    "general": {
        "description": "Pretrained General (multi-adsorbate expert) model",
        "class": "general",
        "default_dir": _ROOT / "models" / "general",
    },
    "co2_sfs": {
        "description": "CO2 Single-Feature Stacking model (expert + downstream MLP)",
        "class": "sfs",
        "default_dir": _ROOT / "models" / "co2_sfs",
    },
    "propane_sfs": {
        "description": "Propane Single-Feature Stacking model (expert + downstream MLP)",
        "class": "sfs",
        "default_dir": _ROOT / "models" / "propane_sfs",
    },
    "co2_scratch": {
        "description": "CO2 scratch MLP (histogram + textural + fugacity)",
        "class": "scratch",
        "default_dir": _ROOT / "models" / "co2_scratch",
    },
    "propane_scratch": {
        "description": "Propane scratch MLP (histogram + textural + fugacity)",
        "class": "scratch",
        "default_dir": _ROOT / "models" / "propane_scratch",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict MOF gas loading from a CSV using bundled models."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(MODELS.keys()),
        help="Which bundled model to use.",
    )
    parser.add_argument("--input-csv", required=True, type=Path, help="Input CSV with MOF rows.")
    parser.add_argument(
        "--output-csv",
        required=True,
        type=Path,
        help="Output CSV (input columns plus a prediction column).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Override the default model directory for the selected --model.",
    )
    parser.add_argument(
        "--general-model-dir",
        type=Path,
        default=_ROOT / "models" / "general",
        help="General expert model directory (used only for SFS models).",
    )
    parser.add_argument(
        "--prediction-column",
        default=None,
        help="Name of the prediction column in the output CSV.",
    )
    return parser.parse_args()


def build_predictor(model_name: str, model_dir: Path, general_model_dir: Path):
    spec = MODELS[model_name]
    if spec["class"] == "general":
        return GeneralPredictor(model_dir)
    if spec["class"] == "sfs":
        return SFSPredictor(model_dir, general_model_dir=general_model_dir)
    if spec["class"] == "scratch":
        return ScratchPredictor(model_dir)
    raise ValueError(f"Unknown model class: {spec['class']}")


def main() -> None:
    args = parse_args()
    spec = MODELS[args.model]
    model_dir = args.model_dir if args.model_dir is not None else spec["default_dir"]

    if not model_dir.is_dir():
        raise SystemExit(
            f"Model directory not found: {model_dir}\n"
            "Run assemble_models.py from the repository root to populate models/."
        )

    df = pd.read_csv(args.input_csv)
    predictor = build_predictor(args.model, model_dir, args.general_model_dir)

    col = args.prediction_column
    if col is None:
        col = "prediction_loading"

    out = predictor.predict_with_column(df, column_name=col)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"Model: {args.model}")
    print(f"Rows predicted: {len(out)}")
    print(f"Saved: {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
