from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export production bundle metadata/artifacts from a run directory.")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Training/HPO run directory containing run_config.json and preprocessor.joblib.",
    )
    parser.add_argument(
        "--production-dir",
        default="production",
        help="Destination production directory.",
    )
    parser.add_argument(
        "--source-model-name",
        default="best_model.keras",
        help="Model filename inside run-dir to copy (e.g. best_model.keras, final_model.keras, best_model_hpo.keras).",
    )
    parser.add_argument(
        "--target-model-name",
        default="best_model.keras",
        help="Model filename to write in production-dir.",
    )
    parser.add_argument(
        "--copy-model",
        action="store_true",
        help="Copy selected model file from run-dir to production-dir.",
    )
    parser.add_argument(
        "--copy-preprocessor",
        action="store_true",
        help="Copy preprocessor.joblib from run-dir to production-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    production_dir = Path(args.production_dir)
    production_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = _load_json(run_dir / "run_config.json")
    features = run_cfg.get("features", {})
    manifest = {
        "model_name": str(features.get("name", "selected_model")),
        "features": {
            "include_textural": bool(features.get("include_textural", True)),
            "include_ads": bool(features.get("include_ads", True)),
            "include_histogram": bool(features.get("include_histogram", True)),
        },
        "prediction_column": "prediction_loading",
        "notes": f"Exported from run dir: {run_dir}",
    }

    with open(production_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if args.copy_preprocessor:
        shutil.copy2(run_dir / "preprocessor.joblib", production_dir / "preprocessor.joblib")

    if args.copy_model:
        shutil.copy2(run_dir / args.source_model_name, production_dir / args.target_model_name)

    print(f"Wrote manifest: {production_dir / 'model_config.json'}")
    if args.copy_preprocessor:
        print(f"Copied: {run_dir / 'preprocessor.joblib'} -> {production_dir / 'preprocessor.joblib'}")
    if args.copy_model:
        print(f"Copied: {run_dir / args.source_model_name} -> {production_dir / args.target_model_name}")


if __name__ == "__main__":
    main()
