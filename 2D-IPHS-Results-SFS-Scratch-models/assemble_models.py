#!/usr/bin/env python3
"""Copy trained model artifacts into prediction_bundle/models/ (no training data)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_ROOT = Path(__file__).resolve().parent
MODELS_ROOT = BUNDLE_ROOT / "models"

SCRATCH_FEATURE_COLUMNS = [
    "feature_106",
    "feature_108",
    "feature_110",
    "feature_112",
    "feature_180",
    "feature_182",
    "feature_184",
    "feature_186",
    "feature_254",
    "feature_258",
    "feature_260",
    "feature_328",
    "feature_1581",
    "feature_1592",
    "feature_1598",
    "feature_1599",
    "feature_1655",
    "feature_1666",
    "feature_1668",
    "feature_1673",
    "feature_1689",
    "feature_1729",
    "feature_1740",
    "feature_1742",
    "feature_1763",
    "feature_1803",
    "feature_1814",
    "feature_1877",
    "feature_3051",
    "feature_3064",
    "feature_3066",
    "feature_3077",
    "feature_3125",
    "feature_3138",
    "feature_3140",
    "feature_3145",
    "feature_3152",
    "feature_3199",
    "feature_3214",
    "feature_3219",
    "feature_3226",
    "feature_3273",
    "feature_3288",
    "feature_3347",
    "fugacity",
    "LPD",
    "PLD",
    "SA_grav",
    "VF",
    "PSSD",
    "density",
]

ARTIFACT_FILES = (
    "best_model_hpo.keras",
    "preprocessor.joblib",
    "metrics.json",
    "best_hyperparameters.json",
)


def copy_file(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(f"Missing artifact: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def copy_general() -> None:
    src_dir = REPO_ROOT / "inference_bundle" / "model"
    dst_dir = MODELS_ROOT / "general"
    for name in (
        "best_model_hpo.keras",
        "preprocessor.joblib",
        "model_config.json",
        "model_metadata.json",
        "run_config.json",
    ):
        copy_file(src_dir / name, dst_dir / name)
    print(f"  general -> {dst_dir}")


def copy_sfs(name: str, source_rel: Path) -> None:
    src_dir = REPO_ROOT / source_rel
    dst_dir = MODELS_ROOT / name

    for fname in ARTIFACT_FILES:
        src = src_dir / fname
        if src.is_file():
            copy_file(src, dst_dir / fname)

    run_config = json.loads((src_dir / "run_config.json").read_text(encoding="utf-8"))
    model_config = {
        "model_name": name,
        "molecule": "co2" if "co2" in name else "propane",
        "pipeline": "sfs",
        "general_model_dir": "../general",
        "artifact_paths": {
            "keras_model": "best_model_hpo.keras",
            "preprocessor": "preprocessor.joblib",
        },
        "prediction_column": "prediction_loading",
        "feature_policy": run_config.get("feature_policy", {}),
        "features_after_drop": run_config.get("features_after_drop", []),
        "training": {
            "train_fraction": run_config.get("train_fraction"),
            "best_hyperparameters": run_config.get("hpo", {}).get("best_hyperparameters"),
        },
    }
    write_json(dst_dir / "model_config.json", model_config)
    print(f"  {name} -> {dst_dir}")


def copy_scratch(name: str, source_rel: Path) -> None:
    src_dir = REPO_ROOT / source_rel
    dst_dir = MODELS_ROOT / name

    for fname in ARTIFACT_FILES:
        src = src_dir / fname
        if src.is_file():
            copy_file(src, dst_dir / fname)

    run_config = {}
    run_path = src_dir / "run_config.json"
    if run_path.is_file():
        run_config = json.loads(run_path.read_text(encoding="utf-8"))

    model_config = {
        "model_name": name,
        "molecule": "co2" if "co2" in name else "propane",
        "pipeline": "scratch",
        "artifact_paths": {
            "keras_model": "best_model_hpo.keras",
            "preprocessor": "preprocessor.joblib",
        },
        "prediction_column": "prediction_loading",
        "feature_columns": SCRATCH_FEATURE_COLUMNS,
        "training": {
            "best_hyperparameters": run_config.get("hpo", {}).get("best_hyperparameters"),
        },
    }
    write_json(dst_dir / "model_config.json", model_config)
    print(f"  {name} -> {dst_dir}")


def main() -> None:
    print("Assembling prediction_bundle/models/ from trained artifacts...")
    copy_general()
    copy_sfs(
        "co2_sfs",
        Path("CO2_bayes_data_availability_analysis/outputs/co2_sfs_bayes_100pct"),
    )
    copy_sfs(
        "propane_sfs",
        Path("Propane_bayes_data_availability_analysis/outputs/propane_sfs_bayes_100pct"),
    )
    copy_scratch(
        "co2_scratch",
        Path("CO2_bayes_data_availability_analysis/outputs/co2_scratch_bayes_100pct"),
    )
    copy_scratch(
        "propane_scratch",
        Path("Propane_bayes_data_availability_analysis/outputs/propane_scratch_bayes_100pct"),
    )
    print("Done.")


if __name__ == "__main__":
    main()
