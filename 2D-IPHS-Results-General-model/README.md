# Adsorption DNN — reproducible `model_all` package

This repository bundles **code, fixed data splits, configuration, and trained artifacts** to reproduce a TensorFlow DNN regressor and use it to predict **`Loading`**.

**Scope:** The **`model_all`** profile makes predictions using the representation consisting of: MOF histogram (`feature_*` columns), textural descriptors, molecular descriptors, fugacity, and optional `ads` handling as in the saved preprocessor.

| Topic | Where to read |
|--------|----------------|
| Environment, layout, quick links | This file |
| Training and HPO step-by-step | [`docs/TRAINING.md`](docs/TRAINING.md) |
| Running predictions on new CSVs | [`docs/INFERENCE.md`](docs/INFERENCE.md) |
| Data file naming and splits | [`data/README.md`](data/README.md) |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Use `PYTHONPATH="$PWD"` (or install the project in editable mode) so `import src` works from the repo root.

## Bundled checkpoint (inference-ready)

All published weights and config for **`model_all`** live in one directory:

**[`model_all/`](model_all/)**

| File | Purpose |
|------|---------|
| `best_model_hpo.keras` | Trained Keras model |
| `preprocessor.joblib` | Fitted preprocessing pipeline |
| `run_config.json` | Merged YAML snapshot used for that run (data paths, feature flags, training/HPO settings) |
| `model_metadata.json` | Hyperparameters, input feature lists, and reference test-set metrics (documentation only) |

Training for that run used the CSVs under **`data/processed/`** listed in `model_all/run_config.json`.

## Quick inference (same repo)

```bash
PYTHONPATH="$PWD" python scripts/predict_saved_model.py \
  --input-csv path/to/your_rows.csv \
  --model-path model_all/best_model_hpo.keras \
  --preprocessor-path model_all/preprocessor.joblib \
  --output-csv path/to/predictions.csv \
  --include-textural \
  --include-ads
```

To write **`predictions.csv`** plus **parity** and **residual** plots in one folder (when the input CSV includes a **`loading`** column for evaluation), use **`--output-dir`** instead of **`--output-csv`**; see [`docs/INFERENCE.md`](docs/INFERENCE.md).

Do **not** pass `--no-histogram` for this model. Required columns match `model_all/model_metadata.json` (`inputs.feature_groups`); see [`docs/INFERENCE.md`](docs/INFERENCE.md).

## Optional production folder

```bash
PYTHONPATH="$PWD" python scripts/export_production_bundle.py \
  --run-dir model_all \
  --production-dir production \
  --copy-model --source-model-name best_model_hpo.keras \
  --copy-preprocessor
```

That writes `model_config.json` (feature flags) plus the model and preprocessor. Inference still needs a driver compatible with those flags (reference: `scripts/predict_saved_model.py`).

## SLURM / HPC (optional)

**[`submit_hpo_all.sh`](submit_hpo_all.sh)** is an **example** batch script for systems using SLURM. It is not tied to a specific machine: read the header checklist and edit resource directives, modules, conda, log paths, and threading for your site before `sbatch`. See also [`docs/TRAINING.md`](docs/TRAINING.md#6-cluster-submission-slurm).

## Training vs inference docs

- **[`docs/TRAINING.md`](docs/TRAINING.md)** — how configs combine, where new runs write output, and reproducibility caveats.  
- **[`docs/INFERENCE.md`](docs/INFERENCE.md)** — required inputs, CLI flags, troubleshooting.
