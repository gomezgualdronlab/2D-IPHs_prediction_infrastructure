# Prediction Bundle

Lightweight, self-contained package for **inference only**. Contains trained
models and Python code to predict MOF gas loading. No training data, no HPO
tuner checkpoints, and no analysis outputs.

Use this folder when sharing predictions capability on GitHub without the large
CSV splits used to train the models.

## Models included

| Model key         | Description                                      |
|-------------------|--------------------------------------------------|
| `general`         | Pretrained General (multi-adsorbate expert) MLP  |
| `co2_sfs`         | CO2 Single-Feature Stacking (expert + downstream) |
| `propane_sfs`     | Propane SFS                                      |
| `co2_scratch`     | CO2 scratch MLP                                  |
| `propane_scratch` | Propane scratch MLP                              |

Bundled SFS and scratch models are the **100% training-set** Bayesian HPO runs
from the main repository analysis pipelines.

## Layout

```
prediction_bundle/
  predict.py              Unified CLI
  assemble_models.py      Copy artifacts from training outputs (maintainers)
  requirements.txt
  inference/              Python predictors (general, SFS, scratch)
  src/                    Shared column constants
  models/
    general/              Expert model weights + preprocessor
    co2_sfs/
    propane_sfs/
    co2_scratch/
    propane_scratch/
```

## Setup

```bash
cd prediction_bundle
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Populate models (maintainers)

If `models/` is empty or you retrained in the main project, run from the
repository root:

```bash
python3 prediction_bundle/assemble_models.py
```

This copies only model weights, preprocessors, and small JSON metadata from
existing training outputs. It does **not** copy any CSV training data.

## CLI usage

Run from inside `prediction_bundle/`:

```bash
# General expert model (any adsorbate row with full descriptor set)
python3 predict.py \
  --model general \
  --input-csv /path/to/rows.csv \
  --output-csv /path/to/predictions.csv

# CO2 SFS (uses General model internally, then downstream MLP)
python3 predict.py \
  --model co2_sfs \
  --input-csv /path/to/co2_rows.csv \
  --output-csv /path/to/co2_sfs_predictions.csv

# Propane scratch baseline
python3 predict.py \
  --model propane_scratch \
  --input-csv /path/to/propane_rows.csv \
  --output-csv /path/to/propane_scratch_predictions.csv
```

Optional flags:

- `--model-dir PATH` — override the default directory for the selected model
- `--general-model-dir PATH` — expert model path (SFS models only)
- `--prediction-column NAME` — output column name (default: `prediction_loading`)

## Python API

```python
import pandas as pd
from inference.general import GeneralPredictor
from inference.sfs import SFSPredictor
from inference.scratch import ScratchPredictor

df = pd.read_csv("rows.csv")

general = GeneralPredictor("models/general")
loading = general.predict(df)

co2_sfs = SFSPredictor("models/co2_sfs", general_model_dir="models/general")
out = co2_sfs.predict_with_column(df)

scratch = ScratchPredictor("models/co2_scratch")
out = scratch.predict_with_column(df)
```

## Required input columns

### General model

MOF histogram `feature_*`, molecular descriptors (`chg`, `bond_length`,
`eps_eff`, `sig_eff`), `fugacity`, textural columns (`LPD`, `PLD`, `SA_grav`,
`VF`, `PSSD`, `density`), and `ads` when present. See
`models/general/model_metadata.json` for the authoritative list.

### SFS models (CO2 / Propane)

Same columns as the General model (expert step), plus all downstream features
listed in `models/<name>/model_config.json` → `features_after_drop` except
`expert_prediction_loading`, which is computed automatically.

Molecule-specific descriptors are required for the expert step even though they
are not passed to the downstream MLP.

### Scratch models (CO2 / Propane)

`feature_*` histogram columns, textural descriptors, and `fugacity` only. No
molecular descriptors or expert model needed.

## Approximate size

The full bundle (code + all five models) is on the order of tens of MB — suitable
for GitHub without Git LFS, unlike the multi-GB training CSVs in the main project.

## Relationship to the main repository

| Main repo                          | This bundle                    |
|------------------------------------|--------------------------------|
| Training scripts + large CSV splits| Inference code + model weights |
| HPO tuner checkpoints              | Best model only                |
| Analysis notebooks / batch jobs    | Single-row / batch CSV predict |

After retraining in the main project, rerun `assemble_models.py` to refresh
the bundled checkpoints.
