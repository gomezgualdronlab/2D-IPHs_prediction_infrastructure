# Inference with the published `model_all` checkpoint

Bundled artifacts (repository root):

**`model_all/`**

| File | Role |
|------|------|
| `best_model_hpo.keras` | Trained Keras model |
| `preprocessor.joblib` | Fitted preprocessing pipeline (column subset + transforms) |

Supporting files in the same folder: **`run_config.json`** (exact training snapshot), **`model_metadata.json`** (feature lists and reference test metrics for documentation).

## 1. Input CSV expectations

Your CSV must include the same logical columns the preprocessor expects **before** preprocessing:

- **Histogram:** all `feature_*` columns listed under `inputs.feature_groups.histogram_columns` in **`model_all/model_metadata.json`** (44 columns in the bundled run).  
- **Molecular:** `chg`, `bond_length`, `eps_eff`, `sig_eff`  
- **Pressure:** `fugacity` (see `pressure_column` in `model_metadata.json`)  
- **Textural:** `LPD`, `PLD`, `SA_grav`, `VF`, `PSSD`, `density`  
- **`ads`:** include when your data carries adsorbate identity and you use the same convention as training (`run_config.json` has `include_ads: true`; the saved preprocessor’s categorical branch follows what was seen at fit time).

The script adds **`prediction_loading`** to a copy of the input table and writes the path you pass as `--output-csv`.

## 2. Command-line inference (in-repo)

Use **either** `--output-csv` **or** `--output-dir` (not both).

### Single CSV path

```bash
PYTHONPATH="$PWD" python scripts/predict_saved_model.py \
  --input-csv /absolute/path/to/input.csv \
  --model-path model_all/best_model_hpo.keras \
  --preprocessor-path model_all/preprocessor.joblib \
  --output-csv /absolute/path/to/output.csv \
  --include-textural \
  --include-ads
```

### Directory layout (predictions + plots)

With **`--output-dir`** the script writes:

| Path | Content |
|------|---------|
| `<output-dir>/predictions.csv` | Input columns plus **`prediction_loading`** |
| `<output-dir>/parity_true_vs_pred.png` | Scatter of true vs predicted loading (only if the input CSV has a **`loading`** column) |
| `<output-dir>/residual_hist.png` | Histogram of `loading - prediction_loading` (same condition) |

Example (test split with ground truth, from repo root):

```bash
PYTHONPATH="$PWD" python scripts/predict_saved_model.py \
  --input-csv data/processed/ml_ready_test.csv \
  --model-path model_all/best_model_hpo.keras \
  --preprocessor-path model_all/preprocessor.joblib \
  --output-dir outputs/example_inference \
  --include-textural \
  --include-ads
```

Generated files are under **`outputs/`**, which is listed in **`.gitignore`** so large prediction runs are not committed by default. Pick any directory you prefer.

**Flags for this bundled model:**

- **`--include-textural`** — required (textural columns are part of `model_all`).  
- **`--include-ads`** — use when your CSV has an `ads` column and you want the same feature layout as training (`run_config.json` / `model_metadata.json`).  
- Do **not** pass **`--no-histogram`**.

If flags do not match how the preprocessor was fit, expect shape errors or invalid predictions.

## 3. Exporting a small artifact bundle

```bash
PYTHONPATH="$PWD" python scripts/export_production_bundle.py \
  --run-dir model_all \
  --production-dir production \
  --copy-model --source-model-name best_model_hpo.keras \
  --copy-preprocessor
```

`production/model_config.json` records `include_textural`, `include_ads`, and `include_histogram`. Any standalone script must mirror `scripts/predict_saved_model.py` for `model_all`.

## 4. Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| KeyError / missing column | CSV missing a column the preprocessor expects (compare to `model_metadata.json` → `inputs.feature_groups`). |
| Shape mismatch on `transform` | Wrong combination of `--include-textural`, `--include-ads`, or `--no-histogram` vs training. |
| NaNs after load | Coerce or impute consistently with training data policy; raw NaNs may propagate. |

For the authoritative feature list and reference metrics, read **`model_all/model_metadata.json`**.
