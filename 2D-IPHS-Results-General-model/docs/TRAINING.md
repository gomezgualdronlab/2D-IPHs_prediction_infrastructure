# Training and hyperparameter search (`model_all`)

This document describes how **training** and **HPO** are run for the **`model_all`** profile. It assumes the repository root as the working directory.

## 1. What gets merged

Entry points load two YAML files and merge them (model overrides base):

- **`configs/base.yaml`** — default paths (`results/`, `results/hpo/`), full-data CSV paths under `data/processed/`, default architecture, training loop, and HPO budget.  
- **`configs/base_pipeline_eval.yaml`** — same layout but writes under **`results_pipeline_eval/`** and uses a larger **`hpo.max_trials`** (20 in the config version that produced the archived checkpoint).  
- **`configs/model_all.yaml`** — sets `features.name: model_all` and enables histogram + textural + ads-related handling consistent with `src/data.py`.

The **exact** settings used for the **bundled** checkpoint are frozen in **`model_all/run_config.json`**.

## 2. Data

Processed splits live in **`data/processed/`** (see [`../data/README.md`](../data/README.md)):

- `ml_ready_training.csv`  
- `ml_ready_validation.csv`  
- `ml_ready_test.csv`  

`model_all/run_config.json` points at those files. Optional subsets under `data/subsets/` are only for faster experiments; create them with `scripts/make_data_subset.py` if needed.

## 3. Single training run (no HPO)

Writes to: `results_dir / model_all / <run_name>/` (see `configs/base.yaml` → `paths.results_dir`).

```bash
PYTHONPATH="$PWD" python scripts/run_train.py \
  --base-config configs/base.yaml \
  --model-config configs/model_all.yaml
```

Optional explicit run name:

```bash
PYTHONPATH="$PWD" python scripts/run_train.py \
  --base-config configs/base.yaml \
  --model-config configs/model_all.yaml \
  --run-name my_experiment
```

Typical artifacts include `metrics.json`, `history.json`, `run_config.json`, `preprocessor.joblib`, `best_model.keras` / `final_model.keras`, and plots or prediction CSVs when enabled. See `src/train.py` for exact outputs.

## 4. Hyperparameter search (HPO)

Writes to: `hpo_dir / model_all / <run_name>/` (for example `results_pipeline_eval/hpo/model_all/<run_name>/` when using `base_pipeline_eval.yaml`).

```bash
PYTHONPATH="$PWD" python scripts/run_hpo.py \
  --base-config configs/base_pipeline_eval.yaml \
  --model-config configs/model_all.yaml \
  --run-name my_hpo_run
```

KerasTuner drives the search (`src/hpo.py`). A finished run includes `best_model_hpo.keras`, `best_trial.json`, trial metrics logs, histories, and plots under that run directory. **The shared repository only ships the flat `model_all/` inference bundle** (model, preprocessor, configs, metadata), not full HPO logs.

## 5. Reproducibility notes

- **Randomness:** set `training.seed` in YAML; hardware and library versions can still change numerical details.  
- **Environment:** pin TensorFlow and dependencies (`requirements.txt`).  
- Re-running HPO produces a **new** run directory; new trials will not match the archived checkpoint unless every source of randomness is controlled.

## 6. Cluster submission (SLURM)

**`submit_hpo_all.sh`** at the repo root is an **example** for SLURM sites. It is not portable as-is. Before `sbatch`, review this checklist:

| Area | What to change |
|------|----------------|
| **Partitions / QoS** | `#SBATCH --partition=…` (and `#SBATCH --qos=…` if your center uses QoS). |
| **Account / reservation** | Add `#SBATCH -A …` or `#SBATCH --reservation=…` if required. |
| **Resources** | `--cpus-per-task`, `--mem`, `--time` to match policy and job size. |
| **GPUs** | This example targets **CPU** jobs. For GPU, switch partition, add `#SBATCH --gres=gpu:…`, and adjust TensorFlow device settings if needed. |
| **Log locations** | `#SBATCH --output` / `--error`: use a directory that exists on the compute nodes and that you are allowed to write (often under `$HOME` or a scratch filesystem). The script does `mkdir -p logs` only on the **shared** project path—ensure that path is visible from compute nodes or write logs to `$SLURM_SUBMIT_DIR` / scratch. |
| **Software stack** | Replace `module load …` with your center’s modules (Python, compiler, MPI if you use it, CUDA for GPU). |
| **Python environment** | Replace `conda activate …` with your venv or conda env where `requirements.txt` is installed. Some sites use `module load python` instead of conda. |
| **Threading** | Align `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `TF_NUM_INTRAOP_THREADS`, etc. with allocated CPUs to avoid oversubscription. |
| **Job identity** | `--job-name` and optional `--array` if you run multiple seeds or configs. |
| **Project path** | Set `PROJECT_DIR` (or `cd` to the repo) so `PYTHONPATH` and config paths resolve on compute nodes. |

Defaults in the script (`BASE_CONFIG`, `MODEL_CONFIG`) point at `configs/base_pipeline_eval.yaml` and `configs/model_all.yaml`; override with `sbatch --export=ALL,BASE_CONFIG=…,MODEL_CONFIG=…` if you use other files.

After a successful HPO run, copy **`best_model_hpo.keras`**, **`preprocessor.joblib`**, and **`run_config.json`** from the new run directory into **`model_all/`** (and refresh **`model_metadata.json`** if you want to publish updated hyperparameters and metrics) before sharing.
