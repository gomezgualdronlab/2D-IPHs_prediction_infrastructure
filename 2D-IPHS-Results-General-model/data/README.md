# Data layout

| Path | Role |
|------|------|
| `data/processed/` | Canonical train / validation / test CSVs used by `configs/base.yaml` and `configs/base_full_thorough.yaml`. |
| `data/subsets/` | Derived samples (e.g. random fractions). Each subdirectory holds the three split files with the same names as in `processed/`. `base_pipeline_eval.yaml` uses `data/subsets/frac0.10_seed42/`. |
| `data/raw/` | Place upstream or intermediate exports here; add preparation scripts that write into `data/processed/` (or versioned dirs). |

Note: csv files are compressed in .zip format for easier download and transfer.

Regenerate a 10% sample from the full processed splits:

```bash
python scripts/make_data_subset.py \
  --input-dir data/processed \
  --output-dir data/subsets/frac0.10_seed42 \
  --frac 0.1 \
  --seed 42
```

With no arguments, `make_data_subset.py` reads from `data/processed/` and writes to `data/subsets/subset_out/` (overwrite that directory or pass a unique `--output-dir` per experiment).

Point any YAML `data.*_csv` entries at the folder you want for that experiment.
