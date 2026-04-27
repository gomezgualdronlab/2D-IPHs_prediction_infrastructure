#!/bin/bash
# =============================================================================
# SLURM example: run hyperparameter search for model_all (OPTIONAL)
# -----------------------------------------------------------------------------
# This script is a starting point from one HPC environment. Your center will
# almost certainly use different partitions, modules, memory/time limits, and
# ways to activate Python. Treat every #SBATCH line and every module/conda line
# as something you must verify or replace.
#
# Checklist before sbatch:
#   - Partition / QoS / account flags match your site policy.
#   - CPUs, memory, walltime are appropriate for your HPO budget.
#   - Log paths point to a filesystem visible and writable from compute nodes.
#   - Module stack (or flat modules) provides a working TensorFlow + deps.
#   - Conda or venv activation matches where you installed requirements.txt.
#   - Thread env vars match allocated CPUs (avoid oversubscription).
#   - PROJECT_DIR points to this repository on the node (shared FS or stage copy).
#
# Submit (from repo root, after editing):
#   sbatch submit_hpo_all.sh
# Or override configs at submit time:
#   sbatch --export=ALL,PROJECT_DIR=$PWD,BASE_CONFIG=configs/base.yaml,MODEL_CONFIG=configs/model_all.yaml submit_hpo_all.sh
# =============================================================================

#SBATCH --job-name=ads_hpo_example
# YOU: set partition / QoS / account to match your cluster documentation.
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
# YOU: match cpus-per-task to what you request and to threading exports below.
#SBATCH --cpus-per-task=28
# YOU: use explicit memory (--mem=64G) if your site dislikes --mem=0.
#SBATCH --mem=0
# YOU: adjust walltime to your queue limits and expected runtime.
#SBATCH --time=4-00:00:00
# YOU: put logs somewhere valid on compute nodes (scratch, home, or project).
#SBATCH --output=logs/hpo_%j.out
#SBATCH --error=logs/hpo_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
BASE_CONFIG="${BASE_CONFIG:-configs/base_pipeline_eval.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model_all.yaml}"

cd "$PROJECT_DIR"
mkdir -p logs

# ---------------------------------------------------------------------------
# Software environment — REPLACE ENTIRELY for your site
# ---------------------------------------------------------------------------
# Example from one contributor system (commented out). Uncomment and edit,
# or delete this block and use your own module / Spack / container / venv flow.

# module purge
# module load apps/python3/2024.05
# module load compilers/gcc/13
# module load mpi/openmpi/gcc/4.1.4
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate machine_learning

# Minimal placeholder: fail fast if you have not wired the environment yet.
if ! command -v python3 >/dev/null 2>&1; then
  echo "Edit submit_hpo_all.sh: load modules or activate conda/venv so python3 is available." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Threading — align with SLURM_CPUS_PER_TASK (or your chosen thread count)
# ---------------------------------------------------------------------------
CPUS="${SLURM_CPUS_PER_TASK:-28}"
export OMP_NUM_THREADS="$CPUS"
export OPENBLAS_NUM_THREADS="$CPUS"
export MKL_NUM_THREADS="$CPUS"
export VECLIB_MAXIMUM_THREADS="$CPUS"
export NUMEXPR_NUM_THREADS="$CPUS"
export TF_NUM_INTRAOP_THREADS="$CPUS"
export TF_NUM_INTEROP_THREADS=2

PYTHONPATH="$PROJECT_DIR" python3 scripts/run_hpo.py \
  --base-config "$BASE_CONFIG" \
  --model-config "$MODEL_CONFIG" \
  --run-name "slurm_${SLURM_JOB_ID:-local}"
