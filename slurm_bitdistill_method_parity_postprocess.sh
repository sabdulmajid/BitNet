#!/bin/bash
#SBATCH --job-name=bd-parity-audit
#SBATCH --partition=dualcard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/slurm_nfs/a6abdulm/projects/BitNet/logs/%x-%j.out
#SBATCH --error=/mnt/slurm_nfs/a6abdulm/projects/BitNet/logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [ -z "${PYTHON_BIN:-}" ]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif [ -x "$HOME/miniconda3/bin/python" ]; then
    PYTHON_BIN="$HOME/miniconda3/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "No Python interpreter found. Set PYTHON_BIN explicitly." >&2
    exit 2
  fi
fi
"$PYTHON_BIN" benchmarks/audit_bitdistill_method_parity.py \
  --submission-job-id "${PARITY_ARRAY_JOB_ID:-unknown}" \
  --output-json benchmarks/results/bitdistill_method_parity_2026-09-04.json \
  --output-md benchmarks/results/bitdistill_method_parity_2026-09-04.md
