#!/bin/bash
#SBATCH --job-name=bitdistill-warmup-finalize
#SBATCH --partition=dualcard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/slurm_nfs/a6abdulm/projects/BitNet/logs/%x-%j.out
#SBATCH --error=/mnt/slurm_nfs/a6abdulm/projects/BitNet/logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs benchmark_results benchmarks/results

DATE="${BITNET_REPORT_DATE:-$(date -u +%F)}"
export BITNET_REPORT_DATE="$DATE"

python benchmarks/run_bitdistill_warmup_finalizer.py \
  --date "${DATE}" \
  --output-json "benchmark_results/bitdistill_warmup_finalizer_${DATE}.json" \
  --output-md "benchmarks/results/bitdistill_warmup_finalizer_${DATE}.md"
