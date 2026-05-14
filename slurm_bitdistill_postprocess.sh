#!/bin/bash
#SBATCH --job-name=bitdistill-postprocess
#SBATCH --partition=dualcard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/slurm_nfs/a6abdulm/projects/BitNet/logs/%x-%j.out
#SBATCH --error=/mnt/slurm_nfs/a6abdulm/projects/BitNet/logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs benchmark_results benchmarks/results

DATE="$(date -u +%F)"

python benchmarks/monitor_bitdistill_jobs.py \
  --output-json "benchmark_results/bitdistill_job_monitor_${DATE}.json" \
  --output-md "benchmarks/results/bitdistill_job_monitor_${DATE}.md"

python benchmarks/gate_bitdistill_reproduction.py \
  --output-json "benchmark_results/bitdistill_reproduction_gate_${DATE}.json" \
  --output-md "benchmarks/results/bitdistill_reproduction_gate_${DATE}.md"

python benchmarks/summarize_bitdistill_variants.py \
  --roots \
    checkpoints/bitdistill-glue-seqcls \
    checkpoints/bitdistill-glue-seqcls-paperlogit \
    checkpoints/bitdistill-glue-seqcls-ablate \
    checkpoints/bitdistill-glue-seqcls-longwarmup \
  --title "BitDistill Variant Summary, ${DATE}" \
  --output-json "benchmark_results/bitdistill_variant_summary_${DATE}.json" \
  --output-md "benchmarks/results/bitdistill_variant_summary_${DATE}.md"

python benchmarks/audit_objective_completion.py \
  --output-json "benchmark_results/objective_completion_audit_${DATE}.json" \
  --output-md "benchmarks/results/objective_completion_audit_${DATE}.md"
