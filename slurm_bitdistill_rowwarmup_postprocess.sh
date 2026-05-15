#!/bin/bash
#SBATCH --job-name=bitdistill-rowwarmup-postprocess
#SBATCH --partition=midcard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/slurm_nfs/a6abdulm/projects/BitNet/logs/%x-%j.out
#SBATCH --error=/mnt/slurm_nfs/a6abdulm/projects/BitNet/logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs benchmark_results benchmarks/results

DATE="${BITNET_REPORT_DATE:-$(date -u +%F)}"
DATE_COMPACT="${DATE//-/}"
export BITNET_REPORT_DATE="$DATE"

ROW_WARMUP_JOB="${ROW_WARMUP_JOB:-10028}"
ROW_WARMUP_LOG="${ROW_WARMUP_LOG:-logs/bitdistill-glue-${ROW_WARMUP_JOB}.out}"
ROW_GAMMA100_TABLE="${ROW_GAMMA100_TABLE:-benchmark_results/bitdistill_rowwarmup_downstream_gamma100_${DATE_COMPACT}.tsv}"
ROW_PAPERGAMMA_TABLE="${ROW_PAPERGAMMA_TABLE:-benchmark_results/bitdistill_rowwarmup_downstream_papergamma_${DATE_COMPACT}.tsv}"

python benchmarks/monitor_bitdistill_jobs.py \
  --job-table "$ROW_GAMMA100_TABLE" \
  --warmup-log "$ROW_WARMUP_LOG" \
  --output-json "benchmark_results/bitdistill_row_warmup_monitor_${DATE}.json" \
  --output-md "benchmarks/results/bitdistill_row_warmup_monitor_${DATE}.md"

python benchmarks/monitor_bitdistill_jobs.py \
  --job-table "$ROW_PAPERGAMMA_TABLE" \
  --warmup-log "$ROW_WARMUP_LOG" \
  --output-json "benchmark_results/bitdistill_row_warmup_papergamma_monitor_${DATE}.json" \
  --output-md "benchmarks/results/bitdistill_row_warmup_papergamma_monitor_${DATE}.md"

python benchmarks/audit_bitdistill_warmup_health.py \
  --log-path "$ROW_WARMUP_LOG" \
  --monitor-json "benchmark_results/bitdistill_row_warmup_monitor_${DATE}.json" \
  --output-json "benchmark_results/bitdistill_row_warmup_health_${DATE}.json" \
  --output-md "benchmarks/results/bitdistill_row_warmup_health_${DATE}.md"

python benchmarks/summarize_bitdistill_variants.py \
  --roots \
    checkpoints/bitdistill-glue-seqcls-rowwarmup-gamma100 \
    checkpoints/bitdistill-glue-seqcls-rowwarmup-papergamma \
  --title "BitDistill Row-Warmup Variant Summary, ${DATE}" \
  --output-json "benchmark_results/bitdistill_rowwarmup_variant_summary_${DATE}.json" \
  --output-md "benchmarks/results/bitdistill_rowwarmup_variant_summary_${DATE}.md"

python benchmarks/gate_bitdistill_rowwarmup.py \
  --output-json "benchmark_results/bitdistill_rowwarmup_gate_${DATE}.json" \
  --output-md "benchmarks/results/bitdistill_rowwarmup_gate_${DATE}.md"
