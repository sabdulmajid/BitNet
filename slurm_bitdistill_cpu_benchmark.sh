#!/bin/bash
#SBATCH --job-name=bitdistill-cpu-bench
#SBATCH --partition=dualcard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/slurm_nfs/a6abdulm/projects/BitNet/logs/%x-%j.out
#SBATCH --error=/mnt/slurm_nfs/a6abdulm/projects/BitNet/logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs benchmark_results benchmarks/results

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

DATE="$(date -u +%F)"
export BITNET_REPORT_DATE="${DATE}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-64}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MODEL_DTYPE="${MODEL_DTYPE:-fp32}"
CHILD_TIMEOUT_SECONDS="${CHILD_TIMEOUT_SECONDS:-900}"
OUTPUT_JSON="benchmark_results/bitdistill_glue_cpu_${DATE}.json"
OUTPUT_MD="benchmarks/results/bitdistill_glue_cpu_${DATE}.md"
LATEST_JSON="benchmark_results/bitdistill_glue_cpu_latest.json"
LATEST_MD="benchmarks/results/bitdistill_glue_cpu_latest.md"

rm -f "$LATEST_JSON" "$LATEST_MD"

python benchmarks/benchmark_bitdistill_glue_cpu.py \
  --tasks mnli qnli sst2 \
  --runs \
    short:fp16_sft-tensor-layer-1 \
    short:bitnet_sft-tensor-layer-1 \
    short:bitdistill-tensor-layer-1 \
    short:bitdistill-row-layer-1 \
    short:bitdistill-tensor-layer-8 \
    longwarmup:bitdistill-longwarmup-tensor-layer-8 \
    longwarmup:bitdistill-longwarmup-row-layer-8 \
    papergamma:bitdistill-longwarmup-tensor-layer-8 \
    papergamma_row:bitdistill-longwarmup-row-layer-8 \
    papergamma_lr1:bitdistill-longwarmup-tensor-layer-8 \
    papergamma_lr5:bitdistill-longwarmup-tensor-layer-8 \
    papergamma_headinit:bitdistill-longwarmup-tensor-layer-8 \
  --threads "${SLURM_CPUS_PER_TASK:-12}" \
  --batch-size "$BATCH_SIZE" \
  --max-eval-samples "$MAX_EVAL_SAMPLES" \
  --model-dtype "$MODEL_DTYPE" \
  --child-timeout-seconds "$CHILD_TIMEOUT_SECONDS" \
  --output-json "$OUTPUT_JSON" \
  --output-md "$OUTPUT_MD"

cp "$OUTPUT_JSON" "$LATEST_JSON"
cp "$OUTPUT_MD" "$LATEST_MD"
