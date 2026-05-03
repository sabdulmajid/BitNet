#!/bin/bash
#SBATCH --job-name=bitnet-mc-bench
#SBATCH --partition=dualcard
#SBATCH --nodelist=ece-nebula10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/slurm_nfs/a6abdulm/projects/BitNet/logs/%x-%j.out
#SBATCH --error=/mnt/slurm_nfs/a6abdulm/projects/BitNet/logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs benchmark_results

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
RUN_ID="${SLURM_JOB_ID:-local}"
OUT_DIR="${OUT_DIR:-benchmark_results/mc-${RUN_ID}}"
TASKS="${TASKS:-piqa,arc_easy,arc_challenge,hellaswag}"
LIMIT="${LIMIT:-200}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"

RUN_FP="${RUN_FP:-true}"
RUN_QAT="${RUN_QAT:-true}"
RUN_PTQ="${RUN_PTQ:-true}"

QWEN05_FP="${QWEN05_FP:-Qwen/Qwen2.5-0.5B}"
QWEN15_FP="${QWEN15_FP:-Qwen/Qwen2.5-1.5B}"
QWEN05_TERNARY="${QWEN05_TERNARY:-checkpoints/qwen2.5-0.5b-fineweb-edu-12/step-1000}"
QWEN15_TERNARY="${QWEN15_TERNARY:-checkpoints/qwen2.5-1.5b-fineweb-edu/step-5000}"
QWEN05_PTQ="${QWEN05_PTQ:-checkpoints/qwen2.5-0.5b-naive-ptq-tensor}"
QWEN15_PTQ="${QWEN15_PTQ:-checkpoints/qwen2.5-1.5b-naive-ptq-tensor}"

mkdir -p "$OUT_DIR"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "OUT_DIR=$OUT_DIR"
echo "DEVICE=$DEVICE DTYPE=$DTYPE LIMIT=$LIMIT MAX_SEQ_LEN=$MAX_SEQ_LEN"
echo "TASKS=$TASKS"
echo "RUN_FP=$RUN_FP RUN_QAT=$RUN_QAT RUN_PTQ=$RUN_PTQ"

run_task() {
  local task="$1"
  local name="$2"
  shift 2
  python benchmarks/run_mc_eval.py \
    --task "$task" \
    --limit "$LIMIT" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --output-json "$OUT_DIR/${name}_${task}.json" \
    --output-jsonl "$OUT_DIR/${name}_${task}.jsonl" \
    "$@"
}

IFS=',' read -ra TASK_ARRAY <<< "$TASKS"
for task in "${TASK_ARRAY[@]}"; do
  if [ "$RUN_FP" = "true" ]; then
    run_task "$task" qwen05b_fp --model-kind hf --model "$QWEN05_FP"
    run_task "$task" qwen15b_fp --model-kind hf --model "$QWEN15_FP"
  fi

  if [ "$RUN_PTQ" = "true" ]; then
    if [ -d "$QWEN05_PTQ" ]; then
      run_task "$task" qwen05b_naive_ptq --model-kind ternary --checkpoint-dir "$QWEN05_PTQ"
    else
      echo "Skipping qwen05b_naive_ptq; directory not found: $QWEN05_PTQ"
    fi
    if [ -d "$QWEN15_PTQ" ]; then
      run_task "$task" qwen15b_naive_ptq --model-kind ternary --checkpoint-dir "$QWEN15_PTQ"
    else
      echo "Skipping qwen15b_naive_ptq; directory not found: $QWEN15_PTQ"
    fi
  fi

  if [ "$RUN_QAT" = "true" ]; then
    run_task "$task" qwen05b_ternary --model-kind ternary --checkpoint-dir "$QWEN05_TERNARY"
    run_task "$task" qwen15b_ternary --model-kind ternary --checkpoint-dir "$QWEN15_TERNARY"
  fi
done

python benchmarks/summarize_results.py \
  --perplexity-glob "$OUT_DIR/*.perplexity-do-not-match.json" \
  --mc-glob "$OUT_DIR/*.json" \
  --generation-glob "$OUT_DIR/*.jsonl" \
  --output-md "$OUT_DIR/summary.md"

echo "wrote $OUT_DIR/summary.md"
