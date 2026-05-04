#!/bin/bash
#SBATCH --job-name=bitnet-lmeval
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
DTYPE="${DTYPE:-bfloat16}"
RUN_ID="${SLURM_JOB_ID:-local}"
OUT_DIR="${OUT_DIR:-benchmark_results/lm-eval-${RUN_ID}}"
TASKS="${TASKS:-piqa,arc_easy,arc_challenge,hellaswag}"
LIMIT="${LIMIT:-100}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"

RUN_FP="${RUN_FP:-true}"
RUN_QAT="${RUN_QAT:-true}"
RUN_PTQ="${RUN_PTQ:-true}"

QWEN15_FP="${QWEN15_FP:-Qwen/Qwen2.5-1.5B}"
QWEN15_TERNARY="${QWEN15_TERNARY:-checkpoints/qwen2.5-1.5b-fineweb-edu/step-5000}"
QWEN15_PTQ="${QWEN15_PTQ:-checkpoints/qwen2.5-1.5b-naive-ptq-tensor}"

mkdir -p "$OUT_DIR"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "OUT_DIR=$OUT_DIR"
echo "DEVICE=$DEVICE DTYPE=$DTYPE LIMIT=$LIMIT BATCH_SIZE=$BATCH_SIZE"
echo "TASKS=$TASKS"
echo "RUN_FP=$RUN_FP RUN_QAT=$RUN_QAT RUN_PTQ=$RUN_PTQ"

if [ "$RUN_FP" = "true" ]; then
  python benchmarks/run_lm_eval.py \
    --model-kind hf \
    --model "$QWEN15_FP" \
    --tasks "$TASKS" \
    --limit "$LIMIT" \
    --batch-size "$BATCH_SIZE" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --output-json "$OUT_DIR/qwen15b_fp.json"
fi

if [ "$RUN_PTQ" = "true" ]; then
  python benchmarks/run_lm_eval.py \
    --model-kind ternary \
    --checkpoint-dir "$QWEN15_PTQ" \
    --tasks "$TASKS" \
    --limit "$LIMIT" \
    --batch-size "$BATCH_SIZE" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --output-json "$OUT_DIR/qwen15b_naive_ptq.json"
fi

if [ "$RUN_QAT" = "true" ]; then
  python benchmarks/run_lm_eval.py \
    --model-kind ternary \
    --checkpoint-dir "$QWEN15_TERNARY" \
    --tasks "$TASKS" \
    --limit "$LIMIT" \
    --batch-size "$BATCH_SIZE" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --output-json "$OUT_DIR/qwen15b_qat_ternary.json"
fi

python benchmarks/summarize_results.py \
  --perplexity-glob "$OUT_DIR/*.perplexity-do-not-match.json" \
  --mc-glob "$OUT_DIR/*.mc-do-not-match.json" \
  --lm-eval-glob "$OUT_DIR/*.json" \
  --runtime-glob "$OUT_DIR/*.runtime-do-not-match.json" \
  --generation-glob "$OUT_DIR/*.jsonl" \
  --output-md "$OUT_DIR/summary.md"

echo "wrote $OUT_DIR/summary.md"
