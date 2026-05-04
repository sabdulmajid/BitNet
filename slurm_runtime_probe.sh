#!/bin/bash
#SBATCH --job-name=bitnet-runtime
#SBATCH --partition=midcard
#SBATCH --nodelist=ece-nebula12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/slurm_nfs/a6abdulm/projects/BitNet/logs/%x-%j.out
#SBATCH --error=/mnt/slurm_nfs/a6abdulm/projects/BitNet/logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs benchmark_results

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

DEVICE="${DEVICE:-cpu}"
DTYPE="${DTYPE:-fp32}"
RUN_ID="${SLURM_JOB_ID:-local}"
OUT_DIR="${OUT_DIR:-benchmark_results/runtime-${RUN_ID}}"
PROMPT_TOKENS="${PROMPT_TOKENS:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
REPEATS="${REPEATS:-3}"
WARMUP="${WARMUP:-1}"
NUM_THREADS="${NUM_THREADS:-${SLURM_CPUS_PER_TASK:-12}}"

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
echo "DEVICE=$DEVICE DTYPE=$DTYPE THREADS=$NUM_THREADS"
echo "PROMPT_TOKENS=$PROMPT_TOKENS MAX_NEW_TOKENS=$MAX_NEW_TOKENS REPEATS=$REPEATS"
echo "RUN_FP=$RUN_FP RUN_QAT=$RUN_QAT RUN_PTQ=$RUN_PTQ"
lscpu | sed -n '1,35p'

run_probe() {
  local name="$1"
  shift
  python benchmarks/run_runtime_probe.py \
    "$@" \
    --output-json "$OUT_DIR/${name}.json" \
    --prompt-tokens "$PROMPT_TOKENS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --warmup "$WARMUP" \
    --repeats "$REPEATS" \
    --num-threads "$NUM_THREADS" \
    --device "$DEVICE" \
    --dtype "$DTYPE"
}

if [ "$RUN_FP" = "true" ]; then
  run_probe qwen05b_fp --model-kind hf --model "$QWEN05_FP"
  run_probe qwen15b_fp --model-kind hf --model "$QWEN15_FP"
fi

if [ "$RUN_PTQ" = "true" ]; then
  run_probe qwen05b_naive_ptq --model-kind ternary --checkpoint-dir "$QWEN05_PTQ"
  run_probe qwen15b_naive_ptq --model-kind ternary --checkpoint-dir "$QWEN15_PTQ"
fi

if [ "$RUN_QAT" = "true" ]; then
  run_probe qwen05b_qat_ternary --model-kind ternary --checkpoint-dir "$QWEN05_TERNARY"
  run_probe qwen15b_qat_ternary --model-kind ternary --checkpoint-dir "$QWEN15_TERNARY"
fi

python benchmarks/summarize_results.py \
  --perplexity-glob "$OUT_DIR/*.perplexity-do-not-match.json" \
  --mc-glob "$OUT_DIR/*.mc-do-not-match.json" \
  --lm-eval-glob "$OUT_DIR/*.lm-eval-do-not-match.json" \
  --runtime-glob "$OUT_DIR/*.json" \
  --gguf-runtime-glob "$OUT_DIR/*.gguf-runtime-do-not-match.json" \
  --gguf-ppl-glob "$OUT_DIR/*.gguf-ppl-do-not-match.log" \
  --generation-glob "$OUT_DIR/*.jsonl" \
  --output-md "$OUT_DIR/summary.md"

echo "wrote $OUT_DIR/summary.md"
