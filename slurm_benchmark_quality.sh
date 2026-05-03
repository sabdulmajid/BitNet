#!/bin/bash
#SBATCH --job-name=bitnet-quality-bench
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
OUT_DIR="${OUT_DIR:-benchmark_results/quality-${RUN_ID}}"

QWEN05_FP="${QWEN05_FP:-Qwen/Qwen2.5-0.5B}"
QWEN15_FP="${QWEN15_FP:-Qwen/Qwen2.5-1.5B}"
QWEN05_TERNARY="${QWEN05_TERNARY:-checkpoints/qwen2.5-0.5b-fineweb-edu-12/step-1000}"
QWEN15_TERNARY="${QWEN15_TERNARY:-checkpoints/qwen2.5-1.5b-fineweb-edu/step-5000}"

WIKITEXT_BLOCKS="${WIKITEXT_BLOCKS:-64}"
WIKITEXT_SEQ="${WIKITEXT_SEQ:-512}"
FINEWEB_BLOCKS="${FINEWEB_BLOCKS:-32}"
FINEWEB_SEQ="${FINEWEB_SEQ:-1024}"
FINEWEB_SKIP_ROWS="${FINEWEB_SKIP_ROWS:-25000}"

mkdir -p "$OUT_DIR"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "OUT_DIR=$OUT_DIR"
echo "DEVICE=$DEVICE DTYPE=$DTYPE"
echo "WIKITEXT_BLOCKS=$WIKITEXT_BLOCKS WIKITEXT_SEQ=$WIKITEXT_SEQ"
echo "FINEWEB_BLOCKS=$FINEWEB_BLOCKS FINEWEB_SEQ=$FINEWEB_SEQ FINEWEB_SKIP_ROWS=$FINEWEB_SKIP_ROWS"

run_wikitext() {
  local name="$1"
  shift
  python benchmarks/run_perplexity.py \
    "$@" \
    --output-json "$OUT_DIR/${name}_wikitext.json" \
    --dataset-name wikitext \
    --dataset-config wikitext-2-raw-v1 \
    --dataset-split test \
    --text-column text \
    --max-blocks "$WIKITEXT_BLOCKS" \
    --max-seq-len "$WIKITEXT_SEQ" \
    --batch-size 1 \
    --device "$DEVICE" \
    --dtype "$DTYPE"
}

run_fineweb() {
  local name="$1"
  shift
  python benchmarks/run_perplexity.py \
    "$@" \
    --output-json "$OUT_DIR/${name}_fineweb_heldout.json" \
    --dataset-name HuggingFaceFW/fineweb-edu \
    --dataset-config sample-10BT \
    --dataset-split train \
    --text-column text \
    --skip-rows "$FINEWEB_SKIP_ROWS" \
    --max-blocks "$FINEWEB_BLOCKS" \
    --max-seq-len "$FINEWEB_SEQ" \
    --batch-size 1 \
    --device "$DEVICE" \
    --dtype "$DTYPE"
}

run_wikitext qwen05b_fp --model-kind hf --model "$QWEN05_FP"
run_wikitext qwen05b_ternary --model-kind ternary --checkpoint-dir "$QWEN05_TERNARY"
run_wikitext qwen15b_fp --model-kind hf --model "$QWEN15_FP"
run_wikitext qwen15b_ternary --model-kind ternary --checkpoint-dir "$QWEN15_TERNARY"

run_fineweb qwen05b_fp --model-kind hf --model "$QWEN05_FP"
run_fineweb qwen05b_ternary --model-kind ternary --checkpoint-dir "$QWEN05_TERNARY"
run_fineweb qwen15b_fp --model-kind hf --model "$QWEN15_FP"
run_fineweb qwen15b_ternary --model-kind ternary --checkpoint-dir "$QWEN15_TERNARY"

python benchmarks/summarize_results.py \
  --perplexity-glob "$OUT_DIR/*.json" \
  --generation-glob "$OUT_DIR/*.jsonl" \
  --output-md "$OUT_DIR/summary.md"

echo "wrote $OUT_DIR/summary.md"
