#!/bin/bash
# Export BitDistill causal-LM checkpoints to row-scale I2_SR GGUF and optionally
# run llama.cpp CPU quality/speed/RSS probes.
#SBATCH --job-name=bitdistill-i2sr
#SBATCH --partition=dualcard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/slurm_nfs/a6abdulm/projects/BitNet/logs/%x-%j.out
#SBATCH --error=/mnt/slurm_nfs/a6abdulm/projects/BitNet/logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs models benchmark_results

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/bitdistill-glue-causal-longwarmup-densehead}"
TASKS=(${TASKS:-mnli qnli sst2})
SCALES=(${SCALES:-tensor row})
LAYER="${LAYER:--8}"
RUN_TEMPLATE="${RUN_TEMPLATE:-bitdistill-longwarmup-{scale}-layer-{safe_layer}}"
OUT_MODEL_DIR="${OUT_MODEL_DIR:-models/bitdistill-causal-longwarmup-i2sr}"
DATE="${BITNET_REPORT_DATE:-$(date -u +%F)}"
export BITNET_REPORT_DATE="$DATE"
RESULTS_DIR="${RESULTS_DIR:-benchmark_results/bitdistill-causal-longwarmup-i2sr-${DATE}}"
LLAMA_BIN_DIR="${LLAMA_BIN_DIR:-build-portable-avx2/bin}"
THREADS="${THREADS:-12}"
PROMPT_TOKENS="${PROMPT_TOKENS:-512}"
GEN_TOKENS="${GEN_TOKENS:-128}"
PPL_CHUNKS="${PPL_CHUNKS:-8}"
CTX_SIZES=(${CTX_SIZES:-512 2048 4096})
PERPLEXITY_FILE="${PERPLEXITY_FILE:-benchmark_results/gguf-ppl/wikitext2_test_excerpt.txt}"
ALLOW_MISSING="${ALLOW_MISSING:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RUN_SUITE="${RUN_SUITE:-1}"
RUN_MEMORY="${RUN_MEMORY:-1}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "MODEL=$MODEL"
echo "CHECKPOINT_ROOT=$CHECKPOINT_ROOT"
echo "TASKS=${TASKS[*]}"
echo "SCALES=${SCALES[*]}"
echo "LAYER=$LAYER RUN_TEMPLATE=$RUN_TEMPLATE"
echo "OUT_MODEL_DIR=$OUT_MODEL_DIR"
echo "RESULTS_DIR=$RESULTS_DIR"
echo "LLAMA_BIN_DIR=$LLAMA_BIN_DIR THREADS=$THREADS PPL_CHUNKS=$PPL_CHUNKS"

args=(
  --root "$CHECKPOINT_ROOT"
  --model "$MODEL"
  --tasks "${TASKS[@]}"
  --scales "${SCALES[@]}"
  --layer "$LAYER"
  --run-template "$RUN_TEMPLATE"
  --out-model-dir "$OUT_MODEL_DIR"
  --results-dir "$RESULTS_DIR"
  --llama-bin-dir "$LLAMA_BIN_DIR"
  --threads "$THREADS"
  --prompt-tokens "$PROMPT_TOKENS"
  --gen-tokens "$GEN_TOKENS"
  --ppl-chunks "$PPL_CHUNKS"
  --perplexity-file "$PERPLEXITY_FILE"
  --ctx-sizes "${CTX_SIZES[@]}"
  --skip-unsupported-architecture
)

if [ "$ALLOW_MISSING" = "1" ]; then
  args+=(--allow-missing)
fi
if [ "$SKIP_EXISTING" = "1" ]; then
  args+=(--skip-existing)
fi
if [ "$RUN_SUITE" = "1" ]; then
  args+=(--run-suite)
fi
if [ "$RUN_MEMORY" = "1" ]; then
  args+=(--run-memory)
fi

python benchmarks/export_bitdistill_i2sr_suite.py "${args[@]}"
