#!/bin/bash
#SBATCH --job-name=bitnet-gguf-static-suite
#SBATCH --partition=dualcard
#SBATCH --nodelist=ece-nebula10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/slurm_nfs/a6abdulm/projects/BitNet/logs/%x-%j.out
#SBATCH --error=/mnt/slurm_nfs/a6abdulm/projects/BitNet/logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:?set CHECKPOINT_DIR to an exported ternary checkpoint directory}"
EXPECT_TERNARY_KEYS="${EXPECT_TERNARY_KEYS:?set EXPECT_TERNARY_KEYS}"
RUN_LABEL="${RUN_LABEL:-qwen15b_static_ternary}"
OUT_MODEL_DIR="${OUT_MODEL_DIR:-models/${RUN_LABEL}}"
RESULTS_DIR="${RESULTS_DIR:-benchmark_results/gguf-${RUN_LABEL}}"
THREADS="${THREADS:-12}"
PPL_CHUNKS="${PPL_CHUNKS:-16}"
PROMPT_TOKENS="${PROMPT_TOKENS:-512}"
GEN_TOKENS="${GEN_TOKENS:-128}"
LLAMA_BUILD_DIR="${LLAMA_BUILD_DIR:-build-portable-avx2}"
LLAMA_BIN_DIR="${LLAMA_BIN_DIR:-$LLAMA_BUILD_DIR/bin}"

mkdir -p "$OUT_MODEL_DIR" "$RESULTS_DIR"

echo "CHECKPOINT_DIR=$CHECKPOINT_DIR"
echo "EXPECT_TERNARY_KEYS=$EXPECT_TERNARY_KEYS"
echo "RUN_LABEL=$RUN_LABEL"
echo "OUT_MODEL_DIR=$OUT_MODEL_DIR"
echo "RESULTS_DIR=$RESULTS_DIR"
echo "THREADS=$THREADS PPL_CHUNKS=$PPL_CHUNKS PROMPT_TOKENS=$PROMPT_TOKENS GEN_TOKENS=$GEN_TOKENS"
echo "LLAMA_BUILD_DIR=$LLAMA_BUILD_DIR"
echo "LLAMA_BIN_DIR=$LLAMA_BIN_DIR"

if [[ ! -x "$LLAMA_BIN_DIR/llama-quantize" || ! -x "$LLAMA_BIN_DIR/llama-cli" || ! -x "$LLAMA_BIN_DIR/llama-bench" || ! -x "$LLAMA_BIN_DIR/llama-perplexity" ]]; then
  cmake -S . -B "$LLAMA_BUILD_DIR" \
    -DGGML_NATIVE=OFF \
    -DGGML_AVX=ON \
    -DGGML_AVX2=ON \
    -DGGML_FMA=ON \
    -DGGML_F16C=ON \
    -DGGML_AVX512=OFF \
    -DGGML_AVX512_VBMI=OFF \
    -DGGML_AVX512_VNNI=OFF \
    -DGGML_AVX512_BF16=OFF \
    -DBITNET_X86_TL2=OFF
  cmake --build "$LLAMA_BUILD_DIR" --target llama-quantize llama-cli llama-bench llama-perplexity -j "$THREADS"
fi

python benchmarks/build_static_ternary_gguf_bridge.py \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --expect-ternary-keys "$EXPECT_TERNARY_KEYS" \
  --run-label "$RUN_LABEL" \
  --out-model-dir "$OUT_MODEL_DIR" \
  --results-dir "$RESULTS_DIR" \
  --llama-bin-dir "$LLAMA_BIN_DIR" \
  --threads "$THREADS" \
  --prompt-tokens "$PROMPT_TOKENS" \
  --gen-tokens "$GEN_TOKENS" \
  --ppl-chunks "$PPL_CHUNKS" \
  --run-suite

echo "wrote $RESULTS_DIR/summary.json"
echo "wrote $RESULTS_DIR/audit.md"
