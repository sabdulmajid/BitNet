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

HF_DENSE_DIR="$OUT_MODEL_DIR/hf_f16"
F16_GGUF="$OUT_MODEL_DIR/${RUN_LABEL}_f16.gguf"
TQ2_GGUF="$OUT_MODEL_DIR/${RUN_LABEL}_tq2_0.gguf"
I2S_GGUF="$OUT_MODEL_DIR/${RUN_LABEL}_i2_s_t1.gguf"
MANIFEST="$OUT_MODEL_DIR/${RUN_LABEL}_manifest.json"

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

python benchmarks/materialize_static_ternary_hf.py \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --output-dir "$HF_DENSE_DIR" \
  --dtype float16 \
  --expect-ternary-keys "$EXPECT_TERNARY_KEYS"

python 3rdparty/llama.cpp/convert_hf_to_gguf.py \
  "$HF_DENSE_DIR" \
  --outfile "$F16_GGUF" \
  --outtype f16

python benchmarks/quantize_gguf_safe.py \
  --llama-quantize "$LLAMA_BIN_DIR/llama-quantize" \
  --input "$F16_GGUF" \
  --output "$TQ2_GGUF" \
  --type TQ2_0 \
  --threads "$THREADS"

python benchmarks/quantize_gguf_safe.py \
  --llama-quantize "$LLAMA_BIN_DIR/llama-quantize" \
  --input "$F16_GGUF" \
  --output "$I2S_GGUF" \
  --type I2_S \
  --threads "$THREADS"

python - <<PY
import json
from pathlib import Path

manifest = [
    {
        "name": "qwen15b_fp_f16",
        "kind": "fp_reference",
        "path": "models/qwen2.5-1.5b-fp/qwen15b_fp_f16.gguf",
    },
    {
        "name": "qwen15b_fp_q8_0",
        "kind": "llama_q8",
        "path": "models/qwen2.5-1.5b-fp/qwen15b_fp_q8_0.gguf",
    },
    {
        "name": "qwen15b_fp_q4_k_m",
        "kind": "llama_q4",
        "path": "models/qwen2.5-1.5b-fp/qwen15b_fp_q4_k_m.gguf",
    },
    {
        "name": "${RUN_LABEL}_f16",
        "kind": "static_ternary_materialized",
        "path": "${F16_GGUF}",
    },
    {
        "name": "${RUN_LABEL}_tq2_0",
        "kind": "static_ternary_tq2",
        "path": "${TQ2_GGUF}",
    },
    {
        "name": "${RUN_LABEL}_i2_s",
        "kind": "static_ternary_i2s_single_thread_quant",
        "path": "${I2S_GGUF}",
    },
]
path = Path("${MANIFEST}")
path.write_text(json.dumps(manifest, indent=2) + "\\n", encoding="utf-8")
print(path)
PY

python benchmarks/run_gguf_suite.py \
  --models-json "$MANIFEST" \
  --out-dir "$RESULTS_DIR" \
  --llama-bin-dir "$LLAMA_BIN_DIR" \
  --perplexity-file benchmark_results/gguf-ppl/wikitext2_test_excerpt.txt \
  --threads "$THREADS" \
  --prompt-tokens "$PROMPT_TOKENS" \
  --gen-tokens "$GEN_TOKENS" \
  --ppl-chunks "$PPL_CHUNKS"

python benchmarks/audit_evidence.py \
  --gguf-summary "${RUN_LABEL}=$RESULTS_DIR/summary.json:6" \
  --output-md "$RESULTS_DIR/audit.md"

echo "wrote $RESULTS_DIR/summary.json"
echo "wrote $RESULTS_DIR/audit.md"
