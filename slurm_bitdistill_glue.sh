#!/bin/bash
#SBATCH --job-name=bitdistill-glue
#SBATCH --partition=dualcard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/slurm_nfs/a6abdulm/projects/BitNet/logs/%x-%j.out
#SBATCH --error=/mnt/slurm_nfs/a6abdulm/projects/BitNet/logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs checkpoints benchmark_results/bitdistill

if [ -n "${PYTHON_BIN:-}" ]; then
  python_candidates=("$PYTHON_BIN")
else
  python_candidates=()
  command -v python >/dev/null 2>&1 && python_candidates+=("$(command -v python)")
  [ -x "$HOME/miniconda3/bin/python" ] && python_candidates+=("$HOME/miniconda3/bin/python")
  command -v python3 >/dev/null 2>&1 && python_candidates+=("$(command -v python3)")
fi
PYTHON_BIN=""
for candidate in "${python_candidates[@]}"; do
  if "$candidate" -c 'import torch, transformers, datasets' >/dev/null 2>&1; then
    PYTHON_BIN="$candidate"
    break
  fi
done
if [ -z "$PYTHON_BIN" ]; then
  echo "No Python interpreter can import torch, transformers, and datasets. Set PYTHON_BIN explicitly." >&2
  exit 2
fi
echo "PYTHON_BIN=$PYTHON_BIN"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B}"
STAGE="${STAGE:-task_sft}"
METHOD="${METHOD:-fp16_sft}"
TASK_NAME="${TASK_NAME:-sst2}"
TASK_FORMAT="${TASK_FORMAT:-causal_lm}"
LABEL_SCHEME="${LABEL_SCHEME:-words}"
CANDIDATE_SCORE="${CANDIDATE_SCORE:-sum}"
DATASET_NAME="${DATASET_NAME:-HuggingFaceFW/fineweb-edu}"
DATASET_CONFIG="${DATASET_CONFIG:-sample-10BT}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
TEXT_COLUMN="${TEXT_COLUMN:-text}"
NUM_TRAIN_SAMPLES="${NUM_TRAIN_SAMPLES:-20000}"
MAX_PACKED_BLOCKS="${MAX_PACKED_BLOCKS:-0}"
TOKENIZER_BATCH_SIZE="${TOKENIZER_BATCH_SIZE:-128}"
TEACHER_MODEL="${TEACHER_MODEL:-}"
INIT_STATE_DICT="${INIT_STATE_DICT:-}"
INIT_STATE_MANIFEST="${INIT_STATE_MANIFEST:-}"
SCALE_MODE="${SCALE_MODE:-tensor}"
TERNARY_INIT_MODE="${TERNARY_INIT_MODE:-absmean}"
TERNARY_INIT_ITERATIONS="${TERNARY_INIT_ITERATIONS:-8}"
TERNARY_INIT_CALIBRATION_BATCHES="${TERNARY_INIT_CALIBRATION_BATCHES:-8}"
ACTIVATION_QUANTIZATION="${ACTIVATION_QUANTIZATION:-1}"
USE_SUBLN="${USE_SUBLN:-}"
EXCLUDE_LINEAR_REGEX="${EXCLUDE_LINEAR_REGEX:-score|classifier}"
DISTILL_LAYER="${DISTILL_LAYER:--1}"
ATTENTION_SPLIT_HEADS="${ATTENTION_SPLIT_HEADS:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
MAX_STEPS="${MAX_STEPS:-1000}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
LR="${LR:-2e-5}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
LOGIT_KD_WEIGHT="${LOGIT_KD_WEIGHT:-10.0}"
ATTENTION_KD_WEIGHT="${ATTENTION_KD_WEIGHT:-100.0}"
LOGIT_TEMPERATURE="${LOGIT_TEMPERATURE:-5.0}"
LOGIT_KD_TEMPERATURE_SCALE="${LOGIT_KD_TEMPERATURE_SCALE:-none}"
ATTENTION_TEMPERATURE="${ATTENTION_TEMPERATURE:-1.0}"
ATTENTION_RELATION_MODE="${ATTENTION_RELATION_MODE:-cosine}"
ATTENTION_KD_BALANCE="${ATTENTION_KD_BALANCE:-fixed}"
ATTENTION_BALANCE_TARGET_RATIO="${ATTENTION_BALANCE_TARGET_RATIO:-1.0}"
ATTENTION_BALANCE_BETA="${ATTENTION_BALANCE_BETA:-0.9}"
ATTENTION_BALANCE_EVERY_STEPS="${ATTENTION_BALANCE_EVERY_STEPS:-20}"
ATTENTION_BALANCE_MIN_WEIGHT="${ATTENTION_BALANCE_MIN_WEIGHT:-0.001}"
ATTENTION_BALANCE_MAX_WEIGHT="${ATTENTION_BALANCE_MAX_WEIGHT:-100000}"
TELEMETRY_EVERY_STEPS="${TELEMETRY_EVERY_STEPS:-0}"
TELEMETRY_COMPONENT_GRAD_NORMS="${TELEMETRY_COMPONENT_GRAD_NORMS:-0}"
TELEMETRY_MAX_ELEMENTS_PER_LAYER="${TELEMETRY_MAX_ELEMENTS_PER_LAYER:-65536}"
INIT_OUTPUT_HEAD_FROM_TEACHER="${INIT_OUTPUT_HEAD_FROM_TEACHER:-0}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"
SAVE_MODEL_ARTIFACTS="${SAVE_MODEL_ARTIFACTS:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-checkpoints/bitdistill-glue}"
MODEL_SLUG="${MODEL//\//-}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/${MODEL_SLUG}/${TASK_NAME}/${METHOD}-${SCALE_MODE}-layer${DISTILL_LAYER}}"
if [ -z "${SAVE_EVERY_STEPS+x}" ]; then
  if [ "$STAGE" = "continued_pretrain" ]; then
    SAVE_EVERY_STEPS=1000
  else
    SAVE_EVERY_STEPS=0
  fi
fi
if [ "$STAGE" = "continued_pretrain" ] && [ "$SAVE_EVERY_STEPS" = "0" ] && [ "${ALLOW_NO_WARMUP_SNAPSHOTS:-0}" != "1" ]; then
  echo "Refusing continued_pretrain with SAVE_EVERY_STEPS=0. Set SAVE_EVERY_STEPS>0 or ALLOW_NO_WARMUP_SNAPSHOTS=1." >&2
  exit 2
fi

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "MODEL=$MODEL"
echo "MODEL_SLUG=$MODEL_SLUG"
echo "STAGE=$STAGE METHOD=$METHOD TASK_NAME=$TASK_NAME TASK_FORMAT=$TASK_FORMAT LABEL_SCHEME=$LABEL_SCHEME CANDIDATE_SCORE=$CANDIDATE_SCORE"
echo "TEACHER_MODEL=${TEACHER_MODEL:-none}"
echo "INIT_STATE_MANIFEST=${INIT_STATE_MANIFEST:-none}"
if [ -n "$INIT_STATE_MANIFEST" ] && [ -z "$INIT_STATE_DICT" ]; then
  INIT_STATE_DICT="$("$PYTHON_BIN" benchmarks/resolve_stage2_state_dict.py "$INIT_STATE_MANIFEST")"
fi
echo "INIT_STATE_DICT=${INIT_STATE_DICT:-none}"
echo "SCALE_MODE=$SCALE_MODE EXCLUDE_LINEAR_REGEX=$EXCLUDE_LINEAR_REGEX DISTILL_LAYER=$DISTILL_LAYER ATTENTION_SPLIT_HEADS=$ATTENTION_SPLIT_HEADS"
echo "TERNARY_INIT_MODE=$TERNARY_INIT_MODE TERNARY_INIT_ITERATIONS=$TERNARY_INIT_ITERATIONS TERNARY_INIT_CALIBRATION_BATCHES=$TERNARY_INIT_CALIBRATION_BATCHES"
echo "ACTIVATION_QUANTIZATION=$ACTIVATION_QUANTIZATION"
echo "USE_SUBLN=${USE_SUBLN:-default}"
echo "LOGIT_KD_WEIGHT=$LOGIT_KD_WEIGHT ATTENTION_KD_WEIGHT=$ATTENTION_KD_WEIGHT LOGIT_TEMPERATURE=$LOGIT_TEMPERATURE LOGIT_KD_TEMPERATURE_SCALE=$LOGIT_KD_TEMPERATURE_SCALE ATTENTION_TEMPERATURE=$ATTENTION_TEMPERATURE ATTENTION_RELATION_MODE=$ATTENTION_RELATION_MODE"
echo "ATTENTION_KD_BALANCE=$ATTENTION_KD_BALANCE ATTENTION_BALANCE_TARGET_RATIO=$ATTENTION_BALANCE_TARGET_RATIO ATTENTION_BALANCE_BETA=$ATTENTION_BALANCE_BETA ATTENTION_BALANCE_EVERY_STEPS=$ATTENTION_BALANCE_EVERY_STEPS ATTENTION_BALANCE_MIN_WEIGHT=$ATTENTION_BALANCE_MIN_WEIGHT ATTENTION_BALANCE_MAX_WEIGHT=$ATTENTION_BALANCE_MAX_WEIGHT"
echo "TELEMETRY_EVERY_STEPS=$TELEMETRY_EVERY_STEPS TELEMETRY_COMPONENT_GRAD_NORMS=$TELEMETRY_COMPONENT_GRAD_NORMS TELEMETRY_MAX_ELEMENTS_PER_LAYER=$TELEMETRY_MAX_ELEMENTS_PER_LAYER"
echo "INIT_OUTPUT_HEAD_FROM_TEACHER=$INIT_OUTPUT_HEAD_FROM_TEACHER"
echo "MAX_SEQ_LEN=$MAX_SEQ_LEN MAX_STEPS=$MAX_STEPS PER_DEVICE_BATCH_SIZE=$PER_DEVICE_BATCH_SIZE GRAD_ACCUM_STEPS=$GRAD_ACCUM_STEPS LR=$LR LR_SCHEDULER=$LR_SCHEDULER"
echo "SAVE_EVERY_STEPS=$SAVE_EVERY_STEPS"
echo "SAVE_MODEL_ARTIFACTS=$SAVE_MODEL_ARTIFACTS"
echo "OUTPUT_DIR=$OUTPUT_DIR"

TEACHER_ARGS=()
if [ -n "$TEACHER_MODEL" ]; then
  TEACHER_ARGS=(--teacher-model "$TEACHER_MODEL")
fi

INIT_ARGS=()
if [ -n "$INIT_STATE_DICT" ]; then
  INIT_ARGS=(--init-state-dict "$INIT_STATE_DICT")
elif [ -n "$INIT_STATE_MANIFEST" ]; then
  INIT_ARGS=(--init-state-manifest "$INIT_STATE_MANIFEST")
fi

OUTPUT_HEAD_ARGS=(--no-init-output-head-from-teacher)
if [ "$INIT_OUTPUT_HEAD_FROM_TEACHER" = "1" ]; then
  OUTPUT_HEAD_ARGS=(--init-output-head-from-teacher)
fi

SAVE_MODEL_ARGS=(--save-model-artifacts)
if [ "$SAVE_MODEL_ARTIFACTS" = "0" ]; then
  SAVE_MODEL_ARGS=(--no-save-model-artifacts)
fi

TELEMETRY_ARGS=(--telemetry-every-steps "$TELEMETRY_EVERY_STEPS" --telemetry-max-elements-per-layer "$TELEMETRY_MAX_ELEMENTS_PER_LAYER" --no-telemetry-component-grad-norms)
if [ "$TELEMETRY_COMPONENT_GRAD_NORMS" = "1" ]; then
  TELEMETRY_ARGS=(--telemetry-every-steps "$TELEMETRY_EVERY_STEPS" --telemetry-max-elements-per-layer "$TELEMETRY_MAX_ELEMENTS_PER_LAYER" --telemetry-component-grad-norms)
fi

ACTIVATION_ARGS=(--activation-quantization)
if [ "$ACTIVATION_QUANTIZATION" = "0" ]; then
  ACTIVATION_ARGS=(--no-activation-quantization)
fi

SUBLN_ARGS=()
if [ "$USE_SUBLN" = "1" ]; then
  SUBLN_ARGS=(--use-subln)
elif [ "$USE_SUBLN" = "0" ]; then
  SUBLN_ARGS=(--no-use-subln)
fi

"$PYTHON_BIN" train_bitdistill.py \
  --stage "$STAGE" \
  --method "$METHOD" \
  --student-model "$MODEL" \
  "${TEACHER_ARGS[@]}" \
  "${INIT_ARGS[@]}" \
  --task-name "$TASK_NAME" \
  --task-format "$TASK_FORMAT" \
  --label-scheme "$LABEL_SCHEME" \
  --candidate-score "$CANDIDATE_SCORE" \
  --dataset-name "$DATASET_NAME" \
  --dataset-config "$DATASET_CONFIG" \
  --dataset-split "$DATASET_SPLIT" \
  --text-column "$TEXT_COLUMN" \
  --num-train-samples "$NUM_TRAIN_SAMPLES" \
  --max-packed-blocks "$MAX_PACKED_BLOCKS" \
  --tokenizer-batch-size "$TOKENIZER_BATCH_SIZE" \
  --scale-mode "$SCALE_MODE" \
  --ternary-init-mode "$TERNARY_INIT_MODE" \
  --ternary-init-iterations "$TERNARY_INIT_ITERATIONS" \
  --ternary-init-calibration-batches "$TERNARY_INIT_CALIBRATION_BATCHES" \
  "${ACTIVATION_ARGS[@]}" \
  "${SUBLN_ARGS[@]}" \
  --exclude-linear-regex "$EXCLUDE_LINEAR_REGEX" \
  --distill-layer "$DISTILL_LAYER" \
  --attention-split-heads "$ATTENTION_SPLIT_HEADS" \
  "${OUTPUT_HEAD_ARGS[@]}" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --max-steps "$MAX_STEPS" \
  --per-device-batch-size "$PER_DEVICE_BATCH_SIZE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  --learning-rate "$LR" \
  --lr-scheduler "$LR_SCHEDULER" \
  --warmup-steps "$WARMUP_STEPS" \
  --min-lr-ratio "$MIN_LR_RATIO" \
  --logit-kd-weight "$LOGIT_KD_WEIGHT" \
  --logit-kd-temperature-scale "$LOGIT_KD_TEMPERATURE_SCALE" \
  --attention-kd-weight "$ATTENTION_KD_WEIGHT" \
  --logit-temperature "$LOGIT_TEMPERATURE" \
  --attention-temperature "$ATTENTION_TEMPERATURE" \
  --attention-relation-mode "$ATTENTION_RELATION_MODE" \
  --attention-kd-balance "$ATTENTION_KD_BALANCE" \
  --attention-balance-target-ratio "$ATTENTION_BALANCE_TARGET_RATIO" \
  --attention-balance-beta "$ATTENTION_BALANCE_BETA" \
  --attention-balance-every-steps "$ATTENTION_BALANCE_EVERY_STEPS" \
  --attention-balance-min-weight "$ATTENTION_BALANCE_MIN_WEIGHT" \
  --attention-balance-max-weight "$ATTENTION_BALANCE_MAX_WEIGHT" \
  "${TELEMETRY_ARGS[@]}" \
  --max-train-samples "$MAX_TRAIN_SAMPLES" \
  --max-eval-samples "$MAX_EVAL_SAMPLES" \
  --model-dtype bf16 \
  --master-weight-dtype fp32 \
  --gradient-checkpointing \
  "${SAVE_MODEL_ARGS[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --log-every-steps 10 \
  --save-every-steps "$SAVE_EVERY_STEPS"
