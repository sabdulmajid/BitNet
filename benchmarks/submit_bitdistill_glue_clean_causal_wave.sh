#!/bin/bash
# Submit a clean GLUE BitDistill wave.
#
# This intentionally reuses an existing Stage-2 BitDistill continued-pretraining
# checkpoint.  It is the preferred launcher after the first word-label wave
# exposed token-length confounds in verbal labels such as "not_entailment".

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B}"
MODEL_SLUG="${MODEL//\//-}"
WARMUP_ROOT="${WARMUP_ROOT:-checkpoints/bitdistill-glue}"
OUTPUT_ROOT="${OUTPUT_ROOT:-checkpoints/bitdistill-glue-letters}"
WARMUP_STATE="${WARMUP_STATE:-$WARMUP_ROOT/$MODEL_SLUG/continued_pretrain/bitdistill-tensor/custom_state_dict.pt}"
TASKS=(${TASKS:-mnli qnli sst2})
SWEEP_TASK="${SWEEP_TASK:-mnli}"
SWEEP_LAYERS=(${SWEEP_LAYERS:--2 -4})
ENABLE_SWEEP="${ENABLE_SWEEP:-1}"

TASK_MAX_STEPS="${TASK_MAX_STEPS:-1000}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
LR="${LR:-2e-5}"
TASK_FORMAT="${TASK_FORMAT:-causal_lm}"
LABEL_SCHEME="${LABEL_SCHEME:-letters}"
CANDIDATE_SCORE="${CANDIDATE_SCORE:-mean}"
EXCLUDE_LINEAR_REGEX="${EXCLUDE_LINEAR_REGEX:-score|classifier}"
LOGIT_KD_WEIGHT="${LOGIT_KD_WEIGHT:-10.0}"
ATTENTION_KD_WEIGHT="${ATTENTION_KD_WEIGHT:-100.0}"
LOGIT_TEMPERATURE="${LOGIT_TEMPERATURE:-5.0}"
ATTENTION_TEMPERATURE="${ATTENTION_TEMPERATURE:-1.0}"
SBATCH_PARTITION="${SBATCH_PARTITION:-}"
SBATCH_MEM="${SBATCH_MEM:-}"

if [ ! -s "$WARMUP_STATE" ]; then
  echo "missing warmup state: $WARMUP_STATE" >&2
  exit 1
fi

common_sbatch_args() {
  if [ -n "$SBATCH_PARTITION" ]; then
    printf "%s\n" "--partition=$SBATCH_PARTITION"
  fi
  if [ -n "$SBATCH_MEM" ]; then
    printf "%s\n" "--mem=$SBATCH_MEM"
  fi
}

mkdir -p benchmark_results
JOB_KIND="${JOB_KIND:-bitdistill_glue_${TASK_FORMAT}_jobs}"
JOB_TABLE="benchmark_results/${JOB_KIND}_$(date -u +%Y%m%d_%H%M%S).tsv"
printf "phase\ttask\tmethod\tscale\tlayer\tjob_id\tdependency\toutput_dir\n" > "$JOB_TABLE"

submit_job() {
  local phase="$1"
  local task="$2"
  local method="$3"
  local scale="$4"
  local layer="$5"
  local dependency="$6"
  local output_dir="$7"
  shift 7

  local sbatch_args=(--parsable)
  while IFS= read -r arg; do
    [ -n "$arg" ] && sbatch_args+=("$arg")
  done < <(common_sbatch_args)
  if [ -n "$dependency" ]; then
    sbatch_args+=(--dependency="$dependency")
  fi

  local job_id
  job_id="$(
    env \
      MODEL="$MODEL" \
      OUTPUT_ROOT="$OUTPUT_ROOT" \
      OUTPUT_DIR="$output_dir" \
      TASK_FORMAT="$TASK_FORMAT" \
      LABEL_SCHEME="$LABEL_SCHEME" \
      CANDIDATE_SCORE="$CANDIDATE_SCORE" \
      TASK_NAME="$task" \
      METHOD="$method" \
      SCALE_MODE="$scale" \
      EXCLUDE_LINEAR_REGEX="$EXCLUDE_LINEAR_REGEX" \
      DISTILL_LAYER="$layer" \
      MAX_STEPS="$TASK_MAX_STEPS" \
      MAX_TRAIN_SAMPLES="$MAX_TRAIN_SAMPLES" \
      MAX_EVAL_SAMPLES="$MAX_EVAL_SAMPLES" \
      PER_DEVICE_BATCH_SIZE="$PER_DEVICE_BATCH_SIZE" \
      EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
      GRAD_ACCUM_STEPS="$GRAD_ACCUM_STEPS" \
      LR="$LR" \
      LOGIT_KD_WEIGHT="$LOGIT_KD_WEIGHT" \
      ATTENTION_KD_WEIGHT="$ATTENTION_KD_WEIGHT" \
      LOGIT_TEMPERATURE="$LOGIT_TEMPERATURE" \
      ATTENTION_TEMPERATURE="$ATTENTION_TEMPERATURE" \
      "$@" \
      sbatch "${sbatch_args[@]}" slurm_bitdistill_glue.sh
  )"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$phase" "$task" "$method" "$scale" "$layer" "$job_id" "${dependency:-none}" "$output_dir" | tee -a "$JOB_TABLE"
}

for task in "${TASKS[@]}"; do
  FP_DIR="$OUTPUT_ROOT/$MODEL_SLUG/$task/fp16_sft-tensor-layer-1"
  BITNET_DIR="$OUTPUT_ROOT/$MODEL_SLUG/$task/bitnet_sft-tensor-layer-1"
  BD_TENSOR_DIR="$OUTPUT_ROOT/$MODEL_SLUG/$task/bitdistill-tensor-layer-1"
  BD_ROW_DIR="$OUTPUT_ROOT/$MODEL_SLUG/$task/bitdistill-row-layer-1"

  FP_JOB="$(
    submit_job paper_baseline "$task" fp16_sft tensor -1 "" "$FP_DIR" | tail -n 1 | cut -f6
  )"
  submit_job paper_baseline "$task" bitnet_sft tensor -1 "" "$BITNET_DIR" >/dev/null
  submit_job paper_baseline "$task" bitdistill tensor -1 "afterok:${FP_JOB}" "$BD_TENSOR_DIR" \
    TEACHER_MODEL="$FP_DIR" INIT_STATE_DICT="$WARMUP_STATE" >/dev/null
  submit_job novelty_row_scale "$task" bitdistill row -1 "afterok:${FP_JOB}" "$BD_ROW_DIR" \
    TEACHER_MODEL="$FP_DIR" INIT_STATE_DICT="$WARMUP_STATE" >/dev/null
done

if [ "$ENABLE_SWEEP" = "1" ]; then
  FP_SWEEP_DIR="$OUTPUT_ROOT/$MODEL_SLUG/$SWEEP_TASK/fp16_sft-tensor-layer-1"
  FP_SWEEP_JOB="$(awk -F '\t' -v task="$SWEEP_TASK" '$1 == "paper_baseline" && $2 == task && $3 == "fp16_sft" { print $6 }' "$JOB_TABLE" | tail -n 1)"
  for layer in "${SWEEP_LAYERS[@]}"; do
    SWEEP_DIR="$OUTPUT_ROOT/$MODEL_SLUG/$SWEEP_TASK/bitdistill-tensor-layer${layer}"
    submit_job attention_layer_sweep "$SWEEP_TASK" bitdistill tensor "$layer" "afterok:${FP_SWEEP_JOB}" "$SWEEP_DIR" \
      TEACHER_MODEL="$FP_SWEEP_DIR" INIT_STATE_DICT="$WARMUP_STATE" >/dev/null
  done
fi

echo "wrote $JOB_TABLE"
