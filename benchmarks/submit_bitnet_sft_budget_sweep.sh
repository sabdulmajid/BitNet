#!/bin/bash
# Submit a focused BitNet-SFT MNLI LR/step-budget sweep.
#
# This is a reproduction-gap diagnostic for the weak BitNet-SFT baseline.  It
# intentionally holds the task formulation, model, quantization mode, SubLN
# setting, and activation quantization fixed while varying only the optimizer
# learning rate and the number of downstream task steps.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B}"
MODEL_SLUG="${MODEL//\//-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-checkpoints/bitdistill-glue-seqcls-bitnet-sft-budget}"
TASK_NAME="${TASK_NAME:-mnli}"
TASK_FORMAT="${TASK_FORMAT:-sequence_classification}"
LABEL_SCHEME="${LABEL_SCHEME:-letters}"
CANDIDATE_SCORE="${CANDIDATE_SCORE:-mean}"
SCALE_MODE="${SCALE_MODE:-tensor}"
TERNARY_INIT_MODE="${TERNARY_INIT_MODE:-absmean}"
TERNARY_INIT_ITERATIONS="${TERNARY_INIT_ITERATIONS:-8}"
TERNARY_INIT_CALIBRATION_BATCHES="${TERNARY_INIT_CALIBRATION_BATCHES:-8}"
EXCLUDE_LINEAR_REGEX="${EXCLUDE_LINEAR_REGEX:-score|classifier}"
ACTIVATION_QUANTIZATION="${ACTIVATION_QUANTIZATION:-1}"
USE_SUBLN="${USE_SUBLN:-0}"

LRS=(${LRS:-5e-6 1e-5 2e-5 5e-5})
STEPS_LIST=(${STEPS_LIST:-1000 3000})
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"
SAVE_MODEL_ARTIFACTS="${SAVE_MODEL_ARTIFACTS:-0}"
SBATCH_PARTITION="${SBATCH_PARTITION:-midcard}"
SBATCH_MEM="${SBATCH_MEM:-24G}"

safe_value() {
  printf "%s" "$1" | sed -e 's/-/m/g' -e 's/+//g' -e 's/\\./p/g'
}

common_sbatch_args() {
  if [ -n "$SBATCH_PARTITION" ]; then
    printf "%s\n" "--partition=$SBATCH_PARTITION"
  fi
  if [ -n "$SBATCH_MEM" ]; then
    printf "%s\n" "--mem=$SBATCH_MEM"
  fi
}

mkdir -p benchmark_results
JOB_TABLE="${JOB_TABLE:-benchmark_results/bitnet_sft_budget_sweep_$(date -u +%Y%m%d_%H%M%S)_$$_${RANDOM}.tsv}"
printf "phase\ttask\tmethod\tscale\tternary_init_mode\tternary_init_iterations\tternary_init_calibration_batches\tsteps\tlr\tjob_id\toutput_dir\tactivation_quantization\tuse_subln\tmax_train_samples\tmax_eval_samples\tper_device_batch_size\tgrad_accum_steps\tsave_model_artifacts\n" > "$JOB_TABLE"

submit_job() {
  local steps="$1"
  local lr="$2"
  local safe_lr
  safe_lr="$(safe_value "$lr")"
  local init_suffix=""
  if [ "$TERNARY_INIT_MODE" != "absmean" ]; then
    init_suffix="-init${TERNARY_INIT_MODE}"
    if [ "$TERNARY_INIT_MODE" = "diag_ls" ]; then
      init_suffix="${init_suffix}-cal${TERNARY_INIT_CALIBRATION_BATCHES}"
    fi
  fi
  local output_dir="$OUTPUT_ROOT/$MODEL_SLUG/$TASK_NAME/bitnet_sft-${SCALE_MODE}${init_suffix}-steps${steps}-lr${safe_lr}"

  local sbatch_args=(--parsable)
  while IFS= read -r arg; do
    [ -n "$arg" ] && sbatch_args+=("$arg")
  done < <(common_sbatch_args)

  local job_id
  job_id="$(
    env \
      MODEL="$MODEL" \
      STAGE=task_sft \
      METHOD=bitnet_sft \
      TASK_NAME="$TASK_NAME" \
      TASK_FORMAT="$TASK_FORMAT" \
      LABEL_SCHEME="$LABEL_SCHEME" \
      CANDIDATE_SCORE="$CANDIDATE_SCORE" \
      SCALE_MODE="$SCALE_MODE" \
      TERNARY_INIT_MODE="$TERNARY_INIT_MODE" \
      TERNARY_INIT_ITERATIONS="$TERNARY_INIT_ITERATIONS" \
      TERNARY_INIT_CALIBRATION_BATCHES="$TERNARY_INIT_CALIBRATION_BATCHES" \
      EXCLUDE_LINEAR_REGEX="$EXCLUDE_LINEAR_REGEX" \
      ACTIVATION_QUANTIZATION="$ACTIVATION_QUANTIZATION" \
      USE_SUBLN="$USE_SUBLN" \
      OUTPUT_ROOT="$OUTPUT_ROOT" \
      OUTPUT_DIR="$output_dir" \
      MAX_STEPS="$steps" \
      MAX_TRAIN_SAMPLES="$MAX_TRAIN_SAMPLES" \
      MAX_EVAL_SAMPLES="$MAX_EVAL_SAMPLES" \
      PER_DEVICE_BATCH_SIZE="$PER_DEVICE_BATCH_SIZE" \
      EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
      GRAD_ACCUM_STEPS="$GRAD_ACCUM_STEPS" \
      LR="$lr" \
      SAVE_MODEL_ARTIFACTS="$SAVE_MODEL_ARTIFACTS" \
      sbatch "${sbatch_args[@]}" slurm_bitdistill_glue.sh
  )"
  printf "bitnet_sft_budget_sweep\t%s\tbitnet_sft\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$TASK_NAME" "$SCALE_MODE" "$TERNARY_INIT_MODE" "$TERNARY_INIT_ITERATIONS" "$TERNARY_INIT_CALIBRATION_BATCHES" "$steps" "$lr" "$job_id" "$output_dir" \
    "$ACTIVATION_QUANTIZATION" "$USE_SUBLN" "$MAX_TRAIN_SAMPLES" "$MAX_EVAL_SAMPLES" \
    "$PER_DEVICE_BATCH_SIZE" "$GRAD_ACCUM_STEPS" "$SAVE_MODEL_ARTIFACTS" | tee -a "$JOB_TABLE"
}

for steps in "${STEPS_LIST[@]}"; do
  for lr in "${LRS[@]}"; do
    submit_job "$steps" "$lr"
  done
done

echo "wrote $JOB_TABLE"
