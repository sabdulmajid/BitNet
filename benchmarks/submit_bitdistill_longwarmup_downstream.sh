#!/bin/bash
# Submit downstream BitDistill jobs that depend on a longer Stage-2 warm-up.
#
# This launcher intentionally does not resubmit FP16-SFT or BitNet-SFT.  It
# consumes already completed FP16 teachers from SOURCE_ROOT and a future or
# existing continued-pretraining checkpoint from WARMUP_STATE.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B}"
MODEL_SLUG="${MODEL//\//-}"
SOURCE_ROOT="${SOURCE_ROOT:-checkpoints/bitdistill-glue-seqcls}"
OUTPUT_ROOT="${OUTPUT_ROOT:-checkpoints/bitdistill-glue-seqcls-longwarmup}"
WARMUP_ROOT="${WARMUP_ROOT:-checkpoints/bitdistill-glue-longwarmup}"
WARMUP_NAME="${WARMUP_NAME:-bitdistill-tensor-20k}"
WARMUP_STATE="${WARMUP_STATE:-$WARMUP_ROOT/$MODEL_SLUG/continued_pretrain/$WARMUP_NAME/custom_state_dict.pt}"

TASKS=(${TASKS:-mnli qnli sst2})
SCALE_MODES=(${SCALE_MODES:-tensor row})
DISTILL_LAYERS=(${DISTILL_LAYERS:--8})
DEPENDENCY="${DEPENDENCY:-}"

TASK_MAX_STEPS="${TASK_MAX_STEPS:-1000}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
LR="${LR:-2e-5}"
TASK_FORMAT="${TASK_FORMAT:-sequence_classification}"
LABEL_SCHEME="${LABEL_SCHEME:-letters}"
CANDIDATE_SCORE="${CANDIDATE_SCORE:-mean}"
EXCLUDE_LINEAR_REGEX="${EXCLUDE_LINEAR_REGEX:-score|classifier}"
LOGIT_KD_WEIGHT="${LOGIT_KD_WEIGHT:-10.0}"
ATTENTION_KD_WEIGHT="${ATTENTION_KD_WEIGHT:-100.0}"
LOGIT_TEMPERATURE="${LOGIT_TEMPERATURE:-5.0}"
LOGIT_KD_TEMPERATURE_SCALE="${LOGIT_KD_TEMPERATURE_SCALE:-none}"
ATTENTION_TEMPERATURE="${ATTENTION_TEMPERATURE:-1.0}"
INIT_OUTPUT_HEAD_FROM_TEACHER="${INIT_OUTPUT_HEAD_FROM_TEACHER:-0}"
SBATCH_PARTITION="${SBATCH_PARTITION:-}"
SBATCH_MEM="${SBATCH_MEM:-}"

if [ -z "$DEPENDENCY" ] && [ ! -s "$WARMUP_STATE" ]; then
  echo "missing warmup state and no DEPENDENCY was provided: $WARMUP_STATE" >&2
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
JOB_TABLE="benchmark_results/bitdistill_longwarmup_downstream_$(date -u +%Y%m%d_%H%M%S).tsv"
printf "phase\ttask\tmethod\tscale\tlayer\tjob_id\tdependency\tteacher\twarmup_state\toutput_dir\ttask_max_steps\tmax_train_samples\tmax_eval_samples\tper_device_batch_size\tgrad_accum_steps\tlr\tlogit_kd_weight\tattention_kd_weight\tlogit_temperature\tlogit_kd_temperature_scale\tattention_temperature\tinit_output_head_from_teacher\n" > "$JOB_TABLE"

submit_job() {
  local task="$1"
  local scale="$2"
  local layer="$3"
  local teacher_dir="$4"
  local output_dir="$5"

  local sbatch_args=(--parsable)
  while IFS= read -r arg; do
    [ -n "$arg" ] && sbatch_args+=("$arg")
  done < <(common_sbatch_args)
  if [ -n "$DEPENDENCY" ]; then
    sbatch_args+=(--dependency="$DEPENDENCY")
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
      METHOD=bitdistill \
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
      LOGIT_KD_TEMPERATURE_SCALE="$LOGIT_KD_TEMPERATURE_SCALE" \
      ATTENTION_TEMPERATURE="$ATTENTION_TEMPERATURE" \
      INIT_OUTPUT_HEAD_FROM_TEACHER="$INIT_OUTPUT_HEAD_FROM_TEACHER" \
      TEACHER_MODEL="$teacher_dir" \
      INIT_STATE_DICT="$WARMUP_STATE" \
      sbatch "${sbatch_args[@]}" slurm_bitdistill_glue.sh
  )"
  printf "longwarmup_downstream\t%s\tbitdistill\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$task" "$scale" "$layer" "$job_id" "${DEPENDENCY:-none}" "$teacher_dir" "$WARMUP_STATE" "$output_dir" \
    "$TASK_MAX_STEPS" "$MAX_TRAIN_SAMPLES" "$MAX_EVAL_SAMPLES" "$PER_DEVICE_BATCH_SIZE" "$GRAD_ACCUM_STEPS" "$LR" \
    "$LOGIT_KD_WEIGHT" "$ATTENTION_KD_WEIGHT" "$LOGIT_TEMPERATURE" "$LOGIT_KD_TEMPERATURE_SCALE" "$ATTENTION_TEMPERATURE" \
    "$INIT_OUTPUT_HEAD_FROM_TEACHER" | tee -a "$JOB_TABLE"
}

for task in "${TASKS[@]}"; do
  teacher_dir="$SOURCE_ROOT/$MODEL_SLUG/$task/fp16_sft-tensor-layer-1"
  if [ ! -s "$teacher_dir/metrics.json" ]; then
    echo "missing FP16 teacher metrics for $task: $teacher_dir/metrics.json" >&2
    exit 1
  fi
  for scale in "${SCALE_MODES[@]}"; do
    for layer in "${DISTILL_LAYERS[@]}"; do
      safe_layer="${layer#-}"
      output_dir="$OUTPUT_ROOT/$MODEL_SLUG/$task/bitdistill-longwarmup-${scale}-layer-${safe_layer}"
      submit_job "$task" "$scale" "$layer" "$teacher_dir" "$output_dir"
    done
  done
done

echo "wrote $JOB_TABLE"
