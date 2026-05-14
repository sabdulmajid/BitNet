#!/bin/bash
# Submit a focused sequence-classification head-initialization probe.
#
# This is a diagnostic, not the primary paper baseline.  It copies the task
# head from an already trained FP16-SFT teacher before training the ternary
# student.  The goal is to separate ternary-backbone failure from a random
# sequence-classification head that simply did not receive enough task steps.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B}"
MODEL_SLUG="${MODEL//\//-}"
SOURCE_ROOT="${SOURCE_ROOT:-checkpoints/bitdistill-glue-seqcls}"
OUTPUT_ROOT="${OUTPUT_ROOT:-checkpoints/bitdistill-glue-seqcls-teacherhead}"
WARMUP_ROOT="${WARMUP_ROOT:-checkpoints/bitdistill-glue}"
WARMUP_STATE="${WARMUP_STATE:-$WARMUP_ROOT/$MODEL_SLUG/continued_pretrain/bitdistill-tensor/custom_state_dict.pt}"
TASKS=(${TASKS:-mnli})

TASK_MAX_STEPS="${TASK_MAX_STEPS:-1000}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
LR="${LR:-2e-5}"
LOGIT_KD_WEIGHT="${LOGIT_KD_WEIGHT:-10.0}"
ATTENTION_KD_WEIGHT="${ATTENTION_KD_WEIGHT:-100.0}"
LOGIT_TEMPERATURE="${LOGIT_TEMPERATURE:-5.0}"
ATTENTION_TEMPERATURE="${ATTENTION_TEMPERATURE:-1.0}"

if [ ! -s "$WARMUP_STATE" ]; then
  echo "missing warmup state: $WARMUP_STATE" >&2
  exit 1
fi

mkdir -p benchmark_results
JOB_TABLE="benchmark_results/bitdistill_seqcls_teacherhead_probe_$(date -u +%Y%m%d_%H%M%S).tsv"
printf "phase\ttask\tmethod\tscale\tlayer\tjob_id\tteacher\toutput_dir\n" > "$JOB_TABLE"

submit_probe() {
  local task="$1"
  local method="$2"
  local scale="$3"
  local teacher_dir="$4"
  local output_dir="$5"
  shift 5

  local job_id
  job_id="$(
    env \
      MODEL="$MODEL" \
      TASK_FORMAT=sequence_classification \
      LABEL_SCHEME=letters \
      CANDIDATE_SCORE=mean \
      TASK_NAME="$task" \
      METHOD="$method" \
      SCALE_MODE="$scale" \
      EXCLUDE_LINEAR_REGEX='score|classifier' \
      DISTILL_LAYER=-1 \
      MAX_STEPS="$TASK_MAX_STEPS" \
      PER_DEVICE_BATCH_SIZE="$PER_DEVICE_BATCH_SIZE" \
      EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
      GRAD_ACCUM_STEPS="$GRAD_ACCUM_STEPS" \
      LR="$LR" \
      LOGIT_KD_WEIGHT="$LOGIT_KD_WEIGHT" \
      ATTENTION_KD_WEIGHT="$ATTENTION_KD_WEIGHT" \
      LOGIT_TEMPERATURE="$LOGIT_TEMPERATURE" \
      ATTENTION_TEMPERATURE="$ATTENTION_TEMPERATURE" \
      INIT_OUTPUT_HEAD_FROM_TEACHER=1 \
      TEACHER_MODEL="$teacher_dir" \
      OUTPUT_ROOT="$OUTPUT_ROOT" \
      OUTPUT_DIR="$output_dir" \
      "$@" \
      sbatch --parsable slurm_bitdistill_glue.sh
  )"
  printf "teacher_head_probe\t%s\t%s\t%s\t-1\t%s\t%s\t%s\n" "$task" "$method" "$scale" "$job_id" "$teacher_dir" "$output_dir" | tee -a "$JOB_TABLE"
}

for task in "${TASKS[@]}"; do
  FP_DIR="$SOURCE_ROOT/$MODEL_SLUG/$task/fp16_sft-tensor-layer-1"
  if [ ! -s "$FP_DIR/metrics.json" ]; then
    echo "missing FP16 teacher metrics for $task: $FP_DIR/metrics.json" >&2
    exit 1
  fi

  submit_probe "$task" bitnet_sft tensor "$FP_DIR" "$OUTPUT_ROOT/$MODEL_SLUG/$task/bitnet_sft-headinit-tensor-layer-1"
  submit_probe "$task" bitdistill tensor "$FP_DIR" "$OUTPUT_ROOT/$MODEL_SLUG/$task/bitdistill-headinit-tensor-layer-1" \
    INIT_STATE_DICT="$WARMUP_STATE"
  submit_probe "$task" bitdistill row "$FP_DIR" "$OUTPUT_ROOT/$MODEL_SLUG/$task/bitdistill-headinit-row-layer-1" \
    INIT_STATE_DICT="$WARMUP_STATE"
done

echo "wrote $JOB_TABLE"
