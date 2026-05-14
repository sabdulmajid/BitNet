#!/bin/bash
# Submit the BitDistill GLUE reproduction and row-scale novelty matrix.
#
# This script deliberately uses Slurm dependencies:
# - Stage-2 continued pretraining runs first.
# - Each BitDistill task run waits for both Stage-2 and its FP16-SFT teacher.
# - BitNet-SFT baselines can run independently of the teacher.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B}"
MODEL_SLUG="${MODEL//\//-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-checkpoints/bitdistill-glue}"
TASKS=(${TASKS:-mnli qnli sst2})
SWEEP_TASK="${SWEEP_TASK:-mnli}"
SWEEP_LAYERS=(${SWEEP_LAYERS:--2 -4})

CONTINUED_PRETRAIN_STEPS="${CONTINUED_PRETRAIN_STEPS:-5000}"
TASK_MAX_STEPS="${TASK_MAX_STEPS:-1000}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
LR="${LR:-2e-5}"
TASK_FORMAT="${TASK_FORMAT:-causal_lm}"

mkdir -p benchmark_results
JOB_TABLE="benchmark_results/bitdistill_glue_jobs_$(date -u +%Y%m%d_%H%M%S).tsv"
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
      TASK_NAME="$task" \
      METHOD="$method" \
      SCALE_MODE="$scale" \
      DISTILL_LAYER="$layer" \
      MAX_STEPS="$TASK_MAX_STEPS" \
      MAX_TRAIN_SAMPLES="$MAX_TRAIN_SAMPLES" \
      MAX_EVAL_SAMPLES="$MAX_EVAL_SAMPLES" \
      PER_DEVICE_BATCH_SIZE="$PER_DEVICE_BATCH_SIZE" \
      EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
      GRAD_ACCUM_STEPS="$GRAD_ACCUM_STEPS" \
      LR="$LR" \
      "$@" \
      sbatch "${sbatch_args[@]}" slurm_bitdistill_glue.sh
  )"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$phase" "$task" "$method" "$scale" "$layer" "$job_id" "${dependency:-none}" "$output_dir" | tee -a "$JOB_TABLE"
}

submit_warmup() {
  local output_dir="$OUTPUT_ROOT/$MODEL_SLUG/continued_pretrain/bitdistill-tensor"
  local job_id
  job_id="$(
    env \
      MODEL="$MODEL" \
      STAGE=continued_pretrain \
      METHOD=bitdistill \
      SCALE_MODE=tensor \
      MAX_STEPS="$CONTINUED_PRETRAIN_STEPS" \
      PER_DEVICE_BATCH_SIZE="$PER_DEVICE_BATCH_SIZE" \
      GRAD_ACCUM_STEPS="$GRAD_ACCUM_STEPS" \
      LR="$LR" \
      OUTPUT_DIR="$output_dir" \
      sbatch --parsable slurm_bitdistill_glue.sh
  )"
  printf "stage2\t-\tbitdistill\ttensor\t-\t%s\tnone\t%s\n" "$job_id" "$output_dir" | tee -a "$JOB_TABLE"
  echo "$job_id"
}

WARMUP_JOB="$(submit_warmup | tail -n 1)"
WARMUP_STATE="$OUTPUT_ROOT/$MODEL_SLUG/continued_pretrain/bitdistill-tensor/custom_state_dict.pt"

for task in "${TASKS[@]}"; do
  FP_DIR="$OUTPUT_ROOT/$MODEL_SLUG/$task/fp16_sft-tensor-layer-1"
  BITNET_DIR="$OUTPUT_ROOT/$MODEL_SLUG/$task/bitnet_sft-tensor-layer-1"
  BD_TENSOR_DIR="$OUTPUT_ROOT/$MODEL_SLUG/$task/bitdistill-tensor-layer-1"
  BD_ROW_DIR="$OUTPUT_ROOT/$MODEL_SLUG/$task/bitdistill-row-layer-1"

  FP_JOB="$(
    submit_job paper_baseline "$task" fp16_sft tensor -1 "" "$FP_DIR" | tail -n 1 | cut -f6
  )"
  submit_job paper_baseline "$task" bitnet_sft tensor -1 "" "$BITNET_DIR" >/dev/null
  submit_job paper_baseline "$task" bitdistill tensor -1 "afterok:${WARMUP_JOB}:${FP_JOB}" "$BD_TENSOR_DIR" \
    TEACHER_MODEL="$FP_DIR" INIT_STATE_DICT="$WARMUP_STATE" >/dev/null
  submit_job novelty_row_scale "$task" bitdistill row -1 "afterok:${WARMUP_JOB}:${FP_JOB}" "$BD_ROW_DIR" \
    TEACHER_MODEL="$FP_DIR" INIT_STATE_DICT="$WARMUP_STATE" >/dev/null
done

FP_SWEEP_DIR="$OUTPUT_ROOT/$MODEL_SLUG/$SWEEP_TASK/fp16_sft-tensor-layer-1"
FP_SWEEP_JOB="$(awk -F '\t' -v task="$SWEEP_TASK" '$1 == "paper_baseline" && $2 == task && $3 == "fp16_sft" { print $6 }' "$JOB_TABLE" | tail -n 1)"
for layer in "${SWEEP_LAYERS[@]}"; do
  SWEEP_DIR="$OUTPUT_ROOT/$MODEL_SLUG/$SWEEP_TASK/bitdistill-tensor-layer${layer}"
  submit_job attention_layer_sweep "$SWEEP_TASK" bitdistill tensor "$layer" "afterok:${WARMUP_JOB}:${FP_SWEEP_JOB}" "$SWEEP_DIR" \
    TEACHER_MODEL="$FP_SWEEP_DIR" INIT_STATE_DICT="$WARMUP_STATE" >/dev/null
done

echo "wrote $JOB_TABLE"
