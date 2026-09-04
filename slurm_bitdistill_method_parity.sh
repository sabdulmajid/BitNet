#!/bin/bash
#SBATCH --job-name=bd-parity
#SBATCH --partition=dualcard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=0-5%1
#SBATCH --output=/mnt/slurm_nfs/a6abdulm/projects/BitNet/logs/%x-%A_%a.out
#SBATCH --error=/mnt/slurm_nfs/a6abdulm/projects/BitNet/logs/%x-%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

case_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
case "$case_id" in
  0)
    RUN_NAME=seqcls-cosine-s8-fixed
    TASK_FORMAT=sequence_classification
    ATTENTION_RELATION_MODE=cosine
    ATTENTION_SPLIT_HEADS=8
    ATTENTION_KD_BALANCE=fixed
    TEACHER_MODEL=checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1
    ;;
  1)
    RUN_NAME=seqcls-cosine-s1-fixed
    TASK_FORMAT=sequence_classification
    ATTENTION_RELATION_MODE=cosine
    ATTENTION_SPLIT_HEADS=1
    ATTENTION_KD_BALANCE=fixed
    TEACHER_MODEL=checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1
    ;;
  2)
    RUN_NAME=seqcls-scaled-dot-s1-fixed
    TASK_FORMAT=sequence_classification
    ATTENTION_RELATION_MODE=scaled_dot
    ATTENTION_SPLIT_HEADS=1
    ATTENTION_KD_BALANCE=fixed
    TEACHER_MODEL=checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1
    ;;
  3)
    RUN_NAME=seqcls-cosine-s1-adaptive
    TASK_FORMAT=sequence_classification
    ATTENTION_RELATION_MODE=cosine
    ATTENTION_SPLIT_HEADS=1
    ATTENTION_KD_BALANCE=gradnorm_ema
    TEACHER_MODEL=checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1
    ;;
  4)
    RUN_NAME=causal-cosine-s1-fixed
    TASK_FORMAT=causal_lm
    ATTENTION_RELATION_MODE=cosine
    ATTENTION_SPLIT_HEADS=1
    ATTENTION_KD_BALANCE=fixed
    TEACHER_MODEL=checkpoints/bitdistill-glue-letters/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1
    ;;
  5)
    RUN_NAME=causal-cosine-s1-adaptive
    TASK_FORMAT=causal_lm
    ATTENTION_RELATION_MODE=cosine
    ATTENTION_SPLIT_HEADS=1
    ATTENTION_KD_BALANCE=gradnorm_ema
    TEACHER_MODEL=checkpoints/bitdistill-glue-letters/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1
    ;;
  *)
    echo "Unsupported parity case: $case_id" >&2
    exit 2
    ;;
esac

export MODEL=Qwen/Qwen2.5-0.5B
export STAGE=task_sft
export METHOD=bitdistill
export TASK_NAME=mnli
export TASK_FORMAT
export LABEL_SCHEME=letters
export CANDIDATE_SCORE=mean
export TEACHER_MODEL
export INIT_STATE_MANIFEST=benchmarks/results/stage2_manifest_655m_2026-05-23.json
export SCALE_MODE=tensor
export EXCLUDE_LINEAR_REGEX='score|classifier'
export DISTILL_LAYER=-1
export ATTENTION_RELATION_MODE
export ATTENTION_SPLIT_HEADS
export ATTENTION_KD_BALANCE
export ATTENTION_KD_WEIGHT=100000
export ATTENTION_BALANCE_TARGET_RATIO=1.0
export ATTENTION_BALANCE_BETA=0.9
export ATTENTION_BALANCE_EVERY_STEPS=20
export ATTENTION_BALANCE_MIN_WEIGHT=0.001
export ATTENTION_BALANCE_MAX_WEIGHT=100000
export ACTIVATION_QUANTIZATION=1
export USE_SUBLN=1
export LOGIT_KD_WEIGHT=10
export LOGIT_TEMPERATURE=5.0
export LOGIT_KD_TEMPERATURE_SCALE=none
export ATTENTION_TEMPERATURE=1.0
export INIT_OUTPUT_HEAD_FROM_TEACHER=1
export MAX_SEQ_LEN=512
export MAX_STEPS=120
export MAX_TRAIN_SAMPLES=8192
export MAX_EVAL_SAMPLES=512
export PER_DEVICE_BATCH_SIZE=4
export EVAL_BATCH_SIZE=16
export GRAD_ACCUM_STEPS=4
export LR=2e-5
export LR_SCHEDULER=cosine
export WARMUP_STEPS=100
export TELEMETRY_EVERY_STEPS=20
export TELEMETRY_COMPONENT_GRAD_NORMS=1
export SAVE_EVERY_STEPS=0
export SAVE_MODEL_ARTIFACTS=0
export OUTPUT_DIR="checkpoints/bitdistill-method-parity/${RUN_NAME}"

echo "METHOD_PARITY_CASE=$case_id RUN_NAME=$RUN_NAME"
source_revision="$(git rev-parse HEAD)"
if [ -n "${EXPECTED_SOURCE_REVISION:-}" ] && [ "$source_revision" != "$EXPECTED_SOURCE_REVISION" ]; then
  echo "Source revision mismatch: expected $EXPECTED_SOURCE_REVISION, found $source_revision" >&2
  exit 2
fi
echo "SOURCE_REVISION=$source_revision"
exec bash slurm_bitdistill_glue.sh
