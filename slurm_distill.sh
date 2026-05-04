#!/bin/bash
#SBATCH --job-name=bitnet-qat-distill
#SBATCH --partition=dualcard
#SBATCH --nodelist=ece-nebula10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=28G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/slurm_nfs/a6abdulm/projects/BitNet/logs/%x-%j.out
#SBATCH --error=/mnt/slurm_nfs/a6abdulm/projects/BitNet/logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs checkpoints

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B}"
STUDENT_INIT_MODEL="${STUDENT_INIT_MODEL:-$MODEL}"
DATASET_NAME="${DATASET_NAME:-wikitext}"
DATASET_CONFIG="${DATASET_CONFIG:-wikitext-2-raw-v1}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
TEXT_COLUMN="${TEXT_COLUMN:-text}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
MAX_STEPS="${MAX_STEPS:-1000}"
NUM_TRAIN_SAMPLES="${NUM_TRAIN_SAMPLES:-10000}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
LR="${LR:-2e-5}"
LR_SCHEDULER="${LR_SCHEDULER:-constant}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
TEMPERATURE="${TEMPERATURE:-2.0}"
KL_WEIGHT="${KL_WEIGHT:-1.0}"
HIDDEN_MSE_WEIGHT="${HIDDEN_MSE_WEIGHT:-1.0}"
HIDDEN_STATE_LAYERS="${HIDDEN_STATE_LAYERS:-last}"
SCALE_MODE="${SCALE_MODE:-tensor}"
QUANT_EPS="${QUANT_EPS:-1e-5}"
EXCLUDE_LINEAR_REGEX="${EXCLUDE_LINEAR_REGEX:-}"
MAX_PACKED_BLOCKS="${MAX_PACKED_BLOCKS:-0}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-10}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-250}"
SAVE_FINAL="${SAVE_FINAL:-true}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
USE_FSDP="${USE_FSDP:-true}"
FSDP_MIXED_PRECISION="${FSDP_MIXED_PRECISION:-true}"
FSDP_CPU_OFFLOAD="${FSDP_CPU_OFFLOAD:-true}"
FSDP_WRAP_CLASS_NAMES="${FSDP_WRAP_CLASS_NAMES:-Qwen2DecoderLayer}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/qwen2.5-1.5b-wikitext2}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "MODEL=$MODEL"
echo "DATASET_NAME=$DATASET_NAME"
echo "DATASET_CONFIG=$DATASET_CONFIG"
echo "MAX_SEQ_LEN=$MAX_SEQ_LEN"
echo "PER_DEVICE_BATCH_SIZE=$PER_DEVICE_BATCH_SIZE"
echo "GRAD_ACCUM_STEPS=$GRAD_ACCUM_STEPS MAX_STEPS=$MAX_STEPS LR=$LR"
echo "LR_SCHEDULER=$LR_SCHEDULER WARMUP_STEPS=$WARMUP_STEPS MIN_LR_RATIO=$MIN_LR_RATIO"
echo "TEMPERATURE=$TEMPERATURE KL_WEIGHT=$KL_WEIGHT HIDDEN_MSE_WEIGHT=$HIDDEN_MSE_WEIGHT"
echo "HIDDEN_STATE_LAYERS=$HIDDEN_STATE_LAYERS SCALE_MODE=$SCALE_MODE QUANT_EPS=$QUANT_EPS"
echo "EXCLUDE_LINEAR_REGEX=$EXCLUDE_LINEAR_REGEX"
echo "NPROC_PER_NODE=$NPROC_PER_NODE USE_FSDP=$USE_FSDP FSDP_CPU_OFFLOAD=$FSDP_CPU_OFFLOAD"
echo "OUTPUT_DIR=$OUTPUT_DIR"

SAVE_FINAL_ARG=(--no-save-final)
if [ "$SAVE_FINAL" = "true" ]; then
  SAVE_FINAL_ARG=(--save-final)
fi

USE_FSDP_ARG=(--no-use-fsdp)
if [ "$USE_FSDP" = "true" ]; then
  USE_FSDP_ARG=(--use-fsdp)
fi

FSDP_MIXED_PRECISION_ARG=(--no-fsdp-mixed-precision)
if [ "$FSDP_MIXED_PRECISION" = "true" ]; then
  FSDP_MIXED_PRECISION_ARG=(--fsdp-mixed-precision)
fi

FSDP_CPU_OFFLOAD_ARG=(--no-fsdp-cpu-offload)
if [ "$FSDP_CPU_OFFLOAD" = "true" ]; then
  FSDP_CPU_OFFLOAD_ARG=(--fsdp-cpu-offload)
fi

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$NPROC_PER_NODE" \
  train_distill.py \
  --teacher-model "$MODEL" \
  --student-init-model "$STUDENT_INIT_MODEL" \
  --dataset-name "$DATASET_NAME" \
  --dataset-config "$DATASET_CONFIG" \
  --dataset-split "$DATASET_SPLIT" \
  --text-column "$TEXT_COLUMN" \
  --num-train-samples "$NUM_TRAIN_SAMPLES" \
  --max-packed-blocks "$MAX_PACKED_BLOCKS" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --per-device-batch-size "$PER_DEVICE_BATCH_SIZE" \
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  --max-steps "$MAX_STEPS" \
  --learning-rate "$LR" \
  --lr-scheduler "$LR_SCHEDULER" \
  --warmup-steps "$WARMUP_STEPS" \
  --min-lr-ratio "$MIN_LR_RATIO" \
  --temperature "$TEMPERATURE" \
  --kl-weight "$KL_WEIGHT" \
  --hidden-mse-weight "$HIDDEN_MSE_WEIGHT" \
  --hidden-state-layers "$HIDDEN_STATE_LAYERS" \
  --model-dtype bf16 \
  --master-weight-dtype fp32 \
  --scale-mode "$SCALE_MODE" \
  --quant-eps "$QUANT_EPS" \
  --exclude-linear-regex "$EXCLUDE_LINEAR_REGEX" \
  --gradient-checkpointing \
  "${USE_FSDP_ARG[@]}" \
  "${FSDP_MIXED_PRECISION_ARG[@]}" \
  "${FSDP_CPU_OFFLOAD_ARG[@]}" \
  --fsdp-wrap-class-names "$FSDP_WRAP_CLASS_NAMES" \
  --dataloader-num-workers "$DATALOADER_NUM_WORKERS" \
  --output-dir "$OUTPUT_DIR" \
  --save-every-steps "$SAVE_EVERY_STEPS" \
  --log-every-steps "$LOG_EVERY_STEPS" \
  "${SAVE_FINAL_ARG[@]}"
