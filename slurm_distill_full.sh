#!/bin/bash
#SBATCH --job-name=bitnet-qat-full
#SBATCH --partition=dualcard
#SBATCH --nodelist=ece-nebula10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=12:00:00
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
DATASET_NAME="${DATASET_NAME:-HuggingFaceFW/fineweb-edu}"
DATASET_CONFIG="${DATASET_CONFIG:-sample-10BT}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
TEXT_COLUMN="${TEXT_COLUMN:-text}"
NUM_TRAIN_SAMPLES="${NUM_TRAIN_SAMPLES:-50000}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
MAX_STEPS="${MAX_STEPS:-5000}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
LR="${LR:-1e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-10}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-1000}"
SAVE_FINAL="${SAVE_FINAL:-true}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/qwen2.5-1.5b-fineweb-edu}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "MODEL=$MODEL"
echo "DATASET_NAME=$DATASET_NAME"
echo "DATASET_CONFIG=$DATASET_CONFIG"
echo "NUM_TRAIN_SAMPLES=$NUM_TRAIN_SAMPLES"
echo "MAX_SEQ_LEN=$MAX_SEQ_LEN"
echo "MAX_STEPS=$MAX_STEPS"
echo "PER_DEVICE_BATCH_SIZE=$PER_DEVICE_BATCH_SIZE"
echo "GRAD_ACCUM_STEPS=$GRAD_ACCUM_STEPS"
echo "LR=$LR"
echo "WARMUP_STEPS=$WARMUP_STEPS"
echo "MIN_LR_RATIO=$MIN_LR_RATIO"
echo "SAVE_EVERY_STEPS=$SAVE_EVERY_STEPS"
echo "OUTPUT_DIR=$OUTPUT_DIR"

SAVE_FINAL_ARG=(--no-save-final)
if [ "$SAVE_FINAL" = "true" ]; then
  SAVE_FINAL_ARG=(--save-final)
fi

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=2 \
  train_distill.py \
  --teacher-model "$MODEL" \
  --student-init-model "$STUDENT_INIT_MODEL" \
  --dataset-name "$DATASET_NAME" \
  --dataset-config "$DATASET_CONFIG" \
  --dataset-split "$DATASET_SPLIT" \
  --dataset-streaming \
  --text-column "$TEXT_COLUMN" \
  --num-train-samples "$NUM_TRAIN_SAMPLES" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --per-device-batch-size "$PER_DEVICE_BATCH_SIZE" \
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  --max-steps "$MAX_STEPS" \
  --learning-rate "$LR" \
  --lr-scheduler cosine \
  --warmup-steps "$WARMUP_STEPS" \
  --min-lr-ratio "$MIN_LR_RATIO" \
  --temperature 2.0 \
  --kl-weight 1.0 \
  --hidden-mse-weight 1.0 \
  --hidden-state-layers last \
  --model-dtype bf16 \
  --master-weight-dtype fp32 \
  --scale-mode tensor \
  --gradient-checkpointing \
  --use-fsdp \
  --fsdp-mixed-precision \
  --fsdp-cpu-offload \
  --fsdp-wrap-class-names Qwen2DecoderLayer \
  --dataloader-num-workers "$DATALOADER_NUM_WORKERS" \
  --output-dir "$OUTPUT_DIR" \
  --save-every-steps "$SAVE_EVERY_STEPS" \
  --log-every-steps "$LOG_EVERY_STEPS" \
  "${SAVE_FINAL_ARG[@]}"
