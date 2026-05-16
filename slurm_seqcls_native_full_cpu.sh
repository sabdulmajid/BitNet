#!/bin/bash
#SBATCH --job-name=seqcls-native-full
#SBATCH --partition=midcard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/slurm_nfs/a6abdulm/projects/BitNet/logs/%x-%j.out
#SBATCH --error=/mnt/slurm_nfs/a6abdulm/projects/BitNet/logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs benchmark_results benchmarks/results

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

DATE="${BITNET_REPORT_DATE:-2026-05-15}"
export BITNET_REPORT_DATE="$DATE"

python benchmarks/benchmark_seqcls_native_i2sr_cpu.py \
  --task mnli \
  --max-samples 0 \
  --prompt-input token_ids \
  --prompt-batch-size 1 \
  --threads "${SLURM_CPUS_PER_TASK:-24}" \
  --batch-size 4096 \
  --ubatch-size 512 \
  --timeout-seconds 7200 \
  --progress-jsonl "benchmark_results/seqcls_native_i2sr_cpu_mnli_full_token_ids_${DATE}.progress.jsonl" \
  --resume-progress \
  --progress-every 64 \
  --output-json "benchmark_results/seqcls_native_i2sr_cpu_mnli_full_token_ids_${DATE}.json" \
  --output-md "benchmarks/results/seqcls_native_i2sr_cpu_mnli_full_token_ids_${DATE}.md"
