#!/bin/bash
#SBATCH --job-name=bitdistill-ctrl-post
#SBATCH --partition=midcard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/slurm_nfs/a6abdulm/projects/BitNet/logs/%x-%j.out
#SBATCH --error=/mnt/slurm_nfs/a6abdulm/projects/BitNet/logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs benchmark_results benchmarks/results

DATE="${BITNET_REPORT_DATE:-$(date -u +%F)}"
export BITNET_REPORT_DATE="$DATE"

python benchmarks/audit_bitnet_sft_budget_sweep.py
python benchmarks/audit_bitnet_sft_budget_paired.py
python benchmarks/audit_bitnet_sft_recipe_alignment.py
python benchmarks/audit_bitnet_sft_baseline.py
python benchmarks/audit_bitnet_sft_mechanics.py

python benchmarks/audit_bitdistill_recovery_run.py
python benchmarks/audit_bitdistill_controlled_curve.py
python benchmarks/audit_bitdistill_stage2_curve.py
python benchmarks/audit_bitdistill_loss_scales.py
python benchmarks/audit_bitdistill_telemetry_coverage.py
python benchmarks/audit_bitdistill_root_cause.py

python benchmarks/audit_product_scope.py
python benchmarks/audit_objective_completion.py

python benchmarks/build_evidence_manifest.py \
  --allow-missing-label benchmark_coverage_gate_report \
  --allow-missing-label benchmark_coverage_gate_json \
  --output-json "benchmarks/results/evidence_manifest_${DATE}.json" \
  --output-md "benchmarks/results/evidence_manifest_${DATE}.md"

python benchmarks/audit_benchmark_coverage.py \
  --manifest-path "benchmarks/results/evidence_manifest_${DATE}.json" \
  --output-json "benchmark_results/benchmark_coverage_gate_${DATE}.json" \
  --output-md "benchmarks/results/benchmark_coverage_gate_${DATE}.md"

python benchmarks/build_evidence_manifest.py \
  --output-json "benchmarks/results/evidence_manifest_${DATE}.json" \
  --output-md "benchmarks/results/evidence_manifest_${DATE}.md"
