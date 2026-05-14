#!/bin/bash
#SBATCH --job-name=bitdistill-postprocess
#SBATCH --partition=dualcard
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

DATE="$(date -u +%F)"

python benchmarks/monitor_bitdistill_jobs.py \
  --output-json "benchmark_results/bitdistill_job_monitor_${DATE}.json" \
  --output-md "benchmarks/results/bitdistill_job_monitor_${DATE}.md"

python benchmarks/audit_bitdistill_dependency_graph.py \
  --output-json "benchmark_results/bitdistill_dependency_graph_${DATE}.json" \
  --output-md "benchmarks/results/bitdistill_dependency_graph_${DATE}.md"

python benchmarks/audit_bitdistill_warmup_health.py \
  --output-json "benchmark_results/bitdistill_warmup_health_${DATE}.json" \
  --output-md "benchmarks/results/bitdistill_warmup_health_${DATE}.md"

python benchmarks/audit_bitdistill_job_matrix.py \
  --monitor-json "benchmark_results/bitdistill_job_monitor_${DATE}.json" \
  --output-json "benchmark_results/bitdistill_job_matrix_audit_${DATE}.json" \
  --output-md "benchmarks/results/bitdistill_job_matrix_audit_${DATE}.md"

python benchmarks/run_bitdistill_smoke_contract.py \
  --work-dir "benchmark_results/bitdistill-smoke-contract-${DATE}" \
  --output-json "benchmark_results/bitdistill_smoke_contract_${DATE}.json" \
  --output-md "benchmarks/results/bitdistill_smoke_contract_${DATE}.md"

python benchmarks/gate_bitdistill_reproduction.py \
  --output-json "benchmark_results/bitdistill_reproduction_gate_${DATE}.json" \
  --output-md "benchmarks/results/bitdistill_reproduction_gate_${DATE}.md"

python benchmarks/audit_bitdistill_paper_alignment.py \
  --output-json "benchmark_results/bitdistill_paper_alignment_${DATE}.json" \
  --output-md "benchmarks/results/bitdistill_paper_alignment_${DATE}.md"

python benchmarks/audit_bitdistill_loss_scales.py \
  --output-json "benchmark_results/bitdistill_loss_scale_audit_${DATE}.json" \
  --output-md "benchmarks/results/bitdistill_loss_scale_audit_${DATE}.md"

python benchmarks/audit_tl2_row_scale_runtime_contract.py \
  --output-json "benchmark_results/tl2_row_scale_runtime_contract_${DATE}.json" \
  --output-md "benchmarks/results/tl2_row_scale_runtime_contract_${DATE}.md"

python benchmarks/summarize_bitdistill_variants.py \
  --roots \
    checkpoints/bitdistill-glue-seqcls \
    checkpoints/bitdistill-glue-seqcls-paperlogit \
    checkpoints/bitdistill-glue-seqcls-ablate \
    checkpoints/bitdistill-glue-seqcls-longwarmup \
    checkpoints/bitdistill-glue-seqcls-longwarmup-headinit \
    checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma \
    checkpoints/bitdistill-glue-seqcls-longwarmup-gamma1k \
    checkpoints/bitdistill-glue-seqcls-longwarmup-gamma10k \
    checkpoints/bitdistill-glue-seqcls-longwarmup-layer-sweep \
  --title "BitDistill Variant Summary, ${DATE}" \
  --output-json "benchmark_results/bitdistill_variant_summary_${DATE}.json" \
  --output-md "benchmarks/results/bitdistill_variant_summary_${DATE}.md"

python benchmarks/summarize_bitdistill_glue.py \
  --root checkpoints/bitdistill-glue-causal-longwarmup-densehead \
  --fp-root checkpoints/bitdistill-glue \
  --bitnet-root checkpoints/bitdistill-glue \
  --tasks mnli qnli sst2 \
  --bitdistill-tensor-template bitdistill-longwarmup-tensor-layer-8 \
  --bitdistill-row-template bitdistill-longwarmup-row-layer-8 \
  --output-json "benchmark_results/bitdistill_causal_longwarmup_densehead_summary_${DATE}.json" \
  --output-md "benchmarks/results/bitdistill_causal_longwarmup_densehead_summary_${DATE}.md"

python benchmarks/gate_bitdistill_cpu_benchmark.py \
  --input-json "benchmark_results/bitdistill_glue_cpu_latest.json" \
  --output-json "benchmark_results/bitdistill_glue_cpu_gate_${DATE}.json" \
  --output-md "benchmarks/results/bitdistill_glue_cpu_gate_${DATE}.md"

python benchmarks/gate_bitdistill_i2sr_export.py \
  --results-dir "benchmark_results/bitdistill-causal-longwarmup-i2sr-${DATE}" \
  --output-json "benchmark_results/bitdistill_i2sr_export_gate_${DATE}.json" \
  --output-md "benchmarks/results/bitdistill_i2sr_export_gate_${DATE}.md"

python benchmarks/run_tiny_qwen2moe_fixture.py \
  --skip-existing \
  --output-json "benchmark_results/tiny_qwen2moe_fixture_${DATE}.json" \
  --output-md "benchmarks/results/tiny_qwen2moe_fixture_${DATE}.md"

python benchmarks/audit_moe_support.py \
  --output-json "benchmark_results/moe_support_audit_${DATE}.json" \
  --output-md "benchmarks/results/moe_support_audit_${DATE}.md"

python benchmarks/audit_unblock_requirements.py \
  --output-json "benchmark_results/unblock_requirements_${DATE}.json" \
  --output-md "benchmarks/results/unblock_requirements_${DATE}.md"

python benchmarks/audit_objective_completion.py \
  --output-json "benchmark_results/objective_completion_audit_${DATE}.json" \
  --output-md "benchmarks/results/objective_completion_audit_${DATE}.md"

python benchmarks/audit_product_scope.py \
  --output-json "benchmark_results/product_scope_gate_${DATE}.json" \
  --output-md "benchmarks/results/product_scope_gate_${DATE}.md"

python benchmarks/audit_bitdistill_active_goal.py \
  --output-json "benchmark_results/bitdistill_active_goal_audit_${DATE}.json" \
  --output-md "benchmarks/results/bitdistill_active_goal_audit_${DATE}.md"

python benchmarks/build_qwen_side_by_side.py \
  --output-md "benchmarks/results/qwen_side_by_side_${DATE}.md"

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
