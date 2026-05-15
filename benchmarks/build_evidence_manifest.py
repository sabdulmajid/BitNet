#!/usr/bin/env python3
"""Build a compact manifest for cited benchmark evidence artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SELECTED_LM_EVAL_METRICS = {
    "arc_challenge": "acc_norm",
    "arc_easy": "acc_norm",
    "hellaswag": "acc_norm",
    "piqa": "acc_norm",
    "winogrande": "acc",
    "boolq": "acc",
    "copa": "acc",
    "openbookqa": "acc_norm",
    "sciq": "acc_norm",
    "truthfulqa_mc1": "acc",
}


CATASTROPHIC_PPL_THRESHOLD = 1.0e4
DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()


ARTIFACTS: list[dict[str, str]] = [
    # Tracked reports.
    {"label": "README", "kind": "tracked_report", "path": "README.md"},
    {"label": "research_redirect_report", "kind": "tracked_report", "path": f"benchmarks/results/research_redirect_{DATE}.md"},
    {"label": "side_by_side_report", "kind": "tracked_report", "path": f"benchmarks/results/qwen_side_by_side_{DATE}.md"},
    {"label": "paired_row_minus_fp_report", "kind": "tracked_report", "path": "benchmarks/results/paired_row_densehead_minus_fp_2026-05-13.md"},
    {"label": "paired_row_minus_ptq_report", "kind": "tracked_report", "path": "benchmarks/results/paired_row_densehead_minus_ptq_2026-05-13.md"},
    {"label": "publishable_claims", "kind": "tracked_report", "path": "benchmarks/results/publishable_claims_2026-05-05.md"},
    {"label": "progress_audit", "kind": "tracked_report", "path": "benchmarks/results/progress_audit_2026-05-05.md"},
    {"label": "active_goal_audit", "kind": "tracked_report", "path": "benchmarks/results/active_goal_completion_audit_2026-05-05.md"},
    {"label": "objective_completion_audit", "kind": "tracked_report", "path": f"benchmarks/results/objective_completion_audit_{DATE}.md"},
    {"label": "product_scope_gate", "kind": "tracked_report", "path": f"benchmarks/results/product_scope_gate_{DATE}.md"},
    {"label": "bitdistill_reproduction_status", "kind": "tracked_report", "path": "benchmarks/results/bitdistill_reproduction_status_2026-05-14.md"},
    {"label": "bitdistill_reproduction_gap_analysis", "kind": "tracked_report", "path": f"benchmarks/results/bitdistill_reproduction_gap_analysis_{DATE}.md"},
    {"label": "bitdistill_root_cause_audit_report", "kind": "tracked_report", "path": f"benchmarks/results/bitdistill_root_cause_audit_{DATE}.md"},
    {"label": "bitdistill_recovery_submission_report", "kind": "tracked_report", "path": f"benchmarks/results/bitdistill_recovery_submission_{DATE}.md"},
    {"label": "bitdistill_recovery_audit_report", "kind": "tracked_report", "path": f"benchmarks/results/bitdistill_recovery_audit_{DATE}.md"},
    {"label": "bitdistill_stage2_curve_report", "kind": "tracked_report", "path": f"benchmarks/results/bitdistill_stage2_curve_{DATE}.md"},
    {"label": "bitdistill_stage2_curve_submission_report", "kind": "tracked_report", "path": f"benchmarks/results/bitdistill_stage2_curve_submission_{DATE}.md"},
    {"label": "bitdistill_controlled_curve_report", "kind": "tracked_report", "path": f"benchmarks/results/bitdistill_controlled_curve_{DATE}.md"},
    {"label": "bitdistill_reproduction_gate_report", "kind": "tracked_report", "path": f"benchmarks/results/bitdistill_reproduction_gate_{DATE}.md"},
    {"label": "bitdistill_paired_predictions_report", "kind": "tracked_report", "path": f"benchmarks/results/bitdistill_paired_predictions_{DATE}.md"},
    {"label": "bitdistill_task_formulation_report", "kind": "tracked_report", "path": f"benchmarks/results/bitdistill_task_formulation_audit_{DATE}.md"},
    {"label": "bitnet_sft_baseline_audit_report", "kind": "tracked_report", "path": f"benchmarks/results/bitnet_sft_baseline_audit_{DATE}.md"},
    {"label": "bitnet_sft_recipe_alignment_report", "kind": "tracked_report", "path": f"benchmarks/results/bitnet_sft_recipe_alignment_{DATE}.md"},
    {"label": "bitnet_sft_mechanics_audit_report", "kind": "tracked_report", "path": f"benchmarks/results/bitnet_sft_mechanics_audit_{DATE}.md"},
    {"label": "bitnet_sft_budget_sweep_report", "kind": "tracked_report", "path": f"benchmarks/results/bitnet_sft_budget_sweep_{DATE}.md"},
    {"label": "bitnet_sft_budget_paired_report", "kind": "tracked_report", "path": f"benchmarks/results/bitnet_sft_budget_paired_{DATE}.md"},
    {"label": "subln_activation_variance_report", "kind": "tracked_report", "path": f"benchmarks/results/subln_activation_variance_{DATE}.md"},
    {"label": "bitdistill_paper_alignment_report", "kind": "tracked_report", "path": f"benchmarks/results/bitdistill_paper_alignment_{DATE}.md"},
    {"label": "bitdistill_loss_scale_report", "kind": "tracked_report", "path": f"benchmarks/results/bitdistill_loss_scale_audit_{DATE}.md"},
    {"label": "bitdistill_cpu_benchmark_report", "kind": "tracked_report", "path": f"benchmarks/results/bitdistill_glue_cpu_{DATE}.md"},
    {"label": "bitdistill_cpu_gate_report", "kind": "tracked_report", "path": f"benchmarks/results/bitdistill_glue_cpu_gate_{DATE}.md"},
    {"label": "bitdistill_cpu_xeon_benchmark_report", "kind": "tracked_report", "path": f"benchmarks/results/bitdistill_glue_cpu_xeon_{DATE}.md"},
    {"label": "bitdistill_cpu_xeon_gate_report", "kind": "tracked_report", "path": f"benchmarks/results/bitdistill_glue_cpu_xeon_gate_{DATE}.md"},
    {"label": "cpu_tradeoff_frontier_report", "kind": "tracked_report", "path": f"benchmarks/results/cpu_tradeoff_frontier_{DATE}.md"},
    {"label": "cpu_speed_uncertainty_report", "kind": "tracked_report", "path": f"benchmarks/results/cpu_speed_uncertainty_{DATE}.md"},
    {"label": "bitdistill_i2sr_gate_report", "kind": "tracked_report", "path": f"benchmarks/results/bitdistill_i2sr_export_gate_{DATE}.md"},
    {"label": "bitdistill_i2sr_local_gate_report", "kind": "tracked_report", "path": f"benchmarks/results/bitdistill_i2sr_export_gate_local_{DATE}.md"},
    {"label": "bitdistill_job_monitor_report", "kind": "tracked_report", "path": f"benchmarks/results/bitdistill_job_monitor_{DATE}.md"},
    {"label": "bitdistill_dependency_graph_report", "kind": "tracked_report", "path": f"benchmarks/results/bitdistill_dependency_graph_{DATE}.md"},
    {"label": "bitdistill_postprocess_submission_report", "kind": "tracked_report", "path": f"benchmarks/results/bitdistill_postprocess_submission_{DATE}.md"},
    {"label": "bitdistill_postprocess_dependency_report", "kind": "tracked_report", "path": f"benchmarks/results/bitdistill_postprocess_dependency_audit_{DATE}.md"},
    {"label": "bitdistill_afterany_postprocess_report", "kind": "tracked_report", "path": f"benchmarks/results/bitdistill_afterany_postprocess_{DATE}.md"},
    {"label": "bitdistill_afterany_postprocess_dependency_report", "kind": "tracked_report", "path": f"benchmarks/results/bitdistill_afterany_postprocess_dependency_{DATE}.md"},
    {"label": "bitdistill_warmup_health_report", "kind": "tracked_report", "path": f"benchmarks/results/bitdistill_warmup_health_{DATE}.md"},
    {"label": "bitdistill_warmup_finalizer_submission_report", "kind": "tracked_report", "path": f"benchmarks/results/bitdistill_warmup_finalizer_submission_{DATE}.md"},
    {"label": "bitdistill_warmup_finalizer_report", "kind": "tracked_report", "path": f"benchmarks/results/bitdistill_warmup_finalizer_{DATE}.md"},
    {"label": "bitdistill_producer_script_audit_report", "kind": "tracked_report", "path": f"benchmarks/results/bitdistill_producer_script_audit_{DATE}.md"},
    {"label": "bitdistill_job_matrix_report", "kind": "tracked_report", "path": f"benchmarks/results/bitdistill_job_matrix_audit_{DATE}.md"},
    {"label": "bitdistill_active_goal_report", "kind": "tracked_report", "path": f"benchmarks/results/bitdistill_active_goal_audit_{DATE}.md"},
    {"label": "bitdistill_snapshot_integrity_report", "kind": "tracked_report", "path": f"benchmarks/results/bitdistill_snapshot_integrity_{DATE}.md"},
    {"label": "bitdistill_smoke_contract_report", "kind": "tracked_report", "path": f"benchmarks/results/bitdistill_smoke_contract_{DATE}.md"},
    {"label": "bitdistill_variant_summary_report", "kind": "tracked_report", "path": f"benchmarks/results/bitdistill_variant_summary_{DATE}.md"},
    {"label": "bitdistill_rowwarmup_variant_summary_report", "kind": "tracked_report", "path": f"benchmarks/results/bitdistill_rowwarmup_variant_summary_{DATE}.md"},
    {"label": "bitdistill_causal_longwarmup_report", "kind": "tracked_report", "path": f"benchmarks/results/bitdistill_causal_longwarmup_densehead_summary_{DATE}.md"},
    {"label": "bitdistill_glue3_summary_report", "kind": "tracked_report", "path": "benchmarks/results/bitdistill_seqcls_glue3_primary_summary_2026-05-14.md"},
    {"label": "bitdistill_mnli_diagnostic_report", "kind": "tracked_report", "path": "benchmarks/results/bitdistill_seqcls_mnli_diagnostic_variant_summary_2026-05-14.md"},
    {"label": "i2sr_submodule_promotion_audit", "kind": "tracked_report", "path": "benchmarks/results/i2sr_submodule_promotion_audit_2026-05-13.md"},
    {"label": "benchmark_coverage_gate_report", "kind": "tracked_report", "path": f"benchmarks/results/benchmark_coverage_gate_{DATE}.md"},
    {"label": "direct_static_ternary_gguf_report", "kind": "tracked_report", "path": "benchmarks/results/direct_static_ternary_gguf_2026-05-13.md"},
    {"label": "direct_packed_gguf_support_report", "kind": "tracked_report", "path": "benchmarks/results/direct_packed_gguf_support_2026-05-13.md"},
    {"label": "direct_i2s_scalar_gguf_report", "kind": "tracked_report", "path": "benchmarks/results/direct_i2s_scalar_gguf_2026-05-13.md"},
    {"label": "direct_row_i2s_qwen05b_report", "kind": "tracked_report", "path": "benchmarks/results/direct_row_i2s_qwen05b_2026-05-13.md"},
    {"label": "tl2_shape_report", "kind": "tracked_report", "path": "benchmarks/results/tl2_shape_support_audit_2026-05-05.md"},
    {"label": "tl2_probe_report", "kind": "tracked_report", "path": "benchmarks/results/qwen05b_tl2_probe_2026-05-05.md"},
    {"label": "tl2_scale_report", "kind": "tracked_report", "path": "benchmarks/results/tl2_scale_semantics_2026-05-05.md"},
    {"label": "tl2_row_scale_design_report", "kind": "tracked_report", "path": "benchmarks/results/tl2_row_scale_design_2026-05-13.md"},
    {"label": "tl2_row_scale_runtime_contract_report", "kind": "tracked_report", "path": f"benchmarks/results/tl2_row_scale_runtime_contract_{DATE}.md"},
    {"label": "i2s_row_scale_format_report", "kind": "tracked_report", "path": "benchmarks/results/i2s_row_scale_format_audit_2026-05-13.md"},
    {"label": "row_scale_qtype_productization_gate_report", "kind": "tracked_report", "path": "benchmarks/results/row_scale_qtype_productization_gate_2026-05-13.md"},
    {"label": "row_scale_qtype_i2sr_active_patch_gate_report", "kind": "tracked_report", "path": "benchmarks/results/row_scale_qtype_productization_gate_i2sr_active_patch_2026-05-13.md"},
    {"label": "row_scale_qtype_i2sr_promotion_rehearsal_report", "kind": "tracked_report", "path": "benchmarks/results/row_scale_qtype_productization_gate_i2sr_promotion_rehearsal_2026-05-13.md"},
    {"label": "i2sr_promotion_handoff_report", "kind": "tracked_report", "path": "benchmarks/results/i2sr_promotion_handoff_2026-05-13.md"},
    {"label": "i2sr_qwen15b_candidate_report", "kind": "tracked_report", "path": "benchmarks/results/i2sr_qwen15b_candidate_2026-05-13.md"},
    {"label": "i2sr_x86act_fix_report", "kind": "tracked_report", "path": "benchmarks/results/i2sr_x86act_fix_2026-05-13.md"},
    {"label": "i2s_packing_layout_verify_report", "kind": "tracked_report", "path": "benchmarks/results/i2s_packing_layout_verify_2026-05-13.md"},
    {"label": "i2sr_rss_report", "kind": "tracked_report", "path": "benchmarks/results/i2sr_rss_2026-05-13.md"},
    {"label": "artifact_prune_application_report", "kind": "tracked_report", "path": "benchmarks/results/artifact_prune_application_2026-05-13.md"},
    {"label": "moe_report", "kind": "tracked_report", "path": f"benchmarks/results/moe_support_audit_{DATE}.md"},
    {"label": "kimi_config_feasibility_report", "kind": "tracked_report", "path": f"benchmarks/results/kimi_config_feasibility_{DATE}.md"},
    {"label": "moe_packing_contract_report", "kind": "tracked_report", "path": f"benchmarks/results/moe_packing_contract_{DATE}.md"},
    {"label": "moe_tl2_runtime_contract_report", "kind": "tracked_report", "path": f"benchmarks/results/moe_tl2_runtime_contract_{DATE}.md"},
    {"label": "tiny_qwen2moe_fixture_report", "kind": "tracked_report", "path": f"benchmarks/results/tiny_qwen2moe_fixture_{DATE}.md"},
    {"label": "tiny_qwen2moe_ternary_i2sr_fixture_report", "kind": "tracked_report", "path": f"benchmarks/results/tiny_qwen2moe_ternary_i2sr_fixture_{DATE}.md"},
    {"label": "tiny_qwen2moe_expert_scaling_report", "kind": "tracked_report", "path": f"benchmarks/results/tiny_qwen2moe_expert_scaling_{DATE}.md"},
    {"label": "unblock_requirements_report", "kind": "tracked_report", "path": f"benchmarks/results/unblock_requirements_{DATE}.md"},
    {"label": "i2sr_combined_patch", "kind": "tracked_report", "path": "patches/llama-i2sr-row-scale-qtype.patch"},
    {"label": "i2sr_root_runtime_patch", "kind": "tracked_report", "path": "patches/bitnet-i2sr-root-runtime.patch"},
    {"label": "i2sr_submodule_patch", "kind": "tracked_report", "path": "patches/llama-i2sr-row-scale-qtype.submodule.patch"},
    # Mechanical audits.
    {"label": "latest_nonrow_audit", "kind": "evidence_audit_md", "path": "benchmark_results/evidence_audit/latest_nonrow.md"},
    {"label": "row_notie_5000_audit", "kind": "evidence_audit_md", "path": "benchmark_results/evidence_audit/qwen15b_row_notie_5000.md"},
    {"label": "row_i2s_heapfix_audit", "kind": "evidence_audit_md", "path": "benchmark_results/evidence_audit/qwen15b_row_i2s_heapfix.md"},
    {"label": "row_i2s_thread_scaling_audit", "kind": "evidence_audit_md", "path": "benchmark_results/evidence_audit/qwen15b_row_i2s_thread_scaling.md"},
    {"label": "context_scaling_rss_audit", "kind": "evidence_audit_md", "path": "benchmark_results/evidence_audit/qwen15b_context_scaling_rss.md"},
    {"label": "qwen05b_tl2_probe_audit", "kind": "evidence_audit_md", "path": "benchmark_results/evidence_audit/qwen05b_tl2_probe.md"},
    # PPL quality artifacts.
    {"label": "fp_wikitext", "kind": "perplexity_json", "path": "benchmark_results/quality-9735/qwen15b_fp_wikitext.json"},
    {"label": "fp_fineweb", "kind": "perplexity_json", "path": "benchmark_results/quality-9735/qwen15b_fp_fineweb_heldout.json"},
    {"label": "ptq_wikitext", "kind": "perplexity_json", "path": "benchmark_results/quality-ptq-qwen15b/qwen15b_naive_ptq_wikitext.json"},
    {"label": "ptq_fineweb", "kind": "perplexity_json", "path": "benchmark_results/quality-ptq-qwen15b/qwen15b_naive_ptq_fineweb_heldout.json"},
    {"label": "hidden_mse_wikitext", "kind": "perplexity_json", "path": "benchmark_results/quality-9735/qwen15b_ternary_wikitext.json"},
    {"label": "hidden_mse_fineweb", "kind": "perplexity_json", "path": "benchmark_results/quality-9735/qwen15b_ternary_fineweb_heldout.json"},
    {"label": "kl_wikitext", "kind": "perplexity_json", "path": "benchmark_results/quality-qwen15b-klonly-5000/qwen15b_ternary_wikitext.json"},
    {"label": "kl_fineweb", "kind": "perplexity_json", "path": "benchmark_results/quality-qwen15b-klonly-5000/qwen15b_ternary_fineweb_heldout.json"},
    {"label": "kl_dense_head_wikitext", "kind": "perplexity_json", "path": "benchmark_results/quality-qwen15b-klonly-notiehead-5000/qwen15b_ternary_wikitext.json"},
    {"label": "kl_dense_head_fineweb", "kind": "perplexity_json", "path": "benchmark_results/quality-qwen15b-klonly-notiehead-5000/qwen15b_ternary_fineweb_heldout.json"},
    {"label": "row_dense_head_wikitext", "kind": "perplexity_json", "path": "benchmark_results/quality-qwen15b-klonly-row-notiehead-5000/qwen15b_ternary_wikitext.json"},
    {"label": "row_dense_head_fineweb", "kind": "perplexity_json", "path": "benchmark_results/quality-qwen15b-klonly-row-notiehead-5000/qwen15b_ternary_fineweb_heldout.json"},
    # Full ten-task lm-eval artifacts.
    {"label": "lm_eval_fp", "kind": "lm_eval_json", "path": "benchmark_results/lm-eval-qwen15b-full10/qwen15b_fp.json"},
    {"label": "lm_eval_ptq", "kind": "lm_eval_json", "path": "benchmark_results/lm-eval-qwen15b-full10/qwen15b_naive_ptq.json"},
    {"label": "lm_eval_hidden_mse", "kind": "lm_eval_json", "path": "benchmark_results/lm-eval-qwen15b-full10/qwen15b_qat_ternary.json"},
    {"label": "lm_eval_kl", "kind": "lm_eval_json", "path": "benchmark_results/lm-eval-qwen15b-klonly-full10/qwen15b_qat_ternary.json"},
    {"label": "lm_eval_kl_dense_head", "kind": "lm_eval_json", "path": "benchmark_results/lm-eval-qwen15b-klonly-notiehead-full10/qwen15b_qat_ternary.json"},
    {"label": "lm_eval_row_dense_head", "kind": "lm_eval_json", "path": "benchmark_results/lm-eval-qwen15b-klonly-row-notiehead-full10/qwen15b_qat_ternary.json"},
    # GGUF summaries and targeted JSON audits.
    {"label": "gguf_kl_suite", "kind": "gguf_summary_json", "path": "benchmark_results/gguf-qwen15b-klonly-suite/summary.json"},
    {"label": "gguf_dense_head_suite", "kind": "gguf_summary_json", "path": "benchmark_results/gguf-qwen15b-klonly-notiehead-suite/summary.json"},
    {"label": "gguf_row_dense_head_suite", "kind": "gguf_summary_json", "path": "benchmark_results/gguf-qwen15b-klonly-row-notiehead-suite/summary.json"},
    {"label": "gguf_row_i2s_heapfix", "kind": "gguf_summary_json", "path": "benchmark_results/gguf-qwen15b-row-i2s-heapfix-confirm/summary.json"},
    {"label": "gguf_row_i2s_thread_scaling", "kind": "thread_scaling_json", "path": "benchmark_results/i2s-row-scale-thread-scaling-fixed-2026-05-05/summary.json"},
    {"label": "gguf_context_rss", "kind": "gguf_memory_json", "path": "benchmark_results/gguf-rss-qwen15b-context-scaling-2026-05-05/summary.json"},
    {"label": "i2sr_x86act_rss", "kind": "gguf_memory_json", "path": "benchmark_results/gguf-rss-qwen15b-i2sr-x86act-context-2026-05-13/summary.json"},
    {"label": "direct_gguf_tiny", "kind": "direct_gguf_json", "path": "benchmark_results/direct-gguf-tiny-2026-05-13/summary.json"},
    {"label": "direct_gguf_qwen05b", "kind": "direct_gguf_json", "path": "benchmark_results/direct-gguf-qwen05b-klonly-notie-2026-05-13/summary.json"},
    {"label": "direct_i2s_tiny", "kind": "direct_i2s_json", "path": "benchmark_results/direct-i2s-tiny-2026-05-13/selfcontained_summary.json"},
    {"label": "direct_i2s_qwen05b_rowgroup_conversion", "kind": "direct_i2s_json", "path": "benchmark_results/direct-i2s-qwen05b-klonly-2026-05-13/conversion_summary.json"},
    {"label": "direct_i2s_qwen05b_rowgroup_suite", "kind": "gguf_summary_json", "path": "benchmark_results/direct-i2s-qwen05b-klonly-2026-05-13/summary.json"},
    {"label": "direct_i2s_qwen05b_x86act_conversion", "kind": "direct_i2s_json", "path": "benchmark_results/direct-i2s-qwen05b-klonly-x86act-2026-05-13/conversion_summary.json"},
    {"label": "direct_i2s_qwen05b_x86act_suite", "kind": "gguf_summary_json", "path": "benchmark_results/direct-i2s-qwen05b-klonly-x86act-2026-05-13/summary.json"},
    {"label": "direct_row_i2s_qwen05b_conversion", "kind": "direct_i2s_json", "path": "benchmark_results/direct-row-i2s-qwen05b-2026-05-13/conversion_summary.json"},
    {"label": "i2sr_writer_smoke_qwen05b", "kind": "direct_i2s_json", "path": "benchmark_results/i2sr-writer-smoke-2026-05-13/summary.json"},
    {"label": "i2sr_row_scale_qwen15b_conversion", "kind": "direct_i2s_json", "path": "benchmark_results/i2sr-row-scale-qwen15b-convert-2026-05-13/summary.json"},
    {"label": "i2sr_row_scale_qwen15b_suite", "kind": "gguf_summary_json", "path": "benchmark_results/i2sr-row-scale-qwen15b-suite-2026-05-13/summary.json"},
    {"label": "i2sr_x86act_qwen15b_conversion", "kind": "direct_i2s_json", "path": "benchmark_results/i2sr-row-scale-qwen15b-x86act-convert-2026-05-13/summary.json"},
    {"label": "i2sr_x86act_qwen15b_suite", "kind": "gguf_summary_json", "path": "benchmark_results/i2sr-row-scale-qwen15b-x86act-suite-2026-05-13/summary.json"},
    {"label": "direct_row_i2s_qwen05b_suite", "kind": "gguf_summary_json", "path": "benchmark_results/direct-row-i2s-qwen05b-portable-2026-05-13/summary.json"},
    {"label": "row_f16_qwen05b_suite", "kind": "gguf_summary_json", "path": "benchmark_results/row-f16-qwen05b-2026-05-13/summary.json"},
    {"label": "row_i2s_quantized_qwen05b_suite", "kind": "gguf_summary_json", "path": "benchmark_results/quantized-row-i2s-qwen05b-2026-05-13/summary.json"},
    {"label": "row_tq2_qwen05b_suite", "kind": "gguf_summary_json", "path": "benchmark_results/row-tq2-qwen05b-2026-05-13/summary.json"},
    {"label": "direct_packed_gguf_support_json", "kind": "direct_packed_support_json", "path": "benchmark_results/direct_packed_gguf_support_2026-05-13.json"},
    {"label": "tl2_shape_json", "kind": "tl2_shape_json", "path": "benchmark_results/tl2_shape_support_audit_2026-05-05.json"},
    {"label": "tl2_scale_json", "kind": "tl2_scale_json", "path": "benchmark_results/tl2_scale_semantics_2026-05-05.json"},
    {"label": "tl2_row_scale_design_json", "kind": "tl2_row_scale_design_json", "path": "benchmark_results/tl2_row_scale_design_2026-05-13.json"},
    {"label": "tl2_row_scale_runtime_contract_json", "kind": "tl2_row_scale_runtime_contract_json", "path": f"benchmark_results/tl2_row_scale_runtime_contract_{DATE}.json"},
    {"label": "i2s_row_scale_format_json", "kind": "i2s_format_json", "path": "benchmark_results/i2s_row_scale_format_audit_2026-05-13.json"},
    {"label": "row_scale_qtype_productization_gate_json", "kind": "row_scale_qtype_gate_json", "path": "benchmark_results/row_scale_qtype_productization_gate_2026-05-13.json"},
    {"label": "row_scale_qtype_i2sr_active_patch_gate_json", "kind": "row_scale_qtype_gate_json", "path": "benchmark_results/row_scale_qtype_productization_gate_i2sr_active_patch_2026-05-13.json"},
    {"label": "row_scale_qtype_i2sr_promotion_rehearsal_json", "kind": "row_scale_qtype_gate_json", "path": "benchmark_results/row_scale_qtype_productization_gate_i2sr_promotion_rehearsal_2026-05-13.json"},
    {"label": "i2sr_promotion_handoff_json", "kind": "i2sr_promotion_handoff_json", "path": "benchmark_results/i2sr_promotion_handoff_2026-05-13.json"},
    {"label": "i2s_packing_layout_verify_json", "kind": "packing_verify_json", "path": "benchmark_results/i2s-packing-layout-verify-2026-05-13/summary.json"},
    {"label": "benchmark_coverage_gate_json", "kind": "benchmark_coverage_gate_json", "path": f"benchmark_results/benchmark_coverage_gate_{DATE}.json"},
    {"label": "objective_completion_audit_json", "kind": "objective_completion_audit_json", "path": f"benchmark_results/objective_completion_audit_{DATE}.json"},
    {"label": "product_scope_gate_json", "kind": "product_scope_gate_json", "path": f"benchmark_results/product_scope_gate_{DATE}.json"},
    {"label": "bitdistill_recovery_submission_json", "kind": "generic_json", "path": f"benchmark_results/bitdistill_recovery_submission_{DATE}.json"},
    {"label": "bitdistill_recovery_audit_json", "kind": "generic_json", "path": f"benchmark_results/bitdistill_recovery_audit_{DATE}.json"},
    {"label": "bitdistill_stage2_curve_json", "kind": "generic_json", "path": f"benchmark_results/bitdistill_stage2_curve_{DATE}.json"},
    {"label": "bitdistill_stage2_curve_submission_json", "kind": "generic_json", "path": f"benchmark_results/bitdistill_stage2_curve_submission_{DATE}.json"},
    {"label": "bitdistill_controlled_curve_json", "kind": "generic_json", "path": f"benchmark_results/bitdistill_controlled_curve_{DATE}.json"},
    {"label": "bitdistill_root_cause_audit_json", "kind": "generic_json", "path": f"benchmark_results/bitdistill_root_cause_audit_{DATE}.json"},
    {"label": "bitdistill_reproduction_gate_json", "kind": "bitdistill_reproduction_gate_json", "path": f"benchmark_results/bitdistill_reproduction_gate_{DATE}.json"},
    {"label": "bitnet_sft_baseline_audit_json", "kind": "generic_json", "path": f"benchmark_results/bitnet_sft_baseline_audit_{DATE}.json"},
    {"label": "bitnet_sft_recipe_alignment_json", "kind": "generic_json", "path": f"benchmark_results/bitnet_sft_recipe_alignment_{DATE}.json"},
    {"label": "bitnet_sft_mechanics_audit_json", "kind": "generic_json", "path": f"benchmark_results/bitnet_sft_mechanics_audit_{DATE}.json"},
    {"label": "bitnet_sft_budget_sweep_json", "kind": "generic_json", "path": f"benchmark_results/bitnet_sft_budget_sweep_{DATE}.json"},
    {"label": "bitnet_sft_budget_paired_json", "kind": "generic_json", "path": f"benchmark_results/bitnet_sft_budget_paired_{DATE}.json"},
    {"label": "subln_activation_variance_json", "kind": "generic_json", "path": f"benchmark_results/subln_activation_variance_{DATE}.json"},
    {"label": "bitdistill_paired_predictions_json", "kind": "bitdistill_paired_predictions_json", "path": f"benchmark_results/bitdistill_paired_predictions_{DATE}.json"},
    {"label": "bitdistill_task_formulation_json", "kind": "bitdistill_task_formulation_json", "path": f"benchmark_results/bitdistill_task_formulation_audit_{DATE}.json"},
    {"label": "bitdistill_cpu_gate_json", "kind": "bitdistill_cpu_gate_json", "path": f"benchmark_results/bitdistill_glue_cpu_gate_{DATE}.json"},
    {"label": "bitdistill_cpu_xeon_gate_json", "kind": "bitdistill_cpu_gate_json", "path": f"benchmark_results/bitdistill_glue_cpu_xeon_gate_{DATE}.json"},
    {"label": "cpu_tradeoff_frontier_json", "kind": "generic_json", "path": f"benchmark_results/cpu_tradeoff_frontier_{DATE}.json"},
    {"label": "cpu_speed_uncertainty_json", "kind": "generic_json", "path": f"benchmark_results/cpu_speed_uncertainty_{DATE}.json"},
    {"label": "bitdistill_i2sr_gate_json", "kind": "bitdistill_i2sr_gate_json", "path": f"benchmark_results/bitdistill_i2sr_export_gate_{DATE}.json"},
    {"label": "bitdistill_i2sr_local_gate_json", "kind": "bitdistill_i2sr_gate_json", "path": f"benchmark_results/bitdistill_i2sr_export_gate_local_{DATE}.json"},
    {"label": "bitdistill_job_monitor_json", "kind": "bitdistill_job_monitor_json", "path": f"benchmark_results/bitdistill_job_monitor_{DATE}.json"},
    {"label": "bitdistill_dependency_graph_json", "kind": "bitdistill_dependency_graph_json", "path": f"benchmark_results/bitdistill_dependency_graph_{DATE}.json"},
    {"label": "bitdistill_postprocess_submission_json", "kind": "bitdistill_postprocess_submission_json", "path": f"benchmark_results/bitdistill_postprocess_submission_{DATE}.json"},
    {"label": "bitdistill_postprocess_dependency_json", "kind": "bitdistill_postprocess_dependency_json", "path": f"benchmark_results/bitdistill_postprocess_dependency_audit_{DATE}.json"},
    {"label": "bitdistill_afterany_postprocess_json", "kind": "bitdistill_afterany_postprocess_submission_json", "path": f"benchmark_results/bitdistill_afterany_postprocess_{DATE}.json"},
    {"label": "bitdistill_afterany_postprocess_dependency_json", "kind": "bitdistill_postprocess_dependency_json", "path": f"benchmark_results/bitdistill_afterany_postprocess_dependency_{DATE}.json"},
    {"label": "bitdistill_warmup_health_json", "kind": "bitdistill_warmup_health_json", "path": f"benchmark_results/bitdistill_warmup_health_{DATE}.json"},
    {"label": "bitdistill_warmup_finalizer_submission_json", "kind": "bitdistill_warmup_finalizer_submission_json", "path": f"benchmark_results/bitdistill_warmup_finalizer_submission_{DATE}.json"},
    {"label": "bitdistill_warmup_finalizer_json", "kind": "bitdistill_warmup_finalizer_json", "path": f"benchmark_results/bitdistill_warmup_finalizer_{DATE}.json"},
    {"label": "bitdistill_producer_script_audit_json", "kind": "bitdistill_producer_script_audit_json", "path": f"benchmark_results/bitdistill_producer_script_audit_{DATE}.json"},
    {"label": "bitdistill_job_matrix_json", "kind": "bitdistill_job_matrix_json", "path": f"benchmark_results/bitdistill_job_matrix_audit_{DATE}.json"},
    {"label": "bitdistill_active_goal_json", "kind": "bitdistill_active_goal_json", "path": f"benchmark_results/bitdistill_active_goal_audit_{DATE}.json"},
    {"label": "bitdistill_snapshot_integrity_json", "kind": "bitdistill_snapshot_integrity_json", "path": f"benchmark_results/bitdistill_snapshot_integrity_{DATE}.json"},
    {"label": "bitdistill_smoke_contract_json", "kind": "bitdistill_smoke_contract_json", "path": f"benchmark_results/bitdistill_smoke_contract_{DATE}.json"},
    {"label": "bitdistill_variant_summary_json", "kind": "bitdistill_variant_summary_json", "path": f"benchmark_results/bitdistill_variant_summary_{DATE}.json"},
    {"label": "bitdistill_rowwarmup_variant_summary_json", "kind": "bitdistill_variant_summary_json", "path": f"benchmark_results/bitdistill_rowwarmup_variant_summary_{DATE}.json"},
    {"label": "bitdistill_causal_longwarmup_json", "kind": "bitdistill_causal_summary_json", "path": f"benchmark_results/bitdistill_causal_longwarmup_densehead_summary_{DATE}.json"},
    {"label": "bitdistill_loss_scale_json", "kind": "bitdistill_loss_scale_json", "path": f"benchmark_results/bitdistill_loss_scale_audit_{DATE}.json"},
    {"label": "i2sr_submodule_promotion_audit_json", "kind": "i2sr_submodule_promotion_audit_json", "path": "benchmark_results/i2sr_submodule_promotion_audit_2026-05-13.json"},
    {"label": "moe_support_json", "kind": "moe_support_json", "path": f"benchmark_results/moe_support_audit_{DATE}.json"},
    {"label": "kimi_config_feasibility_json", "kind": "kimi_config_feasibility_json", "path": f"benchmark_results/kimi_config_feasibility_{DATE}.json"},
    {"label": "moe_packing_contract_json", "kind": "moe_packing_contract_json", "path": f"benchmark_results/moe_packing_contract_{DATE}.json"},
    {"label": "moe_tl2_runtime_contract_json", "kind": "moe_tl2_runtime_contract_json", "path": f"benchmark_results/moe_tl2_runtime_contract_{DATE}.json"},
    {"label": "bitdistill_cpu_benchmark_json", "kind": "bitdistill_cpu_benchmark_json", "path": f"benchmark_results/bitdistill_glue_cpu_{DATE}.json"},
    {"label": "bitdistill_cpu_xeon_benchmark_json", "kind": "bitdistill_cpu_benchmark_json", "path": f"benchmark_results/bitdistill_glue_cpu_xeon_{DATE}.json"},
    {"label": "tiny_qwen2moe_fixture_json", "kind": "tiny_qwen2moe_fixture_json", "path": f"benchmark_results/tiny_qwen2moe_fixture_{DATE}.json"},
    {"label": "tiny_qwen2moe_ternary_i2sr_fixture_json", "kind": "tiny_qwen2moe_ternary_i2sr_fixture_json", "path": f"benchmark_results/tiny_qwen2moe_ternary_i2sr_fixture_{DATE}.json"},
    {"label": "tiny_qwen2moe_expert_scaling_json", "kind": "tiny_qwen2moe_expert_scaling_json", "path": f"benchmark_results/tiny_qwen2moe_expert_scaling_{DATE}.json"},
    {"label": "unblock_requirements_json", "kind": "unblock_requirements_json", "path": f"benchmark_results/unblock_requirements_{DATE}.json"},
    {"label": "tl2_generic_summary", "kind": "gguf_summary_json", "path": "benchmark_results/gguf-qwen05b-tl2-probe-2026-05-05/summary.json"},
    {"label": "tl2_avx512_summary", "kind": "gguf_summary_json", "path": "benchmark_results/gguf-qwen05b-tl2-avx512-2026-05-05/summary.json"},
    {"label": "ptq_math", "kind": "math_json", "path": "benchmark_results/math_viability_gaussian_10trial_2026-05-05.json"},
]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def metric_value(task_results: dict[str, Any], metric: str) -> float:
    for key in (metric, f"{metric},none"):
        value = task_results.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    raise KeyError(metric)


def parse_lm_eval(data: dict[str, Any]) -> dict[str, Any]:
    results = data.get("results", {})
    samples = data.get("samples", {})
    values: list[float] = []
    sample_count = 0
    missing: list[str] = []
    task_metrics: dict[str, float] = {}
    for task, metric in SELECTED_LM_EVAL_METRICS.items():
        task_result = results.get(task)
        if not isinstance(task_result, dict):
            missing.append(task)
            continue
        try:
            value = metric_value(task_result, metric)
        except KeyError:
            missing.append(task)
            continue
        values.append(value)
        task_metrics[task] = value
        task_samples = samples.get(task, [])
        if isinstance(task_samples, list):
            sample_count += len(task_samples)
    return {
        "selected_mean": sum(values) / len(values) if values else None,
        "selected_tasks": len(values),
        "samples": sample_count,
        "missing": missing,
        "task_metrics": task_metrics,
    }


def parse_gguf_summary(data: dict[str, Any]) -> dict[str, Any]:
    rows = data.get("rows", [])
    parsed_rows: list[dict[str, Any]] = []
    failed: list[str] = []
    nan_ppl: list[str] = []
    catastrophic_ppl: list[str] = []
    finite_ppl_values: list[float] = []
    if not isinstance(rows, list):
        return {
            "rows": 0,
            "failed": ["rows-not-list"],
            "nan_ppl": [],
            "catastrophic_ppl": [],
            "catastrophic_ppl_threshold": CATASTROPHIC_PPL_THRESHOLD,
            "max_finite_ppl": None,
            "artifacts": [],
        }
    for row in rows:
        if not isinstance(row, dict):
            failed.append("non-object-row")
            continue
        name = str(row.get("name", ""))
        for key in ("smoke_returncode", "bench_returncode", "ppl_returncode"):
            if key in row and int(row.get(key, 1)) != 0:
                failed.append(f"{name}:{key}={row.get(key)}")
        ppl = row.get("perplexity", {}).get("ppl") if isinstance(row.get("perplexity"), dict) else None
        if ppl is not None and (not isinstance(ppl, (int, float)) or not math.isfinite(float(ppl))):
            nan_ppl.append(name)
        elif isinstance(ppl, (int, float)) and math.isfinite(float(ppl)):
            finite_ppl = float(ppl)
            finite_ppl_values.append(finite_ppl)
            if finite_ppl >= CATASTROPHIC_PPL_THRESHOLD:
                catastrophic_ppl.append(name)
        parsed_rows.append(
            {
                "name": name,
                "kind": row.get("kind"),
                "file_mib": row.get("file_mib"),
                "ppl": ppl,
                "prefill_tok_s": row.get("bench", {}).get("prefill", {}).get("tok_s") if isinstance(row.get("bench"), dict) else None,
                "decode_tok_s": row.get("bench", {}).get("decode", {}).get("tok_s") if isinstance(row.get("bench"), dict) else None,
            }
        )
    return {
        "rows": len(rows),
        "failed": failed,
        "nan_ppl": nan_ppl,
        "catastrophic_ppl": catastrophic_ppl,
        "catastrophic_ppl_threshold": CATASTROPHIC_PPL_THRESHOLD,
        "max_finite_ppl": max(finite_ppl_values, default=None),
        "artifacts": parsed_rows,
    }


def extract_metrics(kind: str, path: Path) -> dict[str, Any]:
    data = read_json(path) if path.suffix == ".json" else None
    if data is None:
        return {}
    if kind == "perplexity_json":
        return {
            "perplexity": data.get("perplexity"),
            "nll": data.get("nll"),
            "eval_tokens": data.get("eval_tokens"),
            "model_kind": data.get("model_kind"),
        }
    if kind == "lm_eval_json":
        return parse_lm_eval(data)
    if kind == "gguf_summary_json":
        return parse_gguf_summary(data)
    if kind == "thread_scaling_json":
        rows = data.get("rows", [])
        return {
            "rows": len(rows) if isinstance(rows, list) else None,
            "threads": [row.get("threads") for row in rows if isinstance(row, dict)],
            "max_prefill_tok_s": max(
                [float(row.get("prefill_tok_s")) for row in rows if isinstance(row, dict) and isinstance(row.get("prefill_tok_s"), (int, float))],
                default=None,
            ),
            "max_decode_tok_s": max(
                [float(row.get("decode_tok_s")) for row in rows if isinstance(row, dict) and isinstance(row.get("decode_tok_s"), (int, float))],
                default=None,
            ),
        }
    if kind == "gguf_memory_json":
        rows = data.get("rows", [])
        return {
            "rows": len(rows) if isinstance(rows, list) else None,
            "contexts": sorted({int(row.get("ctx_size")) for row in rows if isinstance(row, dict) and isinstance(row.get("ctx_size"), (int, float))}),
        }
    if kind == "direct_gguf_json":
        smoke = data.get("smoke", {})
        reader = data.get("gguf_reader", {})
        return {
            "architecture": data.get("architecture"),
            "outtype": data.get("outtype"),
            "ternary_materialized": data.get("ternary_materialized"),
            "copied_tensors": data.get("copied_tensors"),
            "output_tensors": data.get("output_tensors"),
            "outfile_size_bytes": data.get("outfile_size_bytes"),
            "gguf_reader_returncode": reader.get("returncode") if isinstance(reader, dict) else None,
            "gguf_reader_tensors": reader.get("n_tensors") if isinstance(reader, dict) else None,
            "smoke_returncode": smoke.get("returncode") if isinstance(smoke, dict) else None,
        }
    if kind == "direct_i2s_json":
        return {
            "architecture": data.get("architecture"),
            "has_native_gguf_python_constants": data.get("has_native_gguf_python_constants"),
            "has_native_i2s_gguf_python_constants": data.get("has_native_i2s_gguf_python_constants"),
            "has_native_i2sr_gguf_python_constants": data.get("has_native_i2sr_gguf_python_constants"),
            "ternary_i2s_packed": data.get("ternary_i2s_packed"),
            "row_scale_i2s_packed": data.get("row_scale_i2s_packed"),
            "output_f16_tensors": data.get("output_f16_tensors"),
            "copied_tensors": data.get("copied_tensors"),
            "output_tensors": data.get("output_tensors"),
            "outfile_size_bytes": data.get("outfile_size_bytes"),
            "row_scale_rejected": data.get("row_scale_rejected"),
            "row_scale_prototype": data.get("row_scale_prototype"),
            "row_scale_qtype": data.get("row_scale_qtype"),
            "i2_sr_dtype": data.get("i2_sr_dtype"),
            "mostly_i2_sr_file_type": data.get("mostly_i2_sr_file_type"),
        }
    if kind == "direct_packed_support_json":
        verdict = data.get("verdict", {})
        checks = data.get("checks", {})
        return {
            "direct_dense_gguf_supported": verdict.get("direct_dense_gguf_supported"),
            "direct_packed_i2s_supported": verdict.get("direct_packed_i2s_supported"),
            "product_safe_row_scale_packed_supported": verdict.get("product_safe_row_scale_packed_supported"),
            "py_gguf_has_i2s_quant_type": checks.get("py_gguf_has_i2s_quant_type"),
            "requires_stable_row_scale_type_or_version": verdict.get("requires_stable_row_scale_type_or_version"),
        }
    if kind == "tl2_shape_json":
        models = data.get("models", [])
        return {
            "models": [
                {
                    "label": model.get("label"),
                    "eligible_tensors": model.get("eligible_tensors"),
                    "unique_shapes": len(model.get("unique_shapes", [])) if isinstance(model.get("unique_shapes"), list) else None,
                }
                for model in models
                if isinstance(model, dict)
            ],
            "builds": data.get("builds", []),
        }
    if kind == "tl2_scale_json":
        results = data.get("results", [])
        return {
            "results": [
                {
                    "label": result.get("label"),
                    "tensors": result.get("tensors"),
                    "row_scale_tensors": result.get("row_scale_tensors"),
                    "scalar_scale_tensors": result.get("scalar_scale_tensors"),
                    "total_relative_fro_error_if_one_scale": result.get("total_relative_fro_error_if_one_scale"),
                    "max_tensor_relative_fro_error": result.get("max_tensor_relative_fro_error"),
                }
                for result in results
                if isinstance(result, dict)
            ]
        }
    if kind == "tl2_row_scale_design_json":
        parsed = []
        for result in data.get("results", []):
            if not isinstance(result, dict):
                continue
            strategies = {
                row.get("name"): row
                for row in result.get("strategies", [])
                if isinstance(row, dict) and isinstance(row.get("name"), str)
            }
            current = strategies.get("current_tl2_tensor_max_fp32", {})
            row_fp16 = strategies.get("row_exact_fp16", {})
            group32 = strategies.get("group32_l2_optimal_fp16", {})
            parsed.append(
                {
                    "label": result.get("label"),
                    "tensors": result.get("tensors"),
                    "row_scale_tensors": result.get("row_scale_tensors"),
                    "current_tl2_error": current.get("expected_relative_output_rms_error"),
                    "group32_fp16_error": group32.get("expected_relative_output_rms_error"),
                    "row_fp16_error": row_fp16.get("expected_relative_output_rms_error"),
                    "row_fp16_scale_mib": row_fp16.get("scale_mib_fp16"),
                }
            )
        return {"results": parsed}
    if kind == "tl2_row_scale_runtime_contract_json":
        checks = data.get("checks", [])
        failed = [check.get("name") for check in checks if isinstance(check, dict) and not check.get("passed")]
        math_section = data.get("math", {}) if isinstance(data.get("math"), dict) else {}
        benchmark_evidence = data.get("benchmark_evidence", {}) if isinstance(data.get("benchmark_evidence"), dict) else {}
        return {
            "ready": data.get("tl2_row_scale_runtime_ready"),
            "checks": len(checks) if isinstance(checks, list) else None,
            "failed": failed,
            "current_tl2_tensor_max_error": math_section.get("current_tl2_tensor_max_error"),
            "row_fp16_error": math_section.get("row_fp16_error"),
            "row_fp16_scale_mib": math_section.get("row_fp16_scale_mib"),
            "row_scale_tl2_rows": len(benchmark_evidence.get("row_scale_tl2_rows", []))
            if isinstance(benchmark_evidence.get("row_scale_tl2_rows"), list)
            else None,
            "row_scale_tl2_finite_quality_rows": len(benchmark_evidence.get("row_scale_tl2_finite_quality_rows", []))
            if isinstance(benchmark_evidence.get("row_scale_tl2_finite_quality_rows"), list)
            else None,
        }
    if kind == "i2s_format_json":
        metrics = data.get("metrics", {})
        verdict = data.get("verdict", {})
        return {
            "default_i2s_to_tq2_ppl_ratio": metrics.get("default_row_scale_i2s_to_tq2_ppl_ratio"),
            "prototype_i2s_to_tq2_ppl_ratio": metrics.get("prototype_row_scale_i2s_to_tq2_ppl_ratio"),
            "row_scale_i2s_physically_possible": verdict.get("row_scale_i2s_physically_possible"),
            "current_patch_is_product_format_safe": verdict.get("current_patch_is_product_format_safe"),
            "stable_new_format_required": verdict.get("stable_new_format_required"),
            "direct_ternary_gguf_writer_still_required": verdict.get("direct_ternary_gguf_writer_still_required"),
        }
    if kind == "row_scale_qtype_gate_json":
        gates = data.get("gates", [])
        failed = [gate.get("name") for gate in gates if isinstance(gate, dict) and not gate.get("passed")]
        observations = data.get("observations", {})
        return {
            "passed": data.get("passed"),
            "gates": len(gates) if isinstance(gates, list) else None,
            "failed": failed,
            "prototype_ratio": observations.get("prototype_row_scale_i2s_to_tq2_ppl_ratio"),
            "default_ratio": observations.get("default_row_scale_i2s_to_tq2_ppl_ratio"),
            "has_ggml_stable_qtype": observations.get("has_ggml_stable_qtype"),
            "has_llama_stable_ftype": observations.get("has_llama_stable_ftype"),
            "direct_writer_emits_stable_qtype": observations.get("direct_writer_emits_stable_qtype"),
            "stable_benchmark_present": observations.get("stable_benchmark_present"),
            "stable_benchmark_quality_ok": observations.get("stable_benchmark_quality_ok"),
            "stable_benchmark_max_ppl": observations.get("stable_benchmark_max_ppl"),
            "packing_verification_passed": observations.get("packing_verification_passed"),
            "packing_verification_checked_tensors": observations.get("packing_verification_checked_tensors"),
        }
    if kind == "packing_verify_json":
        return {
            "passed": data.get("passed"),
            "checked_tensors": data.get("checked_tensors"),
            "passed_tensors": data.get("passed_tensors"),
        }
    if kind == "benchmark_coverage_gate_json":
        return {
            "passed": data.get("passed"),
            "check_count": data.get("check_count"),
            "failed": data.get("failed"),
        }
    if kind == "objective_completion_audit_json":
        return {
            "objective_achieved": data.get("objective_achieved"),
            "completion_status": data.get("completion_status"),
            "check_count": data.get("check_count"),
            "complete_count": data.get("complete_count"),
            "partial_or_missing": data.get("partial_or_missing"),
        }
    if kind == "product_scope_gate_json":
        return {
            "scope_status": data.get("scope_status"),
            "supported_claim_count": data.get("supported_claim_count"),
            "unsupported_claim_count": data.get("unsupported_claim_count"),
            "publishable_angle": data.get("publishable_angle"),
        }
    if kind == "bitdistill_reproduction_gate_json":
        rows = data.get("rows", [])
        present = [row for row in rows if isinstance(row, dict) and row.get("exists")]
        with_examples = [row for row in present if isinstance(row.get("examples"), int)]
        with_full_eval = [row for row in present if row.get("full_eval_examples") is True]
        with_ci = [row for row in present if isinstance(row.get("accuracy_ci95"), list)]
        return {
            "tasks": data.get("tasks", []),
            "rows": len(rows) if isinstance(rows, list) else None,
            "present_rows": len(present),
            "present_with_examples": len(with_examples),
            "present_with_full_eval": len(with_full_eval),
            "present_with_accuracy_ci95": len(with_ci),
            "paper_style_tensor_complete": data.get("paper_style_tensor_complete"),
            "paper_style_tensor_passed": data.get("paper_style_tensor_passed"),
            "row_scale_complete": data.get("row_scale_complete"),
            "row_scale_passed": data.get("row_scale_passed"),
            "max_fp_gap": data.get("max_fp_gap"),
            "confidence_level": (data.get("confidence") or {}).get("level"),
        }
    if kind == "bitdistill_paired_predictions_json":
        rows = data.get("rows", [])
        return {
            "status": data.get("status"),
            "rows": len(rows) if isinstance(rows, list) else None,
            "complete": data.get("complete"),
            "pending": data.get("pending"),
            "failed": data.get("failed"),
        }
    if kind == "bitdistill_task_formulation_json":
        return {
            "sequence_baselines_full": data.get("sequence_baselines_full"),
            "causal_rows_materialized": data.get("causal_rows_materialized"),
            "pending_paper_candidates": data.get("pending_paper_candidates"),
            "paper_anchor_source": data.get("paper_anchor_source"),
            "rows": len(data.get("rows", [])) if isinstance(data.get("rows"), list) else None,
        }
    if kind == "bitdistill_cpu_gate_json":
        critical = data.get("critical", [])
        complete = [row for row in critical if isinstance(row, dict) and row.get("complete")]
        full_quality = [row for row in critical if isinstance(row, dict) and row.get("full_quality_available")]
        return {
            "passed": data.get("passed"),
            "input_exists": data.get("input_exists"),
            "rows": len(data.get("rows", [])) if isinstance(data.get("rows"), list) else None,
            "critical": len(critical) if isinstance(critical, list) else None,
            "critical_complete": len(complete),
            "critical_full_quality": len(full_quality),
            "max_eval_samples": data.get("max_eval_samples"),
            "blockers": data.get("blockers", []),
        }
    if kind == "bitdistill_i2sr_gate_json":
        rows = data.get("rows", [])
        complete = [row for row in rows if isinstance(row, dict) and row.get("complete")]
        blockers = {
            blocker
            for row in rows
            if isinstance(row, dict)
            for blocker in row.get("blockers", [])
            if isinstance(blocker, str)
        }
        return {
            "passed": data.get("passed"),
            "rows": len(rows) if isinstance(rows, list) else None,
            "complete": len(complete),
            "tasks": data.get("tasks", []),
            "scales": data.get("scales", []),
            "blockers": sorted(blockers),
        }
    if kind == "bitdistill_job_monitor_json":
        warmup = data.get("warmup", {}) if isinstance(data.get("warmup"), dict) else {}
        latest = warmup.get("latest_step", {}) if isinstance(warmup.get("latest_step"), dict) else {}
        downstream = data.get("downstream", [])
        states = {
            row.get("job_status", {}).get("state")
            for row in downstream
            if isinstance(row, dict) and isinstance(row.get("job_status"), dict)
        }
        return {
            "warmup_step": latest.get("step"),
            "warmup_max_steps": warmup.get("max_steps"),
            "warmup_progress": warmup.get("progress"),
            "warmup_latest_ce": latest.get("ce"),
            "warmup_save_every_steps": warmup.get("save_every_steps"),
            "warmup_snapshots": warmup.get("snapshot_count"),
            "warmup_warnings": warmup.get("warnings", []),
            "downstream_jobs": len(downstream) if isinstance(downstream, list) else None,
            "downstream_states": sorted(str(state) for state in states if state),
        }
    if kind == "bitdistill_dependency_graph_json":
        warmup = data.get("warmup", {}) if isinstance(data.get("warmup"), dict) else {}
        latest = warmup.get("latest_step", {}) if isinstance(warmup.get("latest_step"), dict) else {}
        return {
            "ready": data.get("ready"),
            "checks": len(data.get("checks", [])) if isinstance(data.get("checks"), list) else None,
            "failed": len([check for check in data.get("checks", []) if isinstance(check, dict) and not check.get("passed")])
            if isinstance(data.get("checks"), list)
            else None,
            "active_rows": data.get("active_rows"),
            "deduped_rows": data.get("deduped_rows"),
            "raw_rows": data.get("raw_rows"),
            "warmup_step": latest.get("step"),
            "warmup_max_steps": warmup.get("max_steps"),
            "warnings": data.get("warnings", []),
            "blockers": data.get("blockers", []),
        }
    if kind == "bitdistill_postprocess_dependency_json":
        checks = data.get("checks", [])
        expected = data.get("expected_producers", {})
        postprocess = data.get("postprocess", {})
        job = postprocess.get("job", {}) if isinstance(postprocess, dict) else {}
        return {
            "passed": data.get("passed"),
            "checks": len(checks) if isinstance(checks, list) else None,
            "failed": len([check for check in checks if isinstance(check, dict) and not check.get("passed")])
            if isinstance(checks, list)
            else None,
            "expected_jobs": len(expected.get("job_ids", [])) if isinstance(expected.get("job_ids"), list) else None,
            "warmup_jobs": len(expected.get("warmup_job_ids", [])) if isinstance(expected.get("warmup_job_ids"), list) else None,
            "downstream_jobs": len(expected.get("downstream_job_ids", []))
            if isinstance(expected.get("downstream_job_ids"), list)
            else None,
            "extra_jobs": len(expected.get("extra_jobs", [])) if isinstance(expected.get("extra_jobs"), list) else None,
            "missing": data.get("missing_dependency_job_ids", []),
            "postprocess_job": job.get("job_id") if isinstance(job, dict) else None,
        }
    if kind in {"bitdistill_postprocess_submission_json", "bitdistill_afterany_postprocess_submission_json"}:
        return {
            "submitted": data.get("submitted"),
            "job_id": data.get("job_id"),
            "dependency_type": data.get("dependency_type"),
            "producer_jobs": len(data.get("producer_job_ids", [])) if isinstance(data.get("producer_job_ids"), list) else None,
            "warmup_jobs": len(data.get("warmup_job_ids", [])) if isinstance(data.get("warmup_job_ids"), list) else None,
            "downstream_jobs": len(data.get("downstream_job_ids", [])) if isinstance(data.get("downstream_job_ids"), list) else None,
            "extra_jobs": len(data.get("extra_job_ids", [])) if isinstance(data.get("extra_job_ids"), list) else None,
            "note": data.get("note"),
        }
    if kind == "bitdistill_warmup_finalizer_submission_json":
        return {
            "submitted": data.get("submitted"),
            "job_id": data.get("job_id"),
            "dependency_type": data.get("dependency_type"),
            "warmup_jobs": len(data.get("warmup_job_ids", [])) if isinstance(data.get("warmup_job_ids"), list) else None,
            "note": data.get("note"),
        }
    if kind == "bitdistill_warmup_finalizer_json":
        steps = data.get("steps", [])
        failed = data.get("failed_steps", [])
        return {
            "passed": data.get("passed"),
            "steps": len(steps) if isinstance(steps, list) else None,
            "failed": len(failed) if isinstance(failed, list) else None,
        }
    if kind == "bitdistill_warmup_health_json":
        latest = data.get("latest_step", {}) if isinstance(data.get("latest_step"), dict) else {}
        return {
            "passed": data.get("passed"),
            "checks": len(data.get("checks", [])) if isinstance(data.get("checks"), list) else None,
            "failed": len(data.get("failed", [])) if isinstance(data.get("failed"), list) else None,
            "warnings": len(data.get("warnings", [])) if isinstance(data.get("warnings"), list) else None,
            "step": latest.get("step"),
            "max_steps": data.get("max_steps"),
            "progress": data.get("progress"),
            "latest_ce": latest.get("ce"),
            "last_ce_mean": (data.get("last_window") or {}).get("mean"),
            "seconds_per_step": data.get("seconds_per_step"),
            "eta_seconds": data.get("eta_seconds"),
            "snapshot_count": data.get("snapshot_count"),
            "final_state_exists": data.get("final_state_exists"),
        }
    if kind == "bitdistill_producer_script_audit_json":
        checks = data.get("checks", [])
        failed = [check for check in checks if isinstance(check, dict) and not check.get("passed")]
        cpu_job = data.get("cpu_job", {}) if isinstance(data.get("cpu_job"), dict) else {}
        i2sr_job = data.get("i2sr_job", {}) if isinstance(data.get("i2sr_job"), dict) else {}
        strict_post = data.get("strict_postprocess_job", {}) if isinstance(data.get("strict_postprocess_job"), dict) else {}
        any_post = data.get("afterany_postprocess_job", {}) if isinstance(data.get("afterany_postprocess_job"), dict) else {}
        return {
            "passed": data.get("passed"),
            "checks": len(checks) if isinstance(checks, list) else None,
            "failed": len(failed) if isinstance(checks, list) else None,
            "downstream_jobs": len(data.get("downstream_job_ids", [])) if isinstance(data.get("downstream_job_ids"), list) else None,
            "cpu_job": cpu_job.get("job_id"),
            "i2sr_job": i2sr_job.get("job_id"),
            "strict_postprocess_job": strict_post.get("job_id"),
            "afterany_postprocess_job": any_post.get("job_id"),
        }
    if kind == "bitdistill_job_matrix_json":
        return {
            "passed": data.get("passed"),
            "observed_rows": data.get("observed_rows"),
            "expected_rows": data.get("expected_rows"),
            "configured_rows": data.get("configured_rows"),
            "job_states": data.get("job_states", {}),
            "inferred_rows": len(data.get("inferred_field_rows", [])) if isinstance(data.get("inferred_field_rows"), list) else None,
            "blockers": data.get("blockers", []),
        }
    if kind == "bitdistill_active_goal_json":
        paper = data.get("metrics", {}).get("paper_reproduction", {})
        runtime = data.get("metrics", {}).get("row_scale_runtime", {})
        return {
            "objective_achieved": data.get("objective_achieved"),
            "completion_status": data.get("completion_status"),
            "check_count": data.get("check_count"),
            "complete_count": data.get("complete_count"),
            "pending_count": data.get("pending_count"),
            "open_requirements": data.get("open_requirements", []),
            "warmup_step": paper.get("warmup_step") if isinstance(paper, dict) else None,
            "warmup_max_steps": paper.get("warmup_max_steps") if isinstance(paper, dict) else None,
            "row_scale_complete": runtime.get("row_scale_complete") if isinstance(runtime, dict) else None,
            "i2sr_passed": runtime.get("i2sr_passed") if isinstance(runtime, dict) else None,
            "cpu_passed": runtime.get("cpu_passed") if isinstance(runtime, dict) else None,
        }
    if kind == "bitdistill_snapshot_integrity_json":
        snapshots = data.get("snapshots", [])
        snapshot_rows = [row for row in snapshots if isinstance(row, dict)]
        passed_rows = [row for row in snapshot_rows if row.get("passed")]
        first = snapshot_rows[0] if snapshot_rows else {}
        metrics = first.get("metrics", {}) if isinstance(first.get("metrics"), dict) else {}
        state = first.get("state", {}) if isinstance(first.get("state"), dict) else {}
        return {
            "passed": data.get("passed"),
            "snapshot_count": data.get("snapshot_count"),
            "passed_snapshots": len(passed_rows),
            "step": metrics.get("step"),
            "scale_mode": metrics.get("scale_mode"),
            "ce": metrics.get("ce"),
            "expected_ternary": metrics.get("bitlinear_replaced"),
            "ternary_weight_count": state.get("ternary_weight_count"),
            "weight_scale_count": state.get("weight_scale_count"),
            "row_scale_count": state.get("row_scale_count"),
            "tensor_scale_count": state.get("tensor_scale_count"),
            "codes_valid": state.get("codes_valid"),
        }
    if kind == "bitdistill_smoke_contract_json":
        checks = data.get("checks", [])
        failed = data.get("failed", [])
        continued = data.get("continued_pretrain_metrics", {})
        task = data.get("task_sft_metrics", {})
        continued_prep = continued.get("preparation", {}) if isinstance(continued, dict) else {}
        task_prep = task.get("preparation", {}) if isinstance(task, dict) else {}
        return {
            "passed": data.get("passed"),
            "check_count": data.get("check_count"),
            "failed_count": len(failed) if isinstance(failed, list) else None,
            "checks": len(checks) if isinstance(checks, list) else None,
            "continued_bitlinear": continued_prep.get("bitlinear_replaced") if isinstance(continued_prep, dict) else None,
            "continued_subln": continued_prep.get("subln_inserted") if isinstance(continued_prep, dict) else None,
            "task_bitlinear": task_prep.get("bitlinear_replaced") if isinstance(task_prep, dict) else None,
            "task_subln": task_prep.get("subln_inserted") if isinstance(task_prep, dict) else None,
        }
    if kind == "bitdistill_variant_summary_json":
        rows = data.get("rows", [])
        materialized = [
            row
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("accuracy"), (int, float))
        ]
        return {
            "rows": len(rows) if isinstance(rows, list) else None,
            "materialized_rows": len(materialized),
            "roots": data.get("roots", []),
            "tasks": data.get("tasks", []),
        }
    if kind == "bitdistill_causal_summary_json":
        rows = data.get("rows", [])
        verdicts = data.get("verdicts", [])
        materialized = [
            row
            for row in rows
            if isinstance(row, dict) and row.get("exists")
        ]
        passed_verdicts = [
            row
            for row in verdicts
            if isinstance(row, dict) and row.get("passes_fp_gap")
        ]
        return {
            "passed": data.get("passed"),
            "rows": len(rows) if isinstance(rows, list) else None,
            "materialized_rows": len(materialized),
            "verdicts": len(verdicts) if isinstance(verdicts, list) else None,
            "passed_verdicts": len(passed_verdicts),
            "tasks": data.get("tasks", []),
        }
    if kind == "bitdistill_loss_scale_json":
        return {
            "rows": len(data.get("rows", [])) if isinstance(data.get("rows"), list) else None,
            "materialized_rows": data.get("materialized_rows"),
            "paper_gamma": data.get("paper_classification_attention_gamma"),
            "projected_attention_to_ce_min": data.get("projected_paper_attention_to_ce_min"),
            "projected_attention_to_ce_max": data.get("projected_paper_attention_to_ce_max"),
        }
    if kind == "i2sr_submodule_promotion_audit_json":
        return {
            "promotion_ready": data.get("promotion_ready"),
            "active_runtime_support": data.get("active_runtime_support"),
            "patch_applies_cleanly": data.get("patch_applies_cleanly"),
            "submodule_short": data.get("submodule_short"),
            "blockers": data.get("blockers"),
        }
    if kind == "i2sr_promotion_handoff_json":
        fork = data.get("candidate_fork_probe", {})
        return {
            "ready_for_handoff": data.get("ready_for_handoff"),
            "root_clean": data.get("root_clean"),
            "submodule_clean": data.get("submodule_clean"),
            "root_patch_applies": data.get("root_patch_check", {}).get("applies"),
            "submodule_patch_applies": data.get("submodule_patch_check", {}).get("applies"),
            "candidate_fork_reachable": fork.get("reachable") if isinstance(fork, dict) else None,
            "blockers": data.get("blockers", []),
        }
    if kind == "moe_support_json":
        gates = data.get("productization_gates", [])
        failed = [gate.get("name") for gate in gates if isinstance(gate, dict) and not gate.get("passed")]
        checks = data.get("checks", [])
        tiny = data.get("tiny_qwen2moe_fixture", {}) if isinstance(data.get("tiny_qwen2moe_fixture"), dict) else {}
        ternary_tiny = (
            data.get("tiny_qwen2moe_ternary_i2sr_fixture", {})
            if isinstance(data.get("tiny_qwen2moe_ternary_i2sr_fixture"), dict)
            else {}
        )
        return {
            "checks": len(checks) if isinstance(checks, list) else None,
            "present_checks": sum(1 for check in checks if isinstance(check, dict) and check.get("status") == "present"),
            "gates": len(gates) if isinstance(gates, list) else None,
            "failed_gates": failed,
            "kimi_source_matches": len(data.get("kimi_source_matches", [])),
            "local_kimi_artifacts": len(data.get("local_kimi_artifacts", [])),
            "tiny_qwen2moe_passed": tiny.get("passed"),
            "tiny_qwen2moe_ternary_i2sr_passed": ternary_tiny.get("passed"),
        }
    if kind == "kimi_config_feasibility_json":
        architecture = data.get("architecture", {}) if isinstance(data.get("architecture"), dict) else {}
        features = data.get("features", []) if isinstance(data.get("features"), list) else []
        unsupported = data.get("unsupported_features", [])
        return {
            "passed": data.get("passed"),
            "model_type": architecture.get("model_type"),
            "architectures": architecture.get("architectures"),
            "layers": architecture.get("num_hidden_layers"),
            "routed_experts": architecture.get("n_routed_experts"),
            "experts_per_token": architecture.get("num_experts_per_tok"),
            "required_features": len(features),
            "unsupported_features": unsupported if isinstance(unsupported, list) else [],
        }
    if kind == "moe_packing_contract_json":
        verdict = data.get("verdict", {})
        checks = data.get("checks", [])
        layout_checks = [check for check in checks if isinstance(check, dict) and check.get("layout_verified") is not None]
        layout_verified = [check for check in layout_checks if check.get("layout_verified") is True]
        return {
            "checks": len(checks) if isinstance(checks, list) else None,
            "moe_packing_ready": verdict.get("moe_packing_ready"),
            "tl2_3d": verdict.get("merged_3d_tl2_supported"),
            "i2sr_3d": verdict.get("merged_3d_i2s_i2sr_supported"),
            "dense_2d_control": verdict.get("dense_2d_i2s_control_supported"),
            "layout_verified": len(layout_verified),
            "layout_checks": len(layout_checks),
            "blockers": data.get("blockers", []),
        }
    if kind == "moe_tl2_runtime_contract_json":
        byte_probe = data.get("byte_size_probe", {})
        checks = data.get("checks", [])
        failed = [check.get("name") for check in checks if isinstance(check, dict) and not check.get("passed")]
        return {
            "tl2_moe_runtime_ready": data.get("tl2_moe_runtime_ready"),
            "checks": len(checks) if isinstance(checks, list) else None,
            "failed": failed,
            "underreport_bytes": byte_probe.get("underreport_bytes") if isinstance(byte_probe, dict) else None,
            "underreport_ratio": byte_probe.get("underreport_ratio") if isinstance(byte_probe, dict) else None,
            "blockers": data.get("blockers", []),
        }
    if kind == "unblock_requirements_json":
        requirements = data.get("requirements", [])
        missing = [item for item in requirements if isinstance(item, dict) and item.get("status") == "missing"]
        fork = data.get("candidate_fork_probe", {})
        return {
            "missing_count": data.get("missing_count"),
            "requirements": len(requirements) if isinstance(requirements, list) else None,
            "can_continue_productively_without_input": data.get("can_continue_productively_without_input"),
            "objective_status": data.get("objective_status"),
            "candidate_fork_reachable": fork.get("reachable") if isinstance(fork, dict) else None,
            "missing": [item.get("name") for item in missing],
        }
    if kind == "tiny_qwen2moe_fixture_json":
        smoke = data.get("smoke", {}) if isinstance(data.get("smoke"), dict) else {}
        rss = data.get("rss", {}) if isinstance(data.get("rss"), dict) else {}
        gates = data.get("gates", {}) if isinstance(data.get("gates"), dict) else {}
        return {
            "passed": data.get("passed"),
            "gguf_mib": data.get("gguf_mib"),
            "architecture": smoke.get("architecture"),
            "expert_count": smoke.get("expert_count"),
            "expert_used_count": smoke.get("expert_used_count"),
            "model_params_m": smoke.get("model_params_m"),
            "cpu_buffer_mib": smoke.get("cpu_buffer_mib"),
            "prompt_eval_tok_s": smoke.get("prompt_eval_tok_s"),
            "decode_tok_s": smoke.get("decode_tok_s"),
            "max_rss_mib": rss.get("max_rss_mib"),
            "gates": gates,
        }
    if kind == "tiny_qwen2moe_ternary_i2sr_fixture_json":
        smoke = data.get("smoke", {}) if isinstance(data.get("smoke"), dict) else {}
        rss = data.get("rss", {}) if isinstance(data.get("rss"), dict) else {}
        gates = data.get("gates", {}) if isinstance(data.get("gates"), dict) else {}
        conversion = data.get("conversion_summary", {}) if isinstance(data.get("conversion_summary"), dict) else {}
        return {
            "passed": data.get("passed"),
            "gguf_mib": data.get("gguf_mib"),
            "architecture": smoke.get("architecture"),
            "expert_count": smoke.get("expert_count"),
            "expert_used_count": smoke.get("expert_used_count"),
            "model_params_m": smoke.get("model_params_m"),
            "cpu_buffer_mib": smoke.get("cpu_buffer_mib"),
            "prompt_eval_tok_s": smoke.get("prompt_eval_tok_s"),
            "decode_tok_s": smoke.get("decode_tok_s"),
            "max_rss_mib": rss.get("max_rss_mib"),
            "ternary_i2s_packed": conversion.get("ternary_i2s_packed"),
            "row_scale_i2s_packed": conversion.get("row_scale_i2s_packed"),
            "output_ftype_name": conversion.get("output_ftype_name"),
            "packed_i2s_bytes": conversion.get("packed_i2s_bytes"),
            "gates": gates,
        }
    if kind == "tiny_qwen2moe_expert_scaling_json":
        rows = data.get("rows", []) if isinstance(data.get("rows"), list) else []
        passed_rows = [row for row in rows if isinstance(row, dict) and row.get("passed")]
        decode_values = [
            row.get("runtime", {}).get("decode_tok_s")
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("runtime"), dict)
        ]
        finite_decode = [float(value) for value in decode_values if isinstance(value, (int, float)) and math.isfinite(float(value))]
        return {
            "passed": data.get("passed"),
            "rows": len(rows),
            "passed_rows": len(passed_rows),
            "min_decode_tok_s": min(finite_decode) if finite_decode else None,
            "max_decode_tok_s": max(finite_decode) if finite_decode else None,
        }
    if kind == "math_json":
        aggregate = data.get("aggregate", {})
        mean_abs = aggregate.get("mean_abs_ternary_repo_formula", {})
        return {
            "trials": data.get("trials"),
            "theoretical_mean_abs_relative_fro_error": data.get("theoretical_mean_abs_relative_fro_error"),
            "relative_output_fro_error_mean": mean_abs.get("relative_output_fro_error", {}).get("mean") if isinstance(mean_abs, dict) else None,
        }
    return {}


def build_manifest(repo_root: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for item in ARTIFACTS:
        path = repo_root / item["path"]
        entry: dict[str, Any] = {
            "label": item["label"],
            "kind": item["kind"],
            "path": item["path"],
            "exists": path.exists(),
        }
        if path.exists():
            entry["size_bytes"] = path.stat().st_size
            entry["sha256"] = sha256_file(path)
            entry["metrics"] = extract_metrics(item["kind"], path)
        entries.append(entry)
    missing = [entry["label"] for entry in entries if not entry["exists"]]
    return {
        "schema": "bitnet-evidence-manifest-v1",
        "artifact_count": len(entries),
        "missing_count": len(missing),
        "missing": missing,
        "entries": entries,
    }


def fmt_metric(value: Any) -> str:
    if isinstance(value, float) and math.isfinite(value):
        return f"{value:.6g}"
    return "-" if value is None else str(value)


def build_report(manifest: dict[str, Any]) -> str:
    rows = []
    for entry in manifest["entries"]:
        metrics = entry.get("metrics", {})
        summary = ""
        if entry["kind"] == "perplexity_json":
            summary = f"ppl={fmt_metric(metrics.get('perplexity'))}, tokens={metrics.get('eval_tokens', '-')}"
        elif entry["kind"] == "lm_eval_json":
            summary = f"mean={fmt_metric(metrics.get('selected_mean'))}, tasks={metrics.get('selected_tasks', '-')}, samples={metrics.get('samples', '-')}"
        elif entry["kind"] == "gguf_summary_json":
            summary = (
                f"rows={metrics.get('rows', '-')}, failed={len(metrics.get('failed', []))}, "
                f"nan={len(metrics.get('nan_ppl', []))}, catastrophic={len(metrics.get('catastrophic_ppl', []))}, "
                f"max_ppl={fmt_metric(metrics.get('max_finite_ppl'))}"
            )
        elif entry["kind"] == "thread_scaling_json":
            summary = f"rows={metrics.get('rows', '-')}, max_prefill={fmt_metric(metrics.get('max_prefill_tok_s'))}, max_decode={fmt_metric(metrics.get('max_decode_tok_s'))}"
        elif entry["kind"] == "gguf_memory_json":
            summary = f"rows={metrics.get('rows', '-')}, contexts={metrics.get('contexts', '-')}"
        elif entry["kind"] == "direct_gguf_json":
            summary = (
                f"arch={metrics.get('architecture', '-')}, outtype={metrics.get('outtype', '-')}, "
                f"ternary={metrics.get('ternary_materialized', '-')}, tensors={metrics.get('output_tensors', '-')}, "
                f"reader_rc={metrics.get('gguf_reader_returncode', '-')}, smoke_rc={metrics.get('smoke_returncode', '-')}"
            )
        elif entry["kind"] == "direct_packed_support_json":
            summary = (
                f"dense={metrics.get('direct_dense_gguf_supported', '-')}, "
                f"packed_i2s={metrics.get('direct_packed_i2s_supported', '-')}, "
                f"row_safe={metrics.get('product_safe_row_scale_packed_supported', '-')}"
            )
        elif entry["kind"] == "direct_i2s_json":
            native_consts = metrics.get("has_native_gguf_python_constants")
            if native_consts is None:
                native_consts = {
                    "i2s": metrics.get("has_native_i2s_gguf_python_constants"),
                    "i2sr": metrics.get("has_native_i2sr_gguf_python_constants"),
                }
            summary = (
                f"arch={metrics.get('architecture', '-')}, packed={metrics.get('ternary_i2s_packed', '-')}, "
                f"row_packed={metrics.get('row_scale_i2s_packed', '-')}, "
                f"out_f16={metrics.get('output_f16_tensors', '-')}, tensors={metrics.get('output_tensors', '-')}, "
                f"row_qtype={metrics.get('row_scale_qtype', '-')}, "
                f"native_py_consts={native_consts}"
            )
        elif entry["kind"] == "tl2_scale_json":
            values = metrics.get("results", [])
            summary = "; ".join(
                f"{value.get('label')} err={fmt_metric(value.get('total_relative_fro_error_if_one_scale'))}"
                for value in values
                if isinstance(value, dict)
            )
        elif entry["kind"] == "tl2_row_scale_design_json":
            values = metrics.get("results", [])
            summary = "; ".join(
                (
                    f"{value.get('label')} current={fmt_metric(value.get('current_tl2_error'))}, "
                    f"row_fp16={fmt_metric(value.get('row_fp16_error'))}, "
                    f"scaleMiB={fmt_metric(value.get('row_fp16_scale_mib'))}"
                )
                for value in values
                if isinstance(value, dict)
            )
        elif entry["kind"] == "tl2_row_scale_runtime_contract_json":
            summary = (
                f"ready={metrics.get('ready', '-')}, checks={metrics.get('checks', '-')}, "
                f"failed={len(metrics.get('failed', [])) if isinstance(metrics.get('failed'), list) else '-'}, "
                f"current={fmt_metric(metrics.get('current_tl2_tensor_max_error'))}, "
                f"row_fp16={fmt_metric(metrics.get('row_fp16_error'))}, "
                f"scaleMiB={fmt_metric(metrics.get('row_fp16_scale_mib'))}, "
                f"bench_rows={metrics.get('row_scale_tl2_finite_quality_rows', '-')}"
            )
        elif entry["kind"] == "i2s_format_json":
            summary = (
                f"default_ratio={fmt_metric(metrics.get('default_i2s_to_tq2_ppl_ratio'))}, "
                f"prototype_ratio={fmt_metric(metrics.get('prototype_i2s_to_tq2_ppl_ratio'))}, "
                f"stable_format_required={metrics.get('stable_new_format_required', '-')}"
            )
        elif entry["kind"] == "row_scale_qtype_gate_json":
            summary = (
                f"passed={metrics.get('passed', '-')}, gates={metrics.get('gates', '-')}, "
                f"failed={len(metrics.get('failed', []))}, "
                f"stable_qtype={metrics.get('has_ggml_stable_qtype', '-')}, "
                f"writer={metrics.get('direct_writer_emits_stable_qtype', '-')}, "
                f"stable_quality={metrics.get('stable_benchmark_quality_ok', '-')}, "
                f"stable_max_ppl={fmt_metric(metrics.get('stable_benchmark_max_ppl'))}, "
                f"layout_verified={metrics.get('packing_verification_passed', '-')}"
            )
        elif entry["kind"] == "packing_verify_json":
            summary = (
                f"passed={metrics.get('passed', '-')}, "
                f"checked={metrics.get('checked_tensors', '-')}, "
                f"passed_tensors={metrics.get('passed_tensors', '-')}"
            )
        elif entry["kind"] == "benchmark_coverage_gate_json":
            summary = (
                f"passed={metrics.get('passed', '-')}, "
                f"checks={metrics.get('check_count', '-')}, "
                f"failed={len(metrics.get('failed', []))}"
            )
        elif entry["kind"] == "objective_completion_audit_json":
            summary = (
                f"achieved={metrics.get('objective_achieved', '-')}, "
                f"status={metrics.get('completion_status', '-')}, "
                f"complete={metrics.get('complete_count', '-')}/{metrics.get('check_count', '-')}, "
                f"open={len(metrics.get('partial_or_missing', []))}"
            )
        elif entry["kind"] == "product_scope_gate_json":
            summary = (
                f"scope={metrics.get('scope_status', '-')}, "
                f"supported={metrics.get('supported_claim_count', '-')}, "
                f"unsupported={metrics.get('unsupported_claim_count', '-')}"
            )
        elif entry["kind"] == "bitdistill_reproduction_gate_json":
            summary = (
                f"present={metrics.get('present_rows', '-')}/{metrics.get('rows', '-')}, "
                f"examples={metrics.get('present_with_examples', '-')}, "
                f"full_eval={metrics.get('present_with_full_eval', '-')}, "
                f"ci95={metrics.get('present_with_accuracy_ci95', '-')}, "
                f"paper_complete={metrics.get('paper_style_tensor_complete', '-')}, "
                f"paper_passed={metrics.get('paper_style_tensor_passed', '-')}, "
                f"row_complete={metrics.get('row_scale_complete', '-')}, "
                f"row_passed={metrics.get('row_scale_passed', '-')}, "
                f"confidence={fmt_metric(metrics.get('confidence_level'))}"
            )
        elif entry["kind"] == "bitdistill_paired_predictions_json":
            summary = (
                f"status={metrics.get('status', '-')}, "
                f"complete={metrics.get('complete', '-')}/{metrics.get('rows', '-')}, "
                f"pending={metrics.get('pending', '-')}, "
                f"failed={metrics.get('failed', '-')}"
            )
        elif entry["kind"] == "bitdistill_task_formulation_json":
            summary = (
                f"seq_full={metrics.get('sequence_baselines_full', '-')}, "
                f"causal_rows={metrics.get('causal_rows_materialized', '-')}, "
                f"pending_paper={metrics.get('pending_paper_candidates', '-')}, "
                f"rows={metrics.get('rows', '-')}"
            )
        elif entry["kind"] == "bitdistill_cpu_gate_json":
            summary = (
                f"passed={metrics.get('passed', '-')}, "
                f"input={metrics.get('input_exists', '-')}, "
                f"critical={metrics.get('critical_complete', '-')}/{metrics.get('critical', '-')}, "
                f"full_quality={metrics.get('critical_full_quality', '-')}, "
                f"sample_n={metrics.get('max_eval_samples', '-')}, "
                f"blockers={len(metrics.get('blockers', []))}"
            )
        elif entry["kind"] == "bitdistill_i2sr_gate_json":
            summary = (
                f"passed={metrics.get('passed', '-')}, "
                f"complete={metrics.get('complete', '-')}/{metrics.get('rows', '-')}, "
                f"tasks={metrics.get('tasks', '-')}, scales={metrics.get('scales', '-')}, "
                f"blockers={len(metrics.get('blockers', []))}"
            )
        elif entry["kind"] == "bitdistill_job_monitor_json":
            summary = (
                f"warmup={metrics.get('warmup_step', '-')}/{metrics.get('warmup_max_steps', '-')}, "
                f"progress={fmt_metric(metrics.get('warmup_progress'))}, "
                f"ce={fmt_metric(metrics.get('warmup_latest_ce'))}, "
                f"snapshots={metrics.get('warmup_snapshots', '-')}, "
                f"warnings={len(metrics.get('warmup_warnings', [])) if isinstance(metrics.get('warmup_warnings'), list) else '-'}, "
                f"downstream={metrics.get('downstream_jobs', '-')}"
            )
        elif entry["kind"] == "bitdistill_dependency_graph_json":
            summary = (
                f"ready={metrics.get('ready', '-')}, "
                f"checks={metrics.get('checks', '-')}, failed={metrics.get('failed', '-')}, "
                f"active={metrics.get('active_rows', '-')}/{metrics.get('deduped_rows', '-')}, "
                f"warmup={metrics.get('warmup_step', '-')}/{metrics.get('warmup_max_steps', '-')}, "
                f"warnings={len(metrics.get('warnings', [])) if isinstance(metrics.get('warnings'), list) else '-'}, "
                f"blockers={len(metrics.get('blockers', [])) if isinstance(metrics.get('blockers'), list) else '-'}"
            )
        elif entry["kind"] == "bitdistill_postprocess_dependency_json":
            summary = (
                f"passed={metrics.get('passed', '-')}, "
                f"checks={metrics.get('checks', '-')}, failed={metrics.get('failed', '-')}, "
                f"expected={metrics.get('expected_jobs', '-')}, "
                f"warmup={metrics.get('warmup_jobs', '-')}, downstream={metrics.get('downstream_jobs', '-')}, extra={metrics.get('extra_jobs', '-')}, "
                f"missing={len(metrics.get('missing', [])) if isinstance(metrics.get('missing'), list) else '-'}, "
                f"postprocess={metrics.get('postprocess_job', '-')}"
            )
        elif entry["kind"] in {"bitdistill_postprocess_submission_json", "bitdistill_afterany_postprocess_submission_json"}:
            summary = (
                f"submitted={metrics.get('submitted', '-')}, "
                f"job={metrics.get('job_id', '-')}, dep={metrics.get('dependency_type', '-')}, "
                f"producers={metrics.get('producer_jobs', '-')}, "
                f"warmup={metrics.get('warmup_jobs', '-')}, downstream={metrics.get('downstream_jobs', '-')}, "
                f"extra={metrics.get('extra_jobs', '-')}"
            )
        elif entry["kind"] == "bitdistill_warmup_finalizer_submission_json":
            summary = (
                f"submitted={metrics.get('submitted', '-')}, "
                f"job={metrics.get('job_id', '-')}, dep={metrics.get('dependency_type', '-')}, "
                f"warmup={metrics.get('warmup_jobs', '-')}"
            )
        elif entry["kind"] == "bitdistill_warmup_finalizer_json":
            summary = (
                f"passed={metrics.get('passed', '-')}, "
                f"steps={metrics.get('steps', '-')}, failed={metrics.get('failed', '-')}"
            )
        elif entry["kind"] == "bitdistill_warmup_health_json":
            summary = (
                f"passed={metrics.get('passed', '-')}, "
                f"checks={metrics.get('checks', '-')}, failed={metrics.get('failed', '-')}, "
                f"warnings={metrics.get('warnings', '-')}, "
                f"warmup={metrics.get('step', '-')}/{metrics.get('max_steps', '-')}, "
                f"progress={fmt_metric(metrics.get('progress'))}, "
                f"ce={fmt_metric(metrics.get('latest_ce'))}, "
                f"sec_step={fmt_metric(metrics.get('seconds_per_step'))}, "
                f"snapshots={metrics.get('snapshot_count', '-')}"
            )
        elif entry["kind"] == "bitdistill_producer_script_audit_json":
            summary = (
                f"passed={metrics.get('passed', '-')}, "
                f"checks={metrics.get('checks', '-')}, failed={metrics.get('failed', '-')}, "
                f"downstream={metrics.get('downstream_jobs', '-')}, "
                f"cpu={metrics.get('cpu_job', '-')}, i2sr={metrics.get('i2sr_job', '-')}, "
                f"post={metrics.get('strict_postprocess_job', '-')}/{metrics.get('afterany_postprocess_job', '-')}"
            )
        elif entry["kind"] == "bitdistill_job_matrix_json":
            summary = (
                f"passed={metrics.get('passed', '-')}, "
                f"configured={metrics.get('configured_rows', '-')}/{metrics.get('expected_rows', '-')}, "
                f"observed={metrics.get('observed_rows', '-')}, "
                f"states={metrics.get('job_states', '-')}, "
                f"inferred_rows={metrics.get('inferred_rows', '-')}, "
                f"blockers={len(metrics.get('blockers', [])) if isinstance(metrics.get('blockers'), list) else '-'}"
            )
        elif entry["kind"] == "bitdistill_active_goal_json":
            summary = (
                f"achieved={metrics.get('objective_achieved', '-')}, "
                f"status={metrics.get('completion_status', '-')}, "
                f"complete={metrics.get('complete_count', '-')}/{metrics.get('check_count', '-')}, "
                f"pending={metrics.get('pending_count', '-')}, "
                f"warmup={metrics.get('warmup_step', '-')}/{metrics.get('warmup_max_steps', '-')}, "
                f"row_complete={metrics.get('row_scale_complete', '-')}, "
                f"i2sr={metrics.get('i2sr_passed', '-')}, cpu={metrics.get('cpu_passed', '-')}"
            )
        elif entry["kind"] == "bitdistill_snapshot_integrity_json":
            summary = (
                f"passed={metrics.get('passed', '-')}, "
                f"snapshots={metrics.get('passed_snapshots', '-')}/{metrics.get('snapshot_count', '-')}, "
                f"step={metrics.get('step', '-')}, scale={metrics.get('scale_mode', '-')}, "
                f"ternary={metrics.get('ternary_weight_count', '-')}/{metrics.get('expected_ternary', '-')}, "
                f"row_scales={metrics.get('row_scale_count', '-')}, "
                f"tensor_scales={metrics.get('tensor_scale_count', '-')}, "
                f"codes={metrics.get('codes_valid', '-')}"
            )
        elif entry["kind"] == "bitdistill_smoke_contract_json":
            summary = (
                f"passed={metrics.get('passed', '-')}, "
                f"checks={metrics.get('check_count', '-')}, failed={metrics.get('failed_count', '-')}, "
                f"continued=bitlinear{metrics.get('continued_bitlinear', '-')}/subln{metrics.get('continued_subln', '-')}, "
                f"task=bitlinear{metrics.get('task_bitlinear', '-')}/subln{metrics.get('task_subln', '-')}"
            )
        elif entry["kind"] == "bitdistill_variant_summary_json":
            summary = (
                f"rows={metrics.get('materialized_rows', '-')}/{metrics.get('rows', '-')}, "
                f"tasks={metrics.get('tasks', '-')}"
            )
        elif entry["kind"] == "bitdistill_causal_summary_json":
            summary = (
                f"passed={metrics.get('passed', '-')}, "
                f"rows={metrics.get('materialized_rows', '-')}/{metrics.get('rows', '-')}, "
                f"verdicts={metrics.get('passed_verdicts', '-')}/{metrics.get('verdicts', '-')}, "
                f"tasks={metrics.get('tasks', '-')}"
            )
        elif entry["kind"] == "bitdistill_loss_scale_json":
            summary = (
                f"rows={metrics.get('materialized_rows', '-')}/{metrics.get('rows', '-')}, "
                f"gamma={fmt_metric(metrics.get('paper_gamma'))}, "
                f"projected_attn_ce=[{fmt_metric(metrics.get('projected_attention_to_ce_min'))}, "
                f"{fmt_metric(metrics.get('projected_attention_to_ce_max'))}]"
            )
        elif entry["kind"] == "i2sr_submodule_promotion_audit_json":
            summary = (
                f"ready={metrics.get('promotion_ready', '-')}, "
                f"active={metrics.get('active_runtime_support', '-')}, "
                f"patch_applies={metrics.get('patch_applies_cleanly', '-')}, "
                f"submodule={metrics.get('submodule_short', '-')}, "
                f"blockers={len(metrics.get('blockers', []))}"
            )
        elif entry["kind"] == "i2sr_promotion_handoff_json":
            summary = (
                f"ready={metrics.get('ready_for_handoff', '-')}, "
                f"root_clean={metrics.get('root_clean', '-')}, "
                f"submodule_clean={metrics.get('submodule_clean', '-')}, "
                f"root_patch={metrics.get('root_patch_applies', '-')}, "
                f"submodule_patch={metrics.get('submodule_patch_applies', '-')}, "
                f"fork_reachable={metrics.get('candidate_fork_reachable', '-')}, "
                f"blockers={len(metrics.get('blockers', []))}"
            )
        elif entry["kind"] == "moe_support_json":
            summary = (
                f"present={metrics.get('present_checks', '-')}/{metrics.get('checks', '-')}, "
                f"gates={metrics.get('gates', '-')}, failed={len(metrics.get('failed_gates', []))}, "
                f"kimi_artifacts={metrics.get('local_kimi_artifacts', '-')}, "
                f"tiny_qwen2moe={metrics.get('tiny_qwen2moe_passed', '-')}, "
                f"tiny_i2sr_moe={metrics.get('tiny_qwen2moe_ternary_i2sr_passed', '-')}"
            )
        elif entry["kind"] == "kimi_config_feasibility_json":
            summary = (
                f"passed={metrics.get('passed', '-')}, model={metrics.get('model_type', '-')}, "
                f"experts={metrics.get('routed_experts', '-')}, topk={metrics.get('experts_per_token', '-')}, "
                f"unsupported={len(metrics.get('unsupported_features', []))}/{metrics.get('required_features', '-')}"
            )
        elif entry["kind"] == "moe_packing_contract_json":
            summary = (
                f"ready={metrics.get('moe_packing_ready', '-')}, "
                f"tl2_3d={metrics.get('tl2_3d', '-')}, "
                f"i2sr_3d={metrics.get('i2sr_3d', '-')}, "
                f"control_2d={metrics.get('dense_2d_control', '-')}, "
                f"layout={metrics.get('layout_verified', '-')}/{metrics.get('layout_checks', '-')}, "
                f"blockers={len(metrics.get('blockers', []))}"
            )
        elif entry["kind"] == "moe_tl2_runtime_contract_json":
            summary = (
                f"ready={metrics.get('tl2_moe_runtime_ready', '-')}, "
                f"checks={metrics.get('checks', '-')}, "
                f"failed={len(metrics.get('failed', []))}, "
                f"underreport={metrics.get('underreport_bytes', '-')}, "
                f"ratio={fmt_metric(metrics.get('underreport_ratio'))}, "
                f"blockers={len(metrics.get('blockers', []))}"
            )
        elif entry["kind"] == "unblock_requirements_json":
            summary = (
                f"missing={metrics.get('missing_count', '-')}/{metrics.get('requirements', '-')}, "
                f"can_continue={metrics.get('can_continue_productively_without_input', '-')}, "
                f"fork_reachable={metrics.get('candidate_fork_reachable', '-')}"
            )
        elif entry["kind"] == "tiny_qwen2moe_fixture_json":
            summary = (
                f"passed={metrics.get('passed', '-')}, "
                f"arch={metrics.get('architecture', '-')}, "
                f"experts={metrics.get('expert_used_count', '-')}/{metrics.get('expert_count', '-')}, "
                f"file={fmt_metric(metrics.get('gguf_mib'))} MiB, "
                f"decode={fmt_metric(metrics.get('decode_tok_s'))} tok/s, "
                f"rss={fmt_metric(metrics.get('max_rss_mib'))} MiB"
            )
        elif entry["kind"] == "tiny_qwen2moe_ternary_i2sr_fixture_json":
            summary = (
                f"passed={metrics.get('passed', '-')}, "
                f"arch={metrics.get('architecture', '-')}, "
                f"experts={metrics.get('expert_used_count', '-')}/{metrics.get('expert_count', '-')}, "
                f"qtype={metrics.get('output_ftype_name', '-')}, "
                f"packed={metrics.get('ternary_i2s_packed', '-')}, "
                f"row_scale={metrics.get('row_scale_i2s_packed', '-')}, "
                f"file={fmt_metric(metrics.get('gguf_mib'))} MiB, "
                f"decode={fmt_metric(metrics.get('decode_tok_s'))} tok/s, "
                f"rss={fmt_metric(metrics.get('max_rss_mib'))} MiB"
            )
        elif entry["kind"] == "tiny_qwen2moe_expert_scaling_json":
            summary = (
                f"passed={metrics.get('passed', '-')}, "
                f"rows={metrics.get('passed_rows', '-')}/{metrics.get('rows', '-')}, "
                f"decode=[{fmt_metric(metrics.get('min_decode_tok_s'))}, {fmt_metric(metrics.get('max_decode_tok_s'))}] tok/s"
            )
        elif entry["kind"] == "math_json":
            summary = f"trials={metrics.get('trials', '-')}, rel_error={fmt_metric(metrics.get('relative_output_fro_error_mean'))}"
        rows.append(
            [
                str(entry["label"]),
                str(entry["kind"]),
                "yes" if entry["exists"] else "no",
                str(entry.get("size_bytes", "-")),
                str(entry.get("sha256", "-"))[:12],
                summary,
            ]
        )
    lines = [
        f"# Evidence Manifest, {DATE}",
        f"Artifacts: `{manifest['artifact_count']}`. Missing: `{manifest['missing_count']}`.",
        "| label | kind | exists | size bytes | sha256 prefix | parsed summary |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument(
        "--allow-missing-label",
        action="append",
        default=[],
        help="Artifact label that may be missing without making this preflight manifest fail.",
    )
    args = parser.parse_args()

    repo_root = Path.cwd()
    manifest = build_manifest(repo_root)
    report = build_report(manifest)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report + "\n", encoding="utf-8")
    print(report)
    blocking_missing = sorted(set(manifest["missing"]) - set(args.allow_missing_label))
    if blocking_missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
