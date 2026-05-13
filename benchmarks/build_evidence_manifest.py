#!/usr/bin/env python3
"""Build a compact manifest for cited benchmark evidence artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


ARTIFACTS: list[dict[str, str]] = [
    # Tracked reports.
    {"label": "README", "kind": "tracked_report", "path": "README.md"},
    {"label": "side_by_side_report", "kind": "tracked_report", "path": "benchmarks/results/qwen_side_by_side_2026-05-05.md"},
    {"label": "paired_row_minus_fp_report", "kind": "tracked_report", "path": "benchmarks/results/paired_row_densehead_minus_fp_2026-05-13.md"},
    {"label": "paired_row_minus_ptq_report", "kind": "tracked_report", "path": "benchmarks/results/paired_row_densehead_minus_ptq_2026-05-13.md"},
    {"label": "publishable_claims", "kind": "tracked_report", "path": "benchmarks/results/publishable_claims_2026-05-05.md"},
    {"label": "progress_audit", "kind": "tracked_report", "path": "benchmarks/results/progress_audit_2026-05-05.md"},
    {"label": "active_goal_audit", "kind": "tracked_report", "path": "benchmarks/results/active_goal_completion_audit_2026-05-05.md"},
    {"label": "objective_completion_audit", "kind": "tracked_report", "path": "benchmarks/results/objective_completion_audit_2026-05-13.md"},
    {"label": "benchmark_coverage_gate_report", "kind": "tracked_report", "path": "benchmarks/results/benchmark_coverage_gate_2026-05-13.md"},
    {"label": "direct_static_ternary_gguf_report", "kind": "tracked_report", "path": "benchmarks/results/direct_static_ternary_gguf_2026-05-13.md"},
    {"label": "direct_packed_gguf_support_report", "kind": "tracked_report", "path": "benchmarks/results/direct_packed_gguf_support_2026-05-13.md"},
    {"label": "direct_i2s_scalar_gguf_report", "kind": "tracked_report", "path": "benchmarks/results/direct_i2s_scalar_gguf_2026-05-13.md"},
    {"label": "direct_row_i2s_qwen05b_report", "kind": "tracked_report", "path": "benchmarks/results/direct_row_i2s_qwen05b_2026-05-13.md"},
    {"label": "tl2_shape_report", "kind": "tracked_report", "path": "benchmarks/results/tl2_shape_support_audit_2026-05-05.md"},
    {"label": "tl2_probe_report", "kind": "tracked_report", "path": "benchmarks/results/qwen05b_tl2_probe_2026-05-05.md"},
    {"label": "tl2_scale_report", "kind": "tracked_report", "path": "benchmarks/results/tl2_scale_semantics_2026-05-05.md"},
    {"label": "i2s_row_scale_format_report", "kind": "tracked_report", "path": "benchmarks/results/i2s_row_scale_format_audit_2026-05-13.md"},
    {"label": "row_scale_qtype_productization_gate_report", "kind": "tracked_report", "path": "benchmarks/results/row_scale_qtype_productization_gate_2026-05-13.md"},
    {"label": "row_scale_qtype_i2sr_active_patch_gate_report", "kind": "tracked_report", "path": "benchmarks/results/row_scale_qtype_productization_gate_i2sr_active_patch_2026-05-13.md"},
    {"label": "i2sr_qwen15b_candidate_report", "kind": "tracked_report", "path": "benchmarks/results/i2sr_qwen15b_candidate_2026-05-13.md"},
    {"label": "i2sr_x86act_fix_report", "kind": "tracked_report", "path": "benchmarks/results/i2sr_x86act_fix_2026-05-13.md"},
    {"label": "i2s_packing_layout_verify_report", "kind": "tracked_report", "path": "benchmarks/results/i2s_packing_layout_verify_2026-05-13.md"},
    {"label": "i2sr_rss_report", "kind": "tracked_report", "path": "benchmarks/results/i2sr_rss_2026-05-13.md"},
    {"label": "moe_report", "kind": "tracked_report", "path": "benchmarks/results/moe_support_audit_2026-05-05.md"},
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
    {"label": "i2s_row_scale_format_json", "kind": "i2s_format_json", "path": "benchmark_results/i2s_row_scale_format_audit_2026-05-13.json"},
    {"label": "row_scale_qtype_productization_gate_json", "kind": "row_scale_qtype_gate_json", "path": "benchmark_results/row_scale_qtype_productization_gate_2026-05-13.json"},
    {"label": "row_scale_qtype_i2sr_active_patch_gate_json", "kind": "row_scale_qtype_gate_json", "path": "benchmark_results/row_scale_qtype_productization_gate_i2sr_active_patch_2026-05-13.json"},
    {"label": "i2s_packing_layout_verify_json", "kind": "packing_verify_json", "path": "benchmark_results/i2s-packing-layout-verify-2026-05-13/summary.json"},
    {"label": "benchmark_coverage_gate_json", "kind": "benchmark_coverage_gate_json", "path": "benchmark_results/benchmark_coverage_gate_2026-05-13.json"},
    {"label": "objective_completion_audit_json", "kind": "objective_completion_audit_json", "path": "benchmark_results/objective_completion_audit_2026-05-13.json"},
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
        "# Evidence Manifest, 2026-05-13",
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
    args = parser.parse_args()

    repo_root = Path.cwd()
    manifest = build_manifest(repo_root)
    report = build_report(manifest)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report + "\n", encoding="utf-8")
    print(report)
    if manifest["missing_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
