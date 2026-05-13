#!/usr/bin/env python3
"""Audit the current repo against the user-facing benchmark objective.

This is intentionally stricter than the benchmark coverage gate. A green
coverage gate says the cited dense-Qwen evidence is internally consistent; this
audit asks whether the full objective has actually been achieved.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from pathlib import Path
from typing import Any


DATE = "2026-05-13"
EXPECTED_LM_EVAL_SAMPLES = 22382
EXPECTED_RSS_CONTEXTS = [512, 2048, 8192, 32768]
SELECTED_METRICS = {
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


TERNARY_EXPORTS = {
    "Qwen2.5-1.5B repaired hidden-MSE step-5000": {
        "state": "checkpoints/qwen2.5-1.5b-fineweb-edu/step-5000/ternary_state_dict.pt",
        "dense": "checkpoints/qwen2.5-1.5b-fineweb-edu/step-5000/model.safetensors",
        "expected_ternary": 197,
    },
    "Qwen2.5-0.5B step-1000": {
        "state": "checkpoints/qwen2.5-0.5b-fineweb-edu-12/step-1000/ternary_state_dict.pt",
        "dense": "checkpoints/qwen2.5-0.5b-fineweb-edu-12/step-1000/model.safetensors",
        "expected_ternary": 169,
    },
    "Qwen2.5-1.5B KL row-scale dense-head step-5000": {
        "state": "checkpoints/qwen2.5-1.5b-fineweb-edu-klonly-row-notiehead-5000/step-5000/ternary_state_dict.pt",
        "dense": "checkpoints/qwen2.5-1.5b-fineweb-edu-klonly-row-notiehead-5000/step-5000/model.safetensors",
        "expected_ternary": 196,
    },
}


PPL_RUNS = {
    "FP WikiText": "benchmark_results/quality-9735/qwen15b_fp_wikitext.json",
    "FP FineWeb": "benchmark_results/quality-9735/qwen15b_fp_fineweb_heldout.json",
    "naive PTQ WikiText": "benchmark_results/quality-ptq-qwen15b/qwen15b_naive_ptq_wikitext.json",
    "naive PTQ FineWeb": "benchmark_results/quality-ptq-qwen15b/qwen15b_naive_ptq_fineweb_heldout.json",
    "hidden-MSE QAT WikiText": "benchmark_results/quality-9735/qwen15b_ternary_wikitext.json",
    "hidden-MSE QAT FineWeb": "benchmark_results/quality-9735/qwen15b_ternary_fineweb_heldout.json",
    "KL-only QAT WikiText": "benchmark_results/quality-qwen15b-klonly-5000/qwen15b_ternary_wikitext.json",
    "KL-only QAT FineWeb": "benchmark_results/quality-qwen15b-klonly-5000/qwen15b_ternary_fineweb_heldout.json",
    "dense-head QAT WikiText": "benchmark_results/quality-qwen15b-klonly-notiehead-5000/qwen15b_ternary_wikitext.json",
    "dense-head QAT FineWeb": "benchmark_results/quality-qwen15b-klonly-notiehead-5000/qwen15b_ternary_fineweb_heldout.json",
    "row-scale dense-head QAT WikiText": "benchmark_results/quality-qwen15b-klonly-row-notiehead-5000/qwen15b_ternary_wikitext.json",
    "row-scale dense-head QAT FineWeb": "benchmark_results/quality-qwen15b-klonly-row-notiehead-5000/qwen15b_ternary_fineweb_heldout.json",
}


LM_EVAL_RUNS = {
    "FP": "benchmark_results/lm-eval-qwen15b-full10/qwen15b_fp.json",
    "naive PTQ": "benchmark_results/lm-eval-qwen15b-full10/qwen15b_naive_ptq.json",
    "hidden-MSE QAT": "benchmark_results/lm-eval-qwen15b-full10/qwen15b_qat_ternary.json",
    "KL-only QAT": "benchmark_results/lm-eval-qwen15b-klonly-full10/qwen15b_qat_ternary.json",
    "dense-head QAT": "benchmark_results/lm-eval-qwen15b-klonly-notiehead-full10/qwen15b_qat_ternary.json",
    "row-scale dense-head QAT": "benchmark_results/lm-eval-qwen15b-klonly-row-notiehead-full10/qwen15b_qat_ternary.json",
}


CPU_ROWS = {
    "FP F16": ("benchmark_results/gguf-qwen15b-row-i2s-prototype-suite/summary.json", "qwen15b_fp_f16"),
    "FP Q8_0": ("benchmark_results/gguf-qwen15b-row-i2s-prototype-suite/summary.json", "qwen15b_fp_q8_0"),
    "FP Q4_K_M": ("benchmark_results/gguf-qwen15b-row-i2s-prototype-suite/summary.json", "qwen15b_fp_q4_k_m"),
    "row-scale TQ2_0": ("benchmark_results/gguf-qwen15b-row-i2s-prototype-suite/summary.json", "qwen15b_klonly_row_notie_static_ternary_tq2_0"),
    "row-scale I2_S prototype": ("benchmark_results/gguf-qwen15b-row-i2s-heapfix-confirm/summary.json", "qwen15b_klonly_row_notie_static_ternary_i2_s_rowscale"),
    "row-scale I2_SR candidate": ("benchmark_results/i2sr-row-scale-qwen15b-x86act-suite-2026-05-13/summary.json", "qwen15b_klonly_row_notie_static_ternary_i2_sr_x86act"),
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def metric_value(task_results: dict[str, Any], metric: str) -> float | None:
    for key in (metric, f"{metric},none"):
        value = task_results.get(key)
        if finite_number(value):
            return float(value)
    return None


def status_rank(status: str) -> int:
    return {"complete": 0, "partial": 1, "not_complete": 2}.get(status, 2)


def add_row(rows: list[dict[str, Any]], requirement: str, status: str, evidence: str, remaining_gap: str = "") -> None:
    rows.append(
        {
            "requirement": requirement,
            "status": status,
            "evidence": evidence,
            "remaining_gap": remaining_gap if status != "complete" else "",
        }
    )


def load_torch_state_metadata(path: Path) -> dict[str, Any]:
    warnings.filterwarnings("ignore", category=FutureWarning)
    import torch

    try:
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            state = torch.load(path, map_location="cpu")
    except Exception:
        state = torch.load(path, map_location="cpu")

    if not isinstance(state, dict):
        return {"is_dict": False, "key_count": None, "ternary_weight_count": 0, "weight_scale_count": 0}
    keys = list(state)
    ternary_keys = [key for key in keys if key.endswith(".ternary_weight")]
    scale_keys = [key for key in keys if key.endswith(".weight_scale")]
    row_scale_count = 0
    scalar_scale_count = 0
    for key in scale_keys:
        shape = tuple(getattr(state[key], "shape", ()))
        if len(shape) == 0 or shape == (1,):
            scalar_scale_count += 1
        else:
            row_scale_count += 1
    return {
        "is_dict": True,
        "key_count": len(keys),
        "ternary_weight_count": len(ternary_keys),
        "weight_scale_count": len(scale_keys),
        "row_scale_count": row_scale_count,
        "scalar_scale_count": scalar_scale_count,
        "first_ternary_keys": ternary_keys[:3],
    }


def audit_exports(root: Path, rows: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    export_metrics = {}
    all_ok = True
    evidence_parts = []
    for label, config in TERNARY_EXPORTS.items():
        state_path = root / config["state"]
        dense_path = root / config["dense"]
        if not state_path.exists():
            all_ok = False
            export_metrics[label] = {"exists": False}
            evidence_parts.append(f"{label}: missing ternary_state_dict.pt")
            continue
        meta = load_torch_state_metadata(state_path)
        expected = int(config["expected_ternary"])
        ok = meta.get("ternary_weight_count") == expected and dense_path.exists()
        all_ok = all_ok and ok
        export_metrics[label] = {
            **meta,
            "state": config["state"],
            "dense": config["dense"],
            "dense_exists": dense_path.exists(),
            "expected_ternary_weight_count": expected,
        }
        evidence_parts.append(
            f"{label}: ternary={meta.get('ternary_weight_count')}/{expected}, "
            f"scales={meta.get('weight_scale_count')}, dense={dense_path.exists()}"
        )
    metrics["ternary_exports"] = export_metrics
    add_row(
        rows,
        "Fix FSDP ternary export bug and re-export Qwen2.5-1.5B step-5000; target 197 ternary linear keys, not 1",
        "complete" if all_ok else "not_complete",
        "; ".join(evidence_parts),
        "One or more checked ternary exports is missing or has the wrong ternary key count.",
    )


def audit_prompt_suites(root: Path, rows: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    suites = {
        "Qwen2.5-1.5B repaired step-5000": "benchmark_results/generation/qwen15b_step5000_core_cpu_16tok.jsonl",
        "Qwen2.5-0.5B step-1000": "benchmark_results/generation/qwen05b_step1000_core_cpu.jsonl",
    }
    suite_metrics = {}
    evidence_parts = []
    all_ok = True
    for label, rel_path in suites.items():
        path = root / rel_path
        line_count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()) if path.exists() else 0
        suite_metrics[label] = {"path": rel_path, "line_count": line_count, "exists": path.exists()}
        all_ok = all_ok and path.exists() and line_count >= 5
        evidence_parts.append(f"{label}: {line_count} prompts")
    metrics["prompt_suites"] = suite_metrics
    add_row(
        rows,
        "Run fixed prompt suites for repaired 1.5B and complete 0.5B checkpoints",
        "complete" if all_ok else "partial",
        "; ".join(evidence_parts),
        "Prompt suites are sanity checks only; quality claims must come from PPL/lm-eval artifacts.",
    )


def audit_ppl(root: Path, rows: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    ppl_metrics = {}
    all_ok = True
    for label, rel_path in PPL_RUNS.items():
        path = root / rel_path
        data = read_json(path) if path.exists() else {}
        ppl = data.get("perplexity")
        tokens = data.get("eval_tokens")
        ok = path.exists() and finite_number(ppl) and finite_number(tokens) and float(tokens) >= 32704
        all_ok = all_ok and ok
        ppl_metrics[label] = {"path": rel_path, "perplexity": ppl, "eval_tokens": tokens, "ok": ok}
    metrics["ppl"] = ppl_metrics
    best = ppl_metrics.get("row-scale dense-head QAT WikiText", {}).get("perplexity")
    fp = ppl_metrics.get("FP WikiText", {}).get("perplexity")
    ptq = ppl_metrics.get("naive PTQ WikiText", {}).get("perplexity")
    add_row(
        rows,
        "Add WikiText and FineWeb heldout perplexity for FP, naive PTQ, hidden-MSE QAT, KL QAT, dense-head, and row-scale variants",
        "complete" if all_ok else "partial",
        f"12/12 finite artifacts={all_ok}; FP WikiText={fp}; naive PTQ WikiText={ptq}; best row-scale WikiText={best}",
        "Missing or non-finite PPL artifact.",
    )


def audit_lm_eval(root: Path, rows: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    lm_metrics = {}
    all_ok = True
    for label, rel_path in LM_EVAL_RUNS.items():
        path = root / rel_path
        data = read_json(path) if path.exists() else {}
        results = data.get("results", {})
        samples = data.get("samples", {})
        task_values = {}
        sample_count = 0
        missing = []
        for task, metric in SELECTED_METRICS.items():
            value = metric_value(results.get(task, {}), metric) if isinstance(results.get(task), dict) else None
            if value is None:
                missing.append(task)
            else:
                task_values[task] = value
            task_samples = samples.get(task, [])
            if isinstance(task_samples, list):
                sample_count += len(task_samples)
        mean = sum(task_values.values()) / len(task_values) if task_values else None
        ok = path.exists() and len(task_values) == len(SELECTED_METRICS) and sample_count == EXPECTED_LM_EVAL_SAMPLES
        all_ok = all_ok and ok
        lm_metrics[label] = {
            "path": rel_path,
            "selected_tasks": len(task_values),
            "sample_count": sample_count,
            "selected_mean": mean,
            "missing": missing,
            "ok": ok,
        }
    metrics["lm_eval"] = lm_metrics
    fp_mean = lm_metrics.get("FP", {}).get("selected_mean")
    row_mean = lm_metrics.get("row-scale dense-head QAT", {}).get("selected_mean")
    ptq_mean = lm_metrics.get("naive PTQ", {}).get("selected_mean")
    add_row(
        rows,
        "Add HellaSwag/PIQA/ARC and broader ten-task EleutherAI lm-eval comparisons with logged samples",
        "complete" if all_ok else "partial",
        f"models={len(lm_metrics)}, tasks/model={len(SELECTED_METRICS)}, samples/model={EXPECTED_LM_EVAL_SAMPLES}; FP mean={fp_mean:.6f}; PTQ mean={ptq_mean:.6f}; row-scale mean={row_mean:.6f}",
        "Missing task metric or logged-sample coverage.",
    )


def find_summary_row(summary: dict[str, Any], name: str) -> dict[str, Any] | None:
    for row in summary.get("rows", []):
        if isinstance(row, dict) and row.get("name") == name:
            return row
    return None


def audit_cpu(root: Path, rows: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    cpu_metrics = {}
    all_ok = True
    for label, (rel_path, row_name) in CPU_ROWS.items():
        path = root / rel_path
        row = find_summary_row(read_json(path), row_name) if path.exists() else None
        ppl = row.get("perplexity", {}).get("ppl") if row else None
        prefill = row.get("bench", {}).get("prefill", {}).get("tok_s") if row else None
        decode = row.get("bench", {}).get("decode", {}).get("tok_s") if row else None
        file_mib = row.get("file_mib") if row else None
        ok = all(finite_number(value) for value in (ppl, prefill, decode, file_mib))
        all_ok = all_ok and ok
        cpu_metrics[label] = {
            "path": rel_path,
            "row": row_name,
            "ppl": ppl,
            "prefill_tok_s": prefill,
            "decode_tok_s": decode,
            "file_mib": file_mib,
            "ok": ok,
        }
    rss_path = root / "benchmark_results/gguf-rss-qwen15b-i2sr-x86act-context-2026-05-13/summary.json"
    rss_rows = read_json(rss_path).get("rows", []) if rss_path.exists() else []
    contexts = sorted({int(row["ctx_size"]) for row in rss_rows if isinstance(row, dict) and row.get("returncode") == 0 and finite_number(row.get("ctx_size"))})
    all_ok = all_ok and contexts == EXPECTED_RSS_CONTEXTS
    metrics["cpu"] = cpu_metrics
    metrics["i2sr_rss_contexts"] = contexts
    i2sr = cpu_metrics.get("row-scale I2_SR candidate", {})
    q4 = cpu_metrics.get("FP Q4_K_M", {})
    add_row(
        rows,
        "Measure Xeon model size, quality, prompt throughput, decode throughput, and RSS/context scaling",
        "complete" if all_ok else "partial",
        (
            f"I2_SR PPL={i2sr.get('ppl')}, file={i2sr.get('file_mib'):.1f} MiB, "
            f"prompt={i2sr.get('prefill_tok_s'):.2f} tok/s, decode={i2sr.get('decode_tok_s'):.2f} tok/s; "
            f"Q4_K_M file={q4.get('file_mib'):.1f} MiB; RSS contexts={contexts}"
        ),
        "Missing finite CPU row or four-context RSS probe.",
    )


def parse_macro_delta(path: Path) -> str | None:
    if not path.exists():
        return None
    match = re.search(r"\| macro mean delta \| ([^|]+) \|", path.read_text(encoding="utf-8"))
    return match.group(1).strip() if match else None


def audit_baselines(root: Path, rows: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    deltas = {
        "row_minus_fp": parse_macro_delta(root / "benchmarks/results/paired_row_densehead_minus_fp_2026-05-13.md"),
        "row_minus_ptq": parse_macro_delta(root / "benchmarks/results/paired_row_densehead_minus_ptq_2026-05-13.md"),
        "row_minus_tensor_dense_head": parse_macro_delta(root / "benchmark_results/lm-eval-qwen15b-klonly-row-notiehead-full10/paired_row_densehead_minus_tensor_densehead.md"),
        "row_minus_kl_only": parse_macro_delta(root / "benchmark_results/lm-eval-qwen15b-klonly-row-notiehead-full10/paired_row_densehead_minus_klonly.md"),
    }
    metrics["paired_deltas"] = deltas
    complete = all(deltas.values())
    add_row(
        rows,
        "Add baselines: original FP, naive PTQ, llama.cpp Q4_K_M/Q8_0, QAT with/without hidden MSE, row-scale versus tensor-scale",
        "complete" if complete else "partial",
        (
            f"row-FP={deltas['row_minus_fp']}; row-PTQ={deltas['row_minus_ptq']}; "
            f"row-tensor={deltas['row_minus_tensor_dense_head']}; row-KL={deltas['row_minus_kl_only']}"
        ),
        "One or more paired-delta report is missing.",
    )


def audit_productization(root: Path, rows: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    active = read_json(root / "benchmark_results/row_scale_qtype_productization_gate_2026-05-13.json")
    candidate = read_json(root / "benchmark_results/row_scale_qtype_productization_gate_i2sr_active_patch_2026-05-13.json")
    direct = read_json(root / "benchmark_results/direct_packed_gguf_support_2026-05-13.json")
    tl2 = read_json(root / "benchmark_results/tl2_scale_semantics_2026-05-05.json")
    tl2_row = next((item for item in tl2.get("results", []) if item.get("label") == "qwen15b_row_scale"), {})
    tl2_design = read_json(root / "benchmark_results/tl2_row_scale_design_2026-05-13.json")
    tl2_design_row = next((item for item in tl2_design.get("results", []) if item.get("label") == "qwen15b_row_scale"), {})
    tl2_design_strategies = {
        item.get("name"): item
        for item in tl2_design_row.get("strategies", [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    tl2_row_fp16 = tl2_design_strategies.get("row_exact_fp16", {})
    metrics["productization"] = {
        "active_qtype_gate_passed": active.get("passed"),
        "candidate_patch_gate_passed": candidate.get("passed"),
        "direct_packed_verdict": direct.get("verdict", {}),
        "qwen15b_row_scale_tl2_one_scale_error": tl2_row.get("total_relative_fro_error_if_one_scale"),
        "qwen15b_row_scale_tl2_row_fp16_error": tl2_row_fp16.get("expected_relative_output_rms_error"),
        "qwen15b_row_scale_tl2_row_fp16_scale_mib": tl2_row_fp16.get("scale_mib_fp16"),
    }
    status = "partial"
    add_row(
        rows,
        "Convert repaired checkpoints into GGUF/TL2/I2_S and run actual bitnet.cpp/llama.cpp CPU inference",
        status,
        (
            f"direct dense/scalar writers exist; I2_SR candidate gate={candidate.get('passed')}; "
            f"active default gate={active.get('passed')}; TL2 row-scale one-scale error={tl2_row.get('total_relative_fro_error_if_one_scale')}; "
            f"row-fp16 design error={tl2_row_fp16.get('expected_relative_output_rms_error')} at {tl2_row_fp16.get('scale_mib_fp16')} MiB"
        ),
        "Packed row-scale CPU inference is proven only through the downstream I2_SR patch; TL2 quality-preserving row-scale Qwen1.5B requires row/group-scale runtime and kernel support.",
    )


def audit_moe(root: Path, rows: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    moe = read_json(root / "benchmark_results/moe_support_audit_2026-05-05.json")
    local_kimi = moe.get("local_kimi_artifacts", [])
    source_kimi = moe.get("kimi_source_matches", [])
    present_checks = [check for check in moe.get("checks", []) if check.get("status") == "present"]
    productization_gates = moe.get("productization_gates", [])
    failed_gates = [gate for gate in productization_gates if isinstance(gate, dict) and not gate.get("passed")]
    metrics["moe"] = {
        "present_generic_checks": len(present_checks),
        "productization_gate_count": len(productization_gates),
        "failed_productization_gate_count": len(failed_gates),
        "local_kimi_artifact_count": len(local_kimi),
        "kimi_source_match_count": len(source_kimi),
    }
    add_row(
        rows,
        "Evaluate MoE/Kimi feasibility including converter mapping, router/expert execution, quality, throughput, and locality",
        "not_complete",
        (
            f"generic MoE checks present={len(present_checks)}; productization gates failed={len(failed_gates)}/{len(productization_gates)}; "
            f"Kimi artifacts={len(local_kimi)}; Kimi source matches={len(source_kimi)}"
        ),
        "No Kimi/Qwen2MoE BitNet converter mapping, TL2 3D expert packing support, router distillation, MoE quality run, throughput run, or expert-locality benchmark exists. Direct I2_S/I2_SR 3D packing is only a synthetic contract until a real MoE GGUF/runtime artifact exists.",
    )


def audit_publishability(root: Path, rows: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    manifest = read_json(root / "benchmarks/results/evidence_manifest_2026-05-13.json")
    coverage = read_json(root / "benchmark_results/benchmark_coverage_gate_2026-05-13.json")
    prune = read_json(root / "benchmarks/results/artifact_prune_plan_2026-05-13.json")
    metrics["meta"] = {
        "manifest_artifacts": manifest.get("artifact_count"),
        "manifest_missing": manifest.get("missing_count"),
        "coverage_passed": coverage.get("passed"),
        "coverage_checks": coverage.get("check_count"),
        "prune_sizes": prune.get("sizes_bytes", {}),
    }
    complete = manifest.get("missing_count") == 0 and coverage.get("passed") is True
    add_row(
        rows,
        "Produce side-by-side comparison, evidence manifest, prune plan, and honest novelty/product verdict",
        "complete" if complete else "partial",
        (
            f"manifest artifacts={manifest.get('artifact_count')}, missing={manifest.get('missing_count')}; "
            f"coverage={coverage.get('passed')} checks={coverage.get('check_count')}; "
            "publishable ledger scopes negative result plus recovery path"
        ),
        "Manifest or benchmark coverage gate is not clean.",
    )


def md_table(headers: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def render_markdown(result: dict[str, Any]) -> str:
    checklist_rows = [
        [row["requirement"], row["status"], row["evidence"], row["remaining_gap"]]
        for row in result["checklist"]
    ]
    incomplete = [row for row in result["checklist"] if row["status"] != "complete"]
    lines = [
        f"# Objective Completion Audit, {DATE}",
        "This audit maps the active user objective to concrete artifacts in this fork. It is stricter than the benchmark coverage gate and is not a success declaration.",
        "## Success Criteria",
        "1. Repaired ternary checkpoint export has the expected ternary key counts.",
        "2. Fixed prompt sanity suites and quality benchmarks exist for the repaired dense-Qwen checkpoints.",
        "3. WikiText, FineWeb-heldout, and ten-task EleutherAI lm-eval comparisons cover FP, naive PTQ, QAT, dense-head, and row-scale variants.",
        "4. GGUF/TQ2_0/I2_S/I2_SR CPU paths measure quality, file size, throughput, and RSS on the Xeon.",
        "5. Product claims are limited to what the active/default runtime supports, with TL2 and MoE/Kimi gaps called out.",
        "## Verdict",
        f"Objective achieved: `{result['objective_achieved']}`.",
        f"Completion status: `{result['completion_status']}`.",
        f"Complete rows: `{result['complete_count']}` / `{result['check_count']}`.",
        "The dense-Qwen negative result and row-scale recovery path are well supported. The full objective is still incomplete because product-default row-scale qtype support, quality-preserving TL2 for the strong row-scale checkpoint, and MoE/Kimi evidence remain missing.",
        "## Prompt-To-Artifact Checklist",
        md_table(["requirement", "status", "evidence", "remaining gap"], checklist_rows),
        "## Remaining Blockers",
    ]
    lines.extend(f"- {row['requirement']}: {row['remaining_gap']}" for row in incomplete)
    lines.extend(
        [
            "## Practical Next Step",
            "Promote the `I2_SR` candidate patch into the active runtime contract or keep it explicitly documented as a downstream patch. Do not expand product claims to TL2 or MoE/Kimi until those paths have quality-valid CPU benchmark artifacts.",
        ]
    )
    return "\n\n".join(lines) + "\n"


def build_audit(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    audit_exports(root, rows, metrics)
    audit_prompt_suites(root, rows, metrics)
    audit_ppl(root, rows, metrics)
    audit_lm_eval(root, rows, metrics)
    audit_baselines(root, rows, metrics)
    audit_cpu(root, rows, metrics)
    audit_productization(root, rows, metrics)
    audit_moe(root, rows, metrics)
    audit_publishability(root, rows, metrics)
    completion_status = max((row["status"] for row in rows), key=status_rank)
    complete_count = sum(1 for row in rows if row["status"] == "complete")
    return {
        "schema": "bitnet-objective-completion-audit-v1",
        "date": DATE,
        "objective_achieved": all(row["status"] == "complete" for row in rows),
        "completion_status": completion_status,
        "check_count": len(rows),
        "complete_count": complete_count,
        "partial_or_missing": [row["requirement"] for row in rows if row["status"] != "complete"],
        "checklist": rows,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-json", type=Path, default=Path("benchmark_results/objective_completion_audit_2026-05-13.json"))
    parser.add_argument("--output-md", type=Path, default=Path("benchmarks/results/objective_completion_audit_2026-05-13.md"))
    args = parser.parse_args()

    root = args.repo_root.resolve()
    result = build_audit(root)
    output_json = args.output_json if args.output_json.is_absolute() else root / args.output_json
    output_md = args.output_md if args.output_md.is_absolute() else root / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(result), encoding="utf-8")
    print(render_markdown(result))


if __name__ == "__main__":
    main()
