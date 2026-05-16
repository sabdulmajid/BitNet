#!/usr/bin/env python3
"""Gate the redirected research claims against concrete benchmark artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()

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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def metric_value(task_results: dict[str, Any], metric: str) -> float | None:
    for key in (metric, f"{metric},none"):
        value = finite(task_results.get(key))
        if value is not None:
            return value
    return None


def selected_mean(path: Path) -> tuple[float | None, int]:
    data = read_json(path)
    results = data.get("results", {}) if isinstance(data.get("results"), dict) else {}
    values: list[float] = []
    for task, metric in SELECTED_LM_EVAL_METRICS.items():
        task_results = results.get(task, {})
        if isinstance(task_results, dict):
            value = metric_value(task_results, metric)
            if value is not None:
                values.append(value)
    return (sum(values) / len(values), len(values)) if values else (None, 0)


def load_ppl(path: Path) -> float | None:
    data = read_json(path)
    return finite(data.get("perplexity")) or finite(data.get("ppl"))


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value != 0 and (abs(value) >= 1000 or abs(value) < 0.0001):
            return f"{value:.3e}"
        return f"{value:.6f}"
    return str(value).replace("|", "\\|")


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(cell) for cell in row) + " |")
    return "\n".join(lines)


def add_claim(
    claims: list[dict[str, Any]],
    name: str,
    status: str,
    supported: bool,
    evidence: str,
    public_label: str,
    next_gate: str = "",
) -> None:
    claims.append(
        {
            "name": name,
            "status": status,
            "supported": bool(supported),
            "public_label": public_label,
            "evidence": evidence,
            "next_gate": next_gate,
        }
    )


def build_gate(root: Path) -> dict[str, Any]:
    fp_mean, fp_tasks = selected_mean(root / "benchmark_results/lm-eval-qwen15b-full10/qwen15b_fp.json")
    ptq_mean, ptq_tasks = selected_mean(root / "benchmark_results/lm-eval-qwen15b-full10/qwen15b_naive_ptq.json")
    row_mean, row_tasks = selected_mean(
        root / "benchmark_results/lm-eval-qwen15b-klonly-row-notiehead-full10/qwen15b_qat_ternary.json"
    )
    fp_ppl = load_ppl(root / "benchmark_results/quality-9735/qwen15b_fp_wikitext.json")
    ptq_ppl = load_ppl(root / "benchmark_results/quality-ptq-qwen15b/qwen15b_naive_ptq_wikitext.json")
    row_ppl = load_ppl(root / "benchmark_results/quality-qwen15b-klonly-row-notiehead-5000/qwen15b_ternary_wikitext.json")

    controlled = read_json(root / f"benchmark_results/bitdistill_controlled_curve_{DATE}.json")
    control_rows = controlled.get("rows", []) if isinstance(controlled.get("rows"), list) else []
    completed_control_rows = [
        row
        for row in control_rows
        if isinstance(row, dict)
        and row.get("metrics_exists") is True
        and row.get("predictions_exists") is True
        and finite(row.get("metric_accuracy")) is not None
    ]
    best_control = max(
        completed_control_rows,
        key=lambda row: finite(row.get("metric_accuracy")) or float("-inf"),
        default={},
    )
    best_control_delta = None
    paired = best_control.get("paired") if isinstance(best_control.get("paired"), dict) else {}
    best_control_delta = finite(paired.get("delta_vs_reference"))

    cpu = read_json(root / f"benchmark_results/cpu_tradeoff_frontier_{DATE}.json")
    q4_vs_i2sr = cpu.get("q4_vs_i2sr", {}) if isinstance(cpu.get("q4_vs_i2sr"), dict) else {}

    tl2 = read_json(root / f"benchmark_results/tl2_negative_result_{DATE}.json")
    seqcls = read_json(root / f"benchmark_results/seqcls_runtime_gap_{DATE}.json")
    sidecar = (
        seqcls.get("seqcls_sidecar_cpu_benchmark", {})
        if isinstance(seqcls.get("seqcls_sidecar_cpu_benchmark"), dict)
        else {}
    )
    native_cpu = (
        seqcls.get("seqcls_native_cpu_benchmark", {})
        if isinstance(seqcls.get("seqcls_native_cpu_benchmark"), dict)
        else {}
    )
    moe = read_json(root / f"benchmark_results/moe_support_audit_{DATE}.json")
    moe_gates = moe.get("productization_gates", []) if isinstance(moe.get("productization_gates"), list) else []
    moe_failed = [gate for gate in moe_gates if isinstance(gate, dict) and not gate.get("passed")]
    kimi = read_json(root / f"benchmark_results/kimi_config_feasibility_{DATE}.json")

    ptq_delta = ptq_mean - fp_mean if ptq_mean is not None and fp_mean is not None else None
    row_delta = row_mean - fp_mean if row_mean is not None and fp_mean is not None else None
    ptq_ppl_ratio = ptq_ppl / fp_ppl if ptq_ppl is not None and fp_ppl not in (None, 0) else None
    row_recovery = row_mean - ptq_mean if row_mean is not None and ptq_mean is not None else None

    claims: list[dict[str, Any]] = []
    add_claim(
        claims,
        "Blind arbitrary FP/BF16-to-ternary PTQ retrofit",
        "rejected_for_tested_dense_qwen",
        ptq_delta is not None and ptq_delta < -0.10 and ptq_ppl_ratio is not None and ptq_ppl_ratio > 1000,
        (
            f"FP mean={fmt(fp_mean)} over {fp_tasks} tasks; PTQ mean={fmt(ptq_mean)} over {ptq_tasks} tasks; "
            f"delta={fmt(ptq_delta)}; WikiText PPL ratio={fmt(ptq_ppl_ratio)}."
        ),
        "Do not claim a universal converter.",
    )
    add_claim(
        claims,
        "QAT/distillation recovery over blind PTQ",
        "partially_supported",
        row_recovery is not None and row_recovery > 0.10 and row_delta is not None and row_delta < -0.05,
        (
            f"row-scale mean={fmt(row_mean)} over {row_tasks} tasks; recovery over PTQ={fmt(row_recovery)}; "
            f"delta vs FP={fmt(row_delta)}; row WikiText PPL={fmt(row_ppl)}."
        ),
        "Claim partial recovery only, not FP-quality recovery.",
    )
    add_claim(
        claims,
        "Paper-level BitDistill reproduction",
        "not_proven",
        controlled.get("all_passed_fp_recovery_gate") is False,
        (
            f"controlled rows complete={controlled.get('complete')}/{controlled.get('expected')}; "
            f"passed FP recovery={controlled.get('passed_fp_recovery_gate')}; "
            f"best controlled accuracy={fmt(best_control.get('metric_accuracy'))}; "
            f"best paired delta vs FP={fmt(best_control_delta)}."
        ),
        "Keep labeled as paper-inspired until full controlled rows close the FP gap.",
    )
    add_claim(
        claims,
        "Row-scale I2_SR runtime semantics",
        "supported_as_retrofit_variant",
        finite(q4_vs_i2sr.get("prefill_speedup")) is not None
        and finite(q4_vs_i2sr.get("decode_speedup")) is not None
        and q4_vs_i2sr.get("prefill_speedup") > 1.0
        and q4_vs_i2sr.get("decode_speedup") > 1.0
        and finite(q4_vs_i2sr.get("ppl_ratio")) is not None
        and q4_vs_i2sr.get("ppl_ratio") > 1.0,
        (
            f"I2_SR/Q4 prefill={fmt(q4_vs_i2sr.get('prefill_speedup'))}x; "
            f"decode={fmt(q4_vs_i2sr.get('decode_speedup'))}x; "
            f"file={fmt(q4_vs_i2sr.get('file_ratio'))}x; "
            f"PPL={fmt(q4_vs_i2sr.get('ppl_ratio'))}x."
        ),
        "Claim runtime/scale-contract viability, not a Q4 quality/storage win.",
    )
    add_claim(
        claims,
        "TL2 row-scale runtime readiness",
        "blocked",
        tl2.get("runtime_ready") is False and tl2.get("negative_result_supported") is True,
        (
            f"runtime_ready={fmt(tl2.get('runtime_ready'))}; "
            f"current row-scale error={fmt(tl2.get('qwen15b_row_scale_current_tl2_error'))}; "
            f"exact row-scale design error={fmt(tl2.get('qwen15b_row_scale_exact_fp16_error'))}; "
            f"finite TL2 quality={fmt(tl2.get('tl2_probe_has_finite_quality'))}."
        ),
        "Use I2_SR until TL2 has explicit row/group-scale metadata and kernels.",
    )
    add_claim(
        claims,
        "Native packed sequence-classification deployment",
        "full_validation_batching_blocked",
        seqcls.get("same_artifact_quality_cpu_ready") is False
        and native_cpu.get("status") == "pass"
        and native_cpu.get("full_validation_complete") is True
        and native_cpu.get("ready_to_productize") is False,
        (
            f"same artifact ready={fmt(seqcls.get('same_artifact_quality_cpu_ready'))}; "
            f"native examples={fmt(native_cpu.get('examples'))}; "
            f"native accuracy={fmt(native_cpu.get('accuracy'))}; "
            f"agreement={fmt(native_cpu.get('agreement_with_saved_pytorch_predictions'))}; "
            f"batching_ready={fmt(native_cpu.get('batching_parity_ready'))}."
        ),
        "Fix native batching parity before product throughput claims.",
    )
    add_claim(
        claims,
        "Kimi/MoE product support",
        "not_proven",
        kimi.get("passed") is False and len(moe_failed) > 0,
        (
            f"Kimi config supported={fmt(kimi.get('passed'))}; "
            f"local Kimi artifacts={len(moe.get('local_kimi_artifacts', []))}; "
            f"failed MoE gates={len(moe_failed)}/{len(moe_gates)}."
        ),
        "Keep MoE/Kimi in future work until trained quality, routing locality, and runtime are measured.",
    )

    required = {
        "Blind arbitrary FP/BF16-to-ternary PTQ retrofit",
        "QAT/distillation recovery over blind PTQ",
        "Paper-level BitDistill reproduction",
        "Row-scale I2_SR runtime semantics",
        "TL2 row-scale runtime readiness",
        "Native packed sequence-classification deployment",
        "Kimi/MoE product support",
    }
    supported_claims = [claim for claim in claims if claim["supported"]]
    missing = sorted(required - {claim["name"] for claim in claims if claim["supported"]})
    return {
        "schema": "research_redirect_claim_gate.v1",
        "date": DATE,
        "status": "claim_guardrail_passed" if not missing else "claim_guardrail_failed",
        "passed": not missing,
        "missing_guardrails": missing,
        "claims": claims,
        "safe_public_summary": [
            "Blind PTQ-to-ternary is rejected for the tested dense-Qwen setup.",
            "QAT/distillation is a partial recovery path, not an FP-quality result yet.",
            "Row-scale I2_SR is a runtime-semantics contribution for compatible dense causal artifacts.",
            "Paper-level BitDistill, native packed classifier deployment, TL2 row-scale, and Kimi/MoE remain gated.",
        ],
        "claim_count": len(claims),
        "supported_guardrail_count": len(supported_claims),
    }


def render_markdown(result: dict[str, Any]) -> str:
    claim_rows = [
        [
            claim["name"],
            claim["status"],
            claim["public_label"],
            claim["evidence"],
            claim["next_gate"],
        ]
        for claim in result["claims"]
    ]
    summary_rows = [["status", result["status"]], ["passed", result["passed"]], ["guardrails", f"{result['supported_guardrail_count']}/{result['claim_count']}"]]
    return "\n\n".join(
        [
            f"# Research Redirect Claim Gate, {result['date']}",
            "This audit turns the redirected research framing into a machine-checkable claim ledger.",
            md_table(["field", "value"], summary_rows),
            "## Claims",
            md_table(["claim", "status", "safe public label", "evidence", "next gate"], claim_rows),
            "## Safe Public Summary",
            "\n".join(f"- {item}" for item in result["safe_public_summary"]),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/research_redirect_claims_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/research_redirect_claims_{DATE}.md"))
    args = parser.parse_args()

    result = build_gate(args.repo_root.resolve())
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(result) + "\n", encoding="utf-8")
    print(render_markdown(result))


if __name__ == "__main__":
    main()
