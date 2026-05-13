#!/usr/bin/env python3
"""Classify which product and publication claims are supported by artifacts."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


DATE = "2026-05-13"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def find_row(summary: dict[str, Any], name: str) -> dict[str, Any] | None:
    for row in summary.get("rows", []):
        if isinstance(row, dict) and row.get("name") == name:
            return row
    return None


def parse_macro_delta(path: Path) -> str | None:
    if not path.exists():
        return None
    match = re.search(r"\| macro mean delta \| ([^|]+) \|", path.read_text(encoding="utf-8"))
    return match.group(1).strip() if match else None


def selected_mean(path: Path) -> float | None:
    data = read_json(path)
    metrics = {
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
    values: list[float] = []
    for task, metric in metrics.items():
        task_result = data.get("results", {}).get(task, {})
        if not isinstance(task_result, dict):
            continue
        value = task_result.get(metric)
        if value is None:
            value = task_result.get(f"{metric},none")
        if finite_number(value):
            values.append(float(value))
    return sum(values) / len(values) if len(values) == len(metrics) else None


def add_claim(claims: list[dict[str, Any]], name: str, status: str, evidence: str, scope: str, blocker: str = "") -> None:
    claims.append(
        {
            "claim": name,
            "status": status,
            "evidence": evidence,
            "scope": scope,
            "blocker": blocker if status not in {"supported", "supported_with_patch"} else "",
        }
    )


def build_gate(root: Path) -> dict[str, Any]:
    fp_mean = selected_mean(root / "benchmark_results/lm-eval-qwen15b-full10/qwen15b_fp.json")
    ptq_mean = selected_mean(root / "benchmark_results/lm-eval-qwen15b-full10/qwen15b_naive_ptq.json")
    row_mean = selected_mean(root / "benchmark_results/lm-eval-qwen15b-klonly-row-notiehead-full10/qwen15b_qat_ternary.json")
    row_minus_fp = parse_macro_delta(root / "benchmarks/results/paired_row_densehead_minus_fp_2026-05-13.md")
    row_minus_ptq = parse_macro_delta(root / "benchmarks/results/paired_row_densehead_minus_ptq_2026-05-13.md")

    ppl_fp = read_json(root / "benchmark_results/quality-9735/qwen15b_fp_wikitext.json").get("perplexity")
    ppl_ptq = read_json(root / "benchmark_results/quality-ptq-qwen15b/qwen15b_naive_ptq_wikitext.json").get("perplexity")
    ppl_row = read_json(root / "benchmark_results/quality-qwen15b-klonly-row-notiehead-5000/qwen15b_ternary_wikitext.json").get("perplexity")

    i2sr_summary = read_json(root / "benchmark_results/i2sr-row-scale-qwen15b-x86act-suite-2026-05-13/summary.json")
    i2sr = find_row(i2sr_summary, "qwen15b_klonly_row_notie_static_ternary_i2_sr_x86act") or {}
    i2sr_ppl = i2sr.get("perplexity", {}).get("ppl")
    i2sr_prefill = i2sr.get("bench", {}).get("prefill", {}).get("tok_s")
    i2sr_decode = i2sr.get("bench", {}).get("decode", {}).get("tok_s")
    i2sr_file_mib = i2sr.get("file_mib")

    active_gate = read_json(root / "benchmark_results/row_scale_qtype_productization_gate_2026-05-13.json")
    patch_gate = read_json(root / "benchmark_results/row_scale_qtype_productization_gate_i2sr_active_patch_2026-05-13.json")
    packed_support = read_json(root / "benchmark_results/direct_packed_gguf_support_2026-05-13.json")

    tl2_scale = read_json(root / "benchmark_results/tl2_scale_semantics_2026-05-05.json")
    tl2_row = next((item for item in tl2_scale.get("results", []) if item.get("label") == "qwen15b_row_scale"), {})
    tl2_design = read_json(root / "benchmark_results/tl2_row_scale_design_2026-05-13.json")
    tl2_design_row = next((item for item in tl2_design.get("results", []) if item.get("label") == "qwen15b_row_scale"), {})
    tl2_design_strategies = {
        item.get("name"): item
        for item in tl2_design_row.get("strategies", [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    tl2_row_fp16 = tl2_design_strategies.get("row_exact_fp16", {})
    tl2_group2 = tl2_design_strategies.get("group2_l2_optimal_fp16", {})
    tl2_group32 = tl2_design_strategies.get("group32_l2_optimal_fp16", {})
    tl2_avx = read_json(root / "benchmark_results/gguf-qwen05b-tl2-avx512-2026-05-05/summary.json")
    tl2_artifact = find_row(tl2_avx, "qwen05b_qat_tl2") or {}
    tl2_ppl = tl2_artifact.get("perplexity", {}).get("ppl")

    moe = read_json(root / "benchmark_results/moe_support_audit_2026-05-05.json")
    kimi_artifacts = moe.get("local_kimi_artifacts", [])
    kimi_source_matches = moe.get("kimi_source_matches", [])

    claims: list[dict[str, Any]] = []
    add_claim(
        claims,
        "One-click lossless arbitrary FP/BF16-to-ternary retrofit",
        "unsupported",
        f"Qwen1.5B FP mean={fp_mean:.6f}; naive PTQ mean={ptq_mean:.6f}; FP WikiText PPL={ppl_fp}; PTQ WikiText PPL={ppl_ptq}",
        "Do not market as arbitrary lossless conversion.",
        "Blind PTQ destroys quality in both math and model-level artifacts.",
    )
    add_claim(
        claims,
        "Dense Qwen negative result plus QAT/distillation recovery path",
        "supported",
        f"row-scale mean={row_mean:.6f}; row-PTQ paired delta={row_minus_ptq}; row-FP paired delta={row_minus_fp}; row WikiText PPL={ppl_row}",
        "Qwen2.5 dense 1.5B evidence in this fork.",
    )
    add_claim(
        claims,
        "CPU row-scale ternary inference for dense Qwen through fixed I2_SR candidate",
        "supported_with_patch",
        f"I2_SR PPL={i2sr_ppl}; file={i2sr_file_mib:.1f} MiB; prompt={i2sr_prefill:.2f} tok/s; decode={i2sr_decode:.2f} tok/s; patch gate={patch_gate.get('passed')}",
        "Downstream `patches/llama-i2sr-row-scale-qtype.patch` applied to the audited source state.",
    )
    add_claim(
        claims,
        "Default committed runtime supports stable row-scale I2_SR",
        "unsupported",
        f"active productization gate passed={active_gate.get('passed')}; failed_gates={sum(1 for gate in active_gate.get('gates', []) if not gate.get('passed'))}",
        "Do not claim default/submodule support yet.",
        "The committed submodule lacks the separate qtype/file type/runtime routing without applying the downstream patch.",
    )
    add_claim(
        claims,
        "Direct packed row-scale GGUF export is product-safe by default",
        "unsupported",
        f"direct packed verdict={packed_support.get('verdict', {}).get('product_safe_row_scale_packed_supported')}; candidate_i2sr_quality_valid={packed_support.get('verdict', {}).get('candidate_i2sr_quality_valid')}",
        "Direct writer can be used for controlled experiments; product claims need stable qtype support.",
        "The safe row-scale path is still tied to the downstream I2_SR candidate patch.",
    )
    add_claim(
        claims,
        "TL2 product support for the strong row-scale Qwen checkpoint",
        "unsupported",
        (
            f"Qwen0.5B TL2 PPL={tl2_ppl}; Qwen1.5B row-scale one-scale error={tl2_row.get('total_relative_fro_error_if_one_scale')}; "
            f"group2 fp16 design error={tl2_group2.get('expected_relative_output_rms_error')}; "
            f"group32 fp16 design error={tl2_group32.get('expected_relative_output_rms_error')}; "
            f"exact row-fp16 design error={tl2_row_fp16.get('expected_relative_output_rms_error')} "
            f"at {tl2_row_fp16.get('scale_mib_fp16')} MiB"
        ),
        "Exclude TL2 from MVP claims.",
        "Current TL2 scale semantics are incompatible with row-scale checkpoint quality; fixing it requires row/group-scale metadata and generated kernels that index those scales.",
    )
    add_claim(
        claims,
        "MoE/Kimi retrofit and CPU runtime support",
        "unsupported",
        f"Kimi artifacts={len(kimi_artifacts)}; Kimi source matches={len(kimi_source_matches)}",
        "Treat as separate research milestone.",
        "No Kimi-specific converter/runtime mapping, router distillation, quality benchmark, or expert-locality benchmark exists.",
    )

    supported = [claim for claim in claims if claim["status"] in {"supported", "supported_with_patch"}]
    unsupported = [claim for claim in claims if claim["status"] == "unsupported"]
    return {
        "schema": "bitnet-product-scope-gate-v1",
        "date": DATE,
        "scope_status": "research_mvp_only",
        "publishable_angle": "negative arbitrary-retrofit result plus dense-Qwen row-scale recovery path",
        "supported_claim_count": len(supported),
        "unsupported_claim_count": len(unsupported),
        "claims": claims,
        "recommendation": {
            "product": "CPU-first dense-Qwen retrofit evaluator with explicit downstream I2_SR runtime patch requirement.",
            "paper": "Scope as a negative PTQ result plus measured distillation/row-scale/runtime recovery path; do not claim arbitrary or MoE support.",
            "next_engineering_gate": "Make I2_SR active/default or keep it clearly as a patch-distribution runtime.",
        },
    }


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_markdown(result: dict[str, Any]) -> str:
    rows = [
        [claim["claim"], claim["status"], claim["evidence"], claim["scope"], claim["blocker"]]
        for claim in result["claims"]
    ]
    return "\n\n".join(
        [
            f"# Product Scope Gate, {DATE}",
            "This gate separates what can be published or productized from what remains unsupported.",
            f"Scope status: `{result['scope_status']}`.",
            f"Publishable angle: {result['publishable_angle']}.",
            md_table(["claim", "status", "evidence", "scope", "blocker"], rows),
            "## Recommendation",
            f"Product: {result['recommendation']['product']}",
            f"Paper: {result['recommendation']['paper']}",
            f"Next engineering gate: {result['recommendation']['next_engineering_gate']}",
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-json", type=Path, default=Path("benchmark_results/product_scope_gate_2026-05-13.json"))
    parser.add_argument("--output-md", type=Path, default=Path("benchmarks/results/product_scope_gate_2026-05-13.md"))
    args = parser.parse_args()

    root = args.repo_root.resolve()
    result = build_gate(root)
    output_json = args.output_json if args.output_json.is_absolute() else root / args.output_json
    output_md = args.output_md if args.output_md.is_absolute() else root / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(result), encoding="utf-8")
    print(render_markdown(result))


if __name__ == "__main__":
    main()
