#!/usr/bin/env python3
"""Classify which product and publication claims are supported by artifacts."""

from __future__ import annotations

import os
import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_json_path(root: Path, pattern: str, fallback: str) -> Path:
    paths = sorted(root.glob(pattern))
    return paths[-1] if paths else root / fallback


def display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


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
    bitdistill_i2sr_path = latest_json_path(
        root,
        "benchmark_results/bitdistill_i2sr_export_gate_*.json",
        "benchmark_results/bitdistill_i2sr_export_gate_2026-05-14.json",
    )
    bitdistill_i2sr = read_json(bitdistill_i2sr_path) if bitdistill_i2sr_path.exists() else {}
    bitdistill_i2sr_rows = bitdistill_i2sr.get("rows", []) if isinstance(bitdistill_i2sr.get("rows"), list) else []
    bitdistill_i2sr_complete = sum(1 for row in bitdistill_i2sr_rows if isinstance(row, dict) and row.get("complete"))
    bitdistill_i2sr_expected = len(bitdistill_i2sr_rows)
    bitdistill_i2sr_blockers = sorted(
        {
            blocker
            for row in bitdistill_i2sr_rows
            if isinstance(row, dict)
            for blocker in row.get("blockers", [])
            if isinstance(blocker, str)
        }
    )
    bitdistill_reproduction_path = latest_json_path(
        root,
        "benchmark_results/bitdistill_reproduction_gate_*.json",
        "benchmark_results/bitdistill_reproduction_gate_2026-05-14.json",
    )
    bitdistill_reproduction = read_json(bitdistill_reproduction_path) if bitdistill_reproduction_path.exists() else {}
    bitdistill_rows = bitdistill_reproduction.get("rows", []) if isinstance(bitdistill_reproduction.get("rows"), list) else []
    bitdistill_gamma100_rows = [
        row
        for row in bitdistill_rows
        if isinstance(row, dict) and row.get("family") == "longwarmup_gamma100"
    ]
    bitdistill_gamma100_full_rows = sum(1 for row in bitdistill_gamma100_rows if row.get("full_eval_examples") is True)
    bitdistill_gamma100_passed_rows = sum(1 for row in bitdistill_gamma100_rows if row.get("passes_fp_gap") is True)
    bitdistill_paper_rows = [
        row
        for row in bitdistill_rows
        if isinstance(row, dict) and row.get("family") == "paper_hparam_candidate"
    ]
    bitdistill_paper_full_rows = sum(1 for row in bitdistill_paper_rows if row.get("full_eval_examples") is True)
    bitdistill_paper_passed_rows = sum(1 for row in bitdistill_paper_rows if row.get("passes_fp_gap") is True)
    bitdistill_expected_examples = bitdistill_reproduction.get("expected_eval_examples", {})
    bitdistill_paired_path = latest_json_path(
        root,
        "benchmark_results/bitdistill_paired_predictions_*.json",
        "benchmark_results/bitdistill_paired_predictions_2026-05-14.json",
    )
    bitdistill_paired = read_json(bitdistill_paired_path) if bitdistill_paired_path.exists() else {}
    bitdistill_paired_complete = bitdistill_paired.get("complete_count", bitdistill_paired.get("complete"))
    bitdistill_paired_total = bitdistill_paired.get("check_count", bitdistill_paired.get("total"))
    bitdistill_cpu_path = latest_json_path(
        root,
        "benchmark_results/bitdistill_glue_cpu_gate_*.json",
        "benchmark_results/bitdistill_glue_cpu_gate_2026-05-14.json",
    )
    bitdistill_cpu = read_json(bitdistill_cpu_path) if bitdistill_cpu_path.exists() else {}
    bitdistill_cpu_critical = (
        bitdistill_cpu.get("critical", []) if isinstance(bitdistill_cpu.get("critical"), list) else []
    )
    bitdistill_cpu_complete = sum(1 for row in bitdistill_cpu_critical if isinstance(row, dict) and row.get("complete"))
    bitdistill_cpu_full_quality = sum(
        1 for row in bitdistill_cpu_critical if isinstance(row, dict) and row.get("full_quality_available")
    )

    active_gate = read_json(root / "benchmark_results/row_scale_qtype_productization_gate_2026-05-13.json")
    patch_gate = read_json(root / "benchmark_results/row_scale_qtype_productization_gate_i2sr_active_patch_2026-05-13.json")
    packed_support = read_json(root / "benchmark_results/direct_packed_gguf_support_2026-05-13.json")
    promotion_audit = read_json(root / "benchmark_results/i2sr_submodule_promotion_audit_2026-05-13.json")

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

    moe = read_json(latest_json_path(root, "benchmark_results/moe_support_audit_*.json", "benchmark_results/moe_support_audit_2026-05-05.json"))
    kimi_artifacts = moe.get("local_kimi_artifacts", [])
    kimi_source_matches = moe.get("kimi_source_matches", [])
    moe_gates = moe.get("productization_gates", [])
    moe_failed_gates = [gate.get("name") for gate in moe_gates if isinstance(gate, dict) and not gate.get("passed")]
    tiny_qwen2moe = moe.get("tiny_qwen2moe_fixture", {}) if isinstance(moe.get("tiny_qwen2moe_fixture"), dict) else {}
    tiny_qwen2moe_smoke = tiny_qwen2moe.get("smoke", {}) if isinstance(tiny_qwen2moe.get("smoke"), dict) else {}
    tiny_qwen2moe_rss = tiny_qwen2moe.get("rss", {}) if isinstance(tiny_qwen2moe.get("rss"), dict) else {}
    tiny_qwen2moe_scaling = moe.get("tiny_qwen2moe_expert_scaling", {}) if isinstance(moe.get("tiny_qwen2moe_expert_scaling"), dict) else {}
    tiny_qwen2moe_scaling_rows = tiny_qwen2moe_scaling.get("rows", []) if isinstance(tiny_qwen2moe_scaling.get("rows"), list) else []
    moe_tl2 = read_json(
        latest_json_path(
            root,
            "benchmark_results/moe_tl2_runtime_contract_*.json",
            "benchmark_results/moe_tl2_runtime_contract_2026-05-13.json",
        )
    )
    moe_tl2_byte_probe = moe_tl2.get("byte_size_probe", {})

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
        "CPU row-scale ternary inference for dense Qwen through stable I2_SR",
        "supported",
        f"I2_SR PPL={i2sr_ppl}; file={i2sr_file_mib:.1f} MiB; prompt={i2sr_prefill:.2f} tok/s; decode={i2sr_decode:.2f} tok/s; active gate={active_gate.get('passed')}; patch gate={patch_gate.get('passed')}",
        "Dense Qwen2.5-1.5B I2_SR evidence in this fork on Intel Xeon Silver 4116.",
    )
    bitdistill_paper_passed = bool(bitdistill_reproduction.get("paper_style_tensor_passed"))
    add_claim(
        claims,
        "BitDistill paper-level GLUE reproduction on Qwen2.5-0.5B",
        "supported" if bitdistill_paper_passed else "unsupported",
        (
            f"paper tensor complete={bitdistill_reproduction.get('paper_style_tensor_complete')}; "
            f"passed={bitdistill_reproduction.get('paper_style_tensor_passed')}; "
            f"gamma100 full rows={bitdistill_gamma100_full_rows}/{len(bitdistill_gamma100_rows)}; "
            f"gamma100 gap-pass rows={bitdistill_gamma100_passed_rows}/{len(bitdistill_gamma100_rows)}; "
            f"strict paper full rows={bitdistill_paper_full_rows}/{len(bitdistill_paper_rows)}; "
            f"strict paper gap-pass rows={bitdistill_paper_passed_rows}/{len(bitdistill_paper_rows)}; "
            f"expected eval={bitdistill_expected_examples}; "
            f"paired status={bitdistill_paired.get('status')}; "
            f"paired complete={bitdistill_paired_complete}/{bitdistill_paired_total}; "
            f"cpu gate={bitdistill_cpu.get('passed')} ({bitdistill_cpu_complete}/{len(bitdistill_cpu_critical)} rows, "
            f"full-quality={bitdistill_cpu_full_quality}/{len(bitdistill_cpu_critical)})"
        ),
        "Claim only after MNLI/QNLI/SST2 full-validation BitDistill rows are within the configured FP16 gap and paired traces/CPU gates are complete.",
        "Gamma=100 and strict paper-gamma tensor rows are complete but below the FP16-gap gate; row paper-gamma, LR/head-init search, paired-trace coverage, and CPU full-quality rows are still missing or incomplete.",
    )
    bitdistill_i2sr_passed = bool(bitdistill_i2sr.get("passed"))
    add_claim(
        claims,
        "BitDistill task-specific packed ternary runtime support",
        "supported" if bitdistill_i2sr_passed else "unsupported",
        (
            f"causal packed-ternary gate passed={bitdistill_i2sr.get('passed')}; "
            f"complete rows={bitdistill_i2sr_complete}/{bitdistill_i2sr_expected}; "
            f"gate={display_path(bitdistill_i2sr_path, root)}"
        ),
        "Only causal prompt-scoring BitDistill checkpoints can use this packed path; tensor baselines use I2_S, row-scale novelty runs use I2_SR, and sequence-classification heads remain PyTorch-only unless runtime support is added.",
        "; ".join(bitdistill_i2sr_blockers) if bitdistill_i2sr_blockers else "Causal BitDistill packed export and CPU benchmark artifacts are missing or incomplete.",
    )
    default_runtime_supported = bool(active_gate.get("passed")) and bool(promotion_audit.get("promotion_ready"))
    add_claim(
        claims,
        "Default committed runtime supports stable row-scale I2_SR",
        "supported" if default_runtime_supported else "unsupported",
        f"active productization gate passed={active_gate.get('passed')}; promotion_ready={promotion_audit.get('promotion_ready')}; failed_gates={sum(1 for gate in active_gate.get('gates', []) if not gate.get('passed'))}",
        "Stable I2_SR is available in the active source state when this gate is supported.",
        "The active source state is not yet promoted to a reachable submodule branch and clean superproject pointer.",
    )
    direct_row_supported = bool(packed_support.get("verdict", {}).get("product_safe_row_scale_packed_supported"))
    add_claim(
        claims,
        "Direct packed row-scale GGUF export is product-safe by default",
        "supported" if direct_row_supported else "unsupported",
        f"direct packed verdict={packed_support.get('verdict', {}).get('product_safe_row_scale_packed_supported')}; candidate_i2sr_quality_valid={packed_support.get('verdict', {}).get('candidate_i2sr_quality_valid')}",
        "Direct writer can be used for the audited dense-Qwen I2_SR path when this gate is supported.",
        "Direct writer still lacks a promoted stable qtype/runtime or byte-layout/quality evidence.",
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
        (
            f"Kimi artifacts={len(kimi_artifacts)}; Kimi source matches={len(kimi_source_matches)}; "
            f"tiny Qwen2MoE FP16 fixture passed={tiny_qwen2moe.get('passed')}; "
            f"fixture arch={tiny_qwen2moe_smoke.get('architecture')}; fixture RSS MiB={tiny_qwen2moe_rss.get('max_rss_mib')}; "
            f"synthetic expert scaling passed={tiny_qwen2moe_scaling.get('passed')}; scaling rows={len(tiny_qwen2moe_scaling_rows)}; "
            f"failed MoE gates={len(moe_failed_gates)}/{len(moe_gates)}; "
            f"TL2 MoE runtime ready={moe_tl2.get('tl2_moe_runtime_ready')}; "
            f"TL2 expert byte underreport={moe_tl2_byte_probe.get('underreport_bytes')}"
        ),
        "Treat as separate research milestone.",
        "No Kimi-specific mapping, trained Qwen2MoE/Kimi quality artifact, ternary MoE runtime artifact, TL2 3D expert runtime support, router distillation, quality benchmark, or trained expert-locality benchmark exists; the tiny Qwen2MoE fixtures only prove synthetic FP16 converter/runtime plumbing.",
    )

    supported = [claim for claim in claims if claim["status"] in {"supported", "supported_with_patch"}]
    unsupported = [claim for claim in claims if claim["status"] == "unsupported"]
    return {
        "schema": "bitnet-product-scope-gate-v1",
        "date": DATE,
        "scope_status": "research_mvp_only",
        "publishable_angle": "negative arbitrary-retrofit result plus dense-Qwen row-scale recovery path, with BitDistill reproduction and I2_SR row-scale extensions gated separately",
        "supported_claim_count": len(supported),
        "unsupported_claim_count": len(unsupported),
        "claims": claims,
        "recommendation": {
            "product": "CPU-first dense-Qwen retrofit evaluator with stable I2_SR runtime support; keep BitDistill quality claims behind the full GLUE reproduction, paired-trace, and CPU full-quality gates.",
            "paper": "Scope as a negative PTQ result plus measured QAT/row-scale/runtime recovery path; add BitDistill reproduction claims only if the full-validation long-warmup gates pass. Do not claim arbitrary or MoE support.",
            "next_engineering_gate": "Finish the row paper-gamma, LR/head-init search, paired-trace, and CPU-quality dependency chain, then validate row-scale I2_SR export/CPU evidence and keep MoE/Kimi as a separate milestone.",
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
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/product_scope_gate_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/product_scope_gate_{DATE}.md"))
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
