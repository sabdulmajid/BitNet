#!/usr/bin/env python3
"""Consolidate the current BitDistill reproduction root-cause evidence.

This audit is deliberately narrow. It does not decide whether BitDistill works
in general; it records what the local evidence currently supports:

* blind ternary PTQ fails for tested dense Qwen checkpoints,
* basic BitLinear mechanics are not the obvious remaining blocker,
* the weak early BitNet-SFT baseline was substantially budget-sensitive,
* BitDistill recovery toward FP16 remains unresolved under controlled runs,
* row-scale I2_SR is a separate runtime/retrofit contribution.
"""

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


def nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def finite(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value != 0 and (abs(value) >= 1000 or abs(value) < 0.0001):
            return f"{value:.3e}"
        return f"{value:.6f}"
    if isinstance(value, list):
        return "[" + ", ".join(fmt(item) for item in value) + "]"
    return str(value)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(cell).replace("|", "\\|") for cell in row) + " |")
    return "\n".join(lines)


def metric_value(task_results: dict[str, Any], metric: str) -> float | None:
    for key in (metric, f"{metric},none"):
        value = finite(task_results.get(key))
        if value is not None:
            return value
    return None


def lm_eval_mean(path: Path) -> tuple[float | None, int]:
    data = read_json(path)
    results = data.get("results", {}) if isinstance(data.get("results"), dict) else {}
    values: list[float] = []
    for task, metric in SELECTED_LM_EVAL_METRICS.items():
        task_results = results.get(task, {})
        if not isinstance(task_results, dict):
            continue
        value = metric_value(task_results, metric)
        if value is not None:
            values.append(value)
    if not values:
        return None, 0
    return sum(values) / len(values), len(values)


def load_ppl(path: Path) -> float | None:
    data = read_json(path)
    return finite(data.get("ppl")) or finite(data.get("perplexity"))


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    fp_mean, fp_tasks = lm_eval_mean(args.fp_lm_eval)
    ptq_mean, ptq_tasks = lm_eval_mean(args.ptq_lm_eval)
    row_mean, row_tasks = lm_eval_mean(args.row_lm_eval)
    fp_wiki = load_ppl(args.fp_wikitext)
    ptq_wiki = load_ppl(args.ptq_wikitext)
    row_wiki = load_ppl(args.row_wikitext)

    budget = read_json(args.bitnet_budget_paired)
    budget_best = budget.get("best", {}) if isinstance(budget.get("best"), dict) else {}
    mechanics = read_json(args.bitnet_mechanics)
    stage2 = read_json(args.stage2_curve)
    controlled = read_json(args.controlled_curve)
    loss_scale = read_json(args.loss_scale)
    subln = read_json(args.subln_audit)
    runtime = read_json(args.cpu_frontier)
    tl2 = read_json(args.tl2_contract)
    moe = read_json(args.moe_support)
    kimi = read_json(args.kimi_feasibility)
    moe_gates = moe.get("productization_gates", []) if isinstance(moe.get("productization_gates"), list) else []
    moe_failed_gates = sum(1 for gate in moe_gates if isinstance(gate, dict) and not gate.get("passed"))

    stage2_best = stage2.get("best_by_family", {}) if isinstance(stage2.get("best_by_family"), dict) else {}
    tensor_best = stage2_best.get("bitdistill_tensor", {}) if isinstance(stage2_best.get("bitdistill_tensor"), dict) else {}
    row_best = stage2_best.get("retrofit_variant", {}) if isinstance(stage2_best.get("retrofit_variant"), dict) else {}
    q4_vs_i2sr = runtime.get("q4_vs_i2sr", {}) if isinstance(runtime.get("q4_vs_i2sr"), dict) else {}
    controlled_rows = controlled.get("rows", []) if isinstance(controlled.get("rows"), list) else []
    controlled_completed_rows = [
        row
        for row in controlled_rows
        if isinstance(row, dict)
        and row.get("metrics_exists") is True
        and row.get("predictions_exists") is True
        and finite(row.get("metric_accuracy")) is not None
    ]
    controlled_best = max(
        controlled_completed_rows,
        key=lambda row: finite(row.get("metric_accuracy")) or float("-inf"),
        default={},
    )

    ptq_quality_delta = (
        ptq_mean - fp_mean
        if fp_mean is not None and ptq_mean is not None
        else None
    )
    row_quality_delta = (
        row_mean - fp_mean
        if fp_mean is not None and row_mean is not None
        else None
    )
    bitnet_best_delta_vs_fp = finite(budget_best.get("delta_vs_reference"))
    tensor_bitdistill_delta_vs_fp = finite(tensor_best.get("delta_vs_fp16"))
    row_bitdistill_delta_vs_fp = finite(row_best.get("delta_vs_fp16"))

    claims = [
        {
            "claim": "Blind ternary PTQ is not a viable universal retrofit for tested Qwen.",
            "status": "supported_for_tested_setup",
            "evidence": (
                f"WikiText PPL {fmt(fp_wiki)} -> {fmt(ptq_wiki)}; "
                f"ten-task mean {fmt(fp_mean)} -> {fmt(ptq_mean)} "
                f"(delta {fmt(ptq_quality_delta)})."
            ),
            "next_gate": "None for the tested dense-Qwen setup; this is already a negative result.",
        },
        {
            "claim": "The early weak BitNet-SFT baseline was not just broken mechanics.",
            "status": "supported",
            "evidence": (
                f"mechanics passed={fmt(mechanics.get('passed'))}; "
                f"best 10k CE-only MNLI={fmt(budget_best.get('candidate_accuracy'))}, "
                f"paper anchor delta={fmt(budget_best.get('delta_vs_paper_anchor'))}, "
                f"FP paired delta={fmt(bitnet_best_delta_vs_fp)}."
            ),
            "next_gate": "Finish the second 10k LR row and keep paired traces for schedule robustness.",
        },
        {
            "claim": "BitDistill paper-level recovery has not been locally reproduced.",
            "status": "not_proven",
            "evidence": (
                f"best tensor BitDistill MNLI={fmt(tensor_best.get('accuracy'))} "
                f"(delta vs FP {fmt(tensor_bitdistill_delta_vs_fp)}); "
                f"controlled rows complete={controlled.get('complete', 0)}/{controlled.get('expected', 0)}, "
                f"best controlled MNLI={fmt(controlled_best.get('metric_accuracy'))} "
                f"(delta vs FP {fmt(nested(controlled_best, 'paired', 'delta_vs_reference'))})."
            ),
            "next_gate": "Finish the fixed-recipe 5k/20k/40k Stage-2 curve and require a full-validation paired trace within the FP recovery gate.",
        },
        {
            "claim": "Loss normalization is a live reproduction risk.",
            "status": "supported",
            "evidence": (
                "projected paper-gamma attention/CE range "
                f"{fmt(loss_scale.get('projected_paper_attention_to_ce_min'))} to "
                f"{fmt(loss_scale.get('projected_paper_attention_to_ce_max'))}; "
                f"materialized rows={loss_scale.get('materialized_rows')}."
            ),
            "next_gate": "For new jobs, compare CE/logit-KD/attention-KD magnitudes before interpreting gamma sweeps.",
        },
        {
            "claim": "Local SubLN surgery is not identity-preserving before adaptation.",
            "status": "supported",
            "evidence": (
                f"inserted={subln.get('subln_inserted')}; "
                f"logit relative RMS drift={fmt(subln.get('logit_relative_rms'))}; "
                f"cosine={fmt(subln.get('logit_cosine'))}; "
                f"top1 agreement={fmt(subln.get('last_token_top1_agreement'))}."
            ),
            "next_gate": "Treat SubLN timing/init as part of the training recipe, not as harmless module insertion.",
        },
        {
            "claim": "Row-scale I2_SR is a runtime-semantics contribution, not a Q4 quality/storage win.",
            "status": "supported",
            "evidence": (
                f"row-scale ten-task mean={fmt(row_mean)} (delta vs FP {fmt(row_quality_delta)}); "
                f"I2_SR/Q4 prefill={fmt(q4_vs_i2sr.get('prefill_speedup'))}x, "
                f"decode={fmt(q4_vs_i2sr.get('decode_speedup'))}x, "
                f"file={fmt(q4_vs_i2sr.get('file_ratio'))}x, "
                f"PPL={fmt(q4_vs_i2sr.get('ppl_ratio'))}x."
            ),
            "next_gate": "Keep claims scoped to speed and scale-contract fidelity until quality improves.",
        },
        {
            "claim": "TL2 row-scale and real Kimi/MoE product claims remain open.",
            "status": "not_proven",
            "evidence": (
                f"TL2 ready={fmt(tl2.get('tl2_row_scale_runtime_ready'))}; "
                f"TL2 row one-scale error={fmt(nested(tl2, 'math', 'current_tl2_tensor_max_error'))}; "
                f"MoE gates failed={moe_failed_gates}/{len(moe_gates)}; "
                f"Kimi config supported={fmt(kimi.get('passed'))}."
            ),
            "next_gate": "Do not foreground Kimi/MoE until trained quality, routing locality, and runtime support exist.",
        },
    ]

    return {
        "schema": "bitdistill-root-cause-audit-v1",
        "date": DATE,
        "quality": {
            "fp_wikitext_ppl": fp_wiki,
            "ptq_wikitext_ppl": ptq_wiki,
            "row_wikitext_ppl": row_wiki,
            "fp_ten_task_mean": fp_mean,
            "fp_tasks": fp_tasks,
            "ptq_ten_task_mean": ptq_mean,
            "ptq_tasks": ptq_tasks,
            "row_ten_task_mean": row_mean,
            "row_tasks": row_tasks,
            "ptq_delta_vs_fp": ptq_quality_delta,
            "row_delta_vs_fp": row_quality_delta,
        },
        "bitnet_sft": {
            "complete_rows": budget.get("complete"),
            "best": budget_best,
        },
        "bitdistill": {
            "best_tensor": tensor_best,
            "best_row_retrofit": row_best,
            "controlled_complete": controlled.get("complete"),
            "controlled_expected": controlled.get("expected"),
            "controlled_all_complete": controlled.get("all_complete"),
            "controlled_passed_fp_recovery_gate": controlled.get("passed_fp_recovery_gate"),
            "controlled_best": controlled_best,
        },
        "loss_scale": {
            "materialized_rows": loss_scale.get("materialized_rows"),
            "projected_paper_attention_to_ce_min": loss_scale.get("projected_paper_attention_to_ce_min"),
            "projected_paper_attention_to_ce_max": loss_scale.get("projected_paper_attention_to_ce_max"),
        },
        "runtime": {
            "q4_vs_i2sr": q4_vs_i2sr,
            "interpretation": runtime.get("interpretation"),
        },
        "open_runtime_claims": {
            "tl2_row_scale_runtime_ready": tl2.get("tl2_row_scale_runtime_ready"),
            "tl2_current_one_scale_error": nested(tl2, "math", "current_tl2_tensor_max_error"),
            "tl2_row_fp16_error": nested(tl2, "math", "row_fp16_error"),
            "tl2_failed_checks": sum(1 for check in tl2.get("checks", []) if isinstance(check, dict) and not check.get("passed")),
            "moe_productization_gates": len(moe_gates),
            "moe_failed_productization_gates": moe_failed_gates,
            "kimi_config_supported": kimi.get("passed"),
            "kimi_unsupported_features": kimi.get("unsupported_features", []),
        },
        "claims": claims,
        "verdict": (
            "The current evidence supports a negative PTQ result and a positive "
            "row-scale runtime-semantics result. It does not yet support a "
            "paper-level BitDistill reproduction or Kimi/MoE product claim."
        ),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    claim_rows = [
        [claim["claim"], claim["status"], claim["evidence"], claim["next_gate"]]
        for claim in summary["claims"]
    ]
    quality = summary["quality"]
    bitnet = summary["bitnet_sft"]
    bitdistill = summary["bitdistill"]
    runtime = summary["runtime"]
    return "\n\n".join(
        [
            f"# BitDistill Root-Cause Audit, {summary['date']}",
            summary["verdict"],
            "## Claim Ledger",
            md_table(["claim", "status", "evidence", "next gate"], claim_rows),
            "## Quality Anchors",
            md_table(
                ["row", "WikiText PPL", "ten-task mean", "delta vs FP", "tasks"],
                [
                    ["FP reference", quality["fp_wikitext_ppl"], quality["fp_ten_task_mean"], 0.0, quality["fp_tasks"]],
                    ["naive PTQ", quality["ptq_wikitext_ppl"], quality["ptq_ten_task_mean"], quality["ptq_delta_vs_fp"], quality["ptq_tasks"]],
                    ["row-scale QAT", quality["row_wikitext_ppl"], quality["row_ten_task_mean"], quality["row_delta_vs_fp"], quality["row_tasks"]],
                ],
            ),
            "## BitNet-SFT Baseline",
            md_table(
                ["field", "value"],
                [
                    ["complete paired rows", bitnet["complete_rows"]],
                    ["best accuracy", nested(bitnet, "best", "candidate_accuracy")],
                    ["paper anchor", nested(bitnet, "best", "paper_anchor")],
                    ["delta vs paper anchor", nested(bitnet, "best", "delta_vs_paper_anchor")],
                    ["paired delta vs FP", nested(bitnet, "best", "delta_vs_reference")],
                    ["paired CI95", nested(bitnet, "best", "paired_ci95")],
                    ["McNemar exact p", nested(bitnet, "best", "mcnemar_exact_p")],
                ],
            ),
            "## BitDistill Recovery Gate",
            md_table(
                ["field", "value"],
                [
                    ["best tensor MNLI", nested(bitdistill, "best_tensor", "accuracy")],
                    ["best tensor delta vs FP", nested(bitdistill, "best_tensor", "delta_vs_fp16")],
                    ["best row retrofit MNLI", nested(bitdistill, "best_row_retrofit", "accuracy")],
                    ["best row retrofit delta vs FP", nested(bitdistill, "best_row_retrofit", "delta_vs_fp16")],
                    ["controlled rows complete", f"{bitdistill['controlled_complete']}/{bitdistill['controlled_expected']}"],
                    ["controlled all complete", bitdistill["controlled_all_complete"]],
                    ["controlled rows passing FP gate", bitdistill["controlled_passed_fp_recovery_gate"]],
                    ["best controlled job", nested(bitdistill, "controlled_best", "job_id")],
                    ["best controlled MNLI", nested(bitdistill, "controlled_best", "metric_accuracy")],
                    ["best controlled delta vs FP", nested(bitdistill, "controlled_best", "paired", "delta_vs_reference")],
                    ["best controlled CI95", nested(bitdistill, "controlled_best", "paired", "paired_ci95")],
                ],
            ),
            "## Runtime Boundary",
            md_table(
                ["Q4_K_M normalized metric", "I2_SR value"],
                [[key, value] for key, value in sorted(runtime["q4_vs_i2sr"].items())],
            ),
            str(runtime.get("interpretation") or ""),
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp-wikitext", type=Path, default=Path("benchmark_results/quality-9735/qwen15b_fp_wikitext.json"))
    parser.add_argument("--ptq-wikitext", type=Path, default=Path("benchmark_results/quality-ptq-qwen15b/qwen15b_naive_ptq_wikitext.json"))
    parser.add_argument("--row-wikitext", type=Path, default=Path("benchmark_results/quality-qwen15b-klonly-row-notiehead-5000/qwen15b_ternary_wikitext.json"))
    parser.add_argument("--fp-lm-eval", type=Path, default=Path("benchmark_results/lm-eval-qwen15b-full10/qwen15b_fp.json"))
    parser.add_argument("--ptq-lm-eval", type=Path, default=Path("benchmark_results/lm-eval-qwen15b-full10/qwen15b_naive_ptq.json"))
    parser.add_argument("--row-lm-eval", type=Path, default=Path("benchmark_results/lm-eval-qwen15b-klonly-row-notiehead-full10/qwen15b_qat_ternary.json"))
    parser.add_argument("--bitnet-budget-paired", type=Path, default=Path(f"benchmark_results/bitnet_sft_budget_paired_{DATE}.json"))
    parser.add_argument("--bitnet-mechanics", type=Path, default=Path(f"benchmark_results/bitnet_sft_mechanics_audit_{DATE}.json"))
    parser.add_argument("--stage2-curve", type=Path, default=Path(f"benchmark_results/bitdistill_stage2_curve_{DATE}.json"))
    parser.add_argument("--controlled-curve", type=Path, default=Path(f"benchmark_results/bitdistill_controlled_curve_{DATE}.json"))
    parser.add_argument("--loss-scale", type=Path, default=Path(f"benchmark_results/bitdistill_loss_scale_audit_{DATE}.json"))
    parser.add_argument("--subln-audit", type=Path, default=Path(f"benchmark_results/subln_activation_variance_{DATE}.json"))
    parser.add_argument("--cpu-frontier", type=Path, default=Path(f"benchmark_results/cpu_tradeoff_frontier_{DATE}.json"))
    parser.add_argument("--tl2-contract", type=Path, default=Path(f"benchmark_results/tl2_row_scale_runtime_contract_{DATE}.json"))
    parser.add_argument("--moe-support", type=Path, default=Path(f"benchmark_results/moe_support_audit_{DATE}.json"))
    parser.add_argument("--kimi-feasibility", type=Path, default=Path(f"benchmark_results/kimi_config_feasibility_{DATE}.json"))
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/bitdistill_root_cause_audit_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/bitdistill_root_cause_audit_{DATE}.md"))
    args = parser.parse_args()

    summary = build_summary(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_markdown(summary)
    args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
