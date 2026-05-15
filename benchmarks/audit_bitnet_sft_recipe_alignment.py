#!/usr/bin/env python3
"""Audit the narrow BitNet-SFT recipe-alignment blockers.

This is a read-only report. It does not decide whether BitDistill works; it
records whether the local BitNet-SFT implementation has obvious mechanical
divergences before broader distillation sweeps are interpreted.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()
PAPER_BITNET_SFT_MNLI = 0.608000


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


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(fmt(cell) for cell in row) + " |")
    return "\n".join(out)


def source_checks(train_distill: str, train_bitdistill: str) -> list[dict[str, str]]:
    checks = [
        {
            "check": "Ternary weight formula is absmean STE",
            "status": "pass"
            if "alpha = weight.detach().abs().mean().clamp_min(eps)" in train_distill
            and "torch.round(weight.detach() / alpha).clamp_(-1, 1)" in train_distill
            else "review",
            "evidence": "TernaryWeightSTE uses alpha=mean(abs(W)), round(W/alpha), clamp [-1,1], identity backward.",
            "risk": "Formula is paper-style for tensor scale; row scale is a fork variant.",
        },
        {
            "check": "Activation quantization is per-token absmax int8 STE",
            "status": "pass"
            if "amax(dim=-1, keepdim=True).clamp_min(eps) / 127.0" in train_distill
            and "clamp_(-128, 127)" in train_distill
            else "review",
            "evidence": "AbsmaxActivationSTE quantizes x with per-token absmax / 127 and identity backward.",
            "risk": "A8 ablation shows this is not the dominant MNLI gap locally.",
        },
        {
            "check": "Sequence-classification head is excluded from BitLinear replacement by default",
            "status": "pass" if 'default="score|classifier"' in train_bitdistill else "review",
            "evidence": "Default exclude regex is score|classifier.",
            "risk": "Dense-head treatment appears aligned for GLUE sequence classification.",
        },
        {
            "check": "SubLN is inserted before attention output and FFN down projections",
            "status": "pass"
            if 'hasattr(self_attn, "o_proj")' in train_bitdistill
            and 'hasattr(mlp, "down_proj")' in train_bitdistill
            else "review",
            "evidence": "add_subln_to_qwen_blocks wraps o_proj and down_proj with RMSNorm before projection.",
            "risk": "The local SubLN-only control worsens MNLI, so placement/timing/init still need recipe audit.",
        },
        {
            "check": "BitLinear replacement happens after SubLN insertion",
            "status": "pass"
            if "if args.use_subln:" in train_bitdistill
            and "replace_linear_layers(" in train_bitdistill
            and train_bitdistill.find("if args.use_subln:") < train_bitdistill.find("replace_linear_layers(")
            else "review",
            "evidence": "prepare_bitnet_student inserts SubLN first, then replaces nested nn.Linear projections.",
            "risk": "This is mechanically coherent, but may not match the paper's exact initialization/training timing.",
        },
    ]
    return checks


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    train_distill = args.train_distill.read_text(encoding="utf-8")
    train_bitdistill = args.train_bitdistill.read_text(encoding="utf-8")
    baseline = read_json(args.baseline_json)
    sweep = read_json(args.sweep_json)

    best_sweep = sweep.get("best") or {}
    baseline_rows = baseline.get("checkpoints") or []
    by_name = {row.get("name"): row for row in baseline_rows if isinstance(row, dict)}
    bitnet = by_name.get("BitNet-SFT", {})
    weights_only = by_name.get("BitNet-SFT weights-only", {})
    subln = by_name.get("BitNet-SFT SubLN", {})

    bitnet_acc = bitnet.get("accuracy")
    best_acc = best_sweep.get("accuracy")
    checks = source_checks(train_distill, train_bitdistill)
    checks.extend(
        [
            {
                "check": "Default BitNet-SFT reaches paper anchor",
                "status": "fail",
                "evidence": f"default={fmt(bitnet_acc)}, paper={fmt(PAPER_BITNET_SFT_MNLI)}, gap={fmt(PAPER_BITNET_SFT_MNLI - bitnet_acc if isinstance(bitnet_acc, (int, float)) else None)}",
                "risk": "Primary blocker: BitDistill recovery cannot be interpreted until this baseline is explained.",
            },
            {
                "check": "Best completed budget row reaches paper anchor",
                "status": "fail"
                if isinstance(best_acc, (int, float)) and best_acc < PAPER_BITNET_SFT_MNLI
                else "pending",
                "evidence": f"best_completed={fmt(best_acc)}, steps={best_sweep.get('steps', '-')}, lr={best_sweep.get('lr', '-')}",
                "risk": "Pending 10000-step rows are needed to distinguish undertraining from equation mismatch.",
            },
            {
                "check": "Activation quantization explains the gap",
                "status": "fail",
                "evidence": f"weights_only={fmt(weights_only.get('accuracy'))}, W1.58A8={fmt(bitnet_acc)}",
                "risk": "A8 removal improves only slightly, so the problem is mostly ternary training/recipe.",
            },
            {
                "check": "SubLN-only local control explains the gap",
                "status": "fail",
                "evidence": f"subln_only={fmt(subln.get('accuracy'))}, default={fmt(bitnet_acc)}",
                "risk": "SubLN likely requires exact paper timing/budget or current insertion differs in a material way.",
            },
        ]
    )

    return {
        "date": DATE,
        "baseline_json": str(args.baseline_json),
        "sweep_json": str(args.sweep_json),
        "paper_bitnet_sft_mnli": PAPER_BITNET_SFT_MNLI,
        "default_bitnet_sft_accuracy": bitnet_acc,
        "best_completed_sweep_accuracy": best_acc,
        "checks": checks,
        "verdict": "mechanically_plausible_but_not_recipe_matched",
        "next": [
            "Finish the remaining pending budget rows, especially the 10000-step rows.",
            "If the curve saturates low, audit BitLinear/SubLN equation parity against the paper implementation.",
            "Keep row-scale results separate from paper-reproduction labels.",
            "Do not broaden MoE/Kimi claims until dense BitNet-SFT is explained.",
        ],
    }


def render_markdown(summary: dict[str, Any]) -> str:
    checks = [
        [row["check"], row["status"], row["evidence"], row["risk"]]
        for row in summary["checks"]
    ]
    lines = [
        f"# BitNet-SFT Recipe Alignment Audit, {summary['date']}",
        "Verdict: the source-level implementation is mechanically plausible, but the local BitNet-SFT accuracy is not recipe-matched to the paper anchor yet.",
        "",
        f"- Default BitNet-SFT MNLI: `{fmt(summary['default_bitnet_sft_accuracy'])}`.",
        f"- Paper BitNet-SFT MNLI anchor: `{fmt(summary['paper_bitnet_sft_mnli'])}`.",
        f"- Best completed budget-sweep row: `{fmt(summary['best_completed_sweep_accuracy'])}`.",
        "",
        md_table(["check", "status", "evidence", "risk"], checks),
        "",
        "## Next Actions",
        "\n".join(f"{idx}. {item}" for idx, item in enumerate(summary["next"], start=1)),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-distill", type=Path, default=Path("train_distill.py"))
    parser.add_argument("--train-bitdistill", type=Path, default=Path("train_bitdistill.py"))
    parser.add_argument("--baseline-json", type=Path, default=Path(f"benchmark_results/bitnet_sft_baseline_audit_{DATE}.json"))
    parser.add_argument("--sweep-json", type=Path, default=Path(f"benchmark_results/bitnet_sft_budget_sweep_{DATE}.json"))
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/bitnet_sft_recipe_alignment_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/bitnet_sft_recipe_alignment_{DATE}.md"))
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
