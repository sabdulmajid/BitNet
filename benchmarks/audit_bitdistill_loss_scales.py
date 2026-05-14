#!/usr/bin/env python3
"""Audit BitDistill loss component scales.

The BitDistill paper reports a classification attention-relation coefficient
of gamma=1e5.  This fork initially used gamma=100, so this report makes the
scale difference explicit from saved metrics instead of relying on prose.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = datetime.now(timezone.utc).date().isoformat()
TASKS = ["mnli", "qnli", "sst2"]
PAPER_CLASSIFICATION_ATTENTION_GAMMA = 100_000.0


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:.3e}"
        return f"{value:.6f}"
    return str(value)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def metric_paths(args: argparse.Namespace) -> list[tuple[str, Path]]:
    model_dir = args.model.replace("/", "-")
    specs = [
        ("short-tensor-layer-1", args.baseline_root, "{task}/bitdistill-tensor-layer-1"),
        ("short-row-layer-1", args.baseline_root, "{task}/bitdistill-row-layer-1"),
        ("short-tensor-layer-8", args.baseline_root, "{task}/bitdistill-tensor-layer-8"),
        ("paperlogit-tensor-layer-1", args.paperlogit_root, "{task}/bitdistill-paperlogit-tensor-layer-1"),
        ("paperlogit-row-layer-1", args.paperlogit_root, "{task}/bitdistill-paperlogit-row-layer-1"),
        ("paperlogit-headinit-tensor-layer-1", args.paperlogit_root, "{task}/bitdistill-paperlogit-headinit-tensor-layer-1"),
        ("longwarmup-tensor-gamma100", args.longwarmup_root, "{task}/bitdistill-longwarmup-tensor-layer-8"),
        ("longwarmup-row-gamma100", args.longwarmup_root, "{task}/bitdistill-longwarmup-row-layer-8"),
        ("longwarmup-tensor-papergamma", args.paper_hparam_root, "{task}/bitdistill-longwarmup-tensor-layer-8"),
    ]
    paths: list[tuple[str, Path]] = []
    for label, root, template in specs:
        for task in args.tasks:
            paths.append((f"{task}:{label}", root / model_dir / template.format(task=task) / "metrics.json"))
    if args.smoke_metrics.exists():
        paths.append(("smoke:papergamma", args.smoke_metrics))
    return paths


def summarize_path(label: str, path: Path) -> dict[str, Any]:
    data = read_json(path)
    last = data.get("last", {}) if isinstance(data.get("last"), dict) else {}
    weights = data.get("loss_weights", {}) if isinstance(data.get("loss_weights"), dict) else {}
    ce = float(last["ce"]) if finite(last.get("ce")) else None
    logit_kd = float(last["logit_kd"]) if finite(last.get("logit_kd")) else None
    attention_kd = float(last["attention_kd"]) if finite(last.get("attention_kd")) else None
    actual_attention_weight = float(weights["attention_kd_weight"]) if finite(weights.get("attention_kd_weight")) else None
    actual_weighted_attention = (
        float(last["weighted_attention_kd"])
        if finite(last.get("weighted_attention_kd"))
        else (actual_attention_weight * attention_kd if actual_attention_weight is not None and attention_kd is not None else None)
    )
    projected_paper_weighted_attention = (
        PAPER_CLASSIFICATION_ATTENTION_GAMMA * attention_kd if attention_kd is not None else None
    )
    actual_attention_to_ce = (
        actual_weighted_attention / ce if actual_weighted_attention is not None and ce not in (None, 0.0) else None
    )
    projected_paper_attention_to_ce = (
        projected_paper_weighted_attention / ce if projected_paper_weighted_attention is not None and ce not in (None, 0.0) else None
    )
    return {
        "label": label,
        "path": str(path),
        "exists": path.exists(),
        "stage": data.get("stage"),
        "method": data.get("method"),
        "steps": data.get("steps"),
        "task": data.get("task"),
        "scale_mode": data.get("scale_mode"),
        "distill_layer": data.get("distill_layer"),
        "ce": ce,
        "logit_kd": logit_kd,
        "attention_kd": attention_kd,
        "actual_attention_weight": actual_attention_weight,
        "actual_weighted_attention": actual_weighted_attention,
        "projected_paper_attention_weight": PAPER_CLASSIFICATION_ATTENTION_GAMMA,
        "projected_paper_weighted_attention": projected_paper_weighted_attention,
        "actual_attention_to_ce": actual_attention_to_ce,
        "projected_paper_attention_to_ce": projected_paper_attention_to_ce,
        "finite_loss_components": all(
            value is None or finite(value)
            for value in [ce, logit_kd, attention_kd, actual_weighted_attention, projected_paper_weighted_attention]
        ),
    }


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    rows = [summarize_path(label, path) for label, path in metric_paths(args)]
    materialized = [row for row in rows if row["exists"] and row["attention_kd"] is not None]
    projected_ratios = [
        float(row["projected_paper_attention_to_ce"])
        for row in materialized
        if row.get("projected_paper_attention_to_ce") is not None
    ]
    return {
        "schema": "bitdistill-loss-scale-audit-v1",
        "date": DATE,
        "paper_classification_attention_gamma": PAPER_CLASSIFICATION_ATTENTION_GAMMA,
        "rows": rows,
        "materialized_rows": len(materialized),
        "projected_paper_attention_to_ce_min": min(projected_ratios) if projected_ratios else None,
        "projected_paper_attention_to_ce_max": max(projected_ratios) if projected_ratios else None,
        "interpretation": (
            "gamma=1e5 is finite in the local smoke test, but it can dominate CE by orders of magnitude "
            "under this implementation's relation-loss normalization. Treat paper-gamma jobs as strict "
            "paper-hyperparameter stress tests and compare them to gamma=100 diagnostics."
        ),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    rows = [
        [
            row["label"],
            fmt(row["exists"]),
            fmt(row["steps"]),
            row.get("scale_mode") or "-",
            fmt(row.get("distill_layer")),
            fmt(row["ce"]),
            fmt(row["attention_kd"]),
            fmt(row["actual_attention_weight"]),
            fmt(row["actual_weighted_attention"]),
            fmt(row["actual_attention_to_ce"]),
            fmt(row["projected_paper_weighted_attention"]),
            fmt(row["projected_paper_attention_to_ce"]),
            row["path"],
        ]
        for row in summary["rows"]
    ]
    return "\n\n".join(
        [
            f"# BitDistill Loss-Scale Audit, {summary['date']}",
            f"Paper classification attention gamma: `{summary['paper_classification_attention_gamma']}`.",
            summary["interpretation"],
            f"Materialized rows with attention KD: `{summary['materialized_rows']}`.",
            f"Projected paper-gamma attention/CE range: `{fmt(summary['projected_paper_attention_to_ce_min'])}` to `{fmt(summary['projected_paper_attention_to_ce_max'])}`.",
            "## Runs",
            md_table(
                [
                    "run",
                    "exists",
                    "steps",
                    "scale",
                    "layer",
                    "CE",
                    "attention KD",
                    "actual gamma",
                    "actual weighted AD",
                    "actual AD/CE",
                    "projected paper weighted AD",
                    "projected paper AD/CE",
                    "metrics path",
                ],
                rows,
            ),
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--baseline-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls"))
    parser.add_argument("--paperlogit-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls-paperlogit"))
    parser.add_argument("--longwarmup-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls-longwarmup"))
    parser.add_argument("--paper-hparam-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma"))
    parser.add_argument("--smoke-metrics", type=Path, default=Path("benchmark_results/tmp_bitdistill_papergamma_smoke/metrics.json"))
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/bitdistill_loss_scale_audit_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/bitdistill_loss_scale_audit_{DATE}.md"))
    args = parser.parse_args()

    summary = build_summary(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_markdown(summary)
    args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
