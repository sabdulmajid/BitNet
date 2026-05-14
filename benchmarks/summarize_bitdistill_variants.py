#!/usr/bin/env python3
"""Discover and summarize BitDistill variant metrics.

The fixed GLUE summarizer reports the paper matrix.  This script is for
diagnostic branches such as attention-layer sweeps and teacher-head probes,
where run directory names are intentionally not part of the core matrix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TASKS = ["mnli", "qnli", "sst2"]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return "-" if value is None else str(value)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    def cell(value: str) -> str:
        return value.replace("|", "\\|")

    lines = ["| " + " | ".join(cell(header) for header in headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def fp_lookup(fp_root: Path, model: str, tasks: list[str]) -> dict[str, float]:
    lookup: dict[str, float] = {}
    model_dir = model.replace("/", "-")
    for task in tasks:
        path = fp_root / model_dir / task / "fp16_sft-tensor-layer-1" / "metrics.json"
        data = read_json(path)
        value = data.get("eval", {}).get("accuracy") if isinstance(data.get("eval"), dict) else None
        if isinstance(value, (int, float)):
            lookup[task] = float(value)
    return lookup


def discover_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    model_dir = args.model.replace("/", "-")
    fp_acc = fp_lookup(args.fp_root, args.model, args.tasks)
    rows: list[dict[str, Any]] = []
    for root in args.roots:
        for task in args.tasks:
            task_dir = root / model_dir / task
            for path in sorted(task_dir.glob("*/metrics.json")):
                data = read_json(path)
                eval_metrics = data.get("eval", {}) if isinstance(data.get("eval"), dict) else {}
                last = data.get("last", {}) if isinstance(data.get("last"), dict) else {}
                state_load = data.get("state_load", {}) if isinstance(data.get("state_load"), dict) else {}
                output_head_init = data.get("output_head_init", {}) if isinstance(data.get("output_head_init"), dict) else {}
                loss_weights = data.get("loss_weights", {}) if isinstance(data.get("loss_weights"), dict) else {}
                acc = eval_metrics.get("accuracy")
                gap = fp_acc.get(task) - float(acc) if isinstance(acc, (int, float)) and task in fp_acc else None
                mismatches = state_load.get("skipped_shape_mismatches", {})
                rows.append(
                    {
                        "root": str(root),
                        "task": task,
                        "run": path.parent.name,
                        "accuracy": acc,
                        "fp_accuracy": fp_acc.get(task),
                        "fp_minus_run": gap,
                        "examples": eval_metrics.get("eval_examples"),
                        "steps": data.get("steps"),
                        "method": data.get("method"),
                        "format": data.get("task_format"),
                        "scale": data.get("scale_mode"),
                        "distill_layer": data.get("distill_layer"),
                        "head_init": output_head_init.get("copied"),
                        "state_loaded": state_load.get("loaded"),
                        "shape_mismatches": len(mismatches) if isinstance(mismatches, dict) else None,
                        "ce": last.get("ce"),
                        "weighted_logit_kd": last.get("weighted_logit_kd"),
                        "weighted_attention_kd": last.get("weighted_attention_kd"),
                        "logit_kd_weight": loss_weights.get("logit_kd_weight"),
                        "logit_kd_temperature_scale": loss_weights.get("logit_kd_temperature_scale"),
                        "attention_kd_weight": loss_weights.get("attention_kd_weight"),
                        "metrics_path": str(path),
                    }
                )
    return rows


def render_markdown(args: argparse.Namespace, rows: list[dict[str, Any]]) -> str:
    metric_rows = [
        [
            row["root"],
            row["task"],
            row["run"],
            fmt(row["accuracy"]),
            fmt(row["fp_accuracy"]),
            fmt(row["fp_minus_run"]),
            fmt(row["examples"]),
            fmt(row["steps"]),
            row.get("method") or "-",
            row.get("format") or "-",
            row.get("scale") or "-",
            fmt(row["distill_layer"]),
            fmt(row["head_init"]),
            fmt(row["state_loaded"]),
            fmt(row["shape_mismatches"]),
            fmt(row["ce"]),
            fmt(row["weighted_logit_kd"]),
            row.get("logit_kd_temperature_scale") or "-",
            fmt(row["weighted_attention_kd"]),
            row["metrics_path"],
        ]
        for row in rows
    ]
    return "\n\n".join(
        [
            f"# {args.title}",
            f"Model: `{args.model}`.",
            f"FP reference root: `{args.fp_root}`.",
            "## Runs",
            md_table(
                [
                    "root",
                    "task",
                    "run",
                    "accuracy",
                    "FP16",
                    "FP-run",
                    "examples",
                    "steps",
                    "method",
                    "format",
                    "scale",
                    "layer",
                    "head init",
                    "state loaded",
                    "shape mismatches",
                    "last CE",
                    "last wLogitKD",
                    "logit temp scale",
                    "last wAttnKD",
                    "metrics path",
                ],
                metric_rows,
            ),
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--fp-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--title", default="BitDistill Variant Summary, 2026-05-14")
    parser.add_argument("--output-json", type=Path, default=Path("benchmark_results/bitdistill_variant_summary_2026-05-14.json"))
    parser.add_argument("--output-md", type=Path, default=Path("benchmarks/results/bitdistill_variant_summary_2026-05-14.md"))
    args = parser.parse_args()

    rows = discover_rows(args)
    summary = {
        "schema": "bitdistill-variant-summary-v1",
        "model": args.model,
        "roots": [str(root) for root in args.roots],
        "fp_root": str(args.fp_root),
        "tasks": args.tasks,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_markdown(args, rows)
    args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
