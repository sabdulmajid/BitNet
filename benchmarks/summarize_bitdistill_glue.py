#!/usr/bin/env python3
"""Summarize BitDistill GLUE metrics and check reproduction thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TASKS = ["mnli", "qnli", "sst2"]
METHOD_ORDER = ["fp16_sft", "bitnet_sft", "bitdistill_tensor", "bitdistill_row"]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def find_metrics(root: Path, model: str, task: str, method: str, scale: str) -> dict[str, Any]:
    path = root / model.replace("/", "-") / task / f"{method}-{scale}-layer-1" / "metrics.json"
    data = read_json(path)
    return {"path": str(path), "exists": bool(data), "data": data}


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for task in args.tasks:
        specs = [
            ("fp16_sft", "fp16_sft", "tensor"),
            ("bitnet_sft", "bitnet_sft", "tensor"),
            ("bitdistill_tensor", "bitdistill", "tensor"),
            ("bitdistill_row", "bitdistill", "row"),
        ]
        for label, method, scale in specs:
            item = find_metrics(args.root, args.model, task, method, scale)
            data = item["data"]
            eval_metrics = data.get("eval", {}) if isinstance(data.get("eval"), dict) else {}
            rows.append(
                {
                    "task": task,
                    "label": label,
                    "path": item["path"],
                    "exists": item["exists"],
                    "accuracy": eval_metrics.get("accuracy"),
                    "examples": eval_metrics.get("eval_examples"),
                    "steps": data.get("steps"),
                    "task_format": data.get("task_format"),
                    "scale_mode": scale,
                }
            )

    verdicts: list[dict[str, Any]] = []
    for task in args.tasks:
        fp = next((row for row in rows if row["task"] == task and row["label"] == "fp16_sft"), {})
        bd = next((row for row in rows if row["task"] == task and row["label"] == "bitdistill_tensor"), {})
        row = next((item for item in rows if item["task"] == task and item["label"] == "bitdistill_row"), {})
        fp_acc = fp.get("accuracy")
        bd_acc = bd.get("accuracy")
        row_acc = row.get("accuracy")
        gap = None
        row_delta = None
        passed = False
        if isinstance(fp_acc, (int, float)) and isinstance(bd_acc, (int, float)):
            gap = float(fp_acc) - float(bd_acc)
            passed = abs(gap) <= args.max_fp_gap
        if isinstance(row_acc, (int, float)) and isinstance(bd_acc, (int, float)):
            row_delta = float(row_acc) - float(bd_acc)
        verdicts.append(
            {
                "task": task,
                "fp16_accuracy": fp_acc,
                "bitdistill_accuracy": bd_acc,
                "fp_minus_bitdistill": gap,
                "row_minus_tensor_bitdistill": row_delta,
                "passes_fp_gap": passed,
            }
        )
    return {
        "schema": "bitdistill-glue-summary-v1",
        "root": str(args.root),
        "model": args.model,
        "tasks": args.tasks,
        "max_fp_gap": args.max_fp_gap,
        "rows": rows,
        "verdicts": verdicts,
        "passed": all(item["passes_fp_gap"] for item in verdicts),
    }


def fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return "-" if value is None else str(value)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_markdown(summary: dict[str, Any]) -> str:
    metric_rows = [
        [
            row["task"],
            row["label"],
            "yes" if row["exists"] else "no",
            fmt(row["accuracy"]),
            fmt(row["examples"]),
            fmt(row["steps"]),
            row["path"],
        ]
        for row in summary["rows"]
    ]
    verdict_rows = [
        [
            row["task"],
            fmt(row["fp16_accuracy"]),
            fmt(row["bitdistill_accuracy"]),
            fmt(row["fp_minus_bitdistill"]),
            fmt(row["row_minus_tensor_bitdistill"]),
            "pass" if row["passes_fp_gap"] else "fail",
        ]
        for row in summary["verdicts"]
    ]
    return "\n\n".join(
        [
            "# BitDistill GLUE Summary, 2026-05-14",
            f"Model: `{summary['model']}`.",
            f"Overall threshold pass: `{summary['passed']}` with max FP gap `{summary['max_fp_gap']}`.",
            "## Metrics",
            md_table(["task", "run", "exists", "accuracy", "examples", "steps", "metrics path"], metric_rows),
            "## Verdicts",
            md_table(["task", "FP16", "BitDistill", "FP-BitDistill", "row-tensor", "status"], verdict_rows),
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("checkpoints/bitdistill-glue"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    parser.add_argument("--max-fp-gap", type=float, default=0.01)
    parser.add_argument("--output-json", type=Path, default=Path("benchmark_results/bitdistill_glue_summary_2026-05-14.json"))
    parser.add_argument("--output-md", type=Path, default=Path("benchmarks/results/bitdistill_glue_summary_2026-05-14.md"))
    args = parser.parse_args()

    summary = build_summary(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(summary), encoding="utf-8")
    print(render_markdown(summary))


if __name__ == "__main__":
    main()
