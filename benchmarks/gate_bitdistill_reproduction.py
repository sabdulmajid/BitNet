#!/usr/bin/env python3
"""Gate BitDistill GLUE reproduction against FP16-SFT baselines."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = datetime.now(timezone.utc).date().isoformat()
TASKS = ["mnli", "qnli", "sst2"]


@dataclass(frozen=True)
class RunSpec:
    label: str
    root_attr: str
    template: str
    family: str


RUN_SPECS = [
    RunSpec("FP16-SFT", "baseline_root", "{task}/fp16_sft-tensor-layer-1", "baseline"),
    RunSpec("BitNet-SFT", "baseline_root", "{task}/bitnet_sft-tensor-layer-1", "baseline"),
    RunSpec("BitDistill short tensor layer -1", "baseline_root", "{task}/bitdistill-tensor-layer-1", "short"),
    RunSpec("BitDistill short row layer -1", "baseline_root", "{task}/bitdistill-row-layer-1", "short"),
    RunSpec("BitDistill short tensor layer -8", "baseline_root", "{task}/bitdistill-tensor-layer-8", "short"),
    RunSpec("BitDistill longwarmup tensor layer -8 gamma100", "longwarmup_root", "{task}/bitdistill-longwarmup-tensor-layer-8", "longwarmup_gamma100"),
    RunSpec("BitDistill longwarmup row layer -8", "longwarmup_root", "{task}/bitdistill-longwarmup-row-layer-8", "row_scale_candidate"),
    RunSpec(
        "BitDistill longwarmup tensor layer -8 paper gamma",
        "paper_hparam_root",
        "{task}/bitdistill-longwarmup-tensor-layer-8",
        "paper_hparam_candidate",
    ),
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run_path(root: Path, model: str, spec: RunSpec, task: str) -> Path:
    return root / model.replace("/", "-") / spec.template.format(task=task) / "metrics.json"


def accuracy_from_metrics(path: Path) -> float | None:
    data = read_json(path)
    eval_metrics = data.get("eval", {}) if isinstance(data.get("eval"), dict) else {}
    accuracy = eval_metrics.get("accuracy")
    return float(accuracy) if isinstance(accuracy, (int, float)) else None


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    fp_by_task: dict[str, float] = {}
    for task in args.tasks:
        fp_path = run_path(args.baseline_root, args.model, RUN_SPECS[0], task)
        fp_acc = accuracy_from_metrics(fp_path)
        if fp_acc is not None:
            fp_by_task[task] = fp_acc

    for task in args.tasks:
        for spec in RUN_SPECS:
            root = getattr(args, spec.root_attr)
            path = run_path(root, args.model, spec, task)
            acc = accuracy_from_metrics(path)
            fp = fp_by_task.get(task)
            gap = (fp - acc) if fp is not None and acc is not None else None
            rows.append(
                {
                    "task": task,
                    "run": spec.label,
                    "family": spec.family,
                    "path": str(path),
                    "exists": path.exists(),
                    "accuracy": acc,
                    "fp16_accuracy": fp,
                    "fp_minus_run": gap,
                    "passes_fp_gap": (abs(gap) <= args.max_fp_gap) if gap is not None else False,
                }
            )

    paper_rows = [row for row in rows if row["family"] == "paper_hparam_candidate"]
    row_scale_rows = [row for row in rows if row["family"] == "row_scale_candidate"]
    paper_complete = all(row["exists"] and row["accuracy"] is not None for row in paper_rows)
    row_complete = all(row["exists"] and row["accuracy"] is not None for row in row_scale_rows)
    paper_passed = paper_complete and all(row["passes_fp_gap"] for row in paper_rows)
    row_passed = row_complete and all(row["passes_fp_gap"] for row in row_scale_rows)

    comparisons = []
    for task in args.tasks:
        tensor = next(row for row in paper_rows if row["task"] == task)
        row_scale = next(row for row in row_scale_rows if row["task"] == task)
        delta = None
        if tensor["accuracy"] is not None and row_scale["accuracy"] is not None:
            delta = row_scale["accuracy"] - tensor["accuracy"]
        comparisons.append(
            {
                "task": task,
                "row_minus_tensor": delta,
                "tensor_accuracy": tensor["accuracy"],
                "row_accuracy": row_scale["accuracy"],
            }
        )

    return {
        "schema": "bitdistill-reproduction-gate-v1",
        "date": DATE,
        "model": args.model,
        "tasks": args.tasks,
        "max_fp_gap": args.max_fp_gap,
        "baseline_root": str(args.baseline_root),
        "longwarmup_root": str(args.longwarmup_root),
        "paper_hparam_root": str(args.paper_hparam_root),
        "paper_style_tensor_complete": paper_complete,
        "paper_style_tensor_passed": paper_passed,
        "row_scale_complete": row_complete,
        "row_scale_passed": row_passed,
        "rows": rows,
        "row_scale_comparisons": comparisons,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def render_markdown(summary: dict[str, Any]) -> str:
    rows = [
        [
            row["task"],
            row["run"],
            row["family"],
            fmt(row["exists"]),
            fmt(row["accuracy"]),
            fmt(row["fp16_accuracy"]),
            fmt(row["fp_minus_run"]),
            "pass" if row["passes_fp_gap"] else "fail_or_pending",
            row["path"],
        ]
        for row in summary["rows"]
    ]
    comparison_rows = [
        [
            row["task"],
            fmt(row["tensor_accuracy"]),
            fmt(row["row_accuracy"]),
            fmt(row["row_minus_tensor"]),
        ]
        for row in summary["row_scale_comparisons"]
    ]
    return "\n\n".join(
        [
            f"# BitDistill Reproduction Gate, {summary['date']}",
            f"Model: `{summary['model']}`.",
            f"Threshold: absolute FP16-SFT gap <= `{summary['max_fp_gap']}` accuracy.",
            f"Strict paper-hyperparameter tensor candidate complete: `{summary['paper_style_tensor_complete']}`.",
            f"Strict paper-hyperparameter tensor candidate passed: `{summary['paper_style_tensor_passed']}`.",
            f"Row-scale candidate complete: `{summary['row_scale_complete']}`.",
            f"Row-scale candidate passed: `{summary['row_scale_passed']}`.",
            "## Runs",
            md_table(
                ["task", "run", "family", "exists", "accuracy", "FP16", "FP-run", "status", "metrics path"],
                rows,
            ),
            "## Row-Scale Comparison",
            md_table(["task", "tensor", "row", "row-tensor"], comparison_rows),
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls"))
    parser.add_argument("--longwarmup-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls-longwarmup"))
    parser.add_argument("--paper-hparam-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--max-fp-gap", type=float, default=0.01)
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/bitdistill_reproduction_gate_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/bitdistill_reproduction_gate_{DATE}.md"))
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
