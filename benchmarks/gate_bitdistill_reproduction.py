#!/usr/bin/env python3
"""Gate BitDistill GLUE reproduction against FP16-SFT baselines."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = datetime.now(timezone.utc).date().isoformat()
TASKS = ["mnli", "qnli", "sst2"]
EXPECTED_EVAL_EXAMPLES = {
    "mnli": 9815,
    "qnli": 5463,
    "sst2": 872,
}
Z_95 = 1.959963984540054


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
    RunSpec(
        "BitDistill longwarmup row layer -8 paper gamma",
        "paper_hparam_row_root",
        "{task}/bitdistill-longwarmup-row-layer-8",
        "paper_hparam_row_candidate",
    ),
    RunSpec(
        "BitDistill longwarmup tensor layer -8 paper gamma lr1e-5",
        "paper_hparam_lr1_root",
        "{task}/bitdistill-longwarmup-tensor-layer-8",
        "paper_hparam_search",
    ),
    RunSpec(
        "BitDistill longwarmup tensor layer -8 paper gamma lr5e-5",
        "paper_hparam_lr5_root",
        "{task}/bitdistill-longwarmup-tensor-layer-8",
        "paper_hparam_search",
    ),
    RunSpec(
        "BitDistill longwarmup tensor layer -8 paper gamma headinit",
        "paper_hparam_headinit_root",
        "{task}/bitdistill-longwarmup-tensor-layer-8",
        "paper_hparam_search",
    ),
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run_path(root: Path, model: str, spec: RunSpec, task: str) -> Path:
    return root / model.replace("/", "-") / spec.template.format(task=task) / "metrics.json"


def metric_summary_from_path(path: Path, task: str) -> dict[str, Any]:
    data = read_json(path)
    eval_metrics = data.get("eval", {}) if isinstance(data.get("eval"), dict) else {}
    accuracy = eval_metrics.get("accuracy")
    examples = eval_metrics.get("eval_examples")
    accuracy_value = float(accuracy) if isinstance(accuracy, (int, float)) else None
    example_count = int(examples) if isinstance(examples, (int, float)) and examples > 0 else None
    expected_examples = EXPECTED_EVAL_EXAMPLES[task]
    return {
        "accuracy": accuracy_value,
        "examples": example_count,
        "expected_examples": expected_examples,
        "full_eval_examples": example_count == expected_examples,
        "accuracy_ci95": wilson_ci(accuracy_value, example_count),
    }


def wilson_ci(p: float | None, n: int | None, z: float = Z_95) -> list[float] | None:
    if p is None or n is None or n <= 0 or not math.isfinite(p):
        return None
    p = min(max(float(p), 0.0), 1.0)
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def difference_ci(
    left: float | None,
    left_n: int | None,
    right: float | None,
    right_n: int | None,
    z: float = Z_95,
) -> list[float] | None:
    if left is None or right is None or left_n is None or right_n is None or left_n <= 0 or right_n <= 0:
        return None
    if not (math.isfinite(left) and math.isfinite(right)):
        return None
    se = math.sqrt(left * (1.0 - left) / left_n + right * (1.0 - right) / right_n)
    delta = left - right
    return [delta - z * se, delta + z * se]


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    fp_by_task: dict[str, dict[str, Any]] = {}
    for task in args.tasks:
        fp_path = run_path(args.baseline_root, args.model, RUN_SPECS[0], task)
        fp_metrics = metric_summary_from_path(fp_path, task)
        if fp_metrics["accuracy"] is not None and fp_metrics["full_eval_examples"]:
            fp_by_task[task] = fp_metrics

    for task in args.tasks:
        for spec in RUN_SPECS:
            root = getattr(args, spec.root_attr)
            path = run_path(root, args.model, spec, task)
            metrics = metric_summary_from_path(path, task)
            acc = metrics["accuracy"]
            examples = metrics["examples"]
            fp = fp_by_task.get(task, {})
            fp_acc = fp.get("accuracy")
            fp_examples = fp.get("examples")
            gap = (fp_acc - acc) if fp_acc is not None and acc is not None else None
            rows.append(
                {
                    "task": task,
                    "run": spec.label,
                    "family": spec.family,
                    "path": str(path),
                    "exists": path.exists(),
                    "accuracy": acc,
                    "examples": examples,
                    "expected_examples": metrics["expected_examples"],
                    "full_eval_examples": metrics["full_eval_examples"],
                    "accuracy_ci95": metrics["accuracy_ci95"],
                    "fp16_accuracy": fp_acc,
                    "fp16_examples": fp_examples,
                    "fp_minus_run": gap,
                    "fp_minus_run_ci95": difference_ci(fp_acc, fp_examples, acc, examples),
                    "passes_fp_gap": (metrics["full_eval_examples"] and abs(gap) <= args.max_fp_gap) if gap is not None else False,
                }
            )

    paper_rows = [row for row in rows if row["family"] == "paper_hparam_candidate"]
    paper_search_rows = [row for row in rows if row["family"] in {"paper_hparam_candidate", "paper_hparam_search"}]
    tensor_gamma100_rows = [row for row in rows if row["family"] == "longwarmup_gamma100"]
    row_scale_rows = [row for row in rows if row["family"] == "row_scale_candidate"]
    paper_row_rows = [row for row in rows if row["family"] == "paper_hparam_row_candidate"]
    paper_complete = all(row["exists"] and row["accuracy"] is not None and row["full_eval_examples"] for row in paper_rows)
    paper_search_complete = all(row["exists"] and row["accuracy"] is not None and row["full_eval_examples"] for row in paper_search_rows)
    row_complete = all(row["exists"] and row["accuracy"] is not None and row["full_eval_examples"] for row in row_scale_rows)
    paper_passed = paper_complete and all(row["passes_fp_gap"] for row in paper_rows)
    paper_search_best: dict[str, dict[str, Any]] = {}
    for task in args.tasks:
        task_rows = [row for row in paper_search_rows if row["task"] == task and row["accuracy"] is not None]
        if task_rows:
            paper_search_best[task] = max(task_rows, key=lambda row: float(row["accuracy"]))
    paper_search_passed = (
        paper_search_complete
        and len(paper_search_best) == len(args.tasks)
        and all(row["passes_fp_gap"] for row in paper_search_best.values())
    )
    row_passed = row_complete and all(row["passes_fp_gap"] for row in row_scale_rows)

    comparisons = []
    for task in args.tasks:
        tensor = next(row for row in tensor_gamma100_rows if row["task"] == task)
        row_scale = next(row for row in row_scale_rows if row["task"] == task)
        delta = None
        delta_ci95 = None
        if tensor["accuracy"] is not None and row_scale["accuracy"] is not None:
            delta = row_scale["accuracy"] - tensor["accuracy"]
            delta_ci95 = difference_ci(
                row_scale["accuracy"],
                row_scale["examples"],
                tensor["accuracy"],
                tensor["examples"],
            )
        comparisons.append(
            {
                "task": task,
                "row_minus_tensor": delta,
                "row_minus_tensor_ci95": delta_ci95,
                "tensor_accuracy": tensor["accuracy"],
                "tensor_examples": tensor["examples"],
                "row_accuracy": row_scale["accuracy"],
                "row_examples": row_scale["examples"],
                "comparison": "gamma100_row_minus_tensor",
            }
        )
        paper_tensor = next(row for row in paper_rows if row["task"] == task)
        paper_row = next(row for row in paper_row_rows if row["task"] == task)
        paper_delta = None
        paper_delta_ci95 = None
        if paper_tensor["accuracy"] is not None and paper_row["accuracy"] is not None:
            paper_delta = paper_row["accuracy"] - paper_tensor["accuracy"]
            paper_delta_ci95 = difference_ci(
                paper_row["accuracy"],
                paper_row["examples"],
                paper_tensor["accuracy"],
                paper_tensor["examples"],
            )
        comparisons.append(
            {
                "task": task,
                "row_minus_tensor": paper_delta,
                "row_minus_tensor_ci95": paper_delta_ci95,
                "tensor_accuracy": paper_tensor["accuracy"],
                "tensor_examples": paper_tensor["examples"],
                "row_accuracy": paper_row["accuracy"],
                "row_examples": paper_row["examples"],
                "comparison": "paper_gamma_row_minus_tensor",
            }
        )

    return {
        "schema": "bitdistill-reproduction-gate-v1",
        "date": DATE,
        "model": args.model,
        "tasks": args.tasks,
        "max_fp_gap": args.max_fp_gap,
        "confidence": {
            "level": 0.95,
            "accuracy_interval": "Wilson score interval from aggregate accuracy and eval_examples",
            "difference_interval": "Unpaired normal approximation from aggregate accuracies; paired prediction deltas are audited separately when eval_predictions.jsonl exists.",
        },
        "expected_eval_examples": {task: EXPECTED_EVAL_EXAMPLES[task] for task in args.tasks},
        "baseline_root": str(args.baseline_root),
        "longwarmup_root": str(args.longwarmup_root),
        "paper_hparam_root": str(args.paper_hparam_root),
        "paper_hparam_row_root": str(args.paper_hparam_row_root),
        "paper_hparam_lr1_root": str(args.paper_hparam_lr1_root),
        "paper_hparam_lr5_root": str(args.paper_hparam_lr5_root),
        "paper_hparam_headinit_root": str(args.paper_hparam_headinit_root),
        "paper_style_tensor_complete": paper_complete,
        "paper_style_tensor_passed": paper_passed,
        "paper_search_tensor_complete": paper_search_complete,
        "paper_search_tensor_passed": paper_search_passed,
        "paper_search_best_by_task": paper_search_best,
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


def fmt_ci(value: Any) -> str:
    if not isinstance(value, list) or len(value) != 2:
        return "-"
    left, right = value
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return "-"
    return f"[{float(left):.6f}, {float(right):.6f}]"


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
            fmt(row["examples"]),
            fmt(row["expected_examples"]),
            "pass" if row["full_eval_examples"] else "fail_or_pending",
            fmt_ci(row.get("accuracy_ci95")),
            fmt(row["fp16_accuracy"]),
            fmt(row["fp_minus_run"]),
            fmt_ci(row.get("fp_minus_run_ci95")),
            "pass" if row["passes_fp_gap"] else "fail_or_pending",
            row["path"],
        ]
        for row in summary["rows"]
    ]
    comparison_rows = [
        [
            row["task"],
            row.get("comparison", "-"),
            fmt(row["tensor_accuracy"]),
            fmt(row["tensor_examples"]),
            fmt(row["row_accuracy"]),
            fmt(row["row_examples"]),
            fmt(row["row_minus_tensor"]),
            fmt_ci(row.get("row_minus_tensor_ci95")),
        ]
        for row in summary["row_scale_comparisons"]
    ]
    return "\n\n".join(
        [
            f"# BitDistill Reproduction Gate, {summary['date']}",
            f"Model: `{summary['model']}`.",
            f"Threshold: absolute FP16-SFT gap <= `{summary['max_fp_gap']}` accuracy.",
            f"Full-evaluation contract: `{summary['expected_eval_examples']}` examples. Rows that do not match these counts cannot pass this gate.",
            "Confidence intervals: accuracy uses Wilson 95% intervals; this aggregate gate uses unpaired normal delta intervals. The paired-prediction audit is the authoritative example-level comparison when `eval_predictions.jsonl` exists.",
            f"Strict paper-hyperparameter tensor candidate complete: `{summary['paper_style_tensor_complete']}`.",
            f"Strict paper-hyperparameter tensor candidate passed: `{summary['paper_style_tensor_passed']}`.",
            f"Paper tensor LR/headinit search complete: `{summary['paper_search_tensor_complete']}`.",
            f"Paper tensor LR/headinit search passed: `{summary['paper_search_tensor_passed']}`.",
            f"Row-scale candidate complete: `{summary['row_scale_complete']}`.",
            f"Row-scale candidate passed: `{summary['row_scale_passed']}`.",
            "## Runs",
            md_table(
                [
                    "task",
                    "run",
                    "family",
                    "exists",
                    "accuracy",
                    "examples",
                    "expected examples",
                    "full eval",
                    "accuracy 95% CI",
                    "FP16",
                    "FP-run",
                    "FP-run 95% CI",
                    "status",
                    "metrics path",
                ],
                rows,
            ),
            "## Row-Scale Comparison",
            md_table(
                ["task", "comparison", "tensor", "tensor n", "row", "row n", "row-tensor", "row-tensor 95% CI"],
                comparison_rows,
            ),
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls"))
    parser.add_argument("--longwarmup-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls-longwarmup"))
    parser.add_argument("--paper-hparam-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma"))
    parser.add_argument("--paper-hparam-row-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-row"))
    parser.add_argument("--paper-hparam-lr1-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-lr1e-5"))
    parser.add_argument("--paper-hparam-lr5-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-lr5e-5"))
    parser.add_argument("--paper-hparam-headinit-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-headinit"))
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
