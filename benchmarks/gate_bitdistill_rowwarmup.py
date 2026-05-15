#!/usr/bin/env python3
"""Gate the clean row-warmup BitDistill branch against FP16 and tensor branches."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()
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
    family: str
    root_attr: str
    template: str


RUN_SPECS = [
    RunSpec("FP16-SFT", "baseline", "baseline_root", "{task}/fp16_sft-tensor-layer-1"),
    RunSpec("tensor-warmup tensor gamma100", "tensor_warmup_gamma100", "tensor_gamma100_root", "{task}/bitdistill-longwarmup-tensor-layer-8"),
    RunSpec("tensor-warmup row gamma100", "tensor_warmup_row_gamma100", "tensor_gamma100_root", "{task}/bitdistill-longwarmup-row-layer-8"),
    RunSpec("tensor-warmup tensor paper gamma", "tensor_warmup_papergamma", "tensor_papergamma_root", "{task}/bitdistill-longwarmup-tensor-layer-8"),
    RunSpec("tensor-warmup row paper gamma", "tensor_warmup_row_papergamma", "tensor_papergamma_row_root", "{task}/bitdistill-longwarmup-row-layer-8"),
    RunSpec("row-warmup row gamma100", "row_warmup_gamma100", "row_warmup_gamma100_root", "{task}/bitdistill-longwarmup-row-layer-8"),
    RunSpec("row-warmup row paper gamma", "row_warmup_papergamma", "row_warmup_papergamma_root", "{task}/bitdistill-longwarmup-row-layer-8"),
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def run_path(root: Path, model: str, spec: RunSpec, task: str) -> Path:
    return root / model.replace("/", "-") / spec.template.format(task=task) / "metrics.json"


def metric_summary(path: Path, task: str) -> dict[str, Any]:
    data = read_json(path)
    eval_metrics = data.get("eval", {}) if isinstance(data.get("eval"), dict) else {}
    last = data.get("last", {}) if isinstance(data.get("last"), dict) else {}
    loss_weights = data.get("loss_weights", {}) if isinstance(data.get("loss_weights"), dict) else {}
    accuracy = eval_metrics.get("accuracy")
    examples = eval_metrics.get("eval_examples")
    accuracy_value = float(accuracy) if isinstance(accuracy, (int, float)) else None
    example_count = int(examples) if isinstance(examples, (int, float)) and examples > 0 else None
    expected = EXPECTED_EVAL_EXAMPLES[task]
    return {
        "exists": path.exists(),
        "accuracy": accuracy_value,
        "examples": example_count,
        "expected_examples": expected,
        "full_eval_examples": example_count == expected,
        "accuracy_ci95": wilson_ci(accuracy_value, example_count),
        "steps": data.get("steps"),
        "task_format": data.get("task_format"),
        "scale_mode": data.get("scale_mode"),
        "attention_kd_weight": loss_weights.get("attention_kd_weight"),
        "last_ce": last.get("ce"),
        "last_weighted_logit_kd": last.get("weighted_logit_kd"),
        "last_weighted_attention_kd": last.get("weighted_attention_kd"),
    }


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    by_task_family: dict[tuple[str, str], dict[str, Any]] = {}
    for task in args.tasks:
        fp_path = run_path(args.baseline_root, args.model, RUN_SPECS[0], task)
        fp = metric_summary(fp_path, task)
        for spec in RUN_SPECS:
            root = getattr(args, spec.root_attr)
            path = run_path(root, args.model, spec, task)
            metrics = metric_summary(path, task)
            gap = None
            gap_ci95 = None
            if fp.get("accuracy") is not None and metrics.get("accuracy") is not None:
                gap = float(fp["accuracy"]) - float(metrics["accuracy"])
                gap_ci95 = difference_ci(fp["accuracy"], fp["examples"], metrics["accuracy"], metrics["examples"])
            row = {
                "task": task,
                "run": spec.label,
                "family": spec.family,
                "path": str(path),
                **metrics,
                "fp16_accuracy": fp.get("accuracy"),
                "fp_minus_run": gap,
                "fp_minus_run_ci95": gap_ci95,
                "passes_fp_gap": bool(metrics["full_eval_examples"] and gap is not None and abs(gap) <= args.max_fp_gap),
            }
            rows.append(row)
            by_task_family[(task, spec.family)] = row

    comparisons: list[dict[str, Any]] = []
    comparison_specs = [
        ("row_warmup_gamma100_minus_tensor_warmup_tensor_gamma100", "row_warmup_gamma100", "tensor_warmup_gamma100"),
        ("row_warmup_gamma100_minus_tensor_warmup_row_gamma100", "row_warmup_gamma100", "tensor_warmup_row_gamma100"),
        ("row_warmup_papergamma_minus_tensor_warmup_tensor_papergamma", "row_warmup_papergamma", "tensor_warmup_papergamma"),
        ("row_warmup_papergamma_minus_tensor_warmup_row_papergamma", "row_warmup_papergamma", "tensor_warmup_row_papergamma"),
    ]
    for task in args.tasks:
        for label, left_family, right_family in comparison_specs:
            left = by_task_family[(task, left_family)]
            right = by_task_family[(task, right_family)]
            delta = None
            delta_ci95 = None
            if left.get("accuracy") is not None and right.get("accuracy") is not None:
                delta = float(left["accuracy"]) - float(right["accuracy"])
                delta_ci95 = difference_ci(left["accuracy"], left["examples"], right["accuracy"], right["examples"])
            comparisons.append(
                {
                    "task": task,
                    "comparison": label,
                    "left_accuracy": left.get("accuracy"),
                    "left_examples": left.get("examples"),
                    "right_accuracy": right.get("accuracy"),
                    "right_examples": right.get("examples"),
                    "delta": delta,
                    "delta_ci95": delta_ci95,
                }
            )

    family_status: dict[str, dict[str, Any]] = {}
    for family in ["row_warmup_gamma100", "row_warmup_papergamma"]:
        family_rows = [row for row in rows if row["family"] == family]
        complete = all(row["exists"] and row["accuracy"] is not None and row["full_eval_examples"] for row in family_rows)
        passed = complete and all(row["passes_fp_gap"] for row in family_rows)
        family_status[family] = {"complete": complete, "passed": passed}

    return {
        "schema": "bitdistill-rowwarmup-gate-v1",
        "date": DATE,
        "model": args.model,
        "tasks": args.tasks,
        "max_fp_gap": args.max_fp_gap,
        "rows": rows,
        "comparisons": comparisons,
        "family_status": family_status,
        "passed": any(item["passed"] for item in family_status.values()),
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
    return f"[{value[0]:.6f}, {value[1]:.6f}]"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def render_markdown(summary: dict[str, Any]) -> str:
    metric_rows = [
        [
            row["task"],
            row["run"],
            row["family"],
            fmt(row["exists"]),
            fmt(row["accuracy"]),
            fmt(row["examples"]),
            fmt(row["expected_examples"]),
            "pass" if row["full_eval_examples"] else "fail_or_pending",
            fmt_ci(row["accuracy_ci95"]),
            fmt(row["fp16_accuracy"]),
            fmt(row["fp_minus_run"]),
            fmt_ci(row["fp_minus_run_ci95"]),
            "pass" if row["passes_fp_gap"] else "fail_or_pending",
            row["path"],
        ]
        for row in summary["rows"]
    ]
    comparison_rows = [
        [
            item["task"],
            item["comparison"],
            fmt(item["left_accuracy"]),
            fmt(item["left_examples"]),
            fmt(item["right_accuracy"]),
            fmt(item["right_examples"]),
            fmt(item["delta"]),
            fmt_ci(item["delta_ci95"]),
        ]
        for item in summary["comparisons"]
    ]
    status_rows = [
        [family, fmt(status["complete"]), fmt(status["passed"])]
        for family, status in summary["family_status"].items()
    ]
    return "\n\n".join(
        [
            f"# BitDistill Row-Warmup Gate, {summary['date']}",
            f"Model: `{summary['model']}`.",
            f"Threshold: absolute FP16-SFT gap <= `{summary['max_fp_gap']}` accuracy.",
            f"Any row-warmup family passed: `{summary['passed']}`.",
            "## Family Status",
            md_table(["family", "complete", "passed"], status_rows),
            "## Runs",
            md_table(
                [
                    "task",
                    "run",
                    "family",
                    "exists",
                    "accuracy",
                    "examples",
                    "expected",
                    "full eval",
                    "accuracy 95% CI",
                    "FP16",
                    "FP-run",
                    "FP-run 95% CI",
                    "status",
                    "metrics path",
                ],
                metric_rows,
            ),
            "## Comparisons",
            md_table(
                [
                    "task",
                    "comparison",
                    "left",
                    "left n",
                    "right",
                    "right n",
                    "left-right",
                    "left-right 95% CI",
                ],
                comparison_rows,
            ),
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--max-fp-gap", type=float, default=0.01)
    parser.add_argument("--baseline-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls"))
    parser.add_argument("--tensor-gamma100-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls-longwarmup"))
    parser.add_argument("--tensor-papergamma-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma"))
    parser.add_argument("--tensor-papergamma-row-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-row"))
    parser.add_argument("--row-warmup-gamma100-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls-rowwarmup-gamma100"))
    parser.add_argument("--row-warmup-papergamma-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls-rowwarmup-papergamma"))
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/bitdistill_rowwarmup_gate_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/bitdistill_rowwarmup_gate_{DATE}.md"))
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
