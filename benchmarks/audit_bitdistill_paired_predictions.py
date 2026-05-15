#!/usr/bin/env python3
"""Audit paired BitDistill GLUE predictions when per-example traces exist."""

from __future__ import annotations

import os
import argparse
import json
import math
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
class RunRef:
    label: str
    root_attr: str
    template: str


@dataclass(frozen=True)
class ComparisonSpec:
    label: str
    task: str
    reference: RunRef
    candidate: RunRef
    family: str


FP16 = RunRef("FP16-SFT", "baseline_prediction_root", "{task}/fp16_sft-tensor-layer-1")
BITNET = RunRef("BitNet-SFT", "baseline_prediction_root", "{task}/bitnet_sft-tensor-layer-1")
LONG_TENSOR = RunRef("BitDistill gamma100 tensor", "longwarmup_root", "{task}/bitdistill-longwarmup-tensor-layer-8")
LONG_ROW = RunRef("BitDistill gamma100 row", "longwarmup_root", "{task}/bitdistill-longwarmup-row-layer-8")
PAPER_TENSOR = RunRef("BitDistill paper-gamma tensor", "paper_hparam_root", "{task}/bitdistill-longwarmup-tensor-layer-8")
PAPER_ROW = RunRef("BitDistill paper-gamma row", "paper_hparam_row_root", "{task}/bitdistill-longwarmup-row-layer-8")
PAPER_LR1 = RunRef("BitDistill paper-gamma tensor lr1e-5", "paper_hparam_lr1_root", "{task}/bitdistill-longwarmup-tensor-layer-8")
PAPER_LR5 = RunRef("BitDistill paper-gamma tensor lr5e-5", "paper_hparam_lr5_root", "{task}/bitdistill-longwarmup-tensor-layer-8")
PAPER_HEADINIT = RunRef("BitDistill paper-gamma tensor headinit", "paper_hparam_headinit_root", "{task}/bitdistill-longwarmup-tensor-layer-8")


def run_dir(root: Path, model: str, ref: RunRef, task: str) -> Path:
    return root / model.replace("/", "-") / ref.template.format(task=task)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_predictions(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    if not path.exists():
        return [], [f"missing {path}"]
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path}:{lineno}: invalid json: {exc}")
            continue
        index = row.get("index")
        label = row.get("label")
        prediction = row.get("prediction")
        correct = row.get("correct")
        if not isinstance(index, int):
            errors.append(f"{path}:{lineno}: index is not int")
            continue
        if index in seen:
            errors.append(f"{path}:{lineno}: duplicate index {index}")
            continue
        seen.add(index)
        if not isinstance(label, int):
            errors.append(f"{path}:{lineno}: label is not int")
            continue
        if not isinstance(prediction, int):
            errors.append(f"{path}:{lineno}: prediction is not int")
            continue
        if bool(correct) != (prediction == label):
            errors.append(f"{path}:{lineno}: correct flag disagrees with label/prediction")
            continue
        rows.append(row)
    rows.sort(key=lambda row: int(row["index"]))
    for expected, row in enumerate(rows):
        if int(row["index"]) != expected:
            errors.append(f"{path}: non-contiguous index at row {expected}, saw {row['index']}")
            break
    return rows, errors


def metric_summary(run_directory: Path) -> dict[str, Any]:
    metrics = read_json(run_directory / "metrics.json")
    eval_metrics = metrics.get("eval", {}) if isinstance(metrics.get("eval"), dict) else {}
    value = eval_metrics.get("accuracy")
    examples = eval_metrics.get("eval_examples")
    example_count = int(examples) if isinstance(examples, (int, float)) and examples > 0 else None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        accuracy = float(value)
    else:
        accuracy = None
    return {"accuracy": accuracy, "examples": example_count}


def paired_ci(differences: list[float], z: float = Z_95) -> list[float] | None:
    n = len(differences)
    if n <= 1:
        return None
    mean = sum(differences) / n
    variance = sum((value - mean) ** 2 for value in differences) / (n - 1)
    half = z * math.sqrt(variance / n)
    return [mean - half, mean + half]


def logsumexp(values: list[float]) -> float:
    if not values:
        return float("-inf")
    peak = max(values)
    if not math.isfinite(peak):
        return peak
    return peak + math.log(sum(math.exp(value - peak) for value in values))


def binomial_tail(n: int, p: float, *, lower_k: int | None = None, upper_k: int | None = None) -> float:
    if lower_k is None and upper_k is None:
        raise ValueError("one tail bound is required")
    if n <= 0:
        return 1.0
    start = 0 if lower_k is not None else int(upper_k)
    end = int(lower_k) if lower_k is not None else n
    if start > end:
        return 0.0
    logp = math.log(p)
    logq = math.log1p(-p)
    terms = [
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * logp + (n - k) * logq
        for k in range(start, end + 1)
    ]
    return min(1.0, math.exp(logsumexp(terms)))


def exact_mcnemar_pvalue(candidate_wins: int, reference_wins: int) -> float:
    discordant = candidate_wins + reference_wins
    if discordant == 0:
        return 1.0
    low = min(candidate_wins, reference_wins)
    high = max(candidate_wins, reference_wins)
    p_lower = binomial_tail(discordant, 0.5, lower_k=low)
    p_upper = binomial_tail(discordant, 0.5, upper_k=high)
    return min(1.0, 2.0 * min(p_lower, p_upper))


def compare_predictions(args: argparse.Namespace, spec: ComparisonSpec) -> dict[str, Any]:
    expected_examples = EXPECTED_EVAL_EXAMPLES[spec.task]
    reference_dir = run_dir(getattr(args, spec.reference.root_attr), args.model, spec.reference, spec.task)
    candidate_dir = run_dir(getattr(args, spec.candidate.root_attr), args.model, spec.candidate, spec.task)
    reference_path = reference_dir / "eval_predictions.jsonl"
    candidate_path = candidate_dir / "eval_predictions.jsonl"
    reference_rows, reference_errors = read_predictions(reference_path)
    candidate_rows, candidate_errors = read_predictions(candidate_path)
    errors = reference_errors + candidate_errors
    missing = [str(path) for path in (reference_path, candidate_path) if not path.exists()]
    status = "pending" if missing else "pass"

    if not missing and len(reference_rows) != len(candidate_rows):
        errors.append(f"prediction row count mismatch: reference={len(reference_rows)}, candidate={len(candidate_rows)}")
    if not missing and len(reference_rows) != expected_examples:
        errors.append(f"reference prediction rows={len(reference_rows)} expected={expected_examples}")
    if not missing and len(candidate_rows) != expected_examples:
        errors.append(f"candidate prediction rows={len(candidate_rows)} expected={expected_examples}")
    if errors and not missing:
        status = "fail"

    matched = 0
    reference_correct = 0
    candidate_correct = 0
    candidate_wins = 0
    reference_wins = 0
    both_correct = 0
    both_wrong = 0
    differences: list[float] = []

    if not errors and not missing:
        for reference_row, candidate_row in zip(reference_rows, candidate_rows):
            if int(reference_row["index"]) != int(candidate_row["index"]):
                errors.append(f"index mismatch: reference={reference_row['index']}, candidate={candidate_row['index']}")
                status = "fail"
                break
            if int(reference_row["label"]) != int(candidate_row["label"]):
                errors.append(f"label mismatch at index {reference_row['index']}")
                status = "fail"
                break
            ref_ok = bool(reference_row["correct"])
            cand_ok = bool(candidate_row["correct"])
            matched += 1
            reference_correct += int(ref_ok)
            candidate_correct += int(cand_ok)
            both_correct += int(ref_ok and cand_ok)
            both_wrong += int((not ref_ok) and (not cand_ok))
            candidate_wins += int(cand_ok and not ref_ok)
            reference_wins += int(ref_ok and not cand_ok)
            differences.append(float(cand_ok) - float(ref_ok))

    reference_accuracy = reference_correct / matched if matched else None
    candidate_accuracy = candidate_correct / matched if matched else None
    delta = (candidate_accuracy - reference_accuracy) if reference_accuracy is not None and candidate_accuracy is not None else None
    reference_metrics = metric_summary(reference_dir)
    candidate_metrics = metric_summary(candidate_dir)
    reference_metric_accuracy = reference_metrics["accuracy"]
    candidate_metric_accuracy = candidate_metrics["accuracy"]
    reference_metric_examples = reference_metrics["examples"]
    candidate_metric_examples = candidate_metrics["examples"]
    if not missing:
        if reference_metric_accuracy is None:
            errors.append("missing reference metric accuracy")
        elif reference_accuracy is not None and abs(reference_accuracy - reference_metric_accuracy) > 1e-12:
            errors.append(f"reference prediction accuracy {reference_accuracy} disagrees with metrics {reference_metric_accuracy}")
        if candidate_metric_accuracy is None:
            errors.append("missing candidate metric accuracy")
        elif candidate_accuracy is not None and abs(candidate_accuracy - candidate_metric_accuracy) > 1e-12:
            errors.append(f"candidate prediction accuracy {candidate_accuracy} disagrees with metrics {candidate_metric_accuracy}")
        if reference_metric_examples != expected_examples:
            errors.append(f"reference metric eval_examples={reference_metric_examples} expected={expected_examples}")
        if candidate_metric_examples != expected_examples:
            errors.append(f"candidate metric eval_examples={candidate_metric_examples} expected={expected_examples}")
    if errors and not missing:
        status = "fail"

    return {
        "label": spec.label,
        "family": spec.family,
        "task": spec.task,
        "status": status,
        "missing": missing,
        "errors": errors,
        "reference_label": spec.reference.label,
        "candidate_label": spec.candidate.label,
        "reference_dir": str(reference_dir),
        "candidate_dir": str(candidate_dir),
        "reference_predictions": str(reference_path),
        "candidate_predictions": str(candidate_path),
        "expected_examples": expected_examples,
        "matched": matched,
        "reference_accuracy": reference_accuracy,
        "candidate_accuracy": candidate_accuracy,
        "candidate_minus_reference": delta,
        "paired_ci95": paired_ci(differences),
        "candidate_wins": candidate_wins,
        "reference_wins": reference_wins,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "mcnemar_exact_p": exact_mcnemar_pvalue(candidate_wins, reference_wins) if matched else None,
        "reference_metric_accuracy": reference_metric_accuracy,
        "candidate_metric_accuracy": candidate_metric_accuracy,
        "reference_metric_examples": reference_metric_examples,
        "candidate_metric_examples": candidate_metric_examples,
    }


def comparison_specs(tasks: list[str]) -> list[ComparisonSpec]:
    specs: list[ComparisonSpec] = []
    for task in tasks:
        specs.extend(
            [
                ComparisonSpec("BitNet-SFT minus FP16-SFT", task, FP16, BITNET, "baseline_vs_fp"),
                ComparisonSpec("gamma100 row minus tensor", task, LONG_TENSOR, LONG_ROW, "row_vs_tensor"),
                ComparisonSpec("paper-gamma row minus tensor", task, PAPER_TENSOR, PAPER_ROW, "row_vs_tensor"),
                ComparisonSpec("gamma100 tensor minus FP16-SFT", task, FP16, LONG_TENSOR, "bitdistill_vs_fp"),
                ComparisonSpec("gamma100 row minus FP16-SFT", task, FP16, LONG_ROW, "bitdistill_vs_fp"),
                ComparisonSpec("paper-gamma tensor minus FP16-SFT", task, FP16, PAPER_TENSOR, "bitdistill_vs_fp"),
                ComparisonSpec("paper-gamma row minus FP16-SFT", task, FP16, PAPER_ROW, "bitdistill_vs_fp"),
                ComparisonSpec("paper-gamma lr1e-5 tensor minus FP16-SFT", task, FP16, PAPER_LR1, "bitdistill_vs_fp"),
                ComparisonSpec("paper-gamma lr5e-5 tensor minus FP16-SFT", task, FP16, PAPER_LR5, "bitdistill_vs_fp"),
                ComparisonSpec("paper-gamma headinit tensor minus FP16-SFT", task, FP16, PAPER_HEADINIT, "bitdistill_vs_fp"),
                ComparisonSpec("paper-gamma lr1e-5 minus default", task, PAPER_TENSOR, PAPER_LR1, "paper_search"),
                ComparisonSpec("paper-gamma lr5e-5 minus default", task, PAPER_TENSOR, PAPER_LR5, "paper_search"),
                ComparisonSpec("paper-gamma headinit minus default", task, PAPER_TENSOR, PAPER_HEADINIT, "paper_search"),
            ]
        )
    return specs


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    rows = [compare_predictions(args, spec) for spec in comparison_specs(args.tasks)]
    failed = [row for row in rows if row["status"] == "fail"]
    complete = [row for row in rows if row["status"] == "pass"]
    pending = [row for row in rows if row["status"] == "pending"]
    status = "fail" if failed else ("pass" if len(complete) == len(rows) else "pending")
    return {
        "schema": "bitdistill-paired-prediction-audit-v1",
        "date": DATE,
        "model": args.model,
        "tasks": args.tasks,
        "status": status,
        "complete": len(complete),
        "pending": len(pending),
        "failed": len(failed),
        "total": len(rows),
        "rows": rows,
        "notes": [
            "Delta is candidate accuracy minus reference accuracy on matched eval examples.",
            "Rows pass only when both prediction traces cover the full expected task validation split.",
            "The paired 95% interval is a normal interval over per-example paired correctness differences.",
            "McNemar p-values use an exact two-sided binomial test over discordant pairs.",
        ],
        "expected_eval_examples": {task: EXPECTED_EVAL_EXAMPLES[task] for task in args.tasks},
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
    return f"[{float(value[0]):.6f}, {float(value[1]):.6f}]"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def render_markdown(summary: dict[str, Any]) -> str:
    rows = [
        [
            row["task"],
            row["label"],
            row["status"],
            fmt(row["matched"]),
            fmt(row["expected_examples"]),
            fmt(row["reference_accuracy"]),
            fmt(row["candidate_accuracy"]),
            fmt(row["candidate_minus_reference"]),
            fmt_ci(row.get("paired_ci95")),
            fmt(row["candidate_wins"]),
            fmt(row["reference_wins"]),
            fmt(row.get("mcnemar_exact_p")),
            "; ".join(row["missing"][:2]) if row["missing"] else "; ".join(row["errors"][:2]),
        ]
        for row in summary["rows"]
    ]
    return "\n\n".join(
        [
            f"# BitDistill Paired Prediction Audit, {summary['date']}",
            f"Overall status: `{summary['status']}`.",
            f"Rows complete: `{summary['complete']}` / `{summary['total']}`. Pending: `{summary['pending']}`. Failed: `{summary['failed']}`.",
            "This report is designed to become the main statistical comparison once long-warmup jobs write `eval_predictions.jsonl`.",
            f"Full-evaluation contract: `{summary['expected_eval_examples']}` examples. Partial prediction traces cannot pass.",
            "Delta is candidate minus reference on the same eval indices; positive means the candidate is better.",
            md_table(
                [
                    "task",
                    "comparison",
                    "status",
                    "matched n",
                    "expected n",
                    "reference acc",
                    "candidate acc",
                    "delta",
                    "paired 95% CI",
                    "candidate wins",
                    "reference wins",
                    "McNemar p",
                    "pending/error",
                ],
                rows,
            ),
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls"))
    parser.add_argument("--baseline-prediction-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls-predtrace"))
    parser.add_argument("--longwarmup-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls-longwarmup"))
    parser.add_argument("--paper-hparam-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma"))
    parser.add_argument("--paper-hparam-row-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-row"))
    parser.add_argument("--paper-hparam-lr1-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-lr1e-5"))
    parser.add_argument("--paper-hparam-lr5-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-lr5e-5"))
    parser.add_argument("--paper-hparam-headinit-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-headinit"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/bitdistill_paired_predictions_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/bitdistill_paired_predictions_{DATE}.md"))
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
