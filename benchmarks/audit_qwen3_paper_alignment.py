#!/usr/bin/env python3
"""Audit the queued Qwen3-0.6B BitDistill paper-alignment branch.

This report is narrow: it follows the job TSV for the Qwen3 base-model branch
and records which rows have completed full validation, how ternary rows compare
to the FP16 task model for the same task, and whether any row is close enough to
support a paper-reproduction claim. It is safe to run while jobs are pending.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()
EXPECTED_EXAMPLES = {"mnli": 9815, "qnli": 5463, "sst2": 872}
SUCCESS_DELTA_FROM_FP = -0.01
Z_95 = 1.959963984540054


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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
        if value != 0.0 and (abs(value) >= 1000 or abs(value) < 0.0001):
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
        lines.append("| " + " | ".join(fmt(item).replace("|", "\\|") for item in row) + " |")
    return "\n".join(lines)


def latest_tsv(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    return matches[-1] if matches else root / "benchmark_results/bitdistill_glue_jobs_20260515_124507.tsv"


def squeue_state(job_id: str) -> dict[str, str]:
    if not job_id:
        return {"state": "unknown"}
    try:
        result = subprocess.run(
            ["squeue", "-h", "-j", job_id, "-o", "%i\t%T\t%M\t%R"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return {"state": "squeue_unavailable"}
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "Invalid job id" in stderr or "Invalid job id specified" in stderr:
            return {"job_id": job_id, "state": "not_in_squeue"}
        return {"state": "squeue_error", "stderr": stderr}
    line = result.stdout.strip()
    if not line:
        return {"job_id": job_id, "state": "not_in_squeue"}
    parts = line.split("\t")
    return {
        "job_id": parts[0] if len(parts) > 0 else job_id,
        "state": parts[1] if len(parts) > 1 else "unknown",
        "elapsed": parts[2] if len(parts) > 2 else "",
        "reason": parts[3] if len(parts) > 3 else "",
    }


def read_predictions(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.exists():
        return [], [f"missing {path}"]
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
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
        if not isinstance(label, int) or not isinstance(prediction, int):
            errors.append(f"{path}:{lineno}: label/prediction is not int")
            continue
        if bool(correct) != (label == prediction):
            errors.append(f"{path}:{lineno}: correct flag disagrees with label/prediction")
            continue
        rows.append(row)
    rows.sort(key=lambda item: int(item["index"]))
    for expected, row in enumerate(rows):
        if int(row["index"]) != expected:
            errors.append(f"{path}: non-contiguous index at row {expected}, saw {row['index']}")
            break
    return rows, errors


def paired_ci(values: list[float]) -> list[float] | None:
    n = len(values)
    if n <= 1:
        return None
    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    half = Z_95 * math.sqrt(variance / n)
    return [mean - half, mean + half]


def compare_predictions(reference_path: Path, candidate_path: Path, expected: int) -> dict[str, Any]:
    ref_rows, ref_errors = read_predictions(reference_path)
    cand_rows, cand_errors = read_predictions(candidate_path)
    if ref_errors or cand_errors:
        return {"status": "pending", "errors": ref_errors + cand_errors}
    if len(ref_rows) != expected or len(cand_rows) != expected:
        return {
            "status": "pending",
            "errors": [f"expected {expected} rows, got reference={len(ref_rows)}, candidate={len(cand_rows)}"],
        }
    values: list[float] = []
    both_correct = ref_only = cand_only = both_wrong = 0
    for ref, cand in zip(ref_rows, cand_rows):
        if int(ref["index"]) != int(cand["index"]) or int(ref["label"]) != int(cand["label"]):
            return {"status": "fail", "errors": [f"prediction index/label mismatch at {ref.get('index')}"]}
        ref_correct = 1.0 if bool(ref["correct"]) else 0.0
        cand_correct = 1.0 if bool(cand["correct"]) else 0.0
        values.append(cand_correct - ref_correct)
        if ref_correct and cand_correct:
            both_correct += 1
        elif ref_correct and not cand_correct:
            ref_only += 1
        elif cand_correct and not ref_correct:
            cand_only += 1
        else:
            both_wrong += 1
    mean = sum(values) / len(values)
    return {
        "status": "pass",
        "matched": len(values),
        "delta_vs_fp": mean,
        "ci95": paired_ci(values),
        "reference_accuracy": sum(1 for row in ref_rows if row["correct"]) / len(ref_rows),
        "candidate_accuracy": sum(1 for row in cand_rows if row["correct"]) / len(cand_rows),
        "both_correct": both_correct,
        "fp_only": ref_only,
        "candidate_only": cand_only,
        "both_wrong": both_wrong,
    }


def load_jobs(tsv_path: Path) -> list[dict[str, str]]:
    if not tsv_path.exists():
        return []
    with tsv_path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def evidence_label(job: dict[str, str]) -> str:
    """Return the public claim label for this row.

    The Qwen3 queue is a paper-alignment branch, not a completed paper-scale
    reproduction: token budget and remaining task coverage are still pending.
    Row-scale entries are fork-specific runtime/quality variants by definition.
    """
    if job.get("scale") == "row" or job.get("phase") == "novelty_row_scale":
        return "retrofit-variant"
    return "paper-inspired"


def summarize_job(job: dict[str, str], fp_predictions_by_task: dict[str, Path]) -> dict[str, Any]:
    output_dir = Path(job.get("output_dir", ""))
    metrics_path = output_dir / "metrics.json"
    predictions_path = output_dir / "eval_predictions.jsonl"
    metrics = read_json(metrics_path)
    eval_metrics = metrics.get("eval", {}) if isinstance(metrics.get("eval"), dict) else {}
    task = job.get("task", "")
    expected = EXPECTED_EXAMPLES.get(task)
    examples = finite(eval_metrics.get("eval_examples"))
    accuracy = finite(eval_metrics.get("accuracy"))
    complete = bool(
        metrics_path.exists()
        and predictions_path.exists()
        and expected is not None
        and examples == expected
        and accuracy is not None
    )
    paired: dict[str, Any] = {}
    if job.get("method") != "fp16_sft" and task in fp_predictions_by_task and expected is not None:
        paired = compare_predictions(fp_predictions_by_task[task], predictions_path, expected)
    return {
        "phase": job.get("phase"),
        "task": task,
        "method": job.get("method"),
        "scale": job.get("scale"),
        "evidence_label": evidence_label(job),
        "layer": job.get("layer"),
        "job_id": job.get("job_id"),
        "dependency": job.get("dependency"),
        "output_dir": str(output_dir),
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
        "queue": squeue_state(job.get("job_id", "")),
        "metrics_exists": metrics_path.exists(),
        "predictions_exists": predictions_path.exists(),
        "complete": complete,
        "accuracy": accuracy,
        "eval_examples": examples,
        "expected_examples": expected,
        "steps": metrics.get("steps"),
        "last_ce": finite(metrics.get("last", {}).get("ce")) if isinstance(metrics.get("last"), dict) else None,
        "preparation": metrics.get("preparation", {}) if isinstance(metrics.get("preparation"), dict) else {},
        "paired": paired,
        "passes_fp_gap": (
            paired.get("status") == "pass"
            and finite(paired.get("delta_vs_fp")) is not None
            and float(paired["delta_vs_fp"]) >= SUCCESS_DELTA_FROM_FP
        ),
    }


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    tsv_path = args.jobs_tsv if args.jobs_tsv is not None else latest_tsv(args.repo_root, "benchmark_results/bitdistill_glue_jobs_*.tsv")
    jobs = load_jobs(tsv_path)
    fp_predictions_by_task: dict[str, Path] = {}
    for job in jobs:
        if job.get("method") == "fp16_sft" and job.get("task") in EXPECTED_EXAMPLES:
            fp_predictions_by_task[str(job["task"])] = Path(job["output_dir"]) / "eval_predictions.jsonl"
    rows = [summarize_job(job, fp_predictions_by_task) for job in jobs]
    task_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["task"] in EXPECTED_EXAMPLES:
            task_groups[row["task"]].append(row)
    complete_rows = [row for row in rows if row["complete"]]
    gap_pass_rows = [row for row in rows if row["passes_fp_gap"]]
    fp_complete = sorted(row["task"] for row in rows if row["method"] == "fp16_sft" and row["complete"])
    bitnet_complete = sorted(row["task"] for row in rows if row["method"] == "bitnet_sft" and row["complete"])
    tensor_bitdistill_complete = sorted(
        row["task"]
        for row in rows
        if row["method"] == "bitdistill" and row["scale"] == "tensor" and row["phase"] == "paper_baseline" and row["complete"]
    )
    row_bitdistill_complete = sorted(
        row["task"]
        for row in rows
        if row["method"] == "bitdistill" and row["scale"] == "row" and row["complete"]
    )
    scale_comparisons: list[dict[str, Any]] = []
    for task in sorted(EXPECTED_EXAMPLES):
        expected = EXPECTED_EXAMPLES[task]
        tensor_rows = [
            row
            for row in rows
            if row["task"] == task
            and row["method"] == "bitdistill"
            and row["scale"] == "tensor"
            and row["phase"] == "paper_baseline"
            and row["complete"]
        ]
        row_rows = [
            row
            for row in rows
            if row["task"] == task
            and row["method"] == "bitdistill"
            and row["scale"] == "row"
            and row["complete"]
        ]
        if not tensor_rows or not row_rows:
            continue
        tensor_row = tensor_rows[0]
        row_row = row_rows[0]
        paired = compare_predictions(Path(tensor_row["predictions_path"]), Path(row_row["predictions_path"]), expected)
        scale_comparisons.append(
            {
                "task": task,
                "tensor_accuracy": tensor_row["accuracy"],
                "row_accuracy": row_row["accuracy"],
                "paired": {
                    "status": paired.get("status"),
                    "matched": paired.get("matched"),
                    "delta_row_minus_tensor": paired.get("delta_vs_fp"),
                    "ci95": paired.get("ci95"),
                    "tensor_only": paired.get("fp_only"),
                    "row_only": paired.get("candidate_only"),
                    "both_correct": paired.get("both_correct"),
                    "both_wrong": paired.get("both_wrong"),
                    "errors": paired.get("errors", []),
                },
            }
        )
    attention_layer_comparisons: list[dict[str, Any]] = []
    for task in sorted(EXPECTED_EXAMPLES):
        expected = EXPECTED_EXAMPLES[task]
        baseline_rows = [
            row
            for row in rows
            if row["task"] == task
            and row["method"] == "bitdistill"
            and row["scale"] == "tensor"
            and row["phase"] == "paper_baseline"
            and row["layer"] == "-1"
            and row["complete"]
        ]
        if not baseline_rows:
            continue
        baseline = baseline_rows[0]
        sweep_rows = sorted(
            [
                row
                for row in rows
                if row["task"] == task
                and row["method"] == "bitdistill"
                and row["scale"] == "tensor"
                and row["phase"] == "attention_layer_sweep"
                and row["complete"]
            ],
            key=lambda item: str(item["layer"]),
        )
        for sweep in sweep_rows:
            paired = compare_predictions(Path(baseline["predictions_path"]), Path(sweep["predictions_path"]), expected)
            attention_layer_comparisons.append(
                {
                    "task": task,
                    "baseline_layer": baseline["layer"],
                    "candidate_layer": sweep["layer"],
                    "baseline_accuracy": baseline["accuracy"],
                    "candidate_accuracy": sweep["accuracy"],
                    "paired": {
                        "status": paired.get("status"),
                        "matched": paired.get("matched"),
                        "delta_candidate_minus_baseline": paired.get("delta_vs_fp"),
                        "ci95": paired.get("ci95"),
                        "baseline_only": paired.get("fp_only"),
                        "candidate_only": paired.get("candidate_only"),
                        "both_correct": paired.get("both_correct"),
                        "both_wrong": paired.get("both_wrong"),
                        "errors": paired.get("errors", []),
                    },
                }
            )
    return {
        "schema": "qwen3-paper-alignment-audit-v1",
        "date": DATE,
        "jobs_tsv": str(tsv_path),
        "expected_examples": EXPECTED_EXAMPLES,
        "success_delta_from_fp": SUCCESS_DELTA_FROM_FP,
        "rows": rows,
        "job_count": len(rows),
        "complete_count": len(complete_rows),
        "fp_complete_tasks": fp_complete,
        "bitnet_complete_tasks": bitnet_complete,
        "tensor_bitdistill_complete_tasks": tensor_bitdistill_complete,
        "row_bitdistill_complete_tasks": row_bitdistill_complete,
        "scale_comparisons": scale_comparisons,
        "attention_layer_comparisons": attention_layer_comparisons,
        "gap_pass_count": len(gap_pass_rows),
        "paper_reproduction_ready": (
            set(fp_complete) == set(EXPECTED_EXAMPLES)
            and set(bitnet_complete) == set(EXPECTED_EXAMPLES)
            and set(tensor_bitdistill_complete) == set(EXPECTED_EXAMPLES)
            and all(
                row["passes_fp_gap"]
                for row in rows
                if row["method"] == "bitdistill" and row["scale"] == "tensor" and row["phase"] == "paper_baseline"
            )
        ),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    row_table = [
        [
            row["job_id"],
            row["phase"],
            row["task"],
            row["method"],
            row["scale"],
            row["evidence_label"],
            row["layer"],
            row["queue"].get("state"),
            row["complete"],
            row["accuracy"],
            row.get("paired", {}).get("delta_vs_fp"),
            row.get("paired", {}).get("ci95"),
            row["passes_fp_gap"],
            row["output_dir"],
        ]
        for row in summary["rows"]
    ]
    headline = [
        ["jobs", summary["job_count"]],
        ["complete rows", summary["complete_count"]],
        ["FP complete tasks", summary["fp_complete_tasks"]],
        ["BitNet-SFT complete tasks", summary["bitnet_complete_tasks"]],
        ["tensor BitDistill complete tasks", summary["tensor_bitdistill_complete_tasks"]],
        ["row BitDistill complete tasks", summary["row_bitdistill_complete_tasks"]],
        ["gap-pass rows", summary["gap_pass_count"]],
        ["paper reproduction ready", summary["paper_reproduction_ready"]],
    ]
    scale_table = [
        [
            row["task"],
            row["tensor_accuracy"],
            row["row_accuracy"],
            row["paired"].get("delta_row_minus_tensor"),
            row["paired"].get("ci95"),
            row["paired"].get("matched"),
        ]
        for row in summary.get("scale_comparisons", [])
    ]
    attention_table = [
        [
            row["task"],
            row["baseline_layer"],
            row["candidate_layer"],
            row["baseline_accuracy"],
            row["candidate_accuracy"],
            row["paired"].get("delta_candidate_minus_baseline"),
            row["paired"].get("ci95"),
            row["paired"].get("matched"),
        ]
        for row in summary.get("attention_layer_comparisons", [])
    ]
    status = "ready" if summary["paper_reproduction_ready"] else "pending"
    sections = [
        f"# Qwen3 Paper-Alignment Audit, {summary['date']}",
        f"Overall status: **{status}**.",
        (
            "This audit tracks the queued Qwen3-0.6B branch. Rows remain "
            "pending until full validation metrics and prediction traces exist."
        ),
        "## Headline",
        md_table(["field", "value"], headline),
    ]
    if scale_table:
        sections.extend(
            [
                "## Row-Scale Versus Tensor-Scale BitDistill",
                md_table(
                    ["task", "tensor accuracy", "row accuracy", "row minus tensor", "CI95", "matched"],
                    scale_table,
                ),
            ]
        )
    if attention_table:
        sections.extend(
            [
                "## Attention-Layer Sweep",
                md_table(
                    [
                        "task",
                        "baseline layer",
                        "candidate layer",
                        "baseline accuracy",
                        "candidate accuracy",
                        "candidate minus baseline",
                        "CI95",
                        "matched",
                    ],
                    attention_table,
                ),
            ]
        )
    sections.extend(
        [
            "## Rows",
            md_table(
                [
                    "job",
                    "phase",
                    "task",
                    "method",
                    "scale",
                    "label",
                    "layer",
                    "queue",
                    "complete",
                    "accuracy",
                    "delta vs FP",
                    "CI95",
                    "gap pass",
                    "output",
                ],
                row_table,
            ),
        ]
    )
    return "\n\n".join(sections) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--jobs-tsv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/qwen3_paper_alignment_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/qwen3_paper_alignment_{DATE}.md"))
    args = parser.parse_args()

    args.repo_root = args.repo_root.resolve()
    summary = build_summary(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_markdown(summary)
    args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
