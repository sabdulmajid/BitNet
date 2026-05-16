#!/usr/bin/env python3
"""Summarize native I2_SR full-validation progress JSONL."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()
DEFAULT_PROGRESS = Path(f"benchmark_results/seqcls_native_i2sr_cpu_mnli_full_token_ids_{DATE}.progress.jsonl")


def read_progress(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def accuracy_ci_wilson(correct: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    phat = correct / total
    denom = 1.0 + z * z / total
    center = (phat + z * z / (2.0 * total)) / denom
    half = z * ((phat * (1.0 - phat) / total + z * z / (4.0 * total * total)) ** 0.5) / denom
    return [center - half, center + half]


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, list):
        return "[" + ", ".join(fmt(item) for item in value) + "]"
    return str(value).replace("|", "\\|")


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(cell) for cell in row) + " |")
    return "\n".join(lines)


def summarize(rows: list[dict[str, Any]], expected_examples: int, progress_path: Path) -> dict[str, Any]:
    contiguous = True
    valid_rows: list[dict[str, Any]] = []
    for expected_index, row in enumerate(rows):
        if int(row.get("index", -1)) != expected_index:
            contiguous = False
            break
        if not isinstance(row.get("prediction"), int) or not isinstance(row.get("label"), int):
            contiguous = False
            break
        valid_rows.append(row)

    completed = len(valid_rows)
    correct = sum(int(int(row["prediction"]) == int(row["label"])) for row in valid_rows)
    saved = [
        row
        for row in valid_rows
        if isinstance(row.get("saved_pytorch_prediction"), int)
    ]
    saved_agree = sum(
        int(int(row["prediction"]) == int(row["saved_pytorch_prediction"]))
        for row in saved
    )
    elapsed_values = [
        float(row["elapsed_seconds"])
        for row in valid_rows
        if isinstance(row.get("elapsed_seconds"), (int, float)) and math.isfinite(float(row["elapsed_seconds"]))
    ]
    elapsed = max(elapsed_values) if elapsed_values else None
    examples_per_second = completed / elapsed if elapsed and elapsed > 0 else None
    remaining = max(0, expected_examples - completed)
    eta_seconds = remaining / examples_per_second if examples_per_second and examples_per_second > 0 else None
    status = "complete" if completed == expected_examples and contiguous else ("partial" if completed else "missing")

    return {
        "schema": "seqcls_native_full_progress.v1",
        "date": DATE,
        "status": status,
        "progress_path": str(progress_path),
        "expected_examples": expected_examples,
        "completed_examples": completed,
        "remaining_examples": remaining,
        "contiguous_prefix": contiguous,
        "last_index": int(valid_rows[-1]["index"]) if valid_rows else None,
        "accuracy_so_far": correct / completed if completed else None,
        "accuracy_ci95_wilson": accuracy_ci_wilson(correct, completed),
        "correct_so_far": correct,
        "saved_pytorch_prediction_rows": len(saved),
        "agreement_with_saved_pytorch_predictions_so_far": saved_agree / len(saved) if saved else None,
        "saved_pytorch_prediction_agreements": saved_agree,
        "elapsed_seconds": elapsed,
        "examples_per_second": examples_per_second,
        "eta_seconds": eta_seconds,
        "is_product_evidence": False,
        "interpretation": (
            "Full validation progress is complete; ingest the final benchmark JSON for product claims."
            if status == "complete"
            else "This is partial progress evidence only. It is useful for monitoring/resume, not for final quality claims."
        ),
    }


def render_markdown(result: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            f"# Sequence-Classification Native Full CPU Progress, {result['date']}",
            "This report summarizes the resumable per-example progress trace for the native single-artifact MNLI CPU run.",
            md_table(
                ["field", "value"],
                [
                    ["status", result["status"]],
                    ["completed examples", result["completed_examples"]],
                    ["expected examples", result["expected_examples"]],
                    ["remaining examples", result["remaining_examples"]],
                    ["contiguous prefix", result["contiguous_prefix"]],
                    ["accuracy so far", result["accuracy_so_far"]],
                    ["accuracy CI95 so far", result["accuracy_ci95_wilson"]],
                    ["agreement with saved PyTorch predictions so far", result["agreement_with_saved_pytorch_predictions_so_far"]],
                    ["examples/sec", result["examples_per_second"]],
                    ["elapsed seconds", result["elapsed_seconds"]],
                    ["ETA seconds", result["eta_seconds"]],
                    ["product evidence", result["is_product_evidence"]],
                ],
            ),
            "## Interpretation",
            result["interpretation"],
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--progress-jsonl", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--expected-examples", type=int, default=9815)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(f"benchmark_results/seqcls_native_full_progress_{DATE}.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(f"benchmarks/results/seqcls_native_full_progress_{DATE}.md"),
    )
    args = parser.parse_args()

    root = args.repo_root.resolve()
    progress_path = args.progress_jsonl if args.progress_jsonl.is_absolute() else root / args.progress_jsonl
    rows = read_progress(progress_path)
    result = summarize(rows, args.expected_examples, progress_path)

    output_json = args.output_json if args.output_json.is_absolute() else root / args.output_json
    output_md = args.output_md if args.output_md.is_absolute() else root / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({"status": result["status"], "completed": result["completed_examples"]}, sort_keys=True))


if __name__ == "__main__":
    main()
