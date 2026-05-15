#!/usr/bin/env python3
"""Summarize the focused BitNet-SFT MNLI LR/step-budget sweep."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()
FULL_MNLI_VALIDATION = 9815
PAPER_BITNET_SFT_MNLI = 0.6080


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


def finite_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if abs(value) >= 1000 or (abs(value) < 1e-3 and value != 0):
            return f"{value:.3e}"
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


def safe_value(value: str) -> str:
    return value.replace("-", "m").replace("+", "").replace(".", "p")


def read_job_tables(pattern: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                row = dict(row)
                row["job_table"] = path
                rows.append(row)
    return rows


def expected_runs_from_args_and_jobs(args: argparse.Namespace, job_rows: list[dict[str, str]]) -> list[tuple[int, str]]:
    expected = {(steps, lr) for steps in args.steps for lr in args.lrs}
    model_slug = args.model.replace("/", "-")
    root_prefix = str(args.output_root / model_slug / args.task_name)
    for row in job_rows:
        if row.get("task") != args.task_name:
            continue
        if row.get("method") != "bitnet_sft":
            continue
        if row.get("scale") != args.scale_mode:
            continue
        output_dir = row.get("output_dir", "")
        if output_dir and not output_dir.startswith(root_prefix):
            continue
        try:
            steps = int(row.get("steps", ""))
        except ValueError:
            continue
        lr = row.get("lr", "")
        if not lr:
            continue
        expected.add((steps, lr))
    return sorted(expected, key=lambda item: (item[0], float(item[1])))


def summarize_run(root: Path, *, steps: int, lr: str, job_rows: list[dict[str, str]]) -> dict[str, Any]:
    metrics_path = root / "metrics.json"
    metrics = read_json(metrics_path)
    accuracy = finite_float(nested(metrics, "eval", "accuracy"))
    examples = finite_float(nested(metrics, "eval", "eval_examples"))
    matching_jobs = [row for row in job_rows if row.get("output_dir") == str(root)]
    job = matching_jobs[-1] if matching_jobs else {}
    return {
        "root": str(root),
        "metrics_path": str(metrics_path),
        "exists": metrics_path.exists(),
        "job_id": job.get("job_id", ""),
        "job_table": job.get("job_table", ""),
        "steps": steps,
        "lr": lr,
        "method": metrics.get("method"),
        "scale_mode": metrics.get("scale_mode"),
        "accuracy": accuracy,
        "eval_examples": examples,
        "full_eval": int(examples or 0) == FULL_MNLI_VALIDATION,
        "paper_anchor": PAPER_BITNET_SFT_MNLI,
        "paper_anchor_minus_local": PAPER_BITNET_SFT_MNLI - accuracy if accuracy is not None else None,
        "default_baseline_delta": None,
        "last_loss": finite_float(nested(metrics, "last", "loss")),
        "last_ce": finite_float(nested(metrics, "last", "ce")),
        "preparation": metrics.get("preparation", {}),
        "training_budget": metrics.get("training_budget", {}),
    }


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    model_slug = args.model.replace("/", "-")
    job_rows = read_job_tables(args.job_table_glob)
    rows = []
    for steps, lr in expected_runs_from_args_and_jobs(args, job_rows):
        root = args.output_root / model_slug / args.task_name / f"bitnet_sft-{args.scale_mode}-steps{steps}-lr{safe_value(lr)}"
        rows.append(summarize_run(root, steps=steps, lr=lr, job_rows=job_rows))

    baseline_metrics = read_json(args.default_baseline_root / "metrics.json")
    baseline_accuracy = finite_float(nested(baseline_metrics, "eval", "accuracy"))
    for row in rows:
        if baseline_accuracy is not None and row["accuracy"] is not None:
            row["default_baseline_delta"] = row["accuracy"] - baseline_accuracy

    completed = [row for row in rows if row["exists"] and row["accuracy"] is not None]
    best = max(completed, key=lambda row: row["accuracy"]) if completed else None
    return {
        "date": DATE,
        "model": args.model,
        "task": args.task_name,
        "output_root": str(args.output_root),
        "default_baseline_root": str(args.default_baseline_root),
        "default_baseline_accuracy": baseline_accuracy,
        "paper_anchor": PAPER_BITNET_SFT_MNLI,
        "runs": rows,
        "complete": len(completed),
        "expected": len(rows),
        "best": best,
        "job_tables": sorted({row.get("job_table", "") for row in job_rows if row.get("job_table")}),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    best = summary.get("best")
    if best:
        verdict = (
            f"Best completed sweep row is accuracy `{fmt(best['accuracy'])}` at "
            f"steps=`{best['steps']}`, lr=`{best['lr']}`. The paper BitNet-SFT "
            f"anchor remains `{fmt(summary['paper_anchor'])}`, so the remaining "
            f"gap is `{fmt(best['paper_anchor_minus_local'])}`."
        )
    else:
        verdict = "No completed sweep rows are available yet; this report currently tracks submitted jobs and output paths."

    rows = []
    for row in summary["runs"]:
        prep = row.get("preparation", {}) if isinstance(row.get("preparation"), dict) else {}
        rows.append(
            [
                row["steps"],
                row["lr"],
                row["exists"],
                row.get("job_id", ""),
                row.get("accuracy"),
                row.get("eval_examples"),
                row.get("full_eval"),
                row.get("default_baseline_delta"),
                row.get("paper_anchor_minus_local"),
                row.get("last_ce"),
                prep.get("activation_quantization"),
                prep.get("subln_inserted"),
            ]
        )
    lines = [
        f"# BitNet-SFT Budget Sweep Audit, {summary['date']}",
        verdict,
        f"Completed rows: `{summary['complete']}/{summary['expected']}`.",
        f"Default BitNet-SFT MNLI baseline: `{fmt(summary['default_baseline_accuracy'])}`.",
        "## Runs",
        md_table(
            [
                "steps",
                "lr",
                "metrics",
                "job",
                "accuracy",
                "examples",
                "full eval",
                "delta vs default",
                "paper anchor - local",
                "last CE",
                "A8",
                "SubLN",
            ],
            rows,
        ),
    ]
    if summary.get("job_tables"):
        lines.extend(["## Job Tables", "\n".join(f"- `{path}`" for path in summary["job_tables"])])
    return "\n\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--task-name", default="mnli")
    parser.add_argument("--scale-mode", default="tensor")
    parser.add_argument("--output-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls-bitnet-sft-budget"))
    parser.add_argument("--default-baseline-root", type=Path, default=Path("checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/bitnet_sft-tensor-layer-1"))
    parser.add_argument("--steps", nargs="+", type=int, default=[1000, 3000])
    parser.add_argument("--lrs", nargs="+", default=["5e-6", "1e-5", "2e-5", "5e-5"])
    parser.add_argument("--job-table-glob", default="benchmark_results/bitnet_sft_budget_sweep_*.tsv")
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/bitnet_sft_budget_sweep_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/bitnet_sft_budget_sweep_{DATE}.md"))
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
