#!/usr/bin/env python3
"""Compare selected metrics across EleutherAI lm-eval JSON outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_METRICS = {
    "arc_challenge": "acc_norm",
    "arc_easy": "acc_norm",
    "hellaswag": "acc_norm",
    "piqa": "acc_norm",
    "winogrande": "acc",
    "boolq": "acc",
    "copa": "acc",
    "openbookqa": "acc_norm",
    "sciq": "acc_norm",
    "truthfulqa_mc1": "acc",
}


def parse_run(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        label, path = spec.split("=", 1)
        label = label.strip()
    else:
        path = spec
        label = Path(path).stem
    if not label:
        raise ValueError(f"empty run label in {spec!r}")
    return label, Path(path)


def parse_metric(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"--metric expects task=metric, got {spec!r}")
    task, metric = spec.split("=", 1)
    task = task.strip()
    metric = metric.strip()
    if not task or not metric:
        raise ValueError(f"invalid metric spec {spec!r}")
    return task, metric


def load_results(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results")
    if not isinstance(results, dict):
        raise ValueError(f"{path} does not look like an lm-eval result JSON")
    return results


def metric_value(task_results: dict[str, Any], metric: str) -> float:
    candidates = [metric, f"{metric},none"]
    for key in candidates:
        if key in task_results:
            return float(task_results[key])
    raise KeyError(f"metric {metric!r} not found; available keys: {sorted(task_results)}")


def build_table(runs: list[tuple[str, dict[str, Any]]], metrics: dict[str, str]) -> str:
    headers = ["task", "metric", *[label for label, _ in runs]]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---", "---", *["---:" for _ in runs]]) + " |",
    ]
    means = {label: [] for label, _ in runs}

    for task, metric in metrics.items():
        row = [task, metric]
        for label, results in runs:
            if task not in results:
                row.append("-")
                continue
            value = metric_value(results[task], metric)
            means[label].append(value)
            row.append(f"{value:.6f}")
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", "| method | mean |", "| --- | ---: |"])
    for label, values in means.items():
        if values:
            lines.append(f"| {label} | {sum(values) / len(values):.6f} |")
        else:
            lines.append(f"| {label} | - |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="LABEL=path.json, repeatable")
    parser.add_argument(
        "--metric",
        action="append",
        default=[],
        help="task=metric, repeatable; defaults to common selected metrics",
    )
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    run_specs = [parse_run(spec) for spec in args.run]
    runs = [(label, load_results(path)) for label, path in run_specs]
    metrics = dict(DEFAULT_METRICS)
    if args.metric:
        metrics = dict(parse_metric(spec) for spec in args.metric)

    table = build_table(runs, metrics)
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
