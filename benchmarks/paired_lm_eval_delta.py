#!/usr/bin/env python3
"""Compute paired lm-eval metric deltas from logged samples.

This script compares two lm-eval JSON files that were produced with
``log_samples=True``.  It matches examples by ``doc_hash``/``prompt_hash`` and
reports the paired mean difference for selected binary metrics such as
``acc`` and ``acc_norm``.  Paired deltas are more informative than comparing
two independent standard errors because every method sees the same examples.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
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


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data.get("samples")
    if not isinstance(samples, dict):
        raise ValueError(f"{path} has no lm-eval samples; rerun with log_samples=True")
    return data


def sample_key(sample: dict[str, Any]) -> tuple[str, str, str]:
    doc_id = str(sample.get("doc_id", ""))
    doc_hash = str(sample.get("doc_hash", ""))
    prompt_hash = str(sample.get("prompt_hash", ""))
    if doc_hash or prompt_hash:
        return doc_id, doc_hash, prompt_hash
    target_hash = str(sample.get("target_hash", ""))
    return doc_id, target_hash, str(sample.get("target", ""))


def metric_from_sample(sample: dict[str, Any], metric: str) -> float:
    value = sample.get(metric)
    if value is None and metric.endswith(",none"):
        value = sample.get(metric.removesuffix(",none"))
    if value is None:
        metrics = sample.get("metrics")
        raise KeyError(f"metric {metric!r} not found in sample; sample metrics={metrics!r}")
    return float(value)


def task_values(data: dict[str, Any], task: str, metric: str) -> dict[tuple[str, str, str], float]:
    samples = data.get("samples", {}).get(task)
    if not isinstance(samples, list):
        raise KeyError(f"task {task!r} not found in samples")
    values: dict[tuple[str, str, str], float] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        key = sample_key(sample)
        if key in values:
            raise ValueError(f"duplicate sample key for task {task}: {key}")
        values[key] = metric_from_sample(sample, metric)
    return values


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def ci95(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    center = mean(values)
    if len(values) == 1:
        return center, center, center
    stderr = statistics.stdev(values) / math.sqrt(len(values))
    width = 1.96 * stderr
    return center, center - width, center + width


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, help="LABEL=path.json")
    parser.add_argument("--b", required=True, help="LABEL=path.json")
    parser.add_argument(
        "--metric",
        action="append",
        default=[],
        help="task=metric, repeatable; defaults to common selected metrics",
    )
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    label_a, path_a = parse_run(args.a)
    label_b, path_b = parse_run(args.b)
    data_a = load_json(path_a)
    data_b = load_json(path_b)
    metrics = dict(DEFAULT_METRICS)
    if args.metric:
        metrics = dict(parse_metric(spec) for spec in args.metric)

    rows: list[list[str]] = []
    task_deltas: list[float] = []
    task_weights: list[int] = []
    for task, metric in metrics.items():
        try:
            values_a = task_values(data_a, task, metric)
            values_b = task_values(data_b, task, metric)
        except KeyError:
            continue
        common = sorted(set(values_a) & set(values_b))
        if not common:
            raise ValueError(f"no matched samples for task {task}")
        missing_a = len(values_b) - len(common)
        missing_b = len(values_a) - len(common)
        a = [values_a[key] for key in common]
        b = [values_b[key] for key in common]
        deltas = [right - left for left, right in zip(a, b)]
        delta, low, high = ci95(deltas)
        task_deltas.append(delta)
        task_weights.append(len(common))
        rows.append(
            [
                task,
                metric,
                str(len(common)),
                f"{mean(a):.6f}",
                f"{mean(b):.6f}",
                f"{delta:+.6f}",
                f"[{low:+.6f}, {high:+.6f}]",
                str(missing_a),
                str(missing_b),
            ]
        )

    if task_deltas:
        macro_delta, macro_low, macro_high = ci95(task_deltas)
        weighted_delta = sum(delta * n for delta, n in zip(task_deltas, task_weights)) / sum(task_weights)
    else:
        macro_delta = macro_low = macro_high = weighted_delta = float("nan")

    report = "\n\n".join(
        [
            f"# Paired lm-eval Delta: {label_b} - {label_a}",
            md_table(
                [
                    "task",
                    "metric",
                    "matched n",
                    label_a,
                    label_b,
                    "delta",
                    "paired 95% CI",
                    f"missing from {label_a}",
                    f"missing from {label_b}",
                ],
                rows,
            ),
            md_table(
                ["summary", "delta"],
                [
                    ["macro mean delta", f"{macro_delta:+.6f} [{macro_low:+.6f}, {macro_high:+.6f}]"],
                    ["example-weighted delta", f"{weighted_delta:+.6f}"],
                ],
            ),
        ]
    )

    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
