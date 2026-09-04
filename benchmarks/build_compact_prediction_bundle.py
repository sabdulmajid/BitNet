#!/usr/bin/env python3
"""Build a compact, public bundle from aligned classification prediction traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_trace_spec(value: str) -> tuple[str, Path]:
    model_id, separator, raw_path = value.partition("=")
    if not separator or not model_id or not raw_path:
        raise argparse.ArgumentTypeError("trace must use MODEL_ID=PATH")
    if not model_id.replace("_", "").isalnum():
        raise argparse.ArgumentTypeError(f"invalid model ID: {model_id!r}")
    return model_id, Path(raw_path)


def read_trace(path: Path) -> tuple[list[int], list[int]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    labels: list[int] = []
    predictions: list[int] = []
    seen: set[int] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected an object")
        index = row.get("index")
        label = row.get("label")
        prediction = row.get("prediction")
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in (index, label, prediction)):
            raise ValueError(f"{path}:{line_number}: index, label, and prediction must be integers")
        if index in seen:
            raise ValueError(f"{path}:{line_number}: duplicate index {index}")
        if index != len(labels):
            raise ValueError(f"{path}:{line_number}: expected contiguous index {len(labels)}, saw {index}")
        if row.get("correct") is not (label == prediction):
            raise ValueError(f"{path}:{line_number}: incorrect correctness flag")
        seen.add(index)
        labels.append(label)
        predictions.append(prediction)
    return labels, predictions


def build_bundle(trace_specs: list[tuple[str, Path]], expected_examples: int) -> dict[str, Any]:
    if len(trace_specs) < 2:
        raise ValueError("at least two traces are required")
    if len({model_id for model_id, _ in trace_specs}) != len(trace_specs):
        raise ValueError("trace model IDs must be unique")

    labels: list[int] | None = None
    models: dict[str, Any] = {}
    predictions: dict[str, list[int]] = {}
    for model_id, path in trace_specs:
        candidate_labels, candidate_predictions = read_trace(path)
        if len(candidate_labels) != expected_examples:
            raise ValueError(f"{model_id}: rows={len(candidate_labels)}, expected={expected_examples}")
        if labels is None:
            labels = candidate_labels
        elif candidate_labels != labels:
            raise ValueError(f"{model_id}: labels do not align with the first trace")
        models[model_id] = {
            "accuracy": sum(
                prediction == label
                for prediction, label in zip(candidate_predictions, candidate_labels, strict=True)
            )
            / expected_examples,
            "source_bytes": path.stat().st_size,
            "source_sha256": sha256(path),
        }
        predictions[model_id] = candidate_predictions

    assert labels is not None
    return {
        "schema": "compact-classification-predictions-v1",
        "examples": expected_examples,
        "labels": labels,
        "models": models,
        "predictions": predictions,
    }


def render_markdown(bundle: dict[str, Any]) -> str:
    rows = [
        f"| {model_id} | {metadata['accuracy']:.6f} | `{metadata['source_sha256']}` |"
        for model_id, metadata in bundle["models"].items()
    ]
    return "\n".join(
        [
            "# Compact Prediction Evidence",
            "",
            f"This bundle preserves all labels and class predictions for `{bundle['examples']:,}` aligned examples.",
            "It omits logits, prompts, and private paths. Accuracies and paired tests are exactly reconstructible.",
            "",
            "| model | accuracy | source trace SHA-256 |",
            "| --- | ---: | --- |",
            *rows,
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="append", type=parse_trace_spec, required=True)
    parser.add_argument("--expected-examples", type=int, default=9815)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmarks/results/bitdistill_adaptive_prediction_bundle_2026-09-04.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("benchmarks/results/bitdistill_adaptive_prediction_bundle_2026-09-04.md"),
    )
    args = parser.parse_args()
    bundle = build_bundle(args.trace, args.expected_examples)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(bundle, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(render_markdown(bundle), encoding="utf-8")


if __name__ == "__main__":
    main()
