#!/usr/bin/env python3
"""Audit whether separator batching preserves sidecar classifier predictions."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()


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
        return f"{value:.6f}"
    return str(value).replace("|", "\\|")


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(fmt(item) for item in row) + " |" for row in rows)
    return "\n".join(lines)


def row_summary(label: str, data: dict[str, Any]) -> dict[str, Any]:
    summary = data.get("summary", {}) if isinstance(data.get("summary"), dict) else {}
    runtime = data.get("runtime", {}) if isinstance(data.get("runtime"), dict) else {}
    return {
        "label": label,
        "status": data.get("status"),
        "batch_size": data.get("batch_size"),
        "examples": summary.get("examples"),
        "accuracy": summary.get("accuracy"),
        "agreement_with_saved_pytorch_predictions": summary.get("agreement_with_saved_pytorch_predictions"),
        "disagreements_with_saved_pytorch_predictions": summary.get("disagreements_with_saved_pytorch_predictions"),
        "examples_per_second": runtime.get("examples_per_second"),
        "prompt_eval_tokens_per_second_aggregate": runtime.get("prompt_eval_tokens_per_second_aggregate"),
        "prediction_sha256": data.get("prediction_sha256"),
    }


def compare_predictions(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    reference_predictions = reference.get("predictions")
    candidate_predictions = candidate.get("predictions")
    if not isinstance(reference_predictions, list) or not isinstance(candidate_predictions, list):
        return {
            "comparable": False,
            "agreement": None,
            "mismatch_count": None,
            "first_mismatch_indices": [],
            "reason": "one or both inputs are missing full prediction vectors",
        }
    if len(reference_predictions) != len(candidate_predictions):
        return {
            "comparable": False,
            "agreement": None,
            "mismatch_count": None,
            "first_mismatch_indices": [],
            "reason": f"prediction length mismatch: {len(reference_predictions)} vs {len(candidate_predictions)}",
        }
    mismatches = [
        idx
        for idx, (ref, cand) in enumerate(zip(reference_predictions, candidate_predictions, strict=True))
        if ref != cand
    ]
    total = len(reference_predictions)
    return {
        "comparable": True,
        "agreement": (total - len(mismatches)) / total if total else None,
        "mismatch_count": len(mismatches),
        "first_mismatch_indices": mismatches[:20],
        "reason": "",
    }


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    reference = read_json(args.batch1_json)
    candidate = read_json(args.batch4_json)
    comparison = compare_predictions(reference, candidate)
    ref_runtime = reference.get("runtime", {}) if isinstance(reference.get("runtime"), dict) else {}
    cand_runtime = candidate.get("runtime", {}) if isinstance(candidate.get("runtime"), dict) else {}
    ref_eps = finite(ref_runtime.get("examples_per_second"))
    cand_eps = finite(cand_runtime.get("examples_per_second"))
    ref_tps = finite(ref_runtime.get("prompt_eval_tokens_per_second_aggregate"))
    cand_tps = finite(cand_runtime.get("prompt_eval_tokens_per_second_aggregate"))
    speedup = cand_eps / ref_eps if ref_eps and cand_eps else None
    token_speedup = cand_tps / ref_tps if ref_tps and cand_tps else None
    status = (
        "batching_semantics_preserved"
        if comparison.get("comparable") and comparison.get("mismatch_count") == 0
        else "batching_semantics_drift"
    )
    return {
        "schema": "seqcls-i2sr-sidecar-batching-audit-v1",
        "date": DATE,
        "status": status,
        "reference": row_summary("batch1", reference),
        "candidate": row_summary("batch4", candidate),
        "comparison": comparison,
        "speedup_examples_per_second": speedup,
        "speedup_prompt_tokens_per_second": token_speedup,
        "interpretation": (
            "Separator batching improves throughput but changes at least one classifier prediction. "
            "Treat batch size 1 as the semantic reference until native sequence-classification "
            "runtime support is implemented or batching parity is proven."
            if status == "batching_semantics_drift"
            else "Separator batching preserved predictions for this sample set."
        ),
    }


def render_markdown(result: dict[str, Any]) -> str:
    rows = [result["reference"], result["candidate"]]
    return "\n\n".join(
        [
            f"# Sequence-Classification I2_SR Sidecar Batching Audit, {result['date']}",
            "This audit checks whether `llama-embedding --embd-separator` batching preserves the sidecar classifier predictions.",
            md_table(
                ["field", "value"],
                [
                    ["status", result["status"]],
                    ["batch4 vs batch1 prediction agreement", result["comparison"]["agreement"]],
                    ["batch4 vs batch1 mismatches", result["comparison"]["mismatch_count"]],
                    ["first mismatch indices", result["comparison"]["first_mismatch_indices"]],
                    ["examples/sec speedup", result["speedup_examples_per_second"]],
                    ["prompt tokens/sec speedup", result["speedup_prompt_tokens_per_second"]],
                ],
            ),
            md_table(
                [
                    "row",
                    "batch",
                    "status",
                    "examples",
                    "accuracy",
                    "agreement vs PyTorch",
                    "examples/sec",
                    "prediction sha256",
                ],
                [
                    [
                        row["label"],
                        row["batch_size"],
                        row["status"],
                        row["examples"],
                        row["accuracy"],
                        row["agreement_with_saved_pytorch_predictions"],
                        row["examples_per_second"],
                        str(row["prediction_sha256"])[:12],
                    ]
                    for row in rows
                ],
            ),
            "## Interpretation",
            result["interpretation"],
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch1-json",
        type=Path,
        default=Path(f"benchmark_results/seqcls_i2sr_sidecar_cpu_mnli_64_{DATE}.json"),
    )
    parser.add_argument(
        "--batch4-json",
        type=Path,
        default=Path(f"benchmark_results/seqcls_i2sr_sidecar_cpu_mnli64_batch4_{DATE}.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(f"benchmark_results/seqcls_i2sr_sidecar_batching_{DATE}.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(f"benchmarks/results/seqcls_i2sr_sidecar_batching_{DATE}.md"),
    )
    args = parser.parse_args()

    result = build_result(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({"status": result["status"], "comparison": result["comparison"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
