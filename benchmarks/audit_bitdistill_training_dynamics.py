#!/usr/bin/env python3
"""Summarize materialized BitDistill training telemetry.

This audit consumes telemetry.jsonl files emitted by train_bitdistill.py.  It is
deliberately separate from the source-coverage audit: source coverage proves the
hooks exist, while this script proves that a concrete run actually emitted
update-balance, activation, ternary flip-rate, and scale-drift evidence.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


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
        return ", ".join(fmt(item) for item in value)
    return str(value)


def nested(row: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = row
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def ratio(numerator: Any, denominator: Any) -> float | None:
    if finite(numerator) and finite(denominator) and abs(float(denominator)) > 0.0:
        return float(numerator) / float(denominator)
    return None


def finite_values(rows: list[dict[str, Any]], *keys: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = nested(row, *keys)
        if finite(value):
            values.append(float(value))
    return values


def classify_path(path: Path) -> str:
    text = str(path)
    if "bitdistill-smoke-contract" in text or text.startswith("/tmp/"):
        return "smoke"
    if "checkpoints/" in text or text.startswith("checkpoints/"):
        return "controlled"
    return "other"


def summarize_trace(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    kind = classify_path(path)
    steps = [int(row["step"]) for row in rows if isinstance(row.get("step"), int)]
    grad_rows = [
        row
        for row in rows
        if isinstance(row.get("component_grad_norms_microbatch"), dict)
        and bool(row.get("component_grad_norms_microbatch"))
        and finite(nested(row, "component_grad_norms_microbatch", "ce"))
    ]
    activation_rows = [row for row in rows if isinstance(row.get("activation_quantization"), dict)]
    dynamics_rows = [row for row in rows if isinstance(row.get("quantization_dynamics"), dict)]
    dynamics_compared = [
        row
        for row in dynamics_rows
        if nested(row, "quantization_dynamics", "has_previous") is True
        and finite(nested(row, "quantization_dynamics", "flip_fraction"))
    ]

    final = rows[-1] if rows else {}
    final_loss = final.get("loss", {}) if isinstance(final.get("loss"), dict) else {}
    final_grad = final.get("component_grad_norms_microbatch", {}) if isinstance(final.get("component_grad_norms_microbatch"), dict) else {}

    grad_attention_to_ce = [
        value
        for row in grad_rows
        if (value := ratio(
            nested(row, "component_grad_norms_microbatch", "weighted_attention_kd"),
            nested(row, "component_grad_norms_microbatch", "ce"),
        ))
        is not None
    ]
    grad_logit_to_ce = [
        value
        for row in grad_rows
        if (value := ratio(
            nested(row, "component_grad_norms_microbatch", "weighted_logit_kd"),
            nested(row, "component_grad_norms_microbatch", "ce"),
        ))
        is not None
    ]
    loss_attention_to_ce = [
        value
        for row in rows
        if (value := ratio(nested(row, "loss", "weighted_attention_kd"), nested(row, "loss", "ce"))) is not None
    ]
    loss_logit_to_ce = [
        value
        for row in rows
        if (value := ratio(nested(row, "loss", "weighted_logit_kd"), nested(row, "loss", "ce"))) is not None
    ]

    clipped = finite_values(rows, "activation_quantization", "clipped_fraction")
    edge = finite_values(rows, "activation_quantization", "int8_edge_fraction")
    flips = finite_values(dynamics_compared, "quantization_dynamics", "flip_fraction")
    scale_delta_mean = finite_values(dynamics_compared, "quantization_dynamics", "scale_abs_delta_mean")
    scale_delta_max = finite_values(dynamics_compared, "quantization_dynamics", "scale_abs_delta_max")

    qkv_grad = {
        key: final_grad.get(key)
        for key in ("weighted_attention_q_kd", "weighted_attention_k_kd", "weighted_attention_v_kd")
        if finite(final_grad.get(key))
    }
    qkv_loss = {
        key: final_loss.get(key)
        for key in ("weighted_attention_q_kd", "weighted_attention_k_kd", "weighted_attention_v_kd")
        if finite(final_loss.get(key))
    }

    return {
        "path": str(path),
        "kind": kind,
        "rows": len(rows),
        "first_step": min(steps) if steps else None,
        "last_step": max(steps) if steps else None,
        "has_component_grad_norms": bool(grad_rows),
        "has_activation": bool(activation_rows),
        "has_dynamics": bool(dynamics_compared),
        "final_loss_attention_to_ce": loss_attention_to_ce[-1] if loss_attention_to_ce else None,
        "max_loss_attention_to_ce": max(loss_attention_to_ce) if loss_attention_to_ce else None,
        "final_loss_logit_to_ce": loss_logit_to_ce[-1] if loss_logit_to_ce else None,
        "final_grad_attention_to_ce": grad_attention_to_ce[-1] if grad_attention_to_ce else None,
        "max_grad_attention_to_ce": max(grad_attention_to_ce) if grad_attention_to_ce else None,
        "final_grad_logit_to_ce": grad_logit_to_ce[-1] if grad_logit_to_ce else None,
        "max_activation_clipped_fraction": max(clipped) if clipped else None,
        "max_activation_edge_fraction": max(edge) if edge else None,
        "mean_flip_fraction": mean(flips) if flips else None,
        "max_flip_fraction": max(flips) if flips else None,
        "mean_scale_abs_delta_mean": mean(scale_delta_mean) if scale_delta_mean else None,
        "max_scale_abs_delta_max": max(scale_delta_max) if scale_delta_max else None,
        "final_qkv_grad_norms": qkv_grad,
        "final_qkv_weighted_losses": qkv_loss,
    }


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(cell).replace("|", "\\|") for cell in row) + " |")
    return "\n".join(lines)


def expand_inputs(paths: list[str], globs: list[str]) -> list[Path]:
    seen: set[str] = set()
    expanded: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.exists() and str(path) not in seen:
            expanded.append(path)
            seen.add(str(path))
    for pattern in globs:
        for match in sorted(glob.glob(pattern, recursive=True)):
            path = Path(match)
            if path.exists() and str(path) not in seen:
                expanded.append(path)
                seen.add(str(path))
    return expanded


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    paths = expand_inputs(args.telemetry_path, args.telemetry_glob)
    traces = [summarize_trace(path) for path in paths]
    non_smoke = [trace for trace in traces if trace["kind"] != "smoke"]
    controlled = [trace for trace in traces if trace["kind"] == "controlled"]
    materialized = [
        trace
        for trace in controlled
        if trace["has_component_grad_norms"] and trace["has_activation"] and trace["has_dynamics"]
    ]
    smoke_materialized = [
        trace
        for trace in traces
        if trace["kind"] == "smoke"
        and trace["has_component_grad_norms"]
        and trace["has_activation"]
        and trace["has_dynamics"]
    ]
    if materialized:
        status = "controlled_materialized"
    elif smoke_materialized:
        status = "smoke_only"
    elif traces:
        status = "partial_or_legacy_traces"
    else:
        status = "no_traces"

    return {
        "schema": "bitdistill-training-dynamics-audit-v1",
        "date": DATE,
        "status": status,
        "trace_count": len(traces),
        "controlled_trace_count": len(controlled),
        "non_smoke_trace_count": len(non_smoke),
        "materialized_controlled_count": len(materialized),
        "smoke_materialized_count": len(smoke_materialized),
        "traces": traces,
        "verdict": (
            "Controlled training-dynamics telemetry is materialized."
            if materialized
            else (
                "Only smoke telemetry is materialized; this validates the parser and hooks, "
                "but not a real controlled BitDistill run."
                if smoke_materialized
                else "No complete controlled training-dynamics telemetry has been materialized yet."
            )
        ),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    rows = [
        [
            Path(trace["path"]).name,
            trace["kind"],
            trace["rows"],
            trace["first_step"],
            trace["last_step"],
            trace["has_component_grad_norms"],
            trace["has_activation"],
            trace["has_dynamics"],
            trace["final_grad_attention_to_ce"],
            trace["max_activation_clipped_fraction"],
            trace["max_activation_edge_fraction"],
            trace["mean_flip_fraction"],
            trace["max_scale_abs_delta_max"],
        ]
        for trace in summary["traces"]
    ]
    return (
        f"# BitDistill Training Dynamics Audit, {summary['date']}\n\n"
        f"Overall status: **{summary['status']}**.\n\n"
        f"{summary['verdict']}\n\n"
        f"Traces: `{summary['trace_count']}`. Controlled traces: "
        f"`{summary['controlled_trace_count']}`. Materialized controlled traces: "
        f"`{summary['materialized_controlled_count']}`.\n\n"
        + md_table(
            [
                "trace",
                "kind",
                "rows",
                "first",
                "last",
                "grad",
                "A8",
                "dyn",
                "final grad attn/CE",
                "max clipped",
                "max edge",
                "mean flip",
                "max scale delta",
            ],
            rows,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry-path", action="append", default=[])
    parser.add_argument(
        "--telemetry-glob",
        action="append",
        default=[
            f"benchmark_results/bitdistill-smoke-contract-{DATE}/task_sft/telemetry.jsonl",
            f"benchmark_results/bitdistill-smoke-contract-{DATE}/task_sft_row/telemetry.jsonl",
            "checkpoints/bitdistill-glue-seqcls-telemetry/**/telemetry.jsonl",
        ],
    )
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/bitdistill_training_dynamics_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/bitdistill_training_dynamics_{DATE}.md"))
    args = parser.parse_args()

    summary = build_summary(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(summary), encoding="utf-8")
    print(render_markdown(summary))


if __name__ == "__main__":
    main()
