#!/usr/bin/env python3
"""Audit whether TL2 group/tile scales are enough for row-scale checkpoints.

The TL2 runtime blocker could be solved either by exact per-row scales or by a
cheaper row-group/tile-scale compromise. This audit promotes the existing TL2
design measurements into an explicit decision gate so we do not implement a
half-measure kernel unless the math says it is viable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()
GROUP_RE = re.compile(r"^group(?P<size>[0-9]+)_l2_optimal_(?P<dtype>fp16|fp32)$")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value != 0.0 and (abs(value) >= 1000 or abs(value) < 0.0001):
            return f"{value:.6e}"
        return f"{value:.6f}"
    return str(value).replace("|", "\\|")


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(item) for item in row) + " |")
    return "\n".join(lines)


def finite_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def strategy_map(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = result.get("strategies", [])
    if not isinstance(rows, list):
        return {}
    return {str(row.get("name")): row for row in rows if isinstance(row, dict)}


def find_result(data: dict[str, Any], label: str) -> dict[str, Any]:
    rows = data.get("results", [])
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if isinstance(row, dict) and row.get("label") == label:
            return row
    return {}


def group_rows(result: dict[str, Any], dtype: str = "fp16") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, strategy in strategy_map(result).items():
        match = GROUP_RE.match(name)
        if not match or match.group("dtype") != dtype:
            continue
        error = finite_float(strategy.get("expected_relative_output_rms_error"))
        scale_mib = finite_float(strategy.get("scale_mib_fp16") or strategy.get("scale_mib_fp32"))
        if error is None or scale_mib is None:
            continue
        rows.append(
            {
                "name": name,
                "group_size": int(match.group("size")),
                "dtype": dtype,
                "expected_relative_output_rms_error": error,
                "scale_mib": scale_mib,
            }
        )
    return sorted(rows, key=lambda row: row["group_size"])


def named_strategy(result: dict[str, Any], name: str) -> dict[str, Any]:
    strategy = strategy_map(result).get(name, {})
    return {
        "name": name,
        "expected_relative_output_rms_error": finite_float(strategy.get("expected_relative_output_rms_error")),
        "scale_mib": finite_float(strategy.get("scale_mib_fp16") or strategy.get("scale_mib_fp32")),
    }


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    design = read_json(args.design_json)
    row_scale = find_result(design, args.row_label)
    tensor_scale = find_result(design, args.tensor_label)
    groups = group_rows(row_scale, dtype="fp16")
    best_group = min(groups, key=lambda row: row["expected_relative_output_rms_error"]) if groups else None
    smallest_strict_group = next(
        (row for row in groups if row["expected_relative_output_rms_error"] <= args.strict_error_threshold),
        None,
    )
    smallest_loose_group = next(
        (row for row in groups if row["expected_relative_output_rms_error"] <= args.loose_error_threshold),
        None,
    )
    current = named_strategy(row_scale, "current_tl2_tensor_max_fp32")
    tensor_l2 = named_strategy(row_scale, "tensor_l2_optimal_fp32")
    exact_row = named_strategy(row_scale, "row_exact_fp16")
    tensor_exact_row = named_strategy(tensor_scale, "row_exact_fp16")
    tensor_current = named_strategy(tensor_scale, "current_tl2_tensor_max_fp32")
    strict_viable = smallest_strict_group is not None
    loose_viable = smallest_loose_group is not None
    best_group_error = best_group["expected_relative_output_rms_error"] if best_group else None
    exact_row_error = exact_row["expected_relative_output_rms_error"]
    return {
        "schema": "tl2-group-scale-viability-v1",
        "date": DATE,
        "design_json": str(args.design_json),
        "row_label": args.row_label,
        "tensor_label": args.tensor_label,
        "strict_error_threshold": args.strict_error_threshold,
        "loose_error_threshold": args.loose_error_threshold,
        "current_one_scale": current,
        "best_one_tensor_l2_scale": tensor_l2,
        "best_group_fp16": best_group,
        "smallest_group_meeting_strict_threshold": smallest_strict_group,
        "smallest_group_meeting_loose_threshold": smallest_loose_group,
        "exact_row_fp16": exact_row,
        "tensor_scale_current": tensor_current,
        "tensor_scale_exact_row_fp16": tensor_exact_row,
        "group_fp16_rows": groups,
        "strict_group_scale_viable": strict_viable,
        "loose_group_scale_viable": loose_viable,
        "exact_row_required_for_strict_fidelity": not strict_viable
        and exact_row_error is not None
        and exact_row_error <= args.strict_error_threshold,
        "best_group_to_exact_row_error_ratio": (
            best_group_error / exact_row_error
            if best_group_error is not None and exact_row_error not in (None, 0.0)
            else None
        ),
        "recommendation": (
            "Do not implement group/tile-scale TL2 as the quality-preserving row-scale path; "
            "the best available fp16 group-scale row still misses the strict output-fidelity gate."
            if not strict_viable
            else "A group-scale TL2 variant may be viable under the strict output-fidelity gate."
        ),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    group_rows_md = [
        [
            row["group_size"],
            row["expected_relative_output_rms_error"],
            row["scale_mib"],
        ]
        for row in summary["group_fp16_rows"]
    ]
    decision_rows = [
        ["strict threshold", summary["strict_error_threshold"]],
        ["loose threshold", summary["loose_error_threshold"]],
        ["current one-scale TL2 error", summary["current_one_scale"]["expected_relative_output_rms_error"]],
        ["best tensor L2 one-scale error", summary["best_one_tensor_l2_scale"]["expected_relative_output_rms_error"]],
        ["best fp16 group-scale error", summary["best_group_fp16"]["expected_relative_output_rms_error"]],
        ["best fp16 group-scale MiB", summary["best_group_fp16"]["scale_mib"]],
        ["exact row fp16 error", summary["exact_row_fp16"]["expected_relative_output_rms_error"]],
        ["exact row fp16 MiB", summary["exact_row_fp16"]["scale_mib"]],
        ["strict group-scale viable", summary["strict_group_scale_viable"]],
        ["loose group-scale viable", summary["loose_group_scale_viable"]],
        ["exact row required for strict fidelity", summary["exact_row_required_for_strict_fidelity"]],
        ["best group / exact row error ratio", summary["best_group_to_exact_row_error_ratio"]],
    ]
    tensor_rows = [
        ["tensor-scale checkpoint current TL2 error", summary["tensor_scale_current"]["expected_relative_output_rms_error"]],
        ["tensor-scale checkpoint row-fp16 error", summary["tensor_scale_exact_row_fp16"]["expected_relative_output_rms_error"]],
    ]
    return "\n\n".join(
        [
            f"# TL2 Group-Scale Viability Audit, {summary['date']}",
            summary["recommendation"],
            "## Decision Summary",
            md_table(["field", "value"], decision_rows),
            "## Row-Scale Checkpoint Group Sweep",
            md_table(["group size", "relative output RMS error", "scale MiB"], group_rows_md),
            "## Tensor-Scale Control",
            md_table(["field", "value"], tensor_rows),
            "## Interpretation",
            (
                "The grouped-scale rows answer whether TL2 can use one scale per output-row group instead of one "
                "scale per row. For the audited Qwen2.5-1.5B row-scale checkpoint, even the best available fp16 "
                "group-scale setting misses the strict `0.01` relative-output-RMS fidelity gate by roughly two "
                "orders of magnitude compared with exact row fp16 scales. Group-scale TL2 may be useful as a "
                "future speed/quality experiment, but it should not close the current row-scale TL2 objective "
                "blocker. The quality-preserving path is exact row-scale metadata or a different proven scale "
                "model."
            ),
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-json", type=Path, default=Path("benchmark_results/tl2_row_scale_design_2026-05-13.json"))
    parser.add_argument("--row-label", default="qwen15b_row_scale")
    parser.add_argument("--tensor-label", default="qwen15b_tensor_scale")
    parser.add_argument("--strict-error-threshold", type=float, default=0.01)
    parser.add_argument("--loose-error-threshold", type=float, default=0.10)
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/tl2_group_scale_viability_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/tl2_group_scale_viability_{DATE}.md"))
    args = parser.parse_args()

    summary = build_summary(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(summary), encoding="utf-8")
    print(render_markdown(summary))


if __name__ == "__main__":
    main()
