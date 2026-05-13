#!/usr/bin/env python3
"""Quantify scale-metadata options needed for TL2 row-scale checkpoints.

Current TL2 stores one scale per tensor. Row-scale QAT checkpoints store one
scale per output row. This audit computes the expected relative output RMS
error introduced by replacing those row scales with tensor or row-group scales.
For isotropic activations this equals the relative Frobenius error of the
effective linear weight matrix.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


GROUP_SIZES = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def parse_label_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    label, raw_path = value.split("=", 1)
    return label, Path(raw_path)


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}" if math.isfinite(value) else "nan"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def weighted_error_sq(nnz_by_row: Any, true_scales: Any, approx_scales: Any) -> float:
    return float((nnz_by_row * (true_scales - approx_scales).square()).sum().item())


def tensor_optimal_scale(nnz_by_row: Any, row_scales: Any) -> float:
    denom = float(nnz_by_row.sum().item())
    if denom <= 0.0:
        return 0.0
    return float((nnz_by_row * row_scales).sum().item() / denom)


def grouped_optimal_scales(nnz_by_row: Any, row_scales: Any, group_size: int) -> Any:
    import torch

    rows = int(row_scales.numel())
    approx = torch.empty_like(row_scales)
    for start in range(0, rows, group_size):
        end = min(start + group_size, rows)
        weights = nnz_by_row[start:end]
        scales = row_scales[start:end]
        denom = float(weights.sum().item())
        value = float((weights * scales).sum().item() / denom) if denom > 0.0 else 0.0
        approx[start:end] = value
    return approx


def add_strategy(
    totals: dict[str, dict[str, float]],
    name: str,
    *,
    true_sq: float,
    error_sq: float,
    scale_count: int,
    tensors: int = 1,
) -> None:
    row = totals[name]
    row["true_sq"] += true_sq
    row["error_sq"] += error_sq
    row["scale_count"] += scale_count
    row["tensors"] += tensors


def audit_state(label: str, path: Path, group_sizes: list[int]) -> dict[str, Any]:
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    top_current: list[dict[str, Any]] = []
    tensor_count = 0
    row_scale_tensors = 0
    scalar_scale_tensors = 0
    total_rows = 0
    total_nnz = 0
    total_values = 0

    for key in sorted(state):
        if not key.endswith(".ternary_weight"):
            continue
        prefix = key[: -len(".ternary_weight")]
        scale_key = prefix + ".weight_scale"
        if scale_key not in state:
            continue

        codes = state[key]
        if codes.ndim != 2:
            continue
        scale = state[scale_key].float().cpu()
        rows = int(codes.shape[0])
        if scale.numel() == 1:
            row_scales = scale.reshape(1).repeat(rows).double()
            scalar_scale_tensors += 1
            scale_mode = "scalar"
        elif scale.numel() == rows:
            row_scales = scale.reshape(rows).double()
            row_scale_tensors += 1
            scale_mode = "row"
        else:
            raise ValueError(f"{scale_key} shape {tuple(scale.shape)} is incompatible with {tuple(codes.shape)}")

        nnz_by_row = codes.ne(0).sum(dim=1).double().cpu()
        true_sq = float((nnz_by_row * row_scales.square()).sum().item())
        if true_sq <= 0.0:
            continue

        tensor_count += 1
        total_rows += rows
        total_nnz += int(nnz_by_row.sum().item())
        total_values += int(codes.numel())

        tensor_max = float(row_scales[nnz_by_row > 0].max().item())
        current_scales = row_scales.new_full((rows,), tensor_max)
        current_error = weighted_error_sq(nnz_by_row, row_scales, current_scales)
        add_strategy(totals, "current_tl2_tensor_max_fp32", true_sq=true_sq, error_sq=current_error, scale_count=1)

        opt = tensor_optimal_scale(nnz_by_row, row_scales)
        opt_scales = row_scales.new_full((rows,), opt)
        add_strategy(
            totals,
            "tensor_l2_optimal_fp32",
            true_sq=true_sq,
            error_sq=weighted_error_sq(nnz_by_row, row_scales, opt_scales),
            scale_count=1,
        )

        row_fp16 = row_scales.to(dtype=torch.float16).to(dtype=torch.float64)
        add_strategy(
            totals,
            "row_exact_fp32",
            true_sq=true_sq,
            error_sq=0.0,
            scale_count=rows,
        )
        add_strategy(
            totals,
            "row_exact_fp16",
            true_sq=true_sq,
            error_sq=weighted_error_sq(nnz_by_row, row_scales, row_fp16),
            scale_count=rows,
        )

        for group_size in group_sizes:
            grouped = grouped_optimal_scales(nnz_by_row, row_scales, group_size)
            scale_count = math.ceil(rows / group_size)
            add_strategy(
                totals,
                f"group{group_size}_l2_optimal_fp32",
                true_sq=true_sq,
                error_sq=weighted_error_sq(nnz_by_row, row_scales, grouped),
                scale_count=scale_count,
            )
            grouped_fp16 = grouped.to(dtype=torch.float16).to(dtype=torch.float64)
            add_strategy(
                totals,
                f"group{group_size}_l2_optimal_fp16",
                true_sq=true_sq,
                error_sq=weighted_error_sq(nnz_by_row, row_scales, grouped_fp16),
                scale_count=scale_count,
            )

        top_current.append(
            {
                "tensor": prefix,
                "shape": list(codes.shape),
                "scale_mode": scale_mode,
                "current_tl2_rel_error": math.sqrt(current_error / true_sq),
                "tensor_l2_optimal_rel_error": math.sqrt(
                    weighted_error_sq(nnz_by_row, row_scales, opt_scales) / true_sq
                ),
                "row_scale_min": float(row_scales[nnz_by_row > 0].min().item()),
                "row_scale_median": float(row_scales[nnz_by_row > 0].median().item()),
                "row_scale_max": tensor_max,
                "nonzero_fraction": int(nnz_by_row.sum().item()) / int(codes.numel()),
            }
        )

    strategies = []
    for name, values in sorted(totals.items()):
        true_sq = float(values["true_sq"])
        error_sq = float(values["error_sq"])
        scale_count = int(values["scale_count"])
        strategies.append(
            {
                "name": name,
                "expected_relative_output_rms_error": math.sqrt(error_sq / true_sq) if true_sq > 0.0 else float("nan"),
                "relative_frobenius_error": math.sqrt(error_sq / true_sq) if true_sq > 0.0 else float("nan"),
                "scale_count": scale_count,
                "scale_bytes_fp32": scale_count * 4,
                "scale_bytes_fp16": scale_count * 2,
                "scale_mib_fp32": scale_count * 4 / 1024 / 1024,
                "scale_mib_fp16": scale_count * 2 / 1024 / 1024,
                "tensors": int(values["tensors"]),
            }
        )

    return {
        "label": label,
        "path": str(path),
        "tensors": tensor_count,
        "scalar_scale_tensors": scalar_scale_tensors,
        "row_scale_tensors": row_scale_tensors,
        "total_rows": total_rows,
        "nonzero_fraction": total_nnz / total_values if total_values else 0.0,
        "strategies": strategies,
        "top_current_tl2_errors": sorted(top_current, key=lambda row: row["current_tl2_rel_error"], reverse=True)[:10],
    }


def strategy_map(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["name"]: row for row in result["strategies"]}


def build_report(results: list[dict[str, Any]]) -> str:
    sections = [
        "# TL2 Row-Scale Design Audit, 2026-05-13",
        (
            "This audit measures whether the current one-scale TL2 metadata can preserve row-scale "
            "ternary checkpoints, and how much metadata a row/group-scale TL2 variant would need. "
            "For isotropic activations, the reported relative output RMS error equals the relative "
            "Frobenius error of the effective weight matrix."
        ),
    ]

    summary_rows: list[list[str]] = []
    for result in results:
        strategies = strategy_map(result)
        current = strategies["current_tl2_tensor_max_fp32"]
        optimal = strategies["tensor_l2_optimal_fp32"]
        row16 = strategies["row_exact_fp16"]
        group2 = strategies.get("group2_l2_optimal_fp16")
        group32 = strategies.get("group32_l2_optimal_fp16")
        group128 = strategies.get("group128_l2_optimal_fp16")
        summary_rows.append(
            [
                result["label"],
                str(result["tensors"]),
                str(result["row_scale_tensors"]),
                fmt(float(current["expected_relative_output_rms_error"])),
                fmt(float(optimal["expected_relative_output_rms_error"])),
                fmt(float(group2["expected_relative_output_rms_error"])) if group2 else "n/a",
                fmt(float(group32["expected_relative_output_rms_error"])) if group32 else "n/a",
                fmt(float(group128["expected_relative_output_rms_error"])) if group128 else "n/a",
                fmt(float(row16["expected_relative_output_rms_error"])),
                f"{float(row16['scale_mib_fp16']):.3f}",
            ]
        )
    sections.append(
        md_table(
            [
                "checkpoint",
                "tensors",
                "row-scale tensors",
                "current TL2 err",
                "best one-scale err",
                "group2 fp16 err",
                "group32 fp16 err",
                "group128 fp16 err",
                "row fp16 err",
                "row fp16 scale MiB",
            ],
            summary_rows,
        )
    )

    for result in results:
        strategies = strategy_map(result)
        rows = []
        selected = [
            "current_tl2_tensor_max_fp32",
            "tensor_l2_optimal_fp32",
            "group2_l2_optimal_fp16",
            "group4_l2_optimal_fp16",
            "group8_l2_optimal_fp16",
            "group16_l2_optimal_fp16",
            "group32_l2_optimal_fp16",
            "group64_l2_optimal_fp16",
            "group128_l2_optimal_fp16",
            "row_exact_fp16",
            "row_exact_fp32",
        ]
        for name in selected:
            if name not in strategies:
                continue
            item = strategies[name]
            rows.append(
                [
                    name,
                    fmt(float(item["expected_relative_output_rms_error"])),
                    str(item["scale_count"]),
                    f"{float(item['scale_mib_fp16']):.3f}",
                    f"{float(item['scale_mib_fp32']):.3f}",
                ]
            )
        sections.extend([f"## {result['label']} Strategy Detail", md_table(["strategy", "rel output RMS err", "scales", "fp16 MiB", "fp32 MiB"], rows)])

        top_rows = [
            [
                row["tensor"],
                "x".join(str(dim) for dim in row["shape"]),
                row["scale_mode"],
                fmt(float(row["current_tl2_rel_error"])),
                fmt(float(row["tensor_l2_optimal_rel_error"])),
                fmt(float(row["row_scale_max"] / row["row_scale_median"]), 3) if row["row_scale_median"] else "inf",
            ]
            for row in result["top_current_tl2_errors"][:5]
        ]
        sections.append(
            md_table(
                ["tensor", "shape", "scale mode", "current TL2 err", "best one-scale err", "max/median scale"],
                top_rows,
            )
        )

    sections.extend(
        [
            "## Interpretation",
            (
                "A one-scale TL2 representation is mathematically exact for tensor-scale checkpoints, "
                "but not for row-scale checkpoints. For the strong row-scale Qwen checkpoint, the fix "
                "is not more benchmarking of the existing TL2 format; the format/runtime must carry "
                "row or row-group scale metadata and the generated TL2 kernels must index those scales. "
                "The scale-memory cost of exact fp16 row scales is small relative to the model file, "
                "so the blocker is runtime/kernel support rather than storage."
            ),
        ]
    )
    return "\n\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", action="append", required=True, help="LABEL=/path/to/ternary_state_dict.pt")
    parser.add_argument("--group-size", action="append", type=int, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    group_sizes = args.group_size or GROUP_SIZES
    results = [audit_state(label, path, group_sizes) for label, path in (parse_label_path(item) for item in args.state)]
    report = build_report(results)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"results": results}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
