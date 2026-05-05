#!/usr/bin/env python3
"""Audit the error caused by replacing row-wise ternary scales with one TL2 scale."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_label_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    label, raw_path = value.split("=", 1)
    return label, Path(raw_path)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}" if math.isfinite(value) else "nan"


def audit_state(label: str, path: Path) -> dict[str, Any]:
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    tensor_rows: list[dict[str, Any]] = []
    total_true_sq = 0.0
    total_error_sq = 0.0
    total_nnz = 0
    total_values = 0
    row_scale_tensors = 0
    scalar_scale_tensors = 0

    for key in sorted(state):
        if not key.endswith(".ternary_weight"):
            continue
        prefix = key[: -len(".ternary_weight")]
        scale_key = prefix + ".weight_scale"
        if scale_key not in state:
            continue
        codes = state[key]
        scale = state[scale_key].float().cpu()
        if codes.ndim != 2:
            continue
        out_features = int(codes.shape[0])
        if scale.numel() == 1:
            row_scales = scale.reshape(1).repeat(out_features)
            scale_mode = "scalar"
            scalar_scale_tensors += 1
        elif scale.numel() == out_features:
            row_scales = scale.reshape(out_features)
            scale_mode = "row"
            row_scale_tensors += 1
        else:
            raise ValueError(f"{scale_key} has incompatible shape {tuple(scale.shape)} for {tuple(codes.shape)}")

        nnz_by_row = codes.ne(0).sum(dim=1).double().cpu()
        active = nnz_by_row > 0
        if not bool(active.any()):
            continue
        active_scales = row_scales.double()[active]
        tensor_scale = float(active_scales.max().item())
        scale_error = tensor_scale - row_scales.double()

        true_sq = float((nnz_by_row * row_scales.double().square()).sum().item())
        error_sq = float((nnz_by_row * scale_error.square()).sum().item())
        rel_fro = math.sqrt(error_sq / true_sq) if true_sq > 0.0 else float("nan")
        nnz = int(nnz_by_row.sum().item())
        values = int(codes.numel())
        scale_min = float(active_scales.min().item())
        scale_median = float(active_scales.median().item())
        scale_mean = float(active_scales.mean().item())
        scale_std = float(active_scales.std(unbiased=False).item()) if active_scales.numel() > 1 else 0.0

        total_true_sq += true_sq
        total_error_sq += error_sq
        total_nnz += nnz
        total_values += values
        tensor_rows.append(
            {
                "name": prefix,
                "shape": list(codes.shape),
                "scale_mode": scale_mode,
                "rel_fro_error_if_one_scale": rel_fro,
                "expected_rel_output_rms_error": rel_fro,
                "tensor_scale": tensor_scale,
                "row_scale_min": scale_min,
                "row_scale_median": scale_median,
                "row_scale_mean": scale_mean,
                "row_scale_std": scale_std,
                "row_scale_cv": scale_std / scale_mean if scale_mean else 0.0,
                "tensor_scale_over_median": tensor_scale / scale_median if scale_median else float("inf"),
                "nonzero_fraction": nnz / values if values else 0.0,
            }
        )

    rel_values = [float(row["rel_fro_error_if_one_scale"]) for row in tensor_rows]
    aggregate = {
        "label": label,
        "path": str(path),
        "tensors": len(tensor_rows),
        "row_scale_tensors": row_scale_tensors,
        "scalar_scale_tensors": scalar_scale_tensors,
        "total_relative_fro_error_if_one_scale": math.sqrt(total_error_sq / total_true_sq) if total_true_sq > 0.0 else float("nan"),
        "expected_total_relative_output_rms_error": math.sqrt(total_error_sq / total_true_sq) if total_true_sq > 0.0 else float("nan"),
        "median_tensor_relative_fro_error": percentile(rel_values, 0.50),
        "p95_tensor_relative_fro_error": percentile(rel_values, 0.95),
        "max_tensor_relative_fro_error": max(rel_values) if rel_values else float("nan"),
        "nonzero_fraction": total_nnz / total_values if total_values else 0.0,
        "top_tensors": sorted(tensor_rows, key=lambda row: row["rel_fro_error_if_one_scale"], reverse=True)[:10],
    }
    return aggregate


def build_report(results: list[dict[str, Any]]) -> str:
    rows = [
        [
            result["label"],
            str(result["tensors"]),
            str(result["scalar_scale_tensors"]),
            str(result["row_scale_tensors"]),
            fmt(float(result["total_relative_fro_error_if_one_scale"])),
            fmt(float(result["median_tensor_relative_fro_error"])),
            fmt(float(result["p95_tensor_relative_fro_error"])),
            fmt(float(result["max_tensor_relative_fro_error"])),
            fmt(float(result["nonzero_fraction"]), 4),
        ]
        for result in results
    ]

    sections: list[str] = [
        "# TL2 Scale Semantics Audit, 2026-05-05",
        "This audit measures the error introduced if a ternary checkpoint that was trained with row-wise scales is exported through the current TL1/TL2 single-scale convention. For random isotropic activations, the expected relative output RMS error equals the relative Frobenius error of the effective weight matrix, so this is a direct linear-layer error proxy.",
        md_table(
            [
                "checkpoint",
                "tensors",
                "scalar-scale tensors",
                "row-scale tensors",
                "total rel Fro error",
                "median tensor rel error",
                "p95 tensor rel error",
                "max tensor rel error",
                "nonzero frac",
            ],
            rows,
        ),
    ]

    for result in results:
        top_rows = [
            [
                row["name"],
                "x".join(str(dim) for dim in row["shape"]),
                fmt(float(row["rel_fro_error_if_one_scale"])),
                fmt(float(row["tensor_scale_over_median"]), 3),
                fmt(float(row["row_scale_cv"]), 3),
                fmt(float(row["nonzero_fraction"]), 4),
            ]
            for row in result["top_tensors"][:5]
        ]
        sections.extend(
            [
                f"## {result['label']} Largest Single-Scale Errors",
                md_table(
                    [
                        "tensor",
                        "shape",
                        "rel error",
                        "global/median scale",
                        "scale CV",
                        "nonzero frac",
                    ],
                    top_rows,
                ),
            ]
        )

    verdict = (
        "A scalar-scale checkpoint should show near-zero error because TL2's one-scale "
        "assumption matches its scale semantics. A row-scale checkpoint with a large "
        "relative error is not safe to export through the current TL2 scale model; it "
        "needs row-scale metadata/runtime support or a retraining/export recipe that "
        "uses tensor-scale semantics."
    )
    sections.extend(["## Verdict", verdict])
    return "\n\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", action="append", required=True, help="LABEL=/path/to/ternary_state_dict.pt")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    results = [audit_state(label, path) for label, path in (parse_label_path(item) for item in args.state)]
    report = build_report(results)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report + "\n", encoding="utf-8")
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps({"results": results}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
