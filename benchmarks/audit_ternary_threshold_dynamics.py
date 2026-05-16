#!/usr/bin/env python3
"""Measure whether Stage-2 weights concentrate near ternary thresholds.

BitDistill's analysis argues that continued pretraining helps pretrained FP
weights adapt into a BitNet-like regime where many values sit near ternary
transition boundaries.  This audit measures that mechanism directly on saved
Stage-2 custom state snapshots.

For a weight tensor W and scale alpha, ternary rounding changes state at
|W| / alpha = 0.5.  The report tracks the fraction of projection weights near
that boundary under either tensor-scale or row-scale alpha semantics.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()
DEFAULT_ROOT = Path("checkpoints/bitdistill-glue-longwarmup-row/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-row-20k")
DEFAULT_STEPS = [1000, 5000, 10000, 15000, 20000]
PROJECTION_RE = re.compile(
    r"\.(q_proj|k_proj|v_proj|gate_proj|up_proj)\.weight$|"
    r"\.(o_proj|down_proj)\.proj\.weight$"
)


def checkpoint_state_path(root: Path, step: int) -> Path:
    checkpoint_path = root / f"checkpoint-{step}" / "custom_state_dict.pt"
    if checkpoint_path.exists():
        return checkpoint_path
    if (root / "custom_state_dict.pt").exists() and step == max(DEFAULT_STEPS):
        return root / "custom_state_dict.pt"
    return checkpoint_path


def family_name(key: str) -> str:
    match = PROJECTION_RE.search(key)
    if not match:
        return "other"
    return next(group for group in match.groups() if group is not None)


def projection_items(state: dict[str, Any]) -> list[tuple[str, torch.Tensor]]:
    out: list[tuple[str, torch.Tensor]] = []
    for key, value in state.items():
        if PROJECTION_RE.search(key) and isinstance(value, torch.Tensor) and value.ndim == 2:
            out.append((key, value.detach().float().cpu()))
    return out


def empty_counts() -> dict[str, float]:
    return defaultdict(float)


def add_tensor_stats(counts: dict[str, float], weight: torch.Tensor, *, scale_mode: str, bands: list[float]) -> None:
    abs_w = weight.abs()
    if scale_mode == "row":
        alpha = abs_w.mean(dim=1, keepdim=True).clamp_min(1e-12)
        scale_values = alpha.flatten()
    elif scale_mode == "tensor":
        alpha = abs_w.mean().clamp_min(1e-12)
        scale_values = alpha.reshape(1)
    else:
        raise ValueError(f"unsupported scale mode: {scale_mode}")

    normalized = abs_w / alpha
    rounded = torch.round(weight / alpha).clamp(-1, 1)
    elements = int(weight.numel())
    counts["elements"] += elements
    counts["tensors"] += 1
    counts["minus_one"] += int((rounded == -1).sum().item())
    counts["zero"] += int((rounded == 0).sum().item())
    counts["plus_one"] += int((rounded == 1).sum().item())
    counts["zero_region"] += int((normalized < 0.5).sum().item())
    counts["nonzero_region"] += int((normalized >= 0.5).sum().item())
    counts["norm_sum"] += float(normalized.sum().item())
    counts["norm_sq_sum"] += float((normalized * normalized).sum().item())
    counts["scale_count"] += int(scale_values.numel())
    counts["scale_sum"] += float(scale_values.sum().item())
    counts["scale_sq_sum"] += float((scale_values * scale_values).sum().item())
    counts["scale_min"] = min(float(counts.get("scale_min", float("inf"))), float(scale_values.min().item()))
    counts["scale_max"] = max(float(counts.get("scale_max", float("-inf"))), float(scale_values.max().item()))
    for band in bands:
        counts[f"threshold_band_{band}"] += int(((normalized - 0.5).abs() <= band).sum().item())


def finalize(counts: dict[str, float], bands: list[float]) -> dict[str, Any]:
    elements = int(counts.get("elements", 0))
    scale_count = int(counts.get("scale_count", 0))
    mean_norm = counts["norm_sum"] / elements if elements else None
    norm_var = counts["norm_sq_sum"] / elements - mean_norm * mean_norm if elements and mean_norm is not None else None
    scale_mean = counts["scale_sum"] / scale_count if scale_count else None
    scale_var = (
        counts["scale_sq_sum"] / scale_count - scale_mean * scale_mean
        if scale_count and scale_mean is not None
        else None
    )
    return {
        "tensors": int(counts.get("tensors", 0)),
        "elements": elements,
        "code_fractions": {
            "-1": counts["minus_one"] / elements if elements else None,
            "0": counts["zero"] / elements if elements else None,
            "1": counts["plus_one"] / elements if elements else None,
        },
        "zero_region_fraction": counts["zero_region"] / elements if elements else None,
        "nonzero_region_fraction": counts["nonzero_region"] / elements if elements else None,
        "threshold_band_fractions": {
            str(band): counts[f"threshold_band_{band}"] / elements if elements else None for band in bands
        },
        "normalized_abs_mean": mean_norm,
        "normalized_abs_std": math.sqrt(max(norm_var, 0.0)) if norm_var is not None else None,
        "scale_mean": scale_mean,
        "scale_std": math.sqrt(max(scale_var, 0.0)) if scale_var is not None else None,
        "scale_min": counts.get("scale_min") if scale_count else None,
        "scale_max": counts.get("scale_max") if scale_count else None,
    }


def summarize_snapshot(path: Path, step: int, *, scale_mode: str, bands: list[float]) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"{path} did not load to a dict")
    total = empty_counts()
    family_counts: dict[str, dict[str, float]] = defaultdict(empty_counts)
    for key, weight in projection_items(state):
        family = family_name(key)
        add_tensor_stats(total, weight, scale_mode=scale_mode, bands=bands)
        add_tensor_stats(family_counts[family], weight, scale_mode=scale_mode, bands=bands)
    del state
    gc.collect()
    return {
        "step": step,
        "path": str(path),
        "scale_mode": scale_mode,
        "total": finalize(total, bands),
        "families": {family: finalize(counts, bands) for family, counts in sorted(family_counts.items())},
    }


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if 0.0 < abs(value) < 0.0001:
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


def render_markdown(result: dict[str, Any]) -> str:
    band = result["primary_band"]
    snapshot_rows = []
    for row in result["snapshots"]:
        total = row["total"]
        snapshot_rows.append(
            [
                row["step"],
                total["tensors"],
                total["elements"],
                total["threshold_band_fractions"][str(band)],
                total["zero_region_fraction"],
                total["code_fractions"]["0"],
                total["code_fractions"]["-1"],
                total["code_fractions"]["1"],
                total["scale_mean"],
                total["scale_std"],
            ]
        )

    family_rows: list[list[Any]] = []
    final = result["snapshots"][-1] if result["snapshots"] else None
    if final is not None:
        for family, stats in final["families"].items():
            family_rows.append(
                [
                    family,
                    stats["elements"],
                    stats["threshold_band_fractions"][str(band)],
                    stats["zero_region_fraction"],
                    stats["code_fractions"]["0"],
                    stats["scale_mean"],
                    stats["scale_std"],
                ]
            )

    return "\n\n".join(
        [
            f"# Ternary Threshold Dynamics Audit, {result['date']}",
            f"Status: **{result['status']}**.",
            f"Snapshot root: `{result['snapshot_root']}`.",
            f"Scale mode: `{result['scale_mode']}`. Primary boundary band: `±{band}` around `|W|/alpha=0.5`.",
            (
                "This is a mechanism audit for saved Stage-2 weights. It supports or rejects the narrow claim that "
                "continued pretraining moves latent FP weights near ternary transition boundaries; it does not by "
                "itself prove task quality or paper reproduction."
            ),
            "## Snapshot Trend",
            md_table(
                [
                    "step",
                    "projection tensors",
                    "elements",
                    f"threshold ±{band}",
                    "zero-region",
                    "code 0",
                    "code -1",
                    "code +1",
                    "scale mean",
                    "scale std",
                ],
                snapshot_rows,
            ),
            "## Final-Step Family Breakdown",
            md_table(
                ["family", "elements", f"threshold ±{band}", "zero-region", "code 0", "scale mean", "scale std"],
                family_rows,
            ),
            "## Interpretation",
            result["interpretation"],
        ]
    )


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repo_root.resolve()
    snapshot_root = args.snapshot_root if args.snapshot_root.is_absolute() else root / args.snapshot_root
    steps = sorted(set(args.steps))
    paths = [checkpoint_state_path(snapshot_root, step) for step in steps]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("missing snapshots: " + ", ".join(missing))

    snapshots = [
        summarize_snapshot(path, step, scale_mode=args.scale_mode, bands=args.bands)
        for step, path in zip(steps, paths)
    ]
    band = args.primary_band
    first = snapshots[0]["total"]["threshold_band_fractions"][str(band)] if snapshots else None
    last = snapshots[-1]["total"]["threshold_band_fractions"][str(band)] if snapshots else None
    delta = last - first if isinstance(first, float) and isinstance(last, float) else None
    monotonic_non_decreasing = all(
        snapshots[i]["total"]["threshold_band_fractions"][str(band)]
        <= snapshots[i + 1]["total"]["threshold_band_fractions"][str(band)] + 1e-12
        for i in range(len(snapshots) - 1)
    )
    status = "measured_increase" if isinstance(delta, float) and delta > 0 else "measured_no_increase"
    interpretation = (
        f"The primary threshold-band fraction changed by {fmt(delta)} from step {steps[0]} to {steps[-1]}. "
        f"Monotonic non-decreasing: {fmt(monotonic_non_decreasing)}. "
    )
    if isinstance(delta, float) and delta > 0:
        interpretation += (
            "This supports the mechanism that continued pretraining keeps more latent weights close to ternary "
            "transition boundaries, at least for this saved row-scale Stage-2 run."
        )
    else:
        interpretation += (
            "This does not support a simple boundary-concentration mechanism for this saved run; quality recovery "
            "must be explained by other dynamics or by unsaved early movement."
        )
    return {
        "schema": "ternary-threshold-dynamics-v1",
        "date": DATE,
        "status": status,
        "snapshot_root": str(snapshot_root.relative_to(root)),
        "scale_mode": args.scale_mode,
        "steps": steps,
        "bands": args.bands,
        "primary_band": band,
        "snapshots": snapshots,
        "threshold_band_delta_first_to_last": delta,
        "monotonic_non_decreasing": monotonic_non_decreasing,
        "interpretation": interpretation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--steps", type=int, nargs="+", default=DEFAULT_STEPS)
    parser.add_argument("--scale-mode", choices=["tensor", "row"], default="row")
    parser.add_argument("--bands", type=float, nargs="+", default=[0.05, 0.10])
    parser.add_argument("--primary-band", type=float, default=0.05)
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/ternary_threshold_dynamics_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/ternary_threshold_dynamics_{DATE}.md"))
    args = parser.parse_args()

    if args.primary_band not in args.bands:
        args.bands.append(args.primary_band)
        args.bands.sort()

    root = args.repo_root.resolve()
    result = build_summary(args)
    output_json = args.output_json if args.output_json.is_absolute() else root / args.output_json
    output_md = args.output_md if args.output_md.is_absolute() else root / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(result) + "\n", encoding="utf-8")
    print(render_markdown(result))


if __name__ == "__main__":
    main()
