#!/usr/bin/env python3
"""Measure offline ternary-code flip dynamics across saved Stage-2 snapshots."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()
DEFAULT_ROOT = Path("checkpoints/bitdistill-glue-longwarmup-row/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-row-20k")
DEFAULT_STEPS = [1000, 10000, 20000]
FAMILY_RE = re.compile(r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)(?:\.proj)?\.ternary_weight$")


def family_name(key: str) -> str:
    if key == "lm_head.ternary_weight":
        return "lm_head"
    match = FAMILY_RE.search(key)
    return match.group(1) if match else "other"


def load_ternary(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"{path} did not load to a dict")
    return {
        key: value
        for key, value in state.items()
        if key.endswith(".ternary_weight") and isinstance(value, torch.Tensor)
    }


def snapshot_path(root: Path, step: int) -> Path:
    checkpoint = root / f"checkpoint-{step}" / "ternary_state_dict.pt"
    if checkpoint.exists():
        return checkpoint
    if step == 20000 and (root / "ternary_state_dict.pt").exists():
        return root / "ternary_state_dict.pt"
    return checkpoint


def accumulate_counts(tensor: torch.Tensor, counts: dict[str, int]) -> None:
    counts["elements"] += int(tensor.numel())
    counts["minus_one"] += int((tensor == -1).sum().item())
    counts["zero"] += int((tensor == 0).sum().item())
    counts["plus_one"] += int((tensor == 1).sum().item())


def snapshot_summary(root: Path, step: int) -> dict[str, Any]:
    path = snapshot_path(root, step)
    tensors = load_ternary(path)
    counts: dict[str, int] = defaultdict(int)
    family_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for key, tensor in tensors.items():
        accumulate_counts(tensor, counts)
        accumulate_counts(tensor, family_counts[family_name(key)])
    total = counts["elements"]
    return {
        "step": step,
        "path": str(path),
        "tensor_count": len(tensors),
        "elements": total,
        "fractions": {
            "-1": counts["minus_one"] / total if total else None,
            "0": counts["zero"] / total if total else None,
            "1": counts["plus_one"] / total if total else None,
        },
        "families": {
            family: {
                "elements": row["elements"],
                "fractions": {
                    "-1": row["minus_one"] / row["elements"] if row["elements"] else None,
                    "0": row["zero"] / row["elements"] if row["elements"] else None,
                    "1": row["plus_one"] / row["elements"] if row["elements"] else None,
                },
            }
            for family, row in sorted(family_counts.items())
        },
    }


def compare_pair(root: Path, left_step: int, right_step: int) -> dict[str, Any]:
    left = load_ternary(snapshot_path(root, left_step))
    right = load_ternary(snapshot_path(root, right_step))
    keys = sorted(set(left) & set(right))
    family_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total: dict[str, int] = defaultdict(int)
    for key in keys:
        a = left[key]
        b = right[key]
        if tuple(a.shape) != tuple(b.shape):
            raise ValueError(f"shape mismatch for {key}: {tuple(a.shape)} vs {tuple(b.shape)}")
        changed = a != b
        zero_to_nonzero = (a == 0) & (b != 0)
        nonzero_to_zero = (a != 0) & (b == 0)
        opposite_sign = (a * b) == -1
        bucket = family_counts[family_name(key)]
        for target in (total, bucket):
            target["elements"] += int(a.numel())
            target["changed"] += int(changed.sum().item())
            target["zero_to_nonzero"] += int(zero_to_nonzero.sum().item())
            target["nonzero_to_zero"] += int(nonzero_to_zero.sum().item())
            target["opposite_sign"] += int(opposite_sign.sum().item())
    del left, right
    gc.collect()

    def rates(row: dict[str, int]) -> dict[str, Any]:
        elements = row["elements"]
        return {
            "elements": elements,
            "changed": row["changed"],
            "flip_rate": row["changed"] / elements if elements else None,
            "zero_to_nonzero_rate": row["zero_to_nonzero"] / elements if elements else None,
            "nonzero_to_zero_rate": row["nonzero_to_zero"] / elements if elements else None,
            "opposite_sign_rate": row["opposite_sign"] / elements if elements else None,
        }

    return {
        "from_step": left_step,
        "to_step": right_step,
        "shared_tensors": len(keys),
        "total": rates(total),
        "families": {family: rates(row) for family, row in sorted(family_counts.items())},
    }


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).replace("|", "\\|")


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(cell) for cell in row) + " |")
    return "\n".join(lines)


def render_markdown(result: dict[str, Any]) -> str:
    snapshot_rows = [
        [
            row["step"],
            row["tensor_count"],
            row["elements"],
            row["fractions"]["-1"],
            row["fractions"]["0"],
            row["fractions"]["1"],
        ]
        for row in result["snapshots"]
    ]
    pair_rows = [
        [
            f"{row['from_step']}->{row['to_step']}",
            row["shared_tensors"],
            row["total"]["elements"],
            row["total"]["flip_rate"],
            row["total"]["zero_to_nonzero_rate"],
            row["total"]["nonzero_to_zero_rate"],
            row["total"]["opposite_sign_rate"],
        ]
        for row in result["pairs"]
    ]
    family_rows: list[list[Any]] = []
    for row in result["pairs"]:
        for family, stats in row["families"].items():
            family_rows.append(
                [
                    f"{row['from_step']}->{row['to_step']}",
                    family,
                    stats["elements"],
                    stats["flip_rate"],
                    stats["zero_to_nonzero_rate"],
                    stats["nonzero_to_zero_rate"],
                    stats["opposite_sign_rate"],
                ]
            )
    return "\n\n".join(
        [
            f"# Ternary Flip Dynamics Audit, {result['date']}",
            f"Status: **{result['status']}**.",
            f"Snapshot root: `{result['snapshot_root']}`.",
            "This is offline telemetry over saved Stage-2 checkpoints. It does not replace live per-step gradient or activation telemetry, but it measures whether ternary codes are still changing during continued pretraining.",
            "## Snapshot Code Fractions",
            md_table(["step", "tensors", "elements", "-1 fraction", "0 fraction", "+1 fraction"], snapshot_rows),
            "## Pairwise Flip Rates",
            md_table(
                [
                    "pair",
                    "shared tensors",
                    "elements",
                    "flip rate",
                    "0->nonzero",
                    "nonzero->0",
                    "-1<->+1",
                ],
                pair_rows,
            ),
            "## Family Flip Rates",
            md_table(["pair", "family", "elements", "flip rate", "0->nonzero", "nonzero->0", "-1<->+1"], family_rows),
            "## Interpretation",
            (
                "A nonzero flip rate means the Stage-2 student is not merely storing a fixed ternary projection; "
                "the discrete codes continue to move under the warm-up objective. This supports the framing that "
                "ternary retrofit is a training-dynamics problem, not only a packing problem."
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--steps", type=int, nargs="+", default=DEFAULT_STEPS)
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/ternary_flip_dynamics_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/ternary_flip_dynamics_{DATE}.md"))
    args = parser.parse_args()

    root = args.repo_root.resolve()
    snapshot_root = args.snapshot_root if args.snapshot_root.is_absolute() else root / args.snapshot_root
    steps = sorted(set(args.steps))
    missing = [str(snapshot_path(snapshot_root, step)) for step in steps if not snapshot_path(snapshot_root, step).exists()]
    if missing:
        raise FileNotFoundError("missing snapshots: " + ", ".join(missing))
    snapshots = [snapshot_summary(snapshot_root, step) for step in steps]
    pairs = [compare_pair(snapshot_root, left, right) for left, right in zip(steps, steps[1:])]
    finite_pair_rates = [
        row["total"]["flip_rate"]
        for row in pairs
        if isinstance(row.get("total"), dict) and isinstance(row["total"].get("flip_rate"), float)
    ]
    status = "pass" if snapshots and pairs and all(rate > 0.0 for rate in finite_pair_rates) else "incomplete"
    result = {
        "schema": "ternary_flip_dynamics.v1",
        "date": DATE,
        "status": status,
        "snapshot_root": str(snapshot_root.relative_to(root)),
        "steps": steps,
        "snapshots": snapshots,
        "pairs": pairs,
        "max_flip_rate": max(finite_pair_rates) if finite_pair_rates else None,
        "min_flip_rate": min(finite_pair_rates) if finite_pair_rates else None,
    }

    output_json = args.output_json if args.output_json.is_absolute() else root / args.output_json
    output_md = args.output_md if args.output_md.is_absolute() else root / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(result) + "\n", encoding="utf-8")
    print(render_markdown(result))


if __name__ == "__main__":
    main()
