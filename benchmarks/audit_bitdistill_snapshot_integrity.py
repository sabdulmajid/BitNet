#!/usr/bin/env python3
"""Audit BitDistill warm-up snapshot integrity.

This is a lightweight checkpoint contract for long-running Stage-2 jobs.  It
checks that a saved snapshot is loadable, has the expected ternary export count,
and that tensor-scale versus row-scale sidecars match the recorded `scale_mode`.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def snapshot_from_monitor(path: Path, root: Path) -> Path | None:
    monitor = read_json(path)
    warmup = monitor.get("warmup", {}) if isinstance(monitor.get("warmup"), dict) else {}
    snapshot = warmup.get("latest_snapshot")
    if not snapshot:
        return None
    snapshot_path = Path(snapshot)
    if not snapshot_path.is_absolute():
        snapshot_path = root / snapshot_path
    return snapshot_path


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and value == value and value not in (float("inf"), float("-inf"))


def check_codes(state: dict[str, Any], ternary_keys: list[str]) -> tuple[bool, str]:
    bad: list[str] = []
    for key in ternary_keys:
        tensor = state[key]
        if not torch.is_tensor(tensor):
            bad.append(f"{key}:not_tensor")
            continue
        if tensor.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            bad.append(f"{key}:dtype={tensor.dtype}")
            continue
        min_value = int(tensor.min().item()) if tensor.numel() else 0
        max_value = int(tensor.max().item()) if tensor.numel() else 0
        if min_value < -1 or max_value > 1:
            bad.append(f"{key}:range=[{min_value},{max_value}]")
    return not bad, "; ".join(bad[:5])


def audit_snapshot(path: Path, root: Path, validate_codes: bool) -> dict[str, Any]:
    metrics_path = path / "metrics.json"
    custom_state = path / "custom_state_dict.pt"
    ternary_state = path / "ternary_state_dict.pt"
    metrics = read_json(metrics_path)
    scale_mode = metrics.get("scale_mode")
    expected_ternary = (
        metrics.get("preparation", {}).get("bitlinear_replaced")
        if isinstance(metrics.get("preparation"), dict)
        else None
    )

    blockers: list[str] = []
    for required in (metrics_path, custom_state, ternary_state, path / "config.json", path / "tokenizer.json"):
        if not required.exists():
            blockers.append(f"missing {rel(required, root)}")

    state: dict[str, Any] = {}
    load_error = ""
    if ternary_state.exists():
        try:
            loaded = torch.load(ternary_state, map_location="cpu", weights_only=True)
            if isinstance(loaded, dict):
                state = loaded
            else:
                load_error = f"expected dict, got {type(loaded).__name__}"
        except Exception as exc:  # pragma: no cover - recorded in report
            load_error = repr(exc)
    if load_error:
        blockers.append(f"cannot load ternary_state_dict.pt: {load_error}")

    keys = list(state)
    ternary_keys = [key for key in keys if key.endswith(".ternary_weight")]
    scale_keys = [key for key in keys if key.endswith(".weight_scale")]
    tensor_scale_keys = [key for key in scale_keys if torch.is_tensor(state[key]) and state[key].numel() == 1]
    row_scale_keys = [key for key in scale_keys if torch.is_tensor(state[key]) and state[key].numel() > 1]
    scale_dtype_hist = Counter(str(state[key].dtype).replace("torch.", "") for key in scale_keys if torch.is_tensor(state[key]))
    ternary_dtype_hist = Counter(str(state[key].dtype).replace("torch.", "") for key in ternary_keys if torch.is_tensor(state[key]))

    if expected_ternary is not None and len(ternary_keys) != int(expected_ternary):
        blockers.append(f"ternary key count {len(ternary_keys)} != expected {expected_ternary}")
    if len(scale_keys) != len(ternary_keys):
        blockers.append(f"scale key count {len(scale_keys)} != ternary key count {len(ternary_keys)}")
    if scale_mode == "row" and len(row_scale_keys) != len(scale_keys):
        blockers.append(f"row scale mode has {len(row_scale_keys)}/{len(scale_keys)} row-scale sidecars")
    if scale_mode == "tensor" and len(tensor_scale_keys) != len(scale_keys):
        blockers.append(f"tensor scale mode has {len(tensor_scale_keys)}/{len(scale_keys)} scalar sidecars")
    if scale_mode not in {"row", "tensor"}:
        blockers.append(f"unexpected scale_mode={scale_mode!r}")
    if not finite_number(metrics.get("last", {}).get("ce") if isinstance(metrics.get("last"), dict) else None):
        blockers.append("latest CE is missing or non-finite")

    codes_valid = None
    code_error = ""
    if validate_codes and state:
        codes_valid, code_error = check_codes(state, ternary_keys)
        if not codes_valid:
            blockers.append(f"invalid ternary codes: {code_error}")

    return {
        "snapshot_dir": rel(path, root),
        "exists": path.exists(),
        "passed": not blockers,
        "blockers": blockers,
        "metrics": {
            "step": metrics.get("steps"),
            "snapshot_step": metrics.get("snapshot", {}).get("step") if isinstance(metrics.get("snapshot"), dict) else None,
            "stage": metrics.get("stage"),
            "scale_mode": scale_mode,
            "ce": metrics.get("last", {}).get("ce") if isinstance(metrics.get("last"), dict) else None,
            "bitlinear_replaced": expected_ternary,
            "subln_inserted": metrics.get("preparation", {}).get("subln_inserted")
            if isinstance(metrics.get("preparation"), dict)
            else None,
        },
        "files": {
            "custom_state_bytes": custom_state.stat().st_size if custom_state.exists() else None,
            "ternary_state_bytes": ternary_state.stat().st_size if ternary_state.exists() else None,
        },
        "state": {
            "key_count": len(keys),
            "ternary_weight_count": len(ternary_keys),
            "weight_scale_count": len(scale_keys),
            "tensor_scale_count": len(tensor_scale_keys),
            "row_scale_count": len(row_scale_keys),
            "ternary_dtype_histogram": dict(sorted(ternary_dtype_hist.items())),
            "scale_dtype_histogram": dict(sorted(scale_dtype_hist.items())),
            "codes_validated": bool(validate_codes),
            "codes_valid": codes_valid,
            "code_error": code_error,
        },
    }


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def render_markdown(result: dict[str, Any]) -> str:
    rows = []
    blockers = []
    for snapshot in result["snapshots"]:
        metrics = snapshot["metrics"]
        state = snapshot["state"]
        blockers.extend(snapshot["blockers"])
        rows.append(
            [
                snapshot["snapshot_dir"],
                snapshot["passed"],
                metrics.get("step"),
                metrics.get("scale_mode"),
                metrics.get("ce"),
                metrics.get("bitlinear_replaced"),
                state.get("ternary_weight_count"),
                state.get("weight_scale_count"),
                state.get("row_scale_count"),
                state.get("tensor_scale_count"),
                state.get("codes_valid") if state.get("codes_validated") else "not_checked",
                "; ".join(snapshot["blockers"]),
            ]
        )
    return "\n\n".join(
        [
            f"# BitDistill Snapshot Integrity Audit, {result['date']}",
            f"Overall status: `{fmt(result['passed'])}`.",
            "## Snapshots",
            md_table(
                [
                    "snapshot",
                    "passed",
                    "step",
                    "scale",
                    "CE",
                    "expected ternary",
                    "ternary",
                    "scales",
                    "row scales",
                    "tensor scales",
                    "codes",
                    "blockers",
                ],
                rows,
            ),
            "## Blockers",
            md_table(["blocker"], [[blocker] for blocker in blockers] or [["none"]]),
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--monitor-json", type=Path, action="append", default=[])
    parser.add_argument("--snapshot-dir", type=Path, action="append", default=[])
    parser.add_argument("--validate-codes", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/bitdistill_snapshot_integrity_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/bitdistill_snapshot_integrity_{DATE}.md"))
    args = parser.parse_args()

    root = args.repo_root.resolve()
    snapshots: list[Path] = []
    for monitor in args.monitor_json:
        monitor_path = monitor if monitor.is_absolute() else root / monitor
        snapshot = snapshot_from_monitor(monitor_path, root)
        if snapshot is not None:
            snapshots.append(snapshot)
    for snapshot in args.snapshot_dir:
        snapshots.append(snapshot if snapshot.is_absolute() else root / snapshot)
    snapshots = sorted({path.resolve() for path in snapshots})

    result = {
        "schema": "bitdistill-snapshot-integrity-v1",
        "date": DATE,
        "snapshot_count": len(snapshots),
        "snapshots": [audit_snapshot(path, root, validate_codes=args.validate_codes) for path in snapshots],
    }
    result["passed"] = bool(result["snapshots"]) and all(row["passed"] for row in result["snapshots"])

    output_json = args.output_json if args.output_json.is_absolute() else root / args.output_json
    output_md = args.output_md if args.output_md.is_absolute() else root / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_markdown(result)
    output_md.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
