#!/usr/bin/env python3
"""Executable contract check for MoE merged-expert packing gaps.

llama.cpp stores many MoE expert weights as merged 3D tensors with shape
`[experts, out, in]`. Dense Qwen ternary packing in this fork currently handles
2D matrices. This audit uses synthetic tensors to verify whether the TL2 and
direct I2_S/I2_SR packers accept or reject merged expert tensors.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch


DATE = "2026-05-13"


def load_module(path: Path, name: str, extra_path: Path | None = None) -> ModuleType:
    if extra_path is not None:
        sys.path.insert(0, str(extra_path))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def call_contract(label: str, func: Any, *args: Any) -> dict[str, Any]:
    try:
        output = func(*args)
        shape = list(output.shape) if hasattr(output, "shape") else None
        return {
            "label": label,
            "accepted": True,
            "error_type": None,
            "error": None,
            "output_shape": shape,
        }
    except Exception as exc:  # noqa: BLE001 - this is an audit of the failure surface.
        return {
            "label": label,
            "accepted": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "output_shape": None,
        }


def md_bool(value: bool) -> str:
    return "`true`" if value else "`false`"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_audit(root: Path) -> dict[str, Any]:
    bitnet_converter = load_module(
        root / "utils/convert-hf-to-gguf-bitnet.py",
        "bitnet_hf_converter_for_moe_contract",
        root / "utils",
    )
    direct_writer = load_module(
        root / "benchmarks/convert_static_ternary_to_i2s_gguf.py",
        "direct_i2s_writer_for_moe_contract",
    )

    tl2_expert = np.zeros((2, 4, 128), dtype=np.float32)
    i2s_expert = torch.zeros((2, 4, 128), dtype=torch.int8)
    i2s_dense_control = torch.zeros((4, 128), dtype=torch.int8)

    checks = [
        call_contract("tl2_merged_3d_expert", bitnet_converter.preprocess_weights_tl2, tl2_expert),
        call_contract("i2s_merged_3d_expert", direct_writer.pack_i2_s_codes_x86_act, i2s_expert),
        call_contract("i2s_2d_dense_control", direct_writer.pack_i2_s_codes_x86_act, i2s_dense_control),
    ]

    tl2_3d = next(item for item in checks if item["label"] == "tl2_merged_3d_expert")
    i2s_3d = next(item for item in checks if item["label"] == "i2s_merged_3d_expert")
    i2s_2d = next(item for item in checks if item["label"] == "i2s_2d_dense_control")
    verdict = {
        "merged_3d_tl2_supported": bool(tl2_3d["accepted"]),
        "merged_3d_i2s_i2sr_supported": bool(i2s_3d["accepted"]),
        "dense_2d_i2s_control_supported": bool(i2s_2d["accepted"]),
        "moe_packing_ready": bool(tl2_3d["accepted"] and i2s_3d["accepted"] and i2s_2d["accepted"]),
    }
    blockers = []
    if not verdict["merged_3d_tl2_supported"]:
        blockers.append("TL2 preprocessing rejects merged 3D expert tensors before kernel lookup.")
    if not verdict["merged_3d_i2s_i2sr_supported"]:
        blockers.append("Direct I2_S/I2_SR code packing rejects merged 3D expert tensors.")
    if not verdict["dense_2d_i2s_control_supported"]:
        blockers.append("2D dense control failed; direct writer regression must be fixed before interpreting MoE result.")

    return {
        "schema": "bitnet-moe-packing-contract-v1",
        "date": DATE,
        "synthetic_shapes": {
            "merged_expert": [2, 4, 128],
            "dense_control": [4, 128],
        },
        "checks": checks,
        "verdict": verdict,
        "blockers": blockers,
        "required_next_steps": [
            "Define a 3D expert tensor layout for TL2 and I2_SR instead of flattening expert identity away.",
            "Add per-expert or per-expert-row scale metadata semantics.",
            "Add byte-layout regression tests for merged expert tensors.",
            "Only then run Kimi/Qwen2MoE quality, throughput, RSS, and expert-locality benchmarks.",
        ],
    }


def render_markdown(result: dict[str, Any]) -> str:
    rows = []
    for check in result["checks"]:
        rows.append(
            [
                check["label"],
                md_bool(bool(check["accepted"])),
                f"`{check['error_type'] or ''}`",
                f"`{(check['error'] or '').replace('|', '/')}`",
                f"`{check['output_shape'] or ''}`",
            ]
        )
    verdict_rows = [[key, md_bool(bool(value))] for key, value in result["verdict"].items()]
    blocker_rows = [[item] for item in result["blockers"]] or [["none"]]
    next_rows = [[item] for item in result["required_next_steps"]]
    return "\n\n".join(
        [
            f"# MoE Packing Contract Audit, {DATE}",
            "This audit uses synthetic merged expert tensors with shape `[experts, out, in]` to test whether current dense ternary packers support MoE weight layout.",
            "## Checks",
            md_table(["check", "accepted", "error type", "error", "output shape"], rows),
            "## Verdict",
            md_table(["field", "value"], verdict_rows),
            "## Blockers",
            md_table(["blocker"], blocker_rows),
            "## Required Next Steps",
            md_table(["step"], next_rows),
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-json", type=Path, default=Path("benchmark_results/moe_packing_contract_2026-05-13.json"))
    parser.add_argument("--output-md", type=Path, default=Path("benchmarks/results/moe_packing_contract_2026-05-13.md"))
    args = parser.parse_args()

    root = args.repo_root.resolve()
    result = build_audit(root)
    output_json = args.output_json if args.output_json.is_absolute() else root / args.output_json
    output_md = args.output_md if args.output_md.is_absolute() else root / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(result), encoding="utf-8")
    print(render_markdown(result))


if __name__ == "__main__":
    main()
