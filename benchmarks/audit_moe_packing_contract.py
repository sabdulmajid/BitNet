#!/usr/bin/env python3
"""Executable contract check for MoE merged-expert packing gaps.

llama.cpp stores many MoE expert weights as merged 3D tensors with shape
`[experts, out, in]`. This audit uses synthetic tensors to verify whether the
TL2 and direct I2_S/I2_SR packers accept or reject merged expert tensors.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()


def load_module(path: Path, name: str, extra_path: Path | None = None) -> ModuleType:
    if extra_path is not None:
        sys.path.insert(0, str(extra_path))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_prefix(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()[:12]


def expected_i2s_x86_act(codes: torch.Tensor) -> np.ndarray:
    """Independent byte oracle for the active x86 I2_S packing order."""
    q8 = (codes.to(torch.int16).cpu().numpy() + 1).astype(np.uint8, copy=False)
    if q8.ndim not in {2, 3}:
        raise ValueError(f"expected 2D or 3D codes, got shape {q8.shape}")
    if q8.shape[-1] % 128 != 0:
        raise ValueError(f"innermost dimension must be divisible by 128, got {q8.shape[-1]}")

    logical_rows = q8.reshape(-1, q8.shape[-1])
    out = bytearray()
    for row in logical_rows:
        for block_start in range(0, row.shape[0], 128):
            block = row[block_start : block_start + 128]
            for j in range(32):
                out.append(
                    (int(block[j]) << 6)
                    | (int(block[j + 32]) << 4)
                    | (int(block[j + 64]) << 2)
                    | int(block[j + 96])
                )
    return np.frombuffer(bytes(out), dtype=np.uint8).copy()


def expected_i2s_scalar(codes: torch.Tensor, scale: torch.Tensor) -> np.ndarray:
    packed = expected_i2s_x86_act(codes)
    output = np.zeros(packed.size + 32, dtype=np.uint8)
    output[: packed.size] = packed
    output[packed.size : packed.size + 4] = np.asarray([float(scale.reshape(-1)[0].item())], dtype=np.float32).view(np.uint8)
    return output


def expected_i2sr_row_scale(codes: torch.Tensor, scale: torch.Tensor) -> np.ndarray:
    packed = expected_i2s_x86_act(codes)
    scale_bytes = scale.reshape(-1).to(dtype=torch.float32).cpu().numpy().astype(np.float32, copy=False).view(np.uint8)
    output = np.zeros(packed.size + scale_bytes.size + 32, dtype=np.uint8)
    output[: packed.size] = packed
    output[packed.size : packed.size + scale_bytes.size] = scale_bytes
    return output


def call_contract(label: str, func: Any, *args: Any, expected: np.ndarray | None = None) -> dict[str, Any]:
    try:
        output = func(*args)
        output_array = np.asarray(output) if hasattr(output, "shape") else None
        shape = list(output_array.shape) if output_array is not None else None
        nbytes = int(output_array.nbytes) if output_array is not None else None
        layout_verified = None
        output_sha = None
        expected_sha = None
        verification_error = None
        if output_array is not None:
            output_sha = sha256_prefix(output_array)
        if expected is not None:
            expected_sha = sha256_prefix(expected)
            layout_verified = bool(
                output_array is not None
                and output_array.dtype == expected.dtype
                and output_array.shape == expected.shape
                and output_array.tobytes() == expected.tobytes()
            )
            if not layout_verified:
                verification_error = (
                    f"expected dtype/shape/sha {expected.dtype}/{list(expected.shape)}/{expected_sha}, "
                    f"got {getattr(output_array, 'dtype', None)}/{shape}/{output_sha}"
                )
        return {
            "label": label,
            "accepted": True,
            "error_type": None,
            "error": None,
            "output_shape": shape,
            "output_nbytes": nbytes,
            "layout_verified": layout_verified,
            "output_sha256_prefix": output_sha,
            "expected_sha256_prefix": expected_sha,
            "verification_error": verification_error,
        }
    except Exception as exc:  # noqa: BLE001 - this is an audit of the failure surface.
        return {
            "label": label,
            "accepted": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "output_shape": None,
            "output_nbytes": None,
            "layout_verified": False if expected is not None else None,
            "output_sha256_prefix": None,
            "expected_sha256_prefix": sha256_prefix(expected) if expected is not None else None,
            "verification_error": str(exc) if expected is not None else None,
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

    i2s_expert = (torch.arange(4 * 8 * 384, dtype=torch.int16).reshape(4, 8, 384) % 3 - 1).to(torch.int8)
    tl2_expert = i2s_expert.to(torch.float32).numpy()
    i2s_expert_scalar = torch.tensor([0.75], dtype=torch.float32)
    i2s_expert_row_scale = torch.linspace(0.25, 0.95, steps=32, dtype=torch.float32).reshape(4, 8)
    i2s_dense_control = (torch.arange(32 * 384, dtype=torch.int16).reshape(32, 384) % 3 - 1).to(torch.int8)

    with tempfile.TemporaryDirectory() as tmp:
        kernel_config = Path(tmp) / "kernel_config.ini"
        kernel_config.write_text(
            "\n".join(
                [
                    "[synthetic_moe_32x384]",
                    "m = 32",
                    "k = 384",
                    "bm = 32",
                    "bk = 96",
                    "bmm = 32",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        old_kernel_config = os.environ.get("BITNET_KERNEL_CONFIG")
        os.environ["BITNET_KERNEL_CONFIG"] = str(kernel_config)
        try:
            checks = [
                call_contract("tl2_merged_3d_expert", bitnet_converter.preprocess_weights_tl2, tl2_expert),
                call_contract(
                    "i2s_merged_3d_expert_codes",
                    direct_writer.pack_i2_s_codes_x86_act,
                    i2s_expert,
                    expected=expected_i2s_x86_act(i2s_expert),
                ),
                call_contract(
                    "i2s_merged_3d_expert_scalar",
                    direct_writer.pack_i2_s_scalar,
                    i2s_expert,
                    i2s_expert_scalar,
                    expected=expected_i2s_scalar(i2s_expert, i2s_expert_scalar),
                ),
                call_contract(
                    "i2sr_merged_3d_expert_row_scale",
                    direct_writer.pack_i2_s_row_prototype,
                    i2s_expert,
                    i2s_expert_row_scale,
                    expected=expected_i2sr_row_scale(i2s_expert, i2s_expert_row_scale),
                ),
                call_contract(
                    "i2s_2d_dense_control",
                    direct_writer.pack_i2_s_codes_x86_act,
                    i2s_dense_control,
                    expected=expected_i2s_x86_act(i2s_dense_control),
                ),
            ]
        finally:
            if old_kernel_config is None:
                os.environ.pop("BITNET_KERNEL_CONFIG", None)
            else:
                os.environ["BITNET_KERNEL_CONFIG"] = old_kernel_config

    tl2_3d = next(item for item in checks if item["label"] == "tl2_merged_3d_expert")
    i2s_3d_codes = next(item for item in checks if item["label"] == "i2s_merged_3d_expert_codes")
    i2s_3d_scalar = next(item for item in checks if item["label"] == "i2s_merged_3d_expert_scalar")
    i2sr_3d_row = next(item for item in checks if item["label"] == "i2sr_merged_3d_expert_row_scale")
    i2s_2d = next(item for item in checks if item["label"] == "i2s_2d_dense_control")
    verdict = {
        "merged_3d_tl2_supported": bool(tl2_3d["accepted"]),
        "merged_3d_i2s_code_packing_supported": bool(i2s_3d_codes["accepted"] and i2s_3d_codes["layout_verified"]),
        "merged_3d_i2s_scalar_supported": bool(i2s_3d_scalar["accepted"] and i2s_3d_scalar["layout_verified"]),
        "merged_3d_i2sr_row_scale_supported": bool(i2sr_3d_row["accepted"] and i2sr_3d_row["layout_verified"]),
        "merged_3d_i2s_i2sr_supported": bool(
            i2s_3d_codes["accepted"]
            and i2s_3d_codes["layout_verified"]
            and i2s_3d_scalar["accepted"]
            and i2s_3d_scalar["layout_verified"]
            and i2sr_3d_row["accepted"]
            and i2sr_3d_row["layout_verified"]
        ),
        "dense_2d_i2s_control_supported": bool(i2s_2d["accepted"] and i2s_2d["layout_verified"]),
        "moe_packing_ready": bool(
            tl2_3d["accepted"]
            and i2s_3d_codes["accepted"]
            and i2s_3d_codes["layout_verified"]
            and i2s_3d_scalar["accepted"]
            and i2s_3d_scalar["layout_verified"]
            and i2sr_3d_row["accepted"]
            and i2sr_3d_row["layout_verified"]
            and i2s_2d["accepted"]
            and i2s_2d["layout_verified"]
        ),
    }
    blockers = []
    if not verdict["merged_3d_tl2_supported"]:
        blockers.append("TL2 preprocessing still rejects merged 3D expert tensors before kernel lookup.")
    if not verdict["merged_3d_i2s_i2sr_supported"]:
        blockers.append("Direct I2_S/I2_SR code packing rejects merged 3D expert tensors.")
    if not verdict["dense_2d_i2s_control_supported"]:
        blockers.append("2D dense control failed; direct writer regression must be fixed before interpreting MoE result.")

    return {
        "schema": "bitnet-moe-packing-contract-v1",
        "date": DATE,
        "synthetic_shapes": {
            "merged_expert": [4, 8, 384],
            "dense_control": [32, 384],
        },
        "synthetic_payload": "nonzero deterministic ternary pattern with byte-exact independent packing oracle",
        "checks": checks,
        "verdict": verdict,
        "blockers": blockers,
        "required_next_steps": [
            "Fix TL2 runtime byte-size and stride semantics for tensors whose raw shape includes an expert dimension.",
            "Route TL2 `ggml_mul_mat_id` through an expert-aware BitNet LUT kernel.",
            "Add TL2 per-expert or per-expert-row scale metadata semantics before quality claims.",
            "Add full GGUF byte-layout regression tests with a real MoE checkpoint.",
            "Then run Kimi/Qwen2MoE quality, throughput, RSS, and expert-locality benchmarks.",
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
                f"`{check['output_nbytes'] or ''}`",
                md_bool(bool(check["layout_verified"])) if check["layout_verified"] is not None else "`n/a`",
                f"`{check['output_sha256_prefix'] or ''}`",
                f"`{check['expected_sha256_prefix'] or ''}`",
                f"`{(check['verification_error'] or '').replace('|', '/')}`",
            ]
        )
    verdict_rows = [[key, md_bool(bool(value))] for key, value in result["verdict"].items()]
    blocker_rows = [[item] for item in result["blockers"]] or [["none"]]
    next_rows = [[item] for item in result["required_next_steps"]]
    return "\n\n".join(
        [
            f"# MoE Packing Contract Audit, {DATE}",
            "This audit uses nonzero synthetic merged expert tensors with shape `[experts, out, in]` to test whether current dense ternary packers support MoE weight layout and byte order.",
            "## Checks",
            md_table(
                [
                    "check",
                    "accepted",
                    "error type",
                    "error",
                    "output shape",
                    "output bytes",
                    "layout verified",
                    "output sha",
                    "expected sha",
                    "verification error",
                ],
                rows,
            ),
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
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/moe_packing_contract_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/moe_packing_contract_{DATE}.md"))
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
