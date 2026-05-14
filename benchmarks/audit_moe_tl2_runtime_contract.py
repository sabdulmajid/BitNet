#!/usr/bin/env python3
"""Audit whether TL2 can safely represent and execute merged MoE experts.

This is a static/runtime-contract audit, not a quality benchmark. It checks the
current Python converter and the vendored ggml runtime for the minimum contracts
needed by llama.cpp-style merged expert tensors with shape [experts, out, in].
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = datetime.now(timezone.utc).date().isoformat()
SYNTHETIC_SHAPE = {"experts": 4, "out": 256, "in": 384}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def first_line(text: str, pattern: str) -> int | None:
    for lineno, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return lineno
    return None


def slice_between(text: str, start: str, end: str) -> str:
    start_idx = text.find(start)
    if start_idx < 0:
        return ""
    end_idx = text.find(end, start_idx + len(start))
    if end_idx < 0:
        return text[start_idx:]
    return text[start_idx:end_idx]


def tl2_nbytes_for_rows(k: int, rows: int) -> int:
    # Mirror the active ggml_nbytes TL2 integer arithmetic for a logical row count.
    nbytes = (k - 256) * rows // 3 * 5 // 8 + 256 * rows // 2 * 4 // 8
    if nbytes % 32 != 0:
        nbytes = 32 - nbytes % 32 + nbytes
    return nbytes + 32


def make_check(name: str, passed: bool, evidence: str, blocker: str = "") -> dict[str, Any]:
    return {
        "name": name,
        "passed": passed,
        "evidence": evidence,
        "blocker": "" if passed else blocker,
    }


def build_audit(root: Path) -> dict[str, Any]:
    converter_path = root / "utils/convert-hf-to-gguf-bitnet.py"
    ggml_path = root / "3rdparty/llama.cpp/ggml/src/ggml.c"
    llama_path = root / "3rdparty/llama.cpp/src/llama.cpp"

    converter = read_text(converter_path)
    ggml = read_text(ggml_path)
    llama = read_text(llama_path)

    preprocess_tl2 = slice_between(converter, "def preprocess_weights_tl2", "def transform_to_tl1")
    ggml_nbytes = slice_between(ggml, "size_t ggml_nbytes", "size_t ggml_nbytes_pad")
    type_traits_tl2 = slice_between(ggml, "[GGML_TYPE_TL2]", "[GGML_TYPE_I8]")
    mul_mat = slice_between(ggml, "static void ggml_compute_forward_mul_mat(", "// ggml_compute_forward_mul_mat_id")
    mul_mat_id = slice_between(ggml, "static void ggml_compute_forward_mul_mat_id(", "// ggml_compute_forward_out_prod")
    quantize_loop = slice_between(llama, "// quantize only 2D and 3D tensors (experts)", "LLAMA_LOG_INFO(\"%s: model size")

    converter_has_legacy_2d_unpack = "M, K = w.shape" in preprocess_tl2
    converter_has_ndim_branch = "w.ndim" in preprocess_tl2 or "len(w.shape)" in preprocess_tl2
    converter_has_3d_branch = "w.ndim == 3" in preprocess_tl2
    nbytes_uses_nrows = "ggml_nrows(tensor)" in ggml_nbytes
    nbytes_uses_ne2 = nbytes_uses_nrows or "tensor->ne[2]" in ggml_nbytes or "ne[2]" in ggml_nbytes
    nbytes_uses_ne3 = nbytes_uses_nrows or "tensor->ne[3]" in ggml_nbytes or "ne[3]" in ggml_nbytes
    mul_mat_has_tl2_lut = "GGML_BITNET_X86_TL2" in mul_mat and "ggml_bitnet_can_mul_mat" in mul_mat and "ggml_qgemm_lut" in mul_mat
    mul_mat_id_has_tl2_lut = "GGML_BITNET_X86_TL2" in mul_mat_id or "ggml_bitnet_can_mul_mat" in mul_mat_id or "ggml_qgemm_lut" in mul_mat_id
    tl2_vec_dot_f32 = "ggml_vec_dot_f32" in type_traits_tl2 and "vec_dot_type             = GGML_TYPE_F32" in type_traits_tl2
    llama_quantizes_3d = "quantize only 2D and 3D tensors (experts)" in quantize_loop and "tensor->ne[2]" in quantize_loop

    experts = SYNTHETIC_SHAPE["experts"]
    out = SYNTHETIC_SHAPE["out"]
    in_features = SYNTHETIC_SHAPE["in"]
    one_expert_bytes = tl2_nbytes_for_rows(in_features, out)
    active_3d_bytes = tl2_nbytes_for_rows(in_features, out * experts)
    flat_expert_bytes = tl2_nbytes_for_rows(in_features, out * experts)
    underreport_bytes = flat_expert_bytes - active_3d_bytes
    underreport_ratio = flat_expert_bytes / active_3d_bytes if active_3d_bytes else None

    checks = [
        make_check(
            "TL2 Python preprocessor accepts explicit 3D expert tensors",
            converter_has_ndim_branch and converter_has_3d_branch,
            (
                f"preprocess_weights_tl2_line={first_line(converter, 'def preprocess_weights_tl2')}; "
                f"legacy_2d_unpack_present={converter_has_legacy_2d_unpack}; "
                f"has_ndim_branch={converter_has_ndim_branch}; has_3d_branch={converter_has_3d_branch}"
            ),
            "`preprocess_weights_tl2` still lacks an explicit [experts, out, in] branch.",
        ),
        make_check(
            "TL2 ggml_nbytes accounts for expert dimension",
            nbytes_uses_ne2 and nbytes_uses_ne3,
            (
                f"ggml_nbytes_line={first_line(ggml, 'size_t ggml_nbytes')}; "
                f"uses_nrows={nbytes_uses_nrows}; uses_ne2={nbytes_uses_ne2}; uses_ne3={nbytes_uses_ne3}; "
                f"shape=[{experts},{out},{in_features}]; one_expert_bytes={one_expert_bytes}; "
                f"active_3d_bytes={active_3d_bytes}; "
                f"flat_expert_bytes={flat_expert_bytes}; underreport_bytes={underreport_bytes}; "
                f"underreport_ratio={underreport_ratio:.6f}"
            ),
            "The active TL2 byte-size formula ignores tensor->ne[2]/ne[3], so merged expert tensors would be under-sized.",
        ),
        make_check(
            "TL2 LUT kernel is routed for normal dense matmul",
            mul_mat_has_tl2_lut,
            (
                f"mul_mat_line={first_line(ggml, 'static void ggml_compute_forward_mul_mat(')}; "
                f"has_tl2_lut={mul_mat_has_tl2_lut}"
            ),
            "Dense TL2 matmul must route through the generated BitNet LUT kernel.",
        ),
        make_check(
            "TL2 LUT kernel is routed for MoE ggml_mul_mat_id",
            mul_mat_id_has_tl2_lut,
            (
                f"mul_mat_id_line={first_line(ggml, 'static void ggml_compute_forward_mul_mat_id(')}; "
                f"has_tl2_lut={mul_mat_id_has_tl2_lut}; type_traits_vec_dot_f32={tl2_vec_dot_f32}"
            ),
            "`ggml_compute_forward_mul_mat_id` does not route TL2 through the BitNet LUT kernel; TL2 type traits fall back to F32 vec-dot semantics.",
        ),
        make_check(
            "llama.cpp generic quantizer is at least 3D-expert aware",
            llama_quantizes_3d,
            (
                f"quantize_loop_line={first_line(llama, 'quantize only 2D and 3D tensors (experts)')}; "
                f"uses_tensor_ne2={llama_quantizes_3d}"
            ),
            "The generic quantizer must preserve expert identity when it sees merged 3D tensors.",
        ),
    ]

    blockers = [check["blocker"] for check in checks if not check["passed"]]
    return {
        "schema": "bitnet-moe-tl2-runtime-contract-v1",
        "date": DATE,
        "synthetic_shape": SYNTHETIC_SHAPE,
        "byte_size_probe": {
            "tl2_nbytes_one_expert": one_expert_bytes,
            "active_tl2_nbytes_3d": active_3d_bytes,
            "flat_expert_tl2_nbytes": flat_expert_bytes,
            "underreport_bytes": underreport_bytes,
            "underreport_ratio": underreport_ratio,
        },
        "source_paths": {
            "converter": str(converter_path.relative_to(root)),
            "ggml": str(ggml_path.relative_to(root)),
            "llama": str(llama_path.relative_to(root)),
        },
        "checks": checks,
        "tl2_moe_runtime_ready": not blockers,
        "blockers": blockers,
        "required_next_steps": [
            "Route ggml_mul_mat_id for TL2 through expert-aware BitNet LUT kernels, not generic F32 vec-dot fallback.",
            "Define per-expert tensor-scale or row/group-scale metadata semantics before quality benchmarking.",
            "Validate with a real Qwen2MoE/Kimi GGUF and router/expert-locality benchmarks.",
        ],
    }


def md_bool(value: bool) -> str:
    return "`true`" if value else "`false`"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_markdown(result: dict[str, Any]) -> str:
    check_rows = [
        [
            check["name"],
            "pass" if check["passed"] else "fail",
            check["evidence"].replace("|", "/"),
            check["blocker"].replace("|", "/"),
        ]
        for check in result["checks"]
    ]
    byte_probe = result["byte_size_probe"]
    blocker_rows = [[blocker] for blocker in result["blockers"]] or [["none"]]
    next_rows = [[step] for step in result["required_next_steps"]]
    return "\n\n".join(
        [
            f"# MoE TL2 Runtime Contract Audit, {DATE}",
            (
                "This audit checks whether the current TL2 converter and runtime contracts "
                "can safely carry llama.cpp-style merged expert tensors with shape "
                "`[experts, out, in]`."
            ),
            f"TL2 MoE runtime ready: {md_bool(result['tl2_moe_runtime_ready'])}.",
            "## Checks",
            md_table(["check", "status", "evidence", "blocker"], check_rows),
            "## Byte-Size Probe",
            md_table(
                ["field", "value"],
                [
                    ["synthetic shape", f"`{result['synthetic_shape']}`"],
                    ["one-expert bytes", f"`{byte_probe['tl2_nbytes_one_expert']}`"],
                    ["active 3D bytes", f"`{byte_probe['active_tl2_nbytes_3d']}`"],
                    ["flat expert bytes", f"`{byte_probe['flat_expert_tl2_nbytes']}`"],
                    ["underreport bytes", f"`{byte_probe['underreport_bytes']}`"],
                    ["underreport ratio", f"`{byte_probe['underreport_ratio']:.6f}`"],
                ],
            ),
            "## Blockers",
            md_table(["blocker"], blocker_rows),
            "## Required Next Steps",
            md_table(["step"], next_rows),
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/moe_tl2_runtime_contract_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/moe_tl2_runtime_contract_{DATE}.md"))
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
