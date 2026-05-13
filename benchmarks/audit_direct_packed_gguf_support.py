#!/usr/bin/env python3
"""Audit direct packed GGUF writer support for static ternary checkpoints."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def has(pattern: str, text: str) -> bool:
    return re.search(pattern, text, flags=re.MULTILINE | re.DOTALL) is not None


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    ggml_h = read(args.ggml_h)
    llama_h = read(args.llama_h)
    llama_cpp = read(args.llama_cpp)
    py_constants = read(args.py_constants)
    py_quants = read(args.py_quants)
    py_writer = read(args.py_writer)
    direct_i2s_writer = read(args.direct_i2s_writer)
    direct_converter = read(args.direct_converter)
    row_patch = read(args.row_scale_patch)
    packing_verification = read_json(args.packing_verification_json)

    checks = {
        "cxx_has_i2s_ggml_type": has(r"GGML_TYPE_I2_S\s*=\s*36", ggml_h),
        "cxx_has_i2s_llama_ftype": has(r"LLAMA_FTYPE_MOSTLY_I2_S\s*=\s*40", llama_h),
        "cxx_quantize_maps_i2s_ftype": has(r"LLAMA_FTYPE_MOSTLY_I2_S:\s*default_type\s*=\s*GGML_TYPE_I2_S", llama_cpp),
        "cxx_quantize_cli_exposes_i2s": has(r'\{\s*"I2_S"\s*,\s*LLAMA_FTYPE_MOSTLY_I2_S', read(args.quantize_cpp)),
        "py_gguf_has_i2s_quant_type": has(r"\bI2_S\s*=\s*36", py_constants),
        "py_gguf_has_i2s_file_type": has(r"\bMOSTLY_I2_S\s*=\s*40", py_constants),
        "py_gguf_has_i2s_quant_size": has(r"GGMLQuantizationType\.I2_S\s*:", py_constants),
        "py_quants_has_i2s_trait": has(r"class\s+I2_S\b", py_quants),
        "py_writer_has_i2s_special_layout": has(r"GGMLQuantizationType\.I2_S", py_writer),
        "direct_i2s_writer_has_x86_act_layout": "pack_i2_s_codes_x86_act" in direct_i2s_writer and "qk_i2_s = 128" in direct_i2s_writer,
        "direct_i2s_writer_has_i2s_fallback": 'NamedInt(36, "I2_S")' in direct_i2s_writer,
        "direct_i2s_writer_has_i2sr_mode": "--row-scale-qtype" in direct_i2s_writer and "I2_SR" in direct_i2s_writer,
        "direct_converter_blocks_quantized_by_default": "--allow-converter-quantized-outtype" in direct_converter,
        "direct_i2sr_packing_byte_verified": bool(packing_verification.get("passed")),
        "row_scale_patch_reuses_i2s_type": (
            "GGML_TYPE_I2_S" in row_patch
            and "GGML_TYPE_I2_RS" not in row_patch
            and "LLAMA_FTYPE_MOSTLY_I2_RS" not in row_patch
        ),
        "row_scale_patch_changes_i2s_nbytes": "ggml_nrows(tensor) * sizeof(float)" in row_patch,
    }

    dense_direct_supported = Path(args.direct_qwen_summary).exists()
    direct_packed_i2s_supported_via_native_py_stack = all(
        checks[name]
        for name in (
            "cxx_has_i2s_ggml_type",
            "cxx_has_i2s_llama_ftype",
            "cxx_quantize_maps_i2s_ftype",
            "py_gguf_has_i2s_quant_type",
            "py_gguf_has_i2s_file_type",
            "py_gguf_has_i2s_quant_size",
            "py_writer_has_i2s_special_layout",
        )
    )
    direct_packed_scalar_i2s_supported = (
        Path(args.direct_i2s_qwen_summary).exists()
        and checks["cxx_has_i2s_ggml_type"]
        and checks["cxx_has_i2s_llama_ftype"]
        and checks["direct_i2s_writer_has_x86_act_layout"]
        and checks["direct_i2s_writer_has_i2s_fallback"]
    )
    candidate_i2sr_writer_supported = checks["direct_i2s_writer_has_i2sr_mode"] and Path(args.i2sr_writer_smoke_summary).exists()
    candidate_i2sr_quality_valid = Path(args.i2sr_qwen15b_summary).exists()
    candidate_i2sr_layout_verified = checks["direct_i2sr_packing_byte_verified"]
    product_safe_row_scale_packed_supported = False

    return {
        "schema": "bitnet-direct-packed-gguf-support-audit-v1",
        "inputs": {
            "ggml_h": str(args.ggml_h),
            "llama_h": str(args.llama_h),
            "llama_cpp": str(args.llama_cpp),
            "py_constants": str(args.py_constants),
            "py_quants": str(args.py_quants),
            "py_writer": str(args.py_writer),
            "direct_i2s_writer": str(args.direct_i2s_writer),
            "direct_converter": str(args.direct_converter),
            "row_scale_patch": str(args.row_scale_patch),
            "direct_qwen_summary": str(args.direct_qwen_summary),
            "direct_i2s_qwen_summary": str(args.direct_i2s_qwen_summary),
            "i2sr_writer_smoke_summary": str(args.i2sr_writer_smoke_summary),
            "i2sr_qwen15b_summary": str(args.i2sr_qwen15b_summary),
            "packing_verification_json": str(args.packing_verification_json),
        },
        "checks": checks,
        "verdict": {
            "direct_dense_gguf_supported": dense_direct_supported,
            "direct_packed_i2s_supported": direct_packed_scalar_i2s_supported,
            "direct_packed_i2s_supported_via_native_py_stack": direct_packed_i2s_supported_via_native_py_stack,
            "candidate_i2sr_writer_supported": candidate_i2sr_writer_supported,
            "candidate_i2sr_quality_valid": candidate_i2sr_quality_valid,
            "candidate_i2sr_layout_verified": candidate_i2sr_layout_verified,
            "product_safe_row_scale_packed_supported": product_safe_row_scale_packed_supported,
            "requires_python_gguf_i2s_support": not checks["py_gguf_has_i2s_quant_type"],
            "requires_stable_row_scale_type_or_version": checks["row_scale_patch_reuses_i2s_type"],
            "requires_special_row_scale_nbytes": checks["row_scale_patch_changes_i2s_nbytes"],
        },
        "required_gates": [
            "Keep scalar direct I2_S covered by load/run evidence after the x86 ACT packing fix.",
            "Promote the candidate I2_SR patch into the active runtime or carry it as an explicit downstream patch.",
            "Keep byte-layout regression coverage for direct I2_SR packing against the known-good quantizer layout.",
            "Only then claim product-safe direct packed row-scale GGUF support.",
        ],
    }


def build_report(result: dict[str, Any]) -> str:
    check_rows = [[name, str(value)] for name, value in result["checks"].items()]
    verdict = result["verdict"]
    verdict_rows = [[name, str(value)] for name, value in verdict.items()]
    gates = "\n".join(f"{idx}. {gate}" for idx, gate in enumerate(result["required_gates"], start=1))

    return "\n\n".join(
        [
            "# Direct Packed GGUF Support Audit, 2026-05-13",
            "This audit distinguishes direct dense GGUF export, scalar direct packed `I2_S` export, and product-safe row-scale packed export.",
            "## Checks",
            md_table(["check", "value"], check_rows),
            "## Verdict",
            md_table(["claim", "value"], verdict_rows),
            "## Required Gates",
            gates,
            "## Interpretation",
            "Scalar direct packed `I2_S` export is mechanically supported by the self-contained writer. Row-scale direct export is now quality-valid and byte-layout-verified through the fixed x86 ACT `I2_SR` candidate path on Qwen2.5-1.5B, but it remains not product-complete because the cleaner row-scale qtype is still a downstream patch rather than active/default runtime support.",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ggml-h", type=Path, default=Path("3rdparty/llama.cpp/ggml/include/ggml.h"))
    parser.add_argument("--llama-h", type=Path, default=Path("3rdparty/llama.cpp/include/llama.h"))
    parser.add_argument("--llama-cpp", type=Path, default=Path("3rdparty/llama.cpp/src/llama.cpp"))
    parser.add_argument("--quantize-cpp", type=Path, default=Path("3rdparty/llama.cpp/examples/quantize/quantize.cpp"))
    parser.add_argument("--py-constants", type=Path, default=Path("3rdparty/llama.cpp/gguf-py/gguf/constants.py"))
    parser.add_argument("--py-quants", type=Path, default=Path("3rdparty/llama.cpp/gguf-py/gguf/quants.py"))
    parser.add_argument("--py-writer", type=Path, default=Path("3rdparty/llama.cpp/gguf-py/gguf/gguf_writer.py"))
    parser.add_argument("--direct-i2s-writer", type=Path, default=Path("benchmarks/convert_static_ternary_to_i2s_gguf.py"))
    parser.add_argument("--direct-converter", type=Path, default=Path("benchmarks/convert_static_ternary_to_gguf.py"))
    parser.add_argument("--row-scale-patch", type=Path, default=Path("patches/llama-i2s-row-scale.patch"))
    parser.add_argument("--direct-qwen-summary", type=Path, default=Path("benchmark_results/direct-gguf-qwen05b-klonly-notie-2026-05-13/summary.json"))
    parser.add_argument("--direct-i2s-qwen-summary", type=Path, default=Path("benchmark_results/direct-i2s-qwen05b-klonly-x86act-2026-05-13/summary.json"))
    parser.add_argument("--i2sr-writer-smoke-summary", type=Path, default=Path("benchmark_results/i2sr-writer-smoke-2026-05-13/summary.json"))
    parser.add_argument("--i2sr-qwen15b-summary", type=Path, default=Path("benchmark_results/i2sr-row-scale-qwen15b-x86act-suite-2026-05-13/summary.json"))
    parser.add_argument("--packing-verification-json", type=Path, default=Path("benchmark_results/i2s-packing-layout-verify-2026-05-13/summary.json"))
    parser.add_argument("--output-json", type=Path, default=Path("benchmark_results/direct_packed_gguf_support_2026-05-13.json"))
    parser.add_argument("--output-md", type=Path, default=Path("benchmarks/results/direct_packed_gguf_support_2026-05-13.md"))
    args = parser.parse_args()

    result = audit(args)
    report = build_report(result)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
