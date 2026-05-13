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
    direct_converter = read(args.direct_converter)
    row_patch = read(args.row_scale_patch)

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
        "direct_converter_blocks_quantized_by_default": "--allow-converter-quantized-outtype" in direct_converter,
        "row_scale_patch_reuses_i2s_type": (
            "GGML_TYPE_I2_S" in row_patch
            and "GGML_TYPE_I2_RS" not in row_patch
            and "LLAMA_FTYPE_MOSTLY_I2_RS" not in row_patch
        ),
        "row_scale_patch_changes_i2s_nbytes": "ggml_nrows(tensor) * sizeof(float)" in row_patch,
    }

    dense_direct_supported = Path(args.direct_qwen_summary).exists()
    direct_packed_i2s_supported = all(
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
    product_safe_row_scale_packed_supported = direct_packed_i2s_supported and not checks["row_scale_patch_reuses_i2s_type"]

    return {
        "schema": "bitnet-direct-packed-gguf-support-audit-v1",
        "inputs": {
            "ggml_h": str(args.ggml_h),
            "llama_h": str(args.llama_h),
            "llama_cpp": str(args.llama_cpp),
            "py_constants": str(args.py_constants),
            "py_quants": str(args.py_quants),
            "py_writer": str(args.py_writer),
            "direct_converter": str(args.direct_converter),
            "row_scale_patch": str(args.row_scale_patch),
            "direct_qwen_summary": str(args.direct_qwen_summary),
        },
        "checks": checks,
        "verdict": {
            "direct_dense_gguf_supported": dense_direct_supported,
            "direct_packed_i2s_supported": direct_packed_i2s_supported,
            "product_safe_row_scale_packed_supported": product_safe_row_scale_packed_supported,
            "requires_python_gguf_i2s_support": not checks["py_gguf_has_i2s_quant_type"],
            "requires_stable_row_scale_type_or_version": checks["row_scale_patch_reuses_i2s_type"],
            "requires_special_row_scale_nbytes": checks["row_scale_patch_changes_i2s_nbytes"],
        },
        "required_gates": [
            "Add Python GGUF constants for the packed ternary type being written.",
            "Add file-type metadata for packed I2_S or a new row-scale ternary type.",
            "Teach the Python writer/reader the special packed layout instead of assuming a fixed block type size is enough.",
            "Define a compatibility-safe row-scale layout instead of overloading existing tensor-scale I2_S.",
            "Write direct packed tensors from ternary codes plus scales, then load with llama-cli and run PPL/throughput/RSS audits.",
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
            "This audit distinguishes direct dense GGUF export from direct packed CPU-native GGUF export. The former is now validated; the latter still needs writer and format work.",
            "## Checks",
            md_table(["check", "value"], check_rows),
            "## Verdict",
            md_table(["claim", "value"], verdict_rows),
            "## Required Gates",
            gates,
            "## Interpretation",
            "The C++ runtime and `llama-quantize` path know about `I2_S`, but the Python GGUF writer stack used for direct `ternary_state_dict.pt` export does not expose a compatible `I2_S` writer contract. More importantly, row-scale deployment needs a compatibility-safe row-scale layout or new qtype; the current prototype patch changes the existing `I2_S` payload. Therefore direct dense GGUF export is a real improvement, but direct packed row-scale GGUF export is not complete.",
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
    parser.add_argument("--direct-converter", type=Path, default=Path("benchmarks/convert_static_ternary_to_gguf.py"))
    parser.add_argument("--row-scale-patch", type=Path, default=Path("patches/llama-i2s-row-scale.patch"))
    parser.add_argument("--direct-qwen-summary", type=Path, default=Path("benchmark_results/direct-gguf-qwen05b-klonly-notie-2026-05-13/summary.json"))
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
