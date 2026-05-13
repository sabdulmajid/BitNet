#!/usr/bin/env python3
"""Audit whether the row-scale I2_S prototype is format-safe enough to productize."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_row(summary: dict[str, Any], *, kind: str | None = None, name: str | None = None) -> dict[str, Any]:
    for row in summary.get("rows", []):
        if not isinstance(row, dict):
            continue
        if kind is not None and row.get("kind") == kind:
            return row
        if name is not None and row.get("name") == name:
            return row
    raise KeyError(kind or name or "<unspecified>")


def ppl(row: dict[str, Any]) -> float:
    value = row.get("perplexity", {}).get("ppl") if isinstance(row.get("perplexity"), dict) else None
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"row {row.get('name')} has non-finite PPL {value!r}")
    return float(value)


def tok_s(row: dict[str, Any], mode: str) -> float | None:
    value = row.get("bench", {}).get(mode, {}).get("tok_s") if isinstance(row.get("bench"), dict) else None
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def fmt(value: float | int | str | None, digits: int = 4) -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.{digits}f}"
    return "" if value is None else str(value)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    default_summary = read_json(args.default_summary)
    prototype_summary = read_json(args.prototype_summary)

    tq2 = find_row(default_summary, kind="static_ternary_tq2")
    default_i2s = find_row(default_summary, kind="static_ternary_i2s_single_thread_quant")
    prototype_i2s = find_row(prototype_summary, kind="static_ternary_i2s_row_scale_prototype_heap_tmp_fix")

    default_ref_ppl = ppl(tq2)
    default_i2s_ppl = ppl(default_i2s)
    prototype_i2s_ppl = ppl(prototype_i2s)

    patch_text = read_text(args.patch)
    mad_text = read_text(args.current_mad_source)
    ggml_text = read_text(args.current_ggml_source)

    patch_overloads_i2s = (
        "GGML_TYPE_I2_S" in patch_text
        and "ggml_nrows(tensor) * sizeof(float)" in patch_text
        and "scale[src0_start + row]" in patch_text
        and "GGML_TYPE_I2_RS" not in patch_text
        and "LLAMA_FTYPE_MOSTLY_I2_RS" not in patch_text
    )
    current_source_tensor_scale_i2s = (
        "scale_ptr[0]" in mad_text
        and "i2_scale" in mad_text
        and "*scale" in ggml_text
        and "ggml_nrows(tensor) * sizeof(float)" not in ggml_text
    )

    default_ratio = default_i2s_ppl / default_ref_ppl
    prototype_ratio = prototype_i2s_ppl / default_ref_ppl
    patched_vs_default_reduction = default_i2s_ppl / prototype_i2s_ppl
    file_mib_delta = float(prototype_i2s.get("file_mib", 0.0)) - float(default_i2s.get("file_mib", 0.0))

    return {
        "default_summary": str(args.default_summary),
        "prototype_summary": str(args.prototype_summary),
        "patch": str(args.patch),
        "current_mad_source": str(args.current_mad_source),
        "current_ggml_source": str(args.current_ggml_source),
        "metrics": {
            "reference_tq2_ppl": default_ref_ppl,
            "default_row_scale_i2s_ppl": default_i2s_ppl,
            "prototype_row_scale_i2s_ppl": prototype_i2s_ppl,
            "default_row_scale_i2s_to_tq2_ppl_ratio": default_ratio,
            "prototype_row_scale_i2s_to_tq2_ppl_ratio": prototype_ratio,
            "patched_vs_default_i2s_ppl_reduction": patched_vs_default_reduction,
            "default_i2s_file_mib": float(default_i2s.get("file_mib", 0.0)),
            "prototype_i2s_file_mib": float(prototype_i2s.get("file_mib", 0.0)),
            "prototype_minus_default_file_mib": file_mib_delta,
            "prototype_prompt_tok_s": tok_s(prototype_i2s, "prefill"),
            "prototype_decode_tok_s": tok_s(prototype_i2s, "decode"),
        },
        "code_audit": {
            "patch_overloads_existing_i2s_type": patch_overloads_i2s,
            "current_source_is_tensor_scale_i2s": current_source_tensor_scale_i2s,
            "patch_defines_new_rowscale_type": "GGML_TYPE_I2_RS" in patch_text or "LLAMA_FTYPE_MOSTLY_I2_RS" in patch_text,
            "patch_changes_nbytes_for_i2s": "ggml_nrows(tensor) * sizeof(float)" in patch_text,
            "patch_indexes_per_row_scale": "scale[src0_start + row]" in patch_text or "scale[scale_row]" in patch_text,
        },
        "verdict": {
            "row_scale_i2s_physically_possible": prototype_ratio <= args.prototype_max_ppl_ratio,
            "default_i2s_layout_fails_row_scale": default_ratio >= args.default_failure_min_ratio,
            "current_patch_is_product_format_safe": False,
            "stable_new_format_required": patch_overloads_i2s,
            "direct_ternary_gguf_writer_still_required": True,
        },
        "thresholds": {
            "prototype_max_ppl_ratio": args.prototype_max_ppl_ratio,
            "default_failure_min_ratio": args.default_failure_min_ratio,
        },
    }


def build_report(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    code = result["code_audit"]
    verdict = result["verdict"]
    rows = [
        [
            "row-scale TQ2_0 reference",
            fmt(metrics["reference_tq2_ppl"]),
            "1.0000",
            "",
            "",
        ],
        [
            "default row-scale I2_S",
            fmt(metrics["default_row_scale_i2s_ppl"]),
            fmt(metrics["default_row_scale_i2s_to_tq2_ppl_ratio"], 2),
            fmt(metrics["default_i2s_file_mib"], 1),
            "fails row-scale scales",
        ],
        [
            "patched row-scale I2_S prototype",
            fmt(metrics["prototype_row_scale_i2s_ppl"]),
            fmt(metrics["prototype_row_scale_i2s_to_tq2_ppl_ratio"], 4),
            fmt(metrics["prototype_i2s_file_mib"], 1),
            "preserves row-scale quality",
        ],
    ]

    code_rows = [
        ["current source stores one I2_S scale", str(code["current_source_is_tensor_scale_i2s"])],
        ["patch changes I2_S nbytes", str(code["patch_changes_nbytes_for_i2s"])],
        ["patch indexes per-row scales", str(code["patch_indexes_per_row_scale"])],
        ["patch defines a new row-scale qtype", str(code["patch_defines_new_rowscale_type"])],
        ["patch overloads existing I2_S type", str(code["patch_overloads_existing_i2s_type"])],
    ]

    lines = [
        "# I2_S Row-Scale Format Compatibility Audit, 2026-05-13",
        "",
        "This audit separates two claims that must not be conflated: row-scale packed ternary execution is physically possible, but the current prototype is not a stable product format because it reuses the existing `I2_S` type while changing the tensor payload layout.",
        "",
        "## Measured Evidence",
        "",
        md_table(["artifact", "fixed PPL", "ratio vs TQ2_0", "file MiB", "interpretation"], rows),
        "",
        "## Code-Level Format Evidence",
        "",
        md_table(["check", "value"], code_rows),
        "",
        "## Verdict",
        "",
        f"- Row-scale packed `I2_S` is physically possible: `{verdict['row_scale_i2s_physically_possible']}`.",
        f"- The default `I2_S` layout fails row-scale checkpoints: `{verdict['default_i2s_layout_fails_row_scale']}`.",
        f"- The current patch is product-format safe: `{verdict['current_patch_is_product_format_safe']}`.",
        f"- A compatibility-safe new GGUF quantization type or explicit versioned layout is required: `{verdict['stable_new_format_required']}`.",
        f"- Direct `ternary_state_dict.pt` GGUF writing remains required: `{verdict['direct_ternary_gguf_writer_still_required']}`.",
        "",
        "## Production Gate",
        "",
        "Do not market the row-scale `I2_S` result as default/upstream `I2_S`. The benchmark proves the missing scale semantics and CPU kernel are feasible. Productization requires a new row-scale-aware GGUF type, writer, reader, `ggml_nbytes` accounting, matmul/get-rows kernels, backward-compatibility tests for tensor-scale `I2_S`, and direct export from `ternary_state_dict.pt` without materializing dense F16 weights.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--default-summary", type=Path, default=Path("benchmark_results/gguf-qwen15b-klonly-row-notiehead-suite/summary.json"))
    parser.add_argument("--prototype-summary", type=Path, default=Path("benchmark_results/gguf-qwen15b-row-i2s-heapfix-confirm/summary.json"))
    parser.add_argument("--patch", type=Path, default=Path("patches/llama-i2s-row-scale.patch"))
    parser.add_argument("--current-mad-source", type=Path, default=Path("src/ggml-bitnet-mad.cpp"))
    parser.add_argument("--current-ggml-source", type=Path, default=Path("3rdparty/llama.cpp/ggml/src/ggml.c"))
    parser.add_argument("--prototype-max-ppl-ratio", type=float, default=1.01)
    parser.add_argument("--default-failure-min-ratio", type=float, default=10.0)
    parser.add_argument("--output-json", type=Path, default=Path("benchmark_results/i2s_row_scale_format_audit_2026-05-13.json"))
    parser.add_argument("--output-md", type=Path, default=Path("benchmarks/results/i2s_row_scale_format_audit_2026-05-13.md"))
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
