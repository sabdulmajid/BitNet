#!/usr/bin/env python3
"""Gate whether TL2 can represent row-scale ternary checkpoints.

The row-scale design audit quantifies the mathematical error from collapsing
learned row scales to one tensor scale. This script checks the implementation
contract: converter metadata, ggml storage, generated TL2 transform metadata,
and benchmark evidence. It intentionally fails until TL2 has explicit
row/group-scale semantics.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = datetime.now(timezone.utc).date().isoformat()


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def first_line(text: str, pattern: str) -> int | None:
    for lineno, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return lineno
    return None


def slice_between(text: str, start: str, end: str) -> str:
    start_index = text.find(start)
    if start_index < 0:
        return ""
    end_index = text.find(end, start_index + len(start))
    if end_index < 0:
        return text[start_index:]
    return text[start_index:end_index]


def latest_existing(pattern: str) -> Path | None:
    paths = [Path(path) for path in glob.glob(pattern)]
    paths = [path for path in paths if path.exists()]
    return sorted(paths)[-1] if paths else None


def strategy_by_name(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = result.get("strategies", [])
    if not isinstance(rows, list):
        return {}
    return {str(row.get("name")): row for row in rows if isinstance(row, dict)}


def best_row_scale_result(design: dict[str, Any]) -> dict[str, Any]:
    results = design.get("results", [])
    if not isinstance(results, list):
        return {}
    row_results = [
        row
        for row in results
        if isinstance(row, dict) and int(row.get("row_scale_tensors") or 0) > 0
    ]
    return row_results[0] if row_results else (results[0] if results and isinstance(results[0], dict) else {})


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def metric(value: Any) -> float | None:
    return float(value) if finite(value) else None


def make_check(name: str, passed: bool, evidence: str, blocker: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": passed,
        "evidence": evidence,
        "blocker": "" if passed else blocker,
    }


def collect_benchmark_evidence(root: Path) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    for path in sorted(root.glob("benchmark_results/**/*tl2*summary.json")):
        data = read_json(path)
        rows = data.get("rows", []) if isinstance(data.get("rows"), list) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name", ""))
            summaries.append(
                {
                    "path": str(path.relative_to(root)),
                    "name": name,
                    "file_mib": row.get("file_mib"),
                    "ppl": row.get("perplexity", {}).get("ppl")
                    if isinstance(row.get("perplexity"), dict)
                    else None,
                    "prefill_tok_s": row.get("bench", {}).get("prefill", {}).get("tok_s")
                    if isinstance(row.get("bench"), dict)
                    else None,
                    "decode_tok_s": row.get("bench", {}).get("decode", {}).get("tok_s")
                    if isinstance(row.get("bench"), dict)
                    else None,
                    "looks_row_scale": "row" in name.lower() or "i2_sr" in name.lower(),
                    "is_tl2": "tl2" in name.lower() or "tl2" in str(path).lower(),
                }
            )
    row_scale_tl2 = [row for row in summaries if row["is_tl2"] and row["looks_row_scale"]]
    finite_quality = [
        row
        for row in row_scale_tl2
        if finite(row.get("ppl")) and finite(row.get("prefill_tok_s")) and finite(row.get("decode_tok_s"))
    ]
    return {
        "tl2_summary_rows": len(summaries),
        "row_scale_tl2_rows": row_scale_tl2,
        "row_scale_tl2_finite_quality_rows": finite_quality,
    }


def build_audit(root: Path, design_json: Path | None, max_existing_tl2_error: float) -> dict[str, Any]:
    converter_path = root / "utils/convert-hf-to-gguf-bitnet.py"
    ggml_path = root / "3rdparty/llama.cpp/ggml/src/ggml.c"
    ggml_h_path = root / "3rdparty/llama.cpp/ggml/include/ggml.h"
    generated_kernel_path = root / "include/bitnet-lut-kernels.h"
    preset_kernel_path = root / "preset_kernels/bitnet_b1_58-large/bitnet-lut-kernels-tl2.h"
    bitnet_kernel_path = generated_kernel_path if "ggml_bitnet_transform_tensor" in read_text(generated_kernel_path) else preset_kernel_path

    converter = read_text(converter_path)
    ggml = read_text(ggml_path)
    ggml_h = read_text(ggml_h_path)
    bitnet_kernel = read_text(bitnet_kernel_path)

    transform_to_tl2 = slice_between(converter, "def transform_to_tl2", "def read_model_config")
    write_tensors = slice_between(converter, "def write_tensors", "def set_gguf_parameters")
    ggml_nbytes = slice_between(ggml, "size_t ggml_nbytes", "size_t ggml_nbytes_pad")
    bitnet_transform = slice_between(bitnet_kernel, "void ggml_bitnet_transform_tensor", "int ggml_bitnet_get_type_bits")

    design_path = design_json or latest_existing(str(root / "benchmark_results/tl2_row_scale_design_*.json"))
    design = read_json(design_path) if design_path else {}
    design_result = best_row_scale_result(design)
    strategies = strategy_by_name(design_result)
    current = strategies.get("current_tl2_tensor_max_fp32", {})
    optimal = strategies.get("tensor_l2_optimal_fp32", {})
    group32 = strategies.get("group32_l2_optimal_fp16", {})
    row16 = strategies.get("row_exact_fp16", {})
    current_error = metric(current.get("expected_relative_output_rms_error"))
    row16_error = metric(row16.get("expected_relative_output_rms_error"))

    transform_accepts_scale = "scale" in transform_to_tl2.split(":", 1)[0]
    transform_uses_scalar_max = "np.max(np.abs(x))" in transform_to_tl2
    transform_returns_single_scale = "return res, scale" in transform_to_tl2
    converter_passes_scale_metadata = "transform_to_tl2(data," in write_tensors or "transform_to_tl2(data_torch" in write_tensors
    converter_has_learned_scale_map = "scale_map" in write_tensors and "weight_scale" in write_tensors
    writer_emits_scale_sidecar = 'new_name + "_scale"' in write_tensors
    nbytes_has_row_scale_sidecar = "GGML_TYPE_TL2" in ggml_nbytes and ("lut_scales_size" in ggml_nbytes or "row_scale" in ggml_nbytes)
    transform_single_scale = "const int lut_scales_size = 1" in bitnet_transform and "i2_scales[0]" in bitnet_transform
    transform_uses_all_rows = "ggml_nrows(tensor)" in bitnet_transform or "tensor->ne[2]" in bitnet_transform
    has_dedicated_row_tl2_type = "TL2_R" in ggml_h or "TL2_SR" in ggml_h or "TL2_G" in ggml_h
    has_i2sr_fallback = "GGML_TYPE_I2_SR" in ggml_h and "LLAMA_FTYPE_MOSTLY_I2_SR" in read_text(root / "3rdparty/llama.cpp/include/llama.h")
    benchmark_evidence = collect_benchmark_evidence(root)
    has_row_tl2_quality = bool(benchmark_evidence["row_scale_tl2_finite_quality_rows"])

    checks = [
        make_check(
            "Existing TL2 one-scale error is below product threshold",
            current_error is not None and current_error <= max_existing_tl2_error,
            f"design_json={design_path}; current_error={current_error}; threshold={max_existing_tl2_error}",
            "Existing TL2 collapses learned row scales to one scale and exceeds the allowed relative output RMS error.",
        ),
        make_check(
            "TL2 converter accepts learned row/group scale metadata",
            transform_accepts_scale and converter_passes_scale_metadata,
            (
                f"transform_to_tl2_line={first_line(converter, 'def transform_to_tl2')}; "
                f"accepts_scale={transform_accepts_scale}; passes_scale_metadata={converter_passes_scale_metadata}; "
                f"has_scale_map={converter_has_learned_scale_map}"
            ),
            "`transform_to_tl2` is data-only and recomputes a scalar scale instead of carrying checkpoint row/group scales.",
        ),
        make_check(
            "TL2 converter no longer recomputes one scalar max scale",
            not transform_uses_scalar_max and not transform_returns_single_scale,
            (
                f"uses_np_max_abs={transform_uses_scalar_max}; returns_single_scale={transform_returns_single_scale}; "
                f"emits_scale_sidecar={writer_emits_scale_sidecar}"
            ),
            "The converter still derives one scalar `np.max(abs(x))` scale and writes a single sidecar scale tensor.",
        ),
        make_check(
            "ggml TL2 storage accounts for row/group-scale sidecar",
            nbytes_has_row_scale_sidecar or has_dedicated_row_tl2_type,
            (
                f"ggml_nbytes_line={first_line(ggml, 'size_t ggml_nbytes')}; "
                f"nbytes_has_row_scale_sidecar={nbytes_has_row_scale_sidecar}; "
                f"dedicated_row_tl2_type={has_dedicated_row_tl2_type}"
            ),
            "`GGML_TYPE_TL2` storage has no explicit row/group-scale count or dedicated row-scale TL2 qtype.",
        ),
        make_check(
            "TL2 transform metadata is row/group-scale aware",
            transform_uses_all_rows and not transform_single_scale,
            (
                f"bitnet_transform_line={first_line(bitnet_kernel, 'void ggml_bitnet_transform_tensor')}; "
                f"kernel_source={bitnet_kernel_path.relative_to(root)}; "
                f"uses_all_rows={transform_uses_all_rows}; single_scale_metadata={transform_single_scale}"
            ),
            "`ggml_bitnet_transform_tensor` still builds metadata for one scale, not one scale per row or row group.",
        ),
        make_check(
            "A compatibility-safe row-scale runtime path exists",
            has_i2sr_fallback,
            f"GGML_TYPE_I2_SR={has_i2sr_fallback}",
            "No stable fallback row-scale qtype is present.",
        ),
        make_check(
            "Row-scale TL2 has quality and speed benchmark evidence",
            has_row_tl2_quality,
            f"row_scale_tl2_rows={len(benchmark_evidence['row_scale_tl2_rows'])}; finite_quality_rows={len(benchmark_evidence['row_scale_tl2_finite_quality_rows'])}",
            "No row-scale TL2 GGUF has passed PPL, throughput, and RSS-style evidence.",
        ),
    ]
    blockers = [check["blocker"] for check in checks if not check["passed"]]
    return {
        "schema": "tl2-row-scale-runtime-contract-v1",
        "date": DATE,
        "source_paths": {
            "converter": str(converter_path.relative_to(root)),
            "ggml": str(ggml_path.relative_to(root)),
            "ggml_h": str(ggml_h_path.relative_to(root)),
            "bitnet_kernel": str(bitnet_kernel_path.relative_to(root)),
            "design_json": str(design_path.relative_to(root)) if design_path else None,
        },
        "math": {
            "label": design_result.get("label"),
            "row_scale_tensors": design_result.get("row_scale_tensors"),
            "current_tl2_tensor_max_error": current_error,
            "best_one_scale_error": metric(optimal.get("expected_relative_output_rms_error")),
            "group32_fp16_error": metric(group32.get("expected_relative_output_rms_error")),
            "row_fp16_error": row16_error,
            "row_fp16_scale_mib": metric(row16.get("scale_mib_fp16")),
            "max_existing_tl2_error": max_existing_tl2_error,
        },
        "checks": checks,
        "benchmark_evidence": benchmark_evidence,
        "tl2_row_scale_runtime_ready": not blockers,
        "blockers": blockers,
        "required_next_steps": [
            "Add a compatibility-safe TL2 row/group-scale qtype or explicit TL2 metadata version; do not overload current single-scale TL2 silently.",
            "Teach the converter to pass learned tensor/row/group scales into TL2 packing instead of recomputing one `max(abs(W))` scale.",
            "Extend GGUF/ggml byte-size semantics so packed TL2 data carries the exact number of fp16/fp32 row or row-group scales.",
            "Update generated TL2 transform metadata and kernels to index the correct scale for each output row or row group.",
            "Run dense Qwen PPL, lm-eval/task quality, llama-bench throughput, and RSS benchmarks before enabling any product claim.",
        ],
    }


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "`true`" if value else "`false`"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def render_markdown(result: dict[str, Any]) -> str:
    check_rows = [
        [
            check["name"],
            "pass" if check["passed"] else "fail",
            check["evidence"],
            check["blocker"],
        ]
        for check in result["checks"]
    ]
    math_rows = [[key, fmt(value)] for key, value in result["math"].items()]
    blockers = [[blocker] for blocker in result["blockers"]] or [["none"]]
    next_steps = [[step] for step in result["required_next_steps"]]
    row_tl2 = result["benchmark_evidence"]["row_scale_tl2_rows"]
    evidence_rows = [
        [
            row["path"],
            row["name"],
            fmt(row.get("file_mib")),
            fmt(row.get("ppl")),
            fmt(row.get("prefill_tok_s")),
            fmt(row.get("decode_tok_s")),
        ]
        for row in row_tl2[:10]
    ] or [["none", "-", "-", "-", "-", "-"]]
    return "\n\n".join(
        [
            f"# TL2 Row-Scale Runtime Contract, {result['date']}",
            (
                "This gate checks whether TL2 can safely carry row-scale ternary "
                "checkpoints. It is stricter than the design audit: passing requires "
                "both low mathematical error and concrete converter/runtime/benchmark evidence."
            ),
            f"TL2 row-scale runtime ready: {fmt(result['tl2_row_scale_runtime_ready'])}.",
            "## Math Summary",
            md_table(["field", "value"], math_rows),
            "## Checks",
            md_table(["check", "status", "evidence", "blocker"], check_rows),
            "## Row-Scale TL2 Benchmark Evidence",
            md_table(["summary", "name", "file MiB", "PPL", "prefill tok/s", "decode tok/s"], evidence_rows),
            "## Blockers",
            md_table(["blocker"], blockers),
            "## Required Implementation Steps",
            md_table(["step"], next_steps),
            "## Verdict",
            (
                "Current TL2 is not a supported path for the strongest row-scale "
                "checkpoint. The supported packed row-scale path remains `I2_SR`; TL2 "
                "needs a metadata and kernel extension before it can be benchmarked as "
                "a quality-preserving alternative."
            ),
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--design-json", type=Path, default=None)
    parser.add_argument("--max-existing-tl2-error", type=float, default=0.01)
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/tl2_row_scale_runtime_contract_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/tl2_row_scale_runtime_contract_{DATE}.md"))
    args = parser.parse_args()

    root = args.root.resolve()
    result = build_audit(root, args.design_json, args.max_existing_tl2_error)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_markdown(result)
    args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
