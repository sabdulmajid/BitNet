#!/usr/bin/env python3
"""Build a source-mapped implementation plan for row-scale TL2 support.

This is not a success gate.  It converts the failing TL2 row-scale runtime
contract into an actionable engineering plan with required source files,
verification artifacts, and exit criteria.  The point is to keep the remaining
objective blocker concrete and auditable.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value != 0 and (abs(value) >= 1000 or abs(value) < 0.0001):
            return f"{value:.6e}"
        return f"{value:.6f}"
    if isinstance(value, list):
        return "[" + ", ".join(fmt(item) for item in value) + "]"
    return str(value).replace("|", "\\|")


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(item) for item in row) + " |")
    return "\n".join(lines)


def failed_checks(contract: dict[str, Any]) -> list[dict[str, Any]]:
    checks = contract.get("checks", [])
    return [check for check in checks if isinstance(check, dict) and not check.get("passed")]


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    contract = read_json(args.contract_json)
    math = contract.get("math", {}) if isinstance(contract.get("math"), dict) else {}
    failed = failed_checks(contract)
    steps = [
        {
            "step": "1. Define a row/group-scale TL2 representation",
            "files": [
                "3rdparty/llama.cpp/ggml/include/ggml.h",
                "3rdparty/llama.cpp/gguf-py/gguf/constants.py",
            ],
            "work": (
                "Add either a dedicated row-scale TL2 qtype or explicit metadata for the number and layout of "
                "learned scales. Reusing GGML_TYPE_TL2 without changing byte-size semantics is unsafe."
            ),
            "exit_gate": "ggml row-size/nbytes can represent packed codes plus row/group scales without undercounting bytes.",
        },
        {
            "step": "2. Make the converter sidecar-aware for TL2",
            "files": ["utils/convert-hf-to-gguf-bitnet.py", "benchmarks/convert_static_ternary_to_i2s_gguf.py"],
            "work": (
                "Thread checkpoint `weight_scale` tensors into TL2 export instead of recomputing a single "
                "`np.max(abs(x))` scale. Reject ambiguous row-scale exports unless the new qtype/metadata is active."
            ),
            "exit_gate": "A row-scale checkpoint exports learned scales byte-for-byte and scalar TL2 continues to export unchanged.",
        },
        {
            "step": "3. Regenerate TL2 transform metadata for multiple scales",
            "files": [
                "utils/codegen_tl2.py",
                "preset_kernels/bitnet_b1_58-large/bitnet-lut-kernels-tl2.h",
                "include/bitnet-lut-kernels.h",
            ],
            "work": (
                "Replace the one-scale `lut_scales_size = 1` contract with row/group-scale metadata and regenerate "
                "kernels. Generated qgemm must index the output row or row group, not `Scales[0]`."
            ),
            "exit_gate": "Static audit shows no generated row-scale path multiplies by `Scales[0]` for every output row.",
        },
        {
            "step": "4. Offset scales in x86 TL2 matmul dispatch",
            "files": ["3rdparty/llama.cpp/ggml/src/ggml.c"],
            "work": (
                "Pass the correct scale slice into each generated qgemm call, or pass a base pointer plus row offset. "
                "The dispatch must match the kernel's row/group-scale indexing contract."
            ),
            "exit_gate": "Layer-level relative output RMS error falls below 0.01 on the row-scale Qwen1.5B design audit.",
        },
        {
            "step": "5. Expose learned scale sidecars in the BitNet/Qwen loader",
            "files": ["3rdparty/llama.cpp/src/llama.cpp"],
            "work": (
                "Load learned scale tensors for the Qwen-compatible BitNet graph when the model uses row/group-scale TL2. "
                "The current TL2 path hides scale sidecars behind packed tensor metadata."
            ),
            "exit_gate": "Converted GGUF loads without missing scale tensors and without silently falling back to scalar TL2.",
        },
        {
            "step": "6. Run CPU quality, speed, and RSS gates",
            "files": [
                "benchmarks/audit_tl2_row_scale_runtime_contract.py",
                "benchmarks/audit_tl2_negative_result.py",
                "benchmarks/build_qwen_side_by_side.py",
            ],
            "work": (
                "Convert the row-scale Qwen1.5B checkpoint, run perplexity/throughput/RSS on the Xeon, and compare "
                "against I2_SR, TQ2_0, Q4_K_M, and FP16."
            ),
            "exit_gate": (
                "Row-scale TL2 has finite PPL near the I2_SR row-scale artifact, benchmark rows include prefill/decode "
                "throughput and RSS, and the runtime contract audit passes."
            ),
        },
    ]
    blockers = [
        {
            "check": check.get("name"),
            "evidence": check.get("evidence"),
            "blocker": check.get("blocker"),
        }
        for check in failed
    ]
    return {
        "schema": "tl2-row-scale-implementation-plan-v1",
        "date": DATE,
        "contract_json": str(args.contract_json),
        "design_json": str(args.design_json),
        "current_ready": contract.get("tl2_row_scale_runtime_ready"),
        "failed_check_count": len(failed),
        "current_one_scale_error": math.get("current_tl2_tensor_max_error"),
        "exact_row_fp16_error": math.get("row_fp16_error"),
        "row_scale_storage_mib": math.get("row_fp16_scale_mib"),
        "blockers": blockers,
        "implementation_steps": steps,
        "verdict": (
            "Do not productize TL2 for learned row-scale checkpoints yet. The required work is a new row/group-scale "
            "runtime contract, not a benchmark rerun."
        ),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    blocker_rows = [
        [row["check"], row["evidence"], row["blocker"]]
        for row in summary["blockers"]
    ]
    step_rows = [
        [row["step"], ", ".join(row["files"]), row["work"], row["exit_gate"]]
        for row in summary["implementation_steps"]
    ]
    return "\n\n".join(
        [
            f"# TL2 Row-Scale Implementation Plan, {summary['date']}",
            summary["verdict"],
            "## Current Math",
            md_table(
                ["field", "value"],
                [
                    ["contract ready", summary["current_ready"]],
                    ["failed checks", summary["failed_check_count"]],
                    ["current one-scale relative output RMS error", summary["current_one_scale_error"]],
                    ["exact row-scale FP16 relative output RMS error", summary["exact_row_fp16_error"]],
                    ["row-scale storage overhead MiB", summary["row_scale_storage_mib"]],
                ],
            ),
            "## Failed Contract Checks",
            md_table(["check", "evidence", "blocker"], blocker_rows),
            "## Patch Sequence",
            md_table(["step", "files", "work", "exit gate"], step_rows),
            "## Completion Criteria",
            (
                "This blocker is closed only when the row-scale TL2 runtime contract audit passes, a row-scale TL2 "
                "GGUF has finite quality/speed/RSS evidence on the Xeon, and the side-by-side table compares that "
                "artifact against I2_SR and Q4_K_M. Until then, I2_SR remains the deployable row-scale path."
            ),
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--contract-json", type=Path, default=Path(f"benchmark_results/tl2_row_scale_runtime_contract_{DATE}.json"))
    parser.add_argument("--design-json", type=Path, default=Path("benchmark_results/tl2_row_scale_design_2026-05-13.json"))
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/tl2_row_scale_implementation_plan_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/tl2_row_scale_implementation_plan_{DATE}.md"))
    args = parser.parse_args()

    root = args.repo_root.resolve()
    for field in ("contract_json", "design_json", "output_json", "output_md"):
        path = getattr(args, field)
        if isinstance(path, Path) and not path.is_absolute():
            setattr(args, field, root / path)

    summary = build_summary(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(summary), encoding="utf-8")
    print(render_markdown(summary))


if __name__ == "__main__":
    main()
