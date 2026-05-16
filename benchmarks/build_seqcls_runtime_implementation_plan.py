#!/usr/bin/env python3
"""Build an implementation plan for native packed sequence classification.

The repo currently has strict GLUE quality in PyTorch and a packed I2_SR
backbone plus Python score-head sidecar. This report turns that gap into a
source-owned runtime plan with measurable exit gates.
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


def nested(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    gap = read_json(args.runtime_gap_json)
    steps = [
        {
            "step": "1. Define classifier GGUF contract",
            "files": [
                "benchmarks/convert_static_ternary_to_i2s_gguf.py",
                "3rdparty/llama.cpp/gguf-py/gguf/constants.py",
                "3rdparty/llama.cpp/src/llama.cpp",
            ],
            "work": (
                "Persist score-head tensors and label metadata in GGUF instead of an NPZ sidecar. "
                "Record num_labels, label order, pooling policy, and problem type."
            ),
            "exit_gate": "GGUF reader lists classifier head tensors and metadata without requiring sidecar files.",
        },
        {
            "step": "2. Add native Qwen sequence-classification graph path",
            "files": [
                "3rdparty/llama.cpp/src/llama.cpp",
                "3rdparty/llama.cpp/src/llama-model.cpp",
                "3rdparty/llama.cpp/src/llama-graph.cpp",
            ],
            "work": (
                "Reuse the audited `bitnet-qwen` decoder semantics, then pool the last non-padding token "
                "and apply the dense score head in the runtime."
            ),
            "exit_gate": "Single-sample logits match PyTorch/sidecar logits within a strict relative RMS threshold.",
        },
        {
            "step": "3. Implement a CPU GLUE classifier evaluator",
            "files": [
                "benchmarks/benchmark_seqcls_i2sr_sidecar_cpu.py",
                "benchmarks/benchmark_bitdistill_glue_cpu.py",
                "3rdparty/llama.cpp/examples",
            ],
            "work": (
                "Replace the Python sidecar loop with native GGUF classifier inference and preserve the "
                "same prompt/tokenization contract used by PyTorch validation."
            ),
            "exit_gate": "Full MNLI validation runs from one native GGUF artifact and reports accuracy, RSS, and examples/sec.",
        },
        {
            "step": "4. Prove batching parity",
            "files": [
                "benchmarks/audit_seqcls_i2sr_sidecar_batching.py",
                "benchmarks/audit_seqcls_runtime_gap.py",
            ],
            "work": (
                "Native batching must preserve predictions relative to batch size 1. Existing separator "
                "batching changed `3/64` sidecar predictions, so it is not a safe semantic reference."
            ),
            "exit_gate": "Batch-size 1 and production batch size have identical predictions on a fixed validation subset.",
        },
        {
            "step": "5. Promote quality/runtime gate",
            "files": [
                "benchmarks/audit_seqcls_runtime_gap.py",
                "benchmarks/audit_benchmark_coverage.py",
                "README.md",
            ],
            "work": (
                "Only after native logits, full-split accuracy, RSS, and throughput are present should the "
                "sequence-classification product status move from prototype to deployed artifact."
            ),
            "exit_gate": "same_artifact_quality_cpu_ready=true and coverage gate passes with native classifier rows.",
        },
    ]
    return {
        "schema": "seqcls-runtime-implementation-plan-v1",
        "date": DATE,
        "runtime_gap_json": str(args.runtime_gap_json),
        "status": gap.get("status"),
        "same_artifact_quality_cpu_ready": gap.get("same_artifact_quality_cpu_ready"),
        "seqcls_configs": nested(gap, "sequence_classification", "configs"),
        "seqcls_causal_export_compatible": nested(gap, "sequence_classification", "causal_export_compatible"),
        "sidecar_status": nested(gap, "seqcls_sidecar_smoke", "status"),
        "sidecar_sampled_accuracy": nested(gap, "seqcls_sidecar_cpu_benchmark", "accuracy"),
        "sidecar_agreement": nested(gap, "seqcls_sidecar_cpu_benchmark", "agreement_with_saved_pytorch_predictions"),
        "hidden_relative_rms": nested(gap, "seqcls_hidden_contract", "hidden_relative_rms"),
        "hidden_cosine": nested(gap, "seqcls_hidden_contract", "hidden_cosine"),
        "bitnet_qwen_available": nested(gap, "seqcls_arch_contract", "bitnet_qwen_available"),
        "implementation_steps": steps,
        "ready_to_productize": False,
        "verdict": (
            "Do not call the sequence-classification artifact deployed yet. The sidecar proves the path is plausible, "
            "but native GGUF score-head metadata, runtime pooling/head execution, full validation accuracy, RSS, "
            "and batching parity are still required."
        ),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    headline = [
        ["status", summary["status"]],
        ["same artifact quality+CPU ready", summary["same_artifact_quality_cpu_ready"]],
        ["seqcls configs", summary["seqcls_configs"]],
        ["seqcls causal-export compatible", summary["seqcls_causal_export_compatible"]],
        ["sidecar status", summary["sidecar_status"]],
        ["sidecar sampled accuracy", summary["sidecar_sampled_accuracy"]],
        ["sidecar agreement with PyTorch predictions", summary["sidecar_agreement"]],
        ["hidden relative RMS", summary["hidden_relative_rms"]],
        ["hidden cosine", summary["hidden_cosine"]],
        ["bitnet-qwen graph available", summary["bitnet_qwen_available"]],
        ["ready to productize", summary["ready_to_productize"]],
    ]
    steps = [
        [row["step"], ", ".join(row["files"]), row["work"], row["exit_gate"]]
        for row in summary["implementation_steps"]
    ]
    return "\n\n".join(
        [
            f"# Sequence-Classification Runtime Implementation Plan, {summary['date']}",
            summary["verdict"],
            "## Current Evidence",
            md_table(["field", "value"], headline),
            "## Source-Owned Plan",
            md_table(["step", "files", "work", "exit gate"], steps),
            "## Completion Criteria",
            (
                "This blocker closes only when a single packed classifier artifact carries the decoder, "
                "score head, labels, and pooling contract; produces native CPU logits matching PyTorch; "
                "runs full GLUE validation; and reports accuracy, RSS, and throughput from the same GGUF."
            ),
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-gap-json", type=Path, default=Path(f"benchmark_results/seqcls_runtime_gap_{DATE}.json"))
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/seqcls_runtime_implementation_plan_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/seqcls_runtime_implementation_plan_{DATE}.md"))
    args = parser.parse_args()

    summary = build_summary(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(summary), encoding="utf-8")
    print(render_markdown(summary))


if __name__ == "__main__":
    main()
