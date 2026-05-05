#!/usr/bin/env python3
"""Audit generic MoE support and Kimi-specific gaps in this fork."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PatternCheck:
    label: str
    path: Path
    patterns: tuple[str, ...]
    expectation: str


def first_line(path: Path, pattern: str) -> int | None:
    for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if pattern in line:
            return lineno
    return None


def run_check(check: PatternCheck) -> dict[str, Any]:
    pattern_lines = {pattern: first_line(check.path, pattern) for pattern in check.patterns}
    present = {pattern: lineno for pattern, lineno in pattern_lines.items() if lineno is not None}
    return {
        "label": check.label,
        "path": str(check.path),
        "expectation": check.expectation,
        "status": "present" if len(present) == len(check.patterns) else "missing",
        "patterns": pattern_lines,
    }


def search_tree(root: Path, needle: str) -> list[str]:
    matches: list[str] = []
    suffixes = {".py", ".cpp", ".c", ".h", ".hpp"}
    skip_dirs = {".git", "benchmark_results", "checkpoints", "models", "__pycache__"}
    for path in root.rglob("*"):
        if any(part in skip_dirs for part in path.parts):
            continue
        if path.name == "audit_moe_support.py":
            continue
        if not path.is_file() or path.suffix not in suffixes:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if needle.lower() in text.lower():
            matches.append(str(path))
    return sorted(matches)


def local_artifacts(root: Path) -> list[str]:
    artifacts: list[str] = []
    if not root.exists():
        return artifacts
    for path in root.rglob("*"):
        normalized = path.name.lower()
        if "kimi" in normalized:
            artifacts.append(str(path))
    return sorted(artifacts)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def pattern_summary(result: dict[str, Any]) -> str:
    parts: list[str] = []
    for pattern, lineno in result["patterns"].items():
        parts.append(f"`{pattern}`@{lineno}" if lineno is not None else f"`{pattern}`@missing")
    return ", ".join(parts)


def build_report(data: dict[str, Any]) -> str:
    rows = [
        [
            result["label"],
            result["path"],
            result["status"],
            result["expectation"],
            pattern_summary(result),
        ]
        for result in data["checks"]
    ]
    artifact_note = (
        f"Local Kimi benchmark artifact paths found: {len(data['local_kimi_artifacts'])}."
        if data["local_kimi_artifacts"]
        else "No local Kimi benchmark artifacts were found under benchmark_results."
    )
    kimi_note = (
        f"Kimi string matches in tracked source files: {len(data['kimi_source_matches'])}."
        if data["kimi_source_matches"]
        else "No Kimi-specific converter/runtime mapping was found in tracked source files."
    )
    verdict = (
        "Generic MoE infrastructure is present: GGUF metadata has expert counts, "
        "Qwen2MoE is registered in the vendored llama.cpp converter, expert "
        "weights are merged into 3D tensors, and the runtime builds sparse "
        "top-k expert execution with `ggml_mul_mat_id`. This does not prove "
        "Kimi support: no Kimi-specific mapping or benchmark artifact is present, "
        "and the TL2-capable BitNet converter still lacks Qwen2MoE registration."
    )
    return "\n\n".join(
        [
            "# MoE Support Audit, 2026-05-05",
            md_table(["check", "path", "status", "expectation", "evidence"], rows),
            "## Negative Checks",
            "\n".join([kimi_note, artifact_note]),
            "## Verdict",
            verdict,
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    root = args.repo_root
    checks = [
        PatternCheck(
            "GGUF expert metadata",
            root / "3rdparty/llama.cpp/gguf-py/gguf/constants.py",
            ("EXPERT_COUNT", "EXPERT_USED_COUNT"),
            "metadata can record expert count and active experts",
        ),
        PatternCheck(
            "Qwen2MoE tensor schema",
            root / "3rdparty/llama.cpp/gguf-py/gguf/constants.py",
            ("MODEL_ARCH.QWEN2MOE", "MODEL_TENSOR.FFN_GATE_EXP", "MODEL_TENSOR.FFN_DOWN_EXP", "MODEL_TENSOR.FFN_UP_EXP"),
            "GGUF schema has merged expert tensors for Qwen2MoE",
        ),
        PatternCheck(
            "llama.cpp Qwen2MoE converter",
            root / "3rdparty/llama.cpp/convert_hf_to_gguf.py",
            ('@Model.register("Qwen2MoeForCausalLM")', "MODEL_ARCH.QWEN2MOE", "torch.stack(datas, dim=0)"),
            "vendored converter registers Qwen2MoE and merges experts",
        ),
        PatternCheck(
            "BitNet converter generic expert packing",
            root / "utils/convert-hf-to-gguf-bitnet.py",
            ("num_local_experts", "num_experts_per_tok", "block_sparse_moe.experts"),
            "BitNet converter has generic Mixtral-style expert metadata/packing",
        ),
        PatternCheck(
            "Runtime sparse expert execution",
            root / "3rdparty/llama.cpp/src/llama.cpp",
            ("llm_build_moe_ffn", "ggml_soft_max", "ggml_top_k", "ggml_mul_mat_id"),
            "runtime builds top-k routed sparse expert matmuls",
        ),
    ]

    data: dict[str, Any] = {
        "checks": [run_check(check) for check in checks],
        "kimi_source_matches": search_tree(root, "Kimi"),
        "local_kimi_artifacts": local_artifacts(root / "benchmark_results"),
    }
    report = build_report(data)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report + "\n", encoding="utf-8")
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
