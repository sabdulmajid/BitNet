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
    skip_dirs = {".git", "benchmark_results", "benchmarks", "checkpoints", "models", "__pycache__"}
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


def file_contains(path: Path, patterns: tuple[str, ...]) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    return all(pattern in text for pattern in patterns)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def make_gate(name: str, passed: bool, evidence: str, blocker: str = "") -> dict[str, Any]:
    return {
        "name": name,
        "passed": passed,
        "evidence": evidence,
        "blocker": "" if passed else blocker,
    }


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
        f"Kimi string matches in converter/runtime source files: {len(data['kimi_source_matches'])}."
        if data["kimi_source_matches"]
        else "No Kimi-specific converter/runtime mapping was found in converter/runtime source files."
    )
    contract = data.get("moe_packing_contract") or {}
    contract_verdict = contract.get("verdict", {})
    contract_note = (
        "Synthetic MoE packing contract: "
        f"TL2 3D supported={contract_verdict.get('merged_3d_tl2_supported')}; "
        f"I2_S/I2_SR 3D supported={contract_verdict.get('merged_3d_i2s_i2sr_supported')}; "
        f"2D control supported={contract_verdict.get('dense_2d_i2s_control_supported')}."
        if contract
        else "Synthetic MoE packing contract artifact is missing."
    )
    tl2_contract = data.get("moe_tl2_runtime_contract") or {}
    tl2_note = (
        "TL2 MoE runtime contract: "
        f"ready={tl2_contract.get('tl2_moe_runtime_ready')}; "
        f"blockers={len(tl2_contract.get('blockers', []))}."
        if tl2_contract
        else "TL2 MoE runtime contract artifact is missing."
    )
    verdict = (
        "Generic MoE infrastructure is present: GGUF metadata has expert counts, "
        "Qwen2MoE is registered in the vendored llama.cpp converter, expert "
        "weights are merged into 3D tensors, and the runtime builds sparse "
        "top-k expert execution with `ggml_mul_mat_id`. This does not prove "
        "Kimi support: no Kimi-specific mapping or benchmark artifact is present, "
        "Qwen2MoE mapping still lacks a real converted/evaluated artifact, "
        "the TL2 packing path remains 2D-matrix oriented, and the active TL2 "
        "runtime contract does not size or route merged experts correctly. The "
        "direct I2_S/I2_SR path is only a synthetic packing contract until it "
        "is validated with a full MoE GGUF/runtime artifact."
    )
    gate_rows = [
        [
            gate["name"],
            "pass" if gate["passed"] else "fail",
            gate["evidence"],
            gate["blocker"],
        ]
        for gate in data["productization_gates"]
    ]
    required_plan = (
        "Required MoE/Kimi path: validate the new Qwen2MoE BitNet converter "
        "mapping on a real checkpoint, add any Kimi-specific tensor mapping, decide "
        "which router and shared-expert tensors stay dense, extend TL2 packing plus "
        "full GGUF/runtime tests to 3D expert tensors, distill router and expert "
        "weights under ternary constraints, then run quality, throughput, RSS, and "
        "expert-locality benchmarks against dense and llama.cpp quantized MoE baselines."
    )
    return "\n\n".join(
        [
            "# MoE Support Audit, 2026-05-05",
            md_table(["check", "path", "status", "expectation", "evidence"], rows),
            "## Productization Gates",
            md_table(["gate", "status", "evidence", "blocker"], gate_rows),
            "## Negative Checks",
            "\n".join([kimi_note, artifact_note, contract_note, tl2_note]),
            "## Verdict",
            verdict,
            "## Required Plan",
            required_plan,
        ]
    )


def build_productization_gates(
    root: Path,
    checks: list[dict[str, Any]],
    kimi_matches: list[str],
    kimi_artifacts: list[str],
    moe_packing_contract: dict[str, Any],
    moe_tl2_runtime_contract: dict[str, Any],
) -> list[dict[str, Any]]:
    check_by_label = {check["label"]: check for check in checks}
    generic_runtime = check_by_label.get("Runtime sparse expert execution", {}).get("status") == "present"
    gguf_schema = check_by_label.get("Qwen2MoE tensor schema", {}).get("status") == "present"
    llama_qwen2moe = check_by_label.get("llama.cpp Qwen2MoE converter", {}).get("status") == "present"

    bitnet_converter = root / "utils/convert-hf-to-gguf-bitnet.py"
    direct_i2sr_writer = root / "benchmarks/convert_static_ternary_to_i2s_gguf.py"
    tl2_is_2d = file_contains(bitnet_converter, ("def preprocess_weights_tl2", "M, K = w.shape"))
    direct_i2sr_is_2d = file_contains(direct_i2sr_writer, ("codes.ndim != 2", "I2_S packing expects a 2D weight matrix"))
    bitnet_qwen2moe = file_contains(bitnet_converter, ('@Model.register("Qwen2MoeForCausalLM")',))
    bitnet_kimi = file_contains(bitnet_converter, ("Kimi",))
    contract_verdict = moe_packing_contract.get("verdict", {})
    contract_tl2_3d = bool(contract_verdict.get("merged_3d_tl2_supported"))
    contract_i2sr_3d = bool(contract_verdict.get("merged_3d_i2s_i2sr_supported"))
    contract_2d_control = bool(contract_verdict.get("dense_2d_i2s_control_supported"))
    contract_available = bool(contract_verdict)
    runtime_ready = bool(moe_tl2_runtime_contract.get("tl2_moe_runtime_ready"))
    runtime_blockers = moe_tl2_runtime_contract.get("blockers", [])
    byte_probe = moe_tl2_runtime_contract.get("byte_size_probe", {})

    return [
        make_gate(
            "generic GGUF/runtime MoE support exists",
            gguf_schema and generic_runtime and llama_qwen2moe,
            f"gguf_schema={gguf_schema}; runtime={generic_runtime}; llama_qwen2moe_converter={llama_qwen2moe}",
            "The vendored llama.cpp layer must expose MoE metadata, conversion, and routed execution.",
        ),
        make_gate(
            "BitNet converter has explicit Qwen2MoE-or-Kimi registration",
            bitnet_qwen2moe or bitnet_kimi,
            f"qwen2moe_registration={bitnet_qwen2moe}; kimi_converter_match={bitnet_kimi}; tracked_kimi_mentions={len(kimi_matches)}",
            "The TL2-capable BitNet converter must register at least one MoE-family architecture before real MoE conversion can be tested.",
        ),
        make_gate(
            "TL2 converter path is validated for merged 3D expert tensors",
            contract_tl2_3d and runtime_ready,
            (
                f"contract_available={contract_available}; contract_tl2_3d={contract_tl2_3d}; "
                f"preprocess_weights_tl2_uses_2d_unpack={tl2_is_2d}; runtime_ready={runtime_ready}; "
                f"runtime_blockers={len(runtime_blockers)}; "
                f"tl2_expert_byte_underreport={byte_probe.get('underreport_bytes')}"
            ),
            "`preprocess_weights_tl2` rejects 3D tensors, and the active TL2 runtime contract also under-sizes and misroutes merged expert tensors.",
        ),
        make_gate(
            "direct I2_SR writer is validated for merged 3D expert tensors",
            contract_i2sr_3d and contract_2d_control,
            (
                f"contract_available={contract_available}; contract_i2sr_3d={contract_i2sr_3d}; "
                f"contract_2d_control={contract_2d_control}; direct_i2sr_writer_rejects_non_2d={direct_i2sr_is_2d}"
            ),
            "The direct packed I2_S/I2_SR writer must accept synthetic 3D expert tensors without regressing 2D dense packing.",
        ),
        make_gate(
            "local Kimi model/eval artifacts exist",
            bool(kimi_artifacts),
            f"kimi_artifacts={len(kimi_artifacts)}",
            "No local Kimi checkpoint, conversion, quality, throughput, RSS, or expert-locality artifact exists.",
        ),
        make_gate(
            "MoE quality and locality benchmarks exist",
            False,
            "quality_runs=0; throughput_runs=0; expert_locality_runs=0",
            "No benchmark measures router accuracy, expert selection locality, sparse expert throughput, or quality degradation.",
        ),
    ]


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

    check_results = [run_check(check) for check in checks]
    kimi_source_matches = search_tree(root, "Kimi")
    local_kimi_results = local_artifacts(root / "benchmark_results")
    moe_packing_contract = read_json(root / "benchmark_results/moe_packing_contract_2026-05-13.json")
    moe_tl2_runtime_contract = read_json(root / "benchmark_results/moe_tl2_runtime_contract_2026-05-13.json")
    data: dict[str, Any] = {
        "checks": check_results,
        "productization_gates": build_productization_gates(
            root,
            check_results,
            kimi_source_matches,
            local_kimi_results,
            moe_packing_contract,
            moe_tl2_runtime_contract,
        ),
        "kimi_source_matches": kimi_source_matches,
        "local_kimi_artifacts": local_kimi_results,
        "moe_packing_contract": moe_packing_contract,
        "moe_tl2_runtime_contract": moe_tl2_runtime_contract,
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
