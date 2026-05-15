#!/usr/bin/env python3
"""Audit architecture-contract mismatches for the seqcls I2_SR sidecar path."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()
DEFAULT_CHECKPOINT = Path(
    "checkpoints/bitdistill-glue-seqcls-longwarmup/"
    "Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8"
)
DEFAULT_LLAMA_CPP = Path("3rdparty/llama.cpp/src/llama.cpp")
DEFAULT_HIDDEN_AUDIT = Path(f"benchmark_results/seqcls_i2sr_hidden_contract_{DATE}.json")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def line_number(text: str, needle: str) -> int | None:
    index = text.find(needle)
    if index < 0:
        return None
    return text[:index].count("\n") + 1


def line_number_after(text: str, after: str, needle: str) -> int | None:
    after_index = text.find(after)
    if after_index < 0:
        return None
    index = text.find(needle, after_index)
    if index < 0:
        return None
    return text[:index].count("\n") + 1


def block_between(text: str, start: str, end: str) -> str:
    start_index = text.find(start)
    if start_index < 0:
        return ""
    end_index = text.find(end, start_index + len(start))
    if end_index < 0:
        return text[start_index:]
    return text[start_index:end_index]


def block_between_after(text: str, after: str, start: str, end: str) -> str:
    after_index = text.find(after)
    if after_index < 0:
        return ""
    start_index = text.find(start, after_index)
    if start_index < 0:
        return ""
    end_index = text.find(end, start_index + len(start))
    if end_index < 0:
        return text[start_index:]
    return text[start_index:end_index]


def bias_summary(state_path: Path) -> dict[str, Any]:
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"expected state dict at {state_path}")
    projection_biases = [
        key
        for key in state
        if key.endswith(".bias") and any(f".{name}." in key for name in ("q_proj", "k_proj", "v_proj", "o_proj"))
    ]
    families: dict[str, int] = {}
    for key in projection_biases:
        family = key.split(".")[-2]
        families[family] = families.get(family, 0) + 1
    return {
        "projection_bias_count": len(projection_biases),
        "projection_bias_families": dict(sorted(families.items())),
        "first_projection_biases": projection_biases[:12],
    }


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).replace("|", "\\|")


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(cell) for cell in row) + " |")
    return "\n".join(lines)


def render_markdown(result: dict[str, Any]) -> str:
    runtime = result["runtime_source"]
    hidden = result["hidden_contract"]
    bias = result["checkpoint_biases"]
    return "\n\n".join(
        [
            f"# Sequence-Classification I2_SR Architecture-Contract Audit, {result['date']}",
            (
                "This audit explains why the current packed sidecar classifier cannot yet be "
                "treated as a faithful deployment artifact. It checks the PyTorch checkpoint "
                "architecture against the llama.cpp graph selected by the GGUF export."
            ),
            md_table(
                ["field", "value"],
                [
                    ["status", result["status"]],
                    ["checkpoint hidden_act", result["checkpoint_config"].get("hidden_act")],
                    ["GGUF runtime arch under audit", result["runtime_arch_under_audit"]],
                    ["bitnet-25 graph activation", runtime["bitnet25_ffn_activation"]],
                    ["PyTorch/runtime activation mismatch", result["checks"]["activation_mismatch"]],
                    ["Q/K/V projection bias tensors in checkpoint", bias["projection_bias_count"]],
                    ["plain bitnet loader has Q/K/V bias slots", runtime["bitnet_loader_has_qkv_bias"]],
                    ["bitnet-25 loader has Q/K/V bias slots", runtime["bitnet25_loader_has_qkv_bias"]],
                    ["hidden relative RMS", hidden.get("comparisons", {}).get("hidden_relative_rms")],
                    ["hidden cosine", hidden.get("comparisons", {}).get("hidden_cosine")],
                ],
            ),
            "## Source Evidence",
            md_table(
                ["source check", "line"],
                [
                    ["bitnet-25 dispatches to build_bitnet_158", runtime["bitnet25_dispatch_line"]],
                    ["build_bitnet_158 uses ReLU squared FFN", runtime["bitnet25_relu_sqr_line"]],
                    ["build_bitnet uses SiLU FFN", runtime["bitnet_silu_line"]],
                    ["plain bitnet loader block starts", runtime["bitnet_loader_line"]],
                    ["bitnet-25 loader block starts", runtime["bitnet25_loader_line"]],
                ],
            ),
            "## Interpretation",
            (
                "The current seqcls export uses the `bitnet-25` runtime graph, but the "
                "checkpoint is a Qwen2 sequence-classification student whose config says "
                "`hidden_act = silu`. The `bitnet-25` graph uses the BitNet b1.58/2.5 "
                "ReLU-squared FFN path. That is a deterministic architecture mismatch. "
                "The alternative plain `bitnet` graph has the SiLU FFN path, but its loader "
                "does not declare Q/K/V projection-bias tensors, while this Qwen2 checkpoint "
                "has 72 such tensors. Therefore the fix is not just changing a metadata "
                "string; we need a Qwen-BitDistill runtime contract that preserves Qwen2 "
                "SwiGLU/SiLU, optional projection biases, SubLN, RoPE metadata, and I2_SR "
                "row-scale kernels together."
            ),
            "## Required Runtime Work",
            md_table(
                ["requirement", "status"],
                [[item["requirement"], item["status"]] for item in result["required_runtime_work"]],
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--llama-cpp", type=Path, default=DEFAULT_LLAMA_CPP)
    parser.add_argument("--hidden-contract-json", type=Path, default=DEFAULT_HIDDEN_AUDIT)
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/seqcls_i2sr_arch_contract_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/seqcls_i2sr_arch_contract_{DATE}.md"))
    args = parser.parse_args()

    root = args.repo_root.resolve()
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir.is_absolute() else root / args.checkpoint_dir
    llama_cpp = args.llama_cpp if args.llama_cpp.is_absolute() else root / args.llama_cpp
    hidden_path = args.hidden_contract_json if args.hidden_contract_json.is_absolute() else root / args.hidden_contract_json

    config = read_json(checkpoint_dir / "config.json")
    hidden = read_json(hidden_path)
    source = llama_cpp.read_text(encoding="utf-8")
    build_bitnet = block_between(source, "struct ggml_cgraph * build_bitnet() {", "struct ggml_cgraph * build_bitnet_158() {")
    build_bitnet_158 = block_between(source, "struct ggml_cgraph * build_bitnet_158() {", "struct ggml_cgraph * build_t5_encoder() {")
    loader_anchor = "static bool llm_load_tensors("
    bitnet_loader = block_between_after(source, loader_anchor, "case LLM_ARCH_BITNET:", "case LLM_ARCH_BITNET_B158:")
    bitnet25_loader = block_between_after(source, loader_anchor, "case LLM_ARCH_BITNET_B158:", "case LLM_ARCH_T5:")
    graph_dispatch = block_between(source, "case LLM_ARCH_BITNET:", "case LLM_ARCH_T5:")
    biases = bias_summary(checkpoint_dir / "ternary_state_dict.pt")

    runtime = {
        "path": maybe_relative(llama_cpp, root),
        "bitnet25_ffn_activation": "relu_sqr" if "LLM_FFN_RELU_SQR" in build_bitnet_158 else "unknown",
        "bitnet_ffn_activation": "silu" if "LLM_FFN_SILU" in build_bitnet else "unknown",
        "bitnet_loader_has_qkv_bias": all(token in bitnet_loader for token in ("layer.bq", "layer.bk", "layer.bv")),
        "bitnet25_loader_has_qkv_bias": all(token in bitnet25_loader for token in ("layer.bq", "layer.bk", "layer.bv")),
        "bitnet25_dispatch_line": line_number_after(source, "static struct ggml_cgraph * llama_build_graph(", "case LLM_ARCH_BITNET_25:"),
        "bitnet25_relu_sqr_line": line_number_after(source, "struct ggml_cgraph * build_bitnet_158() {", "LLM_FFN_RELU_SQR"),
        "bitnet_silu_line": line_number_after(source, "struct ggml_cgraph * build_bitnet() {", "LLM_FFN_SILU"),
        "bitnet_loader_line": line_number_after(source, loader_anchor, "case LLM_ARCH_BITNET:"),
        "bitnet25_loader_line": line_number_after(source, loader_anchor, "case LLM_ARCH_BITNET_B158:"),
    }
    activation_mismatch = config.get("hidden_act") == "silu" and runtime["bitnet25_ffn_activation"] == "relu_sqr"
    bias_contract_gap = biases["projection_bias_count"] > 0 and not runtime["bitnet_loader_has_qkv_bias"]

    result = {
        "schema": "seqcls_i2sr_arch_contract.v1",
        "date": DATE,
        "status": "architecture_contract_mismatch" if activation_mismatch else "needs_review",
        "checkpoint": {
            "path": maybe_relative(checkpoint_dir, root),
            "state_path": maybe_relative(checkpoint_dir / "ternary_state_dict.pt", root),
        },
        "checkpoint_config": {
            "architectures": config.get("architectures"),
            "model_type": config.get("model_type"),
            "hidden_act": config.get("hidden_act"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "hidden_size": config.get("hidden_size"),
            "num_attention_heads": config.get("num_attention_heads"),
            "num_key_value_heads": config.get("num_key_value_heads"),
        },
        "checkpoint_biases": biases,
        "runtime_arch_under_audit": "bitnet-25",
        "runtime_source": runtime,
        "hidden_contract": {
            "path": maybe_relative(hidden_path, root),
            "status": hidden.get("status"),
            "comparisons": hidden.get("comparisons", {}),
        },
        "checks": {
            "activation_mismatch": activation_mismatch,
            "plain_bitnet_has_silu_graph": runtime["bitnet_ffn_activation"] == "silu",
            "plain_bitnet_bias_contract_gap": bias_contract_gap,
            "bitnet25_has_bias_slots": runtime["bitnet25_loader_has_qkv_bias"],
        },
        "required_runtime_work": [
            {"requirement": "Qwen2/Qwen3 SiLU/SwiGLU FFN in packed graph", "status": "missing in bitnet-25 graph"},
            {"requirement": "Q/K/V projection-bias tensor slots", "status": "present in bitnet-25 loader, missing in plain bitnet loader"},
            {"requirement": "SubLN before attention output and FFN down projections", "status": "present in bitnet-25 graph"},
            {"requirement": "RoPE theta from rope_parameters", "status": "fixed in converter"},
            {"requirement": "row-scale I2_SR kernels", "status": "present for dense matmuls"},
            {"requirement": "native sequence-classification score head", "status": "not implemented"},
        ],
        "interpretation": (
            "The current sidecar mismatch is not just tokenizer metadata. The exported runtime "
            "graph is not the same architecture family as the PyTorch Qwen2 sequence-classification "
            "student: bitnet-25 uses ReLU-squared FFN while the checkpoint uses SiLU/SwiGLU. "
            "The plain bitnet graph has SiLU but lacks Q/K/V bias tensor slots needed by Qwen2. "
            "A dedicated Qwen-BitDistill packed graph is required before full GLUE runtime quality "
            "numbers are meaningful."
        ),
    }

    output_json = args.output_json if args.output_json.is_absolute() else root / args.output_json
    output_md = args.output_md if args.output_md.is_absolute() else root / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(result) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": result["checks"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
