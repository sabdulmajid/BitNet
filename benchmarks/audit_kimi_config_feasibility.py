#!/usr/bin/env python3
"""Audit Kimi-K2 architecture requirements against this fork's runtime.

This is a boundary-report generator, not a model benchmark.  It fetches or reads
the public Hugging Face config for Kimi-K2 and compares the architectural
features in that config with concrete support signals in this repository.  The
goal is to avoid vague "MoE support" claims: Kimi requires DeepSeekV3/Kimi
model loading, MLA-style attention metadata, routed/shared experts, and FP8
checkpoint handling before a ternary CPU runtime claim would be meaningful.
"""

from __future__ import annotations

import os
import argparse
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()
DEFAULT_CONFIG_URL = "https://huggingface.co/moonshotai/Kimi-K2-Instruct/raw/main/config.json"


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def fetch_json(url: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "BitNet-feasibility-audit/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - user-selected public URL
        return json.loads(response.read().decode("utf-8"))


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.config_json:
        return json.loads(args.config_json.read_text(encoding="utf-8"))
    return fetch_json(args.config_url, timeout=args.timeout)


def repo_support(root: Path) -> dict[str, bool]:
    converter_runtime = "\n".join(
        read_text(path)
        for path in [
            root / "utils" / "convert-hf-to-gguf-bitnet.py",
            root / "utils" / "convert-ms-to-gguf-bitnet.py",
            root / "benchmarks" / "audit_moe_support.py",
            root / "benchmarks" / "run_tiny_qwen2moe_fixture.py",
            root / "benchmarks" / "run_tiny_qwen2moe_expert_scaling.py",
            root / "src" / "ggml-bitnet-mad.cpp",
            root / "3rdparty" / "llama.cpp" / "src" / "llama-model-loader.cpp",
            root / "3rdparty" / "llama.cpp" / "src" / "llama-model.cpp",
        ]
    ).lower()
    bitnet_converter_runtime = "\n".join(
        read_text(path)
        for path in [
            root / "utils" / "convert-hf-to-gguf-bitnet.py",
            root / "utils" / "convert-ms-to-gguf-bitnet.py",
            root / "src" / "ggml-bitnet-mad.cpp",
        ]
    ).lower()
    llama_runtime = "\n".join(
        read_text(path)
        for path in [
            root / "3rdparty" / "llama.cpp" / "src" / "llama.cpp",
            root / "3rdparty" / "llama.cpp" / "src" / "llama-model-loader.cpp",
            root / "3rdparty" / "llama.cpp" / "src" / "llama-model.cpp",
        ]
    ).lower()
    return {
        "qwen2moe_fixture": "qwen2moe" in converter_runtime,
        "deepseek2_runtime": "llm_arch_deepseek2" in llama_runtime and "build_deepseek2" in llama_runtime,
        "kimi_k2_direct_loader": "kimi_k2" in bitnet_converter_runtime or "kimi-k2" in bitnet_converter_runtime,
        "deepseek_v3_direct_loader": "deepseekv3" in bitnet_converter_runtime or "deepseek3" in bitnet_converter_runtime,
        "mla_runtime_generic": "q_lora_rank" in llama_runtime and "kv_lora_rank" in llama_runtime,
        "mla_bitnet_converter": "q_lora_rank" in bitnet_converter_runtime or "kv_lora_rank" in bitnet_converter_runtime,
        "qwen_shared_expert_metadata": "shared_expert_intermediate_size" in bitnet_converter_runtime,
        "kimi_shared_expert_metadata": "n_shared_experts" in bitnet_converter_runtime,
        "fp8_block_import": "fp8" in bitnet_converter_runtime and "weight_block_size" in bitnet_converter_runtime,
        "i2sr_named": "i2_sr" in converter_runtime or "i2sr" in converter_runtime,
    }


def required_features(config: dict[str, Any]) -> list[dict[str, Any]]:
    quant = config.get("quantization_config", {}) if isinstance(config.get("quantization_config"), dict) else {}
    return [
        {
            "feature": "Kimi/DeepSeekV3 architecture loader",
            "config_evidence": f"model_type={config.get('model_type')}, architectures={config.get('architectures')}",
            "support_keys": ["kimi_k2_direct_loader", "deepseek_v3_direct_loader"],
            "required": True,
            "note": "Qwen/Qwen2MoE support does not imply Kimi-K2 loader support.",
        },
        {
            "feature": "MLA/Q-LoRA attention metadata",
            "config_evidence": f"q_lora_rank={config.get('q_lora_rank')}, kv_lora_rank={config.get('kv_lora_rank')}, qk_nope={config.get('qk_nope_head_dim')}, qk_rope={config.get('qk_rope_head_dim')}",
            "support_keys": ["mla_runtime_generic", "mla_bitnet_converter"],
            "required": True,
            "note": "llama.cpp has generic DeepSeek2 MLA runtime signals, but BitNet conversion must preserve Kimi/DeepSeekV3 metadata and tensor names.",
        },
        {
            "feature": "Routed MoE experts",
            "config_evidence": f"n_routed_experts={config.get('n_routed_experts')}, num_experts_per_tok={config.get('num_experts_per_tok')}",
            "support_keys": ["qwen2moe_fixture"],
            "required": True,
            "note": "Generic MoE tensor shape support is weaker than Kimi-specific routed execution.",
        },
        {
            "feature": "Shared expert path",
            "config_evidence": f"n_shared_experts={config.get('n_shared_experts')}",
            "support_keys": ["kimi_shared_expert_metadata"],
            "required": True,
            "note": "Shared experts need distinct conversion, packing, and runtime accounting.",
        },
        {
            "feature": "Block-FP8 source checkpoint import",
            "config_evidence": f"quant_method={quant.get('quant_method')}, fmt={quant.get('fmt')}, weight_block_size={quant.get('weight_block_size')}",
            "support_keys": ["fp8_block_import"],
            "required": True,
            "note": "A ternary retrofit pipeline must first correctly dequantize/import the source checkpoint.",
        },
        {
            "feature": "Packed row-scale ternary runtime",
            "config_evidence": "required for this fork's I2_SR product path",
            "support_keys": ["i2sr_named"],
            "required": True,
            "note": "Dense I2_SR support exists; Kimi MoE I2_SR quality/runtime remains unproven.",
        },
    ]


def render_markdown(summary: dict[str, Any]) -> str:
    rows = []
    for row in summary["features"]:
        rows.append(
            [
                row["feature"],
                "pass" if row["support_signal"] else "fail",
                row["config_evidence"],
                row["note"],
            ]
        )
    lines = [
        f"# Kimi Config Feasibility Audit, {DATE}",
        "",
        f"Overall status: `{'pass' if summary['passed'] else 'not_supported'}`.",
        "",
        f"Config source: `{summary['config_source']}`.",
        "",
        "## Architecture Summary",
        "",
        f"- model_type: `{summary['architecture']['model_type']}`",
        f"- architecture: `{summary['architecture']['architectures']}`",
        f"- layers: `{summary['architecture']['num_hidden_layers']}`",
        f"- hidden size: `{summary['architecture']['hidden_size']}`",
        f"- routed experts: `{summary['architecture']['n_routed_experts']}`",
        f"- experts per token: `{summary['architecture']['num_experts_per_tok']}`",
        f"- shared experts: `{summary['architecture']['n_shared_experts']}`",
        f"- context length: `{summary['architecture']['max_position_embeddings']}`",
        f"- source quantization: `{summary['architecture']['quantization_config']}`",
        "",
        "## Required Support Checks",
        "",
        "| required feature | repo signal | config evidence | note |",
        "| --- | --- | --- | --- |",
    ]
    lines.extend("| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |" for row in rows)
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            summary["verdict"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-url", default=DEFAULT_CONFIG_URL)
    parser.add_argument("--config-json", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/kimi_config_feasibility_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/kimi_config_feasibility_{DATE}.md"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config = load_config(args)
    support = repo_support(root)
    features = []
    for feature in required_features(config):
        keys = feature["support_keys"]
        signal = all(bool(support.get(key, False)) for key in keys)
        features.append({**feature, "support_signal": signal, "support_details": {key: support.get(key, False) for key in keys}})

    unsupported = [row["feature"] for row in features if row["required"] and not row["support_signal"]]
    passed = not unsupported
    architecture = {
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "hidden_size": config.get("hidden_size"),
        "n_routed_experts": config.get("n_routed_experts"),
        "num_experts_per_tok": config.get("num_experts_per_tok"),
        "n_shared_experts": config.get("n_shared_experts"),
        "max_position_embeddings": config.get("max_position_embeddings"),
        "quantization_config": config.get("quantization_config"),
    }
    verdict = (
        "Kimi-K2 support is not established in this fork. The config alone requires a Kimi/DeepSeekV3 loader, "
        "MLA attention layout handling, routed and shared expert conversion, block-FP8 import, and MoE-aware "
        "packed row-scale runtime validation before quality or speed claims are defensible."
    )
    if passed:
        verdict = (
            "Static repo signals mention every required feature, but this is still not a trained Kimi benchmark. "
            "A real support claim would require converted Kimi artifacts plus task quality, throughput, RSS, "
            "and expert-locality measurements."
        )

    summary = {
        "date": DATE,
        "config_source": str(args.config_json) if args.config_json else args.config_url,
        "passed": passed,
        "unsupported_features": unsupported,
        "architecture": architecture,
        "repo_support": support,
        "features": features,
        "verdict": verdict,
        "source_urls": [
            "https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/config.json",
            DEFAULT_CONFIG_URL,
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(summary), encoding="utf-8")
    print(render_markdown(summary))


if __name__ == "__main__":
    main()
