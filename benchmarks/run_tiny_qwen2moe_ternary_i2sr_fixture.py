#!/usr/bin/env python3
"""Build and execute a tiny ternary Qwen2MoE I2_SR fixture.

This is a runtime contract probe, not a quality benchmark. It creates a tiny
random Qwen2MoE checkpoint whose expert matrix widths are divisible by the
active I2_S/I2_SR packing group size, replaces the merged expert tensors with
row-scale ternary weights, exports a GGUF with the direct I2_SR writer, and
attempts CPU routed execution through llama.cpp.

The test proves only converter/runtime plumbing for row-scale ternary merged
expert tensors. It does not prove Kimi support, router quality, trained MoE
quality, or task accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, Qwen2MoeConfig, Qwen2MoeForCausalLM


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()
I2_GROUP_SIZE = 128


def run_command(command: list[str], *, stdout_path: Path, stderr_path: Path) -> dict[str, Any]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "elapsed_seconds": time.perf_counter() - start,
    }


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def find_first(pattern: str, text: str, default: str = "") -> str:
    match = re.search(pattern, text, flags=re.MULTILINE)
    return match.group(1).strip() if match else default


def parse_float(pattern: str, text: str) -> float | None:
    value = find_first(pattern, text)
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_int(pattern: str, text: str) -> int | None:
    value = find_first(pattern, text)
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def ternarize_row_scale(weight: torch.Tensor, eps: float = 1.0e-6) -> tuple[torch.Tensor, torch.Tensor]:
    dense = weight.detach().to(dtype=torch.float32)
    scale = dense.abs().mean(dim=-1).clamp_min(eps)
    codes = torch.round(dense / scale.unsqueeze(-1)).clamp(-1, 1).to(dtype=torch.int8)
    return codes, scale.to(dtype=torch.float32)


def make_tiny_model(
    model_dir: Path,
    tokenizer_model: str,
    *,
    seed: int,
    force: bool,
    hidden_size: int,
    intermediate_size: int,
    moe_intermediate_size: int,
    shared_expert_intermediate_size: int,
    num_experts: int,
    num_experts_per_tok: int,
) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    model_path = model_dir / "model.safetensors"
    if config_path.exists() and model_path.exists() and not force:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        return {
            "created": False,
            "model_dir": str(model_dir),
            "config": {
                "architectures": config.get("architectures"),
                "hidden_size": config.get("hidden_size"),
                "num_hidden_layers": config.get("num_hidden_layers"),
                "num_experts": config.get("num_experts"),
                "num_experts_per_tok": config.get("num_experts_per_tok"),
                "moe_intermediate_size": config.get("moe_intermediate_size"),
                "shared_expert_intermediate_size": config.get("shared_expert_intermediate_size"),
            },
        }

    if hidden_size % I2_GROUP_SIZE != 0 or moe_intermediate_size % I2_GROUP_SIZE != 0:
        raise ValueError("hidden_size and moe_intermediate_size must be divisible by 128 for I2_SR expert packing")

    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    tokenizer.save_pretrained(model_dir)

    torch.manual_seed(seed)
    config = Qwen2MoeConfig(
        vocab_size=len(tokenizer),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        moe_intermediate_size=moe_intermediate_size,
        shared_expert_intermediate_size=shared_expert_intermediate_size,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        decoder_sparse_step=1,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        mlp_only_layers=[],
        qkv_bias=True,
        tie_word_embeddings=False,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        torch_dtype="float16",
    )
    model = Qwen2MoeForCausalLM(config)
    model.eval()
    model.save_pretrained(model_dir, safe_serialization=True)

    return {
        "created": True,
        "model_dir": str(model_dir),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "config": {
            "architectures": ["Qwen2MoeForCausalLM"],
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "num_experts": config.num_experts,
            "num_experts_per_tok": config.num_experts_per_tok,
            "moe_intermediate_size": config.moe_intermediate_size,
            "shared_expert_intermediate_size": config.shared_expert_intermediate_size,
        },
    }


def build_ternary_state(model_dir: Path, output_path: Path, *, force: bool) -> dict[str, Any]:
    if output_path.exists() and not force:
        state = torch.load(output_path, map_location="cpu", weights_only=True)
        return summarize_ternary_state(output_path, state, created=False)

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    num_layers = int(config["num_hidden_layers"])
    num_experts = int(config["num_experts"])
    dense_state = load_file(str(model_dir / "model.safetensors"), device="cpu")
    ternary_state: dict[str, torch.Tensor] = {}
    skipped_expert_keys: set[str] = set()

    for layer in range(num_layers):
        for expert in range(num_experts):
            for weight_name in ("down_proj", "gate_proj", "up_proj"):
                skipped_expert_keys.add(f"model.layers.{layer}.mlp.experts.{expert}.{weight_name}.weight")

    for key, tensor in dense_state.items():
        if key in skipped_expert_keys:
            continue
        ternary_state[key] = tensor

    for layer in range(num_layers):
        for weight_name in ("down_proj", "gate_proj", "up_proj"):
            parts = [
                dense_state[f"model.layers.{layer}.mlp.experts.{expert}.{weight_name}.weight"]
                for expert in range(num_experts)
            ]
            merged = torch.stack(parts, dim=0)
            if merged.shape[-1] % I2_GROUP_SIZE != 0:
                raise ValueError(f"merged expert tensor {weight_name} has incompatible shape {tuple(merged.shape)}")
            codes, scales = ternarize_row_scale(merged)
            prefix = f"model.layers.{layer}.mlp.experts.{weight_name}"
            ternary_state[f"{prefix}.ternary_weight"] = codes
            ternary_state[f"{prefix}.weight_scale"] = scales

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ternary_state, output_path)
    return summarize_ternary_state(output_path, ternary_state, created=True)


def summarize_ternary_state(path: Path, state: dict[str, torch.Tensor], *, created: bool) -> dict[str, Any]:
    ternary_keys = sorted(key for key in state if key.endswith(".ternary_weight"))
    scale_keys = sorted(key for key in state if key.endswith(".weight_scale"))
    row_scale_shapes = {
        key: list(state[key].shape)
        for key in scale_keys
        if state[key].numel() > 1
    }
    code_shapes = {key: list(state[key].shape) for key in ternary_keys}
    invalid: dict[str, list[int]] = {}
    for key in ternary_keys:
        values = sorted(int(value) for value in torch.unique(state[key]).tolist())
        bad = [value for value in values if value not in {-1, 0, 1}]
        if bad:
            invalid[key] = bad
    return {
        "path": str(path),
        "created": created,
        "tensor_count": len(state),
        "ternary_key_count": len(ternary_keys),
        "scale_key_count": len(scale_keys),
        "row_scale_key_count": len(row_scale_shapes),
        "ternary_shapes": code_shapes,
        "row_scale_shapes": row_scale_shapes,
        "invalid_ternary_values": invalid,
        "size_bytes": path.stat().st_size if path.exists() else None,
    }


def summarize_output(stdout_path: Path, stderr_path: Path) -> dict[str, Any]:
    stdout = read_text(stdout_path)
    stderr = read_text(stderr_path)
    combined = stdout + "\n" + stderr
    return {
        "generated_prefix": " ".join(stdout.split())[:240],
        "architecture": find_first(r"general\.architecture\s+str\s+=\s+([^\n]+)", combined),
        "expert_count": parse_int(r"qwen2moe\.expert_count\s+u32\s+=\s+([0-9]+)", combined),
        "expert_used_count": parse_int(r"qwen2moe\.expert_used_count\s+u32\s+=\s+([0-9]+)", combined),
        "model_params_m": parse_float(r"model params\s+=\s+([0-9.]+)\s+M", combined),
        "cpu_buffer_mib": parse_float(r"(?:CPU_Mapped model|CPU) buffer size\s+=\s+([0-9.]+)\s+MiB", combined),
        "prompt_eval_tok_s": parse_float(r"prompt eval time =.+?([0-9.]+) tokens per second", combined),
        "decode_tok_s": parse_float(r"llama_perf_context_print:\s+eval time =.+?([0-9.]+) tokens per second", combined),
        "graph_nodes": parse_int(r"graph nodes\s+=\s+([0-9]+)", combined),
        "fatal": find_first(r"(GGML_ASSERT[^\n]+|error:[^\n]+|failed[^\n]+|Aborted[^\n]*)", combined),
    }


def parse_rss(stderr_path: Path) -> dict[str, Any]:
    text = read_text(stderr_path)
    rss_kib = parse_int(r"Maximum resident set size \(kbytes\):\s+([0-9]+)", text)
    return {
        "max_rss_kib": rss_kib,
        "max_rss_mib": rss_kib / 1024 if rss_kib is not None else None,
        "elapsed": find_first(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s+([^\n]+)", text),
    }


def md_bool(value: bool) -> str:
    return "yes" if value else "no"


def render_markdown(result: dict[str, Any]) -> str:
    conversion_summary = result.get("conversion_summary", {})
    smoke = result.get("smoke", {})
    rss = result.get("rss", {})
    gates = result.get("gates", {})
    rows = [
        ["HF checkpoint created", md_bool(gates.get("hf_checkpoint_created", False)), result["hf_model"].get("model_dir", "")],
        ["Merged ternary expert state built", md_bool(gates.get("ternary_state_built", False)), result["ternary_state"].get("path", "")],
        ["I2_SR GGUF conversion finished", md_bool(gates.get("i2sr_converted", False)), result.get("gguf_path", "")],
        [
            "3D expert tensors packed as row-scale I2_SR",
            md_bool(gates.get("merged_expert_i2sr_packed", False)),
            f"packed={conversion_summary.get('ternary_i2s_packed')}; row_scale={conversion_summary.get('row_scale_i2s_packed')}",
        ],
        ["CPU routed smoke executed", md_bool(gates.get("cpu_smoke_executed", False)), str(result["commands"].get("smoke", {}).get("returncode"))],
        ["Peak RSS measured", md_bool(gates.get("rss_measured", False)), f"{rss.get('max_rss_mib')} MiB"],
        [
            "Qwen2MoE metadata present",
            md_bool(gates.get("qwen2moe_metadata_present", False)),
            f"experts={smoke.get('expert_count')}; used={smoke.get('expert_used_count')}",
        ],
    ]
    table = "\n".join(
        [
            "| gate | pass | evidence |",
            "| --- | --- | --- |",
            *("| " + " | ".join(row) + " |" for row in rows),
        ]
    )
    return "\n\n".join(
        [
            f"# Tiny Qwen2MoE Ternary I2_SR Runtime Fixture, {DATE}",
            (
                "This fixture creates a tiny random `Qwen2MoeForCausalLM`, merges expert weights into "
                "3D row-scale ternary tensors, exports `MOSTLY_I2_SR` GGUF, and runs `llama-cli` on CPU."
            ),
            table,
            "## Runtime Snapshot",
            (
                f"Passed: `{result.get('passed')}`; architecture: `{smoke.get('architecture')}`; "
                f"params: `{smoke.get('model_params_m')}` M; file: `{result.get('gguf_mib')}` MiB; "
                f"CPU buffer: `{smoke.get('cpu_buffer_mib')}` MiB; peak RSS: `{rss.get('max_rss_mib')}` MiB; "
                f"prompt: `{smoke.get('prompt_eval_tok_s')}` tok/s; decode: `{smoke.get('decode_tok_s')}` tok/s; "
                f"fatal: `{smoke.get('fatal')}`."
            ),
            "## Interpretation",
            (
                "A pass is positive evidence that the direct I2_SR writer and the current CPU runtime can carry "
                "merged 3D row-scale ternary expert tensors through routed Qwen2MoE execution. A failure is still "
                "a hard boundary artifact because it records the exact failing conversion/runtime command."
            ),
            (
                "This does not prove Kimi support, trained MoE quality, router distillation, task accuracy, TL2 "
                "expert kernels, or expert-locality throughput on a real model."
            ),
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--model-dir", type=Path, default=Path("models/tiny-qwen2moe-ternary-i2sr-fixture"))
    parser.add_argument("--ternary-state", type=Path, default=None)
    parser.add_argument("--gguf", type=Path, default=Path("models/tiny-qwen2moe-ternary-i2sr-fixture/tiny-qwen2moe-ternary-i2sr.gguf"))
    parser.add_argument("--converter", type=Path, default=Path("benchmarks/convert_static_ternary_to_i2s_gguf.py"))
    parser.add_argument("--llama-bin-dir", type=Path, default=Path("build-portable-avx2/bin"))
    parser.add_argument("--out-dir", type=Path, default=Path(f"benchmark_results/tiny_qwen2moe_ternary_i2sr_fixture_{DATE}"))
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/tiny_qwen2moe_ternary_i2sr_fixture_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/tiny_qwen2moe_ternary_i2sr_fixture_{DATE}.md"))
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--rss-tokens", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--intermediate-size", type=int, default=256)
    parser.add_argument("--moe-intermediate-size", type=int, default=128)
    parser.add_argument("--shared-expert-intermediate-size", type=int, default=128)
    parser.add_argument("--num-experts", type=int, default=2)
    parser.add_argument("--num-experts-per-tok", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    ternary_state_path = args.ternary_state or args.model_dir / "ternary_state_dict.pt"
    hf_model = make_tiny_model(
        args.model_dir,
        args.tokenizer_model,
        seed=args.seed,
        force=args.force,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        moe_intermediate_size=args.moe_intermediate_size,
        shared_expert_intermediate_size=args.shared_expert_intermediate_size,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
    )
    ternary_state = build_ternary_state(args.model_dir, ternary_state_path, force=args.force)

    conversion_summary_path = args.out_dir / "conversion_summary.json"
    convert_stdout = args.out_dir / "convert.stdout"
    convert_stderr = args.out_dir / "convert.stderr"
    convert = run_command(
        [
            "python",
            str(args.converter),
            "--checkpoint-dir",
            str(args.model_dir),
            "--ternary-state",
            str(ternary_state_path),
            "--outfile",
            str(args.gguf),
            "--summary-json",
            str(conversion_summary_path),
            "--validate-codes",
            "--row-scale-qtype",
            "i2_sr",
            "--expect-ternary-keys",
            "3",
        ],
        stdout_path=convert_stdout,
        stderr_path=convert_stderr,
    )

    commands: dict[str, Any] = {"convert": convert}
    smoke_summary: dict[str, Any] = {}
    rss_summary: dict[str, Any] = {}

    if convert["returncode"] == 0:
        smoke_stdout = args.out_dir / "smoke.stdout"
        smoke_stderr = args.out_dir / "smoke.stderr"
        smoke = run_command(
            [
                str(args.llama_bin_dir / "llama-cli"),
                "-m",
                str(args.gguf),
                "-p",
                args.prompt,
                "-n",
                str(args.tokens),
                "-t",
                str(args.threads),
                "-ngl",
                "0",
                "--temp",
                "0",
                "--no-display-prompt",
            ],
            stdout_path=smoke_stdout,
            stderr_path=smoke_stderr,
        )
        commands["smoke"] = smoke
        smoke_summary = summarize_output(smoke_stdout, smoke_stderr)

        if smoke["returncode"] == 0:
            rss_stdout = args.out_dir / "rss.stdout"
            rss_stderr = args.out_dir / "rss.stderr"
            rss = run_command(
                [
                    "/usr/bin/time",
                    "-v",
                    str(args.llama_bin_dir / "llama-cli"),
                    "-m",
                    str(args.gguf),
                    "-p",
                    args.prompt,
                    "-n",
                    str(args.rss_tokens),
                    "-t",
                    str(args.threads),
                    "-ngl",
                    "0",
                    "--temp",
                    "0",
                    "--no-display-prompt",
                ],
                stdout_path=rss_stdout,
                stderr_path=rss_stderr,
            )
            commands["rss"] = rss
            rss_summary = parse_rss(rss_stderr)
    else:
        commands["smoke"] = {"returncode": None}

    conversion_summary = read_json(conversion_summary_path)
    gates = {
        "hf_checkpoint_created": args.model_dir.exists() and (args.model_dir / "config.json").exists(),
        "ternary_state_built": ternary_state.get("ternary_key_count") == 3
        and ternary_state.get("row_scale_key_count") == 3
        and not ternary_state.get("invalid_ternary_values"),
        "i2sr_converted": convert["returncode"] == 0 and args.gguf.exists() and args.gguf.stat().st_size > 0,
        "merged_expert_i2sr_packed": conversion_summary.get("ternary_i2s_packed") == 3
        and conversion_summary.get("row_scale_i2s_packed") == 3
        and conversion_summary.get("row_scale_qtype") == "i2_sr",
        "cpu_smoke_executed": commands.get("smoke", {}).get("returncode") == 0,
        "rss_measured": rss_summary.get("max_rss_kib") is not None,
        "qwen2moe_metadata_present": smoke_summary.get("architecture") == "qwen2moe"
        and smoke_summary.get("expert_count") == args.num_experts
        and smoke_summary.get("expert_used_count") == args.num_experts_per_tok,
    }
    result: dict[str, Any] = {
        "schema": "tiny-qwen2moe-ternary-i2sr-runtime-fixture-v1",
        "date": DATE,
        "i2_group_size": I2_GROUP_SIZE,
        "hf_model": hf_model,
        "ternary_state": ternary_state,
        "conversion_summary": conversion_summary,
        "gguf_path": str(args.gguf),
        "gguf_mib": args.gguf.stat().st_size / (1024**2) if args.gguf.exists() else None,
        "commands": commands,
        "smoke": smoke_summary,
        "rss": rss_summary,
        "gates": gates,
        "passed": all(gates.values()),
        "proves": [
            "Merged Qwen2MoE expert tensors can be represented as 3D row-scale ternary weights.",
            "The direct static ternary GGUF writer can emit those merged expert tensors as I2_SR when conversion passes.",
            "The vendored llama.cpp CPU runtime can execute routed Qwen2MoE with I2_SR experts when the smoke gate passes.",
        ],
        "does_not_prove": [
            "Kimi model support.",
            "Task quality or language-model quality.",
            "Router distillation quality.",
            "Expert-locality throughput behavior on a trained model.",
            "TL2 row-scale MoE support.",
        ],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(result), encoding="utf-8")
    print(render_markdown(result))

    if args.fail_on_error and not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
