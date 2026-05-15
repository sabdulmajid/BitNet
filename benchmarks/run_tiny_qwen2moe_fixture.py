#!/usr/bin/env python3
"""Build and execute a tiny random Qwen2MoE GGUF fixture.

This is a runtime contract probe, not a quality benchmark. It proves that the
vendored llama.cpp converter and CPU runtime can carry a minimal Qwen2MoE model
through FP16 GGUF conversion and routed execution. It does not prove Kimi
support, ternary MoE support, router quality, or useful task accuracy.
"""

from __future__ import annotations

import os
import argparse
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, Qwen2MoeConfig, Qwen2MoeForCausalLM


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()


def run_command(command: list[str], *, stdout_path: Path, stderr_path: Path, skip_existing: bool = False) -> dict[str, Any]:
    if skip_existing and stdout_path.exists() and stderr_path.exists():
        return {
            "command": command,
            "returncode": 0,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "skipped": True,
            "elapsed_seconds": 0.0,
        }

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
        "skipped": False,
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


def make_tiny_model(
    model_dir: Path,
    tokenizer_model: str,
    *,
    seed: int,
    force: bool,
    num_experts: int = 2,
    num_experts_per_tok: int = 1,
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
            },
        }

    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    tokenizer.save_pretrained(model_dir)

    torch.manual_seed(seed)
    config = Qwen2MoeConfig(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=96,
        shared_expert_intermediate_size=96,
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

    param_count = sum(parameter.numel() for parameter in model.parameters())
    return {
        "created": True,
        "model_dir": str(model_dir),
        "parameter_count": param_count,
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


def summarize_output(stdout_path: Path, stderr_path: Path) -> dict[str, Any]:
    stdout = read_text(stdout_path)
    stderr = read_text(stderr_path)
    combined = stdout + "\n" + stderr
    generated = " ".join(stdout.split())[:240]
    return {
        "generated_prefix": generated,
        "architecture": find_first(r"general\.architecture\s+str\s+=\s+([^\n]+)", combined),
        "expert_count": parse_int(r"qwen2moe\.expert_count\s+u32\s+=\s+([0-9]+)", combined),
        "expert_used_count": parse_int(r"qwen2moe\.expert_used_count\s+u32\s+=\s+([0-9]+)", combined),
        "model_params_m": parse_float(r"model params\s+=\s+([0-9.]+)\s+M", combined),
        "cpu_buffer_mib": parse_float(r"(?:CPU_Mapped model|CPU) buffer size\s+=\s+([0-9.]+)\s+MiB", combined),
        "prompt_eval_tok_s": parse_float(r"prompt eval time =.+?([0-9.]+) tokens per second", combined),
        "decode_tok_s": parse_float(r"llama_perf_context_print:\s+eval time =.+?([0-9.]+) tokens per second", combined),
        "graph_nodes": parse_int(r"graph nodes\s+=\s+([0-9]+)", combined),
    }


def parse_rss(stderr_path: Path) -> dict[str, Any]:
    text = read_text(stderr_path)
    rss_kib = parse_int(r"Maximum resident set size \(kbytes\):\s+([0-9]+)", text)
    elapsed = find_first(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s+([^\n]+)", text)
    return {
        "max_rss_kib": rss_kib,
        "max_rss_mib": rss_kib / 1024 if rss_kib is not None else None,
        "elapsed": elapsed,
    }


def md_bool(value: bool) -> str:
    return "yes" if value else "no"


def render_markdown(result: dict[str, Any]) -> str:
    smoke = result.get("smoke", {})
    gates = result.get("gates", {})
    rows = [
        ["HF checkpoint created", md_bool(gates.get("hf_checkpoint_created", False)), result["hf_model"].get("model_dir", "")],
        ["GGUF converted", md_bool(gates.get("gguf_converted", False)), result.get("gguf_path", "")],
        ["CPU smoke executed", md_bool(gates.get("cpu_smoke_executed", False)), str(result["commands"]["smoke"].get("returncode"))],
        ["Peak RSS measured", md_bool(gates.get("rss_measured", False)), f"{result.get('rss', {}).get('max_rss_mib')} MiB"],
        ["Qwen2MoE metadata present", md_bool(gates.get("qwen2moe_metadata_present", False)), f"experts={smoke.get('expert_count')}; used={smoke.get('expert_used_count')}"],
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
            f"# Tiny Qwen2MoE Runtime Fixture, {DATE}",
            "This fixture creates a tiny random `Qwen2MoeForCausalLM`, converts it to FP16 GGUF with the vendored llama.cpp converter, and runs `llama-cli` on CPU.",
            table,
            "## Runtime Snapshot",
            (
                f"Architecture: `{smoke.get('architecture')}`; "
                f"params: `{smoke.get('model_params_m')}` M; "
                f"CPU buffer: `{smoke.get('cpu_buffer_mib')}` MiB; "
                f"peak RSS: `{result.get('rss', {}).get('max_rss_mib')}` MiB; "
                f"prompt: `{smoke.get('prompt_eval_tok_s')}` tok/s; "
                f"decode: `{smoke.get('decode_tok_s')}` tok/s."
            ),
            "## Interpretation",
            (
                "This is a positive converter/runtime fixture for a minimal random Qwen2MoE FP16 model. "
                "It proves generic Qwen2MoE metadata and routed CPU execution are reachable in the vendored llama.cpp stack."
            ),
            (
                "It does not prove Kimi support, ternary MoE support, BitDistill MoE training, router quality, "
                "task quality, expert-locality behavior, TL2 expert kernels, or row-scale `I2_SR` MoE quality."
            ),
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--model-dir", type=Path, default=Path("models/tiny-qwen2moe-fixture"))
    parser.add_argument("--gguf", type=Path, default=Path("models/tiny-qwen2moe-fixture/tiny-qwen2moe-f16.gguf"))
    parser.add_argument("--converter", type=Path, default=Path("3rdparty/llama.cpp/convert_hf_to_gguf.py"))
    parser.add_argument("--llama-bin-dir", type=Path, default=Path("build-portable-avx2/bin"))
    parser.add_argument("--out-dir", type=Path, default=Path(f"benchmark_results/tiny_qwen2moe_fixture_{DATE}"))
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/tiny_qwen2moe_fixture_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/tiny_qwen2moe_fixture_{DATE}.md"))
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--rss-tokens", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--num-experts", type=int, default=2)
    parser.add_argument("--num-experts-per-tok", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    hf_model = make_tiny_model(
        args.model_dir,
        args.tokenizer_model,
        seed=args.seed,
        force=args.force,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
    )

    convert_stdout = args.out_dir / "convert.stdout"
    convert_stderr = args.out_dir / "convert.stderr"
    convert = run_command(
        [
            "python",
            str(args.converter),
            str(args.model_dir),
            "--outfile",
            str(args.gguf),
            "--outtype",
            "f16",
        ],
        stdout_path=convert_stdout,
        stderr_path=convert_stderr,
        skip_existing=args.skip_existing and args.gguf.exists(),
    )
    if convert["returncode"] != 0:
        raise RuntimeError(f"conversion failed; see {convert_stdout} and {convert_stderr}")

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
        skip_existing=False,
    )
    if smoke["returncode"] != 0:
        raise RuntimeError(f"llama-cli smoke failed; see {smoke_stdout} and {smoke_stderr}")

    rss_stdout = args.out_dir / "rss.stdout"
    rss_stderr = args.out_dir / "rss.stderr"
    rss_command = [
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
    ]
    rss_run = run_command(rss_command, stdout_path=rss_stdout, stderr_path=rss_stderr, skip_existing=False)
    if rss_run["returncode"] != 0:
        raise RuntimeError(f"llama-cli RSS probe failed; see {rss_stdout} and {rss_stderr}")

    smoke_summary = summarize_output(smoke_stdout, smoke_stderr)
    rss_summary = parse_rss(rss_stderr)
    gates = {
        "hf_checkpoint_created": args.model_dir.exists() and (args.model_dir / "config.json").exists(),
        "gguf_converted": args.gguf.exists() and args.gguf.stat().st_size > 0,
        "cpu_smoke_executed": smoke["returncode"] == 0,
        "rss_measured": rss_summary.get("max_rss_kib") is not None,
        "qwen2moe_metadata_present": smoke_summary.get("architecture") == "qwen2moe"
        and smoke_summary.get("expert_count") == 2
        and smoke_summary.get("expert_used_count") == 1,
    }
    result: dict[str, Any] = {
        "schema": "tiny-qwen2moe-runtime-fixture-v1",
        "date": DATE,
        "hf_model": hf_model,
        "gguf_path": str(args.gguf),
        "gguf_mib": args.gguf.stat().st_size / (1024**2) if args.gguf.exists() else None,
        "commands": {"convert": convert, "smoke": smoke, "rss": rss_run},
        "smoke": smoke_summary,
        "rss": rss_summary,
        "gates": gates,
        "passed": all(gates.values()),
        "proves": [
            "A minimal random Qwen2MoE Hugging Face checkpoint can be converted to FP16 GGUF.",
            "The converted GGUF exposes Qwen2MoE expert metadata.",
            "The vendored llama.cpp CPU runtime can execute a routed Qwen2MoE graph for this fixture.",
        ],
        "does_not_prove": [
            "Kimi model support.",
            "Ternary or BitNet MoE support.",
            "Task quality or language-model quality.",
            "Router distillation quality.",
            "Expert-locality throughput behavior.",
            "TL2 or row-scale I2_SR MoE runtime correctness.",
        ],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(result), encoding="utf-8")
    print(render_markdown(result))


if __name__ == "__main__":
    main()
