#!/usr/bin/env python3
"""Measure load size, RSS, prefill, and generation throughput.

This is a PyTorch runtime probe for HF and exported ternary checkpoints. It is
not a substitute for packed GGUF/TL2/I2_S benchmarking in bitnet.cpp.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from statistics import mean, median
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_ternary import dtype_from_name, load_model as load_ternary_model, select_device  # noqa: E402


def current_rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return None


def directory_size_bytes(path: str | None) -> int | None:
    if not path:
        return None
    root = Path(path)
    if not root.exists():
        return None
    if root.is_file():
        return root.stat().st_size
    return sum(item.stat().st_size for item in root.rglob("*") if item.is_file())


def file_size_bytes(path: Path | None) -> int | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    return path.stat().st_size


def default_ternary_state_path(args: argparse.Namespace) -> Path | None:
    if args.ternary_state:
        return Path(args.ternary_state)
    if args.checkpoint_dir:
        return Path(args.checkpoint_dir) / "ternary_state_dict.pt"
    return None


def checkpoint_safetensors_bytes(args: argparse.Namespace) -> int | None:
    if not args.checkpoint_dir:
        return None
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    return sum(item.stat().st_size for item in checkpoint_dir.glob("*.safetensors") if item.is_file())


def cpu_model_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def storage_bytes(model: torch.nn.Module) -> int:
    total = 0
    seen: set[tuple[int, str]] = set()
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor is None:
            continue
        storage = tensor.untyped_storage()
        key = (storage.data_ptr(), str(tensor.device))
        if key in seen:
            continue
        seen.add(key)
        total += storage.nbytes()
    return int(total)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_hf(args: argparse.Namespace):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_from_name(args.dtype),
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )
    return model, tokenizer


def load_candidate(args: argparse.Namespace):
    if args.model_kind == "hf":
        if not args.model:
            raise ValueError("--model is required for --model-kind hf")
        return load_hf(args)
    load_args = SimpleNamespace(
        checkpoint_dir=args.checkpoint_dir,
        ternary_state=args.ternary_state,
        model=args.model,
        tokenizer=args.tokenizer,
        dtype=args.dtype,
        quant_eps=args.quant_eps,
        trust_remote_code=args.trust_remote_code,
    )
    return load_ternary_model(load_args)


def make_prompt(tokenizer, target_tokens: int) -> torch.Tensor:
    seed = (
        "Machine learning systems should be evaluated with reproducible "
        "benchmarks, exact hardware notes, and clear failure modes. "
    )
    text = seed
    ids = tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    while len(ids) < target_tokens:
        text += seed
        ids = tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    ids = ids[:target_tokens]
    return torch.tensor([ids], dtype=torch.long)


def summarize_times(times: list[float]) -> dict[str, float]:
    return {
        "mean_seconds": mean(times),
        "median_seconds": median(times),
        "min_seconds": min(times),
        "max_seconds": max(times),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a PyTorch runtime probe for HF or ternary checkpoints")
    parser.add_argument("--model-kind", choices=["hf", "ternary"], required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--ternary-state", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="fp32", choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--quant-eps", type=float, default=1e-5)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
        torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)

    rss_before_load = current_rss_bytes()
    load_start = time.perf_counter()
    model, tokenizer = load_candidate(args)
    load_seconds = time.perf_counter() - load_start
    rss_after_load = current_rss_bytes()

    device = select_device(args.device)
    move_start = time.perf_counter()
    model.to(device)
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    sync(device)
    move_seconds = time.perf_counter() - move_start
    rss_after_move = current_rss_bytes()

    input_ids = make_prompt(tokenizer, args.prompt_tokens).to(device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    actual_prompt_tokens = int(input_ids.shape[-1])

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    prefill_times: list[float] = []
    generate_times: list[float] = []
    total_runs = args.warmup + args.repeats
    with torch.inference_mode():
        for index in range(total_runs):
            start = time.perf_counter()
            _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
            sync(device)
            prefill_elapsed = time.perf_counter() - start

            start = time.perf_counter()
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            sync(device)
            generate_elapsed = time.perf_counter() - start
            if index >= args.warmup:
                prefill_times.append(prefill_elapsed)
                generate_times.append(generate_elapsed)

    generated_tokens = int(output_ids.shape[-1] - input_ids.shape[-1])
    prefill = summarize_times(prefill_times)
    generate = summarize_times(generate_times)
    decode_estimate = max(generate["median_seconds"] - prefill["median_seconds"], 1e-9)

    result = {
        "model_kind": args.model_kind,
        "model": args.model,
        "checkpoint_dir": args.checkpoint_dir,
        "ternary_state": args.ternary_state,
        "device": str(device),
        "dtype": args.dtype,
        "torch_version": torch.__version__,
        "cpu_model": cpu_model_name(),
        "python_platform": platform.platform(),
        "torch_num_threads": torch.get_num_threads(),
        "prompt_tokens": actual_prompt_tokens,
        "max_new_tokens": args.max_new_tokens,
        "generated_tokens_observed": generated_tokens,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "load_seconds": load_seconds,
        "move_seconds": move_seconds,
        "rss_before_load_bytes": rss_before_load,
        "rss_after_load_bytes": rss_after_load,
        "rss_after_move_bytes": rss_after_move,
        "model_storage_bytes": storage_bytes(model),
        "checkpoint_disk_bytes": directory_size_bytes(args.checkpoint_dir or args.model),
        "checkpoint_safetensors_bytes": checkpoint_safetensors_bytes(args),
        "ternary_state_bytes": file_size_bytes(default_ternary_state_path(args)),
        "prefill": {
            **prefill,
            "tokens_per_second_median": actual_prompt_tokens / prefill["median_seconds"],
        },
        "generate": {
            **generate,
            "new_tokens_per_second_median_including_prefill": generated_tokens / generate["median_seconds"],
            "decode_tokens_per_second_estimate": generated_tokens / decode_estimate,
        },
    }
    if device.type == "cuda":
        result["cuda_max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        result["cuda_device_name"] = torch.cuda.get_device_name(device)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
