#!/usr/bin/env python3
"""Run a deterministic prompt suite against a ternary checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_ternary import load_model, select_device  # noqa: E402


def load_prompts(path: Path, limit: int | None) -> list[dict[str, str]]:
    prompts: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "id" not in item or "prompt" not in item:
                raise ValueError(f"prompt entries must contain id and prompt: {item}")
            prompts.append({"id": str(item["id"]), "prompt": str(item["prompt"])})
            if limit is not None and len(prompts) >= limit:
                break
    if not prompts:
        raise ValueError(f"no prompts loaded from {path}")
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed prompts against a W1.58A8 ternary checkpoint")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--ternary-state", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--prompts", type=Path, default=ROOT / "benchmarks" / "prompts_core.jsonl")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="fp32", choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--quant-eps", type=float, default=1e-5)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    load_args = SimpleNamespace(
        checkpoint_dir=args.checkpoint_dir,
        ternary_state=args.ternary_state,
        model=args.model,
        tokenizer=args.tokenizer,
        dtype=args.dtype,
        quant_eps=args.quant_eps,
        trust_remote_code=args.trust_remote_code,
    )
    model, tokenizer = load_model(load_args)
    device = select_device(args.device)
    model.to(device)
    model.eval()

    prompts = load_prompts(args.prompts, args.limit)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as out:
        for item in prompts:
            encoded = tokenizer(item["prompt"], return_tensors="pt").to(device)
            input_tokens = int(encoded["input_ids"].shape[-1])
            start = time.perf_counter()
            with torch.inference_mode():
                output_ids = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            elapsed = time.perf_counter() - start
            total_tokens = int(output_ids.shape[-1])
            new_tokens = max(total_tokens - input_tokens, 0)
            full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            record = {
                "id": item["id"],
                "checkpoint_dir": args.checkpoint_dir,
                "device": str(device),
                "dtype": args.dtype,
                "prompt": item["prompt"],
                "full_text": full_text,
                "input_tokens": input_tokens,
                "new_tokens": new_tokens,
                "elapsed_seconds": elapsed,
                "tokens_per_second": new_tokens / elapsed if elapsed > 0 else None,
            }
            out.write(json.dumps(record, ensure_ascii=True) + "\n")
            out.flush()
            print(
                f"{item['id']}: new_tokens={new_tokens} elapsed={elapsed:.2f}s "
                f"tok_s={record['tokens_per_second']:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
