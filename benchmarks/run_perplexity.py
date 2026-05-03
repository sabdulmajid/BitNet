#!/usr/bin/env python3
"""Compute causal-LM perplexity on packed heldout text blocks.

This runner is intentionally model-agnostic enough to compare:

* Hugging Face FP/BF16 baselines via --model-kind hf
* exported W1.58A8 ternary checkpoints via --model-kind ternary
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_ternary import dtype_from_name, load_model as load_ternary_model, select_device  # noqa: E402


def load_hf_model(args: argparse.Namespace):
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
        return load_hf_model(args)
    if args.model_kind == "ternary":
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
    raise ValueError(f"unknown model_kind={args.model_kind}")


def iter_dataset_texts(args: argparse.Namespace) -> Iterator[str]:
    from datasets import load_dataset

    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.dataset_split,
        streaming=args.dataset_streaming,
    )
    iterator = iter(dataset)
    if args.skip_rows > 0:
        iterator = itertools.islice(iterator, args.skip_rows, None)

    text_column = args.text_column
    if text_column is None:
        first_row = next(iterator, None)
        if first_row is None:
            return
        string_columns = [name for name, value in first_row.items() if isinstance(value, str)]
        if not string_columns:
            raise ValueError("no string column found; pass --text-column")
        text_column = string_columns[0]
        iterator = itertools.chain([first_row], iterator)

    rows = iterator if args.max_rows <= 0 else itertools.islice(iterator, args.max_rows)
    for row in rows:
        text = row.get(text_column)
        if isinstance(text, str) and text.strip():
            yield text


def build_blocks(args: argparse.Namespace, tokenizer) -> torch.Tensor:
    eos_token_id = tokenizer.eos_token_id
    token_buffer: list[int] = []
    blocks: list[list[int]] = []
    pending: list[str] = []
    total_tokens = 0
    text_rows = 0

    def encode_pending() -> None:
        nonlocal pending, token_buffer, total_tokens
        if not pending:
            return
        encoded = tokenizer(pending, add_special_tokens=False, return_attention_mask=False, truncation=False)
        pending = []
        for ids in encoded["input_ids"]:
            if not ids:
                continue
            token_buffer.extend(ids)
            total_tokens += len(ids)
            if eos_token_id is not None:
                token_buffer.append(eos_token_id)
                total_tokens += 1
            while len(token_buffer) >= args.max_seq_len and len(blocks) < args.max_blocks:
                blocks.append(token_buffer[: args.max_seq_len])
                token_buffer = token_buffer[args.max_seq_len :]

    for text in iter_dataset_texts(args):
        if len(blocks) >= args.max_blocks:
            break
        text_rows += 1
        pending.append(text)
        if len(pending) >= args.tokenizer_batch_size:
            encode_pending()
    encode_pending()

    if not blocks:
        raise ValueError("dataset did not yield enough tokens for one eval block")
    print(
        f"Packed {text_rows} rows into {len(blocks)} blocks x {args.max_seq_len}; "
        f"total_tokens={total_tokens}; remainder={len(token_buffer)}",
        flush=True,
    )
    return torch.tensor(blocks, dtype=torch.long)


def evaluate(model, blocks: torch.Tensor, device: torch.device, batch_size: int) -> dict[str, float]:
    nll = 0.0
    token_count = 0
    start = time.perf_counter()
    for start_index in range(0, blocks.shape[0], batch_size):
        input_ids = blocks[start_index : start_index + batch_size].to(device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids, use_cache=False)
        shifted_tokens = int(attention_mask[:, 1:].sum().item())
        nll += float(outputs.loss.detach().cpu()) * shifted_tokens
        token_count += shifted_tokens
    elapsed = time.perf_counter() - start
    mean_nll = nll / max(token_count, 1)
    return {
        "nll": mean_nll,
        "perplexity": math.exp(mean_nll) if mean_nll < 100 else float("inf"),
        "eval_tokens": float(token_count),
        "elapsed_seconds": elapsed,
        "tokens_per_second": token_count / elapsed if elapsed > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run packed-text perplexity for HF or ternary causal LMs")
    parser.add_argument("--model-kind", choices=["hf", "ternary"], required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--ternary-state", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--dataset-name", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--dataset-streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--skip-rows", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-blocks", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--tokenizer-batch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="fp32", choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--quant-eps", type=float, default=1e-5)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    model, tokenizer = load_candidate(args)
    device = select_device(args.device)
    model.to(device)
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False

    blocks = build_blocks(args, tokenizer)
    metrics = evaluate(model, blocks, device, args.batch_size)
    result = {
        "model_kind": args.model_kind,
        "model": args.model,
        "checkpoint_dir": args.checkpoint_dir,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "skip_rows": args.skip_rows,
        "max_blocks": args.max_blocks,
        "max_seq_len": args.max_seq_len,
        "batch_size": args.batch_size,
        "device": str(device),
        "dtype": args.dtype,
        **metrics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
