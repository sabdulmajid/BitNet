#!/usr/bin/env python3
"""Run EleutherAI lm-eval on HF or exported ternary checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_ternary import load_model as load_ternary_model, select_device  # noqa: E402


def load_ternary_hflm(args: argparse.Namespace):
    from lm_eval.models.huggingface import HFLM

    load_args = SimpleNamespace(
        checkpoint_dir=args.checkpoint_dir,
        ternary_state=args.ternary_state,
        model=args.model,
        tokenizer=args.tokenizer,
        dtype=args.dtype,
        quant_eps=args.quant_eps,
        trust_remote_code=args.trust_remote_code,
    )
    model, tokenizer = load_ternary_model(load_args)
    device = select_device(args.device)
    model.to(device)
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False
    return HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        backend="causal",
        batch_size=args.batch_size,
        max_length=args.max_seq_len,
        device=str(device),
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lm-eval for HF or ternary checkpoints")
    parser.add_argument("--model-kind", choices=["hf", "ternary"], required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--ternary-state", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--tasks", default="piqa")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--batch-size", default="1")
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--quant-eps", type=float, default=1e-5)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--confirm-run-unsafe-code", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    from lm_eval import evaluator

    torch.set_grad_enabled(False)
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if not tasks:
        raise ValueError("--tasks cannot be empty")

    if args.model_kind == "hf":
        if not args.model:
            raise ValueError("--model is required for --model-kind hf")
        model = "hf"
        model_args = {
            "pretrained": args.model,
            "dtype": args.dtype,
            "trust_remote_code": args.trust_remote_code,
        }
    else:
        model = load_ternary_hflm(args)
        model_args = None

    results = evaluator.simple_evaluate(
        model=model,
        model_args=model_args,
        tasks=tasks,
        limit=args.limit,
        batch_size=args.batch_size,
        device=args.device,
        log_samples=True,
        confirm_run_unsafe_code=args.confirm_run_unsafe_code,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(results.get("results", {}), indent=2, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
