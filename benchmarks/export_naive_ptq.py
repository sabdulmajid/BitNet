#!/usr/bin/env python3
"""Create a naive PTQ ternary checkpoint from a Hugging Face CausalLM.

This baseline intentionally performs no training, calibration, or distillation.
It saves the source model/tokenizer, then exports ternary linear weights using
the same absmean projection used by the QAT exporter. Comparing this checkpoint
against QAT/distilled checkpoints answers whether training under ternary
constraints improved over blind post-training ternarization.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_ternary import dtype_from_name  # noqa: E402
from export_ternary import DEFAULT_LINEAR_KEY_REGEX, export_ternary, safetensor_files  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a naive PTQ W1.58A8 checkpoint from a HF model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--scale-mode", default="tensor", choices=["tensor", "row"])
    parser.add_argument("--quant-eps", type=float, default=1e-5)
    parser.add_argument("--linear-key-regex", default=DEFAULT_LINEAR_KEY_REGEX)
    parser.add_argument("--expect-ternary-keys", type=int, default=None)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(args.output_dir)

    dtype = dtype_from_name(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )
    model.save_pretrained(args.output_dir, safe_serialization=True)
    if hasattr(model, "config"):
        model.config.save_pretrained(args.output_dir)
    if hasattr(model, "generation_config"):
        model.generation_config.save_pretrained(args.output_dir)
    del model

    stats = export_ternary(
        safetensor_files(args.output_dir, None),
        args.output_dir / "ternary_state_dict.pt",
        linear_key_regex=args.linear_key_regex,
        scale_mode=args.scale_mode,
        eps=args.quant_eps,
        backup_existing=True,
        dry_run=False,
    )
    if args.expect_ternary_keys is not None and stats["ternary_keys"] != args.expect_ternary_keys:
        raise SystemExit(f"expected {args.expect_ternary_keys} ternary keys, found {stats['ternary_keys']}")

    manifest = {
        "baseline": "naive_ptq",
        "model": args.model,
        "dtype": args.dtype,
        "scale_mode": args.scale_mode,
        "quant_eps": args.quant_eps,
        **stats,
    }
    (args.output_dir / "ptq_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
