#!/usr/bin/env python3
"""Run small multiple-choice log-likelihood evaluations.

This is a controlled in-repo evaluator for early benchmarking. It is not a
drop-in replacement for EleutherAI lm-eval, but it uses the same core idea:
score each answer choice by conditional log-likelihood under the model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from run_perplexity import load_candidate  # noqa: E402
from eval_ternary import select_device  # noqa: E402


def load_task(task: str, split: str, limit: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    if task == "piqa":
        dataset = load_dataset("piqa", split=split, trust_remote_code=True)
        rows = []
        for row in dataset.select(range(min(limit, len(dataset)))):
            rows.append({
                "id": str(row.get("id", len(rows))),
                "prompt": f"Question: {row['goal']}\nAnswer:",
                "choices": [" " + row["sol1"], " " + row["sol2"]],
                "label": int(row["label"]),
            })
        return rows

    if task in {"arc_easy", "arc_challenge"}:
        config = "ARC-Easy" if task == "arc_easy" else "ARC-Challenge"
        dataset = load_dataset("ai2_arc", config, split=split)
        rows = []
        for row in dataset.select(range(min(limit, len(dataset)))):
            labels = [str(label) for label in row["choices"]["label"]]
            answer = str(row["answerKey"])
            if answer not in labels:
                continue
            rows.append({
                "id": str(row["id"]),
                "prompt": f"Question: {row['question']}\nAnswer:",
                "choices": [" " + text for text in row["choices"]["text"]],
                "label": labels.index(answer),
            })
        return rows

    if task == "hellaswag":
        dataset = load_dataset("Rowan/hellaswag", split=split)
        rows = []
        for row in dataset.select(range(min(limit, len(dataset)))):
            ctx = (row.get("ctx_a", "") + " " + row.get("ctx_b", "")).strip()
            rows.append({
                "id": str(row.get("ind", len(rows))),
                "prompt": ctx,
                "choices": [" " + ending for ending in row["endings"]],
                "label": int(row["label"]),
            })
        return rows

    raise ValueError(f"unsupported task={task}")


def trim_to_length(input_ids: list[int], labels: list[int], max_seq_len: int) -> tuple[list[int], list[int]]:
    if max_seq_len <= 0 or len(input_ids) <= max_seq_len:
        return input_ids, labels
    keep_start = len(input_ids) - max_seq_len
    input_ids = input_ids[keep_start:]
    labels = labels[keep_start:]
    if all(label == -100 for label in labels):
        raise ValueError("max_seq_len truncated away the whole continuation")
    return input_ids, labels


def score_choice(
    model,
    tokenizer,
    *,
    prompt: str,
    continuation: str,
    device: torch.device,
    max_seq_len: int,
) -> tuple[float, float, int]:
    context_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    continuation_ids = tokenizer(continuation, add_special_tokens=False)["input_ids"]
    if not continuation_ids:
        return -math.inf, -math.inf, 0

    input_ids = context_ids + continuation_ids
    labels = [-100] * len(context_ids) + continuation_ids
    input_ids, labels = trim_to_length(input_ids, labels, max_seq_len)
    continuation_tokens = sum(1 for label in labels if label != -100)

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    labels_tensor = torch.tensor([labels], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_tensor, dtype=torch.long, device=device)
    with torch.inference_mode():
        outputs = model(input_ids=input_tensor, attention_mask=attention_mask, labels=labels_tensor, use_cache=False)
    sum_logprob = -float(outputs.loss.detach().cpu()) * float(continuation_tokens)
    avg_logprob = sum_logprob / float(max(continuation_tokens, 1))
    return sum_logprob, avg_logprob, continuation_tokens


def evaluate_task(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer = load_candidate(args)
    device = select_device(args.device)
    model.to(device)
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False

    rows = load_task(args.task, args.split, args.limit)
    if not rows:
        raise ValueError(f"task {args.task} produced no rows")

    records = []
    raw_correct = 0
    norm_correct = 0
    start = time.perf_counter()
    for index, row in enumerate(rows):
        raw_scores = []
        norm_scores = []
        token_counts = []
        for choice in row["choices"]:
            raw, norm, count = score_choice(
                model,
                tokenizer,
                prompt=row["prompt"],
                continuation=choice,
                device=device,
                max_seq_len=args.max_seq_len,
            )
            raw_scores.append(raw)
            norm_scores.append(norm)
            token_counts.append(count)

        raw_pred = max(range(len(raw_scores)), key=lambda i: raw_scores[i])
        norm_pred = max(range(len(norm_scores)), key=lambda i: norm_scores[i])
        label = int(row["label"])
        raw_correct += int(raw_pred == label)
        norm_correct += int(norm_pred == label)
        record = {
            "index": index,
            "id": row["id"],
            "label": label,
            "raw_pred": raw_pred,
            "norm_pred": norm_pred,
            "raw_scores": raw_scores,
            "norm_scores": norm_scores,
            "token_counts": token_counts,
        }
        records.append(record)
        if args.log_every > 0 and (index + 1) % args.log_every == 0:
            print(
                f"{args.task} {index + 1}/{len(rows)} "
                f"acc={raw_correct / (index + 1):.4f} acc_norm={norm_correct / (index + 1):.4f}",
                flush=True,
            )

    elapsed = time.perf_counter() - start
    result = {
        "task": args.task,
        "split": args.split,
        "model_kind": args.model_kind,
        "model": args.model,
        "checkpoint_dir": args.checkpoint_dir,
        "limit": len(rows),
        "device": str(device),
        "dtype": args.dtype,
        "max_seq_len": args.max_seq_len,
        "accuracy": raw_correct / len(rows),
        "accuracy_norm": norm_correct / len(rows),
        "elapsed_seconds": elapsed,
        "examples_per_second": len(rows) / elapsed if elapsed > 0 else 0.0,
    }
    if args.output_jsonl:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_jsonl.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Multiple-choice log-likelihood eval")
    parser.add_argument("--task", choices=["piqa", "arc_easy", "arc_challenge", "hellaswag"], required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--model-kind", choices=["hf", "ternary"], required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--ternary-state", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--quant-eps", type=float, default=1e-5)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    result = evaluate_task(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
