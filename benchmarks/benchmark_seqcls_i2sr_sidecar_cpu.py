#!/usr/bin/env python3
"""Evaluate a packed I2_SR sequence-classification backbone with a dense sidecar head."""

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

import numpy as np


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()
TASK_SPECS = {
    "mnli": {
        "dataset": ("glue", "mnli"),
        "eval_split": "validation_matched",
        "text_keys": ("premise", "hypothesis"),
        "expected_examples": 9815,
    },
    "qnli": {
        "dataset": ("glue", "qnli"),
        "eval_split": "validation",
        "text_keys": ("question", "sentence"),
        "expected_examples": 5463,
    },
    "sst2": {
        "dataset": ("glue", "sst2"),
        "eval_split": "validation",
        "text_keys": ("sentence", None),
        "expected_examples": 872,
    },
}
DEFAULT_CHECKPOINT = Path(
    "checkpoints/bitdistill-glue-seqcls-longwarmup/"
    "Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8"
)
DEFAULT_GGUF = Path(
    "models/seqcls-backbone-i2sr/"
    "Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8_bitnet_qwen_i2_sr.gguf"
)
DEFAULT_HEAD = Path(
    "models/seqcls-backbone-i2sr/"
    "Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8_score_head.npz"
)

PROMPT_EVAL_RE = re.compile(
    r"prompt eval time =\s+(?P<ms>[0-9.]+) ms /\s+(?P<tokens>[0-9]+) tokens.*?"
    r"(?P<tps>[0-9.]+) tokens per second"
)
LOAD_RE = re.compile(r"load time =\s+(?P<ms>[0-9.]+) ms")
TOTAL_RE = re.compile(r"total time =\s+(?P<ms>[0-9.]+) ms")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_prediction_trace(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if limit and len(rows) >= limit:
                break
            rows.append(json.loads(line))
    return rows


def maybe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def load_rows(task: str, limit: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    spec = TASK_SPECS[task]
    dataset_name, dataset_config = spec["dataset"]
    dataset = load_dataset(dataset_name, dataset_config)[spec["eval_split"]]
    if limit > 0:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return [dict(row) for row in dataset]


def render_prompt(tokenizer: Any, task: str, row: dict[str, Any]) -> str:
    spec = TASK_SPECS[task]
    text_a, text_b = spec["text_keys"]
    if text_b is None:
        encoded = tokenizer(row[text_a], truncation=True, max_length=512, add_special_tokens=True)
    else:
        encoded = tokenizer(row[text_a], row[text_b], truncation=True, max_length=512, add_special_tokens=True)
    # Decoding the tokenizer's exact pair input IDs avoids guessing separators.
    return tokenizer.decode(encoded["input_ids"], clean_up_tokenization_spaces=False)


def parse_perf(stderr: str) -> dict[str, Any]:
    prompt_match = PROMPT_EVAL_RE.search(stderr)
    load_match = LOAD_RE.search(stderr)
    total_match = TOTAL_RE.search(stderr)
    return {
        "load_time_ms": float(load_match.group("ms")) if load_match else None,
        "prompt_eval_ms": float(prompt_match.group("ms")) if prompt_match else None,
        "prompt_eval_tokens": int(prompt_match.group("tokens")) if prompt_match else None,
        "prompt_eval_tokens_per_second": float(prompt_match.group("tps")) if prompt_match else None,
        "total_time_ms": float(total_match.group("ms")) if total_match else None,
    }


def run_batch(
    *,
    binary: Path,
    gguf: Path,
    prompts: list[str],
    separator: str,
    threads: int,
    ctx_size: int,
    timeout_seconds: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if any(separator in prompt for prompt in prompts):
        raise ValueError("separator occurs inside a prompt")
    command = [
        str(binary),
        "-m",
        str(gguf),
        "-p",
        separator.join(prompts),
        "--embd-separator",
        separator,
        "--pooling",
        "last",
        "--attention",
        "causal",
        "--embd-output-format",
        "json",
        "--embd-normalize",
        "-1",
        "-ngl",
        "0",
        "-t",
        str(threads),
        "-c",
        str(ctx_size),
    ]
    started = time.perf_counter()
    result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout_seconds)
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-4000:])
    parsed = json.loads(result.stdout)
    embeddings = [row["embedding"] for row in parsed.get("data", [])]
    if len(embeddings) != len(prompts):
        raise RuntimeError(f"expected {len(prompts)} embeddings, got {len(embeddings)}")
    meta = {
        "elapsed_seconds": elapsed,
        "stdout_bytes": len(result.stdout.encode("utf-8")),
        "stderr_bytes": len(result.stderr.encode("utf-8")),
        "perf": parse_perf(result.stderr),
    }
    return np.asarray(embeddings, dtype=np.float32), meta


def accuracy_ci_wilson(correct: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    phat = correct / total
    denom = 1.0 + z * z / total
    center = (phat + z * z / (2.0 * total)) / denom
    half = z * ((phat * (1.0 - phat) / total + z * z / (4.0 * total * total)) ** 0.5) / denom
    return [center - half, center + half]


def summarize_agreement(predictions: list[int], labels: list[int], trace: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(int(pred == label) for pred, label in zip(predictions, labels, strict=False))
    total = len(labels)
    trace_predictions = [
        int(row["prediction"])
        for row in trace[:total]
        if isinstance(row, dict) and isinstance(row.get("prediction"), int)
    ]
    trace_labels = [int(row["label"]) for row in trace[:total] if isinstance(row, dict) and isinstance(row.get("label"), int)]
    trace_agreement = None
    trace_label_match = None
    if len(trace_predictions) == total:
        trace_agreement = sum(int(a == b) for a, b in zip(predictions, trace_predictions, strict=True)) / total
    if len(trace_labels) == total:
        trace_label_match = sum(int(a == b) for a, b in zip(labels, trace_labels, strict=True)) / total
    return {
        "examples": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "accuracy_ci95_wilson": accuracy_ci_wilson(correct, total),
        "saved_trace_predictions": len(trace_predictions),
        "agreement_with_saved_pytorch_predictions": trace_agreement,
        "label_agreement_with_saved_trace": trace_label_match,
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
    summary = result["summary"]
    runtime = result["runtime"]
    checkpoint = result["checkpoint"]
    return "\n\n".join(
        [
            f"# Sequence-Classification I2_SR Sidecar CPU Benchmark, {result['date']}",
            (
                "This benchmark evaluates the packed I2_SR decoder backbone with the dense classifier "
                "head applied outside llama.cpp. It is a sidecar prototype, not native GGUF classifier support."
            ),
            md_table(
                ["field", "value"],
                [
                    ["status", result["status"]],
                    ["task", result["task"]],
                    ["examples", summary["examples"]],
                    ["accuracy", summary["accuracy"]],
                    ["accuracy CI95", summary["accuracy_ci95_wilson"]],
                    ["stored PyTorch accuracy", checkpoint.get("stored_accuracy")],
                    ["agreement with saved PyTorch predictions", summary["agreement_with_saved_pytorch_predictions"]],
                    ["label agreement with saved trace", summary["label_agreement_with_saved_trace"]],
                    ["batch size", result["batch_size"]],
                    ["wall seconds", runtime["wall_seconds"]],
                    ["examples/sec", runtime["examples_per_second"]],
                    ["tokens/sec aggregate", runtime["prompt_eval_tokens_per_second_aggregate"]],
                ],
            ),
            "## Interpretation",
            (
                "If agreement with saved PyTorch predictions is low, the sidecar path is only a load/runtime "
                "prototype and still needs tokenization, pooling, or runtime-head alignment before it can be used "
                "as a deployed classifier."
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--task", choices=sorted(TASK_SPECS), default="mnli")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--classifier-head", type=Path, default=DEFAULT_HEAD)
    parser.add_argument("--embedding-binary", type=Path, default=Path("build-portable-avx2/bin/llama-embedding"))
    parser.add_argument("--max-samples", type=int, default=512, help="0 means full validation split")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Keep this at 1 for the current prototype. llama-embedding separator batching "
            "is not a reliable persistent classifier runtime in this path."
        ),
    )
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--ctx-size", type=int, default=512)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--separator", default="<#BITNET_EVAL_SEP#>")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(f"benchmark_results/seqcls_i2sr_sidecar_cpu_mnli_512_{DATE}.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(f"benchmarks/results/seqcls_i2sr_sidecar_cpu_mnli_512_{DATE}.md"),
    )
    args = parser.parse_args()

    root = args.repo_root.resolve()
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir.is_absolute() else root / args.checkpoint_dir
    gguf = args.gguf if args.gguf.is_absolute() else root / args.gguf
    classifier_head = args.classifier_head if args.classifier_head.is_absolute() else root / args.classifier_head
    embedding_binary = args.embedding_binary if args.embedding_binary.is_absolute() else root / args.embedding_binary

    from transformers import AutoTokenizer

    metrics = read_json(checkpoint_dir / "metrics.json")
    eval_metrics = metrics.get("eval", {}) if isinstance(metrics.get("eval"), dict) else {}
    prediction_trace_path = eval_metrics.get("prediction_path") or str(checkpoint_dir / "eval_predictions.jsonl")
    prediction_trace = read_prediction_trace(root / prediction_trace_path, args.max_samples)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    rows = load_rows(args.task, args.max_samples)
    prompts = [render_prompt(tokenizer, args.task, row) for row in rows]
    labels = [int(row["label"]) for row in rows]

    head = np.load(classifier_head)
    weight_key = "score_weight" if "score_weight" in head.files else "classifier_weight"
    bias_key = "score_bias" if "score_bias" in head.files else "classifier_bias"
    weight = np.asarray(head[weight_key], dtype=np.float32)
    bias = np.asarray(head[bias_key], dtype=np.float32) if bias_key in head.files else None

    batch_metas: list[dict[str, Any]] = []
    predictions: list[int] = []
    started = time.perf_counter()
    for start in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[start : start + args.batch_size]
        embeddings, meta = run_batch(
            binary=embedding_binary,
            gguf=gguf,
            prompts=batch_prompts,
            separator=args.separator,
            threads=args.threads,
            ctx_size=args.ctx_size,
            timeout_seconds=args.timeout_seconds,
        )
        logits = embeddings @ weight.T
        if bias is not None:
            logits = logits + bias
        predictions.extend([int(x) for x in np.argmax(logits, axis=-1)])
        batch_metas.append(meta)
    wall_seconds = time.perf_counter() - started

    total_prompt_tokens = sum(
        int(meta.get("perf", {}).get("prompt_eval_tokens") or 0) for meta in batch_metas
    )
    total_prompt_ms = sum(float(meta.get("perf", {}).get("prompt_eval_ms") or 0.0) for meta in batch_metas)
    summary = summarize_agreement(predictions, labels, prediction_trace)
    runtime = {
        "wall_seconds": wall_seconds,
        "examples_per_second": len(rows) / wall_seconds if wall_seconds > 0 else None,
        "batches": len(batch_metas),
        "prompt_eval_tokens": total_prompt_tokens,
        "prompt_eval_ms": total_prompt_ms,
        "prompt_eval_tokens_per_second_aggregate": total_prompt_tokens / (total_prompt_ms / 1000.0)
        if total_prompt_ms > 0
        else None,
    }
    trace_agreement = summary["agreement_with_saved_pytorch_predictions"]
    if summary["examples"] != len(rows) or not np.isfinite(summary["accuracy"]):
        status = "fail"
    elif trace_agreement is not None and trace_agreement < 0.95:
        status = "quality_mismatch"
    else:
        status = "pass"
    result = {
        "schema": "seqcls_i2sr_sidecar_cpu.v1",
        "date": DATE,
        "status": status,
        "task": args.task,
        "max_samples": args.max_samples,
        "batch_size": args.batch_size,
        "checkpoint": {
            "path": maybe_relative(checkpoint_dir, root),
            "stored_accuracy": eval_metrics.get("accuracy"),
            "stored_eval_examples": eval_metrics.get("eval_examples"),
            "prediction_trace": prediction_trace_path,
        },
        "artifacts": {
            "gguf": maybe_relative(gguf, root),
            "classifier_head": maybe_relative(classifier_head, root),
            "embedding_binary": maybe_relative(embedding_binary, root),
            "head_key": weight_key,
            "head_shape": list(weight.shape),
            "bias_key": bias_key if bias is not None else None,
        },
        "summary": summary,
        "runtime": runtime,
        "sample_predictions": [
            {
                "index": idx,
                "label": labels[idx],
                "prediction": predictions[idx],
                "saved_pytorch_prediction": prediction_trace[idx].get("prediction")
                if idx < len(prediction_trace)
                else None,
            }
            for idx in range(min(20, len(predictions)))
        ],
        "limitations": [
            "Classifier head is applied in Python from an NPZ sidecar.",
            "This benchmark uses llama-embedding last-token pooling, not native llama.cpp sequence-classification code.",
            "Prompt text is decoded from the Hugging Face tokenizer's exact pair input IDs to avoid separator guessing.",
            "A low saved-prediction agreement is a blocking runtime-contract failure, not a model-quality result.",
        ],
    }

    output_json = args.output_json if args.output_json.is_absolute() else root / args.output_json
    output_md = args.output_md if args.output_md.is_absolute() else root / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(result) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "summary": summary, "runtime": runtime}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
