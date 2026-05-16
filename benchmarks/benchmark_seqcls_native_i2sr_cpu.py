#!/usr/bin/env python3
"""Evaluate a native packed I2_SR sequence-classification GGUF."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import resource
import signal
import subprocess
import tempfile
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
    "models/seqcls-native-i2sr/"
    "Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8_bitnet_qwen_i2_sr_cls.gguf"
)
PROMPT_EVAL_RE = re.compile(
    r"prompt eval time =\s+(?P<ms>[0-9.]+) ms /\s+(?P<tokens>[0-9]+) tokens.*?"
    r"(?P<tps>[0-9.]+) tokens per second"
)
LOAD_RE = re.compile(r"load time =\s+(?P<ms>[0-9.]+) ms")
TOTAL_RE = re.compile(r"total time =\s+(?P<ms>[0-9.]+) ms")


def run_clean(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate()
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)


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


def run_native_classifier(
    *,
    binary: Path,
    gguf: Path,
    prompts: list[str],
    separator: str,
    threads: int,
    ctx_size: int,
    batch_size: int,
    ubatch_size: int,
    timeout_seconds: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if any(separator in prompt for prompt in prompts):
        raise ValueError("separator occurs inside a prompt")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
        prompt_file = Path(handle.name)
        handle.write(separator.join(prompts))
    command = [
        str(binary),
        "-m",
        str(gguf),
        "-f",
        str(prompt_file),
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
        "-b",
        str(batch_size),
        "-ub",
        str(ubatch_size),
    ]
    before_rss_kib = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    started = time.perf_counter()
    try:
        result = run_clean(command, timeout=timeout_seconds)
    finally:
        try:
            prompt_file.unlink()
        except FileNotFoundError:
            pass
    elapsed = time.perf_counter() - started
    after_rss_kib = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-4000:])
    parsed = json.loads(result.stdout)
    logits = [row["embedding"] for row in parsed.get("data", [])]
    if len(logits) != len(prompts):
        raise RuntimeError(f"expected {len(prompts)} logits rows, got {len(logits)}")
    meta = {
        "command": command,
        "elapsed_seconds": elapsed,
        "stdout_bytes": len(result.stdout.encode("utf-8")),
        "stderr_bytes": len(result.stderr.encode("utf-8")),
        "perf": parse_perf(result.stderr),
        "child_peak_rss_kib_before": before_rss_kib,
        "child_peak_rss_kib_after": after_rss_kib,
        "stderr_tail": result.stderr[-4000:],
    }
    return np.asarray(logits, dtype=np.float32), meta


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
    trace_scores = [
        [float(value) for value in row["scores"]]
        for row in trace[:total]
        if isinstance(row, dict) and isinstance(row.get("scores"), list)
    ]
    trace_agreement = None
    trace_label_match = None
    if len(trace_predictions) == total:
        trace_agreement = sum(int(a == b) for a, b in zip(predictions, trace_predictions, strict=True)) / total
    if len(trace_labels) == total:
        trace_label_match = sum(int(a == b) for a, b in zip(labels, trace_labels, strict=True)) / total
    trace_mismatches = [
        idx
        for idx, (pred, trace_pred) in enumerate(zip(predictions, trace_predictions, strict=False))
        if pred != trace_pred
    ]
    return {
        "examples": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "accuracy_ci95_wilson": accuracy_ci_wilson(correct, total),
        "saved_trace_predictions": len(trace_predictions),
        "saved_trace_scores": len(trace_scores),
        "agreement_with_saved_pytorch_predictions": trace_agreement,
        "disagreements_with_saved_pytorch_predictions": len(trace_mismatches)
        if len(trace_predictions) == total
        else None,
        "first_saved_prediction_mismatch_indices": trace_mismatches[:20]
        if len(trace_predictions) == total
        else [],
        "label_agreement_with_saved_trace": trace_label_match,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, list):
        return "[" + ", ".join(fmt(item) for item in value) + "]"
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
            f"# Sequence-Classification Native I2_SR CPU Benchmark, {result['date']}",
            (
                "This benchmark evaluates one native GGUF artifact that contains the packed I2_SR "
                "backbone and dense classifier head. It is the same-artifact runtime path, but it "
                "is not product-ready unless full validation, batching parity, RSS, and throughput "
                "gates pass."
            ),
            md_table(
                ["field", "value"],
                [
                    ["status", result["status"]],
                    ["task", result["task"]],
                    ["examples", summary["examples"]],
                    ["expected examples", result["expected_examples"]],
                    ["full validation", result["full_validation_complete"]],
                    ["accuracy", summary["accuracy"]],
                    ["accuracy CI95", summary["accuracy_ci95_wilson"]],
                    ["stored PyTorch accuracy", checkpoint.get("stored_accuracy")],
                    ["agreement with saved PyTorch predictions", summary["agreement_with_saved_pytorch_predictions"]],
                    ["label agreement with saved trace", summary["label_agreement_with_saved_trace"]],
                    ["prompt batch size", result["prompt_batch_size"]],
                    ["llama batch size", result["batch_size"]],
                    ["ubatch size", result["ubatch_size"]],
                    ["wall seconds", runtime["wall_seconds"]],
                    ["examples/sec", runtime["examples_per_second"]],
                    ["tokens/sec", runtime["prompt_eval_tokens_per_second"]],
                    ["child peak RSS MiB", runtime["child_peak_rss_mib"]],
                    ["ready to productize", result["ready_to_productize"]],
                ],
            ),
            "## Interpretation",
            result["verdict"],
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--task", choices=sorted(TASK_SPECS), default="mnli")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--embedding-binary", type=Path, default=Path("build-portable-avx2/bin/llama-embedding"))
    parser.add_argument("--max-samples", type=int, default=512, help="0 means full validation split")
    parser.add_argument(
        "--prompt-batch-size",
        type=int,
        default=1,
        help=(
            "Number of prompts to pass to one llama-embedding process. Keep at 1 until "
            "multi-prompt classifier batching is proven stable."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--ubatch-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--ctx-size", type=int, default=512)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--separator", default="<#BITNET_NATIVE_EVAL_SEP#>")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(f"benchmark_results/seqcls_native_i2sr_cpu_mnli_512_{DATE}.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(f"benchmarks/results/seqcls_native_i2sr_cpu_mnli_512_{DATE}.md"),
    )
    args = parser.parse_args()
    if args.prompt_batch_size <= 0:
        raise SystemExit("--prompt-batch-size must be positive")

    root = args.repo_root.resolve()
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir.is_absolute() else root / args.checkpoint_dir
    gguf = args.gguf if args.gguf.is_absolute() else root / args.gguf
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

    all_logits: list[np.ndarray] = []
    metas: list[dict[str, Any]] = []
    started = time.perf_counter()
    for start in range(0, len(prompts), args.prompt_batch_size):
        batch_prompts = prompts[start : start + args.prompt_batch_size]
        batch_logits, meta = run_native_classifier(
            binary=embedding_binary,
            gguf=gguf,
            prompts=batch_prompts,
            separator=args.separator,
            threads=args.threads,
            ctx_size=args.ctx_size,
            batch_size=args.batch_size,
            ubatch_size=args.ubatch_size,
            timeout_seconds=args.timeout_seconds,
        )
        all_logits.append(batch_logits)
        metas.append(meta)
    wall_seconds = time.perf_counter() - started
    logits = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0, 3), dtype=np.float32)
    predictions = [int(x) for x in np.argmax(logits, axis=-1)]
    summary = summarize_agreement(predictions, labels, prediction_trace)
    expected_examples = int(TASK_SPECS[args.task]["expected_examples"])
    full_validation_complete = args.max_samples == 0 and summary["examples"] == expected_examples
    prompt_eval_tokens = sum(int(meta["perf"].get("prompt_eval_tokens") or 0) for meta in metas)
    prompt_eval_ms = sum(float(meta["perf"].get("prompt_eval_ms") or 0.0) for meta in metas)
    load_time_ms = sum(float(meta["perf"].get("load_time_ms") or 0.0) for meta in metas)
    total_time_ms = sum(float(meta["perf"].get("total_time_ms") or 0.0) for meta in metas)
    child_peak_rss_kib = max((meta.get("child_peak_rss_kib_after") or 0) for meta in metas) if metas else None
    runtime = {
        "wall_seconds": wall_seconds,
        "examples_per_second": len(rows) / wall_seconds if wall_seconds > 0 else None,
        "subprocesses": len(metas),
        "load_time_ms": load_time_ms,
        "prompt_eval_ms": prompt_eval_ms,
        "prompt_eval_tokens": prompt_eval_tokens,
        "prompt_eval_tokens_per_second": prompt_eval_tokens / (prompt_eval_ms / 1000.0)
        if prompt_eval_ms > 0
        else None,
        "total_time_ms": total_time_ms,
        "child_peak_rss_kib": child_peak_rss_kib,
        "child_peak_rss_mib": child_peak_rss_kib / 1024.0
        if isinstance(child_peak_rss_kib, (int, float))
        else None,
    }
    trace_agreement = summary["agreement_with_saved_pytorch_predictions"]
    status = "pass"
    if summary["examples"] != len(rows) or not isinstance(summary["accuracy"], float) or not math.isfinite(summary["accuracy"]):
        status = "fail"
    elif trace_agreement is not None and trace_agreement < 0.95:
        status = "quality_mismatch"
    elif not full_validation_complete:
        status = "sample_only"
    ready_to_productize = (
        status == "pass"
        and full_validation_complete
        and trace_agreement is not None
        and trace_agreement >= 0.99
        and runtime["child_peak_rss_mib"] is not None
        and runtime["examples_per_second"] is not None
        and runtime["examples_per_second"] > 0
    )
    prediction_json = json.dumps(predictions, separators=(",", ":"))
    result = {
        "schema": "seqcls_native_i2sr_cpu.v1",
        "date": DATE,
        "status": status,
        "task": args.task,
        "max_samples": args.max_samples,
        "expected_examples": expected_examples,
        "full_validation_complete": full_validation_complete,
        "ready_to_productize": ready_to_productize,
        "prompt_batch_size": args.prompt_batch_size,
        "batch_size": args.batch_size,
        "ubatch_size": args.ubatch_size,
        "checkpoint": {
            "path": maybe_relative(checkpoint_dir, root),
            "stored_accuracy": eval_metrics.get("accuracy"),
            "stored_eval_examples": eval_metrics.get("eval_examples"),
            "prediction_trace": prediction_trace_path,
        },
        "artifacts": {
            "gguf": maybe_relative(gguf, root),
            "embedding_binary": maybe_relative(embedding_binary, root),
        },
        "summary": summary,
        "runtime": runtime,
        "predictions": predictions,
        "prediction_sha256": hashlib.sha256(prediction_json.encode("utf-8")).hexdigest(),
        "sample_predictions": [
            {
                "index": idx,
                "label": labels[idx],
                "prediction": predictions[idx],
                "saved_pytorch_prediction": prediction_trace[idx].get("prediction")
                if idx < len(prediction_trace)
                else None,
                "logits": [float(value) for value in logits[idx].tolist()],
            }
            for idx in range(min(20, len(predictions)))
        ],
        "limitations": [
            "This uses llama-embedding JSON output as the classifier-logit transport.",
            "The smoke is product-ready only after full validation and batching parity pass.",
            "The child RSS value is a process-level peak from resource.getrusage on Linux.",
        ],
        "verdict": (
            "Native same-artifact classifier validation passed the configured product gate."
            if ready_to_productize
            else "Native same-artifact classifier execution is measurable, but the product gate remains blocked "
            "until full validation, batching parity, RSS, and throughput are audited."
        ),
        "stderr_tail": metas[-1]["stderr_tail"] if metas else "",
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
