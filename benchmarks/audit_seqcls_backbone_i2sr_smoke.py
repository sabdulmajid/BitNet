#!/usr/bin/env python3
"""Smoke-test a sequence-classification checkpoint exported as I2_SR backbone + dense head sidecar."""

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
DEFAULT_CHECKPOINT = Path(
    "checkpoints/bitdistill-glue-seqcls-longwarmup/"
    "Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8"
)
DEFAULT_GGUF = Path(
    "models/seqcls-backbone-i2sr/"
    "Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8_bitnet25_i2_sr.gguf"
)
DEFAULT_HEAD = Path(
    "models/seqcls-backbone-i2sr/"
    "Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8_score_head.npz"
)
DEFAULT_CONVERSION_SUMMARY = Path("benchmark_results/seqcls_longwarmup_backbone_i2sr_smoke_2026-05-15.json")
DEFAULT_PROMPT = "Premise: A person is riding a bicycle. Hypothesis: Someone is outdoors."


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


def maybe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def file_info(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": maybe_relative(path, root),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "size_mib": path.stat().st_size / (1024 * 1024) if path.exists() else None,
    }


def parse_perf(stderr: str) -> dict[str, Any]:
    prompt_match = PROMPT_EVAL_RE.search(stderr)
    load_match = LOAD_RE.search(stderr)
    total_match = TOTAL_RE.search(stderr)
    system_info = next((line for line in stderr.splitlines() if line.startswith("system_info:")), "")
    model_ftype = next((line for line in stderr.splitlines() if "model ftype" in line), "")
    return {
        "system_info": system_info,
        "model_ftype_line": model_ftype,
        "load_time_ms": float(load_match.group("ms")) if load_match else None,
        "prompt_eval_ms": float(prompt_match.group("ms")) if prompt_match else None,
        "prompt_eval_tokens": int(prompt_match.group("tokens")) if prompt_match else None,
        "prompt_eval_tokens_per_second": float(prompt_match.group("tps")) if prompt_match else None,
        "total_time_ms": float(total_match.group("ms")) if total_match else None,
    }


def run_embedding(
    *,
    binary: Path,
    gguf: Path,
    prompt: str,
    threads: int,
    ctx_size: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    command = [
        str(binary),
        "-m",
        str(gguf),
        "-p",
        prompt,
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
    parsed: dict[str, Any] = {}
    embedding: list[float] = []
    parse_error = ""
    if result.stdout.strip():
        try:
            parsed = json.loads(result.stdout)
            data = parsed.get("data", [])
            if data:
                embedding = data[0].get("embedding", [])
        except json.JSONDecodeError as exc:
            parse_error = str(exc)
    return {
        "command": command,
        "returncode": result.returncode,
        "elapsed_seconds": elapsed,
        "stdout_bytes": len(result.stdout.encode("utf-8")),
        "stderr_bytes": len(result.stderr.encode("utf-8")),
        "parse_error": parse_error,
        "embedding": embedding,
        "perf": parse_perf(result.stderr),
        "stderr_tail": "\n".join(result.stderr.splitlines()[-12:]),
    }


def classifier_logits(head_path: Path, embedding: list[float]) -> dict[str, Any]:
    head = np.load(head_path)
    weight_key = "score_weight" if "score_weight" in head.files else "classifier_weight"
    bias_key = "score_bias" if "score_bias" in head.files else "classifier_bias"
    weight = np.asarray(head[weight_key], dtype=np.float32)
    vector = np.asarray(embedding, dtype=np.float32)
    logits = weight @ vector
    has_bias = bias_key in head.files
    if has_bias:
        logits = logits + np.asarray(head[bias_key], dtype=np.float32)
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()
    return {
        "weight_key": weight_key,
        "bias_key": bias_key if has_bias else None,
        "weight_shape": list(weight.shape),
        "embedding_shape": list(vector.shape),
        "logits": [float(x) for x in logits],
        "probabilities": [float(x) for x in probs],
        "prediction": int(np.argmax(logits)),
        "finite_embedding": bool(np.isfinite(vector).all()),
        "finite_logits": bool(np.isfinite(logits).all()),
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
    checkpoint = result["checkpoint"]
    conversion = result["conversion"]
    runtime = result["runtime_smoke"]
    logits = result["classifier_head_application"]
    files = result["files"]
    return "\n\n".join(
        [
            f"# Sequence-Classification I2_SR Backbone Smoke, {result['date']}",
            (
                "This is a prototype bridge for the current product gap: a strict GLUE "
                "sequence-classification checkpoint is exported as a packed I2_SR decoder "
                "backbone, while the dense classifier head is kept as an NPZ sidecar."
            ),
            md_table(
                ["field", "value"],
                [
                    ["status", result["status"]],
                    ["checkpoint accuracy", checkpoint.get("accuracy")],
                    ["checkpoint eval examples", checkpoint.get("eval_examples")],
                    ["GGUF MiB", files["gguf"].get("size_mib")],
                    ["head sidecar KiB", files["classifier_head"].get("size_bytes", 0) / 1024],
                    ["I2_SR tensors", conversion.get("row_scale_i2s_packed")],
                    ["copied dense tensors", conversion.get("copied_tensors")],
                    ["runtime return code", runtime.get("returncode")],
                    ["embedding dim", logits.get("embedding_shape", ["-"])[0]],
                    ["head shape", logits.get("weight_shape")],
                    ["finite embedding", logits.get("finite_embedding")],
                    ["finite logits", logits.get("finite_logits")],
                    ["prompt tok/s", runtime.get("perf", {}).get("prompt_eval_tokens_per_second")],
                    ["predicted class for smoke prompt", logits.get("prediction")],
                ],
            ),
            "## Smoke Prompt",
            f"`{result['prompt']}`",
            "## Logits",
            md_table(
                ["class", "logit", "probability"],
                [
                    [idx, logit, prob]
                    for idx, (logit, prob) in enumerate(
                        zip(logits.get("logits", []), logits.get("probabilities", []), strict=False)
                    )
                ],
            ),
            "## Interpretation",
            (
                "This smoke proves loadability, last-token embedding extraction, sidecar-head "
                "shape compatibility, and finite CPU logits for the same sequence-classification "
                "checkpoint. It is not a full GLUE CPU accuracy result and it is not native "
                "llama.cpp sequence-classification support yet."
            ),
            "## Remaining Runtime Work",
            md_table(
                ["item", "status"],
                [
                    ["Persist classifier head and label metadata inside GGUF", "not implemented"],
                    ["Apply Qwen sequence-classification pooling/head inside llama.cpp", "not implemented"],
                    ["Run full MNLI/QNLI/SST2 accuracy from packed CPU artifact", "not implemented"],
                    ["Benchmark RSS and throughput for full task evaluation", "not implemented"],
                ],
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--classifier-head", type=Path, default=DEFAULT_HEAD)
    parser.add_argument("--conversion-summary", type=Path, default=DEFAULT_CONVERSION_SUMMARY)
    parser.add_argument("--embedding-binary", type=Path, default=Path("build-portable-avx2/bin/llama-embedding"))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--ctx-size", type=int, default=128)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(f"benchmark_results/seqcls_backbone_i2sr_smoke_{DATE}.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(f"benchmarks/results/seqcls_backbone_i2sr_smoke_{DATE}.md"),
    )
    args = parser.parse_args()

    root = args.repo_root.resolve()
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir.is_absolute() else root / args.checkpoint_dir
    gguf = args.gguf if args.gguf.is_absolute() else root / args.gguf
    classifier_head = args.classifier_head if args.classifier_head.is_absolute() else root / args.classifier_head
    conversion_summary = (
        args.conversion_summary if args.conversion_summary.is_absolute() else root / args.conversion_summary
    )
    embedding_binary = args.embedding_binary if args.embedding_binary.is_absolute() else root / args.embedding_binary

    config = read_json(checkpoint_dir / "config.json")
    metrics = read_json(checkpoint_dir / "metrics.json")
    conversion = read_json(conversion_summary)
    runtime = run_embedding(
        binary=embedding_binary,
        gguf=gguf,
        prompt=args.prompt,
        threads=args.threads,
        ctx_size=args.ctx_size,
        timeout_seconds=args.timeout_seconds,
    )
    logits = classifier_logits(classifier_head, runtime["embedding"]) if runtime["embedding"] else {}

    files = {
        "gguf": file_info(gguf, root),
        "classifier_head": file_info(classifier_head, root),
        "conversion_summary": file_info(conversion_summary, root),
    }
    finite_path = bool(logits.get("finite_embedding")) and bool(logits.get("finite_logits"))
    status = "prototype_smoke_passed" if runtime["returncode"] == 0 and finite_path else "prototype_smoke_failed"

    result = {
        "schema": "seqcls_backbone_i2sr_smoke.v1",
        "date": DATE,
        "status": status,
        "prompt": args.prompt,
        "checkpoint": {
            "path": maybe_relative(checkpoint_dir, root),
            "architecture": (config.get("architectures") or [""])[0],
            "task": metrics.get("task"),
            "method": metrics.get("method"),
            "scale_mode": metrics.get("scale_mode"),
            "accuracy": metrics.get("eval", {}).get("accuracy") if isinstance(metrics.get("eval"), dict) else None,
            "eval_examples": metrics.get("eval", {}).get("eval_examples") if isinstance(metrics.get("eval"), dict) else None,
            "prediction_trace": metrics.get("eval", {}).get("prediction_path") if isinstance(metrics.get("eval"), dict) else None,
        },
        "files": files,
        "conversion": conversion,
        "runtime_smoke": runtime,
        "classifier_head_application": logits,
        "limitations": [
            "Classifier head is applied outside llama.cpp from an NPZ sidecar.",
            "GGUF does not yet contain sequence-classification label metadata or native score-head execution.",
            "Smoke validates a single prompt path only; it is not a full GLUE CPU accuracy benchmark.",
            "Default binary is the portable AVX2 build, not an AVX-512-tuned benchmark binary.",
        ],
    }

    output_json = args.output_json if args.output_json.is_absolute() else root / args.output_json
    output_md = args.output_md if args.output_md.is_absolute() else root / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(result) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "output_json": str(output_json), "output_md": str(output_md)}, indent=2))


if __name__ == "__main__":
    main()
