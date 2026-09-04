#!/usr/bin/env python3
"""Run an interleaved, repeated native classifier CPU throughput benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.benchmark_seqcls_native_i2sr_cpu import (
    DEFAULT_CHECKPOINT,
    cpu_environment,
    file_identity,
    load_rows,
    render_prompt,
    run_native_classifier,
    runtime_build_contract,
)


DEFAULT_MODELS = {
    "fp16_teacher": Path("models/seqcls-native-baselines/Qwen-Qwen2.5-0.5B/mnli/fp16_sft_qwen2_f16_cls.gguf"),
    "q4_0_teacher": Path("models/seqcls-native-baselines/Qwen-Qwen2.5-0.5B/mnli/fp16_sft_qwen2_q4_0_cls.gguf"),
    "i2_sr_student": Path(
        "models/seqcls-native-i2sr/Qwen-Qwen2.5-0.5B/mnli/"
        "bitdistill-longwarmup-row-layer-8_bitnet_qwen_i2_sr_cls.gguf"
    ),
    "i2_sr_q8_embedding_student": Path(
        "models/seqcls-native-i2sr/Qwen-Qwen2.5-0.5B/mnli/"
        "bitdistill-longwarmup-row-layer-8_bitnet_qwen_i2_sr_q8_embedding_cls.gguf"
    ),
}


def mean_ci95(values: list[float]) -> list[float]:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    if len(values) == 1:
        return [values[0], values[0]]
    t_critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(len(values), 1.96)
    mean = statistics.fmean(values)
    half = t_critical * statistics.stdev(values) / math.sqrt(len(values))
    return [mean - half, mean + half]


def summarize(values: list[float]) -> dict[str, Any]:
    return {
        "values": values,
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "sample_standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
        "mean_ci95_t": mean_ci95(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def summarize_ratios(candidate: list[float], reference: list[float]) -> dict[str, Any]:
    if len(candidate) != len(reference):
        raise ValueError("paired timing ratios require equal run counts")
    ratios = [left / right for left, right in zip(candidate, reference)]
    log_ratios = [math.log(value) for value in ratios]
    log_ci = mean_ci95(log_ratios)
    return {
        "paired_ratios": ratios,
        "geometric_mean": math.exp(statistics.fmean(log_ratios)),
        "geometric_mean_ci95_t": [math.exp(log_ci[0]), math.exp(log_ci[1])],
    }


def render_markdown(report: dict[str, Any]) -> str:
    rows = []
    fp_tps = report["summaries"]["fp16_teacher"]["prompt_tokens_per_second"]["values"]
    for name, summary in report["summaries"].items():
        tps = summary["prompt_tokens_per_second"]
        ratio = summarize_ratios(tps["values"], fp_tps)
        rows.append(
            [
                name,
                f"{tps['mean']:.3f}",
                f"[{tps['mean_ci95_t'][0]:.3f}, {tps['mean_ci95_t'][1]:.3f}]",
                f"{tps['minimum']:.3f}-{tps['maximum']:.3f}",
                f"{ratio['geometric_mean']:.3f}",
                f"[{ratio['geometric_mean_ci95_t'][0]:.3f}, {ratio['geometric_mean_ci95_t'][1]:.3f}]",
                summary["predictions_stable"],
            ]
        )
    table = [
        "| artifact | mean tok/s | mean 95% CI | range | speed / FP16 | ratio 95% CI | predictions stable |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    table.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(
        [
            "# Repeated Native MNLI CPU Throughput",
            "",
            f"Generated: `{report['created_utc']}`. Status: **{report['status']}**.",
            "",
            f"Protocol: `{report['repetitions']}` interleaved repetitions over the first "
            f"`{report['examples']}` MNLI validation examples, `{report['threads']}` threads pinned "
            f"to `{report['cpu_affinity']}`.",
            "",
            *table,
            "",
            "## Interpretation",
            "",
            report["interpretation"],
            "",
            "## Claim Boundary",
            "",
            "- Intervals use a two-sided Student-t interval over four execution repetitions; they quantify run variability, not model-quality uncertainty.",
            "- Ratios are paired by repetition and summarized on the log scale.",
            "- The I2_SR artifacts are trained students; speed comparisons are valid deployed-artifact comparisons, not isolated kernel microbenchmarks.",
            "- Results apply to this CPU, affinity, executable, shared libraries, prompt set, and sequence-isolated classifier path.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--embedding-binary", type=Path, default=Path("build-portable-avx2/bin/llama-embedding"))
    for name, default in DEFAULT_MODELS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=default)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--cpu-affinity", default="0-11")
    parser.add_argument("--ctx-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--ubatch-size", type=int, default=512)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--cooldown-seconds", type=float, default=2.0)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmarks/results/seqcls_native_cpu_repeated_2026-09-04.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("benchmarks/results/seqcls_native_cpu_repeated_2026-09-04.md"),
    )
    args = parser.parse_args()
    if args.max_samples <= 0 or args.repetitions < 2:
        raise SystemExit("--max-samples must be positive and --repetitions must be at least 2")

    root = args.repo_root.resolve()
    checkpoint = args.checkpoint_dir if args.checkpoint_dir.is_absolute() else root / args.checkpoint_dir
    binary = args.embedding_binary if args.embedding_binary.is_absolute() else root / args.embedding_binary
    model_paths = {
        name: value if value.is_absolute() else root / value
        for name, value in ((name, getattr(args, name)) for name in DEFAULT_MODELS)
    }

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    dataset_rows = load_rows("mnli", args.max_samples)
    prompts = [render_prompt(tokenizer, "mnli", row, prompt_input="token_ids") for row in dataset_rows]
    labels = [int(row["label"]) for row in dataset_rows]
    names = list(DEFAULT_MODELS)
    runs: list[dict[str, Any]] = []

    for repetition in range(args.repetitions):
        order = names[repetition % len(names) :] + names[: repetition % len(names)]
        for order_index, name in enumerate(order):
            load_start = list(os.getloadavg())
            logits, meta = run_native_classifier(
                binary=binary,
                gguf=model_paths[name],
                prompts=prompts,
                separator="<#BITNET_REPEATED_CPU#>",
                threads=args.threads,
                ctx_size=args.ctx_size,
                batch_size=args.batch_size,
                ubatch_size=args.ubatch_size,
                timeout_seconds=args.timeout_seconds,
                embedding_sequential=True,
                cpu_affinity=args.cpu_affinity,
            )
            predictions = [int(value) for value in np.argmax(logits, axis=-1)]
            accuracy = sum(int(pred == label) for pred, label in zip(predictions, labels)) / len(labels)
            runs.append(
                {
                    "repetition": repetition,
                    "order_index": order_index,
                    "artifact": name,
                    "accuracy": accuracy,
                    "prediction_sha256": hashlib.sha256(
                        json.dumps(predictions, separators=(",", ":")).encode("utf-8")
                    ).hexdigest(),
                    "wall_seconds": meta["elapsed_seconds"],
                    "prompt_tokens": meta["perf"]["prompt_eval_tokens"],
                    "prompt_eval_ms": meta["perf"]["prompt_eval_ms"],
                    "prompt_tokens_per_second": meta["perf"]["prompt_eval_tokens_per_second"],
                    "load_time_ms": meta["perf"]["load_time_ms"],
                    "host_load_average_start": load_start,
                    "host_load_average_end": list(os.getloadavg()),
                }
            )
            print(
                f"repeat={repetition} order={order_index} artifact={name} "
                f"tps={meta['perf']['prompt_eval_tokens_per_second']:.3f} accuracy={accuracy:.6f}",
                flush=True,
            )
            if args.cooldown_seconds > 0:
                time.sleep(args.cooldown_seconds)

    summaries: dict[str, Any] = {}
    errors: list[str] = []
    for name in names:
        selected = [row for row in runs if row["artifact"] == name]
        prediction_hashes = {row["prediction_sha256"] for row in selected}
        token_counts = {row["prompt_tokens"] for row in selected}
        stable = len(prediction_hashes) == 1
        if not stable:
            errors.append(f"{name}: predictions changed across repetitions")
        if len(token_counts) != 1:
            errors.append(f"{name}: prompt token count changed across repetitions")
        summaries[name] = {
            "predictions_stable": stable,
            "prediction_sha256": selected[0]["prediction_sha256"],
            "accuracy": selected[0]["accuracy"],
            "prompt_tokens": selected[0]["prompt_tokens"],
            "prompt_tokens_per_second": summarize(
                [float(row["prompt_tokens_per_second"]) for row in selected]
            ),
            "wall_seconds": summarize([float(row["wall_seconds"]) for row in selected]),
            "load_time_ms": summarize([float(row["load_time_ms"]) for row in selected]),
        }

    fp_values = summaries["fp16_teacher"]["prompt_tokens_per_second"]["values"]
    paired_speed_ratios = {
        name: summarize_ratios(summary["prompt_tokens_per_second"]["values"], fp_values)
        for name, summary in summaries.items()
    }
    report = {
        "schema": "seqcls-native-cpu-repeated-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "valid" if not errors else "invalid",
        "errors": errors,
        "examples": len(labels),
        "repetitions": args.repetitions,
        "threads": args.threads,
        "cpu_affinity": args.cpu_affinity,
        "hardware": cpu_environment(args.threads),
        "runtime_build": runtime_build_contract(binary, root),
        "artifacts": {name: file_identity(path, root) for name, path in model_paths.items()},
        "runs": runs,
        "summaries": summaries,
        "paired_speed_ratios_vs_fp16": paired_speed_ratios,
        "interpretation": (
            "All prediction and token-count contracts are stable across repetitions. Throughput "
            "comparisons may be reported with paired run-level intervals."
            if not errors
            else "A repeated-run contract failed; no throughput comparison is permitted."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
