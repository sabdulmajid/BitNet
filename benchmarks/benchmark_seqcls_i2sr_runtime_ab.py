#!/usr/bin/env python3
"""Benchmark two I2_SR runtime revisions with identical classifier artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.benchmark_seqcls_native_cpu_repeated import summarize, summarize_ratios
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
    "i2_sr_student": Path(
        "models/seqcls-native-i2sr/Qwen-Qwen2.5-0.5B/mnli/"
        "bitdistill-longwarmup-row-layer-8_bitnet_qwen_i2_sr_cls.gguf"
    ),
    "i2_sr_q8_embedding_student": Path(
        "models/seqcls-native-i2sr/Qwen-Qwen2.5-0.5B/mnli/"
        "bitdistill-longwarmup-row-layer-8_bitnet_qwen_i2_sr_q8_embedding_cls.gguf"
    ),
}
EXPECTED_SOURCE_DIFFERENCES = ["3rdparty/llama.cpp/ggml/src/ggml.c"]


def changed_source_paths(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> list[str]:
    baseline_sources = {row["path"]: row["sha256"] for row in baseline["source_files"]}
    candidate_sources = {row["path"]: row["sha256"] for row in candidate["source_files"]}
    return sorted(
        path
        for path in baseline_sources.keys() | candidate_sources.keys()
        if baseline_sources.get(path) != candidate_sources.get(path)
    )


def render_markdown(report: dict[str, Any]) -> str:
    rows = []
    for artifact, summary in report["summaries"].items():
        baseline = summary["baseline"]["prompt_tokens_per_second"]
        candidate = summary["candidate"]["prompt_tokens_per_second"]
        ratio = summary["candidate_over_baseline"]
        numeric = summary["numeric_equivalence"]
        rows.append(
            "| "
            + " | ".join(
                [
                    artifact,
                    f"{baseline['mean']:.3f}",
                    f"{candidate['mean']:.3f}",
                    f"{ratio['geometric_mean']:.4f}",
                    (
                        f"[{ratio['geometric_mean_ci95_t'][0]:.4f}, "
                        f"{ratio['geometric_mean_ci95_t'][1]:.4f}]"
                    ),
                    f"{numeric['max_abs_logit_difference']:.3e}",
                    str(numeric["predictions_identical"]),
                ]
            )
            + " |"
        )
    lines = [
        "# I2_SR Runtime A/B Benchmark",
        "",
        f"Generated: `{report['created_utc']}`. Status: **{report['status']}**.",
        "",
        f"Protocol: `{report['repetitions']}` rotated repetitions over the first "
        f"`{report['examples']}` MNLI validation examples, `{report['threads']}` threads "
        f"pinned to `{report['cpu_affinity']}`.",
        "",
        "| artifact | baseline tok/s | candidate tok/s | candidate / baseline | paired 95% CI | max abs logit delta | predictions identical |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        *rows,
        "",
        "## Runtime Revisions",
        "",
        f"- Baseline BitNet: `{report['builds']['baseline']['repositories']['bitnet']['revision']}`",
        f"- Baseline llama.cpp: `{report['builds']['baseline']['repositories']['llama_cpp']['revision']}`",
        f"- Candidate BitNet: `{report['builds']['candidate']['repositories']['bitnet']['revision']}`",
        f"- Candidate llama.cpp: `{report['builds']['candidate']['repositories']['llama_cpp']['revision']}`",
        f"- Fingerprinted source differences: `{', '.join(report['source_differences'])}`",
        "",
        "## Interpretation",
        "",
        report["interpretation"],
        "",
        "## Claim Boundary",
        "",
        "- The estimand is a runtime-implementation effect: model bytes, prompts, thread count, and affinity are identical.",
        "- Ratios are paired by repetition and summarized on the log scale with a Student-t interval.",
        "- Four repetitions characterize local run variability; they do not establish portability to other CPUs or workloads.",
        "- This benchmark does not change or re-evaluate model quality relative to FP16.",
        "",
    ]
    if report["errors"]:
        lines.extend(["## Contract Errors", "", *[f"- {value}" for value in report["errors"]], ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, default=Path.cwd())
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--binary", type=Path, default=Path("build-portable-avx2/bin/llama-embedding"))
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
    parser.add_argument("--cooldown-seconds", type=float, default=3.0)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmark_results/seqcls_i2sr_runtime_ab_2026-09-04.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("benchmarks/results/seqcls_i2sr_runtime_ab_2026-09-04.md"),
    )
    args = parser.parse_args()
    if args.max_samples <= 0 or args.repetitions < 2:
        raise SystemExit("--max-samples must be positive and --repetitions must be at least 2")

    candidate_root = args.candidate_root.resolve()
    baseline_root = args.baseline_root.resolve()
    roots = {"baseline": baseline_root, "candidate": candidate_root}
    binaries = {
        name: value / args.binary if not args.binary.is_absolute() else args.binary
        for name, value in roots.items()
    }
    checkpoint = (
        args.checkpoint_dir
        if args.checkpoint_dir.is_absolute()
        else candidate_root / args.checkpoint_dir
    )
    models = {
        name: value if value.is_absolute() else candidate_root / value
        for name, value in ((key, getattr(args, key)) for key in DEFAULT_MODELS)
    }

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    dataset_rows = load_rows("mnli", args.max_samples)
    prompts = [render_prompt(tokenizer, "mnli", row, prompt_input="token_ids") for row in dataset_rows]
    labels = [int(row["label"]) for row in dataset_rows]
    configurations = [
        (artifact, implementation)
        for artifact in DEFAULT_MODELS
        for implementation in ("baseline", "candidate")
    ]
    runs: list[dict[str, Any]] = []
    logits_by_run: dict[tuple[int, str, str], np.ndarray] = {}

    for repetition in range(args.repetitions):
        order = configurations[repetition % len(configurations) :] + configurations[
            : repetition % len(configurations)
        ]
        for order_index, (artifact, implementation) in enumerate(order):
            load_start = list(os.getloadavg())
            logits, meta = run_native_classifier(
                binary=binaries[implementation],
                gguf=models[artifact],
                prompts=prompts,
                separator="<#BITNET_I2SR_RUNTIME_AB#>",
                threads=args.threads,
                ctx_size=args.ctx_size,
                batch_size=args.batch_size,
                ubatch_size=args.ubatch_size,
                timeout_seconds=args.timeout_seconds,
                embedding_sequential=True,
                cpu_affinity=args.cpu_affinity,
            )
            logits_array = np.asarray(logits, dtype=np.float64)
            predictions = np.argmax(logits_array, axis=-1).astype(np.int64)
            logits_by_run[(repetition, artifact, implementation)] = logits_array
            runs.append(
                {
                    "repetition": repetition,
                    "order_index": order_index,
                    "artifact": artifact,
                    "implementation": implementation,
                    "accuracy": float(np.mean(predictions == labels)),
                    "prediction_sha256": hashlib.sha256(predictions.tobytes()).hexdigest(),
                    "logits_sha256": hashlib.sha256(logits_array.tobytes()).hexdigest(),
                    "prompt_tokens": meta["perf"]["prompt_eval_tokens"],
                    "prompt_tokens_per_second": meta["perf"]["prompt_eval_tokens_per_second"],
                    "wall_seconds": meta["elapsed_seconds"],
                    "host_load_average_start": load_start,
                    "host_load_average_end": list(os.getloadavg()),
                }
            )
            print(
                f"repeat={repetition} order={order_index} artifact={artifact} "
                f"implementation={implementation} "
                f"tps={meta['perf']['prompt_eval_tokens_per_second']:.3f}",
                flush=True,
            )
            if args.cooldown_seconds > 0:
                time.sleep(args.cooldown_seconds)

    errors: list[str] = []
    builds = {
        name: runtime_build_contract(binary, roots[name]) for name, binary in binaries.items()
    }
    if builds["baseline"]["cmake_options"] != builds["candidate"]["cmake_options"]:
        errors.append("baseline and candidate CMake options differ")
    source_differences = changed_source_paths(builds["baseline"], builds["candidate"])
    if source_differences != EXPECTED_SOURCE_DIFFERENCES:
        errors.append(
            "fingerprinted source differences do not match the registered runtime change: "
            f"{source_differences!r}"
        )
    for implementation, build in builds.items():
        for repository, identity in build["repositories"].items():
            if identity["tracked_files_dirty"] is not False:
                errors.append(f"{implementation} {repository} source was dirty")

    summaries: dict[str, Any] = {}
    for artifact in DEFAULT_MODELS:
        implementation_summaries: dict[str, Any] = {}
        for implementation in ("baseline", "candidate"):
            selected = sorted(
                (
                    row
                    for row in runs
                    if row["artifact"] == artifact and row["implementation"] == implementation
                ),
                key=lambda row: row["repetition"],
            )
            prediction_hashes = {row["prediction_sha256"] for row in selected}
            token_counts = {row["prompt_tokens"] for row in selected}
            if len(prediction_hashes) != 1:
                errors.append(f"{artifact} {implementation}: predictions changed across repetitions")
            if len(token_counts) != 1:
                errors.append(f"{artifact} {implementation}: token count changed across repetitions")
            implementation_summaries[implementation] = {
                "accuracy": selected[0]["accuracy"],
                "prediction_sha256": selected[0]["prediction_sha256"],
                "prompt_tokens": selected[0]["prompt_tokens"],
                "prompt_tokens_per_second": summarize(
                    [float(row["prompt_tokens_per_second"]) for row in selected]
                ),
            }

        max_abs = 0.0
        rms_values = []
        for repetition in range(args.repetitions):
            delta = (
                logits_by_run[(repetition, artifact, "candidate")]
                - logits_by_run[(repetition, artifact, "baseline")]
            )
            max_abs = max(max_abs, float(np.max(np.abs(delta))))
            rms_values.append(float(np.sqrt(np.mean(np.square(delta)))))
        predictions_identical = (
            implementation_summaries["candidate"]["prediction_sha256"]
            == implementation_summaries["baseline"]["prediction_sha256"]
        )
        if not predictions_identical:
            errors.append(f"{artifact}: candidate predictions differ from baseline")
        baseline_tps = implementation_summaries["baseline"]["prompt_tokens_per_second"]["values"]
        candidate_tps = implementation_summaries["candidate"]["prompt_tokens_per_second"]["values"]
        summaries[artifact] = {
            **implementation_summaries,
            "candidate_over_baseline": summarize_ratios(candidate_tps, baseline_tps),
            "numeric_equivalence": {
                "predictions_identical": predictions_identical,
                "max_abs_logit_difference": max_abs,
                "maximum_rms_logit_difference": max(rms_values),
            },
        }

    positive_speedups = [
        summary["candidate_over_baseline"]["geometric_mean_ci95_t"][0] > 1.0
        for summary in summaries.values()
    ]
    if errors:
        interpretation = "A build, prediction, or timing contract failed; no speed claim is permitted."
    elif all(positive_speedups):
        interpretation = (
            "The candidate preserves predictions and produces a statistically positive local throughput "
            "ratio for both I2_SR artifacts under this protocol."
        )
    else:
        interpretation = (
            "The candidate preserves predictions, but at least one paired interval does not establish a "
            "positive throughput effect for both I2_SR artifacts."
        )

    report = {
        "schema": "seqcls-i2sr-runtime-ab-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "valid" if not errors else "invalid",
        "errors": errors,
        "examples": len(labels),
        "repetitions": args.repetitions,
        "threads": args.threads,
        "cpu_affinity": args.cpu_affinity,
        "hardware": cpu_environment(args.threads),
        "benchmark_script": file_identity(Path(__file__).resolve(), candidate_root),
        "builds": builds,
        "source_differences": source_differences,
        "artifacts": {name: file_identity(path, candidate_root) for name, path in models.items()},
        "runs": runs,
        "summaries": summaries,
        "interpretation": interpretation,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
