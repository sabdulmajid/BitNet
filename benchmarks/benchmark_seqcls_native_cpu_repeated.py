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


def parse_cpu_list(spec: str) -> list[int]:
    cpus: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", maxsplit=1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"invalid descending CPU range: {part}")
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(part))
    if not cpus:
        raise ValueError("CPU list is empty")
    return sorted(cpus)


def monitored_cpu_set(
    affinity: str,
    topology_root: Path = Path("/sys/devices/system/cpu"),
) -> list[int]:
    cpus = set(parse_cpu_list(affinity))
    for cpu in tuple(cpus):
        sibling_path = topology_root / f"cpu{cpu}" / "topology" / "thread_siblings_list"
        if sibling_path.is_file():
            cpus.update(parse_cpu_list(sibling_path.read_text(encoding="utf-8").strip()))
    return sorted(cpus)


def read_cpu_snapshot(cpus: list[int], proc_stat: Path = Path("/proc/stat")) -> dict[int, tuple[int, int]]:
    selected = set(cpus)
    snapshot: dict[int, tuple[int, int]] = {}
    for line in proc_stat.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields or not fields[0].startswith("cpu") or fields[0] == "cpu":
            continue
        cpu_text = fields[0][3:]
        if not cpu_text.isdigit() or int(cpu_text) not in selected:
            continue
        values = [int(value) for value in fields[1:]]
        if len(values) < 5:
            raise ValueError(f"incomplete CPU counters for {fields[0]}")
        idle = values[3] + values[4]
        snapshot[int(cpu_text)] = (sum(values), idle)
    missing = selected.difference(snapshot)
    if missing:
        raise ValueError(f"missing /proc/stat counters for CPUs {sorted(missing)}")
    return snapshot


def cpu_utilization(
    before: dict[int, tuple[int, int]],
    after: dict[int, tuple[int, int]],
) -> dict[int, float]:
    if before.keys() != after.keys():
        raise ValueError("CPU snapshots cover different logical CPUs")
    utilization: dict[int, float] = {}
    for cpu in before:
        total_delta = after[cpu][0] - before[cpu][0]
        idle_delta = after[cpu][1] - before[cpu][1]
        if total_delta <= 0 or idle_delta < 0:
            raise ValueError(f"non-monotonic CPU counters for cpu{cpu}")
        utilization[cpu] = max(0.0, min(1.0, 1.0 - idle_delta / total_delta))
    return utilization


def wait_for_idle_cpus(
    affinity: str,
    *,
    max_utilization: float,
    sample_seconds: float,
    consecutive_samples: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    cpus = monitored_cpu_set(affinity)
    started = time.monotonic()
    accepted: list[dict[str, Any]] = []
    attempts = 0
    while time.monotonic() - started < timeout_seconds:
        before = read_cpu_snapshot(cpus)
        time.sleep(sample_seconds)
        utilization = cpu_utilization(before, read_cpu_snapshot(cpus))
        attempts += 1
        sample = {
            "maximum": max(utilization.values()),
            "mean": statistics.fmean(utilization.values()),
            "per_cpu": {str(cpu): value for cpu, value in utilization.items()},
        }
        if sample["maximum"] <= max_utilization:
            accepted.append(sample)
            if len(accepted) == consecutive_samples:
                return {
                    "logical_cpus": cpus,
                    "maximum_allowed_utilization": max_utilization,
                    "sample_seconds": sample_seconds,
                    "consecutive_samples": consecutive_samples,
                    "attempts": attempts,
                    "wait_seconds": time.monotonic() - started,
                    "accepted_samples": accepted,
                }
        else:
            accepted = []
    raise RuntimeError(
        f"CPUs {cpus} did not remain below {max_utilization:.1%} utilization for "
        f"{consecutive_samples} consecutive {sample_seconds:g}s samples within {timeout_seconds:g}s"
    )


def mean_ci95(values: list[float]) -> list[float]:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    if len(values) == 1:
        return [values[0], values[0]]
    # Two-sided 95% Student-t critical values indexed by degrees of freedom.
    t_critical_by_df = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
    }
    t_critical = t_critical_by_df.get(len(values) - 1, 1.96)
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
    reference_name = report["speed_reference"]
    reference_tps = report["summaries"][reference_name]["prompt_tokens_per_second"]["values"]
    for name, summary in report["summaries"].items():
        tps = summary["prompt_tokens_per_second"]
        ratio = summarize_ratios(tps["values"], reference_tps)
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
        f"| artifact | mean tok/s | mean 95% CI | range | speed / {reference_name} | ratio 95% CI | predictions stable |",
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
            f"- Intervals use a two-sided Student-t interval over {report['repetitions']} execution repetitions; they quantify run variability, not model-quality uncertainty.",
            "- Ratios are paired by repetition and summarized on the log scale.",
            "- Trained-student speed comparisons are valid deployed-artifact comparisons, not isolated kernel microbenchmarks.",
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
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Explicit artifact set; when present, replaces the built-in model set.",
    )
    parser.add_argument(
        "--speed-reference",
        default="fp16_teacher",
        help="Artifact name used as the denominator for paired speed ratios.",
    )
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--cpu-affinity", default="0-11")
    parser.add_argument("--ctx-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--ubatch-size", type=int, default=512)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--cooldown-seconds", type=float, default=2.0)
    parser.add_argument("--idle-max-utilization", type=float, default=0.20)
    parser.add_argument("--idle-sample-seconds", type=float, default=1.0)
    parser.add_argument("--idle-consecutive-samples", type=int, default=2)
    parser.add_argument("--idle-timeout-seconds", type=float, default=900.0)
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
    if not 0.0 <= args.idle_max_utilization <= 1.0:
        raise SystemExit("--idle-max-utilization must be between 0 and 1")
    if args.idle_sample_seconds <= 0 or args.idle_consecutive_samples <= 0 or args.idle_timeout_seconds <= 0:
        raise SystemExit("idle-check durations and sample count must be positive")

    root = args.repo_root.resolve()
    checkpoint = args.checkpoint_dir if args.checkpoint_dir.is_absolute() else root / args.checkpoint_dir
    binary = args.embedding_binary if args.embedding_binary.is_absolute() else root / args.embedding_binary
    if args.model:
        model_paths = {}
        for item in args.model:
            name, separator, raw_path = item.partition("=")
            if not separator or not name or not raw_path:
                raise SystemExit(f"invalid --model {item!r}; expected NAME=PATH")
            if name in model_paths:
                raise SystemExit(f"duplicate --model name: {name}")
            path = Path(raw_path)
            model_paths[name] = path if path.is_absolute() else root / path
    else:
        model_paths = {
            name: value if value.is_absolute() else root / value
            for name, value in ((name, getattr(args, name)) for name in DEFAULT_MODELS)
        }
    if args.speed_reference not in model_paths:
        raise SystemExit(
            f"--speed-reference {args.speed_reference!r} is not one of {sorted(model_paths)}"
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    dataset_rows = load_rows("mnli", args.max_samples)
    prompts = [render_prompt(tokenizer, "mnli", row, prompt_input="token_ids") for row in dataset_rows]
    labels = [int(row["label"]) for row in dataset_rows]
    names = list(model_paths)
    runs: list[dict[str, Any]] = []
    idle_preflights: list[dict[str, Any]] = []

    for repetition in range(args.repetitions):
        order = names[repetition % len(names) :] + names[: repetition % len(names)]
        for order_index, name in enumerate(order):
            idle_preflight = wait_for_idle_cpus(
                args.cpu_affinity,
                max_utilization=args.idle_max_utilization,
                sample_seconds=args.idle_sample_seconds,
                consecutive_samples=args.idle_consecutive_samples,
                timeout_seconds=args.idle_timeout_seconds,
            )
            idle_preflights.append(
                {
                    "repetition": repetition,
                    "order_index": order_index,
                    "artifact": name,
                    **idle_preflight,
                }
            )
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

    reference_values = summaries[args.speed_reference]["prompt_tokens_per_second"]["values"]
    paired_speed_ratios = {
        name: summarize_ratios(summary["prompt_tokens_per_second"]["values"], reference_values)
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
        "idle_preflights": idle_preflights,
        "speed_reference": args.speed_reference,
        "hardware": cpu_environment(args.threads),
        "runtime_build": runtime_build_contract(binary, root),
        "benchmark_script": file_identity(Path(__file__).resolve(), root),
        "artifacts": {name: file_identity(path, root) for name, path in model_paths.items()},
        "runs": runs,
        "summaries": summaries,
        "paired_speed_ratios_vs_reference": paired_speed_ratios,
        "interpretation": (
            "All prediction and token-count contracts are stable across repetitions. Throughput "
            "comparisons may be reported with paired run-level intervals."
            if not errors
            else "A repeated-run contract failed; no throughput comparison is permitted."
        ),
    }
    if args.speed_reference == "fp16_teacher":
        report["paired_speed_ratios_vs_fp16"] = paired_speed_ratios
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
