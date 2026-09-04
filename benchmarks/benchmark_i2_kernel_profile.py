#!/usr/bin/env python3
"""Profile I2 activation quantization and GEMM at Qwen projection shapes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()
PROJECTION_MULTIPLICITIES = {
    (896, 896): 2,   # q_proj and o_proj
    (896, 128): 2,   # k_proj and v_proj
    (896, 4864): 2,  # gate_proj and up_proj
    (4864, 896): 1,  # down_proj
}
T_CRITICAL_95 = {
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        display = str(resolved.relative_to(root.resolve()))
    except ValueError:
        display = resolved.name
    return {
        "path": display,
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def git_identity(path: Path, display_path: str) -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(path), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    return {
        "path": display_path,
        "revision": git("rev-parse", "HEAD"),
        "tracked_files_dirty": bool(git("status", "--short", "--untracked-files=no")),
    }


def summarize(values: list[float]) -> dict[str, Any]:
    if len(values) < 2:
        raise ValueError("at least two values are required")
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values)
    critical = T_CRITICAL_95.get(len(values), 1.96)
    half_width = critical * standard_deviation / math.sqrt(len(values))
    return {
        "values": values,
        "mean": mean,
        "median": statistics.median(values),
        "standard_deviation": standard_deviation,
        "mean_ci95_t": [mean - half_width, mean + half_width],
        "minimum": min(values),
        "maximum": max(values),
    }


def parse_profile_output(output: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in output.splitlines() if line.strip()]
    expected = set(PROJECTION_MULTIPLICITIES)
    observed = {(int(row["input"]), int(row["output"])) for row in rows}
    if observed != expected or len(rows) != len(expected):
        raise ValueError(f"unexpected projection shapes: {sorted(observed)!r}")
    for row in rows:
        if float(row["max_abs_error"]) != 0.0:
            raise ValueError(f"kernel mismatch for projection row: {row!r}")
    return rows


def aggregate_projection_mix(rows: list[dict[str, Any]]) -> dict[str, float]:
    quantize_us = 0.0
    multiply_us = 0.0
    for row in rows:
        key = (int(row["input"]), int(row["output"]))
        multiplicity = PROJECTION_MULTIPLICITIES[key]
        quantize_us += multiplicity * float(row["quantize_us"])
        multiply_us += multiplicity * float(row["multiply_us"])
    total_us = quantize_us + multiply_us
    fraction = quantize_us / total_us
    return {
        "quantize_us": quantize_us,
        "multiply_us": multiply_us,
        "combined_us": total_us,
        "quantize_fraction": fraction,
        "ideal_speedup_if_quantization_were_free": 1.0 / (1.0 - fraction),
    }


def render_markdown(report: dict[str, Any]) -> str:
    table_rows = []
    for shape in report["shape_summaries"]:
        q = shape["quantize_us"]
        m = shape["multiply_us"]
        f = shape["quantize_fraction"]
        table_rows.append(
            f"| {shape['input']} x {shape['output']} | {shape['multiplicity']} | "
            f"{q['mean']:.3f} | {m['mean']:.3f} | {100.0 * f['mean']:.2f}% |"
        )

    aggregate = report["aggregate_projection_mix"]
    fraction = aggregate["quantize_fraction"]
    ideal = aggregate["ideal_speedup_if_quantization_were_free"]
    return "\n".join(
        [
            "# I2 Kernel Cost Profile",
            "",
            f"Generated: `{report['created_utc']}`. Status: **{report['status']}**.",
            "",
            f"Protocol: `{report['outer_repetitions']}` process repetitions; each reports the "
            f"median of `{report['inner_iterations']}` timed calls over `{report['tokens']}` "
            f"activation rows on CPU `{report['cpu_affinity']}`.",
            "",
            "| projection (input x output) | uses/layer | A8 quantize us | I2 GEMM us | quantize share |",
            "| --- | ---: | ---: | ---: | ---: |",
            *table_rows,
            "",
            "## Aggregate Qwen2.5-0.5B Projection Mix",
            "",
            f"- Activation quantization share: `{100.0 * fraction['mean']:.2f}%` "
            f"(95% t interval `[{100.0 * fraction['mean_ci95_t'][0]:.2f}%, "
            f"{100.0 * fraction['mean_ci95_t'][1]:.2f}%]`).",
            f"- I2 dot/GEMM share: `{100.0 * (1.0 - fraction['mean']):.2f}%`.",
            f"- Ideal upper-bound speedup from deleting A8 quantization entirely: "
            f"`{ideal['mean']:.4f}x`.",
            f"- Maximum scalar-reference error over all runs: "
            f"`{report['maximum_abs_error']:.1f}` raw accumulator units.",
            "",
            "## Interpretation",
            "",
            report["interpretation"],
            "",
            "## Claim Boundary",
            "",
            "- This isolates one CPU core and the four dense projection shapes in a Qwen2.5-0.5B block.",
            "- It measures raw activation quantization and packed I2 GEMM, excluding graph scheduling, normalization, attention, and model loading.",
            "- The aggregate weights projections by architectural use count; it is not an end-to-end latency attribution.",
            "- The reported upper bound is Amdahl's law for this isolated projection mix, not a forecast of model throughput.",
            "",
        ]
    )


def public_compile_command(command: list[str], root: Path) -> list[str]:
    public = []
    skip_next = False
    for value in command:
        if skip_next:
            skip_next = False
            continue
        if value == "-o":
            skip_next = True
            continue
        public.append(value.replace(str(root), "$REPO_ROOT"))
    return public


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--build-dir", type=Path, default=Path("build-portable-avx2"))
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--inner-iterations", type=int, default=31)
    parser.add_argument("--outer-repetitions", type=int, default=5)
    parser.add_argument("--cpu-affinity", default="0")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(f"benchmarks/results/i2_kernel_profile_{DATE}.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(f"benchmarks/results/i2_kernel_profile_{DATE}.md"),
    )
    args = parser.parse_args()
    if args.tokens <= 0 or args.tokens % 4 != 0:
        raise SystemExit("--tokens must be a positive multiple of four")
    if args.inner_iterations < 2 or args.outer_repetitions < 2:
        raise SystemExit("both iteration counts must be at least two")

    root = args.repo_root.resolve()
    build_dir = args.build_dir if args.build_dir.is_absolute() else root / args.build_dir
    library = build_dir / "3rdparty/llama.cpp/ggml/src/libggml.so"
    source = root / "benchmarks/i2_kernel_profile.cpp"
    kernel_source = root / "src/ggml-bitnet-mad.cpp"
    for required in (library, source, kernel_source):
        if not required.is_file():
            raise SystemExit(f"required file is missing: {required}")

    with tempfile.TemporaryDirectory(prefix="bitnet-i2-profile-") as temporary:
        binary = Path(temporary) / "i2-kernel-profile"
        command = [
            "c++",
            "-O3",
            "-std=c++17",
            "-mavx2",
            "-mfma",
            str(source),
            f"-L{library.parent}",
            f"-Wl,-rpath,{library.parent}",
            "-lggml",
            "-o",
            str(binary),
        ]
        subprocess.run(command, cwd=root, check=True)

        runs: list[list[dict[str, Any]]] = []
        for repetition in range(args.outer_repetitions):
            result = subprocess.run(
                [
                    "taskset",
                    "-c",
                    args.cpu_affinity,
                    str(binary),
                    str(args.tokens),
                    str(args.inner_iterations),
                ],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            rows = parse_profile_output(result.stdout)
            runs.append(rows)
            print(f"repetition={repetition + 1}/{args.outer_repetitions}", flush=True)

    rows_by_shape: dict[tuple[int, int], list[dict[str, Any]]] = {
        shape: [] for shape in PROJECTION_MULTIPLICITIES
    }
    aggregates = []
    for rows in runs:
        aggregates.append(aggregate_projection_mix(rows))
        for row in rows:
            rows_by_shape[(int(row["input"]), int(row["output"]))].append(row)

    shape_summaries = []
    for shape, rows in rows_by_shape.items():
        shape_summaries.append(
            {
                "input": shape[0],
                "output": shape[1],
                "multiplicity": PROJECTION_MULTIPLICITIES[shape],
                "quantize_us": summarize([float(row["quantize_us"]) for row in rows]),
                "multiply_us": summarize([float(row["multiply_us"]) for row in rows]),
                "quantize_fraction": summarize(
                    [float(row["quantize_fraction"]) for row in rows]
                ),
            }
        )

    aggregate_summary = {
        key: summarize([float(row[key]) for row in aggregates])
        for key in aggregates[0]
    }
    maximum_abs_error = max(
        float(row["max_abs_error"]) for rows in runs for row in rows
    )
    errors = []
    repositories = {
        "bitnet": git_identity(root, "$REPO_ROOT"),
        "llama_cpp": git_identity(root / "3rdparty/llama.cpp", "$REPO_ROOT/3rdparty/llama.cpp"),
    }
    for name, identity in repositories.items():
        if identity["tracked_files_dirty"]:
            errors.append(f"{name} tracked source was dirty")
    if maximum_abs_error != 0.0:
        errors.append("I2 kernel output differed from the scalar reference")

    report = {
        "schema": "i2-kernel-profile-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "valid" if not errors else "invalid",
        "errors": errors,
        "tokens": args.tokens,
        "inner_iterations": args.inner_iterations,
        "outer_repetitions": args.outer_repetitions,
        "cpu_affinity": args.cpu_affinity,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "repositories": repositories,
        "artifacts": {
            "benchmark_source": file_identity(source, root),
            "driver": file_identity(Path(__file__), root),
            "kernel_source": file_identity(kernel_source, root),
            "ggml_library": file_identity(library, root),
        },
        "compile_command": public_compile_command(command, root),
        "runs": runs,
        "shape_summaries": shape_summaries,
        "aggregate_projection_mix": aggregate_summary,
        "maximum_abs_error": maximum_abs_error,
        "interpretation": (
            "Activation quantization is a minority cost in the isolated projection mix. "
            "The packed I2 dot/GEMM implementation is therefore the material CPU optimization "
            "target; eliminating activation quantization alone cannot close the measured "
            "end-to-end gap to FP16."
        ),
    }

    output_json = args.output_json if args.output_json.is_absolute() else root / args.output_json
    output_md = args.output_md if args.output_md.is_absolute() else root / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output_json)}, indent=2))
    return 0 if report["status"] == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
