#!/usr/bin/env python3
"""Audit a paired native CPU matrix for FP, conventional, and ternary models.

The audit keeps two estimands separate:

* a same-teacher format effect, such as Q4_0 versus F16; and
* a deployed-model effect, such as a trained I2_SR student versus its teacher.

Those are not interchangeable: the latter includes training-recipe differences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.audit_bitdistill_adaptive_full import exact_mcnemar_pvalue


DEFAULT_ARTIFACTS = {
    "fp16_teacher": Path("benchmark_results/seqcls_native_fp16_teacher_cpu_mnli_512_xeon_2026-09-04.json"),
    "q4_0_teacher": Path("benchmark_results/seqcls_native_fp16_teacher_q4_0_cpu_mnli_512_xeon_2026-09-04.json"),
    "i2_sr_student": Path("benchmark_results/seqcls_native_i2sr_cpu_mnli_512_xeon_2026-09-04.json"),
}
SEMANTIC_FAMILY = {
    "fp16_teacher": "fp16_teacher",
    "q4_0_teacher": "fp16_teacher",
    "i2_sr_student": "qat_student",
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_quality(
    candidate: list[int],
    reference: list[int],
    labels: list[int],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    if not (len(candidate) == len(reference) == len(labels)) or not labels:
        raise ValueError("paired quality requires equal, non-empty prediction and label arrays")
    candidate_ok = np.equal(candidate, labels).astype(np.int8)
    reference_ok = np.equal(reference, labels).astype(np.int8)
    deltas = candidate_ok - reference_ok
    candidate_wins = int(np.count_nonzero(deltas == 1))
    reference_wins = int(np.count_nonzero(deltas == -1))
    delta = float(np.mean(deltas))
    if len(deltas) > 1:
        standard_error = float(np.std(deltas, ddof=1) / math.sqrt(len(deltas)))
    else:
        standard_error = 0.0
    normal_ci = [delta - 1.959963984540054 * standard_error, delta + 1.959963984540054 * standard_error]

    counts = np.asarray(
        [reference_wins, len(deltas) - candidate_wins - reference_wins, candidate_wins],
        dtype=np.int64,
    )
    rng = np.random.default_rng(seed)
    bootstrap_counts = rng.multinomial(len(deltas), counts / len(deltas), size=bootstrap_samples)
    bootstrap_deltas = (bootstrap_counts[:, 2] - bootstrap_counts[:, 0]) / len(deltas)
    bootstrap_ci = np.quantile(bootstrap_deltas, [0.025, 0.975]).tolist()
    return {
        "examples": len(labels),
        "candidate_accuracy": float(np.mean(candidate_ok)),
        "reference_accuracy": float(np.mean(reference_ok)),
        "delta_candidate_minus_reference": delta,
        "paired_standard_error": standard_error,
        "paired_normal_ci95": normal_ci,
        "paired_bootstrap_ci95": bootstrap_ci,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "candidate_wins": candidate_wins,
        "reference_wins": reference_wins,
        "prediction_agreement": float(np.mean(np.equal(candidate, reference))),
        "mcnemar_exact_p": exact_mcnemar_pvalue(candidate_wins, reference_wins),
    }


def validate_matrix(artifacts: dict[str, dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    reference = artifacts["fp16_teacher"]
    contract_fields = (
        "task",
        "max_samples",
        "prompt_input",
        "prompt_batch_size",
        "embedding_sequential",
        "batch_size",
        "ubatch_size",
    )
    reference_labels = reference.get("labels")
    reference_predictions = reference.get("predictions")
    if not isinstance(reference_labels, list) or not isinstance(reference_predictions, list):
        errors.append("fp16_teacher lacks labels or predictions")
        return errors
    if len(reference_labels) != len(reference_predictions):
        errors.append("fp16_teacher label/prediction count mismatch")
    reference_build = (reference.get("runtime_build") or {}).get("sha256")
    reference_cpu = reference.get("hardware") or {}

    for name, artifact in artifacts.items():
        if artifact.get("schema") != "seqcls_native_cpu.v2":
            errors.append(f"{name}: schema={artifact.get('schema')!r}, expected seqcls_native_cpu.v2")
        for field in contract_fields:
            if artifact.get(field) != reference.get(field):
                errors.append(
                    f"{name}: {field}={artifact.get(field)!r}, reference={reference.get(field)!r}"
                )
        labels = artifact.get("labels")
        predictions = artifact.get("predictions")
        if labels != reference_labels:
            errors.append(f"{name}: labels differ from fp16_teacher")
        if not isinstance(predictions, list) or len(predictions) != len(reference_labels):
            errors.append(f"{name}: prediction count does not match fp16_teacher")
            continue
        recomputed = sum(int(pred == label) for pred, label in zip(predictions, labels)) / len(labels)
        recorded = (artifact.get("summary") or {}).get("accuracy")
        if not isinstance(recorded, (int, float)) or not math.isclose(recomputed, float(recorded), abs_tol=1e-15):
            errors.append(f"{name}: recorded accuracy disagrees with predictions")
        if (artifact.get("runtime_build") or {}).get("sha256") != reference_build:
            errors.append(f"{name}: runtime build contract differs from fp16_teacher")
        hardware = artifact.get("hardware") or {}
        for field in ("cpu_model", "requested_threads", "logical_cpus_cpuinfo", "physical_cores_cpuinfo"):
            if hardware.get(field) != reference_cpu.get(field):
                errors.append(f"{name}: hardware {field} differs from fp16_teacher")
        for repo_name, identity in ((artifact.get("runtime_build") or {}).get("repositories") or {}).items():
            if isinstance(identity, dict) and identity.get("tracked_files_dirty") is not False:
                errors.append(f"{name}: {repo_name} tracked source was dirty during benchmark")
    return errors


def system_effect(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    candidate_size = float(candidate["artifacts"]["gguf_size_bytes"])
    reference_size = float(reference["artifacts"]["gguf_size_bytes"])
    candidate_tps = float(candidate["runtime"]["prompt_eval_tokens_per_second"])
    reference_tps = float(reference["runtime"]["prompt_eval_tokens_per_second"])
    candidate_rss = float(candidate["runtime"]["child_peak_rss_mib"])
    reference_rss = float(reference["runtime"]["child_peak_rss_mib"])
    return {
        "candidate_size_bytes": int(candidate_size),
        "reference_size_bytes": int(reference_size),
        "size_ratio_reference_over_candidate": reference_size / candidate_size,
        "size_reduction_fraction": 1.0 - candidate_size / reference_size,
        "candidate_prompt_tokens_per_second": candidate_tps,
        "reference_prompt_tokens_per_second": reference_tps,
        "throughput_ratio_candidate_over_reference": candidate_tps / reference_tps,
        "candidate_peak_rss_mib": candidate_rss,
        "reference_peak_rss_mib": reference_rss,
        "rss_reduction_fraction": 1.0 - candidate_rss / reference_rss,
    }


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(fmt(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def render_markdown(report: dict[str, Any]) -> str:
    artifact_rows = []
    for name, row in report["artifacts"].items():
        artifact_rows.append(
            [
                name,
                row["semantic_family"],
                row["accuracy"],
                row["gguf_mib"],
                row["prompt_tokens_per_second"],
                row["examples_per_second"],
                row["peak_rss_mib"],
            ]
        )
    comparison_rows = []
    for name, comparison in report["comparisons"].items():
        quality = comparison["quality"]
        system = comparison["system"]
        comparison_rows.append(
            [
                name,
                comparison["estimand"],
                quality["delta_candidate_minus_reference"],
                quality["paired_bootstrap_ci95"],
                quality["mcnemar_exact_p"],
                system["size_ratio_reference_over_candidate"],
                system["throughput_ratio_candidate_over_reference"],
            ]
        )
    return "\n\n".join(
        [
            "# Native MNLI CPU Deployment Matrix",
            f"Generated: `{report['created_utc']}`. Status: **{report['status']}**.",
            "## Artifacts",
            table(
                ["artifact", "function", "accuracy", "MiB", "prompt tok/s", "examples/s", "peak RSS MiB"],
                artifact_rows,
            ),
            "## Paired Comparisons",
            table(
                ["comparison", "estimand", "accuracy delta", "paired 95% CI", "McNemar p", "size factor", "speed factor"],
                comparison_rows,
            ),
            "## Interpretation",
            report["interpretation"],
            "## Claim Boundary",
            "\n".join(f"- {item}" for item in report["claim_boundary"]),
            "## Validation",
            "No contract violations." if not report["errors"] else "\n".join(f"- {error}" for error in report["errors"]),
        ]
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in DEFAULT_ARTIFACTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=default)
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260904)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmarks/results/seqcls_native_cpu_matrix_2026-09-04.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("benchmarks/results/seqcls_native_cpu_matrix_2026-09-04.md"),
    )
    args = parser.parse_args()

    paths = {name: getattr(args, name) for name in DEFAULT_ARTIFACTS}
    artifacts = {name: read_json(path) for name, path in paths.items()}
    errors = validate_matrix(artifacts)
    reference = artifacts["fp16_teacher"]
    labels = [int(value) for value in reference.get("labels", [])]
    comparisons: dict[str, Any] = {}
    if labels and all(len(value.get("predictions", [])) == len(labels) for value in artifacts.values()):
        for offset, name in enumerate(("q4_0_teacher", "i2_sr_student")):
            same_function = SEMANTIC_FAMILY[name] == SEMANTIC_FAMILY["fp16_teacher"]
            comparisons[f"{name}_vs_fp16_teacher"] = {
                "candidate": name,
                "reference": "fp16_teacher",
                "estimand": "same-teacher format effect" if same_function else "deployed-model effect",
                "same_predeployment_function": same_function,
                "quality": paired_quality(
                    [int(value) for value in artifacts[name]["predictions"]],
                    [int(value) for value in reference["predictions"]],
                    labels,
                    bootstrap_samples=args.bootstrap_samples,
                    seed=args.bootstrap_seed + offset,
                ),
                "system": system_effect(artifacts[name], reference),
            }

    artifact_summary = {
        name: {
            "semantic_family": SEMANTIC_FAMILY[name],
            "source": str(paths[name]),
            "source_sha256": sha256_file(paths[name]),
            "gguf": value["artifacts"]["gguf"],
            "gguf_sha256": value["artifacts"]["gguf_sha256"],
            "gguf_mib": value["artifacts"]["gguf_size_bytes"] / (1024 * 1024),
            "accuracy": value["summary"]["accuracy"],
            "prompt_tokens_per_second": value["runtime"]["prompt_eval_tokens_per_second"],
            "examples_per_second": value["runtime"]["examples_per_second"],
            "peak_rss_mib": value["runtime"]["child_peak_rss_mib"],
            "prediction_sha256": value["prediction_sha256"],
            "runtime_build_sha256": value["runtime_build"]["sha256"],
        }
        for name, value in artifacts.items()
    }
    status = "valid_sample_matrix" if not errors else "invalid"
    interpretation = (
        "The matrix is contract-valid. Q4_0 versus F16 estimates a format-only effect on the same "
        "teacher. I2_SR versus F16 estimates the end-to-end deployed-student tradeoff and must not "
        "be interpreted as a pure quantization-format effect. Statistical and systems conclusions "
        "are limited to this fixed MNLI sample and hardware."
        if not errors
        else "One or more task, build, hardware, or prediction contracts differ; comparative claims are blocked."
    )
    report = {
        "schema": "seqcls-native-cpu-matrix-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "errors": errors,
        "artifacts": artifact_summary,
        "comparisons": comparisons,
        "interpretation": interpretation,
        "claim_boundary": [
            "The fixed sample is the first N MNLI validation_matched rows; it is not a randomized benchmark sample.",
            "Accuracy intervals are paired over examples; throughput is a single-run measurement and has no timing confidence interval.",
            "I2_SR is a separately trained QAT student, so its comparison with the FP16 teacher includes training and format effects.",
            "Peak RSS includes runtime overhead and shared libraries; GGUF bytes are the cleaner storage measurement.",
            "General language-model quality, other tasks, other CPUs, and energy use are outside this matrix.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
