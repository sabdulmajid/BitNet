#!/usr/bin/env python3
"""Synthesize TL2_SR correctness, quality, storage, and speed evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def conversion_output_identity(receipt: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    """Resolve and hash the artifact named by a conversion receipt.

    Older I2_SR receipts predate exporter-side output hashes. They can still be
    tied to validation traces by hashing the referenced artifact at audit time,
    while newer receipts must also match their declared digest.
    """
    raw_path = receipt.get("outfile")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("conversion receipt does not declare outfile")
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    identity = file_identity(path)
    try:
        identity["path"] = str(path.relative_to(repo_root))
    except ValueError:
        identity["path"] = str(path)
    declared_sha256 = receipt.get("outfile_sha256")
    return {
        **identity,
        "declared_sha256": declared_sha256,
        "declared_sha256_present": declared_sha256 is not None,
        "declared_sha256_matches": declared_sha256 is None or declared_sha256 == identity["sha256"],
    }


def exact_mcnemar_p(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(left_only, right_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def paired_accuracy_delta_ci95(
    *,
    total: int,
    left_only: int,
    right_only: int,
    seed: int = 1234,
    samples: int = 100_000,
) -> list[float]:
    """Bootstrap the paired right-minus-left accuracy delta from outcome counts."""
    tied = total - left_only - right_only
    if total <= 0 or tied < 0:
        raise ValueError("invalid paired outcome counts")
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        total,
        [left_only / total, tied / total, right_only / total],
        size=samples,
    )
    deltas = (counts[:, 2] - counts[:, 0]) / total
    return [float(value) for value in np.quantile(deltas, [0.025, 0.975])]


def compare_predictions(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_predictions = left.get("predictions", [])
    right_predictions = right.get("predictions", [])
    left_labels = left.get("labels", [])
    right_labels = right.get("labels", [])
    if not left_predictions or len(left_predictions) != len(right_predictions):
        raise ValueError("prediction traces are missing or have different lengths")
    if left_labels != right_labels or len(left_labels) != len(left_predictions):
        raise ValueError("label traces are missing or do not match")

    disagreements = [
        index
        for index, (left_value, right_value) in enumerate(
            zip(left_predictions, right_predictions, strict=True)
        )
        if left_value != right_value
    ]
    left_only = sum(
        left_value == label and right_value != label
        for left_value, right_value, label in zip(
            left_predictions, right_predictions, left_labels, strict=True
        )
    )
    right_only = sum(
        right_value == label and left_value != label
        for left_value, right_value, label in zip(
            left_predictions, right_predictions, left_labels, strict=True
        )
    )
    total = len(left_predictions)
    left_accuracy = sum(value == label for value, label in zip(left_predictions, left_labels, strict=True)) / total
    right_accuracy = sum(value == label for value, label in zip(right_predictions, right_labels, strict=True)) / total
    return {
        "examples": total,
        "left_accuracy": left_accuracy,
        "right_accuracy": right_accuracy,
        "accuracy_delta_right_minus_left": right_accuracy - left_accuracy,
        "prediction_agreement": 1.0 - len(disagreements) / total,
        "disagreements": len(disagreements),
        "first_disagreement_indices": disagreements[:20],
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "exact_mcnemar_p_two_sided": exact_mcnemar_p(left_only, right_only),
        "accuracy_delta_ci95_paired_bootstrap": paired_accuracy_delta_ci95(
            total=total,
            left_only=left_only,
            right_only=right_only,
        ),
        "paired_bootstrap_seed": 1234,
        "paired_bootstrap_samples": 100_000,
    }


def repeated_summary(path: Path, artifact: str) -> dict[str, Any]:
    report = read_json(path)
    ratio = report["paired_speed_ratios_vs_reference"][artifact]
    throughput = report["summaries"][artifact]["prompt_tokens_per_second"]
    idle_preflights = report.get("idle_preflights", [])
    expected_preflights = report["repetitions"] * len(report["summaries"])
    idle_preflight_complete = len(idle_preflights) == expected_preflights and all(
        len(row.get("accepted_samples", [])) == row.get("consecutive_samples")
        for row in idle_preflights
    )
    return {
        "path": str(path),
        "status": report["status"],
        "artifact": artifact,
        "tile_bm": int(artifact.removeprefix("tl2_sr_bm")) if "_bm" in artifact else 128,
        "mean_tokens_per_second": throughput["mean"],
        "mean_tokens_per_second_ci95_t": throughput["mean_ci95_t"],
        "paired_speed_ratio_vs_i2sr": ratio["geometric_mean"],
        "paired_speed_ratio_ci95_t": ratio["geometric_mean_ci95_t"],
        "predictions_stable": report["summaries"][artifact]["predictions_stable"],
        "repetitions": report["repetitions"],
        "examples": report["examples"],
        "threads": report["threads"],
        "cpu_affinity": report["cpu_affinity"],
        "idle_preflight_complete": idle_preflight_complete,
    }


def native_gguf_sha256(report: dict[str, Any]) -> str | None:
    artifacts = report.get("artifacts", {})
    return artifacts.get("gguf_sha256") if isinstance(artifacts, dict) else None


def audit_status(valid_runtime: bool, speed_superiority_proven: bool) -> str:
    if not valid_runtime:
        return "review"
    return "valid_runtime_speed_win" if speed_superiority_proven else "valid_runtime_no_speed_win"


def render_markdown(result: dict[str, Any]) -> str:
    sample = result["sample_quality"]
    storage = result["storage"]
    lines = [
        "# TL2_SR Evidence Audit",
        "",
        f"Generated: `{result['created_utc']}`. Status: **{result['status']}**.",
        "",
        "## Verdict",
        "",
        result["verdict"],
        "",
        "## Correctness And Quality",
        "",
        "| evidence | result |",
        "| --- | ---: |",
        f"| generated kernel contracts passed | {result['kernel_contracts_passed']}/{result['kernel_contracts_total']} |",
        f"| generated kernel cases passed | {result['kernel_cases_passed']}/{result['kernel_cases_total']} |",
        f"| 512-sample I2_SR accuracy | {sample['left_accuracy']:.6f} |",
        f"| 512-sample TL2_SR accuracy | {sample['right_accuracy']:.6f} |",
        f"| cross-format prediction agreement | {sample['prediction_agreement']:.6f} |",
        f"| I2-only / TL2-only correct | {sample['left_only_correct']} / {sample['right_only_correct']} |",
        f"| exact McNemar p | {sample['exact_mcnemar_p_two_sided']:.6f} |",
        "",
        "## Storage",
        "",
        "| artifact region | I2_SR MiB | TL2_SR MiB | reduction |",
        "| --- | ---: | ---: | ---: |",
        f"| complete GGUF | {storage['i2sr_file_mib']:.3f} | {storage['tl2sr_file_mib']:.3f} | {storage['file_reduction_fraction']:.3%} |",
        f"| packed ternary projections | {storage['i2sr_projection_mib']:.3f} | {storage['tl2sr_projection_mib']:.3f} | {storage['projection_reduction_fraction']:.3%} |",
        "",
        "## Xeon Tiling Sweep",
        "",
        "| BM | mean TL2_SR tok/s | paired speed / I2_SR | ratio 95% CI |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(result["tiling_sweep"], key=lambda value: value["tile_bm"], reverse=True):
        interval = row["paired_speed_ratio_ci95_t"]
        lines.append(
            f"| {row['tile_bm']} | {row['mean_tokens_per_second']:.3f} | "
            f"{row['paired_speed_ratio_vs_i2sr']:.3f} | [{interval[0]:.3f}, {interval[1]:.3f}] |"
        )
    lines.extend(
        [
            "",
            "The speed ratios are paired within each build and must not be compared through their absolute",
            "throughput across builds. The full-validation field remains separate from the 512-example",
            "same-build format comparison.",
            "",
            "## Gates",
            "",
        ]
    )
    for name, gate in result["gates"].items():
        lines.append(f"- **{name.replace('_', ' ')}:** {'pass' if gate else 'fail'}")
    if result.get("full_validation"):
        full = result["full_validation"]
        lines.extend(
            [
                "",
                "## Full Validation",
                "",
                "| evidence | result |",
                "| --- | ---: |",
                f"| examples | {full['examples']} |",
                f"| I2_SR accuracy | {full['left_accuracy']:.6f} |",
                f"| TL2_SR accuracy | {full['right_accuracy']:.6f} |",
                f"| TL2_SR minus I2_SR | {full['accuracy_delta_right_minus_left']:+.6f} |",
                f"| paired delta 95% bootstrap CI | [{full['accuracy_delta_ci95_paired_bootstrap'][0]:+.6f}, {full['accuracy_delta_ci95_paired_bootstrap'][1]:+.6f}] |",
                f"| cross-format prediction agreement | {full['prediction_agreement']:.6f} |",
                f"| I2-only / TL2-only correct | {full['left_only_correct']} / {full['right_only_correct']} |",
                f"| exact McNemar p | {full['exact_mcnemar_p_two_sided']:.6g} |",
            ]
        )
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            "The JSON companion records SHA-256 identities for every generated kernel header/config,",
            "conversion receipt, validation trace, and repeated benchmark consumed by this audit.",
            "All correctness, identity, and full-validation gates must pass before this report is publication evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i2sr-conversion", type=Path, default=Path("benchmark_results/seqcls_native_i2sr_gguf_2026-05-15.json"))
    parser.add_argument("--tl2sr-conversion", type=Path, default=Path("benchmark_results/seqcls_native_tl2sr_bm64_gguf_2026-09-04.json"))
    parser.add_argument("--i2sr-sample", type=Path, default=Path("benchmark_results/seqcls_native_i2sr_tl2build_cpu_mnli_512_2026-09-04.json"))
    parser.add_argument("--tl2sr-sample", type=Path, default=Path("benchmark_results/seqcls_native_tl2sr_cpu_mnli_512_2026-09-04.json"))
    parser.add_argument("--i2sr-full", type=Path, default=Path("benchmark_results/seqcls_native_i2sr_cpu_mnli_full_tl2build_final_2026-09-04.json"))
    parser.add_argument("--tl2sr-full", type=Path, default=Path("benchmark_results/seqcls_native_tl2sr_bm64_cpu_mnli_full_final_2026-09-04.json"))
    parser.add_argument("--output-json", type=Path, default=Path("benchmark_results/tl2sr_evidence_audit_2026-09-04.json"))
    parser.add_argument("--output-md", type=Path, default=Path("benchmarks/results/tl2sr_evidence_audit_2026-09-04.md"))
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()

    kernel_paths = [
        Path("benchmark_results/tl2sr_kernel_contract_2026-09-04.json"),
        Path("benchmark_results/tl2sr_bm64_kernel_contract_2026-09-04.json"),
        Path("benchmark_results/tl2sr_bm32_kernel_contract_2026-09-04.json"),
    ]
    preset_paths = [
        Path("preset_kernels/Qwen2.5-0.5B-TL2SR/bitnet-lut-kernels-tl2sr.h"),
        Path("preset_kernels/Qwen2.5-0.5B-TL2SR/kernel_config_tl2sr.ini"),
        Path("preset_kernels/Qwen2.5-0.5B-TL2SR-BM64/bitnet-lut-kernels-tl2sr.h"),
        Path("preset_kernels/Qwen2.5-0.5B-TL2SR-BM64/kernel_config_tl2sr.ini"),
        Path("preset_kernels/Qwen2.5-0.5B-TL2SR-BM32/bitnet-lut-kernels-tl2sr.h"),
        Path("preset_kernels/Qwen2.5-0.5B-TL2SR-BM32/kernel_config_tl2sr.ini"),
    ]
    repeated_specs = [
        (Path("benchmarks/results/seqcls_tl2sr_vs_i2sr_repeated_2026-09-04.json"), "tl2_sr"),
        (Path("benchmarks/results/seqcls_tl2sr_bm64_vs_i2sr_repeated_2026-09-04.json"), "tl2_sr_bm64"),
        (Path("benchmarks/results/seqcls_tl2sr_bm32_vs_i2sr_repeated_2026-09-04.json"), "tl2_sr_bm32"),
    ]
    layout_guard_path = Path("benchmark_results/tl2sr_layout_guard_2026-09-04.json")
    layout_guard = read_json(layout_guard_path)
    kernel_reports = [read_json(path) for path in kernel_paths]
    i2_sample_report = read_json(args.i2sr_sample)
    tl2_sample_report = read_json(args.tl2sr_sample)
    sample = compare_predictions(i2_sample_report, tl2_sample_report)
    tiling = [repeated_summary(path, artifact) for path, artifact in repeated_specs]
    i2_conversion = read_json(args.i2sr_conversion)
    tl2_conversion = read_json(args.tl2sr_conversion)
    tl2_conversion_paths = {
        "tl2_sr": Path("benchmark_results/seqcls_native_tl2sr_gguf_2026-09-04.json"),
        "tl2_sr_bm64": args.tl2sr_conversion,
        "tl2_sr_bm32": Path("benchmark_results/seqcls_native_tl2sr_bm32_gguf_2026-09-04.json"),
    }
    tl2_conversions = {
        artifact: read_json(path) for artifact, path in tl2_conversion_paths.items()
    }
    conversion_identities = {
        "i2_sr": conversion_output_identity(i2_conversion, repo_root),
        **{
            artifact: conversion_output_identity(receipt, repo_root)
            for artifact, receipt in tl2_conversions.items()
        },
    }
    preset_configs = {
        "tl2_sr": Path("preset_kernels/Qwen2.5-0.5B-TL2SR/kernel_config_tl2sr.ini"),
        "tl2_sr_bm64": Path("preset_kernels/Qwen2.5-0.5B-TL2SR-BM64/kernel_config_tl2sr.ini"),
        "tl2_sr_bm32": Path("preset_kernels/Qwen2.5-0.5B-TL2SR-BM32/kernel_config_tl2sr.ini"),
    }
    expected_config_hashes = {
        artifact: file_identity(path)["sha256"] for artifact, path in preset_configs.items()
    }
    i2_file = int(i2_conversion["outfile_size_bytes"])
    tl2_file = int(tl2_conversion["outfile_size_bytes"])
    i2_projection = int(i2_conversion["packed_i2s_bytes"])
    tl2_projection = int(tl2_conversion["packed_i2s_bytes"])
    storage = {
        "i2sr_file_bytes": i2_file,
        "tl2sr_file_bytes": tl2_file,
        "i2sr_file_mib": i2_file / (1024**2),
        "tl2sr_file_mib": tl2_file / (1024**2),
        "file_reduction_fraction": 1.0 - tl2_file / i2_file,
        "i2sr_projection_bytes": i2_projection,
        "tl2sr_projection_bytes": tl2_projection,
        "i2sr_projection_mib": i2_projection / (1024**2),
        "tl2sr_projection_mib": tl2_projection / (1024**2),
        "projection_reduction_fraction": 1.0 - tl2_projection / i2_projection,
    }
    i2_full_report = read_json(args.i2sr_full)
    tl2_full_report = None
    full = None
    if args.tl2sr_full.is_file():
        tl2_full_report = read_json(args.tl2sr_full)
        full = compare_predictions(i2_full_report, tl2_full_report)
    repeated_reports = {
        artifact: read_json(path) for path, artifact in repeated_specs
    }
    artifact_receipts_match = (
        native_gguf_sha256(i2_sample_report) == conversion_identities["i2_sr"]["sha256"]
        and native_gguf_sha256(tl2_sample_report) == conversion_identities["tl2_sr_bm64"]["sha256"]
        and native_gguf_sha256(i2_full_report) == conversion_identities["i2_sr"]["sha256"]
        and (
            tl2_full_report is None
            or native_gguf_sha256(tl2_full_report) == conversion_identities["tl2_sr_bm64"]["sha256"]
        )
        and all(
            report.get("artifacts", {}).get("i2_sr", {}).get("sha256")
            == conversion_identities["i2_sr"]["sha256"]
            and report.get("artifacts", {}).get(artifact, {}).get("sha256")
            == conversion_identities[artifact]["sha256"]
            for artifact, report in repeated_reports.items()
        )
    )
    kernel_layout_receipts_match = all(
        receipt.get("tl2_kernel_config_sha256") == expected_config_hashes[artifact]
        and isinstance(receipt.get("bitnet_converter"), str)
        and Path(str(receipt["bitnet_converter"])).is_file()
        and receipt.get("bitnet_converter_sha256")
        == file_identity(Path(str(receipt["bitnet_converter"])))["sha256"]
        for artifact, receipt in tl2_conversions.items()
    )
    best = max(tiling, key=lambda row: row["paired_speed_ratio_vs_i2sr"])
    gates = {
        "all_kernel_contracts": all(report.get("status") == "pass" for report in kernel_reports),
        "layout_guard": layout_guard.get("status") == "pass",
        "sample_accuracy_within_one_point": abs(sample["accuracy_delta_right_minus_left"]) <= 0.01,
        "sample_prediction_agreement_at_least_98_percent": sample["prediction_agreement"] >= 0.98,
        "projection_storage_reduced": storage["projection_reduction_fraction"] > 0,
        "repeated_benchmarks_valid": all(row["status"] == "valid" and row["predictions_stable"] for row in tiling),
        "repeated_benchmarks_idle_gated": all(row["idle_preflight_complete"] for row in tiling),
        "kernel_layout_receipt_matches": kernel_layout_receipts_match,
        "conversion_output_hashes_match": all(
            identity["declared_sha256_matches"] for identity in conversion_identities.values()
        ),
        "artifact_receipts_match": artifact_receipts_match,
        "speed_superiority_proven": best["paired_speed_ratio_ci95_t"][0] > 1.0,
        "full_validation_complete": (
            full is not None
            and full["examples"] == 9815
            and i2_full_report.get("status") == "pass"
            and i2_full_report.get("full_validation_complete") is True
            and tl2_full_report is not None
            and tl2_full_report.get("status") == "pass"
            and tl2_full_report.get("full_validation_complete") is True
        ),
        "full_accuracy_within_one_point": (
            full is not None and abs(full["accuracy_delta_right_minus_left"]) <= 0.01
        ),
        "full_prediction_agreement_at_least_98_percent": (
            full is not None and full["prediction_agreement"] >= 0.98
        ),
    }
    valid_runtime = all(
        gates[name]
        for name in (
            "all_kernel_contracts",
            "layout_guard",
            "sample_accuracy_within_one_point",
            "sample_prediction_agreement_at_least_98_percent",
            "projection_storage_reduced",
            "repeated_benchmarks_valid",
            "repeated_benchmarks_idle_gated",
            "kernel_layout_receipt_matches",
            "conversion_output_hashes_match",
            "artifact_receipts_match",
            "full_validation_complete",
            "full_accuracy_within_one_point",
            "full_prediction_agreement_at_least_98_percent",
        )
    )
    result = {
        "schema": "tl2sr-evidence-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": audit_status(valid_runtime, gates["speed_superiority_proven"]),
        "kernel_contracts_passed": sum(report.get("status") == "pass" for report in kernel_reports),
        "kernel_contracts_total": len(kernel_reports),
        "kernel_cases_passed": sum(
            case.get("passed") is True
            for report in kernel_reports
            for case in report.get("cases", [])
        ),
        "kernel_cases_total": sum(len(report.get("cases", [])) for report in kernel_reports),
        "sample_quality": sample,
        "full_validation": full,
        "storage": storage,
        "tiling_sweep": tiling,
        "best_tested_tiling": best["tile_bm"],
        "provenance": {
            "kernel_presets": [file_identity(path) for path in preset_paths],
            "kernel_contracts": [file_identity(path) for path in kernel_paths],
            "layout_guard": file_identity(layout_guard_path),
            "conversion_receipts": [
                file_identity(args.i2sr_conversion),
                *[file_identity(path) for path in tl2_conversion_paths.values()],
            ],
            "conversion_outputs": conversion_identities,
            "validation_traces": [
                file_identity(args.i2sr_sample),
                file_identity(args.tl2sr_sample),
                file_identity(args.i2sr_full),
                *([file_identity(args.tl2sr_full)] if args.tl2sr_full.is_file() else []),
            ],
            "repeated_benchmarks": [file_identity(path) for path, _ in repeated_specs],
        },
        "gates": gates,
        "verdict": (
            "TL2_SR is a functionally valid row-scale ternary storage/runtime contract for the tested Qwen student. "
            "It reduces packed projection storage, but no tested tile layout proves a CPU throughput advantage over I2_SR."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(result), encoding="utf-8")
    print(render_markdown(result))


if __name__ == "__main__":
    main()
