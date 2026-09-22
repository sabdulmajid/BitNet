#!/usr/bin/env python3
"""Build the compact public snapshot used by the README and research status."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(report: dict[str, Any], *, schema: str, status: str | None = None) -> None:
    if report.get("schema") != schema:
        raise ValueError(f"expected schema {schema!r}, found {report.get('schema')!r}")
    if status is not None and report.get("status") != status:
        raise ValueError(f"expected status {status!r}, found {report.get('status')!r}")


def source(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        public_path = resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"evidence source must be inside the repository: {path}") from exc
    return {"path": public_path.as_posix(), "sha256": sha256(resolved)}


def build_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    canonical = load_json(args.canonical)
    matched = load_json(args.matched_audit)
    predictions = load_json(args.prediction_bundle)
    tl2 = load_json(args.tl2_audit)
    cpu_matrix = load_json(args.cpu_matrix)
    cpu_repeated = load_json(args.cpu_repeated)
    initializer = load_json(args.initializer_audit)

    require(canonical, schema="bitnet-canonical-evidence-bundle-v1")
    require(
        matched,
        schema="bitdistill-adaptive-vs-fixed-matched-audit-v1",
        status="complete",
    )
    require(predictions, schema="compact-classification-predictions-v1")
    require(tl2, schema="tl2sr-evidence-audit-v1", status="valid_runtime_no_speed_win")
    require(cpu_matrix, schema="seqcls-native-cpu-matrix-v1", status="valid_sample_matrix")
    require(cpu_repeated, schema="seqcls-native-cpu-repeated-v1", status="valid")
    require(initializer, schema="second-order-ternary-init-audit-v1")

    matched_models = predictions["models"]
    fp16_accuracy = float(matched_models["fp16"]["accuracy"])
    aggregate = matched["aggregate"]
    adaptive_mean = float(aggregate["adaptive_mean_accuracy"])
    fixed_mean = float(aggregate["fixed60_mean_accuracy"])
    blind = canonical["claims"]["blind_ptq"]
    qat = canonical["claims"]["qat_distill"]
    scale_contract = canonical["claims"]["row_scale_runtime_contract"]
    causal_cpu = canonical["claims"]["i2sr_cpu"]
    mixed_artifact = cpu_matrix["artifacts"]["i2_sr_q8_embedding_student"]
    mixed_comparison = cpu_matrix["comparisons"][
        "i2_sr_q8_embedding_student_vs_i2_sr_student"
    ]

    snapshot = {
        "schema": "bitnet-current-evidence-v1",
        "created_utc": args.created_utc,
        "scope": (
            "Dense Qwen retrofit experiments and matching CPU runtime artifacts. "
            "Results do not establish universal model or MoE support."
        ),
        "verdicts": {
            "blind_absmean_ptq": {
                "status": "rejected_in_tested_setup",
                "model": "Qwen2.5-1.5B",
                "fp_wikitext_ppl": blind["fp_wikitext_ppl"],
                "ptq_wikitext_ppl": blind["ptq_wikitext_ppl"],
                "fp_ten_task_mean": blind["fp_ten_task_mean"],
                "ptq_ten_task_mean": blind["ptq_ten_task_mean"],
                "boundary": blind["caveat"],
            },
            "qat_distillation": {
                "status": "partial_recovery_not_fp_quality",
                "best_row_scale_ten_task_mean": qat["best_row_scale_qat_ten_task_mean"],
                "gain_vs_naive_ptq": qat["recovery_vs_ptq"],
                "gap_vs_fp": qat["gap_vs_fp"],
                "boundary": qat["caveat"],
            },
            "matched_bitdistill_mnli": {
                "status": "not_reproduced",
                "model": "Qwen2.5-0.5B",
                "examples": predictions["examples"],
                "training_seeds": 3,
                "fp16_accuracy": fp16_accuracy,
                "adaptive_mean_accuracy": adaptive_mean,
                "adaptive_seed_mean_ci95": aggregate["adaptive_seed_mean_t_ci95"],
                "adaptive_gap_vs_fp16": adaptive_mean - fp16_accuracy,
                "fixed60_mean_accuracy": fixed_mean,
                "fixed60_seed_mean_ci95": aggregate["fixed60_seed_mean_t_ci95"],
                "fixed60_gap_vs_fp16": fixed_mean - fp16_accuracy,
                "adaptive_minus_fixed60": aggregate["mean_adaptive_minus_fixed60"],
                "adaptive_minus_fixed60_seed_ci95": aggregate["seed_level_paired_t_ci95"],
                "decision": matched["decisions"]["recommended_method"],
                "paper_recovery_floor": matched["preregistration"]["paper_recovery_floor"],
                "boundary": matched["claim_boundary"],
            },
            "least_squares_initializer": {
                "status": "rejected_by_task_quality",
                "absmean_mnli_accuracy": initializer["task_quality_audits"]["ls"][
                    "baseline_accuracy"
                ],
                "ls_mnli_accuracy": initializer["task_quality_audits"]["ls"][
                    "candidate_accuracy"
                ],
                "diag_ls_mnli_accuracy": initializer["task_quality_audits"]["diag_ls"][
                    "candidate_accuracy"
                ],
                "boundary": (
                    "Lower synthetic layer-output error did not transfer to MNLI quality; "
                    "this does not test modern end-to-end asymmetric PTQ methods."
                ),
            },
            "row_scale_runtime_contract": {
                "status": "supported",
                "one_scale_relative_rms_error": scale_contract[
                    "one_scale_tl2_relative_rms_error"
                ],
                "exact_row_scale_relative_rms_error": scale_contract[
                    "exact_fp16_row_scale_relative_rms_error"
                ],
            },
            "causal_i2sr_cpu": {
                "status": "working_not_q4_quality_or_size_competitive",
                "fp16": causal_cpu["fp_f16"],
                "q4_k_m": causal_cpu["q4_k_m"],
                "i2_sr": causal_cpu["row_i2sr"],
                "boundary": causal_cpu["caveat"],
            },
            "sequence_classifier_cpu": {
                "status": "smaller_but_slower_than_fp16",
                "hardware": cpu_repeated["hardware"]["cpu_model"],
                "threads": cpu_repeated["hardware"]["requested_threads"],
                "i2_sr_speed_ratio_vs_fp16": cpu_repeated[
                    "paired_speed_ratios_vs_fp16"
                ]["i2_sr_student"],
                "mixed_i2_sr_q8_speed_ratio_vs_fp16": cpu_repeated[
                    "paired_speed_ratios_vs_fp16"
                ]["i2_sr_q8_embedding_student"],
                "mixed_file_mib": mixed_artifact["gguf_mib"],
                "mixed_size_ratio_fp16_over_candidate": cpu_matrix["comparisons"][
                    "i2_sr_q8_embedding_student_vs_fp16_teacher"
                ]["system"]["size_ratio_reference_over_candidate"],
                "mixed_vs_i2sr_accuracy_delta": mixed_comparison["quality"][
                    "delta_candidate_minus_reference"
                ],
                "boundary": cpu_matrix["claim_boundary"],
            },
            "tl2_sr": {
                "status": "fidelity_and_storage_supported_speed_rejected",
                "kernel_cases": [tl2["kernel_cases_passed"], tl2["kernel_cases_total"]],
                "full_validation": tl2["full_validation"],
                "projection_storage_reduction": tl2["storage"][
                    "projection_reduction_fraction"
                ],
                "whole_file_reduction": tl2["storage"]["file_reduction_fraction"],
                "speed_ratios_vs_i2sr": [
                    {
                        "tile_bm": row["tile_bm"],
                        "ratio": row["paired_speed_ratio_vs_i2sr"],
                        "ci95": row["paired_speed_ratio_ci95_t"],
                    }
                    for row in tl2["tiling_sweep"]
                ],
                "boundary": tl2["verdict"],
            },
            "moe_kimi": {
                "status": "not_proven",
                "boundary": "Only tiny Qwen2MoE plumbing exists; no Kimi checkpoint, quality, or routed CPU benchmark exists.",
            },
        },
        "sources": {
            "canonical": source(args.canonical),
            "matched_audit": source(args.matched_audit),
            "prediction_bundle": source(args.prediction_bundle),
            "tl2_audit": source(args.tl2_audit),
            "cpu_matrix": source(args.cpu_matrix),
            "cpu_repeated": source(args.cpu_repeated),
            "initializer_audit": source(args.initializer_audit),
        },
    }
    serialized = json.dumps(snapshot, sort_keys=True)
    for prefix in ("/mnt/", "/local/"):
        if prefix in serialized:
            raise ValueError(f"public snapshot contains private path prefix {prefix!r}")
    return snapshot


def render_markdown(snapshot: dict[str, Any]) -> str:
    verdicts = snapshot["verdicts"]
    ptq = verdicts["blind_absmean_ptq"]
    matched = verdicts["matched_bitdistill_mnli"]
    scale = verdicts["row_scale_runtime_contract"]
    classifier = verdicts["sequence_classifier_cpu"]
    tl2 = verdicts["tl2_sr"]
    rows = [
        (
            "Blind absmean PTQ",
            "Rejected",
            f"WikiText PPL {ptq['fp_wikitext_ppl']:.3f} -> {ptq['ptq_wikitext_ppl']:,.3f}",
        ),
        (
            "Matched BitDistill MNLI",
            "Not reproduced",
            f"FP16 {matched['fp16_accuracy']:.6f}; best arm mean {max(matched['adaptive_mean_accuracy'], matched['fixed60_mean_accuracy']):.6f}",
        ),
        (
            "Adaptive vs fixed-60",
            "Inconclusive",
            f"delta {matched['adaptive_minus_fixed60']:+.6f}; seed 95% CI [{matched['adaptive_minus_fixed60_seed_ci95'][0]:+.6f}, {matched['adaptive_minus_fixed60_seed_ci95'][1]:+.6f}]",
        ),
        (
            "Row-scale runtime contract",
            "Supported",
            f"relative RMS error {scale['one_scale_relative_rms_error']:.6f} -> {scale['exact_row_scale_relative_rms_error']:.6f}",
        ),
        (
            "Packed classifier speed",
            "Rejected on Xeon 4116",
            f"I2_SR/FP16 {classifier['i2_sr_speed_ratio_vs_fp16']['geometric_mean']:.3f}x",
        ),
        (
            "TL2_SR",
            "Fidelity yes; speed no",
            f"projection bytes -{100.0 * tl2['projection_storage_reduction']:.3f}%; best tested speed {max(row['ratio'] for row in tl2['speed_ratios_vs_i2sr']):.3f}x I2_SR",
        ),
        ("Kimi/MoE", "Not proven", verdicts["moe_kimi"]["boundary"]),
    ]
    table = [f"| {area} | **{status}** | {evidence} |" for area, status, evidence in rows]
    return "\n".join(
        [
            "# Current Evidence Snapshot",
            "",
            f"Generated: `{snapshot['created_utc']}`",
            "",
            snapshot["scope"],
            "",
            "| Question | Verdict | Headline evidence |",
            "| --- | --- | --- |",
            *table,
            "",
            "All source reports and SHA-256 hashes are recorded in the adjacent JSON file.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical",
        type=Path,
        default=Path("benchmarks/results/canonical_evidence_bundle_2026-05-20.json"),
    )
    parser.add_argument(
        "--matched-audit",
        type=Path,
        default=Path(
            "benchmarks/results/bitdistill_adaptive_vs_fixed_matched_audit_2026-09-22.json"
        ),
    )
    parser.add_argument(
        "--prediction-bundle",
        type=Path,
        default=Path("benchmarks/results/bitdistill_matched_prediction_bundle_2026-09-22.json"),
    )
    parser.add_argument(
        "--tl2-audit",
        type=Path,
        default=Path("benchmarks/results/tl2sr_evidence_audit_2026-09-04.json"),
    )
    parser.add_argument(
        "--cpu-matrix",
        type=Path,
        default=Path("benchmarks/results/seqcls_native_cpu_matrix_2026-09-04.json"),
    )
    parser.add_argument(
        "--cpu-repeated",
        type=Path,
        default=Path("benchmarks/results/seqcls_native_cpu_repeated_inplace_2026-09-04.json"),
    )
    parser.add_argument(
        "--initializer-audit",
        type=Path,
        default=Path("benchmark_results/second_order_ternary_init_2026-05-15.json"),
    )
    parser.add_argument(
        "--created-utc",
        default=datetime.now(timezone.utc).isoformat(),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmarks/results/current_evidence_2026-09-22.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("benchmarks/results/current_evidence_2026-09-22.md"),
    )
    args = parser.parse_args()
    snapshot = build_snapshot(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(snapshot), encoding="utf-8")


if __name__ == "__main__":
    main()
