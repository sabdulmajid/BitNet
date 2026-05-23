#!/usr/bin/env python3
"""Validate public docs against the canonical evidence bundle.

This is intentionally conservative: it checks that the headline README,
CLAIMS, and runtime-contract numbers are still backed by the canonical JSON
bundle and that every artifact referenced by the bundle exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def fmt_ppl(value: float) -> str:
    if abs(value) >= 10000:
        return f"{value:,.3f}"
    return f"{value:.3f}"


def require_contains(label: str, needle: str, haystack: str, errors: list[str]) -> None:
    if needle not in haystack:
        errors.append(f"{label}: missing `{needle}`")


def validate_artifacts(bundle: dict[str, Any], errors: list[str]) -> None:
    artifacts = bundle.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append("canonical bundle missing artifacts object")
        return
    for label, artifact in artifacts.items():
        if not isinstance(artifact, dict):
            errors.append(f"artifact {label}: not an object")
            continue
        path = artifact.get("path")
        if not isinstance(path, str) or not path:
            errors.append(f"artifact {label}: missing path")
            continue
        if not Path(path).exists():
            errors.append(f"artifact {label}: path does not exist: {path}")
            continue
        expected_sha = artifact.get("sha256")
        if isinstance(expected_sha, str) and expected_sha:
            actual_sha = sha256(Path(path))
            if actual_sha != expected_sha:
                errors.append(f"artifact {label}: sha256 mismatch: {actual_sha} != {expected_sha}")


def validate_reproduction_gap(report: dict[str, Any], errors: list[str]) -> None:
    if report.get("schema") != "bitnet-reproduction-gap-report-v1":
        errors.append(f"reproduction gap: unexpected schema {report.get('schema')}")
    if report.get("status") != "not_reproduced":
        errors.append(f"reproduction gap: unexpected status {report.get('status')}")
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append("reproduction gap: missing artifacts object")
        return
    for label, artifact in artifacts.items():
        if not isinstance(artifact, dict):
            errors.append(f"reproduction gap artifact {label}: not an object")
            continue
        path = artifact.get("path")
        if not isinstance(path, str) or not path:
            errors.append(f"reproduction gap artifact {label}: missing path")
            continue
        if not Path(path).exists():
            errors.append(f"reproduction gap artifact {label}: path does not exist: {path}")
            continue
        expected_sha = artifact.get("sha256")
        if isinstance(expected_sha, str) and expected_sha:
            actual_sha = sha256(Path(path))
            if actual_sha != expected_sha:
                errors.append(
                    f"reproduction gap artifact {label}: sha256 mismatch: {actual_sha} != {expected_sha}"
                )


def validate_stage2_extension_submission(report: dict[str, Any], errors: list[str]) -> None:
    if report.get("schema") != "bitnet-stage2-extension-submission-v1":
        errors.append(f"stage2 extension: unexpected schema {report.get('schema')}")
    if report.get("status") not in {"pending", "running", "complete", "failed", "cancelled"}:
        errors.append(f"stage2 extension: unexpected status {report.get('status')}")
    parent = report.get("parent_manifest", {})
    if not isinstance(parent, dict):
        errors.append("stage2 extension: missing parent_manifest object")
        return
    parent_path = parent.get("path")
    if not isinstance(parent_path, str) or not parent_path:
        errors.append("stage2 extension: missing parent manifest path")
    elif not Path(parent_path).exists():
        errors.append(f"stage2 extension: parent manifest does not exist: {parent_path}")
    state_dict = parent.get("state_dict_path")
    if not isinstance(state_dict, str) or not state_dict:
        errors.append("stage2 extension: missing parent state_dict_path")
    elif not Path(state_dict).exists():
        errors.append(f"stage2 extension: parent state_dict_path does not exist: {state_dict}")
    config = report.get("run_config", {})
    if not isinstance(config, dict):
        errors.append("stage2 extension: missing run_config object")
        return
    segment = config.get("segment_token_presentations")
    cumulative = config.get("cumulative_token_presentations")
    parent_tokens = parent.get("token_presentations")
    if isinstance(segment, int) and isinstance(cumulative, int) and isinstance(parent_tokens, int):
        if cumulative != parent_tokens + segment:
            errors.append(
                "stage2 extension: cumulative tokens do not equal parent + segment: "
                f"{cumulative} != {parent_tokens} + {segment}"
            )
    else:
        errors.append("stage2 extension: token presentation fields must be integers")


def validate_stage2_handoff_submission(report: dict[str, Any], errors: list[str]) -> None:
    if report.get("schema") != "bitnet-stage2-extension-handoff-submission-v1":
        errors.append(f"stage2 handoff: unexpected schema {report.get('schema')}")
    if report.get("status") not in {"dependency_pending", "running", "submitted_downstream", "failed"}:
        errors.append(f"stage2 handoff: unexpected status {report.get('status')}")
    if report.get("dependency") != "afterok:10250":
        errors.append(f"stage2 handoff: unexpected dependency {report.get('dependency')}")
    script = report.get("script")
    if not isinstance(script, str) or not script:
        errors.append("stage2 handoff: missing script")
    elif not Path(script).exists():
        errors.append(f"stage2 handoff: script does not exist: {script}")


def validate_gradient_telemetry_submission(report: dict[str, Any], errors: list[str]) -> None:
    if report.get("schema") != "bitdistill-gradient-telemetry-submission-v1":
        errors.append(f"gamma telemetry: unexpected schema {report.get('schema')}")
    if report.get("status") not in {"dependency_pending", "running", "complete", "failed"}:
        errors.append(f"gamma telemetry: unexpected status {report.get('status')}")
    if report.get("dependency") != "afterok:10250":
        errors.append(f"gamma telemetry: unexpected dependency {report.get('dependency')}")
    config = report.get("run_config", {})
    if not isinstance(config, dict):
        errors.append("gamma telemetry: missing run_config object")
        return
    if config.get("attention_kd_weight") != 60:
        errors.append(f"gamma telemetry: expected attention_kd_weight 60, got {config.get('attention_kd_weight')}")
    if config.get("max_steps") != 200:
        errors.append(f"gamma telemetry: expected max_steps 200, got {config.get('max_steps')}")
    if config.get("telemetry_component_grad_norms") is not True:
        errors.append("gamma telemetry: component gradient telemetry is not enabled")
    output_dir = config.get("output_dir")
    if not isinstance(output_dir, str) or "telemetry-gamma60" not in output_dir:
        errors.append(f"gamma telemetry: unexpected output_dir {output_dir}")
    target = report.get("comparison_target", {})
    if not isinstance(target, dict):
        errors.append("gamma telemetry: missing comparison_target object")
    else:
        existing_report = target.get("existing_report")
        if not isinstance(existing_report, str) or not Path(existing_report).exists():
            errors.append(f"gamma telemetry: comparison report does not exist: {existing_report}")
    caveat = report.get("caveat")
    if not isinstance(caveat, str) or "not a quality benchmark" not in caveat:
        errors.append("gamma telemetry: missing non-quality-benchmark caveat")


def validate_active_stage2_monitor(report: dict[str, Any], errors: list[str]) -> None:
    if report.get("schema") != "bitnet-active-stage2-extension-monitor-v1":
        errors.append(f"stage2 monitor: unexpected schema {report.get('schema')}")
    if report.get("quality_claim") != "none":
        errors.append(f"stage2 monitor: quality_claim must be none, got {report.get('quality_claim')}")
    stage2 = report.get("stage2", {})
    if not isinstance(stage2, dict):
        errors.append("stage2 monitor: missing stage2 object")
        return
    if stage2.get("job_id") != "10250":
        errors.append(f"stage2 monitor: unexpected stage2 job {stage2.get('job_id')}")
    if stage2.get("cumulative_token_presentations") != 655360000:
        errors.append(
            "stage2 monitor: unexpected cumulative tokens "
            f"{stage2.get('cumulative_token_presentations')}"
        )


def validate_readme(bundle: dict[str, Any], readme: str, errors: list[str]) -> None:
    claims = bundle["claims"]
    blind = claims["blind_ptq"]
    qat = claims["qat_distill"]
    bitdistill = claims["bitdistill_reproduction"]
    runtime = claims["row_scale_runtime_contract"]
    i2sr = claims["i2sr_cpu"]
    native = claims["native_classifier"]

    required = {
        "title": "Ternary Retrofit Evaluator and CPU Runtime-Contract Tester",
        "framing": "Extreme ternary quantization is not a file-format conversion problem",
        "not_converter": "not a universal BitNet converter",
        "blind_status": "strong negative result",
        "fp_ppl": fmt_ppl(float(blind["fp_wikitext_ppl"])),
        "ptq_ppl": fmt_ppl(float(blind["ptq_wikitext_ppl"])),
        "fp_mean": fmt(float(blind["fp_ten_task_mean"])),
        "ptq_mean": fmt(float(blind["ptq_ten_task_mean"])),
        "qat_mean": fmt(float(qat["best_row_scale_qat_ten_task_mean"])),
        "qat_recovery": f"{float(qat['recovery_vs_ptq']):+.6f}",
        "qat_gap": f"{float(qat['gap_vs_fp']):+.6f}",
        "bitdistill_status": "No",
        "mnli_40m": fmt(float(bitdistill["controlled_40_96m_mnli"])),
        "mnli_163m": fmt(float(bitdistill["controlled_163_84m_mnli"])),
        "mnli_327m": fmt(float(bitdistill["controlled_327_68m_mnli"])),
        "mnli_delta": f"{float(bitdistill['controlled_327_68m_delta_vs_fp']):+.6f}",
        "stage2_tokens": "327,680,000",
        "stage2_final_ce": "3.784057",
        "tl2_one_scale": fmt(float(runtime["one_scale_tl2_relative_rms_error"])),
        "row_scale": fmt(float(runtime["exact_fp16_row_scale_relative_rms_error"])),
        "i2sr_file": f"{float(i2sr['row_i2sr']['file_mib']):.1f}",
        "i2sr_ppl": f"{float(i2sr['row_i2sr']['ppl']):.4f}",
        "i2sr_prompt": f"{float(i2sr['row_i2sr']['prompt_tok_s']):.2f}",
        "i2sr_decode": f"{float(i2sr['row_i2sr']['decode_tok_s']):.2f}",
        "native_mnli": fmt(float(native["mnli_accuracy"])),
        "native_agreement": fmt(float(native["pytorch_agreement"])),
        "moe_not_supported": "not supported",
    }
    for label, needle in required.items():
        require_contains(f"README {label}", needle, readme, errors)


def validate_reproduction_gap_docs(
    report: dict[str, Any],
    readme: str,
    claims_doc: str,
    errors: list[str],
) -> None:
    metrics = report["metrics"]
    required = {
        "gap_report_md": "bitdistill_reproduction_gap_2026-05-23.md",
        "gap_report_json": "bitdistill_reproduction_gap_2026-05-23.json",
        "bitnet_default": fmt(float(metrics["bitnet_sft_default_mnli"])),
        "bitnet_best": fmt(float(metrics["bitnet_sft_best_mnli"])),
        "bitnet_vs_paper": f"{float(metrics['bitnet_sft_best_delta_vs_paper_anchor']):+.6f}",
        "bitdistill_327": fmt(float(metrics["bitdistill_327_68m_mnli"])),
        "bitdistill_vs_fp": f"{float(metrics['bitdistill_327_68m_delta_vs_fp16']):+.6f}",
    }
    for label, needle in required.items():
        require_contains(f"README reproduction gap {label}", needle, readme, errors)
        require_contains(f"CLAIMS reproduction gap {label}", needle, claims_doc, errors)
    require_contains(
        "README stage2 extension report",
        "stage2_655m_submission_2026-05-23.md",
        readme,
        errors,
    )
    require_contains("README stage2 extension job", "10250", readme, errors)
    require_contains("README stage2 extension tokens", "655.36M", readme, errors)
    require_contains("README stage2 handoff job", "10253", readme, errors)
    require_contains("README stage2 handoff dependency", "afterok:10250", readme, errors)
    require_contains(
        "README gamma telemetry report",
        "gamma60_telemetry_submission_2026-05-23.md",
        readme,
        errors,
    )
    require_contains("README gamma telemetry job", "10254", readme, errors)
    require_contains("README gamma telemetry caveat", "not a quality benchmark", readme, errors)


def validate_claims_doc(bundle: dict[str, Any], claims_doc: str, errors: list[str]) -> None:
    claims = bundle["claims"]
    required = {
        "ptq_rejected": "Rejected",
        "partial": "Supported, partial",
        "bitdistill_not_yet": "Not yet",
        "row_supported": "Supported",
        "i2sr_caveat": "does not beat Q4_K_M",
        "native_not_ready": "Not yet",
        "moe_not_supported": "Not supported",
        "mnli_327": fmt(float(claims["bitdistill_reproduction"]["controlled_327_68m_mnli"])),
        "delta_327": f"{float(claims['bitdistill_reproduction']['controlled_327_68m_delta_vs_fp']):+.6f}",
    }
    for label, needle in required.items():
        require_contains(f"CLAIMS {label}", needle, claims_doc, errors)


def validate_runtime_doc(bundle: dict[str, Any], runtime_doc: str, errors: list[str]) -> None:
    runtime = bundle["claims"]["row_scale_runtime_contract"]
    i2sr = bundle["claims"]["i2sr_cpu"]
    required = {
        "one_scale": fmt(float(runtime["one_scale_tl2_relative_rms_error"])),
        "row_scale": fmt(float(runtime["exact_fp16_row_scale_relative_rms_error"])),
        "fp_file": f"{float(i2sr['fp_f16']['file_mib']):.1f}",
        "fp_ppl": f"{float(i2sr['fp_f16']['ppl']):.4f}",
        "q4_file": f"{float(i2sr['q4_k_m']['file_mib']):.1f}",
        "q4_ppl": f"{float(i2sr['q4_k_m']['ppl']):.4f}",
        "i2sr_file": f"{float(i2sr['row_i2sr']['file_mib']):.1f}",
        "i2sr_ppl": f"{float(i2sr['row_i2sr']['ppl']):.4f}",
        "not_q4": "not quality/storage competitive with Q4_K_M",
    }
    for label, needle in required.items():
        require_contains(f"RUNTIME_CONTRACT {label}", needle, runtime_doc, errors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("benchmarks/results/canonical_evidence_bundle_2026-05-20.json"),
    )
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--claims", type=Path, default=Path("CLAIMS.md"))
    parser.add_argument("--runtime-contract", type=Path, default=Path("RUNTIME_CONTRACT.md"))
    parser.add_argument(
        "--reproduction-gap",
        type=Path,
        default=Path("benchmarks/results/bitdistill_reproduction_gap_2026-05-23.json"),
    )
    parser.add_argument(
        "--stage2-extension",
        type=Path,
        default=Path("benchmarks/results/stage2_655m_submission_2026-05-23.json"),
    )
    parser.add_argument(
        "--stage2-handoff",
        type=Path,
        default=Path("benchmarks/results/stage2_655m_handoff_submission_2026-05-23.json"),
    )
    parser.add_argument(
        "--gamma-telemetry-submission",
        type=Path,
        default=Path("benchmarks/results/gamma60_telemetry_submission_2026-05-23.json"),
    )
    parser.add_argument(
        "--stage2-monitor",
        type=Path,
        default=Path("benchmarks/results/active_stage2_extension_monitor_2026-05-23.json"),
    )
    args = parser.parse_args()

    bundle = load_json(args.bundle)
    reproduction_gap = load_json(args.reproduction_gap)
    stage2_extension = load_json(args.stage2_extension)
    stage2_handoff = load_json(args.stage2_handoff)
    gamma_telemetry = load_json(args.gamma_telemetry_submission)
    stage2_monitor = load_json(args.stage2_monitor)
    readme = read_text(args.readme)
    claims_doc = read_text(args.claims)
    errors: list[str] = []
    validate_artifacts(bundle, errors)
    validate_reproduction_gap(reproduction_gap, errors)
    validate_stage2_extension_submission(stage2_extension, errors)
    validate_stage2_handoff_submission(stage2_handoff, errors)
    validate_gradient_telemetry_submission(gamma_telemetry, errors)
    validate_active_stage2_monitor(stage2_monitor, errors)
    validate_readme(bundle, readme, errors)
    validate_reproduction_gap_docs(reproduction_gap, readme, claims_doc, errors)
    validate_claims_doc(bundle, claims_doc, errors)
    validate_runtime_doc(bundle, read_text(args.runtime_contract), errors)

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"validated public docs against {args.bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
