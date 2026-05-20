#!/usr/bin/env python3
"""Validate public docs against the canonical evidence bundle.

This is intentionally conservative: it checks that the headline README,
CLAIMS, and runtime-contract numbers are still backed by the canonical JSON
bundle and that every artifact referenced by the bundle exists.
"""

from __future__ import annotations

import argparse
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
    args = parser.parse_args()

    bundle = load_json(args.bundle)
    errors: list[str] = []
    validate_artifacts(bundle, errors)
    validate_readme(bundle, read_text(args.readme), errors)
    validate_claims_doc(bundle, read_text(args.claims), errors)
    validate_runtime_doc(bundle, read_text(args.runtime_contract), errors)

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"validated public docs against {args.bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
