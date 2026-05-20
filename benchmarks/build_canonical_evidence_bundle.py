#!/usr/bin/env python3
"""Build the small, claim-oriented evidence bundle used by the README."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LM_EVAL_METRICS = {
    "arc_challenge": "acc_norm",
    "arc_easy": "acc_norm",
    "hellaswag": "acc_norm",
    "piqa": "acc_norm",
    "winogrande": "acc",
    "boolq": "acc",
    "copa": "acc",
    "openbookqa": "acc_norm",
    "sciq": "acc_norm",
    "truthfulqa_mc1": "acc",
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lm_eval_mean(path: Path) -> float:
    results = read_json(path)["results"]
    values: list[float] = []
    for task, metric in LM_EVAL_METRICS.items():
        row = results[task]
        value = row.get(metric, row.get(f"{metric},none"))
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{path}: missing finite {task}/{metric}")
        values.append(float(value))
    return sum(values) / len(values)


def cpu_row(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        if row.get("label") == label:
            return row
    raise KeyError(label)


def build(args: argparse.Namespace) -> dict[str, Any]:
    artifacts = {
        "fp_ppl": Path("benchmark_results/quality-9735/qwen15b_fp_wikitext.json"),
        "ptq_ppl": Path("benchmark_results/quality-ptq-qwen15b/qwen15b_naive_ptq_wikitext.json"),
        "fp_lm_eval": Path("benchmark_results/lm-eval-qwen15b-full10/qwen15b_fp.json"),
        "ptq_lm_eval": Path("benchmark_results/lm-eval-qwen15b-full10/qwen15b_naive_ptq.json"),
        "row_qat_lm_eval": Path("benchmark_results/lm-eval-qwen15b-klonly-row-notiehead-full10/qwen15b_qat_ternary.json"),
        "stage2_curve": Path("benchmark_results/bitdistill_stage2_curve_2026-05-16.json"),
        "controlled_curve": Path("benchmarks/results/bitdistill_controlled_curve_2026-05-20.json"),
        "gamma60": Path("benchmark_results/bitdistill_gamma60_diagnostic_2026-05-15.json"),
        "tl2_contract": Path("benchmark_results/tl2_row_scale_runtime_contract_2026-05-15.json"),
        "cpu_frontier": Path("benchmark_results/cpu_tradeoff_frontier_2026-05-15.json"),
        "native_seqcls": Path("benchmark_results/seqcls_native_i2sr_cpu_mnli_full_token_ids_sequence_isolated_2026-05-15.json"),
        "stage2_manifest": args.stage2_manifest,
    }
    loaded = {key: read_json(path) for key, path in artifacts.items()}
    fp_mean = lm_eval_mean(artifacts["fp_lm_eval"])
    ptq_mean = lm_eval_mean(artifacts["ptq_lm_eval"])
    row_qat_mean = lm_eval_mean(artifacts["row_qat_lm_eval"])
    cpu_rows = loaded["cpu_frontier"]["rows"]
    fp_cpu = cpu_row("FP F16", cpu_rows)
    q4_cpu = cpu_row("FP Q4_K_M", cpu_rows)
    i2sr_cpu = cpu_row("row I2_SR", cpu_rows)
    controlled_rows = loaded["controlled_curve"]["rows"]
    controlled_by_tokens = {
        int(row["stage2_token_presentations"]): row for row in controlled_rows if row.get("metric_accuracy") is not None
    }
    stage2_manifest = loaded["stage2_manifest"]
    bundle = {
        "schema": "bitnet-canonical-evidence-bundle-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            key: {"path": str(path), "sha256": sha256(path)} for key, path in artifacts.items()
        },
        "claims": {
            "blind_ptq": {
                "status": "strong_negative_tested_setup",
                "fp_wikitext_ppl": loaded["fp_ppl"]["perplexity"],
                "ptq_wikitext_ppl": loaded["ptq_ppl"]["perplexity"],
                "fp_ten_task_mean": fp_mean,
                "ptq_ten_task_mean": ptq_mean,
                "caveat": "Dense Qwen2.5-1.5B tested setup; not a theorem for every architecture.",
            },
            "qat_distill": {
                "status": "partial_recovery_not_fp",
                "best_row_scale_qat_ten_task_mean": row_qat_mean,
                "recovery_vs_ptq": row_qat_mean - ptq_mean,
                "gap_vs_fp": row_qat_mean - fp_mean,
                "caveat": "Row-scale QAT is a retrofit variant, not standard BitDistill.",
            },
            "bitdistill_reproduction": {
                "status": "not_reproduced_327m_pending",
                "fp16_sft_mnli": loaded["stage2_curve"]["fp16_accuracy"],
                "controlled_40_96m_mnli": controlled_by_tokens[40_960_000]["metric_accuracy"],
                "controlled_163_84m_mnli": controlled_by_tokens[163_840_000]["metric_accuracy"],
                "controlled_327_68m_stage2_tokens": stage2_manifest["token_presentations"],
                "controlled_327_68m_stage2_final_ce": stage2_manifest["final_ce"],
                "controlled_327_68m_downstream_status": stage2_manifest["downstream"]["status"],
                "state_dict_path": stage2_manifest["state_dict_path"],
                "caveat": "The 327.68M Stage-2 producer finished, but downstream MNLI must be rerun with the snapshot state_dict_path.",
            },
            "gamma_normalization": {
                "status": "local_loss_normalization_mismatch",
                "gamma60_mnli": loaded["gamma60"]["candidate_accuracy"],
                "gamma60_delta_vs_fp": loaded["gamma60"]["fp_comparison"]["delta_vs_reference"],
                "caveat": "This is a local normalization diagnostic, not a claim that the paper coefficient is wrong.",
            },
            "row_scale_runtime_contract": {
                "status": "strong_systems_result",
                "one_scale_tl2_relative_rms_error": loaded["tl2_contract"]["math"]["current_tl2_tensor_max_error"],
                "exact_fp16_row_scale_relative_rms_error": loaded["tl2_contract"]["math"]["row_fp16_error"],
                "caveat": "This supports I2_SR/row-scale contracts; TL2 row-scale support is not implemented.",
            },
            "i2sr_cpu": {
                "status": "working_not_q4_quality_competitive",
                "fp_f16": {
                    "file_mib": fp_cpu["file_mib"],
                    "ppl": fp_cpu["ppl"],
                    "prompt_tok_s": fp_cpu["prefill_tok_s"],
                    "decode_tok_s": fp_cpu["decode_tok_s"],
                },
                "q4_k_m": {
                    "file_mib": q4_cpu["file_mib"],
                    "ppl": q4_cpu["ppl"],
                    "prompt_tok_s": q4_cpu["prefill_tok_s"],
                    "decode_tok_s": q4_cpu["decode_tok_s"],
                },
                "row_i2sr": {
                    "file_mib": i2sr_cpu["file_mib"],
                    "ppl": i2sr_cpu["ppl"],
                    "prompt_tok_s": i2sr_cpu["prefill_tok_s"],
                    "decode_tok_s": i2sr_cpu["decode_tok_s"],
                },
                "caveat": loaded["cpu_frontier"]["interpretation"],
            },
            "native_classifier": {
                "status": "research_demo_not_product_ready",
                "mnli_accuracy": loaded["native_seqcls"]["summary"]["accuracy"],
                "pytorch_agreement": loaded["native_seqcls"]["summary"]["agreement_with_saved_pytorch_predictions"],
                "examples_per_second": loaded["native_seqcls"]["runtime"]["examples_per_second"],
                "rss_mib": loaded["native_seqcls"]["runtime"]["child_peak_rss_mib"],
                "caveat": "Agreement remains below the 0.99 product gate.",
            },
            "moe_kimi": {
                "status": "not_supported",
                "caveat": "Only tiny Qwen2MoE fixture/plumbing exists; no Kimi quality or routed CPU runtime is proven.",
            },
        },
    }
    return bundle


def fmt(value: Any) -> str:
    if isinstance(value, float):
        if abs(value) >= 10000:
            return f"{value:,.6f}"
        return f"{value:.6f}"
    return str(value)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def render_markdown(bundle: dict[str, Any]) -> str:
    claims = bundle["claims"]
    rows = [
        [
            "Blind PTQ",
            claims["blind_ptq"]["status"],
            f"FP PPL {claims['blind_ptq']['fp_wikitext_ppl']:.3f}; PTQ PPL {claims['blind_ptq']['ptq_wikitext_ppl']:,.3f}; FP mean {claims['blind_ptq']['fp_ten_task_mean']:.6f}; PTQ mean {claims['blind_ptq']['ptq_ten_task_mean']:.6f}",
            claims["blind_ptq"]["caveat"],
        ],
        [
            "QAT/distill",
            claims["qat_distill"]["status"],
            f"row-scale QAT mean {claims['qat_distill']['best_row_scale_qat_ten_task_mean']:.6f}; recovery {claims['qat_distill']['recovery_vs_ptq']:+.6f}; gap {claims['qat_distill']['gap_vs_fp']:+.6f}",
            claims["qat_distill"]["caveat"],
        ],
        [
            "BitDistill",
            claims["bitdistill_reproduction"]["status"],
            f"MNLI 40.96M {claims['bitdistill_reproduction']['controlled_40_96m_mnli']:.6f}; 163.84M {claims['bitdistill_reproduction']['controlled_163_84m_mnli']:.6f}; 327.68M downstream {claims['bitdistill_reproduction']['controlled_327_68m_downstream_status']}",
            claims["bitdistill_reproduction"]["caveat"],
        ],
        [
            "Row-scale runtime",
            claims["row_scale_runtime_contract"]["status"],
            f"TL2 one-scale RMS {claims['row_scale_runtime_contract']['one_scale_tl2_relative_rms_error']:.6f}; exact row-scale RMS {claims['row_scale_runtime_contract']['exact_fp16_row_scale_relative_rms_error']:.6f}",
            claims["row_scale_runtime_contract"]["caveat"],
        ],
        [
            "I2_SR CPU",
            claims["i2sr_cpu"]["status"],
            f"I2_SR PPL {claims['i2sr_cpu']['row_i2sr']['ppl']:.4f}, prompt {claims['i2sr_cpu']['row_i2sr']['prompt_tok_s']:.2f}, decode {claims['i2sr_cpu']['row_i2sr']['decode_tok_s']:.2f}",
            "Does not beat Q4_K_M on quality or file size.",
        ],
        [
            "Native classifier",
            claims["native_classifier"]["status"],
            f"MNLI {claims['native_classifier']['mnli_accuracy']:.6f}; agreement {claims['native_classifier']['pytorch_agreement']:.6f}; {claims['native_classifier']['examples_per_second']:.6f} ex/s",
            claims["native_classifier"]["caveat"],
        ],
        [
            "MoE/Kimi",
            claims["moe_kimi"]["status"],
            "No trained Kimi/MoE quality or CPU runtime evidence.",
            claims["moe_kimi"]["caveat"],
        ],
    ]
    return "\n\n".join(
        [
            "# Canonical Evidence Bundle",
            "This bundle is manifest/artifact based. Missing artifacts are fatal while building it.",
            md_table(["claim", "status", "evidence", "caveat"], rows),
            "## Artifact Inventory",
            md_table(
                ["label", "path", "sha256"],
                [[key, value["path"], value["sha256"]] for key, value in sorted(bundle["artifacts"].items())],
            ),
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-manifest", type=Path, default=Path("benchmarks/results/stage2_manifest_2026-05-20.json"))
    parser.add_argument("--output-json", type=Path, default=Path("benchmarks/results/canonical_evidence_bundle_2026-05-20.json"))
    parser.add_argument("--output-md", type=Path, default=Path("benchmarks/results/canonical_evidence_bundle_2026-05-20.md"))
    args = parser.parse_args()

    bundle = build(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(bundle).rstrip() + "\n", encoding="utf-8")
    print(render_markdown(bundle))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
