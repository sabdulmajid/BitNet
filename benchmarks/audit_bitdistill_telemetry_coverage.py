#!/usr/bin/env python3
"""Audit whether BitDistill runs expose enough telemetry for reproduction claims.

This is intentionally non-invasive: it reads existing source, metrics, and
reports.  It does not decide whether BitDistill works; it decides which
diagnostics are currently measured well enough to support a root-cause claim.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value != 0.0 and (abs(value) >= 1000 or abs(value) < 0.0001):
            return f"{value:.3e}"
        return f"{value:.6f}"
    if isinstance(value, list):
        return ", ".join(fmt(item) for item in value)
    return str(value)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(cell).replace("|", "\\|") for cell in row) + " |")
    return "\n".join(lines)


def source_has_all(source: str, snippets: list[str]) -> bool:
    return all(snippet in source for snippet in snippets)


def nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def summarize_measured(args: argparse.Namespace, source: str, launcher_source: str) -> list[dict[str, Any]]:
    loss_scale = read_json(args.loss_scale_json)
    mechanics = read_json(args.mechanics_json)
    subln = read_json(args.subln_json)
    stage2 = read_json(args.stage2_json)

    bitdistill_rows = int(loss_scale.get("materialized_rows") or 0)
    projected_min = loss_scale.get("projected_paper_attention_to_ce_min")
    projected_max = loss_scale.get("projected_paper_attention_to_ce_max")
    ternary = mechanics.get("ternary_state", {}) if isinstance(mechanics.get("ternary_state"), dict) else {}
    code_fractions = ternary.get("code_fractions", {}) if isinstance(ternary.get("code_fractions"), dict) else {}
    stage_rows = stage2.get("rows", []) if isinstance(stage2.get("rows"), list) else []
    rows_with_ratios = sum(
        1
        for row in stage_rows
        if isinstance(row, dict)
        and (
            finite(row.get("weighted_logit_to_ce"))
            or finite(row.get("weighted_attention_to_ce"))
        )
    )

    return [
        {
            "telemetry": "raw task loss components",
            "status": "measured",
            "evidence": (
                f"StepMetrics fields present and {bitdistill_rows} materialized BitDistill rows "
                "record CE, logit KD, attention KD, and weighted KD terms."
            ),
            "supports": "Loss-scale sanity checks and finite-run triage.",
            "passed": source_has_all(
                source,
                [
                    "logit_kd: float",
                    "attention_kd: float",
                    "weighted_logit_kd: float",
                    "weighted_attention_kd: float",
                ],
            )
            and bitdistill_rows > 0,
        },
        {
            "telemetry": "paper-gamma loss magnitude projection",
            "status": "measured",
            "evidence": (
                f"Projected paper-gamma attention/CE range is {fmt(projected_min)} to {fmt(projected_max)}."
            ),
            "supports": "The claim that gamma comparison is normalization-sensitive.",
            "passed": finite(projected_min) and finite(projected_max),
        },
        {
            "telemetry": "weighted KD-to-CE ratios on Stage-2 rows",
            "status": "measured_when_rows_exist",
            "evidence": f"{rows_with_ratios} Stage-2 audit rows include weighted KD/CE ratios.",
            "supports": "Controlled-run interpretation after queued rows finish.",
            "passed": rows_with_ratios > 0,
        },
        {
            "telemetry": "final checkpoint ternary code distribution",
            "status": "measured_offline",
            "evidence": f"code fractions={code_fractions}; entropy={fmt(ternary.get('code_entropy_bits'))}.",
            "supports": "Static export/mechanics checks, not step-by-step training dynamics.",
            "passed": all(key in code_fractions for key in ["-1", "0", "1"]),
        },
        {
            "telemetry": "SubLN activation/logit perturbation",
            "status": "measured_offline",
            "evidence": (
                f"inserted={subln.get('subln_inserted')}; "
                f"logit relative RMS={fmt(subln.get('logit_relative_rms'))}; "
                f"cosine={fmt(subln.get('logit_cosine'))}."
            ),
            "supports": "The claim that untrained SubLN surgery is not identity-preserving locally.",
            "passed": finite(subln.get("logit_relative_rms")) and finite(subln.get("logit_cosine")),
        },
        {
            "telemetry": "opt-in training telemetry hooks",
            "status": "instrumented_not_materialized",
            "evidence": (
                "train_bitdistill.py exposes telemetry.jsonl, total grad norm, optional component "
                "grad norms, ternary code fractions, scale stats, threshold-band fractions, "
                "flip-rate telemetry, and scale-drift telemetry."
            ),
            "supports": "Future controlled rows can record update-balance diagnostics without changing default jobs.",
            "passed": source_has_all(
                source,
                [
                    "--telemetry-every-steps",
                    "--telemetry-component-grad-norms",
                    "component_grad_norms_microbatch",
                    "threshold_band_0p05_fraction",
                    "quantization_dynamics",
                    "telemetry.jsonl",
                ],
            ),
        },
        {
            "telemetry": "Slurm launcher telemetry pass-through",
            "status": "instrumented_not_materialized",
            "evidence": (
                "slurm_bitdistill_glue.sh exposes TELEMETRY_EVERY_STEPS, "
                "TELEMETRY_COMPONENT_GRAD_NORMS, and TELEMETRY_MAX_ELEMENTS_PER_LAYER "
                "to train_bitdistill.py."
            ),
            "supports": "Cluster jobs can materialize the new telemetry without hand-editing the launcher.",
            "passed": source_has_all(
                launcher_source,
                [
                    "TELEMETRY_EVERY_STEPS",
                    "TELEMETRY_COMPONENT_GRAD_NORMS",
                    "TELEMETRY_MAX_ELEMENTS_PER_LAYER",
                    "--telemetry-every-steps",
                    "--telemetry-component-grad-norms",
                    "--telemetry-max-elements-per-layer",
                ],
            ),
        },
        {
            "telemetry": "Q/K/V relation KD split",
            "status": "instrumented_not_materialized",
            "evidence": (
                "train_bitdistill.py records raw and weighted attention_q_kd, "
                "attention_k_kd, and attention_v_kd fields, and optional "
                "component-gradient telemetry can include the weighted Q/K/V terms."
            ),
            "supports": "Future controlled rows can identify which attention-relation term dominates.",
            "passed": source_has_all(
                source,
                [
                    "attention_q_kd",
                    "attention_k_kd",
                    "attention_v_kd",
                    "weighted_attention_q_kd",
                    "weighted_attention_k_kd",
                    "weighted_attention_v_kd",
                ],
            ),
        },
        {
            "telemetry": "BitLinear activation int8 saturation",
            "status": "instrumented_not_materialized",
            "evidence": (
                "train_bitdistill.py records per-telemetry-step activation A8 clipping, "
                "edge occupancy, per-token scale statistics, and absmax statistics for "
                "BitLinear student forwards."
            ),
            "supports": "Future controlled rows can rule activation clipping/saturation in or out as a quality limiter.",
            "passed": source_has_all(
                source,
                [
                    "capture_bitlinear_activation_quantization",
                    "clipped_fraction",
                    "int8_edge_fraction",
                    "activation_quantization",
                ],
            ),
        },
        {
            "telemetry": "ternary flip-rate and scale trajectory",
            "status": "instrumented_not_materialized",
            "evidence": (
                "train_bitdistill.py records sampled ternary code flip fractions and "
                "scale absolute-delta statistics between emitted telemetry steps."
            ),
            "supports": "Future controlled rows can show whether continued pretraining is moving ternary codes and scale semantics.",
            "passed": source_has_all(
                source,
                [
                    "BitLinearDynamicsTracker",
                    "flip_fraction",
                    "scale_abs_delta_mean",
                    "quantization_dynamics",
                ],
            ),
        },
    ]


def summarize_missing(source: str) -> list[dict[str, Any]]:
    return [
        {
            "telemetry": "materialized training-dynamics telemetry rows",
            "status": "missing_materialized_run",
            "evidence": (
                "Component-gradient, Q/K/V, activation-saturation, flip-rate, and scale-drift "
                "hooks exist, but completed controlled benchmark rows have not materialized "
                "the new traces yet."
            ),
            "risk": "Cannot yet make causal claims from completed rows about update direction, A8 saturation, or Stage-2 ternary dynamics.",
        },
    ]


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    source = args.training_source.read_text(encoding="utf-8") if args.training_source.exists() else ""
    launcher_source = args.launcher_source.read_text(encoding="utf-8") if args.launcher_source.exists() else ""
    measured = summarize_measured(args, source, launcher_source)
    missing = summarize_missing(source)
    measured_pass = sum(1 for row in measured if row["passed"])
    return {
        "schema": "bitdistill-telemetry-coverage-v1",
        "date": DATE,
        "training_source": str(args.training_source),
        "status": "partial_observability",
        "measured_count": measured_pass,
        "measured_expected": len(measured),
        "missing_count": len(missing),
        "measured": measured,
        "missing": missing,
        "verdict": (
            "Existing telemetry is sufficient for loss-scale and static-mechanics triage, "
            "and the training script plus Slurm launcher now have opt-in hooks for the next controlled wave. "
            "The completed benchmark artifacts are still not sufficient to prove update-direction "
            "causality, because materialized component-gradient, flip-rate, scale-trajectory, "
            "and activation-saturation traces do not exist yet."
        ),
        "safe_next_step": (
            "After the active queued jobs finish, launch the next controlled rows with "
            "--telemetry-every-steps and --telemetry-component-grad-norms enabled on a sparse cadence."
        ),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    measured_rows = [
        [
            row["telemetry"],
            "pass" if row["passed"] else "fail",
            row["status"],
            row["evidence"],
            row["supports"],
        ]
        for row in summary["measured"]
    ]
    missing_rows = [
        [row["telemetry"], row["status"], row["evidence"], row["risk"]]
        for row in summary["missing"]
    ]
    return "\n\n".join(
        [
            f"# BitDistill Telemetry Coverage Audit, {summary['date']}",
            f"Overall status: **{summary['status']}**.",
            summary["verdict"],
            (
                f"Measured diagnostics passing: `{summary['measured_count']}/"
                f"{summary['measured_expected']}`. Missing advanced diagnostics: "
                f"`{summary['missing_count']}`."
            ),
            "## Measured",
            md_table(["telemetry", "gate", "status", "evidence", "supports"], measured_rows),
            "## Missing Before Stronger Causal Claims",
            md_table(["telemetry", "status", "evidence", "risk"], missing_rows),
            "## Next Step",
            summary["safe_next_step"],
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-source", type=Path, default=Path("train_bitdistill.py"))
    parser.add_argument("--launcher-source", type=Path, default=Path("slurm_bitdistill_glue.sh"))
    parser.add_argument("--loss-scale-json", type=Path, default=Path(f"benchmark_results/bitdistill_loss_scale_audit_{DATE}.json"))
    parser.add_argument("--mechanics-json", type=Path, default=Path(f"benchmark_results/bitnet_sft_mechanics_audit_{DATE}.json"))
    parser.add_argument("--subln-json", type=Path, default=Path(f"benchmark_results/subln_activation_variance_{DATE}.json"))
    parser.add_argument("--stage2-json", type=Path, default=Path(f"benchmark_results/bitdistill_stage2_curve_{DATE}.json"))
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/bitdistill_telemetry_coverage_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/bitdistill_telemetry_coverage_{DATE}.md"))
    args = parser.parse_args()

    summary = build_summary(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_markdown(summary)
    args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
