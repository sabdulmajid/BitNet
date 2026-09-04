#!/usr/bin/env python3
"""Fail-closed quality audit for the completed full MNLI gamma-60 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.audit_bitdistill_adaptive_full import compare_prediction_files


RUN_ROOT = Path("checkpoints/bitdistill-glue-seqcls-recovery/Qwen-Qwen2.5-0.5B/mnli")
DEFAULT_CANDIDATE = RUN_ROOT / "bitdistill-tensor-20kwarmup-steps10000-lr2em5-gamma60-headinit"
DEFAULT_FIXED_163M = RUN_ROOT / "bitdistill-tensor-20kwarmup-steps10000-lr2em5-papergamma-headinit"
DEFAULT_FIXED_655M = RUN_ROOT / "bitdistill-tensor-655mwarmup-steps10000-lr2em5-papergamma-headinit"
DEFAULT_FP16 = Path(
    "checkpoints/bitdistill-glue-seqcls-predtrace/Qwen-Qwen2.5-0.5B/mnli/"
    "fp16_sft-tensor-layer-1"
)
DEFAULT_CANDIDATE_LOG = Path("logs/bitdistill-gamma60-10077.out")
DEFAULT_FIXED_163M_LOG = Path("logs/bitdistill-glue-10068.out")

STEP_PATTERN = re.compile(
    r"^step=(?P<step>\d+) loss=(?P<loss>\S+) ce=(?P<ce>\S+) "
    r"logit_kd=(?P<logit_kd>\S+) attention_kd=(?P<attention_kd>\S+)"
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nested(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def validate_metrics(metrics: dict[str, Any], *, gamma: float, state_fragment: str) -> list[str]:
    expected = {
        "method": (metrics.get("method"), "bitdistill"),
        "task": (metrics.get("task"), "mnli"),
        "task_format": (metrics.get("task_format"), "sequence_classification"),
        "scale_mode": (metrics.get("scale_mode"), "tensor"),
        "steps": (metrics.get("steps"), 10_000),
        "eval_examples": (nested(metrics, "eval", "eval_examples"), 9_815.0),
        "attention_split_heads": (metrics.get("attention_split_heads"), 8),
        "attention_kd_weight": (nested(metrics, "loss_weights", "attention_kd_weight"), gamma),
        "state_loaded": (nested(metrics, "state_load", "loaded"), True),
        "head_copied": (nested(metrics, "output_head_init", "copied"), True),
        "activation_quantization": (nested(metrics, "preparation", "activation_quantization"), True),
        "subln_inserted": (nested(metrics, "preparation", "subln_inserted"), 48),
        "bitlinear_replaced": (nested(metrics, "preparation", "bitlinear_replaced"), 168),
    }
    errors = [f"{name}={actual!r}, expected={wanted!r}" for name, (actual, wanted) in expected.items() if actual != wanted]
    state_path = nested(metrics, "state_load", "path")
    if not isinstance(state_path, str) or state_fragment not in state_path:
        errors.append(f"state_load.path={state_path!r}, expected fragment={state_fragment!r}")
    accuracy = nested(metrics, "eval", "accuracy")
    if not isinstance(accuracy, (int, float)) or not math.isfinite(accuracy):
        errors.append(f"eval.accuracy={accuracy!r} is not finite")
    return errors


def controlled_contract_differences(candidate: dict[str, Any], reference: dict[str, Any]) -> list[str]:
    paths = [
        ("student_model",),
        ("teacher_model",),
        ("task",),
        ("task_format",),
        ("scale_mode",),
        ("exclude_linear_regex",),
        ("distill_layer",),
        ("attention_split_heads",),
        ("label_scheme",),
        ("candidate_score",),
        ("steps",),
        ("training_budget",),
        ("preparation",),
        ("state_load", "path"),
        ("loss_weights", "logit_kd_weight"),
        ("loss_weights", "logit_temperature"),
        ("loss_weights", "logit_kd_temperature_scale"),
        ("loss_weights", "attention_temperature"),
        ("loss_weights", "attention_qkv_reduction"),
    ]
    differences: list[str] = []
    for path in paths:
        candidate_value = nested(candidate, *path)
        reference_value = nested(reference, *path)
        if candidate_value != reference_value:
            name = ".".join(path)
            differences.append(f"{name}: candidate={candidate_value!r}, reference={reference_value!r}")
    return differences


def artifact_manifest(run_dir: Path) -> dict[str, Any]:
    metrics = run_dir / "metrics.json"
    predictions = run_dir / "eval_predictions.jsonl"
    return {
        "run_dir": str(run_dir),
        "metrics": str(metrics),
        "metrics_sha256": sha256(metrics),
        "predictions": str(predictions),
        "predictions_sha256": sha256(predictions),
    }


def first_step_fingerprint(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(path)
    for line in path.read_text(encoding="utf-8").splitlines():
        match = STEP_PATTERN.match(line)
        if match and int(match.group("step")) == 1:
            return {
                name: float(match.group(name))
                for name in ("ce", "logit_kd", "attention_kd")
            }
    raise ValueError(f"{path}: no step-1 training record")


def render_markdown(report: dict[str, Any]) -> str:
    comparisons = report["comparisons"]
    lines = [
        "# Full MNLI Gamma-60 Quality Audit",
        "",
        f"Generated: `{report['created_utc']}`",
        "",
        f"Status: **{report['status']}**.",
        "",
        "## Result",
        "",
        report["conclusion"],
        "",
        "| Comparison | Candidate | Reference | Delta | Paired 95% CI | Exact McNemar p |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, comparison in comparisons.items():
        lines.append(
            f"| {name.replace('_', ' ')} | `{comparison['candidate_accuracy']:.6f}` | "
            f"`{comparison['reference_accuracy']:.6f}` | "
            f"`{comparison['delta_candidate_minus_reference']:+.6f}` | "
            f"`[{comparison['paired_ci95'][0]:.6f}, {comparison['paired_ci95'][1]:.6f}]` | "
            f"`{comparison['mcnemar_exact_p']:.6g}` |"
        )
    lines.extend(
        [
            "",
            "## Contract",
            "",
            "The matched 163.84M comparison holds the checkpoint, model, task, 10,000-step budget, "
            "head initialization, tensor-scale W1.58A8 path, SubLN surgery, relation-head split, "
            "and all available serialized training-budget fields fixed. The logs declare only the "
            "attention-KD coefficient change from `100,000` to `60`. Their step-1 CE, logits-KD, "
            "and attention-KD values are exactly identical, providing an execution fingerprint for "
            "the same initialization and first batch.",
            "",
            "## Limitations",
            "",
            "- This is one historical run pair; neither metrics file serialized seed or source revision.",
            "- The paired intervals measure validation-example uncertainty conditional on fixed checkpoints.",
            "- Gamma 60 is implementation-specific and is not evidence that the paper's coefficient is wrong.",
            "- The comparison against 655.36M changes both Stage-2 budget and gamma; it is not the one-axis test.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--fixed-163m", type=Path, default=DEFAULT_FIXED_163M)
    parser.add_argument("--fixed-655m", type=Path, default=DEFAULT_FIXED_655M)
    parser.add_argument("--fp16", type=Path, default=DEFAULT_FP16)
    parser.add_argument("--candidate-log", type=Path, default=DEFAULT_CANDIDATE_LOG)
    parser.add_argument("--fixed-163m-log", type=Path, default=DEFAULT_FIXED_163M_LOG)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmarks/results/bitdistill_gamma60_quality_2026-09-04.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("benchmarks/results/bitdistill_gamma60_quality_2026-09-04.md"),
    )
    args = parser.parse_args()

    candidate_metrics = read_json(args.candidate / "metrics.json")
    fixed_163m_metrics = read_json(args.fixed_163m / "metrics.json")
    fixed_655m_metrics = read_json(args.fixed_655m / "metrics.json")
    errors = validate_metrics(candidate_metrics, gamma=60.0, state_fragment="bitdistill-tensor-20k")
    errors.extend(validate_metrics(fixed_163m_metrics, gamma=100_000.0, state_fragment="bitdistill-tensor-20k"))
    errors.extend(
        validate_metrics(fixed_655m_metrics, gamma=100_000.0, state_fragment="bitdistill-tensor-655m-from327m")
    )
    errors.extend(controlled_contract_differences(candidate_metrics, fixed_163m_metrics))

    candidate_step1 = first_step_fingerprint(args.candidate_log)
    fixed_step1 = first_step_fingerprint(args.fixed_163m_log)
    if candidate_step1 != fixed_step1:
        errors.append(
            f"step-1 execution fingerprints differ: candidate={candidate_step1!r}, "
            f"reference={fixed_step1!r}"
        )

    candidate_predictions = args.candidate / "eval_predictions.jsonl"
    comparisons = {
        "matched_fixed_gamma_163m": compare_prediction_files(
            args.fixed_163m / "eval_predictions.jsonl",
            candidate_predictions,
        ),
        "fixed_gamma_655m": compare_prediction_files(
            args.fixed_655m / "eval_predictions.jsonl",
            candidate_predictions,
        ),
        "fp16": compare_prediction_files(
            args.fp16 / "eval_predictions.jsonl",
            candidate_predictions,
        ),
    }
    for name, comparison in comparisons.items():
        if comparison["status"] != "pass":
            errors.extend(f"{name}: {error}" for error in comparison["errors"])

    status = "matched_historical_control" if not errors else "invalid"
    matched = comparisons["matched_fixed_gamma_163m"]
    conclusion = (
        "In the matched historical 163.84M Stage-2 comparison, the run declaring local attention-KD "
        f"coefficient 60 improves MNLI over the run declaring 100,000 by "
        f"`{matched['delta_candidate_minus_reference']:+.6f}` with "
        f"paired 95% CI `[{matched['paired_ci95'][0]:.6f}, {matched['paired_ci95'][1]:.6f}]`. "
        "The exact step-1 execution fingerprint and matching serialized contract make this strong "
        "local evidence that loss-scale alignment matters, pending source-pinned seeded replication."
        if not errors
        else "The quality audit failed its artifact or contract checks; no claim is permitted."
    )
    report = {
        "schema": "bitdistill-gamma60-quality-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "errors": errors,
        "claim_scope": "matched_historical_gamma_control_without_seed_or_source_revision",
        "candidate_gamma": 60.0,
        "matched_reference_gamma": 100_000.0,
        "comparisons": comparisons,
        "step1_execution_fingerprint": {
            "candidate": candidate_step1,
            "fixed_163m": fixed_step1,
            "exact_match": candidate_step1 == fixed_step1,
        },
        "artifacts": {
            "candidate": artifact_manifest(args.candidate),
            "fixed_163m": artifact_manifest(args.fixed_163m),
            "fixed_655m": artifact_manifest(args.fixed_655m),
            "fp16": artifact_manifest(args.fp16),
            "candidate_log": {
                "path": str(args.candidate_log),
                "sha256": sha256(args.candidate_log),
            },
            "fixed_163m_log": {
                "path": str(args.fixed_163m_log),
                "sha256": sha256(args.fixed_163m_log),
            },
        },
        "conclusion": conclusion,
        "limitations": [
            "Historical metrics omitted seed and source revision; this is not a fully provenance-pinned replicate.",
            "Paired intervals measure validation-example uncertainty, not training-seed uncertainty.",
            "Gamma is not portable across different loss reductions or method definitions.",
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
