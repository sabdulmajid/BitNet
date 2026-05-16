#!/usr/bin/env python3
"""Synthetic audit for activation-weighted ternary initialization.

This is a reconstruction audit, not a model-quality benchmark.  It tests a
candidate initializer for the next BitDistill/BitNet-SFT runs:

    min_T,s E ||X W^T - X (s * T)^T||_F^2,  T in {-1, 0, +1}

using a diagonal activation covariance approximation.  If this reduces output
RMS error before QAT, it is worth adding as an optional initialization path and
testing on MNLI.  It does not prove GLUE accuracy or general-LM quality.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value != 0 and (abs(value) >= 1000 or abs(value) < 0.0001):
            return f"{value:.6e}"
        return f"{value:.6f}"
    return str(value).replace("|", "\\|")


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(cell) for cell in row) + " |")
    return "\n".join(lines)


def output_rel_rms(x: torch.Tensor, weight: torch.Tensor, quantized: torch.Tensor) -> float:
    reference = x @ weight.T
    delta = x @ (weight - quantized).T
    return float((delta.norm() / reference.norm().clamp_min(1e-12)).item())


def code_zero_fraction(quantized_codes: torch.Tensor) -> float:
    return float((quantized_codes == 0).float().mean().item())


def absmean_quantize(weight: torch.Tensor, *, scale_mode: str, eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if scale_mode == "tensor":
        scale = weight.abs().mean().clamp_min(eps)
    elif scale_mode == "row":
        scale = weight.abs().mean(dim=1, keepdim=True).clamp_min(eps)
    else:
        raise ValueError(f"unknown scale_mode={scale_mode}")
    codes = torch.round(weight / scale).clamp(-1, 1)
    return codes * scale, codes, scale.reshape(-1)


def weighted_ls_quantize(
    weight: torch.Tensor,
    diag_hessian: torch.Tensor,
    *,
    scale_mode: str,
    eps: float,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Coordinate descent for diagonal-Hessian weighted ternary least squares.

    For fixed scale, the optimal ternary code is nonzero iff |w| > scale / 2.
    For fixed codes, the optimal nonnegative scale is the weighted least-squares
    projection of |w| onto the active code mask.
    """

    h = diag_hessian.reshape(1, -1).clamp_min(eps)
    if scale_mode == "tensor":
        scale = weight.abs().mean().clamp_min(eps)
        for _ in range(iterations):
            active = (weight.abs() > scale / 2.0).float()
            denom = (active * h).sum().clamp_min(eps)
            scale = ((active * h) * weight.abs()).sum() / denom
        codes = torch.sign(weight) * (weight.abs() > scale / 2.0).float()
    elif scale_mode == "row":
        scale = weight.abs().mean(dim=1, keepdim=True).clamp_min(eps)
        for _ in range(iterations):
            active = (weight.abs() > scale / 2.0).float()
            denom = (active * h).sum(dim=1, keepdim=True).clamp_min(eps)
            scale = ((active * h) * weight.abs()).sum(dim=1, keepdim=True) / denom
        codes = torch.sign(weight) * (weight.abs() > scale / 2.0).float()
    else:
        raise ValueError(f"unknown scale_mode={scale_mode}")
    return codes * scale, codes, scale.reshape(-1)


def make_activation(
    *,
    profile: str,
    samples: int,
    cols: int,
    generator: torch.Generator,
    log_sigma: float,
) -> torch.Tensor:
    base = torch.randn(samples, cols, generator=generator)
    if profile == "isotropic":
        return base
    if profile == "lognormal_diag":
        feature_scale = torch.exp(torch.randn(cols, generator=generator) * log_sigma)
        return base * feature_scale
    raise ValueError(f"unknown activation profile={profile}")


def summarize(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()) if tensor.numel() > 1 else 0.0,
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


def load_json_or_none(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_task_quality_audit(path: Path) -> dict[str, Any]:
    data = load_json_or_none(path)
    if data is None:
        return {"path": str(path), "exists": False, "status": "missing"}
    paired = data.get("paired", {}) if isinstance(data.get("paired"), dict) else {}
    return {
        "path": str(path),
        "exists": True,
        "status": data.get("status"),
        "comparison_valid": data.get("comparison_valid"),
        "candidate_improves_absmean_baseline": data.get("candidate_improves_absmean_baseline"),
        "baseline_accuracy": data.get("baseline_accuracy"),
        "candidate_accuracy": data.get("candidate_accuracy"),
        "delta_vs_absmean_baseline": data.get("delta_vs_absmean_baseline"),
        "paired_matched": paired.get("matched"),
        "paired_ci95": paired.get("paired_ci95"),
        "verdict": data.get("verdict"),
    }


def run_trial(args: argparse.Namespace, *, profile: str, seed: int) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(args.rows, args.cols, generator=generator)
    x = make_activation(
        profile=profile,
        samples=args.samples,
        cols=args.cols,
        generator=generator,
        log_sigma=args.activation_log_sigma,
    )
    diag_hessian = (x * x).mean(dim=0)

    methods = {
        "tensor_absmean_paper": absmean_quantize(weight, scale_mode="tensor", eps=args.eps),
        "row_absmean_retrofit": absmean_quantize(weight, scale_mode="row", eps=args.eps),
        "tensor_diag_hessian_ls": weighted_ls_quantize(
            weight,
            diag_hessian,
            scale_mode="tensor",
            eps=args.eps,
            iterations=args.iterations,
        ),
        "row_diag_hessian_ls": weighted_ls_quantize(
            weight,
            diag_hessian,
            scale_mode="row",
            eps=args.eps,
            iterations=args.iterations,
        ),
    }

    rows = {}
    for name, (quantized, codes, scale) in methods.items():
        rows[name] = {
            "output_rel_rms": output_rel_rms(x, weight, quantized),
            "zero_fraction": code_zero_fraction(codes),
            "scale_mean": float(scale.float().mean().item()),
            "scale_std": float(scale.float().std(unbiased=False).item()) if scale.numel() > 1 else 0.0,
        }
    return {"seed": seed, "profile": profile, "methods": rows}


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repo_root.resolve()
    train_distill = (root / "train_distill.py").read_text(encoding="utf-8")
    train_bitdistill = (root / "train_bitdistill.py").read_text(encoding="utf-8")
    source_checks = [
        {
            "check": "least-squares initializer helper exists",
            "passed": "def initialize_bitlinear_least_squares" in train_distill
            and "def least_squares_ternary_codes_and_scale" in train_distill,
        },
        {
            "check": "BitDistill CLI exposes opt-in init mode",
            "passed": "--ternary-init-mode" in train_bitdistill
            and "choices=[\"absmean\", \"ls\", \"diag_ls\"]" in train_bitdistill,
        },
        {
            "check": "trained checkpoint loads are not reinitialized",
            "passed": "init_state_dict_will_load_trained_weights" in train_bitdistill,
        },
        {
            "check": "diagonal-Hessian calibration hook exists",
            "passed": "def collect_bitlinear_input_diag_hessians" in train_bitdistill
            and "--ternary-init-calibration-batches" in train_bitdistill
            and "diag_hessians=hessians" in train_bitdistill,
        },
    ]
    ls_training_integrated = all(check["passed"] for check in source_checks[:3])
    diag_hessian_training_integrated = all(check["passed"] for check in source_checks)

    profiles = ["isotropic", "lognormal_diag"]
    trials = [
        run_trial(args, profile=profile, seed=args.seed + profile_index * 1000 + trial)
        for profile_index, profile in enumerate(profiles)
        for trial in range(args.trials)
    ]

    by_profile: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        profile_trials = [trial for trial in trials if trial["profile"] == profile]
        methods = sorted(profile_trials[0]["methods"])
        by_profile[profile] = {"trials": len(profile_trials), "methods": {}}
        for method in methods:
            errors = [trial["methods"][method]["output_rel_rms"] for trial in profile_trials]
            zeros = [trial["methods"][method]["zero_fraction"] for trial in profile_trials]
            by_profile[profile]["methods"][method] = {
                "output_rel_rms": summarize(errors),
                "zero_fraction": summarize(zeros),
            }

        baseline = [trial["methods"]["row_absmean_retrofit"]["output_rel_rms"] for trial in profile_trials]
        candidate = [trial["methods"]["row_diag_hessian_ls"]["output_rel_rms"] for trial in profile_trials]
        deltas = [cand - base for cand, base in zip(candidate, baseline)]
        by_profile[profile]["row_diag_hessian_ls_minus_row_absmean"] = {
            **summarize(deltas),
            "wins": int(sum(delta < 0 for delta in deltas)),
            "trials": len(deltas),
        }

    win_profiles = [
        profile
        for profile, data in by_profile.items()
        if data["row_diag_hessian_ls_minus_row_absmean"]["mean"] < -0.02
        and data["row_diag_hessian_ls_minus_row_absmean"]["wins"] == data["row_diag_hessian_ls_minus_row_absmean"]["trials"]
    ]
    synthetic_promising = set(win_profiles) == set(profiles)

    task_quality_audits = {
        "ls": summarize_task_quality_audit(root / f"benchmark_results/bitnet_sft_ls_init_audit_{DATE}.json"),
        "diag_ls": summarize_task_quality_audit(root / f"benchmark_results/bitnet_sft_diag_ls_init_audit_{DATE}.json"),
    }
    complete_quality_audits = [
        audit
        for audit in task_quality_audits.values()
        if audit.get("status") == "complete" and audit.get("comparison_valid") is True
    ]
    completed_quality_rejected = (
        len(complete_quality_audits) >= 2
        and all(audit.get("candidate_improves_absmean_baseline") is False for audit in complete_quality_audits)
    )
    completed_quality_improved = any(
        audit.get("candidate_improves_absmean_baseline") is True for audit in complete_quality_audits
    )

    if synthetic_promising and completed_quality_improved:
        status = "synthetic_promising_task_quality_supported"
        quality_proven = True
        verdict = (
            "Diagonal-Hessian weighted ternary least-squares reduces synthetic output reconstruction error and "
            "has at least one completed task-quality audit that improves over the matched absmean baseline."
        )
        next_gate = "Promote only the task-quality-supported initializer variant and re-test under BitDistill."
    elif synthetic_promising and completed_quality_rejected:
        status = "synthetic_promising_task_quality_rejected"
        quality_proven = False
        verdict = (
            "Diagonal-Hessian weighted ternary least-squares reduces synthetic output reconstruction error, but "
            "both completed MNLI BitNet-SFT initializer audits are negative versus the matched absmean baseline. "
            "Synthetic reconstruction gains are not sufficient evidence for task quality."
        )
        next_gate = (
            "Do not promote LS or diag-LS initialization in the main recipe. Further initializer work needs a "
            "new hypothesis and must clear a full-validation paired task audit before being used in claims."
        )
    elif synthetic_promising and diag_hessian_training_integrated:
        status = "synthetic_promising_diag_calibration_integrated_quality_pending"
        quality_proven = False
        verdict = (
            "Diagonal-Hessian weighted ternary least-squares reduces synthetic output reconstruction error versus "
            "absmean row-scale initialization, so it is a credible next training initializer. The training hook now "
            "exposes both unweighted least-squares initialization and opt-in activation-calibrated diagonal-Hessian "
            "initialization. This is not a GLUE or language-model quality claim until a full BitNet-SFT/BitDistill "
            "run uses it."
        )
        next_gate = (
            "Run MNLI BitNet-SFT with --ternary-init-mode diag_ls against the matched absmean and unweighted-LS "
            "baselines, then compare full-validation paired predictions."
        )
    elif synthetic_promising and ls_training_integrated:
        status = "synthetic_promising_ls_integrated_diag_calibration_pending"
        quality_proven = False
        verdict = (
            "Diagonal-Hessian weighted ternary least-squares reduces synthetic output reconstruction error, but "
            "the activation-calibrated training hook is not fully integrated yet."
        )
        next_gate = "Integrate the diag-LS training hook before any task-quality run."
    else:
        status = "synthetic_mixed"
        quality_proven = False
        verdict = "The synthetic reconstruction audit is mixed and should not drive a task-quality claim."
        next_gate = "Do not promote this initializer without a better synthetic and task-quality result."

    return {
        "schema": "second-order-ternary-init-audit-v1",
        "date": DATE,
        "status": status,
        "quality_proven": quality_proven,
        "training_integrated": ls_training_integrated,
        "diag_hessian_training_integrated": diag_hessian_training_integrated,
        "source_checks": source_checks,
        "task_quality_audits": task_quality_audits,
        "rows": args.rows,
        "cols": args.cols,
        "samples": args.samples,
        "trials_per_profile": args.trials,
        "iterations": args.iterations,
        "activation_profiles": profiles,
        "profiles": by_profile,
        "verdict": verdict,
        "next_gate": next_gate,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    rows = []
    delta_rows = []
    for profile, profile_data in summary["profiles"].items():
        for method, method_data in profile_data["methods"].items():
            err = method_data["output_rel_rms"]
            zero = method_data["zero_fraction"]
            rows.append([profile, method, err["mean"], err["std"], zero["mean"]])
        delta = profile_data["row_diag_hessian_ls_minus_row_absmean"]
        delta_rows.append([profile, delta["mean"], delta["std"], delta["wins"], delta["trials"]])

    task_quality_rows = []
    for name, audit in summary.get("task_quality_audits", {}).items():
        task_quality_rows.append(
            [
                name,
                audit.get("status"),
                audit.get("baseline_accuracy"),
                audit.get("candidate_accuracy"),
                audit.get("delta_vs_absmean_baseline"),
                audit.get("paired_matched"),
                audit.get("paired_ci95"),
                audit.get("candidate_improves_absmean_baseline"),
            ]
        )

    return "\n\n".join(
        [
            f"# Second-Order Ternary Initialization Audit, {summary['date']}",
            summary["verdict"],
            "## Setup",
            md_table(
                ["field", "value"],
                [
                    ["rows", summary["rows"]],
                    ["cols", summary["cols"]],
                    ["calibration samples", summary["samples"]],
                    ["trials per activation profile", summary["trials_per_profile"]],
                    ["coordinate-descent iterations", summary["iterations"]],
                    ["quality proven", summary["quality_proven"]],
                    ["unweighted LS training integrated", summary["training_integrated"]],
                    ["diagonal-Hessian training integrated", summary["diag_hessian_training_integrated"]],
                ],
            ),
            "## Reconstruction Results",
            md_table(["activation profile", "method", "mean rel RMS", "std", "mean zero fraction"], rows),
            "## Paired Candidate Delta",
            "Negative values mean the diagonal-Hessian row-scale initializer had lower reconstruction error than row absmean.",
            md_table(["activation profile", "mean delta", "std", "wins", "trials"], delta_rows),
            "## Source Integration",
            md_table(
                ["check", "status"],
                [
                    [row["check"], "pass" if row["passed"] else "fail"]
                    for row in summary["source_checks"]
                ],
            ),
            "## Task Quality Follow-Up",
            md_table(
                [
                    "initializer",
                    "status",
                    "absmean accuracy",
                    "candidate accuracy",
                    "delta",
                    "matched",
                    "CI95",
                    "improves absmean",
                ],
                task_quality_rows,
            ),
            "## Math",
            (
                "For diagonal activation covariance `H=diag(h)`, each output row minimizes "
                "`sum_j h_j (w_j - s t_j)^2` with `t_j in {-1,0,+1}`. For fixed `s`, the optimal "
                "code is nonzero when `|w_j| > s/2`. For fixed codes, the optimal row scale is "
                "`s = sum_j h_j |w_j| 1(|w_j|>s/2) / sum_j h_j 1(|w_j|>s/2)`. The script iterates "
                "these two closed-form steps."
            ),
            "## Next Gate",
            summary["next_gate"],
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--cols", type=int, default=512)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--activation-log-sigma", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-json", type=Path, default=Path(f"benchmark_results/second_order_ternary_init_{DATE}.json"))
    parser.add_argument("--output-md", type=Path, default=Path(f"benchmarks/results/second_order_ternary_init_{DATE}.md"))
    args = parser.parse_args()

    summary = build_summary(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(summary), encoding="utf-8")
    print(render_markdown(summary))


if __name__ == "__main__":
    main()
