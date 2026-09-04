#!/usr/bin/env python3
"""Audit the non-equivalent attention-relation definitions in BitDistill v1."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_bitdistill import attention_relation_distillation_components


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()


def relation_result(
    student: dict[str, torch.Tensor],
    teacher: dict[str, torch.Tensor],
    mask: torch.Tensor,
    *,
    relation_mode: str,
    split_heads: int,
) -> dict[str, Any]:
    local_student = {
        key: value.detach().clone().requires_grad_(True)
        for key, value in student.items()
    }
    loss, components = attention_relation_distillation_components(
        local_student,
        teacher,
        mask,
        split_heads=split_heads,
        temperature=1.0,
        qkv_reduction="sum",
        relation_mode=relation_mode,
    )
    gradients = torch.autograd.grad(loss, list(local_student.values()))
    flat_gradient = torch.cat([gradient.detach().float().reshape(-1) for gradient in gradients])
    return {
        "relation_mode": relation_mode,
        "split_heads": split_heads,
        "loss": float(loss.detach()),
        "components": {key: float(value.detach()) for key, value in components.items()},
        "gradient_norm": float(torch.linalg.vector_norm(flat_gradient)),
        "gradient": flat_gradient,
    }


def gradient_cosine(left: dict[str, Any], right: dict[str, Any]) -> float:
    return float(torch.nn.functional.cosine_similarity(left["gradient"], right["gradient"], dim=0))


def public_result(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "gradient"}


def build_report(seed: int) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    student = {
        "q": torch.randn(2, 11, 32, generator=generator, dtype=torch.float64),
        "k": torch.randn(2, 11, 16, generator=generator, dtype=torch.float64),
        "v": torch.randn(2, 11, 16, generator=generator, dtype=torch.float64),
    }
    teacher = {
        key: torch.randn(value.shape, generator=generator, dtype=torch.float64)
        for key, value in student.items()
    }
    mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]],
        dtype=torch.long,
    )

    cosine_split1 = relation_result(student, teacher, mask, relation_mode="cosine", split_heads=1)
    cosine_split8 = relation_result(student, teacher, mask, relation_mode="cosine", split_heads=8)
    scaled_dot_split1 = relation_result(student, teacher, mask, relation_mode="scaled_dot", split_heads=1)

    scaled_student = {
        key: value * torch.linspace(0.25, 4.0, value.shape[1], dtype=value.dtype).view(1, -1, 1)
        for key, value in student.items()
    }
    scaled_teacher = {
        key: value * torch.linspace(3.0, 0.5, value.shape[1], dtype=value.dtype).view(1, -1, 1)
        for key, value in teacher.items()
    }
    cosine_rescaled = relation_result(scaled_student, scaled_teacher, mask, relation_mode="cosine", split_heads=1)
    scaled_dot_rescaled = relation_result(
        scaled_student,
        scaled_teacher,
        mask,
        relation_mode="scaled_dot",
        split_heads=1,
    )

    checks = {
        "equation_and_pseudocode_losses_differ": not math.isclose(
            cosine_split1["loss"], scaled_dot_split1["loss"], rel_tol=1e-3, abs_tol=1e-6
        ),
        "equation_and_pseudocode_gradients_differ": gradient_cosine(cosine_split1, scaled_dot_split1) < 0.999,
        "cosine_definition_is_norm_invariant": math.isclose(
            cosine_split1["loss"], cosine_rescaled["loss"], rel_tol=1e-5, abs_tol=1e-7
        ),
        "scaled_dot_definition_is_not_norm_invariant": not math.isclose(
            scaled_dot_split1["loss"], scaled_dot_rescaled["loss"], rel_tol=1e-3, abs_tol=1e-6
        ),
        "split_count_changes_objective": not math.isclose(
            cosine_split1["loss"], cosine_split8["loss"], rel_tol=1e-3, abs_tol=1e-6
        ),
    }
    passed = all(checks.values())
    return {
        "schema": "bitdistill-attention-relation-equivalence-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "published_specification_ambiguous" if passed else "audit_failed",
        "quality_claim": "mathematical_contract_not_task_quality",
        "seed": seed,
        "paper_source": "https://arxiv.org/abs/2510.13998",
        "definitions": {
            "equation_12": "softmax(A A^T / sqrt(d_r))",
            "algorithm_1": "softmax(normalize(A) normalize(A)^T / temperature)",
            "legacy_local": "Algorithm-1 cosine form with split_heads=8",
        },
        "proof": (
            "Equation 12 scales each dot product by the single factor sqrt(d_r), whereas Algorithm 1 "
            "scales it by the pair-dependent factor temperature*||a_i||*||a_j||. No global temperature "
            "makes those logits equal for arbitrary hidden states unless the relevant norm products are "
            "constant (apart from degenerate softmax-equivalent cases). Their KL scales and gradients "
            "therefore need not agree, so a fixed attention coefficient is not portable between them."
        ),
        "checks": checks,
        "results": {
            "algorithm1_cosine_split1": public_result(cosine_split1),
            "legacy_cosine_split8": public_result(cosine_split8),
            "equation12_scaled_dot_split1": public_result(scaled_dot_split1),
            "algorithm1_cosine_split1_rescaled": public_result(cosine_rescaled),
            "equation12_scaled_dot_split1_rescaled": public_result(scaled_dot_rescaled),
        },
        "comparisons": {
            "equation_vs_algorithm_loss_ratio": scaled_dot_split1["loss"] / cosine_split1["loss"],
            "equation_vs_algorithm_gradient_norm_ratio": (
                scaled_dot_split1["gradient_norm"] / cosine_split1["gradient_norm"]
            ),
            "equation_vs_algorithm_gradient_cosine": gradient_cosine(cosine_split1, scaled_dot_split1),
            "split8_vs_split1_loss_ratio": cosine_split8["loss"] / cosine_split1["loss"],
            "split8_vs_split1_gradient_norm_ratio": (
                cosine_split8["gradient_norm"] / cosine_split1["gradient_norm"]
            ),
            "split8_vs_split1_gradient_cosine": gradient_cosine(cosine_split1, cosine_split8),
            "equation_rescaling_loss_ratio": scaled_dot_rescaled["loss"] / scaled_dot_split1["loss"],
            "algorithm_rescaling_loss_ratio": cosine_rescaled["loss"] / cosine_split1["loss"],
        },
        "decision": (
            "Do not interpret gamma sweeps until relation_mode and split_heads are explicit. "
            "Run short telemetry pilots for cosine/split1, scaled_dot/split1, and the legacy "
            "cosine/split8 control before selecting one full-quality MNLI run."
        ),
    }


def fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "pass" if value else "fail"
    if isinstance(value, float):
        return f"{value:.9g}"
    return str(value)


def table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(fmt(value).replace("|", "\\|") for value in row) + " |" for row in rows)
    return "\n".join(lines)


def render_markdown(report: dict[str, Any]) -> str:
    result_rows = [
        [name, row["relation_mode"], row["split_heads"], row["loss"], row["gradient_norm"]]
        for name, row in report["results"].items()
    ]
    comparison_rows = [[key, value] for key, value in report["comparisons"].items()]
    return "\n\n".join(
        [
            "# BitDistill Attention-Relation Equivalence Audit",
            f"Generated: `{report['created_utc']}`",
            f"Status: **{report['status']}**.",
            f"Quality claim: **{report['quality_claim']}**.",
            "## Result",
            (
                "The relation matrix in Equation 12 is not mathematically equivalent to the "
                "L2-normalized relation matrix in Algorithm 1 for general hidden states. The "
                "number of relation heads also changes both the loss and its gradient."
            ),
            "## Proof",
            report["proof"],
            "## Contract Checks",
            table(["check", "status"], [[key, value] for key, value in report["checks"].items()]),
            "## Deterministic Probe",
            table(["variant", "mode", "split heads", "loss", "gradient norm"], result_rows),
            "## Comparisons",
            table(["quantity", "value"], comparison_rows),
            "## Decision",
            report["decision"],
            (
                "This synthetic audit proves a mathematical contract difference. It does not "
                "establish downstream accuracy or identify which published definition produced "
                "the paper's reported scores."
            ),
        ]
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(f"benchmarks/results/attention_relation_equivalence_{DATE}.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(f"benchmarks/results/attention_relation_equivalence_{DATE}.md"),
    )
    args = parser.parse_args()
    report = build_report(args.seed)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    return 0 if report["status"] != "audit_failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
