from __future__ import annotations

import json
from pathlib import Path

from benchmarks.audit_bitdistill_method_parity import (
    EXPECTED_EVAL_EXAMPLES,
    EXPECTED_TELEMETRY_STEPS,
    compare_predictions,
    summarize_case,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_valid_case(root: Path, case: str) -> None:
    case_dir = root / case
    write_json(
        case_dir / "metrics.json",
        {
            "steps": 120,
            "source_revision": "test-revision",
            "task_format": "sequence_classification",
            "attention_split_heads": 1,
            "eval": {"eval_examples": 512, "accuracy": 0.5},
            "loss_weights": {
                "attention_relation_mode": "cosine",
                "attention_kd_balance": "gradnorm_ema",
                "attention_kd_weight": 100000.0,
            },
        },
    )
    rows = [
        {
            "step": step,
            "component_grad_norms_microbatch": {"ce": 1.0, "weighted_attention_kd": 1.0},
            "loss": {"ce": 1.0, "weighted_attention_kd": 1.0, "effective_attention_kd_weight": 10.0},
        }
        for step in EXPECTED_TELEMETRY_STEPS
    ]
    (case_dir / "telemetry.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    (case_dir / "eval_predictions.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "index": index,
                    "label": index % 3,
                    "prediction": index % 3,
                    "correct": True,
                }
            )
            + "\n"
            for index in range(EXPECTED_EVAL_EXAMPLES)
        ),
        encoding="utf-8",
    )


def test_complete_case_requires_exact_telemetry_schedule(tmp_path: Path) -> None:
    write_valid_case(tmp_path, "case")

    result = summarize_case(tmp_path, "case")

    assert result["status"] == "complete"
    assert result["telemetry_rows"] == len(EXPECTED_TELEMETRY_STEPS)


def test_wrong_telemetry_schedule_fails_closed(tmp_path: Path) -> None:
    write_valid_case(tmp_path, "case")
    telemetry_path = tmp_path / "case" / "telemetry.jsonl"
    rows = [json.loads(line) for line in telemetry_path.read_text(encoding="utf-8").splitlines()]
    rows[-1]["step"] = 119
    telemetry_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    result = summarize_case(tmp_path, "case")

    assert result["status"] == "pending_or_invalid"
    assert any("expected telemetry steps" in blocker for blocker in result["blockers"])


def test_missing_predictions_fail_closed(tmp_path: Path) -> None:
    write_valid_case(tmp_path, "case")
    (tmp_path / "case" / "eval_predictions.jsonl").unlink()

    result = summarize_case(tmp_path, "case")

    assert result["status"] == "pending_or_invalid"
    assert any("prediction" in blocker for blocker in result["blockers"])


def test_paired_comparison_reports_discordant_counts(tmp_path: Path) -> None:
    write_valid_case(tmp_path, "reference")
    write_valid_case(tmp_path, "candidate")
    candidate_path = tmp_path / "candidate" / "eval_predictions.jsonl"
    rows = [json.loads(line) for line in candidate_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["prediction"] = (rows[0]["label"] + 1) % 3
    rows[0]["correct"] = False
    candidate_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    result = compare_predictions(tmp_path, "reference", "candidate")

    assert result["status"] == "pass"
    assert result["matched"] == EXPECTED_EVAL_EXAMPLES
    assert result["candidate_wins"] == 0
    assert result["reference_wins"] == 1
    assert result["delta_candidate_minus_reference"] == -1 / EXPECTED_EVAL_EXAMPLES
