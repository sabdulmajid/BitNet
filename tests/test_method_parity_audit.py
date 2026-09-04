from __future__ import annotations

import json
from pathlib import Path

from benchmarks.audit_bitdistill_method_parity import EXPECTED_TELEMETRY_STEPS, summarize_case


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
