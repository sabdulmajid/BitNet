from __future__ import annotations

from benchmarks.audit_bitdistill_gamma60_quality import (
    controlled_contract_differences,
    first_step_fingerprint,
    validate_metrics,
)


def metrics(gamma: float = 60.0) -> dict:
    return {
        "method": "bitdistill",
        "task": "mnli",
        "task_format": "sequence_classification",
        "scale_mode": "tensor",
        "steps": 10_000,
        "attention_split_heads": 8,
        "student_model": "student",
        "teacher_model": "teacher",
        "exclude_linear_regex": "score|classifier",
        "distill_layer": -1,
        "label_scheme": "letters",
        "candidate_score": "mean",
        "training_budget": {"max_steps": 10_000},
        "eval": {"eval_examples": 9_815.0, "accuracy": 0.7},
        "state_load": {"loaded": True, "path": "root/bitdistill-tensor-20k/state.pt"},
        "output_head_init": {"copied": True},
        "preparation": {
            "activation_quantization": True,
            "subln_inserted": 48,
            "bitlinear_replaced": 168,
        },
        "loss_weights": {
            "attention_kd_weight": gamma,
            "logit_kd_weight": 10.0,
            "logit_temperature": 5.0,
            "logit_kd_temperature_scale": "none",
            "attention_temperature": 1.0,
            "attention_qkv_reduction": "sum",
        },
    }


def test_gamma_is_the_only_allowed_control_difference() -> None:
    candidate = metrics(60.0)
    reference = metrics(100_000.0)

    assert validate_metrics(candidate, gamma=60.0, state_fragment="bitdistill-tensor-20k") == []
    assert controlled_contract_differences(candidate, reference) == []


def test_control_difference_fails_closed() -> None:
    candidate = metrics()
    reference = metrics(100_000.0)
    reference["attention_split_heads"] = 1

    assert controlled_contract_differences(candidate, reference) == [
        "attention_split_heads: candidate=8, reference=1"
    ]


def test_first_step_fingerprint(tmp_path) -> None:
    log = tmp_path / "run.out"
    log.write_text(
        "header\n"
        "step=1 loss=4.9 ce=1.492188 logit_kd=0.090411 "
        "attention_kd=0.043257 weighted_logit_kd=0.9\n",
        encoding="utf-8",
    )

    assert first_step_fingerprint(log) == {
        "ce": 1.492188,
        "logit_kd": 0.090411,
        "attention_kd": 0.043257,
    }
