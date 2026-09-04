from __future__ import annotations

from benchmarks.audit_bitdistill_adaptive_full import EXPECTED_SOURCE_REVISION
from benchmarks.audit_bitdistill_adaptive_vs_fixed import (
    decide_method,
    declared_contract_errors,
    parse_declared_contract,
    run_contract_errors,
)


def valid_metrics(*, arm: str, seed: int = 1234) -> dict:
    adaptive = arm == "adaptive"
    return {
        "source_revision": EXPECTED_SOURCE_REVISION,
        "seed": seed,
        "stage": "task_sft",
        "method": "bitdistill",
        "task": "mnli",
        "steps": 10_000,
        "eval": {"eval_examples": 9_815.0, "accuracy": 0.75},
        "task_format": "sequence_classification",
        "label_scheme": "letters",
        "candidate_score": "mean",
        "scale_mode": "tensor",
        "exclude_linear_regex": "score|classifier",
        "distill_layer": -1,
        "attention_split_heads": 1,
        "preparation": {
            "activation_quantization": True,
            "bitlinear_replaced": 168,
            "subln_inserted": 48,
        },
        "state_load": {"loaded": True, "path": "/local/run/assets/stage2.pt"},
        "output_head_init": {"copied": True},
        "training_budget": {
            "max_train_samples": 0,
            "max_eval_samples": 0,
            "max_seq_len": 512,
            "per_device_batch_size": 4,
            "grad_accum_steps": 4,
            "max_steps": 10_000,
        },
        "loss_weights": {
            "logit_kd_weight": 10.0,
            "attention_kd_weight": 100_000.0 if adaptive else 60.0,
            "attention_kd_balance": "gradnorm_ema" if adaptive else "fixed",
            "attention_balance_target_ratio": 1.0,
            "attention_balance_beta": 0.9,
            "attention_balance_every_steps": 20,
            "effective_attention_kd_weight": 25.0 if adaptive else 60.0,
            "logit_temperature": 5.0,
            "logit_kd_temperature_scale": "none",
            "attention_temperature": 1.0,
            "attention_relation_mode": "cosine",
            "attention_qkv_reduction": "sum",
        },
        "telemetry": {
            "every_steps": 500,
            "component_grad_norms": True,
            "max_elements_per_layer": 65_536,
        },
    }


def test_both_arm_contracts_are_explicit() -> None:
    assert run_contract_errors(valid_metrics(arm="adaptive"), arm="adaptive", seed=1234) == []
    assert run_contract_errors(valid_metrics(arm="fixed60"), arm="fixed60", seed=1234) == []

    invalid = valid_metrics(arm="fixed60")
    invalid["loss_weights"]["effective_attention_kd_weight"] = 59.0
    assert run_contract_errors(invalid, arm="fixed60", seed=1234)[-1] == (
        "effective_attention_kd_weight=59.0, expected=60.0"
    )


def test_declared_contract_parser_and_validation(tmp_path) -> None:
    log = tmp_path / "run.out"
    log.write_text(
        "SLURM_JOB_ID=10399\n"
        "MODEL=/local/a6abdulm/bitnet-b7fc773/assets/base_model\n"
        "step=1 loss=1.0\n",
        encoding="utf-8",
    )
    values, errors = parse_declared_contract(log)

    assert errors == []
    assert values == {
        "SLURM_JOB_ID": "10399",
        "MODEL": "/local/a6abdulm/bitnet-b7fc773/assets/base_model",
    }
    contract_errors = declared_contract_errors(values, arm="fixed60", seed=1234)
    assert not any(error.startswith("declared SLURM_JOB_ID") for error in contract_errors)
    assert any(error.startswith("declared TEACHER_MODEL") for error in contract_errors)


def test_preregistered_method_decision() -> None:
    clear_win = decide_method([0.006, 0.008, 0.010], complete=True)
    assert clear_win["adaptive_superiority_gate"] == "pass"
    assert clear_win["recommended_method"] == "adaptive"

    no_material_gain = decide_method([-0.003, -0.002, -0.001], complete=True)
    assert no_material_gain["fixed_simplicity_gate"] == "pass"
    assert no_material_gain["recommended_method"] == "fixed60"

    assert decide_method([], complete=False)["recommended_method"] == "pending"
