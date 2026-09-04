from __future__ import annotations

import math

from benchmarks.audit_bitdistill_adaptive_full import (
    EXPECTED_SOURCE_REVISION,
    run_contract_errors,
    seed_mean_ci,
    telemetry_health,
)


def test_seed_mean_interval_uses_seed_variation() -> None:
    interval = seed_mean_ci([0.79, 0.80, 0.81])

    assert interval is not None
    assert interval[0] < 0.80 < interval[1]
    assert math.isclose(interval[1] - 0.80, 0.80 - interval[0])


def test_telemetry_health_summarizes_training_contracts() -> None:
    rows = [
        {
            "loss": {"effective_attention_kd_weight": 100.0},
            "attention_balance": {
                "last": {"predicted_weighted_attention_to_ce_gradient_ratio": 0.8}
            },
            "component_grad_norms_microbatch": {
                "ce": 2.0,
                "weighted_attention_kd": 1.0,
                "weighted_logit_kd": 4.0,
            },
            "activation_quantization": {"clipped_fraction": 0.0, "int8_edge_fraction": 0.001},
            "quantization_dynamics": {"flip_fraction": None, "scale_abs_delta_max": None},
        },
        {
            "loss": {"effective_attention_kd_weight": 50.0},
            "attention_balance": {
                "last": {"predicted_weighted_attention_to_ce_gradient_ratio": 1.0}
            },
            "component_grad_norms_microbatch": {
                "ce": 4.0,
                "weighted_attention_kd": 4.0,
                "weighted_logit_kd": 2.0,
            },
            "activation_quantization": {"clipped_fraction": 0.0, "int8_edge_fraction": 0.002},
            "quantization_dynamics": {"flip_fraction": 0.02, "scale_abs_delta_max": 0.01},
        },
    ]

    health = telemetry_health(rows)

    assert health["points"] == 2
    assert health["attention_weight"]["first"] == 100.0
    assert health["attention_weight"]["final"] == 50.0
    assert health["weighted_attention_to_ce_gradient_ratio"]["median"] == 0.75
    assert health["weighted_logit_to_ce_gradient_ratio"]["final"] == 0.5
    assert health["probe_weighted_attention_to_ce_gradient_ratio"]["median"] == 0.9
    assert health["global_to_last_controller_probe_ratio"]["median"] == 0.8125
    assert health["global_to_last_controller_probe_ratio"]["comparison_contract"].startswith(
        "descriptive_only"
    )
    assert health["max_activation_clipped_fraction"] == 0.0
    assert health["mean_ternary_flip_fraction"] == 0.02


def valid_metrics() -> dict:
    return {
        "source_revision": EXPECTED_SOURCE_REVISION,
        "seed": 1234,
        "stage": "task_sft",
        "method": "bitdistill",
        "task": "mnli",
        "steps": 10_000,
        "eval": {"eval_examples": 9_815.0},
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
            "attention_kd_weight": 100_000.0,
            "attention_kd_balance": "gradnorm_ema",
            "attention_balance_target_ratio": 1.0,
            "attention_balance_beta": 0.9,
            "attention_balance_every_steps": 20,
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


def test_full_run_contract_is_fail_closed() -> None:
    metrics = valid_metrics()

    assert run_contract_errors(metrics, 1234) == []

    metrics["loss_weights"]["attention_qkv_reduction"] = "mean"
    assert run_contract_errors(metrics, 1234) == [
        "attention_qkv_reduction='mean', expected='sum'"
    ]
