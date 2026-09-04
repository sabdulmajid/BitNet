from __future__ import annotations

import math

from benchmarks.audit_bitdistill_adaptive_full import seed_mean_ci, telemetry_health


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
    assert health["global_to_probe_attention_gradient_ratio"]["median"] == 0.8125
    assert health["max_activation_clipped_fraction"] == 0.0
    assert health["mean_ternary_flip_fraction"] == 0.02
