from __future__ import annotations

import math
import unittest

import torch
import torch.nn.functional as F

from train_bitdistill import (
    GradNormEmaBalancer,
    attention_relation_distillation_components,
    component_gradient_geometry,
    relation_rows,
)


def reference_relation_rows(
    values: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    split_heads: int,
    temperature: float,
    relation_mode: str,
) -> torch.Tensor:
    batch, seq_len, channels = values.shape
    width = channels // split_heads
    states = values.float().reshape(batch, seq_len, split_heads, width).transpose(1, 2)
    if relation_mode == "cosine":
        states = F.normalize(states, dim=-1)
        logits = states @ states.transpose(-2, -1) / temperature
    elif relation_mode == "scaled_dot":
        logits = states @ states.transpose(-2, -1) / (math.sqrt(width) * temperature)
    else:
        raise AssertionError(relation_mode)
    logits = logits.masked_fill(~attention_mask[:, None, None, :].bool(), -1.0e4)
    probabilities = F.softmax(logits, dim=-1).clamp_min(1.0e-8)
    query_mask = attention_mask[:, None, :].expand(batch, split_heads, seq_len).reshape(-1).bool()
    return probabilities.reshape(batch * split_heads * seq_len, seq_len)[query_mask]


class AttentionRelationContractTest(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(20260904)
        self.student = {
            key: torch.randn(2, 7, channels, generator=generator, dtype=torch.float64)
            for key, channels in (("q", 32), ("k", 16), ("v", 16))
        }
        self.teacher = {
            key: torch.randn(2, 7, channels, generator=generator, dtype=torch.float64)
            for key, channels in (("q", 32), ("k", 16), ("v", 16))
        }
        self.mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0]])

    def test_relation_modes_match_direct_references(self) -> None:
        for relation_mode in ("cosine", "scaled_dot"):
            actual = relation_rows(
                self.student["q"],
                self.mask,
                split_heads=4,
                temperature=1.7,
                relation_mode=relation_mode,
            )
            expected = reference_relation_rows(
                self.student["q"],
                self.mask,
                split_heads=4,
                temperature=1.7,
                relation_mode=relation_mode,
            )
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    def test_batch_duplication_does_not_change_batchmean_loss(self) -> None:
        base, _ = attention_relation_distillation_components(
            self.student,
            self.teacher,
            self.mask,
            split_heads=4,
            temperature=1.0,
            qkv_reduction="sum",
            relation_mode="cosine",
        )
        duplicated_student = {key: value.repeat(2, 1, 1) for key, value in self.student.items()}
        duplicated_teacher = {key: value.repeat(2, 1, 1) for key, value in self.teacher.items()}
        duplicated, _ = attention_relation_distillation_components(
            duplicated_student,
            duplicated_teacher,
            self.mask.repeat(2, 1),
            split_heads=4,
            temperature=1.0,
            qkv_reduction="sum",
            relation_mode="cosine",
        )
        torch.testing.assert_close(base, duplicated, rtol=1e-6, atol=1e-7)

    def test_masked_right_padding_does_not_change_loss(self) -> None:
        base, _ = attention_relation_distillation_components(
            self.student,
            self.teacher,
            self.mask,
            split_heads=4,
            temperature=1.0,
            qkv_reduction="sum",
            relation_mode="cosine",
        )
        padded_student = {key: F.pad(value, (0, 0, 0, 3)) for key, value in self.student.items()}
        padded_teacher = {key: F.pad(value, (0, 0, 0, 3)) for key, value in self.teacher.items()}
        padded_mask = F.pad(self.mask, (0, 3))
        padded, _ = attention_relation_distillation_components(
            padded_student,
            padded_teacher,
            padded_mask,
            split_heads=4,
            temperature=1.0,
            qkv_reduction="sum",
            relation_mode="cosine",
        )
        torch.testing.assert_close(base, padded, rtol=1e-5, atol=1e-7)

    def test_cosine_is_scale_invariant_but_scaled_dot_is_not(self) -> None:
        values = self.student["q"]
        scaled_values = values * torch.linspace(0.25, 4.0, values.shape[1]).view(1, -1, 1)
        cosine = relation_rows(values, self.mask, split_heads=1, temperature=1.0, relation_mode="cosine")
        cosine_scaled = relation_rows(
            scaled_values,
            self.mask,
            split_heads=1,
            temperature=1.0,
            relation_mode="cosine",
        )
        dot = relation_rows(values, self.mask, split_heads=1, temperature=1.0, relation_mode="scaled_dot")
        dot_scaled = relation_rows(
            scaled_values,
            self.mask,
            split_heads=1,
            temperature=1.0,
            relation_mode="scaled_dot",
        )
        torch.testing.assert_close(cosine, cosine_scaled, rtol=1e-6, atol=1e-7)
        self.assertGreater(float(torch.max(torch.abs(dot - dot_scaled))), 1e-3)

    def test_gqa_kv_repetition_is_neutral_only_for_cosine_split1(self) -> None:
        batch, seq_len, kv_heads, repeat_factor, head_dim = 2, 7, 2, 7, 16
        values = self.student["q"].reshape(batch, seq_len, kv_heads, -1)
        self.assertEqual(values.shape[-1], head_dim)
        repeated = values.repeat_interleave(repeat_factor, dim=2).reshape(batch, seq_len, -1)
        flat = values.reshape(batch, seq_len, -1)

        cosine = relation_rows(flat, self.mask, split_heads=1, temperature=1.0, relation_mode="cosine")
        repeated_cosine = relation_rows(
            repeated,
            self.mask,
            split_heads=1,
            temperature=1.0,
            relation_mode="cosine",
        )
        scaled_dot = relation_rows(flat, self.mask, split_heads=1, temperature=1.0, relation_mode="scaled_dot")
        repeated_scaled_dot = relation_rows(
            repeated,
            self.mask,
            split_heads=1,
            temperature=1.0,
            relation_mode="scaled_dot",
        )

        torch.testing.assert_close(cosine, repeated_cosine, rtol=1e-6, atol=1e-7)
        self.assertGreater(float(torch.max(torch.abs(scaled_dot - repeated_scaled_dot))), 1e-6)

    def test_invalid_relation_mode_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported relation_mode"):
            relation_rows(
                self.student["q"],
                self.mask,
                split_heads=1,
                temperature=1.0,
                relation_mode="unknown",
            )

    def test_gradient_balancer_targets_ratio_and_uses_log_ema(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        balancer = GradNormEmaBalancer(
            initial_weight=100000.0,
            target_ratio=1.0,
            beta=0.5,
            min_weight=1e-3,
            max_weight=1e5,
            eps=1e-12,
        )
        first = balancer.update((parameter - 3.0) ** 2, (5.0 * parameter) ** 2, [parameter])
        self.assertAlmostEqual(first["ce_gradient_norm"], 4.0)
        self.assertAlmostEqual(first["raw_attention_gradient_norm"], 50.0)
        self.assertAlmostEqual(first["effective_weight"], 0.08)
        self.assertAlmostEqual(first["predicted_weighted_attention_to_ce_gradient_ratio"], 1.0)

        second = balancer.update((parameter - 3.0) ** 2, parameter**2, [parameter])
        self.assertAlmostEqual(second["candidate_weight"], 2.0)
        self.assertAlmostEqual(second["effective_weight"], math.sqrt(0.08 * 2.0))

    def test_gradient_geometry_distinguishes_orthogonal_and_opposed_objectives(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
        geometry = component_gradient_geometry(
            {
                "ce": parameter[0],
                "orthogonal": parameter[1],
                "opposed": -2.0 * parameter[0],
                "constant": torch.tensor(0.0),
            },
            [parameter],
        )

        self.assertEqual(
            geometry["norms"],
            {"ce": 1.0, "orthogonal": 1.0, "opposed": 2.0, "constant": 0.0},
        )
        self.assertAlmostEqual(geometry["cosines"]["ce__orthogonal"], 0.0)
        self.assertAlmostEqual(geometry["cosines"]["ce__opposed"], -1.0)
        self.assertAlmostEqual(geometry["cosines"]["orthogonal__opposed"], 0.0)
        self.assertIsNone(geometry["cosines"]["ce__constant"])


if __name__ == "__main__":
    unittest.main()
