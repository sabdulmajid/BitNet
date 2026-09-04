from __future__ import annotations

import unittest

import numpy as np

from benchmarks.audit_bitdistill_stage2_saturation import geometric_projection, paired_bootstrap


class Stage2SaturationAuditTest(unittest.TestCase):
    def test_geometric_projection_recovers_known_limit(self) -> None:
        # Gains contract by one half: 0.1, 0.05, then 0.025, ...
        projection = geometric_projection(
            np.asarray([0.6, 0.7, 0.75]),
            current_tokens=4,
            target_tokens=16,
        )

        self.assertIsNotNone(projection)
        contraction, asymptote, target = projection or (0.0, 0.0, 0.0)
        self.assertAlmostEqual(contraction, 0.5)
        self.assertAlmostEqual(asymptote, 0.8)
        self.assertAlmostEqual(target, 0.7875)

    def test_noncontracting_curve_is_rejected(self) -> None:
        self.assertIsNone(
            geometric_projection(
                np.asarray([0.6, 0.65, 0.71]),
                current_tokens=4,
                target_tokens=16,
            )
        )

    def test_paired_bootstrap_is_deterministic(self) -> None:
        correctness = np.asarray(
            [
                [0, 0, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=np.float64,
        )
        first = paired_bootstrap(
            correctness,
            samples=200,
            seed=7,
            current_tokens=4,
            target_tokens=16,
            batch_size=16,
        )
        second = paired_bootstrap(
            correctness,
            samples=200,
            seed=7,
            current_tokens=4,
            target_tokens=16,
            batch_size=16,
        )

        self.assertEqual(first, second)
        self.assertGreater(first["valid_samples"], 0)


if __name__ == "__main__":
    unittest.main()
