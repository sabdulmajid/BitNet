from __future__ import annotations

import math

from benchmarks.audit_bitdistill_adaptive_full import seed_mean_ci


def test_seed_mean_interval_uses_seed_variation() -> None:
    interval = seed_mean_ci([0.79, 0.80, 0.81])

    assert interval is not None
    assert interval[0] < 0.80 < interval[1]
    assert math.isclose(interval[1] - 0.80, 0.80 - interval[0])
