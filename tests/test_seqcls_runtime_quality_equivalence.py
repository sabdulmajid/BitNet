from __future__ import annotations

import math

from benchmarks.audit_seqcls_runtime_quality_equivalence import (
    paired_statistics,
    validate_comparison_rows,
)


def test_paired_statistics_keep_prediction_and_quality_agreement_separate() -> None:
    rows = [
        {"index": 0, "label": 0, "pytorch_prediction": 0, "runtime_prediction": 0},
        {"index": 1, "label": 0, "pytorch_prediction": 1, "runtime_prediction": 0},
        {"index": 2, "label": 0, "pytorch_prediction": 0, "runtime_prediction": 1},
        {"index": 3, "label": 0, "pytorch_prediction": 1, "runtime_prediction": 2},
    ]

    stats = paired_statistics(rows, bootstrap_samples=100, seed=7)

    assert stats["runtime_wins"] == 1
    assert stats["pytorch_wins"] == 1
    assert stats["runtime_accuracy"] == stats["pytorch_accuracy"] == 0.5
    assert stats["quality_agreement"] == 0.5
    assert stats["prediction_agreement"] == 0.25
    assert stats["mcnemar_exact_p"] == 1.0
    assert math.isclose(stats["delta_runtime_minus_pytorch"], 0.0)


def test_trace_validation_fails_on_runtime_prediction_drift() -> None:
    rows = [{"index": 0, "label": 0, "pytorch_prediction": 0, "runtime_prediction": 1}]
    runtime = {"predictions": [2]}

    errors = validate_comparison_rows(rows, runtime)

    assert "comparison rows=1, expected=9815" in errors
    assert "runtime prediction mismatch at index 0" in errors
