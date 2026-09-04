import pytest

from benchmarks.benchmark_seqcls_native_cpu_repeated import mean_ci95, summarize_ratios


def test_mean_interval_collapses_for_constant_measurements() -> None:
    assert mean_ci95([3.0, 3.0, 3.0, 3.0]) == [3.0, 3.0]


def test_speed_ratio_is_paired_by_repetition() -> None:
    summary = summarize_ratios([20.0, 40.0, 80.0, 160.0], [10.0, 20.0, 40.0, 80.0])

    assert summary["paired_ratios"] == [2.0, 2.0, 2.0, 2.0]
    assert summary["geometric_mean"] == pytest.approx(2.0)
    assert summary["geometric_mean_ci95_t"] == pytest.approx([2.0, 2.0])
