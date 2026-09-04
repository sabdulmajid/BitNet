from pathlib import Path

import pytest

from benchmarks.benchmark_seqcls_native_cpu_repeated import (
    cpu_utilization,
    mean_ci95,
    monitored_cpu_set,
    parse_cpu_list,
    read_cpu_snapshot,
    summarize_ratios,
)


def test_mean_interval_collapses_for_constant_measurements() -> None:
    assert mean_ci95([3.0, 3.0, 3.0, 3.0]) == [3.0, 3.0]


def test_mean_interval_uses_student_t_for_six_measurements() -> None:
    interval = mean_ci95([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    assert interval == pytest.approx([1.5364, 5.4636], abs=1e-4)


def test_speed_ratio_is_paired_by_repetition() -> None:
    summary = summarize_ratios([20.0, 40.0, 80.0, 160.0], [10.0, 20.0, 40.0, 80.0])

    assert summary["paired_ratios"] == [2.0, 2.0, 2.0, 2.0]
    assert summary["geometric_mean"] == pytest.approx(2.0)
    assert summary["geometric_mean_ci95_t"] == pytest.approx([2.0, 2.0])


def test_parse_cpu_list_supports_ranges_and_deduplicates() -> None:
    assert parse_cpu_list("0-2,2,4,6-7") == [0, 1, 2, 4, 6, 7]


def test_monitored_cpu_set_includes_hyperthread_siblings(tmp_path: Path) -> None:
    for cpu, siblings in ((0, "0,12"), (1, "1,13")):
        path = tmp_path / f"cpu{cpu}" / "topology"
        path.mkdir(parents=True)
        (path / "thread_siblings_list").write_text(siblings, encoding="utf-8")

    assert monitored_cpu_set("0-1", tmp_path) == [0, 1, 12, 13]


def test_cpu_snapshot_and_utilization(tmp_path: Path) -> None:
    before_path = tmp_path / "before"
    after_path = tmp_path / "after"
    before_path.write_text("cpu 0 0 0 0 0\ncpu0 10 0 10 80 0\ncpu1 20 0 10 70 0\n", encoding="utf-8")
    after_path.write_text("cpu 0 0 0 0 0\ncpu0 20 0 20 160 0\ncpu1 40 0 20 140 0\n", encoding="utf-8")

    before = read_cpu_snapshot([0, 1], before_path)
    after = read_cpu_snapshot([0, 1], after_path)

    assert cpu_utilization(before, after) == pytest.approx({0: 0.2, 1: 0.3})
