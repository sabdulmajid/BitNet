from pathlib import Path

from benchmarks.audit_bitdistill_loss_contract import latest_controlled_curve


def write_curve(root: Path, directory: str, date: str) -> Path:
    path = root / directory / f"bitdistill_controlled_curve_{date}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")
    return path


def test_latest_controlled_curve_uses_latest_date(tmp_path: Path) -> None:
    write_curve(tmp_path, "benchmarks/results", "2026-05-20")
    expected = write_curve(tmp_path, "benchmark_results", "2026-05-23")

    assert latest_controlled_curve(tmp_path) == expected


def test_latest_controlled_curve_prefers_public_copy_for_same_date(tmp_path: Path) -> None:
    write_curve(tmp_path, "benchmark_results", "2026-05-23")
    expected = write_curve(tmp_path, "benchmarks/results", "2026-05-23")

    assert latest_controlled_curve(tmp_path) == expected
