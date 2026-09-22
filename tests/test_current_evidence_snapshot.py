import argparse
import json
from pathlib import Path

import pytest

from benchmarks.build_current_evidence_snapshot import build_snapshot, require


ROOT = Path(__file__).resolve().parents[1]


def snapshot_args() -> argparse.Namespace:
    return argparse.Namespace(
        canonical=ROOT / "benchmarks/results/canonical_evidence_bundle_2026-05-20.json",
        matched_audit=ROOT
        / "benchmarks/results/bitdistill_adaptive_vs_fixed_matched_audit_2026-09-22.json",
        prediction_bundle=ROOT
        / "benchmarks/results/bitdistill_matched_prediction_bundle_2026-09-22.json",
        tl2_audit=ROOT / "benchmarks/results/tl2sr_evidence_audit_2026-09-04.json",
        cpu_matrix=ROOT
        / "benchmarks/results/seqcls_native_cpu_matrix_2026-09-04.json",
        cpu_repeated=ROOT
        / "benchmarks/results/seqcls_native_cpu_repeated_inplace_2026-09-04.json",
        initializer_audit=ROOT
        / "benchmark_results/second_order_ternary_init_2026-05-15.json",
        created_utc="2026-09-22T00:00:00+00:00",
    )


def test_current_snapshot_matches_completed_control() -> None:
    snapshot = build_snapshot(snapshot_args())
    matched = snapshot["verdicts"]["matched_bitdistill_mnli"]

    assert matched["status"] == "not_reproduced"
    assert matched["decision"] == "inconclusive"
    assert matched["adaptive_minus_fixed60"] == pytest.approx(-0.0016980811682798438)
    assert matched["adaptive_minus_fixed60_seed_ci95"] == pytest.approx(
        [-0.01089816023013155, 0.007501997893571862]
    )


def test_current_snapshot_is_public_and_hashed() -> None:
    snapshot = build_snapshot(snapshot_args())
    payload = json.dumps(snapshot, sort_keys=True)

    assert "/mnt/" not in payload
    assert "/local/" not in payload
    assert all(len(item["sha256"]) == 64 for item in snapshot["sources"].values())


def test_require_rejects_incomplete_report() -> None:
    with pytest.raises(ValueError, match="expected status"):
        require({"schema": "example", "status": "running"}, schema="example", status="complete")
