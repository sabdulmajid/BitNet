from __future__ import annotations

import hashlib

from benchmarks.audit_tl2sr_evidence import (
    audit_status,
    conversion_output_identity,
    paired_accuracy_delta_ci95,
)


def test_conversion_output_identity_accepts_matching_declared_hash(tmp_path):
    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(b"row-scale-ternary")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()

    identity = conversion_output_identity(
        {"outfile": artifact.name, "outfile_sha256": digest}, tmp_path
    )

    assert identity["sha256"] == digest
    assert identity["path"] == "model.gguf"
    assert identity["declared_sha256_present"] is True
    assert identity["declared_sha256_matches"] is True


def test_conversion_output_identity_flags_stale_receipt(tmp_path):
    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(b"current")

    identity = conversion_output_identity(
        {"outfile": str(artifact), "outfile_sha256": "0" * 64}, tmp_path
    )

    assert identity["declared_sha256_present"] is True
    assert identity["declared_sha256_matches"] is False


def test_conversion_output_identity_supports_legacy_receipt(tmp_path):
    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(b"legacy")

    identity = conversion_output_identity({"outfile": artifact.name}, tmp_path)

    assert identity["declared_sha256_present"] is False
    assert identity["declared_sha256_matches"] is True


def test_audit_status_fails_closed_and_names_speed_result():
    assert audit_status(False, False) == "review"
    assert audit_status(False, True) == "review"
    assert audit_status(True, False) == "valid_runtime_no_speed_win"
    assert audit_status(True, True) == "valid_runtime_speed_win"


def test_paired_accuracy_delta_interval_collapses_without_disagreement():
    assert paired_accuracy_delta_ci95(total=100, left_only=0, right_only=0) == [0.0, 0.0]
