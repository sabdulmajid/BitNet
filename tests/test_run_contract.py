from __future__ import annotations

import argparse
import hashlib
import json

from train_bitdistill import build_run_contract, reference_fingerprint, write_run_contract


def make_args(tmp_path) -> argparse.Namespace:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    state = tmp_path / "state.pt"
    state.write_bytes(b"state")
    return argparse.Namespace(
        output_dir=str(tmp_path / "output"),
        write_run_contract=True,
        hash_input_artifacts=True,
        student_model=str(model),
        teacher_model="remote/teacher",
        init_state_dict=str(state),
        init_state_manifest="",
        seed=1234,
        lr_scheduler="cosine",
        warmup_steps=100,
        eval_batch_size=16,
    )


def test_reference_fingerprint_hashes_files_and_directories(tmp_path) -> None:
    file_path = tmp_path / "value.bin"
    file_path.write_bytes(b"value")
    directory = tmp_path / "directory"
    directory.mkdir()
    (directory / "a.txt").write_text("a", encoding="utf-8")

    file_result = reference_fingerprint(str(file_path), hash_contents=True)
    directory_result = reference_fingerprint(str(directory), hash_contents=True)

    assert file_result["sha256"] == hashlib.sha256(b"value").hexdigest()
    assert directory_result["file_count"] == 1
    assert len(directory_result["tree_sha256"]) == 64
    assert reference_fingerprint("org/model", hash_contents=True) == {
        "reference": "org/model",
        "local": False,
    }


def test_run_contract_is_written_before_training(tmp_path) -> None:
    arguments = make_args(tmp_path)

    contract = write_run_contract(arguments)
    path = tmp_path / "output" / "run_contract.json"
    stored = json.loads(path.read_text(encoding="utf-8"))

    assert contract == stored
    assert stored["schema"] == "bitdistill-run-contract-v1"
    assert stored["resolved_arguments"]["lr_scheduler"] == "cosine"
    assert stored["resolved_arguments"]["eval_batch_size"] == 16
    expected_state_hash = hashlib.sha256(b"state").hexdigest()
    assert stored["inputs"]["init_state_dict"]["sha256"] == expected_state_hash
    assert arguments._run_contract_reference["path"] == str(path)
    assert len(arguments._run_contract_reference["sha256"]) == 64


def test_build_contract_does_not_serialize_internal_fields(tmp_path) -> None:
    arguments = make_args(tmp_path)
    arguments._internal = "not public"

    contract = build_run_contract(arguments)

    assert "_internal" not in contract["resolved_arguments"]
