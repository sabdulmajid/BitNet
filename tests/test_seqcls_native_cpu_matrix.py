from __future__ import annotations

from copy import deepcopy

from benchmarks.audit_seqcls_native_cpu_matrix import paired_quality, validate_matrix


def artifact(predictions: list[int]) -> dict:
    labels = [0, 0, 0, 0]
    accuracy = sum(int(pred == label) for pred, label in zip(predictions, labels)) / len(labels)
    return {
        "schema": "seqcls_native_cpu.v2",
        "task": "mnli",
        "max_samples": 4,
        "prompt_input": "token_ids",
        "prompt_batch_size": 4,
        "embedding_sequential": True,
        "batch_size": 4096,
        "ubatch_size": 512,
        "labels": labels,
        "predictions": predictions,
        "summary": {"accuracy": accuracy},
        "runtime_build": {
            "sha256": "build",
            "repositories": {
                "bitnet": {"tracked_files_dirty": False},
                "llama_cpp": {"tracked_files_dirty": False},
            },
        },
        "hardware": {
            "cpu_model": "test-cpu",
            "requested_threads": 4,
            "logical_cpus_cpuinfo": 4,
            "physical_cores_cpuinfo": 2,
        },
    }


def test_paired_quality_uses_example_level_discordance() -> None:
    stats = paired_quality(
        candidate=[0, 1, 0, 1],
        reference=[0, 0, 1, 1],
        labels=[0, 0, 0, 0],
        bootstrap_samples=100,
        seed=7,
    )

    assert stats["candidate_accuracy"] == stats["reference_accuracy"] == 0.5
    assert stats["candidate_wins"] == stats["reference_wins"] == 1
    assert stats["delta_candidate_minus_reference"] == 0.0
    assert stats["prediction_agreement"] == 0.5


def test_matrix_validation_rejects_runtime_build_mismatch() -> None:
    reference = artifact([0, 0, 0, 0])
    q4 = deepcopy(reference)
    ternary = deepcopy(reference)
    ternary["runtime_build"]["sha256"] = "different-build"

    errors = validate_matrix(
        {
            "fp16_teacher": reference,
            "q4_0_teacher": q4,
            "i2_sr_student": ternary,
        }
    )

    assert "i2_sr_student: runtime build contract differs from fp16_teacher" in errors


def test_matrix_validation_rejects_dirty_source() -> None:
    reference = artifact([0, 0, 0, 0])
    q4 = deepcopy(reference)
    ternary = deepcopy(reference)
    q4["runtime_build"]["repositories"]["llama_cpp"]["tracked_files_dirty"] = True

    errors = validate_matrix(
        {
            "fp16_teacher": reference,
            "q4_0_teacher": q4,
            "i2_sr_student": ternary,
        }
    )

    assert "q4_0_teacher: llama_cpp tracked source was dirty during benchmark" in errors
