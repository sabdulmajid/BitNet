from benchmarks.validate_public_docs import validate_native_cpu_docs


def evidence() -> tuple[dict, dict]:
    matrix = {
        "schema": "seqcls-native-cpu-matrix-v1",
        "status": "valid_sample_matrix",
        "errors": [],
        "artifacts": {"i2_sr_q8_embedding_student": {"gguf_mib": 230.903564}},
        "comparisons": {
            "i2_sr_q8_embedding_student_vs_i2_sr_student": {
                "quality": {
                    "delta_candidate_minus_reference": -0.001953125,
                    "prediction_agreement": 0.982421875,
                },
                "system": {"size_ratio_reference_over_candidate": 1.5271199},
            }
        },
    }
    repeated = {
        "schema": "seqcls-native-cpu-repeated-v1",
        "status": "valid",
        "errors": [],
        "paired_speed_ratios_vs_fp16": {
            "i2_sr_student": {"geometric_mean": 0.6371295},
            "i2_sr_q8_embedding_student": {"geometric_mean": 0.5280768},
        },
    }
    return matrix, repeated


def test_native_cpu_claims_match_evidence() -> None:
    matrix, repeated = evidence()
    claims = "230.90 1.527 -0.001953 0.982422 0.637 0.528"
    readme = (
        claims
        + " seqcls_native_cpu_matrix_2026-09-04.md"
        + " seqcls_native_cpu_repeated_2026-09-04.md"
    )
    experiments = (
        "python benchmarks/audit_seqcls_native_cpu_matrix.py\n"
        "python benchmarks/benchmark_seqcls_native_cpu_repeated.py"
    )
    errors: list[str] = []

    validate_native_cpu_docs(matrix, repeated, readme, claims, experiments, errors)

    assert errors == []


def test_native_cpu_claims_reject_invalid_timing_report() -> None:
    matrix, repeated = evidence()
    repeated["status"] = "invalid"
    errors: list[str] = []

    validate_native_cpu_docs(matrix, repeated, "", "", "", errors)

    assert "native CPU repeated: expected valid timing evidence with no errors" in errors
