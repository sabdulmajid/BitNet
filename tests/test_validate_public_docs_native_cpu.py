from benchmarks.validate_public_docs import (
    validate_i2_kernel_profile_docs,
    validate_native_cpu_docs,
)


def evidence() -> tuple[dict, dict, dict]:
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
            "i2_sr_student": {"geometric_mean": 0.6495997},
            "i2_sr_q8_embedding_student": {"geometric_mean": 0.6046759},
        },
    }
    runtime_ab = {
        "schema": "seqcls-i2sr-runtime-ab-v1",
        "status": "valid",
        "errors": [],
        "source_differences": ["3rdparty/llama.cpp/ggml/src/ggml.c"],
        "summaries": {
            "i2_sr_student": {
                "candidate_over_baseline": {"geometric_mean": 1.4619},
                "numeric_equivalence": {
                    "max_abs_logit_difference": 0.0,
                    "predictions_identical": True,
                },
            },
            "i2_sr_q8_embedding_student": {
                "candidate_over_baseline": {"geometric_mean": 1.4358},
                "numeric_equivalence": {
                    "max_abs_logit_difference": 0.0,
                    "predictions_identical": True,
                },
            },
        },
    }
    return matrix, repeated, runtime_ab


def test_native_cpu_claims_match_evidence() -> None:
    matrix, repeated, runtime_ab = evidence()
    claims = "230.90 1.527 -0.001953 0.982422 0.650 0.605 1.4619 1.4358"
    readme = (
        claims
        + " seqcls_native_cpu_matrix_2026-09-04.md"
        + " seqcls_native_cpu_repeated_inplace_2026-09-04.md"
        + " seqcls_i2sr_runtime_ab_2026-09-04.md"
    )
    experiments = (
        "python benchmarks/audit_seqcls_native_cpu_matrix.py\n"
        "python benchmarks/benchmark_seqcls_native_cpu_repeated.py\n"
        "python benchmarks/benchmark_seqcls_i2sr_runtime_ab.py"
    )
    errors: list[str] = []

    validate_native_cpu_docs(
        matrix, repeated, runtime_ab, readme, claims, experiments, errors
    )

    assert errors == []


def test_native_cpu_claims_reject_invalid_timing_report() -> None:
    matrix, repeated, runtime_ab = evidence()
    repeated["status"] = "invalid"
    errors: list[str] = []

    validate_native_cpu_docs(matrix, repeated, runtime_ab, "", "", "", errors)

    assert "native CPU repeated: expected valid timing evidence with no errors" in errors


def test_i2_kernel_profile_claims_match_evidence() -> None:
    report = {
        "schema": "i2-kernel-profile-v1",
        "status": "valid",
        "errors": [],
        "maximum_abs_error": 0.0,
        "repositories": {
            "bitnet": {"tracked_files_dirty": False},
            "llama_cpp": {"tracked_files_dirty": False},
        },
        "aggregate_projection_mix": {
            "quantize_fraction": {
                "mean": 0.054911,
                "mean_ci95_t": [0.052924, 0.056898],
            },
            "ideal_speedup_if_quantization_were_free": {"mean": 1.058106},
        },
    }
    claims = "5.49% 5.29% 5.69% 94.51% 1.0581x"
    errors: list[str] = []

    validate_i2_kernel_profile_docs(
        report,
        claims + " i2_kernel_profile_2026-09-04.md",
        claims,
        "python benchmarks/benchmark_i2_kernel_profile.py",
        errors,
    )

    assert errors == []


def test_i2_kernel_profile_rejects_nonzero_reference_error() -> None:
    report = {
        "schema": "i2-kernel-profile-v1",
        "status": "valid",
        "errors": [],
        "maximum_abs_error": 1.0,
        "repositories": {},
        "aggregate_projection_mix": {
            "quantize_fraction": {
                "mean": 0.054911,
                "mean_ci95_t": [0.052924, 0.056898],
            },
            "ideal_speedup_if_quantization_were_free": {"mean": 1.058106},
        },
    }
    errors: list[str] = []

    validate_i2_kernel_profile_docs(report, "", "", "", errors)

    assert "I2 kernel profile: scalar-reference error is nonzero" in errors
