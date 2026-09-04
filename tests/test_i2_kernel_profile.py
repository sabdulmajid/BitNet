import json

import pytest

from benchmarks.benchmark_i2_kernel_profile import (
    aggregate_projection_mix,
    parse_profile_output,
    public_compile_command,
    render_markdown,
    summarize,
)


def profile_rows(value: float = 10.0) -> list[dict]:
    return [
        {
            "input": input_size,
            "output": output_size,
            "tokens": 32,
            "quantize_us": value,
            "multiply_us": 9.0 * value,
            "quantize_fraction": 0.1,
            "checksum": 1.0,
            "max_abs_error": 0.0,
        }
        for input_size, output_size in ((896, 896), (896, 128), (896, 4864), (4864, 896))
    ]


def test_parse_profile_output_enforces_shapes_and_exactness() -> None:
    output = "\n".join(json.dumps(row) for row in profile_rows())
    assert len(parse_profile_output(output)) == 4

    bad_rows = profile_rows()
    bad_rows[0]["max_abs_error"] = 1.0
    with pytest.raises(ValueError, match="kernel mismatch"):
        parse_profile_output("\n".join(json.dumps(row) for row in bad_rows))


def test_aggregate_projection_mix_applies_qwen_use_counts() -> None:
    aggregate = aggregate_projection_mix(profile_rows())
    assert aggregate["quantize_us"] == pytest.approx(70.0)
    assert aggregate["multiply_us"] == pytest.approx(630.0)
    assert aggregate["quantize_fraction"] == pytest.approx(0.1)
    assert aggregate["ideal_speedup_if_quantization_were_free"] == pytest.approx(1.0 / 0.9)


def test_summary_and_markdown_report_measured_values() -> None:
    summary = summarize([0.05, 0.06, 0.055, 0.052, 0.058])
    report = {
        "created_utc": "2026-09-04T00:00:00+00:00",
        "status": "valid",
        "outer_repetitions": 5,
        "inner_iterations": 31,
        "tokens": 32,
        "cpu_affinity": "0",
        "shape_summaries": [
            {
                "input": 896,
                "output": 896,
                "multiplicity": 2,
                "quantize_us": summarize([40.0] * 5),
                "multiply_us": summarize([400.0] * 5),
                "quantize_fraction": summarize([0.1] * 5),
            }
        ],
        "aggregate_projection_mix": {
            "quantize_fraction": summary,
            "ideal_speedup_if_quantization_were_free": summarize([1.06] * 5),
        },
        "maximum_abs_error": 0.0,
        "interpretation": "Kernel arithmetic dominates.",
    }
    markdown = render_markdown(report)
    assert "5.50%" in markdown
    assert "1.0600x" in markdown
    assert "Kernel arithmetic dominates." in markdown


def test_public_compile_command_removes_temporary_output(tmp_path) -> None:
    root = tmp_path / "checkout"
    command = ["c++", str(root / "bench.cpp"), "-o", "/tmp/profile"]
    assert public_compile_command(command, root) == ["c++", "$REPO_ROOT/bench.cpp"]
