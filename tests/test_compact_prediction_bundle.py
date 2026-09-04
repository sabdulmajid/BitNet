from __future__ import annotations

import json

import pytest

from benchmarks.build_compact_prediction_bundle import build_bundle, read_trace


def write_trace(path, predictions):
    rows = [
        {"index": index, "label": index % 2, "prediction": prediction, "correct": prediction == index % 2}
        for index, prediction in enumerate(predictions)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_build_bundle_preserves_aligned_predictions(tmp_path):
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    write_trace(left, [0, 0, 0])
    write_trace(right, [0, 1, 1])

    bundle = build_bundle([("left", left), ("right", right)], expected_examples=3)

    assert bundle["labels"] == [0, 1, 0]
    assert bundle["predictions"] == {"left": [0, 0, 0], "right": [0, 1, 1]}
    assert bundle["models"]["left"]["accuracy"] == pytest.approx(2 / 3)
    assert bundle["models"]["right"]["accuracy"] == pytest.approx(2 / 3)


def test_read_trace_rejects_noncontiguous_indices(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text('{"index":1,"label":0,"prediction":0,"correct":true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="expected contiguous index 0"):
        read_trace(path)
