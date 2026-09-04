from __future__ import annotations

import hashlib
import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_exporter():
    path = ROOT / "benchmarks/convert_static_ternary_to_i2s_gguf.py"
    spec = importlib.util.spec_from_file_location("tl2sr_exporter_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tl2sr_packing_aligns_scales_after_weight_payload() -> None:
    exporter = load_exporter()
    rows = cols = 896
    raw_bytes = exporter.tl2_payload_nbytes(rows, cols)
    aligned_bytes = exporter.tl2_packed_nbytes(rows, cols)

    class FakeConverter:
        @staticmethod
        def preprocess_weights_tl2(weights, config_path):
            assert weights.shape == (rows, cols)
            assert config_path == Path("kernel.ini")
            return np.arange(raw_bytes, dtype=np.uint64).astype(np.uint8)

    codes = torch.zeros((rows, cols), dtype=torch.int8)
    scales = torch.linspace(0.001, 0.05, rows, dtype=torch.float32).reshape(rows, 1)
    packed = exporter.pack_tl2_sr(
        codes,
        scales,
        bitnet_converter=FakeConverter,
        kernel_config=Path("kernel.ini"),
    )

    assert raw_bytes == 176810
    assert aligned_bytes == 176832
    assert packed.size == aligned_bytes + rows * 4 + 32
    np.testing.assert_array_equal(packed[:raw_bytes], np.arange(raw_bytes, dtype=np.uint64).astype(np.uint8))
    assert np.count_nonzero(packed[raw_bytes:aligned_bytes]) == 0
    np.testing.assert_array_equal(
        packed[aligned_bytes : aligned_bytes + rows * 4].view(np.float32),
        scales.numpy().reshape(-1),
    )


def test_tl2sr_packing_rejects_unsupported_small_matrix() -> None:
    exporter = load_exporter()

    with pytest.raises(ValueError, match="at least 256 columns"):
        exporter.tl2_payload_nbytes(32, 128)


def test_tl2sr_qtype_ids_match_runtime_and_python_writer() -> None:
    ggml = (ROOT / "3rdparty/llama.cpp/ggml/include/ggml.h").read_text(encoding="utf-8")
    llama = (ROOT / "3rdparty/llama.cpp/include/llama.h").read_text(encoding="utf-8")
    constants = (ROOT / "3rdparty/llama.cpp/gguf-py/gguf/constants.py").read_text(encoding="utf-8")

    assert re.search(r"GGML_TYPE_TL2_SR\s*=\s*41", ggml)
    assert re.search(r"LLAMA_FTYPE_MOSTLY_TL2_SR\s*=\s*42", llama)
    assert re.search(r"TL2_SR\s*=\s*41", constants)
    assert re.search(r"MOSTLY_TL2_SR\s*=\s*42", constants)


def test_generated_tl2sr_kernel_uses_batch_and_row_scale_strides() -> None:
    header = (
        ROOT / "preset_kernels/Qwen2.5-0.5B-TL2SR/bitnet-lut-kernels-tl2sr.h"
    ).read_text(encoding="utf-8")
    runtime = (ROOT / "3rdparty/llama.cpp/ggml/src/ggml.c").read_text(encoding="utf-8")

    assert "const int output_idx = i + bs * 896;" in header
    assert "((float*)Scales)[i * scale_stride]" in header
    assert "GGML_TYPE_TL2_SR" in header
    assert "const int scale_stride = row_scale ? 1 : 0;" in runtime
    assert not re.search(r"two_qlut \+ bs512_num \* 512 \* two_k / 3", runtime)
    assert not re.search(r"bs256_num \* 256 \* two_k / 3", runtime)


def test_tl2sr_export_receipt_fingerprints_layout_inputs() -> None:
    source = (ROOT / "benchmarks/convert_static_ternary_to_i2s_gguf.py").read_text(
        encoding="utf-8"
    )

    assert '"tl2_kernel_config_sha256"' in source
    assert '"bitnet_converter_sha256"' in source


def test_tl2sr_runtime_rejects_kernel_layout_mismatch() -> None:
    runtime = (ROOT / "3rdparty/llama.cpp/src/llama.cpp").read_text(encoding="utf-8")
    generator = (ROOT / "utils/codegen_tl2.py").read_text(encoding="utf-8")

    assert 'ml.get_key("bitnet.tl2_sr.kernel_config_sha256"' in runtime
    assert "artifact_kernel_config != runtime_kernel_config" in runtime
    assert "TL2_SR requires a runtime built with GGML_BITNET_X86_TL2" in runtime
    assert "BITNET_TL2_KERNEL_CONFIG_SHA256" in generator


def test_generated_kernel_fingerprints_match_packing_configs() -> None:
    presets = (
        "Qwen2.5-0.5B-TL2SR",
        "Qwen2.5-0.5B-TL2SR-BM64",
        "Qwen2.5-0.5B-TL2SR-BM32",
    )
    for preset in presets:
        directory = ROOT / "preset_kernels" / preset
        header = (directory / "bitnet-lut-kernels-tl2sr.h").read_text(encoding="utf-8")
        config = (directory / "kernel_config_tl2sr.ini").read_bytes()
        match = re.search(r'BITNET_TL2_KERNEL_CONFIG_SHA256 "([0-9a-f]{64})"', header)

        assert match is not None
        assert match.group(1) == hashlib.sha256(config).hexdigest()
