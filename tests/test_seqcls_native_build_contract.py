from benchmarks.benchmark_seqcls_native_i2sr_cpu import normalize_ldd_output


def test_ldd_normalization_removes_aslr_addresses() -> None:
    first = "libggml.so => /tmp/libggml.so (0x00007fa6c9826000)"
    second = "libggml.so => /tmp/libggml.so (0x00007fdcbd3f8000)"

    assert normalize_ldd_output(first) == normalize_ldd_output(second)
    assert normalize_ldd_output(first).endswith("(0xADDR)")
