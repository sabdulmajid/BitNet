from pathlib import Path

from benchmarks.benchmark_seqcls_native_i2sr_cpu import normalize_ldd_output, normalize_repo_paths


def test_ldd_normalization_removes_aslr_addresses() -> None:
    first = "libggml.so => /tmp/libggml.so (0x00007fa6c9826000)"
    second = "libggml.so => /tmp/libggml.so (0x00007fdcbd3f8000)"

    assert normalize_ldd_output(first) == normalize_ldd_output(second)
    assert normalize_ldd_output(first).endswith("(0xADDR)")


def test_public_provenance_replaces_checkout_path() -> None:
    root = Path("/private/cluster/user/BitNet")
    command = f"c++ -I{root}/include -c {root}/src/kernel.cpp"

    normalized = normalize_repo_paths(command, root)

    assert normalized == "c++ -I$REPO_ROOT/include -c $REPO_ROOT/src/kernel.cpp"
    assert "/private/cluster/user" not in normalized
