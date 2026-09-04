from benchmarks.benchmark_seqcls_i2sr_runtime_ab import changed_source_paths


def build_contract(**sources: str) -> dict:
    return {
        "source_files": [
            {"path": path, "sha256": digest} for path, digest in sources.items()
        ]
    }


def test_changed_source_paths_reports_only_modified_files() -> None:
    baseline = build_contract(kernel="old", config="same")
    candidate = build_contract(kernel="new", config="same")

    assert changed_source_paths(baseline, candidate) == ["kernel"]


def test_changed_source_paths_reports_added_and_removed_files() -> None:
    baseline = build_contract(old="one", shared="same")
    candidate = build_contract(new="two", shared="same")

    assert changed_source_paths(baseline, candidate) == ["new", "old"]
