from benchmarks.submit_bitdistill_afterany_postprocess import markdown_cell


def test_markdown_cell_escapes_table_separators() -> None:
    assert markdown_cell("Dependency|Resources") == "Dependency\\|Resources"
