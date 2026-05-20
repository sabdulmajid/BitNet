#!/usr/bin/env python3
"""Reject benchmark reports that silently summarize zero expected rows."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ZERO_FRACTION_RE = re.compile(r"\b(?:complete|passed[^|]*)\s*\|\s*0/0\b", re.IGNORECASE)


def json_allows_empty(data: dict[str, Any]) -> bool:
    return bool(data.get("allow_empty") or data.get("empty_expected_reason"))


def validate_json(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return []
    expected = data.get("expected")
    complete = data.get("complete")
    rows = data.get("rows")
    errors: list[str] = []
    if expected == 0 and complete == 0 and rows == [] and not json_allows_empty(data):
        errors.append("silent empty report: expected=0 complete=0 rows=[] without empty_expected_reason")
    return errors


def validate_markdown(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if ZERO_FRACTION_RE.search(text) and "empty_expected_reason" not in text and "No rows expected" not in text:
        return ["silent 0/0 summary without an explicit empty-report reason"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    failures: list[str] = []
    for path in args.paths:
        if not path.exists():
            failures.append(f"{path}: missing")
            continue
        if path.suffix == ".json":
            errors = validate_json(path)
        elif path.suffix in {".md", ".markdown"}:
            errors = validate_markdown(path)
        else:
            errors = []
        failures.extend(f"{path}: {error}" for error in errors)

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        return 1
    print(f"validated {len(args.paths)} report(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
