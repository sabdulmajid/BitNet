#!/usr/bin/env python3
"""Split the monolithic I2_SR patch into root and llama.cpp-submodule patches."""

from __future__ import annotations

import argparse
from pathlib import Path


SUBMODULE_PREFIX = "3rdparty/llama.cpp/"
ROOT_RUNTIME_PATH = "src/ggml-bitnet-mad.cpp"


def patch_segments(text: str) -> list[list[str]]:
    segments: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines(keepends=True):
        if line.startswith("diff --git ") and current:
            segments.append(current)
            current = []
        current.append(line)
    if current:
        segments.append(current)
    return segments


def segment_path(segment: list[str]) -> str:
    if not segment or not segment[0].startswith("diff --git "):
        return ""
    parts = segment[0].strip().split()
    if len(parts) < 4 or not parts[2].startswith("a/"):
        return ""
    return parts[2][2:]


def strip_submodule_prefix(segment: list[str]) -> list[str]:
    stripped: list[str] = []
    for line in segment:
        if line.startswith("diff --git "):
            stripped.append(line.replace(f"a/{SUBMODULE_PREFIX}", "a/").replace(f"b/{SUBMODULE_PREFIX}", "b/"))
        elif line.startswith("--- a/") or line.startswith("+++ b/"):
            stripped.append(line.replace(f"a/{SUBMODULE_PREFIX}", "a/").replace(f"b/{SUBMODULE_PREFIX}", "b/"))
        else:
            stripped.append(line)
    return stripped


def split_patch(text: str) -> tuple[str, str]:
    root_segments: list[str] = []
    submodule_segments: list[str] = []
    for segment in patch_segments(text):
        path = segment_path(segment)
        if path == ROOT_RUNTIME_PATH:
            root_segments.extend(segment)
        elif path.startswith(SUBMODULE_PREFIX):
            submodule_segments.extend(strip_submodule_prefix(segment))
    return "".join(root_segments), "".join(submodule_segments)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("patches/llama-i2sr-row-scale-qtype.patch"))
    parser.add_argument("--root-output", type=Path, default=Path("patches/bitnet-i2sr-root-runtime.patch"))
    parser.add_argument("--submodule-output", type=Path, default=Path("patches/llama-i2sr-row-scale-qtype.submodule.patch"))
    args = parser.parse_args()

    text = args.input.read_text(encoding="utf-8")
    root_patch, submodule_patch = split_patch(text)
    if not root_patch:
        raise SystemExit("root runtime patch segment not found")
    if not submodule_patch:
        raise SystemExit("submodule patch segments not found")
    args.root_output.write_text(root_patch, encoding="utf-8")
    args.submodule_output.write_text(submodule_patch, encoding="utf-8")
    print(f"Wrote {args.root_output}")
    print(f"Wrote {args.submodule_output}")


if __name__ == "__main__":
    main()
