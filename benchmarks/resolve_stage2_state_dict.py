#!/usr/bin/env python3
"""Print the state_dict_path recorded in a Stage-2 manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--no-existence-check", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    path = manifest.get("state_dict_path")
    if not isinstance(path, str) or not path:
        raise SystemExit(f"{args.manifest} does not contain state_dict_path")
    if not args.no_existence_check and not Path(path).exists():
        raise SystemExit(f"state_dict_path does not exist: {path}")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
