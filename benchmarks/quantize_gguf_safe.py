#!/usr/bin/env python3
"""Safe wrapper around llama-quantize for known BitNet GGUF edge cases."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-quantize", type=Path, default=Path("build/bin/llama-quantize"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--type", required=True, help="Quantization type, e.g. Q8_0, TQ2_0, I2_S")
    parser.add_argument("--threads", type=int, default=0, help="llama-quantize thread count; 0 uses its default")
    parser.add_argument(
        "--allow-unsafe-i2s-multithread",
        action="store_true",
        help="Do not force I2_S to one writer thread. Use only for debugging the writer bug.",
    )
    args = parser.parse_args()

    qtype = args.type.upper()
    threads = args.threads
    if qtype == "I2_S" and threads != 1 and not args.allow_unsafe_i2s_multithread:
        print(
            "I2_S multi-thread quantization produced corrupted artifacts in this fork; "
            "forcing llama-quantize thread count to 1.",
            flush=True,
        )
        threads = 1

    command = [
        str(args.llama_quantize),
        str(args.input),
        str(args.output),
        args.type,
    ]
    if threads > 0:
        command.append(str(threads))

    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
