#!/usr/bin/env python3
"""Verify direct I2-style GGUF packing against a known-good BitNet layout.

The bundled GGUFReader currently depends on an ndarray.newbyteorder API that is
not available in this environment's NumPy, so this script uses a minimal GGUF
directory parser. It compares only the raw ternary code payload, not the scale
payload. That is intentional for row-scale checks: the known-good prototype can
recompute row scales from a materialized F16 bridge while direct export keeps
the trained FP32 row scales.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO


GGUF_MAGIC = 0x46554747
DEFAULT_ALIGNMENT = 32
VALUE_SIZES = {
    0: 1,   # UINT8
    1: 1,   # INT8
    2: 2,   # UINT16
    3: 2,   # INT16
    4: 4,   # UINT32
    5: 4,   # INT32
    6: 4,   # FLOAT32
    7: 1,   # BOOL
    10: 8,  # UINT64
    11: 8,  # INT64
    12: 8,  # FLOAT64
}


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dims: tuple[int, ...]
    qtype: int
    data_offset: int

    @property
    def elements(self) -> int:
        return math.prod(self.dims)

    @property
    def code_bytes(self) -> int:
        if self.elements % 4 != 0:
            raise ValueError(f"{self.name} has {self.elements} elements, not divisible by four")
        return self.elements // 4


def read_u32(handle: BinaryIO) -> int:
    return struct.unpack("<I", handle.read(4))[0]


def read_u64(handle: BinaryIO) -> int:
    return struct.unpack("<Q", handle.read(8))[0]


def read_string(handle: BinaryIO) -> str:
    size = read_u64(handle)
    return handle.read(size).decode("utf-8")


def skip_value(handle: BinaryIO, value_type: int) -> None:
    if value_type == 8:  # STRING
        handle.seek(read_u64(handle), 1)
        return
    if value_type == 9:  # ARRAY
        item_type = read_u32(handle)
        count = read_u64(handle)
        for _ in range(count):
            skip_value(handle, item_type)
        return
    if value_type not in VALUE_SIZES:
        raise ValueError(f"unsupported GGUF value type {value_type}")
    handle.seek(VALUE_SIZES[value_type], 1)


def parse_gguf_tensors(path: Path) -> dict[str, TensorInfo]:
    with path.open("rb") as handle:
        magic = read_u32(handle)
        if magic != GGUF_MAGIC:
            raise ValueError(f"{path} is not a GGUF file: bad magic 0x{magic:08x}")
        version = read_u32(handle)
        tensor_count = read_u64(handle)
        kv_count = read_u64(handle)
        alignment = DEFAULT_ALIGNMENT

        for _ in range(kv_count):
            key = read_string(handle)
            value_type = read_u32(handle)
            if key == "general.alignment" and value_type == 4:
                alignment = read_u32(handle)
            else:
                skip_value(handle, value_type)

        infos: list[tuple[str, tuple[int, ...], int, int]] = []
        for _ in range(tensor_count):
            name = read_string(handle)
            dim_count = read_u32(handle)
            dims = tuple(read_u64(handle) for _ in range(dim_count))
            qtype = read_u32(handle)
            relative_offset = read_u64(handle)
            infos.append((name, dims, qtype, relative_offset))

        data_start = handle.tell()
        if data_start % alignment:
            data_start += alignment - (data_start % alignment)

    return {
        name: TensorInfo(name=name, dims=dims, qtype=qtype, data_offset=data_start + relative_offset)
        for name, dims, qtype, relative_offset in infos
    }


def read_bytes(path: Path, offset: int, size: int) -> bytes:
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read(size)
    if len(data) != size:
        raise ValueError(f"short read from {path}: wanted {size} bytes, got {len(data)}")
    return data


def compare_tensor(reference_path: Path, candidate_path: Path, reference: TensorInfo, candidate: TensorInfo) -> dict[str, Any]:
    if reference.dims != candidate.dims:
        return {
            "name": reference.name,
            "passed": False,
            "reason": f"shape mismatch: reference {reference.dims}, candidate {candidate.dims}",
            "reference_qtype": reference.qtype,
            "candidate_qtype": candidate.qtype,
        }

    code_bytes = reference.code_bytes
    reference_codes = read_bytes(reference_path, reference.data_offset, code_bytes)
    candidate_codes = read_bytes(candidate_path, candidate.data_offset, code_bytes)
    reference_sha = hashlib.sha256(reference_codes).hexdigest()
    candidate_sha = hashlib.sha256(candidate_codes).hexdigest()
    equal = reference_codes == candidate_codes
    return {
        "name": reference.name,
        "passed": equal,
        "dims": list(reference.dims),
        "elements": reference.elements,
        "code_bytes": code_bytes,
        "reference_qtype": reference.qtype,
        "candidate_qtype": candidate.qtype,
        "reference_sha256": reference_sha,
        "candidate_sha256": candidate_sha,
        "sha256_prefix": reference_sha[:12] if equal else None,
        "reference_first16_hex": reference_codes[:16].hex(),
        "candidate_first16_hex": candidate_codes[:16].hex(),
    }


def build_markdown(result: dict[str, Any]) -> str:
    rows = [
        "| tensor | passed | qtype ref/cand | code bytes | sha256 prefix | first 16 bytes |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in result["rows"]:
        rows.append(
            "| {name} | {passed} | {ref}/{cand} | {code_bytes} | {sha} | `{first16}` |".format(
                name=row["name"],
                passed=str(row["passed"]).lower(),
                ref=row.get("reference_qtype", "-"),
                cand=row.get("candidate_qtype", "-"),
                code_bytes=row.get("code_bytes", "-"),
                sha=row.get("sha256_prefix") or "-",
                first16=row.get("reference_first16_hex", "-"),
            )
        )

    return "\n".join(
        [
            "# I2S Packing Layout Verification",
            "",
            f"Reference GGUF: `{result['reference_gguf']}`",
            f"Candidate GGUF: `{result['candidate_gguf']}`",
            "",
            f"Passed: `{result['passed']}`",
            f"Tensors checked: `{result['checked_tensors']}`",
            "",
            "This check compares only the packed ternary code payload. It does not require scale bytes to match.",
            "",
            *rows,
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-gguf",
        type=Path,
        default=Path("models/qwen2.5-1.5b-klonly-row-notie-static-ternary-dense/qwen15b_klonly_row_notie_static_ternary_i2_s_rowscale.gguf"),
        help="Known-good GGUF produced through the validated quantizer/runtime path.",
    )
    parser.add_argument(
        "--candidate-gguf",
        type=Path,
        default=Path("models/qwen2.5-1.5b-klonly-row-notie-static-ternary-dense/qwen15b_klonly_row_notie_static_ternary_i2_sr_x86act.gguf"),
        help="Direct-writer GGUF candidate to verify.",
    )
    parser.add_argument(
        "--tensor",
        action="append",
        default=[
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_down.weight",
            "blk.27.ffn_down.weight",
        ],
        help="GGUF tensor name to compare; can be repeated.",
    )
    parser.add_argument("--output-json", type=Path, default=Path("benchmark_results/i2s-packing-layout-verify-2026-05-13/summary.json"))
    parser.add_argument("--output-md", type=Path, default=Path("benchmarks/results/i2s_packing_layout_verify_2026-05-13.md"))
    args = parser.parse_args()

    reference_tensors = parse_gguf_tensors(args.reference_gguf)
    candidate_tensors = parse_gguf_tensors(args.candidate_gguf)
    rows = []
    for tensor_name in args.tensor:
        if tensor_name not in reference_tensors:
            rows.append({"name": tensor_name, "passed": False, "reason": "missing from reference"})
            continue
        if tensor_name not in candidate_tensors:
            rows.append({"name": tensor_name, "passed": False, "reason": "missing from candidate"})
            continue
        rows.append(
            compare_tensor(
                args.reference_gguf,
                args.candidate_gguf,
                reference_tensors[tensor_name],
                candidate_tensors[tensor_name],
            )
        )

    result = {
        "schema": "bitnet-i2s-packing-layout-verify-v1",
        "reference_gguf": str(args.reference_gguf),
        "candidate_gguf": str(args.candidate_gguf),
        "checked_tensors": len(rows),
        "passed_tensors": sum(1 for row in rows if row.get("passed") is True),
        "passed": bool(rows) and all(row.get("passed") is True for row in rows),
        "rows": rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(build_markdown(result), encoding="utf-8")
    print(build_markdown(result), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
