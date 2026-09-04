#!/usr/bin/env python3
"""Verify generated TL2_SR kernels against an explicit W1.58A8 reference."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
from configparser import ConfigParser
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


DATE = os.environ.get("BITNET_REPORT_DATE") or datetime.now(timezone.utc).date().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path) -> ModuleType:
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("bitnet_tl2sr_converter", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import converter from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def kernel_shapes(config_path: Path) -> list[tuple[int, int, int, int]]:
    config = ConfigParser()
    if not config.read(config_path):
        raise FileNotFoundError(config_path)
    rows = []
    for section in config.sections():
        rows.append(
            (
                config.getint(section, "m"),
                config.getint(section, "k"),
                config.getint(section, "bm"),
                config.getint(section, "bk"),
            )
        )
    if not rows:
        raise ValueError(f"no kernels in {config_path}")
    return rows


def ptr(array: np.ndarray, offset: int = 0) -> ctypes.c_void_p:
    return ctypes.c_void_p(array.ctypes.data + offset)


def configure_library(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    pointer = ctypes.c_void_p
    library.ggml_preprocessor.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        pointer,
        pointer,
        pointer,
        pointer,
    ]
    library.ggml_qgemm_lut.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        pointer,
        pointer,
        pointer,
        pointer,
        ctypes.c_int,
        pointer,
        pointer,
    ]
    return library


def run_case(
    *,
    library: ctypes.CDLL,
    converter: ModuleType,
    config_path: Path,
    m: int,
    k: int,
    bm: int,
    bk: int,
    batch: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    codes = rng.choice(
        np.array([-1, 0, 1], dtype=np.int8),
        size=(m, k),
        p=(0.25, 0.50, 0.25),
    )
    row_scales = np.linspace(0.001, 0.05, m, dtype=np.float32)
    activations = rng.normal(size=(batch, k)).astype(np.float32)
    packed = converter.preprocess_weights_tl2(
        codes.astype(np.float32),
        config_path=config_path,
    )

    three_k = (k // bk) * bk
    two_k = k - three_k
    three_bytes = three_k * m // 6
    sign_bytes = three_k * m // 24
    two_offset = three_bytes + sign_bytes
    two_bytes = m * two_k // 4
    tile_count = m // bm
    if any(value % tile_count for value in (three_bytes, sign_bytes, two_bytes)):
        raise ValueError(f"packed regions do not divide into {tile_count} tiles")

    three_lut = np.zeros((batch, three_k // 3 * 32), dtype=np.int8)
    two_lut = np.zeros((batch, two_k // 2 * 32), dtype=np.int8)
    activation_scales = np.zeros(batch, dtype=np.float32)
    output = np.zeros((batch, m), dtype=np.float32)

    library.ggml_preprocessor(
        batch,
        m,
        three_k,
        two_k,
        ptr(activations),
        ptr(activation_scales),
        ptr(three_lut),
        ptr(two_lut),
    )
    for tile in range(tile_count):
        output_offset = tile * bm * output.itemsize
        scale_offset = tile * bm * row_scales.itemsize
        library.ggml_qgemm_lut(
            batch,
            m,
            k,
            three_k,
            ptr(packed, tile * three_bytes // tile_count),
            ptr(packed, three_bytes + tile * sign_bytes // tile_count),
            ptr(three_lut),
            ptr(row_scales, scale_offset),
            1,
            ptr(activation_scales),
            ptr(output, output_offset),
        )
        if two_k:
            library.ggml_qgemm_lut(
                batch,
                m,
                k,
                two_k,
                ptr(packed, two_offset + tile * two_bytes // tile_count),
                None,
                ptr(two_lut),
                ptr(row_scales, scale_offset),
                1,
                ptr(activation_scales),
                ptr(output, output_offset),
            )

    quantized = np.rint(activations * activation_scales[:, None]).clip(-127, 127).astype(np.int32)
    reference = (quantized @ codes.astype(np.int32).T).astype(np.float32)
    reference *= row_scales[None, :] / activation_scales[:, None]
    delta = output - reference
    reference_norm = float(np.linalg.norm(reference))
    relative_rms = float(np.linalg.norm(delta) / reference_norm) if reference_norm else math.inf
    return {
        "m": m,
        "k": k,
        "batch": batch,
        "bm": bm,
        "bk": bk,
        "three_k": three_k,
        "two_k": two_k,
        "relative_rms_error": relative_rms,
        "max_abs_error": float(np.max(np.abs(delta))),
        "passed": relative_rms <= 1e-6,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        f"# TL2_SR Kernel Contract, {result['date']}",
        "",
        "Generated lookup-table kernels are compared with the explicit reference",
        "`Y[b, i] = row_scale[i] * sum_j(T[i, j] * Q8(X[b, j])) / activation_scale[b]`.",
        "The test uses nonuniform row scales, so a scalar-scale implementation cannot pass.",
        "",
        f"Status: **{result['status']}**",
        "",
        "| M | K | batch | relative RMS error | max abs error | status |",
        "| ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in result["cases"]:
        lines.append(
            f"| {row['m']} | {row['k']} | {row['batch']} | "
            f"{row['relative_rms_error']:.9g} | {row['max_abs_error']:.9g} | "
            f"{'pass' if row['passed'] else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "This proves the generated matrix kernels for these shapes and batches. It does not,",
            "by itself, prove end-to-end model quality or throughput.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--library",
        type=Path,
        default=Path("build-qwen05b-tl2sr/3rdparty/llama.cpp/ggml/src/libggml.so"),
    )
    parser.add_argument("--converter", type=Path, default=Path("utils/convert-hf-to-gguf-bitnet.py"))
    parser.add_argument(
        "--kernel-config",
        type=Path,
        default=Path("preset_kernels/Qwen2.5-0.5B-TL2SR/kernel_config_tl2sr.ini"),
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(f"benchmark_results/tl2sr_kernel_contract_{DATE}.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(f"benchmarks/results/tl2sr_kernel_contract_{DATE}.md"),
    )
    args = parser.parse_args()

    root = args.repo_root.resolve()
    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    library_path = resolve(args.library)
    converter_path = resolve(args.converter)
    config_path = resolve(args.kernel_config)
    library = configure_library(library_path)
    converter = load_module(converter_path)

    shapes = kernel_shapes(config_path)
    cases = []
    for index, (m, k, bm, bk) in enumerate(shapes):
        batches = (1, 8, 32) if index == 0 else (1,)
        for batch in batches:
            cases.append(
                run_case(
                    library=library,
                    converter=converter,
                    config_path=config_path,
                    m=m,
                    k=k,
                    bm=bm,
                    bk=bk,
                    batch=batch,
                    seed=args.seed + index * 100 + batch,
                )
            )

    passed = all(row["passed"] for row in cases)
    result = {
        "schema": "tl2sr-kernel-contract-v1",
        "date": DATE,
        "status": "pass" if passed else "fail",
        "formula": "Y[b,i]=row_scale[i]*sum_j(T[i,j]*Q8(X[b,j]))/activation_scale[b]",
        "library": str(args.library),
        "library_sha256": sha256_file(library_path),
        "converter": str(args.converter),
        "converter_sha256": sha256_file(converter_path),
        "kernel_config": str(args.kernel_config),
        "kernel_config_sha256": sha256_file(config_path),
        "seed": args.seed,
        "platform": platform.platform(),
        "cases": cases,
    }
    output_json = resolve(args.output_json)
    output_md = resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(result), encoding="utf-8")
    print(render_markdown(result))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
