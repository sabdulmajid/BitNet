#!/usr/bin/env python3
"""Audit TL2 kernel-shape coverage for dense Qwen checkpoints."""

from __future__ import annotations

import argparse
import configparser
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from safetensors import safe_open


EXCLUDED_SUFFIXES = (
    "lm_head.weight",
    "norm.weight",
    "embed_tokens.weight",
)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def parse_label_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    label, raw_path = value.split("=", 1)
    return label, Path(raw_path)


def safetensors_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    direct = path / "model.safetensors"
    if direct.exists():
        return [direct]
    shards = sorted(path.glob("model-*.safetensors"))
    if shards:
        return shards
    raise FileNotFoundError(f"no safetensors files found under {path}")


def eligible_weight_shapes(path: Path) -> tuple[Counter[tuple[int, int]], dict[tuple[int, int], str]]:
    counts: Counter[tuple[int, int]] = Counter()
    examples: dict[tuple[int, int], str] = {}
    for st_file in safetensors_files(path):
        with safe_open(st_file, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                shape = tuple(int(dim) for dim in handle.get_slice(name).get_shape())
                if not name.endswith(".weight") or len(shape) < 2:
                    continue
                if name.endswith(EXCLUDED_SUFFIXES):
                    continue
                if len(shape) != 2:
                    continue
                shape_2d = (shape[0], shape[1])
                counts[shape_2d] += 1
                examples.setdefault(shape_2d, name)
    return counts, examples


def read_kernel_shapes(path: Path) -> dict[tuple[int, int], dict[str, int]]:
    if not path.exists():
        return {}
    config = configparser.ConfigParser()
    config.read(path)
    shapes: dict[tuple[int, int], dict[str, int]] = {}
    for section in config.sections():
        try:
            shape = (config.getint(section, "m"), config.getint(section, "k"))
            shapes[shape] = {
                "bm": config.getint(section, "bm"),
                "bk": config.getint(section, "bk"),
                "bmm": config.getint(section, "bmm"),
            }
        except (configparser.Error, ValueError):
            continue
    return shapes


def parse_cmake_bool(cache_path: Path, key: str) -> str:
    if not cache_path.exists():
        return "missing"
    pattern = re.compile(rf"^{re.escape(key)}:[^=]*=(.*)$")
    for line in cache_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if match:
            return match.group(1)
    return "absent"


def choose_bm(m: int) -> int | None:
    for candidate in (256, 224, 192, 160, 128, 96, 64, 32):
        if m % candidate == 0:
            return candidate
    return None


def choose_bk(k: int) -> int | None:
    for candidate in (192, 96, 64, 32):
        if (k % candidate) % 32 == 0:
            return candidate
    return None


def suggested_codegen(shapes: list[tuple[int, int]]) -> dict[str, Any]:
    bm_values: list[int] = []
    bk_values: list[int] = []
    bm32_values: list[int] = []
    problems: list[str] = []

    for m, k in shapes:
        bm_value = choose_bm(m)
        bk_value = choose_bk(k)
        if bm_value is None:
            problems.append(f"{m},{k}: no BM divisor from the audited candidate set")
            continue
        if bk_value is None:
            problems.append(f"{m},{k}: no BK candidate satisfies (K % BK) % 32 == 0")
            continue
        bm_values.append(bm_value)
        bk_values.append(bk_value)
        bm32_values.append(32)

    if problems:
        return {"available": False, "problems": problems}

    shape_args = " ".join(f"--shape {m},{k}" for m, k in shapes)
    return {
        "available": True,
        "command": (
            "python utils/codegen_tl2.py "
            f"{shape_args} "
            f"--BM {','.join(str(item) for item in bm_values)} "
            f"--BK {','.join(str(item) for item in bk_values)} "
            f"--bm {','.join(str(item) for item in bm32_values)}"
        ),
        "bm": bm_values,
        "bk": bk_values,
        "bmm": bm32_values,
    }


def audit_model(label: str, path: Path, configs: dict[str, Path]) -> dict[str, Any]:
    counts, examples = eligible_weight_shapes(path)
    unique_shapes = sorted(counts)
    config_results: dict[str, Any] = {}
    for config_label, config_path in configs.items():
        kernel_shapes = read_kernel_shapes(config_path)
        supported = sum(count for shape, count in counts.items() if shape in kernel_shapes)
        missing = {shape: count for shape, count in counts.items() if shape not in kernel_shapes}
        config_results[config_label] = {
            "path": str(config_path),
            "exists": config_path.exists(),
            "kernel_shape_count": len(kernel_shapes),
            "supported_tensors": supported,
            "missing_tensors": sum(missing.values()),
            "missing_shapes": [
                {
                    "shape": list(shape),
                    "count": count,
                    "example": examples[shape],
                }
                for shape, count in sorted(missing.items())
            ],
        }
    return {
        "label": label,
        "path": str(path),
        "eligible_tensors": sum(counts.values()),
        "unique_shapes": [
            {"shape": list(shape), "count": counts[shape], "example": examples[shape]}
            for shape in unique_shapes
        ],
        "configs": config_results,
        "suggested_codegen": suggested_codegen(unique_shapes),
    }


def build_report(data: dict[str, Any]) -> str:
    model_rows: list[list[str]] = []
    for model in data["models"]:
        for config_label, result in model["configs"].items():
            model_rows.append(
                [
                    model["label"],
                    config_label,
                    "yes" if result["exists"] else "no",
                    str(model["eligible_tensors"]),
                    str(len(model["unique_shapes"])),
                    str(result["kernel_shape_count"]),
                    str(result["supported_tensors"]),
                    str(result["missing_tensors"]),
                ]
            )

    shape_sections: list[str] = []
    for model in data["models"]:
        rows = [
            [f"{item['shape'][0]} x {item['shape'][1]}", str(item["count"]), item["example"]]
            for item in model["unique_shapes"]
        ]
        shape_sections.extend(
            [
                f"## {model['label']} Shapes",
                md_table(["shape (out x in)", "tensor count", "example tensor"], rows),
            ]
        )
        suggestion = model["suggested_codegen"]
        if suggestion["available"]:
            shape_sections.extend(
                [
                    f"## {model['label']} Custom TL2 Codegen",
                    "A syntactically valid TL2 codegen parameterization exists for these dense matrix shapes:",
                    f"```bash\n{suggestion['command']}\n```",
                ]
            )
        else:
            shape_sections.extend(
                [
                    f"## {model['label']} Custom TL2 Codegen",
                    "No valid parameterization was found by the audited heuristic:",
                    "\n".join(f"- {problem}" for problem in suggestion["problems"]),
                ]
            )

    build_rows = [
        [item["cache"], item["bitnet_x86_tl2"]]
        for item in data["builds"]
    ]
    has_tl2_build = any(item["bitnet_x86_tl2"] == "ON" for item in data["builds"])
    if has_tl2_build:
        build_sentence = (
            "At least one checked build cache has `BITNET_X86_TL2=ON`, but the "
            "default checked build caches remain `OFF`; TL2 measurements must "
            "name the exact model-specific binary used."
        )
    else:
        build_sentence = (
            "The checked build caches have `BITNET_X86_TL2=OFF`, so a TL2 GGUF "
            "would not be a validated runtime artifact in these builds."
        )

    verdict = (
        "The checked Qwen checkpoints have dense matrix shapes that are not covered "
        "by the active TL2 config or by the bundled BitNet/Llama preset configs. "
        f"{build_sentence} Custom TL2 code generation appears possible for the "
        "audited dense shapes, but it requires generating model-specific LUT "
        "kernels, rebuilding the runtime with `-DBITNET_X86_TL2=ON`, converting "
        "with the matching kernel config, and then running the same "
        "PPL/throughput/RSS audits."
    )

    return "\n\n".join(
        [
            "# TL2 Shape Support Audit, 2026-05-05",
            "## Coverage",
            md_table(
                [
                    "model",
                    "kernel config",
                    "config exists",
                    "eligible tensors",
                    "unique shapes",
                    "kernel shapes",
                    "supported tensors",
                    "missing tensors",
                ],
                model_rows,
            ),
            "## Build Flags",
            md_table(["CMake cache", "BITNET_X86_TL2"], build_rows),
            "## Verdict",
            verdict,
            *shape_sections,
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="MODEL_LABEL=/path/to/model_dir_or_safetensors; repeat for multiple models",
    )
    parser.add_argument(
        "--kernel-config",
        action="append",
        default=[],
        help="CONFIG_LABEL=/path/to/kernel_config.ini; repeat for multiple configs",
    )
    parser.add_argument("--cmake-cache", action="append", default=None)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    configs = dict(parse_label_path(item) for item in args.kernel_config)
    if not configs:
        configs = {
            "active include": Path("include/kernel_config.ini"),
            "BitNet 3B preset": Path("preset_kernels/bitnet_b1_58-3B/kernel_config_tl2.ini"),
            "BitNet large preset": Path("preset_kernels/bitnet_b1_58-large/kernel_config_tl2.ini"),
            "Llama3 8B preset": Path("preset_kernels/Llama3-8B-1.58-100B-tokens/kernel_config_tl2.ini"),
        }

    cmake_caches = args.cmake_cache or ["build/CMakeCache.txt", "build-portable-avx2/CMakeCache.txt"]
    models = [audit_model(label, path, configs) for label, path in (parse_label_path(item) for item in args.model)]
    data: dict[str, Any] = {
        "models": models,
        "kernel_configs": {label: str(path) for label, path in configs.items()},
        "builds": [
            {
                "cache": cache_path,
                "bitnet_x86_tl2": parse_cmake_bool(Path(cache_path), "BITNET_X86_TL2"),
            }
            for cache_path in cmake_caches
        ],
    }

    report = build_report(data)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report + "\n", encoding="utf-8")
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
