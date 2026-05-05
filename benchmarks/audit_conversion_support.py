#!/usr/bin/env python3
"""Audit converter/runtime support for Qwen, TL2, and packed ternary paths."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Any


def registered_architectures(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            func = decorator.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr == "register"
                and isinstance(func.value, ast.Name)
                and func.value.id == "Model"
            ):
                continue
            for arg in decorator.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    names.add(arg.value)
    return sorted(names)


def ftype_map_keys(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    keys: set[str] = set()

    def collect_from_dict(node: ast.AST) -> None:
        if not isinstance(node, ast.Dict):
            return
        for key in node.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys.add(key.value)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "ftype_map" for target in node.targets):
                collect_from_dict(node.value)
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "ftype_map" and node.value is not None:
                collect_from_dict(node.value)
    return sorted(keys)


def llama_quantize_types(path: Path) -> list[str]:
    completed = subprocess.run(
        [str(path), "--help"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    text = completed.stdout
    names = set(re.findall(r"\bor\s+([A-Z0-9_]+)\s*:", text))
    if "COPY" in text:
        names.add("COPY")
    return sorted(names)


def python_help_ok(path: Path) -> bool:
    completed = subprocess.run(
        ["python", str(path), "--help"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.returncode == 0 and "usage:" in completed.stdout


def yes(value: bool) -> str:
    return "yes" if value else "no"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_report(data: dict[str, Any]) -> str:
    rows = [
        [
            "BitNet HF converter",
            str(data["bitnet_converter"]["path"]),
            yes(data["bitnet_converter"]["supports_qwen2"]),
            yes(data["bitnet_converter"]["supports_qwen2_moe"]),
            yes(data["bitnet_converter"]["supports_tl2_outtype"]),
            yes(data["bitnet_converter"]["supports_i2s_outtype"]),
            yes(data["bitnet_converter"]["help_ok"]),
            ", ".join(data["bitnet_converter"]["outtypes"]),
        ],
        [
            "llama.cpp HF converter",
            str(data["llama_converter"]["path"]),
            yes(data["llama_converter"]["supports_qwen2"]),
            yes(data["llama_converter"]["supports_qwen2_moe"]),
            yes(data["llama_converter"]["supports_tl2_outtype"]),
            yes(data["llama_converter"]["supports_i2s_outtype"]),
            yes(data["llama_converter"]["help_ok"]),
            ", ".join(data["llama_converter"]["outtypes"]),
        ],
        [
            "llama-quantize",
            str(data["llama_quantize"]["path"]),
            "n/a",
            "n/a",
            yes(data["llama_quantize"]["supports_tl2_type"]),
            yes(data["llama_quantize"]["supports_i2s_type"]),
            yes(data["llama_quantize"]["help_ok"]),
            ", ".join(data["llama_quantize"]["types"]),
        ],
    ]
    if data["bitnet_converter"]["supports_qwen2"] and data["bitnet_converter"]["supports_tl2_outtype"]:
        verdict = (
            "The BitNet HF converter now has a Qwen2 plus TL2 conversion entry "
            "point, and its help path is runnable in this clone. This is "
            "converter-level support only: Qwen2MoE is still not registered in "
            "that converter, TL2 still requires exact model-specific kernel "
            "config/codegen, and `llama-quantize` still cannot create TL2 from "
            "an existing GGUF. The Qwen2.5-0.5B TL2 probe is a quality-failed "
            "small-model artifact, not a production Qwen TL2 deployment. "
            "`llama-quantize` can produce `I2_S`, but it operates from an existing "
            "GGUF and is not a direct `ternary_state_dict.pt` writer."
        )
    else:
        verdict = (
            "The current toolchain does not provide a single direct path that is both "
            "Qwen2-aware and TL2-capable. The BitNet HF converter exposes `tl2`, but "
            "does not register `Qwen2ForCausalLM` or `Qwen2MoeForCausalLM`. The "
            "vendored llama.cpp converter registers those Qwen architectures, but "
            "its `--outtype` choices stop at `tq2_0` and do not include `tl2` or "
            "`i2_s`. `llama-quantize` can produce `I2_S`, but it operates from an "
            "existing GGUF and is not a direct `ternary_state_dict.pt` writer."
        )
    return "\n\n".join(
        [
            "# Conversion Support Audit, 2026-05-05",
            md_table(
                [
                    "component",
                    "path",
                    "Qwen2",
                    "Qwen2 MoE",
                    "TL2 out/type",
                    "I2_S out/type",
                    "help ok",
                    "advertised output/types",
                ],
                rows,
            ),
            "## Verdict",
            verdict,
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bitnet-converter", type=Path, default=Path("utils/convert-hf-to-gguf-bitnet.py"))
    parser.add_argument("--llama-converter", type=Path, default=Path("3rdparty/llama.cpp/convert_hf_to_gguf.py"))
    parser.add_argument("--llama-quantize", type=Path, default=Path("build-portable-avx2/bin/llama-quantize"))
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    bitnet_arches = registered_architectures(args.bitnet_converter)
    llama_arches = registered_architectures(args.llama_converter)
    bitnet_outtypes = ftype_map_keys(args.bitnet_converter)
    llama_outtypes = ftype_map_keys(args.llama_converter)
    quantize_types = llama_quantize_types(args.llama_quantize)

    data: dict[str, Any] = {
        "bitnet_converter": {
            "path": str(args.bitnet_converter),
            "registered_architectures": bitnet_arches,
            "outtypes": bitnet_outtypes,
            "supports_qwen2": "Qwen2ForCausalLM" in bitnet_arches,
            "supports_qwen2_moe": "Qwen2MoeForCausalLM" in bitnet_arches,
            "supports_tl2_outtype": "tl2" in bitnet_outtypes,
            "supports_i2s_outtype": "i2_s" in bitnet_outtypes or "I2_S" in bitnet_outtypes,
            "help_ok": python_help_ok(args.bitnet_converter),
        },
        "llama_converter": {
            "path": str(args.llama_converter),
            "registered_architectures": llama_arches,
            "outtypes": llama_outtypes,
            "supports_qwen2": "Qwen2ForCausalLM" in llama_arches,
            "supports_qwen2_moe": "Qwen2MoeForCausalLM" in llama_arches,
            "supports_tl2_outtype": "tl2" in llama_outtypes,
            "supports_i2s_outtype": "i2_s" in llama_outtypes or "I2_S" in llama_outtypes,
            "help_ok": python_help_ok(args.llama_converter),
        },
        "llama_quantize": {
            "path": str(args.llama_quantize),
            "types": quantize_types,
            "supports_tl2_type": "TL2" in quantize_types,
            "supports_i2s_type": "I2_S" in quantize_types,
            "supports_tq2_type": "TQ2_0" in quantize_types,
            "help_ok": bool(quantize_types),
        },
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
