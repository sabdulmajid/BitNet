#!/usr/bin/env python3
"""Audit whether row-scale I2-style deployment is productized as a stable qtype."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_bool(value: Any) -> bool:
    return bool(value) if isinstance(value, bool) else False


def fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.{digits}f}"
    return "" if value is None else str(value)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def make_gate(name: str, passed: bool, evidence: str, blocker: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": passed,
        "evidence": evidence,
        "blocker": "" if passed else blocker,
    }


def summarize_stable_qtype_benchmark(path: Path, catastrophic_ppl_threshold: float) -> dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "rows": 0,
            "failed_returncodes": [],
            "finite_ppl": [],
            "max_ppl": None,
            "quality_ok": False,
        }
    data = read_json(path)
    rows = data.get("rows", [])
    failed_returncodes: list[str] = []
    finite_ppl: list[float] = []
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                failed_returncodes.append("non-object-row")
                continue
            name = str(row.get("name", ""))
            for key in ("smoke_returncode", "bench_returncode", "ppl_returncode"):
                if key in row and int(row.get(key, 1)) != 0:
                    failed_returncodes.append(f"{name}:{key}={row.get(key)}")
            perplexity = row.get("perplexity", {})
            ppl = perplexity.get("ppl") if isinstance(perplexity, dict) else None
            if isinstance(ppl, (int, float)) and math.isfinite(float(ppl)):
                finite_ppl.append(float(ppl))
            else:
                failed_returncodes.append(f"{name}:ppl={ppl}")
    max_ppl = max(finite_ppl, default=None)
    quality_ok = bool(rows) and not failed_returncodes and max_ppl is not None and max_ppl < catastrophic_ppl_threshold
    return {
        "exists": True,
        "rows": len(rows) if isinstance(rows, list) else 0,
        "failed_returncodes": failed_returncodes,
        "finite_ppl": finite_ppl,
        "max_ppl": max_ppl,
        "quality_ok": quality_ok,
    }


def summarize_packing_verification(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "passed": False, "checked_tensors": 0, "passed_tensors": 0}
    data = read_json(path)
    return {
        "exists": True,
        "passed": bool(data.get("passed")),
        "checked_tensors": data.get("checked_tensors", 0),
        "passed_tensors": data.get("passed_tensors", 0),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    format_audit = read_json(args.format_audit_json)
    evidence_manifest = read_json(args.evidence_manifest_json)

    ggml_h = read_text(args.ggml_h)
    llama_h = read_text(args.llama_h)
    ggml_c = read_text(args.ggml_c)
    llama_cpp = read_text(args.llama_cpp)
    direct_writer = read_text(args.direct_writer)
    row_patch = read_text(args.row_patch)

    metrics = format_audit.get("metrics", {})
    verdict = format_audit.get("verdict", {})
    entries = evidence_manifest.get("entries", [])
    labels = {entry.get("label") for entry in entries if isinstance(entry, dict)}

    has_ggml_qtype = "GGML_TYPE_I2_SR" in ggml_h or "GGML_TYPE_I2_RS" in ggml_h
    has_llama_ftype = "LLAMA_FTYPE_MOSTLY_I2_SR" in llama_h or "LLAMA_FTYPE_MOSTLY_I2_RS" in llama_h
    source_routes_stable_qtype = (
        ("GGML_TYPE_I2_SR" in ggml_c or "GGML_TYPE_I2_RS" in ggml_c)
        and ("LLAMA_FTYPE_MOSTLY_I2_SR" in llama_cpp or "LLAMA_FTYPE_MOSTLY_I2_RS" in llama_cpp)
    )
    direct_writer_emits_stable_qtype = "I2_SR" in direct_writer or "I2_RS" in direct_writer
    row_patch_overloads_i2s = (
        "GGML_TYPE_I2_S" in row_patch
        and "ggml_nrows(tensor) * sizeof(float)" in row_patch
        and "GGML_TYPE_I2_SR" not in row_patch
        and "GGML_TYPE_I2_RS" not in row_patch
    )

    prototype_ratio = metrics.get("prototype_row_scale_i2s_to_tq2_ppl_ratio")
    default_ratio = metrics.get("default_row_scale_i2s_to_tq2_ppl_ratio")
    prototype_quality_ok = isinstance(prototype_ratio, (int, float)) and float(prototype_ratio) <= args.prototype_max_ppl_ratio
    default_failure_proven = isinstance(default_ratio, (int, float)) and float(default_ratio) >= args.default_failure_min_ratio
    stable_benchmark = summarize_stable_qtype_benchmark(args.stable_qtype_summary_json, args.catastrophic_ppl_threshold)
    packing_verification = summarize_packing_verification(args.packing_verification_json)
    stable_benchmark_present = (
        "i2sr_row_scale_qwen15b_suite" in labels
        or "i2rs_row_scale_qwen15b_suite" in labels
        or stable_benchmark["exists"]
    )
    stable_benchmark_quality_ok = stable_benchmark_present and bool(stable_benchmark["quality_ok"])

    gates = [
        make_gate(
            "row-scale semantics are physically feasible",
            prototype_quality_ok and metric_bool(verdict.get("row_scale_i2s_physically_possible")),
            f"prototype/TQ2_0 PPL ratio={fmt(prototype_ratio, 6)}",
            "No quality-preserving row-scale packed prototype evidence.",
        ),
        make_gate(
            "default tensor-scale I2_S failure is proven",
            default_failure_proven and metric_bool(verdict.get("default_i2s_layout_fails_row_scale")),
            f"default/TQ2_0 PPL ratio={fmt(default_ratio, 2)}",
            "Default I2_S has not been shown to fail row-scale semantics strongly enough.",
        ),
        make_gate(
            "stable GGML row-scale qtype is defined",
            has_ggml_qtype,
            str(args.ggml_h),
            "No separate GGML_TYPE_I2_SR/I2_RS enum is present; current patch overloads GGML_TYPE_I2_S.",
        ),
        make_gate(
            "stable llama file type is defined",
            has_llama_ftype,
            str(args.llama_h),
            "No separate LLAMA_FTYPE_MOSTLY_I2_SR/I2_RS value is present.",
        ),
        make_gate(
            "runtime routes stable qtype without changing I2_S",
            source_routes_stable_qtype and not row_patch_overloads_i2s,
            "ggml.c/llama.cpp source scan plus row-scale patch scan",
            "Runtime evidence still indicates an overloaded I2_S patch rather than a separate row-scale qtype path.",
        ),
        make_gate(
            "direct writer emits stable row-scale qtype",
            direct_writer_emits_stable_qtype,
            str(args.direct_writer),
            "Direct writer only emits existing I2_S numeric type IDs/prototype layout.",
        ),
        make_gate(
            "stable qtype benchmark evidence exists",
            stable_benchmark_present,
            str(args.evidence_manifest_json),
            "Evidence manifest has no stable I2_SR/I2_RS row-scale benchmark suite.",
        ),
        make_gate(
            "stable qtype benchmark preserves quality",
            stable_benchmark_quality_ok,
            str(args.stable_qtype_summary_json),
            f"Stable qtype benchmark is missing or has catastrophic/invalid PPL >= {args.catastrophic_ppl_threshold:g}.",
        ),
        make_gate(
            "direct I2_SR packing matches known-good x86 layout",
            bool(packing_verification["passed"]),
            str(args.packing_verification_json),
            "No passing byte-layout regression comparing direct I2_SR codes to the known-good quantizer layout.",
        ),
    ]

    return {
        "schema": "bitnet-row-scale-qtype-productization-v1",
        "inputs": {
            "format_audit_json": str(args.format_audit_json),
            "evidence_manifest_json": str(args.evidence_manifest_json),
            "ggml_h": str(args.ggml_h),
            "llama_h": str(args.llama_h),
            "ggml_c": str(args.ggml_c),
            "llama_cpp": str(args.llama_cpp),
            "direct_writer": str(args.direct_writer),
            "row_patch": str(args.row_patch),
            "stable_qtype_summary_json": str(args.stable_qtype_summary_json),
            "packing_verification_json": str(args.packing_verification_json),
        },
        "thresholds": {
            "prototype_max_ppl_ratio": args.prototype_max_ppl_ratio,
            "default_failure_min_ratio": args.default_failure_min_ratio,
            "catastrophic_ppl_threshold": args.catastrophic_ppl_threshold,
        },
        "observations": {
            "prototype_row_scale_i2s_to_tq2_ppl_ratio": prototype_ratio,
            "default_row_scale_i2s_to_tq2_ppl_ratio": default_ratio,
            "patch_overloads_existing_i2s": row_patch_overloads_i2s,
            "has_ggml_stable_qtype": has_ggml_qtype,
            "has_llama_stable_ftype": has_llama_ftype,
            "direct_writer_emits_stable_qtype": direct_writer_emits_stable_qtype,
            "stable_benchmark_present": stable_benchmark_present,
            "stable_benchmark_quality_ok": stable_benchmark_quality_ok,
            "stable_benchmark_rows": stable_benchmark["rows"],
            "stable_benchmark_max_ppl": stable_benchmark["max_ppl"],
            "stable_benchmark_failed_returncodes": stable_benchmark["failed_returncodes"],
            "packing_verification_present": packing_verification["exists"],
            "packing_verification_passed": packing_verification["passed"],
            "packing_verification_checked_tensors": packing_verification["checked_tensors"],
            "packing_verification_passed_tensors": packing_verification["passed_tensors"],
        },
        "gates": gates,
        "passed": all(gate["passed"] for gate in gates),
    }


def build_report(result: dict[str, Any]) -> str:
    rows = [
        [
            gate["name"],
            "pass" if gate["passed"] else "fail",
            gate["evidence"],
            gate["blocker"],
        ]
        for gate in result["gates"]
    ]
    obs = result["observations"]
    writer_clause = (
        "the direct writer now has an `I2_SR` emission mode"
        if obs["direct_writer_emits_stable_qtype"]
        else "the direct writer does not emit one"
    )
    benchmark_clause = (
        "a stable-qtype benchmark is present"
        if obs["stable_benchmark_present"]
        else "the manifest has no stable-qtype benchmark"
    )
    quality_clause = (
        "and it passes the catastrophic-PPL gate"
        if obs["stable_benchmark_quality_ok"]
        else "but it does not pass the catastrophic-PPL gate"
    )
    lines = [
        "# Row-Scale Qtype Productization Gate, 2026-05-13",
        "",
        "This gate checks whether the row-scale packed ternary path has moved from a local `I2_S`-overloading prototype to a compatibility-safe deployable qtype.",
        "",
        f"Overall status: `{'pass' if result['passed'] else 'fail'}`.",
        "",
        "## Gates",
        "",
        md_table(["gate", "status", "evidence", "blocker"], rows),
        "",
        "## Observations",
        "",
        f"- Prototype row-scale `I2_S` / `TQ2_0` PPL ratio: `{fmt(obs['prototype_row_scale_i2s_to_tq2_ppl_ratio'], 6)}`.",
        f"- Default row-scale `I2_S` / `TQ2_0` PPL ratio: `{fmt(obs['default_row_scale_i2s_to_tq2_ppl_ratio'], 2)}`.",
        f"- Current row-scale patch overloads existing `I2_S`: `{obs['patch_overloads_existing_i2s']}`.",
        f"- Stable GGML qtype present: `{obs['has_ggml_stable_qtype']}`.",
        f"- Stable llama file type present: `{obs['has_llama_stable_ftype']}`.",
        f"- Direct writer emits stable row-scale qtype: `{obs['direct_writer_emits_stable_qtype']}`.",
        f"- Stable qtype benchmark present in manifest: `{obs['stable_benchmark_present']}`.",
        f"- Stable qtype benchmark quality acceptable: `{obs['stable_benchmark_quality_ok']}`.",
        f"- Stable qtype benchmark max finite PPL: `{fmt(obs['stable_benchmark_max_ppl'], 4)}`.",
        f"- Direct `I2_SR` packing byte-layout verification passed: `{obs['packing_verification_passed']}` "
        f"({obs['packing_verification_passed_tensors']}/{obs['packing_verification_checked_tensors']} tensors).",
        "",
        "## Interpretation",
        "",
        "The feasibility claim is positive: the patched prototype preserves row-scale quality. "
        f"The productization claim is negative: the source tree does not yet define a separate row-scale qtype, {writer_clause}, and {benchmark_clause} {quality_clause}. "
        "This keeps row-scale packed deployment in research/prototype status.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format-audit-json", type=Path, default=Path("benchmark_results/i2s_row_scale_format_audit_2026-05-13.json"))
    parser.add_argument("--evidence-manifest-json", type=Path, default=Path("benchmarks/results/evidence_manifest_2026-05-13.json"))
    parser.add_argument("--ggml-h", type=Path, default=Path("3rdparty/llama.cpp/ggml/include/ggml.h"))
    parser.add_argument("--llama-h", type=Path, default=Path("3rdparty/llama.cpp/include/llama.h"))
    parser.add_argument("--ggml-c", type=Path, default=Path("3rdparty/llama.cpp/ggml/src/ggml.c"))
    parser.add_argument("--llama-cpp", type=Path, default=Path("3rdparty/llama.cpp/src/llama.cpp"))
    parser.add_argument("--direct-writer", type=Path, default=Path("benchmarks/convert_static_ternary_to_i2s_gguf.py"))
    parser.add_argument("--row-patch", type=Path, default=Path("patches/llama-i2s-row-scale.patch"))
    parser.add_argument("--stable-qtype-summary-json", type=Path, default=Path("benchmark_results/i2sr-row-scale-qwen15b-x86act-suite-2026-05-13/summary.json"))
    parser.add_argument("--packing-verification-json", type=Path, default=Path("benchmark_results/i2s-packing-layout-verify-2026-05-13/summary.json"))
    parser.add_argument("--prototype-max-ppl-ratio", type=float, default=1.01)
    parser.add_argument("--default-failure-min-ratio", type=float, default=10.0)
    parser.add_argument("--catastrophic-ppl-threshold", type=float, default=1.0e4)
    parser.add_argument("--output-json", type=Path, default=Path("benchmark_results/row_scale_qtype_productization_gate_2026-05-13.json"))
    parser.add_argument("--output-md", type=Path, default=Path("benchmarks/results/row_scale_qtype_productization_gate_2026-05-13.md"))
    args = parser.parse_args()

    result = audit(args)
    report = build_report(result)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(report + "\n", encoding="utf-8")
    print(report)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
