# Runtime Contract

The runtime contract is the main systems lesson from this fork.

## Contract

A row-scale ternary checkpoint represents weights as:

```text
W ~= s_row * T
T in {-1, 0, +1}
```

The per-row scale is part of the learned function. It is not disposable
metadata.

## Evidence

The TL2 row-scale audit measured:

| Runtime scale choice | Relative output RMS error |
| --- | ---: |
| one-scale TL2 path | `1.904230` |
| exact FP16 row scales | `0.000197` |

This means a runtime that collapses learned row scales into one tensor scale
does not preserve the trained checkpoint.

## Current Runtime Status

| Runtime path | Status |
| --- | --- |
| `I2_SR` row-scale GGUF path | Working for compatible causal artifacts. |
| TL2 one-scale path | Not valid for row-scale checkpoints. |
| Experimental `TL2_SR` path | Exact-shape Qwen2.5-0.5B kernels are numerically valid; storage improves, CPU speed does not. |
| Native sequence-classifier path | Task-quality parity supported for one artifact; numerical parity and batching remain below product gates. |
| MoE/Kimi path | Not supported beyond tiny fixtures. |

## CPU Result

On the Xeon Silver 4116 CPU evidence bundle:

| Artifact | File MiB | PPL | Prompt tok/s | Decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| FP F16 | `2950.4` | `12.2808` | `114.47` | `5.56` |
| FP Q4_K_M | `940.4` | `12.8112` | `92.08` | `16.01` |
| row-scale `I2_SR` | `1211.3` | `38.8477` | `211.67` | `19.07` |

Interpretation: `I2_SR` proves a row-scale packed runtime path and improves
decode speed versus FP16, but it is not quality/storage competitive with Q4_K_M
in this run.

## Sequence-Classification Result

The same-artifact native classifier was also evaluated on all 9,815 MNLI
validation examples using direct token IDs and sequence-isolated execution:

| Metric | Result |
| --- | ---: |
| PyTorch accuracy | `0.653591` |
| native `I2_SR` accuracy | `0.652165` |
| paired delta | `-0.001426` |
| paired 95% CI | `[-0.004193, 0.001341]` |
| exact McNemar p | `0.348171` |
| exact prediction agreement | `0.976668` |
| throughput | `7.456204 examples/s` |
| child peak RSS | `960.15 MiB` |

The packed runtime preserves task accuracy for this artifact within measured
uncertainty, but it is not numerically identical to GPU-BF16 PyTorch and the
underlying model is not competitive with FP16. The 0.5-point non-inferiority
criterion was selected retrospectively. Multi-prompt batching is not part of
this result. See
`benchmarks/results/seqcls_runtime_quality_equivalence_2026-09-04.md`.

## Experimental TL2_SR Contract

`TL2_SR` is a dedicated row-scale lookup-table qtype. Its packed tensor layout
is:

```text
[aligned TL2 ternary payload][M FP32 output-row scales][32 bytes padding]
```

For activation row `b` and output row `i`, the generated kernel computes:

```text
Y[b, i] = row_scale[i] * sum_j(T[i, j] * Q8(X[b, j])) / activation_scale[b]
```

The runtime passes a scale stride of zero for scalar `TL2` and one for
`TL2_SR`. The exporter rejects raw payload sizes that do not match the exact
matrix shape, inserts the required 32-byte alignment before scales, and
requires the same generated kernel configuration used by the runtime.
The exporter embeds the config SHA-256 in GGUF metadata. At load time, the
runtime compares it with the fingerprint compiled into the generated kernel
header and rejects missing or mismatched layouts. Non-TL2 builds reject
`TL2_SR` artifacts outright.

All generated Qwen2.5-0.5B projection shapes pass deterministic scalar-reference
tests for batch sizes 1, 8, and 32 where applicable. Across the three tested
tile layouts, the worst relative RMS error is `5.50e-8`.

On the same 512 MNLI examples and same runtime build, `I2_SR` and `TL2_SR`
both score `0.667969`; they agree on `0.988281` of predictions, with three
examples correct only under each format (exact McNemar `p=1`). Packed ternary
projection storage falls `12.862%`, but complete-model storage falls only
`3.154%` because non-ternary tensors dominate.

The same-binary full validation covers all `9,815` MNLI examples. I2_SR scores
`0.651452`; BM64 TL2_SR scores `0.652878`. The paired delta is `+0.001426`
with bootstrap 95% CI `[-0.000917, +0.003872]`; prediction agreement is
`0.982578`, discordant correctness is `65/79`, and exact McNemar `p=0.278615`.
This supports function preservation, not TL2 quality superiority.

The speed result is negative. Five interleaved runs on 12 pinned Xeon 4116
physical cores give paired TL2_SR/I2_SR throughput ratios of `0.853` for
BM128, `0.866` for BM64, and `0.919` for BM32. The respective 95% intervals
are `[0.848, 0.858]`, `[0.835, 0.897]`, and `[0.917, 0.921]`, all below `1.0`.
Each run passed recorded idle checks across the selected cores and their
hyperthread siblings. See
`benchmarks/results/tl2sr_evidence_audit_2026-09-04.md`.
