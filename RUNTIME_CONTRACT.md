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
