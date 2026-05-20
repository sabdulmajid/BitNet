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
| Native sequence-classifier path | Research demo; agreement below product gate. |
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
