# Row-Scale I2_S Prototype, 2026-05-05

This is a local prototype result for the row-scale Qwen2.5-1.5B dense
`lm_head` checkpoint. It is not a default upstream `I2_S` result. The patch is
tracked at `patches/llama-i2s-row-scale.patch`.

## Finding

Current `I2_S` stores one tensor-level scale after the packed 2-bit weights.
That cannot represent row-scale ternary checkpoints, where each output row has
its own learned scale. The failure mode was measured earlier as fixed-excerpt
PPL `1,197,135.5848` with an incoherent smoke prompt.

The prototype changes the `I2_S` payload to store one float scale per output
row after the packed bytes, updates `ggml_nbytes`, and indexes the matching row
scale in matmul and `get_rows`.

## Validation

Hardware: Intel Xeon Silver 4116, 12 threads, portable AVX2 build, no BLAS.

Artifact:
`models/qwen2.5-1.5b-klonly-row-notie-static-ternary-dense/qwen15b_klonly_row_notie_static_ternary_i2_s_rowscale.gguf`

| metric | value |
| --- | ---: |
| fixed WikiText excerpt PPL | `38.8832 +/- 1.97093` |
| prompt throughput, `llama-bench -p 512` | `215.00 tok/s` |
| decode throughput, `llama-bench -n 128` | `18.83 tok/s` |
| model size | `1,264,210,048 bytes` |

Smoke prompt `The capital of France is` completed coherently:

```text
Paris. The capital of the United States is Washington. The capital of the United Kingdom is London. The capital of the
```

## Interpretation

The row-scale `I2_S` failure is not a fundamental limit of ternary CPU
execution. It is a scale-layout bug/limitation in the current packed format.
With per-row scales, `I2_S` preserves the row-scale checkpoint quality to the
same range as row-scale `TQ2_0` (`38.8224` PPL in the prior suite), while
retaining commodity CPU execution.

This patch changes the binary layout of `I2_S` tensors. Existing tensor-scale
`I2_S` GGUF files should be regenerated with the patched writer before use.
