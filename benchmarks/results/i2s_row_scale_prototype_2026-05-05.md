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
The CPU supports AVX-512, but this validation used the portable AVX2 build so
it is not an AVX-512 peak-throughput claim.

Artifact:
`models/qwen2.5-1.5b-klonly-row-notie-static-ternary-dense/qwen15b_klonly_row_notie_static_ternary_i2_s_rowscale.gguf`

| metric | value |
| --- | ---: |
| fixed WikiText excerpt PPL | `38.8832 +/- 1.97093` |
| PPL prompt-eval throughput | `151.89 tok/s` |
| prompt throughput, `llama-bench -p 512` | `216.03 tok/s` |
| decode throughput, `llama-bench -n 128` | `18.83 tok/s` |
| GGUF file size | `1,211.3 MiB` |
| llama-bench model payload size | `1,264,210,048 bytes` |

Same-hardware generated suite:

| artifact | file MiB | fixed PPL | prompt tok/s | decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| FP F16 | 2,950.4 | 12.2808 | 114.47 | 5.56 |
| FP Q8_0 | 1,570.3 | 12.3056 | 124.86 | 10.13 |
| FP Q4_K_M | 940.4 | 12.8112 | 92.08 | 16.01 |
| row-scale static ternary F16 | 3,395.5 | 38.8651 | 114.75 | 5.49 |
| row-scale static ternary TQ2_0 | 1,218.6 | 38.8224 | 169.46 | 18.68 |
| row-scale static ternary I2_S prototype | 1,211.3 | 38.8832 | 216.03 | 18.83 |

Native `GGML_NATIVE=ON` AVX-512 check on the same Xeon:

| artifact | build | fixed PPL | prompt tok/s | decode tok/s | PPL tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| row-scale static ternary I2_S prototype | portable AVX2 | 38.8832 | 216.03 | 18.83 | 151.89 |
| row-scale static ternary I2_S prototype | native AVX-512 enabled | 38.8853 | 207.35 | 18.37 | 139.71 |

The native run reported `AVX512 = 1` in `system_info` and passed the strict
`I2_S`/reference PPL-ratio audit at `1.00128` under a `1.01` max-ratio
threshold. It did not improve throughput for this prototype on Skylake-SP.

Smoke prompt `The capital of France is` completed coherently:

```text
Paris. The capital of the United States is Washington. The capital of the United Kingdom is London. The capital of the
```

## Interpretation

The row-scale `I2_S` failure is not a fundamental limit of ternary CPU
execution. It is a scale-layout bug/limitation in the current packed format.
With per-row scales, `I2_S` preserves the row-scale checkpoint quality to the
same range as row-scale `TQ2_0` (`38.8224` PPL in the same suite), while
retaining commodity CPU execution.

This patch changes the binary layout of `I2_S` tensors. Existing tensor-scale
`I2_S` GGUF files should be regenerated with the patched writer before use.
The current kernel should not be marketed as AVX-512 accelerated until an
AVX-512-specific speedup is measured.
