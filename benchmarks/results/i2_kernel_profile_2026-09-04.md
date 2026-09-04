# I2 Kernel Cost Profile

Generated: `2026-09-04T10:47:43.256574+00:00`. Status: **valid**.

Protocol: `7` process repetitions; each reports the median of `31` timed calls over `32` activation rows on CPU `0`.

| projection (input x output) | uses/layer | A8 quantize us | I2 GEMM us | quantize share |
| --- | ---: | ---: | ---: | ---: |
| 896 x 896 | 2 | 44.100 | 487.816 | 8.29% |
| 896 x 128 | 2 | 44.749 | 67.103 | 40.00% |
| 896 x 4864 | 2 | 44.544 | 2726.427 | 1.61% |
| 4864 x 896 | 1 | 235.204 | 2089.447 | 10.13% |

## Aggregate Qwen2.5-0.5B Projection Mix

- Activation quantization share: `5.49%` (95% t interval `[5.29%, 5.69%]`).
- I2 dot/GEMM share: `94.51%`.
- Ideal upper-bound speedup from deleting A8 quantization entirely: `1.0581x`.
- Maximum scalar-reference error over all runs: `0.0` raw accumulator units.

## Interpretation

Activation quantization is a minority cost in the isolated projection mix. The packed I2 dot/GEMM implementation is therefore the material CPU optimization target; eliminating activation quantization alone cannot close the measured end-to-end gap to FP16.

## Claim Boundary

- This isolates one CPU core and the four dense projection shapes in a Qwen2.5-0.5B block.
- It measures raw activation quantization and packed I2 GEMM, excluding graph scheduling, normalization, attention, and model loading.
- The aggregate weights projections by architectural use count; it is not an end-to-end latency attribution.
- The reported upper bound is Amdahl's law for this isolated projection mix, not a forecast of model throughput.
