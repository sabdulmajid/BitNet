# Direct Row-Scale I2_S Qwen0.5B Control, 2026-05-13

## Verdict

This is a historical negative control from before the x86 `ACT_PARALLEL`
packing fix. The artifact below used a row-group-of-four packing layout and did
**not** produce a quality-valid Qwen2.5-0.5B row-scale model in the current
default runtime.

The current direct writer now uses the active x86 chunk-128 layout instead. The
quality-valid direct row-scale evidence is the fixed `I2_SR` Qwen2.5-1.5B run
recorded in `benchmarks/results/i2sr_x86act_fix_2026-05-13.md`. This file
remains useful as a negative control, not as the current direct-writer result.

## What Changed

The tested artifact packed four output rows at the same input column per byte:

```text
byte = q(row0, col) << 6 | q(row1, col) << 4 | q(row2, col) << 2 | q(row3, col)
```

Scalar-scale mode still uses the existing tensor-scale `I2_S` payload and
rejects row-scale checkpoints by default. Passing `--row-scale-prototype`
writes one float32 scale per output row after the packed codes. That prototype
requires a matching experimental runtime layout; it is not a stable GGUF
contract.

## Evidence

All rows below used the fixed WikiText excerpt, `ctx=512`, 12 CPU threads, and
three benchmark repetitions on the same Xeon host. They were gathered from
separate targeted suites, so use the quality columns as the primary comparison
and treat throughput as the measured per-artifact run.

| artifact | file MiB | PPL | PPL tok/s | prefill tok/s | decode tok/s | smoke |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen2.5-0.5B FP16 baseline | 948.1 | 18.0986 | 238.25 | 365.32 | 16.61 | `Paris. It is the capital of France. It is the capital of France.` |
| scalar direct `I2_S`, historical row-group layout | 610.6 | NaN | 291.05 | 523.69 | 37.30 | `!!!!!!!!!!!!!!!!` |
| row-scale checkpoint materialized as F16 | 1207.8 | 578.4833 | 238.69 | 367.60 | 16.46 | `a major part of the country's economic development. The country's economic growth is` |
| row-scale direct `I2_S` prototype, historical row-group layout | 611.7 | 59401.5449 | 301.96 | 544.87 | 39.56 | `,,,,,,,,,,,,,,,,` |
| row-scale materialized then `I2_S` quantized | 490.6 | NaN | 300.70 | 548.13 | 56.47 | `!!!!!!!!!!!!!!!!` |
| row-scale materialized then `TQ2_0` quantized single-thread | 491.5 | 5118527.5782 | 257.79 | 430.52 | 57.92 | corrupted repeated `opal` text |

Conversion summary for direct row-scale prototype:

- checkpoint: `checkpoints/qwen2.5-0.5b-fineweb-edu-row-1000/step-1000`;
- output: `models/qwen2.5-0.5b-direct-static-ternary/qwen05b_row_direct_i2_s_rowscale_prototype.gguf`;
- packed ternary tensors: `168`;
- row-scale packed tensors: `168`;
- F16 output tensors: `1`;
- copied tensors: `122`;
- GGUF tensors: `291`;
- packed `I2_S` payload bytes: `90,678,528`;
- output size: `641,448,448` bytes;
- vendored Python GGUF native constants for `I2_S`: `false`.

## Interpretation

The corrected direct writer proves that static ternary weights can be packed
without first materializing dense F16 tensors. It does not prove that the
default `I2_S` runtime can preserve row-scale semantics.

The 0.5B row-scale checkpoint is already weak when materialized as dense F16
on this fixed excerpt (`578.4833` PPL), but the packed `I2_S` and `TQ2_0`
controls are much worse. Therefore this artifact family should be treated as a
failure probe. The quality-preserving row-scale packed results are the separate
Qwen2.5-1.5B experiments: the row-scale `I2_S` prototype reached PPL
`38.8832` versus `38.8224` for `TQ2_0`, and the fixed candidate `I2_SR` path
reached PPL `38.8477` after switching the direct writer to the active x86
`ACT_PARALLEL` packing layout.

Product implication: direct packed export is worth continuing, but row-scale
deployment needs a stable new row-scale-aware qtype or versioned layout plus
matching writer, reader, and matmul kernels. Do not claim default/upstream
`I2_S` preserves row-scale Qwen checkpoints.
