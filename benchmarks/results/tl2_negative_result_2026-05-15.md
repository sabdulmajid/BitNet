# TL2 Negative Result Audit, 2026-05-15

TL2 has CPU execution evidence, but current one-scale TL2 is a negative result for learned row-scale checkpoints. I2_SR remains the supported row-scale packed path until TL2 gains row/group-scale metadata and kernels.

## Status

| field | value |
| --- | --- |
| TL2 CPU probes executed | true |
| TL2 probe has finite quality | false |
| row-scale TL2 runtime ready | false |
| runtime failed checks | 9 |
| negative result supported | true |

## TL2 CPU Probe Rows

| row | file MiB | PPL | prefill tok/s | decode tok/s | CPU executed | finite quality | path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen05b_qat_tl2 | 599.450073 | - | 265.776754 | 21.626749 | true | false | benchmark_results/gguf-qwen05b-tl2-probe-2026-05-05/summary.json |
| qwen05b_qat_tl2 | 599.450073 | NaN | 229.517043 | 22.951100 | true | false | benchmark_results/gguf-qwen05b-tl2-avx512-2026-05-05/summary.json |

## Scale Semantics

| case | relative output RMS error / value |
| --- | --- |
| tensor-scale checkpoint through current TL2 | 0.000000 |
| row-scale checkpoint through current TL2 | 1.904230 |
| row-scale best possible one tensor scale | 0.161173 |
| row-scale group32 fp16 scales | 0.142844 |
| row-scale exact fp16 row scales | 0.000197 |
| exact fp16 row-scale overhead MiB | 1.230469 |

## Blockers

- Existing TL2 collapses learned row scales to one scale and exceeds the allowed relative output RMS error.
- `transform_to_tl2` is data-only and recomputes a scalar scale instead of carrying checkpoint row/group scales.
- The converter still derives one scalar `np.max(abs(x))` scale and writes a single sidecar scale tensor.
- `GGML_TYPE_TL2` storage has no explicit row/group-scale count or dedicated row-scale TL2 qtype.
- `ggml_bitnet_transform_tensor` still builds metadata for one scale, not one scale per row or row group.
- Generated TL2 qgemm multiplies by `Scales[0]`; row-scale support needs kernels that index the learned output-row or row-group scale.
- The x86 TL2 dispatch passes the same `wt->scales` pointer into every generated qgemm call, matching the one-scale kernel contract.
- The active BitNet/Qwen TL2 loader path does not create learned row-scale sidecar tensors; TL2 expects scale metadata inside the packed tensor/kernel path.
- No row-scale TL2 GGUF has passed PPL, throughput, and RSS-style evidence.
