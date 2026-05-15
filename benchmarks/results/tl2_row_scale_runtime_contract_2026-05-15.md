# TL2 Row-Scale Runtime Contract, 2026-05-15

This gate checks whether TL2 can safely carry row-scale ternary checkpoints. It is stricter than the design audit: passing requires both low mathematical error and concrete converter/runtime/benchmark evidence.

TL2 row-scale runtime ready: `false`.

## Math Summary

| field | value |
| --- | --- |
| label | qwen15b_row_scale |
| row_scale_tensors | 196 |
| current_tl2_tensor_max_error | 1.904230 |
| best_one_scale_error | 0.161173 |
| group32_fp16_error | 0.142844 |
| row_fp16_error | 0.000197 |
| row_fp16_scale_mib | 1.230469 |
| max_existing_tl2_error | 0.010000 |

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| Existing TL2 one-scale error is below product threshold | fail | design_json=benchmark_results/tl2_row_scale_design_2026-05-13.json; current_error=1.9042302114103853; threshold=0.01 | Existing TL2 collapses learned row scales to one scale and exceeds the allowed relative output RMS error. |
| TL2 converter accepts learned row/group scale metadata | fail | transform_to_tl2_line=711; accepts_scale=False; passes_scale_metadata=False; has_scale_map=True | `transform_to_tl2` is data-only and recomputes a scalar scale instead of carrying checkpoint row/group scales. |
| TL2 converter no longer recomputes one scalar max scale | fail | uses_np_max_abs=True; returns_single_scale=True; emits_scale_sidecar=True | The converter still derives one scalar `np.max(abs(x))` scale and writes a single sidecar scale tensor. |
| All sidecar-aware TL2 converter call sites refuse row-scale sidecars safely | pass | rejects_row_scale_tl2=True; all_tl2_data_calls=2; sidecar_aware_calls=1; bitnet_native_calls=1; uncovered_calls=0 |  |
| ggml TL2 storage accounts for row/group-scale sidecar | fail | ggml_nbytes_line=3490; nbytes_has_row_scale_sidecar=False; dedicated_row_tl2_type=False | `GGML_TYPE_TL2` storage has no explicit row/group-scale count or dedicated row-scale TL2 qtype. |
| TL2 transform metadata is row/group-scale aware | fail | bitnet_transform_line=1124; kernel_source=preset_kernels/bitnet_b1_58-large/bitnet-lut-kernels-tl2.h; uses_all_rows=False; single_scale_metadata=True | `ggml_bitnet_transform_tensor` still builds metadata for one scale, not one scale per row or row group. |
| Generated TL2 qgemm applies output-row or row-group scales | fail | kernel_source=preset_kernels/bitnet_b1_58-large/bitnet-lut-kernels-tl2.h; generated_uses_Scales0=True; codegen_emits_Scales0=True; kernel_mentions_row_scale=False; codegen_mentions_row_scale=False | Generated TL2 qgemm multiplies by `Scales[0]`; row-scale support needs kernels that index the learned output-row or row-group scale. |
| x86 TL2 matmul dispatch offsets scale metadata per output tile/row group | fail | mul_mat_tl2_line=12780; passes_unoffset_wt_scales=True; dedicated_row_tl2_type=False | The x86 TL2 dispatch passes the same `wt->scales` pointer into every generated qgemm call, matching the one-scale kernel contract. |
| BitNet/Qwen TL2 loader exposes learned scale sidecars when required | fail | bitnet_scale_sidecars_disabled_for_tl2=True; qwen_bitnet_loader_has_scale_sidecars=False | The active BitNet/Qwen TL2 loader path does not create learned row-scale sidecar tensors; TL2 expects scale metadata inside the packed tensor/kernel path. |
| A compatibility-safe row-scale runtime path exists | pass | GGML_TYPE_I2_SR=True |  |
| Row-scale TL2 has quality and speed benchmark evidence | fail | row_scale_tl2_rows=0; finite_quality_rows=0 | No row-scale TL2 GGUF has passed PPL, throughput, and RSS-style evidence. |

## Row-Scale TL2 Benchmark Evidence

| summary | name | file MiB | PPL | prefill tok/s | decode tok/s |
| --- | --- | --- | --- | --- | --- |
| none | - | - | - | - | - |

## Blockers

| blocker |
| --- |
| Existing TL2 collapses learned row scales to one scale and exceeds the allowed relative output RMS error. |
| `transform_to_tl2` is data-only and recomputes a scalar scale instead of carrying checkpoint row/group scales. |
| The converter still derives one scalar `np.max(abs(x))` scale and writes a single sidecar scale tensor. |
| `GGML_TYPE_TL2` storage has no explicit row/group-scale count or dedicated row-scale TL2 qtype. |
| `ggml_bitnet_transform_tensor` still builds metadata for one scale, not one scale per row or row group. |
| Generated TL2 qgemm multiplies by `Scales[0]`; row-scale support needs kernels that index the learned output-row or row-group scale. |
| The x86 TL2 dispatch passes the same `wt->scales` pointer into every generated qgemm call, matching the one-scale kernel contract. |
| The active BitNet/Qwen TL2 loader path does not create learned row-scale sidecar tensors; TL2 expects scale metadata inside the packed tensor/kernel path. |
| No row-scale TL2 GGUF has passed PPL, throughput, and RSS-style evidence. |

## Required Implementation Steps

| step |
| --- |
| Add a compatibility-safe TL2 row/group-scale qtype or explicit TL2 metadata version; do not overload current single-scale TL2 silently. |
| Teach the converter to pass learned tensor/row/group scales into TL2 packing instead of recomputing one `max(abs(W))` scale. |
| Extend GGUF/ggml byte-size semantics so packed TL2 data carries the exact number of fp16/fp32 row or row-group scales. |
| Update generated TL2 transform metadata and kernels to index the correct scale for each output row or row group. |
| Update x86 TL2 matmul dispatch so each tile sees the correct scale span, and add loader-side compatibility guards for old single-scale TL2 files. |
| Run dense Qwen PPL, lm-eval/task quality, llama-bench throughput, and RSS benchmarks before enabling any product claim. |

## Verdict

Current TL2 is not a supported path for the strongest row-scale checkpoint. The supported packed row-scale path remains `I2_SR`; TL2 needs a metadata and kernel extension before it can be benchmarked as a quality-preserving alternative.
