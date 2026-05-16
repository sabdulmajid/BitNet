# TL2 Row-Scale Implementation Plan, 2026-05-15

Do not productize TL2 for learned row-scale checkpoints yet. The required work is a new row/group-scale runtime contract, not a benchmark rerun.

## Current Math

| field | value |
| --- | --- |
| contract ready | false |
| failed checks | 9 |
| current one-scale relative output RMS error | 1.904230 |
| exact row-scale FP16 relative output RMS error | 0.000197 |
| row-scale storage overhead MiB | 1.230469 |

## Failed Contract Checks

| check | evidence | blocker |
| --- | --- | --- |
| Existing TL2 one-scale error is below product threshold | design_json=benchmark_results/tl2_row_scale_design_2026-05-13.json; current_error=1.9042302114103853; threshold=0.01 | Existing TL2 collapses learned row scales to one scale and exceeds the allowed relative output RMS error. |
| TL2 converter accepts learned row/group scale metadata | transform_to_tl2_line=711; accepts_scale=False; passes_scale_metadata=False; has_scale_map=True | `transform_to_tl2` is data-only and recomputes a scalar scale instead of carrying checkpoint row/group scales. |
| TL2 converter no longer recomputes one scalar max scale | uses_np_max_abs=True; returns_single_scale=True; emits_scale_sidecar=True | The converter still derives one scalar `np.max(abs(x))` scale and writes a single sidecar scale tensor. |
| ggml TL2 storage accounts for row/group-scale sidecar | ggml_nbytes_line=3490; nbytes_has_row_scale_sidecar=False; dedicated_row_tl2_type=False | `GGML_TYPE_TL2` storage has no explicit row/group-scale count or dedicated row-scale TL2 qtype. |
| TL2 transform metadata is row/group-scale aware | bitnet_transform_line=1124; kernel_source=preset_kernels/bitnet_b1_58-large/bitnet-lut-kernels-tl2.h; uses_all_rows=False; single_scale_metadata=True | `ggml_bitnet_transform_tensor` still builds metadata for one scale, not one scale per row or row group. |
| Generated TL2 qgemm applies output-row or row-group scales | kernel_source=preset_kernels/bitnet_b1_58-large/bitnet-lut-kernels-tl2.h; generated_uses_Scales0=True; codegen_emits_Scales0=True; kernel_mentions_row_scale=False; codegen_mentions_row_scale=False | Generated TL2 qgemm multiplies by `Scales[0]`; row-scale support needs kernels that index the learned output-row or row-group scale. |
| x86 TL2 matmul dispatch offsets scale metadata per output tile/row group | mul_mat_tl2_line=12780; passes_unoffset_wt_scales=True; dedicated_row_tl2_type=False | The x86 TL2 dispatch passes the same `wt->scales` pointer into every generated qgemm call, matching the one-scale kernel contract. |
| BitNet/Qwen TL2 loader exposes learned scale sidecars when required | bitnet_scale_sidecars_disabled_for_tl2=True; qwen_bitnet_loader_has_scale_sidecars=False | The active BitNet/Qwen TL2 loader path does not create learned row-scale sidecar tensors; TL2 expects scale metadata inside the packed tensor/kernel path. |
| Row-scale TL2 has quality and speed benchmark evidence | row_scale_tl2_rows=0; finite_quality_rows=0 | No row-scale TL2 GGUF has passed PPL, throughput, and RSS-style evidence. |

## Patch Sequence

| step | files | work | exit gate |
| --- | --- | --- | --- |
| 1. Define a row/group-scale TL2 representation | 3rdparty/llama.cpp/ggml/include/ggml.h, 3rdparty/llama.cpp/gguf-py/gguf/constants.py | Add either a dedicated row-scale TL2 qtype or explicit metadata for the number and layout of learned scales. Reusing GGML_TYPE_TL2 without changing byte-size semantics is unsafe. | ggml row-size/nbytes can represent packed codes plus row/group scales without undercounting bytes. |
| 2. Make the converter sidecar-aware for TL2 | utils/convert-hf-to-gguf-bitnet.py, benchmarks/convert_static_ternary_to_i2s_gguf.py | Thread checkpoint `weight_scale` tensors into TL2 export instead of recomputing a single `np.max(abs(x))` scale. Reject ambiguous row-scale exports unless the new qtype/metadata is active. | A row-scale checkpoint exports learned scales byte-for-byte and scalar TL2 continues to export unchanged. |
| 3. Regenerate TL2 transform metadata for multiple scales | utils/codegen_tl2.py, preset_kernels/bitnet_b1_58-large/bitnet-lut-kernels-tl2.h, include/bitnet-lut-kernels.h | Replace the one-scale `lut_scales_size = 1` contract with row/group-scale metadata and regenerate kernels. Generated qgemm must index the output row or row group, not `Scales[0]`. | Static audit shows no generated row-scale path multiplies by `Scales[0]` for every output row. |
| 4. Offset scales in x86 TL2 matmul dispatch | 3rdparty/llama.cpp/ggml/src/ggml.c | Pass the correct scale slice into each generated qgemm call, or pass a base pointer plus row offset. The dispatch must match the kernel's row/group-scale indexing contract. | Layer-level relative output RMS error falls below 0.01 on the row-scale Qwen1.5B design audit. |
| 5. Expose learned scale sidecars in the BitNet/Qwen loader | 3rdparty/llama.cpp/src/llama.cpp | Load learned scale tensors for the Qwen-compatible BitNet graph when the model uses row/group-scale TL2. The current TL2 path hides scale sidecars behind packed tensor metadata. | Converted GGUF loads without missing scale tensors and without silently falling back to scalar TL2. |
| 6. Run CPU quality, speed, and RSS gates | benchmarks/audit_tl2_row_scale_runtime_contract.py, benchmarks/audit_tl2_negative_result.py, benchmarks/build_qwen_side_by_side.py | Convert the row-scale Qwen1.5B checkpoint, run perplexity/throughput/RSS on the Xeon, and compare against I2_SR, TQ2_0, Q4_K_M, and FP16. | Row-scale TL2 has finite PPL near the I2_SR row-scale artifact, benchmark rows include prefill/decode throughput and RSS, and the runtime contract audit passes. |

## Completion Criteria

This blocker is closed only when the row-scale TL2 runtime contract audit passes, a row-scale TL2 GGUF has finite quality/speed/RSS evidence on the Xeon, and the side-by-side table compares that artifact against I2_SR and Q4_K_M. Until then, I2_SR remains the deployable row-scale path.
