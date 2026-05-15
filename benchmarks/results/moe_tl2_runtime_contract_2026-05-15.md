# MoE TL2 Runtime Contract Audit, 2026-05-15

This audit checks whether the current TL2 converter and runtime contracts can safely carry llama.cpp-style merged expert tensors with shape `[experts, out, in]`.

TL2 MoE runtime ready: `false`.

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| TL2 Python preprocessor accepts explicit 3D expert tensors | pass | preprocess_weights_tl2_line=628; legacy_2d_unpack_present=True; has_ndim_branch=True; has_3d_branch=True |  |
| TL2 ggml_nbytes accounts for expert dimension | pass | ggml_nbytes_line=3490; uses_nrows=True; uses_ne2=True; uses_ne3=True; shape=[4,256,384]; one_expert_bytes=23264; active_3d_bytes=92896; flat_expert_bytes=92896; underreport_bytes=0; underreport_ratio=1.000000 |  |
| TL2 LUT kernel is routed for normal dense matmul | pass | mul_mat_line=12634; has_tl2_lut=True |  |
| TL2 LUT kernel is routed for MoE ggml_mul_mat_id | fail | mul_mat_id_line=13363; has_tl2_lut=False; type_traits_vec_dot_f32=True | `ggml_compute_forward_mul_mat_id` does not route TL2 through the BitNet LUT kernel; TL2 type traits fall back to F32 vec-dot semantics. |
| TL2 mul_mat_id scratch buffer sizes BitNet LUT workspace | fail | mul_mat_id_wsize_line=18095; uses_bitnet_wsize=False | `GGML_OP_MUL_MAT_ID` workspace sizing does not reserve BitNet LUT preprocessor buffers, so adding a kernel route also requires a wsize contract. |
| TL2 tensor transform metadata is expert-plane aware | fail | bitnet_transform_line=1124; kernel_source=preset_kernels/bitnet_b1_58-large/bitnet-lut-kernels-tl2.h; uses_3d_rows=False; single_scale_metadata=True | `ggml_bitnet_transform_tensor` derives metadata from only `tensor->ne[1]` and stores one tensor scale, so merged experts lack explicit per-expert offsets/scales. |
| llama.cpp generic quantizer is at least 3D-expert aware | pass | quantize_loop_line=18964; uses_tensor_ne2=True |  |

## Byte-Size Probe

| field | value |
| --- | --- |
| synthetic shape | `{'experts': 4, 'out': 256, 'in': 384}` |
| one-expert bytes | `23264` |
| active 3D bytes | `92896` |
| flat expert bytes | `92896` |
| underreport bytes | `0` |
| underreport ratio | `1.000000` |

## Blockers

| blocker |
| --- |
| `ggml_compute_forward_mul_mat_id` does not route TL2 through the BitNet LUT kernel; TL2 type traits fall back to F32 vec-dot semantics. |
| `GGML_OP_MUL_MAT_ID` workspace sizing does not reserve BitNet LUT preprocessor buffers, so adding a kernel route also requires a wsize contract. |
| `ggml_bitnet_transform_tensor` derives metadata from only `tensor->ne[1]` and stores one tensor scale, so merged experts lack explicit per-expert offsets/scales. |

## Required Next Steps

| step |
| --- |
| Route ggml_mul_mat_id for TL2 through expert-aware BitNet LUT kernels, not generic F32 vec-dot fallback. |
| Extend GGML_OP_MUL_MAT_ID workspace sizing to reserve TL2 LUT preprocessor buffers. |
| Define per-expert tensor-scale or row/group-scale metadata semantics before quality benchmarking. |
| Make TL2 tensor transform metadata expert-plane aware so qweight offsets and scales are explicit per expert. |
| Validate with a real Qwen2MoE/Kimi GGUF and router/expert-locality benchmarks. |
