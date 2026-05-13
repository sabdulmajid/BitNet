# MoE TL2 Runtime Contract Audit, 2026-05-13

This audit checks whether the current TL2 converter and runtime contracts can safely carry llama.cpp-style merged expert tensors with shape `[experts, out, in]`.

TL2 MoE runtime ready: `false`.

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| TL2 Python preprocessor accepts explicit 3D expert tensors | fail | preprocess_weights_tl2_line=628; uses_M_K_unpack=True; has_ndim_branch=False | `preprocess_weights_tl2` still unpacks `M, K = w.shape`, so [experts, out, in] tensors fail before packing. |
| TL2 ggml_nbytes accounts for expert dimension | fail | ggml_nbytes_line=3477; uses_ne2=False; uses_ne3=False; shape=[4,256,384]; active_bytes=23264; flat_expert_bytes=92896; underreport_bytes=69632; underreport_ratio=3.993122 | The active TL2 byte-size formula ignores tensor->ne[2]/ne[3], so merged expert tensors would be under-sized. |
| TL2 LUT kernel is routed for normal dense matmul | pass | mul_mat_line=12611; has_tl2_lut=True |  |
| TL2 LUT kernel is routed for MoE ggml_mul_mat_id | fail | mul_mat_id_line=13328; has_tl2_lut=False; type_traits_vec_dot_f32=True | `ggml_compute_forward_mul_mat_id` does not route TL2 through the BitNet LUT kernel; TL2 type traits fall back to F32 vec-dot semantics. |
| llama.cpp generic quantizer is at least 3D-expert aware | pass | quantize_loop_line=18959; uses_tensor_ne2=True |  |

## Byte-Size Probe

| field | value |
| --- | --- |
| synthetic shape | `{'experts': 4, 'out': 256, 'in': 384}` |
| active one-expert bytes | `23264` |
| flat expert bytes | `92896` |
| underreport bytes | `69632` |
| underreport ratio | `3.993122` |

## Blockers

| blocker |
| --- |
| `preprocess_weights_tl2` still unpacks `M, K = w.shape`, so [experts, out, in] tensors fail before packing. |
| The active TL2 byte-size formula ignores tensor->ne[2]/ne[3], so merged expert tensors would be under-sized. |
| `ggml_compute_forward_mul_mat_id` does not route TL2 through the BitNet LUT kernel; TL2 type traits fall back to F32 vec-dot semantics. |

## Required Next Steps

| step |
| --- |
| Add an explicit TL2 3D tensor preprocessor contract for [experts, out, in] instead of relying on a 2D unpack. |
| Change the TL2 byte-size/stride contract so GGUF loading allocates all expert planes. |
| Route ggml_mul_mat_id for TL2 through expert-aware BitNet LUT kernels, not generic F32 vec-dot fallback. |
| Define per-expert tensor-scale or row/group-scale metadata semantics before quality benchmarking. |
| Validate with a real Qwen2MoE/Kimi GGUF and router/expert-locality benchmarks. |
