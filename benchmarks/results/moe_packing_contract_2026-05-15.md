# MoE Packing Contract Audit, 2026-05-15

This audit uses nonzero synthetic merged expert tensors with shape `[experts, out, in]` to test whether current dense ternary packers support MoE weight layout and byte order.

## Checks

| check | accepted | error type | error | output shape | output bytes | layout verified | output sha | expected sha | verification error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tl2_merged_3d_expert | `true` | `` | `` | `[2901]` | `2901` | `n/a` | `24debdcbcd7a` | `` | `` |
| i2s_merged_3d_expert_codes | `true` | `` | `` | `[3072]` | `3072` | `true` | `f4c90257ad87` | `f4c90257ad87` | `` |
| i2s_merged_3d_expert_scalar | `true` | `` | `` | `[3104]` | `3104` | `true` | `16594d994623` | `16594d994623` | `` |
| i2sr_merged_3d_expert_row_scale | `true` | `` | `` | `[3232]` | `3232` | `true` | `7cffe9949c01` | `7cffe9949c01` | `` |
| i2s_2d_dense_control | `true` | `` | `` | `[3072]` | `3072` | `true` | `f4c90257ad87` | `f4c90257ad87` | `` |

## Verdict

| field | value |
| --- | --- |
| merged_3d_tl2_supported | `true` |
| merged_3d_i2s_code_packing_supported | `true` |
| merged_3d_i2s_scalar_supported | `true` |
| merged_3d_i2sr_row_scale_supported | `true` |
| merged_3d_i2s_i2sr_supported | `true` |
| dense_2d_i2s_control_supported | `true` |
| moe_packing_ready | `true` |

## Blockers

| blocker |
| --- |
| none |

## Required Next Steps

| step |
| --- |
| Fix TL2 runtime byte-size and stride semantics for tensors whose raw shape includes an expert dimension. |
| Route TL2 `ggml_mul_mat_id` through an expert-aware BitNet LUT kernel. |
| Add TL2 per-expert or per-expert-row scale metadata semantics before quality claims. |
| Add full GGUF byte-layout regression tests with a real MoE checkpoint. |
| Then run Kimi/Qwen2MoE quality, throughput, RSS, and expert-locality benchmarks. |
