# MoE Packing Contract Audit, 2026-05-13

This audit uses synthetic merged expert tensors with shape `[experts, out, in]` to test whether current dense ternary packers support MoE weight layout.

## Checks

| check | accepted | error type | error | output shape | output bytes |
| --- | --- | --- | --- | --- | --- |
| tl2_merged_3d_expert | `false` | `ValueError` | `too many values to unpack (expected 2)` | `` | `` |
| i2s_merged_3d_expert_codes | `true` | `` | `` | `[256]` | `256` |
| i2s_merged_3d_expert_scalar | `true` | `` | `` | `[288]` | `288` |
| i2sr_merged_3d_expert_row_scale | `true` | `` | `` | `[320]` | `320` |
| i2s_2d_dense_control | `true` | `` | `` | `[128]` | `128` |

## Verdict

| field | value |
| --- | --- |
| merged_3d_tl2_supported | `false` |
| merged_3d_i2s_code_packing_supported | `true` |
| merged_3d_i2s_scalar_supported | `true` |
| merged_3d_i2sr_row_scale_supported | `true` |
| merged_3d_i2s_i2sr_supported | `true` |
| dense_2d_i2s_control_supported | `true` |
| moe_packing_ready | `false` |

## Blockers

| blocker |
| --- |
| TL2 preprocessing still rejects merged 3D expert tensors before kernel lookup. |

## Required Next Steps

| step |
| --- |
| Define and implement a 3D expert tensor layout for TL2 instead of flattening expert identity away. |
| Add TL2 per-expert or per-expert-row scale metadata semantics. |
| Add full GGUF byte-layout regression tests for merged expert tensors after a real MoE checkpoint exists. |
| Then run Kimi/Qwen2MoE quality, throughput, RSS, and expert-locality benchmarks. |
