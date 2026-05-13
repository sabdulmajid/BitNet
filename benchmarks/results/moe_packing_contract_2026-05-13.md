# MoE Packing Contract Audit, 2026-05-13

This audit uses synthetic merged expert tensors with shape `[experts, out, in]` to test whether current dense ternary packers support MoE weight layout.

## Checks

| check | accepted | error type | error | output shape |
| --- | --- | --- | --- | --- |
| tl2_merged_3d_expert | `false` | `ValueError` | `too many values to unpack (expected 2)` | `` |
| i2s_merged_3d_expert | `false` | `ValueError` | `I2_S packing expects a 2D weight matrix, got shape (2, 4, 128)` | `` |
| i2s_2d_dense_control | `true` | `` | `` | `[128]` |

## Verdict

| field | value |
| --- | --- |
| merged_3d_tl2_supported | `false` |
| merged_3d_i2s_i2sr_supported | `false` |
| dense_2d_i2s_control_supported | `true` |
| moe_packing_ready | `false` |

## Blockers

| blocker |
| --- |
| TL2 preprocessing rejects merged 3D expert tensors before kernel lookup. |
| Direct I2_S/I2_SR code packing rejects merged 3D expert tensors. |

## Required Next Steps

| step |
| --- |
| Define a 3D expert tensor layout for TL2 and I2_SR instead of flattening expert identity away. |
| Add per-expert or per-expert-row scale metadata semantics. |
| Add byte-layout regression tests for merged expert tensors. |
| Only then run Kimi/Qwen2MoE quality, throughput, RSS, and expert-locality benchmarks. |
