# Tiny Qwen2MoE Ternary I2_SR Runtime Fixture, 2026-05-15

This fixture creates a tiny random `Qwen2MoeForCausalLM`, merges expert weights into 3D row-scale ternary tensors, exports `MOSTLY_I2_SR` GGUF, and runs `llama-cli` on CPU.

| gate | pass | evidence |
| --- | --- | --- |
| HF checkpoint created | yes | models/tiny-qwen2moe-ternary-i2sr-fixture |
| Merged ternary expert state built | yes | models/tiny-qwen2moe-ternary-i2sr-fixture/ternary_state_dict.pt |
| I2_SR GGUF conversion finished | yes | models/tiny-qwen2moe-ternary-i2sr-fixture/tiny-qwen2moe-ternary-i2sr.gguf |
| 3D expert tensors packed as row-scale I2_SR | yes | packed=3; row_scale=3 |
| CPU routed smoke executed | yes | 0 |
| Peak RSS measured | yes | 142.48046875 MiB |
| Qwen2MoE metadata present | yes | experts=2; used=1 |

## Runtime Snapshot

Passed: `True`; architecture: `qwen2moe`; params: `39.02` M; file: `79.9215087890625` MiB; CPU buffer: `74.27` MiB; peak RSS: `142.48046875` MiB; prompt: `1906.21` tok/s; decode: `419.29` tok/s; fatal: ``.

## Interpretation

A pass is positive evidence that the direct I2_SR writer and the current CPU runtime can carry merged 3D row-scale ternary expert tensors through routed Qwen2MoE execution. A failure is still a hard boundary artifact because it records the exact failing conversion/runtime command.

This does not prove Kimi support, trained MoE quality, router distillation, task accuracy, TL2 expert kernels, or expert-locality throughput on a real model.
