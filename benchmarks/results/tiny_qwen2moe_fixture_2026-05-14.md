# Tiny Qwen2MoE Runtime Fixture, 2026-05-14

This fixture creates a tiny random `Qwen2MoeForCausalLM`, converts it to FP16 GGUF with the vendored llama.cpp converter, and runs `llama-cli` on CPU.

| gate | pass | evidence |
| --- | --- | --- |
| HF checkpoint created | yes | models/tiny-qwen2moe-fixture |
| GGUF converted | yes | models/tiny-qwen2moe-fixture/tiny-qwen2moe-f16.gguf |
| CPU smoke executed | yes | 0 |
| Peak RSS measured | yes | 105.09765625 MiB |
| Qwen2MoE metadata present | yes | experts=2; used=1 |

## Runtime Snapshot

Architecture: `qwen2moe`; params: `19.48` M; CPU buffer: `37.16` MiB; peak RSS: `105.09765625` MiB; prompt: `2424.83` tok/s; decode: `601.22` tok/s.

## Interpretation

This is a positive converter/runtime fixture for a minimal random Qwen2MoE FP16 model. It proves generic Qwen2MoE metadata and routed CPU execution are reachable in the vendored llama.cpp stack.

It does not prove Kimi support, ternary MoE support, BitDistill MoE training, router quality, task quality, expert-locality behavior, TL2 expert kernels, or row-scale `I2_SR` MoE quality.
