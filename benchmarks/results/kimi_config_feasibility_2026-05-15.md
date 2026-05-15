# Kimi Config Feasibility Audit, 2026-05-15

Overall status: `not_supported`.

Config source: `https://huggingface.co/moonshotai/Kimi-K2-Instruct/raw/main/config.json`.

## Architecture Summary

- model_type: `kimi_k2`
- architecture: `['DeepseekV3ForCausalLM']`
- layers: `61`
- hidden size: `7168`
- routed experts: `384`
- experts per token: `8`
- shared experts: `1`
- context length: `131072`
- source quantization: `{'activation_scheme': 'dynamic', 'fmt': 'e4m3', 'quant_method': 'fp8', 'weight_block_size': [128, 128]}`

## Required Support Checks

| required feature | repo signal | config evidence | note |
| --- | --- | --- | --- |
| Kimi/DeepSeekV3 architecture loader | fail | model_type=kimi_k2, architectures=['DeepseekV3ForCausalLM'] | Qwen/Qwen2MoE support does not imply Kimi-K2 loader support. |
| MLA/Q-LoRA attention metadata | fail | q_lora_rank=1536, kv_lora_rank=512, qk_nope=128, qk_rope=64 | llama.cpp has generic DeepSeek2 MLA runtime signals, but BitNet conversion must preserve Kimi/DeepSeekV3 metadata and tensor names. |
| Routed MoE experts | pass | n_routed_experts=384, num_experts_per_tok=8 | Generic MoE tensor shape support is weaker than Kimi-specific routed execution. |
| Shared expert path | fail | n_shared_experts=1 | Shared experts need distinct conversion, packing, and runtime accounting. |
| Block-FP8 source checkpoint import | fail | quant_method=fp8, fmt=e4m3, weight_block_size=[128, 128] | A ternary retrofit pipeline must first correctly dequantize/import the source checkpoint. |
| Packed row-scale ternary runtime | pass | required for this fork's I2_SR product path | Dense I2_SR support exists; Kimi MoE I2_SR quality/runtime remains unproven. |

## Verdict

Kimi-K2 support is not established in this fork. The config alone requires a Kimi/DeepSeekV3 loader, MLA attention layout handling, routed and shared expert conversion, block-FP8 import, and MoE-aware packed row-scale runtime validation before quality or speed claims are defensible.
