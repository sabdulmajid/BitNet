# MoE Support Audit, 2026-05-05

| check | path | status | expectation | evidence |
| --- | --- | --- | --- | --- |
| GGUF expert metadata | 3rdparty/llama.cpp/gguf-py/gguf/constants.py | present | metadata can record expert count and active experts | `EXPERT_COUNT`@88, `EXPERT_USED_COUNT`@89 |
| Qwen2MoE tensor schema | 3rdparty/llama.cpp/gguf-py/gguf/constants.py | present | GGUF schema has merged expert tensors for Qwen2MoE | `MODEL_ARCH.QWEN2MOE`@378, `MODEL_TENSOR.FFN_GATE_EXP`@448, `MODEL_TENSOR.FFN_DOWN_EXP`@449, `MODEL_TENSOR.FFN_UP_EXP`@450 |
| llama.cpp Qwen2MoE converter | 3rdparty/llama.cpp/convert_hf_to_gguf.py | present | vendored converter registers Qwen2MoE and merges experts | `@Model.register("Qwen2MoeForCausalLM")`@1975, `MODEL_ARCH.QWEN2MOE`@1977, `torch.stack(datas, dim=0)`@1494 |
| BitNet converter generic expert packing | utils/convert-hf-to-gguf-bitnet.py | present | BitNet converter has generic Mixtral-style expert metadata/packing | `num_local_experts`@149, `num_experts_per_tok`@152, `block_sparse_moe.experts`@881 |
| Runtime sparse expert execution | 3rdparty/llama.cpp/src/llama.cpp | present | runtime builds top-k routed sparse expert matmuls | `llm_build_moe_ffn`@9653, `ggml_soft_max`@9675, `ggml_top_k`@9679, `ggml_mul_mat_id`@9474 |

## Negative Checks

No Kimi-specific converter/runtime mapping was found in tracked source files.
No local Kimi benchmark artifacts were found under benchmark_results.

## Verdict

Generic MoE infrastructure is present: GGUF metadata has expert counts, Qwen2MoE is registered in the vendored llama.cpp converter, expert weights are merged into 3D tensors, and the runtime builds sparse top-k expert execution with `ggml_mul_mat_id`. This does not prove Kimi support: no Kimi-specific mapping or benchmark artifact is present, and the TL2-capable BitNet converter still lacks Qwen2MoE registration.
