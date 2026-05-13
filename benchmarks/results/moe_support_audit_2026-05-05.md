# MoE Support Audit, 2026-05-05

| check | path | status | expectation | evidence |
| --- | --- | --- | --- | --- |
| GGUF expert metadata | 3rdparty/llama.cpp/gguf-py/gguf/constants.py | present | metadata can record expert count and active experts | `EXPERT_COUNT`@88, `EXPERT_USED_COUNT`@89 |
| Qwen2MoE tensor schema | 3rdparty/llama.cpp/gguf-py/gguf/constants.py | present | GGUF schema has merged expert tensors for Qwen2MoE | `MODEL_ARCH.QWEN2MOE`@378, `MODEL_TENSOR.FFN_GATE_EXP`@448, `MODEL_TENSOR.FFN_DOWN_EXP`@449, `MODEL_TENSOR.FFN_UP_EXP`@450 |
| llama.cpp Qwen2MoE converter | 3rdparty/llama.cpp/convert_hf_to_gguf.py | present | vendored converter registers Qwen2MoE and merges experts | `@Model.register("Qwen2MoeForCausalLM")`@1975, `MODEL_ARCH.QWEN2MOE`@1977, `torch.stack(datas, dim=0)`@1494 |
| BitNet converter generic expert packing | utils/convert-hf-to-gguf-bitnet.py | present | BitNet converter has generic Mixtral-style expert metadata/packing | `num_local_experts`@155, `num_experts_per_tok`@158, `block_sparse_moe.experts`@913 |
| Runtime sparse expert execution | 3rdparty/llama.cpp/src/llama.cpp | present | runtime builds top-k routed sparse expert matmuls | `llm_build_moe_ffn`@9653, `ggml_soft_max`@9675, `ggml_top_k`@9679, `ggml_mul_mat_id`@9474 |

## Productization Gates

| gate | status | evidence | blocker |
| --- | --- | --- | --- |
| generic GGUF/runtime MoE support exists | pass | gguf_schema=True; runtime=True; llama_qwen2moe_converter=True |  |
| BitNet converter has explicit Qwen2MoE/Kimi registration | fail | qwen2moe_registration=False; kimi_converter_match=False; tracked_kimi_mentions=4 | The TL2-capable BitNet converter registers Qwen2 dense/Mixtral-style models but not Qwen2MoE or Kimi. |
| TL2 converter path is validated for merged 3D expert tensors | fail | contract_available=True; contract_tl2_3d=False; preprocess_weights_tl2_uses_2d_unpack=True | `preprocess_weights_tl2` unpacks `M, K = w.shape`, so merged expert tensors with shape [experts, out, in] are not supported. |
| direct I2_SR writer is validated for merged 3D expert tensors | fail | contract_available=True; contract_i2sr_3d=False; contract_2d_control=True; direct_i2sr_writer_rejects_non_2d=True | The direct packed I2_S/I2_SR writer rejects non-2D ternary weights, so merged expert tensors need new packing/layout tests. |
| local Kimi model/eval artifacts exist | fail | kimi_artifacts=0 | No local Kimi checkpoint, conversion, quality, throughput, RSS, or expert-locality artifact exists. |
| MoE quality and locality benchmarks exist | fail | quality_runs=0; throughput_runs=0; expert_locality_runs=0 | No benchmark measures router accuracy, expert selection locality, sparse expert throughput, or quality degradation. |

## Negative Checks

Kimi string matches in tracked source files: 4.
No local Kimi benchmark artifacts were found under benchmark_results.
Synthetic MoE packing contract: TL2 3D supported=False; I2_S/I2_SR 3D supported=False; 2D control supported=True.

## Verdict

Generic MoE infrastructure is present: GGUF metadata has expert counts, Qwen2MoE is registered in the vendored llama.cpp converter, expert weights are merged into 3D tensors, and the runtime builds sparse top-k expert execution with `ggml_mul_mat_id`. This does not prove Kimi support: no Kimi-specific mapping or benchmark artifact is present, the TL2-capable BitNet converter still lacks Qwen2MoE registration, and the current TL2/I2_SR packing code paths are 2D-matrix oriented rather than validated for merged 3D expert tensors.

## Required Plan

Required MoE/Kimi path: add an explicit Kimi/Qwen2MoE BitNet converter registration, map router/shared-expert/expert tensor names, decide which router and shared-expert tensors stay dense, extend TL2/I2_SR packing and runtime tests to 3D expert tensors, distill router and expert weights under ternary constraints, then run quality, throughput, RSS, and expert-locality benchmarks against dense and llama.cpp quantized MoE baselines.
