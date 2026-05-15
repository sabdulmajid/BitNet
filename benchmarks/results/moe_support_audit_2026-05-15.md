# MoE Support Audit, 2026-05-15

| check | path | status | expectation | evidence |
| --- | --- | --- | --- | --- |
| GGUF expert metadata | 3rdparty/llama.cpp/gguf-py/gguf/constants.py | present | metadata can record expert count and active experts | `EXPERT_COUNT`@88, `EXPERT_USED_COUNT`@89 |
| Qwen2MoE tensor schema | 3rdparty/llama.cpp/gguf-py/gguf/constants.py | present | GGUF schema has merged expert tensors for Qwen2MoE | `MODEL_ARCH.QWEN2MOE`@378, `MODEL_TENSOR.FFN_GATE_EXP`@448, `MODEL_TENSOR.FFN_DOWN_EXP`@449, `MODEL_TENSOR.FFN_UP_EXP`@450 |
| llama.cpp Qwen2MoE converter | 3rdparty/llama.cpp/convert_hf_to_gguf.py | present | vendored converter registers Qwen2MoE and merges experts | `@Model.register("Qwen2MoeForCausalLM")`@1975, `MODEL_ARCH.QWEN2MOE`@1977, `torch.stack(datas, dim=0)`@1494 |
| BitNet converter generic expert packing | utils/convert-hf-to-gguf-bitnet.py | present | BitNet converter has generic Mixtral-style expert metadata/packing | `num_local_experts`@155, `num_experts_per_tok`@158, `block_sparse_moe.experts`@939 |
| BitNet converter Qwen2MoE registration | utils/convert-hf-to-gguf-bitnet.py | present | BitNet converter explicitly registers Qwen2MoE and merges expert tensors | `@Model.register("Qwen2MoeForCausalLM")`@1028, `MODEL_ARCH.QWEN2MOE`@1030, `torch.stack(datas, dim=0)`@961 |
| Runtime sparse expert execution | 3rdparty/llama.cpp/src/llama.cpp | present | runtime builds top-k routed sparse expert matmuls | `llm_build_moe_ffn`@9655, `ggml_soft_max`@9677, `ggml_top_k`@9681, `ggml_mul_mat_id`@9476 |

## Productization Gates

| gate | status | evidence | blocker |
| --- | --- | --- | --- |
| generic GGUF/runtime MoE support exists | pass | gguf_schema=True; runtime=True; llama_qwen2moe_converter=True |  |
| BitNet converter has explicit Qwen2MoE-or-Kimi registration | pass | qwen2moe_registration=True; kimi_converter_match=False; tracked_kimi_mentions=0 |  |
| TL2 converter path is validated for merged 3D expert tensors | fail | contract_available=True; contract_tl2_3d=True; preprocess_weights_tl2_has_legacy_2d_branch=True; runtime_ready=False; runtime_blockers=3; tl2_expert_byte_underreport=0 | The TL2 converter accepts the synthetic 3D packing contract and byte sizing now accounts for expert planes, but routed TL2 experts still lack a `ggml_mul_mat_id` LUT route, matching workspace sizing, and expert-aware transform/scale metadata. |
| direct I2_SR writer is validated for merged 3D expert tensors | pass | contract_available=True; contract_i2sr_3d=True; contract_2d_control=True; direct_i2sr_writer_rejects_non_2d=False |  |
| tiny Qwen2MoE ternary I2_SR GGUF CPU fixture passes | pass | passed=True; arch=qwen2moe; experts=2; used=1; packed=3; row_scale=3; decode_tok_s=419.29; peak_rss_mib=142.48046875 |  |
| tiny Qwen2MoE FP16 GGUF CPU fixture passes | pass | passed=True; arch=qwen2moe; experts=2; used=1; decode_tok_s=601.22; peak_rss_mib=105.09765625 |  |
| synthetic Qwen2MoE expert-scaling CPU probe passes | pass | passed=True; rows=4 |  |
| local Kimi model/eval artifacts exist | fail | kimi_artifacts=0 | No local Kimi checkpoint, conversion, quality, throughput, RSS, or expert-locality artifact exists. |
| MoE quality and locality benchmarks exist | fail | quality_runs=0; trained_throughput_runs=0; synthetic_expert_scaling_rows=4 | No trained MoE benchmark measures router accuracy, expert selection locality, sparse expert throughput, or quality degradation. |

## Negative Checks

No Kimi-specific converter/runtime mapping was found in converter/runtime source files.
No local Kimi benchmark artifacts were found under benchmark_results.
Synthetic MoE packing contract: TL2 3D supported=True; I2_S/I2_SR 3D supported=True; 2D control supported=True.
TL2 MoE runtime contract: ready=False; blockers=3.
Tiny Qwen2MoE FP16 runtime fixture: passed=True; arch=qwen2moe; experts=2; used=1; decode_tok_s=601.22; peak_rss_mib=105.09765625.
Tiny Qwen2MoE ternary I2_SR runtime fixture: passed=True; packed=3; row_scale=3; arch=qwen2moe; experts=2; used=1; decode_tok_s=419.29; peak_rss_mib=142.48046875.
Tiny Qwen2MoE expert-scaling probe: passed=True; rows=4.

## Verdict

Generic MoE infrastructure is present: GGUF metadata has expert counts, Qwen2MoE is registered in the vendored llama.cpp converter, expert weights are merged into 3D tensors, and the runtime builds sparse top-k expert execution with `ggml_mul_mat_id`. A tiny random Qwen2MoE FP16 fixture validates converter-to-runtime plumbing on CPU, and a tiny random Qwen2MoE ternary fixture now validates merged 3D expert GGUF export plus routed CPU execution with row-scale `I2_SR`. This does not prove Kimi support: no Kimi-specific mapping or benchmark artifact is present, no trained MoE checkpoint has been evaluated, the TL2 packing path is only synthetically validated, and the active TL2 runtime contract still does not route merged experts correctly. The new ternary fixture is a runtime contract, not a quality or product benchmark.

## Required Plan

Required MoE/Kimi path: validate the new Qwen2MoE BitNet converter mapping on a real checkpoint, add any Kimi-specific tensor mapping, decide which router and shared-expert tensors stay dense, extend TL2 packing plus full GGUF/runtime tests to 3D expert tensors, distill router and expert weights under ternary constraints, then run quality, throughput, RSS, and expert-locality benchmarks against dense and llama.cpp quantized MoE baselines.
