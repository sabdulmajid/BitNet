# Conversion Support Audit, 2026-05-05

| component | path | Qwen2 | Qwen2 MoE | TL2 out/type | I2_S out/type | help ok | advertised output/types |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BitNet HF converter | utils/convert-hf-to-gguf-bitnet.py | no | no | yes | no | yes | f16, f32, tl1, tl2 |
| llama.cpp HF converter | 3rdparty/llama.cpp/convert_hf_to_gguf.py | yes | yes | no | no | yes | auto, bf16, f16, f32, q8_0, tq1_0, tq2_0 |
| llama-quantize | build-portable-avx2/bin/llama-quantize | n/a | n/a | no | yes | yes | BF16, COPY, F16, F32, I2_S, IQ1_M, IQ1_S, IQ2_M, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_M, IQ3_S, IQ3_XS, IQ3_XXS, IQ4_NL, IQ4_XS, Q2_K, Q2_K_S, Q3_K, Q3_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_0_4_4, Q4_0_4_8, Q4_0_8_8, Q4_1, Q4_K, Q4_K_M, Q4_K_S, Q5_0, Q5_1, Q5_K, Q5_K_M, Q5_K_S, Q6_K, Q8_0, TQ1_0, TQ2_0 |

## Verdict

The current toolchain does not provide a single direct path that is both Qwen2-aware and TL2-capable. The BitNet HF converter exposes `tl2`, but does not register `Qwen2ForCausalLM` or `Qwen2MoeForCausalLM`. The vendored llama.cpp converter registers those Qwen architectures, but its `--outtype` choices stop at `tq2_0` and do not include `tl2` or `i2_s`. `llama-quantize` can produce `I2_S`, but it operates from an existing GGUF and is not a direct `ternary_state_dict.pt` writer.
