# Sequence-Classification I2_SR Architecture-Contract Audit, 2026-05-15

This audit explains why the current packed sidecar classifier cannot yet be treated as a faithful deployment artifact. It checks the PyTorch checkpoint architecture against the llama.cpp graph selected by the GGUF export.

| field | value |
| --- | --- |
| status | bitnet_qwen_contract_available |
| checkpoint hidden_act | silu |
| GGUF runtime arch under audit | bitnet-25 |
| bitnet-25 graph activation | relu_sqr |
| bitnet-qwen contract available | true |
| bitnet-qwen graph activation | silu |
| bitnet-qwen loader has Q/K/V bias slots | true |
| PyTorch/runtime activation mismatch | true |
| Q/K/V projection bias tensors in checkpoint | 72 |
| plain bitnet loader has Q/K/V bias slots | false |
| bitnet-25 loader has Q/K/V bias slots | true |
| hidden relative RMS | 0.108662 |
| hidden cosine | 0.994091 |

## Source Evidence

| source check | line |
| --- | --- |
| bitnet-25 dispatches to build_bitnet_158 | 16894 |
| bitnet-qwen dispatches to build_bitnet_158(true) | 16898 |
| build_bitnet_158 uses ReLU squared FFN | 15544 |
| build_bitnet_158 has Qwen SiLU branch | 15544 |
| build_bitnet uses SiLU FFN | 15396 |
| plain bitnet loader block starts | 8719 |
| bitnet-25 loader block starts | 8759 |

## Interpretation

The original `bitnet-25` export was a deterministic architecture mismatch for this Qwen2 sequence-classification checkpoint because it selected the BitNet ReLU-squared FFN path. This fork now carries a dedicated `bitnet-qwen` runtime contract that keeps the BitNet 2.5 loader/SubLN/bias layout but dispatches the packed graph through Qwen SiLU/SwiGLU FFN semantics. The remaining blocker is no longer the dense backbone graph; it is native sequence-classification head/pooling support plus full-split quality validation.

## Required Runtime Work

| requirement | status |
| --- | --- |
| Qwen2/Qwen3 SiLU/SwiGLU FFN in packed graph | present via bitnet-qwen |
| Q/K/V projection-bias tensor slots | present in bitnet-qwen/bitnet-25 loader, missing in plain bitnet loader |
| SubLN before attention output and FFN down projections | present in bitnet-25 graph |
| RoPE theta from rope_parameters | fixed in converter |
| row-scale I2_SR kernels | present for dense matmuls |
| native sequence-classification score head | not implemented |
