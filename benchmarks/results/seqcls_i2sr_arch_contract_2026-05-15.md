# Sequence-Classification I2_SR Architecture-Contract Audit, 2026-05-15

This audit explains why the current packed sidecar classifier cannot yet be treated as a faithful deployment artifact. It checks the PyTorch checkpoint architecture against the llama.cpp graph selected by the GGUF export.

| field | value |
| --- | --- |
| status | architecture_contract_mismatch |
| checkpoint hidden_act | silu |
| GGUF runtime arch under audit | bitnet-25 |
| bitnet-25 graph activation | relu_sqr |
| PyTorch/runtime activation mismatch | true |
| Q/K/V projection bias tensors in checkpoint | 72 |
| plain bitnet loader has Q/K/V bias slots | false |
| bitnet-25 loader has Q/K/V bias slots | true |
| hidden relative RMS | 7.812774 |
| hidden cosine | 0.012303 |

## Source Evidence

| source check | line |
| --- | --- |
| bitnet-25 dispatches to build_bitnet_158 | 16853 |
| build_bitnet_158 uses ReLU squared FFN | 15503 |
| build_bitnet uses SiLU FFN | 15355 |
| plain bitnet loader block starts | 8679 |
| bitnet-25 loader block starts | 8719 |

## Interpretation

The current seqcls export uses the `bitnet-25` runtime graph, but the checkpoint is a Qwen2 sequence-classification student whose config says `hidden_act = silu`. The `bitnet-25` graph uses the BitNet b1.58/2.5 ReLU-squared FFN path. That is a deterministic architecture mismatch. The alternative plain `bitnet` graph has the SiLU FFN path, but its loader does not declare Q/K/V projection-bias tensors, while this Qwen2 checkpoint has 72 such tensors. Therefore the fix is not just changing a metadata string; we need a Qwen-BitDistill runtime contract that preserves Qwen2 SwiGLU/SiLU, optional projection biases, SubLN, RoPE metadata, and I2_SR row-scale kernels together.

## Required Runtime Work

| requirement | status |
| --- | --- |
| Qwen2/Qwen3 SiLU/SwiGLU FFN in packed graph | missing in bitnet-25 graph |
| Q/K/V projection-bias tensor slots | present in bitnet-25 loader, missing in plain bitnet loader |
| SubLN before attention output and FFN down projections | present in bitnet-25 graph |
| RoPE theta from rope_parameters | fixed in converter |
| row-scale I2_SR kernels | present for dense matmuls |
| native sequence-classification score head | not implemented |
