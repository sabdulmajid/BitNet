# Sequence-Classification Runtime Gap Audit, 2026-05-15

This audit separates the best GLUE quality path from the packed CPU runtime path.

| field | value |
| --- | --- |
| status | sidecar_prototype_available_native_runtime_blocked |
| same artifact quality+CPU ready | false |
| sidecar prototype smoke | prototype_smoke_passed |
| sidecar sampled CPU quality | quality_mismatch |
| sidecar hidden contract | hidden_contract_mismatch |
| sidecar architecture contract | architecture_contract_mismatch |
| seqcls configs | 15 |
| seqcls causal-export compatible | 0 |
| causal runtime configs | 6 |
| causal GGUF exports | 6 |
| llama.cpp fork remote | https://github.com/sabdulmajid/llama.cpp.git |
| llama.cpp worktree clean | true |

## Sequence-Classification Quality Path

| architecture | count |
| --- | --- |
| Qwen2ForSequenceClassification | 15 |

These checkpoints are the strict GLUE reproduction artifacts. They use `Qwen2ForSequenceClassification`. The standard causal export path still does not implement a native sequence-classification head, but the sidecar smoke below shows that a packed decoder backbone plus dense score-head sidecar is now loadable.

## Sidecar Prototype

| field | value |
| --- | --- |
| status | prototype_smoke_passed |
| checkpoint accuracy | 0.653591 |
| checkpoint eval examples | 9815.000000 |
| GGUF MiB | 352.606750 |
| head bytes | 11030 |
| runtime return code | 0 |
| embedding shape | [896] |
| head shape | [3, 896] |
| finite logits | true |
| sampled CPU status | quality_mismatch |
| sampled examples | 64 |
| sampled accuracy | 0.359375 |
| agreement with saved PyTorch predictions | 0.343750 |
| sampled examples/sec | 0.706330 |
| token IDs match | true |
| hidden relative RMS | 7.812774 |
| hidden cosine | 0.012303 |
| logit relative RMS | 7.270589 |
| checkpoint hidden_act | silu |
| bitnet-25 FFN activation | relu_sqr |
| Q/K/V projection bias tensors | 72 |

## Causal Runtime Path

| architecture | count |
| --- | --- |
| Qwen2ForCausalLM | 6 |

These checkpoints are export-compatible with the current GGUF/I2_SR path, but they are not the same artifacts as the sequence-classification quality branch.

## Required Runtime Work

| item | status |
| --- | --- |
| Backbone GGUF + dense head sidecar smoke | prototype implemented |
| Sampled sidecar CPU quality agreement | failing |
| Tokenizer pair formatting parity | passes for audited MNLI sample |
| PyTorch pooled hidden state matches llama.cpp embedding | failing |
| Packed graph matches Qwen2 SiLU/SwiGLU semantics | failing |
| Packed loader supports Qwen2 Q/K/V projection biases | partial: bitnet-25 yes, plain bitnet no |
| GGUF writer persists classifier/score head tensors and label metadata | not implemented |
| llama.cpp pools the last non-padding token for Qwen sequence classification | not implemented |
| CPU evaluator reports GLUE accuracy from the packed classifier artifact | not implemented |
| Quality, RSS, and throughput measured on the same deployed artifact | blocked |

## Interpretation

The current repository has a PyTorch quality proof path and a causal GGUF runtime proof path. It now also has a prototype sequence-classification backbone smoke through I2_SR plus an external dense head sidecar. The sampled sidecar CPU quality probe currently disagrees with saved PyTorch predictions. The hidden-contract audit narrows the issue: token IDs match for the first MNLI sample, but the llama.cpp embedding has high relative RMS error and near-zero cosine versus the PyTorch pooled hidden state. The architecture-contract audit gives a concrete root cause to fix: the current bitnet-25 graph uses ReLU-squared FFN while the Qwen2 checkpoint uses SiLU/SwiGLU; the plain bitnet graph has SiLU but lacks the 72 Q/K/V bias tensor slots present in the checkpoint. This is a runtime/model-state mismatch, not a deployable classifier.
