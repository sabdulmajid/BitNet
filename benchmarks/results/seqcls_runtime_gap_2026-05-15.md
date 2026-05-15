# Sequence-Classification Runtime Gap Audit, 2026-05-15

This audit separates the best GLUE quality path from the packed CPU runtime path.

| field | value |
| --- | --- |
| status | sidecar_qwen_contract_available_native_head_blocked |
| same artifact quality+CPU ready | false |
| sidecar prototype smoke | prototype_smoke_passed |
| sidecar sampled CPU quality | quality_mismatch |
| sidecar hidden contract | hidden_contract_mismatch |
| sidecar architecture contract | bitnet_qwen_contract_available |
| seqcls configs | 15 |
| seqcls causal-export compatible | 0 |
| causal runtime configs | 6 |
| causal GGUF exports | 6 |
| llama.cpp fork remote | https://github.com/sabdulmajid/llama.cpp.git |
| llama.cpp worktree clean | false |

## Sequence-Classification Quality Path

| architecture | count |
| --- | --- |
| Qwen2ForSequenceClassification | 15 |

These checkpoints are the strict GLUE reproduction artifacts. They use `Qwen2ForSequenceClassification`. The standard causal export path still does not implement a native sequence-classification head, but the sidecar smoke below shows that a Qwen-compatible packed decoder backbone plus dense score-head sidecar is now loadable.

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
| sampled accuracy | 0.578125 |
| agreement with saved PyTorch predictions | 0.921875 |
| sampled examples/sec | 0.714970 |
| token IDs match | true |
| hidden relative RMS | 0.108662 |
| hidden cosine | 0.994091 |
| logit relative RMS | 0.091918 |
| checkpoint hidden_act | silu |
| bitnet-25 FFN activation | relu_sqr |
| bitnet-qwen contract available | true |
| bitnet-qwen FFN activation | silu |
| bitnet-qwen has Q/K/V bias slots | true |
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
| Sampled sidecar CPU quality agreement | prototype improved; needs full validation |
| Tokenizer pair formatting parity | passes for audited MNLI sample |
| PyTorch pooled hidden state matches llama.cpp embedding | near pass on audited sample, not exact |
| Packed graph matches Qwen2 SiLU/SwiGLU semantics | implemented via bitnet-qwen |
| Packed loader supports Qwen2 Q/K/V projection biases | implemented via bitnet-qwen |
| GGUF writer persists classifier/score head tensors and label metadata | not implemented |
| llama.cpp pools the last non-padding token for Qwen sequence classification | not implemented |
| CPU evaluator reports GLUE accuracy from the packed classifier artifact | not implemented |
| Quality, RSS, and throughput measured on the same deployed artifact | blocked |

## Interpretation

The current repository has a PyTorch quality proof path and a causal GGUF runtime proof path. It now also has a prototype sequence-classification backbone path through `bitnet-qwen` I2_SR plus an external dense head sidecar. The new graph fixes the dominant architecture mismatch: the packed hidden vector now has high cosine agreement with PyTorch on the audited MNLI sample, and the 64-sample sidecar probe mostly agrees with saved PyTorch predictions. This is still not a deployable classifier: the classifier head is not native GGUF metadata/runtime code, the hidden contract is not bit-exact, and full-split CPU quality/RSS/throughput have not been measured on a single native artifact.
