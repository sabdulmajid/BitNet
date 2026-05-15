# Sequence-Classification Runtime Gap Audit, 2026-05-15

This audit separates the best GLUE quality path from the packed CPU runtime path.

| field | value |
| --- | --- |
| status | sidecar_prototype_available_native_runtime_blocked |
| same artifact quality+CPU ready | false |
| sidecar prototype smoke | prototype_smoke_passed |
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
| GGUF MiB | 352.606689 |
| head bytes | 11030 |
| runtime return code | 0 |
| embedding shape | [896] |
| head shape | [3, 896] |
| finite logits | true |

## Causal Runtime Path

| architecture | count |
| --- | --- |
| Qwen2ForCausalLM | 6 |

These checkpoints are export-compatible with the current GGUF/I2_SR path, but they are not the same artifacts as the sequence-classification quality branch.

## Required Runtime Work

| item | status |
| --- | --- |
| Backbone GGUF + dense head sidecar smoke | prototype implemented |
| GGUF writer persists classifier/score head tensors and label metadata | not implemented |
| llama.cpp pools the last non-padding token for Qwen sequence classification | not implemented |
| CPU evaluator reports GLUE accuracy from the packed classifier artifact | not implemented |
| Quality, RSS, and throughput measured on the same deployed artifact | blocked |

## Interpretation

The current repository has a PyTorch quality proof path and a causal GGUF runtime proof path. It now also has a prototype sequence-classification backbone smoke through I2_SR plus an external dense head sidecar. It still does not have native packed sequence-classification inference or full GLUE CPU accuracy/RSS/throughput on that deployed artifact.
