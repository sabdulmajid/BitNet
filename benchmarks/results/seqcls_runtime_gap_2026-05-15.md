# Sequence-Classification Runtime Gap Audit, 2026-05-15

This audit separates the best GLUE quality path from the packed CPU runtime path.

| field | value |
| --- | --- |
| status | blocked_by_classifier_runtime |
| same artifact quality+CPU ready | false |
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

These checkpoints are the strict GLUE reproduction artifacts. They use `Qwen2ForSequenceClassification`; current packed I2_SR export rejects them because the runtime path is causal-LM only.

## Causal Runtime Path

| architecture | count |
| --- | --- |
| Qwen2ForCausalLM | 6 |

These checkpoints are export-compatible with the current GGUF/I2_SR path, but they are not the same artifacts as the sequence-classification quality branch.

## Required Runtime Work

| item | status |
| --- | --- |
| GGUF writer persists classifier/score head tensors and label metadata | not implemented |
| llama.cpp pools the last non-padding token for Qwen sequence classification | not implemented |
| CPU evaluator reports GLUE accuracy from the packed classifier artifact | not implemented |
| Quality, RSS, and throughput measured on the same deployed artifact | blocked |

## Interpretation

The current repository has a PyTorch quality proof path and a causal GGUF runtime proof path. It does not yet have one task model that simultaneously proves GLUE quality and packed CPU deployment. The product path must either implement packed sequence-classification inference or make causal prompt scoring the primary task formulation.
