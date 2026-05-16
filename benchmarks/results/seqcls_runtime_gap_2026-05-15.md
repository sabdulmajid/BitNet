# Sequence-Classification Runtime Gap Audit, 2026-05-15

This audit separates the best GLUE quality path from the packed CPU runtime path.

| field | value |
| --- | --- |
| status | native_classifier_full_validation_batching_blocked |
| same artifact quality+CPU ready | false |
| sidecar prototype smoke | prototype_smoke_passed |
| native GGUF classifier smoke | pass |
| native CPU quality | pass |
| native CPU benchmark path | /mnt/slurm_nfs/a6abdulm/projects/BitNet/benchmark_results/seqcls_native_i2sr_cpu_mnli_full_token_ids_2026-05-15.json |
| sidecar sampled CPU quality | quality_mismatch |
| sidecar hidden contract | hidden_contract_mismatch |
| sidecar architecture contract | bitnet_qwen_contract_available |
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

These checkpoints are the strict GLUE reproduction artifacts. They use `Qwen2ForSequenceClassification`. The standard causal export path is still not a full classifier evaluator, but the native smoke below shows that a Qwen-compatible packed GGUF can now carry the dense score head and emit classifier logits.

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
| sampled examples | 128 |
| sampled accuracy | 0.609375 |
| agreement with saved PyTorch predictions | 0.914062 |
| sampled examples/sec | 0.683671 |
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

## Native GGUF Classifier Smoke

| field | value |
| --- | --- |
| status | pass |
| single artifact | true |
| logit count | 3 |
| prediction | 2 |
| sidecar prediction | 2 |
| relative RMS logit delta | 0.000000 |
| prompt tok/s | 265.900000 |
| full validation complete | false |
| ready to productize | false |

## Native GGUF CPU Benchmark

| field | value |
| --- | --- |
| path | /mnt/slurm_nfs/a6abdulm/projects/BitNet/benchmark_results/seqcls_native_i2sr_cpu_mnli_full_token_ids_2026-05-15.json |
| status | pass |
| task | mnli |
| prompt input | token_ids |
| examples | 9815 |
| accuracy | 0.652165 |
| agreement with saved PyTorch predictions | 0.976668 |
| examples/sec | 2.724140 |
| child peak RSS MiB | 1021.296875 |
| full validation complete | true |
| batching parity ready | false |
| ready to productize | false |

## Native GGUF Batching Audit

| field | value |
| --- | --- |
| status | batching_parity_mismatch |
| all predictions invariant | false |
| changed cases | 4 |
| max relative RMS vs alone | 0.305153 |
| ready for batched product benchmark | false |

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
| Tokenizer pair formatting parity | requires direct token-ID input for Qwen pair prompts |
| PyTorch pooled hidden state matches llama.cpp embedding | near pass on audited sample, not exact |
| Packed graph matches Qwen2 SiLU/SwiGLU semantics | implemented via bitnet-qwen |
| Packed loader supports Qwen2 Q/K/V projection biases | implemented via bitnet-qwen |
| GGUF writer persists classifier/score head tensors and label metadata | single-prompt smoke implemented |
| llama.cpp pools and applies the Qwen sequence-classification head | single-prompt smoke implemented |
| CPU evaluator reports GLUE accuracy from the packed classifier artifact | full token-ID MNLI validation implemented |
| Batched embedding/classifier parity | blocked: audited rows change logits/predictions by batch position |
| Quality, RSS, and throughput measured on the same deployed artifact | full single-prompt validation measured; product still blocked by batching parity |

## Interpretation

The current repository now has full-split native CPU validation for one packed `bitnet-qwen` sequence-classification artifact. The run uses direct token IDs and reports MNLI accuracy `0.652165`, saved-PyTorch prediction agreement `0.976668`, `2.724140` examples/sec, and `1021.296875` MiB child peak RSS. This is useful runtime-fidelity evidence for the checkpoint, not a product-ready classifier: batching parity remains `batching_parity_mismatch`, and the checkpoint accuracy is still well below the FP16 task model.
