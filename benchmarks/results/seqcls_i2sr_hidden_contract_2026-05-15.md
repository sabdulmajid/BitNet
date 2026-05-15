# Sequence-Classification I2_SR Hidden-Contract Audit, 2026-05-15

This audit tests whether the current sidecar classifier failure is caused by tokenization or by a deeper runtime-contract mismatch between the PyTorch sequence-classification checkpoint and the packed llama.cpp backbone.

| field | value |
| --- | --- |
| status | hidden_contract_mismatch |
| label | 1 |
| token IDs match | true |
| token count | 13 |
| PyTorch prediction | 1 |
| llama sidecar prediction | 1 |
| hidden relative RMS | 0.108662 |
| hidden cosine | 0.994091 |
| PyTorch hidden norm | 26.537035 |
| llama hidden norm | 26.508326 |
| logit relative RMS | 0.091918 |

## Prompt

`The new rights are nice enoughEveryone really likes the newest benefits `

## Logits

| source | logits |
| --- | --- |
| PyTorch model | [-0.51953125, 1.4375, 0.287109375] |
| llama embedding + sidecar head | [-0.5463043451309204, 1.516332983970642, 0.17090028524398804] |
| PyTorch hidden + sidecar head | [-0.5188282132148743, 1.441055178642273, 0.2875688076019287] |

## Interpretation

Token IDs match and the packed decoder now follows the PyTorch pooled hidden state closely (cosine 0.994091), but the relative RMS error remains above the strict pass threshold (0.108662). Treat this as a repaired runtime contract that still needs full-split validation and native classifier-head support, not a final deployable classifier.
