# Sequence-Classification I2_SR Hidden-Contract Audit, 2026-05-15

This audit tests whether the current sidecar classifier failure is caused by tokenization or by a deeper runtime-contract mismatch between the PyTorch sequence-classification checkpoint and the packed llama.cpp backbone.

| field | value |
| --- | --- |
| status | hidden_contract_mismatch |
| label | 1 |
| token IDs match | true |
| token count | 13 |
| PyTorch prediction | 1 |
| llama sidecar prediction | 0 |
| hidden relative RMS | 7.834796 |
| hidden cosine | 0.029207 |
| PyTorch hidden norm | 26.537035 |
| llama hidden norm | 206.988292 |
| logit relative RMS | 3.864149 |

## Prompt

`The new rights are nice enoughEveryone really likes the newest benefits `

## Logits

| source | logits |
| --- | --- |
| PyTorch model | [-0.51953125, 1.4375, 0.287109375] |
| llama embedding + sidecar head | [2.182292938232422, -3.929462432861328, 0.17850446701049805] |
| PyTorch hidden + sidecar head | [-0.5188282132148743, 1.441055178642273, 0.2875688076019287] |

## Interpretation

Token IDs matching rules out the tokenizer pair-format path for this sample. The large hidden-state relative RMS and near-zero cosine show that the packed decoder embedding does not currently reproduce the PyTorch sequence-classification pooled hidden state, so the sidecar is not a deployable classifier yet.
