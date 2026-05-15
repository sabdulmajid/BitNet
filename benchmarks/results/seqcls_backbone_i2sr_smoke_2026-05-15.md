# Sequence-Classification I2_SR Backbone Smoke, 2026-05-15

This is a prototype bridge for the current product gap: a strict GLUE sequence-classification checkpoint is exported as a packed I2_SR decoder backbone, while the dense classifier head is kept as an NPZ sidecar.

| field | value |
| --- | --- |
| status | prototype_smoke_passed |
| checkpoint accuracy | 0.653591 |
| checkpoint eval examples | 9815.000000 |
| GGUF MiB | 352.606750 |
| head sidecar KiB | 10.771484 |
| I2_SR tensors | 168 |
| copied dense tensors | 170 |
| runtime return code | 0 |
| embedding dim | 896 |
| head shape | [3, 896] |
| finite embedding | true |
| finite logits | true |
| prompt tok/s | 139.020000 |
| predicted class for smoke prompt | 0 |

## Smoke Prompt

`Premise: A person is riding a bicycle. Hypothesis: Someone is outdoors.`

## Logits

| class | logit | probability |
| --- | --- | --- |
| 0 | 3.117530 | 0.983521 |
| 1 | -3.164887 | 0.001838 |
| 2 | -1.089805 | 0.014641 |

## Interpretation

This smoke proves loadability, last-token embedding extraction, sidecar-head shape compatibility, and finite CPU logits for the same sequence-classification checkpoint. It is not a full GLUE CPU accuracy result and it is not native llama.cpp sequence-classification support yet.

## Remaining Runtime Work

| item | status |
| --- | --- |
| Persist classifier head and label metadata inside GGUF | not implemented |
| Apply Qwen sequence-classification pooling/head inside llama.cpp | not implemented |
| Run full MNLI/QNLI/SST2 accuracy from packed CPU artifact | not implemented |
| Benchmark RSS and throughput for full task evaluation | not implemented |
