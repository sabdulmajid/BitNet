# Sequence-Classification Native I2_SR Mismatch Audit, 2026-05-15

This audit checks the rows that blocked the native same-artifact classifier product gate. It compares the PyTorch sequence-classification checkpoint, the sidecar GGUF backbone plus dense head, and the native classifier-head GGUF on exactly the same MNLI prompts.

| field | value |
| --- | --- |
| status | runtime_hidden_drift |
| indices | [0, 7, 15] |
| token IDs all match | true |
| text roundtrip all token IDs match | false |
| prompt input | token_ids |
| native/sidecar agreement | 1.000000 |
| saved/native agreement | 0.666667 |
| hidden relative RMS max | 0.108662 |
| hidden cosine min | 0.994091 |
| native-vs-sidecar logit relative RMS max | 0.000000 |

## Row Comparisons

| idx | label | saved | torch | sidecar | native | tokens | hidden rel RMS | hidden cos | torch/sidecar logit rel | native/sidecar logit rel | native margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 1 | 1 | 1 | 1 | true | 0.108662 | 0.994091 | 0.091918 | 0.000000 | 1.345433 |
| 7 | 1 | 1 | 1 | 1 | 1 | true | 0.107469 | 0.994308 | 0.221061 | 0.000000 | 0.820764 |
| 15 | 0 | 2 | 2 | 1 | 1 | true | 0.094616 | 0.995552 | 0.116697 | 0.000000 | 0.019177 |

## Interpretation

Token IDs match on the faithful runtime path and native GGUF logits match the sidecar GGUF+head path for these rows, so the native classifier-head plumbing is not the source of the disagreement. The blocker is packed-runtime hidden/logit drift relative to the PyTorch BitLinear checkpoint. The audit also records whether the older text decode/re-tokenize path is lossless; when it is not, direct token IDs are required for sequence-pair evaluation.
