# Sequence-Classification Native I2_SR CPU Benchmark, 2026-05-15

This benchmark evaluates one native GGUF artifact that contains the packed I2_SR backbone and dense classifier head. It is the same-artifact runtime path, but it is not product-ready unless full validation, batching parity, RSS, and throughput gates pass.

| field | value |
| --- | --- |
| status | pass |
| task | mnli |
| examples | 9815 |
| expected examples | 9815 |
| full validation | true |
| accuracy | 0.652165 |
| accuracy CI95 | [0.642685, 0.661526] |
| stored PyTorch accuracy | 0.653591 |
| agreement with saved PyTorch predictions | 0.976668 |
| label agreement with saved trace | 1.000000 |
| prompt input | token_ids |
| prompt batch size | 1 |
| batching parity ready | false |
| llama batch size | 4096 |
| ubatch size | 512 |
| wall seconds | 3602.972219 |
| examples/sec | 2.724140 |
| tokens/sec | 1092.329829 |
| child peak RSS MiB | 1021.296875 |
| ready to productize | false |

## Interpretation

Native same-artifact classifier execution is measurable, but the product gate remains blocked until full validation, batching parity, RSS, and throughput are audited. Batching parity is a hard gate because the current multi-prompt native classifier path changes low-margin logits.
