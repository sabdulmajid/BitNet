# Sequence-Classification Native I2_SR CPU Benchmark, 2026-05-15

This benchmark evaluates one native GGUF artifact that contains the packed I2_SR backbone and dense classifier head. It is the same-artifact runtime path, but it is not product-ready unless full validation, batching parity, RSS, and throughput gates pass.

| field | value |
| --- | --- |
| status | quality_mismatch |
| task | mnli |
| examples | 16 |
| expected examples | 9815 |
| full validation | false |
| accuracy | 0.625000 |
| accuracy CI95 | [0.386410, 0.815188] |
| stored PyTorch accuracy | 0.653591 |
| agreement with saved PyTorch predictions | 0.937500 |
| label agreement with saved trace | 1.000000 |
| prompt input | token_ids |
| prompt batch size | 1 |
| llama batch size | 4096 |
| ubatch size | 512 |
| wall seconds | 22.529323 |
| examples/sec | 0.710186 |
| tokens/sec | 274.552784 |
| child peak RSS MiB | 950.402344 |
| ready to productize | false |

## Interpretation

Native same-artifact classifier execution is measurable, but the product gate remains blocked until full validation, batching parity, RSS, and throughput are audited.
