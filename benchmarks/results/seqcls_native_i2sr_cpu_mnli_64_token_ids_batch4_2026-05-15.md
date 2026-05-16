# Sequence-Classification Native I2_SR CPU Benchmark, 2026-05-15

This benchmark evaluates one native GGUF artifact that contains the packed I2_SR backbone and dense classifier head. It is the same-artifact runtime path, but it is not product-ready unless full validation, batching parity, RSS, and throughput gates pass.

| field | value |
| --- | --- |
| status | sample_only |
| task | mnli |
| examples | 64 |
| expected examples | 9815 |
| full validation | false |
| accuracy | 0.609375 |
| accuracy CI95 | [0.486919, 0.719444] |
| stored PyTorch accuracy | 0.653591 |
| agreement with saved PyTorch predictions | 1.000000 |
| label agreement with saved trace | 1.000000 |
| prompt input | token_ids |
| prompt batch size | 4 |
| llama batch size | 4096 |
| ubatch size | 512 |
| wall seconds | 24.247920 |
| examples/sec | 2.639402 |
| tokens/sec | 575.984088 |
| child peak RSS MiB | 951.421875 |
| ready to productize | false |

## Interpretation

Native same-artifact classifier execution is measurable, but the product gate remains blocked until full validation, batching parity, RSS, and throughput are audited.
