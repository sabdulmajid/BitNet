# Sequence-Classification Native I2_SR CPU Benchmark, 2026-05-15

This benchmark evaluates one native GGUF artifact that contains the packed I2_SR backbone and dense classifier head. It is the same-artifact runtime path, but it is not product-ready unless full validation, runtime parity, RSS, and throughput gates pass.

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
| prompt batch size | 512 |
| embedding sequential | true |
| batching parity ready | false |
| sequence-isolated parity ready | true |
| runtime parity ready | true |
| llama batch size | 4096 |
| ubatch size | 512 |
| wall seconds | 1316.353428 |
| examples/sec | 7.456204 |
| tokens/sec | 277.484939 |
| child peak RSS MiB | 960.152344 |
| ready to productize | false |

## Interpretation

Native same-artifact classifier execution is measurable, but the product gate remains blocked by: saved_pytorch_agreement=0.976668<0.99. Multi-prompt batched I2_SR remains blocked by position-dependent drift; sequence-isolated mode is a separate mitigation.
