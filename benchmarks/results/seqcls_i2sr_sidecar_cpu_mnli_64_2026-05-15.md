# Sequence-Classification I2_SR Sidecar CPU Benchmark, 2026-05-15

This benchmark evaluates the packed I2_SR decoder backbone with the dense classifier head applied outside llama.cpp. It is a sidecar prototype, not native GGUF classifier support.

| field | value |
| --- | --- |
| status | quality_mismatch |
| task | mnli |
| examples | 64 |
| accuracy | 0.578125 |
| accuracy CI95 | [0.456100315400829, 0.6913021752565451] |
| stored PyTorch accuracy | 0.653591 |
| agreement with saved PyTorch predictions | 0.921875 |
| label agreement with saved trace | 1.000000 |
| batch size | 1 |
| wall seconds | 89.514234 |
| examples/sec | 0.714970 |
| tokens/sec aggregate | 268.808150 |

## Interpretation

If agreement with saved PyTorch predictions is low, the sidecar path is only a load/runtime prototype and still needs tokenization, pooling, or runtime-head alignment before it can be used as a deployed classifier.
