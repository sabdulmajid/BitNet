# Sequence-Classification I2_SR Sidecar CPU Benchmark, 2026-05-15

This benchmark evaluates the packed I2_SR decoder backbone with the dense classifier head applied outside llama.cpp. It is a sidecar prototype, not native GGUF classifier support.

| field | value |
| --- | --- |
| status | quality_mismatch |
| task | mnli |
| examples | 64 |
| accuracy | 0.359375 |
| accuracy CI95 | [0.25288402291451745, 0.48179149390220904] |
| stored PyTorch accuracy | 0.653591 |
| agreement with saved PyTorch predictions | 0.343750 |
| label agreement with saved trace | 1.000000 |
| batch size | 1 |
| wall seconds | 90.609232 |
| examples/sec | 0.706330 |
| tokens/sec aggregate | 242.153265 |

## Interpretation

If agreement with saved PyTorch predictions is low, the sidecar path is only a load/runtime prototype and still needs tokenization, pooling, or runtime-head alignment before it can be used as a deployed classifier.
