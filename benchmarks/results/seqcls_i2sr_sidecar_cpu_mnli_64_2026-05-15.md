# Sequence-Classification I2_SR Sidecar CPU Benchmark, 2026-05-15

This benchmark evaluates the packed I2_SR decoder backbone with the dense classifier head applied outside llama.cpp. It is a sidecar prototype, not native GGUF classifier support.

| field | value |
| --- | --- |
| status | quality_mismatch |
| task | mnli |
| examples | 64 |
| accuracy | 0.343750 |
| accuracy CI95 | [0.23923142078443355, 0.46596359790081815] |
| stored PyTorch accuracy | 0.653591 |
| agreement with saved PyTorch predictions | 0.296875 |
| label agreement with saved trace | 1.000000 |
| batch size | 1 |
| wall seconds | 90.783757 |
| examples/sec | 0.704972 |
| tokens/sec aggregate | 235.643285 |

## Interpretation

If agreement with saved PyTorch predictions is low, the sidecar path is only a load/runtime prototype and still needs tokenization, pooling, or runtime-head alignment before it can be used as a deployed classifier.
