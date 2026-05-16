# Sequence-Classification Native I2_SR Smoke, 2026-05-15

This is a single-prompt native GGUF smoke test. It proves that the packed BitNet-Qwen artifact can carry the dense classifier head and emit finite classifier logits without an NPZ sidecar. It is not a full GLUE quality benchmark.

| field | value |
| --- | --- |
| status | pass |
| single artifact | true |
| logit count | 3 |
| prediction | 2 |
| sidecar prediction | 2 |
| max abs logit delta | 0.000000 |
| relative RMS logit delta | 0.000000 |
| prompt tok/s | 128.050000 |
| full validation complete | false |
| ready to productize | false |

## Verdict

Native single-artifact classifier-head execution works for this smoke prompt, but the product gate remains blocked until full MNLI validation, batching parity, RSS, and throughput are measured from this same GGUF.
