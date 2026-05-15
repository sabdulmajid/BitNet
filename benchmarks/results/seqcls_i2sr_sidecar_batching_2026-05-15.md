# Sequence-Classification I2_SR Sidecar Batching Audit, 2026-05-15

This audit checks whether `llama-embedding --embd-separator` batching preserves the sidecar classifier predictions.

| field | value |
| --- | --- |
| status | batching_semantics_drift |
| batch4 vs batch1 prediction agreement | 0.953125 |
| batch4 vs batch1 mismatches | 3 |
| first mismatch indices | [10, 15, 35] |
| examples/sec speedup | 3.576167 |
| prompt tokens/sec speedup | 2.069435 |

| row | batch | status | examples | accuracy | agreement vs PyTorch | examples/sec | prediction sha256 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| batch1 | 1 | quality_mismatch | 64 | 0.578125 | 0.921875 | 0.706617 | 7843e032050a |
| batch4 | 4 | quality_mismatch | 64 | 0.609375 | 0.937500 | 2.526979 | db386c70a4e8 |

## Interpretation

Separator batching improves throughput but changes at least one classifier prediction. Treat batch size 1 as the semantic reference until native sequence-classification runtime support is implemented or batching parity is proven.
