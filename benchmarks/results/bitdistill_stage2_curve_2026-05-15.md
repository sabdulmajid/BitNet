# BitDistill Stage-2 Budget Curve Audit, 2026-05-15

This is not a controlled proof that Stage-2 tokens alone explain the gap. It does show that moving from the older short warm-up row to the completed 20k tensor warm-up family coincides with a large MNLI gain, while the best completed tensor BitDistill row remains far below FP16.

## Summary

| metric | value |
| --- | --- |
| paper Stage-2 warm-up tokens | 10000000000 |
| best tensor BitDistill MNLI | 0.641671 |
| best tensor delta vs FP16 | -0.166480 |
| best row-scale retrofit MNLI | 0.653591 |
| best row-scale delta vs FP16 | -0.154559 |
| diagnostic tensor short-to-long delta | 0.105960 |
| controlled token curve | false |

## Rows

| run | family | MNLI acc | delta vs FP16 | Stage-2 tokens | paper fraction | token source | scale | gamma | QKV reduction | weighted AD / CE | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FP16-SFT reference | reference | 0.808151 | 0.000000 | - | - | none | tensor | - |  | - | Dense FP16 task model; not a ternary student. |
| BitNet-SFT default | ce_only | 0.487621 | -0.320530 | - | - | none | tensor | - |  | - | Short CE-only baseline; undertrained relative to later budget rows. |
| BitNet-SFT best completed budget | ce_only | 0.628935 | -0.179215 | - | - | none | tensor | - |  | - | CE-only tensor-scale row that clears the paper BitNet-SFT anchor but remains far below FP16. |
| BitDistill tensor short warm-up | bitdistill_tensor | 0.535711 | -0.272440 | 40960000.000000 | 0.004096 | inferred_steps_x_8192 | tensor | 100.000000 | legacy_or_absent | 1.479569 | Uses the older 5k-step Stage-2 checkpoint; attention Q/K/V reduction predates the later explicit sum setting. |
| BitDistill tensor 20k warm-up | bitdistill_tensor | 0.641671 | -0.166480 | 163840000.000000 | 0.016384 | recorded | tensor | 100.000000 | sum | 6.740602 | Uses 20k-step tensor Stage-2 warm-up and explicit Q/K/V-sum attention relation loss. |
| BitDistill tensor 20k paper-gamma | bitdistill_tensor | 0.630260 | -0.177891 | 163840000.000000 | 0.016384 | recorded | tensor | 100000.000000 | sum | 4318.596751 | Same tensor warm-up family with gamma=100000 under the local normalization. |
| BitDistill row downstream, tensor warm-up | retrofit_variant | 0.653591 | -0.154559 | 163840000.000000 | 0.016384 | recorded | row | 100.000000 | sum | 8.201130 | Row-scale downstream retrofit variant loaded from the tensor Stage-2 checkpoint. |
| BitDistill row downstream, row warm-up | retrofit_variant | 0.627713 | -0.180438 | 163840000.000000 | 0.016384 | recorded | row | 100.000000 | sum | 8.372711 | Row-scale downstream variant loaded from the row-scale Stage-2 checkpoint. |
| Controlled recovery, 5k warm-up | controlled_curve | 0.616607 | -0.191544 | 40960000.000000 | 0.004096 | inferred_steps_x_8192 | tensor | - |  | - | Queued fixed-recipe control: tensor scale, layer -1, gamma 100000, head init, 10000 downstream steps. |
| Controlled recovery, 20k warm-up | controlled_curve | 0.691187 | -0.116964 | 163840000.000000 | 0.016384 | recorded | tensor | - |  | - | Queued fixed-recipe control: tensor scale, layer -1, gamma 100000, head init, 10000 downstream steps. |
| Controlled recovery, 40k warm-up | controlled_curve | - | - | 327680000.000000 | 0.032768 | expected_from_submission | - | - |  | - | Queued fixed-recipe control: tensor scale, layer -1, gamma 100000, head init, 10000 downstream steps. |

## Confounders

- The short and 20k tensor warm-up rows changed attention Q/K/V reduction reporting/semantics.
- The row-scale rows are retrofit variants and are not paper-style tensor-scale BitDistill reproduction rows.
- The largest completed local Stage-2 budget is 163.84M token presentations, only 1.6384% of the paper's 10B-token warm-up.
- Downstream Stage-3 budgets also differ from the paper's epoch/LR search.

## Interpretation

The correct next experiment is a fixed-recipe Stage-2 budget curve rather than more broad ablations: keep Qwen2.5-0.5B, MNLI, tensor-scale sequence classification, SubLN policy, dense head policy, attention layer, loss normalization, LR, and downstream steps fixed while varying only the continued-pretraining token budget.

