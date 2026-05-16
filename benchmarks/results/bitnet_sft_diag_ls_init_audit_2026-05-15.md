# BitNet-SFT Diag-LS Init Audit, 2026-05-15

diag_ls initialization has a complete paired MNLI comparison but does not improve the matched absmean baseline.

## Summary

| field | value |
| --- | --- |
| status | complete |
| comparison valid | true |
| candidate improves absmean baseline | false |
| baseline accuracy | 0.628935 |
| candidate accuracy | 0.350993 |
| delta vs absmean baseline | -0.277942 |
| candidate eval examples | 9815.000000 |
| paired matched examples | 9815 |
| paired delta | -0.277942 |
| paired CI95 | [-0.290856, -0.265028] |
| McNemar exact p | 0.000000 |

## Artifacts

| artifact | path |
| --- | --- |
| baseline root | checkpoints/bitdistill-glue-seqcls-bitnet-sft-budget/Qwen-Qwen2.5-0.5B/mnli/bitnet_sft-tensor-steps10000-lr2em5 |
| candidate root | checkpoints/bitdistill-glue-seqcls-bitnet-sft-init/Qwen-Qwen2.5-0.5B/mnli/bitnet_sft-tensor-initdiag_ls-cal8-steps10000-lr2em5 |
| submission | benchmark_results/bitnet_sft_diag_ls_init_submission_2026-05-15.json |

## Blockers

None.
