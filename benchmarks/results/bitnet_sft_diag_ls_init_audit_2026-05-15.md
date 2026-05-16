# BitNet-SFT Diag-LS Init Audit, 2026-05-15

Pending Slurm output.

## Summary

| field | value |
| --- | --- |
| status | pending |
| quality proven | false |
| baseline accuracy | 0.628935 |
| candidate accuracy | - |
| delta vs absmean baseline | - |
| candidate eval examples | - |
| paired matched examples | - |
| paired delta | - |
| paired CI95 | - |
| McNemar exact p | - |

## Artifacts

| artifact | path |
| --- | --- |
| baseline root | checkpoints/bitdistill-glue-seqcls-bitnet-sft-budget/Qwen-Qwen2.5-0.5B/mnli/bitnet_sft-tensor-steps10000-lr2em5 |
| candidate root | checkpoints/bitdistill-glue-seqcls-bitnet-sft-init/Qwen-Qwen2.5-0.5B/mnli/bitnet_sft-tensor-initdiag_ls-cal8-steps10000-lr2em5 |
| submission | benchmark_results/bitnet_sft_diag_ls_init_submission_2026-05-15.json |

## Blockers

- missing candidate metrics: checkpoints/bitdistill-glue-seqcls-bitnet-sft-init/Qwen-Qwen2.5-0.5B/mnli/bitnet_sft-tensor-initdiag_ls-cal8-steps10000-lr2em5/metrics.json
