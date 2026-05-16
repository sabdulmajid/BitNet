# Qwen3 Paper-Alignment Audit, 2026-05-15

Overall status: **pending**.

This audit tracks the queued Qwen3-0.6B branch. Rows remain pending until full validation metrics and prediction traces exist.

## Headline

| field | value |
| --- | --- |
| jobs | 16 |
| complete rows | 15 |
| FP complete tasks | [mnli, qnli, sst2] |
| BitNet-SFT complete tasks | [mnli, qnli, sst2] |
| tensor BitDistill complete tasks | [mnli, qnli, sst2] |
| row BitDistill complete tasks | [mnli, qnli, sst2] |
| gap-pass rows | 0 |
| paper reproduction ready | false |

## Row-Scale Versus Tensor-Scale BitDistill

| task | tensor accuracy | row accuracy | row minus tensor | CI95 | matched |
| --- | --- | --- | --- | --- | --- |
| mnli | 0.723484 | 0.696179 | -0.027305 | [-0.034073, -0.020537] | 9815 |
| qnli | 0.861065 | 0.848435 | -0.012630 | [-0.019826, -0.005435] | 5463 |
| sst2 | 0.871560 | 0.877294 | 0.005734 | [-0.011536, 0.023004] | 872 |

## Attention-Layer Sweep

| task | baseline layer | candidate layer | baseline accuracy | candidate accuracy | candidate minus baseline | CI95 | matched |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mnli | -1 | -2 | 0.723484 | 0.717779 | -0.005706 | [-0.011847, 0.000436] | 9815 |
| mnli | -1 | -4 | 0.723484 | 0.733367 | 0.009883 | [0.003959, 0.015807] | 9815 |
| mnli | -1 | -8 | 0.723484 | 0.752012 | 0.028528 | [0.022008, 0.035048] | 9815 |

## Rows

| job | phase | task | method | scale | label | layer | queue | complete | accuracy | delta vs FP | CI95 | gap pass | output |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10040 | stage2 | - | bitdistill | tensor | paper-inspired | - | not_in_squeue | false | - | - | - | false | checkpoints/bitdistill-glue-qwen3base-seqcls-papergamma/Qwen-Qwen3-0.6B-Base/continued_pretrain/bitdistill-tensor |
| 10041 | paper_baseline | mnli | fp16_sft | tensor | paper-inspired | -1 | not_in_squeue | true | 0.829750 | - | - | false | checkpoints/bitdistill-glue-qwen3base-seqcls-papergamma/Qwen-Qwen3-0.6B-Base/mnli/fp16_sft-tensor-layer-1 |
| 10042 | paper_baseline | mnli | bitnet_sft | tensor | paper-inspired | -1 | not_in_squeue | true | 0.477127 | -0.352624 | [-0.364186, -0.341061] | false | checkpoints/bitdistill-glue-qwen3base-seqcls-papergamma/Qwen-Qwen3-0.6B-Base/mnli/bitnet_sft-tensor-layer-1 |
| 10043 | paper_baseline | mnli | bitdistill | tensor | paper-inspired | -1 | not_in_squeue | true | 0.723484 | -0.106266 | [-0.115044, -0.097488] | false | checkpoints/bitdistill-glue-qwen3base-seqcls-papergamma/Qwen-Qwen3-0.6B-Base/mnli/bitdistill-tensor-layer-1 |
| 10044 | novelty_row_scale | mnli | bitdistill | row | retrofit-variant | -1 | not_in_squeue | true | 0.696179 | -0.133571 | [-0.142757, -0.124385] | false | checkpoints/bitdistill-glue-qwen3base-seqcls-papergamma/Qwen-Qwen3-0.6B-Base/mnli/bitdistill-row-layer-1 |
| 10045 | paper_baseline | qnli | fp16_sft | tensor | paper-inspired | -1 | not_in_squeue | true | 0.921106 | - | - | false | checkpoints/bitdistill-glue-qwen3base-seqcls-papergamma/Qwen-Qwen3-0.6B-Base/qnli/fp16_sft-tensor-layer-1 |
| 10046 | paper_baseline | qnli | bitnet_sft | tensor | paper-inspired | -1 | not_in_squeue | true | 0.587040 | -0.334066 | [-0.348717, -0.319415] | false | checkpoints/bitdistill-glue-qwen3base-seqcls-papergamma/Qwen-Qwen3-0.6B-Base/qnli/bitnet_sft-tensor-layer-1 |
| 10047 | paper_baseline | qnli | bitdistill | tensor | paper-inspired | -1 | not_in_squeue | true | 0.861065 | -0.060040 | [-0.068684, -0.051397] | false | checkpoints/bitdistill-glue-qwen3base-seqcls-papergamma/Qwen-Qwen3-0.6B-Base/qnli/bitdistill-tensor-layer-1 |
| 10048 | novelty_row_scale | qnli | bitdistill | row | retrofit-variant | -1 | not_in_squeue | true | 0.848435 | -0.072671 | [-0.081763, -0.063578] | false | checkpoints/bitdistill-glue-qwen3base-seqcls-papergamma/Qwen-Qwen3-0.6B-Base/qnli/bitdistill-row-layer-1 |
| 10049 | paper_baseline | sst2 | fp16_sft | tensor | paper-inspired | -1 | not_in_squeue | true | 0.930046 | - | - | false | checkpoints/bitdistill-glue-qwen3base-seqcls-papergamma/Qwen-Qwen3-0.6B-Base/sst2/fp16_sft-tensor-layer-1 |
| 10050 | paper_baseline | sst2 | bitnet_sft | tensor | paper-inspired | -1 | not_in_squeue | true | 0.799312 | -0.130734 | [-0.159101, -0.102367] | false | checkpoints/bitdistill-glue-qwen3base-seqcls-papergamma/Qwen-Qwen3-0.6B-Base/sst2/bitnet_sft-tensor-layer-1 |
| 10051 | paper_baseline | sst2 | bitdistill | tensor | paper-inspired | -1 | not_in_squeue | true | 0.871560 | -0.058486 | [-0.080523, -0.036449] | false | checkpoints/bitdistill-glue-qwen3base-seqcls-papergamma/Qwen-Qwen3-0.6B-Base/sst2/bitdistill-tensor-layer-1 |
| 10052 | novelty_row_scale | sst2 | bitdistill | row | retrofit-variant | -1 | not_in_squeue | true | 0.877294 | -0.052752 | [-0.072814, -0.032691] | false | checkpoints/bitdistill-glue-qwen3base-seqcls-papergamma/Qwen-Qwen3-0.6B-Base/sst2/bitdistill-row-layer-1 |
| 10053 | attention_layer_sweep | mnli | bitdistill | tensor | paper-inspired | -8 | not_in_squeue | true | 0.752012 | -0.077738 | [-0.085957, -0.069520] | false | checkpoints/bitdistill-glue-qwen3base-seqcls-papergamma/Qwen-Qwen3-0.6B-Base/mnli/bitdistill-tensor-layer-8 |
| 10054 | attention_layer_sweep | mnli | bitdistill | tensor | paper-inspired | -2 | not_in_squeue | true | 0.717779 | -0.111971 | [-0.120781, -0.103162] | false | checkpoints/bitdistill-glue-qwen3base-seqcls-papergamma/Qwen-Qwen3-0.6B-Base/mnli/bitdistill-tensor-layer-2 |
| 10055 | attention_layer_sweep | mnli | bitdistill | tensor | paper-inspired | -4 | not_in_squeue | true | 0.733367 | -0.096383 | [-0.104928, -0.087838] | false | checkpoints/bitdistill-glue-qwen3base-seqcls-papergamma/Qwen-Qwen3-0.6B-Base/mnli/bitdistill-tensor-layer-4 |
