# Qwen3 Paper-Alignment Audit, 2026-05-15

Overall status: **pending**.

This audit tracks the queued Qwen3-0.6B branch. Rows remain pending until full validation metrics and prediction traces exist.

## Headline

| field | value |
| --- | --- |
| jobs | 16 |
| complete rows | 0 |
| FP complete tasks | [] |
| BitNet-SFT complete tasks | [] |
| tensor BitDistill complete tasks | [] |
| row BitDistill complete tasks | [] |
| gap-pass rows | 0 |
| paper reproduction ready | false |

## Rows

| job | phase | task | method | scale | layer | queue | complete | accuracy | delta vs FP | CI95 | gap pass | output |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10040 | stage2 | - | bitdistill | tensor | - | RUNNING | false | - | - | - | false | checkpoints/bitdistill-glue-qwen3base-seqcls-papergamma/Qwen-Qwen3-0.6B-Base/continued_pretrain/bitdistill-tensor |
| 10041 | paper_baseline | mnli | fp16_sft | tensor | -1 | PENDING | false | - | - | - | false | checkpoints/bitdistill-glue-qwen3base-seqcls-papergamma/Qwen-Qwen3-0.6B-Base/mnli/fp16_sft-tensor-layer-1 |
| 10042 | paper_baseline | mnli | bitnet_sft | tensor | -1 | PENDING | false | - | - | - | false | checkpoints/bitdistill-glue-qwen3base-seqcls-papergamma/Qwen-Qwen3-0.6B-Base/mnli/bitnet_sft-tensor-layer-1 |
| 10043 | paper_baseline | mnli | bitdistill | tensor | -1 | PENDING | false | - | - | - | false | checkpoints/bitdistill-glue-qwen3base-seqcls-papergamma/Qwen-Qwen3-0.6B-Base/mnli/bitdistill-tensor-layer-1 |
| 10044 | novelty_row_scale | mnli | bitdistill | row | -1 | PENDING | false | - | - | - | false | checkpoints/bitdistill-glue-qwen3base-seqcls-papergamma/Qwen-Qwen3-0.6B-Base/mnli/bitdistill-row-layer-1 |
| 10045 | paper_baseline | qnli | fp16_sft | tensor | -1 | PENDING | false | - | - | - | false | checkpoints/bitdistill-glue-qwen3base-seqcls-papergamma/Qwen-Qwen3-0.6B-Base/qnli/fp16_sft-tensor-layer-1 |
| 10046 | paper_baseline | qnli | bitnet_sft | tensor | -1 | PENDING | false | - | - | - | false | checkpoints/bitdistill-glue-qwen3base-seqcls-papergamma/Qwen-Qwen3-0.6B-Base/qnli/bitnet_sft-tensor-layer-1 |
| 10047 | paper_baseline | qnli | bitdistill | tensor | -1 | PENDING | false | - | - | - | false | checkpoints/bitdistill-glue-qwen3base-seqcls-papergamma/Qwen-Qwen3-0.6B-Base/qnli/bitdistill-tensor-layer-1 |
| 10048 | novelty_row_scale | qnli | bitdistill | row | -1 | PENDING | false | - | - | - | false | checkpoints/bitdistill-glue-qwen3base-seqcls-papergamma/Qwen-Qwen3-0.6B-Base/qnli/bitdistill-row-layer-1 |
| 10049 | paper_baseline | sst2 | fp16_sft | tensor | -1 | PENDING | false | - | - | - | false | checkpoints/bitdistill-glue-qwen3base-seqcls-papergamma/Qwen-Qwen3-0.6B-Base/sst2/fp16_sft-tensor-layer-1 |
| 10050 | paper_baseline | sst2 | bitnet_sft | tensor | -1 | PENDING | false | - | - | - | false | checkpoints/bitdistill-glue-qwen3base-seqcls-papergamma/Qwen-Qwen3-0.6B-Base/sst2/bitnet_sft-tensor-layer-1 |
| 10051 | paper_baseline | sst2 | bitdistill | tensor | -1 | PENDING | false | - | - | - | false | checkpoints/bitdistill-glue-qwen3base-seqcls-papergamma/Qwen-Qwen3-0.6B-Base/sst2/bitdistill-tensor-layer-1 |
| 10052 | novelty_row_scale | sst2 | bitdistill | row | -1 | PENDING | false | - | - | - | false | checkpoints/bitdistill-glue-qwen3base-seqcls-papergamma/Qwen-Qwen3-0.6B-Base/sst2/bitdistill-row-layer-1 |
| 10053 | attention_layer_sweep | mnli | bitdistill | tensor | -8 | PENDING | false | - | - | - | false | checkpoints/bitdistill-glue-qwen3base-seqcls-papergamma/Qwen-Qwen3-0.6B-Base/mnli/bitdistill-tensor-layer-8 |
| 10054 | attention_layer_sweep | mnli | bitdistill | tensor | -2 | PENDING | false | - | - | - | false | checkpoints/bitdistill-glue-qwen3base-seqcls-papergamma/Qwen-Qwen3-0.6B-Base/mnli/bitdistill-tensor-layer-2 |
| 10055 | attention_layer_sweep | mnli | bitdistill | tensor | -4 | PENDING | false | - | - | - | false | checkpoints/bitdistill-glue-qwen3base-seqcls-papergamma/Qwen-Qwen3-0.6B-Base/mnli/bitdistill-tensor-layer-4 |
