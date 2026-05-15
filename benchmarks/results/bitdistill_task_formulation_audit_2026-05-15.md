# BitDistill Task Formulation Audit, 2026-05-15

Model: `Qwen/Qwen2.5-0.5B`.

Paper anchor source: BitDistill paper excerpt, Table 3 Qwen2.5-0.5B MNLI row.

This audit prevents sequence-classification, causal prompt scoring, and paper table anchors from being mixed into one overbroad claim.

Sequence-classification full baselines: `6`. Causal diagnostic rows materialized: `17`. Pending paper candidates: `9`.

## Claim Controls

| control |
| --- |
| Current strict reproduction claim should be limited to the sequence-classification branch until paper training code confirms a different task head/prompt formulation. |
| Causal-LM GLUE rows are diagnostics for deployment-style prompting and should not be mixed with sequence-classification rows in one headline accuracy table. |
| The provided excerpt only gives Qwen2.5-0.5B anchors for MNLI; QNLI/SST2 local Qwen2.5 rows are reproduction targets by task, not direct table-value reproductions. |
| BitDistill success remains pending until long-warmup tensor/row candidates finish full validation. |

## Rows

| task | run | formulation | paper role | exists | accuracy | eval n | full eval | paper Qwen2.5 MNLI anchor | local-anchor | metrics path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mnli | FP16-SFT | sequence_classification | baseline | true | 0.807641 | 9815 | true | 0.799100 | 0.008541 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1/metrics.json |
| qnli | FP16-SFT | sequence_classification | baseline | true | 0.898957 | 5463 | true | - | - | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/qnli/fp16_sft-tensor-layer-1/metrics.json |
| sst2 | FP16-SFT | sequence_classification | baseline | true | 0.925459 | 872 | true | - | - | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/sst2/fp16_sft-tensor-layer-1/metrics.json |
| mnli | BitNet-SFT | sequence_classification | baseline | true | 0.487621 | 9815 | true | 0.608000 | -0.120379 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/bitnet_sft-tensor-layer-1/metrics.json |
| qnli | BitNet-SFT | sequence_classification | baseline | true | 0.596925 | 5463 | true | - | - | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/qnli/bitnet_sft-tensor-layer-1/metrics.json |
| sst2 | BitNet-SFT | sequence_classification | baseline | true | 0.770642 | 872 | true | - | - | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/sst2/bitnet_sft-tensor-layer-1/metrics.json |
| mnli | BitDistill short tensor | sequence_classification | diagnostic | true | 0.525217 | 9815 | true | 0.799800 | -0.274583 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-layer-1/metrics.json |
| qnli | BitDistill short tensor | sequence_classification | diagnostic | true | 0.596925 | 5463 | true | - | - | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/qnli/bitdistill-tensor-layer-1/metrics.json |
| sst2 | BitDistill short tensor | sequence_classification | diagnostic | true | 0.815367 | 872 | true | - | - | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/sst2/bitdistill-tensor-layer-1/metrics.json |
| mnli | BitDistill short row | sequence_classification | diagnostic | true | 0.516556 | 9815 | true | 0.799800 | -0.283244 | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/mnli/bitdistill-row-layer-1/metrics.json |
| qnli | BitDistill short row | sequence_classification | diagnostic | true | 0.618525 | 5463 | true | - | - | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/qnli/bitdistill-row-layer-1/metrics.json |
| sst2 | BitDistill short row | sequence_classification | diagnostic | true | 0.808486 | 872 | true | - | - | checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B/sst2/bitdistill-row-layer-1/metrics.json |
| mnli | BitDistill longwarmup tensor gamma100 | sequence_classification | pending_candidate | false | - | - | false | 0.799800 | - | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| qnli | BitDistill longwarmup tensor gamma100 | sequence_classification | pending_candidate | false | - | - | false | - | - | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| sst2 | BitDistill longwarmup tensor gamma100 | sequence_classification | pending_candidate | false | - | - | false | - | - | checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| mnli | BitDistill longwarmup tensor paper gamma | sequence_classification | pending_paper_candidate | false | - | - | false | 0.799800 | - | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| qnli | BitDistill longwarmup tensor paper gamma | sequence_classification | pending_paper_candidate | false | - | - | false | - | - | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| sst2 | BitDistill longwarmup tensor paper gamma | sequence_classification | pending_paper_candidate | false | - | - | false | - | - | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| mnli | BitDistill longwarmup row paper gamma | sequence_classification | pending_row_candidate | false | - | - | false | 0.799800 | - | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-row/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| qnli | BitDistill longwarmup row paper gamma | sequence_classification | pending_row_candidate | false | - | - | false | - | - | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-row/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| sst2 | BitDistill longwarmup row paper gamma | sequence_classification | pending_row_candidate | false | - | - | false | - | - | checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma-row/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-row-layer-8/metrics.json |
| mnli | Causal FP16-SFT words | causal_lm_words | diagnostic | true | 0.829852 | 9815 | true | - | - | checkpoints/bitdistill-glue/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1/metrics.json |
| qnli | Causal FP16-SFT words | causal_lm_words | diagnostic | true | 0.900970 | 5463 | true | - | - | checkpoints/bitdistill-glue/Qwen-Qwen2.5-0.5B/qnli/fp16_sft-tensor-layer-1/metrics.json |
| sst2 | Causal FP16-SFT words | causal_lm_words | diagnostic | true | 0.939220 | 872 | true | - | - | checkpoints/bitdistill-glue/Qwen-Qwen2.5-0.5B/sst2/fp16_sft-tensor-layer-1/metrics.json |
| mnli | Causal BitNet-SFT words | causal_lm_words | diagnostic | true | 0.517983 | 9815 | true | - | - | checkpoints/bitdistill-glue/Qwen-Qwen2.5-0.5B/mnli/bitnet_sft-tensor-layer-1/metrics.json |
| qnli | Causal BitNet-SFT words | causal_lm_words | diagnostic | true | 0.614681 | 5463 | true | - | - | checkpoints/bitdistill-glue/Qwen-Qwen2.5-0.5B/qnli/bitnet_sft-tensor-layer-1/metrics.json |
| sst2 | Causal BitNet-SFT words | causal_lm_words | diagnostic | true | 0.831422 | 872 | true | - | - | checkpoints/bitdistill-glue/Qwen-Qwen2.5-0.5B/sst2/bitnet_sft-tensor-layer-1/metrics.json |
| mnli | Causal BitDistill short tensor words | causal_lm_words | diagnostic | true | 0.534692 | 9815 | true | - | - | checkpoints/bitdistill-glue/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-layer-1/metrics.json |
| qnli | Causal BitDistill short tensor words | causal_lm_words | diagnostic | true | 0.607542 | 5463 | true | - | - | checkpoints/bitdistill-glue/Qwen-Qwen2.5-0.5B/qnli/bitdistill-tensor-layer-1/metrics.json |
| sst2 | Causal BitDistill short tensor words | causal_lm_words | diagnostic | false | - | - | false | - | - | checkpoints/bitdistill-glue/Qwen-Qwen2.5-0.5B/sst2/bitdistill-tensor-layer-1/metrics.json |
| mnli | Causal FP16-SFT letters | causal_lm_letters | diagnostic | true | 0.831482 | 9815 | true | - | - | checkpoints/bitdistill-glue-letters/Qwen-Qwen2.5-0.5B/mnli/fp16_sft-tensor-layer-1/metrics.json |
| qnli | Causal FP16-SFT letters | causal_lm_letters | diagnostic | false | - | - | false | - | - | checkpoints/bitdistill-glue-letters/Qwen-Qwen2.5-0.5B/qnli/fp16_sft-tensor-layer-1/metrics.json |
| sst2 | Causal FP16-SFT letters | causal_lm_letters | diagnostic | false | - | - | false | - | - | checkpoints/bitdistill-glue-letters/Qwen-Qwen2.5-0.5B/sst2/fp16_sft-tensor-layer-1/metrics.json |
| mnli | Causal BitNet-SFT letters | causal_lm_letters | diagnostic | true | 0.465512 | 9815 | true | - | - | checkpoints/bitdistill-glue-letters/Qwen-Qwen2.5-0.5B/mnli/bitnet_sft-tensor-layer-1/metrics.json |
| qnli | Causal BitNet-SFT letters | causal_lm_letters | diagnostic | false | - | - | false | - | - | checkpoints/bitdistill-glue-letters/Qwen-Qwen2.5-0.5B/qnli/bitnet_sft-tensor-layer-1/metrics.json |
| sst2 | Causal BitNet-SFT letters | causal_lm_letters | diagnostic | false | - | - | false | - | - | checkpoints/bitdistill-glue-letters/Qwen-Qwen2.5-0.5B/sst2/bitnet_sft-tensor-layer-1/metrics.json |
| mnli | Causal BitDistill short tensor letters | causal_lm_letters | diagnostic | true | 0.510239 | 9815 | true | - | - | checkpoints/bitdistill-glue-letters/Qwen-Qwen2.5-0.5B/mnli/bitdistill-tensor-layer-1/metrics.json |
| qnli | Causal BitDistill short tensor letters | causal_lm_letters | diagnostic | false | - | - | false | - | - | checkpoints/bitdistill-glue-letters/Qwen-Qwen2.5-0.5B/qnli/bitdistill-tensor-layer-1/metrics.json |
| sst2 | Causal BitDistill short tensor letters | causal_lm_letters | diagnostic | false | - | - | false | - | - | checkpoints/bitdistill-glue-letters/Qwen-Qwen2.5-0.5B/sst2/bitdistill-tensor-layer-1/metrics.json |
| mnli | Causal BitDistill longwarmup tensor letters | causal_lm_letters | deployment_diagnostic | true | 0.615181 | 9815 | true | - | - | checkpoints/bitdistill-glue-causal-longwarmup-densehead/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| qnli | Causal BitDistill longwarmup tensor letters | causal_lm_letters | deployment_diagnostic | true | 0.765697 | 5463 | true | - | - | checkpoints/bitdistill-glue-causal-longwarmup-densehead/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| sst2 | Causal BitDistill longwarmup tensor letters | causal_lm_letters | deployment_diagnostic | true | 0.833716 | 872 | true | - | - | checkpoints/bitdistill-glue-causal-longwarmup-densehead/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| mnli | Causal BitDistill longwarmup row letters | causal_lm_letters | deployment_diagnostic | true | 0.608355 | 9815 | true | - | - | checkpoints/bitdistill-glue-causal-longwarmup-densehead/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| qnli | Causal BitDistill longwarmup row letters | causal_lm_letters | deployment_diagnostic | true | 0.770822 | 5463 | true | - | - | checkpoints/bitdistill-glue-causal-longwarmup-densehead/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| sst2 | Causal BitDistill longwarmup row letters | causal_lm_letters | deployment_diagnostic | true | 0.840596 | 872 | true | - | - | checkpoints/bitdistill-glue-causal-longwarmup-densehead/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-row-layer-8/metrics.json |
