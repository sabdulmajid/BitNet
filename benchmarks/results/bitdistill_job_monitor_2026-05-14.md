# BitDistill Job Monitor, 2026-05-14

Job tables: `/mnt/slurm_nfs/a6abdulm/projects/BitNet/benchmark_results/bitdistill_longwarmup_downstream_20260514_163342.tsv, /mnt/slurm_nfs/a6abdulm/projects/BitNet/benchmark_results/bitdistill_longwarmup_downstream_20260514_171512.tsv`.

## Stage-2 Warm-Up

| log | step | max steps | progress | latest CE | effective tokens | target tokens | ETA |
| --- | --- | --- | --- | --- | --- | --- | --- |
| /mnt/slurm_nfs/a6abdulm/projects/BitNet/logs/bitdistill-glue-9894.out | 1960 | 20000 | 0.098000 | 4.882624 | 16056320 | 163840000 | 9.12h |

## Downstream Jobs

| job | task | scale | layer | state | elapsed | node/reason | metrics | accuracy | metrics path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9899 | mnli | tensor | -8 | PENDING | 0:00 | (Dependency) | false | - | /mnt/slurm_nfs/a6abdulm/projects/BitNet/checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| 9900 | mnli | row | -8 | PENDING | 0:00 | (Dependency) | false | - | /mnt/slurm_nfs/a6abdulm/projects/BitNet/checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| 9901 | qnli | tensor | -8 | PENDING | 0:00 | (Dependency) | false | - | /mnt/slurm_nfs/a6abdulm/projects/BitNet/checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| 9902 | qnli | row | -8 | PENDING | 0:00 | (Dependency) | false | - | /mnt/slurm_nfs/a6abdulm/projects/BitNet/checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-row-layer-8/metrics.json |
| 9903 | sst2 | tensor | -8 | PENDING | 0:00 | (Dependency) | false | - | /mnt/slurm_nfs/a6abdulm/projects/BitNet/checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| 9904 | sst2 | row | -8 | PENDING | 0:00 | (Dependency) | false | - | /mnt/slurm_nfs/a6abdulm/projects/BitNet/checkpoints/bitdistill-glue-seqcls-longwarmup/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-row-layer-8/metrics.json |
| 9906 | mnli | tensor | -8 | PENDING | 0:00 | (Dependency) | false | - | /mnt/slurm_nfs/a6abdulm/projects/BitNet/checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/mnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| 9907 | qnli | tensor | -8 | PENDING | 0:00 | (Dependency) | false | - | /mnt/slurm_nfs/a6abdulm/projects/BitNet/checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/qnli/bitdistill-longwarmup-tensor-layer-8/metrics.json |
| 9908 | sst2 | tensor | -8 | PENDING | 0:00 | (Dependency) | false | - | /mnt/slurm_nfs/a6abdulm/projects/BitNet/checkpoints/bitdistill-glue-seqcls-longwarmup-papergamma/Qwen-Qwen2.5-0.5B/sst2/bitdistill-longwarmup-tensor-layer-8/metrics.json |
