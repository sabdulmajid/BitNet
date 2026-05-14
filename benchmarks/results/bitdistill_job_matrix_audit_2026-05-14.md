# BitDistill Job Matrix Audit, 2026-05-14

Overall status: `pass`.

Monitor JSON: `benchmark_results/bitdistill_job_monitor_2026-05-14.json`.

Warm-up state: `/mnt/slurm_nfs/a6abdulm/projects/BitNet/checkpoints/bitdistill-glue-longwarmup/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor-20k/custom_state_dict.pt`.

Observed rows: `38`. Expected rows: `38`. Configured rows: `38`.

Job states: `{'PENDING': 38}`.

Rows with fields inferred from submitter defaults: `0`.

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| monitor json exists | pass | benchmark_results/bitdistill_job_monitor_2026-05-14.json |  |
| active row count matches design | pass | rows=38, expected=38 |  |
| output directories are unique | pass | duplicates=[] |  |
| all expected experiment rows are present and configured | pass | configured=38/38 |  |
| warm-up progress is finite | pass | step=11530/20000 |  |

## Expected Matrix

| family | task | format | scale | attention gamma | head init | job | state | teacher metrics | inferred fields | issues |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seqcls_gamma100 | mnli | sequence_classification | tensor | 100.000000 | 0 | 9956 | PENDING | true | none | none |
| seqcls_gamma100_headinit | mnli | sequence_classification | tensor | 100.000000 | 1 | 9971 | PENDING | true | none | none |
| seqcls_gamma100 | mnli | sequence_classification | row | 100.000000 | 0 | 9957 | PENDING | true | none | none |
| seqcls_gamma100_headinit | mnli | sequence_classification | row | 100.000000 | 1 | 9972 | PENDING | true | none | none |
| seqcls_gamma100 | qnli | sequence_classification | tensor | 100.000000 | 0 | 9958 | PENDING | true | none | none |
| seqcls_gamma100_headinit | qnli | sequence_classification | tensor | 100.000000 | 1 | 9973 | PENDING | true | none | none |
| seqcls_gamma100 | qnli | sequence_classification | row | 100.000000 | 0 | 9959 | PENDING | true | none | none |
| seqcls_gamma100_headinit | qnli | sequence_classification | row | 100.000000 | 1 | 9974 | PENDING | true | none | none |
| seqcls_gamma100 | sst2 | sequence_classification | tensor | 100.000000 | 0 | 9960 | PENDING | true | none | none |
| seqcls_gamma100_headinit | sst2 | sequence_classification | tensor | 100.000000 | 1 | 9975 | PENDING | true | none | none |
| seqcls_gamma100 | sst2 | sequence_classification | row | 100.000000 | 0 | 9961 | PENDING | true | none | none |
| seqcls_gamma100_headinit | sst2 | sequence_classification | row | 100.000000 | 1 | 9976 | PENDING | true | none | none |
| seqcls_paper_gamma100000_tensor | mnli | sequence_classification | tensor | 100000.000000 | 0 | 9962 | PENDING | true | none | none |
| seqcls_paper_gamma100000_row | mnli | sequence_classification | row | 100000.000000 | 0 | 9981 | PENDING | true | none | none |
| seqcls_paper_gamma100000_tensor | qnli | sequence_classification | tensor | 100000.000000 | 0 | 9963 | PENDING | true | none | none |
| seqcls_paper_gamma100000_row | qnli | sequence_classification | row | 100000.000000 | 0 | 9982 | PENDING | true | none | none |
| seqcls_paper_gamma100000_tensor | sst2 | sequence_classification | tensor | 100000.000000 | 0 | 9964 | PENDING | true | none | none |
| seqcls_paper_gamma100000_row | sst2 | sequence_classification | row | 100000.000000 | 0 | 9983 | PENDING | true | none | none |
| seqcls_paper_gamma100000_tensor_lr1e-05 | mnli | sequence_classification | tensor | 100000.000000 | 0 | 9987 | PENDING | true | none | none |
| seqcls_paper_gamma100000_tensor_lr1e-05 | qnli | sequence_classification | tensor | 100000.000000 | 0 | 9988 | PENDING | true | none | none |
| seqcls_paper_gamma100000_tensor_lr1e-05 | sst2 | sequence_classification | tensor | 100000.000000 | 0 | 9989 | PENDING | true | none | none |
| seqcls_paper_gamma100000_tensor_lr5e-05 | mnli | sequence_classification | tensor | 100000.000000 | 0 | 9990 | PENDING | true | none | none |
| seqcls_paper_gamma100000_tensor_lr5e-05 | qnli | sequence_classification | tensor | 100000.000000 | 0 | 9991 | PENDING | true | none | none |
| seqcls_paper_gamma100000_tensor_lr5e-05 | sst2 | sequence_classification | tensor | 100000.000000 | 0 | 9992 | PENDING | true | none | none |
| seqcls_paper_gamma100000_tensor_headinit | mnli | sequence_classification | tensor | 100000.000000 | 1 | 9993 | PENDING | true | none | none |
| seqcls_paper_gamma100000_tensor_headinit | qnli | sequence_classification | tensor | 100000.000000 | 1 | 9994 | PENDING | true | none | none |
| seqcls_paper_gamma100000_tensor_headinit | sst2 | sequence_classification | tensor | 100000.000000 | 1 | 9995 | PENDING | true | none | none |
| mnli_gamma1000 | mnli | sequence_classification | tensor | 1000.000000 | 0 | 9965 | PENDING | true | none | none |
| mnli_gamma10000 | mnli | sequence_classification | tensor | 10000.000000 | 0 | 9966 | PENDING | true | none | none |
| mnli_layer_sweep_-1 | mnli | sequence_classification | tensor | 100.000000 | 0 | 9978 | PENDING | true | none | none |
| mnli_layer_sweep_-2 | mnli | sequence_classification | tensor | 100.000000 | 0 | 9979 | PENDING | true | none | none |
| mnli_layer_sweep_-4 | mnli | sequence_classification | tensor | 100.000000 | 0 | 9980 | PENDING | true | none | none |
| causal_densehead_gamma100 | mnli | causal_lm | tensor | 100.000000 | 0 | 9943 | PENDING | true | none | none |
| causal_densehead_gamma100 | mnli | causal_lm | row | 100.000000 | 0 | 9944 | PENDING | true | none | none |
| causal_densehead_gamma100 | qnli | causal_lm | tensor | 100.000000 | 0 | 9945 | PENDING | true | none | none |
| causal_densehead_gamma100 | qnli | causal_lm | row | 100.000000 | 0 | 9946 | PENDING | true | none | none |
| causal_densehead_gamma100 | sst2 | causal_lm | tensor | 100.000000 | 0 | 9947 | PENDING | true | none | none |
| causal_densehead_gamma100 | sst2 | causal_lm | row | 100.000000 | 0 | 9948 | PENDING | true | none | none |
