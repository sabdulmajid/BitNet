# BitDistill Job Matrix Audit, 2026-05-14

Overall status: `pass`.

Monitor JSON: `benchmark_results/bitdistill_job_monitor_2026-05-14.json`.

Warm-up state: `/mnt/slurm_nfs/a6abdulm/projects/BitNet/checkpoints/bitdistill-glue-longwarmup/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor-20k/custom_state_dict.pt`.

Observed rows: `17`. Expected rows: `17`. Configured rows: `17`.

Job states: `{'PENDING': 17}`.

Rows with fields inferred from submitter defaults: `11`.

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| monitor json exists | pass | benchmark_results/bitdistill_job_monitor_2026-05-14.json |  |
| active row count matches design | pass | rows=17, expected=17 |  |
| output directories are unique | pass | duplicates=[] |  |
| all expected experiment rows are present and configured | pass | configured=17/17 |  |
| warm-up progress is finite | pass | step=6800/20000 |  |

## Expected Matrix

| family | task | format | scale | attention gamma | job | state | teacher metrics | inferred fields | issues |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seqcls_gamma100 | mnli | sequence_classification | tensor | 100.000000 | 9925 | PENDING | true | task_format,label_scheme,candidate_score,exclude_linear_regex | none |
| seqcls_gamma100 | mnli | sequence_classification | row | 100.000000 | 9926 | PENDING | true | task_format,label_scheme,candidate_score,exclude_linear_regex | none |
| seqcls_gamma100 | qnli | sequence_classification | tensor | 100.000000 | 9927 | PENDING | true | task_format,label_scheme,candidate_score,exclude_linear_regex | none |
| seqcls_gamma100 | qnli | sequence_classification | row | 100.000000 | 9928 | PENDING | true | task_format,label_scheme,candidate_score,exclude_linear_regex | none |
| seqcls_gamma100 | sst2 | sequence_classification | tensor | 100.000000 | 9929 | PENDING | true | task_format,label_scheme,candidate_score,exclude_linear_regex | none |
| seqcls_gamma100 | sst2 | sequence_classification | row | 100.000000 | 9930 | PENDING | true | task_format,label_scheme,candidate_score,exclude_linear_regex | none |
| seqcls_paper_gamma100000 | mnli | sequence_classification | tensor | 100000.000000 | 9931 | PENDING | true | task_format,label_scheme,candidate_score,exclude_linear_regex | none |
| seqcls_paper_gamma100000 | qnli | sequence_classification | tensor | 100000.000000 | 9932 | PENDING | true | task_format,label_scheme,candidate_score,exclude_linear_regex | none |
| seqcls_paper_gamma100000 | sst2 | sequence_classification | tensor | 100000.000000 | 9933 | PENDING | true | task_format,label_scheme,candidate_score,exclude_linear_regex | none |
| mnli_gamma1000 | mnli | sequence_classification | tensor | 1000.000000 | 9934 | PENDING | true | task_format,label_scheme,candidate_score,exclude_linear_regex | none |
| mnli_gamma10000 | mnli | sequence_classification | tensor | 10000.000000 | 9935 | PENDING | true | task_format,label_scheme,candidate_score,exclude_linear_regex | none |
| causal_densehead_gamma100 | mnli | causal_lm | tensor | 100.000000 | 9943 | PENDING | true | none | none |
| causal_densehead_gamma100 | mnli | causal_lm | row | 100.000000 | 9944 | PENDING | true | none | none |
| causal_densehead_gamma100 | qnli | causal_lm | tensor | 100.000000 | 9945 | PENDING | true | none | none |
| causal_densehead_gamma100 | qnli | causal_lm | row | 100.000000 | 9946 | PENDING | true | none | none |
| causal_densehead_gamma100 | sst2 | causal_lm | tensor | 100.000000 | 9947 | PENDING | true | none | none |
| causal_densehead_gamma100 | sst2 | causal_lm | row | 100.000000 | 9948 | PENDING | true | none | none |
