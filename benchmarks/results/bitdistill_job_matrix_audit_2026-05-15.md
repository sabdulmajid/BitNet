# BitDistill Job Matrix Audit, 2026-05-15

Overall status: `pass`.

Monitor JSON: `benchmark_results/bitdistill_job_monitor_2026-05-15.json`.

Warm-up state: `/mnt/slurm_nfs/a6abdulm/projects/BitNet/checkpoints/bitdistill-glue-longwarmup/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor-20k/custom_state_dict.pt`.

train_bitdistill.py sha256: `5442c0d392a1`. Attention Q/K/V reduction default: `sum`.

Observed rows: `38`. Expected rows: `38`. Configured rows: `38`.

Job states: `{'not_in_squeue': 38}`.

Rows with fields inferred from submitter defaults: `0`.

Stored downstream scripts checked: `0` active / `38` total rows. Failures: `0`.

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| monitor json exists | pass | benchmark_results/bitdistill_job_monitor_2026-05-15.json |  |
| train_bitdistill defaults to paper-style Q/K/V attention sum | pass | train_bitdistill.py sha256=5442c0d392a1 |  |
| active row count matches design | pass | rows=38, expected=38 |  |
| output directories are unique | pass | duplicates=[] |  |
| all expected experiment rows are present and configured | pass | configured=38/38 |  |
| downstream stored scripts include critical KD/export arguments | pass | active checked=0, total rows=38, failures=0 |  |
| warm-up progress is finite | pass | step=20000/20000 |  |

## Expected Matrix

| family | task | format | scale | attention gamma | head init | job | state | teacher metrics | inferred fields | issues |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seqcls_gamma100 | mnli | sequence_classification | tensor | 100.000000 | 0 | 9956 | not_in_squeue | true | none | none |
| seqcls_gamma100_headinit | mnli | sequence_classification | tensor | 100.000000 | 1 | 9971 | not_in_squeue | true | none | none |
| seqcls_gamma100 | mnli | sequence_classification | row | 100.000000 | 0 | 9957 | not_in_squeue | true | none | none |
| seqcls_gamma100_headinit | mnli | sequence_classification | row | 100.000000 | 1 | 9972 | not_in_squeue | true | none | none |
| seqcls_gamma100 | qnli | sequence_classification | tensor | 100.000000 | 0 | 9958 | not_in_squeue | true | none | none |
| seqcls_gamma100_headinit | qnli | sequence_classification | tensor | 100.000000 | 1 | 9973 | not_in_squeue | true | none | none |
| seqcls_gamma100 | qnli | sequence_classification | row | 100.000000 | 0 | 9959 | not_in_squeue | true | none | none |
| seqcls_gamma100_headinit | qnli | sequence_classification | row | 100.000000 | 1 | 9974 | not_in_squeue | true | none | none |
| seqcls_gamma100 | sst2 | sequence_classification | tensor | 100.000000 | 0 | 9960 | not_in_squeue | true | none | none |
| seqcls_gamma100_headinit | sst2 | sequence_classification | tensor | 100.000000 | 1 | 9975 | not_in_squeue | true | none | none |
| seqcls_gamma100 | sst2 | sequence_classification | row | 100.000000 | 0 | 9961 | not_in_squeue | true | none | none |
| seqcls_gamma100_headinit | sst2 | sequence_classification | row | 100.000000 | 1 | 9976 | not_in_squeue | true | none | none |
| seqcls_paper_gamma100000_tensor | mnli | sequence_classification | tensor | 100000.000000 | 0 | 9962 | not_in_squeue | true | none | none |
| seqcls_paper_gamma100000_row | mnli | sequence_classification | row | 100000.000000 | 0 | 9981 | not_in_squeue | true | none | none |
| seqcls_paper_gamma100000_tensor | qnli | sequence_classification | tensor | 100000.000000 | 0 | 9963 | not_in_squeue | true | none | none |
| seqcls_paper_gamma100000_row | qnli | sequence_classification | row | 100000.000000 | 0 | 9982 | not_in_squeue | true | none | none |
| seqcls_paper_gamma100000_tensor | sst2 | sequence_classification | tensor | 100000.000000 | 0 | 9964 | not_in_squeue | true | none | none |
| seqcls_paper_gamma100000_row | sst2 | sequence_classification | row | 100000.000000 | 0 | 9983 | not_in_squeue | true | none | none |
| seqcls_paper_gamma100000_tensor_lr1e-05 | mnli | sequence_classification | tensor | 100000.000000 | 0 | 9987 | not_in_squeue | true | none | none |
| seqcls_paper_gamma100000_tensor_lr1e-05 | qnli | sequence_classification | tensor | 100000.000000 | 0 | 9988 | not_in_squeue | true | none | none |
| seqcls_paper_gamma100000_tensor_lr1e-05 | sst2 | sequence_classification | tensor | 100000.000000 | 0 | 9989 | not_in_squeue | true | none | none |
| seqcls_paper_gamma100000_tensor_lr5e-05 | mnli | sequence_classification | tensor | 100000.000000 | 0 | 9990 | not_in_squeue | true | none | none |
| seqcls_paper_gamma100000_tensor_lr5e-05 | qnli | sequence_classification | tensor | 100000.000000 | 0 | 9991 | not_in_squeue | true | none | none |
| seqcls_paper_gamma100000_tensor_lr5e-05 | sst2 | sequence_classification | tensor | 100000.000000 | 0 | 9992 | not_in_squeue | true | none | none |
| seqcls_paper_gamma100000_tensor_headinit | mnli | sequence_classification | tensor | 100000.000000 | 1 | 9993 | not_in_squeue | true | none | none |
| seqcls_paper_gamma100000_tensor_headinit | qnli | sequence_classification | tensor | 100000.000000 | 1 | 9994 | not_in_squeue | true | none | none |
| seqcls_paper_gamma100000_tensor_headinit | sst2 | sequence_classification | tensor | 100000.000000 | 1 | 9995 | not_in_squeue | true | none | none |
| mnli_gamma1000 | mnli | sequence_classification | tensor | 1000.000000 | 0 | 9965 | not_in_squeue | true | none | none |
| mnli_gamma10000 | mnli | sequence_classification | tensor | 10000.000000 | 0 | 9966 | not_in_squeue | true | none | none |
| mnli_layer_sweep_-1 | mnli | sequence_classification | tensor | 100.000000 | 0 | 9978 | not_in_squeue | true | none | none |
| mnli_layer_sweep_-2 | mnli | sequence_classification | tensor | 100.000000 | 0 | 9979 | not_in_squeue | true | none | none |
| mnli_layer_sweep_-4 | mnli | sequence_classification | tensor | 100.000000 | 0 | 9980 | not_in_squeue | true | none | none |
| causal_densehead_gamma100 | mnli | causal_lm | tensor | 100.000000 | 0 | 9943 | not_in_squeue | true | none | none |
| causal_densehead_gamma100 | mnli | causal_lm | row | 100.000000 | 0 | 9944 | not_in_squeue | true | none | none |
| causal_densehead_gamma100 | qnli | causal_lm | tensor | 100.000000 | 0 | 9945 | not_in_squeue | true | none | none |
| causal_densehead_gamma100 | qnli | causal_lm | row | 100.000000 | 0 | 9946 | not_in_squeue | true | none | none |
| causal_densehead_gamma100 | sst2 | causal_lm | tensor | 100.000000 | 0 | 9947 | not_in_squeue | true | none | none |
| causal_densehead_gamma100 | sst2 | causal_lm | row | 100.000000 | 0 | 9948 | not_in_squeue | true | none | none |
