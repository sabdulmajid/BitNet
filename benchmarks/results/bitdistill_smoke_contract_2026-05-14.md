# BitDistill Smoke Contract, 2026-05-14
Overall status: `pass`.
Work dir: `benchmark_results/bitdistill-smoke-contract-2026-05-14`.

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| attention relation KD uses L2-normalized states | pass | F.normalize before relation matmul |  |
| help command exits zero | pass | returncode=0 |  |
| py_compile command exits zero | pass | returncode=0 |  |
| continued_pretrain command exits zero | pass | returncode=0 |  |
| task_sft command exits zero | pass | returncode=0 |  |
| task_sft_row command exits zero | pass | returncode=0 |  |
| continued-pretrain writes metrics | pass | benchmark_results/bitdistill-smoke-contract-2026-05-14/continued_pretrain/metrics.json |  |
| continued-pretrain takes two steps | pass | steps=2 |  |
| continued-pretrain uses BitLinear and SubLN | pass | bitlinear=15, subln=4 |  |
| continued-pretrain CE is finite | pass | ce=5.5875725746154785 |  |
| continued-pretrain ternary export is valid | pass | codes=15, scales=15, bitlinear=15 |  |
| task-sft writes metrics | pass | benchmark_results/bitdistill-smoke-contract-2026-05-14/task_sft/metrics.json |  |
| task-sft takes two steps | pass | steps=2 |  |
| task-sft uses BitLinear and SubLN | pass | bitlinear=15, subln=4 |  |
| task-sft logits KD is finite | pass | weighted_logit_kd=0.005214028060436249 |  |
| task-sft attention KD is finite | pass | weighted_attention_kd=7.201216220855713 |  |
| task-sft eval accuracy is finite | pass | accuracy=0.5 |  |
| task-sft writes per-example predictions | pass | predictions=8, eval_examples=8.0 |  |
| task-sft tensor-scale ternary export is valid | pass | codes=15, tensor_scales=15, row_scales=0 |  |
| row task-sft writes metrics | pass | benchmark_results/bitdistill-smoke-contract-2026-05-14/task_sft_row/metrics.json |  |
| row task-sft takes two steps | pass | steps=2 |  |
| row task-sft uses BitLinear and SubLN | pass | bitlinear=15, subln=4 |  |
| row task-sft logits KD is finite | pass | weighted_logit_kd=0.00562131404876709 |  |
| row task-sft attention KD is finite | pass | weighted_attention_kd=7.589199542999268 |  |
| row task-sft eval accuracy is finite | pass | accuracy=0.5 |  |
| row task-sft writes per-example predictions | pass | predictions=8, eval_examples=8.0 |  |
| row task-sft row-scale ternary export is valid | pass | codes=15, tensor_scales=0, row_scales=15 |  |
