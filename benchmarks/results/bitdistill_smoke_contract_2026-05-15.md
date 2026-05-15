# BitDistill Smoke Contract, 2026-05-15
Overall status: `pass`.
Work dir: `benchmark_results/bitdistill-smoke-contract-2026-05-15`.

GGUF export checks use a smoke-only synthetic tokenizer stub. They validate packed tensor emission, row-scale `I2_SR` metadata, and SubLN key mapping; they do not validate text generation quality.

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| attention relation KD uses L2-normalized states | pass | F.normalize before relation matmul |  |
| attention relation KD sums Q/K/V losses by default | pass | default attention_qkv_reduction=sum |  |
| help command exits zero | pass | returncode=0 |  |
| py_compile command exits zero | pass | returncode=0 |  |
| continued_pretrain command exits zero | pass | returncode=0 |  |
| continued_pretrain_row command exits zero | pass | returncode=0 |  |
| task_sft command exits zero | pass | returncode=0 |  |
| task_sft_row command exits zero | pass | returncode=0 |  |
| continued-pretrain writes metrics | pass | benchmark_results/bitdistill-smoke-contract-2026-05-15/continued_pretrain/metrics.json |  |
| continued-pretrain takes two steps | pass | steps=2 |  |
| continued-pretrain uses BitLinear and SubLN | pass | bitlinear=15, subln=4 |  |
| continued-pretrain CE is finite | pass | ce=5.597681522369385 |  |
| continued-pretrain ternary export is valid | pass | codes=15, scales=15, bitlinear=15 |  |
| row continued-pretrain writes metrics | pass | benchmark_results/bitdistill-smoke-contract-2026-05-15/continued_pretrain_row/metrics.json |  |
| row continued-pretrain takes two steps | pass | steps=2 |  |
| row continued-pretrain uses BitLinear and SubLN | pass | bitlinear=15, subln=4 |  |
| row continued-pretrain CE is finite | pass | ce=5.601589679718018 |  |
| row continued-pretrain ternary export is valid | pass | codes=15, tensor_scales=0, row_scales=15 |  |
| continued_pretrain_i2s_export command exits zero | pass | returncode=0 |  |
| continued_pretrain_row_i2sr_export command exits zero | pass | returncode=0 |  |
| continued-pretrain tensor GGUF export maps SubLN | pass | packed=14, outfile=benchmark_results/bitdistill-smoke-contract-2026-05-15/continued_pretrain_i2s_smoke.gguf |  |
| row continued-pretrain I2_SR GGUF export maps SubLN and row scales | pass | row_packed=14, outfile=benchmark_results/bitdistill-smoke-contract-2026-05-15/continued_pretrain_row_i2sr_smoke.gguf |  |
| task-sft writes metrics | pass | benchmark_results/bitdistill-smoke-contract-2026-05-15/task_sft/metrics.json |  |
| task-sft takes two steps | pass | steps=2 |  |
| task-sft uses BitLinear and SubLN | pass | bitlinear=15, subln=4 |  |
| task-sft logits KD is finite | pass | weighted_logit_kd=0.0093889981508255 |  |
| task-sft attention KD is finite | pass | weighted_attention_kd=11.875125885009766 |  |
| task-sft records paper-style Q/K/V reduction | pass | attention_qkv_reduction=sum |  |
| task-sft eval accuracy is finite | pass | accuracy=0.5 |  |
| task-sft writes per-example predictions | pass | predictions=8, eval_examples=8.0 |  |
| task-sft tensor-scale ternary export is valid | pass | codes=15, tensor_scales=15, row_scales=0 |  |
| row task-sft writes metrics | pass | benchmark_results/bitdistill-smoke-contract-2026-05-15/task_sft_row/metrics.json |  |
| row task-sft takes two steps | pass | steps=2 |  |
| row task-sft uses BitLinear and SubLN | pass | bitlinear=15, subln=4 |  |
| row task-sft logits KD is finite | pass | weighted_logit_kd=0.009045824408531189 |  |
| row task-sft attention KD is finite | pass | weighted_attention_kd=11.805890083312988 |  |
| row task-sft records paper-style Q/K/V reduction | pass | attention_qkv_reduction=sum |  |
| row task-sft eval accuracy is finite | pass | accuracy=0.5 |  |
| row task-sft writes per-example predictions | pass | predictions=8, eval_examples=8.0 |  |
| row task-sft row-scale ternary export is valid | pass | codes=15, tensor_scales=0, row_scales=15 |  |
