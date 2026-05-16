# BitDistill Smoke Contract, 2026-05-15
Overall status: `pass`.
Work dir: `benchmark_results/bitdistill-smoke-contract-2026-05-15`.

GGUF export checks use a smoke-only synthetic tokenizer stub. They validate packed tensor emission, row-scale `I2_SR` metadata, and SubLN key mapping; they do not validate text generation quality.

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| attention relation KD uses L2-normalized states | pass | F.normalize before relation matmul |  |
| attention relation KD sums Q/K/V losses by default | pass | default attention_qkv_reduction=sum |  |
| attention relation KD exposes Q/K/V components | pass | raw and weighted Q/K/V attention KD fields are present |  |
| BitLinear activation quantization telemetry is implemented | pass | activation A8 clipping, edge occupancy, scale, and absmax fields are present |  |
| BitLinear quantization dynamics telemetry is implemented | pass | sampled ternary flip-rate and scale-drift fields are present |  |
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
| task-sft Q/K/V attention KD split is finite | pass | q=0.03362368792295456, k=0.04242571443319321, v=0.04270185902714729 |  |
| task-sft weighted Q/K/V split sums to aggregate | pass | sum=11.875126123428345, aggregate=11.875125885009766 |  |
| task-sft records paper-style Q/K/V reduction | pass | attention_qkv_reduction=sum |  |
| task-sft eval accuracy is finite | pass | accuracy=0.5 |  |
| task-sft writes per-example predictions | pass | predictions=8, eval_examples=8.0 |  |
| task-sft tensor-scale ternary export is valid | pass | codes=15, tensor_scales=15, row_scales=0 |  |
| task-sft activation telemetry is finite | pass | telemetry_rows=2, last={'absmax_max': 6.895327091217041, 'absmax_mean': 3.0311591625213623, 'activation_quantized_modules': 15, 'clipped_fraction': 0.0, 'clipped_values': 0, 'int8_edge_fraction': 0.0049217267552182165, 'int8_edge_values': 332, 'module_count': 15, 'negative_edge_fraction': 0.0, 'positive_edge_fraction': 0.0049217267552182165, 'scale_count': 465, 'scale_max': 0.0542939156293869, 'scale_mean': 0.023867396637797356, 'scale_min': 0.015457858331501484, 'scale_std': 0.006082689855247736, 'total_values': 67456} |  |
| task-sft quantization dynamics telemetry is finite | pass | telemetry_rows=2, last={'compared_modules': 15, 'flip_fraction': 6.101132370167903e-06, 'flipped_values': 2, 'has_previous': True, 'previous_step': 1, 'sampled_code_values': 327808, 'scale_abs_delta_max': 5.587935447692871e-09, 'scale_abs_delta_mean': 1.862645149230957e-09, 'scale_delta_count': 15, 'scale_modules': 15, 'steps_since_previous': 1, 'tracked_modules': 15} |  |
| row task-sft writes metrics | pass | benchmark_results/bitdistill-smoke-contract-2026-05-15/task_sft_row/metrics.json |  |
| row task-sft takes two steps | pass | steps=2 |  |
| row task-sft uses BitLinear and SubLN | pass | bitlinear=15, subln=4 |  |
| row task-sft logits KD is finite | pass | weighted_logit_kd=0.009045824408531189 |  |
| row task-sft attention KD is finite | pass | weighted_attention_kd=11.805890083312988 |  |
| row task-sft Q/K/V attention KD split is finite | pass | q=0.033562831580638885, k=0.041941218078136444, v=0.042554859071969986 |  |
| row task-sft weighted Q/K/V split sums to aggregate | pass | sum=11.805891036987305, aggregate=11.805890083312988 |  |
| row task-sft records paper-style Q/K/V reduction | pass | attention_qkv_reduction=sum |  |
| row task-sft eval accuracy is finite | pass | accuracy=0.5 |  |
| row task-sft writes per-example predictions | pass | predictions=8, eval_examples=8.0 |  |
| row task-sft row-scale ternary export is valid | pass | codes=15, tensor_scales=0, row_scales=15 |  |
| row task-sft activation telemetry is finite | pass | telemetry_rows=2, last={'absmax_max': 7.610891819000244, 'absmax_mean': 3.0551750659942627, 'activation_quantized_modules': 15, 'clipped_fraction': 0.0, 'clipped_values': 0, 'int8_edge_fraction': 0.005336812144212524, 'int8_edge_values': 360, 'module_count': 15, 'negative_edge_fraction': 0.0, 'positive_edge_fraction': 0.005336812144212524, 'scale_count': 465, 'scale_max': 0.0599282830953598, 'scale_mean': 0.02405649796128273, 'scale_min': 0.01609511487185955, 'scale_std': 0.0073115406557917595, 'total_values': 67456} |  |
| row task-sft quantization dynamics telemetry is finite | pass | telemetry_rows=2, last={'compared_modules': 15, 'flip_fraction': 3.0505661850839517e-06, 'flipped_values': 1, 'has_previous': True, 'previous_step': 1, 'sampled_code_values': 327808, 'scale_abs_delta_max': 9.406358003616333e-08, 'scale_abs_delta_mean': 1.9516561527831434e-08, 'scale_delta_count': 2305, 'scale_modules': 15, 'steps_since_previous': 1, 'tracked_modules': 15} |  |
