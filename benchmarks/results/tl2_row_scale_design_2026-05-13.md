# TL2 Row-Scale Design Audit, 2026-05-13

This audit measures whether the current one-scale TL2 metadata can preserve row-scale ternary checkpoints, and how much metadata a row/group-scale TL2 variant would need. For isotropic activations, the reported relative output RMS error equals the relative Frobenius error of the effective weight matrix.

| checkpoint | tensors | row-scale tensors | current TL2 err | best one-scale err | group2 fp16 err | group32 fp16 err | group128 fp16 err | row fp16 err | row fp16 scale MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen15b_tensor_scale | 197 | 0 | 0.000000 | 0.000000 | 0.000175 | 0.000175 | 0.000175 | 0.000175 | 1.520 |
| qwen15b_row_scale | 196 | 196 | 1.904230 | 0.161173 | 0.098692 | 0.142844 | 0.154392 | 0.000197 | 1.230 |

## qwen15b_tensor_scale Strategy Detail

| strategy | rel output RMS err | scales | fp16 MiB | fp32 MiB |
| --- | --- | --- | --- | --- |
| current_tl2_tensor_max_fp32 | 0.000000 | 197 | 0.000 | 0.001 |
| tensor_l2_optimal_fp32 | 0.000000 | 197 | 0.000 | 0.001 |
| group2_l2_optimal_fp16 | 0.000175 | 398528 | 0.760 | 1.520 |
| group4_l2_optimal_fp16 | 0.000175 | 199264 | 0.380 | 0.760 |
| group8_l2_optimal_fp16 | 0.000175 | 99632 | 0.190 | 0.380 |
| group16_l2_optimal_fp16 | 0.000175 | 49816 | 0.095 | 0.190 |
| group32_l2_optimal_fp16 | 0.000175 | 24908 | 0.048 | 0.095 |
| group64_l2_optimal_fp16 | 0.000175 | 12454 | 0.024 | 0.048 |
| group128_l2_optimal_fp16 | 0.000175 | 6227 | 0.012 | 0.024 |
| row_exact_fp16 | 0.000175 | 797056 | 1.520 | 3.041 |
| row_exact_fp32 | 0.000000 | 797056 | 1.520 | 3.041 |

| tensor | shape | scale mode | current TL2 err | best one-scale err | max/median scale |
| --- | --- | --- | --- | --- | --- |
| lm_head | 151936x1536 | scalar | 0.000000 | 0.000000 | 1.000 |
| model.layers.0.mlp.down_proj | 1536x8960 | scalar | 0.000000 | 0.000000 | 1.000 |
| model.layers.0.mlp.gate_proj | 8960x1536 | scalar | 0.000000 | 0.000000 | 1.000 |
| model.layers.0.mlp.up_proj | 8960x1536 | scalar | 0.000000 | 0.000000 | 1.000 |
| model.layers.0.self_attn.k_proj | 256x1536 | scalar | 0.000000 | 0.000000 | 1.000 |

## qwen15b_row_scale Strategy Detail

| strategy | rel output RMS err | scales | fp16 MiB | fp32 MiB |
| --- | --- | --- | --- | --- |
| current_tl2_tensor_max_fp32 | 1.904230 | 196 | 0.000 | 0.001 |
| tensor_l2_optimal_fp32 | 0.161173 | 196 | 0.000 | 0.001 |
| group2_l2_optimal_fp16 | 0.098692 | 322560 | 0.615 | 1.230 |
| group4_l2_optimal_fp16 | 0.120769 | 161280 | 0.308 | 0.615 |
| group8_l2_optimal_fp16 | 0.130677 | 80640 | 0.154 | 0.308 |
| group16_l2_optimal_fp16 | 0.136894 | 40320 | 0.077 | 0.154 |
| group32_l2_optimal_fp16 | 0.142844 | 20160 | 0.038 | 0.077 |
| group64_l2_optimal_fp16 | 0.153960 | 10080 | 0.019 | 0.038 |
| group128_l2_optimal_fp16 | 0.154392 | 5040 | 0.010 | 0.019 |
| row_exact_fp16 | 0.000197 | 645120 | 1.230 | 2.461 |
| row_exact_fp32 | 0.000000 | 645120 | 1.230 | 2.461 |

| tensor | shape | scale mode | current TL2 err | best one-scale err | max/median scale |
| --- | --- | --- | --- | --- | --- |
| model.layers.26.mlp.up_proj | 8960x1536 | row | 8.196455 | 0.142527 | 9.192 |
| model.layers.26.mlp.gate_proj | 8960x1536 | row | 7.439233 | 0.154375 | 8.786 |
| model.layers.1.mlp.up_proj | 8960x1536 | row | 4.938749 | 0.180063 | 6.204 |
| model.layers.15.self_attn.q_proj | 1536x1536 | row | 3.787746 | 0.322505 | 4.911 |
| model.layers.0.self_attn.q_proj | 1536x1536 | row | 3.519773 | 0.492442 | 5.365 |

## Interpretation

A one-scale TL2 representation is mathematically exact for tensor-scale checkpoints, but not for row-scale checkpoints. For the strong row-scale Qwen checkpoint, the fix is not more benchmarking of the existing TL2 format; the format/runtime must carry row or row-group scale metadata and the generated TL2 kernels must index those scales. The scale-memory cost of exact fp16 row scales is small relative to the model file, so the blocker is runtime/kernel support rather than storage.
