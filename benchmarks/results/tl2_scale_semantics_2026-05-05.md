# TL2 Scale Semantics Audit, 2026-05-05

This audit measures the error introduced if a ternary checkpoint that was trained with row-wise scales is exported through the current TL1/TL2 single-scale convention. For random isotropic activations, the expected relative output RMS error equals the relative Frobenius error of the effective weight matrix, so this is a direct linear-layer error proxy.

| checkpoint | tensors | scalar-scale tensors | row-scale tensors | total rel Fro error | median tensor rel error | p95 tensor rel error | max tensor rel error | nonzero frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen15b_tensor_scale | 196 | 196 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.6714 |
| qwen15b_row_scale | 196 | 0 | 196 | 1.904230 | 1.068644 | 2.631890 | 8.196455 | 0.6790 |

## qwen15b_tensor_scale Largest Single-Scale Errors

| tensor | shape | rel error | global/median scale | scale CV | nonzero frac |
| --- | --- | --- | --- | --- | --- |
| model.layers.0.mlp.down_proj | 1536x8960 | 0.000000 | 1.000 | 0.000 | 0.6823 |
| model.layers.0.mlp.gate_proj | 8960x1536 | 0.000000 | 1.000 | 0.000 | 0.6770 |
| model.layers.0.mlp.up_proj | 8960x1536 | 0.000000 | 1.000 | 0.000 | 0.6790 |
| model.layers.0.self_attn.k_proj | 256x1536 | 0.000000 | 1.000 | 0.000 | 0.6572 |
| model.layers.0.self_attn.o_proj | 1536x1536 | 0.000000 | 1.000 | 0.000 | 0.6539 |

## qwen15b_row_scale Largest Single-Scale Errors

| tensor | shape | rel error | global/median scale | scale CV | nonzero frac |
| --- | --- | --- | --- | --- | --- |
| model.layers.26.mlp.up_proj | 8960x1536 | 8.196455 | 9.192 | 0.139 | 0.6807 |
| model.layers.26.mlp.gate_proj | 8960x1536 | 7.439233 | 8.786 | 0.154 | 0.6814 |
| model.layers.1.mlp.up_proj | 8960x1536 | 4.938749 | 6.204 | 0.183 | 0.6838 |
| model.layers.15.self_attn.q_proj | 1536x1536 | 3.787746 | 4.911 | 0.344 | 0.6814 |
| model.layers.0.self_attn.q_proj | 1536x1536 | 3.519773 | 5.365 | 0.562 | 0.6808 |

## Verdict

A scalar-scale checkpoint should show near-zero error because TL2's one-scale assumption matches its scale semantics. A row-scale checkpoint with a large relative error is not safe to export through the current TL2 scale model; it needs row-scale metadata/runtime support or a retraining/export recipe that uses tensor-scale semantics.
