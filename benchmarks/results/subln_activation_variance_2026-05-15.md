# SubLN Activation Variance Audit, 2026-05-15

Local SubLN insertion is not an identity-preserving conversion. It normalizes the tensors entering o_proj/down_proj to unit RMS and can materially perturb logits before any continued pretraining. Therefore a short SubLN-only downstream run is not a decisive test of the paper's Stage-1 recipe.



- Model: `checkpoints/qwen2.5-0.5b-fineweb-edu-12/step-1000`.

- Examples: `4`; tokens: `104`; max length: `64`.

- SubLN modules inserted: `48`.

- Last-token logit relative RMS drift after untrained SubLN insertion: `0.768044`.

- Last-token logit cosine after untrained SubLN insertion: `0.698252`.

- Last-token top-1 agreement: `0.000000`.



## Family Summary

| family | modules | base input RMS | SubLN input RMS | SubLN output RMS | base input absmax | SubLN output absmax |
| --- | --- | --- | --- | --- | --- | --- |
| .self_attn.o_proj | 24 | 0.388311 | 0.468128 | 0.999680 | 4.193092 | 6.833793 |
| .mlp.down_proj | 24 | 0.380883 | 0.159770 | 0.999304 | 172.548341 | 40.855563 |



## First Modules

| module | base input RMS | SubLN input RMS | SubLN output RMS | base input absmax | SubLN output absmax |
| --- | --- | --- | --- | --- | --- |
| model.layers.0.mlp.down_proj | 0.123675 | 0.110310 | 0.999327 | 7.439883 | 39.418404 |
| model.layers.0.self_attn.o_proj | 0.028069 | 0.028069 | 0.993270 | 0.324123 | 9.685069 |
| model.layers.1.mlp.down_proj | 0.155773 | 0.137322 | 0.999702 | 9.975130 | 54.715130 |
| model.layers.1.self_attn.o_proj | 0.264994 | 0.253250 | 0.999919 | 3.060991 | 10.124833 |
| model.layers.10.mlp.down_proj | 0.145638 | 0.048795 | 0.997680 | 7.787725 | 47.696545 |
| model.layers.10.self_attn.o_proj | 0.268651 | 0.267299 | 0.999928 | 2.672425 | 7.242620 |
| model.layers.11.mlp.down_proj | 0.151380 | 0.057847 | 0.998357 | 7.537001 | 26.303518 |
| model.layers.11.self_attn.o_proj | 0.224997 | 0.306330 | 0.999945 | 2.972131 | 6.224750 |
| model.layers.12.mlp.down_proj | 0.132660 | 0.056344 | 0.998209 | 4.533442 | 31.113981 |
| model.layers.12.self_attn.o_proj | 0.243901 | 0.338884 | 0.999955 | 2.677324 | 7.503242 |
| model.layers.13.mlp.down_proj | 0.154311 | 0.059189 | 0.998391 | 10.515611 | 44.286427 |
| model.layers.13.self_attn.o_proj | 0.224287 | 0.302207 | 0.999943 | 2.001734 | 4.119547 |

