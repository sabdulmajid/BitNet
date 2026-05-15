# Tiny Qwen2MoE Expert Scaling, 2026-05-15

Overall status: `pass`.

This is a synthetic CPU runtime scaling probe for random tiny Qwen2MoE models. It does not measure model quality, Kimi compatibility, ternary MoE correctness, or router accuracy.

| variant | pass | experts | top-k | GGUF MiB | params M | CPU buffer MiB | prompt tok/s | decode tok/s | RSS MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| experts2_top1 | true | 2 | 1 | 42.808 | 19.480 | 37.160 | 9765.620 | 2202.320 | 105.055 |
| experts2_top2 | true | 2 | 2 | 42.808 | 19.480 | 37.160 | 9615.380 | 2186.590 | 105.125 |
| experts4_top1 | true | 4 | 1 | 42.878 | 19.520 | 37.230 | 9416.200 | 2192.980 | 105.309 |
| experts4_top2 | true | 4 | 2 | 42.878 | 19.520 | 37.230 | 9652.510 | 2182.770 | 105.176 |

## Interpretation

A passing row means the converter/runtime can execute that routed graph shape on CPU. Publishable/product claims still require a trained MoE checkpoint, quality evaluation, and a Kimi-specific tensor mapping audit.
