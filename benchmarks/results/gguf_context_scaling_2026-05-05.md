# GGUF RSS Context Scaling, 2026-05-05

Hardware: Intel Xeon Silver 4116, portable AVX2 llama.cpp build, no BLAS.

Command:

```bash
python benchmarks/run_gguf_memory_probe.py \
  --models-json benchmarks/gguf_qwen15b_row_i2s_prototype_manifest.json \
  --out-dir benchmark_results/gguf-rss-qwen15b-context-scaling-2026-05-05 \
  --llama-bin-dir build-portable-avx2/bin \
  --threads 12 \
  --ctx-sizes 512 2048 8192 32768 \
  --tokens 1
```

The probe runs `llama-cli` under `/usr/bin/time -v`, CPU-only, with one
generated token. This measures process peak resident set size after loading the
model and allocating the requested context. It is not a throughput benchmark.

| artifact | ctx 512 RSS GiB | ctx 2048 RSS GiB | ctx 8192 RSS GiB | ctx 32768 RSS GiB |
| --- | ---: | ---: | ---: | ---: |
| FP F16 | 2.948 | 2.989 | 3.153 | 3.812 |
| FP Q8_0 | 1.601 | 1.642 | 1.806 | 2.465 |
| FP Q4_K_M | 0.985 | 1.027 | 1.191 | 1.850 |
| row-scale static ternary F16 | 3.383 | 3.424 | 3.588 | 4.247 |
| row-scale static ternary TQ2_0 | 1.257 | 1.298 | 1.462 | 2.121 |
| row-scale static ternary I2_S prototype | 1.250 | 1.291 | 1.455 | 2.114 |

Mechanical audit:
`benchmark_results/evidence_audit/qwen15b_context_scaling_rss.md`

## Interpretation

The row-scale `I2_S` prototype stays within the same memory band as row-scale
`TQ2_0` across all tested context sizes. At `-c 32768`, peak RSS is `2.114 GiB`
for row-scale `I2_S`, `2.121 GiB` for row-scale `TQ2_0`, `3.812 GiB` for FP
F16, and `1.850 GiB` for FP `Q4_K_M`.

This supports a specific product claim: the row-scale packed ternary bridge can
reduce memory substantially versus FP16 even at long context, but it does not
beat an aggressively quantized dense `Q4_K_M` baseline on RSS in the current
artifact. The likely reasons are the dense tied `lm_head`, per-row scale
metadata, and unchanged KV-cache/runtime overheads.
