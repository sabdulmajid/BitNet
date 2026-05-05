# GGUF RSS Probe, 2026-05-05

Hardware: Intel Xeon Silver 4116, portable AVX2 llama.cpp build, no BLAS.

Command:

```bash
python benchmarks/run_gguf_memory_probe.py \
  --models-json benchmarks/gguf_qwen15b_row_i2s_prototype_manifest.json \
  --out-dir benchmark_results/gguf-rss-qwen15b-row-i2s-fixed-2026-05-05 \
  --llama-bin-dir build-portable-avx2/bin \
  --threads 12 \
  --ctx-size 512 \
  --tokens 1
```

The probe runs `llama-cli` under `/usr/bin/time -v`, with `-c 512`, 12 threads,
CPU-only execution, and one generated token so the model is loaded and used.
The reported metric is peak resident set size from `/usr/bin/time`.

| artifact | file MiB | max RSS GiB |
| --- | ---: | ---: |
| FP F16 | 2,950.4 | 2.948 |
| FP Q8_0 | 1,570.3 | 1.601 |
| FP Q4_K_M | 940.4 | 0.985 |
| row-scale static ternary F16 | 3,395.5 | 3.383 |
| row-scale static ternary TQ2_0 | 1,218.6 | 1.257 |
| row-scale static ternary I2_S prototype | 1,211.3 | 1.250 |

Mechanical audit:
`benchmark_results/evidence_audit/qwen15b_row_i2s_rss.md`

The matching context-scaling probe is documented in
`benchmarks/results/gguf_context_scaling_2026-05-05.md`.

## Interpretation

The corrected row-scale `I2_S` artifact has nearly the same RSS as row-scale
`TQ2_0` at this context length and is much smaller than F16: `1.250 GiB` versus
`2.948 GiB` for FP F16. It is larger than the FP `Q4_K_M` control
(`0.985 GiB`) because the current row-scale checkpoint keeps Qwen's tied
`lm_head` dense and the `I2_S` prototype stores per-row scales in a replacement
layout rather than a final compact GGUF type.
