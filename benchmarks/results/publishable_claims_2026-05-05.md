# Publishable Claims Ledger, 2026-05-05

This ledger separates claims supported by artifacts in this fork from claims
that are not yet supported. It is intended for public review: every positive
claim below is scoped to the cited artifact family.

## Supported Claims

| claim | status | evidence | precise scope |
| --- | --- | --- | --- |
| Blind FP/BF16-to-ternary PTQ is not viable for the tested dense Qwen checkpoints | supported | `experiments/math_viability_test.py`; `benchmark_results/math_viability/summary.json`; `benchmark_results/quality-ptq-qwen15b`; `benchmark_results/lm-eval-qwen15b-full10/qwen15b_naive_ptq.json` | Qwen2.5-0.5B and Qwen2.5-1.5B dense checkpoints tested here |
| QAT/distillation recovers meaningful signal versus naive PTQ | supported | `benchmark_results/quality-9735`; `benchmark_results/quality-qwen15b-klonly-5000`; `benchmark_results/lm-eval-qwen15b-klonly-full10` | recovery is measurable but incomplete; not FP-quality |
| KL-only distillation outperforms the hidden-MSE recipe in these runs | supported | `benchmarks/results/qwen_side_by_side_2026-05-05.md`; `benchmark_results/lm-eval-qwen15b-klonly-full10` | Qwen2.5-1.5B, 5000-step student runs |
| Keeping Qwen's tied `lm_head` dense improves likelihood but not the task verdict | supported | `benchmark_results/quality-qwen15b-klonly-notiehead-5000`; `benchmark_results/lm-eval-qwen15b-klonly-notiehead-full10`; paired-delta artifacts summarized in `benchmarks/results/qwen_retrofit_2026-05-03.md` | Qwen2.5-1.5B dense-head ablation |
| Row-wise ternary scales are the strongest tested PyTorch-quality ablation | supported | `benchmark_results/quality-qwen15b-klonly-row-notiehead-5000`; `benchmark_results/lm-eval-qwen15b-klonly-row-notiehead-full10`; `benchmark_results/evidence_audit/qwen15b_row_notie_5000.md` | Qwen2.5-1.5B KL-only dense-`lm_head`; still below FP |
| Static ternary materialization can preserve the trained ternary checkpoint semantics through GGUF F16/TQ2_0 | supported | `benchmark_results/gguf-qwen15b-klonly-row-notiehead-suite/summary.json`; `benchmarks/build_static_ternary_gguf_bridge.py` | bridge path, not direct packed ternary-state GGUF writing |
| Default row-scale `I2_S` is not valid for row-scale checkpoints | supported | `benchmark_results/gguf-qwen15b-klonly-row-notiehead-suite/summary.json`; `benchmark_results/gguf-qwen15b-klonly-row-notiehead-suite/audit.md` | current default layout stores one tensor scale and loses row-scale magnitudes |
| A local row-scale-aware `I2_S` prototype fixes that specific packed-format failure | supported as prototype | `patches/llama-i2s-row-scale.patch`; `benchmark_results/gguf-qwen15b-row-i2s-heapfix-confirm/summary.json`; `benchmark_results/evidence_audit/qwen15b_row_i2s_heapfix.md` | local replacement layout; not a stable upstream GGUF type |
| Row-scale `I2_S` prefill scales with threads while decode saturates early | supported | `benchmarks/results/i2s_thread_scaling_2026-05-05.md`; `benchmark_results/evidence_audit/qwen15b_row_i2s_thread_scaling.md` | Xeon Silver 4116, portable AVX2 build, patched row-scale `I2_S` artifact |
| Row-scale `I2_S` keeps a memory advantage over FP16 at long context but not over Q4_K_M | supported | `benchmarks/results/gguf_context_scaling_2026-05-05.md`; `benchmark_results/evidence_audit/qwen15b_context_scaling_rss.md` | Qwen2.5-1.5B row-scale dense-`lm_head`, contexts 512/2048/8192/32768 |
| Generic MoE infrastructure exists, but Kimi support is unproven | supported | `benchmarks/results/moe_support_audit_2026-05-05.md`; `benchmarks/results/conversion_support_audit_2026-05-05.md` | generic Qwen2MoE/runtime path exists; no Kimi-specific converter/runtime benchmark |

## Unsupported Or Not-Yet Claims

| claim | status | why not supported yet |
| --- | --- | --- |
| Arbitrary pretrained models can be losslessly retrofitted to 1.58-bit ternary | unsupported | math audit and Qwen PTQ runs show catastrophic information loss |
| The current ternary students match FP Qwen quality | unsupported | best ten-task mean is `0.499459` versus FP `0.644169`; best WikiText/FineWeb PPL is `38.580`/`21.333` versus FP `13.901`/`10.269` |
| Row-scale `I2_S` is production-ready | unsupported | the prototype changes the `I2_S` binary layout instead of introducing a compatibility-safe stable type |
| Native AVX-512 speeds up the row-scale `I2_S` path | unsupported | native `GGML_NATIVE=ON`/`AVX512 = 1` run preserved quality but was slightly slower than portable AVX2 for the tested artifact |
| Direct `ternary_state_dict.pt` to GGUF writing is complete | unsupported | current path uses dense HF materialization through `benchmarks/build_static_ternary_gguf_bridge.py` |
| Qwen TL2 deployment is complete | unsupported | BitNet converter exposes TL2 but lacks Qwen2/Qwen2MoE registration; llama.cpp converter supports Qwen2/Qwen2MoE but lacks TL2 outtype |
| Kimi or other MoE models have been successfully ternary-retrofitted | unsupported | no Kimi-specific converter, router distillation, expert-locality benchmark, or quality run exists in this fork |

## Product Implication

The credible product is not a one-click lossless ternarizer. The credible MVP is
a CPU-first dense-model retrofit pipeline:

1. support a narrow decoder family such as Qwen2 dense,
2. train/distill under exact ternary forward constraints,
3. preserve sensitive components such as tied output heads when needed,
4. export static ternary state plus a reproducible GGUF bridge,
5. benchmark against FP, Q8_0, Q4_K_M, naive PTQ, and QAT variants, and
6. publish the quality, throughput, RSS, and context-scaling tradeoff card.

The publishable angle is a negative result plus an engineering recovery path:
post-training ternarization destroys pretrained dense checkpoints, but
distillation with row-wise ternary scales plus packed CPU kernels can recover
partial quality and real CPU-side speed/memory benefits. The result should be
presented as scoped evidence, not as a solved arbitrary-model deployment method.
