# Publishable Claims Ledger, 2026-05-05

This ledger separates claims supported by artifacts in this fork from claims
that are not yet supported. It is intended for public review: every positive
claim below is scoped to the cited artifact family.

The compact artifact manifest is
`benchmarks/results/evidence_manifest_2026-05-13.md`; it records hashes and
parsed headline metrics for cited artifacts with zero missing entries.

## Supported Claims

| claim | status | evidence | precise scope |
| --- | --- | --- | --- |
| Blind FP/BF16-to-ternary PTQ is not viable for the tested dense Qwen checkpoints | supported | `experiments/math_viability_test.py`; `benchmark_results/math_viability/summary.json`; `benchmark_results/quality-ptq-qwen15b`; `benchmark_results/lm-eval-qwen15b-full10/qwen15b_naive_ptq.json` | Qwen2.5-0.5B and Qwen2.5-1.5B dense checkpoints tested here |
| QAT/distillation recovers meaningful signal versus naive PTQ | supported | `benchmark_results/quality-9735`; `benchmark_results/quality-qwen15b-klonly-5000`; `benchmark_results/lm-eval-qwen15b-klonly-full10` | recovery is measurable but incomplete; not FP-quality |
| KL-only distillation outperforms the hidden-MSE recipe in these runs | supported | `benchmarks/results/qwen_side_by_side_2026-05-05.md`; `benchmark_results/lm-eval-qwen15b-klonly-full10` | Qwen2.5-1.5B, 5000-step student runs |
| Keeping Qwen's tied `lm_head` dense improves likelihood but not the task verdict | supported | `benchmark_results/quality-qwen15b-klonly-notiehead-5000`; `benchmark_results/lm-eval-qwen15b-klonly-notiehead-full10`; paired-delta artifacts summarized in `benchmarks/results/qwen_retrofit_2026-05-03.md` | Qwen2.5-1.5B dense-head ablation |
| Row-wise ternary scales are the strongest tested PyTorch-quality ablation | supported | `benchmark_results/quality-qwen15b-klonly-row-notiehead-5000`; `benchmark_results/lm-eval-qwen15b-klonly-row-notiehead-full10`; `benchmark_results/evidence_audit/qwen15b_row_notie_5000.md` | Qwen2.5-1.5B KL-only dense-`lm_head`; still below FP |
| Static ternary materialization can preserve the trained ternary checkpoint semantics through GGUF F16/TQ2_0 | supported | `benchmark_results/gguf-qwen15b-klonly-row-notiehead-suite/summary.json`; `benchmarks/build_static_ternary_gguf_bridge.py` | bridge path, not direct packed ternary-state GGUF writing |
| Direct dense GGUF export from `ternary_state_dict.pt` is possible without writing an intermediate HF checkpoint | supported | `benchmarks/convert_static_ternary_to_gguf.py`; `benchmarks/results/direct_static_ternary_gguf_2026-05-13.md`; `benchmark_results/direct-gguf-qwen05b-klonly-notie-2026-05-13/summary.json` | dense F16 GGUF bridge only; not direct packed `I2_S` |
| Direct scalar `I2_S` GGUF export is runnable but not quality-preserving for the tested Qwen0.5B scalar checkpoint | supported | `benchmarks/convert_static_ternary_to_i2s_gguf.py`; `benchmarks/results/direct_i2s_scalar_gguf_2026-05-13.md`; `benchmark_results/direct-i2s-qwen05b-klonly-2026-05-13/summary.json` | scalar-scale only; Qwen2.5-0.5B PPL is NaN despite better speed/file size |
| Direct row-scale `I2_S` GGUF export is mechanically writable but not quality-valid in the default runtime | supported | `benchmarks/convert_static_ternary_to_i2s_gguf.py`; `benchmarks/results/direct_row_i2s_qwen05b_2026-05-13.md`; `benchmark_results/direct-row-i2s-qwen05b-portable-2026-05-13/summary.json` | `--row-scale-prototype` writes per-row scales, but Qwen2.5-0.5B row PPL is `59401.5449`; production use still needs a stable qtype/layout and matching runtime |
| A candidate stable `I2_SR` writer/runtime path exists | supported as engineering plumbing | `patches/llama-i2sr-row-scale-qtype.patch`; `benchmarks/convert_static_ternary_to_i2s_gguf.py`; `benchmarks/results/i2sr_candidate_patch_2026-05-13.md`; `benchmark_results/i2sr-writer-smoke-2026-05-13/summary.json` | patch applies and builds; writer emits qtype `40`/ftype `41`; not yet quality- or throughput-benchmarked as an applied runtime artifact |
| Direct packed row-scale GGUF export is not product-complete | supported | `benchmarks/results/direct_packed_gguf_support_2026-05-13.md`; `benchmark_results/direct_packed_gguf_support_2026-05-13.json`; `benchmarks/results/direct_i2s_scalar_gguf_2026-05-13.md`; `benchmarks/results/direct_row_i2s_qwen05b_2026-05-13.md` | scalar direct `I2_S` exists, and row prototype exists for debugging, but quality-preserving row-scale deployment needs a stable type/version and matching runtime |
| Default row-scale `I2_S` is not valid for row-scale checkpoints | supported | `benchmark_results/gguf-qwen15b-klonly-row-notiehead-suite/summary.json`; `benchmark_results/gguf-qwen15b-klonly-row-notiehead-suite/audit.md` | current default layout stores one tensor scale and loses row-scale magnitudes |
| A local row-scale-aware `I2_S` prototype fixes that specific packed-format failure | supported as prototype | `patches/llama-i2s-row-scale.patch`; `benchmark_results/gguf-qwen15b-row-i2s-heapfix-confirm/summary.json`; `benchmark_results/evidence_audit/qwen15b_row_i2s_heapfix.md`; `benchmarks/results/i2s_row_scale_format_audit_2026-05-13.md`; `benchmarks/results/row_scale_qtype_productization_gate_2026-05-13.md` | local replacement layout; not a stable upstream GGUF type; productization gate fails until a separate row-scale qtype/file type/writer/benchmark exists |
| Row-scale `I2_S` prefill scales with threads while decode saturates early | supported | `benchmarks/results/i2s_thread_scaling_2026-05-05.md`; `benchmark_results/evidence_audit/qwen15b_row_i2s_thread_scaling.md` | Xeon Silver 4116, portable AVX2 build, patched row-scale `I2_S` artifact |
| Row-scale `I2_S` keeps a memory advantage over FP16 at long context but not over Q4_K_M | supported | `benchmarks/results/gguf_context_scaling_2026-05-05.md`; `benchmark_results/evidence_audit/qwen15b_context_scaling_rss.md` | Qwen2.5-1.5B row-scale dense-`lm_head`, contexts 512/2048/8192/32768 |
| Dense Qwen TL2 export is possible only with model-specific code generation, and the tested 0.5B TL2 checkpoint fails quality | supported | `benchmarks/results/tl2_shape_support_audit_2026-05-05.md`; `benchmarks/results/qwen05b_tl2_probe_2026-05-05.md`; `benchmark_results/evidence_audit/qwen05b_tl2_probe.md` | Qwen2.5-0.5B dense checkpoint; generated TL2 shapes plus a `BITNET_X86_TL2=ON` AVX-512 build; PPL is NaN |
| Current TL2 scale semantics are incompatible with the best row-scale Qwen1.5B checkpoint | supported | `benchmarks/results/tl2_scale_semantics_2026-05-05.md`; `benchmark_results/tl2_scale_semantics_2026-05-05.json` | replacing row scales with one TL2 tensor scale gives total relative Frobenius/output-RMS error `1.904230`; scalar-scale control is `0.0` |
| Generic MoE infrastructure exists, but Kimi support is unproven | supported | `benchmarks/results/moe_support_audit_2026-05-05.md`; `benchmarks/results/conversion_support_audit_2026-05-05.md` | generic Qwen2MoE/runtime path exists; no Kimi-specific converter/runtime benchmark |

## Unsupported Or Not-Yet Claims

| claim | status | why not supported yet |
| --- | --- | --- |
| Arbitrary pretrained models can be losslessly retrofitted to 1.58-bit ternary | unsupported | math audit and Qwen PTQ runs show catastrophic information loss |
| The current ternary students match FP Qwen quality | unsupported | best ten-task mean is `0.499459` versus FP `0.644169`; best WikiText/FineWeb PPL is `38.580`/`21.333` versus FP `13.901`/`10.269` |
| Row-scale `I2_S`/`I2_SR` is production-ready | unsupported | the quality-preserving prototype changes the `I2_S` binary layout; the cleaner `I2_SR` path exists only as an apply-check/build-checked candidate patch and writer smoke, not a full benchmarked runtime artifact |
| Native AVX-512 speeds up the row-scale `I2_S` path | unsupported | native `GGML_NATIVE=ON`/`AVX512 = 1` run preserved quality but was slightly slower than portable AVX2 for the tested artifact |
| Direct packed `ternary_state_dict.pt` to CPU-native GGUF writing is complete | unsupported | direct dense F16 GGUF, direct scalar `I2_S`, and direct row-prototype `I2_S` now work mechanically, but scalar Qwen0.5B is NaN, direct row Qwen0.5B is catastrophic, and quality-preserving row-scale `I2_S` still needs a stable type and runtime |
| Qwen TL2 deployment is complete | unsupported | Qwen2 dense TL2 export now works only after model-specific codegen and a matching TL2 build; tested Qwen2.5-0.5B TL2 has NaN PPL; Qwen2MoE/Kimi and the strong Qwen2.5-1.5B row-scale TL2 path remain unvalidated |
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
