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
| QAT/distillation recovers meaningful signal versus naive PTQ | supported | `benchmark_results/quality-9735`; `benchmark_results/quality-qwen15b-klonly-5000`; `benchmark_results/lm-eval-qwen15b-klonly-full10`; `benchmarks/results/paired_row_densehead_minus_ptq_2026-05-13.md`; `benchmarks/results/paired_row_densehead_minus_fp_2026-05-13.md` | strongest row-scale QAT is `+0.150788` macro mean versus naive PTQ with paired 95% CI `[+0.053427, +0.248149]`, but remains `-0.144710` versus FP with paired 95% CI `[-0.185756, -0.103664]` |
| KL-only distillation outperforms the hidden-MSE recipe in these runs | supported | `benchmarks/results/qwen_side_by_side_2026-05-05.md`; `benchmark_results/lm-eval-qwen15b-klonly-full10` | Qwen2.5-1.5B, 5000-step student runs |
| Keeping Qwen's tied `lm_head` dense improves likelihood but not the task verdict | supported | `benchmark_results/quality-qwen15b-klonly-notiehead-5000`; `benchmark_results/lm-eval-qwen15b-klonly-notiehead-full10`; paired-delta artifacts summarized in `benchmarks/results/qwen_retrofit_2026-05-03.md` | Qwen2.5-1.5B dense-head ablation |
| Row-wise ternary scales are the strongest tested PyTorch-quality ablation | supported | `benchmark_results/quality-qwen15b-klonly-row-notiehead-5000`; `benchmark_results/lm-eval-qwen15b-klonly-row-notiehead-full10`; `benchmark_results/evidence_audit/qwen15b_row_notie_5000.md` | Qwen2.5-1.5B KL-only dense-`lm_head`; still below FP |
| Static ternary materialization can preserve the trained ternary checkpoint semantics through GGUF F16/TQ2_0 | supported | `benchmark_results/gguf-qwen15b-klonly-row-notiehead-suite/summary.json`; `benchmarks/build_static_ternary_gguf_bridge.py` | bridge path, not direct packed ternary-state GGUF writing |
| Direct dense GGUF export from `ternary_state_dict.pt` is possible without writing an intermediate HF checkpoint | supported | `benchmarks/convert_static_ternary_to_gguf.py`; `benchmarks/results/direct_static_ternary_gguf_2026-05-13.md`; `benchmark_results/direct-gguf-qwen05b-klonly-notie-2026-05-13/summary.json` | dense F16 GGUF bridge only; not direct packed `I2_S` |
| Direct scalar `I2_S` GGUF export is runnable but not product-quality for the tested Qwen0.5B scalar checkpoint | supported | `benchmarks/convert_static_ternary_to_i2s_gguf.py`; `benchmarks/results/direct_i2s_scalar_gguf_2026-05-13.md`; `benchmark_results/direct-i2s-qwen05b-klonly-x86act-2026-05-13/summary.json` | scalar-scale only; fixed x86 ACT pack reaches finite PPL `423.4528`, but FP16 is `18.0986` |
| Direct row-scale `I2_S` GGUF export is mechanically writable but not quality-valid in the default runtime | supported | `benchmarks/convert_static_ternary_to_i2s_gguf.py`; `benchmarks/results/direct_row_i2s_qwen05b_2026-05-13.md`; `benchmark_results/direct-row-i2s-qwen05b-portable-2026-05-13/summary.json` | `--row-scale-prototype` writes per-row scales, but Qwen2.5-0.5B row PPL is `59401.5449`; production use still needs a stable qtype/layout and matching runtime |
| A candidate stable `I2_SR` writer/runtime path preserves row-scale quality after fixing x86 packing | supported as downstream engineering candidate | `patches/llama-i2sr-row-scale-qtype.patch`; `benchmarks/convert_static_ternary_to_i2s_gguf.py`; `benchmarks/verify_i2s_packing_layout.py`; `benchmarks/results/i2sr_candidate_patch_2026-05-13.md`; `benchmarks/results/i2sr_qwen15b_candidate_2026-05-13.md`; `benchmarks/results/i2sr_x86act_fix_2026-05-13.md`; `benchmarks/results/i2s_packing_layout_verify_2026-05-13.md`; `benchmarks/results/i2sr_rss_2026-05-13.md`; `benchmark_results/i2sr-row-scale-qwen15b-x86act-suite-2026-05-13/summary.json` | first row-group direct artifact failed at PPL `20,074,699.9423`; fixed x86 ACT pack reaches PPL `38.8477`, `211.67`/`19.07` tok/s, RSS `1.250`/`2.114 GiB` at ctx `512`/`32768`, matching the row-scale `I2_S` prototype quality/memory class; byte verifier passes `5/5` tensors |
| Direct packed row-scale GGUF export is quality-valid only through the downstream `I2_SR` candidate path | supported | `benchmarks/results/direct_packed_gguf_support_2026-05-13.md`; `benchmark_results/direct_packed_gguf_support_2026-05-13.json`; `benchmarks/results/i2sr_x86act_fix_2026-05-13.md`; `benchmark_results/i2sr-row-scale-qwen15b-x86act-suite-2026-05-13/summary.json` | fixed direct `I2_SR` preserves 1.5B row-scale quality, but the qtype is still a patch, not active upstream/default runtime support |
| Default row-scale `I2_S` is not valid for row-scale checkpoints | supported | `benchmark_results/gguf-qwen15b-klonly-row-notiehead-suite/summary.json`; `benchmark_results/gguf-qwen15b-klonly-row-notiehead-suite/audit.md` | current default layout stores one tensor scale and loses row-scale magnitudes |
| A local row-scale-aware `I2_S` prototype fixes that specific packed-format failure | supported as prototype | `patches/llama-i2s-row-scale.patch`; `benchmark_results/gguf-qwen15b-row-i2s-heapfix-confirm/summary.json`; `benchmark_results/evidence_audit/qwen15b_row_i2s_heapfix.md`; `benchmarks/results/i2s_row_scale_format_audit_2026-05-13.md`; `benchmarks/results/row_scale_qtype_productization_gate_2026-05-13.md` | local replacement layout; not a stable upstream GGUF type; productization gate fails until a separate row-scale qtype/file type/writer/benchmark exists |
| Row-scale `I2_S` prefill scales with threads while decode saturates early | supported | `benchmarks/results/i2s_thread_scaling_2026-05-05.md`; `benchmark_results/evidence_audit/qwen15b_row_i2s_thread_scaling.md` | Xeon Silver 4116, portable AVX2 build, patched row-scale `I2_S` artifact |
| Row-scale ternary GGUF keeps a memory advantage over FP16 at long context but not over Q4_K_M | supported | `benchmarks/results/gguf_context_scaling_2026-05-05.md`; `benchmarks/results/i2sr_rss_2026-05-13.md`; `benchmark_results/evidence_audit/qwen15b_context_scaling_rss.md` | Qwen2.5-1.5B row-scale dense-`lm_head`; `I2_S`/`TQ2_0`/fixed `I2_SR` contexts 512/2048/8192/32768 |
| Dense Qwen TL2 export is possible only with model-specific code generation, and the tested 0.5B TL2 checkpoint fails quality | supported | `benchmarks/results/tl2_shape_support_audit_2026-05-05.md`; `benchmarks/results/qwen05b_tl2_probe_2026-05-05.md`; `benchmark_results/evidence_audit/qwen05b_tl2_probe.md` | Qwen2.5-0.5B dense checkpoint; generated TL2 shapes plus a `BITNET_X86_TL2=ON` AVX-512 build; PPL is NaN |
| Current TL2 scale semantics are incompatible with the best row-scale Qwen1.5B checkpoint | supported | `benchmarks/results/tl2_scale_semantics_2026-05-05.md`; `benchmark_results/tl2_scale_semantics_2026-05-05.json` | replacing row scales with one TL2 tensor scale gives total relative Frobenius/output-RMS error `1.904230`; scalar-scale control is `0.0` |
| Generic MoE infrastructure exists, but Kimi support is unproven | supported | `benchmarks/results/moe_support_audit_2026-05-05.md`; `benchmarks/results/conversion_support_audit_2026-05-05.md` | generic Qwen2MoE/runtime path exists; no Kimi-specific converter/runtime benchmark |

## Unsupported Or Not-Yet Claims

| claim | status | why not supported yet |
| --- | --- | --- |
| Arbitrary pretrained models can be losslessly retrofitted to 1.58-bit ternary | unsupported | math audit and Qwen PTQ runs show catastrophic information loss |
| The current ternary students match FP Qwen quality | unsupported | best ten-task mean is `0.499459` versus FP `0.644169`; best WikiText/FineWeb PPL is `38.580`/`21.333` versus FP `13.901`/`10.269` |
| Row-scale `I2_S`/`I2_SR` is production-ready | unsupported | the quality-preserving `I2_S` prototype changes the `I2_S` binary layout; the cleaner fixed `I2_SR` path preserves quality but still requires applying a downstream runtime patch |
| Native AVX-512 speeds up the row-scale `I2_S` path | unsupported | native `GGML_NATIVE=ON`/`AVX512 = 1` run preserved quality but was slightly slower than portable AVX2 for the tested artifact |
| Direct packed `ternary_state_dict.pt` to CPU-native GGUF writing is complete | unsupported | fixed direct `I2_SR` now preserves row-scale 1.5B quality and has byte-layout regression coverage, but scalar direct `I2_S` and 0.5B controls include quality failures, and the qtype is not active by default |
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
