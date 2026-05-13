# Active Goal Completion Audit, 2026-05-05

This audit checks the active user goal against concrete artifacts in this fork.
It is not a success declaration. The goal is not complete because several
requirements are still partial or unproven.

## Success Criteria

1. Repair and re-export the Qwen2.5-1.5B FSDP ternary checkpoint with the
   expected ternary key count.
2. Run fixed prompt and quality evaluations for repaired Qwen2.5-1.5B and
   Qwen2.5-0.5B ternary checkpoints.
3. Add perplexity and lm-eval style downstream benchmarks.
4. Compare against FP, naive PTQ, Q4_K_M/Q8_0, hidden-MSE QAT, KL-only QAT,
   row-scale, and tensor-scale baselines.
5. Convert repaired checkpoints into GGUF/TL2/I2_S and run real CPU inference.
6. Measure CPU throughput, prompt throughput, RSS, model size, and quality loss
   on the Xeon.
7. Produce a side-by-side comparison and an honest verdict on novelty and
   publishability.

## Prompt-To-Artifact Checklist

| requirement | status | evidence | remaining gap |
| --- | --- | --- | --- |
| FSDP export bug fixed for Qwen2.5-1.5B step-5000 | complete | `checkpoints/qwen2.5-1.5b-fineweb-edu/step-5000/ternary_state_dict.pt`; `benchmark_results/evidence_audit/latest_nonrow.md` | none for this checkpoint |
| Re-export repaired 1.5B from checkpoint state | complete | repaired step-5000 directory contains ternary state, `model.safetensors`, config, tokenizer files | none |
| Fixed prompt suites for repaired 1.5B and 0.5B | complete as sanity check | `benchmark_results/generation/qwen15b_step5000_core_cpu_16tok.jsonl`; `benchmark_results/generation/qwen05b_step1000_core_cpu.jsonl` | prompt suites are sanity checks, not quality proof |
| WikiText and FineWeb heldout perplexity | complete for cited dense-Qwen families | `benchmark_results/quality-*/*.json`; summarized in `benchmarks/results/qwen_side_by_side_2026-05-05.md` | broader corpora still optional future work |
| HellaSwag/PIQA/ARC and broader lm-eval | complete for current ten selected tasks | `benchmark_results/lm-eval-qwen15b-full10`; `benchmark_results/lm-eval-qwen15b-klonly-full10`; `benchmark_results/lm-eval-qwen15b-klonly-row-notiehead-full10` | not a full leaderboard suite |
| FP baseline | complete | FP PPL, ten-task lm-eval, GGUF F16/Q8/Q4 rows in side-by-side report | none for current dense-Qwen scope |
| Naive PTQ baseline | complete | PTQ checkpoints and PPL/lm-eval files under `benchmark_results/quality-ptq-*` and `benchmark_results/lm-eval-qwen15b-full10/qwen15b_naive_ptq.json` | none for current dense-Qwen scope |
| llama.cpp Q4_K_M and Q8_0 baselines | complete for Qwen2.5-1.5B | GGUF summaries and RSS probe in `benchmark_results/gguf-*` and `benchmark_results/gguf-rss-qwen15b-row-i2s-fixed-2026-05-05` | none for current dense-Qwen scope |
| QAT with and without hidden MSE | complete | hidden-MSE, KL-only, dense-head, and row-scale checkpoints/evals summarized in side-by-side report | longer training remains research, not a completed proof |
| Row-scale versus tensor-scale | complete for Qwen2.5-1.5B dense-head | row-scale full ten-task mean `0.499459`; paired CI `[+0.009028, +0.021134]` vs tensor-scale dense-head | still below FP |
| GGUF conversion and packed CPU inference | partial | static-ternary materialization, reusable bridge runner `benchmarks/build_static_ternary_gguf_bridge.py`, direct dense GGUF bridge `benchmarks/convert_static_ternary_to_gguf.py`, direct scalar `I2_S` writer `benchmarks/convert_static_ternary_to_i2s_gguf.py`, direct packed support audit `benchmarks/results/direct_packed_gguf_support_2026-05-13.md`, TQ2_0, tensor-scale I2_S, row-scale I2_S prototype; `patches/llama-i2s-row-scale.patch` | direct scalar `I2_S` export loads/runs but Qwen0.5B quality fails as NaN PPL; direct packed row-scale `I2_S` GGUF writing is not complete and needs a stable row-scale format |
| TL2 conversion and CPU inference | partial | `benchmarks/results/conversion_support_audit_2026-05-05.md`; `benchmarks/results/tl2_shape_support_audit_2026-05-05.md`; `benchmarks/results/tl2_scale_semantics_2026-05-05.md`; Qwen0.5B TL2 probe `benchmark_results/gguf-qwen05b-tl2-avx512-2026-05-05/summary.json` | dense Qwen0.5B TL2 is model-specific and quality-failed; current TL2 one-scale semantics are mathematically invalid for the strong row-scale Qwen1.5B checkpoint; Qwen2MoE and Kimi remain unvalidated |
| Row-scale I2_S quality preservation | complete as prototype | heap-fix confirmation `benchmark_results/gguf-qwen15b-row-i2s-heapfix-confirm/summary.json`; audit `benchmark_results/evidence_audit/qwen15b_row_i2s_heapfix.md`; format audit `benchmarks/results/i2s_row_scale_format_audit_2026-05-13.md` | not an upstream/default stable GGUF format; current patch overloads `I2_S` rather than defining a new row-scale qtype |
| Row-scale I2_S thread scaling | complete as prototype | `benchmarks/results/i2s_thread_scaling_2026-05-05.md`; audit `benchmark_results/evidence_audit/qwen15b_row_i2s_thread_scaling.md` | decode remains around `18-20 tok/s` after 4 threads |
| Native AVX-512 check | complete | `benchmark_results/gguf-qwen15b-row-i2s-prototype-native-suite/summary.json`; audit `benchmark_results/evidence_audit/qwen15b_row_i2s_native.md` | no AVX-512 speedup shown |
| Packed GGUF RSS | complete for current row-scale suite | `benchmarks/results/gguf_memory_2026-05-05.md`; `benchmarks/results/gguf_context_scaling_2026-05-05.md`; audits `benchmark_results/evidence_audit/qwen15b_row_i2s_rss.md` and `benchmark_results/evidence_audit/qwen15b_context_scaling_rss.md` | none for dense Qwen2.5-1.5B GGUF RSS; still not MoE/TL2 |
| Side-by-side comparison | complete for current artifacts | `benchmarks/results/qwen_side_by_side_2026-05-05.md`; generated by `benchmarks/build_qwen_side_by_side.py` | does not cover unfinished TL2/MoE paths |
| Publishability verdict | partial | README, `benchmarks/results/qwen_retrofit_2026-05-03.md`, and `benchmarks/results/publishable_claims_2026-05-05.md` state negative arbitrary-retrofit verdict, supported claims, unsupported claims, and measured recovery path | paper-grade claim still needs stable row-scale format, direct exporter, and possibly more tasks |
| MoE/Kimi feasibility | not complete | `benchmarks/results/moe_support_audit_2026-05-05.md` identifies generic MoE support and Qwen2MoE converter/runtime infrastructure | no Kimi converter, router distillation, MoE quality run, or expert-locality benchmark |

## Current Verdict

The dense-Qwen evidence is strong enough to support a negative result plus a
measured recovery path:

- blind FP/BF16-to-ternary PTQ fails mathematically and empirically;
- QAT/distillation under ternary-forward constraints recovers significant
  quality but does not reach FP quality;
- row-wise scales and a dense tied `lm_head` are the strongest tested recipe;
- a row-scale-aware `I2_S` prototype preserves row-scale quality and runs on
  commodity CPU.

The active goal is not complete because the repo still lacks a production
row-scale GGUF type, quality-preserving direct packed row-scale ternary-state
GGUF ingestion, a quality-preserving Qwen TL2 path, and MoE/Kimi proof.

## Next Required Gates

1. Promote row-scale `I2_S` into a stable GGUF type or compatibility-safe
   format rather than replacing the existing `I2_S` layout. The format audit
   shows default row-scale `I2_S` is `30836.21x` worse than row-scale `TQ2_0`
   by PPL, while the patched prototype is `1.0016x`, but the patch reuses the
   existing `I2_S` type and is not product-format safe.
2. Extend direct GGUF ingestion for `ternary_state_dict.pt` from the current
   dense F16 and scalar `I2_S` bridges to a packed row-scale-aware writer. The
   scalar `I2_S` writer proves the C++ runtime can load direct packed tensors,
   but Qwen2.5-0.5B scalar quality fails with NaN PPL. Row-scale quality still
   requires a compatibility-safe per-row-scale layout or new GGUF qtype.
3. Implement and benchmark a row-scale-aware Qwen2.5-1.5B TL2 path, or keep
   TL2 out of the supported product claim. The current Qwen0.5B TL2 probe is a
   model-specific engineering result with NaN PPL, and the current TL2
   one-scale convention induces `1.904230` relative error on the strong
   row-scale checkpoint.
4. Extend quality evaluation only if the product/paper scope requires a broader
   leaderboard; the current ten-task set is sufficient for the negative
   retrofit verdict but not a full model card.
5. Treat MoE/Kimi as a separate research milestone requiring converter,
   router-distillation, expert-layout, and expert-locality benchmarks.
