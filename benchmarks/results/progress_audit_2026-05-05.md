# BitNet Retrofit Progress Audit, 2026-05-05

This audit maps the original six-item benchmark plan to concrete artifacts in
this fork. It is intentionally conservative: a requirement is marked complete
only when there is a file, log, or mechanical audit supporting it.

## Verdict State

Current evidence still supports the negative retrofit verdict:

- Blind FP/BF16 to ternary PTQ is not viable for the tested Qwen checkpoints.
- QAT/distillation recovers substantial signal over blind PTQ.
- The strongest current Qwen2.5-1.5B PyTorch-quality path is KL-only
  row-scale distillation with a dense tied `lm_head`.
- The strongest default packed row-scale path that preserves quality is
  currently GGUF `TQ2_0`. The default row-scale `I2_S` artifact fails quality
  audit, but a local per-row-scale `I2_S` prototype patch fixes the failure.
- The strongest current CPU-side checkpoint is still far below FP/Q8/Q4
  language-modeling quality.

## Requirement Checklist

| requirement | status | evidence |
| --- | --- | --- |
| Fix FSDP ternary export bug for Qwen2.5-1.5B step-5000 | complete | `checkpoints/qwen2.5-1.5b-fineweb-edu/step-5000/ternary_state_dict.pt` audits at 197 ternary keys / 197 scales, scalar scale, codes in {-1,0,1}, `tie_word_embeddings=false` |
| Re-export repaired 1.5B from checkpoint state | complete | same repaired step-5000 directory contains `ternary_state_dict.pt`, `model.safetensors`, tokenizer/config files |
| Run fixed prompt suites for repaired 1.5B and complete 0.5B | complete as sanity check | `benchmark_results/generation/qwen15b_step5000_core_cpu_16tok.jsonl` and `benchmark_results/generation/qwen05b_step1000_core_cpu.jsonl`, each 5 prompts from `benchmarks/prompts_core.jsonl`; this is not a quality benchmark |
| Add WikiText and FineWeb heldout perplexity | complete for FP/PTQ/QAT families cited in report | `benchmark_results/quality-*/*.json`; latest dense-head 1.5B files are `benchmark_results/quality-qwen15b-klonly-notiehead-5000/qwen15b_ternary_wikitext.json` and `...fineweb_heldout.json` |
| Add HellaSwag/PIQA/ARC task evals | complete via fast MC slices and stronger lm-eval runs | fast MC files under `benchmark_results/mc-*`; full ten-task lm-eval files under `benchmark_results/lm-eval-qwen15b-full10`, `...klonly-full10`, and `...klonly-notiehead-full10` |
| Add FP baseline | complete | HF FP PPL in `benchmark_results/quality-9735`; ten-task FP in `benchmark_results/lm-eval-qwen15b-full10/qwen15b_fp.json`; GGUF F16/Q8/Q4 runtime in `benchmark_results/gguf-qwen15b-klonly-suite/summary.json` |
| Add naive PTQ BitNet baseline | complete | `checkpoints/qwen2.5-0.5b-naive-ptq-tensor`, `checkpoints/qwen2.5-1.5b-naive-ptq-tensor`, PPL files under `benchmark_results/quality-ptq-*`, ten-task PTQ in `benchmark_results/lm-eval-qwen15b-full10/qwen15b_naive_ptq.json` |
| Add llama.cpp Q4_K_M and Q8_0 baselines | complete for Qwen2.5-1.5B | `benchmark_results/gguf-qwen15b-klonly-suite/summary.json` includes F16, Q8_0, Q4_K_M, blind TQ2_0, blind I2_S, and trained static-ternary artifacts |
| Add QAT with and without hidden MSE | complete | hidden-MSE run `checkpoints/qwen2.5-1.5b-fineweb-edu/step-5000`; KL-only run `checkpoints/qwen2.5-1.5b-fineweb-edu-klonly-5000/step-5000`; full ten-task comparison in `benchmark_results/lm-eval-qwen15b-klonly-full10/selected_metrics_with_baselines.md` |
| Add row-scale versus tensor-scale | complete for Qwen2.5-1.5B dense-head ablation | Qwen2.5-1.5B row-scale dense-head job `9771` completed 5000 steps; checkpoints step 1000/2000/3000/4000/5000 all passed audit at 196 ternary keys / 196 row-scale tensors; final PPL, MC200, full ten-task lm-eval, paired deltas, and row GGUF suite completed under `benchmark_results/quality-qwen15b-klonly-row-notiehead-5000`, `benchmark_results/mc-qwen15b-klonly-row-notiehead-5000-200`, `benchmark_results/lm-eval-qwen15b-klonly-row-notiehead-full10`, and `benchmark_results/gguf-qwen15b-klonly-row-notiehead-suite` |
| Convert repaired checkpoints into GGUF/TL2/I2_S | partial | static-ternary materialization to GGUF and packed `TQ2_0`/`I2_S` complete for tensor-scale checkpoints; row-scale materialization and `TQ2_0` preserve quality; default row-scale `I2_S` fails audit; per-row-scale `I2_S` prototype patch preserves quality but is not yet the default format; native direct `ternary_state_dict.pt` GGUF writer and Qwen TL2 path are not yet complete |
| Run actual bitnet.cpp / llama.cpp CPU inference | complete for packed GGUF probes | `benchmark_results/gguf-qwen15b-klonly-suite/summary.json`, `benchmark_results/gguf-qwen15b-klonly-i2s-mt-fixed/summary.json`, dense-head suite `benchmark_results/gguf-qwen15b-klonly-notiehead-suite/summary.json`, and row-scale dense-head suite `benchmark_results/gguf-qwen15b-klonly-row-notiehead-suite/summary.json` |
| Measure CPU tokens/sec, prompt throughput, RSS, model size, quality loss | complete for current baselines | PyTorch RSS/runtime in `benchmark_results/runtime-qwen-xeon4116-512x32/summary.md`; Xeon packed GGUF throughput/file size/PPL in `benchmark_results/gguf-qwen15b-klonly-suite/summary.json`; AMD dense-head packed GGUF in `benchmark_results/gguf-qwen15b-klonly-notiehead-suite/summary.json`; AMD row-scale dense-head packed GGUF in `benchmark_results/gguf-qwen15b-klonly-row-notiehead-suite/summary.json`; patched I2_S confirmation in `benchmark_results/gguf-qwen15b-klonly-i2s-mt-fixed/summary.json`; RSS context scaling in `benchmark_results/gguf-rss-qwen15b-context-scaling-2026-05-05/summary.json` |

## Mechanical Audit Evidence

The latest non-row evidence audit was generated at:

`benchmark_results/evidence_audit/latest_nonrow.md`

It passed the cited checkpoint counts, full ten-task lm-eval sample counts,
dense-head PPL files, MC200 files, patched I2_S GGUF summary, and Gaussian PTQ
math artifact.

The row-scale evidence audit was generated at:

`benchmark_results/evidence_audit/qwen15b_row_notie_5000.md`

It passed the row-scale checkpoint counts, full ten-task lm-eval sample counts,
PPL files, MC200 files, and paired-delta artifacts. The row GGUF audit at
`benchmark_results/gguf-qwen15b-klonly-row-notiehead-suite/audit.md` correctly
fails the current row-scale `I2_S` artifact because its PPL is more than
30,000x worse than row-scale `TQ2_0`.

The row-scale `I2_S` prototype result is recorded at:

`benchmarks/results/i2s_row_scale_prototype_2026-05-05.md`

It was generated by applying `patches/llama-i2s-row-scale.patch`, rebuilding the
portable AVX2 binaries, requantizing the row-scale F16 GGUF to `I2_S`, and
running the same fixed WikiText excerpt plus `llama-bench` on the Xeon 4116.
The generated six-model suite is under
`benchmark_results/gguf-qwen15b-row-i2s-prototype-suite/summary.json`; the
tracked manifest is `benchmarks/gguf_qwen15b_row_i2s_prototype_manifest.json`.
The mechanical suite audit is
`benchmark_results/evidence_audit/qwen15b_row_i2s_prototype.md`, which passes
with six rows, no failed/NaN entries, and an `I2_S`/row-scale-reference PPL
ratio of `1.00157` under a strict `1.01` max-ratio threshold.

The native `GGML_NATIVE=ON` AVX-512-enabled suite is under
`benchmark_results/gguf-qwen15b-row-i2s-prototype-native-suite/summary.json`.
Its mechanical audit is
`benchmark_results/evidence_audit/qwen15b_row_i2s_native.md`, which passes with
six rows and an `I2_S`/row-scale-reference PPL ratio of `1.00128`. Native
AVX-512 preserved quality but did not improve row-scale `I2_S` throughput on
the Xeon 4116.

The row-scale `I2_S` thread-scaling probe is tracked at
`benchmarks/results/i2s_thread_scaling_2026-05-05.md`. The row-scale patch now
uses heap temporary buffers in the I2_S prompt GEMM/GEMV path; after that fix,
`llama-bench` passes at 1/2/4/8/12/16/24 threads. Its mechanical audit is
`benchmark_results/evidence_audit/qwen15b_row_i2s_thread_scaling.md`, which
passes all seven expected thread rows with no failed return codes.

The post-heap-fix row-scale `I2_S` quality confirmation is under
`benchmark_results/gguf-qwen15b-row-i2s-heapfix-confirm/summary.json`. Its
mechanical audit is `benchmark_results/evidence_audit/qwen15b_row_i2s_heapfix.md`,
which passes with one row, PPL `38.8832`, and no failed return codes.

The packed GGUF RSS probe is tracked at
`benchmarks/results/gguf_memory_2026-05-05.md`. Its mechanical audit is
`benchmark_results/evidence_audit/qwen15b_row_i2s_rss.md`, which passes all six
expected rows with positive RSS and zero return-code failures.

The GGUF RSS context-scaling probe is tracked at
`benchmarks/results/gguf_context_scaling_2026-05-05.md`. Its mechanical audit is
`benchmark_results/evidence_audit/qwen15b_context_scaling_rss.md`, which passes
all 24 expected rows across contexts `512,2048,8192,32768` with positive RSS
and zero return-code failures.

Key audited values:

| artifact | audited value |
| --- | ---: |
| Qwen2.5-1.5B FP ten-task mean | 0.644169 |
| Qwen2.5-1.5B naive PTQ ten-task mean | 0.348671 |
| Qwen2.5-1.5B hidden-MSE QAT ten-task mean | 0.464809 |
| Qwen2.5-1.5B KL-only QAT ten-task mean | 0.483438 |
| Qwen2.5-1.5B KL-only dense-head ten-task mean | 0.484378 |
| Qwen2.5-1.5B KL-only row-scale dense-head ten-task mean | 0.499459 |
| Qwen2.5-1.5B KL-only dense-head WikiText PPL | 43.372 |
| Qwen2.5-1.5B KL-only dense-head FineWeb-heldout PPL | 22.759 |
| Qwen2.5-1.5B KL-only row-scale dense-head WikiText PPL | 38.580 |
| Qwen2.5-1.5B KL-only row-scale dense-head FineWeb-heldout PPL | 21.333 |
| Row-scale dense-head minus tensor-scale dense-head ten-task macro delta | +0.015081 |
| Row-scale dense-head minus tensor-scale dense-head paired 95% CI | [+0.009028, +0.021134] |
| Qwen2.5-1.5B KL-only static-ternary patched I2_S PPL | 54.7366 |
| Qwen2.5-1.5B KL-only static-ternary patched I2_S decode tok/s | 18.63 |
| Qwen2.5-1.5B KL-only dense-head static-ternary I2_S PPL | 47.3435 |
| Qwen2.5-1.5B KL-only dense-head static-ternary I2_S decode tok/s on AMD 5945WX | 45.50 |
| Qwen2.5-1.5B KL-only row-scale dense-head static-ternary TQ2_0 PPL on AMD 5945WX | 38.8224 |
| Qwen2.5-1.5B KL-only row-scale dense-head static-ternary TQ2_0 decode tok/s on AMD 5945WX | 44.85 |
| Qwen2.5-1.5B KL-only row-scale dense-head static-ternary I2_S PPL on AMD 5945WX | 1.197e6 |
| Qwen2.5-1.5B row-scale dense-head I2_S prototype PPL on Xeon 4116 | 38.8832 |
| Qwen2.5-1.5B row-scale dense-head I2_S prototype prompt tok/s on Xeon 4116 | 218.17 |
| Qwen2.5-1.5B row-scale dense-head I2_S prototype decode tok/s on Xeon 4116 | 18.97 |
| Qwen2.5-1.5B row-scale dense-head I2_S prototype GGUF file size | 1,211.3 MiB |
| Qwen2.5-1.5B row-scale dense-head native AVX-512 I2_S prototype PPL on Xeon 4116 | 38.8853 |
| Qwen2.5-1.5B row-scale dense-head native AVX-512 I2_S prototype prompt tok/s on Xeon 4116 | 207.35 |
| Qwen2.5-1.5B row-scale dense-head native AVX-512 I2_S prototype decode tok/s on Xeon 4116 | 18.37 |
| Row-scale dense-head I2_S portable AVX2 prompt tok/s at 1 thread | 22.02 |
| Row-scale dense-head I2_S portable AVX2 prompt tok/s at 24 threads | 245.31 |
| Row-scale dense-head I2_S portable AVX2 decode tok/s range, all thread rows | 8.57-19.49 |
| Qwen2.5-1.5B FP F16 GGUF max RSS at `-c 512` | 2.948 GiB |
| Qwen2.5-1.5B FP Q4_K_M GGUF max RSS at `-c 512` | 0.985 GiB |
| Qwen2.5-1.5B row-scale dense-head I2_S max RSS at `-c 512` | 1.250 GiB |
| Qwen2.5-1.5B FP F16 GGUF max RSS at `-c 32768` | 3.812 GiB |
| Qwen2.5-1.5B FP Q4_K_M GGUF max RSS at `-c 32768` | 1.850 GiB |
| Qwen2.5-1.5B row-scale dense-head TQ2_0 max RSS at `-c 32768` | 2.121 GiB |
| Qwen2.5-1.5B row-scale dense-head I2_S max RSS at `-c 32768` | 2.114 GiB |
| Gaussian absmean ternary relative output Frobenius error | 0.512542 |

## Current Open Gaps

1. Row-scale `I2_S` is not yet a default stable format. The current default
   `I2_S` packing path loses row-scale magnitudes and fails the GGUF audit with
   PPL `1.197e6`; the prototype patch stores one scale per output row and
   recovers PPL `38.8832`, but the format change still needs integration,
   compatibility policy, and regeneration of affected artifacts.
2. Native direct GGUF writing from `ternary_state_dict.pt` is not complete.
   Static-ternary materialization is a validated bridge, not the final storage
   path.
3. Qwen TL2 is not complete. The current `llama-quantize` CLI does not expose
   TL2 and its generic quantize switch has no TL2 quantization case. The
   BitNet-specific HF converter exposes `--outtype tl2`, but it only registers
   LLaMA/Mistral/Mixtral-style and BitNet model classes, not `Qwen2ForCausalLM`.
   Custom TL2 code generation now supports arbitrary shapes, but a Qwen-aware
   TL2 GGUF writer/loader path still needs implementation and validation.
4. The validated threaded `I2_S` writer fix exists as
   `patches/llama-i2s-threaded-quantization.patch`, but the llama.cpp submodule
   has not been advanced to a commit containing that fix.
5. MoE/Kimi remains unproven. The backend has generic MoE execution support,
   but this fork has not implemented a Kimi-compatible ternary converter,
   router distillation, expert-locality benchmark, or MoE quality run.
6. The current result is not a publishable "arbitrary model retrofit works"
   claim. The publishable angle, if any, is the negative result plus a measured
   recovery path: PTQ fails mathematically and empirically; QAT/distillation
   plus static ternary packing partially recovers quality and yields real CPU
   throughput on a 2017 Xeon.
