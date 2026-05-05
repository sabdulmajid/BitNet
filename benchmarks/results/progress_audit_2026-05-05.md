# BitNet Retrofit Progress Audit, 2026-05-05

This audit maps the original six-item benchmark plan to concrete artifacts in
this fork. It is intentionally conservative: a requirement is marked complete
only when there is a file, log, or mechanical audit supporting it.

## Verdict State

Current evidence still supports the negative retrofit verdict:

- Blind FP/BF16 to ternary PTQ is not viable for the tested Qwen checkpoints.
- QAT/distillation recovers substantial signal over blind PTQ.
- The strongest current Qwen2.5-1.5B ternary path is KL-only distillation with
  a dense tied `lm_head`, followed by static-ternary materialization and packed
  GGUF `I2_S` or `TQ2_0`.
- The strongest current CPU-side checkpoint is still far below FP/Q8/Q4
  language-modeling quality.
- The Qwen2.5-1.5B row-scale dense-head ablation is still running and must not
  be cited until its checkpoint and dependent evals pass audit.

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
| Add row-scale versus tensor-scale | partial | Qwen2.5-0.5B row-scale evidence complete; Qwen2.5-1.5B row-scale dense-head job `9771` reached steps 1000, 2000, 3000, and 4000, and all four checkpoints passed audit at 196 ternary keys / 196 row-scale tensors with first scale shape `(1536, 1)`; early step-1000 PPL probe is WikiText `88.634` and FineWeb `45.680`, not a final result; final step 5000 quality jobs `9776`, `9779`, `9780`, postprocess `9781`, and row GGUF job `9794` are still pending |
| Convert repaired checkpoints into GGUF/TL2/I2_S | partial | static-ternary materialization to GGUF and packed `TQ2_0`/`I2_S` complete; native direct `ternary_state_dict.pt` GGUF writer and Qwen TL2 path are not yet complete |
| Run actual bitnet.cpp / llama.cpp CPU inference | complete for packed GGUF probes | `benchmark_results/gguf-qwen15b-klonly-suite/summary.json`, `benchmark_results/gguf-qwen15b-klonly-i2s-mt-fixed/summary.json`, and dense-head suite `benchmark_results/gguf-qwen15b-klonly-notiehead-suite/summary.json` |
| Measure CPU tokens/sec, prompt throughput, RSS, model size, quality loss | complete for current baselines | PyTorch RSS/runtime in `benchmark_results/runtime-qwen-xeon4116-512x32/summary.md`; Xeon packed GGUF throughput/file size/PPL in `benchmark_results/gguf-qwen15b-klonly-suite/summary.json`; AMD dense-head packed GGUF in `benchmark_results/gguf-qwen15b-klonly-notiehead-suite/summary.json`; patched I2_S confirmation in `benchmark_results/gguf-qwen15b-klonly-i2s-mt-fixed/summary.json` |

## Mechanical Audit Evidence

The latest non-row evidence audit was generated at:

`benchmark_results/evidence_audit/latest_nonrow.md`

It passed the cited checkpoint counts, full ten-task lm-eval sample counts,
dense-head PPL files, MC200 files, patched I2_S GGUF summary, and Gaussian PTQ
math artifact.

Key audited values:

| artifact | audited value |
| --- | ---: |
| Qwen2.5-1.5B FP ten-task mean | 0.644169 |
| Qwen2.5-1.5B naive PTQ ten-task mean | 0.348671 |
| Qwen2.5-1.5B hidden-MSE QAT ten-task mean | 0.464809 |
| Qwen2.5-1.5B KL-only QAT ten-task mean | 0.483438 |
| Qwen2.5-1.5B KL-only dense-head ten-task mean | 0.484378 |
| Qwen2.5-1.5B KL-only dense-head WikiText PPL | 43.372 |
| Qwen2.5-1.5B KL-only dense-head FineWeb-heldout PPL | 22.759 |
| Qwen2.5-1.5B KL-only static-ternary patched I2_S PPL | 54.7366 |
| Qwen2.5-1.5B KL-only static-ternary patched I2_S decode tok/s | 18.63 |
| Qwen2.5-1.5B KL-only dense-head static-ternary I2_S PPL | 47.3435 |
| Qwen2.5-1.5B KL-only dense-head static-ternary I2_S decode tok/s on AMD 5945WX | 45.50 |
| Gaussian absmean ternary relative output Frobenius error | 0.512542 |

## Current Open Gaps

1. The Qwen2.5-1.5B row-scale dense-head ablation has not completed. Its
   step-1000 checkpoint proves the row-scale export path mechanically, but its
   final results are queued for automatic quality, MC200, full ten-task
   lm-eval, GGUF packing, and postprocess auditing. No row-scale 1.5B quality
   claim should be made yet.
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
