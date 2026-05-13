# bitnet.cpp
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![version](https://img.shields.io/badge/version-1.0-blue)

[<img src="./assets/header_model_release.png" alt="BitNet Model on Hugging Face" width="800"/>](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T)

Try it out via this [demo](https://demo-bitnet-h0h8hcfqeqhrf5gf.canadacentral-01.azurewebsites.net/), or build and run it on your own [CPU](https://github.com/microsoft/BitNet?tab=readme-ov-file#build-from-source) or [GPU](https://github.com/microsoft/BitNet/blob/main/gpu/README.md).

bitnet.cpp is the official inference framework for 1-bit LLMs (e.g., BitNet b1.58). It offers a suite of optimized kernels, that support **fast** and **lossless** inference of 1.58-bit models on CPU and GPU (NPU support will coming next).

The first release of bitnet.cpp is to support inference on CPUs. bitnet.cpp achieves speedups of **1.37x** to **5.07x** on ARM CPUs, with larger models experiencing greater performance gains. Additionally, it reduces energy consumption by **55.4%** to **70.0%**, further boosting overall efficiency. On x86 CPUs, speedups range from **2.37x** to **6.17x** with energy reductions between **71.9%** to **82.2%**. Furthermore, bitnet.cpp can run a 100B BitNet b1.58 model on a single CPU, achieving speeds comparable to human reading (5-7 tokens per second), significantly enhancing the potential for running LLMs on local devices. Please refer to the [technical report](https://arxiv.org/abs/2410.16144) for more details.

**Latest optimization** introduces parallel kernel implementations with configurable tiling and embedding quantization support, achieving **1.15x to 2.1x** additional speedup over the original implementation across different hardware platforms and workloads. For detailed technical information, see the [optimization guide](src/README.md).

## Experimental Qwen Retrofit Work in This Fork

This section describes experiments in this fork, not an upstream `bitnet.cpp`
release claim. It tests whether pretrained dense Hugging Face models can be
retrofitted into BitNet-style W1.58A8 models. The current answer is deliberately
conservative:

- **Verdict so far: no lossless arbitrary retrofit.** The evidence supports
  QAT/distillation under ternary forward constraints, not blind conversion of
  existing FP16/BF16 checkpoints into 1.58-bit form with acceptable degradation.
- **Blind post-training ternarization is not viable** for the tested Qwen
  checkpoints. It causes catastrophic perplexity collapse. On a 10-trial
  Gaussian matrix audit, tensor absmean and row absmean ternarization both
  displace about `51.2%` of the output Frobenius norm before any model-level
  evaluation.
- **QAT/distillation is materially better than naive PTQ**, which shows that
  training under ternary forward constraints recovers real signal.
- **The current QAT checkpoints are not FP-quality yet.** The strongest
  Qwen2.5-1.5B static ternary checkpoint is usable enough for measurement, but
  it remains significantly worse than the FP reference on perplexity and task
  accuracy.
- **KL-only distillation is stronger than the original hidden-MSE recipe.** At
  5000 steps, it improves Qwen2.5-1.5B WikiText/FineWeb PPL from
  `86.414`/`40.398` to `50.595`/`26.599` versus the hidden-MSE QAT run, and
  improves uncapped ten-task lm-eval mean from `0.465` to `0.483`. FP remains
  `0.644`.
- **Keeping Qwen's tied output head dense improves 1.5B likelihood, not the
  task verdict.** The dense-`lm_head` KL-only 1.5B ablation improves
  WikiText/FineWeb PPL again to `43.372`/`22.759`, but its uncapped ten-task
  mean is only `0.484`; the paired macro delta versus the all-ternary KL-only
  run is `+0.00094` with 95% CI crossing zero.
- **Row-wise ternary scales are the strongest PyTorch quality ablation so
  far.** Qwen2.5-1.5B KL-only row-scale training with a dense tied `lm_head`
  reaches WikiText/FineWeb PPL `38.580`/`21.333` and an uncapped ten-task mean
  of `0.499459`. It improves over the tensor-scale dense-head run by
  `+0.015081` macro mean with paired 95% CI `[+0.009028, +0.021134]`, but it
  still trails FP by about `0.145` macro mean.
- **Packed row-scale deployment is format-limited.** Row-scale static ternary
  materialization and generic `TQ2_0` preserve the row-scale quality on the
  fixed GGUF WikiText excerpt: PPL `38.8651` and `38.8224`, respectively. The
  default `I2_S` path does **not** preserve it: row-scale `I2_S` explodes to
  PPL `1.197e6` and produces a failed smoke completion. A local prototype patch
  that stores one scale per output row fixes this layout issue: row-scale
  `I2_S` reaches PPL `38.8832`, `218.17` prompt tok/s, and
  `18.97` decode tok/s on the Xeon 4116 portable-AVX2 heap-fix confirmation
  run. That patch is not yet an upstreamed default. A native
  `GGML_NATIVE=ON` build reported
  `AVX512 = 1` but was slightly slower for this row-scale `I2_S` prototype
  (`207.35` prompt tok/s, `18.37` decode tok/s), so there is no current
  AVX-512 speedup claim.
- **Row-wise ternary scales help likelihood but are not enough alone.** A
  Qwen2.5-0.5B row-scale ablation cut heldout perplexity by about 2.4x versus
  the tensor-scale QAT checkpoint, and the final 1.5B row-scale run improved
  ten-task accuracy. Neither result reaches FP quality.
- **KL-only distillation plus a dense tied `lm_head` is the current base recipe.**
  Removing the hidden-state MSE term improved quality, leaving Qwen's tied
  output head dense improved it again at both 0.5B and 1.5B scale, and row-wise
  scales improved the 1.5B recipe further. This checkpoint family is not an
  all-linear W1.58 model, but it is the more architecturally honest Qwen
  retrofit policy tested so far.
- **PyTorch ternary simulation is not the speed path.** On the Xeon Silver 4116
  host, the exported ternary checkpoints are smaller in memory but slower than
  FP under PyTorch because the probe dequantizes into dense matmuls. Real
  product claims require GGUF/I2_S or TL2 execution through `bitnet.cpp`.
- **Packed ternary runtime is fast, but speed does not rescue blind
  ternarization.** On the same Xeon, blind I2_S/TQ2_0 conversion runs faster
  than F16 but destroys quality. Materializing the trained
  `ternary_state_dict.pt` and packing it as llama.cpp `TQ2_0` preserves both
  tensor-scale and row-scale QAT perplexity; tensor-scale single-thread `I2_S`
  also preserves quality. The default row-scale `I2_S` bridge fails because it
  stores only one tensor scale; `patches/llama-i2s-row-scale.patch` prototypes a
  per-row-scale layout that fixes the row-scale quality failure. The original
  tensor-scale multi-thread I2_S writer path was unsafe for this artifact; this
  fork also includes a validated patch file for that threaded packing/scaling
  bug, while the safe wrapper remains single-threaded until the submodule is
  advanced.
- **TL2 is now a partial dense-Qwen engineering probe, not a product claim.**
  This fork can generate a Qwen2.5-0.5B TL2 GGUF only after exact
  model-specific TL2 code generation and a matching `BITNET_X86_TL2=ON`
  runtime rebuild. The Qwen2.5-0.5B TL2 artifact is `599.5 MiB` and runs on an
  AVX-512-enabled TL2 build, but quality fails: PPL is `NaN` and smoke text is
  nonsensical. The generic AVX2 build can benchmark the artifact but standard
  smoke/perplexity segfault. The strong Qwen2.5-1.5B row-scale checkpoint has
  not been validated through TL2, and the current TL1/TL2 converter uses one
  tensor scale rather than row scales. The scale-semantics audit shows why a
  naive row-scale TL2 export is invalid: replacing row-wise scales with one TL2
  tensor scale would introduce relative Frobenius/output-RMS error `1.904230`
  on the best row-scale 1.5B checkpoint; the scalar-scale control is `0.0`.
- **MoE remains unproven in this fork.** The vendored llama.cpp backend contains
  generic expert routing and merged expert-tensor support, and the BitNet HF
  converter has partial Qwen-style expert packing. This repo has not yet shown a
  Kimi-compatible ternary converter, expert-router distillation, or a Kimi/MoE
  benchmark. The mechanical MoE audit confirms generic Qwen2MoE infrastructure
  exists, but no Kimi-specific converter/runtime mapping or benchmark artifact
  is present.

Current evidence is tracked in
[benchmarks/results/qwen_retrofit_2026-05-03.md](benchmarks/results/qwen_retrofit_2026-05-03.md).
The current prompt-to-artifact progress audit is
[benchmarks/results/progress_audit_2026-05-05.md](benchmarks/results/progress_audit_2026-05-05.md).
The active-goal completion audit is
[benchmarks/results/active_goal_completion_audit_2026-05-05.md](benchmarks/results/active_goal_completion_audit_2026-05-05.md).
The artifact-generated side-by-side Qwen summary is
[benchmarks/results/qwen_side_by_side_2026-05-05.md](benchmarks/results/qwen_side_by_side_2026-05-05.md).
The row-scale `I2_S` prototype note is
[benchmarks/results/i2s_row_scale_prototype_2026-05-05.md](benchmarks/results/i2s_row_scale_prototype_2026-05-05.md).
The row-scale `I2_S` thread-scaling note is
[benchmarks/results/i2s_thread_scaling_2026-05-05.md](benchmarks/results/i2s_thread_scaling_2026-05-05.md).
The GGUF RSS note is
[benchmarks/results/gguf_memory_2026-05-05.md](benchmarks/results/gguf_memory_2026-05-05.md).
The GGUF context-scaling RSS note is
[benchmarks/results/gguf_context_scaling_2026-05-05.md](benchmarks/results/gguf_context_scaling_2026-05-05.md).
The conversion support audit is
[benchmarks/results/conversion_support_audit_2026-05-05.md](benchmarks/results/conversion_support_audit_2026-05-05.md).
The TL2 shape support audit is
[benchmarks/results/tl2_shape_support_audit_2026-05-05.md](benchmarks/results/tl2_shape_support_audit_2026-05-05.md).
The Qwen2.5-0.5B TL2 probe is
[benchmarks/results/qwen05b_tl2_probe_2026-05-05.md](benchmarks/results/qwen05b_tl2_probe_2026-05-05.md).
The TL2 scale-semantics audit is
[benchmarks/results/tl2_scale_semantics_2026-05-05.md](benchmarks/results/tl2_scale_semantics_2026-05-05.md).
The row-scale `I2_S` format compatibility audit is
[benchmarks/results/i2s_row_scale_format_audit_2026-05-13.md](benchmarks/results/i2s_row_scale_format_audit_2026-05-13.md).
The MoE support audit is
[benchmarks/results/moe_support_audit_2026-05-05.md](benchmarks/results/moe_support_audit_2026-05-05.md).
The reusable static-ternary GGUF bridge note is
[benchmarks/results/static_ternary_gguf_bridge_2026-05-05.md](benchmarks/results/static_ternary_gguf_bridge_2026-05-05.md).
The direct static-ternary GGUF bridge note is
[benchmarks/results/direct_static_ternary_gguf_2026-05-13.md](benchmarks/results/direct_static_ternary_gguf_2026-05-13.md).
The direct packed GGUF support audit is
[benchmarks/results/direct_packed_gguf_support_2026-05-13.md](benchmarks/results/direct_packed_gguf_support_2026-05-13.md).
The publishable-claims ledger is
[benchmarks/results/publishable_claims_2026-05-05.md](benchmarks/results/publishable_claims_2026-05-05.md).
The compact evidence manifest with hashes and parsed metrics is
[benchmarks/results/evidence_manifest_2026-05-13.md](benchmarks/results/evidence_manifest_2026-05-13.md).
The benchmark harnesses are in [benchmarks/](benchmarks/).

### Current Perplexity Snapshot

| model | method | WikiText PPL | FineWeb-heldout PPL |
| --- | --- | ---: | ---: |
| Qwen2.5-0.5B | FP reference | 20.461 | 14.124 |
| Qwen2.5-0.5B | naive PTQ ternary | 169,414.428 | 608,726.749 |
| Qwen2.5-0.5B | QAT/distilled ternary | 1,079.167 | 373.775 |
| Qwen2.5-0.5B | QAT/distilled ternary, row scale | 444.691 | 152.821 |
| Qwen2.5-0.5B | QAT/distilled ternary, KL only | 296.602 | 108.366 |
| Qwen2.5-0.5B | QAT/distilled ternary, KL only, dense tied `lm_head` | 270.345 | 97.337 |
| Qwen2.5-1.5B | FP reference | 13.901 | 10.269 |
| Qwen2.5-1.5B | naive PTQ ternary | 3,813,121.803 | 9,582,923.269 |
| Qwen2.5-1.5B | QAT/distilled ternary | 86.414 | 40.398 |
| Qwen2.5-1.5B | QAT/distilled ternary, KL only | 50.595 | 26.599 |
| Qwen2.5-1.5B | QAT/distilled ternary, KL only, dense tied `lm_head` | 43.372 | 22.759 |
| Qwen2.5-1.5B | QAT/distilled ternary, KL only, row scale, dense tied `lm_head` | 38.580 | 21.333 |

These numbers are BF16/CUDA quality measurements using PyTorch simulation of
W1.58A8 math. They are **not** `bitnet.cpp` CPU throughput claims.

### Current Packed GGUF CPU Snapshot

Packed GGUF results below are fixed-excerpt `llama.cpp` CPU measurements. The
PPL values are comparable across rows; throughput is only comparable within the
same CPU family shown in the table.

| CPU | artifact | file MiB | PPL | prompt tok/s | decode tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| Intel Xeon Silver 4116 | FP F16 | 2,950.4 | 12.2808 | 114.47 | 5.56 |
| Intel Xeon Silver 4116 | FP Q8_0 | 1,570.3 | 12.3056 | 124.86 | 10.13 |
| Intel Xeon Silver 4116 | FP Q4_K_M | 940.4 | 12.8112 | 92.08 | 16.01 |
| Intel Xeon Silver 4116 | blind FP-to-I2_S | 766.1 | 1.206e51 | 204.57 | 18.34 |
| Intel Xeon Silver 4116 | KL-only static ternary I2_S, all-linear | 1,208.9 | 54.7366 | 205.76 | 18.60 |
| Intel Xeon Silver 4116 | KL-only row-scale static ternary F16, dense tied `lm_head` | 3,395.5 | 38.8651 | 114.75 | 5.49 |
| Intel Xeon Silver 4116 | KL-only row-scale static ternary TQ2_0, dense tied `lm_head` | 1,218.6 | 38.8224 | 169.46 | 18.68 |
| Intel Xeon Silver 4116 | KL-only row-scale static ternary I2_S prototype, dense tied `lm_head` | 1,211.3 | 38.8832 | 218.17 | 18.97 |
| AMD Ryzen Threadripper PRO 5945WX | KL-only static ternary I2_S, dense tied `lm_head` | 1,208.9 | 47.3435 | 464.19 | 45.50 |
| AMD Ryzen Threadripper PRO 5945WX | KL-only row-scale static ternary TQ2_0, dense tied `lm_head` | 1,218.6 | 38.8224 | 345.32 | 44.85 |
| AMD Ryzen Threadripper PRO 5945WX | KL-only row-scale static ternary I2_S, dense tied `lm_head` | 1,208.9 | 1.197e6 | 465.34 | 46.13 |

The AMD row-scale `I2_S` row is the failed default layout result. The Xeon
row-scale `I2_S` prototype row uses `patches/llama-i2s-row-scale.patch`, which
changes the packed tensor layout to store one scale per output row.
The same patched row-scale `I2_S` artifact also passed a native AVX-512-enabled
Xeon run at PPL `38.8853`, `207.35` prompt tok/s, and `18.37` decode tok/s;
quality is preserved, but throughput did not beat the portable AVX2 build.
Thread scaling on the portable AVX2 row-scale `I2_S` artifact shows prefill
scaling from `22.02` tok/s at 1 thread to `245.31` tok/s at 24 threads, while
decode improves through 4 threads and then stays near `18-20` tok/s. The
current row-scale patch includes a heap-buffer fix for the earlier low-thread
`llama-bench` crash.
At `-c 512`, `/usr/bin/time -v` reports peak RSS `1.250 GiB` for row-scale
`I2_S`, `1.257 GiB` for row-scale `TQ2_0`, `2.948 GiB` for FP F16, and
`0.985 GiB` for FP Q4_K_M. The same probe at `-c 32768` reports `2.114 GiB`
for row-scale `I2_S`, `2.121 GiB` for row-scale `TQ2_0`, `3.812 GiB` for FP
F16, and `1.850 GiB` for FP Q4_K_M. This preserves the ternary-vs-FP16 memory
advantage at long context, but it still does not beat the dense `Q4_K_M`
baseline on RSS.

### Practical Product Direction

This fork does **not** support a credible one-click, lossless ternarizer for
arbitrary Hugging Face models. A realistic product direction is a CPU-first
retrofit pipeline with measured guarantees:

- ingest a supported dense decoder checkpoint,
- replace eligible projections with BitLinear-style ternary-forward modules,
- distill against the FP teacher under the exact ternary constraint,
- export `ternary_state_dict.pt` plus a static-ternary GGUF bridge,
- pack `TQ2_0` and `I2_S` artifacts for commodity CPU inference,
- use `benchmarks/build_static_ternary_gguf_bridge.py` as the current
  reproducible bridge runner while direct GGUF export is still unfinished,
- promote the row-scale-aware `I2_S` prototype into a stable packed format or
  new GGUF quantization type before claiming row-scale `I2_S` deployment,
- publish a benchmark card with FP/Q8/Q4/blind-ternary/QAT comparisons.

The current MVP should target dense Qwen-style models first. MoE models such as
Kimi, production TL2 export for strong row-scale Qwen checkpoints, direct GGUF
ingestion of ternary state dicts, and quality guarantees for arbitrary
architectures remain research tasks.
The current toolchain split is mechanical: the BitNet HF converter now exposes
`tl2`, registers dense `Qwen2ForCausalLM`, and accepts `--kernel-config` for
model-specific TL2 shape tables, but it still does not register Qwen2MoE/Kimi
and TL2 still needs a matching generated LUT runtime. The vendored llama.cpp HF
converter registers Qwen2/Qwen2MoE but exposes only
`f32/f16/bf16/q8_0/tq1_0/tq2_0/auto`.
`llama-quantize` exposes `I2_S`, but it requires an existing GGUF input.

### Qwen2.5-0.5B Row-Scale Ablation

The row-scale ablation kept the same 1000-step Qwen2.5-0.5B setup as the
tensor-scale run, but exported one ternary scale per output row. Final training
metrics were loss `15.3338`, KL `2.0530`, and hidden MSE `13.2808`.

Compared with the tensor-scale QAT checkpoint, row scale improved WikiText PPL
from `1,079.167` to `444.691` and FineWeb-heldout PPL from `373.775` to
`152.821`. The fast 100-example multiple-choice slices did not show a matching
task-accuracy gain:

| task | tensor-scale acc_norm | row-scale acc_norm |
| --- | ---: | ---: |
| PIQA | 0.500 | 0.460 |
| ARC-Easy | 0.270 | 0.260 |
| ARC-Challenge | 0.300 | 0.240 |
| HellaSwag | 0.260 | 0.240 |

Interpretation: row scales reduce quantization error in the language-modeling
objective, but this training budget still does not produce a competitive
downstream checkpoint.

### Qwen2.5-0.5B KL-only and `lm_head` Ablations

A second 1000-step ablation disabled hidden-state MSE and trained only on
teacher KL. Final training metrics were loss/KL `1.6375` and hidden MSE `0`.

That improved WikiText PPL to `296.602` and FineWeb-heldout PPL to `108.366`.
A follow-up run also excluded `lm_head` from BitLinear replacement, preserving
Qwen's tied output embedding as dense FP/BF16. It replaced 168 linear modules
instead of 169, exported 168 ternary matrices, kept `tie_word_embeddings=true`,
and improved PPL again to `270.345` on WikiText and `97.337` on FineWeb-heldout.
The same 100-example multiple-choice slices were also stronger:

| task | tensor-scale acc_norm | row-scale acc_norm | KL-only acc_norm | KL-only dense `lm_head` acc_norm |
| --- | ---: | ---: | ---: | ---: |
| PIQA | 0.500 | 0.460 | 0.560 | 0.570 |
| ARC-Easy | 0.270 | 0.260 | 0.290 | 0.300 |
| ARC-Challenge | 0.300 | 0.240 | 0.240 | 0.290 |
| HellaSwag | 0.260 | 0.240 | 0.260 | 0.290 |

Interpretation: at this short budget, hidden-state MSE appears to overconstrain
the ternary student relative to KL-only distillation, and Qwen's tied output
head should not be blindly ternarized without a specific output-head training
policy. The dense-`lm_head` checkpoint is the best 0.5B ablation so far, but it
is not a fully ternary linear stack and remains far from FP quality.

### Early 100-example lm-eval Snapshot

EleutherAI `lm-eval` 0.4.11 was run on 100-example slices for ten tasks using
Qwen2.5-1.5B FP, naive PTQ ternary, and QAT/distilled ternary. Where a task
reports `acc_norm`, that metric is shown; otherwise raw `acc` is shown.

| task | metric | FP | naive PTQ | QAT ternary |
| --- | --- | ---: | ---: | ---: |
| ARC-Challenge | acc_norm | 0.410 | 0.300 | 0.300 |
| ARC-Easy | acc_norm | 0.760 | 0.220 | 0.510 |
| BoolQ | acc | 0.690 | 0.400 | 0.700 |
| COPA | acc | 0.830 | 0.510 | 0.640 |
| HellaSwag | acc_norm | 0.660 | 0.290 | 0.460 |
| OpenBookQA | acc_norm | 0.350 | 0.290 | 0.280 |
| PIQA | acc_norm | 0.800 | 0.580 | 0.590 |
| SciQ | acc_norm | 0.960 | 0.210 | 0.640 |
| TruthfulQA MC1 | acc | 0.280 | 0.220 | 0.200 |
| WinoGrande | acc | 0.720 | 0.490 | 0.580 |

Mean over these displayed metrics: FP 0.646, naive PTQ 0.351, QAT ternary
0.490. This supports the narrow claim that QAT/distillation recovers substantial
signal over blind ternarization. It does **not** support a claim that the
current ternary model preserves FP quality.

### Larger Core lm-eval Slice

The same Qwen2.5-1.5B comparison was rerun on 1000-example caps for the five
core tasks below. This is still capped, but it is a stronger estimate than the
100-example smoke slice above.

| task | metric | FP | naive PTQ | QAT ternary |
| --- | --- | ---: | ---: | ---: |
| ARC-Challenge | acc_norm | 0.448 | 0.266 | 0.259 |
| ARC-Easy | acc_norm | 0.713 | 0.246 | 0.486 |
| HellaSwag | acc_norm | 0.584 | 0.262 | 0.389 |
| PIQA | acc_norm | 0.756 | 0.510 | 0.614 |
| WinoGrande | acc | 0.647 | 0.509 | 0.531 |

Mean over these displayed metrics: FP 0.630, naive PTQ 0.359, QAT ternary
0.456. QAT recovers clear signal over blind PTQ on the mean, but it recovers
only about 36% of the FP-vs-PTQ gap and remains far below the FP reference.
Paired sample-level deltas on the same capped examples give QAT minus naive PTQ
`+0.097` macro mean with 95% CI `[+0.012, +0.182]`; QAT minus FP is `-0.174`
with 95% CI `[-0.213, -0.135]`. This supports both conclusions at once:
distillation is doing real work, and the current ternary checkpoint is still
not an FP-quality replacement.

### Full Core lm-eval Run

The five core tasks above were also run without `lm-eval` example caps
(`LIMIT=0`). This is the first uncapped task-accuracy evidence in this fork.

| task | metric | FP | naive PTQ | QAT ternary |
| --- | --- | ---: | ---: | ---: |
| ARC-Challenge | acc_norm | 0.450 | 0.262 | 0.264 |
| ARC-Easy | acc_norm | 0.720 | 0.244 | 0.478 |
| HellaSwag | acc_norm | 0.678 | 0.264 | 0.362 |
| PIQA | acc_norm | 0.758 | 0.508 | 0.622 |
| WinoGrande | acc | 0.638 | 0.498 | 0.523 |

Mean over these displayed metrics: FP 0.649, naive PTQ 0.355, QAT ternary
0.450. QAT improves over blind PTQ by `+0.095` macro mean with paired 95% CI
`[+0.015, +0.175]`, recovering about 32% of the FP-vs-PTQ gap. QAT remains
far below FP: QAT minus FP is `-0.199` macro mean with paired 95% CI
`[-0.270, -0.127]`.

### Full Ten-Task lm-eval Run

The full core run was merged with uncapped BoolQ, COPA, OpenBookQA, SciQ, and
TruthfulQA MC1 runs for the same artifacts.

| task | metric | FP | naive PTQ | QAT hidden-MSE | QAT KL-only | QAT KL-only dense `lm_head` | QAT KL-only row scale dense `lm_head` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ARC-Challenge | acc_norm | 0.450 | 0.262 | 0.264 | 0.271 | 0.264 | 0.272 |
| ARC-Easy | acc_norm | 0.720 | 0.244 | 0.478 | 0.483 | 0.501 | 0.518 |
| HellaSwag | acc_norm | 0.678 | 0.264 | 0.362 | 0.378 | 0.391 | 0.412 |
| PIQA | acc_norm | 0.758 | 0.508 | 0.622 | 0.637 | 0.647 | 0.650 |
| WinoGrande | acc | 0.638 | 0.498 | 0.523 | 0.521 | 0.523 | 0.537 |
| BoolQ | acc | 0.726 | 0.506 | 0.593 | 0.596 | 0.597 | 0.605 |
| COPA | acc | 0.830 | 0.510 | 0.640 | 0.700 | 0.680 | 0.690 |
| OpenBookQA | acc_norm | 0.404 | 0.276 | 0.312 | 0.312 | 0.308 | 0.316 |
| SciQ | acc_norm | 0.934 | 0.199 | 0.613 | 0.695 | 0.700 | 0.733 |
| TruthfulQA MC1 | acc | 0.305 | 0.220 | 0.241 | 0.241 | 0.233 | 0.261 |

Mean over these displayed metrics: FP `0.644`, naive PTQ `0.349`,
QAT hidden-MSE `0.465`, QAT KL-only `0.483`, QAT KL-only dense `lm_head`
`0.484`, and QAT KL-only row scale with dense `lm_head` `0.499`. The row-scale
dense-head run improves over tensor-scale dense-head by `+0.015081` macro mean
with paired 95% CI `[+0.009028, +0.021134]`; it improves over all-ternary
KL-only by `+0.016021` with 95% CI `[+0.006145, +0.025897]`. It remains far
below FP by about `-0.14471` macro mean.

### Fast MC200 Dense-Head Check

The in-repo multiple-choice harness was rerun on 200-example validation slices
for the Qwen2.5-1.5B KL-only dense-`lm_head` checkpoint. This is a regression
check, not a replacement for the uncapped `lm-eval` table above.

| task | acc | acc_norm | n |
| --- | ---: | ---: | ---: |
| PIQA | 0.640 | 0.655 | 200 |
| ARC-Easy | 0.585 | 0.450 | 200 |
| ARC-Challenge | 0.225 | 0.255 | 200 |
| HellaSwag | 0.380 | 0.410 | 200 |

### Packed GGUF CPU Runtime Snapshot

CPU runtime: Intel Xeon Silver 4116, 12 threads, `llama-bench -p 512 -n 128
-ngl 0 -r 3`, no BLAS, llama.cpp submodule commit `1f86f058`. The host CPU
supports AVX-512; build-specific AVX usage is stated for the row-scale
prototype suites. The I2_S path is a real packed GGUF CPU measurement.
Dense-control rows were created by converting HF checkpoints to F16 GGUF and
then running `llama-quantize`; static-ternary rows were created by
materializing `ternary_state_dict.pt` back into an HF-shaped checkpoint before
GGUF conversion and packing.

| source | GGUF type | file size | prefill tok/s | decode tok/s | smoke prompt result |
| --- | --- | ---: | ---: | ---: | --- |
| Qwen2.5-0.5B FP | F16 | 948 MiB | 331.82 | 16.39 | sensible |
| Qwen2.5-0.5B FP | Q8_0 | 507 MiB | 391.40 | 28.84 | sensible |
| Qwen2.5-0.5B FP | Q4_K_M | 379 MiB | 213.67 | 35.70 | sensible |
| Qwen2.5-0.5B FP | I2_S | 230 MiB | 532.24 | 53.11 | degenerate punctuation |
| Qwen2.5-0.5B QAT step-1000 | F16 | 1,208 MiB | 332.13 | 16.26 | degenerate text |
| Qwen2.5-0.5B QAT step-1000 | I2_S | 490 MiB | 525.52 | 49.97 | degenerate punctuation |
| Qwen2.5-1.5B FP | F16 | 2,950 MiB | 105.43 | 5.46 | sensible |
| Qwen2.5-1.5B FP | Q8_0 | 1,570 MiB | 132.58 | 10.09 | sensible |
| Qwen2.5-1.5B FP | Q4_K_M | 940 MiB | 94.96 | 15.73 | sensible |
| Qwen2.5-1.5B FP | TQ2_0 | 773 MiB | 160.84 | 18.38 | gibberish |
| Qwen2.5-1.5B FP | I2_S | 766 MiB | 205.17 | 18.45 | repeated-token collapse |
| Qwen2.5-1.5B QAT step-5000 | F16 | 3,396 MiB | 105.21 | 5.52 | degenerate text |
| Qwen2.5-1.5B QAT step-5000 | I2_S | 1,211 MiB | 203.59 | 17.97 | repeated-token collapse |
| Qwen2.5-1.5B static ternary | F16 materialized | 3,396 MiB | 104.54 | 5.50 | sensible |
| Qwen2.5-1.5B static ternary | TQ2_0 | 1,219 MiB | 160.94 | 18.39 | sensible |
| Qwen2.5-1.5B static ternary | I2_S single-thread quant | 1,209 MiB | 206.15 | 18.58 | sensible |
| Qwen2.5-1.5B KL-only static ternary | F16 materialized | 3,396 MiB | 105.34 | 5.50 | sensible |
| Qwen2.5-1.5B KL-only static ternary | TQ2_0 | 1,219 MiB | 160.93 | 18.43 | sensible |
| Qwen2.5-1.5B KL-only static ternary | I2_S single-thread quant | 1,209 MiB | 205.76 | 18.60 | sensible |
| Qwen2.5-1.5B KL-only static ternary | I2_S patched 12-thread quant | 1,209 MiB | 208.10 | 18.63 | sensible |

Interpretation: the CPU backend can execute packed I2_S quickly on this 2017
Xeon. The blocking problem is quality, not kernel availability. Blind I2_S
does not preserve the dense FP checkpoint, but single-thread I2_S preserves the
trained static-ternary artifact. The stronger KL-only static ternary checkpoint
also survives GGUF materialization and packed ternary quantization, reducing
the fixed-excerpt I2_S PPL from `84.5277` to `54.7366` while keeping decode
throughput at about `18.6` tok/s. Q4_K_M should be read with care for
Qwen2.5-0.5B because many tensors require fallback quantization due column
divisibility constraints; Qwen2.5-1.5B did not report that fallback warning.

### Packed GGUF Perplexity Snapshot

`llama-perplexity` was run on a fixed 16-chunk WikiText-2 test excerpt
(8,192 tokens, `-c 512`, 12 threads) for Qwen2.5-1.5B GGUF artifacts.

| source | GGUF type | WikiText excerpt PPL | prompt-eval tok/s |
| --- | --- | ---: | ---: |
| Qwen2.5-1.5B FP | F16 | 12.2806 | 84.13 |
| Qwen2.5-1.5B FP | Q8_0 | 12.3207 | 104.77 |
| Qwen2.5-1.5B FP | Q4_K_M | 12.8452 | 75.66 |
| Qwen2.5-1.5B FP | TQ2_0 | 18,041,439.0235 | 113.43 |
| Qwen2.5-1.5B FP | I2_S | 1.206e51 | 139.16 |
| Qwen2.5-1.5B QAT step-5000 dense GGUF | F16 | 2728.9322 | 83.79 |
| Qwen2.5-1.5B QAT step-5000 dense GGUF | I2_S | 7.619e59 | 137.73 |
| Qwen2.5-1.5B static ternary | F16 materialized | 83.8300 | 77.11 |
| Qwen2.5-1.5B static ternary | TQ2_0 | 84.0553 | 116.64 |
| Qwen2.5-1.5B static ternary | I2_S single-thread quant | 84.5277 | 140.13 |
| Qwen2.5-1.5B KL-only static ternary | F16 materialized | 55.0971 | 82.65 |
| Qwen2.5-1.5B KL-only static ternary | TQ2_0 | 55.1562 | 116.16 |
| Qwen2.5-1.5B KL-only static ternary | I2_S single-thread quant | 54.7366 | 140.95 |
| Qwen2.5-1.5B KL-only static ternary | I2_S patched 12-thread quant | 54.7366 | 141.14 |

A later AMD 5945WX portable-AVX2 suite tested the stronger dense-head and
row-scale dense-head bridges on the same fixed WikiText excerpt:

| source | GGUF type | WikiText excerpt PPL | prompt-eval tok/s | decode tok/s |
| --- | --- | ---: | ---: | ---: |
| Qwen2.5-1.5B KL-only dense-`lm_head` static ternary | I2_S single-thread quant | 47.3435 | 464.19 | 45.50 |
| Qwen2.5-1.5B KL-only row-scale dense-`lm_head` static ternary | TQ2_0 | 38.8224 | 345.32 | 44.85 |
| Qwen2.5-1.5B KL-only row-scale dense-`lm_head` static ternary | I2_S single-thread quant | 1.197e6 | 465.34 | 46.13 |

The packed-runtime conclusion is now narrower: conventional Q8_0 and Q4_K_M
retain the FP likelihood, while blind ternarization destroys it. Materializing
`ternary_state_dict.pt` as dense F16 recovers the static-ternary quality, and
llama.cpp `TQ2_0` preserves both tensor-scale and row-scale static ternary
artifacts. Tensor-scale `I2_S` also preserves static-ternary quality, but the
current row-scale-to-`I2_S` bridge fails; row-scale deployment needs a
row-scale-aware packed ternary writer and kernel rather than the current
I2_S path. The included `patches/llama-i2s-row-scale.patch` proves the local
format fix by recovering row-scale `I2_S` PPL `38.8832` with `218.17` prompt
tok/s and `18.97` decode tok/s on the Xeon portable-AVX2 heap-fix
confirmation. The included
native AVX-512-enabled run preserves quality but is slightly slower for this
kernel (`207.35` prompt tok/s / `18.37` decode tok/s), so CPU-feature-based
speed claims still need kernel-specific measurement. Thread scaling shows
prompt ingestion scaling to `245.31` tok/s at 24 threads, but decode stays near
`18-20` tok/s after 4 threads. The row-scale patch also fixes the earlier
low-thread `llama-bench` crash by moving large I2_S prompt temporary buffers
from stack to heap. The included
`patches/llama-i2s-threaded-quantization.patch` fixes
the tensor-scale threaded I2_S packing path locally: a 12-thread quantized
artifact matches the single-thread PPL (`54.7366`) and reaches `208.10` prompt
tok/s / `18.63` decode tok/s. The safe wrapper remains conservative until the
submodule is advanced to a commit with that fix.

### Xeon PyTorch Runtime Probe

CPU probe: Intel Xeon Silver 4116, 12 physical cores / 24 threads, AVX-512,
PyTorch FP32, 12 Torch threads, 512-token prompt, 32 generated tokens, median of
three measured repeats. These are PyTorch loader numbers, not packed
`bitnet.cpp` kernel numbers.

| model | method | prefill tok/s | gen tok/s | RSS GiB | model GiB | ternary GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B | FP reference | 330.69 | 5.20 | 2.716 | 1.840 | - |
| Qwen2.5-0.5B | naive PTQ ternary | 244.82 | 2.03 | 3.370 | 0.841 | 0.587 |
| Qwen2.5-0.5B | QAT/distilled ternary | 219.71 | 1.41 | 2.173 | 0.967 | 0.968 |
| Qwen2.5-0.5B | QAT KL-only, dense tied `lm_head` | 226.33 | 1.85 | 2.048 | 0.841 | 1.348 |
| Qwen2.5-1.5B | FP reference | 118.74 | 1.95 | 6.631 | 5.751 | - |
| Qwen2.5-1.5B | naive PTQ ternary | 79.93 | 0.48 | 4.748 | 2.090 | 1.655 |
| Qwen2.5-1.5B | QAT/distilled ternary | 74.34 | 0.41 | 4.405 | 2.307 | 2.308 |

### Current Task-Accuracy Snapshot

The in-repo multiple-choice evaluator covers 100-example validation slices
for PIQA, ARC-Easy, ARC-Challenge, and HellaSwag. This is a fast regression
tool; the official `lm-eval` snapshot above is the stronger current evidence.
This older 100-example table is kept for historical comparison with the first
Qwen2.5-1.5B hidden-MSE QAT checkpoint:

| task | FP Qwen2.5-1.5B acc | naive PTQ acc | QAT ternary acc |
| --- | ---: | ---: | ---: |
| PIQA | 0.760 | 0.550 | 0.650 |
| ARC-Easy | 0.760 | 0.300 | 0.550 |
| ARC-Challenge | 0.440 | 0.190 | 0.320 |
| HellaSwag | 0.470 | 0.230 | 0.360 |

<img src="./assets/performance.png" alt="performance_comparison" width="800"/>


## Demo

A demo of bitnet.cpp running a BitNet b1.58 3B model on Apple M2:

https://github.com/user-attachments/assets/7f46b736-edec-4828-b809-4be780a3e5b1

## What's New:
- 01/15/2026 [BitNet CPU Inference Optimization](https://github.com/microsoft/BitNet/blob/main/src/README.md) ![NEW](https://img.shields.io/badge/NEW-red)
- 05/20/2025 [BitNet Official GPU inference kernel](https://github.com/microsoft/BitNet/blob/main/gpu/README.md)
- 04/14/2025 [BitNet Official 2B Parameter Model on Hugging Face](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T)
- 02/18/2025 [Bitnet.cpp: Efficient Edge Inference for Ternary LLMs](https://arxiv.org/abs/2502.11880)
- 11/08/2024 [BitNet a4.8: 4-bit Activations for 1-bit LLMs](https://arxiv.org/abs/2411.04965)
- 10/21/2024 [1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1.58 Inference on CPUs](https://arxiv.org/abs/2410.16144)
- 10/17/2024 bitnet.cpp 1.0 released.
- 03/21/2024 [The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ](https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf)
- 02/27/2024 [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)
- 10/17/2023 [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453)

## Acknowledgements

This project is based on the [llama.cpp](https://github.com/ggerganov/llama.cpp) framework. We would like to thank all the authors for their contributions to the open-source community. Also, bitnet.cpp's kernels are built on top of the Lookup Table methodologies pioneered in [T-MAC](https://github.com/microsoft/T-MAC/). For inference of general low-bit LLMs beyond ternary models, we recommend using T-MAC.
## Official Models
<table>
    </tr>
    <tr>
        <th rowspan="2">Model</th>
        <th rowspan="2">Parameters</th>
        <th rowspan="2">CPU</th>
        <th colspan="3">Kernel</th>
    </tr>
    <tr>
        <th>I2_S</th>
        <th>TL1</th>
        <th>TL2</th>
    </tr>
    <tr>
        <td rowspan="2"><a href="https://huggingface.co/microsoft/BitNet-b1.58-2B-4T">BitNet-b1.58-2B-4T</a></td>
        <td rowspan="2">2.4B</td>
        <td>x86</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
        <td>&#9989;</td>
    </tr>
    <tr>
        <td>ARM</td>
        <td>&#9989;</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
    </tr>
</table>

## Supported Models
❗️**We use existing 1-bit LLMs available on [Hugging Face](https://huggingface.co/) to demonstrate the inference capabilities of bitnet.cpp. We hope the release of bitnet.cpp will inspire the development of 1-bit LLMs in large-scale settings in terms of model size and training tokens.**

<table>
    </tr>
    <tr>
        <th rowspan="2">Model</th>
        <th rowspan="2">Parameters</th>
        <th rowspan="2">CPU</th>
        <th colspan="3">Kernel</th>
    </tr>
    <tr>
        <th>I2_S</th>
        <th>TL1</th>
        <th>TL2</th>
    </tr>
    <tr>
        <td rowspan="2"><a href="https://huggingface.co/1bitLLM/bitnet_b1_58-large">bitnet_b1_58-large</a></td>
        <td rowspan="2">0.7B</td>
        <td>x86</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
        <td>&#9989;</td>
    </tr>
    <tr>
        <td>ARM</td>
        <td>&#9989;</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
    </tr>
    <tr>
        <td rowspan="2"><a href="https://huggingface.co/1bitLLM/bitnet_b1_58-3B">bitnet_b1_58-3B</a></td>
        <td rowspan="2">3.3B</td>
        <td>x86</td>
        <td>&#10060;</td>
        <td>&#10060;</td>
        <td>&#9989;</td>
    </tr>
    <tr>
        <td>ARM</td>
        <td>&#10060;</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
    </tr>
    <tr>
        <td rowspan="2"><a href="https://huggingface.co/HF1BitLLM/Llama3-8B-1.58-100B-tokens">Llama3-8B-1.58-100B-tokens</a></td>
        <td rowspan="2">8.0B</td>
        <td>x86</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
        <td>&#9989;</td>
    </tr>
    <tr>
        <td>ARM</td>
        <td>&#9989;</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
    </tr>
    <tr>
        <td rowspan="2"><a href="https://huggingface.co/collections/tiiuae/falcon3-67605ae03578be86e4e87026">Falcon3 Family</a></td>
        <td rowspan="2">1B-10B</td>
        <td>x86</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
        <td>&#9989;</td>
    </tr>
    <tr>
        <td>ARM</td>
        <td>&#9989;</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
    </tr>
    <tr>
        <td rowspan="2"><a href="https://huggingface.co/collections/tiiuae/falcon-edge-series-6804fd13344d6d8a8fa71130">Falcon-E Family</a></td>
        <td rowspan="2">1B-3B</td>
        <td>x86</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
        <td>&#9989;</td>
    </tr>
    <tr>
        <td>ARM</td>
        <td>&#9989;</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
    </tr>
</table>



## Installation

### Requirements
- python>=3.9
- cmake>=3.22
- clang>=18
    - For Windows users, install [Visual Studio 2022](https://visualstudio.microsoft.com/downloads/). In the installer, toggle on at least the following options(this also automatically installs the required additional tools like CMake):
        -  Desktop-development with C++
        -  C++-CMake Tools for Windows
        -  Git for Windows
        -  C++-Clang Compiler for Windows
        -  MS-Build Support for LLVM-Toolset (clang)
    - For Debian/Ubuntu users, you can download with [Automatic installation script](https://apt.llvm.org/)

        `bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"`
- conda (highly recommend)

### Build from source

> [!IMPORTANT]
> If you are using Windows, please remember to always use a Developer Command Prompt / PowerShell for VS2022 for the following commands. Please refer to the FAQs below if you see any issues.

1. Clone the repo
```bash
git clone --recursive https://github.com/microsoft/BitNet.git
cd BitNet
```
2. Install the dependencies
```bash
# (Recommended) Create a new conda environment
conda create -n bitnet-cpp python=3.9
conda activate bitnet-cpp

pip install -r requirements.txt
```
3. Build the project
```bash
# Manually download the model and run with local path
huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir models/BitNet-b1.58-2B-4T
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s

```
<pre>
usage: setup_env.py [-h] [--hf-repo {1bitLLM/bitnet_b1_58-large,1bitLLM/bitnet_b1_58-3B,HF1BitLLM/Llama3-8B-1.58-100B-tokens,tiiuae/Falcon3-1B-Instruct-1.58bit,tiiuae/Falcon3-3B-Instruct-1.58bit,tiiuae/Falcon3-7B-Instruct-1.58bit,tiiuae/Falcon3-10B-Instruct-1.58bit}] [--model-dir MODEL_DIR] [--log-dir LOG_DIR] [--quant-type {i2_s,tl1}] [--quant-embd]
                    [--use-pretuned]

Setup the environment for running inference

optional arguments:
  -h, --help            show this help message and exit
  --hf-repo {1bitLLM/bitnet_b1_58-large,1bitLLM/bitnet_b1_58-3B,HF1BitLLM/Llama3-8B-1.58-100B-tokens,tiiuae/Falcon3-1B-Instruct-1.58bit,tiiuae/Falcon3-3B-Instruct-1.58bit,tiiuae/Falcon3-7B-Instruct-1.58bit,tiiuae/Falcon3-10B-Instruct-1.58bit}, -hr {1bitLLM/bitnet_b1_58-large,1bitLLM/bitnet_b1_58-3B,HF1BitLLM/Llama3-8B-1.58-100B-tokens,tiiuae/Falcon3-1B-Instruct-1.58bit,tiiuae/Falcon3-3B-Instruct-1.58bit,tiiuae/Falcon3-7B-Instruct-1.58bit,tiiuae/Falcon3-10B-Instruct-1.58bit}
                        Model used for inference
  --model-dir MODEL_DIR, -md MODEL_DIR
                        Directory to save/load the model
  --log-dir LOG_DIR, -ld LOG_DIR
                        Directory to save the logging info
  --quant-type {i2_s,tl1}, -q {i2_s,tl1}
                        Quantization type
  --quant-embd          Quantize the embeddings to f16
  --use-pretuned, -p    Use the pretuned kernel parameters
</pre>
## Usage
### Basic usage
```bash
# Run inference with the quantized model
python run_inference.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -p "You are a helpful assistant" -cnv
```
<pre>
usage: run_inference.py [-h] [-m MODEL] [-n N_PREDICT] -p PROMPT [-t THREADS] [-c CTX_SIZE] [-temp TEMPERATURE] [-cnv]

Run inference

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Path to model file
  -n N_PREDICT, --n-predict N_PREDICT
                        Number of tokens to predict when generating text
  -p PROMPT, --prompt PROMPT
                        Prompt to generate text from
  -t THREADS, --threads THREADS
                        Number of threads to use
  -c CTX_SIZE, --ctx-size CTX_SIZE
                        Size of the prompt context
  -temp TEMPERATURE, --temperature TEMPERATURE
                        Temperature, a hyperparameter that controls the randomness of the generated text
  -cnv, --conversation  Whether to enable chat mode or not (for instruct models.)
                        (When this option is turned on, the prompt specified by -p will be used as the system prompt.)
</pre>

### Benchmark
We provide scripts to run the inference benchmark providing a model.

```  
usage: e2e_benchmark.py -m MODEL [-n N_TOKEN] [-p N_PROMPT] [-t THREADS]  
   
Setup the environment for running the inference  
   
required arguments:  
  -m MODEL, --model MODEL  
                        Path to the model file. 
   
optional arguments:  
  -h, --help  
                        Show this help message and exit. 
  -n N_TOKEN, --n-token N_TOKEN  
                        Number of generated tokens. 
  -p N_PROMPT, --n-prompt N_PROMPT  
                        Prompt to generate text from. 
  -t THREADS, --threads THREADS  
                        Number of threads to use. 
```  
   
Here's a brief explanation of each argument:  
   
- `-m`, `--model`: The path to the model file. This is a required argument that must be provided when running the script.  
- `-n`, `--n-token`: The number of tokens to generate during the inference. It is an optional argument with a default value of 128.  
- `-p`, `--n-prompt`: The number of prompt tokens to use for generating text. This is an optional argument with a default value of 512.  
- `-t`, `--threads`: The number of threads to use for running the inference. It is an optional argument with a default value of 2.  
- `-h`, `--help`: Show the help message and exit. Use this argument to display usage information.  
   
For example:  
   
```sh  
python utils/e2e_benchmark.py -m /path/to/model -n 200 -p 256 -t 4  
```  
   
This command would run the inference benchmark using the model located at `/path/to/model`, generating 200 tokens from a 256 token prompt, utilizing 4 threads.  

For the model layout that do not supported by any public model, we provide scripts to generate a dummy model with the given model layout, and run the benchmark on your machine:

```bash
python utils/generate-dummy-bitnet-model.py models/bitnet_b1_58-large --outfile models/dummy-bitnet-125m.tl1.gguf --outtype tl1 --model-size 125M

# Run benchmark with the generated model, use -m to specify the model path, -p to specify the prompt processed, -n to specify the number of token to generate
python utils/e2e_benchmark.py -m models/dummy-bitnet-125m.tl1.gguf -p 512 -n 128
```

### Convert from `.safetensors` Checkpoints

```sh
# Prepare the .safetensors model file
huggingface-cli download microsoft/bitnet-b1.58-2B-4T-bf16 --local-dir ./models/bitnet-b1.58-2B-4T-bf16

# Convert to gguf model
python ./utils/convert-helper-bitnet.py ./models/bitnet-b1.58-2B-4T-bf16
```

### FAQ (Frequently Asked Questions)📌 

#### Q1: The build dies with errors building llama.cpp due to issues with std::chrono in log.cpp?

**A:**
This is an issue introduced in recent version of llama.cpp. Please refer to this [commit](https://github.com/tinglou/llama.cpp/commit/4e3db1e3d78cc1bcd22bcb3af54bd2a4628dd323) in the [discussion](https://github.com/abetlen/llama-cpp-python/issues/1942) to fix this issue.

#### Q2: How to build with clang in conda environment on windows?

**A:** 
Before building the project, verify your clang installation and access to Visual Studio tools by running:
```
clang -v
```

This command checks that you are using the correct version of clang and that the Visual Studio tools are available. If you see an error message such as:
```
'clang' is not recognized as an internal or external command, operable program or batch file.
```

It indicates that your command line window is not properly initialized for Visual Studio tools.

• If you are using Command Prompt, run:
```
"C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat" -startdir=none -arch=x64 -host_arch=x64
```

• If you are using Windows PowerShell, run the following commands:
```
Import-Module "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\Microsoft.VisualStudio.DevShell.dll" Enter-VsDevShell 3f0e31ad -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64"
```

These steps will initialize your environment and allow you to use the correct Visual Studio tools.
