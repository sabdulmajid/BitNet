# Qwen W1.58A8 Retrofit Results, 2026-05-03

This note records the current evidence for retrofitting pretrained Qwen dense
models into BitNet-style W1.58A8 checkpoints. It is intentionally conservative:
these are quality and loader results, not final CPU product benchmarks.

## Setup

- Benchmark tooling commit before this report update: `61f140e`
- Packed runtime binary: llama.cpp submodule commit `1f86f058`
- Dataset for QAT/distillation: `HuggingFaceFW/fineweb-edu`, `sample-10BT`
- QAT student forward math: ternary weights plus dynamic 8-bit activation quantization
- QAT loss: teacher KL divergence plus last-hidden-state MSE
- Main scale mode: per-tensor absmean scale
- Additional ablation: per-output-row absmean scale for Qwen2.5-0.5B
- Evaluation dtype/device for perplexity: BF16 on CUDA
- WikiText eval: `wikitext-2-raw-v1`, test split, 64 blocks x 512 tokens
- FineWeb heldout eval: `sample-10BT`, train stream after `--skip-rows 25000`, 32 blocks x 1024 tokens
- Official task eval: EleutherAI `lm-eval` 0.4.11, 100-example ten-task
  slices, 1000-example five-task core slices, and uncapped five-task core runs
- PyTorch runtime probe: Intel Xeon Silver 4116, PyTorch FP32, 12 Torch threads,
  512-token prompt, 32 generated tokens
- Packed GGUF runtime probe: Intel Xeon Silver 4116, `llama-bench -p 512
  -n 128 -t 12 -ngl 0 -r 3`, no BLAS, AVX-512 available

The FineWeb heldout skip is beyond the 1.5B training prefix. Job 9730 packed
19,968 rows, so skipping 25,000 rows avoids direct train-prefix reuse for this
smoke-scale heldout test.

## Completed Training Runs

| run | model | steps | data blocks | final loss | final KL | final hidden MSE | status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 9730 | Qwen2.5-1.5B QAT student | 5000 | 20,000 x 1024 | 4.0420 | 1.7478 | 2.2943 | complete |
| 9734 | Qwen2.5-0.5B QAT student | 1000 | 4,000 x 512 | 24.6557 | 2.8857 | 21.7699 | complete |
| 9747 | Qwen2.5-0.5B QAT student, row scale | 1000 | 4,000 x 512 | 15.3338 | 2.0530 | 13.2808 | complete |

## Export Status

| checkpoint | expected ternary keys | observed ternary keys | notes |
| --- | ---: | ---: | --- |
| `qwen2.5-0.5b-fineweb-edu-12/step-1000` | 169 | 169 | complete QAT ternary export |
| `qwen2.5-0.5b-fineweb-edu-row-1000/step-1000` | 169 | 169 | row-scale QAT ternary export; one scale per output row |
| `qwen2.5-1.5b-fineweb-edu/step-5000` | 197 | 197 | repaired after FSDP name-mapping bug |
| `qwen2.5-0.5b-naive-ptq-tensor` | 168 | 168 | original HF checkpoint has tied `lm_head` |
| `qwen2.5-1.5b-naive-ptq-tensor` | 196 | 196 | original HF checkpoint has tied `lm_head` |

## Perplexity

Lower is better.

| model | method | WikiText PPL | FineWeb-heldout PPL | interpretation |
| --- | --- | ---: | ---: | --- |
| Qwen2.5-0.5B | FP reference | 20.461 | 14.124 | baseline quality |
| Qwen2.5-0.5B | naive PTQ ternary | 169,414.428 | 608,726.749 | blind ternarization destroys quality |
| Qwen2.5-0.5B | QAT/distilled ternary | 1,079.167 | 373.775 | QAT recovers a lot versus PTQ, but remains unusable |
| Qwen2.5-0.5B | QAT/distilled ternary, row scale | 444.691 | 152.821 | row scales improve PPL about 2.4x versus tensor-scale QAT |
| Qwen2.5-1.5B | FP reference | 13.901 | 10.269 | baseline quality |
| Qwen2.5-1.5B | naive PTQ ternary | 3,813,121.803 | 9,582,923.269 | blind ternarization destroys quality |
| Qwen2.5-1.5B | QAT/distilled ternary | 86.414 | 40.398 | QAT is orders better than PTQ, but still far from FP |

The row-scale ablation is meaningful but not decisive. It reduces 0.5B
WikiText PPL from `1,079.167` to `444.691` and FineWeb-heldout PPL from
`373.775` to `152.821`, but it remains far from the FP reference and does not
yet establish downstream quality.

## Prompt-Suite Sanity Check

The deterministic prompt suite loads the ternary checkpoints and confirms
autoregressive generation works through `StaticTernaryLinear`. It is not a
publishable quality benchmark.

Observed behavior:

- 0.5B QAT ternary output is mostly repetitive and low quality.
- 1.5B QAT ternary output is more coherent on generic continuation and
  summarization prompts, but fails simple arithmetic and code prompts.

## Multiple-Choice Accuracy

These are 100-example validation slices using the in-repo log-likelihood
multiple-choice evaluator, not EleutherAI `lm-eval`. They are useful for fast
triage and regression detection. They should be replaced by full `lm-eval`
runs before making paper-grade claims.

Accuracy-normalized (`acc_norm`) is included because continuation length can
change raw log-likelihood rankings.

| task | model | method | acc | acc_norm |
| --- | --- | --- | ---: | ---: |
| PIQA | Qwen2.5-0.5B | FP reference | 0.700 | 0.720 |
| PIQA | Qwen2.5-0.5B | naive PTQ ternary | 0.530 | 0.490 |
| PIQA | Qwen2.5-0.5B | QAT/distilled ternary | 0.520 | 0.500 |
| PIQA | Qwen2.5-1.5B | FP reference | 0.760 | 0.780 |
| PIQA | Qwen2.5-1.5B | naive PTQ ternary | 0.550 | 0.590 |
| PIQA | Qwen2.5-1.5B | QAT/distilled ternary | 0.650 | 0.670 |
| ARC-Easy | Qwen2.5-0.5B | FP reference | 0.680 | 0.580 |
| ARC-Easy | Qwen2.5-0.5B | naive PTQ ternary | 0.290 | 0.270 |
| ARC-Easy | Qwen2.5-0.5B | QAT/distilled ternary | 0.290 | 0.270 |
| ARC-Easy | Qwen2.5-1.5B | FP reference | 0.760 | 0.680 |
| ARC-Easy | Qwen2.5-1.5B | naive PTQ ternary | 0.300 | 0.260 |
| ARC-Easy | Qwen2.5-1.5B | QAT/distilled ternary | 0.550 | 0.410 |
| ARC-Challenge | Qwen2.5-0.5B | FP reference | 0.340 | 0.370 |
| ARC-Challenge | Qwen2.5-0.5B | naive PTQ ternary | 0.190 | 0.200 |
| ARC-Challenge | Qwen2.5-0.5B | QAT/distilled ternary | 0.290 | 0.300 |
| ARC-Challenge | Qwen2.5-1.5B | FP reference | 0.440 | 0.450 |
| ARC-Challenge | Qwen2.5-1.5B | naive PTQ ternary | 0.190 | 0.280 |
| ARC-Challenge | Qwen2.5-1.5B | QAT/distilled ternary | 0.320 | 0.330 |
| HellaSwag | Qwen2.5-0.5B | FP reference | 0.400 | 0.430 |
| HellaSwag | Qwen2.5-0.5B | naive PTQ ternary | 0.250 | 0.300 |
| HellaSwag | Qwen2.5-0.5B | QAT/distilled ternary | 0.310 | 0.260 |
| HellaSwag | Qwen2.5-1.5B | FP reference | 0.470 | 0.570 |
| HellaSwag | Qwen2.5-1.5B | naive PTQ ternary | 0.230 | 0.180 |
| HellaSwag | Qwen2.5-1.5B | QAT/distilled ternary | 0.360 | 0.390 |

The Qwen2.5-0.5B row-scale ablation was also checked on the same in-repo
100-example slices:

| task | tensor-scale QAT acc | tensor-scale QAT acc_norm | row-scale QAT acc | row-scale QAT acc_norm |
| --- | ---: | ---: | ---: | ---: |
| PIQA | 0.520 | 0.500 | 0.480 | 0.460 |
| ARC-Easy | 0.290 | 0.270 | 0.300 | 0.260 |
| ARC-Challenge | 0.290 | 0.300 | 0.240 | 0.240 |
| HellaSwag | 0.310 | 0.260 | 0.340 | 0.240 |

This is a cautionary ablation: row-wise scales improve likelihood, but the
small multiple-choice slices are mixed and do not justify a quality claim.

## Official lm-eval Accuracy

These are EleutherAI `lm-eval` 0.4.11 results for Qwen2.5-1.5B on ten
100-example task slices. Where a task reports `acc_norm`, that metric is shown;
otherwise raw `acc` is shown. This is still not a full official benchmark run,
but it uses the standard lm-eval task implementations and result format.

| task | metric | FP reference | naive PTQ ternary | QAT/distilled ternary |
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

Mean displayed metric:

| method | mean |
| --- | ---: |
| FP reference | 0.646 |
| naive PTQ ternary | 0.351 |
| QAT/distilled ternary | 0.490 |

## Larger Core lm-eval Accuracy

The same Qwen2.5-1.5B comparison was rerun with 1000-example caps for five
core tasks. This is still capped, but it is a stronger estimate than the
100-example smoke slice above.

| task | metric | FP reference | naive PTQ ternary | QAT/distilled ternary |
| --- | --- | ---: | ---: | ---: |
| ARC-Challenge | acc_norm | 0.448 | 0.266 | 0.259 |
| ARC-Easy | acc_norm | 0.713 | 0.246 | 0.486 |
| HellaSwag | acc_norm | 0.584 | 0.262 | 0.389 |
| PIQA | acc_norm | 0.756 | 0.510 | 0.614 |
| WinoGrande | acc | 0.647 | 0.509 | 0.531 |

Mean displayed metric:

| method | mean |
| --- | ---: |
| FP reference | 0.630 |
| naive PTQ ternary | 0.359 |
| QAT/distilled ternary | 0.456 |

QAT/distillation improves the mean by 0.097 over naive PTQ and recovers about
36% of the FP-vs-PTQ gap on these five capped tasks. It still remains far below
the FP reference, and ARC-Challenge is a small counterexample where QAT is
slightly below naive PTQ on `acc_norm`.

Because the lm-eval runs log per-example samples, the same five-task slice can
be evaluated as paired deltas on exactly matched examples:

| comparison | macro mean delta | paired 95% CI | interpretation |
| --- | ---: | ---: | --- |
| QAT minus naive PTQ | +0.097 | [+0.012, +0.182] | QAT/distillation is a real recovery signal on this capped slice |
| QAT minus FP reference | -0.174 | [-0.213, -0.135] | QAT remains decisively below the dense teacher |

The paired result strengthens the negative verdict: this is not random table
noise, and it is not enough evidence for arbitrary lossless retrofit.

## Full Core lm-eval Accuracy

The same five core tasks were then rerun with `LIMIT=0`, so lm-eval used the
full task splits rather than the 1000-example cap. This is the strongest
task-accuracy comparison in the current artifact set.

| task | metric | FP reference | naive PTQ ternary | QAT/distilled ternary |
| --- | --- | ---: | ---: | ---: |
| ARC-Challenge | acc_norm | 0.449659 | 0.261945 | 0.263652 |
| ARC-Easy | acc_norm | 0.719697 | 0.244108 | 0.478114 |
| HellaSwag | acc_norm | 0.677953 | 0.264190 | 0.362378 |
| PIQA | acc_norm | 0.757889 | 0.507617 | 0.621872 |
| WinoGrande | acc | 0.637727 | 0.498027 | 0.523283 |

Mean displayed metric:

| method | mean |
| --- | ---: |
| FP reference | 0.648585 |
| naive PTQ ternary | 0.355177 |
| QAT/distilled ternary | 0.449860 |

Paired deltas on matched examples:

| comparison | macro mean delta | paired 95% CI | example-weighted delta |
| --- | ---: | ---: | ---: |
| QAT minus naive PTQ | +0.094682 | [+0.014740, +0.174624] | +0.106978 |
| QAT minus FP reference | -0.198725 | [-0.270323, -0.127127] | -0.260916 |

QAT/distillation recovers about 32% of the FP-vs-naive-PTQ macro gap on this
full five-task run. The recovery is real, but the absolute gap to FP remains
large, especially on HellaSwag and ARC-Easy. The full run therefore confirms
the same core verdict as the capped slice: training under ternary constraints
is necessary and useful, but the current QAT recipe is not yet FP-quality.

## Xeon PyTorch Runtime Probe

These are CPU measurements on the Intel Xeon Silver 4116 host with PyTorch
FP32 and 12 Torch threads. They are intentionally labeled as PyTorch probe
numbers because the ternary path dequantizes into dense PyTorch matmuls. They
do not measure packed `bitnet.cpp` TL2/I2_S kernels.

| model | method | prefill tok/s | gen tok/s | RSS GiB | model GiB | ternary GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B | FP reference | 330.69 | 5.20 | 2.716 | 1.840 | - |
| Qwen2.5-0.5B | naive PTQ ternary | 244.82 | 2.03 | 3.370 | 0.841 | 0.587 |
| Qwen2.5-0.5B | QAT/distilled ternary | 219.71 | 1.41 | 2.173 | 0.967 | 0.968 |
| Qwen2.5-1.5B | FP reference | 118.74 | 1.95 | 6.631 | 5.751 | - |
| Qwen2.5-1.5B | naive PTQ ternary | 79.93 | 0.48 | 4.748 | 2.090 | 1.655 |
| Qwen2.5-1.5B | QAT/distilled ternary | 74.34 | 0.41 | 4.405 | 2.307 | 2.308 |

## Packed GGUF CPU Runtime Probe

These are actual `llama-bench` CPU measurements through the GGUF runtime. The
Qwen2.5-0.5B QAT GGUF was created from the dense `model.safetensors` file in
the QAT checkpoint and then quantized with `llama-quantize`; this is not yet a
bit-exact loader for `ternary_state_dict.pt`.

| source | GGUF type | file size | prefill tok/s | decode tok/s | smoke prompt behavior |
| --- | --- | ---: | ---: | ---: | --- |
| Qwen2.5-0.5B FP | F16 | 948 MiB | 331.82 | 16.39 | sensible completion |
| Qwen2.5-0.5B FP | Q8_0 | 507 MiB | 391.40 | 28.84 | sensible completion |
| Qwen2.5-0.5B FP | Q4_K_M | 379 MiB | 213.67 | 35.70 | sensible completion |
| Qwen2.5-0.5B FP | I2_S | 230 MiB | 532.24 | 53.11 | punctuation collapse |
| Qwen2.5-0.5B QAT step-1000 | F16 | 1,208 MiB | 332.13 | 16.26 | degenerate text |
| Qwen2.5-0.5B QAT step-1000 | I2_S | 490 MiB | 525.52 | 49.97 | punctuation collapse |
| Qwen2.5-1.5B FP | F16 | 2,950 MiB | 105.43 | 5.46 | sensible completion |
| Qwen2.5-1.5B FP | Q8_0 | 1,570 MiB | 132.58 | 10.09 | sensible completion |
| Qwen2.5-1.5B FP | Q4_K_M | 940 MiB | 94.96 | 15.73 | sensible completion |
| Qwen2.5-1.5B FP | TQ2_0 | 773 MiB | 160.84 | 18.38 | gibberish |
| Qwen2.5-1.5B FP | I2_S | 766 MiB | 205.17 | 18.45 | repeated-token collapse |
| Qwen2.5-1.5B QAT step-5000 | F16 | 3,396 MiB | 105.21 | 5.52 | degenerate text |
| Qwen2.5-1.5B QAT step-5000 | I2_S | 1,211 MiB | 203.59 | 17.97 | repeated-token collapse |
| Qwen2.5-1.5B static ternary | F16 materialized | 3,396 MiB | 104.54 | 5.50 | sensible completion |
| Qwen2.5-1.5B static ternary | TQ2_0 | 1,219 MiB | 160.94 | 18.39 | sensible completion |
| Qwen2.5-1.5B static ternary | I2_S single-thread quant | 1,209 MiB | 206.15 | 18.58 | sensible completion |

Smoke prompt: `The capital of France is`, greedy decoding, 24 generated tokens.
The FP/Q8_0/Q4_K_M controls complete with `Paris` and related capital-city
continuations. The 0.5B I2_S artifacts collapse to repeated exclamation marks;
the blind 1.5B I2_S artifact collapses to repeated `is` tokens. The
single-thread-written static-ternary I2_S artifact produces a sensible
Paris/French-government continuation.

Important caveats:

- I2_S is fast on this CPU, so the execution layer is not the bottleneck for a
  Qwen-shaped model.
- Blind I2_S conversion is not quality-preserving, but I2_S can preserve a
  trained static-ternary artifact when the GGUF is written correctly.
- Q4_K_M is not a pure Q4_K_M baseline for the Qwen2.5-0.5B shape: 144 of 168
  tensors required fallback quantization in the original FP conversion. The
  Qwen2.5-1.5B Q4_K_M conversions did not report fallback warnings.
- The QAT 0.5B checkpoint is intrinsically weak, which matches its high
  perplexity. The 1.5B QAT checkpoint is stronger under the PyTorch
  static-ternary path, but dense GGUF export from `model.safetensors` is not a
  valid proxy for that static ternary artifact.
- Materializing `ternary_state_dict.pt` back into dense F16 recovers the
  expected static-ternary behavior. Quantizing that materialized artifact to
  generic `TQ2_0` preserves quality. I2_S also preserves quality when the GGUF
  is written with a single quantization thread; the earlier multi-thread I2_S
  artifact was corrupted.

## Packed GGUF Perplexity Probe

`llama-perplexity` was run on a fixed WikiText-2 test excerpt generated from
123 non-empty test rows: 16 chunks, 512-token context, 8,192 evaluated tokens,
12 CPU threads, `-ngl 0`.

| source | GGUF type | PPL | stderr | prompt-eval tok/s |
| --- | --- | ---: | ---: | ---: |
| Qwen2.5-1.5B FP | F16 | 12.2806 | 0.52969 | 84.13 |
| Qwen2.5-1.5B FP | Q8_0 | 12.3207 | 0.53098 | 104.77 |
| Qwen2.5-1.5B FP | Q4_K_M | 12.8452 | 0.55781 | 75.66 |
| Qwen2.5-1.5B FP | TQ2_0 | 18,041,439.0235 | 1,022,722.63659 | 113.43 |
| Qwen2.5-1.5B FP | I2_S | 1.206e51 | 5.898e50 | 139.16 |
| Qwen2.5-1.5B QAT step-5000 dense GGUF | F16 | 2728.9322 | 262.72596 | 83.79 |
| Qwen2.5-1.5B QAT step-5000 dense GGUF | I2_S | 7.619e59 | 4.502e59 | 137.73 |
| Qwen2.5-1.5B static ternary | F16 materialized | 83.8300 | 4.60205 | 77.11 |
| Qwen2.5-1.5B static ternary | TQ2_0 | 84.0553 | 4.61363 | 116.64 |
| Qwen2.5-1.5B static ternary | I2_S single-thread quant | 84.5277 | 4.63470 | 140.13 |

Interpretation:

- Standard Q8_0 and Q4_K_M preserve the FP perplexity on this packed GGUF
  excerpt.
- Blind ternary quantization mathematically destroys language-modeling
  likelihood even though it improves CPU throughput. This is true for both
  I2_S and generic `TQ2_0` on the original dense Qwen2.5-1.5B artifact.
- Dense `model.safetensors` export from the QAT checkpoint is not the deployment
  object we need. The PyTorch QAT result relies on static ternary weights and
  scales from `ternary_state_dict.pt`; until GGUF ingests those exactly, the
  GGUF QAT numbers above should be treated as a failed conversion path rather
  than a final measure of the QAT recipe.
- A dense materialization bridge for `ternary_state_dict.pt` validates the
  exported static ternary checkpoint in GGUF F16 form: PPL 83.83, matching the
  earlier PyTorch-scale result.
- Generic llama.cpp `TQ2_0` preserves that static-ternary PPL while giving a
  2.06 bpw ternary file and about 18.38 decode tok/s on the Xeon. The same
  `TQ2_0` format does not rescue a dense model without QAT/distillation.
- Single-thread llama.cpp `I2_S` quantization also preserves that
  static-ternary PPL while giving the fastest prompt throughput in this GGUF
  slice. The earlier multi-thread I2_S artifact produced NaN PPL, which points
  to a writer/chunking bug rather than a fundamental runtime math failure.

## What This Proves

1. The pretrained Qwen dense architecture can be trained under BitNet-style
   W1.58A8 forward constraints without immediate optimization collapse.
2. The repaired export path can produce complete static ternary checkpoints for
   Qwen2.5-0.5B and Qwen2.5-1.5B QAT students.
3. Naive post-training ternarization is not viable for these pretrained dense
   checkpoints. The quality collapse is visible across both WikiText and
   FineWeb-heldout perplexity.
4. QAT/distillation substantially recovers quality relative to naive PTQ, but
   current quality is still not close enough to the FP references for a strong
   deployment claim.
5. On fast multiple-choice slices, 1.5B QAT generally recovers meaningful
   accuracy relative to naive PTQ, but still trails FP.
6. On ten 100-example lm-eval slices, 1.5B QAT improves the mean displayed task
   metric from 0.351 for naive PTQ to 0.490, while FP remains 0.646.
7. On five 1000-example core lm-eval slices, 1.5B QAT improves the mean
   displayed task metric from 0.359 for naive PTQ to 0.456, while FP remains
   0.630. This is a measurable recovery, not FP-quality preservation.
8. On the full uncapped five-task core lm-eval run, 1.5B QAT improves the mean
   displayed task metric from 0.355 for naive PTQ to 0.450, while FP remains
   0.649. The paired QAT-minus-PTQ macro delta is +0.095 with 95% CI
   [+0.015, +0.175], so the recovery is measurable but incomplete.
9. Row-wise ternary scales improve 0.5B heldout perplexity by about 2.4x versus
   tensor-scale QAT, but the small multiple-choice slices remain mixed.
10. PyTorch ternary inference is slower than FP on the Xeon probe. The product
   speed thesis depends on packed `bitnet.cpp` kernels, not on the PyTorch
   simulation path.
11. Packed I2_S GGUF execution is fast on the Xeon for Qwen2.5-0.5B-shaped
   and Qwen2.5-1.5B-shaped models, but naive I2_S conversion fails the smoke
   prompt and explodes WikiText excerpt perplexity.
12. A deployable intermediate path exists through static ternary materialization
   plus llama.cpp `TQ2_0` or single-thread-written `I2_S`: both preserve the QAT
   PPL and run much faster than F16 decode, but the path requires
   QAT/distillation.

## What This Does Not Prove Yet

1. It does not prove acceptable downstream task accuracy.
2. It does not replace a broader full `lm-eval` leaderboard suite; only five
   core tasks have been run uncapped so far.
3. It does not prove bit-exact GGUF ingestion of `ternary_state_dict.pt`.
4. It does not prove multi-threaded I2_S writer correctness; the safe artifact
   above was produced with `llama-quantize ... I2_S 1`.
5. It does not prove the materialized dense-F16 bridge is storage-optimal; it
   is a validation bridge until a native ternary-state GGUF writer exists.
6. It does not prove the approach works for MoE models.
7. It does not prove that the current training recipe is publishable as a final
   result.

## Next Gates

The next benchmark gates are:

1. Fix the multi-thread I2_S quantization writer/chunking path, then rerun the
   corrected suite without forcing `nthreads=1`.
2. Build a native GGUF writer for `ternary_state_dict.pt` so I2_S can ingest
   trained ternary codes and scales directly.
3. Keep `TQ2_0` and single-thread `I2_S` as the current working packed ternary
   baselines and compare them against Q8_0 and Q4_K_M on quality, model size,
   RSS, prompt throughput, and decode throughput.
4. Extend uncapped `lm-eval` beyond the five core tasks and run the same paired
   analysis for the strongest exact static-ternary checkpoint.
5. Run ablations: longer QAT, KL-only, hidden-MSE weighting, group/row scales, and
   larger FineWeb samples.
