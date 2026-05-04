# Qwen W1.58A8 Retrofit Results, 2026-05-03

This note records the current evidence for retrofitting pretrained Qwen dense
models into BitNet-style W1.58A8 checkpoints. It is intentionally conservative:
these are quality and loader results, not final CPU product benchmarks.

## Setup

- Repository commit after benchmark tooling: `6501778`
- Packed runtime binary: llama.cpp submodule commit `1f86f058`
- Dataset for QAT/distillation: `HuggingFaceFW/fineweb-edu`, `sample-10BT`
- QAT student forward math: ternary weights plus dynamic 8-bit activation quantization
- QAT loss: teacher KL divergence plus last-hidden-state MSE
- Scale mode: per-tensor absmean scale
- Evaluation dtype/device for perplexity: BF16 on CUDA
- WikiText eval: `wikitext-2-raw-v1`, test split, 64 blocks x 512 tokens
- FineWeb heldout eval: `sample-10BT`, train stream after `--skip-rows 25000`, 32 blocks x 1024 tokens
- Official task eval: EleutherAI `lm-eval` 0.4.11, 100-example slices
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

## Export Status

| checkpoint | expected ternary keys | observed ternary keys | notes |
| --- | ---: | ---: | --- |
| `qwen2.5-0.5b-fineweb-edu-12/step-1000` | 169 | 169 | complete QAT ternary export |
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
| Qwen2.5-1.5B | FP reference | 13.901 | 10.269 | baseline quality |
| Qwen2.5-1.5B | naive PTQ ternary | 3,813,121.803 | 9,582,923.269 | blind ternarization destroys quality |
| Qwen2.5-1.5B | QAT/distilled ternary | 86.414 | 40.398 | QAT is orders better than PTQ, but still far from FP |

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
| Qwen2.5-1.5B FP | F16 | 2,950 MiB | 105.30 | 5.52 | sensible completion |
| Qwen2.5-1.5B FP | Q8_0 | 1,570 MiB | 135.45 | 10.07 | sensible completion |
| Qwen2.5-1.5B FP | Q4_K_M | 940 MiB | 95.17 | 15.72 | sensible completion |
| Qwen2.5-1.5B FP | I2_S | 766 MiB | 205.66 | 18.41 | repeated-token collapse |
| Qwen2.5-1.5B QAT step-5000 | F16 | 3,396 MiB | 105.21 | 5.52 | degenerate text |
| Qwen2.5-1.5B QAT step-5000 | I2_S | 1,211 MiB | 203.59 | 17.97 | repeated-token collapse |
| Qwen2.5-1.5B static ternary | F16 materialized | 3,396 MiB | 105.28 | 5.51 | sensible completion |
| Qwen2.5-1.5B static ternary | TQ2_0 | 1,219 MiB | 158.52 | 18.38 | sensible completion |
| Qwen2.5-1.5B static ternary | I2_S | 1,211 MiB | 190.79 | 18.61 | punctuation collapse |

Smoke prompt: `The capital of France is`, greedy decoding, 24 generated tokens.
The FP/Q8_0/Q4_K_M controls complete with `Paris` and related capital-city
continuations. The 0.5B I2_S artifacts collapse to repeated exclamation marks;
the 1.5B I2_S artifacts collapse to repeated `is` tokens.

Important caveats:

- I2_S is fast on this CPU, so the execution layer is not the bottleneck for a
  0.5B dense Qwen-shaped model.
- Blind I2_S conversion is not quality-preserving.
- Q4_K_M is not a pure Q4_K_M baseline for the Qwen2.5-0.5B shape: 144 of 168
  tensors required fallback quantization in the original FP conversion. The
  Qwen2.5-1.5B Q4_K_M conversions did not report fallback warnings.
- The QAT 0.5B checkpoint is intrinsically weak, which matches its high
  perplexity. The 1.5B QAT checkpoint is stronger under the PyTorch
  static-ternary path, but dense GGUF export from `model.safetensors` is not a
  valid proxy for that static ternary artifact.
- Materializing `ternary_state_dict.pt` back into dense F16 recovers the
  expected static-ternary behavior. Quantizing that materialized artifact to
  generic `TQ2_0` preserves quality, while quantizing it to I2_S does not.

## Packed GGUF Perplexity Probe

`llama-perplexity` was run on a fixed WikiText-2 test excerpt generated from
123 non-empty test rows: 16 chunks, 512-token context, 8,192 evaluated tokens,
12 CPU threads, `-ngl 0`.

| source | GGUF type | PPL | stderr | prompt-eval tok/s |
| --- | --- | ---: | ---: | ---: |
| Qwen2.5-1.5B FP | F16 | 12.2806 | 0.52969 | 84.11 |
| Qwen2.5-1.5B FP | Q8_0 | 12.3207 | 0.53098 | 104.28 |
| Qwen2.5-1.5B FP | Q4_K_M | 12.8452 | 0.55781 | 75.53 |
| Qwen2.5-1.5B FP | I2_S | 1.206e51 | 5.898e50 | 140.03 |
| Qwen2.5-1.5B QAT step-5000 dense GGUF | F16 | 2728.9322 | 262.72596 | 83.79 |
| Qwen2.5-1.5B QAT step-5000 dense GGUF | I2_S | 7.619e59 | 4.502e59 | 137.73 |
| Qwen2.5-1.5B static ternary | F16 materialized | 83.8300 | 4.60205 | 83.27 |
| Qwen2.5-1.5B static ternary | TQ2_0 | 84.0553 | 4.61363 | 113.75 |
| Qwen2.5-1.5B static ternary | I2_S | NaN | NaN | 132.97 |

Interpretation:

- Standard Q8_0 and Q4_K_M preserve the FP perplexity on this packed GGUF
  excerpt.
- Blind I2_S quantization mathematically destroys language-modeling likelihood
  even though it improves CPU throughput.
- Dense `model.safetensors` export from the QAT checkpoint is not the deployment
  object we need. The PyTorch QAT result relies on static ternary weights and
  scales from `ternary_state_dict.pt`; until GGUF ingests those exactly, the
  GGUF QAT numbers above should be treated as a failed conversion path rather
  than a final measure of the QAT recipe.
- A dense materialization bridge for `ternary_state_dict.pt` validates the
  exported static ternary checkpoint in GGUF F16 form: PPL 83.83, matching the
  earlier PyTorch-scale result.
- Generic llama.cpp `TQ2_0` preserves that static-ternary PPL while giving a
  2.06 bpw ternary file and about 18.38 decode tok/s on the Xeon.
- Current I2_S quantization of the same materialized artifact produces NaN PPL,
  so I2_S needs a writer/kernel audit before being used for trained QAT Qwen.

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
6. On ten lm-eval slices, 1.5B QAT improves the mean displayed task metric from
   0.351 for naive PTQ to 0.490, while FP remains 0.646.
7. PyTorch ternary inference is slower than FP on the Xeon probe. The product
   speed thesis depends on packed `bitnet.cpp` kernels, not on the PyTorch
   simulation path.
8. Packed I2_S GGUF execution is fast on the Xeon for Qwen2.5-0.5B-shaped
   and Qwen2.5-1.5B-shaped models, but naive I2_S conversion fails the smoke
   prompt and explodes WikiText excerpt perplexity.
9. A deployable intermediate path exists through static ternary materialization
   plus llama.cpp `TQ2_0`: it preserves the QAT PPL and runs much faster than
   F16 decode, but it is not the optimized BitNet I2_S path.

## What This Does Not Prove Yet

1. It does not prove acceptable downstream task accuracy.
2. It does not replace full, unsliced `lm-eval` task accuracy.
3. It does not prove bit-exact GGUF ingestion of `ternary_state_dict.pt`.
4. It does not prove real `bitnet.cpp` quality for the stronger Qwen2.5-1.5B
   QAT checkpoint yet because dense GGUF export is not bit-exact to
   `ternary_state_dict.pt`.
5. It does not prove I2_S correctness for trained sparse ternary Qwen; the
   materialized static-ternary I2_S artifact produced NaN perplexity.
6. It does not prove the approach works for MoE models.
7. It does not prove that the current training recipe is publishable as a final
   result.

## Next Gates

The next benchmark gates are:

1. Audit and fix I2_S quantization/runtime for materialized sparse ternary Qwen
   weights.
2. Build a native GGUF writer for `ternary_state_dict.pt` so I2_S can ingest
   trained ternary codes and scales directly.
3. Keep `TQ2_0` as the current working packed ternary baseline and compare it
   against Q8_0 and Q4_K_M on quality, model size, RSS, prompt throughput, and
   decode throughput.
4. Run full, unsliced `lm-eval` tasks for the strongest exact static-ternary
   checkpoint.
5. Run ablations: longer QAT, row scale, KL-only, hidden-MSE weighting, and
   larger FineWeb samples.
