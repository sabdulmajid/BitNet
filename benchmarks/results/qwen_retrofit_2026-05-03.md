# Qwen W1.58A8 Retrofit Results, 2026-05-03

This note records the current evidence for retrofitting pretrained Qwen dense
models into BitNet-style W1.58A8 checkpoints. It is intentionally conservative:
these are quality and loader results, not final CPU product benchmarks.

## Setup

- Repository commit after benchmark tooling: `6501778`
- Dataset for QAT/distillation: `HuggingFaceFW/fineweb-edu`, `sample-10BT`
- QAT student forward math: ternary weights plus dynamic 8-bit activation quantization
- QAT loss: teacher KL divergence plus last-hidden-state MSE
- Scale mode: per-tensor absmean scale
- Evaluation dtype/device for perplexity: BF16 on CUDA
- WikiText eval: `wikitext-2-raw-v1`, test split, 64 blocks x 512 tokens
- FineWeb heldout eval: `sample-10BT`, train stream after `--skip-rows 25000`, 32 blocks x 1024 tokens
- Official task eval: EleutherAI `lm-eval` 0.4.11, 100-example slices
- Runtime probe: Intel Xeon Silver 4116, PyTorch FP32, 12 Torch threads,
  512-token prompt, 32 generated tokens

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
   thesis still depends on packed `bitnet.cpp` kernels, not on the PyTorch
   simulation path.

## What This Does Not Prove Yet

1. It does not prove acceptable downstream task accuracy.
2. It does not replace full, unsliced `lm-eval` task accuracy.
3. It does not prove real `bitnet.cpp` CPU speedups; the current evaluation path
   simulates W1.58A8 math in PyTorch and does not use packed TL2/I2_S kernels.
4. It does not prove the approach works for MoE models.
5. It does not prove that the current training recipe is publishable as a final
   result.

## Next Gates

The next benchmark gates are:

1. Convert repaired checkpoints to GGUF/TL2/I2_S and run actual `bitnet.cpp` CPU
   inference.
2. Compare against llama.cpp Q8_0 and Q4_K_M on quality, model size, RSS, prompt
   throughput, and decode throughput.
3. Run full, unsliced `lm-eval` tasks for the strongest checkpoint.
4. Run ablations: longer QAT, row scale, KL-only, hidden-MSE weighting, and
   larger FineWeb samples.
