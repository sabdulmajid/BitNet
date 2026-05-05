# Qwen W1.58A8 Retrofit Results, 2026-05-03, Updated 2026-05-05

This note records the current evidence for retrofitting pretrained Qwen dense
models into BitNet-style W1.58A8 checkpoints. It is intentionally conservative:
these are quality and loader results, not final CPU product benchmarks.

## Setup

- Latest pushed tooling commit before this report update: `272e24e`
- Packed runtime binary: llama.cpp submodule commit `1f86f058`
- Dataset for QAT/distillation: `HuggingFaceFW/fineweb-edu`, `sample-10BT`
- QAT student forward math: ternary weights plus dynamic 8-bit activation quantization
- Main QAT loss: teacher KL divergence plus last-hidden-state MSE
- Additional ablation: teacher KL only for Qwen2.5-0.5B and Qwen2.5-1.5B
- Additional ablation: teacher KL only with tied dense `lm_head` preserved for
  Qwen2.5-0.5B and Qwen2.5-1.5B
- Main scale mode: per-tensor absmean scale
- Additional ablation: per-output-row absmean scale for Qwen2.5-0.5B and
  Qwen2.5-1.5B KL-only dense tied-`lm_head`
- Evaluation dtype/device for perplexity: BF16 on CUDA
- WikiText eval: `wikitext-2-raw-v1`, test split, 64 blocks x 512 tokens
- FineWeb heldout eval: `sample-10BT`, train stream after `--skip-rows 25000`, 32 blocks x 1024 tokens
- Official task eval: EleutherAI `lm-eval` 0.4.11, 100-example ten-task
  slices, 1000-example five-task core slices, and uncapped ten-task merged runs
- PyTorch runtime probe: Intel Xeon Silver 4116, PyTorch FP32, 12 Torch threads,
  512-token prompt, 32 generated tokens
- Packed GGUF runtime probe: Intel Xeon Silver 4116, `llama-bench -p 512
  -n 128 -t 12 -ngl 0 -r 3`, no BLAS, AVX-512 available
- Additional packed GGUF dense-head and row-scale probes: AMD Ryzen
  Threadripper PRO 5945WX via portable AVX2 llama.cpp build

The FineWeb heldout skip is beyond the 1.5B training prefix. Job 9730 packed
19,968 rows, so skipping 25,000 rows avoids direct train-prefix reuse for this
smoke-scale heldout test.

## Completed Training Runs

| run | model | steps | data blocks | final loss | final KL | final hidden MSE | status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 9730 | Qwen2.5-1.5B QAT student | 5000 | 20,000 x 1024 | 4.0420 | 1.7478 | 2.2943 | complete |
| 9734 | Qwen2.5-0.5B QAT student | 1000 | 4,000 x 512 | 24.6557 | 2.8857 | 21.7699 | complete |
| 9747 | Qwen2.5-0.5B QAT student, row scale | 1000 | 4,000 x 512 | 15.3338 | 2.0530 | 13.2808 | complete |
| 9748 | Qwen2.5-0.5B QAT student, KL only | 1000 | 4,000 x 512 | 1.6375 | 1.6375 | 0.0000 | complete |
| 9764 | Qwen2.5-0.5B QAT student, KL only, dense tied `lm_head` | 1000 | 4,000 x 512 | 1.6821 | 1.6821 | 0.0000 | complete |
| 9758 | Qwen2.5-1.5B QAT student, KL only | 5000 | 20,000 x 1024 | 1.5172 | 1.5172 | 0.0000 | complete |
| 9767 | Qwen2.5-1.5B QAT student, KL only, dense tied `lm_head` | 5000 | 20,000 x 1024 | 1.3353 | 1.3353 | 0.0000 | complete |
| 9771 | Qwen2.5-1.5B QAT student, KL only, row scale, dense tied `lm_head` | 5000 | 20,000 x 1024 | 1.2569 | 1.2569 | 0.0000 | complete |

## Export Status

| checkpoint | expected ternary keys | observed ternary keys | notes |
| --- | ---: | ---: | --- |
| `qwen2.5-0.5b-fineweb-edu-12/step-1000` | 169 | 169 | complete QAT ternary export |
| `qwen2.5-0.5b-fineweb-edu-row-1000/step-1000` | 169 | 169 | row-scale QAT ternary export; one scale per output row |
| `qwen2.5-0.5b-fineweb-edu-klonly-1000/step-1000` | 169 | 169 | KL-only QAT ternary export; tensor scale |
| `qwen2.5-0.5b-fineweb-edu-klonly-notiehead-1000/step-1000` | 168 | 168 | KL-only export with tied dense `lm_head` preserved; config keeps `tie_word_embeddings=true` |
| `qwen2.5-1.5b-fineweb-edu/step-5000` | 197 | 197 | repaired after FSDP name-mapping bug |
| `qwen2.5-1.5b-fineweb-edu-klonly-5000/step-5000` | 197 | 197 | KL-only QAT ternary export; `lm_head` is ternary so config is patched to `tie_word_embeddings=false` |
| `qwen2.5-1.5b-fineweb-edu-klonly-notiehead-5000/step-5000` | 196 | 196 | KL-only export with tied dense `lm_head` preserved; config keeps `tie_word_embeddings=true` |
| `qwen2.5-1.5b-fineweb-edu-klonly-row-notiehead-5000/step-5000` | 196 | 196 | KL-only export with tied dense `lm_head` preserved and one scale per output row |
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
| Qwen2.5-0.5B | QAT/distilled ternary, KL only | 296.602 | 108.366 | best 0.5B ablation so far, still far from FP |
| Qwen2.5-0.5B | QAT/distilled ternary, KL only, dense tied `lm_head` | 270.345 | 97.337 | strongest 0.5B ablation so far, but output head remains dense/tied |
| Qwen2.5-1.5B | FP reference | 13.901 | 10.269 | baseline quality |
| Qwen2.5-1.5B | naive PTQ ternary | 3,813,121.803 | 9,582,923.269 | blind ternarization destroys quality |
| Qwen2.5-1.5B | QAT/distilled ternary | 86.414 | 40.398 | QAT is orders better than PTQ, but still far from FP |
| Qwen2.5-1.5B | QAT/distilled ternary, KL only | 50.595 | 26.599 | best 1.5B tensor-scale QAT result so far, still far from FP |
| Qwen2.5-1.5B | QAT/distilled ternary, KL only, dense tied `lm_head` | 43.372 | 22.759 | strong 1.5B tensor-scale likelihood result, but output head remains dense/tied |
| Qwen2.5-1.5B | QAT/distilled ternary, KL only, row scale, dense tied `lm_head` | 38.580 | 21.333 | best 1.5B PyTorch likelihood result so far, still far from FP |

The 0.5B row-scale ablation is meaningful but not decisive. It reduces
WikiText PPL from `1,079.167` to `444.691` and FineWeb-heldout PPL from
`373.775` to `152.821`, but it remains far from the FP reference and does not
establish downstream quality by itself.

The KL-only ablation is stronger. It reduces 0.5B WikiText PPL to `296.602`
and FineWeb-heldout PPL to `108.366`. Preserving Qwen's tied `lm_head` as dense
FP/BF16 improves the same setup again to `270.345` WikiText PPL and `97.337`
FineWeb-heldout PPL. That suggests the hidden-state MSE term can overconstrain
a short-budget ternary student, and that tied output-head policy matters. The
dense-`lm_head` result does not close the FP gap and is not an all-linear W1.58
checkpoint.

The same KL-only loss at Qwen2.5-1.5B scale also improves over the earlier
hidden-MSE QAT run: WikiText PPL drops from `86.414` to `50.595`, and
FineWeb-heldout PPL drops from `40.398` to `26.599`. This is the strongest
1.5B all-linear ternary QAT result so far, but it still remains far from FP
PPL (`13.901` and `10.269`). Preserving the tied output head as dense in the
same 1.5B KL-only setup improves likelihood again to `43.372` WikiText PPL and
`22.759` FineWeb-heldout PPL. That tied-head checkpoint is not an all-linear
W1.58 model, and its full ten-task accuracy gain over the all-ternary KL-only
checkpoint is not statistically decisive.

Adding per-output-row scales to the same KL-only dense-head 1.5B setup improves
likelihood further to `38.580` WikiText PPL and `21.333` FineWeb-heldout PPL.
It is the best PyTorch-quality ablation in this report, but it still remains
far from the FP PPL baseline and, as shown below, current `I2_S` packing does
not preserve it.

## PTQ Math Audit

The ternary export path used by `export_ternary.py` and the naive PTQ baseline
uses absmean scaling:

`alpha = mean(abs(W))`, `Q(W) = clamp(round(W / alpha), -1, 1) * alpha`.

The QAT path in `train_distill.py` uses the same forward projection inside
`TernaryWeightSTE`, but backpropagates through it with a straight-through
estimator. That distinction matters: naive PTQ applies the projection once to
a dense model that never learned under this constraint; QAT trains master
weights while the forward path is already ternary.

The empirical test in `experiments/math_viability_test.py` simulates standard
normally distributed FP16 projection matrices with shape `2048 x 2048` and a
random activation batch. The current audit uses 10 trials from seed 0:

| projection | relative weight Frobenius error | relative output Frobenius error | output cosine | zero fraction |
| --- | ---: | ---: | ---: | ---: |
| tensor absmean ternary repo formula | 0.512679 +/- 0.000128 | 0.512542 +/- 0.000478 | 0.887496 +/- 0.000230 | 0.309943 +/- 0.000153 |
| row absmean ternary QAT formula | 0.512539 +/- 0.000129 | 0.512417 +/- 0.000484 | 0.887538 +/- 0.000222 | 0.309882 +/- 0.000152 |
| sign/max TL-I2 generic path | 3.166747 +/- 0.000940 | 3.166292 +/- 0.004750 | 0.798595 +/- 0.000579 | 0.000740 +/- 0.000019 |

For Gaussian weights, the theoretical relative Frobenius error of the absmean
ternary projection is `0.513207`, matching the simulation. This proves the
blind conversion is not functionally lossless even before model-level
perplexity is measured: roughly half the matrix energy is displaced by the
projection for an ordinary untrained FP weight distribution. Row-wise absmean
scales do not change that conclusion for ordinary dense weights; they mostly
increase the number of possible dequantized values while leaving relative
output error essentially unchanged in this audit.

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

| task | tensor acc | tensor acc_norm | row acc | row acc_norm | KL-only acc | KL-only acc_norm | KL-only dense `lm_head` acc | KL-only dense `lm_head` acc_norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PIQA | 0.520 | 0.500 | 0.480 | 0.460 | 0.570 | 0.560 | 0.580 | 0.570 |
| ARC-Easy | 0.290 | 0.270 | 0.300 | 0.260 | 0.330 | 0.290 | 0.360 | 0.300 |
| ARC-Challenge | 0.290 | 0.300 | 0.240 | 0.240 | 0.290 | 0.240 | 0.270 | 0.290 |
| HellaSwag | 0.310 | 0.260 | 0.340 | 0.240 | 0.360 | 0.260 | 0.350 | 0.290 |

This is a cautionary ablation set. Row-wise scales improve likelihood but not
these fast MC slices. KL-only improves likelihood further. Preserving the tied
dense `lm_head` is best on all four `acc_norm` values in this small harness, but
the slices remain small and the output head is no longer ternary. This result
justifies a larger tied-head ablation, not a deployment-quality claim.

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
full task splits rather than the 1000-example cap. This was the first
uncapped task-accuracy comparison in the current artifact set.

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

## Full Ten-Task lm-eval Accuracy

The full five-task core run was merged with a second uncapped run over BoolQ,
COPA, OpenBookQA, SciQ, and TruthfulQA MC1. The merge was done with
`benchmarks/merge_lm_eval_results.py`, preserving logged samples for paired
analysis.

| task | metric | FP reference | naive PTQ ternary | QAT hidden-MSE | QAT KL-only | QAT KL-only dense `lm_head` | QAT KL-only row-scale dense `lm_head` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ARC-Challenge | acc_norm | 0.449659 | 0.261945 | 0.263652 | 0.271331 | 0.263652 | 0.272184 |
| ARC-Easy | acc_norm | 0.719697 | 0.244108 | 0.478114 | 0.483165 | 0.500842 | 0.517677 |
| HellaSwag | acc_norm | 0.677953 | 0.264190 | 0.362378 | 0.377714 | 0.390759 | 0.412169 |
| PIQA | acc_norm | 0.757889 | 0.507617 | 0.621872 | 0.637106 | 0.647443 | 0.650163 |
| WinoGrande | acc | 0.637727 | 0.498027 | 0.523283 | 0.520916 | 0.523283 | 0.537490 |
| BoolQ | acc | 0.725994 | 0.505505 | 0.592661 | 0.596024 | 0.597248 | 0.605199 |
| COPA | acc | 0.830000 | 0.510000 | 0.640000 | 0.700000 | 0.680000 | 0.690000 |
| OpenBookQA | acc_norm | 0.404000 | 0.276000 | 0.312000 | 0.312000 | 0.308000 | 0.316000 |
| SciQ | acc_norm | 0.934000 | 0.199000 | 0.613000 | 0.695000 | 0.700000 | 0.733000 |
| TruthfulQA MC1 | acc | 0.304774 | 0.220318 | 0.241126 | 0.241126 | 0.232558 | 0.260710 |

Mean displayed metric:

| method | mean |
| --- | ---: |
| FP reference | 0.644169 |
| naive PTQ ternary | 0.348671 |
| QAT hidden-MSE | 0.464809 |
| QAT KL-only | 0.483438 |
| QAT KL-only dense `lm_head` | 0.484378 |
| QAT KL-only row-scale dense `lm_head` | 0.499459 |

Paired deltas on matched examples:

| comparison | macro mean delta | paired 95% CI | example-weighted delta |
| --- | ---: | ---: | ---: |
| QAT hidden-MSE minus naive PTQ | +0.116138 | [+0.038603, +0.193672] | +0.113171 |
| QAT hidden-MSE minus FP reference | -0.179361 | [-0.234751, -0.123971] | -0.233670 |
| naive PTQ minus FP reference | -0.295498 | [-0.418827, -0.172169] | -0.346841 |
| QAT KL-only minus QAT hidden-MSE | +0.018630 | [+0.000830, +0.036429] | +0.013359 |
| QAT KL-only minus naive PTQ | +0.134767 | [+0.042874, +0.226660] | +0.126530 |
| QAT KL-only minus FP reference | -0.160731 | [-0.207484, -0.113978] | -0.220311 |
| QAT KL-only dense `lm_head` minus QAT KL-only | +0.000940 | [-0.006100, +0.007980] | +0.008221 |
| QAT KL-only dense `lm_head` minus QAT hidden-MSE | +0.019570 | [+0.001767, +0.037373] | +0.021580 |
| QAT KL-only dense `lm_head` minus naive PTQ | +0.135707 | [+0.041607, +0.229808] | +0.134751 |
| QAT KL-only dense `lm_head` minus FP reference | -0.159791 | [-0.202734, -0.116847] | -0.212090 |
| QAT KL-only row-scale dense `lm_head` minus QAT KL-only dense `lm_head` | +0.015081 | [+0.009028, +0.021134] | +0.016755 |
| QAT KL-only row-scale dense `lm_head` minus QAT KL-only | +0.016021 | [+0.006145, +0.025897] | +0.024975 |

The row-scale dense tied-head run is the strongest task result in the current
fork. It reaches selected mean `0.499459`, improves over the tensor-scale
dense-head run by `+0.015081` macro mean with paired 95% CI
`[+0.009028, +0.021134]`, and improves over the all-ternary KL-only run by
`+0.016021` with paired 95% CI `[+0.006145, +0.025897]`. It still remains far
below FP on the same tasks: FP mean is `0.644169`, so the remaining macro gap
is about `0.14471`. That supports a publication-quality negative result for
blind retrofit and a partial-positive result for QAT/distillation with better
scaling policy, not a claim of acceptable FP-preserving conversion.

## Fast Dense-Head MC200 Check

The in-repo multiple-choice evaluator was rerun on 200-example validation
slices for the Qwen2.5-1.5B KL-only dense tied-`lm_head` checkpoint. This
corrects an earlier 100-example smoke run that inherited a stale `LIMIT`
environment variable. These MC200 numbers are useful for regression tracking;
the uncapped `lm-eval` table above is stronger evidence for task quality.

| task | acc | acc_norm | n |
| --- | ---: | ---: | ---: |
| PIQA | 0.6400 | 0.6550 | 200 |
| ARC-Easy | 0.5850 | 0.4500 | 200 |
| ARC-Challenge | 0.2250 | 0.2550 | 200 |
| HellaSwag | 0.3800 | 0.4100 | 200 |

The same MC200 probe was rerun for the final row-scale dense tied-`lm_head`
checkpoint:

| task | acc | acc_norm | n |
| --- | ---: | ---: | ---: |
| PIQA | 0.6450 | 0.6850 | 200 |
| ARC-Easy | 0.6100 | 0.5100 | 200 |
| ARC-Challenge | 0.2550 | 0.2900 | 200 |
| HellaSwag | 0.3700 | 0.3950 | 200 |

This quick probe is consistent with the uncapped lm-eval result on PIQA,
ARC-Easy, and ARC-Challenge, while HellaSwag is slightly lower than the
tensor-scale dense-head MC200 slice. The uncapped ten-task table above remains
the stronger evidence.

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
| Qwen2.5-0.5B | QAT KL-only, dense tied `lm_head` | 226.33 | 1.85 | 2.048 | 0.841 | 1.348 |
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
| Qwen2.5-1.5B KL-only static ternary | F16 materialized | 3,396 MiB | 105.34 | 5.50 | sensible completion |
| Qwen2.5-1.5B KL-only static ternary | TQ2_0 | 1,219 MiB | 160.93 | 18.43 | sensible completion |
| Qwen2.5-1.5B KL-only static ternary | I2_S single-thread quant | 1,209 MiB | 205.76 | 18.60 | sensible completion |
| Qwen2.5-1.5B KL-only static ternary | I2_S patched 12-thread quant | 1,209 MiB | 208.10 | 18.63 | sensible completion |

Smoke prompt: `The capital of France is`, greedy decoding, 24 generated tokens.
The FP/Q8_0/Q4_K_M controls complete with `Paris` and related capital-city
continuations. The 0.5B I2_S artifacts collapse to repeated exclamation marks;
the blind 1.5B I2_S artifact collapses to repeated `is` tokens. The
single-thread-written static-ternary I2_S artifact produces a sensible
Paris/French-government continuation. The KL-only static-ternary artifacts
produce sensible capital-city continuations and preserve the stronger KL-only
quality signal.

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
  is written with a single quantization thread. The earlier multi-thread I2_S
  artifact was corrupted because the generic chunked writer did not respect the
  I2_S packed-byte layout or its tensor-level scale.
- Repeating the same static-ternary bridge for the stronger KL-only QAT
  checkpoint improves fixed-excerpt I2_S PPL from `84.5277` to `54.7366` while
  keeping decode throughput at `18.60` tok/s on the Xeon.
- Applying `patches/llama-i2s-threaded-quantization.patch` to the llama.cpp
  submodule fixes the threaded I2_S writer locally. A 12-thread quantized
  KL-only static-ternary artifact preserves PPL `54.7366`, reaches `208.10`
  prompt tok/s and `18.63` decode tok/s, and produces a sensible smoke
  completion.

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
| Qwen2.5-1.5B KL-only static ternary | F16 materialized | 55.0971 | 3.00700 | 82.65 |
| Qwen2.5-1.5B KL-only static ternary | TQ2_0 | 55.1562 | 3.00939 | 116.16 |
| Qwen2.5-1.5B KL-only static ternary | I2_S single-thread quant | 54.7366 | 2.98318 | 140.95 |
| Qwen2.5-1.5B KL-only static ternary | I2_S patched 12-thread quant | 54.7366 | 2.98318 | 141.14 |

A later dense-`lm_head` static-ternary GGUF suite was run on the same fixed
WikiText excerpt but on `ece-nebula12`, an AMD Ryzen Threadripper PRO 5945WX
node. Its PPL values are directly useful; its throughput should not be compared
against the Xeon rows as a same-hardware speedup.

| source | GGUF type | CPU | PPL | prompt-eval tok/s | decode tok/s |
| --- | --- | --- | ---: | ---: | ---: |
| Qwen2.5-1.5B FP | F16 | AMD 5945WX | 12.2808 | 218.46 | 12.46 |
| Qwen2.5-1.5B FP | Q8_0 | AMD 5945WX | 12.3056 | 215.00 | 23.14 |
| Qwen2.5-1.5B FP | Q4_K_M | AMD 5945WX | 12.8112 | 172.14 | 36.50 |
| Qwen2.5-1.5B KL-only dense-`lm_head` static ternary | F16 materialized | AMD 5945WX | 47.2994 | 222.07 | 12.48 |
| Qwen2.5-1.5B KL-only dense-`lm_head` static ternary | TQ2_0 | AMD 5945WX | 47.2823 | 348.88 | 44.03 |
| Qwen2.5-1.5B KL-only dense-`lm_head` static ternary | I2_S single-thread quant | AMD 5945WX | 47.3435 | 464.19 | 45.50 |

The final row-scale dense-head suite was run on the same AMD node and fixed
WikiText excerpt:

| source | GGUF type | CPU | PPL | prompt-eval tok/s | decode tok/s | smoke prompt |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Qwen2.5-1.5B FP | F16 | AMD 5945WX | 12.2808 | 219.27 | 11.99 | sensible |
| Qwen2.5-1.5B FP | Q8_0 | AMD 5945WX | 12.3056 | 214.69 | 22.51 | sensible |
| Qwen2.5-1.5B FP | Q4_K_M | AMD 5945WX | 12.8112 | 172.37 | 36.82 | sensible |
| Qwen2.5-1.5B KL-only row-scale dense-`lm_head` static ternary | F16 materialized | AMD 5945WX | 38.8651 | 221.64 | 12.49 | sensible |
| Qwen2.5-1.5B KL-only row-scale dense-`lm_head` static ternary | TQ2_0 | AMD 5945WX | 38.8224 | 345.32 | 44.85 | sensible |
| Qwen2.5-1.5B KL-only row-scale dense-`lm_head` static ternary | I2_S single-thread quant | AMD 5945WX | 1,197,135.5848 | 465.34 | 46.13 | failed |

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
  static-ternary PPL. The included threaded-writer patch preserves the same PPL
  with 12 quantization threads, which confirms the earlier NaN PPL was a
  writer/chunking bug rather than a fundamental runtime math failure.
- The all-linear KL-only static-ternary bridge is the best same-hardware Xeon
  result: patched 12-thread `I2_S` gives `54.7366` PPL on the fixed excerpt,
  `141.14` prompt-eval tok/s, and `18.63` decode tok/s.
- The dense-`lm_head` KL-only static-ternary bridge improves the fixed-excerpt
  PPL to `47.3435` when packed as `I2_S`. That run was measured on AMD
  5945WX, so it is quality evidence and a separate hardware throughput data
  point, not a same-machine Xeon speed comparison. Both trained ternary bridges
  are still far worse than FP/Q8/Q4 likelihood.
- The row-scale dense-`lm_head` bridge improves fixed-excerpt PPL further to
  `38.8651` as materialized F16 and `38.8224` as `TQ2_0`, so row scales do
  survive the dense bridge and generic ternary packing.
- The current row-scale `I2_S` bridge fails despite high throughput: PPL rises
  to `1,197,135.5848` and the smoke prompt is incoherent. This is a concrete
  format/kernel limitation, not a training-quality result. A row-scale-aware
  packed ternary writer and kernel are required before claiming row-scale
  `I2_S` deployment.

## MoE Status

The codebase has partial infrastructure for MoE-shaped GGUF artifacts, but this
fork has not proven a ternary MoE retrofit.

What exists:

- The vendored llama.cpp runtime stores expert metadata such as
  `expert_count` and `expert_used_count`.
- It has merged expert tensor names such as `ffn_gate_exps`, `ffn_down_exps`,
  and `ffn_up_exps`.
- Its MoE graph builder computes router logits, applies softmax and top-k
  expert selection, uses `ggml_mul_mat_id` for selected expert matrices, and
  aggregates the selected expert outputs.
- `utils/convert-hf-to-gguf-bitnet.py` writes Qwen-style expert metadata from
  `num_local_experts` and `num_experts_per_tok`, and packs
  `block_sparse_moe.experts.*.{w1,w2,w3}.weight` tensors into merged 3D expert
  tensors.

What is missing:

- No Kimi architecture mapping has been implemented or tested in this fork.
- No MoE checkpoint has been trained with ternary expert forward constraints.
- No expert-router distillation or load-balancing objective has been evaluated.
- No native ternary-state GGUF writer has been proven for expert tensors.
- No MoE quality, throughput, memory paging, or expert locality benchmark has
  been run.

Conclusion: MoE is not mathematically excluded. Static expert matrices can in
principle use the same ternary projection and packed kernels as dense FFN
matrices. The unproven part is the system: router quality under ternary expert
constraints, expert tensor layout, dynamic expert selection overhead, and memory
locality on CPU.

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
9. On the full uncapped ten-task lm-eval merge, 1.5B QAT improves the mean
   displayed task metric from 0.349 for naive PTQ to 0.465 with hidden-MSE,
   0.483 with KL-only, 0.484 with KL-only dense `lm_head`, and 0.499 with
   KL-only row-scale dense `lm_head`, while FP remains 0.644. The best ternary
   student still trails FP by about 0.14471 macro mean.
10. Row-wise ternary scales improve 0.5B heldout perplexity by about 2.4x versus
   tensor-scale QAT, but the small multiple-choice slices remain mixed.
11. KL-only 0.5B distillation improves heldout perplexity further. Keeping
   Qwen's tied `lm_head` dense improves it again and is the best short-budget
   0.5B ablation so far, but it remains far from FP quality and is not a fully
   ternary linear stack.
12. Keeping Qwen2.5-1.5B's tied `lm_head` dense improves WikiText/FineWeb PPL
    from 50.595/26.599 to 43.372/22.759 versus the all-ternary KL-only run, but
    its ten-task macro gain over that all-ternary KL-only run is only +0.00094
    with a paired 95% CI crossing zero.
13. Combining row-wise scales with KL-only distillation and a dense tied
    `lm_head` improves Qwen2.5-1.5B WikiText/FineWeb PPL further to
    38.580/21.333 and improves ten-task macro mean over the tensor-scale
    dense-head run by +0.015081 with paired 95% CI [+0.009028, +0.021134].
14. PyTorch ternary inference is slower than FP on the Xeon probe. The product
   speed thesis depends on packed `bitnet.cpp` kernels, not on the PyTorch
   simulation path.
15. Packed I2_S GGUF execution is fast on the Xeon for Qwen2.5-0.5B-shaped
   and Qwen2.5-1.5B-shaped models, but naive I2_S conversion fails the smoke
   prompt and explodes WikiText excerpt perplexity.
16. A deployable intermediate path exists through static ternary materialization
   plus llama.cpp `TQ2_0` or single-thread-written `I2_S`: both preserve the QAT
   PPL and run much faster than F16 decode, but the path requires
   QAT/distillation. The strongest same-hardware Xeon all-linear checkpoint is
   the KL-only static ternary `I2_S` artifact: fixed-excerpt PPL 54.7366,
   prompt-eval 140.95 tok/s, and decode 18.60 tok/s. The strongest packed
   `I2_S` quality result so far is the dense-`lm_head` KL-only static ternary
   artifact: fixed-excerpt PPL 47.3435, measured on AMD 5945WX at 464.19
   prompt-eval tok/s and 45.50 decode tok/s. The strongest packed row-scale
   quality result is `TQ2_0`, not `I2_S`: fixed-excerpt PPL 38.8224 on AMD
   5945WX at 345.32 prompt-eval tok/s and 44.85 decode tok/s.
17. The current row-scale `I2_S` path fails. It reaches high throughput but
    explodes to fixed-excerpt PPL 1,197,135.5848 and fails the smoke prompt.
    This identifies a packed-format/kernel problem rather than a training
    problem.
18. The original tensor-scale multi-thread I2_S writer corruption is fixable.
    A local llama.cpp patch that packs I2_S chunks at compressed offsets and
    writes one tensor-level scale preserves fixed-excerpt PPL 54.7366 with 12
    quantization threads.

## What This Does Not Prove Yet

1. It does not prove acceptable downstream task accuracy.
2. It does not replace a broader full `lm-eval` leaderboard suite; ten selected
   tasks have been run uncapped so far.
3. It does not prove bit-exact GGUF ingestion of `ternary_state_dict.pt`.
4. It does not prove that the multi-threaded I2_S writer fix is upstreamed in
   the llama.cpp submodule yet; the fix is included as
   `patches/llama-i2s-threaded-quantization.patch` and validated locally.
5. It does not prove the materialized dense-F16 bridge is storage-optimal; it
   is a validation bridge until a native ternary-state GGUF writer exists.
6. It does not prove the approach works for MoE models. The backend has generic
   MoE infrastructure, but no Kimi-compatible ternary MoE path has been
   benchmarked.
7. It does not prove that the current training recipe is publishable as a final
   result.
8. It does not prove that a dense tied `lm_head` is acceptable for every product
   constraint; it improves likelihood at 0.5B and 1.5B scale but gives up full
   output-head ternarization.
9. It does not prove row-scale `I2_S` deployment. The current row-scale
   materialized and `TQ2_0` artifacts work; the current row-scale `I2_S`
   artifact fails quality audit.

## Next Gates

The next benchmark gates are:

1. Build a row-scale-aware packed ternary writer and kernel path. The current
   highest-quality row-scale checkpoint works as materialized F16 and `TQ2_0`
   but fails as `I2_S`.
2. Advance or fork the llama.cpp submodule to include the validated
   tensor-scale multi-thread I2_S writer patch, then keep
   `quantize_gguf_safe.py` from forcing `nthreads=1`.
3. Build a native GGUF writer for `ternary_state_dict.pt` so packed formats can
   ingest trained ternary codes and per-tensor or per-row scales directly.
4. Keep `TQ2_0` and tensor-scale single-thread `I2_S` as the current working
   packed ternary baselines and compare them against Q8_0 and Q4_K_M on
   quality, model size, RSS, prompt throughput, and decode throughput.
5. Extend uncapped `lm-eval` beyond the current ten selected tasks and keep the
   same paired analysis for the strongest exact static-ternary checkpoint.
6. Run ablations: longer QAT, hidden-MSE weighting, group/row scales, and
   larger FineWeb samples.
