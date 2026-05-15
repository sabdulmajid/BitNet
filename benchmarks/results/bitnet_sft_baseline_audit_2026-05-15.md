# BitNet-SFT Baseline Audit, 2026-05-15

Verdict: the local FP16-SFT MNLI baseline is close to the paper anchor, but local BitNet-SFT is far below the paper's BitNet-SFT anchor. Static checkpoint checks do not show a missing projection-export bug; the next blockers are recipe alignment, SubLN interpretation, LR/epoch search, and training-budget matching. The completed weights-only control shows activation quantization is not the dominant cause of the gap.

## Accuracy And Mechanical Summary

| run | method | accuracy | examples | FP16 gap | paper anchor | paper anchor - local | BitLinear replaced | SubLN inserted | A8 activations | ternary tensors | scale numel histogram |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FP16-SFT | fp16_sft | 0.807641 | 9815.000000 | 0.000000 | 0.799100 | -0.008541 | 0 | 0 | - | - | - |
| BitNet-SFT | bitnet_sft | 0.487621 | 9815.000000 | 0.320020 | 0.608000 | 0.120379 | 168 | 0 | true | 168 | {'1': 168} |
| BitNet-SFT weights-only | bitnet_sft | 0.493734 | 9815.000000 | 0.313907 | - | - | 168 | 0 | false | 168 | {'1': 168} |
| BitNet-SFT SubLN | bitnet_sft | 0.350280 | 9815.000000 | 0.457361 | - | - | 168 | 48 | true | 168 | {'1': 168} |
| BitNet-SFT head-init | bitnet_sft | 0.496994 | 9815.000000 | 0.310647 | - | - | 168 | 0 | true | 168 | {'1': 168} |
| Best local row-scale BitDistill | bitdistill | 0.653591 | 9815.000000 | 0.154050 | - | - | 168 | 48 | true | 168 | {'128': 48, '896': 72, '4864': 48} |

## Ternary Code Summary

| run | codes | -1 frac | 0 frac | +1 frac | scale mean | scale std |
| --- | --- | --- | --- | --- | --- | --- |
| BitNet-SFT | 357826560 | 0.333243 | 0.333176 | 0.333580 | 0.015992 | 0.004402 |
| BitNet-SFT weights-only | 357826560 | 0.333239 | 0.333178 | 0.333583 | 0.015992 | 0.004402 |
| BitNet-SFT SubLN | 357826560 | 0.333235 | 0.333185 | 0.333579 | 0.015992 | 0.004401 |
| BitNet-SFT head-init | 357826560 | 0.333238 | 0.333181 | 0.333581 | 0.015992 | 0.004402 |
| Best local row-scale BitDistill | 357826560 | 0.338735 | 0.322195 | 0.339070 | 0.015881 | 0.004492 |

## Checks

| check | status | evidence | implication |
| --- | --- | --- | --- |
| FP16-SFT local task is learnable | pass | local=0.807641, paper_anchor=0.799100, delta=0.008541 | The weak BitNet-SFT result is unlikely to be caused only by task formatting or dataset split. |
| BitNet-SFT local baseline matches paper anchor | fail | local=0.487621, paper_anchor=0.608000, delta=-0.120379 | This is the primary reproduction blocker to explain before broadening sweeps. |
| BitNet-SFT ternary projection count matches Qwen2.5-0.5B decoder projections | pass | ternary=168, expected=168 | The low baseline is not explained by exporting only one ternary tensor or missing whole decoder projection families. |
| Classifier head remains dense | pass | score.weight | The poor score is not caused by accidentally ternarizing the classification head in this checkpoint. |
| SubLN-only BitNet-SFT control explains the gap | fail | default_subln_inserted=0, subln_accuracy=0.350280, delta_vs_default=-0.137341 | Current SubLN insertion alone does not recover the paper anchor; either the SubLN recipe differs or it requires matched warmup/search. |
| Weights-only vs W1.58A8 ablation exists | pass | weights_only_accuracy=0.493734, A8_accuracy=0.487621 | This separates weight-code collapse from activation-quantization damage. |
| Stage-3 budget is paper-matched | warn | local steps=1000, batch=4, grad_accum=4 | The paper used greedy LR/epoch search and larger hardware batch; local 1000-step budget may undertrain BitNet-SFT. |

## Next Narrow Experiments

1. Run a small LR/epoch budget sweep for BitNet-SFT before more BitDistill ablations.
2. Audit SubLN placement, initialization, and whether it should be enabled before/after continued pretraining.
3. Add activation variance, int8 saturation, ternary flip-rate, and loss-gradient telemetry to the Stage-3 loop.
4. Keep row-scale results labeled as a retrofit variant, not as a paper-reproduction result.
