# BitNet-SFT Baseline Audit, 2026-05-15

Verdict: the local FP16-SFT MNLI baseline is close to the paper anchor, and the best completed BitNet-SFT budget row now clears the paper's BitNet-SFT anchor. The original 1000-step default row was undertrained. Static checkpoint checks do not show a missing projection-export bug; the remaining reproduction blocker is BitDistill/FP16-level recovery, especially SubLN and distillation-loss parity.

Best completed budget row: `0.628935` at steps=`10000`, lr=`2e-5`. Paired candidate-minus-FP16 delta: `-0.179215` with CI `[-0.189580, -0.168851]`.

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
| Default 1000-step BitNet-SFT matches paper anchor | fail | local=0.487621, paper_anchor=0.608000, delta=-0.120379 | The short/default run is undertrained and should not be used as the final BitNet-SFT anchor. |
| Best completed budget BitNet-SFT matches paper anchor | pass | best=0.628935, paper_anchor=0.608000, steps=10000, lr=2e-5 | The local blocker has shifted from BitNet-SFT viability to BitDistill/FP16-level recovery. |
| Best completed budget BitNet-SFT is still below paired FP16 | fail | candidate_minus_fp16=-0.179215, ci95=[-0.189580, -0.168851], mcnemar=3.438389e-240 | Clearing the paper BitNet-SFT anchor is not enough; the BitDistill stage must recover the remaining FP16 gap. |
| BitNet-SFT ternary projection count matches Qwen2.5-0.5B decoder projections | pass | ternary=168, expected=168 | The low baseline is not explained by exporting only one ternary tensor or missing whole decoder projection families. |
| Classifier head remains dense | pass | score.weight | The poor score is not caused by accidentally ternarizing the classification head in this checkpoint. |
| SubLN-only BitNet-SFT control explains the gap | fail | default_subln_inserted=0, subln_accuracy=0.350280, delta_vs_default=-0.137341 | Current SubLN insertion alone does not recover the paper anchor; either the SubLN recipe differs or it requires matched warmup/search. |
| Weights-only vs W1.58A8 ablation exists | pass | weights_only_accuracy=0.493734, A8_accuracy=0.487621 | This separates weight-code collapse from activation-quantization damage. |
| BitNet-SFT budget issue is explained | pass | default_steps=1000, default_acc=0.487621, best_steps=10000, best_acc=0.628935 | The next controlled comparison should use the best budget row, not the default 1000-step row. |

## Next Narrow Experiments

1. Finish the pending 10000-step BitNet-SFT LR row and update the paired audit.
2. Use the cleared BitNet-SFT budget row as the controlled CE-only baseline for BitDistill recovery.
3. Audit SubLN placement, initialization, and whether it should be enabled before or after continued pretraining.
4. Add activation variance, int8 saturation, ternary flip-rate, and loss-gradient telemetry to the Stage-3 loop.
5. Keep row-scale results labeled as a retrofit variant, not as a paper-reproduction result.
