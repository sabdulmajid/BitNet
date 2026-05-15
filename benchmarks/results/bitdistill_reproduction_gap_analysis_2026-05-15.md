# BitDistill Reproduction Gap Analysis, 2026-05-15

This report explains why the current local BitDistill results do not match the
paper-level claim, and why that is not the same thing as disproving BitDistill.
It separates completed evidence from active hypotheses.

## Binary Status

Paper-level GLUE reproduction in this fork: `not achieved`.

Current best local BitDistill class of method: `useful recovery, not FP parity`.

Current product-safe claim: `dense-Qwen row-scale I2_SR runtime works for the
audited causal path; arbitrary FP16/BF16 retrofit and Kimi/MoE are not proven`.

## Completed Evidence

| task | FP16-SFT | BitNet-SFT | best completed BitDistill | gap to FP16 | paper target gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| MNLI | `0.807641` | `0.487621` | `0.653591` row, gamma=100 | `0.154050` | `<=0.010000` |
| QNLI | `0.898957` | `0.596925` | `0.796998` row, gamma=100 | `0.101959` | `<=0.010000` |
| SST2 | `0.925459` | `0.770642` | `0.866972` tensor, gamma=100 | `0.058486` | `<=0.010000` |

The strict paper-gamma tensor branch (`attention_kd_weight=1e5`) is complete
on GLUE3 and is worse than the gamma=100 branch in this local setting:

| task | gamma=100 tensor | paper-gamma tensor | paper-gamma gap to FP16 |
| --- | ---: | ---: | ---: |
| MNLI | `0.641671` | `0.630260` | `0.177382` |
| QNLI | `0.787846` | `0.759656` | `0.139301` |
| SST2 | `0.866972` | `0.841743` | `0.083716` |

The strict paper-gamma row branch is complete on GLUE3. It does not close the
gap:

| task | paper-gamma tensor | paper-gamma row | row-tensor delta | row gap to FP16 |
| --- | ---: | ---: | ---: | ---: |
| MNLI | `0.630260` | `0.617626` | `-0.012634` | `0.190015` |
| QNLI | `0.759656` | `0.760937` | `+0.001281` | `0.138019` |
| SST2 | `0.841743` | `0.837156` | `-0.004587` | `0.088303` |

The clean row-scale Stage-2 warm-up gamma=100 branch is also complete on
GLUE3. It is negative against both FP16-SFT and the earlier tensor-warmup row
branch:

| task | row-warmup row gamma=100 | FP gap | delta vs tensor-warmup row gamma=100 |
| --- | ---: | ---: | ---: |
| MNLI | `0.627713` | `0.179929` | `-0.025879` |
| QNLI | `0.779791` | `0.119165` | `-0.017207` |
| SST2 | `0.846330` | `0.079128` | `-0.008028` |

The clean row-scale Stage-2 warm-up paper-gamma branch is complete as well.
It remains far outside the paper target:

| task | row-warmup row paper-gamma | FP gap | delta vs tensor-warmup row paper-gamma |
| --- | ---: | ---: | ---: |
| MNLI | `0.617830` | `0.189812` | `+0.000204` |
| QNLI | `0.777046` | `0.121911` | `+0.016108` |
| SST2 | `0.830275` | `0.095183` | `-0.006881` |

## What Went Wrong Locally

| gap | local fact | implication |
| --- | --- | --- |
| Continued-pretraining budget | Current tensor and row warm-up targets are `163,840,000` token presentations; the paper reports `10,000,000,000`. | Local warm-up is only `1.6384%` of the paper budget. It is enough to test directionality, not enough to claim recipe failure. |
| Backbone/scale | Local strict branch uses `Qwen2.5-0.5B`; the paper centers Qwen3 `0.6B/1.7B/4B` and reports Qwen2.5 as a robustness target. | The local result is relevant, but not the primary paper scale ladder. |
| Hyperparameter search | Local completed branches use fixed `1000` downstream steps. LR=`1e-5`, LR=`5e-5`, and strict paper-gamma head-init searches are complete and negative. | The paper explicitly uses greedy LR/epoch search, so the remaining mismatch is broader epoch/budget/backbone search, not the two tested LR points or head initialization. |
| Attention KD scale | Local loss-scale probes show paper `gamma=1e5` can dominate CE by orders of magnitude. | The coefficient is not plug-and-play under smaller budget/backbone/runtime conditions; the sweep is required evidence, not optional tuning. |
| Output-head treatment | Gamma=100 head-initialization diagnostics are complete and mixed/negative overall: MNLI row got worse (`0.653591` to `0.646052`), QNLI row improved slightly (`0.796998` to `0.800476`), QNLI tensor got worse (`0.787846` to `0.777778`), SST2 tensor was unchanged (`0.866972`), and SST2 row got worse (`0.854358` to `0.847477`). | Initializing task heads from the teacher is not currently the missing fix. |
| Attention layer choice | Long-warmup MNLI layer sweep: layer `-1` reached `0.645950`, layer `-2` reached `0.642894`, layer `-4` reached `0.640754`, gamma=1k tensor remains `0.647275`, and row gamma=100 remains `0.653591`. | Layer choice affects the margin but has not closed the FP16 gap under the current budget. |
| Runtime target mismatch | Strict GLUE reproduction uses `Qwen2ForSequenceClassification`; packed `I2_SR` export is valid today for causal-LM checkpoints, not classification heads. | Quality reproduction and packed CPU productization are coupled but distinct engineering gates. |

## What The Paper Changes

The paper does not support blind post-training ternarization. It supports a
task-specific QAT/distillation path:

1. Insert SubLN modules for ternary optimization stability.
2. Continue pretrain under ternary forward constraints before downstream SFT.
3. Distill with CE + logits KL + single-layer Q/K/V relation KD.
4. Use downstream tasks as the target, not general language modeling parity.

That is compatible with the negative PTQ result in this fork. The local
negative result is specifically that a short-budget, Qwen2.5-0.5B reproduction
has not recovered FP-level GLUE quality yet.

## Active Tests That Decide The Next Claim

| branch | purpose | current state |
| --- | --- | --- |
| paper-gamma row | Test whether row scales help under the literal paper attention coefficient. | complete on GLUE3 and negative or neutral; no task closes the FP16 gap |
| LR search at paper gamma | Test whether fixed `2e-5` LR is the failure mode. | LR=`1e-5` and LR=`5e-5` complete on GLUE3; neither closes the FP16 gap |
| paper-gamma headinit | Test whether classifier-head transfer closes the gap. | complete on GLUE3 and negative overall |
| layer sweep | Test whether the selected attention-distillation layer is wrong. | layer `-1`, `-2`, `-4`, and `-8` evidence complete; no layer closes the FP16 gap |
| clean row warm-up gamma=100 | Test whether row-scale Stage-2 pretraining improves row downstream quality at the best completed local coefficient. | complete on GLUE3 and negative; no task closes the FP16 gap, and all three tasks trail the earlier tensor-warmup row branch |
| clean row warm-up paper-gamma | Test whether row-scale Stage-2 pretraining helps under the literal paper attention coefficient. | complete on GLUE3 and negative; QNLI improves over tensor-warmup paper-gamma row but remains `0.121911` behind FP16 |
| CPU/I2_SR producers | Test full runtime speed/RSS/quality gates for completed causal export candidates. | local causal I2_SR gate passes; scoped Xeon PyTorch CPU GLUE slice passes; full Xeon 512-sample PyTorch CPU gate passes `33/33` critical rows |

## Publishable Path

If a pending branch closes the FP16 gap to within one accuracy point, the
publishable claim becomes an independent BitDistill reproduction plus a
row-scale `I2_SR` CPU runtime extension.

If no pending branch closes the gap, the publishable claim becomes a boundary
study: BitDistill-style training is necessary and helpful, but paper-level
quality is resource- and recipe-sensitive; the fork contributes reproducible
negative evidence, statistical gates, and a CPU runtime path that preserves
row-scale ternary semantics for dense causal models.

Both outcomes are useful. Neither outcome supports a one-click arbitrary
FP16/BF16-to-1.58-bit converter today.
