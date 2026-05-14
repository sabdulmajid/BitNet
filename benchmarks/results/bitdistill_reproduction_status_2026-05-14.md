# BitDistill Reproduction Status, 2026-05-14

This report tracks the current attempt to reproduce the BitDistill paper result
on Qwen2.5-0.5B with GLUE sequence-classification tasks.

## Binary Status

**Not reproduced yet.** The current short-budget implementation does not reach
the target of being within 0.5-1.0 accuracy point of the FP16-SFT baseline on
MNLI, QNLI, or SST2.

This is **not** a disproof of the BitDistill paper. The main known gap is
training budget: the paper reports a 10B-token continued-pretraining warm-up,
while the current completed warm-up has 40.96M effective token presentations.
That is about 244x smaller. A second paper-faithfulness issue was also found:
the first completed GLUE3 wave used the common KD convention of multiplying
logits KL by `temperature**2`, but the BitDistill equations provided in the
paper do not include that multiplier. The code now exposes
`--logit-kd-temperature-scale {none,square}` and defaults to paper-style
`none` for new runs.

The current result is therefore evidence about the failure boundary of direct
or short-warm-up ternary retrofit, not evidence that paper-faithful BitDistill
cannot work.

## Implemented Components

The local reproduction code now includes:

- Qwen block refinement with SubLN inserted before attention `o_proj` and MLP
  `down_proj`.
- Ternary `BitLinear` replacement with paper-style per-tensor scales and this
  fork's experimental row-scale mode.
- Stage-2 continued pretraining with causal-LM cross entropy.
- Stage-3 downstream fine-tuning with CE, logits KL, and MiniLM-style Q/K/V
  attention-relation distillation from a task-tuned FP16 teacher.
- Sequence-classification GLUE path for MNLI, QNLI, and SST2.
- Causal prompt-scoring GLUE path used as an earlier diagnostic.
- Optional sequence-classification output-head initialization from the FP16
  teacher, added after the first negative GLUE3 wave.
- Attention layer sweep and CE-only ablation launchers.
- Paper-style logits KL without a tau-squared multiplier, with the older
  tau-squared convention still available as an explicit compatibility option.

## Primary Sequence-Classification GLUE3 Results

Model: `Qwen/Qwen2.5-0.5B`.

Completed root:
`checkpoints/bitdistill-glue-seqcls/Qwen-Qwen2.5-0.5B`.

Training setting:

- `task_format=sequence_classification`
- `max_steps=1000`
- `per_device_batch_size=4`
- `grad_accum_steps=4`
- `max_seq_len=512`
- `lr=2e-5`
- BitDistill weights: logits KL `10.0`, attention relation KD `100.0`
- Logits KL scaling: legacy `temperature**2` convention in this completed wave;
  paper-style `none` reruns are queued separately
- Warm-up state:
  `checkpoints/bitdistill-glue/Qwen-Qwen2.5-0.5B/continued_pretrain/bitdistill-tensor/custom_state_dict.pt`

| task | FP16-SFT | BitNet-SFT | BitDistill tensor | BitDistill row | FP - tensor BitDistill | row - tensor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MNLI | `0.807641` | `0.487621` | `0.525217` | `0.516556` | `0.282425` | `-0.008660` |
| QNLI | `0.898957` | `0.596925` | `0.596925` | `0.618525` | `0.302032` | `+0.021600` |
| SST2 | `0.925459` | `0.770642` | `0.815367` | `0.808486` | `0.110092` | `-0.006881` |

Threshold result: **fail**. The best current BitDistill variant remains
11.0-30.2 accuracy points below FP16-SFT on these tasks.

## Interpretation

The results support three conservative conclusions:

1. Direct BitNet-SFT from a pretrained dense model is not enough for this setup.
2. Short-warm-up BitDistill recovers some signal on MNLI and SST2, but not
   enough to match FP16.
3. Row-scale BitDistill is not yet a quality win in task fine-tuning. It helps
   QNLI by 2.16 points but hurts MNLI and SST2 slightly in the completed wave.

The results do not support claiming a successful BitDistill reproduction, a
general 1.58-bit model converter, or a row-scale task-quality improvement.

## MNLI Diagnostics After Paper-Style Logits KL

After the first GLUE3 wave, the logits-KL implementation was corrected to match
the BitDistill equations by removing the tau-squared multiplier. A focused MNLI
diagnostic wave then tested tensor scale, row scale, and FP16-teacher head
initialization.

| run | accuracy | FP16 gap | logits temp scale | head init | note |
| --- | ---: | ---: | --- | --- | --- |
| FP16-SFT | `0.807641` | `0.000000` | - | - | task teacher/reference |
| BitNet-SFT | `0.487621` | `0.320020` | - | false | direct ternary SFT |
| BitNet-SFT + teacher head | `0.496994` | `0.310647` | - | true | head copy gives only `+0.009373` |
| BitDistill tensor, first wave | `0.525217` | `0.282425` | legacy tau-squared | false | previous best tensor |
| BitDistill row, first wave | `0.516556` | `0.291085` | legacy tau-squared | false | row hurt in first wave |
| BitDistill tensor, paper logits | `0.528477` | `0.279165` | none | false | small gain from paper-style KL |
| BitDistill row, paper logits | `0.532043` | `0.275599` | none | false | best row result |
| BitDistill tensor, paper logits + teacher head | `0.532960` | `0.274682` | none | true | best short-budget result |
| BitDistill row, paper logits + teacher head | `0.530005` | `0.277636` | none | true | head did not help row |

Interpretation: the corrected loss and head initialization improve MNLI by
`+0.007743` over the first tensor BitDistill wave, and row scale helps when
using paper-style logits without head initialization. These are real but small
effects. They do not close the roughly 27.5-point FP16 gap.

## Why This Differs From The Paper

The largest experimental differences are:

- Continued pretraining budget: current completed run is 40.96M effective token
  presentations; the paper reports 10B tokens.
- Hardware: current jobs are one-GPU constrained; the paper reports 8x MI300X.
- Search: current runs use a fixed 1000-step downstream setting; the paper
  uses learning-rate and epoch selection.
- Logits KD scaling: first completed wave used tau-squared scaling; paper-style
  no-tau-squared reruns are now queued.
- Backbone: current primary run uses Qwen2.5-0.5B; the paper reports Qwen3
  0.6B/1.7B/4B and additional backbones.
- Task head handling: first sequence-classification wave trained a new head
  from scratch. A teacher-head initialization diagnostic is now running.

## Active Follow-Up Jobs

Queued or running diagnostics at report time:

- MNLI teacher-head initialization probe:
  BitNet-SFT, BitDistill tensor, BitDistill row.
- MNLI attention-layer sweep:
  layers `-2`, `-4`, `-8`.
- MNLI CE-only ablation with and without teacher-head initialization.
- Longer Stage-2 warm-up pilot:
  20k steps, 163.84M effective token presentations.

The long warm-up pilot is still far below the paper's 10B-token budget. At the
observed Stage-2 rate of roughly 4.5k tokens/s on this setup, a true 10B-token
single-GPU warm-up would take on the order of weeks, not hours.

## Current Publishable Boundary Claim

The credible claim from the completed evidence is:

> A naive or short-warm-up retrofit of Qwen2.5-0.5B to 1.58-bit BitNet does not
> reproduce FP16 task quality on GLUE3. This reinforces that BitDistill's
> continued-pretraining and distillation budget is not optional.

The publishable opportunity remains:

- independent reproduction of BitDistill under explicitly documented resource
  limits,
- an open training implementation if upstream does not expose the full recipe,
- a row-scale `I2_SR` CPU runtime format for quality-preserving packed ternary
  inference,
- and a boundary study separating task-specific BitDistill success from
  general-LM and MoE/Kimi failure modes.
