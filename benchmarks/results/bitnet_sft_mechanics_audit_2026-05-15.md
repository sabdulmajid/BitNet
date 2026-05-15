# BitNet-SFT Mechanics Audit, 2026-05-15

Overall status: **PASS**.

The audited implementation clears the basic mechanical checks for the paper-style tensor-scale BitNet-SFT baseline: exact absmean STE equation, A8 path, dense classifier head, 168 decoder projection tensors, scalar tensor scales, and no accidental embedding/norm/head ternarization. The weak early BitNet-SFT row is best explained by budget/schedule, while the remaining research blocker is BitDistill recovery toward FP16 accuracy and the non-identity behavior of SubLN without matched warmup.



## Headline Numbers

| quantity | value |
| --- | --- |
| default BitNet-SFT MNLI | 0.487621 |
| weights-only BitNet-SFT MNLI | 0.493734 |
| SubLN-only BitNet-SFT MNLI | 0.350280 |
| best budget BitNet-SFT MNLI | 0.628935 |
| paper BitNet-SFT MNLI anchor | 0.608000 |
| ternary tensors | 168 |
| code fractions | {"-1": 0.33324317512931406, "0": 0.33317633827964027, "1": 0.33358048659104567} |
| code entropy bits | 1.584962 |

## A-J Focused Checks

| item | status | evidence | implication |
| --- | --- | --- | --- |
| C. BitLinear equation | pass | TernaryWeightSTE implements alpha=mean(abs(W)), round(W/alpha), clamp [-1,1], with identity STE backward. | The tensor-scale path matches the paper-style absmean ternary projection. Row-scale remains a fork variant. |
| B. Activation quantization equation | pass | AbsmaxActivationSTE uses per-token absmax / 127 with int8 clamp and identity backward. | A8 is present and can be ablated with --no-activation-quantization. |
| E. Dense classifier head default | pass | Default exclude_linear_regex is score\|classifier. | Sequence-classification head treatment is not the obvious reason for the weak early BitNet-SFT row. |
| D. SubLN insertion points | pass | add_subln_to_qwen_blocks wraps self_attn.o_proj and mlp.down_proj. | Placement is mechanically consistent with the paper description, but the exact timing/init still matters. |
| F. Projection replacement count | pass | ternary=168/168; families={'down_proj': 24, 'gate_proj': 24, 'k_proj': 24, 'o_proj': 24, 'q_proj': 24, 'up_proj': 24, 'v_proj': 24} | All Q/K/V/O and MLP gate/up/down decoder projections are represented as ternary tensors. |
| G. Non-projection tensors stay dense | pass | score_dense=True, score_ternary=False, forbidden=[] | The audit does not find accidental ternarization of embeddings, norms, or the sequence-classification head. |
| Tensor-scale checkpoint scales | pass | scale_numel_by_key={'1': 168} | The audited paper-style checkpoint stores one scalar scale per projection tensor, not row-scale metadata. |
| H. Ternary code distribution is three-symbol | pass | fractions={'-1': 0.33324317512931406, '0': 0.33317633827964027, '1': 0.33358048659104567}, entropy_bits=1.584962/1.584963 | Storage has the expected 1.58-bit ternary alphabet; this is also the information bottleneck that blind PTQ cannot avoid. |
| A/B. A8 ablation is not the primary gap | pass | weights_only=0.493734, W1.58A8=0.487621, delta=0.006113 | Disabling activation quantization changes MNLI only slightly in the local control; ternary training/recipe dominates the gap. |
| I/J. SubLN is not identity-preserving locally | pass | logit_rel_rms=0.768044, top1_agreement=0.000000, subln_only=0.350280, default=0.487621 | SubLN should be treated as architecture surgery requiring matched warmup/distillation, not as a harmless drop-in for short SFT. |
| Budget explanation for weak early baseline | pass | best_budget_accuracy=0.628935, paper_bitnet_sft_anchor=0.608000, default=0.487621 | The earliest BitNet-SFT failure was substantially undertraining/schedule. The active blocker is now FP16-level BitDistill recovery. |

## Redirected Next Step

Do not broaden MoE/Kimi or additional row-scale claims from this audit. The next decisive experiment is a controlled BitDistill recovery run on the now-validated BitNet-SFT baseline: fixed MNLI sequence classification, tensor-scale paper-style first, full validation traces, and loss-component telemetry.

