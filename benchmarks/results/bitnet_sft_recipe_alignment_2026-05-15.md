# BitNet-SFT Recipe Alignment Audit, 2026-05-15
Verdict: the source-level implementation is mechanically plausible, but the local BitNet-SFT accuracy is not recipe-matched to the paper anchor yet.

- Default BitNet-SFT MNLI: `0.487621`.
- Paper BitNet-SFT MNLI anchor: `0.608000`.
- Best completed budget-sweep row: `0.523892`.

| check | status | evidence | risk |
| --- | --- | --- | --- |
| Ternary weight formula is absmean STE | pass | TernaryWeightSTE uses alpha=mean(abs(W)), round(W/alpha), clamp [-1,1], identity backward. | Formula is paper-style for tensor scale; row scale is a fork variant. |
| Activation quantization is per-token absmax int8 STE | pass | AbsmaxActivationSTE quantizes x with per-token absmax / 127 and identity backward. | A8 ablation shows this is not the dominant MNLI gap locally. |
| Sequence-classification head is excluded from BitLinear replacement by default | pass | Default exclude regex is score|classifier. | Dense-head treatment appears aligned for GLUE sequence classification. |
| SubLN is inserted before attention output and FFN down projections | pass | add_subln_to_qwen_blocks wraps o_proj and down_proj with RMSNorm before projection. | The local SubLN-only control worsens MNLI, so placement/timing/init still need recipe audit. |
| BitLinear replacement happens after SubLN insertion | pass | prepare_bitnet_student inserts SubLN first, then replaces nested nn.Linear projections. | This is mechanically coherent, but may not match the paper's exact initialization/training timing. |
| Default BitNet-SFT reaches paper anchor | fail | default=0.487621, paper=0.608000, gap=0.120379 | Primary blocker: BitDistill recovery cannot be interpreted until this baseline is explained. |
| Best completed budget row reaches paper anchor | fail | best_completed=0.523892, steps=1000, lr=5e-5 | Longer 3000/10000-step rows are needed to distinguish undertraining from equation mismatch. |
| Activation quantization explains the gap | fail | weights_only=0.493734, W1.58A8=0.487621 | A8 removal improves only slightly, so the problem is mostly ternary training/recipe. |
| SubLN-only local control explains the gap | fail | subln_only=0.350280, default=0.487621 | SubLN likely requires exact paper timing/budget or current insertion differs in a material way. |

## Next Actions
1. Finish 3000/10000-step BitNet-SFT budget curve.
2. If the curve saturates low, audit BitLinear/SubLN equation parity against the paper implementation.
3. Keep row-scale results separate from paper-reproduction labels.
4. Do not broaden MoE/Kimi claims until dense BitNet-SFT is explained.
