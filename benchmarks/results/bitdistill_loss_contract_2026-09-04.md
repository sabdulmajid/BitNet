# BitDistill Loss Contract Audit, 2026-09-04

Status: **loss_normalization_risk**. Verdict: **loss-normalization risk**.

This audit is not a quality result. It checks whether the local implementation and live logs make the paper-gamma setting numerically risky under the current loss normalization.

## Static Contract

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| SubLN wraps projection inputs before BitLinear replacement | pass | first_line=676, second_line=678 |  |
| Attention relation KD uses batchmean KL | pass | line=947, needle=F.kl_div(torch.log(student_rows), teacher_rows, reduction="batchmean", log_target=False) |  |
| Attention Q/K/V reduction defaults to sum | pass | line=2224, needle=parser.add_argument("--attention-qkv-reduction", choices=["sum", "mean"], default="sum") |  |
| Attention relation definition is explicit | pass | line=2223, needle=parser.add_argument("--attention-relation-mode", choices=["cosine", "scaled_dot"], default="cosine") |  |
| Gradient-balanced attention weighting is available | pass | line=2225, needle=parser.add_argument("--attention-kd-balance", choices=["fixed", "gradnorm_ema"], default="fixed") |  |
| Task formulation provenance is recorded | pass | line=2117, needle="task_formulation_contract": task_formulation_contract(args, eval_metrics) |  |
| Logits KD temperature scaling defaults to none | pass | line=2219, needle=parser.add_argument("--logit-kd-temperature-scale", choices=["none", "square"], default="none") |  |
| Stage-3 loss is direct weighted sum | pass | line=1992, needle=loss = ce + weighted_logit_kd + weighted_attention_kd |  |
| Attention weight default is local-safe, not paper gamma | pass | line=2221, needle=parser.add_argument("--attention-kd-weight", type=float, default=100.0) |  |

## Live Loss Balance

| job | label | state | step | CE | attention KD | weighted attention KD | weighted attention / CE | max weighted attention / CE | median weighted attention / CE | p95 weighted attention / CE | median CE/attention gamma | p95 CE/attention gamma | parsed steps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10069 | 5k-warmup downstream control | not_in_squeue | 10000 | 0.279297 | 0.013180 | 1317.987915 | 4718.947626 | 15427.580675 | 1637.956123 | 3908.755310 | 61.052380 | 120.050236 | 1001 |
| 10068 | 20k-warmup downstream control | not_in_squeue | 10000 | 0.200195 | 0.011902 | 1190.173462 | 5945.070866 | 37819.641342 | 1729.105844 | 6080.253825 | 57.831811 | 135.086455 | 1001 |
| 10169 | 40k-warmup downstream control | not_in_squeue | 10000 | 0.339844 | 0.010844 | 1084.440674 | 3190.995498 | 24038.027696 | 1755.253104 | 7327.265686 | 56.969902 | 137.370117 | 1001 |

## Interpretation

The risk threshold is weighted-attention/CE >= `100.0`. The max observed ratio is `37819.641342`. The CE/attention gamma columns estimate the attention weight that would put raw attention KD on the same scale as CE for the observed live steps. If final BitDistill quality remains weak, the first follow-up is loss-normalization and gradient-balance telemetry, not another broad model/task sweep.
