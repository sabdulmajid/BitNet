# BitDistill Loss Contract Audit, 2026-05-15

Status: **loss_normalization_risk**. Verdict: **loss-normalization risk**.

This audit is not a quality result. It checks whether the local implementation and live logs make the paper-gamma setting numerically risky under the current loss normalization.

## Static Contract

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| SubLN wraps projection inputs before BitLinear replacement | pass | first_line=309, second_line=311 |  |
| Attention relation KD uses batchmean KL | pass | line=404, needle=F.kl_div(torch.log(student_rows), teacher_rows, reduction="batchmean", log_target=False) |  |
| Attention Q/K/V reduction defaults to sum | pass | line=1443, needle=parser.add_argument("--attention-qkv-reduction", choices=["sum", "mean"], default="sum") |  |
| Logits KD temperature scaling defaults to none | pass | line=1439, needle=parser.add_argument("--logit-kd-temperature-scale", choices=["none", "square"], default="none") |  |
| Stage-3 loss is direct weighted sum | pass | line=1300, needle=loss = ce + weighted_logit_kd + weighted_attention_kd |  |
| Attention weight default is local-safe, not paper gamma | pass | line=1441, needle=parser.add_argument("--attention-kd-weight", type=float, default=100.0) |  |

## Live Loss Balance

| job | label | state | step | CE | attention KD | weighted attention KD | weighted attention / CE | max weighted attention / CE | median weighted attention / CE | p95 weighted attention / CE | median CE/attention gamma | p95 CE/attention gamma | parsed steps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10068 | 20k-warmup downstream control | RUNNING | 1860 | 0.644531 | 0.016696 | 1669.633057 | 2590.461990 | 16326.870293 | 1874.689599 | 5307.021380 | 53.341231 | 113.075519 | 187 |

## Interpretation

The risk threshold is weighted-attention/CE >= `100.0`. The max observed ratio is `16326.870293`. The CE/attention gamma columns estimate the attention weight that would put raw attention KD on the same scale as CE for the observed live steps. If final BitDistill quality remains weak, the first follow-up is loss-normalization and gradient-balance telemetry, not another broad model/task sweep.
