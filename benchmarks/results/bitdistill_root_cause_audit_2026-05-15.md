# BitDistill Root-Cause Audit, 2026-05-15

The current evidence supports a negative PTQ result and a positive row-scale runtime-semantics result. The CE-only BitNet-SFT baseline is now budget-viable, but paper-level BitDistill recovery and Kimi/MoE product claims remain unproven.

## Claim Ledger

| claim | status | evidence | next gate |
| --- | --- | --- | --- |
| Blind ternary PTQ is not a viable universal retrofit for tested Qwen. | supported_for_tested_setup | WikiText PPL 13.901475 -> 3.813e+06; ten-task mean 0.644169 -> 0.348671 (delta -0.295498). | None for the tested dense-Qwen setup; this is already a negative result. |
| The early weak BitNet-SFT baseline was not just broken mechanics. | supported | mechanics passed=true; best 10k CE-only MNLI=0.628935, paper anchor delta=0.020935, FP paired delta=-0.179215. | Treat CE-only BitNet-SFT as budget-viable but schedule-sensitive; use it as the baseline for BitDistill recovery, not as a success claim. |
| BitDistill paper-level recovery has not been locally reproduced. | not_proven | best tensor BitDistill MNLI=0.641671 (delta vs FP -0.166480); controlled rows complete=2/3, best controlled MNLI=0.691187 (delta vs FP -0.116964). | Finish the fixed-recipe 5k/20k/40k Stage-2 curve and require a full-validation paired trace within the FP recovery gate. |
| Loss normalization is a live reproduction risk. | supported | projected paper-gamma attention/CE range 890.466502 to 1.578e+04; materialized rows=22. | For new jobs, compare CE/logit-KD/attention-KD magnitudes before interpreting gamma sweeps. |
| Local SubLN surgery is not identity-preserving before adaptation. | supported | inserted=48; logit relative RMS drift=0.768044; cosine=0.698252; top1 agreement=0.000000. | Treat SubLN timing/init as part of the training recipe, not as harmless module insertion. |
| Row-scale I2_SR is a runtime-semantics contribution, not a Q4 quality/storage win. | supported | row-scale ten-task mean=0.499459 (delta vs FP -0.144710); I2_SR/Q4 prefill=2.298818x, decode=1.190617x, file=1.288133x, PPL=3.032323x. | Keep claims scoped to speed and scale-contract fidelity until quality improves. |
| TL2 row-scale and real Kimi/MoE product claims remain open. | not_proven | TL2 ready=false; TL2 row one-scale error=1.904230; MoE gates failed=3/9; Kimi config supported=false. | Do not foreground Kimi/MoE until trained quality, routing locality, and runtime support exist. |

## Immediate Decision Gate

Finish the fixed-recipe tensor-scale BitDistill Stage-2 curve. If accuracy rises with warm-up budget, scale continued pretraining. If it saturates far below FP16, prioritize loss normalization, SubLN timing, optimizer schedule, and attention-distillation update balance.

## Quality Anchors

| row | WikiText PPL | ten-task mean | delta vs FP | tasks |
| --- | --- | --- | --- | --- |
| FP reference | 13.901475 | 0.644169 | 0.000000 | 10 |
| naive PTQ | 3.813e+06 | 0.348671 | -0.295498 | 10 |
| row-scale QAT | 38.580065 | 0.499459 | -0.144710 | 10 |

## BitNet-SFT Baseline

| field | value |
| --- | --- |
| complete paired rows | 10 |
| best accuracy | 0.628935 |
| paper anchor | 0.608000 |
| delta vs paper anchor | 0.020935 |
| paired delta vs FP | -0.179215 |
| paired CI95 | [-0.189580, -0.168851] |
| McNemar exact p | 3.438e-240 |

## BitDistill Recovery Gate

| field | value |
| --- | --- |
| best tensor MNLI | 0.641671 |
| best tensor delta vs FP | -0.166480 |
| best row retrofit MNLI | 0.653591 |
| best row retrofit delta vs FP | -0.154559 |
| controlled rows complete | 2/3 |
| controlled all complete | false |
| controlled rows passing FP gate | 0 |
| best controlled job | 10068 |
| best controlled MNLI | 0.691187 |
| best controlled delta vs FP | -0.116964 |
| best controlled CI95 | [-0.126110, -0.107817] |

## Runtime Boundary

| Q4_K_M normalized metric | I2_SR value |
| --- | --- |
| decode_speedup | 1.190617 |
| file_ratio | 1.288133 |
| ppl_ratio | 3.032323 |
| prefill_speedup | 2.298818 |
| rss512_ratio | 1.268574 |

I2_SR is a speed-oriented proof of row-scale ternary runtime semantics. It improves decode speed versus FP16 and is faster than Q4_K_M in the audited run, but it is larger than Q4_K_M and has much worse PPL. It should not be claimed as a quality/storage win over mature Q4 quantization.

