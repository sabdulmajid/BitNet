# BitDistill Method-Parity Pilots

Generated: `2026-09-04T05:50:13.343791+00:00`

Status: **complete_diagnostic**.

Quality claim: **none_diagnostic_subset_only**.

Evidence scope: **exploratory_cross_environment**.

These bounded pilots compare numerical contracts. Their partial evaluation is not a task benchmark.

Environment note: NVIDIA RTX A4500; torch 2.6.0+cu118; Transformers 5.7.0; Datasets 2.18.0; parser-default seed 1234; source 18ec2c9 includes the SubLN dtype portability fix. This screens numerical contracts and is not the reference-environment quality run.

| case | status | task format | relation | split | balance | median grad AD/CE | max grad AD/CE | median gamma | diagnostic accuracy | blockers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seqcls-cosine-s8-fixed | complete | sequence_classification | cosine | 8 | fixed | 119.817 | 177.285 | 100000 | 0.498047 | none |
| seqcls-cosine-s1-fixed | complete | sequence_classification | cosine | 1 | fixed | 69.2248 | 108.878 | 100000 | 0.470703 | none |
| seqcls-scaled-dot-s1-fixed | complete | sequence_classification | scaled_dot | 1 | fixed | 61810.6 | 109257 | 100000 | 0.328125 | none |
| seqcls-cosine-s1-adaptive | complete | sequence_classification | cosine | 1 | gradnorm_ema | 0.273975 | 0.713386 | 146.271 | 0.492188 | none |
| causal-cosine-s1-fixed | complete | causal_lm | cosine | 1 | fixed | 57.3127 | 172.975 | 100000 | 0.349609 | none |
| causal-cosine-s1-adaptive | complete | causal_lm | cosine | 1 | gradnorm_ema | 0.105523 | 0.34735 | 109.98 | 0.371094 | none |

## Paired Diagnostics

| comparison | status | n | delta | paired 95% CI | candidate wins | reference wins | McNemar p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| seqcls_split_s1_minus_s8 | pass | 512 | -0.0273438 | -0.06890016317213879, 0.01421266317213879 | 52 | 66 | 0.231266 |
| seqcls_scaled_dot_minus_cosine | pass | 512 | -0.142578 | -0.19948927214017392, -0.08566697785982608 | 79 | 152 | 1.79485e-06 |
| seqcls_adaptive_minus_fixed | pass | 512 | 0.0214844 | -0.035448939099578365, 0.07841768909957836 | 116 | 105 | 0.501245 |
| causal_adaptive_minus_fixed | pass | 512 | 0.0214844 | -0.04203150451862761, 0.08500025451862761 | 143 | 132 | 0.546572 |

## Decision Rule

Use these pilots to reject numerically unstable contracts and verify adaptive balancing. Do not select a downstream-quality winner from 512 examples or 120 steps. A full run requires an explicit paper-definition choice, all 9,815 MNLI examples, paired predictions, and replication.
