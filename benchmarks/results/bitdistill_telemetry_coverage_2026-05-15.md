# BitDistill Telemetry Coverage Audit, 2026-05-15

Overall status: **partial_observability**.

Existing telemetry is sufficient for loss-scale and static-mechanics triage, and the training script plus Slurm launcher now have opt-in hooks for the next controlled wave. The completed benchmark artifacts are still not sufficient to prove update-direction causality, because materialized gradient-component, flip-rate, scale-trajectory, and activation-saturation traces do not exist yet.

Measured diagnostics passing: `7/7`. Missing advanced diagnostics: `5`.

## Measured

| telemetry | gate | status | evidence | supports |
| --- | --- | --- | --- | --- |
| raw task loss components | pass | measured | StepMetrics fields present and 22 materialized BitDistill rows record CE, logit KD, attention KD, and weighted KD terms. | Loss-scale sanity checks and finite-run triage. |
| paper-gamma loss magnitude projection | pass | measured | Projected paper-gamma attention/CE range is 890.466502 to 1.578e+04. | The claim that gamma comparison is normalization-sensitive. |
| weighted KD-to-CE ratios on Stage-2 rows | pass | measured_when_rows_exist | 5 Stage-2 audit rows include weighted KD/CE ratios. | Controlled-run interpretation after queued rows finish. |
| final checkpoint ternary code distribution | pass | measured_offline | code fractions={'-1': 0.33324317512931406, '0': 0.33317633827964027, '1': 0.33358048659104567}; entropy=1.584962. | Static export/mechanics checks, not step-by-step training dynamics. |
| SubLN activation/logit perturbation | pass | measured_offline | inserted=48; logit relative RMS=0.768044; cosine=0.698252. | The claim that untrained SubLN surgery is not identity-preserving locally. |
| opt-in training telemetry hooks | pass | instrumented_not_materialized | train_bitdistill.py exposes telemetry.jsonl, total grad norm, optional component grad norms, ternary code fractions, scale stats, and threshold-band fractions. | Future controlled rows can record update-balance diagnostics without changing default jobs. |
| Slurm launcher telemetry pass-through | pass | instrumented_not_materialized | slurm_bitdistill_glue.sh exposes TELEMETRY_EVERY_STEPS, TELEMETRY_COMPONENT_GRAD_NORMS, and TELEMETRY_MAX_ELEMENTS_PER_LAYER to train_bitdistill.py. | Cluster jobs can materialize the new telemetry without hand-editing the launcher. |

## Missing Before Stronger Causal Claims

| telemetry | status | evidence | risk |
| --- | --- | --- | --- |
| gradient norm by loss component | missing_materialized_run | Opt-in component-gradient telemetry exists, but no completed benchmark row has materialized those traces yet. | Cannot prove which objective term dominates the update direction. |
| ternary flip rate per step/layer | missing_materialized_run | Final ternary code fractions are audited and opt-in code telemetry exists, but consecutive-step code transitions are not materialized for completed benchmark rows. | Cannot tell whether continued pretraining is moving codes or only tuning dense residual/head parameters. |
| scale trajectory per layer | missing_materialized_run | Final scale histograms and opt-in scale telemetry exist; per-step tensor/row scale drift is not materialized for completed benchmark rows. | Cannot directly verify whether Stage-2 learns BitNet-like scale semantics over time. |
| activation int8 saturation rate | missing | A8 code path is audited statically; runtime saturation statistics are not recorded. | Cannot rule out activation clipping/saturation as a hidden quality limiter. |
| Q/K/V relation KD split | missing | The saved metric records aggregate attention KD, not separate Q, K, and V relation losses. | Cannot identify which attention relation term is failing or dominating. |

## Next Step

After the active queued jobs finish, launch the next controlled rows with --telemetry-every-steps and --telemetry-component-grad-norms enabled on a sparse cadence.
