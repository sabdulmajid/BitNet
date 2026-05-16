# BitDistill Telemetry Coverage Audit, 2026-05-15

Overall status: **partial_observability**.

Existing telemetry is sufficient for loss-scale and static-mechanics triage, and the training script plus Slurm launcher now have opt-in hooks for the next controlled wave. The completed benchmark artifacts are still not sufficient to prove update-direction causality, because materialized component-gradient, flip-rate, scale-trajectory, and activation-saturation traces do not exist yet.

Measured diagnostics passing: `10/10`. Missing advanced diagnostics: `1`.

## Measured

| telemetry | gate | status | evidence | supports |
| --- | --- | --- | --- | --- |
| raw task loss components | pass | measured | StepMetrics fields present and 22 materialized BitDistill rows record CE, logit KD, attention KD, and weighted KD terms. | Loss-scale sanity checks and finite-run triage. |
| paper-gamma loss magnitude projection | pass | measured | Projected paper-gamma attention/CE range is 890.466502 to 1.578e+04. | The claim that gamma comparison is normalization-sensitive. |
| weighted KD-to-CE ratios on Stage-2 rows | pass | measured_when_rows_exist | 5 Stage-2 audit rows include weighted KD/CE ratios. | Controlled-run interpretation after queued rows finish. |
| final checkpoint ternary code distribution | pass | measured_offline | code fractions={'-1': 0.33324317512931406, '0': 0.33317633827964027, '1': 0.33358048659104567}; entropy=1.584962. | Static export/mechanics checks, not step-by-step training dynamics. |
| SubLN activation/logit perturbation | pass | measured_offline | inserted=48; logit relative RMS=0.768044; cosine=0.698252. | The claim that untrained SubLN surgery is not identity-preserving locally. |
| opt-in training telemetry hooks | pass | instrumented_not_materialized | train_bitdistill.py exposes telemetry.jsonl, total grad norm, optional component grad norms, ternary code fractions, scale stats, threshold-band fractions, flip-rate telemetry, and scale-drift telemetry. | Future controlled rows can record update-balance diagnostics without changing default jobs. |
| Slurm launcher telemetry pass-through | pass | instrumented_not_materialized | slurm_bitdistill_glue.sh exposes TELEMETRY_EVERY_STEPS, TELEMETRY_COMPONENT_GRAD_NORMS, and TELEMETRY_MAX_ELEMENTS_PER_LAYER to train_bitdistill.py. | Cluster jobs can materialize the new telemetry without hand-editing the launcher. |
| Q/K/V relation KD split | pass | instrumented_not_materialized | train_bitdistill.py records raw and weighted attention_q_kd, attention_k_kd, and attention_v_kd fields, and optional component-gradient telemetry can include the weighted Q/K/V terms. | Future controlled rows can identify which attention-relation term dominates. |
| BitLinear activation int8 saturation | pass | instrumented_not_materialized | train_bitdistill.py records per-telemetry-step activation A8 clipping, edge occupancy, per-token scale statistics, and absmax statistics for BitLinear student forwards. | Future controlled rows can rule activation clipping/saturation in or out as a quality limiter. |
| ternary flip-rate and scale trajectory | pass | instrumented_not_materialized | train_bitdistill.py records sampled ternary code flip fractions and scale absolute-delta statistics between emitted telemetry steps. | Future controlled rows can show whether continued pretraining is moving ternary codes and scale semantics. |

## Missing Before Stronger Causal Claims

| telemetry | status | evidence | risk |
| --- | --- | --- | --- |
| materialized training-dynamics telemetry rows | missing_materialized_run | Component-gradient, Q/K/V, activation-saturation, flip-rate, and scale-drift hooks exist, but completed controlled benchmark rows have not materialized the new traces yet. | Cannot yet make causal claims from completed rows about update direction, A8 saturation, or Stage-2 ternary dynamics. |

## Next Step

After the active queued jobs finish, launch the next controlled rows with --telemetry-every-steps and --telemetry-component-grad-norms enabled on a sparse cadence.
