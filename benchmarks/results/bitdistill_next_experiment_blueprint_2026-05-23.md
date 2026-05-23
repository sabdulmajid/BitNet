# BitDistill Next Experiment Blueprint

Generated: `2026-05-23T18:57:53.334354+00:00`

Status: **pending_655m_downstream**.

Quality claim: **experiment_blueprint_not_benchmark**.

## Current Recommendation

Wait for the active 655.36M Stage-2 producer, downstream MNLI, and postprocess reports.

## Current Action

| field | value |
| --- | --- |
| action | wait_and_watch_655m_gate |
| runnable now | true |
| why | The active 655.36M producer/downstream chain is already queued; launching another broad run would confound the token-budget curve. |
| claim boundary | status only; quality_claim remains none until ingestion is ingested_reports_rebuilt |

## Evidence Required

| required evidence |
| --- |
| 655M Stage-2 manifest |
| 655M downstream metrics.json |
| 655M downstream eval_predictions.jsonl |
| rebuilt controlled curve and next-decision report |

## Commands

```bash
python benchmarks/run_active_gate_watchdog.py
```

```bash
python benchmarks/audit_stage2_655m_ingestion.py
```

## Action Catalog

| decision status | action | runnable now | claim boundary |
| --- | --- | --- | --- |
| pending_no_controlled_rows | materialize_controlled_row | false | status repair only; no quality claim |
| pending_655m_downstream | wait_and_watch_655m_gate | true | status only; quality_claim remains none until ingestion is ingested_reports_rebuilt |
| hold_for_gamma_balance | wait_for_gamma60_telemetry | true | diagnostic only; gamma60 telemetry is not a quality benchmark |
| run_gamma_balanced_downstream | run_matched_gamma60_mnli_downstream | false | single MNLI ablation; do not broaden to QNLI/SST2 until paired MNLI result is ingested |
| extend_stage2_curve | prepare_next_controlled_stage2_point | false | budget-curve extension only; keep recipe fixed and do not add new task axes |
| replicate_recovery_gate | replicate_passing_mnli_then_expand_glue | false | reproducibility gate; QNLI/SST2 remain gated behind replicated MNLI |
| pause_broad_stage2_audit_recipe | stop_broad_scaling_and_audit_recipe | true | root-cause audit only; do not submit larger Stage-2 runs before resolving recipe mismatch |
| ambiguous_recovery_continue_with_controls | choose_one_narrow_ablation | false | one-axis ablation only |

## Nonclaims

| nonclaim |
| --- |
| This report does not add benchmark evidence. |
| A runnable command is not permission to update quality claims. |
| Broad sweeps remain disallowed until the 655M gate is ingested. |

## Source Paths

| artifact | path |
| --- | --- |
| next_decision | benchmarks/results/bitdistill_next_decision_2026-05-23.json |
| stage2_ingestion | benchmarks/results/stage2_655m_ingestion_2026-05-23.json |
| gamma_balance | benchmarks/results/gamma60_gradient_balance_2026-05-23.json |

This blueprint is decision support. It should be regenerated after the 655M downstream row and gamma telemetry complete.
