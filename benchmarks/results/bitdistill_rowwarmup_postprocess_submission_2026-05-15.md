# BitDistill Row-Warmup Postprocess Submission, 2026-05-15

Purpose: automatically summarize and gate the clean row-scale Stage-2 -> row Stage-3 branch after its downstream GLUE3 jobs finish.

## Submitted Jobs

| job | dependency mode | dependency jobs | partition | output |
| --- | --- | --- | --- | --- |
| `10035` | `afterany` | `10029:10030:10031:10032:10033:10034` | `midcard` | `logs/bitdistill-rowwarmup-postprocess-any-10035.out` |
| `10036` | `afterok` | `10029:10030:10031:10032:10033:10034` | `midcard` | `logs/bitdistill-rowwarmup-postprocess-10036.out` |

## Inputs

| input | value |
| --- | --- |
| row warm-up job | `10028` |
| row gamma100 TSV | `benchmark_results/bitdistill_rowwarmup_downstream_gamma100_20260515.tsv` |
| row paper-gamma TSV | `benchmark_results/bitdistill_rowwarmup_downstream_papergamma_20260515.tsv` |
| report date | `2026-05-15` |

## Postprocess Outputs

| output | path |
| --- | --- |
| row gamma100 monitor | `benchmarks/results/bitdistill_row_warmup_monitor_2026-05-15.md` |
| row paper-gamma monitor | `benchmarks/results/bitdistill_row_warmup_papergamma_monitor_2026-05-15.md` |
| row warm-up health | `benchmarks/results/bitdistill_row_warmup_health_2026-05-15.md` |
| row variant summary | `benchmarks/results/bitdistill_rowwarmup_variant_summary_2026-05-15.md` |
| row gate | `benchmarks/results/bitdistill_rowwarmup_gate_2026-05-15.md` |

## Provenance Checks

| check | status | evidence |
| --- | --- | --- |
| afterany postprocess dependency includes all six row downstream jobs | pass | `10035` depends on `10029`-`10034` with `afterany` |
| strict postprocess dependency includes all six row downstream jobs | pass | `10036` depends on `10029`-`10034` with `afterok` |
| stored postprocess scripts match current script | pass | `0884481545f1` for both stored scripts and `slurm_bitdistill_rowwarmup_postprocess.sh` |
| postprocess runs row-warmup gate | pass | stored script calls `benchmarks/gate_bitdistill_rowwarmup.py` |

## Interpretation

- `10035` is the diagnostic guardrail: it should still emit monitor and gate reports if one downstream job fails.
- `10036` is the strict-success path: it only runs if all six row downstream jobs complete successfully.
- These postprocess jobs do not alter the strict tensor-warmup matrix; they report the separate row-scale novelty branch.
