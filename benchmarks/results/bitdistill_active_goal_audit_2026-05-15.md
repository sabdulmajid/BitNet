# BitDistill Active Goal Audit, 2026-05-15

This audit maps the current BitDistill reproduction/productization goal to concrete artifacts. It is not a success declaration.

## Verdict

Objective achieved: `False`.

Completion status: `partial`.

Complete rows: `2` / `5`.

Pending rows: `0`.

Tensor warm-up progress: `20000` / `20000` (`1.000000`).

Row warm-up progress: `20000` / `20000` (`1.000000`).

Runtime gates: row-scale complete=`True`, row-warmup complete=`True`, I2_SR=`True`, local I2_SR=`True`, CPU=`True`, Xeon CPU=`True`, scoped CPU=`True`.

## Prompt-To-Artifact Checklist

| requirement | status | evidence | remaining gap |
| --- | --- | --- | --- |
| Reproduce BitDistill GLUE3 baseline on Qwen2.5-0.5B with FP16-SFT, BitNet-SFT, and BitDistill | partial | FP16 tasks=['mnli', 'qnli', 'sst2']; BitNet tasks=['mnli', 'qnli', 'sst2']; gamma100 rows=3/3; strict paper rows=3/3; paper row rows=3/3; matrix=38/38, inferred=0; warm-up=20000/20000; LR/head-init search is complete, did not pass | Gamma=100, strict paper-gamma tensor, strict paper-gamma row, LR/head-init, clean row-warmup gamma=100, and clean row-warmup paper-gamma searches are complete and below the FP16-gap target; full-budget and Qwen3/backbone-scale candidates remain pending. |
| Implement SubLN, Stage-2 CE, Stage-3 CE+logits KL+attention-relation KD, and layer selection | complete | smoke=True, smoke checks=40, failed features=[] |  |
| Compare paper-style per-tensor BitDistill against row-scale BitDistill | partial | tensor-warmup row gate complete=True, passed=False; row-warmup gate complete=True, passed=False | Gamma=100 and paper-gamma tensor-warmup row comparisons are complete and do not pass; clean row-warmup gamma=100 and clean row-warmup paper-gamma are complete and also do not pass. |
| Export row-scale checkpoints through I2_SR and benchmark CPU speed, memory/RSS, and task quality on Xeon | complete | I2_SR gate=True (6/6 rows); local isolated I2_SR=True (6/6 rows); full CPU gate=True on AMD Ryzen Threadripper PRO 5945WX 12-Cores (33/33 critical rows); Xeon full CPU gate=True on Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz (33/33 critical rows); scoped CPU slice=True on Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz (15/15 critical rows) |  |
| Define publishable scope: independent reproduction, open training implementation, row-scale I2_SR extension, boundary study, and MoE/Kimi limits | partial | product scope=research_mvp_only; supported=5; unsupported=4; paper gaps=['Backbone', 'Baselines', 'Stage-2 warm-up', 'Hyperparameter search', 'Hardware/resources'] | Strict tensor LR/head-init, clean row-warmup gamma=100, and clean row-warmup paper-gamma searches are complete and negative; remaining quality claims need full-budget and Qwen3/backbone-scale evidence. |

## Open Requirements

| requirement |
| --- |
| Reproduce BitDistill GLUE3 baseline on Qwen2.5-0.5B with FP16-SFT, BitNet-SFT, and BitDistill |
| Compare paper-style per-tensor BitDistill against row-scale BitDistill |
| Define publishable scope: independent reproduction, open training implementation, row-scale I2_SR extension, boundary study, and MoE/Kimi limits |

## Inputs

| input | path |
| --- | --- |
| reproduction_json | benchmark_results/bitdistill_reproduction_gate_2026-05-15.json |
| matrix_json | benchmark_results/bitdistill_job_matrix_audit_2026-05-15.json |
| monitor_json | benchmark_results/bitdistill_job_monitor_2026-05-15.json |
| row_monitor_json | benchmark_results/bitdistill_row_warmup_monitor_2026-05-15.json |
| rowwarmup_json | benchmark_results/bitdistill_rowwarmup_gate_2026-05-15.json |
| smoke_json | benchmark_results/bitdistill_smoke_contract_2026-05-15.json |
| paper_alignment_json | benchmark_results/bitdistill_paper_alignment_2026-05-15.json |
| i2sr_json | benchmark_results/bitdistill_i2sr_export_gate_2026-05-15.json |
| i2sr_local_json | benchmark_results/bitdistill_i2sr_export_gate_local_2026-05-15.json |
| cpu_json | benchmark_results/bitdistill_glue_cpu_gate_2026-05-15.json |
| cpu_xeon_json | benchmark_results/bitdistill_glue_cpu_xeon_gate_2026-05-15.json |
| cpu_fast_json | benchmark_results/bitdistill_glue_cpu_fast_gate_2026-05-15.json |
| product_json | benchmark_results/product_scope_gate_2026-05-15.json |
