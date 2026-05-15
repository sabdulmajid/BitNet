# BitDistill Producer Script Audit, 2026-05-15

Overall status: `pass`.

CPU producer: `10020`.

I2_SR producer: `10019`.

Downstream rows: `38`.

Producer helper hashes: `{'benchmarks/benchmark_bitdistill_glue_cpu.py': 'b47e163aeeb5', 'benchmarks/gate_bitdistill_cpu_benchmark.py': 'fb5b2825afb8', 'benchmarks/export_bitdistill_i2sr_suite.py': '541cbe634492', 'benchmarks/convert_static_ternary_to_i2s_gguf.py': 'e61e8a35abd4', 'benchmarks/audit_bitdistill_task_formulation.py': 'd283e65243dd'}`.

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| exactly one active CPU producer | pass | jobs=[{'job_id': '10020', 'name': 'bitdistill-cpu-bench', 'state': 'PENDING', 'reason': '(Dependency)'}] |  |
| CPU producer stored script matches current script | pass | job=10020, stored=bdfc43f3e801, current=bdfc43f3e801 |  |
| CPU producer depends on all downstream jobs with afterany | pass | job=10020, deps=38/38, dependency=afterany:9943(unfulfilled),afterany:9944(unfulfilled),afterany:9945(unfulfilled) |  |
| CPU parser accepts queued run families | pass | short,longwarmup,papergamma,papergamma_row,papergamma_lr1,papergamma_lr5,papergamma_headinit |  |
| CPU producer script invokes the audited Python benchmark | pass | helper=benchmarks/benchmark_bitdistill_glue_cpu.py, sha256=b47e163aeeb5 |  |
| stale CPU jobs are not active | pass | stale=[] |  |
| exactly one active I2_SR producer | pass | jobs=[{'job_id': '10019', 'name': 'bitdistill-i2sr', 'state': 'PENDING', 'reason': '(Dependency)'}] |  |
| I2_SR producer stored script matches current script | pass | job=10019, stored=aac0b7779394, current=aac0b7779394 |  |
| I2_SR producer script invokes the audited export helper | pass | helper=benchmarks/export_bitdistill_i2sr_suite.py, sha256=541cbe634492 |  |
| producer Python helpers compile | pass | benchmarks/benchmark_bitdistill_glue_cpu.py,benchmarks/gate_bitdistill_cpu_benchmark.py,benchmarks/export_bitdistill_i2sr_suite.py,benchmarks/convert_static_ternary_to_i2s_gguf.py,benchmarks/audit_bitdistill_task_formulation.py |  |
| postprocess finalizers depend on current CPU producer | pass | cpu=10020, strict=['9993', '9994', '9995', '10019', '10020'], afterany=['9993', '9994', '9995', '10019', '10020'] |  |
| current postprocess script includes task-formulation audit | pass | slurm_bitdistill_postprocess.sh |  |
| latest strict postprocess stored script matches current script | pass | job=10022, stored=26f9a6b7969f, current=26f9a6b7969f |  |
| latest afterany postprocess stored script matches current script | pass | job=10021, stored=26f9a6b7969f, current=26f9a6b7969f |  |
