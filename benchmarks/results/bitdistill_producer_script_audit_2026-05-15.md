# BitDistill Producer Script Audit, 2026-05-15

Overall status: `pass`.

CPU producer: `10025`.

I2_SR producer: `10037`.

Downstream rows: `30`.

Producer helper hashes: `{'benchmarks/benchmark_bitdistill_glue_cpu.py': 'b47e163aeeb5', 'benchmarks/gate_bitdistill_cpu_benchmark.py': 'fb5b2825afb8', 'benchmarks/export_bitdistill_i2sr_suite.py': '541cbe634492', 'benchmarks/convert_static_ternary_to_i2s_gguf.py': 'e61e8a35abd4', 'benchmarks/audit_bitdistill_task_formulation.py': '223e0603fe8e'}`.

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| exactly one active CPU producer | pass | jobs=[{'job_id': '10025', 'name': 'bitdistill-cpu-bench', 'state': 'PENDING', 'reason': '(Dependency)'}] |  |
| CPU producer stored script matches current script | pass | job=10025, stored=6579a8e68096, current=6579a8e68096 |  |
| CPU producer depends on all downstream jobs with afterany | pass | job=10025, deps=30/30, dependency=afterany:9958(unfulfilled),afterany:9959(unfulfilled),afterany:9960(unfulfilled) |  |
| CPU parser accepts queued run families | pass | short,longwarmup,papergamma,papergamma_row,papergamma_lr1,papergamma_lr5,papergamma_headinit |  |
| CPU producer script invokes the audited Python benchmark | pass | helper=benchmarks/benchmark_bitdistill_glue_cpu.py, sha256=b47e163aeeb5 |  |
| stale CPU jobs are not active | pass | stale=[] |  |
| exactly one active I2_SR producer | pass | jobs=[{'job_id': '10037', 'name': 'bitdistill-i2sr', 'state': 'PENDING', 'reason': '(Priority)'}] |  |
| I2_SR producer stored script matches current script | pass | job=10037, stored=04211cdffff2, current=04211cdffff2 |  |
| I2_SR producer script invokes the audited export helper | pass | helper=benchmarks/export_bitdistill_i2sr_suite.py, sha256=541cbe634492 |  |
| producer Python helpers compile | pass | benchmarks/benchmark_bitdistill_glue_cpu.py,benchmarks/gate_bitdistill_cpu_benchmark.py,benchmarks/export_bitdistill_i2sr_suite.py,benchmarks/convert_static_ternary_to_i2s_gguf.py,benchmarks/audit_bitdistill_task_formulation.py |  |
| postprocess finalizers depend on current CPU producer | pass | cpu=10025, strict=['9993', '9994', '9995', '10025', '10037'], afterany=['9993', '9994', '9995', '10025', '10037'] |  |
| current postprocess script includes task-formulation audit | pass | slurm_bitdistill_postprocess.sh |  |
| latest strict postprocess stored script matches current script | pass | job=10039, stored=3b309e8daea5, current=3b309e8daea5 |  |
| latest afterany postprocess stored script matches current script | pass | job=10038, stored=3b309e8daea5, current=3b309e8daea5 |  |
