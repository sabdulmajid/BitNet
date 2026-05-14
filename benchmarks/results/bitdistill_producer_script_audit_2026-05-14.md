# BitDistill Producer Script Audit, 2026-05-14

Overall status: `pass`.

CPU producer: `10006`.

I2_SR producer: `9949`.

Downstream rows: `38`.

Producer helper hashes: `{'benchmarks/benchmark_bitdistill_glue_cpu.py': '6d0a046ccc6d', 'benchmarks/gate_bitdistill_cpu_benchmark.py': '7ac4365c86e3', 'benchmarks/export_bitdistill_i2sr_suite.py': '541cbe634492', 'benchmarks/convert_static_ternary_to_i2s_gguf.py': 'e61e8a35abd4'}`.

## Checks

| check | status | evidence | blocker |
| --- | --- | --- | --- |
| exactly one active CPU producer | pass | jobs=[{'job_id': '10006', 'name': 'bitdistill-cpu-bench', 'state': 'PENDING', 'reason': '(Dependency)'}] |  |
| CPU producer stored script matches current script | pass | job=10006, stored=37b640d41ee6, current=37b640d41ee6 |  |
| CPU producer depends on all downstream jobs with afterany | pass | job=10006, deps=38/38, dependency=afterany:9943(unfulfilled),afterany:9944(unfulfilled),afterany:9945(unfulfilled) |  |
| CPU parser accepts queued run families | pass | short,longwarmup,papergamma,papergamma_row,papergamma_lr1,papergamma_lr5,papergamma_headinit |  |
| CPU producer script invokes the audited Python benchmark | pass | helper=benchmarks/benchmark_bitdistill_glue_cpu.py, sha256=6d0a046ccc6d |  |
| stale CPU jobs are not active | pass | stale=[] |  |
| exactly one active I2_SR producer | pass | jobs=[{'job_id': '9949', 'name': 'bitdistill-i2sr', 'state': 'PENDING', 'reason': '(Dependency)'}] |  |
| I2_SR producer stored script matches current script | pass | job=9949, stored=57ec02ab1d06, current=57ec02ab1d06 |  |
| I2_SR producer script invokes the audited export helper | pass | helper=benchmarks/export_bitdistill_i2sr_suite.py, sha256=541cbe634492 |  |
| producer Python helpers compile | pass | benchmarks/benchmark_bitdistill_glue_cpu.py,benchmarks/gate_bitdistill_cpu_benchmark.py,benchmarks/export_bitdistill_i2sr_suite.py,benchmarks/convert_static_ternary_to_i2s_gguf.py |  |
| postprocess finalizers depend on current CPU producer | pass | cpu=10006, strict=['9992', '9993', '9994', '9995', '10006'], afterany=['9992', '9993', '9994', '9995', '10006'] |  |
