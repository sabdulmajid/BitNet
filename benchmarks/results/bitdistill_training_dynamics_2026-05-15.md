# BitDistill Training Dynamics Audit, 2026-05-15

Overall status: **smoke_only**.

Only smoke telemetry is materialized; this validates the parser and hooks, but not a real controlled BitDistill run.

Traces: `2`. Controlled traces: `0`. Materialized controlled traces: `0`.

| trace | kind | rows | first | last | grad | A8 | dyn | final grad attn/CE | max clipped | max edge | mean flip | max scale delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| telemetry.jsonl | smoke | 2 | 1 | 2 | true | true | true | 7.838156 | 0.000000 | 0.005441 | 6.101e-06 | 5.588e-09 |
| telemetry.jsonl | smoke | 2 | 1 | 2 | true | true | true | 8.847890 | 0.000000 | 0.005337 | 3.051e-06 | 9.406e-08 |
