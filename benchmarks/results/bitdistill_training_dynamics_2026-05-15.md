# BitDistill Training Dynamics Audit, 2026-05-15

Overall status: **controlled_materialized**.

Controlled training-dynamics telemetry is materialized.

Traces: `3`. Controlled traces: `1`. Materialized controlled traces: `1`.

| trace | kind | rows | first | last | grad | A8 | dyn | final grad attn/CE | max clipped | max edge | mean flip | max scale delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| telemetry.jsonl | smoke | 2 | 1 | 2 | true | true | true | 7.838156 | 0.000000 | 0.005441 | 6.101e-06 | 5.588e-09 |
| telemetry.jsonl | smoke | 2 | 1 | 2 | true | true | true | 8.847890 | 0.000000 | 0.005337 | 3.051e-06 | 9.406e-08 |
| telemetry.jsonl | controlled | 9 | 1 | 200 | true | true | true | 221.384986 | 0.000000 | 0.000362 | 0.002593 | 8.845e-06 |
