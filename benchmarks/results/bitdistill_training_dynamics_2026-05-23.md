# BitDistill Training Dynamics Audit, 2026-05-23

Overall status: **controlled_materialized**.

Controlled training-dynamics telemetry is materialized.

Traces: `3`. Controlled traces: `3`. Materialized controlled traces: `3`.

| trace | kind | rows | first | last | attn KD weight | grad | A8 | dyn | final grad attn/CE | max clipped | max edge | mean flip | max scale delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bitdistill-tensor-20kwarmup-papergamma-headinit-telemetry-steps200 | controlled | 9 | 1 | 200 | 1.000e+05 | true | true | true | 221.384986 | 0.000000 | 0.000362 | 0.002593 | 8.845e-06 |
| bitdistill-tensor-20kwarmup-papergamma-headinit-after10069-steps200 | controlled | 9 | 1 | 200 | 1.000e+05 | true | true | true | 221.384986 | 0.000000 | 0.000362 | 0.002593 | 8.845e-06 |
| bitdistill-tensor-20kwarmup-gamma60-headinit-steps200 | controlled | 9 | 1 | 200 | 60.000001 | true | true | true | 0.346044 | 0.000000 | 0.000401 | 0.002206 | 9.844e-06 |
