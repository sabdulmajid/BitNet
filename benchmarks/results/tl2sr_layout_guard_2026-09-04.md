# TL2_SR Layout Guard

Generated: `2026-09-04T12:37:34.524445+00:00`. Status: **pass**.

| case | exit code | expected | result |
| --- | ---: | --- | --- |
| matching BM64 runtime | 0 | accept | pass |
| mismatched BM32 runtime | 1 | reject | pass |

The mismatch is accepted as evidence only when the loader exits nonzero and reports
`TL2_SR kernel-layout mismatch` with the artifact and runtime fingerprints.
