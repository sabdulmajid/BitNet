# Current Evidence Snapshot

Generated: `2026-09-22T00:00:00+00:00`

Dense Qwen retrofit experiments and matching CPU runtime artifacts. Results do not establish universal model or MoE support.

| Question | Verdict | Headline evidence |
| --- | --- | --- |
| Blind absmean PTQ | **Rejected** | WikiText PPL 13.901 -> 3,813,121.803 |
| Matched BitDistill MNLI | **Not reproduced** | FP16 0.808151; best arm mean 0.757039 |
| Adaptive vs fixed-60 | **Inconclusive** | delta -0.001698; seed 95% CI [-0.010898, +0.007502] |
| Row-scale runtime contract | **Supported** | relative RMS error 1.904230 -> 0.000197 |
| Packed classifier speed | **Rejected on Xeon 4116** | I2_SR/FP16 0.650x |
| TL2_SR | **Fidelity yes; speed no** | projection bytes -12.862%; best tested speed 0.919x I2_SR |
| Kimi/MoE | **Not proven** | Only tiny Qwen2MoE plumbing exists; no Kimi checkpoint, quality, or routed CPU benchmark exists. |

All source reports and SHA-256 hashes are recorded in the adjacent JSON file.
