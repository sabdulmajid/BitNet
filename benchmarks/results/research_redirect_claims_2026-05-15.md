# Research Redirect Claim Gate, 2026-05-15

This audit turns the redirected research framing into a machine-checkable claim ledger.

| field | value |
| --- | --- |
| status | claim_guardrail_passed |
| passed | true |
| guardrails | 7/7 |

## Claims

| claim | status | safe public label | evidence | next gate |
| --- | --- | --- | --- | --- |
| Blind arbitrary FP/BF16-to-ternary PTQ retrofit | rejected_for_tested_dense_qwen | Do not claim a universal converter. | FP mean=0.644169 over 10 tasks; PTQ mean=0.348671 over 10 tasks; delta=-0.295498; WikiText PPL ratio=2.743e+05. |  |
| QAT/distillation recovery over blind PTQ | partially_supported | Claim partial recovery only, not FP-quality recovery. | row-scale mean=0.499459 over 10 tasks; recovery over PTQ=0.150788; delta vs FP=-0.144710; row WikiText PPL=38.580065. |  |
| Paper-level BitDistill reproduction | not_proven | Keep labeled as paper-inspired until full controlled rows close the FP gap. | controlled rows complete=2/3; passed FP recovery=0; best controlled accuracy=0.691187; best paired delta vs FP=-0.116964. |  |
| Row-scale I2_SR runtime semantics | supported_as_retrofit_variant | Claim runtime/scale-contract viability, not a Q4 quality/storage win. | I2_SR/Q4 prefill=2.298818x; decode=1.190617x; file=1.288133x; PPL=3.032323x. |  |
| TL2 row-scale runtime readiness | blocked | Use I2_SR until TL2 has explicit row/group-scale metadata and kernels. | runtime_ready=false; current row-scale error=1.904230; exact row-scale design error=0.000197; finite TL2 quality=false. |  |
| Native packed sequence-classification deployment | prototype_only | Implement native classifier head metadata/runtime before product claims. | same artifact ready=false; sidecar examples=128; sidecar accuracy=0.609375; agreement=0.914062. |  |
| Kimi/MoE product support | not_proven | Keep MoE/Kimi in future work until trained quality, routing locality, and runtime are measured. | Kimi config supported=false; local Kimi artifacts=0; failed MoE gates=3/9. |  |

## Safe Public Summary

- Blind PTQ-to-ternary is rejected for the tested dense-Qwen setup.
- QAT/distillation is a partial recovery path, not an FP-quality result yet.
- Row-scale I2_SR is a runtime-semantics contribution for compatible dense causal artifacts.
- Paper-level BitDistill, native packed classifier deployment, TL2 row-scale, and Kimi/MoE remain gated.
