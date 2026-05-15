# Product Scope Gate, 2026-05-15

This gate separates what can be published or productized from what remains unsupported.

Scope status: `research_mvp_only`.

Publishable angle: negative arbitrary-retrofit result plus dense-Qwen row-scale recovery path, with BitDistill reproduction and I2_SR row-scale extensions gated separately.

| claim | status | evidence | scope | blocker |
| --- | --- | --- | --- | --- |
| One-click lossless arbitrary FP/BF16-to-ternary retrofit | unsupported | Qwen1.5B FP mean=0.644169; naive PTQ mean=0.348671; FP WikiText PPL=13.901475466625362; PTQ WikiText PPL=3813121.80332679 | Do not market as arbitrary lossless conversion. | Blind PTQ destroys quality in both math and model-level artifacts. |
| Dense Qwen negative result plus QAT/distillation recovery path | supported | row-scale mean=0.499459; row-PTQ paired delta=+0.150788 [+0.053427, +0.248149]; row-FP paired delta=-0.144710 [-0.185756, -0.103664]; row WikiText PPL=38.58006540148406 | Qwen2.5 dense 1.5B evidence in this fork. |  |
| CPU row-scale ternary inference for dense Qwen through stable I2_SR | supported | I2_SR PPL=38.8477; file=1211.3 MiB; prompt=211.67 tok/s; decode=19.07 tok/s; active gate=True; patch gate=True | Dense Qwen2.5-1.5B I2_SR evidence in this fork on Intel Xeon Silver 4116. |  |
| BitDistill paper-level GLUE reproduction on Qwen2.5-0.5B | unsupported | paper tensor complete=True; passed=False; LR/headinit complete=True; LR/headinit passed=False; gamma100 full rows=3/3; gamma100 gap-pass rows=0/3; strict paper full rows=3/3; strict paper gap-pass rows=0/3; expected eval={'mnli': 9815, 'qnli': 5463, 'sst2': 872}; paired status=pending; paired complete=20/44; cpu gate=False (0/33 rows, full-quality=0/33) | Claim only after MNLI/QNLI/SST2 full-validation BitDistill rows are within the configured FP16 gap and paired traces/CPU gates are complete. | Gamma=100, strict paper-gamma tensor, strict paper-gamma row, and LR/head-init runs are complete but below the FP16-gap gate; paired-trace coverage and CPU full-quality rows are still missing or incomplete. |
| BitDistill task-specific packed ternary runtime support | supported | causal packed-ternary gate passed=True; complete rows=6/6; gate=benchmark_results/bitdistill_i2sr_export_gate_2026-05-15.json | Only causal prompt-scoring BitDistill checkpoints can use this packed path; tensor baselines use I2_S, row-scale novelty runs use I2_SR, and sequence-classification heads remain PyTorch-only unless runtime support is added. |  |
| Default committed runtime supports stable row-scale I2_SR | supported | active productization gate passed=True; promotion_ready=True; failed_gates=0 | Stable I2_SR is available in the active source state when this gate is supported. |  |
| Direct packed row-scale GGUF export is product-safe by default | supported | direct packed verdict=True; candidate_i2sr_quality_valid=True | Direct writer can be used for the audited dense-Qwen I2_SR path when this gate is supported. |  |
| TL2 product support for the strong row-scale Qwen checkpoint | unsupported | Qwen0.5B TL2 PPL=NaN; Qwen1.5B row-scale one-scale error=1.9042302114103853; group2 fp16 design error=0.0986922473661489; group32 fp16 design error=0.1428438042225288; exact row-fp16 design error=0.00019744640689756221 at 1.23046875 MiB | Exclude TL2 from MVP claims. | Current TL2 scale semantics are incompatible with row-scale checkpoint quality; fixing it requires row/group-scale metadata and generated kernels that index those scales. |
| MoE/Kimi retrofit and CPU runtime support | unsupported | Kimi artifacts=0; Kimi source matches=0; tiny Qwen2MoE FP16 fixture passed=True; fixture arch=qwen2moe; fixture RSS MiB=105.09765625; tiny Qwen2MoE ternary I2_SR fixture passed=True; ternary fixture decode tok/s=419.29; ternary fixture RSS MiB=142.48046875; synthetic expert scaling passed=True; scaling rows=4; failed MoE gates=3/9; TL2 MoE runtime ready=False; TL2 expert byte underreport=0 | Treat as separate research milestone. | No Kimi-specific mapping, trained Qwen2MoE/Kimi quality artifact, TL2 3D expert runtime support, router distillation, quality benchmark, or trained expert-locality benchmark exists; the tiny Qwen2MoE fixtures only prove synthetic FP16 and row-scale I2_SR converter/runtime plumbing. |

## Recommendation

Product: CPU-first dense-Qwen retrofit evaluator with stable I2_SR runtime support; keep BitDistill quality claims behind the full GLUE reproduction, paired-trace, and CPU full-quality gates.

Paper: Scope as a negative PTQ result plus measured QAT/row-scale/runtime recovery path; add BitDistill reproduction claims only if the full-validation long-warmup gates pass. Do not claim arbitrary or MoE support.

Next engineering gate: Finish paired-trace coverage, the CPU-quality dependency chain, and clean row-warmup comparisons; keep MoE/Kimi as a separate milestone.
