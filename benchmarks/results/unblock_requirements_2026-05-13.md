# Unblock Requirements Audit, 2026-05-13

This audit consolidates the external inputs required before the remaining I2_SR promotion and MoE/Kimi benchmark claims can be completed honestly.

Objective status: `not_complete` (`7/9` complete).

Product scope: `research_mvp_only`.

Can continue productively without new input: `false`.

Next required input: Provide a licensed Kimi or Qwen2MoE checkpoint/tokenizer artifact plus its FP and quantized baselines.

## Requirements

| requirement | status | evidence | unblock action |
| --- | --- | --- | --- |
| Writable llama.cpp fork or branch | ready | promotion_ready=True; candidate_fork_reachable=True; submodule_patch_applies=False; local_handoff_prepared=True; local_handoff_commit=cb38942486a05e3a1212e6eba1462e178478f459 | No action needed; the fork branch is reachable and promotion_ready is true. |
| GitHub automation credential | optional_missing | gh_path=not_found | Install/authenticate GitHub CLI or refresh the GitHub connector token if repository creation/push automation is desired. |
| Local Kimi/Qwen2MoE model artifact | missing | artifacts=0 | Provide a licensed Kimi or Qwen2MoE checkpoint/tokenizer artifact plus its FP and quantized baselines. |
| MoE 3D expert tensor packing support | missing | tl2_3d=False; i2sr_3d=True; 2d_control=True | Implement remaining TL2 3D expert packing and full MoE GGUF/runtime byte tests before any Kimi runtime benchmark. |
| MoE quality/locality benchmark artifacts | missing | failed_moe_gates=3; kimi_artifacts=0 | Run router accuracy, expert locality, quality, throughput, and RSS benchmarks after model and packing support exist. |

## Candidate Fork Probe

| field | value |
| --- | --- |
| url | `https://github.com/sabdulmajid/llama.cpp.git` |
| returncode | `0` |
| reachable | `true` |
| stderr | `` |
