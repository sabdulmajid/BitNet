# Fixed-60 Control Recovery

Generated: `2026-09-04T14:10:00Z`. Status: **running**.

This is infrastructure recovery for the preregistered matched fixed-60 control.
It is not a method result. No adaptive-versus-fixed conclusion is valid until
all three controls and the `afterany` fail-closed audit complete.

## Root Cause

1. Original job `10399` embedded `EXCLUDE_LINEAR_REGEX=score|classifier` in
   executable shell text; the shell attempted to run `classifier`.
2. The first corrected wrappers inherited an NFS `SLURM_SUBMIT_DIR` that
   `ece-nebula12` cannot mount.
3. The first local-directory wrapper did not pin the node-local Hugging Face
   cache and failed before training while resolving GLUE.

Jobs `10399`, `10423`, `10428`, and `10433` completed zero optimization steps.

## Active Chain

| arm | seed | job | dependency |
| --- | ---: | ---: | --- |
| fixed60 | 1234 | `10440` | - |
| fixed60 | 1235 | `10441` | `afterok:10440` |
| fixed60 | 1236 | `10442` | `afterok:10441` |
| fail-closed audit | - | `10443` | `afterany:10442` |

Job `10440` passed model loading, the exact declared training contract, and
cached GLUE resolution (`392,702` train; `9,815` matched validation rows). It
entered optimization; step `490` was observed before this status snapshot.

## Repairs

- The regex now travels through `sbatch --export`, not executable shell text.
- The wrapper overrides `SLURM_SUBMIT_DIR` before starting training.
- `HF_HOME` and `HF_DATASETS_CACHE` point to the verified node-local cache;
  dataset and model loading run offline.
- The audit accepts explicit resubmitted job IDs and is staged outside the
  immutable training checkout.

The JSON companion records launcher, audit, and batch-script SHA-256 values.
