# BitDistill method-parity pilot submission

- Status: reference-environment array queued
- Source revision: `526ede7b2c3f33c6a9638de54bdae91e8afe39c6`
- Reference array: `10389`; fail-closed audit: `10390`
- Target array: `0-5%1` on `dualcard`
- Postprocessor: fail-closed and dependent on the complete array
- Per-pilot resources: 1 GPU, 12 CPUs, 24 GiB RAM, 6-hour limit
- Shared starting point: verified 655M-token Stage-2 manifest
- Scope: six 120-step MNLI diagnostics over at most 8,192 train and 512 validation examples

The matrix isolates three previously conflated choices: published Algorithm 1 cosine relations versus Equation 12 scaled-dot relations, one versus eight relation heads, and fixed versus gradient-balanced attention loss. It also compares the local sequence-classification and causal answer-token formulations without asserting that either is paper-exact.

These runs may select a method for full validation. They cannot establish a task-quality result because the validation subset and training budget are intentionally bounded.

The first submission attempt requested 32 GiB and was rejected before creating a job because the target node has 30,000 MiB physical memory. Commit `bb8e92b` changed only that scheduler request to 24 GiB, matching the existing proven wrapper.

The resulting midcard array `10298` and audit `10299` then failed before script execution. Compute-node probes `10309` through `10312` proved that the shared project and log paths were unavailable on `ece-nebula12`; a `/dev/null` control completed as job `10305`. The idle bigcard node had the same missing mount, while dualcard probe `10315` passed.

Dualcard array `10316` and audit `10317` were queued but cancelled before allocation when a final preflight found that the login environment had no `python` alias and queued jobs were not pinned against source drift. No training step ran. The launcher now resolves an interpreter that can import the required ML packages and rejects a checkout whose Git revision differs from `EXPECTED_SOURCE_REVISION`.

Revision-pinned array `10318` and audit `10319` were also cancelled before allocation after a postprocessor review found an off-by-one contract: telemetry emits at steps `1,20,40,60,80,100,120`, while the auditor expected six rows. The auditor now verifies all seven exact step numbers, with regression tests.

Attempts to use the non-NFS GPU nodes were treated as infrastructure probes, not results. Initial bigcard jobs `10335` and `10336` stopped before training because Datasets 2.18 attempted to resolve the legacy unnamespaced GLUE URI through an incompatible Hub client. The exact cached MNLI Arrow data was then transferred and SHA-256 verified. Jobs `10341`-`10346` stopped in the Slurm wrapper before Python because `--wrap` used `/bin/sh` with a Bash-only `pipefail`; corrected jobs `10348`-`10353` stopped before step 1 because an unmanaged Ollama service occupied approximately 47.6 GiB on each 49.1 GiB A6000. The service was not modified.

Midcard job `10368` reached the first model forward and exposed a PyTorch 2.6 portability defect: FP32 SubLN output could feed a BF16 projection. Commit `18ec2c9` makes `SubLNLinear` preserve the incoming activation dtype and adds a forced-promotion regression test. The corrected exploratory jobs `10373` and `10376`-`10380` then completed all six contracts on the RTX A4500. Their cross-environment diagnostic is published separately in [bitdistill_method_parity_midcard_exploratory_2026-09-04.md](bitdistill_method_parity_midcard_exploratory_2026-09-04.md); it carries no task-quality claim.

Pending dualcard array `10331` and audit `10332` were cancelled before allocation when the validated dtype fix and stricter prediction auditor superseded their source revision. Replacement jobs `10387` and `10388` were also cancelled before allocation after a provenance review found that the wrapper did not explicitly forward or serialize the random seed. The completed exploratory runs used the parser default `1234`; the inaccurate `42` manifest entry was corrected. The live reference jobs `10389` and `10390` explicitly set and validate seed `1234` and run from a detached worktree pinned to `526ede7b2c3f33c6a9638de54bdae91e8afe39c6`, with symlinks only to the shared checkpoint and Hugging Face caches. Advancing `main` therefore cannot alter their source tree.
