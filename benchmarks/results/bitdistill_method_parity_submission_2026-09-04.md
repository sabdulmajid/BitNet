# BitDistill method-parity pilot submission

- Status: ready for revision-pinned resubmission
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
