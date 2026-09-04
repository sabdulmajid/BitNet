# Stage-2 655.36M Afterany Audit

Generated: 2026-05-24T11:57:25.527151+00:00

Status: **failed**.

Quality claim: **none**.

This historical watcher refreshed postmortem/salvage status only. Its watchdog
subcommand failed because the dependency audit still expected satisfied
dependencies to remain visible after jobs started. The later ingestion and
paired-quality reports completed successfully.

| field | value |
| --- | --- |
| stage2_job_id | 10250 |
| afterany_job_id | 10258 |
| dependency | afterany:10250 |
| snapshot_salvage_rc | 0 |
| ingestion_rc | 0 |
| watchdog_rc | 1 |
