# Changelog

## 0.3.0
- Live-editable queue: the jobs file is user-owned, re-read every poll;
  append / delete / reorder pending lines mid-run. Pending = in-file and
  neither running nor terminal in the journal.
- Journal (`journal.jsonl`): per-job identities, attempts, outcomes.
  Resume-after-restart falls out of it; `--fresh` resets.
- CUDA-OOM-aware retry (`[retries=N]` / `--oom-retries`) with the
  declaration auto-bumped to ~1.25x of observed peak; non-OOM failures
  never consume retries.
- Opt-in per-job walltime (`[timeout=...]`): SIGTERM, then SIGKILL after a
  grace period. No heuristic hang detection by design.
- Per-job average device utilization in completion reports.
- Live status board rendered to `<log_dir>/status.txt` every poll.
- `--watch`: keep running after drain, picking up appended lines.
- Parser hardening: an unterminated leading `[...]` block is a parse error
  (previously it fell through and was executed as a shell command); a
  malformed mid-edit jobs file keeps the last good queue.

## 0.2.0
- High-water-mark scheduling: external processes are held to their observed
  per-GPU peaks (buffered by `--spike-buffer`) until they exit, so momentary
  VRAM troughs are not treated as packable space; a scheduled job that
  exceeds its declaration has its budget escalated to its buffered peak.
- Idle detection for undeclared jobs judged on effective (peak-aware)
  external usage.

## 0.1.0
- Initial release: VRAM-declaration-based placement with launch-race
  reservations, packing, `--exclusive`, multi-GPU jobs (`gpus=N`),
  per-job VRAM attribution by process group, immediate under-declaration
  warnings, completion-time over-declaration reports, backfill, fail-fast
  infeasibility, simulated backends and `--sim` dry-run mode.
