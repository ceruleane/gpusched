"""Scheduler core: dispatch loop, per-job VRAM attribution, notifications,
live-editable queue, journal-backed resume, OOM retry, opt-in timeouts.

Queue model (v0.3)
------------------
The jobs file is **user-owned**: the scheduler re-reads it every tick and
never writes to it. Each line has a stable identity (command hash +
occurrence); the journal records attempts and terminal outcomes per identity.
"Pending" = lines present in the file whose identity is neither running nor
terminal in the journal — so appending, deleting, or reordering pending
lines mid-run is fully supported, and re-running the same jobs file skips
completed work (resume). A live status board is rendered to
``<log_dir>/status.txt`` each tick.

Notification contract
---------------------
* OVER (actual > declared * (1 + tolerance)): warned immediately.
* UNDER (peak < declared * (1 - tolerance)): reported at completion.
* CUDA OOM with ``[retries=N]`` (or --oom-retries): requeued with the
  declaration bumped to ~1.25x of max(observed peak, old declaration).
* ``[timeout=...]`` (strictly opt-in): SIGTERM at walltime, SIGKILL +10s.
* Attribution failure reports peaks as "n/a", never a wrong number.
"""

from __future__ import annotations

import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

from .allocation import AllocOptions, Occupant, find_allocation
from .backend import BackendError, GpuBackend, GpuSnapshot, pgid_of
from .jobspec import JobSpec, JobSpecError, assign_keys, parse_jobs_file
from .journal import Journal

OOM_PATTERNS = (
    "cuda out of memory",
    "torch.outofmemoryerror",
    "cuda error: out of memory",
    "cuda_error_out_of_memory",
    "cudaerrormemoryallocation",
    "resource_exhausted",
)


def looks_like_oom(log_path: str, tail_bytes: int = 16384) -> bool:
    try:
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - tail_bytes))
            tail = f.read().decode("utf-8", "replace").lower()
        return any(p in tail for p in OOM_PATTERNS)
    except OSError:
        return False


@dataclass
class SchedulerOptions:
    alloc: AllocOptions = field(default_factory=AllocOptions)
    poll_interval: float = 5.0
    tolerance: float = 0.10          # relative band for over/under verdicts
    verbose: bool = False
    log_dir: str = "gpusched_logs"
    verbose_step_mib: int = 256      # min peak growth before a verbose line
    watch: bool = False              # idle-wait for new jobs when drained
    oom_retries_default: int = 0     # global default; [retries=N] overrides
    oom_bump_factor: float = 1.25    # declaration multiplier after an OOM
    kill_grace_s: float = 10.0       # SIGTERM -> SIGKILL grace on timeout


@dataclass
class RunningJob:
    spec: JobSpec                    # with any journal vram bump applied
    no: int                          # persistent display number
    proc: subprocess.Popen
    pgid: int
    gpu_indices: tuple[int, ...]
    log_path: str
    started_at: float
    last_mib: dict[int, int] = field(default_factory=dict)
    peak_mib: dict[int, int] = field(default_factory=dict)
    attributed_ever: bool = False
    over_warned: bool = False
    last_verbose_peak: int = -1
    util_sum: float = 0.0
    util_n: int = 0
    timed_out: bool = False
    killed_at: float | None = None

    @property
    def label(self) -> str:
        return f"job {self.no}"

    @property
    def overall_peak(self) -> int:
        return max(self.peak_mib.values(), default=0)

    @property
    def avg_util(self) -> float | None:
        return self.util_sum / self.util_n if self.util_n else None

    def as_occupant(self) -> Occupant:
        return Occupant(
            gpu_indices=self.gpu_indices,
            vram_mib=self.spec.vram_mib,
            actual_mib=dict(self.last_mib),
            peak_mib=dict(self.peak_mib),
        )


@dataclass
class JobResult:
    spec: JobSpec
    no: int
    gpu_indices: tuple[int, ...]
    returncode: int
    peak_mib: dict[int, int]
    attributed: bool
    duration_s: float
    verdict: str        # ok | over | under | unattributed | infeasible | timeout | failed_oom
    avg_util: float | None = None


class Scheduler:
    def __init__(
        self,
        backend: GpuBackend,
        jobs: list[JobSpec] | None = None,
        allowed_gpus: set[int] | None = None,
        options: SchedulerOptions | None = None,
        out=None,
        jobs_path: str | None = None,
    ):
        if (jobs is None) == (jobs_path is None):
            raise ValueError("provide exactly one of `jobs` (static) or `jobs_path` (live)")
        self.backend = backend
        self.jobs_path = jobs_path
        self._static = assign_keys(list(jobs)) if jobs is not None else None
        self._cached_specs: list[JobSpec] = self._static or []
        self.allowed_gpus = allowed_gpus  # None = discover from first snapshot
        self.opts = options or SchedulerOptions()
        self.journal = Journal(Path(self.opts.log_dir) / "journal.jsonl")
        self.running: dict[str, RunningJob] = {}   # key -> RunningJob
        self.results: list[JobResult] = []
        # (gpu, pid) -> observed peak MiB for processes NOT launched by us.
        self.external_peaks: dict[tuple[int, int], int] = {}
        self._ext_view: dict[int, int] = {}
        self._pending_now: list[JobSpec] = []
        self._out = out or sys.stdout
        self._backend_failures = 0
        self._parse_error_warned: str | None = None
        # Only announce journal-skips for work completed in a PREVIOUS run;
        # jobs finishing within this run are reported by _finalize already.
        self._preexisting_ok: set[str] = {
            k for k, st in self.journal.states.items() if st.status == "ok"
        }
        self._skip_logged: set[str] = set()

    @property
    def drained(self) -> bool:
        """True iff nothing is running and nothing was pending as of the last tick."""
        return not self._pending_now and not self.running

    # ------------------------------------------------------------- logging
    def _log(self, msg: str) -> None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", file=self._out, flush=True)

    def _warn(self, msg: str) -> None:
        self._log(f"WARN  {msg}")

    # ------------------------------------------------------------- job source
    def _source(self) -> list[JobSpec]:
        """Current job list: re-read the user-owned file (live mode) or the
        static list. A malformed mid-edit file keeps the last good parse."""
        if self.jobs_path is None:
            return self._cached_specs
        try:
            specs = parse_jobs_file(self.jobs_path)
            self._cached_specs = specs
            self._parse_error_warned = None
        except (JobSpecError, OSError) as e:
            if self._parse_error_warned != str(e):
                self._parse_error_warned = str(e)
                self._warn(f"jobs file unreadable, keeping previous queue: {e}")
        return self._cached_specs

    def _pending(self, specs: list[JobSpec]) -> list[JobSpec]:
        pending: list[JobSpec] = []
        for spec in specs:
            no = self.journal.ensure_seen(spec.key, spec.command)
            st = self.journal.state(spec.key)
            if spec.key in self.running:
                continue
            if st.terminal:
                if spec.key in self._preexisting_ok and spec.key not in self._skip_logged:
                    self._skip_logged.add(spec.key)
                    self._log(f"job {no} already completed ok — skipping (journal): {spec.command}")
                continue
            if st.next_vram is not None and st.next_vram != spec.vram_mib:
                spec = replace(spec, vram_mib=st.next_vram)
            pending.append(spec)
        return pending

    # ------------------------------------------------------------- tick
    def tick(self) -> None:
        """One round: snapshot -> attribute -> warn -> timeouts -> reap -> dispatch."""
        try:
            snapshot = self.backend.snapshot()
            self._backend_failures = 0
        except BackendError as e:
            self._backend_failures += 1
            self._warn(f"GPU query failed ({e}); attempt {self._backend_failures}")
            if self._backend_failures >= 5:
                raise
            return

        if self.allowed_gpus is None:
            self.allowed_gpus = set(snapshot.gpus)
            self._log(f"discovered GPUs: {sorted(self.allowed_gpus)}")

        self._attribute(snapshot)
        self._check_over_declaration()
        self._enforce_timeouts()
        self._reap()
        specs = self._source()
        self._pending_now = self._pending(specs)
        self._dispatch(snapshot, specs)
        self._render_status()

    # ------------------------------------------------------------- attribution
    def _attribute(self, snapshot: GpuSnapshot) -> None:
        pgid_index = {rj.pgid: rj for rj in self.running.values()}
        for rj in self.running.values():
            rj.last_mib = {g: 0 for g in rj.gpu_indices}

        pid_cache: dict[int, int | None] = {}
        seen_external: set[tuple[int, int]] = set()
        for proc in snapshot.procs:
            if proc.pid not in pid_cache:
                pid_cache[proc.pid] = pgid_of(proc.pid)
            rj = pgid_index.get(pid_cache[proc.pid])
            if rj is not None and proc.gpu_index in rj.last_mib:
                rj.last_mib[proc.gpu_index] += proc.used_mib
                rj.attributed_ever = True
            else:
                key = (proc.gpu_index, proc.pid)
                seen_external.add(key)
                if proc.used_mib > self.external_peaks.get(key, 0):
                    self.external_peaks[key] = proc.used_mib
        for key in list(self.external_peaks):
            if key not in seen_external:
                del self.external_peaks[key]

        buf = 1 + self.opts.alloc.spike_buffer
        self._ext_view = {}
        for g, stat in snapshot.gpus.items():
            own_actual = sum(rj.last_mib.get(g, 0) for rj in self.running.values())
            instantaneous = max(0, stat.used_mib - own_actual)
            peak_sum = sum(m for (gpu, _), m in self.external_peaks.items() if gpu == g)
            self._ext_view[g] = max(instantaneous, math.ceil(round(peak_sum * buf, 6)))

        for rj in self.running.values():
            for g, mib in rj.last_mib.items():
                if mib > rj.peak_mib.get(g, 0):
                    rj.peak_mib[g] = mib
            utils = [
                snapshot.gpus[g].util_pct for g in rj.gpu_indices
                if g in snapshot.gpus and snapshot.gpus[g].util_pct is not None
            ]
            if utils:
                rj.util_sum += sum(utils) / len(utils)
                rj.util_n += 1
            if (
                self.opts.verbose
                and rj.attributed_ever
                and rj.overall_peak >= rj.last_verbose_peak + self.opts.verbose_step_mib
            ):
                rj.last_verbose_peak = rj.overall_peak
                usage = ", ".join(f"gpu{g}:{m} MiB" for g, m in sorted(rj.last_mib.items()))
                self._log(f"{rj.label} vram now {usage} (peak {rj.overall_peak} MiB)")

    def _check_over_declaration(self) -> None:
        for rj in self.running.values():
            est = rj.spec.vram_mib
            if est is None or rj.over_warned:
                continue
            if rj.overall_peak > est * (1 + self.opts.tolerance):
                rj.over_warned = True
                gpu = max(rj.peak_mib, key=rj.peak_mib.get)
                self._warn(
                    f"{rj.label} EXCEEDS declared VRAM: "
                    f"{rj.peak_mib[gpu]} MiB observed on gpu {gpu} vs {est} MiB declared "
                    f"(+{100 * (rj.peak_mib[gpu] / est - 1):.0f}%) — neighbors may OOM"
                )

    # ------------------------------------------------------------- timeouts
    def _enforce_timeouts(self) -> None:
        now = time.monotonic()
        for rj in self.running.values():
            limit = rj.spec.timeout_s
            if limit is None or rj.proc.poll() is not None:
                continue
            elapsed = now - rj.started_at
            if not rj.timed_out and elapsed > limit:
                rj.timed_out = True
                rj.killed_at = now
                self._warn(f"{rj.label} TIMEOUT after {elapsed:.0f}s (limit {limit:.0f}s) — SIGTERM")
                self._killpg(rj, signal.SIGTERM)
            elif rj.timed_out and rj.killed_at and now - rj.killed_at > self.opts.kill_grace_s:
                rj.killed_at = now + 1e9  # only escalate once
                self._warn(f"{rj.label} did not exit after SIGTERM — SIGKILL")
                self._killpg(rj, signal.SIGKILL)

    @staticmethod
    def _killpg(rj: RunningJob, sig: int) -> None:
        try:
            os.killpg(rj.pgid, sig)
        except ProcessLookupError:
            pass

    # ------------------------------------------------------------- reap
    def _reap(self) -> None:
        for key in list(self.running):
            rj = self.running[key]
            if rj.proc.poll() is None:
                continue
            del self.running[key]
            self._finalize(rj)

    def _finalize(self, rj: RunningJob) -> None:
        rc = rj.proc.returncode
        spec, est = rj.spec, rj.spec.vram_mib
        peak = rj.overall_peak
        attributed = rj.attributed_ever
        duration = time.monotonic() - rj.started_at
        st = self.journal.state(spec.key)

        # --- OOM retry path (non-terminal) -------------------------------
        oom = rc != 0 and not rj.timed_out and looks_like_oom(rj.log_path)
        retries_allowed = max(spec.retries, self.opts.oom_retries_default)
        attempts = st.attempts + 1
        if oom and attempts <= retries_allowed:
            next_vram = self._bumped_vram(est, peak)
            self.journal.record_oom_retry(spec.key, attempts, next_vram)
            bump_txt = f" with vram {est or 'n/a'} → {next_vram} MiB" if next_vram else ""
            self._log(
                f"↻ {rj.label} hit CUDA OOM (attempt {attempts}/{retries_allowed + 1}) "
                f"— requeued{bump_txt}"
            )
            return

        # --- terminal -----------------------------------------------------
        if rj.timed_out:
            status, verdict = "timeout", "timeout"
        elif oom:
            status, verdict = "failed_oom", "failed_oom"
        elif rc != 0:
            status, verdict = "failed", "failed"
        elif not attributed:
            status, verdict = "ok", "unattributed"
        elif est is None:
            status, verdict = "ok", "ok"
        elif peak > est * (1 + self.opts.tolerance):
            status, verdict = "ok", "over"
        elif peak < est * (1 - self.opts.tolerance):
            status, verdict = "ok", "under"
        else:
            status, verdict = "ok", "ok"

        if rj.timed_out:
            label_status = "TIMEOUT"
        elif oom:
            label_status = f"OOM (exit {rc}, retries exhausted)"
        else:
            label_status = "OK" if rc == 0 else f"EXIT {rc}"
        peaks = (
            ", ".join(f"gpu{g}:{m} MiB" for g, m in sorted(rj.peak_mib.items()))
            if attributed else "n/a (attribution failed — see README caveats)"
        )
        line = f"{rj.label} finished [{label_status}] in {duration:.0f}s — peak vram {peaks}"
        if est is not None and attributed and status == "ok":
            if verdict == "over":
                line += f" | declared {est} MiB → UNDER-DECLARED (+{100 * (peak / est - 1):.0f}%); raise it"
            elif verdict == "under":
                line += f" | declared {est} MiB → over-declared (-{100 * (1 - peak / est):.0f}%); lowering it frees packing headroom"
            else:
                line += f" | declared {est} MiB → within ±{self.opts.tolerance:.0%}"
        if rj.avg_util is not None:
            line += f" | avg gpu util {rj.avg_util:.0f}%"
        self._log(line)

        self.journal.record_done(spec.key, status, rc, rj.peak_mib, rj.avg_util)
        self.results.append(JobResult(
            spec=spec, no=rj.no, gpu_indices=rj.gpu_indices, returncode=rc,
            peak_mib=dict(rj.peak_mib), attributed=attributed,
            duration_s=duration, verdict=verdict, avg_util=rj.avg_util,
        ))

    def _bumped_vram(self, declared: int | None, peak: int) -> int | None:
        base = max(peak, declared or 0)
        if base <= 0:
            return None  # nothing to bump from (undeclared + unattributed)
        bumped = math.ceil(round(base * self.opts.oom_bump_factor, 6))
        if declared is not None and bumped <= declared:
            bumped = math.ceil(round(declared * self.opts.oom_bump_factor, 6))
        return bumped

    # ------------------------------------------------------------- dispatch
    def _dispatch(self, snapshot: GpuSnapshot, specs: list[JobSpec]) -> None:
        assert self.allowed_gpus is not None
        occupants = [rj.as_occupant() for rj in self.running.values()]
        still_pending: list[JobSpec] = []
        for spec in self._pending_now:
            reason = self._infeasible_reason(spec, snapshot)
            if reason is not None:
                no = self.journal.ensure_seen(spec.key, spec.command)
                self._warn(f"job {no} INFEASIBLE — {reason}; marking failed: {spec.command}")
                self.journal.record_done(spec.key, "infeasible", 1, {}, None)
                self.results.append(JobResult(
                    spec=spec, no=no, gpu_indices=(), returncode=1, peak_mib={},
                    attributed=False, duration_s=0.0, verdict="infeasible",
                ))
                continue
            alloc = find_allocation(
                spec, snapshot, occupants, self.allowed_gpus, self.opts.alloc,
                external_mib=self._ext_view,
            )
            if alloc is None:
                still_pending.append(spec)  # blocked now; smaller jobs backfill
                continue
            rj = self._launch(spec, alloc, n_known=len(specs))
            occupants.append(rj.as_occupant())
        self._pending_now = still_pending

    def _infeasible_reason(self, spec: JobSpec, snapshot: GpuSnapshot) -> str | None:
        """Infeasible = could not start even on fully EMPTY allowed GPUs —
        distinct from 'blocked right now', which is worth waiting out."""
        allowed = [snapshot.gpus[g] for g in snapshot.gpus if g in self.allowed_gpus]
        if spec.n_gpus > len(allowed):
            return f"needs {spec.n_gpus} GPUs but only {len(allowed)} are usable"
        if spec.vram_mib is not None:
            need = spec.vram_mib + self.opts.alloc.margin_mib
            fitting = sum(1 for g in allowed if g.total_mib >= need)
            if fitting < spec.n_gpus:
                return (
                    f"declared {spec.vram_mib} MiB + {self.opts.alloc.margin_mib} MiB margin "
                    f"exceeds total capacity of {spec.n_gpus - fitting} required GPU(s)"
                )
        return None

    def _launch(self, spec: JobSpec, gpus: list[int], n_known: int) -> RunningJob:
        os.makedirs(self.opts.log_dir, exist_ok=True)
        no = self.journal.ensure_seen(spec.key, spec.command)
        gpu_str = ",".join(str(g) for g in gpus)
        log_path = os.path.join(
            self.opts.log_dir, f"job{no:03d}_gpu{gpu_str.replace(',', '-')}.log"
        )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_str
        logf = open(log_path, "a")
        proc = subprocess.Popen(
            spec.command, shell=True, env=env,
            stdout=logf, stderr=subprocess.STDOUT,
            start_new_session=True,  # own pgid/session -> attribution + clean kill
        )
        rj = RunningJob(
            spec=spec, no=no, proc=proc, pgid=proc.pid, gpu_indices=tuple(gpus),
            log_path=log_path, started_at=time.monotonic(),
        )
        self.running[spec.key] = rj
        declared = f", declared {spec.vram_mib} MiB/gpu" if spec.vram_mib else ", undeclared (exclusive idle GPU)"
        done = len(self.results)
        self._log(
            f"{rj.label} started on gpu [{gpu_str}] "
            f"({done + len(self.running)}/{n_known} dispatched{declared}) — {spec.command}"
        )
        return rj

    # ------------------------------------------------------------- status board
    def _render_status(self) -> None:
        try:
            lines = [
                f"gpusched status @ {datetime.now().strftime('%H:%M:%S')} — "
                f"{len(self.running)} running, {len(self._pending_now)} pending, "
                f"{len(self.results)} done"
            ]
            for rj in sorted(self.running.values(), key=lambda r: r.no):
                cur = ", ".join(f"gpu{g}:{m}" for g, m in sorted(rj.last_mib.items()))
                lines.append(
                    f"▶ job {rj.no:<3} gpu[{','.join(map(str, rj.gpu_indices))}] "
                    f"{time.monotonic() - rj.started_at:5.0f}s  {cur} MiB (peak {rj.overall_peak}) — {rj.spec.command}"
                )
            for spec in self._pending_now:
                st = self.journal.state(spec.key)
                mark = "↻" if st.attempts > 0 else "·"
                extra = f" retry {st.attempts}, vram→{st.next_vram}" if st.attempts > 0 else ""
                vram = f" vram {spec.vram_mib}" if spec.vram_mib else ""
                lines.append(f"{mark} job {st.no or '?':<3} pending{vram}{extra} — {spec.command}")
            for r in sorted(self.results, key=lambda r: r.no):
                mark = "✓" if r.returncode == 0 else "✗"
                peak = max(r.peak_mib.values(), default=0)
                lines.append(
                    f"{mark} job {r.no:<3} {r.verdict} (exit {r.returncode}, peak {peak} MiB) — {r.spec.command}"
                )
            path = Path(self.opts.log_dir) / "status.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text("\n".join(lines) + "\n")
            tmp.replace(path)
        except OSError:
            pass  # status board is best-effort, never fatal

    # ------------------------------------------------------------- run
    def run(self) -> int:
        """Blocking loop. Exits when drained (or runs forever with watch=True)."""
        n0 = len(self._source())
        mode = "live queue (file re-read each tick)" if self.jobs_path else "static queue"
        self._log(
            f"{n0} jobs queued [{mode}] (poll every {self.opts.poll_interval}s; "
            f"status board: {Path(self.opts.log_dir) / 'status.txt'})"
        )
        announced_wait = False
        try:
            while True:
                self.tick()
                if not self._pending_now and not self.running:
                    if not self.opts.watch:
                        break
                    if not announced_wait:
                        announced_wait = True
                        self._log("queue drained — watching jobs file for new lines (Ctrl-C to stop)")
                else:
                    announced_wait = False
                time.sleep(self.opts.poll_interval)
        except KeyboardInterrupt:
            self._warn("interrupted — terminating running jobs")
            self._terminate_all()
            raise
        return self._summarize()

    def _terminate_all(self) -> None:
        for rj in self.running.values():
            self._killpg(rj, signal.SIGTERM)
        deadline = time.monotonic() + self.opts.kill_grace_s
        for rj in self.running.values():
            try:
                rj.proc.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                self._killpg(rj, signal.SIGKILL)

    def _summarize(self) -> int:
        failed = [r for r in self.results if r.returncode != 0]
        over = [r for r in self.results if r.verdict == "over"]
        under = [r for r in self.results if r.verdict == "under"]
        self._log(
            f"all {len(self.results)} jobs done — "
            f"{len(self.results) - len(failed)} ok, {len(failed)} failed, "
            f"{len(over)} under-declared vram, {len(under)} over-declared vram"
        )
        for r in failed:
            self._log(f"  failed: job {r.no} ({r.verdict}, exit {r.returncode}) — {r.spec.command}")
        return 0 if not failed else max(1, max(r.returncode for r in failed))
