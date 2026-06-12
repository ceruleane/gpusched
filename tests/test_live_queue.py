import os
import time
from io import StringIO
from pathlib import Path

import pytest

from gpusched.allocation import AllocOptions
from gpusched.jobspec import JobSpec, JobSpecError, assign_keys, parse_duration
from gpusched.scheduler import Scheduler, SchedulerOptions
from gpusched.testing import SIM_DIR_ENV, FakeBackend, SimBackend


@pytest.mark.parametrize("val,sec", [("90", 90), ("90s", 90), ("15m", 900), ("2h", 7200), ("1.5h", 5400), ("1d", 86400)])
def test_parse_duration(val, sec):
    assert parse_duration(val) == sec


def test_parse_duration_rejects_garbage():
    with pytest.raises(JobSpecError):
        parse_duration("soon", 3)


def test_timeout_and_retries_attrs():
    from gpusched.jobspec import parse_line
    s = parse_line("[vram=8G timeout=2h retries=2] python x.py", 1, 1)
    assert s.timeout_s == 7200 and s.retries == 2 and s.vram_mib == 8192


def opts(tmp_path, **kw):
    base = dict(
        alloc=AllocOptions(idle_threshold_mib=200, margin_mib=512),
        poll_interval=0.05, log_dir=str(tmp_path / "logs"),
    )
    base.update(kw)
    return SchedulerOptions(**base)


def drive(sched, mutate=None, max_s=25, settle_ticks=0):
    """Tick until drained (plus optional settle ticks for --watch-like checks)."""
    deadline = time.monotonic() + max_s
    n = 0
    while time.monotonic() < deadline:
        if mutate:
            mutate(n, sched)
        sched.tick()
        n += 1
        if sched.drained:
            if settle_ticks <= 0:
                return n
            settle_ticks -= 1
        time.sleep(0.03)
    raise AssertionError("did not drain in time")


def sim_backend(tmp_path, n_gpus=1):
    sim_dir = str(tmp_path / "sim")
    os.environ[SIM_DIR_ENV] = sim_dir
    return SimBackend(n_gpus=n_gpus, total_mib=24000, sim_dir=sim_dir)


def simcmd(vram, ramp=0.15, hold=0.3, extra=""):
    return f"python3 -m gpusched.simjob --vram {vram} --ramp {ramp} --hold {hold} {extra}".strip()


# ---------------------------------------------------------------- live queue
def test_live_append_is_picked_up(tmp_path):
    jf = tmp_path / "jobs.txt"
    jf.write_text(f"[vram=4G] {simcmd(4000)}\n")
    sched = Scheduler(sim_backend(tmp_path), jobs_path=str(jf), options=opts(tmp_path), out=StringIO())

    def mutate(n, s):
        if n == 3:  # mid-run append, as a user would from another terminal
            with open(jf, "a") as f:
                f.write(f"[vram=2G] {simcmd(2000)}\n")

    drive(sched, mutate)
    assert len(sched.results) == 2
    assert all(r.returncode == 0 for r in sched.results)


def test_live_delete_dequeues_pending(tmp_path):
    jf = tmp_path / "jobs.txt"
    # job 2 declared bigger than job 1 + margin allows alongside -> pends.
    jf.write_text(
        f"[vram=14G] {simcmd(14000, hold=0.8)}\n"
        f"[vram=14G] {simcmd(14001, hold=0.8)}\n"   # distinct command -> distinct identity
    )
    sched = Scheduler(sim_backend(tmp_path), jobs_path=str(jf), options=opts(tmp_path), out=StringIO())
    deleted = []

    def mutate(n, s):
        if n == 3 and not deleted:  # delete the still-pending second line
            jf.write_text(f"[vram=14G] {simcmd(14000, hold=0.8)}\n")
            deleted.append(True)

    drive(sched, mutate)
    assert len(sched.results) == 1            # second job never ran
    assert sched.results[0].returncode == 0


def test_live_reorder_controls_dispatch_order(tmp_path):
    jf = tmp_path / "jobs.txt"
    a, b = simcmd(20000, hold=0.4), simcmd(3000, hold=0.4)
    # Both blocked initially (external load), then user swaps order before
    # the GPU frees: line order among pending IS the priority.
    jf.write_text(f"[vram=20G] {a}\n[vram=3G] {b}\n")
    sim_dir = str(tmp_path / "sim")
    os.environ[SIM_DIR_ENV] = sim_dir
    backend = SimBackend(n_gpus=1, total_mib=24000, external_mib={0: 23000}, sim_dir=sim_dir)
    sched = Scheduler(backend, jobs_path=str(jf), options=opts(tmp_path), out=StringIO())
    order = []

    def mutate(n, s):
        if n == 2:
            jf.write_text(f"[vram=3G] {b}\n[vram=20G] {a}\n")  # user swaps
        if n == 4:
            backend.external = {}                               # GPU frees
        for rj in s.running.values():
            if rj.spec.command not in order:
                order.append(rj.spec.command)

    drive(sched, mutate)
    assert order and order[0] == b, "reordered file did not control dispatch order"


# ---------------------------------------------------------------- resume
def test_resume_skips_completed_jobs(tmp_path):
    jf = tmp_path / "jobs.txt"
    jf.write_text(f"[vram=4G] {simcmd(4000)}\n")
    o = opts(tmp_path)
    sched1 = Scheduler(sim_backend(tmp_path), jobs_path=str(jf), options=o, out=StringIO())
    drive(sched1)
    assert len(sched1.results) == 1

    out2 = StringIO()
    sched2 = Scheduler(sim_backend(tmp_path), jobs_path=str(jf), options=o, out=out2)
    drive(sched2)  # same journal dir -> nothing to do
    assert sched2.results == []
    assert "already completed ok — skipping" in out2.getvalue()


# ---------------------------------------------------------------- OOM retry
def test_oom_retry_with_declaration_bump(tmp_path):
    marker = tmp_path / "oom.marker"
    jf = tmp_path / "jobs.txt"
    jf.write_text(f"[vram=2000 retries=1] {simcmd(2000, extra=f'--oom-once {marker}')}\n")
    out = StringIO()
    sched = Scheduler(sim_backend(tmp_path), jobs_path=str(jf), options=opts(tmp_path), out=out)
    drive(sched)
    log = out.getvalue()
    assert "hit CUDA OOM (attempt 1/2)" in log
    assert "vram 2000 → 2500" in log               # ceil(2000 * 1.25)
    [r] = sched.results                            # only the terminal attempt
    assert r.returncode == 0
    assert sched.journal.state(r.spec.key).attempts == 2


def test_oom_without_retries_is_terminal(tmp_path):
    marker = tmp_path / "oom.marker"
    jf = tmp_path / "jobs.txt"
    jf.write_text(f"[vram=2000] {simcmd(2000, extra=f'--oom-once {marker}')}\n")
    out = StringIO()
    sched = Scheduler(sim_backend(tmp_path), jobs_path=str(jf), options=opts(tmp_path), out=out)
    drive(sched)
    [r] = sched.results
    assert r.verdict == "failed_oom" and r.returncode == 1
    assert "retries exhausted" in out.getvalue()


def test_plain_failure_is_not_treated_as_oom(tmp_path):
    jf = tmp_path / "jobs.txt"
    jf.write_text(f"[vram=2000 retries=3] {simcmd(2000, extra='--exit-code 7')}\n")
    sched = Scheduler(sim_backend(tmp_path), jobs_path=str(jf), options=opts(tmp_path), out=StringIO())
    drive(sched)
    [r] = sched.results
    assert r.verdict == "failed" and r.returncode == 7
    assert sched.journal.state(r.spec.key).attempts == 1   # no retries burned


# ---------------------------------------------------------------- timeout
def test_timeout_kills_job_and_marks_terminal(tmp_path):
    jf = tmp_path / "jobs.txt"
    jf.write_text("[vram=2000 timeout=0.4s] sleep 60\n")
    out = StringIO()
    t0 = time.monotonic()
    sched = Scheduler(sim_backend(tmp_path), jobs_path=str(jf), options=opts(tmp_path), out=out)
    drive(sched)
    assert time.monotonic() - t0 < 10, "timeout did not actually kill the job"
    [r] = sched.results
    assert r.verdict == "timeout" and r.returncode != 0
    assert "TIMEOUT" in out.getvalue()


def test_no_timeout_attribute_means_no_timeout():
    spec = assign_keys([JobSpec(index=1, command="sleep 999")])[0]
    assert spec.timeout_s is None  # discernment is by declaration, never heuristic


# ---------------------------------------------------------------- util report
def test_util_reported_in_completion_line(tmp_path):
    jf = tmp_path / "jobs.txt"
    jf.write_text(f"[vram=4G] {simcmd(4000)}\n")
    out = StringIO()
    sched = Scheduler(sim_backend(tmp_path), jobs_path=str(jf), options=opts(tmp_path), out=out)
    drive(sched)
    [r] = sched.results
    assert r.avg_util is not None and r.avg_util > 0
    assert "avg gpu util" in out.getvalue()


# ---------------------------------------------------------------- status board
def test_status_board_rendered_and_user_file_untouched(tmp_path):
    jf = tmp_path / "jobs.txt"
    content = f"[vram=4G] {simcmd(4000)}\n"
    jf.write_text(content)
    sched = Scheduler(sim_backend(tmp_path), jobs_path=str(jf), options=opts(tmp_path), out=StringIO())
    drive(sched)
    board = (Path(sched.opts.log_dir) / "status.txt").read_text()
    assert "✓ job 1" in board
    assert jf.read_text() == content, "scheduler must NEVER write the user's jobs file"


def test_malformed_edit_keeps_last_good_queue(tmp_path):
    jf = tmp_path / "jobs.txt"
    jf.write_text(f"[vram=4G] {simcmd(4000, hold=0.6)}\n")
    out = StringIO()
    sched = Scheduler(sim_backend(tmp_path), jobs_path=str(jf), options=opts(tmp_path), out=out)

    def mutate(n, s):
        if n == 2:
            jf.write_text("[vram=oops this is broken\n")  # user mid-edit typo

    drive(sched, mutate)
    assert "jobs file unreadable, keeping previous queue" in out.getvalue()
    [r] = sched.results
    assert r.returncode == 0  # the in-flight job was unaffected
