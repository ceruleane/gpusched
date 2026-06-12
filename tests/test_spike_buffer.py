import os
import time
from io import StringIO

from gpusched.allocation import AllocOptions, Occupant, effective_budget, find_allocation
from gpusched.jobspec import JobSpec
from gpusched.scheduler import Scheduler, SchedulerOptions
from gpusched.testing import FakeBackend

DEAD_PID = 2**21 + 9999  # certainly not a live pid -> classified external

OPTS = AllocOptions(idle_threshold_mib=200, margin_mib=512, spike_buffer=0.10)


def spec(i=1, vram=None, gpus=1, cmd="sleep 0.2"):
    return JobSpec(index=i, command=cmd, vram_mib=vram, n_gpus=gpus)


def test_budget_is_declaration_while_peak_within_it():
    occ = Occupant((0,), vram_mib=8000, actual_mib={0: 3000}, peak_mib={0: 7500})
    assert effective_budget(occ, 0, 0.10) == 8000


def test_budget_escalates_to_buffered_peak_after_violation():
    occ = Occupant((0,), vram_mib=8000, actual_mib={0: 4000}, peak_mib={0: 12000})
    assert effective_budget(occ, 0, 0.10) == 13200  # ceil(12000 * 1.1)


def test_escalated_budget_shrinks_headroom_for_newcomers():
    # GPU 24000; running job declared 8000 but peaked at 12000, currently 4000.
    fb = FakeBackend({0: (24000, 4000)})
    occ = [Occupant((0,), vram_mib=8000, actual_mib={0: 4000}, peak_mib={0: 12000})]
    s = fb.snapshot()
    # headroom = 24000 - 0 external - 13200 = 10800
    assert find_allocation(spec(vram=10000), s, occ, {0}, OPTS) == [0]   # 10512 fits
    assert find_allocation(spec(vram=10300), s, occ, {0}, OPTS) is None  # 10812 doesn't


def running_nos(sched):
    return {rj.no for rj in sched.running.values()}


def run_tick_loop(sched, backend, mutate, max_s=20):
    """Drive ticks manually; `mutate(tick_no)` adjusts the fake world."""
    deadline = time.monotonic() + max_s
    n = 0
    while time.monotonic() < deadline:
        mutate(n)
        sched.tick()
        n += 1
        if sched.drained:
            break
        time.sleep(0.03)
    assert sched.drained, "did not drain"


def make_sched(backend, jobs, tmp_path):
    return Scheduler(
        backend, jobs,
        options=SchedulerOptions(
            alloc=OPTS, poll_interval=0.03, log_dir=str(tmp_path / "logs"),
        ),
        out=StringIO(),
    )


def test_external_trough_is_not_packable(tmp_path):
    """External proc peaks at 12000, dips to 1000; a 13000-MiB job must NOT
    start during the dip (held to 12000*1.1=13200 -> headroom 10800 < 13512),
    and MUST start once the external process exits."""
    fb = FakeBackend({0: (24000, 12000)})
    fb.procs = [(0, DEAD_PID, 12000)]
    sched = make_sched(fb, [spec(1, vram=13000)], tmp_path)
    started_at_tick = {}

    def mutate(n):
        if n == 2:  # dip: trough that v0.1 would have packed into
            fb.set_used(0, 1000)
            fb.procs = [(0, DEAD_PID, 1000)]
        if n == 5:  # external process exits -> peak pruned
            fb.set_used(0, 0)
            fb.procs = []
        if 1 in running_nos(sched) and 1 not in started_at_tick:
            started_at_tick[1] = n

    run_tick_loop(sched, fb, mutate)
    assert started_at_tick[1] >= 6, f"packed into a trough at tick {started_at_tick[1]}"
    assert sched.results[0].returncode == 0


def test_undeclared_job_waits_for_external_peak_to_clear(tmp_path):
    """GPU instantaneously 'idle' (150 MiB) but a live external process once
    used 5000 MiB -> not idle for an undeclared job until that process exits."""
    fb = FakeBackend({0: (24000, 5000)})
    fb.procs = [(0, DEAD_PID, 5000)]
    sched = make_sched(fb, [spec(1, vram=None)], tmp_path)
    started_at_tick = {}

    def mutate(n):
        if n == 2:  # process idles down but stays alive
            fb.set_used(0, 150)
            fb.procs = [(0, DEAD_PID, 150)]
        if n == 5:  # process exits
            fb.set_used(0, 30)
            fb.procs = []
        if 1 in running_nos(sched) and 1 not in started_at_tick:
            started_at_tick[1] = n

    run_tick_loop(sched, fb, mutate)
    assert started_at_tick[1] >= 6, f"treated peak-laden GPU as idle at tick {started_at_tick[1]}"


def test_own_jobs_do_not_count_as_external(tmp_path):
    """A scheduler-launched job's VRAM must be attributed, not double-counted
    as external load (which would wrongly block further packing)."""
    fb = FakeBackend({0: (24000, 0)})
    jobs = [spec(1, vram=6000, cmd="sleep 0.6"), spec(2, vram=6000, cmd="sleep 0.6")]
    sched = make_sched(fb, jobs, tmp_path)
    both_running_seen = []

    def mutate(n):
        # Fabricate the running jobs' usage under their REAL pgids so the
        # attribution path matches them as ours.
        procs, used = [], 0
        for rj in sched.running.values():
            procs.append((0, rj.pgid, 5800))
            used += 5800
        fb.procs = procs
        fb.set_used(0, used)
        if len(sched.running) == 2:
            both_running_seen.append(n)

    run_tick_loop(sched, fb, mutate)
    assert both_running_seen, "jobs failed to co-locate: own usage was treated as external"
    assert all(r.attributed for r in sched.results)
    assert all(5700 <= max(r.peak_mib.values()) <= 5900 for r in sched.results)
