import os
import signal
import subprocess
import time
from io import StringIO
from pathlib import Path

from gpusched.allocation import AllocOptions
from gpusched.jobspec import parse_line
from gpusched.journal import Journal
from gpusched.scheduler import Scheduler, SchedulerOptions
from gpusched.testing import SIM_DIR_ENV, SimBackend


def opts(tmp_path, **kw):
    base = dict(
        alloc=AllocOptions(idle_threshold_mib=200, margin_mib=512),
        poll_interval=0.05, log_dir=str(tmp_path / "logs"),
    )
    base.update(kw)
    return SchedulerOptions(**base)


def drive(sched, mutate=None, max_s=25):
    deadline = time.monotonic() + max_s
    n = 0
    while time.monotonic() < deadline:
        if mutate:
            mutate(n, sched)
        sched.tick()
        n += 1
        if sched.drained:
            return n
        time.sleep(0.03)
    raise AssertionError("did not drain in time")


def sim_backend(tmp_path):
    sim_dir = str(tmp_path / "sim")
    os.environ[SIM_DIR_ENV] = sim_dir
    return SimBackend(n_gpus=1, total_mib=24000, sim_dir=sim_dir)


def simcmd(vram, hold):
    return f"python3 -m gpusched.simjob --vram {vram} --ramp 0.2 --hold {hold}"


# ---------------------------------------------------------------- parsing
def test_cancel_token_parses_and_preserves_identity():
    a = parse_line("[vram=4G] python x.py", 1, 1)
    b = parse_line("[vram=4G cancel] python x.py", 1, 1)
    assert not a.cancel and b.cancel
    assert a.command == b.command  # identity hashes command -> same job targeted


def test_unknown_bare_token_rejected():
    import pytest
    from gpusched.jobspec import JobSpecError
    with pytest.raises(JobSpecError):
        parse_line("[kancel] python x.py", 1, 1)


# ---------------------------------------------------------------- cancel running
def test_cancel_running_job_via_file_edit(tmp_path):
    jf = tmp_path / "jobs.txt"
    cmd = simcmd(4000, hold=30)            # would run ~30s if not cancelled
    jf.write_text(f"[vram=4G] {cmd}\n")
    out = StringIO()
    sched = Scheduler(sim_backend(tmp_path), jobs_path=str(jf), options=opts(tmp_path), out=out)
    t0 = time.monotonic()

    def mutate(n, s):
        if n == 4:  # user edits the line: adds the cancel token
            jf.write_text(f"[vram=4G cancel] {cmd}\n")

    drive(sched, mutate)
    assert time.monotonic() - t0 < 12, "cancellation did not actually stop the job"
    [r] = sched.results
    assert r.verdict == "cancelled" and r.returncode != 0
    log = out.getvalue()
    assert "CANCELLED by user" in log and "finished [CANCELLED]" in log


def test_cancelled_is_not_counted_failed_and_exit_code_is_zero(tmp_path):
    jf = tmp_path / "jobs.txt"
    cmd = simcmd(4000, hold=30)
    jf.write_text(f"[vram=4G retries=3] {cmd}\n")   # retries must NOT trigger
    out = StringIO()
    sched = Scheduler(sim_backend(tmp_path), jobs_path=str(jf), options=opts(tmp_path), out=out)

    def mutate(n, s):
        if n == 4:
            jf.write_text(f"[vram=4G retries=3 cancel] {cmd}\n")

    drive(sched, mutate)
    [r] = sched.results
    assert r.verdict == "cancelled"
    assert sched.journal.state(r.spec.key).attempts == 1     # no retry burned
    rc = sched._summarize()
    assert rc == 0, "user cancellation must not fail the scheduler run"
    assert "1 cancelled" in out.getvalue()


def test_cancel_escalates_to_sigkill_for_term_trapping_job(tmp_path):
    jf = tmp_path / "jobs.txt"
    # A single process that ignores SIGTERM. (A bash 'trap' wrapper is NOT
    # enough: killpg signals the whole group, and bash's sleep child dies.)
    # It touches a ready-file AFTER installing the handler; the test must not
    # cancel before that, or SIGTERM races Python interpreter startup.
    ready = tmp_path / "handler.ready"
    # 'exec' makes python the GROUP LEADER (sh otherwise forks here, and
    # killpg would reap the unprotected sh instantly — see straggler sweep).
    cmd = ("exec python3 -c \"import signal,time,pathlib; "
           "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
           f"pathlib.Path('{ready}').touch(); time.sleep(60)\"")
    jf.write_text(f"[vram=1000] {cmd}\n")
    out = StringIO()
    sched = Scheduler(
        sim_backend(tmp_path), jobs_path=str(jf),
        options=opts(tmp_path, kill_grace_s=0.3), out=out,
    )
    t0 = time.monotonic()

    def mutate(n, s):
        if ready.exists() and "cancel]" not in jf.read_text():
            jf.write_text(f"[vram=1000 cancel] {cmd}\n")

    drive(sched, mutate)
    assert time.monotonic() - t0 < 15, "SIGKILL escalation did not happen"
    assert "SIGKILL" in out.getvalue()
    [r] = sched.results
    assert r.verdict == "cancelled"


# ---------------------------------------------------------------- cancel pending
def test_cancel_pending_job_never_starts(tmp_path):
    jf = tmp_path / "jobs.txt"
    a, b = simcmd(4000, hold=0.8), simcmd(4001, hold=0.8)
    # undeclared-style serialization not needed: one GPU, both declared 14G
    jf.write_text(f"[vram=14G] {a}\n[vram=14G] {b}\n")
    out = StringIO()
    sched = Scheduler(sim_backend(tmp_path), jobs_path=str(jf), options=opts(tmp_path), out=out)

    def mutate(n, s):
        if n == 2:  # job 2 is still pending (job 1 holds the GPU) -> cancel it
            jf.write_text(f"[vram=14G] {a}\n[vram=14G cancel] {b}\n")

    drive(sched, mutate)
    by_verdict = {r.verdict for r in sched.results}
    assert "cancelled" in by_verdict and len(sched.results) == 2
    survivor = [r for r in sched.results if r.verdict != "cancelled"][0]
    assert survivor.returncode == 0   # (its declaration-accuracy verdict is not under test)
    cancelled = [r for r in sched.results if r.verdict == "cancelled"][0]
    assert cancelled.gpu_indices == () and cancelled.duration_s == 0.0
    assert "cancelled before start" in out.getvalue()


def test_removing_cancel_token_does_not_resurrect(tmp_path):
    jf = tmp_path / "jobs.txt"
    cmd = simcmd(4000, hold=0.5)
    jf.write_text(f"[vram=4G cancel] {cmd}\n")      # cancelled before ever running
    o = opts(tmp_path)
    sched1 = Scheduler(sim_backend(tmp_path), jobs_path=str(jf), options=o, out=StringIO())
    drive(sched1)
    assert sched1.results[0].verdict == "cancelled"

    jf.write_text(f"[vram=4G] {cmd}\n")             # user removes the token
    sched2 = Scheduler(sim_backend(tmp_path), jobs_path=str(jf), options=o, out=StringIO())
    drive(sched2)
    assert sched2.results == [], "terminal cancelled job must stay terminal"


# ---------------------------------------------------------------- orphans
def test_orphan_alive_blocks_redispatch_then_dead_requeues(tmp_path):
    jf = tmp_path / "jobs.txt"
    cmd = simcmd(2000, hold=0.4)
    jf.write_text(f"[vram=2G] {cmd}\n")
    o = opts(tmp_path)

    # Fabricate a previous scheduler's death mid-run: journal says 'started'
    # with the pgid of a real live process we control.
    orphan = subprocess.Popen(["sleep", "30"], start_new_session=True)
    from gpusched.jobspec import parse_jobs_file
    key = parse_jobs_file(str(jf))[0].key
    j = Journal(Path(o.log_dir) / "journal.jsonl")
    j.ensure_seen(key, cmd)
    j.record_started(key, orphan.pid, [0])

    out = StringIO()
    sched = Scheduler(sim_backend(tmp_path), jobs_path=str(jf), options=o, out=out)
    for _ in range(4):
        sched.tick()
        time.sleep(0.03)
    assert sched.running == {}, "must not double-launch while the orphan lives"
    assert "ORPHAN" in out.getvalue() and f"-{orphan.pid}" in out.getvalue()

    os.killpg(orphan.pid, signal.SIGKILL)           # orphan dies
    orphan.wait()
    drive(sched)                                    # now it re-queues and runs
    assert "interrupted mid-run" in out.getvalue()
    [r] = sched.results
    assert r.returncode == 0 and r.verdict in ("ok", "under", "over")


def test_started_event_cleared_on_normal_completion(tmp_path):
    jf = tmp_path / "jobs.txt"
    jf.write_text(f"[vram=4G] {simcmd(4000, hold=0.4)}\n")
    o = opts(tmp_path)
    sched = Scheduler(sim_backend(tmp_path), jobs_path=str(jf), options=o, out=StringIO())
    drive(sched)
    # Reload the journal as a fresh scheduler would: no dangling pgid.
    j = Journal(Path(o.log_dir) / "journal.jsonl")
    [st] = j.states.values()
    assert st.running_pgid is None and st.status == "ok"


def test_no_stragglers_survive_a_cancel(tmp_path):
    """sh forks the real program here; killing the group reaps sh first. The
    straggler sweep must ensure the forked child dies too."""
    jf = tmp_path / "jobs.txt"
    cmd = 'python3 -c "import time; time.sleep(60)"'
    jf.write_text(f"[vram=1000] {cmd}\n")
    sched = Scheduler(sim_backend(tmp_path), jobs_path=str(jf), options=opts(tmp_path), out=StringIO())
    seen_pgid = []

    def mutate(n, s):
        for rj in s.running.values():
            if not seen_pgid:
                seen_pgid.append(rj.pgid)
        if seen_pgid and "cancel]" not in jf.read_text():
            jf.write_text(f"[vram=1000 cancel] {cmd}\n")

    drive(sched, mutate)
    assert seen_pgid, "job never started"
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if not Scheduler._pgid_alive(seen_pgid[0]):
            return
        time.sleep(0.05)
    raise AssertionError("a process from the cancelled job's group is still alive")
