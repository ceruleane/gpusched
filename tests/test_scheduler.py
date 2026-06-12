import os
import re
import time
from io import StringIO

from gpusched.backend import pgid_of
from gpusched.jobspec import JobSpec
from gpusched.scheduler import Scheduler, SchedulerOptions
from gpusched.allocation import AllocOptions
from gpusched.testing import SIM_DIR_ENV, FakeBackend, SimBackend


def test_pgid_of_self_matches_os():
    assert pgid_of(os.getpid()) == os.getpgid(os.getpid())


def test_pgid_of_dead_pid_is_none():
    assert pgid_of(2**22 + 12345) is None


def fast_opts(**kw) -> SchedulerOptions:
    base = dict(
        alloc=AllocOptions(idle_threshold_mib=200, margin_mib=512),
        poll_interval=0.05, verbose=True, verbose_step_mib=128,
    )
    base.update(kw)
    return SchedulerOptions(**base)


def run_sim(jobs, n_gpus=2, total=24000, tmp_path=None, external=None, **opt_kw):
    sim_dir = str(tmp_path / "sim")
    os.environ[SIM_DIR_ENV] = sim_dir  # inherited by launched sim jobs
    backend = SimBackend(n_gpus=n_gpus, total_mib=total, external_mib=external, sim_dir=sim_dir)
    out = StringIO()
    opts = fast_opts(log_dir=str(tmp_path / "logs"), **opt_kw)
    sched = Scheduler(backend, jobs, options=opts, out=out)
    rc = sched.run()
    return rc, sched, out.getvalue()


def simjob(i, vram_peak, declared=None, ramp=0.2, hold=0.4, gpus=1, exit_code=0):
    cmd = (
        f"python3 -m gpusched.simjob --vram {vram_peak} "
        f"--ramp {ramp} --hold {hold} --exit-code {exit_code}"
    )
    return JobSpec(index=i, command=cmd, vram_mib=declared, n_gpus=gpus)


def test_end_to_end_accurate_declaration(tmp_path):
    jobs = [simjob(1, vram_peak=8000, declared=8000)]
    rc, sched, log = run_sim(jobs, tmp_path=tmp_path)
    assert rc == 0
    [r] = sched.results
    assert r.verdict == "ok" and r.attributed
    assert 7200 <= r.peak_mib[r.gpu_indices[0]] <= 8000
    assert "within ±10%" in log


def test_under_declared_warns_immediately_and_verdict_over(tmp_path):
    jobs = [simjob(1, vram_peak=12000, declared=8000, hold=0.6)]
    rc, sched, log = run_sim(jobs, tmp_path=tmp_path)
    [r] = sched.results
    assert r.verdict == "over"
    assert "EXCEEDS declared VRAM" in log          # live warning
    assert "UNDER-DECLARED" in log                 # completion verdict


def test_over_declared_reported_at_completion_only(tmp_path):
    jobs = [simjob(1, vram_peak=4000, declared=12000)]
    rc, sched, log = run_sim(jobs, tmp_path=tmp_path)
    [r] = sched.results
    assert r.verdict == "under"
    assert "EXCEEDS" not in log
    assert "over-declared" in log


def test_verbose_streams_per_job_not_batched(tmp_path):
    jobs = [simjob(1, 6000, declared=6000), simjob(2, 6000, declared=6000)]
    rc, sched, log = run_sim(jobs, tmp_path=tmp_path)
    # each job's live vram lines must appear before the LAST completion line
    last_finish = max(log.rfind("job 1 finished"), log.rfind("job 2 finished"))
    first_live = log.find("vram now")
    assert 0 <= first_live < last_finish


def test_packing_two_declared_jobs_one_gpu(tmp_path):
    # 24000-MiB GPUs; two 9000-declared jobs fit on gpu0 together (9512*2 < 24000).
    jobs = [simjob(1, 9000, declared=9000, hold=0.8), simjob(2, 9000, declared=9000, hold=0.8)]
    rc, sched, log = run_sim(jobs, n_gpus=1, tmp_path=tmp_path)
    assert rc == 0
    starts = re.findall(r"job \d started on gpu \[0\]", log)
    assert len(starts) == 2
    # both dispatched before either finished (true concurrency on one GPU)
    assert log.find("job 2 started") < log.find("job 1 finished")


def test_undeclared_jobs_serialize_on_one_gpu(tmp_path):
    jobs = [simjob(1, 5000, declared=None, hold=0.5), simjob(2, 5000, declared=None, hold=0.5)]
    rc, sched, log = run_sim(jobs, n_gpus=1, tmp_path=tmp_path)
    assert rc == 0
    assert log.find("job 1 finished") < log.find("job 2 started")


def test_backfill_smaller_job_skips_blocked_head(tmp_path):
    # Head job needs 20000 on a GPU with 18000 external -> blocked for now;
    # the 3000-MiB job behind it must start anyway (idle-time minimization).
    # External load is then released so the head job eventually runs too.
    sim_dir = str(tmp_path / "sim")
    os.environ[SIM_DIR_ENV] = sim_dir
    backend = SimBackend(n_gpus=1, total_mib=24000, external_mib={0: 18000}, sim_dir=sim_dir)
    jobs = [simjob(1, 20000, declared=20000, hold=0.3), simjob(2, 3000, declared=3000, hold=0.3)]
    out = StringIO()
    sched = Scheduler(backend, jobs, options=fast_opts(log_dir=str(tmp_path / "logs")), out=out)

    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        sched.tick()
        if any(rj.no == 2 for rj in sched.running.values()) and backend.external:
            backend.external = {}  # job 2 started -> free the GPU
        if sched.drained:
            break
        time.sleep(0.05)

    log = out.getvalue()
    assert sched.drained, "scheduler did not drain in time"
    assert all(r.returncode == 0 for r in sched.results)
    assert log.find("job 2 started") < log.find("job 1 started")


def test_truly_infeasible_job_fails_fast(tmp_path):
    # 30000 MiB declared on 24000 MiB GPUs can never fit -> immediate failure,
    # and the feasible job behind it still runs.
    jobs = [simjob(1, 30000, declared=30000), simjob(2, 2000, declared=2000)]
    rc, sched, log = run_sim(jobs, n_gpus=1, tmp_path=tmp_path)
    by_idx = {r.spec.index: r for r in sched.results}
    assert by_idx[1].verdict == "infeasible" and by_idx[1].returncode == 1
    assert by_idx[2].returncode == 0
    assert "INFEASIBLE" in log
    assert rc == 1


def test_multi_gpu_job_gets_two_gpus_and_per_gpu_peaks(tmp_path):
    jobs = [simjob(1, 10000, declared=10000, gpus=2)]
    rc, sched, log = run_sim(jobs, n_gpus=2, tmp_path=tmp_path)
    [r] = sched.results
    assert r.gpu_indices == (0, 1)
    assert set(r.peak_mib) == {0, 1}
    assert all(9000 <= m <= 10000 for m in r.peak_mib.values())


def test_failed_job_exit_code_propagates(tmp_path):
    jobs = [simjob(1, 2000, declared=2000, exit_code=3)]
    rc, sched, log = run_sim(jobs, tmp_path=tmp_path)
    assert rc == 3 and "EXIT 3" in log


def test_cpu_only_job_reports_unattributed(tmp_path):
    jobs = [JobSpec(index=1, command="sleep 0.3", vram_mib=4000, n_gpus=1)]
    rc, sched, log = run_sim(jobs, tmp_path=tmp_path)
    [r] = sched.results
    assert r.verdict == "unattributed"
    assert "n/a" in log


def test_attribution_isolates_concurrent_jobs(tmp_path):
    # Two different-sized jobs sharing one GPU: peaks must not bleed.
    jobs = [simjob(1, 9000, declared=9000, hold=0.8), simjob(2, 3000, declared=3000, hold=0.8)]
    rc, sched, log = run_sim(jobs, n_gpus=1, tmp_path=tmp_path)
    by_idx = {r.spec.index: r for r in sched.results}
    assert 8100 <= max(by_idx[1].peak_mib.values()) <= 9000
    assert 2700 <= max(by_idx[2].peak_mib.values()) <= 3000
