"""Command-line interface: ``gpusched jobs.txt [options]``."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .allocation import AllocOptions
from .backend import NvidiaSmiBackend
from .jobspec import JobSpecError, parse_jobs_file
from .scheduler import Scheduler, SchedulerOptions


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gpusched",
        description=(
            "VRAM-aware GPU job scheduler. Reads one shell command per line; "
            "lines may declare expected max VRAM: '[vram=18G] python train.py' "
            "or multiple GPUs: '[vram=30G gpus=2] torchrun ...'. Declared jobs "
            "are packed onto GPUs with enough free memory; undeclared jobs get "
            "a fully idle GPU. Actual per-job VRAM is monitored and compared "
            "against declarations."
        ),
    )
    p.add_argument("jobs_file", help="file with one shell command per line")
    p.add_argument("--gpus", default=None, metavar="0,1,3",
                   help="comma-separated GPU indices to use (default: all visible)")
    p.add_argument("--idle-threshold", type=int, default=200, metavar="MIB",
                   help="GPU counts as idle below this usage, for undeclared jobs (default: 200)")
    p.add_argument("--margin", type=int, default=512, metavar="MIB",
                   help="safety margin added to every vram declaration (default: 512)")
    p.add_argument("--tolerance", type=float, default=0.10, metavar="FRAC",
                   help="relative band before flagging over/under-declaration (default: 0.10)")
    p.add_argument("--poll", type=float, default=5.0, metavar="SEC",
                   help="polling interval in seconds (default: 5)")
    p.add_argument("--spike-buffer", type=float, default=0.10, metavar="FRAC",
                   help="headroom buffer applied over observed VRAM maxima of fluctuating "
                        "processes — both external ones and own jobs that exceeded their "
                        "declaration (default: 0.10)")
    p.add_argument("--exclusive", action="store_true",
                   help="never co-locate two scheduler jobs on one GPU, even if declarations fit")
    p.add_argument("--log-dir", default="gpusched_logs",
                   help="directory for per-job stdout/stderr logs (default: gpusched_logs)")
    p.add_argument("--watch", action="store_true",
                   help="keep running after the queue drains, picking up lines appended "
                        "to the jobs file (the jobs file is re-read every poll either way)")
    p.add_argument("--oom-retries", type=int, default=0, metavar="N",
                   help="default CUDA-OOM auto-retries per job; [retries=N] overrides (default: 0)")
    p.add_argument("--fresh", action="store_true",
                   help="ignore and remove the existing journal: re-run everything")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="stream live per-job VRAM usage as peaks grow")
    p.add_argument("--sim", type=int, default=None, metavar="N_GPUS",
                   help="dry-run against N simulated 24 GiB GPUs (no hardware needed); "
                        "pair with 'python -m gpusched.simjob' commands")
    p.add_argument("--version", action="version", version=f"gpusched {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        jobs = parse_jobs_file(args.jobs_file)
    except JobSpecError as e:
        print(f"gpusched: {e}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"gpusched: cannot read jobs file: {e}", file=sys.stderr)
        return 2
    if not jobs and not args.watch:
        print("gpusched: jobs file contains no jobs (use --watch to wait for some)", file=sys.stderr)
        return 2

    if args.fresh:
        import pathlib
        pathlib.Path(args.log_dir, "journal.jsonl").unlink(missing_ok=True)

    if args.sim is not None:
        from .testing import SimBackend
        backend = SimBackend(n_gpus=args.sim)
    else:
        backend = NvidiaSmiBackend()

    allowed = (
        {int(g) for g in args.gpus.split(",")} if args.gpus else None
    )
    options = SchedulerOptions(
        alloc=AllocOptions(
            idle_threshold_mib=args.idle_threshold,
            margin_mib=args.margin,
            exclusive=args.exclusive,
            spike_buffer=args.spike_buffer,
        ),
        poll_interval=args.poll,
        tolerance=args.tolerance,
        verbose=args.verbose,
        log_dir=args.log_dir,
        watch=args.watch,
        oom_retries_default=args.oom_retries,
    )
    sched = Scheduler(backend, jobs_path=args.jobs_file, allowed_gpus=allowed, options=options)
    try:
        return sched.run()
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
