# gpusched

A VRAM-aware GPU job scheduler for a single machine. You write shell commands
in a text file, optionally declaring how much GPU memory each needs; gpusched
runs them, placing each job on a GPU with enough free memory the moment one
opens up, measures how much VRAM each job actually used, and tells you when
your declarations were wrong.

```
$ cat jobs.txt
[vram=18G] python train.py --config a.yaml
[vram=18G] python train.py --config b.yaml
python preprocess.py

$ gpusched jobs.txt --watch -v
[14:02:10] job 1 started on gpu [0] (1/3 dispatched, declared 18432 MiB/gpu) — ...
[14:02:10] job 2 started on gpu [1] (2/3 dispatched, declared 18432 MiB/gpu) — ...
[14:31:44] job 1 finished [OK] in 1774s — peak vram gpu0:17910 MiB | declared 18432 MiB → within ±10% | avg gpu util 96%
```

**New here? Read [USAGE.md](USAGE.md)** — a plain-language, start-to-finish
guide with worked examples and a list of common mistakes. This README is the
technical reference: design rationale, exact semantics, and limitations.

## What this is, and what it is not

gpusched is a **single-node, single-user research tool**: roughly a thousand
lines you can read in an afternoon, with no daemon, no database, and zero
Python dependencies. It exists for one workflow — you have a box with a few
GPUs, a pile of training/eval commands, and you want them to run unattended
with minimal GPU idle time and without OOM-ing each other.

It is **not** a cluster scheduler, and several of its guarantees are honest
best-effort rather than enforcement:

- **Declarations are advisory.** gpusched places jobs based on what you
  declare and warns loudly when reality diverges, but it cannot cap a
  process's VRAM (nothing can, short of MIG partitions). A job that blows
  through its declaration can still OOM a neighbor — you get a warning, not
  protection. `--exclusive` (one scheduled job per GPU) is the zero-risk mode.
- **It polls.** GPU state is sampled every `--poll` seconds (default 5) via
  `nvidia-smi`. VRAM spikes shorter than the poll interval are invisible to
  both placement and peak reporting.
- **Co-locating compute-heavy jobs can be slower than serializing them.**
  Packing optimizes VRAM occupancy, not throughput. Two trainers that each
  saturate the SMs will roughly halve each other's speed; packing pays off
  for memory-light or compute-light neighbors (eval scripts, dataloader-bound
  jobs, inference). The per-job utilization report helps you see which case
  you're in.
- **Per-job VRAM attribution can fail in some container setups** where
  `nvidia-smi` reports host-namespace PIDs that don't match the container's.
  gpusched then reports the peak as `n/a` rather than a wrong number;
  placement still works because it uses device-level totals.
- **Testing honesty:** the test suite (65 tests) exercises the full scheduler
  against simulated GPU backends and real subprocesses; the `nvidia-smi`
  parsing layer is straightforward but thin. Do a small smoke run on your
  hardware before trusting it with a week of compute.

## If your needs are bigger than this

Use the right tool instead of stretching this one:

| Need | Look at |
|---|---|
| Multiple machines, multiple users, fairness, accounting | Slurm (or PBS/LSF) |
| Hyperparameter optimization with early stopping / ASHA / PBT | Ray Tune, Optuna, W&B Sweeps |
| Distributed training orchestration | Ray, torchrun + Slurm, Kubernetes + device plugins |
| Workflow DAGs (job B consumes job A's output) | Snakemake, Makefile, Airflow |
| Hard VRAM/compute isolation between jobs | NVIDIA MIG (partitioning), MPS (limits) |
| Just a per-GPU FIFO queue, even simpler than this | task-spooler (`ts`), simple_gpu_scheduler |

For sweeps specifically, the intended pattern is: a 10-line script (or
Optuna in ask-and-tell mode) *generates* the jobs file; gpusched stays a dumb
command queue. The moment you want trials stopped early based on metrics, you
have outgrown this tool — that requires bidirectional communication with
running jobs, which is deliberately out of scope.

## Install

```
uv tool install gpusched        # isolated env, `gpusched` on PATH
# or: pip install gpusched
# or, from a clone: uv tool install .
```

Requires Python ≥ 3.10 and Linux with `nvidia-smi` (the scheduler itself has
zero Python dependencies). `uvx gpusched jobs.txt` runs it without installing.

## Sixty-second tour (no GPU required)

```
echo '[vram=8G] python3 -m gpusched.simjob --vram 8000 --ramp 2 --hold 5' > jobs.txt
gpusched jobs.txt --sim 2 --poll 0.5 -v
```

`--sim N` runs against N simulated 24 GiB GPUs; `gpusched.simjob` is a fake
GPU job that ramps and holds a declared amount of fake VRAM. Everything below
behaves identically in sim and on real hardware.

## The jobs file

One shell command per line; blank lines and `#` comments are skipped. An
optional leading `[...]` block declares per-job attributes:

```
# no declaration -> runs only on a fully idle GPU, alone
python preprocess.py

# declared max VRAM (per GPU): may share a GPU when the declared amount fits
[vram=18000] python train.py --config a.yaml
[vram=22G]   bash run_eval.sh

# multi-GPU: 2 GPUs, EACH with >= 30 GiB free; CUDA_VISIBLE_DEVICES gets both
[vram=30G gpus=2] torchrun --nproc_per_node=2 train_big.py

# auto-retry on CUDA OOM, declaration bumped ~1.25x of observed peak per retry
[vram=8G retries=2] python sweep.py --seed 3

# opt-in walltime: SIGTERM at 2h, SIGKILL +10s. No timeout attribute = runs
# forever; the scheduler never guesses which long-running jobs are hung.
[timeout=2h] python flaky_eval.py

# cancel: stops the job if running (SIGTERM, then SIGKILL), or prevents it
# from ever starting if pending. Add the token to the line; identity is the
# command text, so the same job is targeted.
[vram=18G cancel] python train.py --config a.yaml
```

`vram` accepts MiB integers or `G`/`GiB` suffixes and is always per GPU.
Declare your honest worst case; the completion report tells you how close you
were, so declarations converge after a run or two.

## How placement works

A job **without** a declaration gets a GPU only when it is fully idle (below
`--idle-threshold`, default 200 MiB, with no other scheduled job) — the safe
default when you don't know what a job needs.

A job **with** a declaration of E MiB can be placed wherever *effective
headroom* ≥ E + `--margin` (default 512 MiB). Effective headroom accounts for
three things. First, a just-launched job that hasn't allocated its CUDA
context yet still reserves everything it declared — this closes the classic
double-booking race in poll-based schedulers, where a GPU looks empty for the
few seconds before a process materializes. Second, every process gpusched did
not launch is tracked to its observed per-GPU peak and held to
`peak × (1 + --spike-buffer)` until it exits, so a fluctuating external
process's momentary trough is not treated as packable space. Third, a
scheduled job that exceeds its own declaration stops being trusted: its
budget escalates from the declaration to its buffered observed peak.

Queue order is file order with backfill: if the next job can't fit right now,
smaller jobs behind it run first. A job that could not fit even on a
completely empty GPU fails immediately as `INFEASIBLE` rather than stalling
the queue. Multi-GPU jobs take N distinct GPUs, each meeting the per-GPU
requirement, chosen best-fit to preserve large contiguous headroom.

(One physical reality worth knowing: PyTorch's caching allocator rarely
returns VRAM to the driver, so for torch jobs `nvidia-smi` already reads near
the high-water mark — the spike-buffer machinery matters most for processes
that genuinely release memory between phases.)

## Monitoring: declared vs actual

Each scheduled job runs in its own session (`setsid`), so all its descendant
processes share one process-group id; each poll, gpusched maps `nvidia-smi`'s
per-process VRAM onto jobs by pgid and tracks per-GPU peaks. Two asymmetric
notifications, tuned by `--tolerance` (default ±10%):

**Under-declaration warns immediately** — the first poll where actual exceeds
declared, you get `WARN job N EXCEEDS declared VRAM ... neighbors may OOM`,
because at that moment the packing math other jobs were placed under is
already violated. **Over-declaration is reported at completion** — a
fluctuating job may legitimately peak late, so it can only be judged once it
exits: `declared 12288 MiB → over-declared (-59%); lowering it frees packing
headroom`.

Every completion line is streamed the moment the job finishes and includes
its per-GPU peak and average device-level GPU utilization (device-level: when
two jobs share a GPU the number is confounded — treat it as a diagnostic for
spotting dataloader-bound runs, not a per-process metric). `--verbose` adds
live usage lines as a job's peak grows.

## Live queue, resume, and the status board

The jobs file is **user-owned and never written by the scheduler**; it is
re-read every poll. Each line has a stable identity (hash of its command text
plus an occurrence counter for duplicate lines), and an append-only journal
(`<log_dir>/journal.jsonl`) records attempts and outcomes per identity.
"Pending" is defined as: lines in the file that are neither running nor
terminal in the journal. Everything follows from that one definition —
append a line from any terminal and it is dispatched within a poll; delete a
pending line and it is dequeued; reorder pending lines and you have reordered
the queue (file order among pending IS the priority; there is no separate
priority mechanism); edit a pending line and you have replaced it. Edits to
running or completed lines do nothing. A malformed mid-edit save is rejected
with a warning and the last good queue is kept; in-flight jobs are never
affected.

**Cancelling.** To stop a running job — the loss curve told you the run is
doomed and the GPU-hours are better spent elsewhere — edit its line to add
the bare `cancel` token: `[vram=18G] python train.py` becomes
`[vram=18G cancel] python train.py`. The job's process group gets SIGTERM,
then SIGKILL after a grace period, with a final sweep so no descendant that
traps SIGTERM outlives it. On a pending job, `cancel` marks it terminal
without running. Cancellation is deliberately explicit: **deleting a running
job's line does nothing** — killing real compute requires typing the word, so
an accidental save can't destroy a two-day run. A cancellation is your
decision, so it is reported separately from failures, never consumes OOM
retries, and does not make the scheduler's exit code nonzero. Removing the
token later does not resurrect the job (terminal is terminal, like any
completed job — re-run via a trivial line change or `--fresh`).

Re-running the same command after a crash or Ctrl-C skips everything the
journal marks done — that is resume. `--fresh` wipes the journal to re-run
all; to re-run one job, change its line trivially (new identity). `--watch`
keeps the scheduler alive after the queue drains, waiting for appended lines.

A live board is rendered to `<log_dir>/status.txt` every poll
(`▶` running, `·` pending, `↻` retrying after OOM, `✓`/`✗` done, `⊘` cancelled):

```
watch -n2 cat gpusched_logs/status.txt
```

Known limitation: with continuous submission, a large blocked job can be
starved by a stream of small backfilled ones. There is no aging policy — you
are the priority mechanism. Move the big job's line up and hold small
submissions, or run it under `--exclusive`.

## OOM retry and timeouts

A failed job whose log tail matches CUDA OOM signatures and that declared
`[retries=N]` (or ran under `--oom-retries N`) is requeued instead of
terminal-failed: its declaration is bumped to ~1.25× of max(observed peak,
old declaration), recorded in the journal (so it survives scheduler
restarts), and applied on the next attempt — the retry is scheduled with
honest requirements instead of repeating the same collision. Non-OOM failures
never consume retries, so retry loops cannot mask code bugs.

Timeouts are strictly per-job opt-in (`[timeout=90s|15m|2h|1d]`).
Distinguishing a hung process from a legitimate three-day run is your
declaration, never a heuristic — heuristics (e.g. "0% util for 10 minutes")
kill legitimate CPU phases like preprocessing and checkpoint serialization,
so none are included.

## Running detached (tmux)

The scheduler is an ordinary foreground process — run it inside tmux and
disconnect freely:

```
tmux new -s sched
gpusched jobs.txt --watch -v            # pane 1
# Ctrl-b c -> pane 2:
watch -n2 cat gpusched_logs/status.txt
# Ctrl-b d to detach; later: tmux attach -t sched
```

You can also append jobs over ssh without attaching at all — the file is
re-read every poll. Failure-mode hierarchy, honestly: tmux protects
everything from SSH drops; the journal protects queue state from scheduler
death; and orphans are guarded against double-runs — every launch is recorded
in the journal with its process-group id, so a restarted scheduler that finds
a previous attempt **still alive** refuses to re-dispatch it (and tells you
the pgid to wait out or kill), while a previous attempt that **died with the
scheduler** is marked interrupted and re-queued automatically. What is still
genuinely unrecoverable: a live orphan's exit code — the scheduler cannot
re-attach to a process it didn't spawn.

## Examples

See [examples/](examples/) for a complete, runnable demo: parallel
hyperparameter sweeps of a small GPT and a DDPM across multiple GPUs,
with optional Weights & Biases logging.

## CLI reference

```
gpusched jobs.txt
  --gpus 0,1,3          restrict to these GPU indices (default: all visible)
  --idle-threshold 200  MiB below which a GPU counts as idle (undeclared jobs)
  --margin 512          MiB safety margin added to every declaration
  --tolerance 0.10      band before flagging over/under-declaration
  --spike-buffer 0.10   buffer over observed VRAM maxima of fluctuating processes
  --poll 5              seconds between scheduling rounds
  --exclusive           one scheduled job per GPU, even when declarations fit
  --watch               keep running after drain; pick up appended lines
  --oom-retries N       default CUDA-OOM auto-retries ([retries=N] overrides)
  --fresh               ignore + remove the journal: re-run everything
  --log-dir DIR         per-job logs, journal, status board (default: gpusched_logs)
  -v / --verbose        stream live per-job VRAM as peaks grow
  --sim N               dry-run on N simulated 24 GiB GPUs (no hardware)
```

Exit code: 0 if every job succeeded, otherwise the max failing job exit code.

## Architecture and extending

```
src/gpusched/
  jobspec.py     parsing: [vram=.. gpus=.. timeout=.. retries=..] cmd -> JobSpec
  backend.py     GpuBackend protocol; NvidiaSmiBackend (2 queries/poll); pgid attribution
  allocation.py  PURE placement function: headroom, reservations, best-fit
  journal.py     append-only JSONL: attempts + terminal outcomes per job identity
  scheduler.py   tick loop: snapshot -> attribute -> warn -> timeouts -> reap -> dispatch
  testing.py     FakeBackend (unit tests), SimBackend (integration / --sim)
  simjob.py      simulated GPU job for tests and dry runs
  cli.py         argparse front end
```

The deliberate seams: `allocation.find_allocation` is pure (snapshot +
occupants in, GPU list out), so new placement rules — GPU-type constraints,
NVLink-aware pairing — are filters there plus an attribute in the jobspec
parser, with nothing else touched. Alternative monitors (pynvml, DCGM)
implement the two-method `GpuBackend` protocol. The journal is the only
persistent state.

## Development

```
git clone <repo> && cd gpusched
uv venv && uv pip install -e ".[dev]"
uv run pytest -q          # 65 tests, ~20s, no GPU required
```

Tests drive the scheduler against fake/simulated backends with real
subprocesses; several are timing-based (sub-second sim jobs with fast polls),
so a heavily loaded machine can occasionally need a re-run.

## License

MIT.
