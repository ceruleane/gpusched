# Using gpusched, start to finish

This guide walks through everything gpusched does, in the order you'll
actually meet it: install, first run, daily use, and the mistakes people
make. It assumes nothing beyond comfort with a terminal. The
[README](README.md) covers internals, design rationale, and limitations in
depth; this file is the practical path.

The running example: you have one machine with two 24 GB GPUs and a handful
of training commands to get through — say, a small GPT you're sweeping
learning rates on.

## 1. The mental model (three sentences)

You keep a plain text file of shell commands, one per line. gpusched runs
them for you, starting each command on a GPU that has enough free memory,
as soon as one does. While they run, it measures what each job really used
and writes a live scoreboard you can glance at.

That's the whole program. The text file is yours — gpusched reads it every
few seconds and never writes to it. Everything gpusched knows is kept in a
separate folder (`gpusched_logs/`): per-job output logs, the scoreboard
(`status.txt`), and a journal of what finished.

## 2. Install

```
uv tool install gpusched        # or: pip install gpusched
gpusched --version
```

If the shell says `gpusched: command not found` after a uv install, your
PATH is missing `~/.local/bin` — add `export PATH="$HOME/.local/bin:$PATH"`
to your shell rc. Requirements: Linux, Python ≥ 3.10, and `nvidia-smi`
working (run it once to check). No other dependencies.

## 3. Try it in sixty seconds, no GPU needed

`--sim N` pretends you have N empty 24 GB GPUs, and `gpusched.simjob` is a
pretend GPU program. This is the safe sandbox for learning every feature in
this guide:

```
mkdir demo && cd demo
cat > jobs.txt << 'EOF'
[vram=8G]  python3 -m gpusched.simjob --vram 7800 --ramp 1 --hold 3
[vram=8G]  python3 -m gpusched.simjob --vram 7900 --ramp 1 --hold 3
[vram=20G] python3 -m gpusched.simjob --vram 19000 --ramp 1 --hold 2
EOF
gpusched jobs.txt --sim 2 --poll 0.5
```

Real output:

```
[16:24:07] 3 jobs queued [live queue (file re-read each tick)] (poll every 0.5s; ...)
[16:24:07] discovered GPUs: [0, 1]
[16:24:07] job 1 started on gpu [0] (1/3 dispatched, declared 8192 MiB/gpu) — ...
[16:24:07] job 2 started on gpu [0] (2/3 dispatched, declared 8192 MiB/gpu) — ...
[16:24:07] job 3 started on gpu [1] (3/3 dispatched, declared 20480 MiB/gpu) — ...
[16:24:10] job 3 finished [OK] in 4s — peak vram gpu1:19000 MiB | declared 20480 MiB → within ±10% | avg gpu util 77%
[16:24:11] job 1 finished [OK] in 5s — peak vram gpu0:7800 MiB | declared 8192 MiB → within ±10% | avg gpu util 80%
[16:24:11] job 2 finished [OK] in 5s — peak vram gpu0:7900 MiB | declared 8192 MiB → within ±10% | avg gpu util 80%
[16:24:11] all 3 jobs done — 3 ok, 0 failed, 0 cancelled, 0 under-declared vram, 0 over-declared vram
```

Read what happened: jobs 1 and 2 each said "I need 8 GiB", so gpusched put
**both on gpu 0** — they fit together with room to spare. Job 3 said
"20 GiB" and got gpu 1 to itself. All three ran at once; nothing waited.
That sharing-when-it-fits is the entire point of declaring memory.

## 4. Your first real run

Same idea with real commands. Each job's line may start with a `[...]`
block declaring what it needs:

```
# jobs.txt — one shell command per line, # comments allowed
[vram=14G] python train.py --lr 3e-4 --out runs/lr3e4
[vram=14G] python train.py --lr 1e-4 --out runs/lr1e4
[vram=14G] python train.py --lr 3e-5 --out runs/lr3e5
python make_dataset.py        # no declaration — see below
```

```
gpusched jobs.txt -v
```

Rules of thumb that cover 95% of use:

- **`vram` is per GPU. Bare numbers are MiB; write `14G` for gibibytes.**
- **A job with no declaration only ever runs alone on a completely idle
  GPU.** That's the safe default for unknown jobs (and fine for CPU-only
  ones like `make_dataset.py`), but it means undeclared jobs never share —
  if everything in your file is undeclared, nothing packs and your queue
  serializes one-per-GPU.
- Each job sees only its assigned GPU(s): gpusched sets
  `CUDA_VISIBLE_DEVICES`, so your scripts need no per-GPU logic.
- Each job's stdout/stderr goes to `gpusched_logs/jobNNN_gpuX.log` —
  `tail -f` the one you care about.
- Commands run under `/bin/sh` from the directory where you started
  gpusched. Two consequences: write paths relative to that directory (or
  absolute), and activate environments in a sh-compatible way — the robust
  pattern is an absolute interpreter path:

```
[vram=14G] /home/you/proj/.venv/bin/python train.py --lr 3e-4
[vram=14G] conda run -n gpt python train.py --lr 1e-4
```

## 5. Where do the vram numbers come from?

You don't need to be right the first time — the report tells you, and
declarations converge in a run or two:

1. **First time, guess high** (or check `nvidia-smi` during a manual run).
2. Run it. The finish line is your calibration:

```
job 2 finished [OK] in 1774s — peak vram gpu0:11890 MiB | declared 18432 MiB → over-declared (-35%); lowering it frees packing headroom
```

3. Set the declaration to that peak plus ~10% and move on. (`13G` here.)

If you guessed **too low**, you find out the moment it happens, not at the
end:

```
WARN  job 2 EXCEEDS declared VRAM: 15210 MiB observed on gpu 0 vs 12288 MiB declared (+24%) — neighbors may OOM
```

Take that warning seriously: gpusched placed other jobs assuming your
number. It cannot cap a job's memory (nothing can, short of hardware
partitioning) — it can only tell you. If you'd rather never share GPUs at
all, run with `--exclusive`: declarations are then used only to coexist
with *other people's* processes, never to co-locate your own jobs.

## 6. Running overnight (tmux)

gpusched is a normal foreground program; tmux is how you walk away from it:

```
tmux new -s sched
gpusched jobs.txt --watch -v          # pane 1: the scheduler
# Ctrl-b c  -> new pane:
watch -n2 cat gpusched_logs/status.txt   # pane 2: live scoreboard
# Ctrl-b d  -> detach. Close the laptop. Go home.
```

Later, from anywhere: `ssh box`, then `tmux attach -t sched`. The
scoreboard looks like:

```
gpusched status @ 02:13:09 — 2 running, 3 pending, 4 done
▶ job 5   gpu[0]  6841s  gpu0:13880 MiB (peak 13911) — python train.py --lr 3e-4 ...
▶ job 6   gpu[1]  6840s  gpu1:13902 MiB (peak 13955) — python train.py --lr 1e-4 ...
· job 7   pending vram 14336 — python train.py --lr 3e-5 ...
✓ job 1   ok (exit 0, peak 13899 MiB) — ...
✗ job 3   failed (exit 1, peak 2110 MiB) — ...
```

`--watch` means: when the queue empties, don't exit — keep watching the
file for new lines. Without it, gpusched exits when everything is done.

**Detaching (Ctrl-b d) and quitting (Ctrl-C) are very different.** Ctrl-C
tells gpusched to shut down, and it takes its running jobs with it (that's
deliberate — no zombie GPU jobs). Detach to leave; Ctrl-C to stop.

## 7. Driving the queue while it runs

The jobs file stays live the whole time. You edit it — from inside tmux,
from a second ssh session, from your editor — and gpusched notices within
one poll:

**Add work** — append a line:

```
echo '[vram=14G] python train.py --lr 1e-5 --out runs/lr1e5' >> jobs.txt
```

**Reprioritize** — reorder lines. Among jobs that haven't started, top of
file runs first. Move the urgent one up; that's the entire priority system.

**Remove queued work** — delete its line (only affects jobs that haven't
started; see the mistakes section).

**Replace queued work** — just edit the line. To gpusched, a changed
command is a different job: the old one vanishes from the queue, the new
one joins it.

If you save the file in a broken state mid-edit, nothing bad happens —
gpusched warns once, keeps the last good version of the queue, and running
jobs are never touched.

## 8. Cancelling a run that's no longer worth the electricity

Twenty minutes in, the loss curve tells you `lr=3e-4` is garbage. Stop
paying for it: open `jobs.txt` and add the word `cancel` inside that line's
brackets —

```
before:  [vram=14G] python train.py --lr 3e-4 --out runs/lr3e4
after:   [vram=14G cancel] python train.py --lr 3e-4 --out runs/lr3e4
```

Within a poll:

```
WARN  job 1 CANCELLED by user — SIGTERM to its process group
job 1 finished [CANCELLED] in 1306s — peak vram gpu0:13911 MiB | avg gpu util 96%
```

The job gets a polite SIGTERM (your training code can catch it to save a
final checkpoint), a SIGKILL ten seconds later if it ignores that, and a
final sweep so nothing it spawned lingers. Its GPU is immediately offered
to the next pending job. A `cancel` on a job that hasn't started simply
prevents it from ever starting. Cancellations are your decision, so they're
counted separately from failures and don't trip retries or exit codes.

Why a word instead of deleting the line? So that an accidental edit can
never kill a two-day run. Destroying real compute requires typing `cancel`.

## 9. Set-and-forget extras: OOM retries and timeouts

**OOM retry** — for sweeps where an occasional out-of-memory is expected:

```
[vram=12G retries=2] python train.py --lr 3e-4 --batch 64
```

If the job dies with a CUDA out-of-memory error, gpusched re-queues it with
the declaration raised to ~1.25× of what it actually observed, and tries
again (up to 2 more times here) — so the retry is scheduled with honest
requirements instead of repeating the same collision:

```
↻ job 4 hit CUDA OOM (attempt 1/3) — requeued with vram 12288 → 17420 MiB
```

Only genuine OOMs trigger this. A crash from a bug (any other nonzero exit)
fails immediately and burns no retries, so retries can't hide broken code.

**Timeout** — for jobs that are only ever legitimate for a bounded time:

```
[timeout=30m] python quick_eval.py --ckpt runs/lr1e4/last.pt
```

At the limit: SIGTERM, then SIGKILL. **Don't put timeouts on training runs
by habit** — a job with no timeout is never touched, no matter how long it
runs, and gpusched deliberately has no "this looks hung" guessing (any such
heuristic would also kill legitimate slow phases like preprocessing or
checkpointing). You declare which jobs are bounded; everything else is
trusted.

## 10. Multi-GPU jobs

```
[vram=20G gpus=2] torchrun --nproc_per_node=2 train_big.py
```

gpusched waits until **two** GPUs each have ≥ 20 GiB free, takes both, and
sets `CUDA_VISIBLE_DEVICES` to the pair. `vram` is per GPU, always.

## 11. When things crash (including gpusched itself)

The journal makes restarts boring, which is the goal.

**The scheduler died / you Ctrl-C'd / the box rebooted:** run the exact
same command again. Finished jobs are skipped, the rest continue:

```
$ gpusched jobs.txt --watch -v
job 1 already completed ok — skipping (journal): python train.py --lr 3e-4 ...
job 3 started on gpu [0] ...
```

If a job from the previous scheduler is *still alive* (jobs survive
scheduler death on purpose), gpusched will **not** start a second copy —
that would mean two trainers writing the same checkpoints. It tells you
instead:

```
WARN  job 5 is still running as an ORPHAN of a previous scheduler (pgid 41822) — not re-dispatching. Its exit code is unrecoverable; wait for it or kill it: kill -TERM -41822
```

Wait for it to finish (then mark it however you like) or kill it and let
gpusched re-run it cleanly.

**Re-run everything from scratch:** `gpusched jobs.txt --fresh` (wipes the
journal). **Re-run just one finished job:** change its line trivially —
add a space, a comment-like flag your script ignores, anything — and it's a
new job. (Yes, this is a hack; it's also two keystrokes.)

## 12. Reading the finish lines

```
job 2 finished [OK] in 1774s — peak vram gpu0:13911 MiB | declared 14336 MiB → within ±10% | avg gpu util 96%
```

Three diagnostics per job: the **peak** is your next declaration; the
**verdict** (`within` / `over-declared` / `UNDER-DECLARED`) tells you which
direction to adjust; the **util** percentage is the early-warning for
wasted money — a trainer showing 35% util is probably dataloader-bound, and
no scheduler fixes that. Caveats: util is measured per *device*, so it's
muddled when two jobs share a GPU; and in some Docker setups per-job memory
can't be attributed, in which case the peak honestly reads
`n/a (attribution failed)` rather than a made-up number.

## 13. The dials (you rarely need these)

| Flag | Default | When to touch it |
|---|---|---|
| `--poll 5` | 5 s | Lower for snappier dispatch/cancel and finer peak tracking; the cost is more `nvidia-smi` calls |
| `--gpus 0,1` | all | Leave some GPUs for interactive work |
| `--exclusive` | off | Never co-locate two of your jobs, period |
| `--margin 512` | 512 MiB | Extra safety pad on every declaration |
| `--tolerance 0.10` | ±10% | How wrong a declaration must be before it's flagged |
| `--spike-buffer 0.10` | +10% | Headroom held above the observed peaks of fluctuating processes |
| `--oom-retries N` | 0 | Default retries for every job (`retries=` per line overrides) |
| `--idle-threshold 200` | 200 MiB | What "idle" means for undeclared jobs (raise if a display server holds memory) |
| `--log-dir` | `gpusched_logs` | **One per experiment** — see mistake #4 |

## 14. What NOT to do — the common mistakes

**1. Deleting a running job's line to stop it.** Deleting does nothing to a
job that already started; it keeps burning GPU. Stopping a run requires the
explicit `cancel` token (section 8). This asymmetry is a safety feature,
not an oversight.

**2. `[vram=8]` when you meant 8 gigabytes.** Bare numbers are **MiB** —
that line declares 8 *mebibytes*, so the job gets packed absolutely
anywhere and immediately triggers an EXCEEDS warning. Write `8G`.

**3. Declaring nothing and wondering why GPUs sit half-empty.** Undeclared
jobs require a fully idle GPU to themselves — that's the safe default, and
it means an all-undeclared file serializes one job per GPU. Sharing is what
declarations buy you.

**4. Reusing one `--log-dir` across different experiments.** The journal
identifies jobs by their command text. If today's experiment contains a
command identical to one that finished last month *in the same log dir*,
it's "already done" and silently skipped. One log dir per experiment (or
`--fresh`) avoids ever hitting this.

**5. `source venv/bin/activate && python ...` in a job line.** Jobs run
under `/bin/sh` (dash on Ubuntu), which has no `source`. Use the absolute
interpreter path (`/path/.venv/bin/python train.py`) or `conda run -n env
python ...` — both also survive you changing your default env later.

**6. Ctrl-C "to leave it running overnight."** Ctrl-C *shuts gpusched
down and terminates its jobs* — intentionally, so nothing runs unsupervised
by accident. Leaving things running is tmux **detach** (`Ctrl-b d`).

**7. Editing a running job's line and expecting the change to apply.** A
job launched with the command it had at launch. Edits to running lines are
ignored — with exactly one exception, the `cancel` token.

**8. Low-balling a declaration to get scheduled sooner.** gpusched will
place neighbors trusting your number; when you blow through it, *they* OOM.
You get a warning, they get a crash. Declare honestly; the report makes
honest cheap.

**9. Packing two heavy trainers and expecting double throughput.**
Declarations make jobs *fit*; they don't make SMs bigger. Two
compute-saturated trainers on one GPU each run at roughly half speed. Check
the util numbers: pack high-VRAM/low-util jobs (eval, dataloader-bound,
inference) together; give saturating trainers their own card or use
`--exclusive`.

**10. Habitual `[timeout=...]` on real training runs.** Walltime is a
guillotine, not a watchdog — at 48h00m01s your 48h run is dead. Timeouts
are for jobs that are *only ever legitimate* for a bounded time.

**11. Two schedulers managing the same GPUs.** They don't coordinate; each
sees the other's jobs only as external load, and simultaneous dispatch can
race. One scheduler per machine; one jobs file is plenty (it's editable
live, after all).

**12. Expecting a cancel to land instantly.** Everything happens on the
poll cycle — default 5 s. Against an hours-long doomed run that's nothing;
if it ever matters, `--poll 1`.

**13. Removing `cancel` from a line to re-run the job.** Cancelled is
final, like any finished job. Re-run it the normal way: change the line
trivially, or `--fresh`.

## 15. Pocket reference

```
# line attributes (inside leading [...]):
vram=14G | vram=14336      max VRAM per GPU (G = GiB, bare = MiB)
gpus=2                     number of GPUs (each needs vram free)
retries=2                  auto-retry on CUDA OOM, declaration auto-raised
timeout=30m                walltime: s/m/h/d (opt-in only)
cancel                     stop if running / never start if pending

# the three files that matter:
jobs.txt                   yours; edited live; never written by gpusched
gpusched_logs/status.txt   the scoreboard:  watch -n2 cat ...
gpusched_logs/jobNNN_*.log each job's stdout/stderr

# everyday commands:
gpusched jobs.txt --watch -v          run + keep watching for new lines
gpusched jobs.txt --sim 2 --poll 0.5  rehearse anything with fake GPUs
gpusched jobs.txt --fresh             forget history, run everything again
```

When this tool stops being enough — multiple machines, metric-based early
stopping, DAG pipelines, hard isolation — the README's "If your needs are
bigger than this" table points to the right tools. gpusched's job is to
stay small enough that everything in this guide fits in your head.
