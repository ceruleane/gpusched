# gpusched ML demo — parallel hyperparameter sweeps on 2× RTX 4090

A complete, self-contained example of using [gpusched](https://github.com/YOURUSER/gpusched)
to run **real ML training in parallel** across two GPUs: two different
architectures (a small **GPT** and a **DDPM**), each swept over a small
hyperparameter grid, packed onto your cards by declared VRAM.

Everything trains on **synthetic data generated on the fly** — no downloads,
no dataset paths, nothing written outside this folder. The models are small
enough to finish in minutes but are genuine training loops (real forward /
backward / optimizer steps, losses that actually decrease).

```
train_gpt.py      small causal transformer; learns to SORT integer sequences
train_ddpm.py     denoising diffusion model; learns a 2-D two-moons distribution
common.py         shared helpers (SIGTERM-safe checkpointing, metrics, VRAM)
gen_sweep.py      expands hyperparameter grids -> a gpusched jobs.txt
summarize.py      ranks finished runs into a per-architecture leaderboard
```

## 0. Prerequisites

- gpusched installed (`gpusched --version` works).
- A Python environment with **PyTorch + CUDA**. The demo never hard-codes it;
  you pass its absolute path to `gen_sweep.py`. If you don't have one:

```bash
# example: a dedicated venv on persistent disk
uv venv /workspace/envs/ml --python 3.11
/workspace/envs/ml/bin/python -m pip install torch    # CUDA build for your driver
```

Throughout, `PY` is that interpreter's absolute path, e.g.
`/workspace/envs/ml/bin/python`. **Never a bare `python3`** — under a `uv tool`
install the system Python won't have torch, and gpusched runs job lines under
`/bin/sh` where bare names resolve to the system Python (this is mistake #5 in
gpusched's USAGE.md).

```bash
export PY=/workspace/envs/ml/bin/python    # adjust to your env
```

## 1. Smoke-test one run first (always do this)

Before committing both GPUs to a sweep, prove one job works and see its real
VRAM. Run a single short GPT training directly:

```bash
$PY train_gpt.py --d-model 256 --n-layer 4 --batch-size 128 --epochs 3 --out /tmp/smoke_gpt
```

You should see the loss fall and `sort_acc` climb across 3 epochs, ending with
a `[done]` line. Do the same for DDPM (its `energy_dist` should fall):

```bash
$PY train_ddpm.py --hidden 256 --depth 4 --epochs 3 --out /tmp/smoke_ddpm
```

If both print decreasing metrics and exit cleanly, you're ready. (On CPU they
still run, just slowly — fine for a smoke test.)

## 2. Generate the sweep

```bash
python gen_sweep.py --python "$PY" --out-root runs --epochs 30 --retries 1
```

This writes `jobs.txt`: 8 GPT configs (sweeping `lr`, `d_model`, `n_layer`)
and 8 DDPM configs (sweeping `lr`, `hidden`, `depth`) — 16 jobs total. Each
line has a VRAM declaration, the absolute interpreter path, `retries=1` (auto-
retry once on CUDA OOM), and a unique `--out runs/<name>` directory. Inspect it:

```bash
cat jobs.txt
```

The declared VRAM values are deliberate **over-estimates** (GPT ≈ 4–8 GiB,
DDPM ≈ 2–3 GiB). Over-declaring is the safe direction — it wastes a little
packing headroom but never risks OOM-ing a neighbor. You'll tighten them in
step 5.

Variations:
```bash
python gen_sweep.py --python "$PY" --only gpt      # just the GPT sweep
python gen_sweep.py --python "$PY" --only ddpm     # just the DDPM sweep
python gen_sweep.py --python "$PY" --retries 0     # no OOM auto-retry
```

## 3. Run it across both GPUs

```bash
gpusched jobs.txt --gpus 0,1 --watch -v
```

gpusched packs as many jobs onto each 4090 as the declarations allow — on
24 GiB cards that's typically 3+ of these at once per GPU — and starts the
next pending job the instant one finishes. With 16 small jobs and ~5–8 GiB
each, expect several running concurrently per card.

To watch it unattended, use tmux (so it survives SSH disconnects):

```bash
tmux new -s sweep
gpusched jobs.txt --gpus 0,1 --watch -v        # pane 1
# Ctrl-b c, then in pane 2:
watch -n2 cat gpusched_logs/status.txt          # live scoreboard
# Ctrl-b d to detach; `tmux attach -t sweep` to return
nvidia-smi                                       # confirm both GPUs are busy
```

`--watch` keeps gpusched alive after the queue drains so you can append more
configs (step 6). Drop it if you just want it to exit when done.

## 4. While it runs

**Follow one job's training output:**
```bash
tail -f gpusched_logs/job003_gpu0.log
```

**Check which configs are where:** the scoreboard (`status.txt`) shows `▶`
running, `·` pending, `↻` retrying after OOM, `✓`/`✗` finished.

## 5. Calibrate the VRAM declarations (the real lesson)

Every finish line reports the measured peak and whether your declaration was
honest:

```
job 4 finished [OK] in 248s — peak vram gpu0:7120 MiB | declared 7936 MiB → within ±10% | avg gpu util 94%
job 9 finished [OK] in 143s — peak vram gpu1:1980 MiB | declared 2048 MiB → over-declared (-3%) ...
```

Read the peaks, then either edit `gen_sweep.py`'s `gpt_vram_mib` /
`ddpm_vram_mib` formulas to match reality, or just hand-edit the numbers in
`jobs.txt` for a rerun. Tighter (but still safe) declarations let gpusched pack
more jobs per GPU. If you ever see a `WARN ... EXCEEDS declared VRAM`, that
config under-declared — raise it; a co-located job could have OOM'd.

Also watch `avg gpu util`: these jobs are small, so two packed on one card may
each show lower util — that's the packing-vs-throughput tradeoff in action.
For tiny models like these, packing wins (they'd waste a whole card alone);
for full-size training that saturates the SMs, you'd give each its own card or
use `--exclusive`.

## 6. Add or cancel configs mid-run

**Add a config** without stopping anything — append a line (gpusched picks it
up within a poll):

```bash
echo "[vram=8704 retries=1] $PY train_gpt.py --lr 5e-4 --d-model 512 --n-layer 6 --batch-size 128 --epochs 30 --out runs/gpt_d512_l6_lr5e4" >> jobs.txt
```

**Cancel a config** that's clearly going nowhere — open `jobs.txt`, find its
line, and add the word `cancel` inside the brackets:

```
[vram=5120 retries=1 cancel] /workspace/envs/ml/bin/python train_gpt.py --lr 0.001 --d-model 256 --n-layer 6 ...
```

The training script catches SIGTERM and saves a final checkpoint before
exiting, and the GPU is immediately freed for the next pending config.
(Deleting the line does **not** stop a running job — that's the safety design;
cancelling requires the explicit word.)

## 7. Weights & Biases (optional, but how most devs actually work)

Everything above works with zero external services. To mirror metrics to
W&B — live loss curves, a sweep dashboard, run comparison — opt in.

**One-time setup** (in your torch env):
```bash
$PY -m pip install wandb
$PY -m wandb login            # paste your API key from wandb.ai/authorize
```

**Then generate the sweep with W&B forwarded to every job:**
```bash
python gen_sweep.py --python "$PY" --out-root runs --wandb online \
    --wandb-project gpt-vs-ddpm-sweep
gpusched jobs.txt --gpus 0,1 --watch -v
```

Each job becomes its own W&B run, **grouped by architecture** (`gpt` / `ddpm`)
so the dashboard compares a whole sweep at a glance, tagged with its model,
and annotated with the physical `CUDA_VISIBLE_DEVICES` gpusched assigned it
(handy for spotting a slow card across the sweep). Per-epoch metrics stream
live; the final accuracy/energy-distance and `status` (completed vs cancelled)
land in each run's summary. When you cancel a config (step 6), its W&B run is
finalized with a nonzero exit code so it's filterable.

Three modes:
- `--wandb online` — sync live to wandb.ai (needs the API key above).
- `--wandb offline` — log to a local `./wandb/` dir on the box; sync later
  with `wandb sync ./wandb/offline-run-*`. Best when the GPU box has no
  outbound internet (common on locked-down clusters).
- `--wandb off` — the default; no W&B at all.

The integration is deliberately **fail-safe**: if wandb isn't installed, the
key is missing, or wandb errors at init, the job prints one line and trains
anyway. A logging service can never take down a run gpusched is managing. And
the local `metrics.jsonl` / `summary.json` are written regardless, so
`summarize.py` works with or without W&B.

A note on what W&B and gpusched each see: gpusched monitors **VRAM and
scheduling** from the outside (via nvidia-smi) and reports per-GPU peaks in
its own logs; W&B logs **your model's metrics** from the inside. They're
complementary and never conflict — gpusched never reads W&B, and W&B never
touches the scheduler.

## 8. Summarize results

Once runs finish (or any time — it reads whatever's done):

```bash
python summarize.py --out-root runs
```

```
=== GPT sweep — ranked by sort accuracy (higher is better) ===
run                              acc     status  epochs   peakMiB    sec
gpt_d384_l6_lr0.001            0.970  completed      30      7100    265
gpt_d256_l4_lr0.0003           0.910  completed      30      3900    182
...

=== DDPM sweep — ranked by energy distance (lower is better) ===
run                            enDist     status  epochs   peakMiB    sec
ddpm_h512_d6_lr0.001           0.0410  completed      30      2100    143
...
```

Each run dir also has `config.json`, `metrics.jsonl` (per-epoch), `summary.json`,
and `ckpt.pt` if you want to dig deeper or resume.

## 9. If something crashes

gpusched journals progress, so just **re-run the same command** — finished
configs are skipped, the rest continue. To force a clean rerun of everything,
add `--fresh`. To rerun one finished config, change its line trivially (a new
command is a new job). See gpusched's USAGE.md §11 for the full story,
including the orphan guard that stops a restarted scheduler from launching a
second copy of a still-running job.

## Common mistakes (demo-specific)

1. **Bare `python3` in the sweep** — always pass `--python "$PY"` with the
   absolute path to your torch env. The generated lines bake it in.
2. **Reusing `--out-root runs` across different sweeps in the same log dir** —
   gpusched identifies jobs by command text, so an identical command from a
   previous sweep is "already done" and skipped. Use a fresh `--out-root` (and
   `--log-dir`) per experiment, or `--fresh`.
3. **Trusting the default VRAM numbers as exact** — they're ceilings to start
   from, not measurements. Calibrate from the finish lines (step 5).
4. **Expecting these tiny models to saturate a 4090** — they won't; that's why
   packing several per card is the right move here. Don't read low util as a
   bug.

When you outgrow this — real datasets, metric-based early stopping, multi-node
— see the "If your needs are bigger than this" table in gpusched's README.
This demo's job is to show the scheduling mechanics on something real but small.

## Exercising gpusched's safety features (optional)

The normal sweep has flat, well-declared memory — which means it never
actually triggers gpusched's spike-buffer or OOM-retry. To see those fire, add
`--stress` when generating:

```bash
python gen_sweep.py --python "$PY" --stress
gpusched jobs.txt --gpus 0,1 --watch -v
```

This appends two configs:

- **`stress_spike`** trains with a periodic VRAM spike (`--spike-every 3
  --spike-mib 2048`): every 3 epochs it transiently allocates a 2 GiB scratch
  tensor, mimicking a job whose memory fluctuates. It declares enough to cover
  the spike, so it runs fine — but if you watch `nvidia-smi` you'll see the
  sawtooth, and this is the pattern gpusched's `--spike-buffer` is designed to
  absorb when deciding whether to pack a neighbor (it holds fluctuating
  processes to their observed *peak*, not their trough, so a momentary dip
  doesn't trick it into double-booking).
- **`stress_underdeclared`** declares only 2048 MiB on a model that needs far
  more. Watch gpusched's log: you'll get a live
  `WARN ... EXCEEDS declared VRAM ... neighbors may OOM` the moment it blows
  past 2 GiB, and because the line carries `retries=1`, if it actually hits a
  CUDA OOM gpusched requeues it with the declaration auto-bumped to ~1.25× the
  observed peak (`↻ ... requeued with vram 2048 → NNNN MiB`). This is the
  honest demonstration of why you declare truthfully — and how gpusched copes
  when you don't.

(Real configs also carry things like gradient accumulation and mixed
precision; those are ordinary script flags you'd add to the sweep grids in
`gen_sweep.py` — they don't change anything about how gpusched schedules, so
they're left out of the demo to keep the focus on the scheduler.)
