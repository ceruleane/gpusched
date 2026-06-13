"""Generate a gpusched jobs.txt from hyperparameter sweeps.

Defines two sweeps — a GPT sweep and a DDPM sweep — as small grids, expands
them into one command per config, and writes a jobs.txt that gpusched can run.
Each line carries:

  * a per-job VRAM declaration tuned to the config (so gpusched packs them),
  * an absolute interpreter path (USAGE.md mistake #5: never a bare `python3`),
  * a unique --out directory so runs don't clobber each other.

Run:
    python gen_sweep.py --python /abs/path/to/python --out-root runs
    # then: gpusched jobs.txt --gpus 0,1 --watch -v

The VRAM numbers are deliberate over-estimates of measured peaks; calibrate
them from the finish-line reports after your first run (USAGE.md §5).
"""

from __future__ import annotations

import argparse
import itertools
import shlex
from pathlib import Path

# --------------------------------------------------------------------------- #
# VRAM declaration model.
#
# These are rough ceilings, NOT measured truth. The point of the demo is that
# you run once, read the peak vram in each finish line, and tighten these. The
# formulas below scale with the knobs that drive memory so the relative sizes
# are sensible out of the box on a 24 GiB card.
# --------------------------------------------------------------------------- #
def gpt_vram_mib(d_model: int, n_layer: int, batch: int) -> int:
    # Conservative CEILING, intentionally high. Attention activations grow fast
    # with width and batch; under-declaring risks OOM-ing a co-located job, so
    # we err high and let you calibrate DOWN from the measured peak (README §4).
    base = 1500
    est = base + (d_model / 256) ** 2 * n_layer * 380 + batch * 9
    return _round_up(est)


def ddpm_vram_mib(hidden: int, depth: int, batch: int) -> int:
    # The 2-D MLP is tiny; even a generous ceiling stays small.
    base = 900
    est = base + (hidden / 256) * depth * 130 + batch * 1.5
    return _round_up(est)


def _round_up(mib: float, step: int = 256) -> int:
    import math
    return int(math.ceil(mib / step) * step)


# --------------------------------------------------------------------------- #
# The sweeps. Edit these grids to taste; everything downstream is generic.
# --------------------------------------------------------------------------- #
def gpt_sweep():
    grid = {
        "lr": [3e-4, 1e-3],
        "d_model": [256, 384],
        "n_layer": [4, 6],
        "batch_size": [128],
    }
    for combo in _product(grid):
        vram = gpt_vram_mib(combo["d_model"], combo["n_layer"], combo["batch_size"])
        name = f"gpt_d{combo['d_model']}_l{combo['n_layer']}_lr{combo['lr']:g}"
        yield "train_gpt.py", combo, vram, name


def ddpm_sweep():
    grid = {
        "lr": [1e-3, 3e-3],
        "hidden": [256, 512],
        "depth": [4, 6],
        "batch_size": [256],
    }
    for combo in _product(grid):
        vram = ddpm_vram_mib(combo["hidden"], combo["depth"], combo["batch_size"])
        name = f"ddpm_h{combo['hidden']}_d{combo['depth']}_lr{combo['lr']:g}"
        yield "train_ddpm.py", combo, vram, name


def _product(grid: dict):
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


def stress_sweep():
    """Two configs that exercise gpusched features the normal sweep doesn't:
    a fluctuating-VRAM job (periodic spike) and an intentionally under-declared
    job that will trip the EXCEEDS warning (and, with retries, the auto-bump)."""
    # Fluctuating VRAM: declare for the spike so it's honest, watch the troughs.
    spike_combo = {"lr": 3e-4, "d_model": 256, "n_layer": 4, "batch_size": 128}
    spike_vram = gpt_vram_mib(256, 4, 128) + 2048  # room for the 2 GiB spike
    yield ("train_gpt.py", spike_combo, spike_vram, "stress_spike",
           "--spike-every 3 --spike-mib 2048")
    # Intentionally LOW declaration on a big model: this will exceed its
    # declaration at runtime -> EXCEEDS warning; with retries it auto-bumps.
    under_combo = {"lr": 3e-4, "d_model": 384, "n_layer": 6, "batch_size": 128}
    yield ("train_gpt.py", under_combo, 2048, "stress_underdeclared", "")


# --------------------------------------------------------------------------- #
# Emit jobs.txt
# --------------------------------------------------------------------------- #
def build_lines(python: str, out_root: str, epochs: int, retries: int,
                include_gpt: bool, include_ddpm: bool,
                wandb_mode: str = "off", wandb_project: str = "gpusched-demo",
                include_stress: bool = False) -> list[str]:
    lines: list[str] = []
    sweeps = []
    if include_gpt:
        sweeps.append(("GPT sweep", ((s, c, v, n, "") for s, c, v, n in gpt_sweep())))
    if include_ddpm:
        sweeps.append(("DDPM sweep", ((s, c, v, n, "") for s, c, v, n in ddpm_sweep())))
    if include_stress:
        sweeps.append(("Stress configs (exercise spike-buffer + OOM-retry)", stress_sweep()))

    wandb_suffix = ""
    if wandb_mode != "off":
        wandb_suffix = f" --wandb {wandb_mode} --wandb-project {shlex.quote(wandb_project)}"

    for title, sweep in sweeps:
        lines.append(f"# ===== {title} =====")
        for script, combo, vram, name, extra in sweep:
            out_dir = str(Path(out_root) / name)
            args = " ".join(f"--{k.replace('_', '-')} {v:g}" if isinstance(v, float)
                            else f"--{k.replace('_', '-')} {v}"
                            for k, v in combo.items())
            attrs = f"vram={vram}"
            if retries > 0:
                attrs += f" retries={retries}"
            extra_suffix = f" {extra}" if extra else ""
            cmd = (f"[{attrs}] {shlex.quote(python)} {script} "
                   f"{args} --epochs {epochs} --out {shlex.quote(out_dir)}"
                   f"{extra_suffix}{wandb_suffix}")
            lines.append(cmd)
        lines.append("")
    return lines


def main():
    p = argparse.ArgumentParser(description="Generate gpusched jobs.txt for the ML demo")
    p.add_argument("--python", required=True,
                   help="ABSOLUTE path to the interpreter that has torch "
                        "(e.g. /workspace/envs/ml/bin/python). NOT a bare 'python3'.")
    p.add_argument("--out-root", default="runs", help="root dir for per-run outputs")
    p.add_argument("--jobs-file", default="jobs.txt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--retries", type=int, default=1,
                   help="OOM auto-retries per job (0 to disable)")
    p.add_argument("--only", choices=["gpt", "ddpm"], default=None,
                   help="emit only one architecture's sweep")
    p.add_argument("--wandb", choices=["off", "online", "offline"], default="off",
                   help="forward W&B logging to every job (off by default)")
    p.add_argument("--wandb-project", default="gpusched-demo")
    p.add_argument("--stress", action="store_true",
                   help="append two stress configs that exercise gpusched's "
                        "spike-buffer (fluctuating VRAM) and OOM-retry (intentional "
                        "under-declaration) features")
    args = p.parse_args()

    include_gpt = args.only in (None, "gpt")
    include_ddpm = args.only in (None, "ddpm")

    if not args.python.strip():
        p.error("--python is empty. Pass the ABSOLUTE path to your torch "
                "interpreter, e.g. --python /workspace/envs/ml/bin/python")
    if not args.python.startswith("/"):
        p.error(f"--python must be an absolute path (got {args.python!r}). "
                "A bare name like 'python3' resolves to the system Python "
                "under /bin/sh and won't have torch (USAGE.md mistake #5).")

    lines = build_lines(args.python, args.out_root, args.epochs, args.retries,
                        include_gpt, include_ddpm, args.wandb, args.wandb_project,
                        include_stress=args.stress)
    Path(args.jobs_file).write_text("\n".join(lines) + "\n")

    n_jobs = sum(1 for ln in lines if ln and not ln.startswith("#"))
    print(f"Wrote {n_jobs} jobs to {args.jobs_file}")
    print(f"Run with:  gpusched {args.jobs_file} --gpus 0,1 --watch -v")


if __name__ == "__main__":
    main()
