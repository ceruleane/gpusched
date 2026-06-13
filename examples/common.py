"""Shared helpers for the gpusched ML demo trainers.

Both train_gpt.py and train_ddpm.py use these so the two scripts stay focused
on their model. Nothing here is gpusched-specific except `install_sigterm_handler`,
which makes a job checkpoint cleanly when gpusched cancels it (USAGE.md §8).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path

import torch


# --------------------------------------------------------------------------- #
# Graceful cancellation: gpusched sends SIGTERM, then SIGKILL after a grace
# period. We flip a flag so the training loop can finish the current step,
# save a checkpoint, and exit cleanly instead of dying mid-backward.
# --------------------------------------------------------------------------- #
class StopFlag:
    def __init__(self) -> None:
        self.stop = False

    def install(self) -> "StopFlag":
        def handler(signum, frame):
            self.stop = True
            print(f"[signal] received {signal.Signals(signum).name} "
                  f"— will checkpoint and exit after this step", flush=True)
        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)
        return self


def install_sigterm_handler() -> StopFlag:
    return StopFlag().install()


# --------------------------------------------------------------------------- #
# Device / VRAM
# --------------------------------------------------------------------------- #
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def vram_mb(device: torch.device) -> float:
    """Peak allocated VRAM in MiB (CUDA only); 0.0 on CPU."""
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 * 1024)


def reset_vram_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


# --------------------------------------------------------------------------- #
# Output: per-epoch metrics line + final summary JSON, plus OPTIONAL wandb.
#
# wandb is strictly additive: if it isn't installed, or --wandb isn't passed,
# or no API key is configured, the logger silently falls back to stdout+JSONL
# so the demo still works fully offline. gpusched's own monitoring reads the
# stdout lines and nvidia-smi, never wandb, so the two never conflict.
# --------------------------------------------------------------------------- #
class RunLogger:
    """Writes a human-readable line per epoch to stdout (captured by gpusched
    into gpusched_logs/jobNNN_*.log) and a machine-readable metrics.jsonl +
    final summary.json into the run's output directory. Optionally mirrors
    everything to Weights & Biases."""

    def __init__(self, out_dir: str, config: dict,
                 wandb_mode: str = "off", wandb_project: str = "gpusched-demo",
                 run_name: str | None = None) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.t0 = time.time()
        self.metrics_path = self.out_dir / "metrics.jsonl"
        (self.out_dir / "config.json").write_text(json.dumps(config, indent=2))
        # Truncate any prior metrics so a resumed/rerun job starts clean.
        self.metrics_path.write_text("")
        self._wandb = self._init_wandb(wandb_mode, wandb_project, run_name)

    def _init_wandb(self, mode: str, project: str, run_name: str | None):
        """Returns a live wandb run, or None if wandb is off/unavailable.

        mode: 'off' (default, no wandb), 'online' (sync to wandb.ai; needs an
        API key via `wandb login` or WANDB_API_KEY), or 'offline' (log to a
        local ./wandb dir, sync later with `wandb sync`)."""
        if mode == "off":
            return None
        try:
            import wandb
        except ImportError:
            print("[wandb] not installed (pip install wandb) — continuing without it", flush=True)
            return None
        try:
            run = wandb.init(
                project=project,
                name=run_name,
                config=self.config,
                mode=mode,                       # 'online' or 'offline'
                # group runs of the same architecture so the wandb UI can
                # compare a whole sweep at a glance:
                group=self.config.get("model", "run"),
                tags=[self.config.get("model", "run")],
                # GPU the scheduler pinned us to (CUDA renumbers to 0, but the
                # physical index is in the env gpusched set) — handy for spotting
                # a bad card across a sweep:
                notes=f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '?')}",
            )
            print(f"[wandb] logging to {run.url if mode == 'online' else 'local ./wandb'}", flush=True)
            return run
        except Exception as e:  # never let a wandb hiccup kill a training run
            print(f"[wandb] init failed ({e}) — continuing without it", flush=True)
            return None

    def log_epoch(self, epoch: int, **metrics) -> None:
        rec = {"epoch": epoch, "elapsed_s": round(time.time() - self.t0, 1), **metrics}
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        pretty = "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in metrics.items())
        print(f"[epoch {epoch:>4}] {pretty}  ({rec['elapsed_s']:.0f}s)", flush=True)
        if self._wandb is not None:
            self._wandb.log({**metrics, "elapsed_s": rec["elapsed_s"]}, step=epoch)

    def finalize(self, status: str, **summary) -> None:
        out = {
            "status": status,
            "config": self.config,
            "elapsed_s": round(time.time() - self.t0, 1),
            **summary,
        }
        (self.out_dir / "summary.json").write_text(json.dumps(out, indent=2))
        print(f"[done] status={status}  " +
              "  ".join(f"{k}={v}" for k, v in summary.items()), flush=True)
        if self._wandb is not None:
            # summary panel + a status tag you can filter on (completed/cancelled)
            for k, v in summary.items():
                self._wandb.summary[k] = v
            self._wandb.summary["status"] = status
            self._wandb.finish(exit_code=0 if status == "completed" else 1)


def save_checkpoint(path: str, model, optimizer, epoch: int, extra: dict | None = None) -> None:
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        **(extra or {}),
    }
    tmp = str(path) + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic: a kill mid-save can't corrupt the checkpoint


def common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", required=True, help="output directory for this run")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    # Optional Weights & Biases logging (off by default; demo works without it).
    parser.add_argument("--wandb", choices=["off", "online", "offline"], default="off",
                        help="off (default), online (sync to wandb.ai; needs API key), "
                             "or offline (local ./wandb, sync later)")
    parser.add_argument("--wandb-project", default="gpusched-demo")


def make_logger(args, config: dict, run_name: str) -> "RunLogger":
    """Build a RunLogger wired to the run's wandb settings from common_args."""
    return RunLogger(args.out, config, wandb_mode=args.wandb,
                     wandb_project=args.wandb_project, run_name=run_name)
