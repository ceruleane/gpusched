"""Simulated GPU job: `python -m gpusched.simjob --vram 8000 --ramp 1 --hold 2`.

Ramps fake per-GPU VRAM from 0 to --vram MiB over --ramp seconds, holds for
--hold seconds, then exits. Reports usage on every GPU in
CUDA_VISIBLE_DEVICES by writing $GPUSCHED_SIM_DIR/<pid>.json every 50 ms,
mirroring how a real process would appear to nvidia-smi.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .testing import SIM_DIR_ENV


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--vram", type=int, required=True, help="peak fake VRAM per GPU (MiB)")
    p.add_argument("--ramp", type=float, default=0.5, help="seconds to ramp 0 -> peak")
    p.add_argument("--hold", type=float, default=1.0, help="seconds to hold at peak")
    p.add_argument("--exit-code", type=int, default=0)
    p.add_argument("--oom-once", default=None, metavar="MARKER",
                   help="fail with a CUDA OOM message unless MARKER file exists "
                        "(created on the failing run) — for retry testing")
    args = p.parse_args(argv)

    if args.oom_once:
        marker = Path(args.oom_once)
        if not marker.exists():
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()
            print("RuntimeError: CUDA out of memory. Tried to allocate 2.50 GiB", flush=True)
            return 1

    gpus = [int(g) for g in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if g != ""]
    sim_dir = Path(os.environ.get(SIM_DIR_ENV, "/tmp/gpusched_sim"))
    sim_dir.mkdir(parents=True, exist_ok=True)
    state_file = sim_dir / f"{os.getpid()}.json"

    start = time.monotonic()
    try:
        while True:
            t = time.monotonic() - start
            if t >= args.ramp + args.hold:
                break
            frac = min(1.0, t / args.ramp) if args.ramp > 0 else 1.0
            usage = {str(g): int(args.vram * frac) for g in gpus}
            tmp = state_file.with_suffix(".tmp")
            tmp.write_text(json.dumps({"pid": os.getpid(), "usage": usage}))
            tmp.replace(state_file)  # atomic: backend never reads partial JSON
            time.sleep(0.05)
    finally:
        state_file.unlink(missing_ok=True)
    return args.exit_code


if __name__ == "__main__":
    sys.exit(main())
