"""Test and simulation backends (no GPU required).

* :class:`FakeBackend` — fully scripted snapshots for unit tests.
* :class:`SimBackend` — integration/demo backend: simulated jobs (see
  ``gpusched.simjob``) write their fake VRAM usage to ``$GPUSCHED_SIM_DIR``;
  the backend assembles snapshots from those files. Because the reported
  PIDs are real child processes of the scheduler, this exercises the exact
  pgid-attribution path used in production.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .backend import GpuSnapshot, GpuStat, ProcStat

SIM_DIR_ENV = "GPUSCHED_SIM_DIR"


class FakeBackend:
    def __init__(self, gpus: dict[int, tuple[int, int]]):
        """gpus: {index: (total_mib, used_mib)}"""
        self.gpus = dict(gpus)
        self.procs: list[tuple[int, int, int]] = []  # (gpu, pid, mib)
        self.util: dict[int, int] = {}               # optional per-gpu util %

    def set_used(self, gpu: int, used_mib: int) -> None:
        total, _ = self.gpus[gpu]
        self.gpus[gpu] = (total, used_mib)

    def snapshot(self) -> GpuSnapshot:
        snap = GpuSnapshot()
        for idx, (total, used) in self.gpus.items():
            snap.gpus[idx] = GpuStat(index=idx, total_mib=total, used_mib=used,
                                     util_pct=self.util.get(idx))
        snap.procs = [ProcStat(gpu_index=g, pid=p, used_mib=m) for g, p, m in self.procs]
        return snap


class SimBackend:
    """Snapshot = configured externals + usage self-reported by sim jobs."""

    def __init__(
        self,
        n_gpus: int = 2,
        total_mib: int = 24_000,
        external_mib: dict[int, int] | None = None,
        sim_dir: str | None = None,
    ):
        self.totals = {i: total_mib for i in range(n_gpus)}
        self.external = dict(external_mib or {})
        self.sim_dir = Path(sim_dir or os.environ.get(SIM_DIR_ENV, "/tmp/gpusched_sim"))
        self.sim_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    def snapshot(self) -> GpuSnapshot:
        snap = GpuSnapshot()
        used = {i: self.external.get(i, 0) for i in self.totals}
        for f in sorted(self.sim_dir.glob("*.json")):
            try:
                rec = json.loads(f.read_text())
                pid = int(rec["pid"])
            except (ValueError, KeyError, OSError):
                continue
            if not self._alive(pid):
                f.unlink(missing_ok=True)
                continue
            for gpu_s, mib in rec.get("usage", {}).items():
                gpu = int(gpu_s)
                if gpu in used:
                    used[gpu] += int(mib)
                    snap.procs.append(ProcStat(gpu_index=gpu, pid=pid, used_mib=int(mib)))
        busy = {p.gpu_index for p in snap.procs}
        for i, total in self.totals.items():
            snap.gpus[i] = GpuStat(index=i, total_mib=total, used_mib=min(used[i], total),
                                   util_pct=90 if i in busy else 0)
        return snap
