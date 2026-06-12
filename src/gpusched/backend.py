"""GPU state backends.

The scheduler only ever sees a :class:`GpuSnapshot`; how it is produced is
behind the :class:`GpuBackend` protocol. The real backend shells out to
``nvidia-smi``; test backends fabricate snapshots (see ``gpusched.testing``).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class GpuStat:
    index: int
    total_mib: int
    used_mib: int
    util_pct: int | None = None     # device-level SM utilization, if known

    @property
    def free_mib(self) -> int:
        return self.total_mib - self.used_mib


@dataclass(frozen=True)
class ProcStat:
    gpu_index: int
    pid: int
    used_mib: int


@dataclass
class GpuSnapshot:
    gpus: dict[int, GpuStat] = field(default_factory=dict)
    procs: list[ProcStat] = field(default_factory=list)


class GpuBackend(Protocol):
    def snapshot(self) -> GpuSnapshot: ...


class BackendError(RuntimeError):
    pass


def pgid_of(pid: int) -> int | None:
    """Process-group id of *pid* via /proc (Linux). None if the process is gone.

    Used to attribute GPU compute processes to scheduler-launched jobs: each
    job is started in its own session (``os.setsid``), so every descendant —
    including those spawned through ``sh -c`` — shares the job's pgid.
    """
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            data = f.read().decode("utf-8", "replace")
        # /proc/[pid]/stat: "pid (comm) state ppid pgrp session ...".
        # comm may itself contain spaces/parens, so split after the LAST ')':
        after_comm = data.rsplit(")", 1)[1].split()
        return int(after_comm[2])  # [0]=state, [1]=ppid, [2]=pgrp
    except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError, ValueError):
        return None


class NvidiaSmiBackend:
    """Real backend: two nvidia-smi queries per snapshot.

    * ``--query-gpu=index,uuid,memory.total,memory.used`` for GPU-level stats
    * ``--query-compute-apps=gpu_uuid,pid,used_memory`` for per-process stats
    """

    def __init__(self, nvidia_smi: str = "nvidia-smi", timeout: float = 10.0):
        self._bin = nvidia_smi
        self._timeout = timeout

    def _run(self, args: list[str]) -> str:
        try:
            return subprocess.check_output(
                [self._bin, *args], text=True, timeout=self._timeout,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as e:
            raise BackendError(f"{self._bin} not found — is the NVIDIA driver installed?") from e
        except subprocess.CalledProcessError as e:
            raise BackendError(f"{self._bin} failed: {e.stderr or e}") from e
        except subprocess.TimeoutExpired as e:
            raise BackendError(f"{self._bin} timed out after {self._timeout}s") from e

    def snapshot(self) -> GpuSnapshot:
        snap = GpuSnapshot()
        uuid_to_index: dict[str, int] = {}

        out = self._run([
            "--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 5:
                continue
            idx, uuid, total, used, util = parts
            try:
                util_pct: int | None = int(util)
            except ValueError:
                util_pct = None  # "[N/A]" on some virtualized setups
            gpu = GpuStat(index=int(idx), total_mib=int(total),
                          used_mib=int(used), util_pct=util_pct)
            snap.gpus[gpu.index] = gpu
            uuid_to_index[uuid] = gpu.index

        out = self._run([
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ])
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3 or parts[0] not in uuid_to_index:
                continue
            try:
                snap.procs.append(ProcStat(
                    gpu_index=uuid_to_index[parts[0]],
                    pid=int(parts[1]),
                    used_mib=int(parts[2]),
                ))
            except ValueError:
                # used_memory can be "[N/A]" (e.g. inside some containers /
                # for graphics processes); skip — attribution degrades to n/a.
                continue
        return snap
