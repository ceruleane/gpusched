"""Pure GPU allocation logic.

Separated from the scheduler loop so placement rules are testable with no
subprocesses and no clock.

Semantics (v0.2: high-water-mark aware)
---------------------------------------
* A job **without** a vram declaration requires a *fully idle* GPU:
  effective external usage < idle_threshold and no scheduler job placed there.
* A job **with** a declaration of E MiB (per GPU) can be placed on any GPU
  whose *effective headroom* >= E + margin.
* Effective headroom = total - effective_external - sum(own effective budgets).
* A running job's effective budget per GPU is its declaration while its
  observed peak stays within it; once the peak exceeds the declaration, the
  declaration is no longer trusted and the budget escalates to
  ``peak * (1 + spike_buffer)``. A job that has not ramped yet (actual 0)
  still blocks its full budget — this closes the launch double-booking race.
* ``effective_external`` is supplied by the scheduler as
  ``max(instantaneous_external, sum(per-pid external peaks) * (1 + spike_buffer))``
  so that fluctuating external processes are held to their observed maxima,
  not their momentary troughs. When not supplied (unit tests / simple use),
  it falls back to instantaneous usage minus attributed own usage.
* A GPU hosting an *undeclared* scheduler job is never eligible for others.
* ``exclusive=True`` additionally forbids two scheduler jobs on one GPU,
  while still honoring headroom vs external processes.
* Multi-GPU jobs (n_gpus=N) need N distinct GPUs, each individually
  eligible; selection is best-fit (smallest sufficient headroom first) to
  preserve large contiguous headroom for big jobs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .backend import GpuSnapshot
from .jobspec import JobSpec


@dataclass(frozen=True)
class Occupant:
    """Lightweight view of a running scheduler job, for allocation purposes."""

    gpu_indices: tuple[int, ...]
    vram_mib: int | None                 # declared estimate (per GPU); None = undeclared
    actual_mib: dict[int, int]           # last attributed usage per GPU (MiB)
    peak_mib: dict[int, int] = field(default_factory=dict)  # observed max per GPU


@dataclass(frozen=True)
class AllocOptions:
    idle_threshold_mib: int = 200   # GPU counts as idle below this usage
    margin_mib: int = 512           # safety margin on top of declarations
    exclusive: bool = False         # one scheduler job per GPU
    spike_buffer: float = 0.10      # headroom buffer over observed maxima


def effective_budget(occ: Occupant, gpu: int, spike_buffer: float) -> int | None:
    """MiB to hold against `occ` on `gpu`. None = the whole device (undeclared)."""
    if occ.vram_mib is None:
        return None
    peak = occ.peak_mib.get(gpu, 0)
    if peak <= occ.vram_mib:
        budget = occ.vram_mib
    else:  # declaration violated -> trust the empirical max, buffered
        budget = math.ceil(round(peak * (1 + spike_buffer), 6))
    return max(budget, occ.actual_mib.get(gpu, 0))


def _fallback_external(gpu: int, snapshot: GpuSnapshot, occupants: list[Occupant]) -> int:
    own_actual = sum(occ.actual_mib.get(gpu, 0) for occ in occupants if gpu in occ.gpu_indices)
    return max(0, snapshot.gpus[gpu].used_mib - own_actual)


def effective_headroom(
    gpu: int,
    snapshot: GpuSnapshot,
    occupants: list[Occupant],
    opts: AllocOptions,
    external_mib: dict[int, int] | None = None,
) -> int | None:
    """Headroom available for a *new declared* job on `gpu`.

    Returns None if the GPU is off-limits (hosts an undeclared job).
    """
    stat = snapshot.gpus[gpu]
    own = 0
    for occ in occupants:
        if gpu not in occ.gpu_indices:
            continue
        budget = effective_budget(occ, gpu, opts.spike_buffer)
        if budget is None:
            return None  # undeclared job owns the whole device
        own += budget
    external = (
        external_mib[gpu] if external_mib is not None
        else _fallback_external(gpu, snapshot, occupants)
    )
    return stat.total_mib - external - own


def find_allocation(
    spec: JobSpec,
    snapshot: GpuSnapshot,
    occupants: list[Occupant],
    allowed_gpus: set[int],
    opts: AllocOptions,
    external_mib: dict[int, int] | None = None,
) -> list[int] | None:
    """Return the GPU indices to run `spec` on, or None if it cannot start now."""
    candidates = sorted(g for g in snapshot.gpus if g in allowed_gpus)
    scheduler_used = {g for occ in occupants for g in occ.gpu_indices}

    def external(g: int) -> int:
        return (
            external_mib[g] if external_mib is not None
            else _fallback_external(g, snapshot, occupants)
        )

    if spec.vram_mib is None:
        # Undeclared: fully idle GPUs only (judged on EFFECTIVE external usage,
        # so a live external process is held to its peak), lowest index first.
        eligible = [
            g for g in candidates
            if g not in scheduler_used and external(g) < opts.idle_threshold_mib
        ]
        return eligible[: spec.n_gpus] if len(eligible) >= spec.n_gpus else None

    need = spec.vram_mib + opts.margin_mib
    scored: list[tuple[int, int]] = []  # (headroom, gpu)
    for g in candidates:
        if opts.exclusive and g in scheduler_used:
            continue
        headroom = effective_headroom(g, snapshot, occupants, opts, external_mib)
        if headroom is not None and headroom >= need:
            scored.append((headroom, g))

    if len(scored) < spec.n_gpus:
        return None
    # Best-fit: smallest sufficient headroom first; tie-break on index.
    scored.sort()
    return sorted(g for _, g in scored[: spec.n_gpus])
