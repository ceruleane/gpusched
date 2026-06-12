"""gpusched — VRAM-aware single-node GPU job scheduler."""

__version__ = "0.4.0"

from .allocation import AllocOptions
from .backend import NvidiaSmiBackend
from .jobspec import JobSpec, parse_jobs_file
from .scheduler import Scheduler, SchedulerOptions

__all__ = [
    "AllocOptions", "JobSpec", "NvidiaSmiBackend",
    "Scheduler", "SchedulerOptions", "parse_jobs_file", "__version__",
]
