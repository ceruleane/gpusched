"""Job specification parsing.

Jobs-file syntax (one job per line):

    # comment / blank lines are skipped
    python train.py --config a.yaml          # no estimate -> needs an idle GPU
    [vram=18000] python train.py             # declares max 18000 MiB
    [vram=22G] bash run_eval.sh              # G / GiB suffix accepted
    [vram=30G gpus=2] torchrun train_big.py  # 2 GPUs, each with >= 30 GiB free
    [timeout=2h] python flaky_eval.py        # SIGTERM after 2 hours (opt-in only)
    [vram=8G retries=2] python sweep.py      # auto-retry on CUDA OOM, up to 2x

Attribute block must be a single leading ``[key=value ...]`` group.
``vram`` is interpreted **per GPU** for multi-GPU jobs. ``timeout`` accepts
s/m/h/d suffixes (default seconds). Jobs are identified by a hash of their
command text (plus an occurrence counter for duplicate lines), which is what
makes live-edited queue files and resume-after-restart well defined.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_ATTR_BLOCK = re.compile(r"^\[(?P<attrs>[^\]]*)\]\s*(?P<cmd>.*)$", re.DOTALL)
_VRAM_VALUE = re.compile(r"^(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>g|gb|gib|m|mb|mib)?$", re.IGNORECASE)
_DURATION = re.compile(r"^(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>s|m|h|d)?$", re.IGNORECASE)


class JobSpecError(ValueError):
    """Raised on malformed jobs-file lines; carries the 1-based line number."""

    def __init__(self, lineno: int, message: str):
        super().__init__(f"jobs file line {lineno}: {message}")
        self.lineno = lineno


@dataclass(frozen=True)
class JobSpec:
    """A single queued job."""

    index: int                      # 1-based position in the jobs file
    command: str                    # shell command to execute
    vram_mib: int | None = None     # declared max VRAM per GPU (MiB); None = undeclared
    n_gpus: int = 1                 # number of GPUs required
    timeout_s: float | None = None  # walltime before SIGTERM; None = never (opt-in)
    retries: int = 0                # auto-retries on CUDA OOM
    key: str = field(default="", compare=False)   # stable identity (hash#occurrence)
    lineno: int = field(default=0, compare=False)

    @property
    def label(self) -> str:
        return f"job {self.index}"


def parse_vram(value: str, lineno: int = 0) -> int:
    """Parse a vram value like '12000', '12.5G', '24GiB', '8000MiB' -> MiB."""
    m = _VRAM_VALUE.match(value.strip())
    if not m:
        raise JobSpecError(lineno, f"cannot parse vram value {value!r} (use MiB or a G/GiB suffix)")
    num = float(m.group("num"))
    unit = (m.group("unit") or "m").lower()
    mib = num * 1024 if unit.startswith("g") else num
    mib_int = int(round(mib))
    if mib_int <= 0:
        raise JobSpecError(lineno, f"vram must be positive, got {value!r}")
    return mib_int


def parse_duration(value: str, lineno: int = 0) -> float:
    """Parse '90', '90s', '15m', '2h', '1.5d' -> seconds."""
    m = _DURATION.match(value.strip())
    if not m:
        raise JobSpecError(lineno, f"cannot parse duration {value!r} (use s/m/h/d)")
    mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}[(m.group("unit") or "s").lower()]
    sec = float(m.group("num")) * mult
    if sec <= 0:
        raise JobSpecError(lineno, f"timeout must be positive, got {value!r}")
    return sec


def job_key(command: str, occurrence: int) -> str:
    """Stable identity for a command line; occurrence disambiguates duplicates."""
    import hashlib
    h = hashlib.sha1(command.strip().encode()).hexdigest()[:12]
    return f"{h}#{occurrence}"


def parse_line(line: str, lineno: int, index: int) -> JobSpec | None:
    """Parse one jobs-file line. Returns None for blanks/comments."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    vram_mib: int | None = None
    n_gpus = 1
    timeout_s: float | None = None
    retries = 0
    command = stripped

    if stripped.startswith("[") and not _ATTR_BLOCK.match(stripped):
        # A line beginning with '[' but never closed is almost certainly a
        # torn mid-edit attribute block — refuse to execute it as shell.
        raise JobSpecError(lineno, "unterminated '[...]' attribute block")

    m = _ATTR_BLOCK.match(stripped)
    if m:
        command = m.group("cmd").strip()
        if not command:
            raise JobSpecError(lineno, "attribute block present but command is empty")
        for token in m.group("attrs").split():
            if "=" not in token:
                raise JobSpecError(lineno, f"malformed attribute {token!r} (expected key=value)")
            key, _, value = token.partition("=")
            key = key.lower()
            if key == "vram":
                vram_mib = parse_vram(value, lineno)
            elif key == "timeout":
                timeout_s = parse_duration(value, lineno)
            elif key == "retries":
                try:
                    retries = int(value)
                except ValueError:
                    raise JobSpecError(lineno, f"retries must be an integer, got {value!r}") from None
                if retries < 0:
                    raise JobSpecError(lineno, f"retries must be >= 0, got {retries}")
            elif key == "gpus":
                try:
                    n_gpus = int(value)
                except ValueError:
                    raise JobSpecError(lineno, f"gpus must be an integer, got {value!r}") from None
                if n_gpus < 1:
                    raise JobSpecError(lineno, f"gpus must be >= 1, got {n_gpus}")
            else:
                raise JobSpecError(lineno, f"unknown attribute {key!r} (known: vram, gpus, timeout, retries)")

    return JobSpec(index=index, command=command, vram_mib=vram_mib, n_gpus=n_gpus,
                   timeout_s=timeout_s, retries=retries, lineno=lineno)


def assign_keys(specs: list[JobSpec]) -> list[JobSpec]:
    """Attach stable identity keys (command hash + occurrence index)."""
    from dataclasses import replace
    seen: dict[str, int] = {}
    out = []
    for spec in specs:
        occ = seen.get(spec.command, 0)
        seen[spec.command] = occ + 1
        out.append(replace(spec, key=job_key(spec.command, occ)))
    return out


def parse_jobs_file(path: str) -> list[JobSpec]:
    """Parse a jobs file into an ordered list of JobSpecs with identity keys."""
    specs: list[JobSpec] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            spec = parse_line(line, lineno, index=len(specs) + 1)
            if spec is not None:
                specs.append(spec)
    return assign_keys(specs)
