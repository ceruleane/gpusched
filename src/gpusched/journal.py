"""Append-only job journal (JSONL).

One file per log directory. Each line is an event:

    {"key": "<hash#occ>", "event": "seen",   "no": 7, "command": "..."}
    {"key": "...",        "event": "oom",    "attempts": 1, "next_vram": 9750}
    {"key": "...",        "event": "done",   "status": "ok", "returncode": 0,
     "peak_mib": {"0": 7800}, "avg_util": 84.0}

Folding the events yields per-key state. Terminal statuses are never
re-dispatched; non-terminal keys remain eligible (this is what makes both
``--resume``-style restarts and OOM retries fall out of one mechanism).
The jobs file itself is never written by the scheduler — it stays entirely
user-owned, and this journal is the scheduler's only persistent state.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

TERMINAL = {"ok", "failed", "timeout", "infeasible", "failed_oom"}


@dataclass
class JobState:
    no: int | None = None           # persistent display number (first-seen order)
    command: str = ""
    attempts: int = 0               # completed attempts so far
    status: str = "pending"         # pending | running | <terminal>
    returncode: int | None = None
    next_vram: int | None = None    # bumped declaration for the next attempt
    peak_mib: dict[str, int] = field(default_factory=dict)

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL


class Journal:
    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.states: dict[str, JobState] = {}
        self._next_no = 1
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    try:
                        self._fold(json.loads(line))
                    except (json.JSONDecodeError, KeyError):
                        continue  # tolerate a torn last line after a crash

    def _fold(self, ev: dict) -> None:
        st = self.states.setdefault(ev["key"], JobState())
        kind = ev.get("event")
        if kind == "seen":
            st.no = ev.get("no")
            st.command = ev.get("command", "")
            self._next_no = max(self._next_no, (st.no or 0) + 1)
        elif kind == "oom":
            st.attempts = ev.get("attempts", st.attempts + 1)
            st.next_vram = ev.get("next_vram", st.next_vram)
            st.status = "pending"
        elif kind == "done":
            st.attempts += 1
            st.status = ev.get("status", "failed")
            st.returncode = ev.get("returncode")
            st.peak_mib = ev.get("peak_mib", {})

    def _append(self, ev: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(ev) + "\n")
            f.flush()
            os.fsync(f.fileno())

    # ------------------------------------------------------------- API
    def state(self, key: str) -> JobState:
        return self.states.get(key, JobState())

    def ensure_seen(self, key: str, command: str) -> int:
        """Assign a persistent display number on first sight; return it."""
        st = self.states.get(key)
        if st is not None and st.no is not None:
            return st.no
        no = self._next_no
        ev = {"key": key, "event": "seen", "no": no, "command": command}
        self._append(ev)
        self._fold(ev)
        return no

    def record_oom_retry(self, key: str, attempts: int, next_vram: int | None) -> None:
        ev = {"key": key, "event": "oom", "attempts": attempts, "next_vram": next_vram}
        self._append(ev)
        self._fold(ev)

    def record_done(self, key: str, status: str, returncode: int,
                    peak_mib: dict[int, int], avg_util: float | None) -> None:
        ev = {
            "key": key, "event": "done", "status": status, "returncode": returncode,
            "peak_mib": {str(g): m for g, m in peak_mib.items()}, "avg_util": avg_util,
        }
        self._append(ev)
        self._fold(ev)
