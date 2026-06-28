"""Tiny in-process job runner for long pipeline stages.

The batch stages (upload, prelabel, verify, GPU spin-up) take seconds to
minutes, so the API runs them on background threads and the browser polls for
progress. A Job carries live counters (done/total/failures), a bounded log, and
a terminal result/error. Single-process and ephemeral by design — restart =
clean slate; storage (S3/local) remains the source of truth.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any, Callable

_LOG_MAX = 500


class Job:
    """One background unit of work. All mutating access is lock-guarded so the
    worker thread and the polling request never tear a read."""

    def __init__(self, job_id: str, kind: str):
        self.id = job_id
        self.kind = kind
        self.status = "pending"  # pending -> running -> done | error
        self.total = 0
        self.done = 0
        self.failures = 0
        self.result: Any = None
        self.error: str | None = None
        self._log: deque[str] = deque(maxlen=_LOG_MAX)
        self._lock = threading.Lock()

    def set_progress(self, done: int, total: int) -> None:
        with self._lock:
            self.done, self.total = done, total

    def log(self, line: str) -> None:
        with self._lock:
            self._log.append(str(line))

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "id": self.id,
                "kind": self.kind,
                "status": self.status,
                "done": self.done,
                "total": self.total,
                "failures": self.failures,
                "log": list(self._log),
                "result": self.result,
                "error": self.error,
            }


class JobRegistry:
    """Thread-safe registry that submits jobs to daemon threads."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._counter = 0

    def submit(self, kind: str, fn: Callable[[Job], Any]) -> Job:
        """Run `fn(job)` on a background thread. `fn` reports progress via
        job.set_progress / job.log and returns the JSON-able result."""
        with self._lock:
            self._counter += 1
            job = Job(str(self._counter), kind)
            self._jobs[job.id] = job

        def run() -> None:
            job.status = "running"
            try:
                job.result = fn(job)
                job.status = "done"
            except Exception as exc:  # noqa: BLE001 - surfaced to the client
                job.error = f"{type(exc).__name__}: {exc}"
                job.log(job.error)
                job.status = "error"

        threading.Thread(target=run, name=f"job-{job.id}-{kind}", daemon=True).start()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)
