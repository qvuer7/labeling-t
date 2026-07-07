"""Agent-facing output envelope for both CLIs.

Every subcommand takes --json. With it, stdout carries EXACTLY ONE line —
`{"ok": true, "result": {...}}` or `{"ok": false, "error": {"message": ...}}` —
and all prose (summaries, progress, warnings) goes to stderr. Without it,
output is the human prose it always was. Invariant either way: `ok` mirrors
the exit code (ok=true <=> rc 0), so agents may check either.

Handlers use these three helpers instead of print():

    return emit(a, {"labeled": n, ...}, f"labeled {n} ...")   # success -> 0
    return fail(a, "no frames under ...")                     # error   -> 1
    note(a, f"[{i}/{n}] {group}")                             # mid-run prose
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time


def json_flag() -> argparse.ArgumentParser:
    """Parent parser adding --json to a subcommand (parents=[json_flag()])."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--json", action="store_true",
                   help="machine-readable output: one JSON envelope on stdout, prose to stderr")
    return p


def emit(a: argparse.Namespace, result: dict, message: str = "") -> int:
    """Success: envelope on stdout (--json) or `message` on stdout. Returns 0."""
    if getattr(a, "json", False):
        if message:
            print(message, file=sys.stderr)
        print(json.dumps({"ok": True, "result": result}))
    elif message:
        print(message)
    return 0


def fail(a: argparse.Namespace, message: str, *, result: dict | None = None, **extra) -> int:
    """Error: message on stderr, plus an ok=false envelope on stdout with --json.
    `extra` keys join the error object (structured detail, e.g. candidate pods);
    `result` carries partial success (e.g. a pod that came up but timed out).
    Returns 1 so handlers can `return fail(...)`."""
    print(message, file=sys.stderr)
    if getattr(a, "json", False):
        env: dict = {"ok": False, "error": {"message": message, **extra}}
        if result is not None:
            env["result"] = result
        print(json.dumps(env))
    return 1


def note(a: argparse.Namespace, message: str) -> None:
    """Mid-command prose (progress lines, hints): stdout for humans, stderr
    under --json so the envelope stays alone on stdout."""
    print(message, file=sys.stderr if getattr(a, "json", False) else sys.stdout, flush=True)


def progress_flag() -> argparse.ArgumentParser:
    """Parent parser adding --progress-file to a long-running subcommand."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--progress-file", default=None,
                   help="also atomically rewrite the latest progress event into this local file "
                        "(poll it from another process)")
    return p


def progress_reporter(a: argparse.Namespace, stage: str, *,
                      every: int = 25, min_interval: float = 5.0):
    """An on_progress(done, total) callback for the batch stages.

    Emits one JSON line per report to STDERR in both modes (the --json stdout
    envelope stays alone; humans see a heartbeat instead of 40 silent minutes):

        {"event": "progress", "stage": "<stage>", "done": n, "total": N,
         "elapsed_s": s}

    Throttled: the first item, then every `every` items or `min_interval`
    seconds (whichever comes first), and always the final item. With
    --progress-file the latest event also lands in that file via atomic
    rewrite (tmp + os.replace), so pollers never read a torn line."""
    start = time.monotonic()
    state = {"done": 0, "at": start}
    path = getattr(a, "progress_file", None)

    def report(done: int, total: int) -> None:
        now = time.monotonic()
        if not (done == 1 or done == total
                or done - state["done"] >= every or now - state["at"] >= min_interval):
            return
        state["done"], state["at"] = done, now
        event = {"event": "progress", "stage": stage, "done": done, "total": total,
                 "elapsed_s": round(now - start, 1)}
        line = json.dumps(event)
        print(line, file=sys.stderr, flush=True)
        if path:
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)) or ".")
            try:
                with os.fdopen(fd, "w") as f:
                    f.write(line + "\n")
                os.replace(tmp, path)
            except BaseException:
                if os.path.exists(tmp):
                    os.unlink(tmp)
                raise

    return report
