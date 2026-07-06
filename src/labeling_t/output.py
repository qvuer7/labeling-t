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
import sys


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
