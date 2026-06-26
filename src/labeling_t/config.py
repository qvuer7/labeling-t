"""Environment loading. Infra config (per-model endpoints/keys) lives in .env
and is read by each ModelSpec (see models.py). Model behavior is NOT here."""

from __future__ import annotations


def load_env(path: str = ".env") -> None:
    """Load .env into os.environ if python-dotenv is available. No-op otherwise."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dotenv is a declared dep
        return
    load_dotenv(path)
