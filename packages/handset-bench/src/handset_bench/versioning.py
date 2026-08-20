"""Attribution helpers."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

__all__ = ["git_describe", "iso_now", "run_id", "sha256_file"]


@lru_cache(maxsize=1)
def git_describe(repo: Path | None = None) -> str:
    """`git describe --tags --always --dirty`, or a clear marker if unavailable."""
    try:
        out = subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=repo or Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return out.stdout.strip() or "unknown-revision"
    except (subprocess.SubprocessError, OSError):
        return "unknown-revision"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def iso_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def run_id(system: str, version: str, condition: str, stamp: str) -> str:
    """A stable id for one matrix cell, derived rather than random."""
    payload = f"{system}|{version}|{condition}|{stamp}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]
