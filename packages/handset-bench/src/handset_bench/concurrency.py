"""The simultaneous-call driver."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["LEVELS", "ConcurrencyResult", "run_concurrent"]

#: The concurrency levels every system is measured at.
LEVELS: tuple[int, ...] = (1, 8, 16, 32)


@dataclass(frozen=True)
class ConcurrencyResult:
    n_concurrent: int
    successes: int
    failures: int
    wall_clock_s: float
    audio_seconds_produced: float
    errors: tuple[str, ...] = ()

    @property
    def failure_rate(self) -> float:
        total = self.successes + self.failures
        return self.failures / total if total else 0.0

    @property
    def throughput_rtf(self) -> float:
        """Speech produced per second of wall-clock time."""
        return (
            self.audio_seconds_produced / self.wall_clock_s
            if self.wall_clock_s > 0
            else 0.0
        )


async def _call(adapter, text: str, voice: str | None):
    """Call an adapter, preferring its async path and falling back to a thread."""
    if hasattr(adapter, "asynthesize"):
        return await adapter.asynthesize(text, voice=voice)
    return await asyncio.to_thread(adapter.synthesize, text, voice=voice)


async def run_concurrent(
    adapter,
    texts: Sequence[str],
    n: int,
    *,
    voice: str | None = None,
) -> ConcurrencyResult:
    """Issue `len(texts)` requests with at most `n` in flight at once."""
    if not texts:
        raise ValueError("cannot run a concurrency measurement with no requests")
    if n < 1:
        raise ValueError(f"concurrency level must be at least 1, got {n}")

    semaphore = asyncio.Semaphore(n)

    async def one(text: str):
        async with semaphore:
            return await _call(adapter, text, voice)

    started = time.perf_counter()
    outcomes = await asyncio.gather(
        *(one(text) for text in texts), return_exceptions=True
    )
    wall_clock = time.perf_counter() - started

    successes = 0
    failures = 0
    audio_seconds = 0.0
    errors: list[str] = []

    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            failures += 1
            errors.append(f"{type(outcome).__name__}: {outcome}")
            continue
        if getattr(outcome, "status", "error") != "ok":
            failures += 1
            errors.append(outcome.error or "non-ok status")
            continue
        successes += 1
        audio_seconds += outcome.pcm.size / outcome.sample_rate

    return ConcurrencyResult(
        n_concurrent=n,
        successes=successes,
        failures=failures,
        wall_clock_s=wall_clock,
        audio_seconds_produced=audio_seconds,
        errors=tuple(errors),
    )
