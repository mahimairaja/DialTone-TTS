"""Tests for the simultaneous-call driver."""

import asyncio

import numpy as np
import pytest
from handset_bench.adapters.base import SynthResult, Timings
from handset_bench.concurrency import LEVELS, run_concurrent


class FixedAdapter:
    """Produces a fixed amount of audio after a fixed delay."""

    def __init__(self, audio_secs: float = 1.0, delay_s: float = 0.005):
        self.audio_secs = audio_secs
        self.delay_s = delay_s
        self.sample_rate = 8000

    def version_string(self) -> str:
        return "fixed-1.0.0"

    async def asynthesize(self, text: str, *, voice: str | None = None) -> SynthResult:
        import time

        submitted = time.perf_counter_ns()
        await asyncio.sleep(self.delay_s)
        first = time.perf_counter_ns()
        return SynthResult(
            pcm=np.zeros(int(self.sample_rate * self.audio_secs), dtype=np.float32),
            sample_rate=self.sample_rate,
            timings=Timings(submitted, first, time.perf_counter_ns()),
            status="ok",
        )


class FlakyAdapter(FixedAdapter):
    """Fails every nth request, to prove failures are counted rather than dropped."""

    def __init__(self, fail_every: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.fail_every = fail_every
        self._n = 0
        self._lock = asyncio.Lock()

    async def asynthesize(self, text: str, *, voice: str | None = None) -> SynthResult:
        async with self._lock:
            self._n += 1
            n = self._n
        if n % self.fail_every == 0:
            raise RuntimeError("simulated backend failure")
        return await super().asynthesize(text, voice=voice)


def test_levels_are_1_8_16_32():
    assert LEVELS == (1, 8, 16, 32)


async def test_failures_are_counted_not_dropped():
    """Dropping failures from the average would flatter whichever system fails most."""
    result = await run_concurrent(FlakyAdapter(fail_every=2), ["a"] * 10, n=4)
    assert result.successes + result.failures == 10
    assert result.failures == 5
    assert abs(result.failure_rate - 0.5) < 1e-9


async def test_all_requests_accounted_for_at_every_level():
    for n in LEVELS:
        result = await run_concurrent(FixedAdapter(), ["a"] * 32, n=n)
        assert result.successes + result.failures == 32
        assert result.n_concurrent == n


async def test_throughput_is_audio_seconds_over_wall_clock():
    """Speech produced per second of wall-clock time."""
    result = await run_concurrent(FixedAdapter(audio_secs=2.0), ["a"] * 4, n=4)
    expected = result.audio_seconds_produced / result.wall_clock_s
    assert abs(result.throughput_rtf - expected) < 1e-6


async def test_failed_requests_contribute_no_audio():
    result = await run_concurrent(
        FlakyAdapter(fail_every=2, audio_secs=1.0), ["a"] * 10, n=4
    )
    assert abs(result.audio_seconds_produced - 5.0) < 0.01


async def test_concurrency_actually_overlaps():
    """Eight requests at 8-way concurrency must finish faster than serialised."""
    serial = await run_concurrent(FixedAdapter(delay_s=0.02), ["a"] * 8, n=1)
    parallel = await run_concurrent(FixedAdapter(delay_s=0.02), ["a"] * 8, n=8)
    assert parallel.wall_clock_s < serial.wall_clock_s


async def test_zero_requests_is_rejected():
    with pytest.raises(ValueError):
        await run_concurrent(FixedAdapter(), [], n=4)


async def test_failure_rate_of_a_fully_failing_system_is_one():
    result = await run_concurrent(FlakyAdapter(fail_every=1), ["a"] * 4, n=2)
    assert result.failure_rate == 1.0
    assert result.successes == 0
