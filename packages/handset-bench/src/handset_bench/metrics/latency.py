"""Latency aggregation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from handset_bench.adapters.base import AudioChunk, Timings

__all__ = ["LatencyAggregate", "aggregate", "first_audio_ns", "percentile"]


@dataclass(frozen=True)
class LatencyAggregate:
    ttfb_generation_p50_ms: float
    ttfb_generation_p95_ms: float
    n: int


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile. `q` is a fraction in [0, 1]."""
    if not values:
        raise ValueError("percentile of an empty sequence")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def first_audio_ns(chunks: Sequence[AudioChunk]) -> int:
    """The timestamp of the first piece, never the last."""
    if not chunks:
        raise ValueError("cannot take first audio of an empty stream")
    return min(chunk.received_ns for chunk in chunks)


def aggregate(timings: Sequence[Timings]) -> LatencyAggregate:
    if not timings:
        raise ValueError("cannot aggregate an empty set of timings")
    values = [t.ttfb_generation_ms for t in timings]
    return LatencyAggregate(
        ttfb_generation_p50_ms=percentile(values, 0.50),
        ttfb_generation_p95_ms=percentile(values, 0.95),
        n=len(values),
    )
