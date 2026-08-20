"""Tests for latency aggregation."""

import numpy as np
import pytest
from handset_bench.adapters.base import AudioChunk, Timings
from handset_bench.metrics.latency import aggregate, first_audio_ns, percentile


def ms(value: float) -> int:
    return int(value * 1e6)


def test_p50_and_p95_on_a_known_distribution():
    timings = [Timings(0, ms(v), ms(v * 2)) for v in range(1, 101)]
    agg = aggregate(timings)
    assert 50.0 <= agg.ttfb_generation_p50_ms <= 51.0
    assert 95.0 <= agg.ttfb_generation_p95_ms <= 96.0
    assert agg.n == 100


def test_percentile_endpoints():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(values, 0.0) == 1.0
    assert percentile(values, 1.0) == 5.0


def test_percentile_of_a_single_value():
    assert percentile([7.0], 0.95) == 7.0


def test_field_is_named_ttfb_generation_not_ttfb():
    """The harness measures a warm in-process call, not an end-to-end call through a"""
    agg = aggregate([Timings(0, ms(1), ms(2))])
    assert hasattr(agg, "ttfb_generation_p50_ms")
    assert not hasattr(agg, "ttfb_p50_ms")


def test_streaming_measurement_stops_at_the_first_chunk():
    """Time to first audio stops at the first chunk, not the last."""
    pcm = np.zeros(4, dtype=np.float32)
    chunks = [
        AudioChunk(pcm=pcm, sample_rate=8000, received_ns=ms(5)),
        AudioChunk(pcm=pcm, sample_rate=8000, received_ns=ms(90)),
    ]
    assert first_audio_ns(chunks) == ms(5)


def test_first_audio_of_an_empty_stream_is_rejected():
    with pytest.raises(ValueError):
        first_audio_ns([])


def test_aggregate_of_nothing_is_rejected():
    with pytest.raises(ValueError):
        aggregate([])
