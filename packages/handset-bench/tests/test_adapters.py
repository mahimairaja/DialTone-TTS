"""Tests for the adapter protocol and the dotted-path registry."""

import time

import numpy as np
import pytest
from handset_bench.adapters.base import (
    AudioChunk,
    MissingVersionString,
    SynthResult,
    Timings,
    validate_adapter,
)
from handset_bench.adapters.registry import resolve


class FixtureAdapter:
    """A recorded adapter. No network, no model, deterministic."""

    def __init__(self, sample_rate: int = 22050, secs: float = 0.5):
        self.sample_rate = sample_rate
        self._pcm = np.zeros(int(sample_rate * secs), dtype=np.float32)

    def version_string(self) -> str:
        return "fixture-1.0.0"

    def synthesize(self, text: str, *, voice: str | None = None) -> SynthResult:
        t0 = time.perf_counter_ns()
        t1 = t0 + 1_000_000
        return SynthResult(
            pcm=self._pcm,
            sample_rate=self.sample_rate,
            timings=Timings(t0, t1, t1 + 2_000_000),
            status="ok",
        )


class NoVersionAdapter(FixtureAdapter):
    def version_string(self) -> str:
        return ""


class PlaceholderVersionAdapter(FixtureAdapter):
    def version_string(self) -> str:
        return "unknown"


def test_ttfb_is_computed_from_timings():
    t = Timings(submitted_ns=0, first_audio_ns=41_200_000, completed_ns=90_000_000)
    assert abs(t.ttfb_generation_ms - 41.2) < 1e-9
    assert abs(t.total_ms - 90.0) < 1e-9


def test_timings_reject_a_first_audio_before_submission():
    with pytest.raises(ValueError):
        Timings(submitted_ns=100, first_audio_ns=50, completed_ns=200)


def test_empty_version_string_aborts():
    """A run that cannot be attributed to a version is discarded, not published."""
    with pytest.raises(MissingVersionString):
        validate_adapter(NoVersionAdapter())


def test_placeholder_version_string_aborts():
    with pytest.raises(MissingVersionString):
        validate_adapter(PlaceholderVersionAdapter())


def test_valid_adapter_passes_validation():
    validate_adapter(FixtureAdapter())


def test_fixture_adapter_conforms_to_the_protocol():
    result = FixtureAdapter().synthesize("hello world")
    assert result.status == "ok"
    assert result.pcm.dtype == np.float32
    assert result.timings.first_audio_ns >= result.timings.submitted_ns


def test_synth_result_rejects_non_float32_pcm():
    with pytest.raises(ValueError):
        SynthResult(
            pcm=np.zeros(10, dtype=np.float64),
            sample_rate=8000,
            timings=Timings(0, 1, 2),
            status="ok",
        )


def test_error_status_requires_an_error_message():
    with pytest.raises(ValueError):
        SynthResult(
            pcm=np.zeros(0, dtype=np.float32),
            sample_rate=8000,
            timings=Timings(0, 1, 2),
            status="error",
            error=None,
        )


def test_registry_resolves_a_dotted_path():
    cls = resolve("handset_bench.adapters.base:SynthResult")
    assert cls is SynthResult


def test_registry_rejects_a_path_without_a_colon():
    with pytest.raises(ValueError, match="module:attribute"):
        resolve("no_colon_here")


def test_registry_reports_a_missing_attribute_clearly():
    with pytest.raises(AttributeError):
        resolve("handset_bench.adapters.base:NoSuchThing")


def test_audio_chunk_carries_a_receive_timestamp():
    chunk = AudioChunk(
        pcm=np.zeros(4, dtype=np.float32), sample_rate=8000, received_ns=123
    )
    assert chunk.received_ns == 123


def test_piper_version_states_the_noise_mode():
    """Zeroed noise changes the waveform, so it has to change the identity too."""
    from handset_bench.adapters.piper import PiperAdapter

    deterministic = PiperAdapter(voice="en_US-lessac-medium")
    sampled = PiperAdapter(voice="en_US-lessac-medium", deterministic=False)
    deterministic._piper_version = sampled._piper_version = "1.6.1"

    assert deterministic.version_string().endswith("+det")
    assert sampled.version_string().endswith("+sampled")
    assert deterministic.version_string() != sampled.version_string()


def test_piper_defaults_to_deterministic():
    from handset_bench.adapters.piper import PiperAdapter

    assert PiperAdapter().deterministic is True
