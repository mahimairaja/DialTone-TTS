"""The adapter protocol."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "Adapter",
    "AudioChunk",
    "MissingVersionString",
    "PLACEHOLDER_VERSIONS",
    "Status",
    "SynthResult",
    "Timings",
    "validate_adapter",
]

Status = Literal["ok", "empty", "error"]

#: Version strings that look filled in but carry no attribution.
PLACEHOLDER_VERSIONS = frozenset({"unknown", "none", "n/a", "na", "latest", "dev", "-"})


class MissingVersionString(RuntimeError):
    """Raised when an adapter cannot attribute its output to a specific version."""


@dataclass(frozen=True)
class Timings:
    """Wall-clock timestamps in monotonic nanoseconds."""

    submitted_ns: int
    first_audio_ns: int
    completed_ns: int

    def __post_init__(self) -> None:
        if self.first_audio_ns < self.submitted_ns:
            raise ValueError("first_audio_ns precedes submitted_ns")
        if self.completed_ns < self.first_audio_ns:
            raise ValueError("completed_ns precedes first_audio_ns")

    @property
    def ttfb_generation_ms(self) -> float:
        """Generation-side time to first audio byte."""
        return (self.first_audio_ns - self.submitted_ns) / 1e6

    @property
    def total_ms(self) -> float:
        return (self.completed_ns - self.submitted_ns) / 1e6


@dataclass(frozen=True)
class AudioChunk:
    """One piece of a streamed response, stamped when the client received it."""

    pcm: np.ndarray
    sample_rate: int
    received_ns: int


@dataclass(frozen=True)
class SynthResult:
    """The outcome of one synthesis request."""

    pcm: np.ndarray
    sample_rate: int
    timings: Timings
    status: Status = "ok"
    error: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.pcm.dtype != np.float32:
            raise ValueError(f"pcm must be float32, got {self.pcm.dtype}")
        if self.status == "error" and not self.error:
            raise ValueError("status 'error' requires an error message")
        if self.status == "ok" and self.pcm.size == 0:
            raise ValueError("status 'ok' with empty audio: use status 'empty'")


@runtime_checkable
class Adapter(Protocol):
    """What every speech system under test must expose."""

    def version_string(self) -> str:
        """A pinned, specific version. Never a tag, a branch, or a placeholder."""
        ...

    def synthesize(self, text: str, *, voice: str | None = None) -> SynthResult: ...

    def synthesize_stream(
        self, text: str, *, voice: str | None = None
    ) -> Iterator[AudioChunk]: ...


def validate_adapter(adapter: object) -> None:
    """Abort a run rather than produce a record nobody can attribute."""
    getter = getattr(adapter, "version_string", None)
    if getter is None:
        raise MissingVersionString(
            f"{type(adapter).__name__} does not expose version_string()"
        )
    version = getter()
    if not version or not version.strip():
        raise MissingVersionString(
            f"{type(adapter).__name__}.version_string() returned an empty string"
        )
    if version.strip().lower() in PLACEHOLDER_VERSIONS:
        raise MissingVersionString(
            f"{type(adapter).__name__}.version_string() returned the placeholder "
            f"{version!r}. Runs that cannot be attributed to a specific version "
            "are discarded, not published."
        )
