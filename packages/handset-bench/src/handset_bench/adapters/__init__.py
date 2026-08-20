"""Adapters: one thin wrapper per speech system under test."""

from handset_bench.adapters.base import (
    Adapter,
    AudioChunk,
    MissingVersionString,
    SynthResult,
    Timings,
    validate_adapter,
)
from handset_bench.adapters.registry import resolve

__all__ = [
    "Adapter",
    "AudioChunk",
    "MissingVersionString",
    "SynthResult",
    "Timings",
    "resolve",
    "validate_adapter",
]
