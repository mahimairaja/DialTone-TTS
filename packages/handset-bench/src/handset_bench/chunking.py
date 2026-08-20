"""Sentence chunking, and the driver that makes a non-streaming model streamable."""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from handset_bench.adapters.base import Status, SynthResult

__all__ = ["ChunkedDriver", "ChunkedResult", "DEFAULT_MAX_CHARS", "split_sentences"]

DEFAULT_MAX_CHARS = 200

#: Tokens that end in a period without ending a sentence.
_ABBREVIATIONS = frozenset(
    {
        "dr",
        "mr",
        "mrs",
        "ms",
        "prof",
        "rev",
        "hon",
        "st",
        "ave",
        "blvd",
        "rd",
        "ln",
        "ct",
        "dept",
        "inc",
        "ltd",
        "co",
        "corp",
        "jr",
        "sr",
        "vs",
        "etc",
        "approx",
        "apt",
        "no",
        "fig",
        "vol",
        "est",
        "min",
        "max",
        "sec",
    }
)

# A sentence boundary is terminal punctuation followed by whitespace. The lookbehind
_BOUNDARY = re.compile(r"(?<=[.!?])(?<!\d\.)[\"')\]]*\s+")


def _last_token(text: str) -> str:
    stripped = text.rstrip()
    if not stripped.endswith("."):
        return ""
    return re.split(r"[\s(\[\"']", stripped[:-1])[-1].lower()


def _split_long(chunk: str, max_chars: int) -> list[str]:
    """Break an over-long chunk on word boundaries, never mid-word."""
    words = chunk.split()
    out: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        addition = len(word) + (1 if current else 0)
        if current and length + addition > max_chars:
            out.append(" ".join(current))
            current, length = [word], len(word)
        else:
            current.append(word)
            length += addition
    if current:
        out.append(" ".join(current))
    return out


def split_sentences(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    """Split into chunks at sentence boundaries, respecting a maximum length."""
    if not text or not text.strip():
        return []

    pieces = _BOUNDARY.split(text.strip())

    # Re-join pieces whose split was caused by an abbreviation rather than a
    merged: list[str] = []
    for piece in pieces:
        if merged and _last_token(merged[-1]) in _ABBREVIATIONS:
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)

    out: list[str] = []
    for chunk in merged:
        chunk = chunk.strip()
        if not chunk:
            continue
        out.extend(_split_long(chunk, max_chars) if len(chunk) > max_chars else [chunk])
    return out


@dataclass(frozen=True)
class ChunkedResult:
    """First-chunk and whole-message time, reported separately."""

    pcm: np.ndarray
    sample_rate: int
    ttfb_first_chunk_ms: float
    ttft_full_message_ms: float
    n_chunks: int
    status: Status = "ok"
    error: str | None = None


class ChunkedDriver:
    """Wrap a non-streaming adapter and expose chunked timing."""

    def __init__(self, adapter, max_chars: int = DEFAULT_MAX_CHARS):
        self.adapter = adapter
        self.max_chars = max_chars

    def version_string(self) -> str:
        return self.adapter.version_string()

    def synthesize_chunked(
        self, text: str, *, voice: str | None = None
    ) -> ChunkedResult:
        chunks = split_sentences(text, self.max_chars)
        if not chunks:
            raise ValueError("cannot synthesize empty text")

        started_ns = time.perf_counter_ns()
        first_audio_ns: int | None = None
        parts: list[np.ndarray] = []
        sample_rate: int | None = None

        results: Sequence[SynthResult] = []
        for chunk in chunks:
            result = self.adapter.synthesize(chunk, voice=voice)
            results = [*results, result]

            if result.status != "ok":
                now = time.perf_counter_ns()
                return ChunkedResult(
                    pcm=np.zeros(0, dtype=np.float32),
                    sample_rate=result.sample_rate,
                    ttfb_first_chunk_ms=(now - started_ns) / 1e6,
                    ttft_full_message_ms=(now - started_ns) / 1e6,
                    n_chunks=len(chunks),
                    status=result.status,
                    error=result.error or "chunk synthesis failed",
                )

            if sample_rate is None:
                sample_rate = result.sample_rate
            elif result.sample_rate != sample_rate:
                raise ValueError(
                    "adapter changed sample rate mid-message: "
                    f"{sample_rate} then {result.sample_rate}"
                )

            if first_audio_ns is None:
                first_audio_ns = time.perf_counter_ns()
            parts.append(result.pcm)

        completed_ns = time.perf_counter_ns()
        assert first_audio_ns is not None and sample_rate is not None
        return ChunkedResult(
            pcm=np.concatenate(parts),
            sample_rate=sample_rate,
            ttfb_first_chunk_ms=(first_audio_ns - started_ns) / 1e6,
            ttft_full_message_ms=(completed_ns - started_ns) / 1e6,
            n_chunks=len(chunks),
            status="ok",
        )
