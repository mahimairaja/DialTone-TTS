"""Pinned identities of the two listeners."""

from __future__ import annotations

__all__ = [
    "PARAKEET_MODEL",
    "PARAKEET_REVISION",
    "WHISPER_MODEL",
    "WHISPER_REVISION",
]

#: Headline listener. NVIDIA publishes an evaluation on exactly our condition:
PARAKEET_MODEL = "nvidia/parakeet-tdt-0.6b-v2"
PARAKEET_REVISION = "main"  # pin to a commit SHA before the first published run

#: Second column, deliberately not the headline. Whisper hallucinates on
WHISPER_MODEL = "large-v3"
WHISPER_REVISION = "main"
