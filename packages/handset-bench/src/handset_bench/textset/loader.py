"""Loader for the frozen text set."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "DEFAULT_PATH",
    "TEXTSET_NAME",
    "TEXTSET_SHA256",
    "TextsetHashMismatch",
    "Utterance",
    "load",
    "sample",
    "textset_hash",
]

TEXTSET_NAME = "dialtone_v1"
DEFAULT_PATH = Path(__file__).parent / "dialtone_v1.jsonl"

#: sha256 of dialtone_v1.jsonl. Regenerate with:
TEXTSET_SHA256 = "851cb1b30397921da76aa3bfae4ec97994cbb4f5a32fe3616f9502c847b73306"


class TextsetHashMismatch(RuntimeError):
    """The text set on disk is not the one the pinned numbers were measured on."""


@dataclass(frozen=True)
class Utterance:
    utterance_id: str
    text: str
    category: str
    n_words: int


def textset_hash(path: Path | None = None) -> str:
    target = path or DEFAULT_PATH
    return hashlib.sha256(target.read_bytes()).hexdigest()


def load(path: Path | None = None, *, verify: bool = True) -> list[Utterance]:
    """Load the frozen text set, refusing to proceed on a hash mismatch."""
    target = path or DEFAULT_PATH
    if not target.exists():
        raise FileNotFoundError(
            f"text set not found at {target}. "
            "Build it with: python -m handset_bench.textset.build_v1"
        )

    digest = textset_hash(target)
    if verify and digest != TEXTSET_SHA256:
        raise TextsetHashMismatch(
            f"text set at {target} has sha256 {digest}, expected {TEXTSET_SHA256}. "
            "Every published result record pins the text set hash, so a changed "
            "text set invalidates prior entries rather than silently shifting them."
        )

    utterances: list[Utterance] = []
    for line_number, line in enumerate(
        target.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            utterances.append(
                Utterance(
                    utterance_id=record["utterance_id"],
                    text=record["text"],
                    category=record["category"],
                    n_words=record["n_words"],
                )
            )
        except (json.JSONDecodeError, KeyError) as exc:
            raise ValueError(f"{target}:{line_number} is malformed: {exc}") from exc

    return utterances


def sample(utterances: list[Utterance], limit: int) -> list[Utterance]:
    """Take `limit` utterances spread evenly across categories."""
    # The file is grouped by category, so a plain slice returns only digits.
    if limit <= 0 or limit >= len(utterances):
        return list(utterances)

    buckets: dict[str, list[Utterance]] = {}
    for utterance in utterances:
        buckets.setdefault(utterance.category, []).append(utterance)

    picked: list[Utterance] = []
    categories = list(buckets)
    depth = 0
    while len(picked) < limit:
        progressed = False
        for category in categories:
            if depth < len(buckets[category]):
                picked.append(buckets[category][depth])
                progressed = True
                if len(picked) == limit:
                    break
        if not progressed:  # every bucket exhausted
            break
        depth += 1

    return sorted(picked, key=lambda u: u.utterance_id)
