"""The frozen train and heldout split."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dialtone.data.manifest import ManifestRecord, manifest_hash

__all__ = [
    "HELDOUT_BASIS",
    "HELDOUT_SUBSETS",
    "HeldoutLeak",
    "Split",
    "load_split",
    "make_split",
    "write_split",
]

#: Subsets whose speakers are permanently excluded from training.
HELDOUT_SUBSETS = ("dev-clean", "test-clean")
HELDOUT_BASIS = "corpus dev-clean + test-clean"

#: Recorded for provenance. Nothing is drawn at random, but a future v2 split that
SPLIT_SEED = 20260814


class HeldoutLeak(RuntimeError):
    """A heldout utterance reached a training manifest. Non-recoverable."""


@dataclass(frozen=True)
class Split:
    heldout_basis: str
    seed: int
    manifest_hash: str
    train_speaker_ids: tuple[str, ...]
    heldout_speaker_ids: tuple[str, ...]
    heldout_utterance_ids: tuple[str, ...] = field(default=())

    @property
    def counts(self) -> dict[str, int]:
        return {
            "train_speakers": len(self.train_speaker_ids),
            "heldout_speakers": len(self.heldout_speaker_ids),
            "heldout_utterances": len(self.heldout_utterance_ids),
        }


def make_split(records: list[ManifestRecord]) -> Split:
    heldout_speakers = sorted(
        {r.speaker_id for r in records if r.subset in HELDOUT_SUBSETS}
    )
    if not heldout_speakers:
        raise ValueError(
            "no heldout speakers found. Expected records from subsets "
            f"{HELDOUT_SUBSETS}. Without a heldout portion every quality claim "
            "would be measured on speech the model had already seen."
        )

    train_speakers = sorted(
        {r.speaker_id for r in records if r.subset not in HELDOUT_SUBSETS}
    )
    overlap = set(train_speakers) & set(heldout_speakers)
    if overlap:
        raise ValueError(
            f"speakers appear in both train and heldout: {sorted(overlap)}. "
            "Train and heldout speakers must be disjoint."
        )

    heldout_utterances = sorted(
        r.utterance_id for r in records if r.subset in HELDOUT_SUBSETS
    )
    return Split(
        heldout_basis=HELDOUT_BASIS,
        seed=SPLIT_SEED,
        manifest_hash=manifest_hash(records),
        train_speaker_ids=tuple(train_speakers),
        heldout_speaker_ids=tuple(heldout_speakers),
        heldout_utterance_ids=tuple(heldout_utterances),
    )


def dumps_split(payload: dict) -> str:
    """Serialise with one line per top-level key."""
    keys = sorted(payload)
    body = ",\n".join(
        f"  {json.dumps(key)}: {json.dumps(payload[key], sort_keys=True)}"
        for key in keys
    )
    return "{\n" + body + "\n}\n"


def write_split(split: Split, path: Path) -> str:
    """Write the split and return its sha256, which is the frozen identity."""
    payload = asdict(split)
    payload["counts"] = split.counts
    text = dumps_split(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode()).hexdigest()


def load_split(path: Path) -> Split:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("counts", None)
    return Split(
        heldout_basis=payload["heldout_basis"],
        seed=payload["seed"],
        manifest_hash=payload["manifest_hash"],
        train_speaker_ids=tuple(payload["train_speaker_ids"]),
        heldout_speaker_ids=tuple(payload["heldout_speaker_ids"]),
        heldout_utterance_ids=tuple(payload.get("heldout_utterance_ids", ())),
    )
