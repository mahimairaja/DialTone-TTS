"""Voice selection, and prompt-speaker selection for the benchmark."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from dialtone.data.manifest import ManifestRecord
from dialtone.data.split import Split

__all__ = [
    "N_PROMPT_SPEAKERS",
    "N_VOICES",
    "VOICE_SELECTION_RULE",
    "VOICE_SUBSET",
    "Voice",
    "select_prompt_speakers",
    "select_voices",
]

N_VOICES = 4
N_PROMPT_SPEAKERS = 3
VOICE_SUBSET = "train-clean-100"

VOICE_SELECTION_RULE = (
    "the speakers with the greatest total clean duration in train-clean-100, "
    "ordered by duration descending then by speaker id, taking the first four"
)

PROMPT_SELECTION_RULE = (
    "the heldout speakers with the greatest total duration, ordered by duration "
    "descending then by speaker id, taking the first three"
)


@dataclass(frozen=True)
class Voice:
    id: str
    speaker_id: str
    subset: str
    total_duration_s: float
    selection_rule: str = VOICE_SELECTION_RULE


def _duration_by_speaker(records: list[ManifestRecord], predicate) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for record in records:
        if predicate(record):
            totals[record.speaker_id] += record.duration_s
    return dict(totals)


def _rank(totals: dict[str, float]) -> list[str]:
    """Descending by duration, then ascending by id."""
    return [
        speaker for speaker, _ in sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


def select_voices(records: list[ManifestRecord], n: int = N_VOICES) -> list[Voice]:
    """The DialTone voice set."""
    totals = _duration_by_speaker(records, lambda r: r.subset == VOICE_SUBSET)
    if len(totals) < n:
        raise ValueError(
            f"need at least {n} speakers in {VOICE_SUBSET}, found {len(totals)}"
        )
    chosen = _rank(totals)[:n]
    return [
        Voice(
            id=f"dt-voice-{i + 1}",
            speaker_id=speaker,
            subset=VOICE_SUBSET,
            total_duration_s=round(totals[speaker], 3),
        )
        for i, speaker in enumerate(chosen)
    ]


def select_prompt_speakers(
    records: list[ManifestRecord], split: Split, n: int = N_PROMPT_SPEAKERS
) -> list[str]:
    """Prompt audio for the benchmark's cloning-capable systems."""
    heldout = set(split.heldout_speaker_ids)
    totals = _duration_by_speaker(records, lambda r: r.speaker_id in heldout)
    if len(totals) < n:
        raise ValueError(f"need at least {n} heldout speakers, found {len(totals)}")
    return _rank(totals)[:n]
