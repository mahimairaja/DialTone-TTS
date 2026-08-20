"""The heldout preflight."""

from __future__ import annotations

from collections.abc import Iterable

from dialtone.data.manifest import ManifestRecord
from dialtone.data.split import HeldoutLeak, Split

__all__ = ["verify_no_heldout"]


def verify_no_heldout(records: Iterable[ManifestRecord], split: Split) -> None:
    """Raise if any record belongs to a heldout speaker or utterance."""
    heldout_speakers = set(split.heldout_speaker_ids)
    heldout_utterances = set(split.heldout_utterance_ids)

    offending_speakers: set[str] = set()
    offending_utterances: set[str] = set()

    for record in records:
        if record.speaker_id in heldout_speakers:
            offending_speakers.add(record.speaker_id)
        if record.utterance_id in heldout_utterances:
            offending_utterances.add(record.utterance_id)

    if offending_speakers or offending_utterances:
        raise HeldoutLeak(
            "heldout material found in a training manifest. "
            f"Speakers: {sorted(offending_speakers)}. "
            f"Utterances: {sorted(offending_utterances)[:5]}. "
            "This invalidates every quality number the run would produce, so the "
            "run is aborted rather than continued."
        )
