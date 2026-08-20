"""DATA_CARD.md generation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from dialtone.data.manifest import ManifestRecord

__all__ = ["EXCLUSIONS", "render_data_card"]

#: Corpora considered and rejected, with the reason each failed.
EXCLUSIONS: tuple[tuple[str, str], ...] = (
    (
        "GigaSpeech",
        "Rests on a fair-use argument and its licence is documented as preventing "
        "industrial research. Fails the requirement that every recording permit "
        "public release of derived work.",
    ),
    (
        "People's Speech",
        "Permits commercial use but is partly CC-BY-SA. Share-alike on a derived "
        "model is exactly the ambiguity the licence requirement exists to avoid.",
    ),
    ("Emilia", "CC BY-NC. Non-commercial excludes it outright."),
    (
        "Libri-Light",
        "Overwhelmingly unlabelled. The wrong tool for a corpus that needs "
        "transcripts.",
    ),
    (
        "MLS English",
        "Licence-clean under CC BY 4.0 but 16kHz. Upsampling to 24kHz would give a "
        "mel with no real energy above 8kHz, a distribution mismatch against what "
        "the frozen acoustic model emits at inference. Kept on the shelf.",
    ),
    (
        "LDC Switchboard and Fisher",
        "Paid. Out of scope for this milestone by decision, and reopened only if "
        "augmentation proves insufficient.",
    ),
)


def render_data_card(
    records: Sequence[ManifestRecord],
    removals: Sequence[dict],
) -> str:
    by_source: dict[str, list[ManifestRecord]] = defaultdict(list)
    for record in records:
        by_source[record.source].append(record)

    total_seconds = sum(r.duration_s for r in records)
    total_speakers = len({r.speaker_id for r in records})

    lines = [
        "# DialTone data card",
        "",
        "Generated. Do not edit by hand.",
        "",
        "## Summary",
        "",
        f"- Total duration: {total_seconds / 3600:.2f} hours",
        f"- Speakers: {total_speakers}",
        f"- Utterances: {len(records)}",
        "",
        "## Sources",
        "",
        "| Source | Licence | Utterances | Speakers | Hours | Share |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for source in sorted(by_source):
        group = by_source[source]
        seconds = sum(r.duration_s for r in group)
        licences = sorted({r.licence_spdx or "unknown" for r in group})
        share = (seconds / total_seconds * 100) if total_seconds else 0.0
        lines.append(
            f"| {source} | {', '.join(licences)} | {len(group)} | "
            f"{len({r.speaker_id for r in group})} | {seconds / 3600:.2f} | "
            f"{share:.1f}% |"
        )

    lines += ["", "## Excluded corpora", ""]
    for name, reason in EXCLUSIONS:
        lines.append(f"- **{name}**: {reason}")

    lines += ["", "## Removals", ""]
    if not removals:
        lines.append("None.")
    else:
        lines += ["| Utterance | Removed at | Reason |", "| --- | --- | --- |"]
        for removal in removals:
            lines.append(
                f"| {removal.get('utterance_id', '?')} | "
                f"{removal.get('removed_at', '?')} | "
                f"{removal.get('reason', '?')} |"
            )

    return "\n".join(lines) + "\n"
