"""The licence gate."""

from __future__ import annotations

from dataclasses import dataclass

from dialtone.data.manifest import ManifestRecord

__all__ = [
    "ALLOWED_SPDX",
    "DEFAULT_REJECTION_THRESHOLD",
    "Rejection",
    "RejectionRateExceeded",
    "gate",
]

#: SPDX identifiers permitting training and derivative redistribution.
ALLOWED_SPDX = frozenset({"CC-BY-4.0", "CC0-1.0"})

#: Above this share of rejections, the run aborts instead of continuing.
DEFAULT_REJECTION_THRESHOLD = 0.05


class RejectionRateExceeded(RuntimeError):
    """Too much of the corpus was rejected to trust the ingest adapter."""


@dataclass(frozen=True)
class Rejection:
    utterance_id: str
    licence_spdx: str | None
    reason: str


def gate(
    records: list[ManifestRecord],
    threshold: float = DEFAULT_REJECTION_THRESHOLD,
) -> tuple[list[ManifestRecord], list[Rejection]]:
    """Split records into accepted and rejected."""
    accepted: list[ManifestRecord] = []
    rejected: list[Rejection] = []

    for record in records:
        if not record.licence_spdx:
            rejected.append(Rejection(record.utterance_id, None, "licence_absent"))
        elif record.licence_spdx not in ALLOWED_SPDX:
            rejected.append(
                Rejection(
                    record.utterance_id,
                    record.licence_spdx,
                    "licence_not_allowlisted",
                )
            )
        else:
            accepted.append(record)

    if records:
        rate = len(rejected) / len(records)
        if rate > threshold:
            raise RejectionRateExceeded(
                f"{rate:.1%} of records were rejected, above the {threshold:.1%} "
                "threshold. This usually means a broken ingest adapter rather than "
                "a genuinely unlicensed corpus. Refusing to continue with a corpus "
                "that may have been silently emptied."
            )

    return accepted, rejected
