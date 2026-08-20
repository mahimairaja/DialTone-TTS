"""The unified manifest."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

__all__ = [
    "ManifestRecord",
    "manifest_hash",
    "read_manifest",
    "utterance_id",
    "write_manifest",
]


def utterance_id(source: str, original_path: str) -> str:
    """Content-addressed id, so the pipeline is idempotent."""
    digest = hashlib.sha256(f"{source}|{original_path}".encode()).hexdigest()
    return f"{source}-{digest[:16]}"


@dataclass(frozen=True)
class ManifestRecord:
    utterance_id: str
    audio_path: str
    text: str
    speaker_id: str
    duration_s: float
    sample_rate: int
    source: str
    subset: str
    licence_spdx: str | None
    licence_url: str | None


def write_manifest(records: list[ManifestRecord], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(asdict(r), sort_keys=True) for r in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def read_manifest(path: Path) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(ManifestRecord(**json.loads(line)))
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"{path}:{number} is malformed: {exc}") from exc
    return records


def manifest_hash(records: list[ManifestRecord]) -> str:
    """Hash the manifest content, not the file, so formatting cannot change it."""
    payload = "\n".join(
        json.dumps(asdict(r), sort_keys=True)
        for r in sorted(records, key=lambda r: r.utterance_id)
    )
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()
