"""LibriTTS-R ingest."""

from __future__ import annotations

import os
import wave
from collections.abc import Iterator
from pathlib import Path

from dialtone.data.manifest import ManifestRecord, utterance_id

__all__ = [
    "LICENCE_SPDX",
    "LICENCE_URL",
    "SOURCE",
    "SUBSET_URLS",
    "ingest_subset",
    "list_speakers",
    "probe_format",
    "tarball_url",
]

SOURCE = "libritts-r"
LICENCE_SPDX = "CC-BY-4.0"
LICENCE_URL = "https://creativecommons.org/licenses/by/4.0/"

#: openslr resource ids. Sizes are the compressed tarballs.
SUBSET_URLS: dict[str, str] = {
    "dev-clean": "dev_clean",
    "test-clean": "test_clean",
    "train-clean-100": "train_clean_100",
    "train-clean-360": "train_clean_360",
}


def tarball_url(subset: str) -> str:
    if subset not in SUBSET_URLS:
        raise ValueError(f"unknown subset {subset!r}. Known: {sorted(SUBSET_URLS)}")
    return f"https://www.openslr.org/resources/141/{SUBSET_URLS[subset]}.tar.gz"


def probe_format(subset_root: Path) -> tuple[int, int, int]:
    """Read one WAV header to establish the format of an entire subset."""
    for candidate in subset_root.rglob("*.wav"):
        with wave.open(str(candidate), "rb") as handle:
            return (
                handle.getframerate(),
                handle.getnchannels(),
                handle.getsampwidth(),
            )
    raise FileNotFoundError(f"no wav files under {subset_root}")


def list_speakers(root: Path, subset: str) -> list[str]:
    """Speaker directory names in a subset, sorted."""
    subset_root = root / subset
    if not subset_root.is_dir():
        raise FileNotFoundError(f"subset directory not found: {subset_root}")
    return sorted(p.name for p in subset_root.iterdir() if p.is_dir())


def ingest_subset(
    root: Path,
    subset: str,
    speakers: list[str] | None = None,
) -> Iterator[ManifestRecord]:
    """Walk one extracted subset and emit a record per utterance."""
    subset_root = root / subset
    if not subset_root.is_dir():
        raise FileNotFoundError(f"subset directory not found: {subset_root}")

    sample_rate, channels, width = probe_format(subset_root)
    bytes_per_second = sample_rate * channels * width
    header_bytes = 44

    wanted = set(speakers) if speakers is not None else None
    speaker_dirs = [p for p in sorted(subset_root.iterdir()) if p.is_dir()]
    if wanted is not None:
        speaker_dirs = [p for p in speaker_dirs if p.name in wanted]

    for speaker_dir in speaker_dirs:
        for chapter_dir in sorted(p for p in speaker_dir.iterdir() if p.is_dir()):
            entries = {e.name: e for e in os.scandir(chapter_dir)}
            for name in sorted(entries):
                if not name.endswith(".wav"):
                    continue
                stem = name[: -len(".wav")]
                text_entry = entries.get(f"{stem}.normalized.txt")
                if text_entry is None:
                    continue

                text = Path(text_entry.path).read_text(encoding="utf-8").strip()
                if not text:
                    continue

                size = entries[name].stat().st_size
                duration = max(size - header_bytes, 0) / bytes_per_second
                relative = Path(entries[name].path).relative_to(root).as_posix()

                yield ManifestRecord(
                    utterance_id=utterance_id(SOURCE, relative),
                    audio_path=relative,
                    # <speaker>_<chapter>_<a>_<b>.wav
                    speaker_id=stem.split("_")[0],
                    text=text,
                    duration_s=round(duration, 4),
                    sample_rate=sample_rate,
                    source=SOURCE,
                    subset=subset,
                    licence_spdx=LICENCE_SPDX,
                    licence_url=LICENCE_URL,
                )
