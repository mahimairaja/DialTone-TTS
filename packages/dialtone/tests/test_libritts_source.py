"""Tests for the LibriTTS-R ingest adapter."""

import struct
import wave

import pytest
from dialtone.data.sources.libritts_r import (
    LICENCE_SPDX,
    SOURCE,
    ingest_subset,
    probe_format,
    tarball_url,
)


def _write_wav(path, seconds=2.0, rate=24000):
    frames = int(rate * seconds)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(struct.pack("<h", 0) * frames)


@pytest.fixture
def corpus(tmp_path):
    """LibriTTS_R/<subset>/<speaker>/<chapter>/<speaker>_<chapter>_<a>_<b>.wav"""
    root = tmp_path / "libritts_r"
    for speaker, seconds in (("103", 2.0), ("1034", 5.0)):
        chapter = root / "dev-clean" / speaker / "1240"
        chapter.mkdir(parents=True)
        for index, secs in enumerate([seconds, seconds + 1.0]):
            stem = f"{speaker}_1240_00000{index}_000000"
            _write_wav(chapter / f"{stem}.wav", seconds=secs)
            (chapter / f"{stem}.normalized.txt").write_text("hello there friend")
    return root


def test_tarball_url_is_openslr(corpus):
    assert tarball_url("dev-clean").endswith("/141/dev_clean.tar.gz")


def test_unknown_subset_is_rejected():
    with pytest.raises(ValueError, match="unknown subset"):
        tarball_url("train-clean-999")


def test_probe_format_reads_one_header(corpus):
    assert probe_format(corpus / "dev-clean") == (24000, 1, 2)


def test_ingest_emits_one_record_per_utterance(corpus):
    assert len(list(ingest_subset(corpus, "dev-clean"))) == 4


def test_speaker_id_is_parsed_from_the_filename(corpus):
    speakers = {r.speaker_id for r in ingest_subset(corpus, "dev-clean")}
    assert speakers == {"103", "1034"}


def test_duration_derived_from_file_size_is_accurate(corpus):
    """The size-based shortcut must agree with the header to well under a frame."""
    for record in ingest_subset(corpus, "dev-clean"):
        with wave.open(str(corpus / record.audio_path), "rb") as handle:
            true_duration = handle.getnframes() / handle.getframerate()
        assert abs(record.duration_s - true_duration) < 0.01


def test_licence_is_attached_to_every_record(corpus):
    for record in ingest_subset(corpus, "dev-clean"):
        assert record.licence_spdx == LICENCE_SPDX == "CC-BY-4.0"
        assert record.licence_url
        assert record.source == SOURCE


def test_utterance_without_a_transcript_is_skipped(corpus):
    chapter = corpus / "dev-clean" / "103" / "1240"
    _write_wav(chapter / "103_1240_999999_000000.wav")
    assert len(list(ingest_subset(corpus, "dev-clean"))) == 4


def test_utterance_with_an_empty_transcript_is_skipped(corpus):
    chapter = corpus / "dev-clean" / "103" / "1240"
    _write_wav(chapter / "103_1240_888888_000000.wav")
    (chapter / "103_1240_888888_000000.normalized.txt").write_text("   ")
    assert len(list(ingest_subset(corpus, "dev-clean"))) == 4


def test_ids_are_stable_across_runs(corpus):
    first = [r.utterance_id for r in ingest_subset(corpus, "dev-clean")]
    second = [r.utterance_id for r in ingest_subset(corpus, "dev-clean")]
    assert first == second == sorted(set(first), key=first.index)


def test_missing_subset_raises(corpus):
    with pytest.raises(FileNotFoundError):
        list(ingest_subset(corpus, "test-clean"))
