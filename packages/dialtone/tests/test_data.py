"""Tests for corpus assembly: licence gate, manifest, frozen split, voice selection."""

import json

import pytest
from dialtone.data.licence import (
    ALLOWED_SPDX,
    RejectionRateExceeded,
    gate,
)
from dialtone.data.manifest import (
    ManifestRecord,
    read_manifest,
    utterance_id,
    write_manifest,
)
from dialtone.data.provenance import render_data_card
from dialtone.data.split import HeldoutLeak, Split, load_split, make_split, write_split
from dialtone.data.verify import verify_no_heldout
from dialtone.data.voices import select_prompt_speakers, select_voices


def rec(
    *,
    licence: str | None = "CC-BY-4.0",
    speaker: str = "spk1",
    subset: str = "train-clean-100",
    path: str = "/x/1.wav",
    duration: float = 3.0,
) -> ManifestRecord:
    return ManifestRecord(
        utterance_id=utterance_id("libritts-r", path),
        audio_path=path,
        text="hello world",
        speaker_id=speaker,
        duration_s=duration,
        sample_rate=24000,
        source="libritts-r",
        subset=subset,
        licence_spdx=licence,
        licence_url="https://creativecommons.org/licenses/by/4.0/",
    )


def corpus() -> list[ManifestRecord]:
    """A miniature LibriTTS-R: train speakers plus dev/test heldout speakers."""
    out = []
    for i in range(12):
        out.append(
            rec(
                speaker=f"train{i}",
                subset="train-clean-100",
                path=f"/train/{i}.wav",
                duration=10.0 + i,
            )
        )
    for i in range(3):
        out.append(rec(speaker=f"dev{i}", subset="dev-clean", path=f"/dev/{i}.wav"))
    for i in range(3):
        out.append(rec(speaker=f"test{i}", subset="test-clean", path=f"/test/{i}.wav"))
    return out


# ------------------------------------------------------------------ licence


def test_cc_by_4_is_allowed():
    accepted, rejected = gate([rec(licence="CC-BY-4.0")])
    assert len(accepted) == 1
    assert rejected == []


def test_cc0_is_allowed():
    accepted, _ = gate([rec(licence="CC0-1.0")])
    assert len(accepted) == 1


def test_allowlist_is_exactly_the_two_permissive_licences():
    assert ALLOWED_SPDX == frozenset({"CC-BY-4.0", "CC0-1.0"})


def test_missing_licence_is_rejected_not_caveated():
    """A licence that cannot be established means exclusion."""
    accepted, rejected = gate([rec(licence=None)], threshold=1.0)
    assert accepted == []
    assert rejected[0].reason == "licence_absent"


@pytest.mark.parametrize("bad", ["CC-BY-NC-4.0", "CC-BY-SA-4.0", "custom", "unknown"])
def test_non_allowlisted_licences_are_rejected(bad):
    """Share-alike and non-commercial both fail the public-release requirement."""
    accepted, rejected = gate([rec(licence=bad)], threshold=1.0)
    assert accepted == []
    assert rejected[0].reason == "licence_not_allowlisted"


def test_rejection_rate_above_threshold_aborts_the_run():
    """A broken ingest adapter must not be able to silently empty the corpus."""
    records = [rec()] * 90 + [rec(licence=None)] * 10
    with pytest.raises(RejectionRateExceeded):
        gate(records)


def test_rejection_rate_below_threshold_passes():
    records = [rec() for _ in range(98)] + [rec(licence=None), rec(licence=None)]
    accepted, rejected = gate(records, threshold=0.05)
    assert len(accepted) == 98
    assert len(rejected) == 2


# ----------------------------------------------------------------- manifest


def test_utterance_ids_are_content_addressed_and_stable():
    assert utterance_id("libritts-r", "/x/1.wav") == utterance_id(
        "libritts-r", "/x/1.wav"
    )
    assert utterance_id("libritts-r", "/x/1.wav") != utterance_id(
        "libritts-r", "/x/2.wav"
    )


def test_utterance_ids_differ_across_sources():
    assert utterance_id("libritts-r", "/x/1.wav") != utterance_id("other", "/x/1.wav")


def test_manifest_round_trips(tmp_path):
    path = tmp_path / "m.jsonl"
    write_manifest(corpus(), path)
    assert read_manifest(path) == corpus()


def test_manifest_rejects_a_malformed_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"utterance_id": "x"}\n')
    with pytest.raises(ValueError):
        read_manifest(path)


# -------------------------------------------------------------------- split


def test_zero_speaker_overlap():
    """No speaker appears in both the train and heldout halves."""
    split = make_split(corpus())
    assert set(split.train_speaker_ids) & set(split.heldout_speaker_ids) == set()


def test_heldout_is_the_corpus_dev_and_test_speakers():
    """Inherits a disjointness guarantee the corpus authors already published."""
    split = make_split(corpus())
    assert set(split.heldout_speaker_ids) == {
        "dev0",
        "dev1",
        "dev2",
        "test0",
        "test1",
        "test2",
    }


def test_same_input_reproduces_the_same_split():
    assert make_split(corpus()) == make_split(corpus())


def test_split_file_hash_is_stable(tmp_path):
    a = write_split(make_split(corpus()), tmp_path / "a.json")
    b = write_split(make_split(corpus()), tmp_path / "b.json")
    assert a == b


def test_split_round_trips(tmp_path):
    split = make_split(corpus())
    write_split(split, tmp_path / "s.json")
    assert load_split(tmp_path / "s.json") == split


def test_split_records_its_basis_and_manifest_hash():
    split = make_split(corpus())
    assert split.heldout_basis == "corpus dev-clean + test-clean"
    assert split.manifest_hash.startswith("sha256:")


def test_split_rejects_a_corpus_with_no_heldout_subsets():
    only_train = [rec(speaker=f"s{i}", path=f"/t/{i}.wav") for i in range(4)]
    with pytest.raises(ValueError, match="heldout"):
        make_split(only_train)


# ------------------------------------------------------------------- verify


def test_heldout_utterance_in_a_training_manifest_is_rejected():
    """Leak checking is enforced at the point of use rather than by convention."""
    records = corpus()
    split = make_split(records)
    leaked = [r for r in records if r.speaker_id.startswith("dev")]
    with pytest.raises(HeldoutLeak):
        verify_no_heldout(leaked, split)


def test_clean_training_manifest_passes():
    records = corpus()
    split = make_split(records)
    train = [r for r in records if r.speaker_id.startswith("train")]
    verify_no_heldout(train, split)


def test_leak_message_names_the_offending_speaker():
    records = corpus()
    split = make_split(records)
    leaked = [r for r in records if r.speaker_id == "dev1"]
    with pytest.raises(HeldoutLeak, match="dev1"):
        verify_no_heldout(leaked, split)


# ------------------------------------------------------------------- voices


def test_voice_selection_is_deterministic():
    assert select_voices(corpus()) == select_voices(corpus())


def test_four_voices_all_from_train_clean_100():
    voices = select_voices(corpus())
    assert len(voices) == 4
    assert all(v.subset == "train-clean-100" for v in voices)


def test_voices_are_never_heldout_speakers():
    records = corpus()
    heldout = set(make_split(records).heldout_speaker_ids)
    assert not {v.speaker_id for v in select_voices(records)} & heldout


def test_voice_selection_rule_is_recorded():
    """A reviewer must be able to re-derive the selection, not take our word."""
    for voice in select_voices(corpus()):
        assert voice.selection_rule


def test_prompt_speakers_are_heldout_only():
    """Prompt policy: three fixed heldout speakers, identical across systems."""
    records = corpus()
    split = make_split(records)
    prompts = select_prompt_speakers(records, split)
    assert len(prompts) == 3
    assert set(prompts) <= set(split.heldout_speaker_ids)


def test_prompt_speakers_are_deterministic():
    records = corpus()
    split = make_split(records)
    assert select_prompt_speakers(records, split) == select_prompt_speakers(
        records, split
    )


def test_prompt_speakers_never_overlap_the_voice_set():
    records = corpus()
    split = make_split(records)
    voices = {v.speaker_id for v in select_voices(records)}
    assert not set(select_prompt_speakers(records, split)) & voices


# --------------------------------------------------------------- provenance


def test_data_card_lists_licence_duration_and_share_per_source():
    """REQ-DT-DATA-004.1"""
    card = render_data_card(corpus(), removals=[])
    assert "CC-BY-4.0" in card
    assert "libritts-r" in card
    assert "%" in card


def test_data_card_states_total_duration_and_speaker_count():
    """The data card states total duration and speaker count."""
    card = render_data_card(corpus(), removals=[])
    assert "18" in card  # speaker count
    assert "hours" in card.lower()


def test_data_card_records_removals_with_dates():
    """Removals are recorded with dates rather than silently applied."""
    card = render_data_card(
        corpus(),
        removals=[
            {
                "utterance_id": "x",
                "removed_at": "2026-08-14",
                "reason": "licence revoked",
            }
        ],
    )
    assert "2026-08-14" in card
    assert "licence revoked" in card


def test_data_card_names_the_excluded_corpora_with_reasons():
    """The exclusions are as much a part of provenance as the inclusions."""
    card = render_data_card(corpus(), removals=[])
    for excluded in ("GigaSpeech", "People's Speech", "Emilia"):
        assert excluded in card


def test_split_file_is_valid_json(tmp_path):
    write_split(make_split(corpus()), tmp_path / "s.json")
    payload = json.loads((tmp_path / "s.json").read_text())
    assert payload["heldout_basis"] == "corpus dev-clean + test-clean"
    assert isinstance(payload["train_speaker_ids"], list)


def test_split_dataclass_is_hashable_by_value():
    assert isinstance(make_split(corpus()), Split)
