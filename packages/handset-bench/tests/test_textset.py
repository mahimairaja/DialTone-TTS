"""Tests for the frozen text set."""

from collections import Counter

import pytest
from handset_bench.textset import loader
from handset_bench.textset.build_v1 import EXPECTED_COUNTS


def test_three_hundred_utterances():
    assert len(loader.load()) == 300


def test_category_counts_match_the_spec():
    assert Counter(u.category for u in loader.load()) == EXPECTED_COUNTS


def test_no_duplicate_ids():
    ids = [u.utterance_id for u in loader.load()]
    assert len(set(ids)) == len(ids)


def test_no_duplicate_texts():
    """A repeated utterance would weight one sentence twice in the corpus WER."""
    texts = [u.text for u in loader.load()]
    assert len(set(texts)) == len(texts)


def test_ids_are_sequential_and_zero_padded():
    ids = [u.utterance_id for u in loader.load()]
    assert ids[0] == "dt-0001"
    assert ids[-1] == "dt-0300"


def test_word_counts_are_consistent():
    for u in loader.load():
        assert u.n_words == len(u.text.split())


def test_corpus_is_large_enough_to_resolve_the_reproducibility_band():
    """Two runs must agree within 0.1 percentage points."""
    total = sum(u.n_words for u in loader.load())
    assert total > 2000
    assert 100.0 / total < 0.05


def test_no_em_dashes():
    for u in loader.load():
        assert "—" not in u.text
        assert "–" not in u.text


def test_hash_matches_the_pinned_constant():
    assert loader.textset_hash() == loader.TEXTSET_SHA256


def test_hash_mismatch_aborts(tmp_path):
    """A silently edited text set would make every published number wrong."""
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        '{"utterance_id":"x","text":"hi","category":"general","n_words":1}\n'
    )
    with pytest.raises(loader.TextsetHashMismatch):
        loader.load(bad)


def test_verify_false_allows_an_unpinned_file(tmp_path):
    """Only the builder and the tests may bypass the gate."""
    bad = tmp_path / "other.jsonl"
    bad.write_text(
        '{"utterance_id":"x","text":"hi","category":"general","n_words":1}\n'
    )
    assert len(loader.load(bad, verify=False)) == 1


def test_missing_file_reports_how_to_build_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_v1"):
        loader.load(tmp_path / "absent.jsonl")


def test_malformed_line_reports_its_line_number(tmp_path):
    bad = tmp_path / "broken.jsonl"
    bad.write_text('{"utterance_id":"x"}\n')
    with pytest.raises(ValueError, match=":1"):
        loader.load(bad, verify=False)


def test_builder_is_reproducible(tmp_path):
    """The file is committed data, but its construction must be re-derivable."""
    from handset_bench.textset.build_v1 import write

    first = write(tmp_path / "a.jsonl")
    second = write(tmp_path / "b.jsonl")
    assert first == second == loader.TEXTSET_SHA256


def test_sample_spreads_across_categories():
    """A smoke run must exercise every category, not just the first one."""
    utterances = loader.load()
    picked = loader.sample(utterances, 12)

    assert len(picked) == 12
    categories = {u.category for u in picked}
    assert categories == {u.category for u in utterances}


def test_sample_is_a_no_op_for_the_full_set():
    utterances = loader.load()
    assert loader.sample(utterances, 0) == utterances
    assert loader.sample(utterances, len(utterances)) == utterances


def test_sample_is_deterministic():
    utterances = loader.load()
    assert loader.sample(utterances, 18) == loader.sample(utterances, 18)
