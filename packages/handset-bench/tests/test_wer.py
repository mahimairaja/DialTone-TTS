"""Tests for word error rate."""

import pytest
from handset_bench.metrics.wer import corpus_wer, normalize, utterance_errors


def test_perfect_match_is_zero():
    assert corpus_wer(["hello world"], ["hello world"], ["ok"]) == 0.0


def test_one_substitution_in_four_words():
    assert abs(corpus_wer(["a b c d"], ["a b x d"], ["ok"]) - 0.25) < 1e-9


def test_one_deletion_in_four_words():
    assert abs(corpus_wer(["a b c d"], ["a b d"], ["ok"]) - 0.25) < 1e-9


def test_one_insertion_in_four_words():
    assert abs(corpus_wer(["a b c d"], ["a b c x d"], ["ok"]) - 0.25) < 1e-9


def test_empty_status_counts_every_reference_word_as_an_error():
    """A system producing no audio is a total failure, not a skip."""
    assert corpus_wer(["a b c d"], [""], ["empty"]) == 1.0


def test_error_status_counts_as_a_total_failure():
    assert corpus_wer(["a b"], [""], ["error"]) == 1.0


def test_a_failure_cannot_be_hidden_by_a_success():
    """Two utterances, one perfect and one absent, is 50% not 0%."""
    assert abs(corpus_wer(["a b", "c d"], ["a b", ""], ["ok", "empty"]) - 0.5) < 1e-9


def test_corpus_level_differs_from_the_per_utterance_mean():
    """Aggregation is total errors over total reference words."""
    refs = ["x", " ".join(["w"] * 99)]
    hyps = ["y", " ".join(["w"] * 99)]
    assert abs(corpus_wer(refs, hyps, ["ok", "ok"]) - 0.01) < 1e-9


def test_normalisation_is_applied_to_both_sides():
    assert corpus_wer(["Hello, World!"], ["hello world"], ["ok"]) == 0.0


def test_normalize_lowercases_and_strips_punctuation():
    out = normalize("Hello, World!")
    assert out == out.lower()
    assert "," not in out and "!" not in out


def test_normalize_is_idempotent():
    once = normalize("The total is $1,247.50 on March 3rd.")
    assert normalize(once) == once


def test_normalize_collapses_whitespace():
    assert normalize("hello    world") == normalize("hello world")


def test_utterance_errors_returns_counts_not_a_rate():
    """corpus_wer needs raw counts to aggregate correctly."""
    errors, ref_words = utterance_errors("a b c d", "a b x d", "ok")
    assert errors == 1
    assert ref_words == 4


def test_utterance_errors_for_a_failure_is_all_reference_words():
    errors, ref_words = utterance_errors("a b c d", "", "error")
    assert errors == 4
    assert ref_words == 4


# ------------------------------------------------- numeric form convergence


@pytest.mark.parametrize(
    ("reference", "spoken"),
    [
        (
            "Please call back on 6 1 3 5 5 5 0 1 9 8.",
            "please call back on six one three five five five zero one nine eight",
        ),
        (
            "Your confirmation code is 4 7 B 2 9 K.",
            "your confirmation code is four seven b two nine k",
        ),
        ("The security code is 9 0 5 5.", "the security code is nine zero five five"),
        (
            "The total comes to $1,247.50.",
            "the total comes to one thousand two hundred forty seven dollars "
            "and fifty cents",
        ),
        (
            "The total comes to $1,247.50.",
            "the total comes to twelve forty seven fifty",
        ),
        (
            "Your appointment is March 3rd at 2:15 pm.",
            "your appointment is march third at two fifteen p m",
        ),
        (
            "The deadline is the 31st of January, 2027.",
            "the deadline is the thirty first of january twenty twenty seven",
        ),
        (
            "The address is 1427 Elm Street, apartment 3B.",
            "the address is fourteen twenty seven elm street apartment three b",
        ),
    ],
)
def test_written_and_spoken_numeric_forms_converge(reference, spoken):
    """A correct transcription must score zero regardless of which form it uses."""
    assert normalize(reference) == normalize(spoken)
    assert corpus_wer([reference], [spoken], ["ok"]) == 0.0


def test_a_real_digit_error_still_counts():
    """Convergence must not be achieved by flattening genuine mistakes."""
    wer = corpus_wer(
        ["The security code is 9 0 5 5."],
        ["the security code is nine zero five six"],
        ["ok"],
    )
    assert wer > 0.0


def test_a_real_word_substitution_still_counts():
    wer = corpus_wer(
        ["She placed the book on the table."],
        ["she placed a book on the table"],
        ["ok"],
    )
    assert wer > 0.0


def test_digits_are_atomised_so_each_digit_costs_one_error():
    """A phone agent cares per digit, so that is the granularity errors use."""
    assert normalize("9055") == "9 0 5 5"


def test_ordinals_are_kept_whole():
    """Atomising 1st into '1 st' would make one misheard ordinal cost two errors."""
    assert normalize("the first attempt") == "the 1st attempt"
    assert normalize("March 3rd") == "march 3rd"


def test_normalisation_is_a_no_op_for_prose():
    """Comparability with published prose-based figures is preserved."""
    text = "the committee reviewed all the submissions before deciding"
    assert normalize(text) == text


def test_every_textset_utterance_normalises_to_a_stable_canonical_form():
    from handset_bench.textset import loader

    for utterance in loader.load():
        once = normalize(utterance.text)
        assert normalize(once) == once, utterance.text
        assert "$" not in once and "¢" not in once


def test_mismatched_input_lengths_are_rejected():
    with pytest.raises(ValueError):
        corpus_wer(["a"], ["a", "b"], ["ok", "ok"])


def test_empty_reference_corpus_is_rejected():
    """Returning 0.0 for an empty run would look like a perfect score."""
    with pytest.raises(ValueError):
        corpus_wer([], [], [])


@pytest.mark.parametrize(
    "hypothesis",
    [
        "613 555 0198",
        "613-555-0198",
        "6135550198",
        "(613) 555-0198",
        "six one three five five five zero one nine eight",
        "6 1 3 5 5 5 0 1 9 8",
    ],
)
def test_every_written_form_of_a_phone_number_converges(hypothesis):
    """A correct transcription must score zero however the ASR chose to write it."""
    reference = "six one three five five five zero one nine eight"
    errors, ref_words = utterance_errors(reference, hypothesis, "ok")
    assert (errors, ref_words) == (0, 10)


def test_leading_zeros_survive_normalisation():
    assert normalize("007") == "0 0 7"
    assert normalize("0500") == "0 5 0 0"
