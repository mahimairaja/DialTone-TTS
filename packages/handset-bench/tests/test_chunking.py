"""Tests for sentence chunking and the chunked driver."""

import time

import numpy as np
import pytest
from handset_bench.adapters.base import SynthResult, Timings
from handset_bench.chunking import ChunkedDriver, split_sentences


class SlowFixtureAdapter:
    """Takes measurable time per call so ordering assertions are meaningful."""

    def __init__(self, delay_s: float = 0.01, sample_rate: int = 8000):
        self.delay_s = delay_s
        self.sample_rate = sample_rate
        self.calls = 0

    def version_string(self) -> str:
        return "slow-fixture-1.0.0"

    def synthesize(self, text: str, *, voice: str | None = None) -> SynthResult:
        submitted = time.perf_counter_ns()
        time.sleep(self.delay_s)
        first = time.perf_counter_ns()
        self.calls += 1
        return SynthResult(
            pcm=np.zeros(self.sample_rate // 10, dtype=np.float32),
            sample_rate=self.sample_rate,
            timings=Timings(submitted, first, time.perf_counter_ns()),
            status="ok",
        )


# ---------------------------------------------------------------- splitter


def test_single_sentence_produces_exactly_one_chunk():
    """The wrapper must add zero overhead in the common case."""
    assert split_sentences("Your appointment is confirmed.") == [
        "Your appointment is confirmed."
    ]


def test_title_abbreviations_do_not_split():
    assert len(split_sentences("Dr. Smith will call you back.")) == 1


def test_street_abbreviations_do_not_split():
    assert len(split_sentences("The office is on Elm St. near the park.")) == 1


def test_decimals_do_not_split():
    assert len(split_sentences("The total is 3.14 dollars.")) == 1


def test_currency_with_decimals_does_not_split():
    assert len(split_sentences("That comes to $1,247.50 today.")) == 1


def test_ellipsis_does_not_split_into_three():
    assert len(split_sentences("Let me check... one moment.")) <= 2


def test_no_terminal_punctuation_still_yields_one_chunk():
    assert split_sentences("hello there") == ["hello there"]


def test_two_sentences_split():
    assert len(split_sentences("First one. Second one.")) == 2


def test_question_and_exclamation_split():
    assert len(split_sentences("Are you there? Yes I am!")) == 2


def test_empty_text_yields_no_chunks():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_max_chars_is_respected():
    long_text = " ".join(["word"] * 300)
    for chunk in split_sentences(long_text, max_chars=200):
        assert len(chunk) <= 200


def test_long_sentence_is_split_on_word_boundaries():
    long_text = " ".join(["word"] * 300)
    for chunk in split_sentences(long_text, max_chars=200):
        assert not chunk.startswith(" ")
        assert "wordword" not in chunk


def test_no_text_is_lost_when_splitting():
    text = "First sentence here. Second sentence here. Third one."
    assert "".join(split_sentences(text)).replace(" ", "") == text.replace(" ", "")


# ---------------------------------------------------------------- driver


def test_driver_reports_first_chunk_before_full_message():
    """First-chunk and whole-message figures are reported separately."""
    driver = ChunkedDriver(SlowFixtureAdapter(delay_s=0.02))
    result = driver.synthesize_chunked("First one. Second one. Third one.")
    assert result.ttfb_first_chunk_ms < result.ttft_full_message_ms


def test_driver_calls_the_adapter_once_per_chunk():
    adapter = SlowFixtureAdapter(delay_s=0.001)
    ChunkedDriver(adapter).synthesize_chunked("One. Two. Three.")
    assert adapter.calls == 3


def test_driver_on_a_single_sentence_makes_exactly_one_call():
    adapter = SlowFixtureAdapter(delay_s=0.001)
    ChunkedDriver(adapter).synthesize_chunked("Only one sentence here.")
    assert adapter.calls == 1


def test_driver_concatenates_audio_in_order():
    driver = ChunkedDriver(SlowFixtureAdapter(delay_s=0.001))
    result = driver.synthesize_chunked("One. Two.")
    assert result.pcm.size == 2 * (8000 // 10)


def test_driver_rejects_mixed_sample_rates():
    class WobblyAdapter(SlowFixtureAdapter):
        def synthesize(self, text, *, voice=None):
            self.sample_rate = 8000 if self.calls % 2 == 0 else 16000
            return super().synthesize(text, voice=voice)

    with pytest.raises(ValueError, match="sample rate"):
        ChunkedDriver(WobblyAdapter(delay_s=0.001)).synthesize_chunked("One. Two.")


def test_driver_propagates_a_failed_chunk_as_a_failed_result():
    class FailingAdapter(SlowFixtureAdapter):
        def synthesize(self, text, *, voice=None):
            self.calls += 1
            now = time.perf_counter_ns()
            return SynthResult(
                pcm=np.zeros(0, dtype=np.float32),
                sample_rate=8000,
                timings=Timings(now, now, now),
                status="error",
                error="boom",
            )

    result = ChunkedDriver(FailingAdapter()).synthesize_chunked("One. Two.")
    assert result.status == "error"
