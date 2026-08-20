"""Word error rate."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import jiwer

from handset_bench.asr_models import (
    PARAKEET_MODEL,
    PARAKEET_REVISION,
    WHISPER_MODEL,
    WHISPER_REVISION,
)

__all__ = [
    "PARAKEET_MODEL",
    "PARAKEET_REVISION",
    "WHISPER_MODEL",
    "WHISPER_REVISION",
    "ASRBackend",
    "WerBreakdown",
    "corpus_wer",
    "normalize",
    "utterance_errors",
]


class ASRBackend(Protocol):
    """A listener. Greedy decode only, so a rerun reproduces the transcript."""

    def transcribe(self, pcm, sample_rate: int) -> str: ...

    def name(self) -> str: ...

    def revision(self) -> str: ...


@dataclass(frozen=True)
class WerBreakdown:
    errors: int
    reference_words: int

    @property
    def rate(self) -> float:
        return self.errors / self.reference_words if self.reference_words else 0.0


@lru_cache(maxsize=1)
def _normalizer():
    from whisper_normalizer.english import EnglishTextNormalizer

    return EnglishTextNormalizer()


#: An ordinal is kept whole; any other digit is split out on its own. Whisper turns
_DIGIT_RUN = re.compile(r"\d+(?:st|nd|rd|th)\b|\d")
_NON_WORD = re.compile(r"[^\w\s]")

#: Brackets are stripped before normalisation. Whisper's normaliser treats a
_BRACKET = re.compile(r"[()\[\]{}]")

#: Single-digit words mapped to numerals. Whisper's normaliser is context-dependent:
_DIGIT_WORDS = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}

#: "2:15 pm" normalises to "pm" while "two fifteen p m" normalises to "p m".
_MERIDIEM = re.compile(r"\b[ap] m\b")


def normalize(text: str) -> str:
    """Whisper's English normalisation, plus digit atomisation, on both sides."""
    # Split digits before the normaliser too: it reads "0198" as a number and
    # drops the leading zero, and it deletes bracketed spans like "(613)".
    presplit = _BRACKET.sub(" ", text)
    presplit = _DIGIT_RUN.sub(lambda m: f" {m.group()} ", presplit)
    normalised = _normalizer()(presplit)
    stripped = _NON_WORD.sub(" ", normalised)
    atomised = _DIGIT_RUN.sub(lambda m: f" {m.group()} ", stripped)
    tokens = [_DIGIT_WORDS.get(tok, tok) for tok in atomised.split()]
    return _MERIDIEM.sub(lambda m: m.group().replace(" ", ""), " ".join(tokens))


def utterance_errors(reference: str, hypothesis: str, status: str) -> tuple[int, int]:
    """Return (errors, reference_words) for one utterance."""
    ref = normalize(reference)
    ref_words = len(ref.split())

    if status != "ok":
        # A total failure, never a silent skip.
        return ref_words, ref_words

    hyp = normalize(hypothesis)
    if not ref_words:
        return 0, 0

    out = jiwer.process_words([ref], [hyp])
    errors = out.substitutions + out.deletions + out.insertions
    return errors, ref_words


def corpus_wer(
    references: list[str],
    hypotheses: list[str],
    statuses: list[str],
) -> float:
    """Corpus-level word error rate: total errors over total reference words."""
    if not (len(references) == len(hypotheses) == len(statuses)):
        raise ValueError(
            "references, hypotheses and statuses must have equal length, got "
            f"{len(references)}, {len(hypotheses)}, {len(statuses)}"
        )
    if not references:
        raise ValueError(
            "cannot score an empty run: returning 0.0 would look like a perfect score"
        )

    total_errors = 0
    total_words = 0
    for ref, hyp, status in zip(references, hypotheses, statuses, strict=True):
        errors, words = utterance_errors(ref, hyp, status)
        total_errors += errors
        total_words += words

    if total_words == 0:
        raise ValueError("reference corpus contains no words after normalisation")
    return total_errors / total_words
