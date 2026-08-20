"""Assembly of result records from a generation manifest plus transcripts."""

from __future__ import annotations

from collections.abc import Sequence

from handset_bench import versioning
from handset_bench.conditions import CONDITIONS
from handset_bench.metrics import wer as wer_metrics
from handset_bench.textset.loader import TEXTSET_SHA256, Utterance

__all__ = ["build_quality_record", "build_latency_record"]


def build_quality_record(
    *,
    system: str,
    version: str,
    condition: str,
    utterances: Sequence[Utterance],
    hypotheses: dict[str, str],
    generation: dict[str, dict],
    asr_backend: str,
    asr_revision: str,
    modal_environment: str = "modal",
) -> dict:
    """One `(system, version, condition)` quality record."""
    references: list[str] = []
    hyps: list[str] = []
    statuses: list[str] = []
    per_utterance: list[dict] = []

    for utterance in utterances:
        generated = generation.get(utterance.utterance_id, {})
        gen_status = generated.get("status", "error")
        hypothesis = hypotheses.get(utterance.utterance_id, "")

        if gen_status != "ok":
            status = gen_status
        elif not hypothesis.strip():
            status = "empty"
        else:
            status = "ok"

        errors, ref_words = wer_metrics.utterance_errors(
            utterance.text, hypothesis, status
        )
        references.append(utterance.text)
        hyps.append(hypothesis)
        statuses.append(status)
        per_utterance.append(
            {
                "utterance_id": utterance.utterance_id,
                "status": status,
                "ref_words": ref_words,
                "errors": errors,
                "wer": errors / ref_words if ref_words else 0.0,
                "error": generated.get("error"),
                # The transcript is the expensive artefact: it costs GPU time and
                # Stored so a scoring change is re-scored offline, not re-run.
                "hypothesis": hypothesis,
                "reference": utterance.text,
            }
        )

    started = versioning.iso_now()
    return {
        "system": system,
        "version": version,
        "condition": condition,
        "mode": "quality",
        "status": "ok",
        "textset_hash": f"sha256:{TEXTSET_SHA256}",
        "asr_backend": asr_backend,
        "asr_revision": asr_revision,
        "run_id": versioning.run_id(system, version, condition, started),
        "started_at": started,
        "git_describe": versioning.git_describe(),
        "modal_environment": modal_environment,
        "condition_description": CONDITIONS[condition].description,
        "per_utterance": per_utterance,
        "aggregate": {
            "wer": wer_metrics.corpus_wer(references, hyps, statuses),
            "failure_rate": sum(s != "ok" for s in statuses) / len(statuses),
            "n_utterances": float(len(statuses)),
        },
    }


def build_latency_record(
    *,
    system: str,
    version: str,
    generation: dict[str, dict],
    modal_environment: str = "modal",
) -> dict:
    """Generation-side latency, taken from the generation pass timings."""
    from handset_bench.metrics.latency import percentile

    values = [
        entry["ttfb_generation_ms"]
        for entry in generation.values()
        if entry.get("status") == "ok"
    ]
    started = versioning.iso_now()
    record = {
        "system": system,
        "version": version,
        "condition": "clean",
        "mode": "latency",
        "status": "ok" if values else "unavailable",
        "textset_hash": f"sha256:{TEXTSET_SHA256}",
        "asr_backend": "none",
        "asr_revision": "none",
        "run_id": versioning.run_id(system, version, "clean-latency", started),
        "started_at": started,
        "git_describe": versioning.git_describe(),
        "modal_environment": modal_environment,
        "condition_description": (
            "Generation-side timing only. No phone line and no transcription "
            "involved: the codec chain runs after generation and cannot affect "
            "time to first audio."
        ),
        "per_utterance": [],
        "aggregate": {},
    }
    if values:
        record["aggregate"] = {
            "ttfb_generation_p50_ms": percentile(values, 0.50),
            "ttfb_generation_p95_ms": percentile(values, 0.95),
            "n_utterances": float(len(values)),
        }
    else:
        record["unavailable_reason"] = "no successful generations to time"
    return record
