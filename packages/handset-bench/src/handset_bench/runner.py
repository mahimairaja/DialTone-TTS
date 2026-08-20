"""Matrix orchestration and the result record."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import torch

from handset_bench import codec, versioning
from handset_bench.adapters.base import validate_adapter
from handset_bench.conditions import Condition, apply
from handset_bench.metrics import latency as latency_metrics
from handset_bench.metrics import wer as wer_metrics
from handset_bench.textset.loader import Utterance

__all__ = [
    "Mode",
    "ModeConflict",
    "ResultRecord",
    "UtteranceResult",
    "run_quality",
    "unavailable_record",
    "write_record",
]

Mode = Literal["quality", "latency", "concurrency"]


class ModeConflict(RuntimeError):
    """Raised when a run is asked to measure timing and quality at once."""


@dataclass(frozen=True)
class UtteranceResult:
    utterance_id: str
    status: str
    #: Reference word count after normalisation.
    ref_words: int | None = None
    wer: float | None = None
    ttfb_generation_ms: float | None = None
    audio_path: str | None = None
    error: str | None = None


@dataclass
class ResultRecord:
    """One `(system, version, condition)` cell."""

    system: str
    version: str
    condition: str
    mode: Mode
    textset_hash: str
    asr_backend: str
    asr_revision: str
    run_id: str
    started_at: str
    git_describe: str
    modal_environment: str = "local"
    status: str = "ok"
    unavailable_reason: str | None = None
    per_utterance: list[UtteranceResult] = field(default_factory=list)
    aggregate: dict[str, float] = field(default_factory=dict)
    condition_description: str = ""

    def to_json(self) -> str:
        payload = asdict(self)
        return json.dumps(payload, indent=2, sort_keys=True)


def _seed_for(utterance: Utterance, condition: Condition, run_seed: int) -> int:
    return codec.condition_seed(utterance.utterance_id, condition.name, run_seed)


def run_quality(
    adapter,
    *,
    system: str,
    condition: Condition,
    utterances: Sequence[Utterance],
    asr,
    run_seed: int = 0,
    collect_timings: bool = False,
    audio_dir: Path | None = None,
    modal_environment: str = "local",
) -> ResultRecord:
    """Score one system on one condition."""
    if collect_timings:
        raise ModeConflict(
            "a quality run cannot collect timings. Timing and quality are measured "
            "in separate runs because scoring competes with generation for the GPU "
            "and distorts the timings."
        )

    validate_adapter(adapter)
    version = adapter.version_string()
    started = versioning.iso_now()

    from handset_bench.textset.loader import TEXTSET_SHA256

    record = ResultRecord(
        system=system,
        version=version,
        condition=condition.name,
        mode="quality",
        textset_hash=f"sha256:{TEXTSET_SHA256}",
        asr_backend=asr.name(),
        asr_revision=asr.revision(),
        run_id=versioning.run_id(system, version, condition.name, started),
        started_at=started,
        git_describe=versioning.git_describe(),
        modal_environment=modal_environment,
        condition_description=condition.description,
    )

    references: list[str] = []
    hypotheses: list[str] = []
    statuses: list[str] = []

    for utterance in utterances:
        result = adapter.synthesize(utterance.text)
        if result.status != "ok":
            references.append(utterance.text)
            hypotheses.append("")
            statuses.append(result.status)
            record.per_utterance.append(
                UtteranceResult(
                    utterance_id=utterance.utterance_id,
                    status=result.status,
                    wer=1.0,
                    error=result.error,
                )
            )
            continue

        pcm = torch.from_numpy(result.pcm.copy()).float()
        conditioned = apply(
            condition,
            pcm,
            result.sample_rate,
            seed=_seed_for(utterance, condition, run_seed),
        )
        for_asr = codec.to_asr_rate(conditioned.pcm, conditioned.sample_rate)
        hypothesis = asr.transcribe(for_asr.numpy(), codec.ASR_RATE)

        errors, ref_words = wer_metrics.utterance_errors(
            utterance.text, hypothesis, "ok"
        )
        audio_path = None
        if audio_dir is not None:
            audio_path = str(
                audio_dir / f"{system}_{condition.name}_{utterance.utterance_id}.wav"
            )

        references.append(utterance.text)
        hypotheses.append(hypothesis)
        statuses.append("ok")
        record.per_utterance.append(
            UtteranceResult(
                utterance_id=utterance.utterance_id,
                status="ok",
                wer=errors / ref_words if ref_words else 0.0,
                audio_path=audio_path,
            )
        )

    record.aggregate = {
        "wer": wer_metrics.corpus_wer(references, hypotheses, statuses),
        "failure_rate": sum(s != "ok" for s in statuses) / len(statuses),
        "n_utterances": float(len(statuses)),
    }
    return record


def run_latency(
    adapter,
    *,
    system: str,
    condition: Condition,
    utterances: Sequence[Utterance],
    modal_environment: str = "local",
    warmup: int = 2,
) -> ResultRecord:
    """Measure generation-side latency only. No transcription happens here."""
    validate_adapter(adapter)
    for utterance in utterances[:warmup]:
        adapter.synthesize(utterance.text)
    version = adapter.version_string()
    started = versioning.iso_now()

    from handset_bench.textset.loader import TEXTSET_SHA256

    record = ResultRecord(
        system=system,
        version=version,
        condition=condition.name,
        mode="latency",
        textset_hash=f"sha256:{TEXTSET_SHA256}",
        asr_backend="none",
        asr_revision="none",
        run_id=versioning.run_id(system, version, f"{condition.name}-lat", started),
        started_at=started,
        git_describe=versioning.git_describe(),
        modal_environment=modal_environment,
        condition_description=condition.description,
    )

    timings = []
    failures = 0
    for utterance in utterances:
        result = adapter.synthesize(utterance.text)
        if result.status != "ok":
            failures += 1
            record.per_utterance.append(
                UtteranceResult(
                    utterance_id=utterance.utterance_id,
                    status=result.status,
                    error=result.error,
                )
            )
            continue
        timings.append(result.timings)
        record.per_utterance.append(
            UtteranceResult(
                utterance_id=utterance.utterance_id,
                status="ok",
                ttfb_generation_ms=result.timings.ttfb_generation_ms,
            )
        )

    if timings:
        agg = latency_metrics.aggregate(timings)
        record.aggregate = {
            "ttfb_generation_p50_ms": agg.ttfb_generation_p50_ms,
            "ttfb_generation_p95_ms": agg.ttfb_generation_p95_ms,
            "failure_rate": failures / len(utterances),
            "n_utterances": float(len(utterances)),
        }
    else:
        record.status = "unavailable"
        record.unavailable_reason = "every synthesis request failed"
    return record


def unavailable_record(
    system: str, condition: str, reason: str, *, version: str = "unavailable"
) -> ResultRecord:
    """A system that could not be run at all."""
    started = versioning.iso_now()
    from handset_bench.textset.loader import TEXTSET_SHA256

    return ResultRecord(
        system=system,
        version=version,
        condition=condition,
        mode="quality",
        textset_hash=f"sha256:{TEXTSET_SHA256}",
        asr_backend="none",
        asr_revision="none",
        run_id=versioning.run_id(system, version, condition, started),
        started_at=started,
        git_describe=versioning.git_describe(),
        status="unavailable",
        unavailable_reason=reason,
    )


def write_record(record: ResultRecord, root: Path) -> Path:
    """Write to `results/<system>/<version>/<condition>.json`."""
    safe_version = record.version.replace("/", "_").replace(" ", "_")
    directory = root / record.system / safe_version
    directory.mkdir(parents=True, exist_ok=True)
    suffix = "" if record.mode == "quality" else f".{record.mode}"
    path = directory / f"{record.condition}{suffix}.json"
    path.write_text(record.to_json(), encoding="utf-8")
    return path
