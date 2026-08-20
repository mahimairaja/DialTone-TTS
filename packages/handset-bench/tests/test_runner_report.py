"""Tests for the runner, the result record, and scorecard rendering."""

import json
import time

import numpy as np
import pytest
from handset_bench.adapters.base import SynthResult, Timings
from handset_bench.conditions import CONDITIONS
from handset_bench.metrics.wer import PARAKEET_MODEL
from handset_bench.report import (
    drop_size,
    render_csv,
    render_numbers_to_beat,
    render_scorecard,
)
from handset_bench.runner import (
    ModeConflict,
    run_latency,
    run_quality,
    unavailable_record,
    write_record,
)
from handset_bench.textset.loader import Utterance


class EchoAdapter:
    """Returns silence, and its paired ASR echoes the reference exactly."""

    def __init__(self, sample_rate: int = 22050, version: str = "echo-1.0.0"):
        self.sample_rate = sample_rate
        self._version = version

    def version_string(self) -> str:
        return self._version

    def synthesize(self, text: str, *, voice: str | None = None) -> SynthResult:
        now = time.perf_counter_ns()
        return SynthResult(
            pcm=np.zeros(self.sample_rate // 2, dtype=np.float32),
            sample_rate=self.sample_rate,
            timings=Timings(now, now + 5_000_000, now + 9_000_000),
            status="ok",
            metadata={"text": text},
        )


class PerfectASR:
    """A listener that always hears exactly what was written."""

    def __init__(self, utterances):
        self._texts = [u.text for u in utterances]
        self._i = 0

    def transcribe(self, pcm, sample_rate: int) -> str:
        text = self._texts[self._i % len(self._texts)]
        self._i += 1
        return text

    def name(self) -> str:
        return "perfect-asr"

    def revision(self) -> str:
        return "test"


class DeafASR(PerfectASR):
    def transcribe(self, pcm, sample_rate: int) -> str:
        self._i += 1
        return "completely wrong words here"


UTTERANCES = [
    Utterance("dt-0001", "the committee reviewed all the submissions", "general", 6),
    Utterance("dt-0002", "she placed the book on the table", "general", 7),
]


def test_quality_run_refuses_to_collect_timings():
    """Timing and quality are separate runs. Measuring both distorts the timings."""
    with pytest.raises(ModeConflict):
        run_quality(
            EchoAdapter(),
            system="echo",
            condition=CONDITIONS["clean"],
            utterances=UTTERANCES,
            asr=PerfectASR(UTTERANCES),
            collect_timings=True,
        )


def test_perfect_transcription_scores_zero():
    record = run_quality(
        EchoAdapter(),
        system="echo",
        condition=CONDITIONS["clean"],
        utterances=UTTERANCES,
        asr=PerfectASR(UTTERANCES),
    )
    assert record.aggregate["wer"] == 0.0
    assert record.aggregate["failure_rate"] == 0.0


def test_wrong_transcription_scores_above_zero():
    record = run_quality(
        EchoAdapter(),
        system="echo",
        condition=CONDITIONS["clean"],
        utterances=UTTERANCES,
        asr=DeafASR(UTTERANCES),
    )
    assert record.aggregate["wer"] > 0.0


def test_record_carries_full_attribution():
    record = run_quality(
        EchoAdapter(),
        system="echo",
        condition=CONDITIONS["clean"],
        utterances=UTTERANCES,
        asr=PerfectASR(UTTERANCES),
    )
    assert record.textset_hash.startswith("sha256:")
    assert record.asr_backend == "perfect-asr"
    assert record.asr_revision == "test"
    assert record.git_describe
    assert record.run_id
    assert record.started_at


def test_record_has_no_wer_wideband_field():
    """wideband is its own condition record. One source of truth per number."""
    record = run_quality(
        EchoAdapter(),
        system="echo",
        condition=CONDITIONS["clean"],
        utterances=UTTERANCES,
        asr=PerfectASR(UTTERANCES),
    )
    assert "wer_wideband" not in record.aggregate
    assert "wer_wideband" not in json.loads(record.to_json())["aggregate"]


def test_missing_version_string_aborts_the_run():
    class NoVersion(EchoAdapter):
        def version_string(self) -> str:
            return ""

    from handset_bench.adapters.base import MissingVersionString

    with pytest.raises(MissingVersionString):
        run_quality(
            NoVersion(),
            system="x",
            condition=CONDITIONS["clean"],
            utterances=UTTERANCES,
            asr=PerfectASR(UTTERANCES),
        )


def test_condition_description_travels_with_the_record():
    """The degradation is stated wherever the number appears."""
    record = run_quality(
        EchoAdapter(),
        system="echo",
        condition=CONDITIONS["loss_3pct"],
        utterances=UTTERANCES,
        asr=PerfectASR(UTTERANCES),
    )
    assert "20ms" in record.condition_description


def test_latency_run_reports_generation_side_percentiles():
    record = run_latency(
        EchoAdapter(),
        system="echo",
        condition=CONDITIONS["clean"],
        utterances=UTTERANCES,
    )
    assert "ttfb_generation_p50_ms" in record.aggregate
    assert "ttfb_p50_ms" not in record.aggregate
    assert record.asr_backend == "none"


def test_record_path_layout(tmp_path):
    record = run_quality(
        EchoAdapter(version="1.2.0"),
        system="piper",
        condition=CONDITIONS["clean"],
        utterances=UTTERANCES,
        asr=PerfectASR(UTTERANCES),
    )
    path = write_record(record, tmp_path)
    assert path.relative_to(tmp_path).as_posix() == "piper/1.2.0/clean.json"


def test_latency_record_does_not_overwrite_the_quality_record(tmp_path):
    quality = run_quality(
        EchoAdapter(version="1.2.0"),
        system="piper",
        condition=CONDITIONS["clean"],
        utterances=UTTERANCES,
        asr=PerfectASR(UTTERANCES),
    )
    latency = run_latency(
        EchoAdapter(version="1.2.0"),
        system="piper",
        condition=CONDITIONS["clean"],
        utterances=UTTERANCES,
    )
    assert write_record(quality, tmp_path) != write_record(latency, tmp_path)


# ------------------------------------------------------------------ report


def _records(tmp_path):
    out = []
    for condition in ("wideband", "clean", "loss_1pct", "loss_3pct"):
        record = run_quality(
            EchoAdapter(version="1.2.0"),
            system="piper",
            condition=CONDITIONS[condition],
            utterances=UTTERANCES,
            asr=PerfectASR(UTTERANCES),
        )
        out.append(json.loads(record.to_json()))
    latency = run_latency(
        EchoAdapter(version="1.2.0"),
        system="piper",
        condition=CONDITIONS["clean"],
        utterances=UTTERANCES,
    )
    out.append(json.loads(latency.to_json()))
    return out


def test_drop_size_is_clean_minus_wideband():
    """The size of the drop from wideband must be visible."""
    assert abs(drop_size(0.0412, 0.0193) - 0.0219) < 1e-9


def test_scorecard_states_textset_listener_and_versions(tmp_path):
    """The text set, the listener, and every system version are recorded."""
    md = render_scorecard(_records(tmp_path))
    assert "dialtone_v1" in md
    assert "perfect-asr" in md
    assert "1.2.0" in md


def test_scorecard_labels_latency_as_generation_side(tmp_path):
    md = render_scorecard(_records(tmp_path))
    assert "ttfb_generation_ms" in md
    assert "not an end-to-end figure" in md


def test_scorecard_marks_an_unavailable_system_with_a_reason():
    """An unavailable system is a stated row with a reason, never a missing one."""
    record = json.loads(
        unavailable_record("cosyvoice", "clean", "image build failed").to_json()
    )
    md = render_scorecard([record])
    assert "n/a" in md.lower()
    assert "image build failed" in md


def test_numbers_to_beat_names_the_system_that_set_each(tmp_path):
    """Each best figure names the system that set it."""
    md = render_numbers_to_beat(_records(tmp_path))
    assert "set by" in md.lower()
    assert "piper" in md


def test_numbers_to_beat_records_the_pending_manual_gate(tmp_path):
    """An autonomous run must not imply a listening check was performed."""
    md = render_numbers_to_beat(_records(tmp_path))
    assert "NOT been performed" in md


def test_adding_a_system_does_not_change_existing_rows(tmp_path):
    """Adding a system appends a row and leaves the existing ones byte-identical."""
    base = _records(tmp_path)
    before = render_csv(base)
    extra = json.loads(unavailable_record("newsystem", "clean", "not run").to_json())
    after = render_csv([*base, extra])
    for line in before.splitlines()[1:]:
        assert line in after


def test_csv_has_a_stable_header(tmp_path):
    header = render_csv(_records(tmp_path)).splitlines()[0]
    assert header.startswith("system,version,condition,mode,status,wer")


def test_scorecard_names_the_listener_from_the_records_not_a_default(tmp_path):
    """The harness must never claim a listener that did not produce the numbers."""
    records = _records(tmp_path)
    md = render_scorecard(records)
    assert "perfect-asr" in md
    assert "nvidia/parakeet-tdt-0.6b-v2" not in md.split("Designated headline")[0]


def test_scorecard_flags_that_the_headline_listener_was_not_run(tmp_path):
    md = render_scorecard(_records(tmp_path))
    assert "has NOT been" in md


def test_each_listener_gets_its_own_table(tmp_path):
    """Two listeners must not collide: keying without one silently drops it."""
    base = _records(tmp_path)
    parakeet = [
        {**json.loads(json.dumps(r)), "asr_backend": PARAKEET_MODEL}
        for r in base
        if r.get("mode") == "quality"
    ]
    md = render_scorecard([*base, *parakeet])
    assert md.count("## Word error rate by condition, listener") == 2
    assert PARAKEET_MODEL in md


def test_scorecard_says_when_records_span_several_runs(tmp_path):
    """Two runs an hour apart on the same day are still two runs."""
    base = _records(tmp_path)
    stale = json.loads(json.dumps(base[0]))
    stale["started_at"] = "2020-01-01T00:00:00+00:00"
    md = render_scorecard([*base, stale])
    assert "more than\none run" in md or "more than one run" in md
