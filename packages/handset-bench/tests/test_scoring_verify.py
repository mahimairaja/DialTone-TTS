"""Tests for record assembly and the reproducibility gate."""

import pytest
from handset_bench.scoring import build_latency_record, build_quality_record
from handset_bench.textset.loader import Utterance
from handset_bench.verify_repro import (
    DEFAULT_TOLERANCE_PP,
    verify_entry,
    write_withheld,
)

UTTERANCES = [
    Utterance("dt-0001", "the committee reviewed all the submissions", "general", 6),
    Utterance("dt-0002", "she placed the book on the table", "general", 7),
]

GEN_OK = {
    "dt-0001": {
        "status": "ok",
        "error": None,
        "ttfb_generation_ms": 40.0,
        "total_ms": 90.0,
        "audio_seconds": 2.0,
    },
    "dt-0002": {
        "status": "ok",
        "error": None,
        "ttfb_generation_ms": 60.0,
        "total_ms": 120.0,
        "audio_seconds": 2.0,
    },
}


def _record(hypotheses, generation=None):
    return build_quality_record(
        system="fake",
        version="1.0.0",
        condition="clean",
        utterances=UTTERANCES,
        hypotheses=hypotheses,
        generation=generation or GEN_OK,
        asr_backend="fake-asr",
        asr_revision="0",
    )


def test_perfect_transcription_scores_zero():
    record = _record({u.utterance_id: u.text for u in UTTERANCES})
    assert record["aggregate"]["wer"] == 0.0
    assert record["aggregate"]["failure_rate"] == 0.0


def test_missing_hypothesis_counts_as_a_total_failure():
    """A silent skip would flatter whichever system fails most often."""
    record = _record({"dt-0001": UTTERANCES[0].text, "dt-0002": ""})
    assert record["aggregate"]["wer"] > 0.0
    assert record["per_utterance"][1]["status"] == "empty"
    assert record["per_utterance"][1]["wer"] == 1.0


def test_failed_generation_is_marked_with_its_generation_status():
    generation = dict(GEN_OK)
    generation["dt-0002"] = {
        "status": "error",
        "error": "boom",
        "ttfb_generation_ms": 0.0,
        "total_ms": 0.0,
        "audio_seconds": 0.0,
    }
    record = _record({"dt-0001": UTTERANCES[0].text}, generation)
    assert record["per_utterance"][1]["status"] == "error"
    assert record["per_utterance"][1]["error"] == "boom"


def test_every_utterance_carries_ref_words_for_category_aggregation():
    record = _record({u.utterance_id: u.text for u in UTTERANCES})
    assert all(entry["ref_words"] > 0 for entry in record["per_utterance"])


def test_record_has_full_attribution():
    record = _record({u.utterance_id: u.text for u in UTTERANCES})
    for key in (
        "textset_hash",
        "asr_backend",
        "asr_revision",
        "git_describe",
        "run_id",
        "started_at",
        "condition_description",
    ):
        assert record[key]


def test_record_has_no_wer_wideband_field():
    record = _record({u.utterance_id: u.text for u in UTTERANCES})
    assert "wer_wideband" not in record["aggregate"]


def test_latency_record_uses_generation_timings():
    record = build_latency_record(system="fake", version="1.0.0", generation=GEN_OK)
    assert record["mode"] == "latency"
    assert record["asr_backend"] == "none"
    assert 40.0 <= record["aggregate"]["ttfb_generation_p50_ms"] <= 60.0
    assert "ttfb_p50_ms" not in record["aggregate"]


def test_latency_record_is_unavailable_when_nothing_generated():
    generation = {
        "dt-0001": {
            "status": "error",
            "error": "boom",
            "ttfb_generation_ms": 0.0,
            "total_ms": 0.0,
            "audio_seconds": 0.0,
        }
    }
    record = build_latency_record(system="fake", version="1.0.0", generation=generation)
    assert record["status"] == "unavailable"
    assert record["unavailable_reason"]


# --------------------------------------------------------------- verify gate


def test_tolerance_is_the_documented_band():
    assert DEFAULT_TOLERANCE_PP == 0.1


def test_drift_above_tolerance_withholds_the_entry():
    outcome = verify_entry("piper", "clean", stored_wer=0.0412, rerun_wer=0.0430)
    assert outcome.withheld is True
    assert "0.1" in outcome.reason


def test_drift_below_tolerance_passes():
    outcome = verify_entry("piper", "clean", stored_wer=0.0412, rerun_wer=0.0417)
    assert outcome.withheld is False


def test_drift_exactly_at_tolerance_passes():
    outcome = verify_entry("piper", "clean", stored_wer=0.0400, rerun_wer=0.0410)
    assert abs(outcome.drift_pp - 0.1) < 1e-9
    assert outcome.withheld is False


def test_withheld_entries_are_written_outside_the_scorecard(tmp_path):
    """Never publish with a caveat: a caveat gets dropped when quoted."""
    outcome = verify_entry("piper", "clean", stored_wer=0.04, rerun_wer=0.09)
    path = write_withheld(outcome, tmp_path)
    assert path.parent.name == "withheld"
    assert "drift_pp" in path.read_text()


def test_report_ignores_withheld_records(tmp_path):
    from handset_bench.report import load_records

    outcome = verify_entry("piper", "clean", stored_wer=0.04, rerun_wer=0.09)
    write_withheld(outcome, tmp_path)
    assert load_records(tmp_path) == []


@pytest.mark.parametrize("stored,rerun", [(0.05, 0.05), (0.0, 0.0)])
def test_identical_runs_never_withhold(stored, rerun):
    assert verify_entry("x", "clean", stored, rerun).withheld is False


def test_load_records_ignores_non_record_json(tmp_path):
    """results/ holds more than records: CHUNKING.json lives there too."""
    import json as _json

    from handset_bench.report import load_records

    (tmp_path / "piper" / "1.0.0").mkdir(parents=True)
    (tmp_path / "piper" / "1.0.0" / "clean.json").write_text(
        _json.dumps({"system": "piper", "condition": "clean", "aggregate": {}})
    )
    (tmp_path / "CHUNKING.json").write_text(
        _json.dumps({"status": "ok", "rows": [], "median_ttfb_first_chunk_ms": 1})
    )
    records = load_records(tmp_path)
    assert len(records) == 1
    assert records[0]["system"] == "piper"


def test_load_records_skips_malformed_json(tmp_path):
    from handset_bench.report import load_records

    (tmp_path / "broken.json").write_text("{not json")
    assert load_records(tmp_path) == []
