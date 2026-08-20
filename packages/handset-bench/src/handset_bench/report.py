"""Scorecard rendering."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Sequence
from pathlib import Path

from handset_bench.asr_models import PARAKEET_MODEL, WHISPER_MODEL
from handset_bench.textset.loader import TEXTSET_NAME

__all__ = [
    "drop_size",
    "load_records",
    "render_csv",
    "render_numbers_to_beat",
    "render_scorecard",
]

_NA = "n/a"

#: The latency figure this harness produces, and what it does not prove.
_LATENCY_CAVEAT = (
    "Latency is `ttfb_generation_ms`: wall clock from handing text to the adapter "
    "until the first audio frame, measured in a warm process. It is not an "
    "end-to-end figure. The project's 150ms target additionally includes network, "
    "warm-pool queueing, and framing, and is only measurable against a deployed "
    "serving stack. A system passing the target here has not yet passed it there."
)


def drop_size(clean_wer: float, wideband_wer: float) -> float:
    """How much the phone line costs."""
    return clean_wer - wideband_wer


#: Keys every result record carries. Used to tell records apart from the other
_RECORD_KEYS = ("system", "condition")


def load_records(root: Path) -> list[dict]:
    """Load result records from `results/`, ignoring anything else."""
    records: list[dict] = []
    for path in sorted(root.rglob("*.json")):
        if path.parent.name == "withheld":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and all(k in payload for k in _RECORD_KEYS):
            records.append(payload)
    return records


def _fmt_pct(value: float | None) -> str:
    return _NA if value is None else f"{value * 100:.2f}"


def _fmt_ms(value: float | None) -> str:
    return _NA if value is None else f"{value:.1f}"


def _index(records: Sequence[dict]) -> dict[tuple[str, str, str, str, str], dict]:
    # The listener is part of the key. Without it a second listener overwrites the
    # first and the scorecard silently reports whichever record sorted last.
    return {
        (
            r["system"],
            r["version"],
            r["condition"],
            r.get("mode", "quality"),
            r.get("asr_backend", "none"),
        ): r
        for r in records
    }


def listeners_in(records: Sequence[dict]) -> list[str]:
    """Quality listeners present in the records, headline first."""
    seen = {
        r["asr_backend"]
        for r in records
        if r.get("mode") == "quality" and r.get("asr_backend") not in (None, "none")
    }
    return sorted(seen, key=lambda name: (PARAKEET_MODEL not in name, name))


def run_span_hours(records: Sequence[dict]) -> float:
    """Hours between the oldest and newest record, so a blended scorecard can say so."""
    from datetime import datetime

    stamps = []
    for record in records:
        raw = record.get("started_at")
        if not raw:
            continue
        try:
            stamps.append(datetime.fromisoformat(raw))
        except ValueError:
            continue
    if len(stamps) < 2:
        return 0.0
    return (max(stamps) - min(stamps)).total_seconds() / 3600.0


def _systems(records: Sequence[dict]) -> list[tuple[str, str]]:
    seen: list[tuple[str, str]] = []
    for record in records:
        key = (record["system"], record["version"])
        if key not in seen:
            seen.append(key)
    return sorted(seen)


def _category_table(records: Sequence[dict], listener: str | None = None) -> list[str]:
    """Per-category corpus WER for one listener, over stored reference word counts."""
    if listener is not None:
        records = [r for r in records if r.get("asr_backend") == listener]
    try:
        from handset_bench.textset import loader

        category_of = {u.utterance_id: u.category for u in loader.load()}
    except Exception:  # noqa: BLE001 - the table is optional, the scorecard is not
        return []

    conditions = ("wideband", "clean", "loss_1pct", "loss_3pct")
    per_condition: dict[str, dict[str, tuple[float, float]]] = {}

    for record in records:
        if record.get("mode", "quality") != "quality" or record.get("status") != "ok":
            continue
        totals: dict[str, list[float]] = {}
        for entry in record.get("per_utterance", []):
            ref_words = entry.get("ref_words")
            if not ref_words:
                return []  # older records lack the field; skip rather than mislead
            category = category_of.get(entry["utterance_id"], "unknown")
            bucket = totals.setdefault(category, [0.0, 0.0])
            bucket[0] += (entry.get("wer") or 0.0) * ref_words
            bucket[1] += ref_words
        per_condition[record["condition"]] = {
            k: (v[0], v[1]) for k, v in totals.items()
        }

    if "clean" not in per_condition or "wideband" not in per_condition:
        return []

    lines = [
        "",
        f"## Word error rate by category, listener `{listener or 'all'}`",
        "",
        "The aggregate can hide opposite effects that cancel. This table is where "
        "the phone line's real cost shows up.",
        "",
        "| Category | wideband | clean | drop | loss 3% |",
        "| --- | --- | --- | --- | --- |",
    ]
    for category in sorted(per_condition["clean"]):

        def pct(condition: str, cat: str = category) -> float | None:
            bucket = per_condition.get(condition, {}).get(cat)
            if not bucket or bucket[1] == 0:
                return None
            return bucket[0] / bucket[1]

        wideband, clean, loss3 = pct("wideband"), pct("clean"), pct("loss_3pct")
        drop = (
            f"{(clean - wideband) * 100:+.2f}"
            if clean is not None and wideband is not None
            else _NA
        )
        lines.append(
            f"| {category} | {_fmt_pct(wideband)} | {_fmt_pct(clean)} | {drop} | "
            f"{_fmt_pct(loss3)} |"
        )
    _ = conditions
    return lines


def render_scorecard(records: Sequence[dict], *, asr_backend: str | None = None) -> str:
    """Render `SCORECARD.md`."""
    index = _index(records)

    # The listener is read from the records, never assumed. Falling back to the
    observed = sorted(
        {
            r["asr_backend"]
            for r in records
            if r.get("asr_backend") and r["asr_backend"] != "none"
        }
    )
    listener = asr_backend or (
        observed[0] if len(observed) == 1 else (", ".join(observed) or "unknown")
    )

    lines: list[str] = [
        "# handset-bench scorecard",
        "",
        "Text-to-speech systems scored on what survives a G.711 telephone line.",
        "",
        f"- Text set: `{TEXTSET_NAME}`, 300 utterances",
        f"- **Listener that produced these numbers: `{listener}`**",
    ]

    # Say plainly when the designated headline listener did not produce the
    if PARAKEET_MODEL not in listener:
        lines.append(
            f"- **Designated headline listener `{PARAKEET_MODEL}` has NOT been "
            "run.** These figures come from the second-column listener only and "
            "are not the headline ranking."
        )
    else:
        lines.append(f"- Second listener: `faster-whisper {WHISPER_MODEL}`")

    span = run_span_hours(records)
    if span > 2.0:
        lines.append(
            f"- **These records span {span:.1f} hours, so they come from more than "
            "one run.** Rows from different runs are not directly comparable; rerun "
            "the whole matrix before quoting them together."
        )

    lines += [
        "",
        _LATENCY_CAVEAT,
        "",
    ]

    for listener_name in listeners_in(records) or ["unknown"]:
        lines += _wer_table(records, index, listener_name)

    headline = next(
        (name for name in listeners_in(records) if PARAKEET_MODEL in name), None
    )
    lines += _category_table(records, headline or (listeners_in(records) or [None])[0])

    lines += [
        "",
        "## Generation-side latency",
        "",
        "| System | Version | p50 ms | p95 ms |",
        "| --- | --- | --- | --- |",
    ]
    return _finish_scorecard(lines, records, index)


def _wer_table(records, index, listener_name: str) -> list[str]:
    """The word error rate table for one listener."""
    lines = [
        f"## Word error rate by condition, listener `{listener_name}`",
        "",
        "Lower is better. `wideband` is the pre-codec control; `drop` is how much "
        "the phone line costs.",
        "",
        "| System | Version | wideband | clean | drop | loss 1% | loss 3% |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for system, version in _systems(records):

        def wer_for(
            condition: str,
            _system: str = system,
            _version: str = version,
            _listener: str = listener_name,
        ) -> float | None:
            # Loop variables are bound as defaults rather than captured.
            record = index.get((_system, _version, condition, "quality", _listener))
            if record is None or record.get("status") != "ok":
                return None
            return record["aggregate"].get("wer")

        wideband, clean = wer_for("wideband"), wer_for("clean")
        drop = (
            _fmt_pct(drop_size(clean, wideband))
            if clean is not None and wideband is not None
            else _NA
        )
        unavailable = next(
            (
                r.get("unavailable_reason")
                for r in records
                if r["system"] == system and r.get("status") == "unavailable"
            ),
            None,
        )
        row = (
            f"| {system} | `{version}` | {_fmt_pct(wideband)} | {_fmt_pct(clean)} | "
            f"{drop} | {_fmt_pct(wer_for('loss_1pct'))} | "
            f"{_fmt_pct(wer_for('loss_3pct'))} |"
        )
        lines.append(row)
        if unavailable:
            # Not-applicable is stated with a reason, never left
            lines.append(f"| {system} | `{version}` | {_NA}: {unavailable} | | | | |")

    lines.append("")
    return lines


def _finish_scorecard(lines: list[str], records, index) -> str:
    """Latency table and conditions section, then join."""
    for system, version in _systems(records):
        record = index.get((system, version, "clean", "latency", "none"))
        agg = record["aggregate"] if record else {}
        lines.append(
            f"| {system} | `{version}` | "
            f"{_fmt_ms(agg.get('ttfb_generation_p50_ms'))} | "
            f"{_fmt_ms(agg.get('ttfb_generation_p95_ms'))} |"
        )

    lines += ["", "## Conditions", ""]
    described: set[str] = set()
    for record in records:
        name, description = record["condition"], record.get("condition_description")
        if description and name not in described:
            described.add(name)
            lines.append(f"- **{name}**: {description}")

    return "\n".join(lines) + "\n"


def render_csv(records: Sequence[dict]) -> str:
    """Render `scorecard.csv`. Row order is stable so a diff is readable."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "system",
            "version",
            "condition",
            "mode",
            "status",
            "wer",
            "ttfb_generation_p50_ms",
            "ttfb_generation_p95_ms",
            "failure_rate",
            "asr_backend",
            "textset_hash",
            "git_describe",
        ]
    )
    for record in sorted(
        records,
        key=lambda r: (r["system"], r["version"], r["condition"], r.get("mode", "")),
    ):
        agg = record.get("aggregate", {})
        writer.writerow(
            [
                record["system"],
                record["version"],
                record["condition"],
                record.get("mode", "quality"),
                record.get("status", "ok"),
                agg.get("wer", ""),
                agg.get("ttfb_generation_p50_ms", ""),
                agg.get("ttfb_generation_p95_ms", ""),
                agg.get("failure_rate", ""),
                record.get("asr_backend", ""),
                record.get("textset_hash", ""),
                record.get("git_describe", ""),
            ]
        )
    return buffer.getvalue()


def render_numbers_to_beat(records: Sequence[dict]) -> str:
    """Render `NUMBERS_TO_BEAT.md`, naming the system that set each figure."""
    index = _index(records)
    lines = [
        "# Numbers to beat",
        "",
        "The best value per measure across every baseline, and which system set it.",
        "Later work is measured against these with no change to how they were derived.",
        "",
        _LATENCY_CAVEAT,
        "",
    ]

    # The best figure is taken from the headline listener when it has run, so a
    # second-column number can never set a target the headline did not reach.
    headline = next(
        (name for name in listeners_in(records) if PARAKEET_MODEL in name), None
    )

    best_wer: tuple[float, str, str] | None = None
    for (system, version, condition, mode, listener), record in index.items():
        if condition != "clean" or mode != "quality" or record.get("status") != "ok":
            continue
        if headline is not None and listener != headline:
            continue
        value = record["aggregate"].get("wer")
        if value is not None and (best_wer is None or value < best_wer[0]):
            best_wer = (value, system, version)

    best_latency: tuple[float, str, str] | None = None
    for (system, version, _condition, mode, _listener), record in index.items():
        if mode != "latency" or record.get("status") != "ok":
            continue
        value = record["aggregate"].get("ttfb_generation_p95_ms")
        if value is not None and (best_latency is None or value < best_latency[0]):
            best_latency = (value, system, version)

    if best_wer:
        value, system, version = best_wer
        lines.append(
            f"- **Word error rate after the codec round trip: {value * 100:.2f}%**, "
            f"set by `{system}` at version `{version}` on condition `clean`."
        )
    else:
        lines.append(f"- Word error rate: {_NA}, no system produced a clean result.")

    if best_latency:
        value, system, version = best_latency
        lines.append(
            f"- **Generation-side time to first audio, p95: {value:.1f} ms**, "
            f"set by `{system}` at version `{version}`. This is a generation-side "
            "figure and does not by itself establish the 150ms end-to-end target, "
            "which needs a deployed serving stack to measure."
        )
    else:
        lines.append(f"- Generation-side latency: {_NA}, no latency run completed.")

    lines += [
        f"- Behaviour at 32 simultaneous calls: {_NA}, pending a concurrency run.",
        "",
        "## Pending manual gates",
        "",
        "These have NOT been performed and must not be treated as passed:",
        "",
        "- Listening to one sample per system per condition to confirm the audio is "
        "not obviously broken. A word error rate can look plausible while the audio "
        "is garbage.",
    ]
    return "\n".join(lines) + "\n"
