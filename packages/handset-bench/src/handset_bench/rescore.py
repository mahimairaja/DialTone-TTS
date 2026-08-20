"""Re-score existing result records without re-running the benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from handset_bench.metrics import wer as wer_metrics

__all__ = ["rescore_file", "rescore_tree"]


class NoStoredTranscripts(RuntimeError):
    """The record predates transcript storage, so its score cannot be recomputed."""


def rescore_file(path: Path, *, write: bool = False) -> dict:
    """Recompute one record's scores from its stored transcripts."""
    record = json.loads(path.read_text())
    if record.get("mode") != "quality":
        return {"path": str(path), "skipped": "not a quality record"}

    rows = record["per_utterance"]
    missing = [r for r in rows if "hypothesis" not in r]
    if missing:
        raise NoStoredTranscripts(
            f"{path} has {len(missing)} utterances with no stored transcript. "
            "Records written before transcript storage cannot be re-scored; "
            "re-run the matrix instead."
        )

    references = [r["reference"] for r in rows]
    hypotheses = [r["hypothesis"] for r in rows]
    statuses = [r["status"] for r in rows]

    before = record["aggregate"]["wer"]
    for row in rows:
        errors, ref_words = wer_metrics.utterance_errors(
            row["reference"], row["hypothesis"], row["status"]
        )
        row["errors"] = errors
        row["ref_words"] = ref_words
        row["wer"] = errors / ref_words if ref_words else 0.0

    after = wer_metrics.corpus_wer(references, hypotheses, statuses)
    record["aggregate"]["wer"] = after

    if write:
        path.write_text(json.dumps(record, indent=2, sort_keys=True))

    return {
        "path": str(path),
        "system": record["system"],
        "condition": record["condition"],
        "before": before,
        "after": after,
        "delta_pp": (after - before) * 100,
    }


def rescore_tree(root: Path, *, write: bool = False) -> list[dict]:
    """Re-score every quality record under `root`."""
    out = []
    for path in sorted(root.glob("*/*/*.json")):
        result = rescore_file(path, write=write)
        if "skipped" not in result:
            out.append(result)
    return out
