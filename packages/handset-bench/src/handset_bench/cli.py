"""Command line entry points."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from handset_bench import report as report_mod
from handset_bench.conditions import CONDITIONS

app = typer.Typer(help="Score TTS systems under telephone-line conditions.")
console = Console()

#: Rough per-unit costs used only for the pre-flight estimate.
_COST_PER_UTTERANCE_USD = {
    "generate_cpu": 0.00005,
    "generate_gpu": 0.00030,
    "transcribe_gpu": 0.00025,
}


def estimate_cost(n_utterances: int, n_conditions: int, mode: str) -> float:
    """A conservative upper estimate in US dollars."""
    if mode == "quality":
        return n_utterances * (
            _COST_PER_UTTERANCE_USD["generate_cpu"]
            + n_conditions * _COST_PER_UTTERANCE_USD["transcribe_gpu"]
        )
    if mode == "latency":
        return n_utterances * _COST_PER_UTTERANCE_USD["generate_cpu"]
    if mode == "concurrency":
        return n_utterances * _COST_PER_UTTERANCE_USD["generate_gpu"] * 4
    raise typer.BadParameter(f"unknown mode {mode!r}")


@app.command()
def run(
    system: str = typer.Option(..., help="System under test, e.g. piper"),
    condition: str = typer.Option("all", help="Condition name or 'all'"),
    mode: str = typer.Option("quality", help="quality, latency, or concurrency"),
    max_cost_usd: float = typer.Option(
        5.0, help="Refuse to launch if the estimate exceeds this"
    ),
    limit: int = typer.Option(0, help="Score only the first N utterances"),
) -> None:
    """Estimate the cost, then hand off to the Modal app."""
    from handset_bench.textset import loader

    if mode not in ("quality", "latency", "concurrency"):
        raise typer.BadParameter(f"unknown mode {mode!r}")
    if condition != "all" and condition not in CONDITIONS:
        raise typer.BadParameter(
            f"unknown condition {condition!r}. Known: {sorted(CONDITIONS)}"
        )

    n_utterances = limit or len(loader.load())
    n_conditions = len(CONDITIONS) if condition == "all" else 1
    estimate = estimate_cost(n_utterances, n_conditions, mode)

    console.print(
        f"[bold]{system}[/bold] / {condition} / {mode}: "
        f"{n_utterances} utterances, estimated ${estimate:.2f}"
    )
    if estimate > max_cost_usd:
        console.print(
            f"[red]Refusing to launch.[/red] Estimate ${estimate:.2f} exceeds "
            f"--max-cost-usd ${max_cost_usd:.2f}."
        )
        raise typer.Exit(code=2)

    console.print(f"Launch with:\n  modal run modal/bench_app.py --limit {limit}")


@app.command()
def report(
    results: Path = typer.Option(Path("results"), help="Results directory"),
    write: bool = typer.Option(True, help="Write SCORECARD.md and scorecard.csv"),
) -> None:
    """Render the scorecard from stored result records."""
    records = report_mod.load_records(results)
    if not records:
        console.print(f"[red]No result records found under {results}[/red]")
        raise typer.Exit(code=1)

    table = Table(title="handset-bench")
    for column in ("system", "version", "condition", "mode", "WER %"):
        table.add_column(column)
    for record in sorted(
        records, key=lambda r: (r["system"], r["condition"], r.get("mode", ""))
    ):
        wer = record.get("aggregate", {}).get("wer")
        table.add_row(
            record["system"],
            record["version"],
            record["condition"],
            record.get("mode", "quality"),
            "n/a" if wer is None else f"{wer * 100:.2f}",
        )
    console.print(table)

    if write:
        (results / "SCORECARD.md").write_text(report_mod.render_scorecard(records))
        (results / "scorecard.csv").write_text(report_mod.render_csv(records))
        (results / "NUMBERS_TO_BEAT.md").write_text(
            report_mod.render_numbers_to_beat(records)
        )
        console.print(f"wrote {results / 'SCORECARD.md'}")


@app.command()
def verify(
    results: Path = typer.Option(Path("results"), help="Results directory"),
    tolerance_pp: float = typer.Option(0.1, help="Allowed WER drift, points"),
) -> None:
    """Report which entries would be withheld for drifting beyond tolerance."""
    records = report_mod.load_records(results)
    console.print(
        f"{len(records)} records loaded. Reproducibility tolerance "
        f"{tolerance_pp} percentage points."
    )
    for record in records:
        console.print(
            f"  {record['system']}/{record['condition']}: "
            f"{json.dumps(record.get('aggregate', {}))}"
        )


@app.command()
def rescore(
    results: Path = typer.Option(Path("results"), help="Results directory"),
    write: bool = typer.Option(False, help="Write the recomputed scores back"),
) -> None:
    """Recompute scores from stored transcripts, without re-running the benchmark."""
    from handset_bench.rescore import NoStoredTranscripts, rescore_tree

    try:
        rows = rescore_tree(results, write=write)
    except NoStoredTranscripts as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    table = Table(title="re-scored")
    for column in ("system", "condition", "before %", "after %", "delta pp"):
        table.add_column(column)
    for row in rows:
        table.add_row(
            row["system"],
            row["condition"],
            f"{row['before'] * 100:.2f}",
            f"{row['after'] * 100:.2f}",
            f"{row['delta_pp']:+.2f}",
        )
    console.print(table)
    if not write:
        console.print("[yellow]dry run. Pass --write to update the records.[/yellow]")


if __name__ == "__main__":
    app()
