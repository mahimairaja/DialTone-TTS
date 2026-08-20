# DialTone

Telephony-native text to speech. Most TTS is built for 24kHz headphones and then
squeezed down a phone line: 8kHz, G.711 mu-law, a 300 to 3400 Hz passband, and
packet loss. DialTone measures what that squeeze costs and replaces the part of
the stack that pays for it.

Two packages, one uv workspace:

| Package | What it does |
| --- | --- |
| `handset-bench` | The benchmark. Synthesises a fixed 300-utterance text set, pushes it through a real G.711 codec chain, transcribes it, and scores word error rate and latency per condition |
| `dialtone` | The corpus tooling and the narrowband vocoder that replaces the stock 24kHz one |

`handset-bench` does not import `dialtone`. The benchmark has to be able to score
a system it knows nothing about, so the dependency only runs one way.

## Status

The baseline is partial. The harness runs end to end on Modal; the full scorecard
has not been produced. See [docs/baseline-findings.md](docs/baseline-findings.md),
which records a measured result that contradicts the project's original latency
assumption.

## Running it

Requires [uv](https://docs.astral.sh/uv/) and a logged-in
[Modal](https://modal.com) CLI.

```bash
uv sync --all-packages
./scripts/run_baseline.sh                     # smoke run, 12 utterances
./scripts/run_baseline.sh --full              # all 300
./scripts/run_baseline.sh --full --skip-ingest  # corpus already on the Volume
```

The first run downloads roughly 10GB of LibriTTS-R and walks about 44,000 files
on a network volume: budget 30 to 60 minutes for that stage alone. Everything
after it caches on Modal Volumes, so later runs pass `--skip-ingest` and start at
the matrix. The matrix refuses to launch when its own cost estimate exceeds
`--max-cost` (default $30).

Outputs land in `results/`, which is not tracked. What is meant to survive a run
lives in `baselines.yaml` (the pinned systems, listeners and settings) and in
`docs/`.

## Reproducibility

[docs/REPRODUCING.md](docs/REPRODUCING.md) lists everything pinned and why.
[DATA_CARD.md](DATA_CARD.md) records corpus provenance, including which corpora
were excluded on licence grounds. `manifests/split_v1.json` is the frozen speaker
split; `configs/voices.yaml` is the fixed voice and prompt selection.

Two gates are manual and are never passed by a script: listening to one sample
per system per condition, and listening to augmented clips to confirm they sound
like a phone call. A word error rate can look plausible while the audio is
broken.

## Licence

MIT. See [LICENSE](LICENSE).
