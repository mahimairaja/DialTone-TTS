# DialTone

A benchmark for text to speech that has to survive a phone line: 8kHz, G.711 mu-law, a
300 to 3400 Hz passband, packet loss. Most TTS is built for 24kHz headphones and
evaluated there. This measures what the squeeze actually costs.

It was started to justify building a telephony-native vocoder. The measurement says
that vocoder is not worth building, at least not for intelligibility, so it was not
built. The benchmark is the result.

Two packages, one uv workspace:

| Package | What it does |
| --- | --- |
| `handset-bench` | The benchmark. Synthesises a fixed 300-utterance text set, pushes it through a real G.711 codec chain, transcribes it, and scores word error rate and latency per condition |
| `dialtone` | Corpus tooling: licence gating, the frozen split, provenance. Also an untrained narrowband vocoder, kept because the differentiable codec layer in it is useful on its own |

`handset-bench` does not import `dialtone`. The benchmark has to be able to score
a system it knows nothing about, so the dependency only runs one way.

## Explainer

<!-- Replace VIDEO_ID once the video is uploaded. GitHub cannot embed a player in a
     README, so this renders as a clickable thumbnail. -->
[![DialTone explainer](https://img.youtube.com/vi/VIDEO_ID/maxresdefault.jpg)](https://www.youtube.com/watch?v=VIDEO_ID)

Five minutes on what a phone line does to speech, the scoring bug that nearly
invalidated the whole benchmark, and the measurement that killed the design it was
meant to validate. Animated with Manim, narrated locally with Kokoro-82M; sources in
`video/`, regenerate with `./video/render.sh -qh`.

## What it found

Both systems measured over 300 utterances and four conditions, reproducible to 0.0
percentage points, and listened to.

| System | wideband | clean | loss 1% | loss 3% | latency p50 | hardware |
| --- | --- | --- | --- | --- | --- | --- |
| piper `+det` | 3.48 | **3.06** | 3.13 | 3.24 | 158 ms | CPU |
| zipvoice_distill `+det` | 4.68 | 4.27 | 4.34 | 4.30 | 342 ms | A10G |

Three results, in order of how much they matter.

**The phone line does not cost intelligibility. It improves it.** Both systems score
0.41 points *better* after the G.711 chain than before it, and on both, `datetime_money`
carries it: 9 errors to 1 for Piper, 38 to 27 for ZipVoice. Two unrelated
architectures, the same improvement, the same category. That points at the recogniser,
not either model: band-limiting to 300-3400 Hz removes high-frequency content that
faster-whisper trips over on times and amounts.

**Chunking at punctuation is strictly harmful.** It saves 30 ms on time to first audio
and costs 678 ms on total generation. ZipVoice's per-call cost barely depends on text
length, so splitting a turn into three chunks pays that cost three times. Realtime
factor is 3.8x to 4.9x, not the ~150x the design assumed.

**The small CPU model wins on both axes.** Piper beats a 123M flow-matching model on
GPU by 1.21 points of word error rate and by more than double on latency, at no GPU
cost. ZipVoice's advantage is voice cloning, which this benchmark does not score.

The consequence is that this project's original premise does not survive its own
benchmark. DialTone was built on the idea that the telephone line destroys quality and
a telephony-native model would recover it. For intelligibility there is nothing to
recover. See [docs/baseline-findings.md](docs/baseline-findings.md).

## Status

The benchmark is complete and reproducible. **No vocoder has been trained.** The code
in `packages/dialtone/src/dialtone/vocoder/` is architecture and a differentiable codec
layer, verified by tests, with no training run and no checkpoint behind it. It stayed
that way deliberately: the measurement above removed the case for training it.

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

Two consecutive runs of the same system produced **1,200 identical transcripts**, so
the band is 0.0 points. Getting there required making synthesis deterministic: both
Piper and ZipVoice sampled noise per call and produced different audio every run, which
made the benchmark compare transcripts of different recordings without anything looking
wrong. System versions carry `+det` or `+sampled` because the waveform differs between
them.

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
