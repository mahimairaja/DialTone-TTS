# Reproducing the handset-bench scorecard

Every number on the scorecard is meant to be reproducible by a third party on their
own hardware. If you follow this document and get a materially different answer,
that is a bug in the harness and we want to hear about it.

## What you need

- A [Modal](https://modal.com) account. Everything runs there; nothing needs a
  local GPU.
- Python 3.12 and [uv](https://docs.astral.sh/uv/). Not 3.13 or later: the torch,
  k2, and NeMo ecosystem does not support them yet.
- No HuggingFace token. Every dataset and model used here is public and ungated.

```bash
git clone <this repo> && cd DialTone-TTS
uv sync --all-packages
uv run modal setup        # one time, opens a browser
```

## Running it

```bash
./scripts/run_baseline.sh --full --parakeet
```

That is the whole thing. It runs the local test suite, ingests the corpus, runs
the matrix cheap-first, and renders the scorecard.

For a cheap smoke run over twelve utterances, drop the flags:

```bash
./scripts/run_baseline.sh
```

## What it actually does

1. **Local tests.** No network, no GPU. If these fail, stop: the codec chain or the
   scoring rules are broken and every number downstream would be wrong.
2. **Corpus ingest.** Downloads LibriTTS-R `dev-clean`, `test-clean`, and
   `train-clean-100` from openslr, applies the licence gate, and freezes the
   speaker split. Idempotent: subsets already present are skipped.
3. **The matrix.** Generates audio once per system, derives all four conditions
   from that single waveform, transcribes each, and scores.
4. **The scorecard.** Renders `results/SCORECARD.md`, `scorecard.csv`, and
   `NUMBERS_TO_BEAT.md`.

## What is pinned, and why it matters

| Thing | Pinned as | Why |
| --- | --- | --- |
| Text set | `dialtone_v1.jsonl`, sha256 checked at load | A changed text set invalidates every prior entry. The loader refuses to run on a mismatch rather than silently shifting the numbers |
| System versions | `baselines.yaml`, never "latest" | A run that cannot be attributed to a specific version is discarded, not published |
| ASR decoding | greedy, no temperature fallback, no VAD filter | Any of these would make the transcript depend on sampling |
| Codec seed | derived from `sha256(utterance_id, condition, run_seed)` | Packet loss is deterministic per utterance and varied across the set at the same time |
| Harness revision | `git describe --tags --always --dirty` in every record | The `--dirty` suffix is deliberate: a run against uncommitted changes is not reproducible by anyone else |

## The measurement, precisely

The phone-line chain, in this order, because the order is physical:

1. Band-limit to 300-3400 Hz with an 8th-order Butterworth cascade (48 dB/octave,
   -3 dB exactly at the corner). A single biquad is 12 dB/octave and is not a
   telephone line.
2. Resample to 8000 Hz. The lowpass sits below Nyquist, so it doubles as the
   anti-alias filter.
3. G.711 mu-law encode to 8-bit codes.
4. Drop 20 ms frames independently at the condition's rate, **on the codes**,
   because packet loss happens on the wire after encoding. Dropped frames are
   filled with code 128, which is what silence encodes to.
5. Mu-law decode.

Both the codec arm and the `wideband` control are then resampled to 16 kHz for the
listener, identically. That is the point: the only difference between them is the
phone line itself.

### Normalisation is not plain Whisper

Word error rate uses Whisper's `EnglishTextNormalizer` **plus three further passes**,
applied identically to reference and hypothesis. This is not a stylistic choice.

Whisper's normaliser collapses a spelled-out digit run into a single concatenated
token, so `"six one three five five five zero one nine eight"` becomes
`6135550198`, while leaving space-separated numerals as separate tokens. A
reference written either way therefore fails to match a *perfect* transcription of
itself. Digit strings are a fifth of this text set and the most telephony-critical
part of it, so used alone the normaliser would make every system score near 100%
there regardless of audio quality.

The three passes:

1. Every maximal run of digits is split into individual digits, so all spellings
   converge and a digit error costs one word.
2. Ordinals are kept whole (`1st`, not `1 st`).
3. Single-digit words map to numerals, and `a m` / `p m` collapse to one token.

For an utterance containing no digits, all of this is a no-op, so prose figures stay
comparable to published ones.

## Verifying a published entry

```bash
uv run handset-bench verify --results results
```

Re-run any `(system, condition)` pair and compare. An entry whose aggregate WER
moves more than the band between runs is withheld to `results/withheld/` rather than
published with a caveat. A caveated number gets quoted without its caveat eventually.

**The band is currently unset, and 0.1 points is not achievable.** Measured, the same
Piper configuration run twice moved by up to 0.28 points, and three of four
conditions fell outside 0.1. The cause is the listener: on `wideband`, where no codec
and no packet loss are applied and the audio is bit-identical between runs, 63 of 300
transcripts still differed. faster-whisper on GPU is not bit-reproducible at greedy
decode. Until the listener is pinned to deterministic decoding and the band
re-measured, treat differences under roughly half a point as noise. See
`docs/baseline-findings.md`, finding 4.

## What the numbers do not say

- **Latency is generation-side.** `ttfb_generation_ms` is wall clock from handing
  text to the adapter until the first audio frame, in a warm process. It excludes
  network, warm-pool queueing, and framing. It is not an end-to-end figure and does
  not establish any end-to-end target.
- **No listening check is included.** A word error rate can look entirely plausible
  while the audio is garbage. Sample audio should be listened to before any entry
  is trusted, and no automated run in this repository claims otherwise.
- **The scorecard reports, it does not advise.** There is no recommendation about
  which system to choose.

## Adding a system

Implement the protocol in `handset_bench.adapters.base`, add an entry to
`baselines.yaml` naming your adapter by dotted path, and add an image to
`modal/bench_app.py`. Adding a system requires no change to the scoring rules, and
existing entries stay valid.

Adapters are resolved at runtime by dotted path, so `handset-bench` never imports
the systems it scores. That is also why the model this benchmark was built
alongside lives in a separate package and gets no special handling.
