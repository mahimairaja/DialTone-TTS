# Baseline findings

Status: **partial**. 300 utterances, four conditions. Whisper has scored two systems;
Parakeet has now scored one. A run adding Parakeet and ZipVoice 16-step was
interrupted partway, so the ZipVoice rows below still come from the earlier
Whisper-only run.

**Read finding 4 before quoting any number here.** The benchmark does not reproduce
within its own stated tolerance.

## Word error rate

| System | wideband | clean | loss 1% | loss 3% | latency p50 | p95 |
| --- | --- | --- | --- | --- | --- | --- |
| piper `1.6.1+en_US-lessac-medium` | 4.44 | **3.17** | 3.17 | 3.48 | 157.9 ms | 224.9 ms |
| zipvoice_distill `step8+cfg3` | 4.44 | 4.06 | 4.06 | 4.06 | 342.0 ms | 370.7 ms |

2,904 reference words, so one word error is worth 0.034 points.

## Three findings, in order of how much they change the plan

### 1. The phone line costs nothing measurable

Both systems score **better** after the codec than before it. Piper drops from 3.44
to 3.17, ZipVoice from 4.44 to 4.06. Packet loss at 3 percent costs Piper 0.31
points and ZipVoice nothing at all.

This is not noise and it is not a bug: the per-category table shows where it comes
from.

| Category | piper wide | piper clean | zipvoice wide | zipvoice clean |
| --- | --- | --- | --- | --- |
| datetime_money | 6/429 | 2/429 | 28/429 | 19/429 |
| proper_nouns | 40/326 | 36/326 | 34/326 | 33/326 |
| digits | 33/670 | 32/670 | 39/670 | 41/670 |
| conversational | 0/693 | 1/693 | 0/693 | 0/693 |

Band-limiting to 300-3400 Hz removes high-frequency content, and both systems emit
artefacts up there that the recogniser trips over. Stripping them helps, most of all
on `datetime_money`, which is where ZipVoice loses 9 of its 11 recovered errors.

**The consequence is uncomfortable.** This project exists on the premise that the
telephone line destroys quality and a telephony-native model would recover it. For
intelligibility, measured by word error rate, the line destroys nothing. The case
for a narrowband vocoder has to rest on something else: naturalness, or the compute
saved by generating 8kHz directly instead of 24kHz and resampling. It cannot rest on
word error rate, because there is no gap to close.

### 2. The small CPU model beats the large GPU model on both axes

Piper is a CPU model. It wins on word error rate (3.17 against 4.06) and on
generation latency (157.9 ms p50 against 342.0 ms). It also costs nothing per hour
to run.

ZipVoice's advantage is voice cloning, which this benchmark does not score. Any
argument for it has to be made on that basis, not on the numbers here.

### 3. Chunking is strictly harmful

| Message | first chunk | whole, chunked | whole, unchunked | rtf |
| --- | --- | --- | --- | --- |
| 3.87 s | 381.8 ms | 1081.0 ms | 382.0 ms | 3.6x |
| 5.22 s | 375.2 ms | 1109.8 ms | 392.7 ms | 4.7x |
| 4.83 s | 363.3 ms | 1057.6 ms | 387.6 ms | 4.6x |

Chunking saves 14 ms on time to first audio and costs 695 ms on total generation.
Across four runs the saving has ranged from 14 to 39 ms and the cost from 638 to 695
ms: the saving is inside the noise, the cost is not.

Realtime factor is 3.6x to 4.7x, against the 150x the plan assumed. Generation alone
is 342 ms p50 against a 150 ms budget for the entire round trip.

## Caveats that could move these numbers

- **k2 is not installed.** ZipVoice falls back to a pure PyTorch implementation of
  its activations and warns this is slower. Every ZipVoice latency figure is
  pessimistic by an unmeasured amount.
- **A10G, not H100.**
- **faster-whisper, not Parakeet.** The designated headline listener has not run.
  Whisper hallucinates on non-speech, which the loss conditions manufacture.
- **No listening check has been performed.** A word error rate can look plausible
  while the audio is broken.

## What has not been run

- Parakeet, the headline listener
- ZipVoice 16-step, Chatterbox, CosyVoice2
- The concurrency measurement at 1, 8, 16, 32
- The reproducibility rerun against the 0.1 point band
- Every manual listening gate

### 4. The benchmark does not reproduce within its stated band

`docs/REPRODUCING.md` claims two runs of the same entry agree within 0.1 percentage
points. Running the identical Piper configuration twice:

| Condition | run 1 | run 2 | delta | inside 0.1pp |
| --- | --- | --- | --- | --- |
| wideband | 3.44 | 3.20 | -0.24 | no |
| clean | 3.17 | 3.27 | +0.10 | no |
| loss_1pct | 3.17 | 3.24 | +0.07 | yes |
| loss_3pct | 3.48 | 3.20 | -0.28 | no |

Three of four are outside the band.

`wideband` narrows it. It applies no codec and no packet loss, and the references are
identical, yet **63 of 300 transcripts differ** between runs, including content and
not just casing:

```
dt-0021  run 1: The postal code is K7 of 3M9.
         run 2: The postal code is K7A3M9.
```

**The cause is the system under test, not the listener.** Two probes settled it.

Transcribing the same 40 clips twice inside one process returns identical text 40 out
of 40, under `cuda-float16`, `cuda-float32` and `cpu-int8` alike. The listener is
deterministic. (`cpu-int8` is also 56x slower per pass, so CPU inference is not a
practical fallback: 1,200 clips would take about five hours.)

Synthesising the same text twice returns **0 of 8 identical waveforms**, with
differing lengths. Piper is VITS-based and samples noise for its stochastic duration
predictor, so every call produces different audio. The benchmark was re-synthesising
each run and comparing transcripts of different recordings.

**Fixed** by zeroing both noise terms, which makes synthesis reproducible: 8 of 8
identical, rms difference 0.0. Because that changes the waveform, it changes the
system identity too, so the version string now carries the mode: `+det` or
`+sampled`. Records made before this are not comparable to records made after, and
their bare version string says so.

**What follows.** Every word error rate in this document predates the fix and was
measured on audio that changed between runs. The band has to be re-measured on
deterministic synthesis before any tolerance is stated. The Piper against ZipVoice
gap of 0.89 points is larger than the 0.28 of drift and probably survives, but that
is an expectation, not a measurement.

ZipVoice is prompt-conditioned flow matching and is likely to have the same problem
from a different direction. It has not been probed.

### 5. The headline listener disagrees with the second column

Parakeet has now run, on Piper only.

| Category | whisper, clean | parakeet, clean |
| --- | --- | --- |
| digits | 35/670 | 31/670 |
| proper_nouns | 40/326 | 56/326 |
| addresses | 17/414 | 28/414 |
| datetime_money | 1/429 | 4/429 |
| general | 2/372 | 0/372 |

Parakeet scores Piper worse overall (4.13 against 3.27) and the two disagree by
category: Parakeet is better on digits and general, clearly worse on proper nouns and
addresses. Neither is ground truth. Which listener is designated headline therefore
changes the ranking, which is an argument for reporting both columns rather than
picking one.

## Recommendation

Do the listening gate first. It is free and it is the only check that can invalidate
everything above.

Then settle the premise before spending on training. If the phone line costs nothing
in word error rate, the narrowband vocoder has to be justified on naturalness or on
compute, and neither is measured yet. Deciding that is cheaper than training against
a target that does not exist.
