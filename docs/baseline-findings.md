# Baseline findings

Status: **Piper measured and reproducible.** 300 utterances, four conditions,
faster-whisper large-v3, on deterministic synthesis. Two consecutive runs produced
1,200 identical transcripts, so the reproducibility band is 0.0 points and every
figure below is exact rather than approximate.

ZipVoice has not been re-measured since the determinism fix. Every ZipVoice number in
this document predates it and should not be quoted.

## Piper, measured twice, identical both times

| Condition | WER | change |
| --- | --- | --- |
| wideband (no phone line) | 3.48 | control |
| clean (full G.711 chain) | **3.06** | -0.41 |
| loss 1% | 3.13 | +0.07 |
| loss 3% | 3.24 | +0.10 |

Generation-side latency: p50 157.9 ms, p95 224.9 ms.

The loss series is monotonic, as physics requires: more dropped frames, more errors.
That ordering failed on the pre-fix numbers, which was the first sign something was
wrong.

## Systems not yet re-measured

| System | wideband | clean | loss 1% | loss 3% | latency p50 | p95 |
| --- | --- | --- | --- | --- | --- | --- |
| zipvoice_distill `step8+cfg3` | 4.44 | 4.06 | 4.06 | 4.06 | 342.0 ms | 370.7 ms |

Pre-fix, on sampled synthesis. Not comparable to the Piper figures above and not to
be quoted. ZipVoice is flow matching and starts from noise, so it may have the same
defect; `probe_zipvoice_determinism` exists to answer that and has not been run.

## Three findings, in order of how much they change the plan

### 1. The phone line does not cost anything. It helps.

The codec chain **improves** word error rate by 0.41 points, and this is now exact
rather than within noise. The per-category table shows where it comes from:

| Category | wideband | clean | loss 3% |
| --- | --- | --- | --- |
| datetime_money | 9/429 | **1/429** | 1/429 |
| addresses | 19/414 | 17/414 | 20/414 |
| digits | 34/670 | 32/670 | 32/670 |
| proper_nouns | 39/326 | 37/326 | 38/326 |
| conversational | 0/693 | 0/693 | 0/693 |
| general | 0/372 | 2/372 | 3/372 |

`datetime_money` alone accounts for 8 of the 12 recovered errors, going from 9 errors
to 1. Band-limiting to 300-3400 Hz removes high-frequency content, and Piper emits
artefacts up there that the recogniser trips over. Stripping them helps.

Packet loss does cost something, but very little: 0.07 points at 1 percent and 0.10
more at 3 percent. A 3 percent loss rate is a bad line.

**The consequence is uncomfortable, and it is now backed by an exactly reproducible
measurement.** This project exists on the premise that the telephone line destroys
quality and a telephony-native model would recover it. For intelligibility, the line
destroys nothing: it is worth 0.41 points in the other direction. The case for a
narrowband vocoder has to rest on naturalness, or on the compute saved by generating
8 kHz directly instead of 24 kHz and resampling. It cannot rest on word error rate,
because there is no gap to close.

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

### 4. The benchmark now reproduces exactly, after one fix

`docs/REPRODUCING.md` claimed a 0.1 point band. Before the fix that was not
achievable: the same Piper configuration run twice moved by up to 0.28 points, with
three of four conditions outside the band, and the loss series came out non-monotonic.

Two probes found the cause. Transcribing the same 40 clips twice inside one process
returns identical text 40 of 40, under `cuda-float16`, `cuda-float32` and `cpu-int8`
alike, so the listener was never the problem. Synthesising the same text twice
returned **0 of 8 identical waveforms**, with differing lengths. Piper is VITS-based
and samples noise for its stochastic duration predictor, so every call produced
different audio. The benchmark was comparing transcripts of different recordings.

Zeroing both noise terms fixed it. Two full runs since: **1,200 identical
transcripts, 0.0000 points of drift.** The band is 0.0, and the version string carries
`+det` or `+sampled` because the waveform differs between them.

Worth stating plainly: the first diagnosis blamed faster-whisper for being
non-deterministic on GPU. That was wrong, and the probe that disproved it cost about
fifteen cents. The listener is deterministic across separate containers and separate
GPUs.

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

The benchmark is now trustworthy for Piper. Three things follow, in order.

**Do the listening gate.** It is free and it is the only check that can invalidate
everything above. A word error rate can look perfect while the audio is broken.

**Re-measure ZipVoice on deterministic synthesis**, and probe it first with
`probe_zipvoice_determinism`. It is flow matching and starts from noise, so it may
have exactly the defect Piper had. Until then no ZipVoice number is quotable and the
Piper-beats-ZipVoice conclusion is unverified.

**Then settle the premise.** The phone line costs nothing in word error rate. It is
worth 0.41 points in the wrong direction for the argument this project rests on. A
narrowband vocoder has to be justified on naturalness or on compute, and neither is
measured. That decision is cheaper to make now than after training against a target
that does not exist.
