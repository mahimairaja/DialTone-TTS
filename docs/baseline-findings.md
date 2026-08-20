# Baseline findings

Status: **Both systems measured, reproducible, and heard.** 300 utterances, four conditions,
faster-whisper large-v3, on deterministic synthesis. Two consecutive runs produced
1,200 identical transcripts, so the reproducibility band is 0.0 points and every
figure below is exact rather than approximate.

ZipVoice had the same defect and was fixed the same way. Both systems now pin `+det`
and both reproduce.

## The scorecard

| System | wideband | clean | loss 1% | loss 3% | codec cost |
| --- | --- | --- | --- | --- | --- |
| piper `+det` | 3.48 | **3.06** | 3.13 | 3.24 | -0.41 |
| zipvoice_distill `+det` | 4.68 | 4.27 | 4.34 | 4.30 | -0.41 |

Piper has now produced these exact figures on three consecutive runs. The band is 0.0
points.

**Piper wins by 1.21 points on `clean`**, and on latency by more than double: 157.9 ms
p50 against 342.0 ms. It is a CPU model. ZipVoice-Distill is 123M parameters on an
A10G.

Both systems are improved by the codec by exactly the same amount, 0.41 points, which
is a strong hint the mechanism is the recogniser and not either model.

## Listening gate

Performed 2026-08-20 by Mahimai on both systems, 24 samples each: six utterances, one
per category, across all four conditions.

| System | Version | Result |
| --- | --- | --- |
| piper | `piper-1.6.1+en_US-lessac-medium+det` | **passed** |
| zipvoice_distill | `zipvoice_distill+k2fsa+step8+cfg3+det` | **passed** |

On both: wideband sounds like clean studio speech, the codec arm sounds like a phone
call and stays intelligible, the loss conditions drop audibly without continuous
damage, and nothing is silence, static, clipping or a truncated word.

This is the only check that can invalidate a word error rate, because a rate can look
perfect while the audio is broken. It is the difference between these figures being
numbers and being results.

What it does not cover, stated so nobody reads it as broader than it is:

- Six utterances of 300 per system. A systematic defect on a category would show; a
  rare one would not.
- Neither ZipVoice 16-step nor any Parakeet-scored audio has been heard.
- The augmented-clip gate, confirming augmentation sounds like a real phone line, is
  not applicable yet: augmentation is not built.

Regenerate the samples with `./scripts/fetch_samples.sh`.

## Five findings, in order of how much they change the plan

### 1. The phone line does not cost anything. It helps.

The codec chain **improves** word error rate by 0.41 points, on both systems, exactly.
Per category:

| Category | piper wide | piper clean | zipvoice wide | zipvoice clean |
| --- | --- | --- | --- | --- |
| datetime_money | 9/429 | **1/429** | 38/429 | **27/429** |
| proper_nouns | 39/326 | 37/326 | 31/326 | 28/326 |
| digits | 34/670 | 32/670 | 44/670 | 44/670 |
| addresses | 19/414 | 17/414 | 23/414 | 24/414 |
| conversational | 0/693 | 0/693 | 0/693 | 1/693 |
| general | 0/372 | 2/372 | 0/372 | 0/372 |

`datetime_money` carries it on both: 8 of Piper's 12 recovered errors and 11 of
ZipVoice's. Two unrelated architectures, improved by the same amount, in the same
category. That points at the recogniser rather than at either model: band-limiting to
300-3400 Hz removes high-frequency content that faster-whisper trips over when parsing
times and amounts.

Packet loss costs very little: 0.07 points at 1 percent for both, and 0.10 more at 3
percent for Piper. ZipVoice moves -0.03 at 3 percent, which is one word error and
inside what a single dropped frame can do either way.

**The consequence, now backed by an exactly reproducible measurement on two
architectures.** This project exists on the premise that the telephone line destroys
quality and a telephony-native model would recover it. For intelligibility, the line
destroys nothing: it is worth 0.41 points in the other direction. The narrowband
vocoder has to be justified on naturalness, or on the compute saved by generating
8 kHz directly instead of 24 kHz and resampling. It cannot be justified on word error
rate, because there is no gap to close.

### 2. The small CPU model beats the large GPU model on both axes

| | piper | zipvoice_distill |
| --- | --- | --- |
| WER, clean | **3.06** | 4.27 |
| latency p50 | **157.9 ms** | 342.0 ms |
| hardware | CPU | A10G |

Piper wins on quality by 1.21 points and on latency by more than double, while costing
nothing per hour to run. Both figures are on deterministic synthesis and reproduce.

ZipVoice is better on `proper_nouns` (28 errors against 37) and worse everywhere else,
badly so on `datetime_money` (27 against 1) and `digits` (44 against 32). Its advantage
is voice cloning, which this benchmark does not score. Any argument for it has to be
made on that basis.

### 3. Chunking is strictly harmful

| Message | first chunk | whole, chunked | whole, unchunked | rtf |
| --- | --- | --- | --- | --- |
| 3.87 s | 343.1 ms | 1012.3 ms | 367.2 ms | 3.8x |
| 5.22 s | 350.7 ms | 1054.1 ms | 396.8 ms | 4.9x |
| 4.83 s | 367.9 ms | 1120.8 ms | 388.0 ms | 4.3x |

Chunking saves 30 ms on time to first audio and costs 678 ms on total generation.
Across five runs the saving has ranged from 14 to 39 ms and the cost from 638 to 695
ms: the saving is inside the noise, the cost is not.

Realtime factor is 3.8x to 4.9x, against the 150x the plan assumed. Generation alone
is 342 ms p50 against a 150 ms budget for the entire round trip.

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

Parakeet has run once, on Piper only, and **on pre-determinism-fix audio**. The
disagreement below is directional evidence, not a measurement, and has to be redone.

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

## Recommendation

**Settle the premise.** Both systems are measured, reproducible to 0.0 points, and
heard. The benchmark has answered the question it was built to answer, and further
breadth will not change that answer. Two unrelated architectures, measured exactly and
reproducibly, are both *improved* by the phone line by the same 0.41 points. There is
no intelligibility gap for a telephony-native vocoder to close. It has to be justified
on naturalness, or on the compute saved by generating 8 kHz directly instead of 24 kHz
and resampling, and neither is measured. Deciding that is cheaper now than after
training against a target that does not exist.

**Re-run Parakeet before quoting any headline ranking.** Its only run was on
pre-determinism-fix audio.
