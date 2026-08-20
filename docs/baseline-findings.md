# Baseline findings

Status: **complete for two systems**. 300 utterances, four conditions, faster-whisper
large-v3 as the listener. The designated headline listener, Parakeet, has not been
run, so nothing here is the headline ranking.

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

## Recommendation

Do the listening gate first. It is free and it is the only check that can invalidate
everything above.

Then settle the premise before spending on training. If the phone line costs nothing
in word error rate, the narrowband vocoder has to be justified on naturalness or on
compute, and neither is measured yet. Deciding that is cheaper than training against
a target that does not exist.
