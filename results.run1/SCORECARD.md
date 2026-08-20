# handset-bench scorecard

Text-to-speech systems scored on what survives a G.711 telephone line.

- Text set: `dialtone_v1`, 300 utterances
- **Listener that produced these numbers: `faster-whisper-large-v3`**
- **Designated headline listener `nvidia/parakeet-tdt-0.6b-v2` has NOT been run.** These figures come from the second-column listener only and are not the headline ranking.

Latency is `ttfb_generation_ms`: wall clock from handing text to the adapter until the first audio frame, measured in a warm process. It is not an end-to-end figure. The project's 150ms target additionally includes network, warm-pool queueing, and framing, and is only measurable against a deployed serving stack. A system passing the target here has not yet passed it there.

## Word error rate by condition, listener `faster-whisper-large-v3`

Lower is better. `wideband` is the pre-codec control; `drop` is how much the phone line costs.

| System | Version | wideband | clean | drop | loss 1% | loss 3% |
| --- | --- | --- | --- | --- | --- | --- |
| piper | `piper-1.6.1+en_US-lessac-medium+det` | 3.48 | 3.06 | -0.41 | 3.13 | 3.24 |


## Word error rate by category, listener `faster-whisper-large-v3`

The aggregate can hide opposite effects that cancel. This table is where the phone line's real cost shows up.

| Category | wideband | clean | drop | loss 3% |
| --- | --- | --- | --- | --- |
| addresses | 4.59 | 4.11 | -0.48 | 4.83 |
| conversational | 0.00 | 0.00 | +0.00 | 0.00 |
| datetime_money | 2.10 | 0.23 | -1.86 | 0.23 |
| digits | 5.07 | 4.78 | -0.30 | 4.78 |
| general | 0.00 | 0.54 | +0.54 | 0.81 |
| proper_nouns | 11.96 | 11.35 | -0.61 | 11.66 |

## Generation-side latency

| System | Version | p50 ms | p95 ms |
| --- | --- | --- | --- |
| piper | `piper-1.6.1+en_US-lessac-medium+det` | 110.1 | 156.9 |

## Conditions

- **clean**: Generation-side timing only. No phone line and no transcription involved: the codec chain runs after generation and cannot affect time to first audio.
- **loss_1pct**: Band-limited to 300-3400 Hz, resampled to 8kHz, G.711 mu-law encoded and decoded. 20ms frames dropped independently at 1 percent, each replaced with the mu-law code for silence (about -81 dBFS).
- **loss_3pct**: Band-limited to 300-3400 Hz, resampled to 8kHz, G.711 mu-law encoded and decoded. 20ms frames dropped independently at 3 percent, each replaced with the mu-law code for silence (about -81 dBFS).
- **wideband**: No phone line applied. The system's own output, used as the pre-codec control so the size of the drop is visible.
