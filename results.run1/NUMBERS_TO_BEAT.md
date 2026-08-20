# Numbers to beat

The best value per measure across every baseline, and which system set it.
Later work is measured against these with no change to how they were derived.

Latency is `ttfb_generation_ms`: wall clock from handing text to the adapter until the first audio frame, measured in a warm process. It is not an end-to-end figure. The project's 150ms target additionally includes network, warm-pool queueing, and framing, and is only measurable against a deployed serving stack. A system passing the target here has not yet passed it there.

- **Word error rate after the codec round trip: 3.06%**, set by `piper` at version `piper-1.6.1+en_US-lessac-medium+det` on condition `clean`.
- **Generation-side time to first audio, p95: 156.9 ms**, set by `piper` at version `piper-1.6.1+en_US-lessac-medium+det`. This is a generation-side figure and does not by itself establish the 150ms end-to-end target, which needs a deployed serving stack to measure.
- Behaviour at 32 simultaneous calls: n/a, pending a concurrency run.

## Pending manual gates

These have NOT been performed and must not be treated as passed:

- Listening to one sample per system per condition to confirm the audio is not obviously broken. A word error rate can look plausible while the audio is garbage.
