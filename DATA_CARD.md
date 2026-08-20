# DialTone data card

Generated. Do not edit by hand.

## Summary

- Total duration: 71.02 hours
- Speakers: 326
- Utterances: 43805

## Sources

| Source | Licence | Utterances | Speakers | Hours | Share |
| --- | --- | --- | --- | --- | --- |
| libritts-r | CC-BY-4.0 | 43805 | 326 | 71.02 | 100.0% |

## Excluded corpora

- **GigaSpeech**: Rests on a fair-use argument and its licence is documented as preventing industrial research. Fails the requirement that every recording permit public release of derived work.
- **People's Speech**: Permits commercial use but is partly CC-BY-SA. Share-alike on a derived model is exactly the ambiguity the licence requirement exists to avoid.
- **Emilia**: CC BY-NC. Non-commercial excludes it outright.
- **Libri-Light**: Overwhelmingly unlabelled. The wrong tool for a corpus that needs transcripts.
- **MLS English**: Licence-clean under CC BY 4.0 but 16kHz. Upsampling to 24kHz would give a mel with no real energy above 8kHz, a distribution mismatch against what the frozen acoustic model emits at inference. Kept on the shelf.
- **LDC Switchboard and Fisher**: Paid. Out of scope for this milestone by decision, and reopened only if augmentation proves insufficient.

## Removals

None.
