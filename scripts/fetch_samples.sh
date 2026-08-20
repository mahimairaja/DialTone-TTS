#!/usr/bin/env bash
#
# Fetch listening samples for the manual gate.
#
#   ./scripts/fetch_samples.sh                          # one per category, all conditions
#   ./scripts/fetch_samples.sh dt-0008 dt-0112          # specific utterances
#   SYSTEM=zipvoice_distill ./scripts/fetch_samples.sh  # a different system
#
# Audio is stored on the Modal volume as float32 .npy at 16kHz, which nothing plays.
# This pulls it down and writes .wav next to a listening checklist.
#
# The gate this serves cannot be automated and is not passed by any script: a word
# error rate can look perfect while the audio is broken.

set -euo pipefail

cd "$(dirname "$0")/.."

SYSTEM="${SYSTEM:-piper}"
VERSION="${VERSION:-}"
OUT="${OUT:-samples}"
CONDITIONS=(wideband clean loss_1pct loss_3pct)

# One per category, taking the first of each. Digits and datetime_money are the
# cases where the codec measurably changes the score, so they matter most.
DEFAULT_IDS=(dt-0001 dt-0061 dt-0111 dt-0151 dt-0191 dt-0261)
IDS=("$@")
[[ ${#IDS[@]} -eq 0 ]] && IDS=("${DEFAULT_IDS[@]}")

MODAL=./.venv/bin/modal

# Take the version pinned in baselines.yaml, not whatever the volume lists first.
# Stale audio from a superseded version is still on the volume, and listening to it
# would be checking something the scorecard never measured.
if [[ -z "$VERSION" ]]; then
  VERSION=$(uv run python -c "
import sys, yaml
name = sys.argv[1]
spec = yaml.safe_load(open('baselines.yaml'))
match = [s for s in spec['systems'] if s['name'] == name]
print(match[0]['version'] if match else '')
" "$SYSTEM")
fi
if [[ -z "$VERSION" ]]; then
  echo "no audio on the volume for system '$SYSTEM'. Run the matrix first." >&2
  exit 1
fi

echo "system:  $SYSTEM"
echo "version: $VERSION"
mkdir -p "$OUT"

for condition in "${CONDITIONS[@]}"; do
  for id in "${IDS[@]}"; do
    remote="$SYSTEM/$VERSION/$condition/$id.npy"
    local="$OUT/${id}.${condition}.npy"
    [[ -f "$local" ]] && continue
    "$MODAL" volume get --force handset-bench-audio "$remote" "$local" >/dev/null 2>&1 \
      || { echo "  missing: $remote" >&2; continue; }
  done
done

uv run python - "$OUT" "$SYSTEM" "$VERSION" <<'PY'
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from handset_bench.codec import ASR_RATE
from handset_bench.textset import loader

out, system, version = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
text_of = {u.utterance_id: u.text for u in loader.load()}
category_of = {u.utterance_id: u.category for u in loader.load()}

written = []
for path in sorted(out.glob("*.npy")):
    utterance_id, condition, _ = path.name.split(".", 2)
    wav = path.with_suffix(".wav")
    sf.write(wav, np.load(path).astype(np.float32), ASR_RATE)
    path.unlink()
    written.append((utterance_id, condition, wav.name))

lines = [
    "# Listening gate",
    "",
    f"System `{system}`, version `{version}`.",
    "",
    "Audio is 16kHz because that is what the listener receives. The codec arm really",
    "is 8kHz mu-law underneath; it is upsampled so the comparison is like for like.",
    "",
    "## What you are checking",
    "",
    "1. `wideband` should sound like clean studio speech.",
    "2. `clean` should sound like a phone call, and should still be intelligible.",
    "3. `loss_3pct` should have audible brief dropouts, not continuous damage.",
    "4. `loss_1pct` may be byte-identical to `clean` on a short clip. One percent of",
    "   138 frames is about 1.4 frames, so sometimes zero are drawn. That is correct",
    "   behaviour, not a bug. Judge the loss conditions on `loss_3pct`.",
    "5. Nothing should be silence, static, clipping, or a truncated word.",
    "",
    "A word error rate can look perfect while the audio is broken. That is the only",
    "reason this gate exists.",
    "",
    "## Samples",
    "",
]

for utterance_id in sorted({w[0] for w in written}):
    lines += [
        f"### {utterance_id} ({category_of.get(utterance_id, 'unknown')})",
        "",
        f"> {text_of.get(utterance_id, '')}",
        "",
    ]
    for _, condition, name in sorted(w for w in written if w[0] == utterance_id):
        lines.append(f"- [ ] `{condition}` — `{name}`")
    lines.append("")

(out / "CHECKLIST.md").write_text("\n".join(lines))
print(f"wrote {len(written)} wav files and {out}/CHECKLIST.md")
PY

echo
echo "Now listen. On macOS:"
echo "  open $OUT"
echo "  afplay $OUT/dt-0001.clean.wav"
echo
echo "Work through $OUT/CHECKLIST.md and tick each box."
