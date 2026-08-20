#!/usr/bin/env bash
#
# The baseline run: corpus ingest, then every baseline system pushed through the
# phone line and scored. See README.md for what it costs and how long it takes.
#
#   ./scripts/run_baseline.sh                      # smoke run, 12 utterances
#   ./scripts/run_baseline.sh --full               # all 300
#   ./scripts/run_baseline.sh --full --parakeet    # adds the headline listener
#   ./scripts/run_baseline.sh --skip-ingest        # corpus already on the Volume
#
# Every stage is skippable and idempotent. The matrix refuses to launch when its
# cost estimate exceeds --max-cost.
#
# k2 is not installed in the ZipVoice image: it must be version-matched to torch
# and CUDA and is only needed for training. Upstream falls back to slower pure
# PyTorch Swoosh activations, so every ZipVoice latency figure here is pessimistic.

set -euo pipefail

cd "$(dirname "$0")/.."

LIMIT=12
SYSTEMS="piper,zipvoice_distill"
LISTENERS="whisper"
MAX_COST_USD=30
SKIP_INGEST=0
SKIP_TESTS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full)        LIMIT=0 ;;
    --limit)       LIMIT="$2"; shift ;;
    --systems)     SYSTEMS="$2"; shift ;;
    --parakeet)    LISTENERS="whisper,parakeet"; export DIALTONE_PARAKEET=1 ;;
    --listeners)   LISTENERS="$2"; shift ;;
    --max-cost)    MAX_COST_USD="$2"; shift ;;
    --skip-ingest) SKIP_INGEST=1 ;;
    --skip-tests)  SKIP_TESTS=1 ;;
    -h|--help)     sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
  shift
done

MODAL=./.venv/bin/modal

banner() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }

banner "0. environment"
if [[ ! -x "$MODAL" ]]; then
  echo "modal is not installed in the project venv. Run: uv sync --all-packages" >&2
  exit 1
fi
"$MODAL" profile current
uv --version

if [[ "$SKIP_TESTS" -eq 0 ]]; then
  banner "1. local test suite (no network, no GPU)"
  uv run ruff check .
  uv run ruff format --check .
  uv run pytest packages/ -q
fi

if [[ "$SKIP_INGEST" -eq 0 ]]; then
  banner "2. corpus: LibriTTS-R ingest, licence gate, frozen split"
  # Idempotent: subsets already on the Volume are skipped. Produces
  # manifests/split_v1.json, configs/voices.yaml, DATA_CARD.md, and the three
  # fixed heldout prompt references the benchmark hands to cloning systems.
  "$MODAL" run modal/data_app.py
fi

banner "3. benchmark matrix, cheap systems first"
# Piper runs first and gates everything else. A system that cannot be built or run
# is recorded as unavailable with its reason and the matrix continues.
"$MODAL" run modal/bench_app.py::run_matrix \
  --systems "$SYSTEMS" \
  --listeners "$LISTENERS" \
  --limit "$LIMIT" \
  --max-cost-usd "$MAX_COST_USD"

banner "4. scorecard"
uv run handset-bench report --results results

banner "done"
cat <<'EOF'
Artefacts:
  results/SCORECARD.md        the scorecard, per condition and per category
  results/NUMBERS_TO_BEAT.md  the target every later change must beat
  results/CHUNKING.json       time to first chunk for the non-streaming model
  results/scorecard.csv       machine readable
  manifests/split_v1.json     the frozen speaker split
  configs/voices.yaml         the four DialTone voices and three prompt speakers
  DATA_CARD.md                corpus provenance, including what was excluded

Manual gates NOT performed by this script, and not to be treated as passed:
  - listen to one sample per system per condition and confirm the audio is not
    garbage. A word error rate can look plausible while the audio is broken
  - listen to ~20 augmented clips and confirm they sound like a phone call
EOF
