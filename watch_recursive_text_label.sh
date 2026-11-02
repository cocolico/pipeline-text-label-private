#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  export ARK_API_KEY='your-key'
  source /path/to/venv/bin/activate
  ./watch_recursive_text_label.sh SEARCH_ROOT [--once] [extra pipeline arguments]

The default mode scans forever, once per hour, and processes up to four videos
in parallel inside one pipeline process. Every */raw_data/rgb.mp4 writes to its
own sibling processed_text_label directory. Existing completed outputs are
skipped and incomplete/failed outputs are resumed without --overwrite.

Optional environment variables:
  TEXT_LABEL_POLL_SECONDS=3600
  TEXT_LABEL_WORKERS=4
  TEXT_LABEL_MIN_AGE_SECONDS=300
  TEXT_LABEL_STABILITY_CHECK_SECONDS=60
  TEXT_LABEL_VLM_FRAME_PADDING_SEC=0.5
  TEXT_LABEL_VLM_FRAME_BATCH_SIZE=12
  TEXT_LABEL_PYTHON=python3
  TEXT_LABEL_ONCE=0
  TEXT_LABEL_LOCK_FILE=/tmp/custom.lock
EOF
}

if (( $# < 1 )); then
  usage >&2
  exit 2
fi

SEARCH_ROOT=$1
shift

RUN_ONCE=${TEXT_LABEL_ONCE:-0}
EXTRA_ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--once" ]]; then
    RUN_ONCE=1
  else
    EXTRA_ARGS+=("$arg")
  fi
done

if [[ ! -d "$SEARCH_ROOT" ]]; then
  echo "ERROR: SEARCH_ROOT is not a directory: $SEARCH_ROOT" >&2
  exit 2
fi
SEARCH_ROOT=$(realpath "$SEARCH_ROOT")

if [[ -z "${ARK_API_KEY:-${API_KEY:-}}" ]]; then
  echo "ERROR: set ARK_API_KEY (preferred) or API_KEY in the environment." >&2
  exit 2
fi

POLL_SECONDS=${TEXT_LABEL_POLL_SECONDS:-3600}
WORKERS=${TEXT_LABEL_WORKERS:-4}
MIN_AGE_SECONDS=${TEXT_LABEL_MIN_AGE_SECONDS:-300}
STABILITY_CHECK_SECONDS=${TEXT_LABEL_STABILITY_CHECK_SECONDS:-60}
PYTHON_BIN=${TEXT_LABEL_PYTHON:-python3}

for numeric in "$POLL_SECONDS" "$MIN_AGE_SECONDS" "$STABILITY_CHECK_SECONDS"; do
  if [[ ! "$numeric" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "ERROR: polling, worker, age, and stability values must be non-negative numbers." >&2
    exit 2
  fi
done
if [[ ! "$WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: TEXT_LABEL_WORKERS must be at least 1." >&2
  exit 2
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "ERROR: flock is required for single-instance protection." >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PIPELINE="$SCRIPT_DIR/api/api_seed_openai_easy_no_short_step/pipeline.py"
if [[ ! -f "$PIPELINE" ]]; then
  echo "ERROR: pipeline.py not found relative to this script: $PIPELINE" >&2
  exit 2
fi

ROOT_ID=$(printf '%s' "$SEARCH_ROOT" | sha256sum | cut -c1-16)
LOCK_FILE=${TEXT_LABEL_LOCK_FILE:-"/tmp/text_label_recursive_${UID}_${ROOT_ID}.lock"}
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "ERROR: another watcher is already running for this search root." >&2
  exit 3
fi

echo "Search root: $SEARCH_ROOT"
echo "Workers: $WORKERS; poll: ${POLL_SECONDS}s; min age: ${MIN_AGE_SECONDS}s; stability check: ${STABILITY_CHECK_SECONDS}s"
echo "API key: provided through environment (value hidden)"

while true; do
  echo "[$(date '+%F %T')] Starting recursive scan"

  # Avoid treating an entirely empty tree as a pipeline failure. No path list
  # is persisted; the pipeline performs its own recursive discovery.
  if find "$SEARCH_ROOT" -type f -path '*/raw_data/rgb.mp4' -print -quit | grep -q .; then
    if ! "$PYTHON_BIN" "$PIPELINE" \
      --input-video-dir "$SEARCH_ROOT" \
      --output-root "$SEARCH_ROOT" \
      --output-per-video-sibling \
      --input-min-age-sec "$MIN_AGE_SECONDS" \
      --input-stability-check-sec "$STABILITY_CHECK_SECONDS" \
      --num-workers "$WORKERS" \
      --stages 1,2,3,4 \
      --retry-failed \
      --failed-report-json - \
      --continue-on-error \
      "${EXTRA_ARGS[@]}"; then
      echo "[$(date '+%F %T')] Scan completed with one or more errors; incomplete videos will retry next scan." >&2
    fi
  else
    echo "[$(date '+%F %T')] No raw_data/rgb.mp4 found; nothing to process."
  fi

  if [[ "$RUN_ONCE" == "1" ]]; then
    break
  fi
  echo "[$(date '+%F %T')] Sleeping ${POLL_SECONDS}s before the next scan"
  sleep "$POLL_SECONDS"
done
