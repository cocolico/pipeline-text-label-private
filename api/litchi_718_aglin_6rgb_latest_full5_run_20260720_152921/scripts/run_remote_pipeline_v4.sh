#!/usr/bin/env bash
set -Eeuo pipefail

# Stage1-4-only runner for version4 (Stage5 removed; supports single-video and PARALLEL batch modes).
#
# Paths point at the .59 "version4" code tree as seen from .57 via the NFS mount
# (/mnt/storage59 on .57 == /data on .59), and use the Python venv on .57.
#
# Output for each video is written two levels above rgb.mp4 (i.e. rgb.mp4 sits under a "raw_data" subdir), under a "processed_text_label" dir:
#   <dirname of dirname of rgb.mp4>/processed_text_label/  (skips the "raw_data" level)
#
# Usage:
#   run_remote_pipeline_v4.sh --input-video <path-to-rgb.mp4>
#   run_remote_pipeline_v4.sh --batch-dir <dir-to-scan-for-rgb.mp4> [--parallel N]

CODE_DIR="/mnt/storage59/wangxinyi_code/version4/api/api_seed_openai_easy_no_short_step"
PYTHON_BIN="/home/user/wangxinyi/.venv/bin/python"

MODE=""
TARGET=""
PARALLEL=4
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-video)
      MODE="single"
      TARGET="$2"
      shift 2
      ;;
    --batch-dir)
      MODE="batch"
      TARGET="$2"
      shift 2
      ;;
    --parallel)
      PARALLEL="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$MODE" ]]; then
  echo "Usage: $0 --input-video <path-to-rgb.mp4> | --batch-dir <dir> [--parallel N]" >&2
  exit 2
fi

export CODE_DIR PYTHON_BIN

run_one() {
  local video="$1"
  local out_root
  out_root="$(dirname "$(dirname "$video")")/processed_text_label"
  mkdir -p "$out_root"

  echo "=== [$(date -u +'%Y-%m-%dT%H:%M:%SZ')] START: $video ===" >&2
  echo "=== output-root: $out_root ===" >&2

  (
    cd "$CODE_DIR"
    "$PYTHON_BIN" pipeline.py \
      --input-video "$video" \
      --output-root "$out_root" \
      --video-fps 5.0 \
      --frame-index-fps 5.0 \
      --api-call-retries 8 \
      --api-call-retry-backoff-sec 2.0 \
      --api-call-timeout-sec 420 \
      --continue-on-error \
      --stages 1,2,3 \
      --num-workers 1 < /dev/null
  )
  local rc=$?

  echo "=== [$(date -u +'%Y-%m-%dT%H:%M:%SZ')] END: $video (exit=${rc}) ===" >&2
  return "$rc"
}
export -f run_one

if [[ "$MODE" == "single" ]]; then
  if [[ ! -f "$TARGET" ]]; then
    echo "Input video not found: $TARGET" >&2
    exit 1
  fi
  run_one "$TARGET"
  exit $?
fi

# Batch mode: find every rgb.mp4 under TARGET, process up to $PARALLEL videos concurrently.
if [[ ! -d "$TARGET" ]]; then
  echo "Batch dir not found: $TARGET" >&2
  exit 1
fi

find "$TARGET" -type f -name "rgb.mp4" -print0 | sort -z | \
  xargs -0 -P "$PARALLEL" -I{} bash -c 'run_one "$1"' _ {}
