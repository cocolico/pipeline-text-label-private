#!/usr/bin/env bash
set -Eeuo pipefail

# Stage1-4-only runner for version2 (Stage5 removed; supports single-video and batch modes).
#
# Paths point at the .59 "version2" code tree as seen from .57 via the NFS mount
# (/mnt/storage59 on .57 == /data on .59), and use the Python venv on .57.
#
# Usage:
#   run_remote_pipeline.sh --input-video <path-to-rgb.mp4>
#   run_remote_pipeline.sh --batch-dir <dir-to-scan-for-rgb.mp4>

CODE_DIR="/mnt/storage59/wangxinyi_code/version2/api/api_seed_openai_easy_no_short_step"
PYTHON_BIN="/home/user/wangxinyi/.venv/bin/python"

MODE=""
TARGET=""
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
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$MODE" ]]; then
  echo "Usage: $0 --input-video <path-to-rgb.mp4> | --batch-dir <dir>" >&2
  exit 2
fi

COMMON_ARGS_BASE=(
  --video-fps 5.0
  --frame-index-fps 5.0
  --api-call-retries 8
  --api-call-retry-backoff-sec 2.0
  --api-call-timeout-sec 420
  --continue-on-error
  --stages 1,2,3
  --num-workers 1
)

run_one() {
  local video="$1"
  local out_root
  out_root="$(dirname "$video")/stage1_4_outputs_new_version3"
  mkdir -p "$out_root"

  echo "=== [$(date -u +'%Y-%m-%dT%H:%M:%SZ')] START: $video ===" >&2
  echo "=== output-root: $out_root ===" >&2

  (
    cd "$CODE_DIR"
    "$PYTHON_BIN" pipeline.py \
      --input-video "$video" \
      --output-root "$out_root" \
      "${COMMON_ARGS_BASE[@]}"
  )
  local rc=$?

  echo "=== [$(date -u +'%Y-%m-%dT%H:%M:%SZ')] END: $video (exit=${rc}) ===" >&2
  return "$rc"
}

if [[ "$MODE" == "single" ]]; then
  if [[ ! -f "$TARGET" ]]; then
    echo "Input video not found: $TARGET" >&2
    exit 1
  fi
  run_one "$TARGET"
  exit $?
fi

# Batch mode: find every rgb.mp4 under TARGET, process one at a time, continue past failures.
if [[ ! -d "$TARGET" ]]; then
  echo "Batch dir not found: $TARGET" >&2
  exit 1
fi

overall_rc=0
while IFS= read -r -d '' video; do
  if ! run_one "$video"; then
    overall_rc=1
  fi
done < <(find "$TARGET" -type f -name "rgb.mp4" -print0 | sort -z)

exit "$overall_rc"
