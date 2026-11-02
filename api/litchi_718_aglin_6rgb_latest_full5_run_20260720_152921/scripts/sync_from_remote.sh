#!/usr/bin/env bash
set -Eeuo pipefail

HOST="descfly-192-168-10-113"
REMOTE_ROOT="/home/descfly/luzheng/ego_data_visualization/api/litchi_718_aglin_6rgb_latest_full5_run_20260720_152921"
LOCAL_ROOT="/Users/lululzz/Desktop/litchi/api/litchi_718_aglin_6rgb_latest_full5_run_20260720_152921"

sync_once() {
  mkdir -p "$LOCAL_ROOT/outputs" "$LOCAL_ROOT/logs"
  rsync -a --partial \
    --exclude '.runner_lock/' \
    --exclude '*.tmp' \
    --exclude 'input_videos/' \
    "$HOST:$REMOTE_ROOT/" "$LOCAL_ROOT/"
  printf 'updated_at\tstate\n%s\trunning\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" > "$LOCAL_ROOT/sync_status.tsv"
}

while true; do
  sync_once || true
  if ssh -o BatchMode=yes "$HOST" \
      "test -e '$REMOTE_ROOT/.run_complete' -o -e '$REMOTE_ROOT/.run_failed' -o -e '$REMOTE_ROOT/.needs_review'"; then
    sync_once
    printf 'updated_at\tstate\n%s\tterminal_synced\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" > "$LOCAL_ROOT/sync_status.tsv"
    exit 0
  fi
  sleep 180
done
