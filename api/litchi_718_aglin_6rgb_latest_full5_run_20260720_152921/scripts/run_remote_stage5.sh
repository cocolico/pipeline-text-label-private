#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT="/home/descfly/luzheng/ego_data_visualization/api/litchi_718_aglin_6rgb_latest_full5_run_20260720_152921"
CODE_DIR="/home/descfly/luzheng/ego_data_visualization/api/api_seed_openai_easy_no_short_step/luzheng1/Qwen-PC/ECCV/four_stage_100frames_refine_v2"
PYTHON_BIN="/home/descfly/luzheng/ego_data_visualization/api/.venv_bbox_tracking/bin/python"
SAM3_PYTHON="/home/descfly/miniconda3/envs/sam3/bin/python"
SAM3_REPO="/home/descfly/caoheng/sam3"
SAM3_CHECKPOINT="/home/descfly/luzheng/ego_data_visualization/api/models/sam3/sam3.pt"
INPUT_DIR="$RUN_ROOT/input_videos"
OUTPUT_ROOT="$RUN_ROOT/outputs"
LOG_DIR="$RUN_ROOT/logs"
MANIFEST="$RUN_ROOT/INPUT_MANIFEST.tsv"
AUDITOR="$RUN_ROOT/scripts/audit_outputs.py"
LOCK_DIR="$RUN_ROOT/.stage5_runner_lock"

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

if [[ -e "$RUN_ROOT/.stage5_disabled" ]]; then
  printf 'updated_at_utc\tphase\tdetail\n%s\t%s\t%s\n' \
    "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "completed" "stage1_4_only_stage5_disabled" > "$RUN_ROOT/status.tsv"
  touch "$RUN_ROOT/.stage5_paused"
  exit 0
fi

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Another Stage 5 runner already owns $LOCK_DIR" >&2
  exit 73
fi

status() {
  local phase=$1
  local detail=${2:-}
  printf 'updated_at_utc\tphase\tdetail\n%s\t%s\t%s\n' \
    "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$phase" "$detail" > "$RUN_ROOT/status.tsv"
}

cleanup() {
  local rc=$?
  if [[ "$rc" -ne 0 ]] && [[ ! -e "$RUN_ROOT/.needs_review" ]] && [[ ! -e "$RUN_ROOT/.run_failed" ]]; then
    status failed "stage5_runner_exit_${rc}"
    touch "$RUN_ROOT/.run_failed"
  fi
  date -u +'%Y-%m-%dT%H:%M:%SZ' > "$RUN_ROOT/finished_at_utc.txt"
  printf '%s\n' "$rc" > "$RUN_ROOT/runner_exit_code.txt"
  rm -rf "$LOCK_DIR"
}
trap cleanup EXIT

run_logged() {
  local log_file=$1
  shift
  set +e
  "$@" 2>&1 | tee "$log_file"
  local rc=${PIPESTATUS[0]}
  set -e
  return "$rc"
}

COMMON_ARGS=(
  --input-video-dir "$INPUT_DIR"
  --output-root "$OUTPUT_ROOT"
  --video-fps 5.0
  --frame-index-fps 5.0
  --api-call-retries 8
  --api-call-retry-backoff-sec 2.0
  --api-call-timeout-sec 420
  --continue-on-error
)

STAGE5_ARGS=(
  --bbox-python "$PYTHON_BIN"
  --hand-device cpu
  --bbox-device cuda
  --bbox-backend sam3
  --sam3-python "$SAM3_PYTHON"
  --sam3-repo "$SAM3_REPO"
  --sam3-checkpoint "$SAM3_CHECKPOINT"
  --sam3-threshold 0.5
  --sam3-chunk-size 500
  --sam3-max-instances 8
  --sam3-hand-fusion
  --no-sam3-legacy-fallback
  --api-hand-assist
  --api-grounding
  --api-grounding-thinking disabled
  --bbox-overlay
)

cd "$CODE_DIR"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$RUN_ROOT/stage5_started_at_utc.txt"
rm -f "$RUN_ROOT/.run_complete" "$RUN_ROOT/.run_failed" "$RUN_ROOT/.needs_review"

status stage5_preflight running
run_logged "$LOG_DIR/stage5_preflight.log" \
  "$PYTHON_BIN" pipeline.py \
  "${COMMON_ARGS[@]}" "${STAGE5_ARGS[@]}" \
  --stages 5 --num-workers 1 --preflight-only
status stage5_preflight passed

stage5_ok=0
for attempt in 1 2 3; do
  status stage5 "attempt_${attempt}_serial"
  run_logged "$LOG_DIR/stage5_attempt_${attempt}.log" \
    "$PYTHON_BIN" pipeline.py \
    "${COMMON_ARGS[@]}" "${STAGE5_ARGS[@]}" \
    --stages 5 --num-workers 1 --no-retry-failed \
    --failed-report-json "$RUN_ROOT/stage5_failed_report.json" || true
  if "$PYTHON_BIN" "$AUDITOR" \
      --manifest "$MANIFEST" \
      --output-root "$OUTPUT_ROOT" \
      --phase final \
      --report "$RUN_ROOT/audit_final.json"; then
    stage5_ok=1
    break
  fi
  sleep $((attempt * 60))
done

if [[ "$stage5_ok" -ne 1 ]]; then
  status needs_review stage5_quality_gate_not_passed_for_all_videos
  touch "$RUN_ROOT/.needs_review"
  exit 42
fi

status completed all_6_videos_passed_full_5_stage_audit
touch "$RUN_ROOT/.run_complete"
