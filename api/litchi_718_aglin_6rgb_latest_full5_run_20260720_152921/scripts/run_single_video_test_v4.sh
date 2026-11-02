#!/usr/bin/env bash
set -Eeuo pipefail

# Single-video test runner for version4 (Stage1-4 only, output -> processed_text_label).
# Usage: run_single_video_test_v4.sh <path-to-rgb.mp4>

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <path-to-rgb.mp4>" >&2
  exit 2
fi

VIDEO="$1"

source /home/user/wangxinyi/.venv/bin/activate

LOG_DIR=/home/user/wangxinyi/litchi_run/code
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/run_single_test_v4_$(date +%Y%m%d_%H%M%S).log"
TIME_LOG="$LOG_DIR/pipeline_time.log"

echo "========== Single-Video Test (v4) Start: $(date "+%Y-%m-%d %H:%M:%S") ==========" | tee "$MAIN_LOG"
echo "Video: $VIDEO" | tee -a "$MAIN_LOG"
start=$(date +%s)

bash /mnt/storage59/wangxinyi_code/version4/api/litchi_718_aglin_6rgb_latest_full5_run_20260720_152921/scripts/run_remote_pipeline_v4.sh \
  --input-video "$VIDEO" < /dev/null 2>&1 | tee -a "$MAIN_LOG"

exit_code=${PIPESTATUS[0]}

end=$(date +%s)
duration=$((end - start))
echo "========== Single-Video Test (v4) End: $(date "+%Y-%m-%d %H:%M:%S") (耗时: ${duration}秒, 退出码: ${exit_code}) ==========" | tee -a "$MAIN_LOG"
echo "$(date "+%Y-%m-%d %H:%M:%S") | video=${VIDEO} | duration=${duration}s | exit_code=${exit_code}" >> "$TIME_LOG"

echo "完整日志已保存至: $MAIN_LOG"
echo "耗时记录已追加至: $TIME_LOG"
