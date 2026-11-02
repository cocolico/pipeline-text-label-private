#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================================
# 资源监控版 Pipeline 运行脚本
# 功能：运行 pipeline.py 并收集资源使用情况（CPU、内存、显存、GPU卡数、磁盘I/O、网络、Tokens）
# ============================================================================

PIPELINE_DIR="/mnt/storage59/wangxinyi_code/version4.1.2/api/api_seed_openai_easy_no_short_step"
VENV_ACTIVATE="/home/user/wangxinyi/.venv/bin/activate"
SEARCH_ROOT="/mnt/storage59/test-1/Accommodation_and_Food_Service_Activities/Food_and_Beverage_Services"
OUTPUT_DIR_NAME="processed_text_label"

VIDEO_FPS=5.0
FRAME_INDEX_FPS=5.0
API_CALL_RETRIES=8
API_CALL_RETRY_BACKOFF=2.0
API_CALL_TIMEOUT=420
STAGES="1,2,3"
NUM_WORKERS=1

MONITOR_INTERVAL=5
MONITOR_LOG_DIR="./resource_monitor_logs"

get_gpu_info() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null || echo "GPU: nvidia-smi failed"
    else
        echo "GPU: nvidia-smi not found"
    fi
}

get_system_resources() {
    local timestamp
    timestamp=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
    
    local cpu_percent mem_percent mem_used mem_total
    cpu_percent=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1 || echo "0")
    mem_used=$(free -m | awk '/Mem:/ {print $3}')
    mem_total=$(free -m | awk '/Mem:/ {print $2}')
    mem_percent=$(awk "BEGIN {printf \"%.2f\", ($mem_used/$mem_total)*100}")
    
    local gpu_info gpu_count
    if command -v nvidia-smi &> /dev/null; then
        gpu_count=$(nvidia-smi --list-gpus | wc -l || echo "0")
        gpu_info=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null || echo "N/A")
    else
        gpu_count="0"
        gpu_info="N/A"
    fi
    
    local disk_read disk_write
    disk_read=$(cat /proc/diskstats | awk '{sum+=$6} END {print sum}' || echo "0")
    disk_write=$(cat /proc/diskstats | awk '{sum+=$10} END {print sum}' || echo "0")
    
    local net_recv net_send
    net_recv=$(cat /proc/net/dev | awk '{sum+=$2} END {print sum}' || echo "0")
    net_send=$(cat /proc/net/dev | awk '{sum+=$10} END {print sum}' || echo "0")
    
    echo "$timestamp|$cpu_percent|$mem_percent|$mem_used|$mem_total|$gpu_count|$gpu_info|$disk_read|$disk_write|$net_recv|$net_send"
}

parse_tokens_from_summary() {
    local summary_file="$1"
    if [[ -f "$summary_file" ]]; then
        python3 -c "
import json
import sys
try:
    with open('$summary_file', 'r') as f:
        data = json.load(f)
    stage1 = data.get('stage1', {})
    stage2 = data.get('stage2', {})
    stage3 = data.get('stage3', {})
    t1 = stage1.get('token_usage', {})
    t2 = stage2.get('token_usage', {})
    t3 = stage3.get('token_usage', {})
    print(f\"Stage1: prompt={t1.get('prompt_tokens', 0)}, completion={t1.get('completion_tokens', 0)}, total={t1.get('total_tokens', 0)}\")
    print(f\"Stage2: prompt={t2.get('prompt_tokens', 0)}, completion={t2.get('completion_tokens', 0)}, total={t2.get('total_tokens', 0)}\")
    print(f\"Stage3: prompt={t3.get('prompt_tokens', 0)}, completion={t3.get('completion_tokens', 0)}, total={t3.get('total_tokens', 0)}\")
    total_pt = t1.get('prompt_tokens', 0) + t2.get('prompt_tokens', 0) + t3.get('prompt_tokens', 0)
    total_ct = t1.get('completion_tokens', 0) + t2.get('completion_tokens', 0) + t3.get('completion_tokens', 0)
    total_t = t1.get('total_tokens', 0) + t2.get('total_tokens', 0) + t3.get('total_tokens', 0)
    print(f'TOTAL: prompt={total_pt}, completion={total_ct}, total={total_t}')
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
"
    else
        echo "Tokens: run_summary.json not found"
    fi
}

init_monitor_log() {
    mkdir -p "$MONITOR_LOG_DIR"
    local log_file="$MONITOR_LOG_DIR/resource_monitor_$(date +%Y%m%d_%H%M%S).csv"
    echo "timestamp|cpu_percent|mem_percent|mem_used_MB|mem_total_MB|gpu_count|gpu_mem_used_MB|gpu_mem_total_MB|gpu_utilization_percent|gpu_mem_utilization_percent|disk_read_sectors|disk_write_sectors|net_recv_bytes|net_send_bytes" > "$log_file"
    echo "$log_file"
}

monitor_resources() {
    local log_file="$1"
    local pid="$2"
    local end_time="$3"
    
    while kill -0 "$pid" 2>/dev/null; do
        if [[ $(date +%s) -ge $end_time ]]; then
            break
        fi
        get_system_resources >> "$log_file"
        sleep "$MONITOR_INTERVAL"
    done
}

calculate_resource_usage() {
    local log_file="$1"
    if [[ ! -f "$log_file" ]] || [[ $(wc -l < "$log_file") -le 1 ]]; then
        echo "Insufficient data for resource calculation"
        return
    fi
    
    python3 -c "
import csv
import sys

log_file = '$log_file'
try:
    with open(log_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        print('No data')
        return
    
    cpu_vals = [float(r['cpu_percent']) for r in rows if r['cpu_percent'].replace('.','').isdigit()]
    mem_used_vals = [int(r['mem_used_MB']) for r in rows if r['mem_used_MB'].isdigit()]
    disk_read_vals = [int(r['disk_read_sectors']) for r in rows if r['disk_read_sectors'].isdigit()]
    disk_write_vals = [int(r['disk_write_sectors']) for r in rows if r['disk_write_sectors'].isdigit()]
    net_recv_vals = [int(r['net_recv_bytes']) for r in rows if r['net_recv_bytes'].isdigit()]
    net_send_vals = [int(r['net_send_bytes']) for r in rows if r['net_send_bytes'].isdigit()]
    
    print('=== Resource Usage Summary ===')
    print(f'CPU: avg={sum(cpu_vals)/len(cpu_vals):.2f}% max={max(cpu_vals):.2f}%')
    print(f'Memory: avg={sum(mem_used_vals)/len(mem_used_vals):.0f}MB max={max(mem_used_vals)}MB')
    print(f'Disk I/O: read={disk_read_vals[-1]-disk_read_vals[0] if len(disk_read_vals)>1 else 0} sectors write={disk_write_vals[-1]-disk_write_vals[0] if len(disk_write_vals)>1 else 0} sectors')
    print(f'Network: recv={net_recv_vals[-1]-net_recv_vals[0] if len(net_recv_vals)>1 else 0} bytes send={net_send_vals[-1]-net_send_vals[0] if len(net_send_vals)>1 else 0} bytes')
    
    gpu_count = rows[0]['gpu_count']
    gpu_mem = rows[0]['gpu_mem_used_MB']
    gpu_util = rows[0]['gpu_utilization_percent']
    print(f'GPU Count: {gpu_count}')
    if gpu_count != '0':
        print(f'GPU Memory: {gpu_mem}MB')
        print(f'GPU Utilization: {gpu_util}%')
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
"
}

echo "=========================================="
echo "Resource Monitoring Pipeline Runner"
echo "=========================================="

source "$VENV_ACTIVATE"
cd "$PIPELINE_DIR"

LOG_FILE=$(init_monitor_log)
echo "Monitor log: $LOG_FILE"

echo "=== Initial System Resources ==="
echo "GPU Info:"
get_gpu_info
echo ""

START_TIME=$(date +%s)
START_DISK_READ=$(cat /proc/diskstats | awk '{sum+=$6} END {print sum}')
START_NET_RECV=$(cat /proc/net/dev | awk '{sum+=$2} END {print sum}')

echo "=== Starting Pipeline Execution ==="

find "$SEARCH_ROOT" -type f -name "rgb.mp4" | head -n 1 | while IFS= read -r INPUT_VIDEO; do
    PARENT_DIR=$(dirname "$INPUT_VIDEO")
    CAMERA_DIR=$(dirname "$PARENT_DIR")
    OUTPUT_ROOT="${CAMERA_DIR}/${OUTPUT_DIR_NAME}"

    echo "===== 处理: $INPUT_VIDEO ====="
    echo "输出: $OUTPUT_ROOT"

    PIPELINE_PID=""
    
    python3 pipeline.py \
        --input-video "$INPUT_VIDEO" \
        --output-root "$OUTPUT_ROOT" \
        --video-fps "$VIDEO_FPS" \
        --frame-index-fps "$FRAME_INDEX_FPS" \
        --api-call-retries "$API_CALL_RETRIES" \
        --api-call-retry-backoff-sec "$API_CALL_RETRY_BACKOFF" \
        --api-call-timeout-sec "$API_CALL_TIMEOUT" \
        --continue-on-error \
        --stages "$STAGES" \
        --num-workers "$NUM_WORKERS" \
        --overwrite \
        --no-retry-failed &
    
    PIPELINE_PID=$!
    END_TIME=$(($(date +%s) + 7200))
    
    monitor_resources "$LOG_FILE" "$PELINE_PID" "$END_TIME" &
    MONITOR_PID=$!
    
    wait $PELINE_PID
    PIPELINE_EXIT=$?
    
    kill $MONITOR_PID 2>/dev/null || true
    
    echo "===== Pipeline finished with exit code: $PIPELINE_EXIT ====="
    echo ""
    
    echo "=== Resource Usage (calculated from monitor log) ==="
    calculate_resource_usage "$LOG_FILE"
    echo ""
    
    echo "=== Tokens Usage ==="
    parse_tokens_from_summary "${OUTPUT_ROOT}/run_summary.json"
    echo ""
    
    echo "=========================================="
    echo "资源需求汇总"
    echo "=========================================="
    echo "CPU: 查看上方 CPU 平均/峰值使用率"
    echo "内存: 查看上方 Memory 平均/峰值"
    echo "显存: 查看 GPU Memory (如适用)"
    echo "GPU 卡数: $(head -2 "$LOG_FILE" | tail -1 | cut -d'|' -f6)"
    echo "磁盘 I/O: 查看上方 Disk I/O 读写扇区数"
    echo "网络: 查看上方 Network 收发字节数"
    echo "Tokens: 查看上方 Tokens Usage"
    echo "=========================================="
done