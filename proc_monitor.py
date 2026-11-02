#!/usr/bin/env python3
"""Per-process (pipeline.py + all descendants) resource sampler.

Usage: proc_monitor.py <root_pid> <interval_sec> <csv_path>
Samples until the root pid exits.
"""
import sys
import csv
import re
import time
import threading
import subprocess

import psutil

root_pid = int(sys.argv[1])
interval = float(sys.argv[2])
csv_path = sys.argv[3]

FIELDS = [
    "timestamp", "num_processes", "cpu_percent_sum", "rss_mb_sum",
    "disk_read_bytes_total", "disk_write_bytes_total",
    "disk_read_bytes_delta", "disk_write_bytes_delta",
    "gpu_mem_mb_sum",
    "net_sent_kb_total", "net_recv_kb_total",
    "net_sent_kb_delta", "net_recv_kb_delta",
    "pids",
]

# ---------------------------------------------------------------------------
# nethogs integration: nethogs -t -v1 prints cumulative KB sent/received per
# process, refreshed every NETHOGS_DELAY seconds. There is no /proc/<pid>/net
# equivalent on Linux, so this external tool is the only way to get real
# per-process network attribution. Requires cap_net_raw/cap_net_admin on the
# nethogs binary (already granted via setcap so it runs without sudo).
NETHOGS_BIN = "/usr/sbin/nethogs"
NETHOGS_DELAY = max(1, int(round(interval)))

# pid -> (sent_kb_total, recv_kb_total), updated continuously by the reader thread
net_totals = {}
net_lock = threading.Lock()

_line_re = re.compile(r"^(.+)/(\d+)/(\d+)\t([\d.]+)\t([\d.]+)$")


def _nethogs_reader(proc):
    for raw_line in proc.stdout:
        line = raw_line.rstrip("\n")
        m = _line_re.match(line)
        if not m:
            continue
        pid_str, sent_str, recv_str = m.group(2), m.group(4), m.group(5)
        try:
            pid = int(pid_str)
            sent_kb = float(sent_str)
            recv_kb = float(recv_str)
        except ValueError:
            continue
        if pid == 0:
            continue
        with net_lock:
            net_totals[pid] = (sent_kb, recv_kb)


def start_nethogs():
    try:
        proc = subprocess.Popen(
            [NETHOGS_BIN, "-t", "-v", "1", "-d", str(NETHOGS_DELAY)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
    except Exception:
        return None
    thread = threading.Thread(target=_nethogs_reader, args=(proc,), daemon=True)
    thread.start()
    return proc


def net_kb_for_pids(pid_set):
    sent = 0.0
    recv = 0.0
    with net_lock:
        for pid in pid_set:
            s, r = net_totals.get(pid, (0.0, 0.0))
            sent += s
            recv += r
    return sent, recv


def tree_pids(pid):
    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return []
    pids = [root.pid]
    try:
        pids.extend(c.pid for c in root.children(recursive=True))
    except psutil.Error:
        pass
    return pids


def gpu_mem_for_pids(pid_set):
    if not pid_set:
        return 0
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except Exception:
        return 0
    total = 0
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 2:
            continue
        p, mem = parts
        if p.isdigit() and int(p) in pid_set and mem.isdigit():
            total += int(mem)
    return total


# pid -> psutil.Process, kept alive across iterations so cpu_percent(None)
# reports a real delta instead of always seeing a "first call" on a fresh object.
tracked = {}


def sync_tracked(pid):
    current = set(tree_pids(pid))
    for gone_pid in set(tracked) - current:
        del tracked[gone_pid]
    for new_pid in current - set(tracked):
        try:
            proc = psutil.Process(new_pid)
            proc.cpu_percent(None)  # prime
            tracked[new_pid] = proc
        except psutil.Error:
            pass
    return current


with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f, delimiter="|")
    writer.writerow(FIELDS)

sync_tracked(root_pid)
prev_read = None
prev_write = None
prev_net_sent = None
prev_net_recv = None

nethogs_proc = start_nethogs()

while psutil.pid_exists(root_pid):
    current_pids = sync_tracked(root_pid)
    if not current_pids:
        break

    time.sleep(interval)

    cpu_sum = 0.0
    rss_sum = 0
    read_total = 0
    write_total = 0
    have_io = False
    live_pids = []
    for pid, proc in list(tracked.items()):
        try:
            cpu_sum += proc.cpu_percent(None)
            rss_sum += proc.memory_info().rss
            live_pids.append(pid)
        except psutil.Error:
            continue
        try:
            io = proc.io_counters()
            read_total += io.read_bytes
            write_total += io.write_bytes
            have_io = True
        except (psutil.Error, AttributeError):
            pass

    read_delta = (read_total - prev_read) if (have_io and prev_read is not None) else 0
    write_delta = (write_total - prev_write) if (have_io and prev_write is not None) else 0
    if have_io:
        prev_read, prev_write = read_total, write_total

    gpu_mem = gpu_mem_for_pids(set(live_pids))

    net_sent, net_recv = net_kb_for_pids(set(live_pids))
    net_sent_delta = (net_sent - prev_net_sent) if prev_net_sent is not None else 0.0
    net_recv_delta = (net_recv - prev_net_recv) if prev_net_recv is not None else 0.0
    prev_net_sent, prev_net_recv = net_sent, net_recv

    row = [
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        len(live_pids),
        f"{cpu_sum:.2f}",
        f"{rss_sum / (1024 * 1024):.1f}",
        read_total if have_io else "NA",
        write_total if have_io else "NA",
        read_delta if have_io else "NA",
        write_delta if have_io else "NA",
        gpu_mem,
        f"{net_sent:.2f}",
        f"{net_recv:.2f}",
        f"{net_sent_delta:.2f}",
        f"{net_recv_delta:.2f}",
        ";".join(str(x) for x in live_pids),
    ]
    with open(csv_path, "a", newline="") as f:
        csv.writer(f, delimiter="|").writerow(row)

if nethogs_proc is not None:
    nethogs_proc.terminate()
