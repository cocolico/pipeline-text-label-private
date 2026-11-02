#!/usr/bin/env python3
"""Render native LeRobot task_index labels beside video, outside the dataset."""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import sys

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lerobot_annotation.dataset import atomic_json, inspect, path_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--html", type=Path, required=True)
    args = parser.parse_args()
    data = inspect(args.input)
    output = args.html.resolve()
    if data["root"] == output or data["root"] in output.parents:
        parser.error("Preview must be outside the delivered dataset")
    if not 0 <= args.episode < len(data["episodes"]):
        parser.error("Episode does not exist")
    episode = data["episodes"][args.episode]
    table = pq.read_table(data["root"] / path_for(data["info"], args.episode), columns=["task_index"])
    indices = table["task_index"].to_pylist()
    vocabulary = {r["task_index"]: r["task"] for r in data["tasks"]}
    segments = []
    for frame, index in enumerate(indices):
        if not segments or segments[-1]["task_index"] != index:
            segments.append({"start_frame": frame, "end_frame": frame + 1,
                             "task_index": index, "text": vocabulary[index]})
        else:
            segments[-1]["end_frame"] = frame + 1
    fps = data["info"]["fps"]
    output.parent.mkdir(parents=True, exist_ok=True)
    videos = {name: Path(os.path.relpath(data["root"] / path_for(data["info"], args.episode, video_key=name), output.parent)).as_posix()
              for name in data["cameras"]}
    buttons = " ".join(f'<button data-video="{html.escape(path, quote=True)}">{html.escape(name)}</button>' for name, path in videos.items())
    initial = videos.get("observation.images.cam0", next(iter(videos.values())))
    rows = []
    for number, segment in enumerate(segments, 1):
        start, end = segment["start_frame"] / fps, segment["end_frame"] / fps
        rows.append(f'<tr data-start="{start}" data-end="{end}"><td>{number}</td>'
                    f'<td><button data-seek="{start}">{start:.2f}–{end:.2f}s</button></td>'
                    f'<td>[{segment["start_frame"]}, {segment["end_frame"]})</td>'
                    f'<td>{segment["task_index"]}</td><td>{html.escape(segment["text"])}</td></tr>')
    document = '''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>LeRobot 真实标注预览</title><style>
body{font-family:system-ui;margin:24px;background:#f7f8fa;color:#17212b}
video{width:min(960px,100%);background:#111}table{border-collapse:collapse;width:100%;background:white}
td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}button{padding:7px;cursor:pointer}
tr.active{background:#dceefe}.note{color:#555}</style>
<h1>LeRobot 真实模型标注预览</h1>'''
    document += (f'<p>Episode {args.episode:06d} · {episode["length"]} 帧 · {fps} FPS · '
                 f'{len(segments)} 个连续任务区间。标签直接读取 Parquet 的 task_index。</p>'
                 '<p class="note">点击时间跳转；可以切换三路视频。动作边界为模型估计，尚未逐帧人工验证。</p>'
                 f'<div>{buttons}</div><p><video id="video" controls src="{html.escape(initial,quote=True)}"></video></p>'
                 '<table><thead><tr><th>步骤</th><th>时间</th><th>帧区间</th><th>task_index</th><th>任务描述</th></tr></thead><tbody>'
                 + ''.join(rows) + '</tbody></table>')
    document += '''<script>
const video=document.getElementById('video');
document.querySelectorAll('[data-seek]').forEach(b=>b.onclick=()=>{video.currentTime=Number(b.dataset.seek)});
document.querySelectorAll('[data-video]').forEach(b=>b.onclick=()=>{const t=video.currentTime;video.src=b.dataset.video;video.addEventListener('loadedmetadata',()=>{video.currentTime=t},{once:true})});
video.ontimeupdate=()=>document.querySelectorAll('tr[data-start]').forEach(r=>r.classList.toggle('active',video.currentTime>=Number(r.dataset.start)&&video.currentTime<Number(r.dataset.end)));
</script></html>'''
    output.write_text(document)
    atomic_json(output.with_suffix(".json"), {"episode_index": args.episode, "frames": episode["length"],
                                             "fps": fps, "segments": segments})
    print(output)


if __name__ == "__main__": main()
