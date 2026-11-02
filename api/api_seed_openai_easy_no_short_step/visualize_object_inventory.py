#!/usr/bin/env python3
import argparse
import glob
import json
import os
import subprocess
import textwrap

import cv2
import numpy as np


def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def in_intervals(frame_id, intervals):
    return any(int(x["start_frame_id"]) <= frame_id <= int(x["end_frame_id"]) for x in intervals or [])


def lines_for_object(obj, frame_id, width=48):
    visible = in_intervals(frame_id, obj.get("visible_intervals"))
    missing = in_intervals(frame_id, obj.get("temporarily_missing_intervals"))
    status = "VISIBLE" if visible else "TEMPORARILY MISSING" if missing else "OUTSIDE REPORTED VISIBILITY"
    color = (80, 235, 120) if visible else (30, 190, 255) if missing else (165, 165, 165)
    head = f"{obj.get('entity_ref', '?')}  {status}"
    body = [
        str(obj.get("canonical_name", "unnamed object")),
        f"identity={obj.get('identity_status', '?')}  confidence={float(obj.get('confidence', 0)):.2f}",
    ]
    interactions = [x for x in obj.get("hand_interactions", []) if int(x.get("start_frame_id", -1)) <= frame_id <= int(x.get("end_frame_id", -1))]
    if interactions:
        body.append("interaction: " + "; ".join(f"{x.get('hand')}: {x.get('action')}" for x in interactions))
    transitions = [x for x in obj.get("state_transitions", []) if abs(int(x.get("frame_id", -999999)) - frame_id) <= 15]
    if transitions:
        body.append("transition: " + "; ".join(f"{x.get('from')} -> {x.get('to')}" for x in transitions))
    wrapped = []
    for line in body:
        wrapped.extend(textwrap.wrap(line, width=width) or [""])
    return head, wrapped, color


def put(frame, text, xy, scale=.52, color=(235, 235, 235), thickness=1):
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def choose_clip(rgb_out, step_id):
    record_path = os.path.join(rgb_out, "stage1", f"step_{step_id:03d}", "action_semantics.json")
    if os.path.isfile(record_path):
        rec = load_json(record_path)
        rel = rec.get("visualization_clip_path") or rec.get("clip_path")
        if rel and os.path.isfile(os.path.join(rgb_out, rel)):
            return os.path.join(rgb_out, rel)
    found = sorted(glob.glob(os.path.join(rgb_out, "stage1", "step_clips", f"step_{step_id:03d}_*_visualization.mp4")))
    if not found:
        raise FileNotFoundError(f"No visualization clip for step {step_id}")
    return found[-1]


def render_step(rgb_out, inventory_path):
    data = load_json(inventory_path)
    step_id = int(data["step_id"])
    start = int(data["source_frame_range"]["start_frame_id"])
    end = int(data["source_frame_range"]["end_frame_id"])
    clip = choose_clip(rgb_out, step_id)
    output_dir = os.path.join(rgb_out, "object_inventory_visualizations")
    os.makedirs(output_dir, exist_ok=True)
    final_path = os.path.join(output_dir, f"step_{step_id:03d}_object_inventory_visualization.mp4")
    raw_path = final_path + ".raw.mp4"
    cap = cv2.VideoCapture(clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    count = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    vw, vh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    panel = max(620, int(vw * .58))
    writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (vw + panel, vh))
    index = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        source_frame = start + int(round(index * max(0, end - start) / max(1, count - 1)))
        canvas = np.full((vh, vw + panel, 3), 20, dtype=np.uint8)
        canvas[:, :vw] = image
        cv2.line(canvas, (vw, 0), (vw, vh), (75, 75, 75), 2)
        put(canvas, f"OBJECT INVENTORY  |  STEP {step_id:03d}", (vw + 22, 38), .72, (255, 255, 255), 2)
        put(canvas, f"original frame: {source_frame}   step range: [{start}, {end}]", (vw + 22, 68), .5, (190, 190, 190))
        y = 108
        for obj in data.get("object_inventory", []):
            head, body, color = lines_for_object(obj, source_frame)
            put(canvas, head, (vw + 22, y), .56, color, 2); y += 25
            for line in body:
                put(canvas, line, (vw + 34, y), .45, (225, 225, 225)); y += 21
            y += 15
            if y > vh - 70:
                break
        x0, x1, ty = vw + 22, vw + panel - 22, vh - 34
        cv2.line(canvas, (x0, ty), (x1, ty), (95, 95, 95), 5)
        for obj_i, obj in enumerate(data.get("object_inventory", [])):
            color = [(80, 235, 120), (255, 180, 70), (180, 100, 255), (255, 120, 170)][obj_i % 4]
            for interval in obj.get("visible_intervals", []):
                a = x0 + int((int(interval["start_frame_id"]) - start) * (x1 - x0) / max(1, end - start))
                b = x0 + int((int(interval["end_frame_id"]) - start) * (x1 - x0) / max(1, end - start))
                cv2.line(canvas, (a, ty - 8 - obj_i * 5), (b, ty - 8 - obj_i * 5), color, 3)
        now_x = x0 + int((source_frame - start) * (x1 - x0) / max(1, end - start))
        cv2.line(canvas, (now_x, ty - 30), (now_x, ty + 10), (255, 255, 255), 2)
        writer.write(canvas)
        index += 1
    cap.release(); writer.release()
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", raw_path,
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", final_path], check=True)
    os.remove(raw_path)
    os.chmod(final_path, 0o666)
    return final_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rgb_out")
    args = parser.parse_args()
    paths = sorted(glob.glob(os.path.join(args.rgb_out, "stage2", "step_*", "object_inventory.json")))
    for path in paths:
        print(render_step(args.rgb_out, path), flush=True)
    os.chmod(os.path.join(args.rgb_out, "object_inventory_visualizations"), 0o777)


if __name__ == "__main__":
    main()
