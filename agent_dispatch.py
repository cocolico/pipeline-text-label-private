#!/usr/bin/env python3
"""Route each video to the glove or natural-hand Docker pipeline.

The router makes one VLM-only classification call per video, then starts the
selected image.  API keys, image names, model and paths are runtime options.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path

from api.api_seed_openai_easy_no_short_step.api_client_qwen_v2 import (
    build_request_payload_input, call_model_api, create_api_client,
)

DEFAULT_BASE = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_MODEL = "doubao-seed-2-0-lite-260428"

def _head_clip(video: Path, seconds: float) -> Path:
    fd, name = tempfile.mkstemp(prefix="text_label_agent_", suffix=".mp4")
    os.close(fd)
    out = Path(name)
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video), "-t", str(seconds), "-c", "copy", str(out),
    ], check=True)
    return out

def classify(video: Path, key: str, base: str, model: str, seconds: float) -> str:
    clip = _head_clip(video, seconds)
    client = create_api_client(base, key)
    prompt = (
        "Classify the hands in this video. Return JSON only: "
        '{"hand_state":"gloved"|"natural"|"uncertain","confidence":0.0}. '
        "Use gloved when a work glove or thick glove covers the hand; use natural "
        "only when skin/natural hand is clearly visible. If uncertain, choose uncertain."
    )
    payload = build_request_payload_input([{"role": "user", "content": [
        {"type": "input_text", "text": prompt},
        {"type": "input_video_path", "video_path": str(clip), "fps": 2},
    ]}])
    try:
        text, _ = call_model_api(client, model_name=model, payload_input=payload,
                                 timeout_sec=420, video_fps=2, max_output_tokens=300,
                                 reasoning_effort="low")
        try:
            obj = json.loads(text.strip().replace("```json", "").replace("```", "").strip())
            state = str(obj.get("hand_state", "uncertain")).lower()
            return state if state in {"gloved", "natural"} else "uncertain"
        except Exception:
            return "uncertain"
    finally:
        clip.unlink(missing_ok=True)

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    p.add_argument("--glove-image", default=os.getenv("TEXT_LABEL_GLOVE_IMAGE", "text-label:2.7"))
    p.add_argument("--natural-image", default=os.getenv("TEXT_LABEL_NATURAL_IMAGE", "text-label:2.6"))
    p.add_argument("--uncertain-image", default=os.getenv("TEXT_LABEL_UNCERTAIN_IMAGE", "text-label:2.7"))
    p.add_argument("--api-key", default=os.getenv("ARK_API_KEY") or os.getenv("API_KEY"))
    p.add_argument("--api-base", default=os.getenv("API_BASE_URL", DEFAULT_BASE))
    p.add_argument("--model", default=os.getenv("MODEL_NAME", DEFAULT_MODEL))
    p.add_argument("--classify-seconds", type=float,
                   default=float(os.getenv("TEXT_LABEL_AGENT_CLASSIFY_SECONDS", "3")),
                   help="Only the first N seconds are sent for hand/glove routing.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--", dest="pipeline_args", nargs=argparse.REMAINDER)
    a = p.parse_args()
    if not a.video.is_file(): p.error(f"video not found: {a.video}")
    if not a.api_key and not a.dry_run: p.error("set ARK_API_KEY or API_KEY")
    if a.classify_seconds <= 0: p.error("--classify-seconds must be positive")
    state = "uncertain" if a.dry_run else classify(a.video, a.api_key, a.api_base, a.model, a.classify_seconds)
    image = a.glove_image if state == "gloved" else a.natural_image if state == "natural" else a.uncertain_image
    raw = a.video.parent
    video_dir = raw.parent if raw.name == "raw_data" else raw
    out = video_dir / "processed_text_label"
    # Mount the parent of the camera directory, preserving a short stable path.
    cont_video = "/data/in/" + video_dir.name
    cmd = ["docker", "run", "--rm", "-e", "ARK_API_KEY", "-v", f"{video_dir.parent}:/data/in:rw", image,
           "--input-video-dir", cont_video, "--output-root", cont_video + "/processed_text_label",
           "--output-per-video-sibling", "--stages", "1,2,3", "--continue-on-error"]
    if a.pipeline_args: cmd.extend(a.pipeline_args)
    print(json.dumps({"video": str(a.video), "hand_state": state, "image": image, "output": str(out)}, ensure_ascii=False), flush=True)
    if a.dry_run: return 0
    return subprocess.call(cmd)

if __name__ == "__main__": raise SystemExit(main())
