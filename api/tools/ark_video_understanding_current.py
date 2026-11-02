#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI


DEFAULT_ARK_API_KEY = ""
DEFAULT_ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_ARK_MODEL = "doubao-seed-2-1-pro-260628"
DEFAULT_ARK_REASONING_EFFORT = "medium"
DEFAULT_VIDEO_FPS = 5.0
DEFAULT_PROMPT = (
    "Summarize the visible human operation in this video as an action task. "
    "Be accurate, grounded in the visible frames, and avoid inventing off-screen details."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Ark video-understanding example.")
    parser.add_argument("video", help="Local video path.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt sent with the video.")
    parser.add_argument("--system", default="You are a precise video understanding assistant.")
    parser.add_argument("--api-key", default=os.environ.get("ARK_API_KEY") or os.environ.get("API_KEY", DEFAULT_ARK_API_KEY))
    parser.add_argument("--api-base", default=os.environ.get("API_BASE_URL", DEFAULT_ARK_BASE_URL))
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", DEFAULT_ARK_MODEL))
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        default=os.environ.get("ARK_REASONING_EFFORT", DEFAULT_ARK_REASONING_EFFORT),
    )
    parser.add_argument("--video-fps", type=float, default=float(os.environ.get("VIDEO_FPS", str(DEFAULT_VIDEO_FPS))))
    parser.add_argument("--max-output-tokens", type=int, default=int(os.environ.get("MAX_OUTPUT_TOKENS", "2048")))
    parser.add_argument("--timeout-sec", type=float, default=float(os.environ.get("API_TIMEOUT_SEC", "420")))
    parser.add_argument("--json-output", action="store_true", help="Print response text and usage as JSON.")
    parser.add_argument("--keep-uploaded-file", action="store_true", help="Do not delete the uploaded file after the call.")
    return parser.parse_args()


def resolve_base_url(api_base: str) -> str:
    base = str(api_base or "").strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base[: -len("/chat/completions")]
    if base.endswith("/responses"):
        return base[: -len("/responses")]
    return base


def wait_for_file_ready(client: OpenAI, file_id: str, timeout_sec: float) -> None:
    deadline = time.time() + max(float(timeout_sec), 60.0)
    while True:
        file_obj = client.with_options(timeout=float(timeout_sec)).files.retrieve(file_id)
        status = str(getattr(file_obj, "status", "") or "").lower()
        if status in {"failed", "error", "cancelled", "canceled"}:
            raise RuntimeError(f"File processing failed: file_id={file_id}, status={status}")
        if status != "processing":
            return
        if time.time() >= deadline:
            raise TimeoutError(f"Timed out waiting for file processing: file_id={file_id}, status={status}")
        time.sleep(2.0)


def upload_video(client: OpenAI, video_path: Path, video_fps: float, timeout_sec: float) -> str:
    if not video_path.exists():
        raise FileNotFoundError(str(video_path))
    if video_path.stat().st_size <= 0:
        raise ValueError(f"Video file is empty: {video_path}")

    with video_path.open("rb") as f:
        uploaded = client.with_options(timeout=float(timeout_sec)).files.create(
            file=f,
            purpose="user_data",
            extra_body={"preprocess_configs": {"video": {"fps": float(video_fps)}}},
        )
    file_id = str(getattr(uploaded, "id", "") or "")
    if not file_id:
        raise RuntimeError("Files API upload did not return a file id.")

    wait_for_file_ready(client, file_id, timeout_sec)
    return file_id


def extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    parts: List[str] = []
    for item in getattr(response, "output", []) or []:
        for part in getattr(item, "content", []) or []:
            text = getattr(part, "text", None)
            if isinstance(text, str) and text.strip():
                parts.append(text)
    if parts:
        return "\n".join(parts).strip()

    if hasattr(response, "model_dump_json"):
        return response.model_dump_json()
    return str(response)


def extract_usage(response: Any) -> Dict[str, int]:
    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", 0)) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", getattr(usage, "output_tokens", 0)) or 0)
    total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def main() -> int:
    args = parse_args()
    client = OpenAI(api_key=args.api_key, base_url=resolve_base_url(args.api_base), max_retries=0)
    file_id = upload_video(client, Path(args.video), float(args.video_fps), float(args.timeout_sec))

    try:
        response = client.with_options(timeout=float(args.timeout_sec)).responses.create(
            model=args.model,
            instructions=args.system,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_video", "file_id": file_id},
                        {"type": "input_text", "text": args.prompt},
                    ],
                }
            ],
            max_output_tokens=int(args.max_output_tokens),
            stream=False,
            reasoning={"effort": args.reasoning_effort},
            extra_body={"thinking": {"type": "enabled"}},
        )
    finally:
        if not args.keep_uploaded_file:
            try:
                client.with_options(timeout=min(float(args.timeout_sec), 60.0)).files.delete(file_id)
            except Exception:
                pass

    text = extract_response_text(response)
    usage = extract_usage(response)
    if args.json_output:
        print(json.dumps({"text": text, "usage": usage, "file_id": file_id}, ensure_ascii=False, indent=2))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
