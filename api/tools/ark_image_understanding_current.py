#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI


DEFAULT_ARK_API_KEY = ""
DEFAULT_ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_ARK_MODEL = "doubao-seed-2-1-pro-260628"
DEFAULT_ARK_REASONING_EFFORT = "medium"
DEFAULT_PROMPT = (
    "Describe the visible content of this image accurately and concisely. "
    "Focus on objects, hands/tools, spatial relationships, and any visible action evidence."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Ark image-understanding example.")
    parser.add_argument("image", help="Local image path, remote image URL, or data URL.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt sent with the image.")
    parser.add_argument("--system", default="You are a precise visual understanding assistant.")
    parser.add_argument("--api-key", default=os.environ.get("ARK_API_KEY") or os.environ.get("API_KEY", DEFAULT_ARK_API_KEY))
    parser.add_argument("--api-base", default=os.environ.get("API_BASE_URL", DEFAULT_ARK_BASE_URL))
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", DEFAULT_ARK_MODEL))
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        default=os.environ.get("ARK_REASONING_EFFORT", DEFAULT_ARK_REASONING_EFFORT),
    )
    parser.add_argument("--max-output-tokens", type=int, default=int(os.environ.get("MAX_OUTPUT_TOKENS", "2048")))
    parser.add_argument("--timeout-sec", type=float, default=float(os.environ.get("API_TIMEOUT_SEC", "420")))
    parser.add_argument("--json-output", action="store_true", help="Print response text and usage as JSON.")
    return parser.parse_args()


def resolve_base_url(api_base: str) -> str:
    base = str(api_base or "").strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base[: -len("/chat/completions")]
    if base.endswith("/responses"):
        return base[: -len("/responses")]
    return base


def image_to_url(value: str) -> str:
    if value.startswith(("http://", "https://", "data:")):
        return value

    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(value)

    mime_type = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


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
    image_url = image_to_url(str(args.image))

    response = client.with_options(timeout=float(args.timeout_sec)).responses.create(
        model=args.model,
        instructions=args.system,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": image_url},
                    {"type": "input_text", "text": args.prompt},
                ],
            }
        ],
        max_output_tokens=int(args.max_output_tokens),
        stream=False,
        reasoning={"effort": args.reasoning_effort},
        extra_body={"thinking": {"type": "enabled"}},
    )

    text = extract_response_text(response)
    usage = extract_usage(response)
    if args.json_output:
        print(json.dumps({"text": text, "usage": usage}, ensure_ascii=False, indent=2))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
