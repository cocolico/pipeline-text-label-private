from __future__ import annotations

import base64
import mimetypes
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


LOCAL_FILE_UPLOAD_LIMIT_BYTES = 512 * 1024 * 1024
_FILE_ID_CACHE: Dict[Tuple[str, int, int, float], str] = {}


def resolve_sdk_base_url(api_base_url: str) -> str:
    base = str(api_base_url or "").strip().rstrip("/")
    if not base:
        return ""
    if base.endswith("/chat/completions"):
        return base[: -len("/chat/completions")]
    if base.endswith("/responses"):
        return base[: -len("/responses")]
    return base


def create_api_client(api_base_url: str, api_key: str) -> OpenAI:
    return OpenAI(api_key=str(api_key or ""), base_url=resolve_sdk_base_url(api_base_url), max_retries=0)


def build_request_payload_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        normalized.append(
            {
                "role": str(msg.get("role") or "user"),
                "content": _normalize_message_content(msg.get("content")),
            }
        )
    return normalized


def call_model_api(
    client: OpenAI,
    *,
    model_name: str,
    payload_input: List[Dict[str, Any]],
    timeout_sec: float = 180.0,
    video_fps: float = 10.0,
    max_output_tokens: Optional[int] = None,
    reasoning_effort: str = "low",
    temperature: Optional[float] = None,
) -> Tuple[str, Dict[str, int]]:
    """Call Volcengine Ark Responses API and return (text_content, usage_dict)."""
    uploaded_file_ids: List[str] = []
    try:
        instructions, response_input = _prepare_responses_input(
            client,
            payload_input,
            timeout_sec=timeout_sec,
            video_fps=video_fps,
            uploaded_file_ids=uploaded_file_ids,
        )
        kwargs: Dict[str, Any] = {
            "model": str(model_name or ""),
            "input": response_input,
            "stream": False,
            "reasoning": {"effort": _normalize_reasoning_effort(reasoning_effort)},
            "extra_body": {"thinking": {"type": "enabled"}},
        }
        if instructions:
            kwargs["instructions"] = instructions
        if max_output_tokens is not None and int(max_output_tokens) > 0:
            kwargs["max_output_tokens"] = int(max_output_tokens)
        if temperature is not None:
            kwargs["temperature"] = float(temperature)

        response = client.with_options(timeout=float(timeout_sec)).responses.create(**kwargs)
        return _extract_response_text(response), _extract_usage(response)
    finally:
        _delete_uploaded_files(client, uploaded_file_ids, timeout_sec=timeout_sec)


def _normalize_reasoning_effort(value: Any) -> str:
    effort = str(value or "medium").strip().lower()
    if effort not in {"low", "medium", "high"}:
        raise ValueError(
            f"Invalid Ark reasoning effort: {value!r}; expected low, medium, or high."
        )
    return effort


def _prepare_responses_input(
    client: OpenAI,
    payload_input: List[Dict[str, Any]],
    *,
    timeout_sec: float,
    video_fps: float,
    uploaded_file_ids: Optional[List[str]] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    instructions: List[str] = []
    response_input: List[Dict[str, Any]] = []
    for msg in payload_input or []:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user").strip().lower() or "user"
        content = msg.get("content")
        content_items = content if isinstance(content, list) else [{"type": "input_text", "text": str(content or "")}]
        prepared_items = [
            _prepare_content_item(
                client,
                item,
                timeout_sec=timeout_sec,
                video_fps=video_fps,
                uploaded_file_ids=uploaded_file_ids,
            )
            for item in content_items
        ]
        prepared_items = [item for item in prepared_items if item]
        if not prepared_items:
            continue
        if role in {"system", "developer"}:
            instructions.extend(_content_items_to_text(prepared_items))
            continue
        response_input.append({"role": role, "content": prepared_items})
    if not response_input:
        response_input.append({"role": "user", "content": [{"type": "input_text", "text": ""}]})
    return "\n\n".join(x for x in instructions if x.strip()), response_input


def _prepare_content_item(
    client: OpenAI,
    item: Any,
    *,
    timeout_sec: float,
    video_fps: float,
    uploaded_file_ids: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    if isinstance(item, str):
        return {"type": "input_text", "text": item}
    if not isinstance(item, dict):
        return {"type": "input_text", "text": str(item or "")}

    kind = str(item.get("type") or "").strip().lower()
    if kind in {"text", "input_text"}:
        return {"type": "input_text", "text": str(item.get("text") or "")}

    if kind in {"input_video_path", "video_path"}:
        path = str(item.get("video_path") or item.get("path") or "").strip()
        fps = _coerce_fps(item.get("fps"), video_fps)
        file_id = _upload_video_file(client, path, fps=fps, timeout_sec=timeout_sec)
        if uploaded_file_ids is not None:
            uploaded_file_ids.append(file_id)
        return {"type": "input_video", "file_id": file_id}

    if kind == "input_video":
        if item.get("file_id"):
            return {"type": "input_video", "file_id": str(item.get("file_id"))}
        if item.get("video_url"):
            out = {"type": "input_video", "video_url": str(item.get("video_url"))}
            if item.get("fps") is not None:
                out["fps"] = _coerce_fps(item.get("fps"), video_fps)
            return out

    if kind == "video_url":
        video_url = item.get("video_url")
        url = video_url.get("url") if isinstance(video_url, dict) else video_url
        if isinstance(url, str) and url:
            out = {"type": "input_video", "video_url": _maybe_to_data_url(url)}
            fps_raw = video_url.get("fps") if isinstance(video_url, dict) else item.get("fps")
            if fps_raw is not None:
                out["fps"] = _coerce_fps(fps_raw, video_fps)
            return out

    if kind in {"image_url", "input_image"}:
        image_url = item.get("image_url")
        url = image_url.get("url") if isinstance(image_url, dict) else image_url
        if isinstance(url, str) and url:
            return {"type": "input_image", "image_url": _maybe_to_data_url(url)}

    return item


def _upload_video_file(client: OpenAI, video_path: str, *, fps: float, timeout_sec: float) -> str:
    if not video_path:
        raise ValueError("Missing video_path for input_video.")
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)
    size = os.path.getsize(video_path)
    if size <= 0:
        raise ValueError(f"Video file is empty: {video_path}")
    if size > LOCAL_FILE_UPLOAD_LIMIT_BYTES:
        mib = size / 1024 / 1024
        raise ValueError(
            f"Local Files API upload exceeds 512 MiB limit: {video_path} ({mib:.2f} MiB). "
            "Use a smaller file or the TOS-based Files API path."
        )

    stat = os.stat(video_path)
    cache_key = (os.path.abspath(video_path), int(size), int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))), float(fps))
    # Uploaded Files API entries are deleted after each model call to avoid
    # exhausting Ark account storage quota, so cached file ids may be stale.
    _FILE_ID_CACHE.pop(cache_key, None)

    extra_body = {"preprocess_configs": {"video": {"fps": float(fps)}}}
    with open(video_path, "rb") as f:
        uploaded = client.with_options(timeout=float(timeout_sec)).files.create(
            file=f,
            purpose="user_data",
            extra_body=extra_body,
        )
    file_id = str(getattr(uploaded, "id", "") or "")
    if not file_id:
        raise RuntimeError(f"Files API upload did not return file id for: {video_path}")

    _wait_for_file_processing(client, file_id, timeout_sec=timeout_sec)
    return file_id


def _delete_uploaded_files(client: OpenAI, file_ids: List[str], *, timeout_sec: float) -> None:
    seen = set()
    for file_id in file_ids or []:
        fid = str(file_id or "").strip()
        if not fid or fid in seen:
            continue
        seen.add(fid)
        try:
            client.with_options(timeout=min(float(timeout_sec), 60.0)).files.delete(fid)
        except Exception:
            pass


def _wait_for_file_processing(client: OpenAI, file_id: str, *, timeout_sec: float) -> None:
    deadline = time.time() + max(float(timeout_sec), 60.0)
    last_status = ""
    while True:
        file_obj = client.with_options(timeout=float(timeout_sec)).files.retrieve(file_id)
        last_status = str(getattr(file_obj, "status", "") or "").strip().lower()
        if last_status in {"failed", "error", "cancelled", "canceled"}:
            raise RuntimeError(f"File processing failed for {file_id}: status={last_status}")
        if last_status != "processing":
            return
        if time.time() >= deadline:
            raise TimeoutError(f"Timed out waiting for file processing: {file_id} (last_status={last_status})")
        time.sleep(2.0)


def _content_items_to_text(items: List[Dict[str, Any]]) -> List[str]:
    text_parts: List[str] = []
    for item in items:
        if isinstance(item, dict) and str(item.get("type")) == "input_text":
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                text_parts.append(text)
    return text_parts


def _extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    parts: List[str] = []
    output = getattr(response, "output", None)
    if isinstance(output, list):
        for item in output:
            content = _get_field(item, "content")
            if isinstance(content, list):
                for part in content:
                    text = _get_field(part, "text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text)
            text = _get_field(item, "text")
            if isinstance(text, str) and text.strip():
                parts.append(text)
    if parts:
        return "\n".join(parts).strip()

    choices = getattr(response, "choices", None)
    if isinstance(choices, list) and choices:
        msg = getattr(choices[0], "message", None)
        content = getattr(msg, "content", None) if msg is not None else None
        if isinstance(content, str):
            return content

    if hasattr(response, "model_dump_json"):
        return response.model_dump_json()
    return str(response or "")


def _extract_usage(response: Any) -> Dict[str, int]:
    usage_raw = getattr(response, "usage", None)
    prompt = _usage_int(usage_raw, "prompt_tokens", "input_tokens")
    completion = _usage_int(usage_raw, "completion_tokens", "output_tokens")
    total = _usage_int(usage_raw, "total_tokens")
    if total <= 0:
        total = prompt + completion
    return {
        "prompt_tokens": int(prompt),
        "completion_tokens": int(completion),
        "total_tokens": int(total),
    }


def _usage_int(obj: Any, *names: str) -> int:
    for name in names:
        value = _get_field(obj, name)
        try:
            if value is not None:
                return int(value or 0)
        except Exception:
            continue
    return 0


def _get_field(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _normalize_message_content(content: Any) -> Any:
    if isinstance(content, list):
        return [_normalize_content_item(item) for item in content]
    return content


def _normalize_content_item(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    kind = str(item.get("type") or "").strip().lower()
    if kind != "image_url":
        return item

    image_url = item.get("image_url")
    if not isinstance(image_url, dict):
        return item
    url = image_url.get("url")
    if not isinstance(url, str) or not url:
        return item

    normalized_url = _maybe_to_data_url(url)
    if normalized_url == url:
        return item

    normalized = dict(item)
    normalized["image_url"] = dict(image_url)
    normalized["image_url"]["url"] = normalized_url
    return normalized


def _maybe_to_data_url(value: str) -> str:
    if str(value).startswith("data:"):
        return value
    if not os.path.exists(value):
        return value
    mime_type = mimetypes.guess_type(value)[0] or "application/octet-stream"
    with open(value, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _coerce_fps(value: Any, default: float) -> float:
    try:
        fps = float(value)
    except Exception:
        fps = float(default)
    if fps <= 0:
        return float(default)
    return fps
