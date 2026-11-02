from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple

from api_client_qwen_v2 import build_request_payload_input, call_model_api, create_api_client

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


logger = logging.getLogger("three_stage")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Reduce noise from API clients
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


DEFAULT_MAX_FRAMES = 50
VIDEO_EXTS = (".mp4", ".m4v", ".avi", ".mov", ".mkv", ".webm")
DEFAULT_ARK_API_KEY = ""  # no baked-in fallback; must come from ARK_API_KEY/API_KEY env or --api-key
DEFAULT_ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_ARK_MODEL = "doubao-seed-2-0-lite-260428"
DEFAULT_ARK_REASONING_EFFORT = "medium"
DEFAULT_VIDEO_UNDERSTANDING_FPS = 5.0
HARDCODED_ARK_API_KEY = ""


def format_duration(seconds: float) -> str:
    """Format seconds into a short human-readable duration string."""
    try:
        total = float(seconds)
    except Exception:
        return str(seconds)
    if total < 0:
        total = 0.0
    if total < 60:
        return f"{total:.1f}s"
    total_int = int(round(total))
    s = total_int % 60
    m = (total_int // 60) % 60
    h = (total_int // 3600) % 24
    d = total_int // 86400
    parts: List[str] = []
    if d:
        parts.append(f"{d}d")
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return "".join(parts)


@dataclass
class ApiConfig:
    api_key: str = os.environ.get("ARK_API_KEY") or os.environ.get("API_KEY", HARDCODED_ARK_API_KEY)
    api_base_url: str = os.environ.get("API_BASE_URL", DEFAULT_ARK_BASE_URL)
    model_provider_id: str = os.environ.get("MODEL_PROVIDER_ID", "volcengine_maas")
    model_name: str = os.environ.get("MODEL_NAME", DEFAULT_ARK_MODEL)
    reasoning_effort: str = os.environ.get(
        "ARK_REASONING_EFFORT",
        os.environ.get("REASONING_EFFORT", DEFAULT_ARK_REASONING_EFFORT),
    )
    max_tokens: int = int(os.environ.get("MAX_TOKENS", "32000"))
    temperature: float = float(os.environ.get("TEMPERATURE", "0.8"))
    video_fps: float = float(os.environ.get("VIDEO_UNDERSTANDING_FPS", str(DEFAULT_VIDEO_UNDERSTANDING_FPS)))
    embed_index_on_api_images: bool = os.environ.get("EMBED_INDEX_ON_API_IMAGES", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    verbose: bool = os.environ.get("VERBOSE_LOGGING", "0").strip().lower() in {"1", "true", "yes", "y"}
    api_call_retries: int = int(os.environ.get("API_CALL_RETRIES", "3"))
    api_call_retry_backoff_sec: float = float(os.environ.get("API_CALL_RETRY_BACKOFF_SEC", "1.0"))
    api_call_timeout_sec: float = float(os.environ.get("API_CALL_TIMEOUT_SEC", "420"))


def add_api_cli_args(parser: Any, *, include_no_embed_index: bool = True) -> None:
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ARK_API_KEY") or os.environ.get("API_KEY", DEFAULT_ARK_API_KEY),
        help="API key (or set env: ARK_API_KEY/API_KEY).",
    )
    parser.add_argument(
        "--api-base",
        default=os.environ.get("API_BASE_URL", DEFAULT_ARK_BASE_URL),
        help="API base URL (supports /api/v3, /chat/completions, or /responses; or set env: API_BASE_URL).",
    )
    parser.add_argument(
        "--provider",
        default=os.environ.get("MODEL_PROVIDER_ID", "volcengine_maas"),
        help="Legacy provider arg kept for CLI compatibility; ignored by the Qwen v2 adapter.",
    )
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", DEFAULT_ARK_MODEL), help="Model name.")
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        default=os.environ.get(
            "ARK_REASONING_EFFORT",
            os.environ.get("REASONING_EFFORT", DEFAULT_ARK_REASONING_EFFORT),
        ),
        help="Ark reasoning effort. Defaults to medium.",
    )
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", "32000")), help="Max tokens.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("TEMPERATURE", "0.8")),
        help="Sampling temperature (lower is usually more stable).",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=float(os.environ.get("VIDEO_UNDERSTANDING_FPS", str(DEFAULT_VIDEO_UNDERSTANDING_FPS))),
        help="Files API video preprocessing FPS for Ark video understanding.",
    )
    parser.add_argument(
        "--api-call-retries",
        type=int,
        default=int(os.environ.get("API_CALL_RETRIES", "3")),
        help="Retries for a single model call (transport/service errors).",
    )
    parser.add_argument(
        "--api-call-retry-backoff-sec",
        type=float,
        default=float(os.environ.get("API_CALL_RETRY_BACKOFF_SEC", "1.0")),
        help="Backoff seconds for model-call retries (exponential).",
    )
    parser.add_argument(
        "--api-call-timeout-sec",
        type=float,
        default=float(os.environ.get("API_CALL_TIMEOUT_SEC", "420")),
        help="Timeout seconds for a single model call.",
    )
    if include_no_embed_index:
        parser.add_argument(
            "--no-embed-index",
            action="store_true",
            help="Do not draw 1-based frame indices onto images (will use text labels instead).",
        )
    parser.add_argument("--verbose", action="store_true", help="Log raw model outputs.")


def api_config_from_args(args: Any) -> ApiConfig:
    return ApiConfig(
        api_key=getattr(args, "api_key"),
        api_base_url=getattr(args, "api_base"),
        model_provider_id=getattr(args, "provider"),
        model_name=getattr(args, "model"),
        reasoning_effort=str(getattr(args, "reasoning_effort", DEFAULT_ARK_REASONING_EFFORT)),
        max_tokens=int(getattr(args, "max_tokens")),
        temperature=float(getattr(args, "temperature")),
        video_fps=float(getattr(args, "video_fps")),
        api_call_retries=int(getattr(args, "api_call_retries")),
        api_call_retry_backoff_sec=float(getattr(args, "api_call_retry_backoff_sec")),
        api_call_timeout_sec=float(getattr(args, "api_call_timeout_sec")),
        embed_index_on_api_images=not bool(getattr(args, "no_embed_index", False)),
        verbose=bool(getattr(args, "verbose", False)),
    )


@dataclass
class SamplingConfig:
    max_frames: int = DEFAULT_MAX_FRAMES
    resize_dimension: Optional[Tuple[int, int]] = None
    jpeg_quality: int = 95
    frame_index_fps: Optional[float] = None


def _parse_resize_dimension(value: Any) -> Optional[Tuple[int, int]]:
    text = str(value or "").strip().lower()
    if not text:
        return None
    m = re.fullmatch(r"(\d+)\s*[x,]\s*(\d+)", text)
    if not m:
        raise ValueError(f"Invalid resize dimension: {value!r} (expected WIDTHxHEIGHT)")
    width = int(m.group(1))
    height = int(m.group(2))
    if width <= 0 or height <= 0:
        raise ValueError(f"Resize dimensions must be positive: {value!r}")
    return (width, height)


def _parse_optional_positive_float(value: Any) -> Optional[float]:
    text = str(value or "").strip()
    if not text:
        return None
    val = float(text)
    return val if val > 0 else None


def add_sampling_cli_args(parser: Any, *, default_max_frames: int = 100, default_jpeg_quality: int = 95) -> None:
    default_frame_index_fps = _parse_optional_positive_float(os.environ.get("FRAME_INDEX_FPS", "5.0"))
    parser.add_argument(
        "--max-frames",
        type=int,
        default=int(default_max_frames),
        help="Fallback max frames per pool when dense --frame-index-fps is disabled.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=int(default_jpeg_quality), help="JPEG quality (1-100).")
    parser.add_argument(
        "--frame-index-fps",
        type=float,
        default=default_frame_index_fps,
        help=(
            "Dense frame-index timeline FPS. Default: use --video-fps/API video preprocessing FPS. "
            "Set <=0 to fall back to --max-frames uniform sampling."
        ),
    )
    parser.add_argument(
        "--resize-dimension",
        default=os.environ.get("SAMPLING_RESIZE_DIM", ""),
        help="Optional resize for sampled frames before JPEG encode, format WIDTHxHEIGHT (e.g. 768x768).",
    )


def sampling_config_from_args(args: Any) -> SamplingConfig:
    return SamplingConfig(
        max_frames=int(getattr(args, "max_frames")),
        resize_dimension=_parse_resize_dimension(getattr(args, "resize_dimension", "")),
        jpeg_quality=int(getattr(args, "jpeg_quality")),
        frame_index_fps=_parse_optional_positive_float(getattr(args, "frame_index_fps", None)),
    )


def ensure_frame_index_fps(sampling: SamplingConfig, fallback_fps: Any) -> SamplingConfig:
    if _parse_optional_positive_float(getattr(sampling, "frame_index_fps", None)) is None:
        sampling.frame_index_fps = _parse_optional_positive_float(fallback_fps)
    return sampling


def default_output_root() -> str:
    return os.path.join(os.path.dirname(__file__), "spafa_plan_dataset_long_easy")


def video_id_from_path(video_path: str) -> str:
    base = os.path.basename(video_path)
    name, _ = os.path.splitext(base)
    return name


def build_source_video_meta(video_path: str) -> Dict[str, str]:
    """Return the persisted source-video metadata object used by all JSON outputs."""
    return {"id": video_id_from_path(video_path), "path": os.path.abspath(video_path)}


def source_video_path(source_video: Any) -> str:
    """Read a source video path from the new object shape or the legacy string shape."""
    if isinstance(source_video, dict):
        raw = source_video.get("path")
        return str(raw or "").strip()
    if isinstance(source_video, str):
        return source_video.strip()
    return ""


def source_video_id(source_video: Any) -> str:
    """Read a source video id from the new object shape, falling back to the path stem."""
    if isinstance(source_video, dict):
        raw = source_video.get("id")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        path = source_video_path(source_video)
        return video_id_from_path(path) if path else ""
    if isinstance(source_video, str) and source_video.strip():
        return video_id_from_path(source_video)
    return ""


def sanitize_filename(text: str, *, max_len: Optional[int] = 80) -> str:
    text = re.sub(r"[^\w\s-]", "", (text or "")).strip().lower()
    text = re.sub(r"[-\s]+", "_", text)
    text = text or "unnamed"
    if max_len is None or int(max_len) <= 0:
        return text or "unnamed"
    # Truncate to max_len to avoid OSError: File name too long on macOS (255 byte limit).
    # Break at underscore boundary to keep slug readable.
    if len(text) > int(max_len):
        truncated = text[: int(max_len)]
        # Try to break at last underscore for readability
        last_sep = truncated.rfind("_")
        if last_sep > int(max_len) // 2:
            truncated = truncated[:last_sep]
        text = truncated.rstrip("_")
    return text or "unnamed"


def _split_slug_for_path_components(slug: str, max_bytes: int) -> List[str]:
    """Split a full slug into filesystem-safe path components without dropping text."""
    slug = str(slug or "unnamed")
    max_bytes = max(16, int(max_bytes))
    parts: List[str] = []
    remaining = slug
    while remaining:
        chunk = _truncate_utf8_to_max_bytes(remaining, max_bytes).rstrip("_")
        if not chunk:
            chunk = _truncate_utf8_to_max_bytes(remaining, max_bytes)
        if not chunk:
            break
        parts.append(chunk)
        remaining = remaining[len(chunk) :].lstrip("_")
    return parts or ["unnamed"]


def _flat_slug_filename(
    prefix: str,
    slug: str,
    suffix: str,
    *,
    max_component_bytes: int = 240,
    hash_len: int = 10,
) -> str:
    slug = str(slug or "unnamed")
    candidate = str(prefix) + slug + str(suffix)
    if len(candidate.encode("utf-8")) <= int(max_component_bytes):
        return candidate

    digest = hashlib.sha1(slug.encode("utf-8")).hexdigest()[: int(hash_len)]
    hash_suffix = f"__{digest}"
    budget = (
        int(max_component_bytes)
        - len(str(prefix).encode("utf-8"))
        - len(hash_suffix.encode("utf-8"))
        - len(str(suffix).encode("utf-8"))
    )
    truncated = _truncate_utf8_to_max_bytes(slug, budget).rstrip("_")
    if not truncated:
        truncated = "unnamed"
    return str(prefix) + truncated + hash_suffix + str(suffix)


def stage2_step_clip_relpath(step_id: int, step_goal: str, *, max_component_bytes: int = 240) -> str:
    """Return a flat Stage-2 clip relpath with a filesystem-safe mp4 filename."""
    sid = int(step_id)
    slug = sanitize_filename(step_goal, max_len=None)
    prefix = f"step{sid:02d}_"
    filename = _flat_slug_filename(prefix, slug, ".mp4", max_component_bytes=max_component_bytes)
    return os.path.join("step_clips", filename)


def stage4_atomic_clip_relpath(
    step_dir_name: str,
    atomic_action_id: int,
    action: str,
    *,
    max_component_bytes: int = 240,
) -> str:
    """Return a flat Stage-4 atomic clip relpath with a filesystem-safe mp4 filename."""
    aid = int(atomic_action_id)
    slug = sanitize_filename(action, max_len=None)
    prefix = f"atomic_{aid:02d}_"
    filename = _flat_slug_filename(prefix, slug, ".mp4", max_component_bytes=max_component_bytes)
    return os.path.join(str(step_dir_name), "atomic_clips", filename)


def _truncate_utf8_to_max_bytes(text: str, max_bytes: int) -> str:
    if not isinstance(text, str):
        text = str(text or "")
    if max_bytes <= 0:
        return ""
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text
    raw = raw[:max_bytes]
    while raw:
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            raw = raw[:-1]
    return ""


def stage3_step_folder_basename(step_id: int, step_goal: str, *, max_bytes: int = 220, hash_len: int = 10) -> str:
    """Return a filesystem-safe Stage-3 per-step folder basename.

    Naming scheme is `{step_id:02d}_{sanitize_filename(step_goal)}` but truncated by UTF-8 bytes to
    avoid Linux NAME_MAX (typically 255 bytes) errors. When truncation occurs, append a stable hash
    suffix to avoid collisions.
    """
    prefix = f"{int(step_id):02d}_"
    slug = sanitize_filename(step_goal)
    base = prefix + slug
    if len(base.encode("utf-8")) <= int(max_bytes):
        return base

    digest = hashlib.sha1(str(step_goal or "").encode("utf-8")).hexdigest()[: int(hash_len)]
    suffix = f"__{digest}"
    budget = int(max_bytes) - len(prefix.encode("utf-8")) - len(suffix.encode("utf-8"))
    truncated = _truncate_utf8_to_max_bytes(slug, budget).rstrip("_")
    if not truncated:
        truncated = "unnamed"
    out = prefix + truncated + suffix
    if len(out.encode("utf-8")) <= int(max_bytes):
        return out

    # Last-resort: keep a minimal readable prefix.
    budget2 = int(max_bytes) - len(prefix.encode("utf-8")) - len(suffix.encode("utf-8"))
    truncated2 = _truncate_utf8_to_max_bytes("unnamed", budget2) or "unnamed"
    return prefix + truncated2 + suffix


_PLACEHOLDER_STRINGS = {
    "n/a",
    "na",
    "none",
    "null",
    "unknown",
    "unspecified",
    "tbd",
    "todo",
    "-",
    "...",
}


def _is_placeholder_str(s: str) -> bool:
    if not isinstance(s, str):
        return False
    t = re.sub(r"\s+", " ", s.strip().lower())
    return t in _PLACEHOLDER_STRINGS


_FRAME_REF_RE = re.compile(r"\b(frame|image|img|picture)\s*\d+\b", re.IGNORECASE)
_FRAME_REF_ORDINAL_RE = re.compile(
    r"\b(initial|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last|final|beginning|ending)\s+"
    r"(frame|image|img|picture)\b",
    re.IGNORECASE,
)


def _contains_frame_ref(text: Any) -> bool:
    s = str(text or "")
    return bool(_FRAME_REF_RE.search(s) or _FRAME_REF_ORDINAL_RE.search(s))


_TIME_REF_RE = re.compile(
    r"(?:"
    # Explicit timestamp assignment like "t=3.2s"
    r"\bt\s*=\s*\d+(?:\.\d+)?\s*(?:s|sec|secs|second|seconds|ms|msec|milliseconds?)\b"
    r"|"
    # Numeric duration like "3.2s", "3 seconds"
    r"\b\d+(?:\.\d+)?\s*(?:s|sec|secs|second|seconds|ms|msec|milliseconds?)\b"
    r"|"
    # Timecode like "00:03" or "1:02:03.5"
    r"\b\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?\b"
    r")",
    re.IGNORECASE,
)


def _contains_time_ref(text: Any) -> bool:
    s = str(text or "")
    return bool(_TIME_REF_RE.search(s))


def _text_dedupe_key(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _dedupe_keep_order(items: List[str], *, key_fn: Optional[Callable[[str], str]] = None) -> List[str]:
    if not items:
        return []
    if key_fn is None:
        key_fn = lambda x: x  # noqa: E731
    seen: set[str] = set()
    out: List[str] = []
    for x in items:
        k = key_fn(x)
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


def initialize_api_client(cfg: ApiConfig) -> Any:
    if not cfg.api_key:
        logger.error(
            "No API key configured. Set the ARK_API_KEY (or API_KEY) environment variable, "
            "or pass --api-key explicitly."
        )
        return None
    try:
        return create_api_client(cfg.api_base_url, cfg.api_key)
    except ImportError as e:
        logger.error(f"Missing dependency: {e}. Install with: pip install requests")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize Qwen v2 API client: {e}")
        return None


def extract_json_from_response(response_text: str) -> str:
    if not isinstance(response_text, str):
        raise ValueError("Response was not a string.")

    match = re.search(r"```json\s*([\s\S]+?)\s*```", response_text)
    if match:
        return match.group(1).strip()

    start = response_text.find("{")
    end = response_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return response_text[start : end + 1].strip()

    raise ValueError("Could not find a valid JSON object in the response.")


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=dir_path or ".",
            prefix=os.path.basename(path) + ".tmp.",
            delete=False,
        ) as f:
            tmp_path = f.name
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def write_text(path: str, text: str) -> None:
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=dir_path or ".",
            prefix=os.path.basename(path) + ".tmp.",
            delete=False,
        ) as f:
            tmp_path = f.name
            f.write(text or "")
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


class OutputDirCollisionError(RuntimeError):
    """Raised when the output directory is unsafe to use for the current source video."""


def same_source_video(path_a: str, path_b: str) -> bool:
    """Return True if two paths refer to the same source video.

    We consider:
    - identical absolute paths
    - identical real paths (symlink-resolved)
    - os.path.samefile (inode-level identity; handles hardlinks)

    This is important because datasets are often reorganized via symlinks or multiple mount points, and we
    want to avoid false "output dir collision" errors when the underlying file is the same.
    """
    a = str(path_a or "").strip()
    b = str(path_b or "").strip()
    if not a or not b:
        return False

    try:
        if os.path.abspath(a) == os.path.abspath(b):
            return True
    except Exception:
        pass
    try:
        if os.path.realpath(a) == os.path.realpath(b):
            return True
    except Exception:
        pass
    try:
        if os.path.exists(a) and os.path.exists(b) and os.path.samefile(a, b):
            return True
    except Exception:
        pass
    return False


def ensure_video_out_dir_safe(video_out: str, video_path: str) -> None:
    """Ensure `video_out` is usable for the given `video_path`.

    In this repo, `<video_out>/` is keyed by `video_id_from_path()` (filename stem). In practice, datasets
    are often reorganized (moved folders, new mount points, symlinks), so `run_summary.json.source_video`
    can become stale even when the output artifacts are valid.

    Per the dataset-generation workflow, we treat output completion based on the existence of the expected
    artifacts (e.g. Stage-3 `step_plan.json`), and we only keep `source_video` as a
    *bookkeeping* field. Therefore, if `source_video` differs, we update it to the current path instead
    of aborting the run.
    """
    if not os.path.exists(video_out):
        return
    try:
        entries = [x for x in os.listdir(video_out) if x not in {".", ".."}]
    except Exception:
        return
    if not entries:
        return

    run_summary_path = os.path.join(video_out, "run_summary.json")
    if os.path.exists(run_summary_path):
        try:
            rs = read_json(run_summary_path)
        except Exception:
            return
        src = source_video_path(rs.get("source_video"))
        if src and same_source_video(video_path, src):
            return
        # Update stale/missing bookkeeping path and continue.
        try:
            update_run_summary(run_summary_path, {"source_video": build_source_video_meta(video_path)})
        except Exception:
            pass
        return

    # No run_summary.json: allow (do not fail fast). Artifact existence checks happen elsewhere.
    return


def build_retry_prefix(errors: List[str], prev_output: str) -> str:
    err_text = "\n".join(f"- {e}" for e in (errors or [])[:50])
    prev = (prev_output or "")[:12000]
    return (
        "Your previous output was invalid and failed strict validation.\n"
        "Fix ALL errors and return ONLY the corrected strict JSON.\n\n"
        f"Validation errors:\n{err_text}\n\n"
        "Previous output (for reference; correct it):\n"
        f"{prev}\n\n"
    )


def collect_videos(input_dir: str, exts: Tuple[str, ...]) -> List[str]:
    """Collect video file paths under `input_dir`.

    Note: this function is intentionally recursive (and follows symlink directories safely).
    Many datasets (e.g. EPIC/Ego4D exports) store videos in nested subfolders; a non-recursive scan
    can incorrectly report "No videos found" or cause the pipeline to think "everything is complete"
    (because it never sees the remaining unprocessed videos).
    """
    paths: List[str] = []
    seen_dirs: set[str] = set()
    seen_files: set[str] = set()
    walk_errors: List[str] = []

    def _onerror(e: OSError) -> None:
        walk_errors.append(f"{type(e).__name__}: {e}")

    for root, dirnames, filenames in os.walk(input_dir, followlinks=True, onerror=_onerror):
        # Prevent infinite recursion when followlinks=True (symlink loops).
        real_root = os.path.realpath(root)
        if real_root in seen_dirs:
            dirnames[:] = []
            continue
        seen_dirs.add(real_root)

        # Deterministic ordering (important for reproducibility and logs).
        # Deduplicate directory traversal by realpath, preferring non-symlink dirs when both exist.
        chosen: Dict[str, Tuple[bool, str]] = {}
        for d in dirnames:
            if str(d).startswith("."):
                continue
            full = os.path.join(root, d)
            try:
                real_d = os.path.realpath(full)
            except Exception:
                continue
            if real_d in seen_dirs:
                continue
            is_link = bool(os.path.islink(full))
            rank = (is_link, str(d))
            if real_d not in chosen or rank < chosen[real_d]:
                chosen[real_d] = rank
        dirnames[:] = [rank[1] for rank in chosen.values()]
        dirnames.sort()
        for name in sorted(filenames):
            if not str(name).lower().endswith(exts):
                continue
            p = os.path.join(root, name)
            if os.path.isfile(p):
                # De-duplicate symlinked/hardlinked files.
                try:
                    real_p = os.path.realpath(p)
                except Exception:
                    real_p = p
                if real_p in seen_files:
                    continue
                seen_files.add(real_p)
                paths.append(p)

    if walk_errors:
        sample = " | ".join(walk_errors[:3])
        more = "" if len(walk_errors) <= 3 else f" | ... (+{len(walk_errors) - 3})"
        logger.warning(
            "[collect_videos] Filesystem errors while scanning "
            f"{os.path.abspath(input_dir)} (errors={len(walk_errors)}): {sample}{more}"
        )
    return paths


def can_open_video(video_path: str) -> bool:
    """Best-effort check whether a video is openable by our decoder stack.

    This is used to avoid resuming from corrupted/incomplete MP4s that can exist if a previous
    ffmpeg cut was interrupted (common symptom: 'moov atom not found').
    """
    if not isinstance(video_path, str) or not video_path:
        return False
    if not os.path.exists(video_path):
        return False
    try:
        if os.path.getsize(video_path) <= 0:
            return False
    except Exception:
        return False

    if cv2 is not None:
        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                return False
            # Treat videos with clearly invalid metadata as not openable for our pipeline.
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if total_frames <= 0 or fps <= 0:
                return False
            return True
        except Exception:
            return False
        finally:
            try:
                cap.release()
            except Exception:
                pass

    # Fallback when cv2 is unavailable: lightweight MP4 sanity heuristic.
    lower = video_path.lower()
    if lower.endswith((".mp4", ".mov", ".m4v")):
        try:
            size = int(os.path.getsize(video_path))
            chunk = min(2 * 1024 * 1024, size)
            with open(video_path, "rb") as f:
                head = f.read(chunk)
                tail = b""
                if size > chunk:
                    try:
                        f.seek(max(0, size - chunk), os.SEEK_SET)
                        tail = f.read(chunk)
                    except Exception:
                        tail = b""
            blob = head + tail
            if b"ftyp" not in blob:
                return False
            if b"moov" not in blob:
                return False
            return True
        except Exception:
            return False

    # Unknown container: can't validate without cv2.
    return True


def get_video_metadata(video_path: str) -> Dict[str, Any]:
    if cv2 is None:
        raise RuntimeError("opencv-python (cv2) is required to read video metadata.")
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        if total_frames <= 0 or fps <= 0:
            raise RuntimeError(f"Video has invalid metadata (frames={total_frames}, fps={fps}).")
        return {
            "total_frames": int(total_frames),
            "fps": float(fps),
            "width": int(width),
            "height": int(height),
            "duration_sec": float(total_frames) / float(fps),
        }
    finally:
        cap.release()


def _dense_frame_indices(total_frames: int, source_fps: float, target_fps: float) -> List[int]:
    total = int(total_frames)
    src_fps = float(source_fps)
    tgt_fps = float(target_fps)
    if total <= 1:
        return [0]
    if tgt_fps <= 0:
        raise ValueError(f"target_fps must be positive (got {target_fps})")
    if tgt_fps >= src_fps:
        return list(range(total))

    last_ts = float(total - 1) / src_fps
    count = int(math.floor(last_ts * tgt_fps + 1e-9)) + 1
    raw = [int(round((i / tgt_fps) * src_fps)) for i in range(max(1, count))]
    raw.append(total - 1)

    out: List[int] = []
    seen = set()
    for idx in raw:
        clamped = min(max(0, int(idx)), total - 1)
        if clamped in seen:
            continue
        seen.add(clamped)
        out.append(clamped)
    return sorted(out)


def sample_video_to_frames(
    video_path: str,
    sampling: SamplingConfig,
) -> Tuple[List[Dict[str, Any]], Tuple[int, int]]:
    if cv2 is None:
        raise RuntimeError("opencv-python (cv2) is required to sample frames.")
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_dimensions = (width, height)
        if total_frames <= 0 or fps <= 0:
            raise RuntimeError(f"Video has invalid metadata (frames={total_frames}, fps={fps}).")

        frame_index_fps = _parse_optional_positive_float(getattr(sampling, "frame_index_fps", None))
        if frame_index_fps is not None:
            indices = _dense_frame_indices(total_frames, fps, frame_index_fps)
            sampling_mode = "fps_dense"
        else:
            if sampling.max_frames <= 0:
                raise ValueError(f"max_frames must be positive (got {sampling.max_frames})")
            if sampling.max_frames == 1:
                indices = [0]
            else:
                # Inclusive uniform sampling over [0, total_frames - 1].
                denom = float(sampling.max_frames - 1)
                indices = [int(round(i * (total_frames - 1) / denom)) for i in range(sampling.max_frames)]
            sampling_mode = "uniform_count"
        indices = [min(max(0, idx), total_frames - 1) for idx in indices]
        expected_count = len(indices)
        frames: List[Dict[str, Any]] = []
        last_good: Optional[Dict[str, Any]] = None
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                if last_good is None:
                    continue
                # Keep frame-count stable by reusing previous bytes if a read fails.
                # Duplicate the full previous entry (including timestamp/index) to avoid
                # mismatching image bytes with a different timestamp.
                frames.append({**last_good})
                continue

            if sampling.resize_dimension:
                frame = cv2.resize(frame, sampling.resize_dimension)
            ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(sampling.jpeg_quality)])
            if not ok2:
                if last_good is None:
                    continue
                frames.append({**last_good})
                continue

            b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
            last_good = {
                "base64": b64,
                "timestamp_sec": float(frame_idx) / fps,
                "original_frame_index": int(frame_idx),
                "source_frame_0based": int(frame_idx),
                "source_video_fps": float(fps),
                "source_total_frames": int(total_frames),
                "source_video_duration_sec": float(total_frames) / float(fps),
                "frame_index_sampling_fps": float(frame_index_fps) if frame_index_fps is not None else None,
                "frame_index_sampling_mode": sampling_mode,
            }
            frames.append({**last_good})

        if len(frames) != expected_count and frames:
            # Pad to the target frame count by repeating the last frame.
            while len(frames) < expected_count:
                frames.append({**frames[-1]})
            frames = frames[:expected_count]

        if len(frames) != expected_count:
            raise RuntimeError(f"Failed to sample {expected_count} frames from {video_path} (got {len(frames)}).")
        return frames, original_dimensions
    finally:
        cap.release()


def sample_video_to_frame_index(
    video_path: str,
    sampling: SamplingConfig,
) -> Tuple[List[Dict[str, Any]], Tuple[int, int]]:
    """Build a dense timestamp/frame index without decoding or saving local images."""
    meta = get_video_metadata(video_path)
    total_frames = int(meta["total_frames"])
    fps = float(meta["fps"])
    width = int(meta.get("width", 0) or 0)
    height = int(meta.get("height", 0) or 0)
    frame_index_fps = _parse_optional_positive_float(getattr(sampling, "frame_index_fps", None))
    if frame_index_fps is not None:
        indices = _dense_frame_indices(total_frames, fps, frame_index_fps)
        sampling_mode = "fps_dense"
    else:
        if sampling.max_frames <= 0:
            raise ValueError(f"max_frames must be positive (got {sampling.max_frames})")
        if sampling.max_frames == 1:
            indices = [0]
        else:
            denom = float(sampling.max_frames - 1)
            indices = [int(round(i * (total_frames - 1) / denom)) for i in range(sampling.max_frames)]
        sampling_mode = "uniform_count"

    indices = [min(max(0, int(idx)), total_frames - 1) for idx in indices]
    frames: List[Dict[str, Any]] = []
    for frame_idx in indices:
        frames.append(
            {
                "timestamp_sec": float(frame_idx) / fps,
                "original_frame_index": int(frame_idx),
                "source_frame_0based": int(frame_idx),
                "source_video_fps": float(fps),
                "source_total_frames": int(total_frames),
                "source_video_duration_sec": float(total_frames) / float(fps),
                "frame_index_sampling_fps": float(frame_index_fps) if frame_index_fps is not None else None,
                "frame_index_sampling_mode": sampling_mode,
            }
        )
    return frames, (width, height)


def save_sampled_frames_jpegs(frames: List[Dict[str, Any]], output_dir: str) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    rel_paths: List[str] = []
    for i, fr in enumerate(frames, start=1):
        ts = float(fr.get("timestamp_sec", 0.0))
        name = f"sample_{i:03d}_ts_{ts:.2f}s.jpg"
        path = os.path.join(output_dir, name)
        data = base64.b64decode(fr["base64"]) if isinstance(fr.get("base64"), str) else None
        if not data:
            raise RuntimeError(f"Missing base64 for sampled frame {i}.")
        with open(path, "wb") as f:
            f.write(data)
        rel_paths.append(name)
    return rel_paths


def write_frame_manifest(
    frames: List[Dict[str, Any]],
    sampled_frames_dir: Optional[str],
    manifest_path: str,
) -> Dict[str, Any]:
    manifest_dir = os.path.dirname(manifest_path)
    os.makedirs(manifest_dir, exist_ok=True)

    first = frames[0] if frames else {}
    entries: List[Dict[str, Any]] = []
    for i, fr in enumerate(frames, start=1):
        ts = float(fr.get("timestamp_sec", 0.0))
        source_frame_0based = int(fr.get("source_frame_0based", fr.get("original_frame_index", -1)))
        entry = {
            "frame_index_1based": i,
            "timestamp_sec": ts,
            "original_frame_index": int(fr.get("original_frame_index", -1)),
            "source_frame_0based": source_frame_0based,
        }
        if sampled_frames_dir and isinstance(fr.get("base64"), str):
            name = f"sample_{i:03d}_ts_{ts:.2f}s.jpg"
            abs_img = os.path.join(sampled_frames_dir, name)
            entry["image_relpath"] = os.path.relpath(abs_img, manifest_dir)
        entries.append(entry)

    manifest = {
        "num_frames": len(entries),
        "source_video_fps": float(first.get("source_video_fps", 0.0) or 0.0),
        "source_total_frames": int(first.get("source_total_frames", 0) or 0),
        "source_video_duration_sec": float(first.get("source_video_duration_sec", 0.0) or 0.0),
        "frame_index_sampling_fps": first.get("frame_index_sampling_fps"),
        "frame_index_sampling_mode": str(first.get("frame_index_sampling_mode", "")),
        "note": (
            "frame_index_1based is the 1-based boundary index used in prompts/model outputs; "
            "source_frame_0based is the corresponding 0-based frame number in the original full video."
        ),
        "frames": entries,
    }
    write_json(manifest_path, manifest)
    return manifest


def load_frames_from_manifest(manifest_path: str) -> List[Dict[str, Any]]:
    manifest = read_json(manifest_path)
    base_dir = os.path.dirname(manifest_path)
    frames: List[Dict[str, Any]] = []
    for entry in manifest.get("frames", []):
        rel = entry.get("image_relpath")
        item = {
            "timestamp_sec": float(entry.get("timestamp_sec", 0.0)),
            "original_frame_index": int(entry.get("original_frame_index", -1)),
            "source_frame_0based": int(entry.get("source_frame_0based", entry.get("original_frame_index", -1))),
            "source_video_fps": float(manifest.get("source_video_fps", 0.0) or 0.0),
            "source_total_frames": int(manifest.get("source_total_frames", 0) or 0),
            "source_video_duration_sec": float(manifest.get("source_video_duration_sec", 0.0) or 0.0),
            "frame_index_sampling_fps": manifest.get("frame_index_sampling_fps"),
            "frame_index_sampling_mode": manifest.get("frame_index_sampling_mode"),
        }
        if rel:
            img_path = os.path.join(base_dir, rel)
            if os.path.exists(img_path):
                with open(img_path, "rb") as f:
                    item["base64"] = base64.b64encode(f.read()).decode("utf-8")
        frames.append(item)
    if len(frames) != int(manifest.get("num_frames", len(frames))):
        logger.warning("Manifest frame count mismatch; continuing with loaded frames.")
    return frames


def build_index_manifest_text(frames: List[Dict[str, Any]]) -> str:
    lines = ["Dense Frame Index Manifest (1-based boundary index):"]
    for i, fr in enumerate(frames, start=1):
        ts = float(fr.get("timestamp_sec", 0.0))
        source_frame_0based = int(fr.get("source_frame_0based", fr.get("original_frame_index", -1)))
        lines.append(f"- Frame {i}: t={ts:.3f}s, source_frame_0based={source_frame_0based}")
    return "\n".join(lines)


def _overlay_index_on_base64_image(b64_img: str, index_1based: int) -> str:
    if cv2 is None:
        return b64_img
    try:
        import numpy as np

        data = base64.b64decode(b64_img)
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return b64_img
        # Only overlay the 1-based index to avoid tempting the model to output timestamps.
        # (Timestamps remain available in the on-disk frame_manifest.json for script-side conversion.)
        text = f"Frame {index_1based:02d}"
        cv2.putText(img, text, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            return b64_img
        return base64.b64encode(buf.tobytes()).decode("utf-8")
    except Exception:
        return b64_img


def build_api_content(
    frames: List[Dict[str, Any]],
    embed_index: bool,
    *,
    include_manifest: bool = True,
    include_frame_labels: bool = True,
    label_prefix: str = "Frame",
    label_numbers: bool = True,
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    if include_manifest:
        content.append({"type": "text", "text": build_index_manifest_text(frames)})
    for i, fr in enumerate(frames, start=1):
        b64 = fr.get("base64")
        if embed_index and isinstance(b64, str):
            b64 = _overlay_index_on_base64_image(b64, i)
        if include_frame_labels:
            if label_numbers:
                content.append({"type": "text", "text": f"{label_prefix} {i}"})
            else:
                content.append({"type": "text", "text": str(label_prefix)})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return content


def build_video_api_content(
    video_path: str,
    frames: List[Dict[str, Any]],
    *,
    video_fps: float = DEFAULT_VIDEO_UNDERSTANDING_FPS,
    include_manifest: bool = True,
) -> List[Dict[str, Any]]:
    """Build Responses API content for Ark video understanding while preserving frame-index semantics."""
    content: List[Dict[str, Any]] = [
        {
            "type": "input_video_path",
            "video_path": str(video_path),
            "fps": float(video_fps),
        }
    ]
    if include_manifest:
        manifest = build_index_manifest_text(frames)
        content.append(
            {
                "type": "text",
                "text": (
                    "Frame-index mapping for the existing dataset pipeline:\n"
                    "The attached video is the visual source. When the prompt refers to provided frames, "
                    "interpret those frames as this dense 1-based timestamp boundary timeline. "
                    "Use the video content around each listed timestamp/source_frame_0based pair, and output "
                    "frame_index values only from this manifest.\n\n"
                    f"{manifest}"
                ),
            }
        )
    return content


def cut_video_segment_ffmpeg(
    ffmpeg_bin: str,
    src_video: str,
    start_sec: float,
    end_sec: float,
    dst_video: str,
    overwrite: bool,
    *,
    mode: str = "reencode",
    seek_slop_sec: float = 1.0,
    crf: int = 18,
    preset: str = "veryfast",
    keep_audio: bool = False,
) -> None:
    duration = float(end_sec) - float(start_sec)
    if duration <= 0:
        raise ValueError(f"Non-positive clip duration: start={start_sec}, end={end_sec}")

    out_dir = os.path.dirname(dst_video)
    os.makedirs(out_dir, exist_ok=True)

    # Avoid leaving partially-written/corrupted MP4s at the final path if ffmpeg is interrupted.
    # Write to a temp file then atomically replace on success.
    if not overwrite and os.path.exists(dst_video):
        raise FileExistsError(dst_video)
    suffix = os.path.splitext(dst_video)[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(prefix=".tmp_cut_", suffix=suffix, dir=out_dir, delete=False)
    tmp_path = tmp.name
    tmp.close()

    # For high-quality alignment, default to re-encoding with a hybrid seek:
    # - A fast pre-seek near the target time (start - seek_slop_sec)
    # - An accurate post-seek within the decoded window
    # This avoids the common stream-copy behavior of snapping to previous keyframes.
    mode = (mode or "").strip().lower()
    if mode not in {"copy", "reencode"}:
        raise ValueError(f"Unknown cut mode: {mode} (expected 'copy' or 'reencode')")

    if mode == "copy":
        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_sec:.3f}",
            "-i",
            src_video,
            "-t",
            f"{duration:.3f}",
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            tmp_path,
        ]
    else:
        pre = max(0.0, float(start_sec) - float(seek_slop_sec))
        post = float(start_sec) - pre
        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{pre:.3f}",
            "-i",
            src_video,
            "-ss",
            f"{post:.3f}",
            "-t",
            f"{duration:.3f}",
            "-map",
            "0:v:0",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(int(crf)),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]
        if keep_audio:
            # `0:a?` makes audio optional (some videos have no audio track).
            cmd += ["-map", "0:a?", "-c:a", "aac", "-b:a", "128k"]
        else:
            cmd += ["-an"]
        cmd.append(tmp_path)

    # Always overwrite the temp output.
    cmd.insert(1, "-y")
    logger.info("[ffmpeg] " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)
        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) <= 0:
            raise RuntimeError(f"ffmpeg produced an empty clip: {tmp_path}")
        os.replace(tmp_path, dst_video)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"ffmpeg binary not found: '{ffmpeg_bin}'. Install ffmpeg or pass a valid path via --ffmpeg-bin."
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed (exit={e.returncode}). Command: " + " ".join(cmd)) from e
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def update_run_summary(path: str, updates: Dict[str, Any]) -> None:
    data: Dict[str, Any] = {}
    if os.path.exists(path):
        try:
            data = read_json(path)
        except Exception:
            data = {}
    data.update(updates)
    write_json(path, data)


def normalize_draft_plan(plan: Any) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []

    if not isinstance(plan, dict):
        warnings.append(f"Top-level JSON must be an object; got {type(plan).__name__}.")
        plan = {}

    def _norm_str(v: Any) -> str:
        s = str(v).strip() if v is not None else ""
        return "" if _is_placeholder_str(s) else s

    out: Dict[str, Any] = {
        "high_level_goal": _norm_str(plan.get("high_level_goal", "")),
        "steps": [],
    }
    if not out["high_level_goal"]:
        warnings.append("Missing/empty high_level_goal.")

    steps = plan.get("steps", [])
    if not isinstance(steps, list):
        steps = []
        warnings.append("Top-level 'steps' is not a list; replaced with empty list.")

    # If the model provided a coherent step_id sequence (1..N), prefer that ordering to preserve chronology.
    non_object_steps = sum(1 for s in steps if not isinstance(s, dict))
    if non_object_steps:
        warnings.append(f"Found {non_object_steps} non-object step entries; they will be skipped.")

    dict_steps = [s for s in steps if isinstance(s, dict)]
    parsed_step_ids: List[int] = []
    step_ids_ok = bool(dict_steps)
    for s in dict_steps:
        try:
            sid = int(s.get("step_id"))
        except Exception:
            step_ids_ok = False
            break
        if sid <= 0:
            step_ids_ok = False
            break
        parsed_step_ids.append(sid)
    if (
        step_ids_ok
        and len(set(parsed_step_ids)) == len(parsed_step_ids)
        and set(parsed_step_ids) == set(range(1, len(parsed_step_ids) + 1))
    ):
        if parsed_step_ids != sorted(parsed_step_ids):
            warnings.append("Reordered steps by provided step_id (1..N) to preserve chronology.")
        steps = sorted(dict_steps, key=lambda x: int(x.get("step_id")))

    seen_goals: Dict[str, int] = {}
    for idx, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            warnings.append(f"Step #{idx} is not an object; skipped.")
            continue
        for forbidden in ("critical_frames", "frame_index", "interaction", "keyframe_image_path"):
            if forbidden in step:
                warnings.append(f"Removed unexpected '{forbidden}' in step_id={step.get('step_id')}.")
                step.pop(forbidden, None)

        step_goal = _norm_str(step.get("step_goal", ""))
        if not step_goal:
            step_goal = f"unnamed_step_{idx:02d}"
            warnings.append(f"Empty step_goal at step #{idx}; replaced with '{step_goal}'.")
        if step_goal in seen_goals:
            warnings.append(f"Duplicate step_goal detected: '{step_goal}' (first at step {seen_goals[step_goal]}).")
        else:
            seen_goals[step_goal] = idx

        out["steps"].append(
            {
                # Enforce canonical step_id to avoid downstream ambiguity.
                "step_id": idx,
                "step_goal": step_goal,
            }
        )

    if not out["steps"]:
        warnings.append("Draft contains 0 usable steps after normalization.")
    return out, warnings


def estimate_min_positive_delta_sec(timestamps: List[float]) -> float:
    uniq = sorted({float(x) for x in (timestamps or [])})
    deltas = [b - a for a, b in zip(uniq, uniq[1:]) if b > a]
    if not deltas:
        return 0.1
    # Avoid epsilon too close to 0 which may produce empty clips after seeking/rounding.
    return max(min(deltas), 0.05)


def validate_stage2_localization(
    draft_plan: Dict[str, Any],
    localization: Any,
    num_frames: int,
    *,
    frame_timestamps: Optional[List[float]] = None,
) -> Tuple[bool, List[str], Dict[int, Dict[str, int]]]:
    errors: List[str] = []

    def _is_strict_int(v: Any) -> bool:
        return isinstance(v, int) and not isinstance(v, bool)

    steps = draft_plan.get("steps", [])
    if not isinstance(steps, list) or not steps:
        return False, ["Draft plan has no steps."], {}

    if not isinstance(localization, dict):
        return False, ["Localization output must be a JSON object with a 'steps' list."], {}
    extra_top = sorted(set(localization.keys()) - {"steps"})
    if extra_top:
        errors.append(f"Localization output contains extra top-level keys (not allowed): {extra_top}")

    loc_steps = localization.get("steps", [])
    if not isinstance(loc_steps, list):
        return False, ["Localization JSON missing 'steps' list."], {}

    step_ids: List[int] = []
    for s in steps:
        if not isinstance(s, dict):
            continue
        try:
            step_ids.append(int(s.get("step_id")))
        except Exception:
            continue
    if not step_ids:
        return False, ["Draft plan contains no valid step_id values."], {}
    if len(set(step_ids)) != len(step_ids):
        errors.append("Draft plan has duplicate step_id values (unexpected).")

    expected_ids = set(step_ids)
    ordered_ids = sorted(step_ids)

    # Enforce output ordering as instructed by the Stage-2 prompt: ascending step_id.
    loc_sids_in_list: List[int] = []
    for obj in loc_steps:
        if not isinstance(obj, dict):
            continue
        raw_sid = obj.get("step_id")
        if _is_strict_int(raw_sid):
            loc_sids_in_list.append(int(raw_sid))
    if loc_sids_in_list and loc_sids_in_list != ordered_ids:
        errors.append("Localization 'steps' entries must be in ascending step_id order (match the draft order).")

    allowed_entry_keys = {"step_id", "start_frame_index", "end_frame_index", "value"}
    by_id: Dict[int, Dict[str, int]] = {}
    for i, obj in enumerate(loc_steps):
        if not isinstance(obj, dict):
            errors.append(f"localization.steps[{i}] is not an object.")
            continue
        extra = sorted(set(obj.keys()) - allowed_entry_keys)
        if extra:
            errors.append(f"step_id={obj.get('step_id')} contains extra keys (not allowed): {extra}")
        raw_sid = obj.get("step_id")
        if not _is_strict_int(raw_sid):
            errors.append(f"localization.steps[{i}].step_id missing/invalid (expected int).")
            continue
        sid = int(raw_sid)
        if sid not in expected_ids:
            errors.append(f"Unexpected step_id in localization output: {sid}")
            continue
        if sid in by_id:
            errors.append(f"Duplicate localization entries for step_id={sid}")
            continue
        raw_s = obj.get("start_frame_index")
        raw_e = obj.get("end_frame_index")
        if not _is_strict_int(raw_s) or not _is_strict_int(raw_e):
            errors.append(f"step_id={sid} start_frame_index/end_frame_index missing/invalid (expected int).")
            continue
        raw_v = obj.get("value")
        if not isinstance(raw_v, str) or raw_v not in {"yes", "no"}:
            errors.append(f"step_id={sid} value field missing/invalid (expected 'yes' or 'no').")
            continue
        s = int(raw_s)
        e = int(raw_e)
        by_id[sid] = {"start_frame_index": s, "end_frame_index": e, "value": raw_v}

    # Ensure all steps present and no extras
    for sid in step_ids:
        if sid not in by_id:
            errors.append(f"Missing localization for step_id={sid}")

    # Validate constraints in step_id order
    ordered = ordered_ids
    prev_end: Optional[int] = None
    prev_end_ts: Optional[float] = None
    for pos, sid in enumerate(ordered):
        seg = by_id.get(sid)
        if not seg:
            continue
        s = seg["start_frame_index"]
        e = seg["end_frame_index"]
        if s < 1 or s > num_frames:
            errors.append(f"step_id={sid} start_frame_index out of range: {s}")
        if e < 1 or e > num_frames + 1:
            errors.append(f"step_id={sid} end_frame_index out of range: {e}")
        if not (s < e):
            errors.append(f"step_id={sid} requires start_frame_index < end_frame_index (got {s}, {e})")
        is_first = pos == 0
        is_last = pos == len(ordered) - 1
        if is_first and s != 1:
            errors.append(f"HARD full-video coverage violated: first step start_frame_index must be 1 (got {s})")
        if prev_end is not None and not is_first and s != prev_end:
            errors.append(
                f"Contiguity constraint violated: step_id={sid} requires start_frame_index == prev_end (got {s} vs {prev_end})"
            )
        if not is_last and e == num_frames + 1:
            errors.append(
                f"Only the LAST step may use end_frame_index={num_frames + 1} (got step_id={sid} with end_frame_index={e})"
            )
        if is_last and e != num_frames + 1:
            errors.append(f"HARD full-video coverage violated: last step end_frame_index must be {num_frames + 1} (got {e})")

        if frame_timestamps and 1 <= s <= len(frame_timestamps) and 1 <= e <= len(frame_timestamps) + 1:
            s_ts = float(frame_timestamps[s - 1])
            if e == len(frame_timestamps) + 1:
                e_ts = float(frame_timestamps[-1]) + estimate_min_positive_delta_sec(frame_timestamps)
            else:
                e_ts = float(frame_timestamps[e - 1])
            if not (s_ts < e_ts):
                if abs(s_ts - e_ts) < 1e-9:
                    errors.append(
                        f"step_id={sid} selected indices map to identical timestamps (got {s_ts:.2f}, {e_ts:.2f}); "
                        "this often happens when sampled frames are duplicates. Choose a larger end_frame_index that shows clear progress."
                    )
                else:
                    errors.append(f"step_id={sid} requires start_sec < end_sec (got {s_ts:.2f}, {e_ts:.2f})")
            if prev_end_ts is not None and s_ts < prev_end_ts:
                errors.append(
                    f"Monotonic constraint violated in seconds: step_id={sid} start_sec {s_ts:.2f} < prev_end_sec {prev_end_ts:.2f}"
                )
            prev_end_ts = e_ts
        prev_end = e

    return len(errors) == 0, errors, by_id


def save_keyframe_images_from_manifest(
    manifest_path: str,
    frame_indices_1based: List[int],
    output_dir: str,
) -> Dict[int, str]:
    manifest = read_json(manifest_path)
    base_dir = os.path.dirname(manifest_path)

    by_idx: Dict[int, Dict[str, Any]] = {}
    for entry in manifest.get("frames", []):
        if not isinstance(entry, dict):
            continue
        try:
            idx1 = int(entry.get("frame_index_1based"))
        except Exception:
            continue
        by_idx[idx1] = entry

    os.makedirs(output_dir, exist_ok=True)
    out: Dict[int, str] = {}
    for idx1 in frame_indices_1based:
        if idx1 not in by_idx:
            raise ValueError(f"frame_index not found in manifest: {idx1}")
        entry = by_idx[idx1]
        rel = entry.get("image_relpath")
        if not isinstance(rel, str) or not rel:
            raise ValueError(f"Manifest entry missing image_relpath for frame_index={idx1}")
        src = os.path.join(base_dir, rel)
        if not os.path.exists(src):
            raise FileNotFoundError(src)
        ts = float(entry.get("timestamp_sec", 0.0))
        name = f"frame_{idx1:03d}_ts_{ts:.2f}s.jpg"
        dst = os.path.abspath(os.path.join(output_dir, name))
        shutil.copyfile(src, dst)
        out[idx1] = dst
    return out


def now_utc_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def get_run_summary_schema_fingerprint(run_summary_path: str) -> Optional[str]:
    if not os.path.exists(run_summary_path):
        return None
    try:
        rs = read_json(run_summary_path)
    except Exception:
        return None
    raw = rs.get("schema_fingerprint")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


@lru_cache(maxsize=1)
def three_stage_schema_fingerprint() -> str:
    """Fingerprint the prompts/schema spec used by the three-stage pipeline.

    This is primarily used to prevent unsafe "resume" across prompt/schema changes.
    """
    module_dir = os.path.dirname(__file__)
    schema_paths = [
        os.path.join(module_dir, "prompts.py"),
        os.path.join(module_dir, "hand_step_pipeline.py"),
    ]
    try:
        data = b""
        for schema_path in schema_paths:
            with open(schema_path, "rb") as f:
                data += os.path.basename(schema_path).encode("utf-8") + b"\0" + f.read() + b"\0"
    except Exception:
        return "sha256:unknown"
    return "sha256:" + hashlib.sha256(data).hexdigest()


def guard_schema_fingerprint(
    run_summary_path: str,
    video_out: str,
    *,
    stage: str,
    overwrite: bool,
    allow_legacy_resume: bool,
    will_resume: bool,
) -> str:
    current_fp = three_stage_schema_fingerprint()
    existing_fp = get_run_summary_schema_fingerprint(run_summary_path)
    if existing_fp is not None and existing_fp != current_fp and not overwrite:
        raise RuntimeError(
            f"Refusing to run {stage}: schema_fingerprint mismatch.\n"
            f"- video_out: {os.path.abspath(video_out)}\n"
            f"- existing: {existing_fp}\n"
            f"- current : {current_fp}\n"
            "Use --overwrite to regenerate outputs, or use a different --output-root."
        )
    if will_resume and existing_fp is None and not allow_legacy_resume:
        raise RuntimeError(
            f"Refusing to resume {stage} legacy outputs without schema_fingerprint in run_summary.json.\n"
            f"- video_out: {os.path.abspath(video_out)}\n"
            "Re-run with --overwrite, or pass --allow-legacy-resume to bypass this check."
        )
    return current_fp


def _is_non_retryable_model_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in (
        "total tokens of multi-modal content and text exceed max message tokens",
        "context_length_exceeded",
        "maximum context length",
    ))


def call_chat_completion(
    client: Any,
    cfg: ApiConfig,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    *,
    video_fps: Optional[float] = None,
    temperature: Optional[float] = None,
) -> Tuple[str, Dict[str, int]]:
    """Call VLM and return (text_content, usage_dict).

    usage_dict has keys: prompt_tokens, completion_tokens, total_tokens (all int).
    If the API does not return usage, values default to 0.
    """
    max_attempts = int(getattr(cfg, "api_call_retries", 1) or 1)
    max_attempts = max(1, max_attempts)
    backoff_sec = float(getattr(cfg, "api_call_retry_backoff_sec", 1.0) or 0.0)
    timeout_sec = float(getattr(cfg, "api_call_timeout_sec", 420.0) or 420.0)

    start = time.time()
    _ = max_tokens
    payload_input = build_request_payload_input(messages)
    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for attempt in range(1, max_attempts + 1):
        try:
            content, usage = call_model_api(
                client,
                model_name=cfg.model_name,
                payload_input=payload_input,
                timeout_sec=timeout_sec,
                video_fps=float(
                    video_fps
                    if video_fps is not None
                    else getattr(cfg, "video_fps", DEFAULT_VIDEO_UNDERSTANDING_FPS)
                ),
                max_output_tokens=int(max_tokens),
                reasoning_effort=str(
                    getattr(cfg, "reasoning_effort", DEFAULT_ARK_REASONING_EFFORT)
                ),
                temperature=float(
                    temperature if temperature is not None else getattr(cfg, "temperature", 0.8)
                ),
            )
            break
        except Exception as e:
            if _is_non_retryable_model_error(e):
                raise RuntimeError(f"Non-retryable model input error: {e}") from e
            if attempt >= max_attempts:
                raise RuntimeError(f"Model call failed after {max_attempts} attempts: {e}") from e
            sleep_sec = min(max(0.0, backoff_sec) * (2 ** (attempt - 1)), 8.0)
            logger.warning(f"Model call error (attempt {attempt}/{max_attempts}): {e}; retrying in {sleep_sec:.1f}s")
            time.sleep(sleep_sec)

    end = time.time()
    logger.info(f"Model call finished in {end - start:.2f}s (tokens: in={usage.get('prompt_tokens', 0)}, out={usage.get('completion_tokens', 0)})")
    if cfg.verbose:
        logger.info("Raw model output:\n" + content)
    return content, usage


# ---- Cost Calculation ----

# Pricing table: model_name -> (input_price_per_1k_tokens_yuan, output_price_per_1k_tokens_yuan)
MODEL_PRICING = {
    "Doubao-Seed-2.0-pro":  (0.0032, 0.016),
    "Doubao-Seed-2.0-lite": (0.0006, 0.0036),
    "Doubao-Seed-1.8":      (0.0008, 0.002),
}


def calculate_cost_yuan(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost in CNY given model name and token counts."""
    pricing = MODEL_PRICING.get(model_name)
    if not pricing:
        return 0.0
    input_price, output_price = pricing
    cost = (prompt_tokens / 1000.0) * input_price + (completion_tokens / 1000.0) * output_price
    return round(cost, 6)


def compute_cost_summary_from_run_summary(run_summary_path: str) -> Dict[str, Any]:
    """Read run_summary.json and compute total cost across all stages.

    Returns a dict with per-stage and total token usage and cost in CNY.
    """
    try:
        summary = read_json(run_summary_path)
    except Exception:
        return {}

    model_name = ""
    api_config = summary.get("api_config", {})
    if isinstance(api_config, dict):
        model_name = str(api_config.get("model_name", ""))

    total_prompt = 0
    total_completion = 0
    total_calls = 0
    per_stage = {}

    for stage_key in ("stage1", "stage2", "stage3", "stage4", "stage5"):
        stage_data = summary.get(stage_key, {})
        if not isinstance(stage_data, dict):
            continue
        tu = stage_data.get("token_usage", {})
        if not isinstance(tu, dict):
            continue
        pt = int(tu.get("prompt_tokens", 0) or 0)
        ct = int(tu.get("completion_tokens", 0) or 0)
        calls = int(tu.get("api_calls", 0) or 0)
        cost = calculate_cost_yuan(model_name, pt, ct)
        per_stage[stage_key] = {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "api_calls": calls,
            "cost_yuan": cost,
        }
        total_prompt += pt
        total_completion += ct
        total_calls += calls

    total_cost = calculate_cost_yuan(model_name, total_prompt, total_completion)

    return {
        "model_name": model_name,
        "per_stage": per_stage,
        "total": {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "api_calls": total_calls,
            "cost_yuan": total_cost,
        },
    }


def print_cost_summary(run_summary_path: str) -> None:
    """Print a human-readable cost summary to the logger."""
    cs = compute_cost_summary_from_run_summary(run_summary_path)
    if not cs or not cs.get("total"):
        return

    model = cs.get("model_name", "unknown")
    total = cs["total"]
    per_stage = cs.get("per_stage", {})

    lines = [
        f"===== Cost Summary (model: {model}) =====",
    ]
    for sk in ("stage1", "stage2", "stage3", "stage4", "stage5"):
        sd = per_stage.get(sk)
        if sd:
            lines.append(
                f"  {sk}: {sd['api_calls']} calls, "
                f"in={sd['prompt_tokens']:,} out={sd['completion_tokens']:,} tokens, "
                f"cost={sd['cost_yuan']:.4f} CNY"
            )
    lines.append(
        f"  TOTAL: {total['api_calls']} calls, "
        f"in={total['prompt_tokens']:,} out={total['completion_tokens']:,} tokens "
        f"(total={total['total_tokens']:,}), "
        f"cost={total['cost_yuan']:.4f} CNY"
    )
    lines.append("=" * 50)

    for line in lines:
        logger.info(line)
