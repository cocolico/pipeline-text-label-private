#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import html
import json
import math
import mimetypes
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from openai import OpenAI

from ark_image_understanding_current import (
    DEFAULT_ARK_API_KEY,
    DEFAULT_ARK_BASE_URL,
    DEFAULT_ARK_MODEL,
    DEFAULT_ARK_REASONING_EFFORT,
)


OFFICIAL_COORD_MIN = 0
OFFICIAL_COORD_MAX = 999
OFFICIAL_COORD_DIVISOR = 1000

POINT_TAG_PATTERN = re.compile(
    r"\s*<point>\s*(-?\d+)\s+(-?\d+)\s*</point>\s*",
    re.IGNORECASE,
)
BBOX_TAG_PATTERN = re.compile(
    r"\s*<bbox>\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*</bbox>\s*",
    re.IGNORECASE,
)


def add_api_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ARK_API_KEY") or os.environ.get("API_KEY", DEFAULT_ARK_API_KEY),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--api-base", default=os.environ.get("API_BASE_URL", DEFAULT_ARK_BASE_URL))
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", DEFAULT_ARK_MODEL))
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        default=os.environ.get("ARK_REASONING_EFFORT", DEFAULT_ARK_REASONING_EFFORT),
    )
    parser.add_argument(
        "--output-dir",
        help="Artifact directory. Defaults to ark_visual_grounding_results/<image>/<test>.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write/print the official request without calling the API.",
    )


def require_local_image(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    return path


def image_dimensions(path: Path) -> Tuple[int, int]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            payload = json.loads(completed.stdout)
            streams = payload.get("streams") or []
            if streams:
                width = int(streams[0]["width"])
                height = int(streams[0]["height"])
                if width > 0 and height > 0:
                    return width, height

    identify = shutil.which("magick") or shutil.which("identify")
    if identify:
        command = [identify]
        if Path(identify).name == "magick":
            command.append("identify")
        command.extend(["-format", "%w %h", str(path)])
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode == 0:
            width_text, height_text = completed.stdout.strip().split()[:2]
            width, height = int(width_text), int(height_text)
            if width > 0 and height > 0:
                return width, height

    raise RuntimeError("Cannot read image dimensions. Install ffprobe or ImageMagick.")


def image_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def call_openai_grounding(
    *,
    image_path: Path,
    prompt: str,
    api_key: str,
    api_base: str,
    model: str,
    reasoning_effort: str = DEFAULT_ARK_REASONING_EFFORT,
) -> Tuple[str, Dict[str, int]]:
    """Use OpenAI SDK Chat Completions with Ark's official Grounding tags."""
    client = OpenAI(api_key=api_key, base_url=api_base, max_retries=0)
    response = client.chat.completions.create(
        model=model,
        reasoning_effort=reasoning_effort,
        extra_body={"thinking": {"type": "enabled"}},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url(image_path)},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    content = response.choices[0].message.content
    if not isinstance(content, str):
        raise ValueError(f"Expected a text response, got {type(content).__name__}.")
    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
    return content.strip(), {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _validate_official_coordinates(values: Sequence[int], field: str) -> Tuple[int, ...]:
    coordinates = tuple(int(value) for value in values)
    for coordinate in coordinates:
        if not OFFICIAL_COORD_MIN <= coordinate <= OFFICIAL_COORD_MAX:
            raise ValueError(
                f"{field} coordinate {coordinate} is outside the official [0, 999] range."
            )
    return coordinates


def parse_official_point_tag(text: str, *, full_response: bool = True) -> Tuple[int, int]:
    match = POINT_TAG_PATTERN.fullmatch(text) if full_response else POINT_TAG_PATTERN.search(text)
    if not match:
        raise ValueError("Expected official format: <point>x y</point>.")
    x, y = _validate_official_coordinates(tuple(map(int, match.groups())), "point")
    return x, y


def parse_official_bbox_tag(text: str, *, full_response: bool = True) -> Tuple[int, int, int, int]:
    match = BBOX_TAG_PATTERN.fullmatch(text) if full_response else BBOX_TAG_PATTERN.search(text)
    if not match:
        raise ValueError("Expected official format: <bbox>x1 y1 x2 y2</bbox>.")
    x1, y1, x2, y2 = _validate_official_coordinates(tuple(map(int, match.groups())), "bbox")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Official bbox must satisfy x1 < x2 and y1 < y2.")
    return x1, y1, x2, y2


def parse_json_array(text: str) -> List[Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        lines = lines[1:] if lines else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    value = json.loads(stripped)
    if not isinstance(value, list):
        raise ValueError("Expected a JSON array for official multi-object grounding output.")
    return value


def point_to_pixels(point: Sequence[int], width: int, height: int) -> Tuple[int, int]:
    """Use the exact official formula: int(coord * image_size / 1000)."""
    x, y = point
    return (
        min(width - 1, int(x * width / OFFICIAL_COORD_DIVISOR)),
        min(height - 1, int(y * height / OFFICIAL_COORD_DIVISOR)),
    )


def box_to_pixels(box: Sequence[int], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        min(width - 1, int(x1 * width / OFFICIAL_COORD_DIVISOR)),
        min(height - 1, int(y1 * height / OFFICIAL_COORD_DIVISOR)),
        min(width - 1, int(x2 * width / OFFICIAL_COORD_DIVISOR)),
        min(height - 1, int(y2 * height / OFFICIAL_COORD_DIVISOR)),
    )


def parse_csv_numbers(value: str, count: int, label: str) -> Tuple[float, ...]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != count:
        raise argparse.ArgumentTypeError(f"{label} requires {count} comma-separated numbers.")
    try:
        numbers = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} contains a non-numeric value.") from exc
    if not all(math.isfinite(number) for number in numbers):
        raise argparse.ArgumentTypeError(f"{label} values must be finite.")
    return numbers


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def result_directory(image_path: Path, test_name: str, output_dir: str | None) -> Path:
    if output_dir:
        path = Path(output_dir).expanduser().resolve()
    else:
        path = Path.cwd() / "ark_visual_grounding_results" / image_path.stem / test_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_overlay_svg(
    image_path: Path,
    width: int,
    height: int,
    shapes: Iterable[Dict[str, Any]],
    output_path: Path,
) -> None:
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    image_href = f"data:{mime_type};base64,{encoded}"
    stroke_width = max(2.0, min(width, height) * 0.004)
    radius = max(7.0, min(width, height) * 0.012)
    font_size = max(16.0, min(width, height) * 0.022)

    elements: List[str] = [
        f'<image href="{image_href}" x="0" y="0" width="{width}" height="{height}"/>',
    ]
    for shape in shapes:
        kind = shape["kind"]
        color = html.escape(str(shape.get("color", "#00e676")), quote=True)
        label = html.escape(str(shape.get("label", "")))
        if kind == "point":
            x, y = float(shape["x"]), float(shape["y"])
            elements.extend(
                [
                    f'<circle cx="{x}" cy="{y}" r="{radius}" fill="none" stroke="{color}" stroke-width="{stroke_width}"/>',
                    f'<line x1="{x-radius*1.5}" y1="{y}" x2="{x+radius*1.5}" y2="{y}" stroke="{color}" stroke-width="{stroke_width}"/>',
                    f'<line x1="{x}" y1="{y-radius*1.5}" x2="{x}" y2="{y+radius*1.5}" stroke="{color}" stroke-width="{stroke_width}"/>',
                ]
            )
            text_x = min(width - 4.0, x + radius * 1.8)
            text_y = max(font_size, min(height - 4.0, y - radius * 1.2))
        elif kind == "box":
            x1, y1, x2, y2 = (float(shape[key]) for key in ("x1", "y1", "x2", "y2"))
            elements.append(
                f'<rect x="{x1}" y="{y1}" width="{max(1.0, x2-x1)}" height="{max(1.0, y2-y1)}" '
                f'fill="none" stroke="{color}" stroke-width="{stroke_width}"/>'
            )
            text_x = max(4.0, x1)
            text_y = y1 - 7.0 if y1 > font_size + 8.0 else y1 + font_size + 7.0
        else:
            raise ValueError(f"Unknown overlay shape: {kind}")

        if label:
            elements.append(
                f'<text x="{text_x}" y="{text_y}" fill="{color}" stroke="#000" stroke-width="{max(2.0, stroke_width/2)}" '
                f'paint-order="stroke" font-family="Arial, sans-serif" font-size="{font_size}" font-weight="700">{label}</text>'
            )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        + "\n".join(elements)
        + "\n</svg>\n"
    )
    output_path.write_text(svg, encoding="utf-8")


def write_overlay_png(
    image_path: Path,
    width: int,
    height: int,
    shapes: Iterable[Dict[str, Any]],
    output_path: Path,
) -> bool:
    magick = shutil.which("magick")
    if not magick:
        return False

    stroke_width = max(2.0, min(width, height) * 0.004)
    radius = max(7.0, min(width, height) * 0.012)
    command: List[str] = [magick, str(image_path)]
    for shape in shapes:
        color = str(shape.get("color", "#00e676"))
        command.extend(["-stroke", color, "-fill", "none", "-strokewidth", str(stroke_width)])
        if shape["kind"] == "point":
            x, y = float(shape["x"]), float(shape["y"])
            command.extend(
                [
                    "-draw",
                    f"circle {x},{y} {x + radius},{y}",
                    "-draw",
                    f"line {x - radius * 1.5},{y} {x + radius * 1.5},{y}",
                    "-draw",
                    f"line {x},{y - radius * 1.5} {x},{y + radius * 1.5}",
                ]
            )
        elif shape["kind"] == "box":
            x1, y1, x2, y2 = (float(shape[key]) for key in ("x1", "y1", "x2", "y2"))
            command.extend(["-draw", f"rectangle {x1},{y1} {x2},{y2}"])
        else:
            raise ValueError(f"Unknown overlay shape: {shape['kind']}")
    command.append(str(output_path))
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    return completed.returncode == 0 and output_path.is_file()


def dry_run_payload(
    *,
    test_type: str,
    image_path: Path,
    width: int,
    height: int,
    model: str,
    prompt: str,
    reasoning_effort: str = DEFAULT_ARK_REASONING_EFFORT,
) -> Dict[str, Any]:
    return {
        "dry_run": True,
        "test_type": test_type,
        "model": model,
        "image": str(image_path),
        "image_size": {"width": width, "height": height},
        "api_sdk": "openai.OpenAI",
        "api_method": "chat.completions.create",
        "thinking": {"type": "enabled"},
        "reasoning_effort": reasoning_effort,
        "coordinate_system": {
            "origin": "top_left",
            "normalized_range": [OFFICIAL_COORD_MIN, OFFICIAL_COORD_MAX],
            "normalization_grid": [OFFICIAL_COORD_DIVISOR, OFFICIAL_COORD_DIVISOR],
            "x_direction": "right",
            "y_direction": "down",
        },
        "prompt": prompt,
    }
