#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any, Dict

from ark_visual_grounding_common import (
    OFFICIAL_COORD_DIVISOR,
    OFFICIAL_COORD_MAX,
    OFFICIAL_COORD_MIN,
    add_api_arguments,
    call_openai_grounding,
    dry_run_payload,
    image_dimensions,
    parse_csv_numbers,
    parse_official_point_tag,
    point_to_pixels,
    require_local_image,
    result_directory,
    write_json,
    write_overlay_png,
    write_overlay_svg,
    write_text,
)


def build_prompt(target: str) -> str:
    return (
        f"定位图像中的“{target}”，输出该目标点的坐标。"
        "只输出 <point>x y</point>，不要输出解释或其他文字。"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test point grounding with the official Ark <point>x y</point> protocol."
    )
    parser.add_argument("image", help="Local image path.")
    parser.add_argument("--target", required=True, help="Object or landmark to locate.")
    parser.add_argument(
        "--gt-point",
        type=lambda value: parse_csv_numbers(value, 2, "--gt-point"),
        help="Optional ground-truth pixel point as X,Y.",
    )
    parser.add_argument(
        "--max-error-px",
        type=float,
        default=25.0,
        help="Pass threshold when --gt-point is provided (default: 25 pixels).",
    )
    parser.add_argument("--require-pass", action="store_true")
    add_api_arguments(parser)
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    image_path = require_local_image(args.image)
    width, height = image_dimensions(image_path)
    prompt = build_prompt(args.target)
    output_dir = result_directory(image_path, "point_grounding_official", args.output_dir)
    write_text(output_dir / "request_prompt.txt", prompt)

    if args.dry_run:
        payload = dry_run_payload(
            test_type="point_grounding_official",
            image_path=image_path,
            width=width,
            height=height,
            model=args.model,
            prompt=prompt,
            reasoning_effort=args.reasoning_effort,
        )
        payload.update(
            {
                "target": args.target,
                "official_response_format": "<point>x y</point>",
            }
        )
        write_json(output_dir / "dry_run.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    raw_text, usage = call_openai_grounding(
        image_path=image_path,
        prompt=prompt,
        api_key=args.api_key,
        api_base=args.api_base,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
    )
    write_text(output_dir / "raw_response.txt", raw_text)
    point_normalized = parse_official_point_tag(raw_text)
    point_pixel = point_to_pixels(point_normalized, width, height)

    evaluation = None
    if args.gt_point is not None:
        gt_x, gt_y = args.gt_point
        if not (0 <= gt_x < width and 0 <= gt_y < height):
            raise ValueError(f"--gt-point must be inside the {width}x{height} image.")
        error = math.hypot(point_pixel[0] - gt_x, point_pixel[1] - gt_y)
        evaluation = {
            "ground_truth_pixel": {"x": gt_x, "y": gt_y},
            "euclidean_error_pixels": error,
            "error_fraction_of_image_diagonal": error / math.hypot(width, height),
            "max_error_pixels": args.max_error_px,
            "passed": error <= args.max_error_px,
        }

    result: Dict[str, Any] = {
        "test_type": "point_grounding_official",
        "model": args.model,
        "api_sdk": "openai.OpenAI",
        "api_method": "chat.completions.create",
        "reasoning_effort": args.reasoning_effort,
        "model_response_format": "<point>x y</point>",
        "image": str(image_path),
        "image_size": {"width": width, "height": height},
        "target": args.target,
        "coordinate_system": {
            "origin": "top_left",
            "normalized_range": [OFFICIAL_COORD_MIN, OFFICIAL_COORD_MAX],
            "normalization_grid": [OFFICIAL_COORD_DIVISOR, OFFICIAL_COORD_DIVISOR],
        },
        "point_normalized_0_999": {"x": point_normalized[0], "y": point_normalized[1]},
        "point_pixel": {"x": point_pixel[0], "y": point_pixel[1]},
        "pixel_conversion": "int(coord * image_size / 1000)",
        "evaluation": evaluation,
        "usage": usage,
    }
    write_json(output_dir / "result.json", result)

    shapes = [
        {
            "kind": "point",
            "x": point_pixel[0],
            "y": point_pixel[1],
            "color": "#00e676",
            "label": "prediction",
        }
    ]
    if args.gt_point is not None:
        shapes.append(
            {
                "kind": "point",
                "x": args.gt_point[0],
                "y": args.gt_point[1],
                "color": "#ff2d95",
                "label": "ground truth",
            }
        )
    write_overlay_svg(image_path, width, height, shapes, output_dir / "overlay.svg")
    write_overlay_png(image_path, width, height, shapes, output_dir / "overlay.png")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.require_pass and evaluation is not None and not evaluation["passed"]:
        return 1
    return 0


def main() -> int:
    try:
        return run()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
