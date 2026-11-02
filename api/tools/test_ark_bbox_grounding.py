#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict

from ark_visual_grounding_common import (
    OFFICIAL_COORD_DIVISOR,
    OFFICIAL_COORD_MAX,
    OFFICIAL_COORD_MIN,
    add_api_arguments,
    box_iou,
    box_to_pixels,
    call_openai_grounding,
    dry_run_payload,
    image_dimensions,
    parse_csv_numbers,
    parse_official_bbox_tag,
    require_local_image,
    result_directory,
    write_json,
    write_overlay_png,
    write_overlay_svg,
    write_text,
)


def build_prompt(target: str) -> str:
    return (
        f"框出图像中的“{target}”，输出 bounding box 的坐标。"
        "只输出 <bbox>x_min y_min x_max y_max</bbox>，不要输出解释或其他文字。"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test 2D bounding-box grounding with the official Ark <bbox> protocol."
    )
    parser.add_argument("image", help="Local image path.")
    parser.add_argument("--target", required=True, help="Object to bound.")
    parser.add_argument(
        "--gt-box",
        type=lambda value: parse_csv_numbers(value, 4, "--gt-box"),
        help="Optional ground-truth pixel box as X1,Y1,X2,Y2.",
    )
    parser.add_argument(
        "--min-iou",
        type=float,
        default=0.5,
        help="Pass threshold when --gt-box is provided (default: 0.5).",
    )
    parser.add_argument("--require-pass", action="store_true")
    add_api_arguments(parser)
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    image_path = require_local_image(args.image)
    width, height = image_dimensions(image_path)
    prompt = build_prompt(args.target)
    output_dir = result_directory(image_path, "bbox_grounding_official", args.output_dir)
    write_text(output_dir / "request_prompt.txt", prompt)

    if args.dry_run:
        payload = dry_run_payload(
            test_type="bbox_grounding_official",
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
                "official_response_format": "<bbox>x_min y_min x_max y_max</bbox>",
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
    box_normalized = parse_official_bbox_tag(raw_text)
    box_pixel = box_to_pixels(box_normalized, width, height)

    evaluation = None
    if args.gt_box is not None:
        gt_x1, gt_y1, gt_x2, gt_y2 = args.gt_box
        if not (0 <= gt_x1 < gt_x2 < width and 0 <= gt_y1 < gt_y2 < height):
            raise ValueError(f"--gt-box must satisfy 0 <= x1 < x2 < {width} and 0 <= y1 < y2 < {height}.")
        iou = box_iou(box_pixel, args.gt_box)
        evaluation = {
            "ground_truth_pixel": {
                "x_min": gt_x1,
                "y_min": gt_y1,
                "x_max": gt_x2,
                "y_max": gt_y2,
            },
            "iou": iou,
            "min_iou": args.min_iou,
            "passed": iou >= args.min_iou,
        }

    result: Dict[str, Any] = {
        "test_type": "bbox_grounding_official",
        "model": args.model,
        "api_sdk": "openai.OpenAI",
        "api_method": "chat.completions.create",
        "reasoning_effort": args.reasoning_effort,
        "model_response_format": "<bbox>x_min y_min x_max y_max</bbox>",
        "image": str(image_path),
        "image_size": {"width": width, "height": height},
        "target": args.target,
        "coordinate_system": {
            "origin": "top_left",
            "normalized_range": [OFFICIAL_COORD_MIN, OFFICIAL_COORD_MAX],
            "normalization_grid": [OFFICIAL_COORD_DIVISOR, OFFICIAL_COORD_DIVISOR],
        },
        "bbox_normalized_0_999": {
            "x_min": box_normalized[0],
            "y_min": box_normalized[1],
            "x_max": box_normalized[2],
            "y_max": box_normalized[3],
        },
        "bbox_pixel": {
            "x_min": box_pixel[0],
            "y_min": box_pixel[1],
            "x_max": box_pixel[2],
            "y_max": box_pixel[3],
        },
        "pixel_conversion": "int(coord * image_size / 1000)",
        "evaluation": evaluation,
        "usage": usage,
    }
    write_json(output_dir / "result.json", result)

    shapes = [
        {
            "kind": "box",
            "x1": box_pixel[0],
            "y1": box_pixel[1],
            "x2": box_pixel[2],
            "y2": box_pixel[3],
            "color": "#00e676",
            "label": "prediction",
        }
    ]
    if args.gt_box is not None:
        shapes.append(
            {
                "kind": "box",
                "x1": args.gt_box[0],
                "y1": args.gt_box[1],
                "x2": args.gt_box[2],
                "y2": args.gt_box[3],
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
