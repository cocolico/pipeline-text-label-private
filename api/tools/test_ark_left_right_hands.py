#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

from ark_visual_grounding_common import (
    OFFICIAL_COORD_DIVISOR,
    OFFICIAL_COORD_MAX,
    OFFICIAL_COORD_MIN,
    add_api_arguments,
    box_to_pixels,
    call_openai_grounding,
    dry_run_payload,
    image_dimensions,
    parse_json_array,
    parse_official_bbox_tag,
    parse_official_point_tag,
    point_to_pixels,
    require_local_image,
    result_directory,
    write_json,
    write_overlay_png,
    write_overlay_svg,
    write_text,
)


CATEGORY_TO_SIDE = {
    "left_hand": "left",
    "right_hand": "right",
    "uncertain_hand": "uncertain",
}


def build_prompt(scope: str) -> str:
    return f"""识别图像中{scope}。

判断的是人物自身的解剖学左手或右手，不能把画面左侧直接当作左手，也不能把画面右侧直接当作右手。请结合手部结构、拇指方向、手臂连续性、抓握方向和第一人称视角判断。证据不足时使用 uncertain_hand，不要猜测。

这是多目标 Grounding 任务。对每只清晰可见的手：
- category 只能是 left_hand、right_hand 或 uncertain_hand。
- bbox 使用官方格式 <bbox>x1 y1 x2 y2</bbox>，紧密框住可见手部或手套，不包含所持工具。
- point 使用官方格式 <point>x y</point>，落在对应手部中心附近且必须位于 bbox 内。
- 坐标为官方 1000×1000 归一化坐标，范围 [0,999]，左上角为原点。
- 按 point 的 x 坐标从小到大排列。

只输出以下 JSON 数组，不要输出 Markdown、解释或其他字段：
[
  {{"category":"left_hand","bbox":"<bbox>x1 y1 x2 y2</bbox>","point":"<point>x y</point>"}},
  {{"category":"right_hand","bbox":"<bbox>x1 y1 x2 y2</bbox>","point":"<point>x y</point>"}}
]"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test handedness using official Ark multi-object Grounding tags."
    )
    parser.add_argument("image", help="Local image path.")
    parser.add_argument(
        "--scope",
        default="主要人物或第一人称相机佩戴者的所有清晰可见手部",
        help="Which person's hands should be evaluated.",
    )
    parser.add_argument(
        "--expected-order",
        help="Optional anatomical labels sorted image-left to image-right, e.g. left,right.",
    )
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Exit nonzero when expected handedness or point/bbox consistency fails.",
    )
    add_api_arguments(parser)
    return parser.parse_args()


def parse_hands(raw_text: str, width: int, height: int) -> List[Dict[str, Any]]:
    items = parse_json_array(raw_text)
    hands: List[Dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"hands[{index - 1}] must be an object.")
        category = str(item.get("category", "")).strip().lower()
        if category not in CATEGORY_TO_SIDE:
            raise ValueError(f"Invalid category for hand {index}: {category!r}")
        bbox_tag = item.get("bbox")
        point_tag = item.get("point")
        if not isinstance(bbox_tag, str) or not isinstance(point_tag, str):
            raise ValueError(f"Hand {index} must contain string bbox and point tags.")
        box = parse_official_bbox_tag(bbox_tag)
        point = parse_official_point_tag(point_tag)
        point_inside_bbox = box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]
        box_px = box_to_pixels(box, width, height)
        point_px = point_to_pixels(point, width, height)
        if point[0] < 333:
            image_region = "left"
        elif point[0] < 667:
            image_region = "center"
        else:
            image_region = "right"
        hands.append(
            {
                "hand_id": f"hand_{index}",
                "anatomical_side": CATEGORY_TO_SIDE[category],
                "official_category": category,
                "image_horizontal_region": image_region,
                "bbox_tag": bbox_tag,
                "point_tag": point_tag,
                "bbox_normalized_0_999": {
                    "x_min": box[0],
                    "y_min": box[1],
                    "x_max": box[2],
                    "y_max": box[3],
                },
                "point_normalized_0_999": {"x": point[0], "y": point[1]},
                "bbox_pixel": {
                    "x_min": box_px[0],
                    "y_min": box_px[1],
                    "x_max": box_px[2],
                    "y_max": box_px[3],
                },
                "point_pixel": {"x": point_px[0], "y": point_px[1]},
                "point_inside_bbox": point_inside_bbox,
            }
        )
    hands.sort(key=lambda hand: hand["point_normalized_0_999"]["x"])
    for index, hand in enumerate(hands, start=1):
        hand["hand_id"] = f"hand_{index}"
    return hands


def run() -> int:
    args = parse_args()
    image_path = require_local_image(args.image)
    width, height = image_dimensions(image_path)
    prompt = build_prompt(args.scope)
    output_dir = result_directory(image_path, "left_right_hands_official", args.output_dir)
    write_text(output_dir / "request_prompt.txt", prompt)

    if args.dry_run:
        payload = dry_run_payload(
            test_type="left_right_hands_official",
            image_path=image_path,
            width=width,
            height=height,
            model=args.model,
            prompt=prompt,
            reasoning_effort=args.reasoning_effort,
        )
        payload["official_response_format"] = (
            'multi-object JSON array with "bbox":"<bbox>...</bbox>" and '
            '"point":"<point>...</point>"'
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
    hands = parse_hands(raw_text, width, height)

    expected = None
    predicted = [hand["anatomical_side"] for hand in hands]
    handedness_passed = None
    if args.expected_order:
        expected = [part.strip().lower() for part in args.expected_order.split(",") if part.strip()]
        invalid = [side for side in expected if side not in {"left", "right", "uncertain"}]
        if invalid:
            raise ValueError(f"Invalid --expected-order labels: {invalid}")
        handedness_passed = predicted == expected
    geometry_passed = all(hand["point_inside_bbox"] for hand in hands)
    evaluation = {
        "expected_order": expected,
        "predicted_order": predicted,
        "handedness_passed": handedness_passed,
        "all_points_inside_bboxes": geometry_passed,
        "passed": geometry_passed and (handedness_passed is not False),
    }

    result = {
        "test_type": "left_right_hands_official",
        "model": args.model,
        "api_sdk": "openai.OpenAI",
        "api_method": "chat.completions.create",
        "reasoning_effort": args.reasoning_effort,
        "model_response_format": "official multi-object JSON containing <bbox> and <point> tags",
        "image": str(image_path),
        "image_size": {"width": width, "height": height},
        "coordinate_system": {
            "origin": "top_left",
            "normalized_range": [OFFICIAL_COORD_MIN, OFFICIAL_COORD_MAX],
            "normalization_grid": [OFFICIAL_COORD_DIVISOR, OFFICIAL_COORD_DIVISOR],
        },
        "hands": hands,
        "evaluation": evaluation,
        "pixel_conversion": "int(coord * image_size / 1000)",
        "usage": usage,
    }
    write_json(output_dir / "result.json", result)

    colors = {"left": "#00e676", "right": "#00b0ff", "uncertain": "#ffd600"}
    shapes: List[Dict[str, Any]] = []
    for hand in hands:
        box = hand["bbox_pixel"]
        point = hand["point_pixel"]
        side = hand["anatomical_side"]
        color = colors[side]
        shapes.append(
            {
                "kind": "box",
                "x1": box["x_min"],
                "y1": box["y_min"],
                "x2": box["x_max"],
                "y2": box["y_max"],
                "color": color,
                "label": f"{hand['hand_id']}: {side}",
            }
        )
        shapes.append(
            {
                "kind": "point",
                "x": point["x"],
                "y": point["y"],
                "color": color,
                "label": "",
            }
        )
    write_overlay_svg(image_path, width, height, shapes, output_dir / "overlay.svg")
    write_overlay_png(image_path, width, height, shapes, output_dir / "overlay.png")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.require_pass and not evaluation["passed"]:
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
