#!/usr/bin/env python3
"""Text-only hand-step discovery, fusion, and ROI-level annotation.

This module deliberately never materializes video clips or frame images.  It
uses the full source video for the independent Stage-1 VLM proposal, fuses
those proposals with upstream ``processed_hand_screening`` frame/bbox data in
Stage 2, and sends representative in-memory ROI crops to the VLM in Stage 3.
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import tempfile
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - production dependency
    cv2 = None

from text_annotation_goal import NO_SUBTASK_GOAL, compose_video_goal

from common import (
    build_retry_prefix,
    build_source_video_meta,
    build_video_api_content,
    call_chat_completion,
    cut_video_segment_ffmpeg,
    ensure_video_out_dir_safe,
    extract_json_from_response,
    format_duration,
    get_video_metadata,
    initialize_api_client,
    logger,
    now_utc_iso,
    read_json,
    update_run_summary,
    video_id_from_path,
    write_json,
    write_text,
)


HAND_EVENTS_SCHEMA = "vlm_hand_frames.v2"
HAND_BOUNDARIES_SCHEMA = "hand_step_boundaries.v1"
HAND_STEPS_SCHEMA = "hand_steps.v1"
UPSTREAM_SCHEMA = "hand_frame_screening.v2"
MIN_VALID_KEEP_FRAMES = 30
MERGE_GAP_FRAMES_EXCLUSIVE = 90
MAX_STEP_VIDEO_SAMPLES = 120
MIN_STEP_VIDEO_REQUEST_FPS = 0.5
MAX_PROMPT_BBOX_SAMPLES_PER_TRACK = 12
BBOX_SHRINK_WINDOW_FRAMES = int(os.environ.get("TEXT_LABEL_BBOX_SHRINK_WINDOW_FRAMES", "30"))
BBOX_SHRINK_WIDTH_RATIO = float(os.environ.get("TEXT_LABEL_BBOX_SHRINK_WIDTH_RATIO", "0.68"))
BBOX_SHRINK_HEIGHT_RATIO = float(os.environ.get("TEXT_LABEL_BBOX_SHRINK_HEIGHT_RATIO", "0.68"))
BBOX_SHRINK_AREA_RATIO = float(os.environ.get("TEXT_LABEL_BBOX_SHRINK_AREA_RATIO", "0.46"))
ANALYSIS_BBOX_PADDING_RATIO_X = float(os.environ.get("TEXT_LABEL_BBOX_PADDING_RATIO_X", "0.10"))
ANALYSIS_BBOX_PADDING_RATIO_Y = float(os.environ.get("TEXT_LABEL_BBOX_PADDING_RATIO_Y", "0.10"))
ANALYSIS_BBOX_MIN_PADDING_PX = int(os.environ.get("TEXT_LABEL_BBOX_MIN_PADDING_PX", "4"))


def _step_action_slug(description: Any, *, max_length: int = 80) -> str:
    text = re.sub(r"\s+", "_", str(description or "").strip())
    text = re.sub(r"[^\w\-]+", "_", text, flags=re.UNICODE).strip("_.-")
    text = re.sub(r"_+", "_", text)
    return (text[:max_length].rstrip("_.-") or "action")


def _step_video_request_fps(
    step: Dict[str, Any], *, source_fps: float, configured_fps: float
) -> float:
    if source_fps <= 0 or configured_fps <= 0:
        raise ValueError("source_fps and configured_fps must be positive")
    frame_count = int(step["end_frame_id"]) - int(step["start_frame_id"]) + 1
    duration_sec = max(1.0 / source_fps, float(frame_count) / source_fps)
    capped_fps = float(MAX_STEP_VIDEO_SAMPLES) / duration_sec
    return min(float(configured_fps), max(MIN_STEP_VIDEO_REQUEST_FPS, capped_fps))


def _compact_stage3_bbox_tracks_for_prompt(step: Dict[str, Any]) -> Dict[str, Any]:
    tracks = step.get("stage3_bbox_tracks", step.get("bbox_tracks", {}))
    tracks = tracks if isinstance(tracks, dict) else {}
    compact: Dict[str, List[Dict[str, Any]]] = {"left_hand": [], "right_hand": []}
    for hand in ("left_hand", "right_hand"):
        by_source: Dict[str, List[Dict[str, Any]]] = {}
        for record in tracks.get(hand, []) if isinstance(tracks.get(hand), list) else []:
            if not isinstance(record, dict) or "frame_id" not in record:
                continue
            source = str(record.get("bbox_source") or "upstream_sam3")
            by_source.setdefault(source, []).append(record)
        for source, records in sorted(by_source.items()):
            records.sort(key=lambda item: int(item["frame_id"]))
            count = min(MAX_PROMPT_BBOX_SAMPLES_PER_TRACK, len(records))
            if count == len(records):
                sample_indices = list(range(len(records)))
            elif count <= 1:
                sample_indices = [0]
            else:
                sample_indices = sorted(set(
                    int(round(index * (len(records) - 1) / (count - 1)))
                    for index in range(count)
                ))
            compact[hand].append({
                "bbox_source": source,
                "available_frame_intervals": [
                    {"start_frame_id": start, "end_frame_id": end}
                    for start, end in _ranges_from_frame_ids(
                        (int(item["frame_id"]) for item in records), max_gap=0
                    )
                ],
                "representative_bboxes": [
                    {
                        "frame_id": int(records[index]["frame_id"]),
                        "bbox_xyxy": list(records[index].get("bbox_xyxy", [])),
                    }
                    for index in sample_indices
                ],
            })
    return compact


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = int(round(min(1.0, max(0.0, float(fraction))) * (len(ordered) - 1)))
    return ordered[index]


def _bbox_size(bbox: Sequence[int]) -> Tuple[int, int, int]:
    width = max(0, int(bbox[2]) - int(bbox[0]))
    height = max(0, int(bbox[3]) - int(bbox[1]))
    return width, height, width * height


def _expand_analysis_bbox(
    bbox: Sequence[int], *, frame_width: int, frame_height: int,
    ratio_x: float = ANALYSIS_BBOX_PADDING_RATIO_X,
    ratio_y: float = ANALYSIS_BBOX_PADDING_RATIO_Y,
    min_padding_px: int = ANALYSIS_BBOX_MIN_PADDING_PX,
) -> List[int]:
    """Add deterministic glove-margin context without changing source geometry."""
    x1, y1, x2, y2 = [int(value) for value in bbox]
    bbox_width, bbox_height, _ = _bbox_size((x1, y1, x2, y2))
    pad_x = max(int(min_padding_px), int(round(float(bbox_width) * float(ratio_x))))
    pad_y = max(int(min_padding_px), int(round(float(bbox_height) * float(ratio_y))))
    return [
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(int(frame_width), x2 + pad_x),
        min(int(frame_height), y2 + pad_y),
    ]


def _repair_temporarily_shrunken_bboxes(
    tracks: Dict[str, List[Dict[str, Any]]], *, width: int, height: int,
    window_frames: int = BBOX_SHRINK_WINDOW_FRAMES,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Keep upstream boxes immutable and derive interpolated analysis boxes.

    A box is repairable only when width, height, and area all shrink sharply
    relative to the local 75th-percentile baseline and larger reliable boxes
    exist on both temporal sides within ``window_frames``.  This conservative
    two-sided rule targets occasional SAM3 under-coverage and avoids treating a
    sustained natural scale change as a wrist-only detector failure.
    """
    analysis: Dict[str, List[Dict[str, Any]]] = {"left_hand": [], "right_hand": []}
    annotations: List[Dict[str, Any]] = []
    for hand in ("left_hand", "right_hand"):
        records = sorted(
            [dict(item) for item in tracks.get(hand, []) if isinstance(item, dict)],
            key=lambda item: int(item["frame_id"]),
        )
        by_frame = {int(item["frame_id"]): item for item in records}
        candidate_ids: set[int] = set()
        metrics: Dict[int, Dict[str, float]] = {}
        for record in records:
            frame_id = int(record["frame_id"])
            bbox = list(record.get("bbox_xyxy", []))
            if len(bbox) != 4:
                continue
            current_w, current_h, current_area = _bbox_size(bbox)
            neighbors = [
                other for other in records
                if 0 < abs(int(other["frame_id"]) - frame_id) <= int(window_frames)
                and len(other.get("bbox_xyxy", [])) == 4
            ]
            before = [other for other in neighbors if int(other["frame_id"]) < frame_id]
            after = [other for other in neighbors if int(other["frame_id"]) > frame_id]
            if not before or not after:
                continue
            widths, heights, areas = zip(*(_bbox_size(other["bbox_xyxy"]) for other in neighbors))
            base_w = _percentile(widths, 0.75)
            base_h = _percentile(heights, 0.75)
            base_area = _percentile(areas, 0.75)
            if min(base_w, base_h, base_area) <= 0:
                continue
            width_ratio = float(current_w) / base_w
            height_ratio = float(current_h) / base_h
            area_ratio = float(current_area) / base_area
            if (
                width_ratio <= BBOX_SHRINK_WIDTH_RATIO
                and height_ratio <= BBOX_SHRINK_HEIGHT_RATIO
                and area_ratio <= BBOX_SHRINK_AREA_RATIO
            ):
                candidate_ids.add(frame_id)
                metrics[frame_id] = {
                    "width_ratio_to_local_baseline": round(width_ratio, 6),
                    "height_ratio_to_local_baseline": round(height_ratio, 6),
                    "area_ratio_to_local_baseline": round(area_ratio, 6),
                }

        for frame_id in sorted(candidate_ids):
            previous_ids = [
                candidate for candidate in by_frame
                if frame_id - int(window_frames) <= candidate < frame_id and candidate not in candidate_ids
            ]
            next_ids = [
                candidate for candidate in by_frame
                if frame_id < candidate <= frame_id + int(window_frames) and candidate not in candidate_ids
            ]
            if not previous_ids or not next_ids:
                continue
            previous_id, next_id = max(previous_ids), min(next_ids)
            previous_bbox = list(by_frame[previous_id]["bbox_xyxy"])
            next_bbox = list(by_frame[next_id]["bbox_xyxy"])
            current_bbox = list(by_frame[frame_id]["bbox_xyxy"])
            current_w, current_h, current_area = _bbox_size(current_bbox)
            previous_w, previous_h, previous_area = _bbox_size(previous_bbox)
            next_w, next_h, next_area = _bbox_size(next_bbox)
            if (
                previous_w <= 0 or previous_h <= 0 or previous_area <= 0
                or next_w <= 0 or next_h <= 0 or next_area <= 0
                or float(current_w) / previous_w > BBOX_SHRINK_WIDTH_RATIO
                or float(current_h) / previous_h > BBOX_SHRINK_HEIGHT_RATIO
                or float(current_area) / previous_area > BBOX_SHRINK_AREA_RATIO
                or float(current_w) / next_w > BBOX_SHRINK_WIDTH_RATIO
                or float(current_h) / next_h > BBOX_SHRINK_HEIGHT_RATIO
                or float(current_area) / next_area > BBOX_SHRINK_AREA_RATIO
            ):
                continue
            span = max(1, next_id - previous_id)
            weight = float(frame_id - previous_id) / float(span)
            repaired = [
                int(round(float(a) + (float(b) - float(a)) * weight))
                for a, b in zip(previous_bbox, next_bbox)
            ]
            repaired[0] = min(max(0, repaired[0]), width)
            repaired[2] = min(max(0, repaired[2]), width)
            repaired[1] = min(max(0, repaired[1]), height)
            repaired[3] = min(max(0, repaired[3]), height)
            if repaired[2] <= repaired[0] or repaired[3] <= repaired[1]:
                continue
            original = current_bbox
            by_frame[frame_id]["analysis_bbox_xyxy"] = repaired
            by_frame[frame_id]["bbox_repair_applied"] = True
            by_frame[frame_id]["bbox_quality_label"] = "possible_wrist_only_undercoverage"
            by_frame[frame_id]["repair_method"] = "two_sided_linear_interpolation"
            annotation = {
                "frame_id": frame_id,
                "hand": hand,
                "bbox_source": str(by_frame[frame_id].get("bbox_source") or "upstream_sam3"),
                "source_bbox_xyxy": original,
                "analysis_bbox_xyxy": repaired,
                "possible_wrist_only": True,
                "source_bbox_may_not_contain_complete_hand": True,
                "repair_applied": True,
                "repair_method": "two_sided_linear_interpolation",
                "previous_reliable_frame_id": previous_id,
                "next_reliable_frame_id": next_id,
                **metrics[frame_id],
            }
            annotations.append(annotation)

        for record in records:
            pre_expansion = list(record.get("analysis_bbox_xyxy", record.get("bbox_xyxy", [])))
            record["pre_expansion_analysis_bbox_xyxy"] = pre_expansion
            record["analysis_bbox_xyxy"] = _expand_analysis_bbox(
                pre_expansion, frame_width=width, frame_height=height
            )
            record["bbox_expansion_applied"] = record["analysis_bbox_xyxy"] != pre_expansion
            record["bbox_expansion_method"] = "symmetric_relative_glove_padding"
            record["bbox_expansion_ratio_x"] = ANALYSIS_BBOX_PADDING_RATIO_X
            record["bbox_expansion_ratio_y"] = ANALYSIS_BBOX_PADDING_RATIO_Y
            record["bbox_min_padding_px"] = ANALYSIS_BBOX_MIN_PADDING_PX
            record.setdefault("bbox_repair_applied", False)
            record.setdefault("bbox_quality_label", "expanded_for_glove_margin")
            analysis[hand].append(record)
        annotation_by_frame = {
            (str(item["hand"]), int(item["frame_id"])): item for item in annotations
        }
        for record in analysis[hand]:
            annotation = annotation_by_frame.get((hand, int(record["frame_id"])))
            if annotation is not None:
                annotation["temporally_repaired_bbox_xyxy"] = list(record["pre_expansion_analysis_bbox_xyxy"])
                annotation["analysis_bbox_xyxy"] = list(record["analysis_bbox_xyxy"])
                annotation["glove_margin_expansion_applied"] = True
    annotations.sort(key=lambda item: (int(item["frame_id"]), str(item["hand"])))
    return analysis, annotations


def _compact_analysis_bbox_tracks_for_prompt(step: Dict[str, Any]) -> Dict[str, Any]:
    tracks = step.get("analysis_bbox_tracks", step.get("bbox_tracks", {}))
    compact: Dict[str, List[Dict[str, Any]]] = {"left_hand": [], "right_hand": []}
    for hand in ("left_hand", "right_hand"):
        records = [item for item in tracks.get(hand, []) if isinstance(item, dict)]
        if not records:
            continue
        records.sort(key=lambda item: int(item["frame_id"]))
        count = min(MAX_PROMPT_BBOX_SAMPLES_PER_TRACK, len(records))
        indices = sorted(set(
            int(round(index * (len(records) - 1) / max(1, count - 1))) for index in range(count)
        ))
        compact[hand] = [{
            "frame_id": int(records[index]["frame_id"]),
            "source_bbox_xyxy": list(records[index].get("bbox_xyxy", [])),
            "analysis_bbox_xyxy": list(records[index].get("analysis_bbox_xyxy", records[index].get("bbox_xyxy", []))),
            "bbox_repair_applied": bool(records[index].get("bbox_repair_applied")),
            "bbox_expansion_applied": bool(records[index].get("bbox_expansion_applied")),
            "bbox_expansion_method": str(records[index].get("bbox_expansion_method") or ""),
            "bbox_quality_label": str(records[index].get("bbox_quality_label") or ""),
        } for index in indices]
    return compact


def _step_clip_path(
    video_out: str,
    step_id: int,
    start_frame: int,
    end_frame: int,
    action_description: Any = None,
) -> str:
    action_suffix = f"_{_step_action_slug(action_description)}" if action_description else ""
    return os.path.join(
        video_out,
        "stage1",
        "step_clips",
        f"step_{int(step_id):03d}_source_frames_{int(start_frame)}_{int(end_frame)}{action_suffix}.mp4",
    )


def _step_visualization_path(video_out: str, step: Dict[str, Any]) -> str:
    return os.path.join(
        video_out,
        "stage1",
        "step_clips",
        f"step_{int(step['hand_step_id']):03d}_source_frames_"
        f"{int(step['start_frame_id'])}_{int(step['end_frame_id'])}_visualization.mp4",
    )


def _ensure_step_visualization(
    *, clip_path: str, video_out: str, step: Dict[str, Any], fps: float,
) -> str:
    """Render original and inference-only repaired boxes on the Step video."""
    if cv2 is None:
        raise RuntimeError("OpenCV is required to render step_clip_visualization.mp4")
    output_path = _step_visualization_path(video_out, step)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    capture = cv2.VideoCapture(clip_path)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open Step clip for visualization: {clip_path}")
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_start = int(step["start_frame_id"])
    source_end = int(step["end_frame_id"])
    original_lookup: Dict[Tuple[str, int], List[int]] = {}
    analysis_lookup: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for hand in ("left_hand", "right_hand"):
        for item in step.get("bbox_tracks", {}).get(hand, []):
            original_lookup[(hand, int(item["frame_id"]))] = list(item["bbox_xyxy"])
        for item in step.get("analysis_bbox_tracks", {}).get(hand, []):
            analysis_lookup[(hand, int(item["frame_id"]))] = item
    raw_temp = tempfile.NamedTemporaryFile(
        prefix=".tmp_bbox_vis_", suffix=".mp4", dir=os.path.dirname(output_path), delete=False
    )
    raw_temp_path = raw_temp.name
    raw_temp.close()
    final_temp = output_path + ".tmp.mp4"
    writer = cv2.VideoWriter(
        raw_temp_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (frame_width, frame_height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot create Step visualization: {raw_temp_path}")
    local_frame_id = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            source_frame_id = source_start + local_frame_id
            for hand, color in (("left_hand", (255, 180, 0)), ("right_hand", (0, 220, 80))):
                original = original_lookup.get((hand, source_frame_id))
                if original:
                    x1, y1, x2, y2 = [int(value) for value in original]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{hand} source", (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
                analysis_item = analysis_lookup.get((hand, source_frame_id), {})
                analysis_bbox = analysis_item.get("analysis_bbox_xyxy", [])
                if len(analysis_bbox) == 4 and list(analysis_bbox) != list(original or []):
                    repaired = bool(analysis_item.get("bbox_repair_applied"))
                    box_color = (0, 80, 255) if repaired else (0, 170, 255)
                    x1, y1, x2, y2 = [int(value) for value in analysis_bbox]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3 if repaired else 2)
                    label = "REPAIRED+EXPANDED" if repaired else "EXPANDED"
                    cv2.putText(frame, f"{hand} {label} inference bbox", (x1, min(frame_height - 8, y2 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, box_color, 1, cv2.LINE_AA)
            cv2.putText(frame, f"source_frame_id={source_frame_id}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "source=blue/green; expanded=orange; repaired+expanded=red", (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(frame)
            local_frame_id += 1
    finally:
        capture.release()
        writer.release()
    expected = source_end - source_start + 1
    if local_frame_id != expected:
        if os.path.exists(raw_temp_path):
            os.remove(raw_temp_path)
        raise RuntimeError(f"Visualization frame count mismatch: expected={expected}, got={local_frame_id}")
    command = [
        os.environ.get("FFMPEG_BIN", "ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
        "-i", raw_temp_path, "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", final_temp,
    ]
    try:
        subprocess.run(command, check=True, stdin=subprocess.DEVNULL)
        rendered_meta = get_video_metadata(final_temp)
        if int(rendered_meta["total_frames"]) != expected:
            raise RuntimeError(
                f"Visualization encode expected {expected} frames, got {rendered_meta['total_frames']}"
            )
        os.replace(final_temp, output_path)
    finally:
        for temporary in (raw_temp_path, final_temp):
            if os.path.exists(temporary):
                os.remove(temporary)
    return output_path


def _ensure_step_clip(
    *, video_path: str, video_out: str, step: Dict[str, Any], fps: float
) -> str:
    """Materialize one exact, reusable clip while keeping source frame IDs authoritative."""
    step_id = int(step["hand_step_id"])
    start_frame = int(step["start_frame_id"])
    end_frame = int(step["end_frame_id"])
    if fps <= 0:
        raise ValueError(f"Invalid source FPS for step clip: {fps}")
    clip_path = _step_clip_path(video_out, step_id, start_frame, end_frame)
    if os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
        return clip_path
    _cut_exact_source_frames(
        video_path=video_path,
        output_path=clip_path,
        start_frame_id=start_frame,
        end_frame_id=end_frame,
        fps=fps,
    )
    return clip_path


def _cut_exact_source_frames(
    *, video_path: str, output_path: str, start_frame_id: int,
    end_frame_id: int, fps: float,
) -> None:
    """Decode the source and select the inclusive source-frame interval exactly.

    This intentionally does not translate frame IDs into seek timestamps.  The
    select filter operates on ffmpeg's decoded, zero-based frame number ``n``.
    """
    start = int(start_frame_id)
    end = int(end_frame_id)
    if start < 0 or end < start:
        raise ValueError(f"Invalid inclusive source-frame interval: [{start}, {end}]")
    if float(fps) <= 0:
        raise ValueError(f"Invalid source FPS for exact frame cut: {fps}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    suffix = os.path.splitext(output_path)[1] or ".mp4"
    handle = tempfile.NamedTemporaryFile(
        prefix=".tmp_cut_", suffix=suffix,
        dir=os.path.dirname(output_path), delete=False,
    )
    temp_path = handle.name
    handle.close()
    expected_frames = end - start + 1
    select_expr = f"select=between(n\\,{start}\\,{end}),setpts=N/({float(fps):.12g}*TB)"
    command = [
        os.environ.get("FFMPEG_BIN", "ffmpeg"), "-y", "-hide_banner",
        "-loglevel", "error", "-i", video_path, "-map", "0:v:0",
        "-vf", select_expr, "-fps_mode", "passthrough", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-an", temp_path,
    ]
    logger.info("[stage1 exact-frame-cut] " + " ".join(command))
    try:
        subprocess.run(command, check=True, stdin=subprocess.DEVNULL)
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) <= 0:
            raise RuntimeError(f"ffmpeg produced an empty clip: {temp_path}")
        clip_meta = get_video_metadata(temp_path)
        actual_frames = int(clip_meta["total_frames"])
        if actual_frames != expected_frames:
            raise RuntimeError(
                f"Exact frame cut [{start}, {end}] expected {expected_frames} frames, "
                f"but ffmpeg produced {actual_frames}."
            )
        os.replace(temp_path, output_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _finalize_step_clip_name(
    *, clip_path: str, video_out: str, step: Dict[str, Any], overall_description: Any
) -> str:
    final_path = _step_clip_path(
        video_out,
        int(step["hand_step_id"]),
        int(step["start_frame_id"]),
        int(step["end_frame_id"]),
        overall_description,
    )
    if os.path.normpath(clip_path) != os.path.normpath(final_path):
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        os.replace(clip_path, final_path)
    return final_path


def _require_stage1_step_clip(
    *, video_out: str, step: Dict[str, Any]
) -> str:
    # Stage 1's manifest is authoritative after the overall-description rename.
    # Consumers must reuse that exact file instead of reconstructing a name.
    manifest_path = os.path.join(video_out, "stage1", "step_clips.json")
    if os.path.exists(manifest_path):
        manifest = read_json(manifest_path)
        step_id = int(step["hand_step_id"])
        for record in manifest.get("steps", []):
            if not isinstance(record, dict) or int(record.get("step_id", -1)) != step_id:
                continue
            expected = (int(step["start_frame_id"]), int(step["end_frame_id"]))
            actual = (int(record.get("start_frame_id", -1)), int(record.get("end_frame_id", -1)))
            if actual != expected:
                raise RuntimeError(
                    f"Stage 1 clip manifest interval mismatch for step_id={step_id}: "
                    f"expected={expected}, manifest={actual}"
                )
            clip_path = os.path.join(video_out, str(record.get("clip_path") or ""))
            if not os.path.exists(clip_path) or os.path.getsize(clip_path) <= 0:
                raise FileNotFoundError(
                    f"Stage 1 manifest clip is missing for step_id={step_id}: {clip_path}"
                )
            return clip_path
        raise KeyError(f"Stage 1 clip manifest has no record for step_id={step_id}: {manifest_path}")

    # Legacy fallback for outputs created before step_clips.json existed.
    semantics = step.get("draft_action_semantics", {})
    overall = semantics.get("overall") if isinstance(semantics, dict) else semantics
    clip_path = _step_clip_path(
        video_out,
        int(step["hand_step_id"]),
        int(step["start_frame_id"]),
        int(step["end_frame_id"]),
        overall,
    )
    if not os.path.exists(clip_path) or os.path.getsize(clip_path) <= 0:
        raise FileNotFoundError(
            f"Stage 1 step clip is missing for hand_step_id={step['hand_step_id']}: {clip_path}"
        )
    return clip_path


def _step_clip_mapping_text(step: Dict[str, Any]) -> str:
    start_frame = int(step["start_frame_id"])
    end_frame = int(step["end_frame_id"])
    return (
        f"The attached short video contains only original source frames {start_frame} through "
        f"{end_frame}, inclusive. Clip-local frame 0 maps to original source frame_id={start_frame}; "
        f"for every clip-local frame n, original frame_id={start_frame}+n. All requested and returned "
        "frame IDs and all bbox tracks use original-source coordinates, never clip-local IDs."
    )


STAGE1_HAND_EVENT_SYSTEM_PROMPT = """
You are an expert frame-level annotator for first-person hand-manipulation video.
Return exactly one valid JSON object and no Markdown or additional prose. Judge
every supplied source frame independently. Upstream data is used only to select
which frame IDs are supplied; do not use or assume its bbox, depth, detection,
or semantic values. Mere hand presence, hovering, resting,
natural arm swing, or an object merely being visible is not an effective hand
action.
""".strip()


def build_stage1_frame_prompt(*, frame_ids: Sequence[int], total_frames: int, fps: float, width: int, height: int) -> str:
    return f"""
Analyze every attached original source frame independently. The source is
{width}x{height}, {fps:.8g} FPS, with {total_frames} frames. The attached frame
IDs, in image order, are: {json.dumps([int(x) for x in frame_ids])}.

For each frame decide whether the primary operator is visibly performing an
EFFECTIVE hand action. Effective means manipulating, contacting, operating,
moving, supporting, or changing an object. Exclude mere presence, hovering,
resting, natural swing, and background people. Independently generate tight
pixel-xyxy bboxes for every visible primary left/right hand. Never infer or
copy coordinates from another frame.

For every generated bbox judge that exact bbox crop conceptually:
- arm_visible=1 only if a connected wrist/forearm is visibly included;
- hand_completeness_score is in [0,1], where 0 is absent/unrecognizable and 1
  is a fully exposed hand including the major fingers, palm/back, and wrist.

Return exactly:
{{
  "frames": [
    {{
      "frame_id": 0,
      "has_effective_hand_action": 1,
      "active_hands": ["left_hand", "right_hand"],
      "action_semantics": {{"left_hand": "...", "right_hand": "...", "overall": "..."}},
      "confidence": 0.0,
      "vlm_generated_bboxes": [
        {{"hand": "left_hand", "bbox_xyxy": [0, 0, 1, 1], "arm_visible": 0, "hand_completeness_score": 0.0}}
      ]
    }}
  ]
}}

Constraints:
- return exactly one record for every supplied frame_id, in the supplied order;
- has_effective_hand_action is exactly 0 or 1;
- active_hands contains only left_hand and/or right_hand;
- use "not active" for an inactive hand;
- confidence is a real number in [0,1];
- no extra keys.
""".strip()


def build_stage1_hand_event_prompt(*, total_frames: int, fps: float, width: int, height: int) -> str:
    """Compatibility wrapper retained for callers that inspect the prompt."""
    return build_stage1_frame_prompt(
        frame_ids=list(range(total_frames)), total_frames=total_frames, fps=fps, width=width, height=height
    )


def _usage_zero() -> Dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "api_calls": 0}


def _add_usage(total: Dict[str, int], usage: Dict[str, Any]) -> None:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        total[key] = int(total.get(key, 0)) + int(usage.get(key, 0) or 0)
    total["api_calls"] = int(total.get("api_calls", 0)) + 1


def _clamp_float(value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        out = float(value)
    except Exception:
        out = lo
    return min(hi, max(lo, out))


def _normalize_vlm_events(obj: Any, *, total_frames: int) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    errors: List[str] = []
    if not isinstance(obj, dict) or set(obj.keys()) != {"events"}:
        return None, ["Top-level output must contain exactly the key 'events'."]
    raw_events = obj.get("events")
    if not isinstance(raw_events, list):
        return None, ["events must be a list."]
    events: List[Dict[str, Any]] = []
    previous_end = -1
    for i, raw in enumerate(raw_events, start=1):
        if not isinstance(raw, dict):
            errors.append(f"events[{i - 1}] must be an object.")
            continue
        required = {"event_id", "start_frame_id", "end_frame_id", "active_hands", "action_description", "confidence"}
        if set(raw.keys()) != required:
            errors.append(f"events[{i - 1}] must contain exactly {sorted(required)}.")
            continue
        try:
            event_id = int(raw.get("event_id"))
            start = int(raw.get("start_frame_id"))
            end = int(raw.get("end_frame_id"))
        except Exception:
            errors.append(f"events[{i - 1}] has non-integer id/boundary.")
            continue
        hands_raw = raw.get("active_hands")
        hands = [str(x) for x in hands_raw] if isinstance(hands_raw, list) else []
        hands = list(dict.fromkeys(x for x in hands if x in {"left_hand", "right_hand"}))
        description = str(raw.get("action_description") or "").strip()
        if event_id != i:
            errors.append(f"events[{i - 1}].event_id must be {i}.")
        if total_frames <= 0 or not (0 <= start <= end < total_frames):
            errors.append(f"events[{i - 1}] boundaries are outside source video.")
        if start <= previous_end:
            errors.append(f"events[{i - 1}] overlaps or is not chronological.")
        if not hands:
            errors.append(f"events[{i - 1}].active_hands is empty/invalid.")
        if not description:
            errors.append(f"events[{i - 1}].action_description is empty.")
        previous_end = end
        events.append(
            {
                "event_id": event_id,
                "start_frame_id": start,
                "end_frame_id": end,
                "active_hands": hands,
                "action_description": description,
                "confidence": _clamp_float(raw.get("confidence")),
            }
        )
    if errors:
        return None, errors
    return {"events": events}, []


def _normalize_vlm_frame_batch(
    obj: Any, *, expected_frame_ids: Sequence[int], width: int, height: int
) -> Tuple[Optional[List[Dict[str, Any]]], List[str]]:
    errors: List[str] = []
    if not isinstance(obj, dict) or set(obj.keys()) != {"frames"} or not isinstance(obj.get("frames"), list):
        return None, ["Top-level output must contain exactly the list key 'frames'."]
    expected = [int(x) for x in expected_frame_ids]
    out: List[Dict[str, Any]] = []
    for index, raw in enumerate(obj["frames"]):
        required = {
            "frame_id", "has_effective_hand_action", "active_hands",
            "action_semantics", "confidence", "vlm_generated_bboxes",
        }
        if not isinstance(raw, dict) or set(raw.keys()) != required:
            errors.append(f"frames[{index}] has invalid keys.")
            continue
        try:
            frame_id = int(raw["frame_id"])
            action_flag = int(raw["has_effective_hand_action"])
        except Exception:
            errors.append(f"frames[{index}] has invalid frame_id/action flag.")
            continue
        if action_flag not in {0, 1}:
            errors.append(f"frames[{index}].has_effective_hand_action must be 0 or 1.")
            action_flag = 0
        hands_raw = raw.get("active_hands")
        hands = list(dict.fromkeys(str(x) for x in hands_raw if str(x) in {"left_hand", "right_hand"})) if isinstance(hands_raw, list) else []
        semantics_raw = raw.get("action_semantics")
        if not isinstance(semantics_raw, dict) or set(semantics_raw.keys()) != {"left_hand", "right_hand", "overall"}:
            errors.append(f"frames[{index}].action_semantics has invalid keys.")
            semantics_raw = {}
        semantics = {
            hand: str(semantics_raw.get(hand) or ("not active" if hand != "overall" else "No effective hand action.")).strip()
            for hand in ("left_hand", "right_hand", "overall")
        }
        boxes_out: List[Dict[str, Any]] = []
        boxes_raw = raw.get("vlm_generated_bboxes")
        if not isinstance(boxes_raw, list):
            errors.append(f"frames[{index}].vlm_generated_bboxes must be a list.")
            boxes_raw = []
        seen_hands: set[str] = set()
        for box_index, box_raw in enumerate(boxes_raw):
            if not isinstance(box_raw, dict) or set(box_raw.keys()) != {
                "hand", "bbox_xyxy", "arm_visible", "hand_completeness_score"
            }:
                errors.append(f"frames[{index}].vlm_generated_bboxes[{box_index}] has invalid keys.")
                continue
            hand = str(box_raw.get("hand") or "")
            bbox = _valid_bbox(box_raw.get("bbox_xyxy"), width=width, height=height)
            try:
                arm_visible = int(box_raw.get("arm_visible"))
            except Exception:
                arm_visible = -1
            if hand not in {"left_hand", "right_hand"} or hand in seen_hands or bbox is None or arm_visible not in {0, 1}:
                errors.append(f"frames[{index}].vlm_generated_bboxes[{box_index}] is invalid.")
                continue
            seen_hands.add(hand)
            boxes_out.append({
                "frame_id": frame_id,
                "hand": hand,
                "bbox_xyxy": bbox,
                "bbox_source": "vlm_generated",
                "arm_visible": arm_visible,
                "hand_completeness_score": _clamp_float(box_raw.get("hand_completeness_score")),
            })
        out.append({
            "frame_id": frame_id,
            "has_effective_hand_action": action_flag,
            "active_hands": hands if action_flag else [],
            "action_semantics": semantics,
            "confidence": _clamp_float(raw.get("confidence")),
            "vlm_generated_bboxes": boxes_out,
        })
    actual = [int(item["frame_id"]) for item in out]
    if actual != expected:
        errors.append(f"Frame IDs must exactly match the request: expected={expected}, got={actual}.")
    return (None, errors) if errors else (out, [])


def _frame_annotations_to_events(frames: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create the legacy interval view deterministically from authoritative frame flags."""
    positive = [item for item in frames if int(item.get("has_effective_hand_action", 0)) == 1]
    events: List[Dict[str, Any]] = []
    group: List[Dict[str, Any]] = []
    for item in positive:
        hands = tuple(item.get("active_hands", []))
        if group and (int(item["frame_id"]) != int(group[-1]["frame_id"]) + 1 or hands != tuple(group[-1].get("active_hands", []))):
            first, last = group[0], group[-1]
            events.append({
                "event_id": len(events) + 1, "start_frame_id": int(first["frame_id"]),
                "end_frame_id": int(last["frame_id"]), "active_hands": list(first.get("active_hands", [])),
                "action_description": str(first.get("action_semantics", {}).get("overall") or "Effective hand action."),
                "confidence": sum(float(x.get("confidence", 0.0)) for x in group) / len(group),
            })
            group = []
        group.append(item)
    if group:
        first, last = group[0], group[-1]
        events.append({
            "event_id": len(events) + 1, "start_frame_id": int(first["frame_id"]),
            "end_frame_id": int(last["frame_id"]), "active_hands": list(first.get("active_hands", [])),
            "action_description": str(first.get("action_semantics", {}).get("overall") or "Effective hand action."),
            "confidence": sum(float(x.get("confidence", 0.0)) for x in group) / len(group),
        })
    return events


def _expanded_upstream_frame_ids(
    upstream: Dict[str, Any], *, total_frames: int, fps: float, padding_sec: float
) -> List[int]:
    """Expand each upstream-provided frame ID by a symmetric source-time window."""
    upstream_ids: List[int] = []
    for frame_id, frame in upstream.get("frames", {}).items():
        if not isinstance(frame, dict):
            continue
        hands = frame.get("hands", {}) if isinstance(frame.get("hands"), dict) else {}
        detected = any(
            isinstance(hands.get(hand_name), dict)
            and bool(hands[hand_name].get("detected"))
            and isinstance(hands[hand_name].get("bbox_xyxy"), list)
            for hand_name in ("left_hand", "right_hand")
        )
        if bool(frame.get("keep")) or detected:
            upstream_ids.append(int(frame_id))
    upstream_ids = sorted(set(upstream_ids))
    if not upstream_ids:
        return []
    padding_frames = max(0, int(round(float(fps) * max(0.0, float(padding_sec)))))
    selected: set[int] = set()
    for frame_id in upstream_ids:
        selected.update(
            range(max(0, frame_id - padding_frames), min(int(total_frames), frame_id + padding_frames + 1))
        )
    return sorted(selected)


def _read_selected_frame_batches(
    video_path: str, *, frame_ids: Sequence[int], batch_size: int
) -> Iterable[Tuple[List[int], List[Dict[str, Any]]]]:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for frame-level VLM analysis.")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for frame-level analysis: {video_path}")
    ids: List[int] = []
    images: List[Dict[str, Any]] = []
    try:
        for frame_id in sorted(set(int(x) for x in frame_ids)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Could not decode selected source frame {frame_id}.")
            encoded = _encode_crop(frame, [0, 0, int(frame.shape[1]), int(frame.shape[0])])
            if encoded is None:
                raise RuntimeError(f"Could not JPEG-encode source frame {frame_id}.")
            ids.append(frame_id)
            images.extend([
                {"type": "text", "text": f"Original source frame_id={frame_id}."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
            ])
            if len(ids) >= batch_size:
                yield ids, images
                ids, images = [], []
        if ids:
            yield ids, images
    finally:
        cap.release()


def _normalize_step_semantics(obj: Any, *, step_id: int) -> Tuple[Optional[Dict[str, str]], List[str]]:
    if not isinstance(obj, dict) or set(obj.keys()) != {"step_id", "action_semantics"}:
        return None, ["Response must contain exactly step_id and action_semantics."]
    try:
        response_step_id = int(obj["step_id"])
    except Exception:
        return None, ["step_id must be an integer."]
    if response_step_id != int(step_id):
        return None, [f"Expected step_id={step_id}, got {response_step_id}."]
    semantics = obj.get("action_semantics")
    if not isinstance(semantics, dict) or set(semantics.keys()) != {"left_hand", "right_hand", "overall"}:
        return None, ["action_semantics must contain exactly left_hand/right_hand/overall."]
    normalized = {key: str(semantics[key]).strip() for key in ("left_hand", "right_hand", "overall")}
    errors: List[str] = []
    for key, value in normalized.items():
        if not value:
            errors.append(f"action_semantics.{key} must not be empty.")
            continue
        if any(ord(char) > 127 for char in value):
            errors.append(f"action_semantics.{key} must be pure English ASCII text.")
        if not re.search(r"[A-Za-z]", value):
            errors.append(f"action_semantics.{key} must contain an English description.")
    return (None, errors) if errors else (normalized, [])


def _normalize_multiple_action_subjects(value: Any) -> Tuple[Optional[bool], List[str]]:
    """Accept only an explicit boolean (or unambiguous 0/1 during repair)."""
    if isinstance(value, bool):
        return value, []
    if isinstance(value, int) and value in (0, 1):
        return bool(value), []
    return None, ["multiple_action_subjects must be a JSON boolean."]


def _conservative_json_repairs(text: str) -> List[str]:
    """Return syntax-only JSON repair candidates without inventing annotations."""
    source = extract_json_from_response(str(text or "")).strip()
    candidates: List[str] = []
    repaired = source.translate(str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"}))
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    repaired = re.sub(r"\bTrue\b", "true", repaired)
    repaired = re.sub(r"\bFalse\b", "false", repaired)
    repaired = re.sub(r"\bNone\b", "null", repaired)
    if repaired and repaired != source:
        candidates.append(repaired)
    return candidates


def _generate_stage1_keep_step_semantics(
    *, video_path: str, video_out: str, client: Any, api_cfg: Any,
    overwrite: bool, max_retries: int,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    meta = get_video_metadata(video_path)
    upstream = load_upstream_hand_data(video_path, meta)
    boundaries = build_hand_step_boundaries(
        video_path=video_path, vlm_events={}, upstream=upstream, meta=meta, max_gap_frames=0
    )
    stage1_dir = os.path.join(video_out, "stage1")
    usage_total = _usage_zero()
    results: List[Dict[str, Any]] = []
    system_prompt = (
        "Inspect one annotated Stage-1 step-only video exactly once and return all downstream annotations in one strict "
        "JSON object: English hand-action semantics, a temporally grounded step-local object identity inventory, "
        "hand-object relations, "
        "and a boolean decision on whether two or more distinct human/agent operators actively perform the step, "
        "and complete original-frame intervals where each supplied upstream bbox track contains a connected "
        "wrist or forearm. Most operators wear thick work gloves: treat the visible glove as the hand, use its "
        "silhouette, motion and object contact rather than expecting bare fingers or skin. Never change frame IDs "
        "or source bboxes. Do not return per-frame visibility flags."
    )
    for step in boundaries["hand_steps"]:
        step_id = int(step["hand_step_id"])
        logger.info(
            f"[stage1] Doubao Seed step clip {step_id}/{len(boundaries['hand_steps'])} "
            f"source_frames=[{int(step['start_frame_id'])}, {int(step['end_frame_id'])}]"
        )
        step_dir = os.path.join(stage1_dir, f"step_{step_id:03d}")
        stage2_step_dir = os.path.join(video_out, "stage2", f"step_{step_id:03d}")
        stage3_step_dir = os.path.join(video_out, "stage3", f"step_{step_id:03d}")
        for output_dir in (step_dir, stage2_step_dir, stage3_step_dir):
            os.makedirs(output_dir, exist_ok=True)
        clip_path = _ensure_step_clip(
            video_path=video_path, video_out=video_out, step=step, fps=float(meta["fps"])
        )
        visualization_path = _ensure_step_visualization(
            clip_path=clip_path, video_out=video_out, step=step, fps=float(meta["fps"])
        )
        request_video_fps = _step_video_request_fps(
            step,
            source_fps=float(meta["fps"]),
            configured_fps=float(api_cfg.video_fps),
        )
        logger.info(
            f"[stage1] step_id={step_id} model_video_fps={request_video_fps:.4g} "
            f"configured_video_fps={float(api_cfg.video_fps):.4g} "
            f"max_video_samples={MAX_STEP_VIDEO_SAMPLES}"
        )
        video_content = build_video_api_content(
            visualization_path, [], video_fps=request_video_fps, include_manifest=False
        )
        compact_stage3_tracks = _compact_stage3_bbox_tracks_for_prompt(step)
        compact_analysis_tracks = _compact_analysis_bbox_tracks_for_prompt(step)
        prompt = f"""
Analyze exactly step_id={step_id}, inclusive original source-frame interval
[{int(step['start_frame_id'])}, {int(step['end_frame_id'])}]. Do not change or
split this interval. {_step_clip_mapping_text(step)}

Stage-3 evaluation is restricted to these original valid-action intervals:
{json.dumps(step.get('valid_action_intervals', []), ensure_ascii=False)}

Original authoritative upstream bbox tracks for bbox_arm_visibility. Judge arm visibility ONLY inside these
original source boxes; available_frame_intervals are exhaustive and representative_bboxes are sampled:
{json.dumps(compact_stage3_tracks, ensure_ascii=False)}

Inference bbox evidence for action_semantics, object_inventory, and hand_object_relations:
{json.dumps(compact_analysis_tracks, ensure_ascii=False)}

Bbox quality annotations. A possible_wrist_only record means width, height and area all suddenly shrank,
then recovered, so the red two-sided interpolated and expanded analysis bbox must be preferred for semantic/object reasoning.
The original box remains authoritative for bbox_arm_visibility and final output:
{json.dumps(step.get('bbox_quality_annotations', []), ensure_ascii=False)}

Return exactly:
{{
  "step_id": {step_id},
  "action_semantics": {{"left_hand": "...", "right_hand": "...", "overall": "..."}},
  "multiple_action_subjects": false,
  "hand_object_relations": [{{"object":"...","hand":"left hand","relation":"...","entity_ref":"step_{step_id:03d}_obj_001"}}],
  "object_inventory": [{{
    "entity_ref":"step_{step_id:03d}_obj_001",
    "canonical_name":"...",
    "description":"...",
    "selection_reason":"...",
    "priority":"direct_hand_interaction",
    "aliases":["..."],
    "identity_profile":{{
      "stable_features":{{"category":"...","material":"...","base_color":"...","distinctive_features":["..."]}},
      "mutable_features":["position","orientation","shape","open_closed_state","contents"]
    }},
    "visible_intervals":[{{"start_frame_id":0,"end_frame_id":0}}],
    "representative_frame_ids":[0],
    "hand_interactions":[{{"hand":"left_hand","action":"...","start_frame_id":0,"end_frame_id":0}}],
    "state_transitions":[{{"frame_id":0,"from":"...","to":"...","cause":"..."}}],
    "identity_evidence":"...",
    "identity_status":"resolved",
    "confidence":0.0
  }}],
  "bbox_arm_visibility": [{{"hand":"left_hand","bbox_source":"upstream_sam3","arm_visible_intervals":[{{"start_frame_id":0,"end_frame_id":0}}]}}]
}}
Use "not active" when a hand does not meaningfully participate.
Write left_hand, right_hand, and overall descriptions in English ASCII only.
Set multiple_action_subjects=true only when at least two distinct people or agents actively perform or manipulate
something during this step. The two hands/gloves of one person count as one subject. A person merely visible,
passing by, reflected, printed on a poster, or standing without participating does not count. Do not infer a second
operator from object motion, shadows, reflections, or multiple detected hand boxes. Set it false for one active
operator or when evidence is insufficient.
Do not use Chinese characters, bilingual text, or non-ASCII punctuation.
object_inventory priority must be direct_hand_interaction or moved_or_shape_changed; exclude unchanged background objects,
hands, arms, people, shadows and reflections. Inventory means physical instances, not category names. Include every object
mentioned by either hand description, overall, or hand_object_relations. Use exactly one entity_ref for the same physical
instance across the entire step, including when it is picked up, moved, rotated, opened, closed, folded, deformed,
partly occluded, carried out of view, or later brought back. Record every reappearance as another visible_intervals item.
Do not create a new entity merely because position, orientation, apparent size, shape, contents or open/closed state changed.
Use stable material, markings and structure plus the hand-action causal chain to preserve identity. Conversely, never merge
two simultaneously visible or otherwise distinguishable instances of the same category. visible_intervals are inclusive,
chronological, non-overlapping original frame IDs. representative_frame_ids must provide evidence from separate appearances.
identity_status must be resolved, uncertain, or new_instance. If identity is ambiguous, keep instances separate and use
uncertain rather than silently merging them. confidence is in [0,1].
For thick gloves, regard the glove body as the hand and infer contact from continuous glove/object motion, not skin or
individual-finger visibility. Orange inference boxes are deterministic 10-percent-per-side expansions intended to
cover a glove whose outer silhouette was slightly missed. Red boxes were first temporally repaired and then expanded.
Use orange/red inference boxes only for action semantics and object reasoning.
Return exactly one bbox_arm_visibility record for every supplied hand+bbox_source track. Intervals are inclusive,
chronological original frame IDs and may contain only frames for which that upstream bbox track exists. Return an
empty interval list if a track never contains a connected wrist or forearm. Do not return bbox coordinates because
all bbox geometry is copied verbatim from upstream. For Stage 3, ignore every frame outside the supplied
valid-action intervals, even though those bridge frames remain visible in the step video.
""".strip()
        last_content = ""
        errors: List[str] = []
        combined: Optional[Dict[str, Any]] = None

        def validate_combined_response(content: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
            try:
                parsed = json.loads(extract_json_from_response(content))
                required_keys = {
                    "step_id", "action_semantics", "hand_object_relations",
                    "object_inventory", "bbox_arm_visibility", "multiple_action_subjects",
                }
                if not isinstance(parsed, dict) or set(parsed.keys()) != required_keys:
                    raise ValueError(f"Combined response must contain exactly {sorted(required_keys)}")
                semantics, semantic_errors = _normalize_step_semantics(
                    {"step_id": parsed.get("step_id"), "action_semantics": parsed.get("action_semantics")},
                    step_id=step_id,
                )
                normalized_bbox_arm, bbox_errors = _normalize_bbox_arm_intervals(
                    parsed.get("bbox_arm_visibility"), step=step
                )
                if semantics is None or normalized_bbox_arm is None:
                    raise ValueError(" | ".join((semantic_errors + bbox_errors)[:10]))
                multiple_subjects, subject_errors = _normalize_multiple_action_subjects(
                    parsed.get("multiple_action_subjects")
                )
                if multiple_subjects is None:
                    raise ValueError(" | ".join(subject_errors[:10]))
                visible_by_hand: Dict[str, List[int]] = {"left_hand": [], "right_hand": []}
                for track_eval in normalized_bbox_arm:
                    visible_by_hand[track_eval["hand"]].extend(
                        frame_id
                        for interval in track_eval["arm_visible_intervals"]
                        for frame_id in range(int(interval["start_frame_id"]), int(interval["end_frame_id"]) + 1)
                    )
                arm_intervals = {
                    hand: _contiguous_frame_intervals(frame_ids)
                    for hand, frame_ids in visible_by_hand.items()
                }
                wrapped = {"steps": [{
                    "step_id": step_id,
                    "action_semantics": semantics,
                    "hand_object_relations": parsed.get("hand_object_relations"),
                    "object_inventory": parsed.get("object_inventory"),
                    "arm_visibility_intervals": arm_intervals,
                }]}
                normalized_review, review_errors = _normalize_whole_video_step_review(
                    wrapped, boundaries={"hand_steps": [step]}
                )
                if not normalized_review:
                    raise ValueError(" | ".join(review_errors[:10]))
                result = dict(normalized_review[0])
                result["bbox_arm_visibility"] = normalized_bbox_arm
                result["multiple_action_subjects"] = multiple_subjects
                return result, []
            except Exception as exc:
                return None, [f"JSON/validation error: {exc}"]

        # Most steps upload the video once. Only a response which cannot be made
        # valid by conservative syntax repair and two text-only repair calls is
        # allowed one final video re-analysis.
        for video_attempt in range(1, 3):
            prefix = build_retry_prefix(errors, last_content) if video_attempt > 1 else ""
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "text", "text": prefix + prompt}] + video_content},
            ]
            last_content, usage = call_chat_completion(
                client,
                api_cfg,
                messages,
                max_tokens=min(api_cfg.max_tokens, 10000),
                video_fps=request_video_fps,
                temperature=0.2,
            )
            _add_usage(usage_total, usage)
            write_text(
                os.path.join(step_dir, f"semantics_raw_response_video_attempt_{video_attempt}.txt"),
                last_content,
            )
            combined, errors = validate_combined_response(last_content)
            if combined is None:
                for candidate_index, candidate in enumerate(_conservative_json_repairs(last_content), start=1):
                    write_text(
                        os.path.join(step_dir, f"semantics_local_repair_video_{video_attempt}_{candidate_index}.json"),
                        candidate,
                    )
                    combined, errors = validate_combined_response(candidate)
                    if combined is not None:
                        last_content = candidate
                        break
            for repair_attempt in range(1, 3) if combined is None else ():
                repair_prompt = f"""
Repair the following model response into one valid JSON object. This is a formatting/schema repair task only.
Do not add, remove, merge, rename, reinterpret, or guess any physical object or visual fact. Preserve all frame IDs,
entity_ref values, descriptions, intervals, relations and confidence values whenever present. Fix quoting, brackets,
commas, JSON types and schema placement only. If information needed by the schema is absent, return the closest
faithful JSON without inventing it; strict validation will reject it and the video will be re-analyzed.

Validation error:
{errors[0] if errors else 'invalid JSON'}

Required contract:
{prompt}

Response to repair:
{last_content}

Return JSON only.
""".strip()
                repaired_content, repair_usage = call_chat_completion(
                    client,
                    api_cfg,
                    [
                        {"role": "system", "content": "You repair JSON syntax and schema without changing visual facts."},
                        {"role": "user", "content": repair_prompt},
                    ],
                    max_tokens=min(api_cfg.max_tokens, 10000),
                    temperature=0.1,
                )
                _add_usage(usage_total, repair_usage)
                write_text(
                    os.path.join(
                        step_dir,
                        f"semantics_text_repair_video_{video_attempt}_attempt_{repair_attempt}.txt",
                    ),
                    repaired_content,
                )
                combined, errors = validate_combined_response(repaired_content)
                if combined is not None:
                    last_content = repaired_content
                    break
            if combined is not None:
                break
            write_text(
                os.path.join(step_dir, f"semantics_validation_errors_video_attempt_{video_attempt}.txt"),
                "\n".join(errors),
            )
        write_text(os.path.join(step_dir, "semantics_system_prompt.txt"), system_prompt)
        write_text(os.path.join(step_dir, "semantics_user_prompt.txt"), prompt)
        write_text(os.path.join(step_dir, "semantics_raw_response.txt"), last_content)
        if combined is None:
            raise RuntimeError(f"Combined step_id={step_id} analysis failed: " + " | ".join(errors[:10]))
        semantics = combined["action_semantics"]
        clip_path = _finalize_step_clip_name(
            clip_path=clip_path,
            video_out=video_out,
            step=step,
            overall_description=semantics.get("overall"),
        )
        record = {
            "step_id": step_id,
            "start_frame_id": int(step["start_frame_id"]),
            "end_frame_id": int(step["end_frame_id"]),
            "clip_path": os.path.relpath(clip_path, video_out),
            "visualization_clip_path": os.path.relpath(visualization_path, video_out),
            "clip_local_start_frame_id": 0,
            "source_frame_id_offset": int(step["start_frame_id"]),
            "action_semantics": semantics,
            "multiple_action_subjects": bool(combined["multiple_action_subjects"]),
            "hand_object_relations": combined["hand_object_relations"],
            "object_inventory": combined["object_inventory"],
            "arm_visibility_intervals": combined["arm_visibility_intervals"],
            "bbox_arm_visibility": combined["bbox_arm_visibility"],
            "bbox_tracks": step.get("bbox_tracks", {}),
            "analysis_bbox_tracks": step.get("analysis_bbox_tracks", {}),
            "bbox_quality_annotations": step.get("bbox_quality_annotations", []),
            "bbox_repair_summary": step.get("bbox_repair_summary", {}),
            "model_video_fps": request_video_fps,
        }
        write_json(os.path.join(step_dir, "action_semantics.json"), record)
        inventory_validation = _build_step_inventory_validation(
            combined["object_inventory"], combined["hand_object_relations"]
        )
        write_json(os.path.join(stage2_step_dir, "object_inventory.json"), {
            "schema": "step_object_identity_inventory.v2",
            "step_id": step_id,
            "source_frame_range": {
                "start_frame_id": int(step["start_frame_id"]),
                "end_frame_id": int(step["end_frame_id"]),
            },
            "hand_object_relations": combined["hand_object_relations"],
            "object_inventory": combined["object_inventory"],
            "validation": inventory_validation,
            "generated_by": "single_pass_step_video_analysis",
        })
        if inventory_validation["status"] != "passed":
            raise RuntimeError(
                f"step_id={step_id} object identity inventory needs review: "
                + json.dumps(inventory_validation, ensure_ascii=False)
            )
        write_json(os.path.join(stage3_step_dir, "bbox_arm_visibility.json"), {
            "schema": "step_bbox_arm_visibility.v1",
            "step_id": step_id,
            "start_frame_id": int(step["start_frame_id"]),
            "end_frame_id": int(step["end_frame_id"]),
            "endpoint_semantics": "inclusive",
            "evaluation_scope": "valid_keep_true_actions_only",
            "valid_action_intervals": list(step.get("valid_action_intervals", [])),
            "bbox_arm_visibility": combined["bbox_arm_visibility"],
            "arm_visibility_intervals": combined["arm_visibility_intervals"],
            "bbox_quality_annotations": list(step.get("bbox_quality_annotations", [])),
            "source_bbox_reliability_policy": (
                "arm visibility is evaluated only inside immutable upstream bbox; "
                "flagged shrink intervals may be unreliable"
            ),
            "generated_by": "single_pass_step_video_analysis",
        })
        results.append(record)
    # The deterministic cut manifest is initially written before model calls.
    # Keep it authoritative after clips gain their overall-description suffix.
    clips_manifest_path = os.path.join(stage1_dir, "step_clips.json")
    if os.path.exists(clips_manifest_path):
        clips_manifest = read_json(clips_manifest_path)
        result_by_id = {int(item["step_id"]): item for item in results}
        for clip_record in clips_manifest.get("steps", []):
            described = result_by_id.get(int(clip_record.get("step_id", -1)))
            if not described:
                continue
            clip_record["clip_path"] = described["clip_path"]
            clip_record["visualization_clip_path"] = described["visualization_clip_path"]
            clip_record["left_hand"] = described["action_semantics"]["left_hand"]
            clip_record["right_hand"] = described["action_semantics"]["right_hand"]
            clip_record["overall"] = described["action_semantics"]["overall"]
            clip_record["multiple_action_subjects"] = bool(described.get("multiple_action_subjects", False))
        write_json(clips_manifest_path, clips_manifest)
    payload = {
        "schema": "upstream_keep_step_semantics.v1",
        "video_id": video_id_from_path(video_path),
        "source_video": build_source_video_meta(video_path),
        "boundary_rule": (
            f"discard keep=true runs shorter than {MIN_VALID_KEEP_FRAMES} frames; "
            f"merge consecutive valid runs when their total intervening frame gap is "
            f"shorter than {MERGE_GAP_FRAMES_EXCLUSIVE} frames"
        ),
        "step_semantics": results,
        "events": [],
        "status": "completed" if results else "no_effective_action",
    }
    write_json(os.path.join(stage1_dir, "vlm_hand_events.json"), payload)
    return payload, usage_total


def materialize_stage1_keep_clips(
    *, video_path: str, video_out: str
) -> Dict[str, Any]:
    """Create Stage-1 clips from filtered and short-gap-merged keep=true runs."""
    meta = get_video_metadata(video_path)
    upstream = load_upstream_hand_data(video_path, meta)
    if not upstream.get("available"):
        raise FileNotFoundError(
            f"Upstream hand-screening JSON is required for Stage 1: {upstream.get('path')}"
        )
    boundaries = build_hand_step_boundaries(
        video_path=video_path,
        vlm_events={},
        upstream=upstream,
        meta=meta,
    )
    steps = boundaries.get("hand_steps", [])
    records: List[Dict[str, Any]] = []
    for step in steps:
        clip_path = _ensure_step_clip(
            video_path=video_path,
            video_out=video_out,
            step=step,
            fps=float(meta["fps"]),
        )
        start = int(step["start_frame_id"])
        end = int(step["end_frame_id"])
        records.append({
            "step_id": int(step["hand_step_id"]),
            "start_frame_id": start,
            "end_frame_id": end,
            "end_frame_id_exclusive": end + 1,
            "frame_count": end - start + 1,
            "clip_path": os.path.relpath(clip_path, video_out),
            "source_frame_id_offset": start,
            "valid_action_intervals": list(step.get("valid_action_intervals", [])),
            "visualization_clip_path": os.path.relpath(
                _step_visualization_path(video_out, step), video_out
            ),
            "bbox_quality_annotations": list(step.get("bbox_quality_annotations", [])),
            "bbox_repair_summary": dict(step.get("bbox_repair_summary", {})),
        })
    payload = {
        "schema": "stage1_keep_clips.v1",
        "video_id": video_id_from_path(video_path),
        "source_video": build_source_video_meta(video_path),
        "upstream_json": upstream.get("path"),
        "boundary_rule": (
            f"discard keep=true runs shorter than {MIN_VALID_KEEP_FRAMES} frames; "
            f"merge consecutive valid runs when their total intervening frame gap is "
            f"shorter than {MERGE_GAP_FRAMES_EXCLUSIVE} frames"
        ),
        "minimum_valid_keep_frames": MIN_VALID_KEEP_FRAMES,
        "merge_when_intervening_frames_less_than": MERGE_GAP_FRAMES_EXCLUSIVE,
        "candidate_frame_ranges": boundaries.get("candidate_frame_union", {}),
        "frame_index_base": 0,
        "endpoint_semantics": "inclusive",
        "steps": records,
        "status": "completed" if records else "no_effective_action",
    }
    stage1_dir = os.path.join(video_out, "stage1")
    write_json(os.path.join(stage1_dir, "step_clips.json"), payload)
    logger.info(
        f"[stage1] materialized {len(records)} filtered/merged keep clips from "
        f"{upstream.get('path')}"
    )
    return payload


def generate_stage1_vlm_hand_events(
    *,
    video_path: str,
    video_out: str,
    client: Any,
    api_cfg: Any,
    frames: List[Dict[str, Any]],
    overwrite: bool,
    max_retries: int,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """VLM-review upstream frame IDs plus a configurable symmetric time window."""
    return _generate_stage1_keep_step_semantics(
        video_path=video_path,
        video_out=video_out,
        client=client,
        api_cfg=api_cfg,
        overwrite=overwrite,
        max_retries=max_retries,
    )
    stage1_dir = os.path.join(video_out, "stage1")
    events_path = os.path.join(stage1_dir, "vlm_hand_events.json")
    raw_path = os.path.join(stage1_dir, "vlm_hand_events_raw_response.txt")
    system_path = os.path.join(stage1_dir, "vlm_hand_events_system_prompt.txt")
    user_path = os.path.join(stage1_dir, "vlm_hand_events_user_prompt.txt")
    if not overwrite and os.path.exists(events_path):
        cached = read_json(events_path)
        annotations = cached.get("frame_annotations") if isinstance(cached, dict) else None
        selected_ids = cached.get("vlm_selected_frame_ids") if isinstance(cached, dict) else None
        if (
            cached.get("analysis_mode") == "upstream_frame_ids_with_time_padding"
            and isinstance(annotations, list)
            and isinstance(selected_ids, list)
            and [int(item.get("frame_id", -1)) for item in annotations if isinstance(item, dict)] == [int(x) for x in selected_ids]
        ):
            return cached, _usage_zero()

    meta = get_video_metadata(video_path)
    write_text(system_path, STAGE1_HAND_EVENT_SYSTEM_PROMPT)
    batch_size = max(1, int(os.environ.get("TEXT_LABEL_VLM_FRAME_BATCH_SIZE", "12")))
    padding_sec = max(0.0, float(os.environ.get("TEXT_LABEL_VLM_FRAME_PADDING_SEC", "0.5")))
    upstream = load_upstream_hand_data(video_path, meta)
    upstream_frame_ids = _expanded_upstream_frame_ids(
        upstream,
        total_frames=int(meta["total_frames"]),
        fps=float(meta["fps"]),
        padding_sec=0.0,
    )
    selected_frame_ids = _expanded_upstream_frame_ids(
        upstream,
        total_frames=int(meta["total_frames"]),
        fps=float(meta["fps"]),
        padding_sec=padding_sec,
    )
    if not selected_frame_ids:
        raise RuntimeError(
            "Stage 1 requires upstream frame IDs in processed_hand_screening/hand_screening.json; "
            "full-video per-frame VLM fallback is intentionally disabled."
        )
    usage_total = _usage_zero()
    annotations: List[Dict[str, Any]] = []
    raw_responses: List[str] = []
    prompts: List[str] = []
    for batch_number, (frame_ids, image_content) in enumerate(
        _read_selected_frame_batches(
            video_path, frame_ids=selected_frame_ids, batch_size=batch_size
        ), start=1
    ):
        prompt = build_stage1_frame_prompt(
            frame_ids=frame_ids, total_frames=int(meta["total_frames"]), fps=float(meta["fps"]),
            width=int(meta["width"]), height=int(meta["height"]),
        )
        prompts.append(f"=== batch {batch_number} frames {frame_ids[0]}-{frame_ids[-1]} ===\n{prompt}")
        last_content = ""
        last_errors: List[str] = []
        normalized_batch: Optional[List[Dict[str, Any]]] = None
        for attempt in range(1, max(1, int(max_retries)) + 1):
            prefix = build_retry_prefix(last_errors, last_content) if attempt > 1 else ""
            messages = [
                {"role": "system", "content": STAGE1_HAND_EVENT_SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "text", "text": prefix + prompt}] + image_content},
            ]
            last_content, usage = call_chat_completion(
                client, api_cfg, messages, max_tokens=min(api_cfg.max_tokens, max(4000, batch_size * 600))
            )
            _add_usage(usage_total, usage)
            try:
                obj = json.loads(extract_json_from_response(last_content))
                normalized_batch, last_errors = _normalize_vlm_frame_batch(
                    obj, expected_frame_ids=frame_ids, width=int(meta["width"]), height=int(meta["height"])
                )
            except Exception as exc:
                normalized_batch, last_errors = None, [f"JSON parse error: {exc}"]
            if normalized_batch is not None:
                break
        raw_responses.append(f"=== batch {batch_number} frames {frame_ids[0]}-{frame_ids[-1]} ===\n{last_content}")
        if normalized_batch is None:
            raise RuntimeError(
                f"Stage 1 frame batch {frame_ids[0]}-{frame_ids[-1]} failed: " + " | ".join(last_errors[:10])
            )
        annotations.extend(normalized_batch)
    expected_ids = selected_frame_ids
    actual_ids = [int(item["frame_id"]) for item in annotations]
    if actual_ids != expected_ids:
        raise RuntimeError(
            f"Stage 1 did not analyze every selected frame: expected {len(expected_ids)}, got {len(actual_ids)}."
        )
    write_text(user_path, "\n\n".join(prompts))
    write_text(raw_path, "\n\n".join(raw_responses))
    events = _frame_annotations_to_events(annotations)
    payload = {
        "schema": HAND_EVENTS_SCHEMA,
        "video_id": video_id_from_path(video_path),
        "source_video": build_source_video_meta(video_path),
        "fps": float(meta["fps"]),
        "total_frames": int(meta["total_frames"]),
        "frame_index_base": 0,
        "endpoint_semantics": "inclusive",
        "detector_independence": "vlm_only_no_upstream_data",
        "analysis_mode": "upstream_frame_ids_with_time_padding",
        "vlm_frame_batch_size": batch_size,
        "vlm_frame_padding_sec": padding_sec,
        "vlm_frame_padding_frames": int(round(float(meta["fps"]) * padding_sec)),
        "upstream_frame_ids": upstream_frame_ids,
        "vlm_selected_frame_ids": selected_frame_ids,
        "vlm_action_frame_ids": [
            int(item["frame_id"]) for item in annotations if int(item["has_effective_hand_action"]) == 1
        ],
        "frame_annotations": annotations,
        "events": events,
    }
    write_json(events_path, payload)
    return payload, usage_total


def upstream_hand_screening_path(video_path: str) -> str:
    raw_dir = os.path.dirname(os.path.abspath(video_path))
    unit_dir = os.path.dirname(raw_dir)
    return os.path.join(unit_dir, "processed_hand_screening", "hand_screening.json")


def _valid_bbox(value: Any, *, width: int, height: int) -> Optional[List[int]]:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in value]
    except Exception:
        return None
    x1, x2 = min(max(0, x1), width), min(max(0, x2), width)
    y1, y2 = min(max(0, y1), height), min(max(0, y2), height)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def load_upstream_hand_data(video_path: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    path = upstream_hand_screening_path(video_path)
    frames: Dict[int, Dict[str, Any]] = {}
    if not os.path.exists(path):
        return {"path": path, "available": False, "schema": None, "frames": frames}
    data = read_json(path)
    width, height = int(meta["width"]), int(meta["height"])
    for raw in data.get("frames", []) if isinstance(data, dict) else []:
        if not isinstance(raw, dict):
            continue
        try:
            frame_id = int(raw.get("frame_index"))
        except Exception:
            continue
        if not (0 <= frame_id < int(meta["total_frames"])):
            continue
        item: Dict[str, Any] = {"keep": bool(raw.get("keep")), "hands": {}}
        for upstream_key, public_key in (("left_hand", "left_hand"), ("right_hand", "right_hand")):
            hand = raw.get(upstream_key)
            if not isinstance(hand, dict):
                hand = {}
            bbox = _valid_bbox(hand.get("bbox_xyxy"), width=width, height=height)
            detected = bool(hand.get("detected")) and bbox is not None
            item["hands"][public_key] = {
                "detected": detected,
                "bbox_xyxy": bbox,
                "depth_mm": hand.get("depth_mm"),
                "depth_status": hand.get("depth_status"),
            }
        frames[frame_id] = item
    return {
        "path": path,
        "available": True,
        "schema": data.get("schema") if isinstance(data, dict) else None,
        "frames": frames,
    }


def _ranges_from_frame_ids(frame_ids: Iterable[int], *, max_gap: int = 6) -> List[Tuple[int, int]]:
    ids = sorted(set(int(x) for x in frame_ids))
    if not ids:
        return []
    out: List[Tuple[int, int]] = []
    start = prev = ids[0]
    for frame_id in ids[1:]:
        if frame_id - prev <= max_gap + 1:
            prev = frame_id
            continue
        out.append((start, prev))
        start = prev = frame_id
    out.append((start, prev))
    return out


def _subtract_ranges(base: Sequence[Tuple[int, int]], masks: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    result: List[Tuple[int, int]] = []
    for start, end in base:
        pieces = [(start, end)]
        for mask_start, mask_end in masks:
            next_pieces: List[Tuple[int, int]] = []
            for a, b in pieces:
                if mask_end < a or mask_start > b:
                    next_pieces.append((a, b))
                    continue
                if a < mask_start:
                    next_pieces.append((a, mask_start - 1))
                if mask_end < b:
                    next_pieces.append((mask_end + 1, b))
            pieces = next_pieces
        result.extend((a, b) for a, b in pieces if a <= b)
    return result


def _visible_hand_state(upstream: Dict[str, Any], frame_id: int) -> Tuple[str, ...]:
    frame = upstream.get("frames", {}).get(frame_id, {})
    hands = frame.get("hands", {}) if isinstance(frame, dict) else {}
    return tuple(
        hand_name
        for hand_name in ("left_hand", "right_hand")
        if isinstance(hands, dict) and bool(hands.get(hand_name, {}).get("detected"))
    )


def _bbox_center(upstream: Dict[str, Any], frame_id: int, hand_name: str) -> Optional[Tuple[float, float]]:
    frame = upstream.get("frames", {}).get(frame_id, {})
    hands = frame.get("hands", {}) if isinstance(frame, dict) else {}
    bbox = hands.get(hand_name, {}).get("bbox_xyxy") if isinstance(hands, dict) else None
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    return ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)


def _split_upstream_ranges_by_bbox_state(
    ranges: Sequence[Tuple[int, int]],
    *,
    upstream: Dict[str, Any],
    width: int,
    height: int,
) -> List[Tuple[int, int]]:
    """Split upstream tracks on hand-state or major workspace discontinuity.

    This is deliberately conservative: ordinary within-action movement stays in
    one range, while a left/right participation change or a >25% frame-diagonal
    center jump establishes a new bbox-defined candidate hand_step.
    """
    diagonal = max(1.0, (float(width) ** 2 + float(height) ** 2) ** 0.5)
    out: List[Tuple[int, int]] = []
    for start, end in ranges:
        part_start = start
        previous_frame = start
        previous_state = _visible_hand_state(upstream, start)
        for frame_id in range(start + 1, end + 1):
            state = _visible_hand_state(upstream, frame_id)
            # Do not split merely because one hand detector flickers on/off;
            # semantic left/right participation changes are supplied by Stage 1.
            # Bbox-only fallback boundaries are reserved for a major workspace
            # discontinuity.
            split = False
            if not split:
                for hand_name in set(state).intersection(previous_state):
                    prev_center = _bbox_center(upstream, previous_frame, hand_name)
                    center = _bbox_center(upstream, frame_id, hand_name)
                    if prev_center is None or center is None:
                        continue
                    displacement = ((center[0] - prev_center[0]) ** 2 + (center[1] - prev_center[1]) ** 2) ** 0.5
                    if displacement / diagonal > 0.25:
                        split = True
                        break
            if split:
                out.append((part_start, frame_id - 1))
                part_start = frame_id
            if state:
                previous_state = state
                previous_frame = frame_id
        out.append((part_start, end))
    return [(a, b) for a, b in out if a <= b]


def _bbox_motion_frame_ids(
    upstream: Dict[str, Any],
    *,
    width: int,
    height: int,
    center_threshold_diagonal: float = 0.015,
    area_change_threshold: float = 0.25,
    dilation_frames: int = 3,
) -> List[int]:
    """Return frames with bbox-level evidence of hand motion.

    Detector presence alone is deliberately insufficient.  A frame becomes a
    bbox-action candidate when the same hand's center moves by at least 1.5% of
    the image diagonal or its bbox area changes by at least 25% versus the prior
    adjacent observation.  A small temporal dilation keeps action onset/end.
    Fine finger motion with a stable bbox is expected to be recovered by the
    independent Stage-1 VLM event stream.
    """
    diagonal = max(1.0, (float(width) ** 2 + float(height) ** 2) ** 0.5)
    frame_map = upstream.get("frames", {})
    active: set[int] = set()
    for hand_name in ("left_hand", "right_hand"):
        previous: Optional[Tuple[int, List[int]]] = None
        for frame_id in sorted(int(x) for x in frame_map.keys()):
            frame = frame_map.get(frame_id, {})
            hands = frame.get("hands", {}) if isinstance(frame, dict) else {}
            bbox = hands.get(hand_name, {}).get("bbox_xyxy") if isinstance(hands, dict) else None
            if not isinstance(bbox, list) or len(bbox) != 4:
                previous = None
                continue
            bbox = [int(v) for v in bbox]
            if previous is not None and frame_id - previous[0] <= 2:
                prev_id, prev_bbox = previous
                prev_center = ((prev_bbox[0] + prev_bbox[2]) / 2.0, (prev_bbox[1] + prev_bbox[3]) / 2.0)
                center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
                displacement = ((center[0] - prev_center[0]) ** 2 + (center[1] - prev_center[1]) ** 2) ** 0.5
                prev_area = max(1, (prev_bbox[2] - prev_bbox[0]) * (prev_bbox[3] - prev_bbox[1]))
                area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                area_change = abs(float(area) - float(prev_area)) / float(max(area, prev_area))
                if displacement / diagonal >= center_threshold_diagonal or area_change >= area_change_threshold:
                    active.add(prev_id)
                    active.add(frame_id)
            previous = (frame_id, bbox)
    if dilation_frames > 0 and active:
        total = max(frame_map.keys(), default=-1) + 1
        expanded: set[int] = set()
        for frame_id in active:
            expanded.update(range(max(0, frame_id - dilation_frames), min(total, frame_id + dilation_frames + 1)))
        active = expanded
    return sorted(active)


def _bbox_tracks_for_range(upstream: Dict[str, Any], start: int, end: int) -> Dict[str, List[Dict[str, Any]]]:
    tracks: Dict[str, List[Dict[str, Any]]] = {"left_hand": [], "right_hand": []}
    frame_map = upstream.get("frames", {})
    for frame_id in range(start, end + 1):
        frame = frame_map.get(frame_id, {}) if isinstance(frame_map, dict) else {}
        hands = frame.get("hands", {}) if isinstance(frame, dict) else {}
        for hand_name in ("left_hand", "right_hand"):
            hand = hands.get(hand_name, {}) if isinstance(hands, dict) else {}
            bbox = hand.get("bbox_xyxy") if isinstance(hand, dict) else None
            if bbox:
                tracks[hand_name].append(
                    {
                        "frame_id": frame_id,
                        "bbox_xyxy": bbox,
                        "bbox_source": "upstream_sam3",
                    }
                )
    return tracks


def _vlm_bbox_tracks_for_range(vlm_events: Dict[str, Any], start: int, end: int) -> Dict[str, List[Dict[str, Any]]]:
    tracks: Dict[str, List[Dict[str, Any]]] = {"left_hand": [], "right_hand": []}
    for frame in vlm_events.get("frame_annotations", []) if isinstance(vlm_events, dict) else []:
        if not isinstance(frame, dict):
            continue
        frame_id = int(frame.get("frame_id", -1))
        if not (start <= frame_id <= end):
            continue
        for bbox in frame.get("vlm_generated_bboxes", []) if isinstance(frame.get("vlm_generated_bboxes"), list) else []:
            hand = str(bbox.get("hand") or "") if isinstance(bbox, dict) else ""
            if hand not in tracks:
                continue
            tracks[hand].append({
                "frame_id": frame_id,
                "bbox_xyxy": list(bbox["bbox_xyxy"]),
                "bbox_source": "vlm_generated",
                "arm_visible": int(bbox.get("arm_visible", 0)),
                "hand_completeness_score": _clamp_float(bbox.get("hand_completeness_score")),
            })
    return tracks


def build_hand_step_boundaries(
    *,
    video_path: str,
    vlm_events: Dict[str, Any],
    upstream: Dict[str, Any],
    meta: Dict[str, Any],
    max_gap_frames: int = 6,
    min_valid_keep_frames: int = MIN_VALID_KEEP_FRAMES,
    merge_gap_frames_exclusive: int = MERGE_GAP_FRAMES_EXCLUSIVE,
) -> Dict[str, Any]:
    """Build steps from valid keep runs, bridging only short total gaps.

    A strictly-contiguous keep=true run is valid when it contains at least
    ``min_valid_keep_frames`` frames.  After invalid runs are discarded as
    standalone actions, two consecutive valid runs are merged when the total
    number of source frames between them is strictly smaller than
    ``merge_gap_frames_exclusive``.  Every intervening frame is retained in the
    merged step, including keep=false frames and discarded short true runs.
    """
    del max_gap_frames  # retained only for backward-compatible callers
    if min_valid_keep_frames <= 0:
        raise ValueError("min_valid_keep_frames must be positive")
    if merge_gap_frames_exclusive <= 0:
        raise ValueError("merge_gap_frames_exclusive must be positive")
    total_frames = int(meta["total_frames"])
    keep_ids: List[int] = []
    left_ids: List[int] = []
    right_ids: List[int] = []
    for frame_id, frame in upstream.get("frames", {}).items():
        hands = frame.get("hands", {}) if isinstance(frame, dict) else {}
        left = bool(hands.get("left_hand", {}).get("detected")) if isinstance(hands, dict) else False
        right = bool(hands.get("right_hand", {}).get("detected")) if isinstance(hands, dict) else False
        keep = bool(frame.get("keep")) if isinstance(frame, dict) else False
        if keep:
            keep_ids.append(int(frame_id))
        if left:
            left_ids.append(int(frame_id))
        if right:
            right_ids.append(int(frame_id))
    raw_keep_ranges = _ranges_from_frame_ids(keep_ids, max_gap=0)
    valid_keep_ranges = [
        (start, end)
        for start, end in raw_keep_ranges
        if end - start + 1 >= min_valid_keep_frames
    ]
    merged_keep_groups: List[Dict[str, Any]] = []
    for start, end in valid_keep_ranges:
        if (
            merged_keep_groups
            and start - int(merged_keep_groups[-1]["end_frame_id"]) - 1 < merge_gap_frames_exclusive
        ):
            merged_keep_groups[-1]["end_frame_id"] = end
            merged_keep_groups[-1]["valid_action_ranges"].append((start, end))
        else:
            merged_keep_groups.append({
                "start_frame_id": start,
                "end_frame_id": end,
                "valid_action_ranges": [(start, end)],
            })
    merged_keep_ranges = [
        (int(group["start_frame_id"]), int(group["end_frame_id"]))
        for group in merged_keep_groups
    ]
    semantics_by_id = {
        int(item["step_id"]): item.get("action_semantics", {})
        for item in vlm_events.get("step_semantics", [])
        if isinstance(item, dict) and "step_id" in item
    }
    multiple_subjects_by_id = {
        int(item["step_id"]): bool(item.get("multiple_action_subjects", False))
        for item in vlm_events.get("step_semantics", [])
        if isinstance(item, dict) and "step_id" in item
    }

    hand_steps: List[Dict[str, Any]] = []
    for step_id, group in enumerate(merged_keep_groups, start=1):
        start = int(group["start_frame_id"])
        end = int(group["end_frame_id"])
        valid_action_ranges = list(group["valid_action_ranges"])
        tracks = _bbox_tracks_for_range(upstream, start, end)
        analysis_tracks, bbox_quality_annotations = _repair_temporarily_shrunken_bboxes(
            tracks, width=int(meta["width"]), height=int(meta["height"])
        )
        stage3_tracks: Dict[str, List[Dict[str, Any]]] = {"left_hand": [], "right_hand": []}
        for action_start, action_end in valid_action_ranges:
            action_tracks = _bbox_tracks_for_range(upstream, action_start, action_end)
            for hand_name in ("left_hand", "right_hand"):
                stage3_tracks[hand_name].extend(action_tracks[hand_name])
        active_hands = [hand_name for hand_name in ("left_hand", "right_hand") if tracks[hand_name]]
        hand_steps.append(
            {
                "hand_step_id": step_id,
                "start_frame_id": start,
                "end_frame_id": end,
                "end_frame_id_exclusive": end + 1,
                "frame_ids": list(range(start, end + 1)),
                "detection_sources": ["upstream_keep"],
                "active_hands": active_hands,
                "draft_action_semantics": semantics_by_id.get(step_id, {}),
                "multiple_action_subjects": multiple_subjects_by_id.get(step_id, False),
                "vlm_confidence": None,
                "bbox_tracks": tracks,
                "analysis_bbox_tracks": analysis_tracks,
                "bbox_quality_annotations": bbox_quality_annotations,
                "bbox_repair_summary": {
                    "window_frames": BBOX_SHRINK_WINDOW_FRAMES,
                    "width_ratio_threshold": BBOX_SHRINK_WIDTH_RATIO,
                    "height_ratio_threshold": BBOX_SHRINK_HEIGHT_RATIO,
                    "area_ratio_threshold": BBOX_SHRINK_AREA_RATIO,
                    "analysis_bbox_padding_ratio_x": ANALYSIS_BBOX_PADDING_RATIO_X,
                    "analysis_bbox_padding_ratio_y": ANALYSIS_BBOX_PADDING_RATIO_Y,
                    "analysis_bbox_min_padding_px": ANALYSIS_BBOX_MIN_PADDING_PX,
                    "repaired_frame_count": len(bbox_quality_annotations),
                    "policy": (
                        "source bbox is immutable; analysis bbox is inference-only; every analysis bbox receives "
                        "symmetric glove-margin padding after optional temporal repair"
                    ),
                },
                "valid_action_intervals": [
                    {
                        "action_run_id": action_id,
                        "start_frame_id": action_start,
                        "end_frame_id": action_end,
                    }
                    for action_id, (action_start, action_end) in enumerate(valid_action_ranges, start=1)
                ],
                "stage3_bbox_tracks": stage3_tracks,
            }
        )

    return {
        "schema": HAND_BOUNDARIES_SCHEMA,
        "video_id": video_id_from_path(video_path),
        "source_video": build_source_video_meta(video_path),
        "fps": float(meta["fps"]),
        "total_frames": total_frames,
        "frame_index_base": 0,
        "endpoint_semantics": "inclusive",
        "fusion_rule": "filtered_upstream_keep_true_runs_with_short_total_gap_bridging",
        "hand_step_boundary_rule": (
            f"keep=true runs of at least {min_valid_keep_frames} frames are valid "
            f"actions; consecutive valid actions are one step when all intervening "
            f"frames total fewer than {merge_gap_frames_exclusive}"
        ),
        "minimum_valid_keep_frames": min_valid_keep_frames,
        "merge_gap_frames_exclusive": merge_gap_frames_exclusive,
        "upstream": {
            "available": bool(upstream.get("available")),
            "schema": upstream.get("schema"),
            "source_json": upstream.get("path"),
            "keep_frame_count": len(set(keep_ids)),
            "left_detected_frame_count": len(set(left_ids)),
            "right_detected_frame_count": len(set(right_ids)),
        },
        "candidate_frame_union": {
            "upstream_keep_ranges": [
                {"start_frame_id": a, "end_frame_id": b} for a, b in raw_keep_ranges
            ],
            "valid_keep_ranges": [
                {"start_frame_id": a, "end_frame_id": b} for a, b in valid_keep_ranges
            ],
            "merged_step_ranges": [
                {"start_frame_id": a, "end_frame_id": b} for a, b in merged_keep_ranges
            ],
        },
        "hand_steps": hand_steps,
    }


def _generate_stage2_step_objects(
    *, video_path: str, boundaries: Dict[str, Any], api_cfg: Any, max_retries: int,
    stage2_dir: str,
) -> Dict[int, Dict[str, Any]]:
    # Stage 1's single model request already generated the Stage-2 payload.
    # Stage 2 is now a validation/assembly phase and must never upload video.
    cached_results: Dict[int, Dict[str, Any]] = {}
    for boundary in boundaries.get("hand_steps", []):
        step_id = int(boundary["hand_step_id"])
        inventory_path = os.path.join(stage2_dir, f"step_{step_id:03d}", "object_inventory.json")
        if not os.path.exists(inventory_path):
            raise FileNotFoundError(
                f"Single-pass Stage-2 result is missing for step_id={step_id}: {inventory_path}. "
                "Rerun Stage 1 for this video."
            )
        result = read_json(inventory_path)
        if int(result.get("step_id", -1)) != step_id:
            raise RuntimeError(f"Stage-2 cached step_id mismatch: {inventory_path}")
        if not isinstance(result.get("hand_object_relations"), list) or not isinstance(result.get("object_inventory"), list):
            raise RuntimeError(f"Invalid single-pass Stage-2 result: {inventory_path}")
        cached_results[step_id] = {
            "step_id": step_id,
            "hand_object_relations": result["hand_object_relations"],
            "object_inventory": result["object_inventory"],
        }
    return cached_results

    # Legacy multi-call implementation retained below for old source-level
    # reference only; it is unreachable from the active pipeline.
    client = initialize_api_client(api_cfg)
    if not client:
        raise SystemExit("Failed to initialize API client for Stage 2 object mapping.")
    system_prompt = (
        "Inspect the attached Stage-1 step-only video and its upstream bbox tracks. "
        "Return strict JSON. Build a step-local object map: direct hand-interaction objects first, then objects "
        "that move or change shape. Exclude unchanged background objects. Never alter frame IDs or bboxes."
    )
    results: Dict[int, Dict[str, Any]] = {}
    video_out = os.path.dirname(stage2_dir)
    for boundary in boundaries.get("hand_steps", []):
        step_id = int(boundary["hand_step_id"])
        step_dir = os.path.join(stage2_dir, f"step_{step_id:03d}")
        os.makedirs(step_dir, exist_ok=True)
        clip_path = _require_stage1_step_clip(video_out=video_out, step=boundary)
        video_content = build_video_api_content(
            clip_path, [], video_fps=api_cfg.video_fps, include_manifest=False
        )
        prompt = f"""
Analyze exactly step_id={step_id}, inclusive original source frames
[{int(boundary['start_frame_id'])}, {int(boundary['end_frame_id'])}].
{_step_clip_mapping_text(boundary)}

Authoritative upstream bbox tracks in original-frame IDs and original-image pixel coordinates:
{json.dumps(boundary.get('bbox_tracks', {}), ensure_ascii=False)}

Return exactly:
{{"step_id": {step_id}, "hand_object_relations": [{{"object":"...","hand":"left hand","relation":"...","entity_ref":"step_{step_id:03d}_obj_001"}}], "object_inventory": [{{"entity_ref":"step_{step_id:03d}_obj_001","canonical_name":"...","description":"...","selection_reason":"...","priority":"direct_hand_interaction"}}]}}
priority must be direct_hand_interaction or moved_or_shape_changed.
""".strip()
        last_content = ""
        errors: List[str] = []
        result: Optional[Dict[str, Any]] = None
        for attempt in range(1, max(1, int(max_retries)) + 1):
            prefix = build_retry_prefix(errors, last_content) if attempt > 1 else ""
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": [{"type": "text", "text": prefix + prompt}] + video_content}]
            last_content, _usage = call_chat_completion(client, api_cfg, messages, max_tokens=min(api_cfg.max_tokens, 5000))
            try:
                parsed = json.loads(extract_json_from_response(last_content))
                if not isinstance(parsed, dict) or set(parsed.keys()) != {"step_id", "hand_object_relations", "object_inventory"}:
                    raise ValueError("Invalid Stage 2 object-map keys.")
                wrapped = {"steps": [{
                    **parsed,
                    "action_semantics": {"left_hand": "not active", "right_hand": "not active", "overall": "not active"},
                    "arm_visibility_intervals": {"left_hand": [], "right_hand": []},
                }]}
                normalized, errors = _normalize_whole_video_step_review(wrapped, boundaries={"hand_steps": [boundary]})
                if normalized:
                    result = {
                        "step_id": step_id,
                        "hand_object_relations": normalized[0]["hand_object_relations"],
                        "object_inventory": normalized[0]["object_inventory"],
                    }
            except Exception as exc:
                result, errors = None, [f"JSON/validation error: {exc}"]
            if result is not None:
                break
        write_text(os.path.join(step_dir, "object_mapping_system_prompt.txt"), system_prompt)
        write_text(os.path.join(step_dir, "object_mapping_user_prompt.txt"), prompt)
        write_text(os.path.join(step_dir, "object_mapping_raw_response.txt"), last_content)
        if result is None:
            raise RuntimeError(f"Stage 2 step_id={step_id} object mapping failed: " + " | ".join(errors[:10]))
        write_json(os.path.join(step_dir, "object_inventory.json"), result)
        results[step_id] = result
    return results

def run_stage2_text_only(
    *,
    video_path: str,
    output_root: str,
    overwrite: bool,
    api_cfg: Any = None,
    max_retries: int = 1,
) -> str:
    """Generate Stage-2 JSON boundaries without cutting or writing any media."""
    started = time.perf_counter()
    vid = video_id_from_path(video_path)
    video_out = os.path.join(output_root, vid)
    ensure_video_out_dir_safe(video_out, video_path)
    stage1_dir = os.path.join(video_out, "stage1")
    stage2_dir = os.path.join(video_out, "stage2")
    events_path = os.path.join(stage1_dir, "vlm_hand_events.json")
    boundaries_path = os.path.join(stage2_dir, "hand_step_boundaries.json")
    segments_path = os.path.join(stage2_dir, "step_segments.json")
    localization_path = os.path.join(stage2_dir, "localization_raw.json")
    raw_path = os.path.join(stage2_dir, "stage2_raw_response.txt")
    system_path = os.path.join(stage2_dir, "stage2_system_prompt.txt")
    user_path = os.path.join(stage2_dir, "stage2_user_prompt.txt")
    run_summary_path = os.path.join(video_out, "run_summary.json")
    if not overwrite and os.path.exists(boundaries_path) and os.path.exists(segments_path):
        cached_boundaries = read_json(boundaries_path)
        cached_steps = cached_boundaries.get("hand_steps", [])
        inventories_complete = isinstance(cached_steps, list) and all(
            os.path.exists(os.path.join(
                stage2_dir, f"step_{int(step.get('hand_step_id', 0)):03d}", "object_inventory.json"
            ))
            for step in cached_steps if isinstance(step, dict)
        )
        if inventories_complete:
            logger.info(f"[stage2] video_id={vid} resume complete per-step object inventories")
            return video_out
        logger.warning(
            f"[stage2] video_id={vid} cached boundaries are incomplete; regenerating per-step object inventories"
        )
    meta = get_video_metadata(video_path)
    vlm_events = read_json(events_path) if os.path.exists(events_path) else {}
    upstream = load_upstream_hand_data(video_path, meta)
    boundaries = build_hand_step_boundaries(
        video_path=video_path,
        vlm_events=vlm_events,
        upstream=upstream,
        meta=meta,
    )
    if api_cfg is None:
        raise RuntimeError("Stage 2 object mapping requires api_cfg.")
    object_maps = _generate_stage2_step_objects(
        video_path=video_path,
        boundaries=boundaries,
        api_cfg=api_cfg,
        max_retries=max_retries,
        stage2_dir=stage2_dir,
    )
    for step in boundaries.get("hand_steps", []):
        mapping = object_maps[int(step["hand_step_id"])]
        step["hand_object_relations"] = mapping["hand_object_relations"]
        step["object_inventory"] = mapping["object_inventory"]
    write_json(localization_path, boundaries.get("candidate_frame_union", {}))
    write_text(raw_path, json.dumps(boundaries.get("candidate_frame_union", {}), ensure_ascii=False, indent=2))
    write_text(system_path, "Deterministic text-only segmentation from strictly contiguous upstream keep=true frames; no media is cut.")
    write_text(user_path, "Each uninterrupted keep=true run is one complete hand_step. Do not split or merge runs.")

    segments: List[Dict[str, Any]] = []
    for step in boundaries.get("hand_steps", []):
        start = int(step["start_frame_id"])
        end = int(step["end_frame_id"])
        clip_path = _require_stage1_step_clip(video_out=video_out, step=step)
        clip_relpath = os.path.relpath(clip_path, video_out)
        step["clip_materialized"] = True
        step["clip_path"] = clip_relpath
        step["clip_relpath"] = clip_relpath
        step["source_frame_id_offset"] = start
        segments.append(
            {
                "step_id": int(step["hand_step_id"]),
                "step_goal": str(step.get("draft_action_semantics") or "Hand action pending Stage 3 ROI review."),
                "value": "yes",
                "start_frame_0based": start,
                "end_frame_0based_exclusive": end + 1,
                "end_frame_0based_inclusive": end,
                "start_sec": float(start) / float(meta["fps"]),
                "end_sec": float(end + 1) / float(meta["fps"]),
                "clip_materialized": True,
                "clip_path": clip_relpath,
                "clip_relpath": clip_relpath,
                "clip_local_start_frame_id": 0,
                "source_frame_id_offset": start,
                "bbox_tracks": step.get("bbox_tracks", {}),
                "analysis_bbox_tracks": step.get("analysis_bbox_tracks", {}),
                "bbox_quality_annotations": step.get("bbox_quality_annotations", []),
                "bbox_repair_summary": step.get("bbox_repair_summary", {}),
                "detection_sources": step.get("detection_sources", []),
            }
        )
    write_json(boundaries_path, boundaries)
    write_json(
        segments_path,
        {
            "source_video": build_source_video_meta(video_path),
            "source_video_fps": float(meta["fps"]),
            "source_total_frames": int(meta["total_frames"]),
            "source_video_duration_sec": float(meta["duration_sec"]),
            "cut": {"materialized": True, "producer_stage": 1, "mode": "reencode"},
            "segments": segments,
        },
    )
    update_run_summary(
        run_summary_path,
        {
            "source_video": build_source_video_meta(video_path),
            "updated_at_utc": now_utc_iso(),
            "stage2": {
                "status": "completed",
                "mode": "text_only_hand_steps",
                "clips_materialized": True,
                "clips_producer_stage": 1,
                "hand_step_boundaries_path": os.path.relpath(boundaries_path, video_out),
                "segments_path": os.path.relpath(segments_path, video_out),
            },
        },
    )
    logger.info(
        f"[stage2] video_id={vid} completed text-only hand_steps={len(segments)} "
        f"elapsed={format_duration(time.perf_counter() - started)}"
    )
    return video_out


def _representative_frame_ids(start: int, end: int, *, limit: int = 7) -> List[int]:
    if end <= start:
        return [start]
    count = min(limit, end - start + 1)
    if count <= 1:
        return [start]
    return sorted(set(int(round(start + (end - start) * i / (count - 1))) for i in range(count)))


def _track_lookup(step: Dict[str, Any], hand_name: str) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    tracks = step.get("bbox_tracks", {})
    records = tracks.get(hand_name, []) if isinstance(tracks, dict) else []
    for record in records if isinstance(records, list) else []:
        if isinstance(record, dict) and isinstance(record.get("bbox_xyxy"), list):
            out[int(record.get("frame_id"))] = [int(x) for x in record["bbox_xyxy"]]
    return out


def _expand_bbox(bbox: List[int], width: int, height: int, factor: float = 1.5) -> List[int]:
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = max(2.0, (x2 - x1) * factor), max(2.0, (y2 - y1) * factor)
    return [
        max(0, int(round(cx - bw / 2))),
        max(0, int(round(cy - bh / 2))),
        min(width, int(round(cx + bw / 2))),
        min(height, int(round(cy + bh / 2))),
    ]


def _encode_crop(frame: Any, bbox: List[int]) -> Optional[str]:
    if cv2 is None:
        return None
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    if crop is None or getattr(crop, "size", 0) <= 0:
        return None
    ok, buffer = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return None
    return base64.b64encode(buffer.tobytes()).decode("ascii")


BBOX_VISIBILITY_SYSTEM_PROMPT = """
You evaluate one exact hand bbox crop at a time. Return strict JSON only. Judge
only pixels inside each supplied bbox crop. arm_visible is 1 only when a
connected wrist or forearm is included. hand_completeness_score is a real
number in [0,1], from absent/unrecognizable to fully exposed hand with major
fingers, palm/back, and wrist. Do not transfer a score between bbox sources.
""".strip()


def _normalize_bbox_evaluations(
    obj: Any, *, expected: Sequence[Dict[str, Any]]
) -> Tuple[Optional[List[Dict[str, Any]]], List[str]]:
    if not isinstance(obj, dict) or set(obj.keys()) != {"evaluations"} or not isinstance(obj.get("evaluations"), list):
        return None, ["bbox evaluation response must contain exactly evaluations."]
    expected_keys = [
        (int(x["frame_id"]), str(x["hand"]), str(x["bbox_source"])) for x in expected
    ]
    out: List[Dict[str, Any]] = []
    errors: List[str] = []
    for index, raw in enumerate(obj["evaluations"]):
        if not isinstance(raw, dict) or set(raw.keys()) != {
            "frame_id", "hand", "bbox_source", "arm_visible", "hand_completeness_score"
        }:
            errors.append(f"evaluations[{index}] has invalid keys.")
            continue
        try:
            arm_visible = int(raw["arm_visible"])
            key = (int(raw["frame_id"]), str(raw["hand"]), str(raw["bbox_source"]))
        except Exception:
            errors.append(f"evaluations[{index}] has invalid identifiers.")
            continue
        if arm_visible not in {0, 1}:
            errors.append(f"evaluations[{index}].arm_visible must be 0 or 1.")
        out.append({
            "frame_id": key[0], "hand": key[1], "bbox_source": key[2],
            "bbox_xyxy": list(expected[index]["bbox_xyxy"]) if index < len(expected) else None,
            "arm_visible": arm_visible,
            "hand_completeness_score": _clamp_float(raw.get("hand_completeness_score")),
        })
    actual_keys = [(x["frame_id"], x["hand"], x["bbox_source"]) for x in out]
    if actual_keys != expected_keys:
        errors.append(f"bbox evaluation identifiers/order mismatch: expected={expected_keys}, got={actual_keys}.")
    return (None, errors) if errors else (out, [])


def evaluate_all_bbox_tracks(
    *, video_path: str, tracks: Dict[str, List[Dict[str, Any]]], client: Any,
    api_cfg: Any, max_retries: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """VLM-evaluate every VLM and upstream bbox independently, in bounded batches."""
    if cv2 is None:
        raise RuntimeError("opencv-python is required for bbox visibility evaluation.")
    records: List[Dict[str, Any]] = []
    for hand in ("left_hand", "right_hand"):
        for raw in tracks.get(hand, []):
            if str(raw.get("bbox_source")) not in {"vlm_generated", "upstream_sam3"}:
                continue
            records.append({
                "frame_id": int(raw["frame_id"]), "hand": hand,
                "bbox_source": str(raw["bbox_source"]), "bbox_xyxy": list(raw["bbox_xyxy"]),
            })
    records.sort(key=lambda x: (x["frame_id"], x["hand"], x["bbox_source"]))
    batch_size = max(1, int(os.environ.get("TEXT_LABEL_BBOX_BATCH_SIZE", "12")))
    usage_total = _usage_zero()
    results: List[Dict[str, Any]] = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for bbox evaluation: {video_path}")
    try:
        for offset in range(0, len(records), batch_size):
            batch = records[offset : offset + batch_size]
            content: List[Dict[str, Any]] = []
            for record in batch:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(record["frame_id"]))
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Cannot read bbox frame {record['frame_id']}.")
                encoded = _encode_crop(frame, record["bbox_xyxy"])
                if encoded is None:
                    raise RuntimeError(f"Cannot encode bbox crop for {record}.")
                content.extend([
                    {"type": "text", "text": (
                        f"Evaluate frame_id={record['frame_id']} hand={record['hand']} "
                        f"bbox_source={record['bbox_source']} bbox_xyxy={record['bbox_xyxy']}."
                    )},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                ])
            prompt = (
                "Return exactly {\"evaluations\":[...]} with one record in the supplied order. "
                "Each record has exactly frame_id, hand, bbox_source, arm_visible, "
                "hand_completeness_score. Preserve identifiers exactly."
            )
            last_content = ""
            errors: List[str] = []
            normalized: Optional[List[Dict[str, Any]]] = None
            for attempt in range(1, max(1, int(max_retries)) + 1):
                prefix = build_retry_prefix(errors, last_content) if attempt > 1 else ""
                messages = [
                    {"role": "system", "content": BBOX_VISIBILITY_SYSTEM_PROMPT},
                    {"role": "user", "content": [{"type": "text", "text": prefix + prompt}] + content},
                ]
                last_content, usage = call_chat_completion(
                    client, api_cfg, messages, max_tokens=min(api_cfg.max_tokens, max(3000, len(batch) * 300))
                )
                _add_usage(usage_total, usage)
                try:
                    parsed = json.loads(extract_json_from_response(last_content))
                    normalized, errors = _normalize_bbox_evaluations(parsed, expected=batch)
                except Exception as exc:
                    normalized, errors = None, [f"JSON parse error: {exc}"]
                if normalized is not None:
                    break
            if normalized is None:
                raise RuntimeError("BBox visibility batch failed: " + " | ".join(errors[:10]))
            results.extend(normalized)
    finally:
        cap.release()
    return results, usage_total


def build_hand_step_roi_content(
    *,
    video_path: str,
    step: Dict[str, Any],
    meta: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return API image content plus a text-only representative manifest."""
    if cv2 is None:
        raise RuntimeError("opencv-python is required for in-memory hand ROI extraction.")
    start, end = int(step["start_frame_id"]), int(step["end_frame_id"])
    frame_ids = _representative_frame_ids(start, end)
    left_lookup = _track_lookup(step, "left_hand")
    right_lookup = _track_lookup(step, "right_hand")
    width, height = int(meta["width"]), int(meta["height"])
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for ROI extraction: {video_path}")
    content: List[Dict[str, Any]] = []
    manifest: List[Dict[str, Any]] = []
    try:
        for frame_id in frame_ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            left_bbox = left_lookup.get(frame_id)
            right_bbox = right_lookup.get(frame_id)
            manifest_record: Dict[str, Any] = {
                "frame_id": frame_id,
                "left_hand_bbox_xyxy": left_bbox,
                "right_hand_bbox_xyxy": right_bbox,
            }
            # Keep the complete frame only as semantic context. Authoritative
            # VLM bboxes are generated on the upstream-centered selected frame
            # set in Stage 1; this full frame is semantic context only.
            full_b64 = _encode_crop(frame, [0, 0, width, height])
            content.append({"type": "text", "text": f"Representative source frame {frame_id} (0-based)."})
            if full_b64:
                content.extend(
                    [
                        {
                            "type": "text",
                            "text": (
                                "Complete original frame for hand-action semantic context only."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{full_b64}"}},
                    ]
                )
            bboxes = [b for b in (left_bbox, right_bbox) if b]
            if bboxes:
                union = [
                    min(b[0] for b in bboxes),
                    min(b[1] for b in bboxes),
                    max(b[2] for b in bboxes),
                    max(b[3] for b in bboxes),
                ]
                semantic_bbox = _expand_bbox(union, width, height, 1.5)
                semantic_b64 = _encode_crop(frame, semantic_bbox)
                manifest_record["semantic_roi_xyxy"] = semantic_bbox
                if semantic_b64:
                    content.extend(
                        [
                            {"type": "text", "text": "Expanded 1.5x semantic ROI for hand-action interpretation."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{semantic_b64}"}},
                        ]
                    )
                for hand_name, bbox in (("left_hand", left_bbox), ("right_hand", right_bbox)):
                    if not bbox:
                        continue
                    exact_b64 = _encode_crop(frame, bbox)
                    if exact_b64:
                        content.extend(
                            [
                                {"type": "text", "text": f"Exact upstream {hand_name} bbox; use ONLY this crop for arm visibility and completeness."},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{exact_b64}"}},
                            ]
                        )
            else:
                manifest_record["upstream_bbox_missing"] = True
            manifest.append(manifest_record)
    finally:
        cap.release()
    if not content:
        raise RuntimeError(f"No representative ROI evidence could be extracted for hand_step_id={step.get('hand_step_id')}")
    return content, manifest


HAND_STEP_REVIEW_SYSTEM_PROMPT = """
You are an expert hand-action ROI reviewer. Return one strict JSON object only.
Evaluate the supplied ROI sequence as ONE complete hand_step, never as unrelated
per-frame annotations. Expanded ROIs are only for action semantics. Arm presence
and hand-completeness scores must use the exact original bboxes. Ignore all
background-person and non-hand events.
""".strip()


def build_hand_step_review_prompt(step: Dict[str, Any], manifest: List[Dict[str, Any]], width: int, height: int) -> str:
    step_id = int(step["hand_step_id"])
    return f"""
Review hand_step_id={step_id}, inclusive source-frame interval
[{int(step['start_frame_id'])}, {int(step['end_frame_id'])}] in a {width}x{height}
video. The images form one chronological ROI sequence for this whole hand_step.

Draft VLM semantics: {step.get('draft_action_semantics')}
Representative bbox manifest:
{json.dumps(manifest, ensure_ascii=False)}

Tasks:
1. Describe the purposeful primary-operator action in professional English,
   separately as left_hand, right_hand, and overall. Use "not active" for a
   hand that does not meaningfully participate. Preserve simultaneous actions.
2. Return visible hand-object relations using object, hand, relation only.
3. For EVERY representative frame listed in the manifest, independently judge
   left_hand/right_hand arm_visible and hand completeness. arm_visible is 1
   only when a connected wrist/forearm is clearly visible IN THAT EXACT FRAME.
   Do not copy one frame's judgment to another frame.
4. Score each frame's completeness in [0,1]: 0 absent/unrecognizable; 0.25 fragment only;
   0.5 identifiable but substantially truncated/occluded; 0.75 mostly complete;
   1 fully visible hand, major fingers, palm/back, and wrist across the action.
5. Do not generate bboxes here. Stage 1 has already generated authoritative
   per-frame VLM bboxes and the separate bbox-evaluation pass scores both VLM
   and upstream SAM3 bboxes. Return fallback_bboxes as an empty list.

Return exactly:
{{
  "step_id": {step_id},
  "action_semantics": {{
    "left_hand": "...",
    "right_hand": "...",
    "overall": "..."
  }},
  "hand_object_relations": [
    {{"object": "...", "hand": "left hand", "relation": "holding"}}
  ],
  "visibility": {{
    "left_hand": {{"arm_visible": 0, "hand_completeness_score": 0.0}},
    "right_hand": {{"arm_visible": 0, "hand_completeness_score": 0.0}}
  }},
  "frame_visibility": [
    {{
      "frame_id": 0,
      "left_hand": {{"arm_visible": 0, "hand_completeness_score": 0.0}},
      "right_hand": {{"arm_visible": 0, "hand_completeness_score": 0.0}}
    }}
  ],
  "fallback_bboxes": [
    {{"frame_id": 0, "hand": "left_hand", "bbox_xyxy": [0, 0, 1, 1], "bbox_source": "vlm_grounding"}}
  ]
}}
No extra keys. frame_visibility must contain exactly one record for every
representative frame_id in the manifest, in chronological order. The top-level
visibility and frame_visibility are compatibility semantic-review summaries.
The per-source bbox_evaluations records are authoritative for the live interface.
fallback_bboxes must be empty.
""".strip()


def _normalize_review(
    obj: Any,
    *,
    step: Dict[str, Any],
    width: int,
    height: int,
    representative_frame_ids: Optional[Sequence[int]] = None,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    errors: List[str] = []
    required = {
        "step_id",
        "action_semantics",
        "hand_object_relations",
        "visibility",
        "frame_visibility",
        "fallback_bboxes",
    }
    if not isinstance(obj, dict) or set(obj.keys()) != required:
        return None, [f"Review output must contain exactly {sorted(required)}."]
    if int(obj.get("step_id", -1)) != int(step["hand_step_id"]):
        errors.append("step_id mismatch.")
    semantics = obj.get("action_semantics")
    if not isinstance(semantics, dict) or set(semantics.keys()) != {"left_hand", "right_hand", "overall"}:
        errors.append("action_semantics must contain exactly left_hand/right_hand/overall.")
        semantics = {"left_hand": "not active", "right_hand": "not active", "overall": "Hand action unclear."}
    else:
        semantics = {k: str(semantics.get(k) or "").strip() for k in ("left_hand", "right_hand", "overall")}
        if not all(semantics.values()):
            errors.append("action_semantics strings must be non-empty.")
    relations_out: List[Dict[str, Any]] = []
    relations = obj.get("hand_object_relations")
    if not isinstance(relations, list):
        errors.append("hand_object_relations must be a list.")
        relations = []
    for rel in relations:
        if not isinstance(rel, dict) or set(rel.keys()) != {"object", "hand", "relation"}:
            errors.append("Each hand_object_relation must contain exactly object/hand/relation.")
            continue
        item = {k: str(rel.get(k) or "").strip() for k in ("object", "hand", "relation")}
        if not all(item.values()) or item["hand"] not in {"left hand", "right hand", "both hands", "unknown hand"}:
            errors.append("Invalid hand_object_relation value.")
            continue
        relations_out.append(item)
    visibility = obj.get("visibility")
    normalized_visibility: Dict[str, Dict[str, Any]] = {}
    if not isinstance(visibility, dict) or set(visibility.keys()) != {"left_hand", "right_hand"}:
        errors.append("visibility must contain exactly left_hand/right_hand.")
        visibility = {}
    for hand_name in ("left_hand", "right_hand"):
        hand_vis = visibility.get(hand_name, {}) if isinstance(visibility, dict) else {}
        if not isinstance(hand_vis, dict) or set(hand_vis.keys()) != {"arm_visible", "hand_completeness_score"}:
            errors.append(f"visibility.{hand_name} has invalid keys.")
            hand_vis = {}
        try:
            arm_visible = int(hand_vis.get("arm_visible", 0))
        except Exception:
            arm_visible = -1
        if arm_visible not in {0, 1}:
            errors.append(f"visibility.{hand_name}.arm_visible must be 0 or 1.")
            arm_visible = 0
        normalized_visibility[hand_name] = {
            "arm_visible": arm_visible,
            "hand_completeness_score": _clamp_float(hand_vis.get("hand_completeness_score")),
        }

    expected_frame_ids = (
        sorted(set(int(x) for x in representative_frame_ids))
        if representative_frame_ids is not None
        else _representative_frame_ids(int(step["start_frame_id"]), int(step["end_frame_id"]))
    )
    frame_visibility_raw = obj.get("frame_visibility")
    if not isinstance(frame_visibility_raw, list):
        errors.append("frame_visibility must be a list.")
        frame_visibility_raw = []
    normalized_frame_visibility: List[Dict[str, Any]] = []
    seen_frame_ids: set[int] = set()
    for i, record in enumerate(frame_visibility_raw):
        if not isinstance(record, dict) or set(record.keys()) != {"frame_id", "left_hand", "right_hand"}:
            errors.append(
                f"frame_visibility[{i}] must contain exactly frame_id/left_hand/right_hand."
            )
            continue
        try:
            frame_id = int(record.get("frame_id"))
        except Exception:
            errors.append(f"frame_visibility[{i}].frame_id must be an integer.")
            continue
        if frame_id in seen_frame_ids:
            errors.append(f"frame_visibility has duplicate frame_id={frame_id}.")
            continue
        seen_frame_ids.add(frame_id)
        normalized_record: Dict[str, Any] = {"frame_id": frame_id}
        for hand_name in ("left_hand", "right_hand"):
            frame_hand = record.get(hand_name)
            if not isinstance(frame_hand, dict) or set(frame_hand.keys()) != {
                "arm_visible",
                "hand_completeness_score",
            }:
                errors.append(f"frame_visibility[{i}].{hand_name} has invalid keys.")
                frame_hand = {}
            try:
                arm_visible = int(frame_hand.get("arm_visible", 0))
            except Exception:
                arm_visible = -1
            if arm_visible not in {0, 1}:
                errors.append(f"frame_visibility[{i}].{hand_name}.arm_visible must be 0 or 1.")
                arm_visible = 0
            normalized_record[hand_name] = {
                "arm_visible": arm_visible,
                "hand_completeness_score": _clamp_float(
                    frame_hand.get("hand_completeness_score")
                ),
            }
        normalized_frame_visibility.append(normalized_record)
    actual_frame_ids = sorted(seen_frame_ids)
    if actual_frame_ids != expected_frame_ids:
        errors.append(
            "frame_visibility frame_ids must exactly match representative manifest: "
            f"expected={expected_frame_ids}, got={actual_frame_ids}."
        )
    normalized_frame_visibility.sort(key=lambda item: int(item["frame_id"]))

    # The per-frame records are authoritative. Recompute the legacy summary so
    # it can never disagree with what the live interface displays.
    if normalized_frame_visibility:
        for hand_name in ("left_hand", "right_hand"):
            frame_values = [item[hand_name] for item in normalized_frame_visibility]
            normalized_visibility[hand_name] = {
                "arm_visible": int(any(int(item["arm_visible"]) == 1 for item in frame_values)),
                "hand_completeness_score": float(
                    sum(float(item["hand_completeness_score"]) for item in frame_values)
                    / len(frame_values)
                ),
            }
    fallback_out: List[Dict[str, Any]] = []
    fallback = obj.get("fallback_bboxes")
    if not isinstance(fallback, list):
        errors.append("fallback_bboxes must be a list.")
        fallback = []
    if fallback:
        errors.append("fallback_bboxes must be empty; Stage 1 frame annotations are authoritative.")
    if errors:
        return None, errors
    active_hands = [x for x in step.get("active_hands", []) if x in {"left_hand", "right_hand"}]
    if not active_hands:
        active_hands = [h for h in ("left_hand", "right_hand") if semantics.get(h, "").lower() != "not active"]
    participating_scores = [normalized_visibility[h]["hand_completeness_score"] for h in active_hands]
    overall_score = min(participating_scores) if participating_scores else 0.0
    return {
        "step_id": int(step["hand_step_id"]),
        "action_semantics": semantics,
        "hand_object_relations": relations_out,
        "visibility": {
            **normalized_visibility,
            "overall_hand_completeness_score": float(overall_score),
            "arm_visible_any": int(any(normalized_visibility[h]["arm_visible"] for h in active_hands)),
            "aggregation": "mean_over_representative_frames_then_min_over_participating_hands",
        },
        "frame_visibility": normalized_frame_visibility,
        "fallback_bboxes": fallback_out,
    }, []


WHOLE_VIDEO_STEP_REVIEW_SYSTEM_PROMPT = """
You are an expert annotator of primary-operator hand actions. Inspect the entire
attached source video and return one strict JSON object only. Step boundaries
and every bbox are authoritative upstream facts and must never be changed.
Describe left hand, right hand, and overall action for each step. Locate
inclusive source-frame intervals where a connected wrist/forearm is visible
inside that hand's upstream bbox. Build a separate object mapping for each
step: prioritize objects directly interacting with a hand, then objects that
move or change shape; exclude objects that remain unchanged for the whole video.
""".strip()


def _normalize_step_identity_inventory(
    inventory_raw: Any, *, step_id: int, start_frame_id: int, end_frame_id: int
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Validate VLM identity evidence and derive gaps without inventing observations."""
    errors: List[str] = []
    if not isinstance(inventory_raw, list):
        return [], ["object_inventory must be a list."]
    out: List[Dict[str, Any]] = []
    refs: set[str] = set()
    allowed_priorities = {"direct_hand_interaction", "moved_or_shape_changed"}
    allowed_statuses = {"resolved", "uncertain", "new_instance"}

    def normalize_intervals(value: Any, label: str) -> List[Dict[str, int]]:
        normalized: List[Dict[str, int]] = []
        if not isinstance(value, list):
            errors.append(f"{label} must be a list.")
            return normalized
        previous_end = start_frame_id - 1
        for interval_index, interval in enumerate(value):
            if not isinstance(interval, dict) or set(interval) != {"start_frame_id", "end_frame_id"}:
                errors.append(f"{label}[{interval_index}] has invalid keys.")
                continue
            try:
                interval_start = int(interval["start_frame_id"])
                interval_end = int(interval["end_frame_id"])
            except Exception:
                errors.append(f"{label}[{interval_index}] has invalid frame IDs.")
                continue
            if not (start_frame_id <= interval_start <= interval_end <= end_frame_id):
                errors.append(f"{label}[{interval_index}] is outside step bounds.")
                continue
            if interval_start <= previous_end:
                errors.append(f"{label}[{interval_index}] overlaps or is not chronological.")
                continue
            previous_end = interval_end
            normalized.append({"start_frame_id": interval_start, "end_frame_id": interval_end})
        return normalized

    for object_index, item in enumerate(inventory_raw):
        label = f"object_inventory[{object_index}]"
        if not isinstance(item, dict):
            errors.append(f"{label} must be an object.")
            continue
        legacy_keys = {"entity_ref", "canonical_name", "description", "selection_reason", "priority"}
        required_identity_keys = {
            "aliases", "identity_profile", "visible_intervals", "representative_frame_ids",
            "hand_interactions", "state_transitions", "identity_evidence", "identity_status", "confidence",
        }
        unknown = set(item) - legacy_keys - required_identity_keys
        missing = legacy_keys - set(item)
        if unknown or missing:
            errors.append(f"{label} invalid keys; missing={sorted(missing)}, unknown={sorted(unknown)}.")
            continue
        ref = str(item.get("entity_ref") or "").strip()
        if not ref or ref in refs or not ref.startswith(f"step_{step_id:03d}_obj_"):
            errors.append(f"{label}.entity_ref is empty, duplicated, or outside this step namespace.")
            continue
        refs.add(ref)
        priority = str(item.get("priority") or "").strip()
        if priority not in allowed_priorities:
            errors.append(f"{label}.priority is invalid.")

        aliases_raw = item.get("aliases", [])
        if not isinstance(aliases_raw, list):
            errors.append(f"{label}.aliases must be a list.")
            aliases_raw = []
        aliases: List[str] = []
        seen_aliases: set[str] = set()
        for alias in aliases_raw:
            alias_text = str(alias or "").strip()
            alias_key = re.sub(r"\s+", " ", alias_text).casefold()
            if alias_text and alias_key not in seen_aliases:
                seen_aliases.add(alias_key)
                aliases.append(alias_text)

        profile_raw = item.get("identity_profile", {})
        if not isinstance(profile_raw, dict) or set(profile_raw) != {"stable_features", "mutable_features"}:
            errors.append(f"{label}.identity_profile has invalid keys.")
            profile_raw = {"stable_features": {}, "mutable_features": []}
        stable_features = profile_raw.get("stable_features", {})
        mutable_features = profile_raw.get("mutable_features", [])
        if not isinstance(stable_features, dict):
            errors.append(f"{label}.identity_profile.stable_features must be an object.")
            stable_features = {}
        if not isinstance(mutable_features, list):
            errors.append(f"{label}.identity_profile.mutable_features must be a list.")
            mutable_features = []

        visible = normalize_intervals(item.get("visible_intervals", []), f"{label}.visible_intervals")
        if not visible:
            errors.append(f"{label}.visible_intervals must contain video evidence.")
        missing_intervals = [
            {"start_frame_id": int(left["end_frame_id"]) + 1, "end_frame_id": int(right["start_frame_id"]) - 1}
            for left, right in zip(visible, visible[1:])
            if int(left["end_frame_id"]) + 1 <= int(right["start_frame_id"]) - 1
        ]

        representative_raw = item.get("representative_frame_ids", [])
        if not isinstance(representative_raw, list):
            errors.append(f"{label}.representative_frame_ids must be a list.")
            representative_raw = []
        representative: List[int] = []
        for frame in representative_raw:
            try:
                frame_id = int(frame)
            except Exception:
                errors.append(f"{label}.representative_frame_ids contains an invalid frame ID.")
                continue
            if not any(int(interval["start_frame_id"]) <= frame_id <= int(interval["end_frame_id"]) for interval in visible):
                errors.append(f"{label}.representative frame {frame_id} is not inside a visible interval.")
                continue
            if frame_id not in representative:
                representative.append(frame_id)

        interactions: List[Dict[str, Any]] = []
        interactions_raw = item.get("hand_interactions", [])
        if not isinstance(interactions_raw, list):
            errors.append(f"{label}.hand_interactions must be a list.")
            interactions_raw = []
        for interaction_index, interaction in enumerate(interactions_raw):
            expected_keys = {"hand", "action", "start_frame_id", "end_frame_id"}
            if not isinstance(interaction, dict) or set(interaction) != expected_keys:
                errors.append(f"{label}.hand_interactions[{interaction_index}] has invalid keys.")
                continue
            hand = str(interaction.get("hand") or "")
            try:
                interaction_start = int(interaction["start_frame_id"])
                interaction_end = int(interaction["end_frame_id"])
            except Exception:
                errors.append(f"{label}.hand_interactions[{interaction_index}] has invalid frame IDs.")
                continue
            if hand not in {"left_hand", "right_hand"} or not (
                start_frame_id <= interaction_start <= interaction_end <= end_frame_id
            ):
                errors.append(f"{label}.hand_interactions[{interaction_index}] is invalid/out of bounds.")
                continue
            interactions.append({
                "hand": hand, "action": str(interaction.get("action") or "").strip(),
                "start_frame_id": interaction_start, "end_frame_id": interaction_end,
            })

        transitions: List[Dict[str, Any]] = []
        transitions_raw = item.get("state_transitions", [])
        if not isinstance(transitions_raw, list):
            errors.append(f"{label}.state_transitions must be a list.")
            transitions_raw = []
        for transition_index, transition in enumerate(transitions_raw):
            expected_keys = {"frame_id", "from", "to", "cause"}
            if not isinstance(transition, dict) or set(transition) != expected_keys:
                errors.append(f"{label}.state_transitions[{transition_index}] has invalid keys.")
                continue
            try:
                frame_id = int(transition["frame_id"])
            except Exception:
                errors.append(f"{label}.state_transitions[{transition_index}] has an invalid frame ID.")
                continue
            if not start_frame_id <= frame_id <= end_frame_id:
                errors.append(f"{label}.state_transitions[{transition_index}] is outside step bounds.")
                continue
            transitions.append({
                "frame_id": frame_id, "from": str(transition.get("from") or "").strip(),
                "to": str(transition.get("to") or "").strip(),
                "cause": str(transition.get("cause") or "").strip(),
            })

        status = str(item.get("identity_status", "uncertain") or "uncertain").strip()
        if status not in allowed_statuses:
            errors.append(f"{label}.identity_status is invalid.")
            status = "uncertain"
        try:
            confidence = float(item.get("confidence", 0.0))
        except Exception:
            confidence = -1.0
        if not 0.0 <= confidence <= 1.0:
            errors.append(f"{label}.confidence must be in [0,1].")
            confidence = max(0.0, min(1.0, confidence))
        if len(visible) > 1 and not str(item.get("identity_evidence") or "").strip():
            errors.append(f"{label}.identity_evidence is required for a reacquired object.")

        out.append({
            "entity_ref": ref,
            "canonical_name": str(item.get("canonical_name") or "").strip(),
            "description": str(item.get("description") or "").strip(),
            "selection_reason": str(item.get("selection_reason") or "").strip(),
            "priority": priority,
            "aliases": aliases,
            "identity_profile": {
                "stable_features": stable_features,
                "mutable_features": [str(value).strip() for value in mutable_features if str(value).strip()],
            },
            "visible_intervals": visible,
            "temporarily_missing_intervals": missing_intervals,
            "representative_frame_ids": sorted(representative),
            "hand_interactions": interactions,
            "state_transitions": transitions,
            "identity_evidence": str(item.get("identity_evidence") or "").strip(),
            "identity_status": status,
            "confidence": confidence,
        })
    return out, errors


def _build_step_inventory_validation(
    inventory: Sequence[Dict[str, Any]], relations: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    refs = {str(item.get("entity_ref") or "") for item in inventory}
    missing_refs = sorted({
        str(relation.get("entity_ref") or "") for relation in relations
        if str(relation.get("entity_ref") or "") not in refs
    })
    by_name: Dict[str, List[Dict[str, Any]]] = {}
    for item in inventory:
        key = re.sub(r"\s+", " ", str(item.get("canonical_name") or "").strip()).casefold()
        if key:
            by_name.setdefault(key, []).append(item)
    duplicate_risks: List[Dict[str, Any]] = []
    for name, entries in by_name.items():
        if len(entries) < 2:
            continue
        duplicate_risks.append({
            "canonical_name": name,
            "entity_refs": [str(entry.get("entity_ref")) for entry in entries],
            "reason": "same canonical name; retained as separate instances unless temporal identity evidence proves a merge",
        })
    uncertain = [str(item.get("entity_ref")) for item in inventory if item.get("identity_status") == "uncertain"]
    warnings = []
    if duplicate_risks:
        warnings.append("same-category instances require review when identity evidence is weak")
    if uncertain:
        warnings.append("one or more physical identities remain uncertain")
    return {
        "status": "passed" if not missing_refs and not uncertain else "needs_review",
        "referenced_objects_all_resolved": not missing_refs,
        "missing_relation_entity_refs": missing_refs,
        "same_name_instance_risks": duplicate_risks,
        "unresolved_candidates": uncertain,
        "warnings": warnings,
    }


def _normalize_whole_video_step_review(
    obj: Any, *, boundaries: Dict[str, Any]
) -> Tuple[Optional[List[Dict[str, Any]]], List[str]]:
    errors: List[str] = []
    if not isinstance(obj, dict) or set(obj.keys()) != {"steps"} or not isinstance(obj.get("steps"), list):
        return None, ["Top-level response must contain exactly the list key steps."]
    expected = {int(step["hand_step_id"]): step for step in boundaries.get("hand_steps", [])}
    out: List[Dict[str, Any]] = []
    seen: set[int] = set()
    for index, raw in enumerate(obj["steps"]):
        required = {"step_id", "action_semantics", "hand_object_relations", "arm_visibility_intervals", "object_inventory"}
        if not isinstance(raw, dict) or set(raw.keys()) != required:
            errors.append(f"steps[{index}] has invalid keys.")
            continue
        try:
            step_id = int(raw["step_id"])
        except Exception:
            errors.append(f"steps[{index}].step_id is invalid.")
            continue
        if step_id not in expected or step_id in seen:
            errors.append(f"steps[{index}].step_id is unexpected or duplicated: {step_id}.")
            continue
        seen.add(step_id)
        boundary = expected[step_id]
        start, end = int(boundary["start_frame_id"]), int(boundary["end_frame_id"])
        semantics = raw.get("action_semantics")
        if not isinstance(semantics, dict) or set(semantics.keys()) != {"left_hand", "right_hand", "overall"}:
            errors.append(f"steps[{index}].action_semantics has invalid keys.")
            semantics = {}
        normalized_semantics = {key: str(semantics.get(key) or "not active").strip() for key in ("left_hand", "right_hand", "overall")}
        arm_raw = raw.get("arm_visibility_intervals")
        if not isinstance(arm_raw, dict) or set(arm_raw.keys()) != {"left_hand", "right_hand"}:
            errors.append(f"steps[{index}].arm_visibility_intervals has invalid keys.")
            arm_raw = {}
        arm_out: Dict[str, List[Dict[str, int]]] = {"left_hand": [], "right_hand": []}
        for hand in ("left_hand", "right_hand"):
            intervals = arm_raw.get(hand, [])
            if not isinstance(intervals, list):
                errors.append(f"steps[{index}].arm_visibility_intervals.{hand} must be a list.")
                continue
            previous_end = start - 1
            for interval_index, interval in enumerate(intervals):
                if not isinstance(interval, dict) or set(interval.keys()) != {"start_frame_id", "end_frame_id"}:
                    errors.append(f"steps[{index}].{hand}[{interval_index}] has invalid keys.")
                    continue
                try:
                    a, b = int(interval["start_frame_id"]), int(interval["end_frame_id"])
                except Exception:
                    errors.append(f"steps[{index}].{hand}[{interval_index}] has invalid frame IDs.")
                    continue
                if not (start <= a <= b <= end) or a <= previous_end:
                    errors.append(f"steps[{index}].{hand}[{interval_index}] is outside/overlapping step bounds.")
                    continue
                previous_end = b
                arm_out[hand].append({"start_frame_id": a, "end_frame_id": b})
        inventory_out, inventory_errors = _normalize_step_identity_inventory(
            raw.get("object_inventory"), step_id=step_id,
            start_frame_id=start, end_frame_id=end,
        )
        errors.extend(f"steps[{index}].{message}" for message in inventory_errors)
        refs = {str(item.get("entity_ref") or "") for item in inventory_out}
        relations_raw = raw.get("hand_object_relations")
        relations_out: List[Dict[str, str]] = []
        if not isinstance(relations_raw, list):
            errors.append(f"steps[{index}].hand_object_relations must be a list.")
            relations_raw = []
        for relation_index, relation in enumerate(relations_raw):
            keys = {"object", "hand", "relation", "entity_ref"}
            if not isinstance(relation, dict) or set(relation.keys()) != keys:
                errors.append(f"steps[{index}].hand_object_relations[{relation_index}] has invalid keys.")
                continue
            if str(relation.get("entity_ref")) not in refs:
                errors.append(f"steps[{index}].relation[{relation_index}] references an unknown step object.")
                continue
            relations_out.append({key: str(relation[key]) for key in keys})
        out.append({
            "step_id": step_id,
            "action_semantics": normalized_semantics,
            "hand_object_relations": relations_out,
            "arm_visibility_intervals": arm_out,
            "object_inventory": inventory_out,
        })
    if seen != set(expected):
        errors.append(f"Response step IDs must exactly match {sorted(expected)}; got {sorted(seen)}.")
    out.sort(key=lambda item: int(item["step_id"]))
    return (None, errors) if errors else (out, [])


OBJECT_REID_SYSTEM_PROMPT = """
You are an expert temporal object re-identification annotator. Analyze only the
attached short step video. For every supplied step-local entity_ref, detect
temporary disappearance and redetect the same physical instance when it
reappears. Match identity using appearance, color, shape, size, motion context,
hand interaction, and surrounding spatial relationships. Do not merge two
similar-looking instances. All frame IDs must use the original source-video
mapping supplied by the user, and every redetection bbox must use original-image
pixel xyxy coordinates. Return strict JSON only.
""".strip()


def _normalize_object_reidentification(
    obj: Any, *, step: Dict[str, Any], object_inventory: Sequence[Dict[str, Any]],
    width: int, height: int,
) -> Tuple[Optional[List[Dict[str, Any]]], List[str]]:
    errors: List[str] = []
    if not isinstance(obj, dict) or set(obj.keys()) != {"step_id", "objects"}:
        return None, ["Object ReID output must contain exactly step_id and objects."]
    if int(obj.get("step_id", -1)) != int(step["hand_step_id"]):
        errors.append("Object ReID step_id mismatch.")
    raw_objects = obj.get("objects")
    if not isinstance(raw_objects, list):
        return None, errors + ["Object ReID objects must be a list."]
    expected = {
        str(item.get("entity_ref") or ""): item for item in object_inventory
        if isinstance(item, dict) and str(item.get("entity_ref") or "")
    }
    start, end = int(step["start_frame_id"]), int(step["end_frame_id"])
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    allowed_status = {"continuous", "reacquired", "not_reacquired", "not_visible_enough"}
    for index, raw in enumerate(raw_objects):
        required = {"entity_ref", "canonical_name", "continuity_status", "disappearance_intervals", "reappearance_detections"}
        if not isinstance(raw, dict) or set(raw.keys()) != required:
            errors.append(f"objects[{index}] has invalid keys.")
            continue
        ref = str(raw.get("entity_ref") or "").strip()
        status = str(raw.get("continuity_status") or "").strip()
        if ref not in expected or ref in seen:
            errors.append(f"objects[{index}].entity_ref is unknown or duplicated: {ref}.")
            continue
        seen.add(ref)
        if status not in allowed_status:
            errors.append(f"objects[{index}].continuity_status is invalid.")
        intervals_out: List[Dict[str, int]] = []
        intervals = raw.get("disappearance_intervals")
        if not isinstance(intervals, list):
            errors.append(f"objects[{index}].disappearance_intervals must be a list.")
            intervals = []
        previous_reappearance = start - 1
        for interval_index, interval in enumerate(intervals):
            if not isinstance(interval, dict) or set(interval.keys()) != {"last_visible_frame_id", "first_reappeared_frame_id"}:
                errors.append(f"objects[{index}].disappearance_intervals[{interval_index}] has invalid keys.")
                continue
            try:
                last_visible = int(interval["last_visible_frame_id"])
                first_reappeared = int(interval["first_reappeared_frame_id"])
            except Exception:
                errors.append(f"objects[{index}].disappearance_intervals[{interval_index}] has invalid frame IDs.")
                continue
            if not (start <= last_visible < first_reappeared <= end) or last_visible < previous_reappearance:
                errors.append(f"objects[{index}].disappearance_intervals[{interval_index}] is outside/overlapping step bounds.")
                continue
            previous_reappearance = first_reappeared
            intervals_out.append({"last_visible_frame_id": last_visible, "first_reappeared_frame_id": first_reappeared})
        detections_out: List[Dict[str, Any]] = []
        detections = raw.get("reappearance_detections")
        if not isinstance(detections, list):
            errors.append(f"objects[{index}].reappearance_detections must be a list.")
            detections = []
        for detection_index, detection in enumerate(detections):
            if not isinstance(detection, dict) or set(detection.keys()) != {"frame_id", "bbox_xyxy", "match_confidence", "identity_evidence"}:
                errors.append(f"objects[{index}].reappearance_detections[{detection_index}] has invalid keys.")
                continue
            try:
                frame_id = int(detection["frame_id"])
            except Exception:
                frame_id = -1
            bbox = _valid_bbox(detection.get("bbox_xyxy"), width=width, height=height)
            evidence = str(detection.get("identity_evidence") or "").strip()
            if not (start <= frame_id <= end) or bbox is None or not evidence:
                errors.append(f"objects[{index}].reappearance_detections[{detection_index}] is invalid.")
                continue
            detections_out.append({
                "frame_id": frame_id, "bbox_xyxy": bbox,
                "bbox_source": "vlm_reid_redetection",
                "match_confidence": _clamp_float(detection.get("match_confidence")),
                "identity_evidence": evidence,
            })
        reappeared_ids = {item["first_reappeared_frame_id"] for item in intervals_out}
        detection_ids = {item["frame_id"] for item in detections_out}
        if status == "reacquired" and (not intervals_out or not reappeared_ids.issubset(detection_ids)):
            errors.append(f"objects[{index}] reacquired status requires a bbox at every first_reappeared_frame_id.")
        if status == "continuous" and (intervals_out or detections_out):
            errors.append(f"objects[{index}] continuous status cannot contain disappearance/redetection records.")
        out.append({
            "entity_ref": ref,
            "canonical_name": str(expected[ref].get("canonical_name") or raw.get("canonical_name") or "").strip(),
            "continuity_status": status,
            "disappearance_intervals": intervals_out,
            "reappearance_detections": detections_out,
            "identity_preserved_within_step": status in {"continuous", "reacquired"},
        })
    if seen != set(expected):
        errors.append(f"Object ReID must return every inventory entity_ref: expected={sorted(expected)}, got={sorted(seen)}.")
    return (None, errors) if errors else (out, [])


def _contiguous_frame_intervals(frame_ids: Iterable[int]) -> List[Dict[str, int]]:
    ids = sorted(set(int(value) for value in frame_ids))
    if not ids:
        return []
    intervals: List[Dict[str, int]] = []
    start = previous = ids[0]
    for frame_id in ids[1:]:
        if frame_id == previous + 1:
            previous = frame_id
            continue
        intervals.append({"start_frame_id": start, "end_frame_id": previous})
        start = previous = frame_id
    intervals.append({"start_frame_id": start, "end_frame_id": previous})
    return intervals


def _normalize_bbox_arm_evaluations(
    raw: Any, *, step: Dict[str, Any]
) -> Tuple[Optional[List[Dict[str, Any]]], List[str]]:
    errors: List[str] = []
    tracks = step.get("stage3_bbox_tracks", step.get("bbox_tracks", {}))
    tracks = tracks if isinstance(tracks, dict) else {}
    expected: Dict[Tuple[str, str], List[int]] = {}
    for hand in ("left_hand", "right_hand"):
        for record in tracks.get(hand, []) if isinstance(tracks.get(hand), list) else []:
            if not isinstance(record, dict) or "frame_id" not in record:
                continue
            source = str(record.get("bbox_source") or "upstream_sam3")
            expected.setdefault((hand, source), []).append(int(record["frame_id"]))
    if not isinstance(raw, list):
        return None, ["bbox_arm_evaluations must be a list."]
    out: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str]] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or set(item.keys()) != {"hand", "bbox_source", "frames"}:
            errors.append(f"bbox_arm_evaluations[{index}] has invalid keys.")
            continue
        hand = str(item.get("hand") or "")
        source = str(item.get("bbox_source") or "")
        key = (hand, source)
        if key not in expected or key in seen:
            errors.append(f"bbox_arm_evaluations[{index}] has unknown/duplicate track {key}.")
            continue
        seen.add(key)
        frames_raw = item.get("frames")
        if not isinstance(frames_raw, list):
            errors.append(f"bbox_arm_evaluations[{index}].frames must be a list.")
            frames_raw = []
        frames_out: List[Dict[str, int]] = []
        actual_ids: List[int] = []
        for frame_index, frame in enumerate(frames_raw):
            if not isinstance(frame, dict) or set(frame.keys()) != {"frame_id", "arm_visible"}:
                errors.append(f"bbox_arm_evaluations[{index}].frames[{frame_index}] has invalid keys.")
                continue
            try:
                frame_id = int(frame["frame_id"])
                arm_visible = int(frame["arm_visible"])
            except Exception:
                errors.append(f"bbox_arm_evaluations[{index}].frames[{frame_index}] has invalid values.")
                continue
            if arm_visible not in {0, 1}:
                errors.append(f"bbox_arm_evaluations[{index}].frames[{frame_index}].arm_visible must be 0 or 1.")
                continue
            actual_ids.append(frame_id)
            frames_out.append({"frame_id": frame_id, "arm_visible": arm_visible})
        expected_ids = sorted(set(expected[key]))
        if actual_ids != expected_ids:
            errors.append(
                f"bbox_arm_evaluations[{index}] frame IDs must exactly match bbox track: "
                f"expected={expected_ids}, got={actual_ids}."
            )
        visible_ids = [frame["frame_id"] for frame in frames_out if frame["arm_visible"] == 1]
        out.append({
            "hand": hand,
            "bbox_source": source,
            "frame_evaluations": frames_out,
            "arm_visible_intervals": _contiguous_frame_intervals(visible_ids),
        })
    if seen != set(expected):
        errors.append(
            f"bbox_arm_evaluations must return every bbox track: "
            f"expected={sorted(expected)}, got={sorted(seen)}."
        )
    return (None, errors) if errors else (out, [])


def _normalize_bbox_arm_intervals(
    raw: Any, *, step: Dict[str, Any]
) -> Tuple[Optional[List[Dict[str, Any]]], List[str]]:
    errors: List[str] = []
    tracks = step.get("stage3_bbox_tracks", step.get("bbox_tracks", {}))
    tracks = tracks if isinstance(tracks, dict) else {}
    expected: Dict[Tuple[str, str], set[int]] = {}
    for hand in ("left_hand", "right_hand"):
        for record in tracks.get(hand, []) if isinstance(tracks.get(hand), list) else []:
            if not isinstance(record, dict) or "frame_id" not in record:
                continue
            source = str(record.get("bbox_source") or "upstream_sam3")
            expected.setdefault((hand, source), set()).add(int(record["frame_id"]))
    if not isinstance(raw, list):
        return None, ["bbox_arm_visibility must be a list."]
    out: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str]] = set()
    step_start, step_end = int(step["start_frame_id"]), int(step["end_frame_id"])
    for index, item in enumerate(raw):
        required_keys = {"hand", "bbox_source", "arm_visible_intervals"}
        allowed_keys = required_keys | {
            "bbox_may_not_contain_complete_hand", "result_may_be_unreliable",
            "unreliable_frame_intervals", "reliability_reason",
        }
        if (
            not isinstance(item, dict)
            or not required_keys.issubset(set(item.keys()))
            or bool(set(item.keys()) - allowed_keys)
        ):
            errors.append(f"bbox_arm_visibility[{index}] has invalid keys.")
            continue
        hand = str(item.get("hand") or "")
        source = str(item.get("bbox_source") or "")
        key = (hand, source)
        if key not in expected or key in seen:
            errors.append(f"bbox_arm_visibility[{index}] has unknown/duplicate track {key}.")
            continue
        seen.add(key)
        intervals_raw = item.get("arm_visible_intervals")
        if not isinstance(intervals_raw, list):
            errors.append(f"bbox_arm_visibility[{index}].arm_visible_intervals must be a list.")
            intervals_raw = []
        visible_frame_ids: List[int] = []
        previous_end = step_start - 1
        available_frames = expected[key]
        for interval_index, interval in enumerate(intervals_raw):
            if not isinstance(interval, dict) or set(interval.keys()) != {"start_frame_id", "end_frame_id"}:
                errors.append(f"bbox_arm_visibility[{index}].arm_visible_intervals[{interval_index}] has invalid keys.")
                continue
            try:
                start_frame = int(interval["start_frame_id"])
                end_frame = int(interval["end_frame_id"])
            except Exception:
                errors.append(f"bbox_arm_visibility[{index}].arm_visible_intervals[{interval_index}] has invalid frame IDs.")
                continue
            if not (step_start <= start_frame <= end_frame <= step_end) or start_frame <= previous_end:
                errors.append(f"bbox_arm_visibility[{index}].arm_visible_intervals[{interval_index}] is outside/overlapping step bounds.")
                continue
            previous_end = end_frame
            interval_frames = set(range(start_frame, end_frame + 1))
            retained_frames = sorted(interval_frames & available_frames)
            removed_count = len(interval_frames) - len(retained_frames)
            if removed_count:
                logger.warning(
                    f"[stage3] clipped bbox arm interval step_id={step.get('hand_step_id')} "
                    f"hand={hand} source={source} requested=[{start_frame},{end_frame}] "
                    f"removed_frames_without_bbox={removed_count}"
                )
            visible_frame_ids.extend(retained_frames)
        intervals_out = _contiguous_frame_intervals(visible_frame_ids)
        unreliable_ids = sorted({
            int(annotation["frame_id"])
            for annotation in step.get("bbox_quality_annotations", [])
            if isinstance(annotation, dict)
            and str(annotation.get("hand")) == hand
            and str(annotation.get("bbox_source") or "upstream_sam3") == source
            and int(annotation.get("frame_id", -1)) in available_frames
            and bool(annotation.get("source_bbox_may_not_contain_complete_hand"))
        })
        unreliable_intervals = _contiguous_frame_intervals(unreliable_ids)
        out.append({
            "hand": hand,
            "bbox_source": source,
            "arm_visible_intervals": intervals_out,
            "bbox_may_not_contain_complete_hand": bool(unreliable_ids),
            "result_may_be_unreliable": bool(unreliable_ids),
            "unreliable_frame_intervals": unreliable_intervals,
            "reliability_reason": (
                "source_bbox_width_height_and_area_temporarily_shrank; arm visibility still refers only to source bbox"
                if unreliable_ids else "no_temporal_bbox_shrink_detected"
            ),
        })
    if seen != set(expected):
        errors.append(
            f"bbox_arm_visibility must return every bbox track: "
            f"expected={sorted(expected)}, got={sorted(seen)}."
        )
    return (None, errors) if errors else (out, [])


def _refine_keep_steps_whole_video(
    *, video_path: str, output_root: str, api_cfg: Any, overwrite: bool, max_retries: int
) -> Tuple[str, Dict[str, Any], Dict[str, int]]:
    vid = video_id_from_path(video_path)
    video_out = os.path.join(output_root, vid)
    ensure_video_out_dir_safe(video_out, video_path)
    stage2_path = os.path.join(video_out, "stage2", "hand_step_boundaries.json")
    stage3_dir = os.path.join(video_out, "stage3")
    if not os.path.exists(stage2_path):
        raise FileNotFoundError(f"Stage 2 hand-step boundaries not found: {stage2_path}")
    boundaries = read_json(stage2_path)
    meta = get_video_metadata(video_path)
    client = initialize_api_client(api_cfg)
    if not client:
        raise SystemExit("Failed to initialize API client.")
    compact_steps = [
        {
            "step_id": int(step["hand_step_id"]),
            "start_frame_id": int(step["start_frame_id"]),
            "end_frame_id": int(step["end_frame_id"]),
            "upstream_bbox_tracks": step.get("bbox_tracks", {}),
        }
        for step in boundaries.get("hand_steps", [])
    ]
    prompt = f"""
The attached complete source video is {int(meta['width'])}x{int(meta['height'])},
{float(meta['fps']):.8g} FPS, with 0-based frame IDs. Review these authoritative
upstream keep=true steps and bbox tracks without changing any frame ID or bbox:
{json.dumps(compact_steps, ensure_ascii=False)}

Return exactly {{"steps":[...]}}. Every step object must contain exactly:
- step_id
- action_semantics with left_hand/right_hand/overall strings
- hand_object_relations: object/hand/relation/entity_ref
- arm_visibility_intervals with left_hand/right_hand lists of inclusive
  start_frame_id/end_frame_id intervals inside this step; judge arm visibility
  from the complete video in temporal context, not isolated bbox crops
- object_inventory entries with entity_ref, canonical_name, description,
  selection_reason, priority. priority is direct_hand_interaction or
  moved_or_shape_changed. Include direct hand-interaction objects first, then
  moved/shape-changed objects. Exclude unchanged background objects.
""".strip()
    video_content = build_video_api_content(video_path, [], video_fps=api_cfg.video_fps, include_manifest=False)
    usage_total = _usage_zero()
    last_content = ""
    errors: List[str] = []
    normalized: Optional[List[Dict[str, Any]]] = None
    for attempt in range(1, max(1, int(max_retries)) + 1):
        prefix = build_retry_prefix(errors, last_content) if attempt > 1 else ""
        messages = [
            {"role": "system", "content": WHOLE_VIDEO_STEP_REVIEW_SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": prefix + prompt}] + video_content},
        ]
        last_content, usage = call_chat_completion(client, api_cfg, messages, max_tokens=min(api_cfg.max_tokens, 16000))
        _add_usage(usage_total, usage)
        try:
            parsed = json.loads(extract_json_from_response(last_content))
            normalized, errors = _normalize_whole_video_step_review(parsed, boundaries=boundaries)
        except Exception as exc:
            normalized, errors = None, [f"JSON parse error: {exc}"]
        if normalized is not None:
            break
    os.makedirs(stage3_dir, exist_ok=True)
    write_text(os.path.join(stage3_dir, "whole_video_step_review_system_prompt.txt"), WHOLE_VIDEO_STEP_REVIEW_SYSTEM_PROMPT)
    write_text(os.path.join(stage3_dir, "whole_video_step_review_user_prompt.txt"), prompt)
    write_text(os.path.join(stage3_dir, "whole_video_step_review_raw_response.txt"), last_content)
    if normalized is None:
        raise RuntimeError("Whole-video step review failed: " + " | ".join(errors[:10]))
    review_by_id = {int(item["step_id"]): item for item in normalized}
    reviewed_steps: List[Dict[str, Any]] = []
    plan_steps: List[Dict[str, Any]] = []
    for boundary in boundaries.get("hand_steps", []):
        step_id = int(boundary["hand_step_id"])
        review = review_by_id[step_id]
        step_dir = os.path.join(stage3_dir, f"step_{step_id:03d}")
        os.makedirs(step_dir, exist_ok=True)
        write_json(os.path.join(step_dir, "object_inventory.json"), {
            "step_id": step_id, "object_inventory": review["object_inventory"]
        })
        write_json(os.path.join(step_dir, "step_review.json"), review)
        reviewed = {
            "hand_step_id": step_id,
            "start_frame_id": int(boundary["start_frame_id"]),
            "end_frame_id": int(boundary["end_frame_id"]),
            "end_frame_id_exclusive": int(boundary["end_frame_id"]) + 1,
            "frame_ids": list(boundary.get("frame_ids", [])),
            "detection_sources": ["upstream_keep"],
            "active_hands": list(boundary.get("active_hands", [])),
            "bbox_tracks": boundary.get("bbox_tracks", {}),
            "action_semantics": review["action_semantics"],
            "multiple_action_subjects": bool(boundary.get("multiple_action_subjects", False)),
            "hand_object_relations": review["hand_object_relations"],
            "arm_visibility_intervals": review["arm_visibility_intervals"],
            "object_inventory": review["object_inventory"],
        }
        reviewed_steps.append(reviewed)
        plan_steps.append({
            "step_id": step_id,
            "step_goal": review["action_semantics"]["overall"],
            "action_semantics": review["action_semantics"],
            "multiple_action_subjects": bool(boundary.get("multiple_action_subjects", False)),
            "active_hands": reviewed["active_hands"],
            "hand_object_relations": review["hand_object_relations"],
            "arm_visibility_intervals": review["arm_visibility_intervals"],
            "object_inventory": review["object_inventory"],
            "bbox_tracks": boundary.get("bbox_tracks", {}),
            # Analysis/repaired/padded bboxes are exposed separately; the
            # bbox_tracks field above remains the authoritative upstream copy.
            "bbox_tracks_modified": boundary.get("analysis_bbox_tracks", {}),
            "frame_ids": list(boundary.get("frame_ids", [])),
            "start_frame_0based": int(boundary["start_frame_id"]),
            "end_frame_0based_inclusive": int(boundary["end_frame_id"]),
            "end_frame_0based_exclusive": int(boundary["end_frame_id"]) + 1,
            "start_sec": float(boundary["start_frame_id"]) / float(meta["fps"]),
            "end_sec": float(int(boundary["end_frame_id"]) + 1) / float(meta["fps"]),
            "clip_materialized": False, "clip_path": None, "clip_relpath": None,
        })
    write_json(os.path.join(stage3_dir, "hand_steps.json"), {
        "schema": "hand_steps.keep_runs.v2", "video_id": vid,
        "source_video": build_source_video_meta(video_path), "fps": float(meta["fps"]),
        "total_frames": int(meta["total_frames"]), "frame_index_base": 0,
        "endpoint_semantics": "inclusive", "media_artifacts_materialized": False,
        "hand_steps": reviewed_steps,
    })
    plan = {
        "source_video": build_source_video_meta(video_path),
        "high_level_goal": "Primary operator hand actions in upstream keep=true intervals.",
        "steps": plan_steps,
    }
    write_json(os.path.join(stage3_dir, "step_plan.json"), plan)
    return video_out, plan, usage_total


def _refine_keep_steps_per_step_video(
    *, video_path: str, output_root: str, api_cfg: Any, overwrite: bool, max_retries: int
) -> Tuple[str, Dict[str, Any], Dict[str, int]]:
    vid = video_id_from_path(video_path)
    video_out = os.path.join(output_root, vid)
    ensure_video_out_dir_safe(video_out, video_path)
    stage2_path = os.path.join(video_out, "stage2", "hand_step_boundaries.json")
    stage3_dir = os.path.join(video_out, "stage3")
    if not os.path.exists(stage2_path):
        raise FileNotFoundError(f"Stage 2 hand-step boundaries not found: {stage2_path}")
    boundaries = read_json(stage2_path)
    meta = get_video_metadata(video_path)
    # Stage 1 already uploaded each clip once and wrote Stage-3 interval output.
    # Stage 3 only validates and assembles those cached results; no API client
    # is initialized and no video is uploaded here.
    os.makedirs(stage3_dir, exist_ok=True)
    reviewed_steps: List[Dict[str, Any]] = []
    plan_steps: List[Dict[str, Any]] = []
    for boundary in boundaries.get("hand_steps", []):
        step_id = int(boundary["hand_step_id"])
        step_dir = os.path.join(stage3_dir, f"step_{step_id:03d}")
        result_path = os.path.join(step_dir, "bbox_arm_visibility.json")
        if not os.path.exists(result_path):
            raise FileNotFoundError(
                f"Single-pass Stage-3 result is missing for step_id={step_id}: {result_path}. "
                "Rerun Stage 1 for this video."
            )
        cached = read_json(result_path)
        if int(cached.get("step_id", -1)) != step_id:
            raise RuntimeError(f"Stage-3 cached step_id mismatch: {result_path}")
        normalized_bbox_arm, interval_errors = _normalize_bbox_arm_intervals(
            cached.get("bbox_arm_visibility"), step=boundary
        )
        if normalized_bbox_arm is None:
            raise RuntimeError(
                f"Invalid single-pass Stage-3 intervals for step_id={step_id}: "
                + " | ".join(interval_errors[:10])
            )
        visible_by_hand: Dict[str, List[int]] = {"left_hand": [], "right_hand": []}
        for track_eval in normalized_bbox_arm:
            visible_by_hand[track_eval["hand"]].extend(
                frame_id
                for interval in track_eval["arm_visible_intervals"]
                for frame_id in range(int(interval["start_frame_id"]), int(interval["end_frame_id"]) + 1)
            )
        arm_intervals = {
            hand: _contiguous_frame_intervals(frame_ids)
            for hand, frame_ids in visible_by_hand.items()
        }
        semantics = boundary.get("draft_action_semantics", {})
        if not isinstance(semantics, dict):
            raise RuntimeError(f"Stage-1 semantics missing for step_id={step_id}")
        clip_path = _require_stage1_step_clip(video_out=video_out, step=boundary)
        common = {
            "start_frame_id": int(boundary["start_frame_id"]),
            "end_frame_id": int(boundary["end_frame_id"]),
            "frame_ids": list(boundary.get("frame_ids", [])),
            "stage3_evaluation_scope": "valid_keep_true_actions_only",
            "valid_action_intervals": list(boundary.get("valid_action_intervals", [])),
            "active_hands": list(boundary.get("active_hands", [])),
            "bbox_tracks": boundary.get("bbox_tracks", {}),
            "analysis_bbox_tracks": boundary.get("analysis_bbox_tracks", {}),
            "bbox_quality_annotations": boundary.get("bbox_quality_annotations", []),
            "bbox_repair_summary": boundary.get("bbox_repair_summary", {}),
            "action_semantics": semantics,
            "multiple_action_subjects": bool(boundary.get("multiple_action_subjects", False)),
            "hand_object_relations": boundary.get("hand_object_relations", []),
            "arm_visibility_intervals": arm_intervals,
            "bbox_arm_visibility": normalized_bbox_arm,
            "object_inventory": boundary.get("object_inventory", []),
            "clip_materialized": True,
            "clip_path": os.path.relpath(clip_path, video_out),
            "clip_relpath": os.path.relpath(clip_path, video_out),
            "visualization_clip_path": os.path.relpath(
                _step_visualization_path(video_out, boundary), video_out
            ),
            "clip_local_start_frame_id": 0,
            "source_frame_id_offset": int(boundary["start_frame_id"]),
        }
        reviewed_steps.append({
            "hand_step_id": step_id,
            "end_frame_id_exclusive": int(boundary["end_frame_id"]) + 1,
            "detection_sources": ["upstream_keep"],
            **common,
        })
        plan_steps.append({
            "step_id": step_id,
            "step_goal": str(semantics.get("overall", "")),
            **common,
            "start_frame_0based": int(boundary["start_frame_id"]),
            "end_frame_0based_inclusive": int(boundary["end_frame_id"]),
            "end_frame_0based_exclusive": int(boundary["end_frame_id"]) + 1,
            "start_sec": float(boundary["start_frame_id"]) / float(meta["fps"]),
            "end_sec": float(int(boundary["end_frame_id"]) + 1) / float(meta["fps"]),
        })
    write_json(os.path.join(stage3_dir, "hand_steps.json"), {
        "schema": "hand_steps.keep_runs.v3.single_pass",
        "video_id": vid,
        "source_video": build_source_video_meta(video_path),
        "fps": float(meta["fps"]),
        "total_frames": int(meta["total_frames"]),
        "frame_index_base": 0,
        "endpoint_semantics": "inclusive",
        "media_artifacts_materialized": True,
        "model_video_uploads_in_stage3": 0,
        "hand_steps": reviewed_steps,
    })
    plan = {
        "source_video": build_source_video_meta(video_path),
        "high_level_goal": compose_video_goal(step["step_goal"] for step in plan_steps) if plan_steps else NO_SUBTASK_GOAL,
        "steps": plan_steps,
    }
    write_json(os.path.join(stage3_dir, "step_plan.json"), plan)
    return video_out, plan, _usage_zero()

    # Legacy multi-call Stage-3 implementation is unreachable.
    client = initialize_api_client(api_cfg)
    if not client:
        raise SystemExit("Failed to initialize API client.")
    os.makedirs(stage3_dir, exist_ok=True)
    usage_total = _usage_zero()
    reviewed_steps: List[Dict[str, Any]] = []
    plan_steps: List[Dict[str, Any]] = []
    system_prompt = (
        "Inspect the attached step-only video and use its upstream bbox "
        "trajectory. Return strict JSON only. Directly locate complete original-frame intervals "
        "where each bbox track contains a connected wrist or forearm. "
        "Never change frame IDs or bboxes. Do not identify objects and do not score hand completeness."
    )
    for boundary in boundaries.get("hand_steps", []):
        step_id = int(boundary["hand_step_id"])
        semantics = boundary.get("draft_action_semantics", {})
        if not isinstance(semantics, dict):
            semantics = {"left_hand": "not active", "right_hand": "not active", "overall": str(semantics)}
        step_dir = os.path.join(stage3_dir, f"step_{step_id:03d}")
        os.makedirs(step_dir, exist_ok=True)
        clip_path = _require_stage1_step_clip(video_out=video_out, step=boundary)
        video_content = build_video_api_content(
            clip_path, [], video_fps=api_cfg.video_fps, include_manifest=False
        )
        prompt = f"""
Analyze exactly step_id={step_id}, inclusive original source-frame interval
[{int(boundary['start_frame_id'])}, {int(boundary['end_frame_id'])}]. Do not
split or expand it. {_step_clip_mapping_text(boundary)}

Stage-3 evaluation is restricted to these original valid-action intervals:
{json.dumps(boundary.get('valid_action_intervals', []), ensure_ascii=False)}

Authoritative upstream bbox tracks inside those valid-action intervals only, in original-frame IDs and original-image pixel coordinates:
{json.dumps(boundary.get('stage3_bbox_tracks', boundary.get('bbox_tracks', {})), ensure_ascii=False)}

Return exactly:
{{
  "step_id": {step_id},
  "bbox_arm_visibility": [
    {{
      "hand": "left_hand",
      "bbox_source": "upstream_sam3",
      "arm_visible_intervals": [
        {{"start_frame_id": 0, "end_frame_id": 0}}
      ]
    }}
  ]
}}
Return exactly one record for every supplied hand+bbox_source track. Judge the
whole short video directly and return complete, non-overlapping, chronological
inclusive original-frame intervals. Return an empty interval list when that bbox
track never contains a connected wrist or forearm. Do not return per-frame flags.
Ignore every frame outside the supplied valid-action intervals.
""".strip()
        last_content = ""
        errors: List[str] = []
        normalized_step: Optional[Dict[str, Any]] = None
        one_boundary = {"hand_steps": [boundary]}
        for attempt in range(1, max(1, int(max_retries)) + 1):
            prefix = build_retry_prefix(errors, last_content) if attempt > 1 else ""
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "text", "text": prefix + prompt}] + video_content},
            ]
            last_content, usage = call_chat_completion(client, api_cfg, messages, max_tokens=min(api_cfg.max_tokens, 6000))
            _add_usage(usage_total, usage)
            try:
                parsed = json.loads(extract_json_from_response(last_content))
                if not isinstance(parsed, dict) or set(parsed.keys()) != {"step_id", "bbox_arm_visibility"}:
                    raise ValueError("Stage 3 response has invalid top-level keys.")
                if int(parsed.get("step_id", -1)) != step_id:
                    raise ValueError("Stage 3 response step_id mismatch.")
                normalized_bbox_arm, bbox_arm_errors = _normalize_bbox_arm_intervals(
                    parsed.get("bbox_arm_visibility"), step=boundary
                )
                if normalized_bbox_arm is None:
                    raise ValueError(" | ".join(bbox_arm_errors[:10]))
                visible_by_hand: Dict[str, List[int]] = {"left_hand": [], "right_hand": []}
                for track_eval in normalized_bbox_arm:
                    visible_by_hand[track_eval["hand"]].extend(
                        frame_id
                        for interval in track_eval["arm_visible_intervals"]
                        for frame_id in range(
                            int(interval["start_frame_id"]), int(interval["end_frame_id"]) + 1
                        )
                    )
                arm_visibility_intervals = {
                    hand: _contiguous_frame_intervals(frame_ids)
                    for hand, frame_ids in visible_by_hand.items()
                }
                wrapped = {"steps": [{
                    "step_id": step_id,
                    "arm_visibility_intervals": arm_visibility_intervals,
                    "action_semantics": semantics,
                    "hand_object_relations": boundary.get("hand_object_relations", []),
                    "object_inventory": boundary.get("object_inventory", []),
                }]}
                normalized_list, errors = _normalize_whole_video_step_review(wrapped, boundaries=one_boundary)
                normalized_step = normalized_list[0] if normalized_list else None
                if normalized_step is not None:
                    normalized_step["bbox_arm_visibility"] = normalized_bbox_arm
            except Exception as exc:
                normalized_step, errors = None, [f"JSON/validation error: {exc}"]
            if normalized_step is not None:
                break
        write_text(os.path.join(step_dir, "stage3_system_prompt.txt"), system_prompt)
        write_text(os.path.join(step_dir, "stage3_user_prompt.txt"), prompt)
        write_text(os.path.join(step_dir, "stage3_raw_response.txt"), last_content)
        if normalized_step is None:
            raise RuntimeError(f"Stage 3 step_id={step_id} failed: " + " | ".join(errors[:10]))
        write_json(os.path.join(step_dir, "bbox_arm_visibility.json"), {
            "schema": "step_bbox_arm_visibility.v1",
            "step_id": step_id,
            "start_frame_id": int(boundary["start_frame_id"]),
            "end_frame_id": int(boundary["end_frame_id"]),
            "endpoint_semantics": "inclusive",
            "evaluation_scope": "valid_keep_true_actions_only",
            "valid_action_intervals": list(boundary.get("valid_action_intervals", [])),
            "bbox_arm_visibility": normalized_step["bbox_arm_visibility"],
            "arm_visibility_intervals": normalized_step["arm_visibility_intervals"],
        })
        object_inventory = [
            item for item in boundary.get("object_inventory", []) if isinstance(item, dict)
        ]
        object_reidentification: List[Dict[str, Any]] = []
        if object_inventory:
            reid_prompt = f"""
Analyze object identity continuity for step_id={step_id}, inclusive original
source-frame interval [{int(boundary['start_frame_id'])}, {int(boundary['end_frame_id'])}].
{_step_clip_mapping_text(boundary)}

The authoritative step-local inventory is:
{json.dumps(object_inventory, ensure_ascii=False)}

For every inventory entity_ref, inspect the entire short video. If the same
physical object disappears and later reappears, redetect it on the first
reappeared original frame, output a tight original-image pixel bbox, and keep
the exact original entity_ref. Distinguish multiple similar instances using
appearance and temporal/spatial context. If it never disappears, use
continuity_status="continuous". If it disappears but cannot be matched safely,
use "not_reacquired" and do not invent a bbox.

Return exactly:
{{
  "step_id": {step_id},
  "objects": [
    {{
      "entity_ref": "...",
      "canonical_name": "...",
      "continuity_status": "continuous|reacquired|not_reacquired|not_visible_enough",
      "disappearance_intervals": [
        {{"last_visible_frame_id": 0, "first_reappeared_frame_id": 0}}
      ],
      "reappearance_detections": [
        {{"frame_id": 0, "bbox_xyxy": [0, 0, 1, 1], "match_confidence": 0.0,
          "identity_evidence": "English explanation of the instance match"}}
      ]
    }}
  ]
}}
Return exactly one object record for every supplied entity_ref and no extra keys.
""".strip()
            reid_content = ""
            reid_errors: List[str] = []
            normalized_reid: Optional[List[Dict[str, Any]]] = None
            for attempt in range(1, max(1, int(max_retries)) + 1):
                prefix = build_retry_prefix(reid_errors, reid_content) if attempt > 1 else ""
                messages = [
                    {"role": "system", "content": OBJECT_REID_SYSTEM_PROMPT},
                    {"role": "user", "content": [{"type": "text", "text": prefix + reid_prompt}] + video_content},
                ]
                reid_content, usage = call_chat_completion(
                    client, api_cfg, messages, max_tokens=min(api_cfg.max_tokens, 7000)
                )
                _add_usage(usage_total, usage)
                try:
                    parsed = json.loads(extract_json_from_response(reid_content))
                    normalized_reid, reid_errors = _normalize_object_reidentification(
                        parsed, step=boundary, object_inventory=object_inventory,
                        width=int(meta["width"]), height=int(meta["height"]),
                    )
                except Exception as exc:
                    normalized_reid, reid_errors = None, [f"JSON/validation error: {exc}"]
                if normalized_reid is not None:
                    break
            write_text(os.path.join(step_dir, "object_reid_system_prompt.txt"), OBJECT_REID_SYSTEM_PROMPT)
            write_text(os.path.join(step_dir, "object_reid_user_prompt.txt"), reid_prompt)
            write_text(os.path.join(step_dir, "object_reid_raw_response.txt"), reid_content)
            if normalized_reid is None:
                raise RuntimeError(f"Stage 3 object ReID step_id={step_id} failed: " + " | ".join(reid_errors[:10]))
            object_reidentification = normalized_reid
            write_json(os.path.join(step_dir, "object_reidentification.json"), {
                "schema": "step_object_reidentification.v1", "step_id": step_id,
                "objects": object_reidentification,
            })
        write_json(os.path.join(step_dir, "step_review.json"), normalized_step)
        reid_by_ref = {item["entity_ref"]: item for item in object_reidentification}
        enriched_inventory = []
        for item in object_inventory:
            enriched = dict(item)
            if str(item.get("entity_ref") or "") in reid_by_ref:
                enriched["reidentification"] = reid_by_ref[str(item["entity_ref"])]
            enriched_inventory.append(enriched)
        reviewed = {
            "hand_step_id": step_id,
            "start_frame_id": int(boundary["start_frame_id"]),
            "end_frame_id": int(boundary["end_frame_id"]),
            "end_frame_id_exclusive": int(boundary["end_frame_id"]) + 1,
            "frame_ids": list(boundary.get("frame_ids", [])),
            "detection_sources": ["upstream_keep"],
            "active_hands": list(boundary.get("active_hands", [])),
            "bbox_tracks": boundary.get("bbox_tracks", {}),
            "action_semantics": semantics,
            "hand_object_relations": boundary.get("hand_object_relations", []),
            "arm_visibility_intervals": normalized_step["arm_visibility_intervals"],
            "bbox_arm_visibility": normalized_step["bbox_arm_visibility"],
            "object_inventory": enriched_inventory,
            "object_reidentification": object_reidentification,
            "clip_materialized": True,
            "clip_path": os.path.relpath(clip_path, video_out),
            "clip_relpath": os.path.relpath(clip_path, video_out),
            "clip_local_start_frame_id": 0,
            "source_frame_id_offset": int(boundary["start_frame_id"]),
        }
        reviewed_steps.append(reviewed)
        plan_steps.append({
            "step_id": step_id, "step_goal": semantics.get("overall", ""),
            "action_semantics": semantics,
            "multiple_action_subjects": bool(boundary.get("multiple_action_subjects", False)),
            "active_hands": reviewed["active_hands"],
            "hand_object_relations": boundary.get("hand_object_relations", []),
            "arm_visibility_intervals": normalized_step["arm_visibility_intervals"],
            "bbox_arm_visibility": normalized_step["bbox_arm_visibility"],
            # final_plan geometry must remain upstream-authoritative. ReID
            # detections (including model-generated reacquisition bboxes) stay
            # in Stage-3 diagnostic artifacts and are not promoted downstream.
            "object_inventory": object_inventory,
            "bbox_tracks": boundary.get("bbox_tracks", {}),
            "frame_ids": list(boundary.get("frame_ids", [])),
            "start_frame_0based": int(boundary["start_frame_id"]),
            "end_frame_0based_inclusive": int(boundary["end_frame_id"]),
            "end_frame_0based_exclusive": int(boundary["end_frame_id"]) + 1,
            "start_sec": float(boundary["start_frame_id"]) / float(meta["fps"]),
            "end_sec": float(int(boundary["end_frame_id"]) + 1) / float(meta["fps"]),
            "clip_materialized": True,
            "clip_path": os.path.relpath(clip_path, video_out),
            "clip_relpath": os.path.relpath(clip_path, video_out),
            "clip_local_start_frame_id": 0,
            "source_frame_id_offset": int(boundary["start_frame_id"]),
        })
    hand_payload = {
        "schema": "hand_steps.keep_runs.v2", "video_id": vid,
        "source_video": build_source_video_meta(video_path), "fps": float(meta["fps"]),
        "total_frames": int(meta["total_frames"]), "frame_index_base": 0,
        "endpoint_semantics": "inclusive", "media_artifacts_materialized": True,
        "hand_steps": reviewed_steps,
    }
    write_json(os.path.join(stage3_dir, "hand_steps.json"), hand_payload)
    plan = {"source_video": build_source_video_meta(video_path), "high_level_goal": "Primary operator hand actions in upstream keep=true intervals.", "steps": plan_steps}
    write_json(os.path.join(stage3_dir, "step_plan.json"), plan)
    return video_out, plan, usage_total


def refine_hand_steps_text_only(
    *,
    video_path: str,
    output_root: str,
    api_cfg: Any,
    overwrite: bool,
    max_retries: int,
) -> Tuple[str, Dict[str, Any], Dict[str, int]]:
    """Review each Stage-2 hand_step using in-memory ROIs and write JSON/TXT only."""
    return _refine_keep_steps_per_step_video(
        video_path=video_path,
        output_root=output_root,
        api_cfg=api_cfg,
        overwrite=overwrite,
        max_retries=max_retries,
    )
    started = time.perf_counter()
    vid = video_id_from_path(video_path)
    video_out = os.path.join(output_root, vid)
    ensure_video_out_dir_safe(video_out, video_path)
    stage2_path = os.path.join(video_out, "stage2", "hand_step_boundaries.json")
    stage3_dir = os.path.join(video_out, "stage3")
    hand_steps_path = os.path.join(stage3_dir, "hand_steps.json")
    step_plan_path = os.path.join(stage3_dir, "step_plan.json")
    run_summary_path = os.path.join(video_out, "run_summary.json")
    if not os.path.exists(stage2_path):
        raise FileNotFoundError(f"Stage 2 hand-step boundaries not found: {stage2_path}")
    if not overwrite and os.path.exists(hand_steps_path) and os.path.exists(step_plan_path):
        return video_out, read_json(step_plan_path), _usage_zero()
    boundaries = read_json(stage2_path)
    stage1_frames_path = os.path.join(video_out, "stage1", "vlm_hand_events.json")
    stage1_payload = read_json(stage1_frames_path) if os.path.exists(stage1_frames_path) else {}
    stage1_frames_by_id = {
        int(item["frame_id"]): item
        for item in stage1_payload.get("frame_annotations", [])
        if isinstance(item, dict) and "frame_id" in item
    }
    meta = get_video_metadata(video_path)
    client = initialize_api_client(api_cfg)
    if not client:
        raise SystemExit("Failed to initialize API client.")
    total_usage = _usage_zero()
    reviewed_steps: List[Dict[str, Any]] = []
    plan_steps: List[Dict[str, Any]] = []
    for step in boundaries.get("hand_steps", []):
        step_id = int(step["hand_step_id"])
        step_dir = os.path.join(stage3_dir, f"step_{step_id:03d}")
        result_path = os.path.join(step_dir, "step_goal_refined.json")
        raw_path = os.path.join(step_dir, "stage3_raw_response.txt")
        system_path = os.path.join(step_dir, "stage3_system_prompt.txt")
        user_path = os.path.join(step_dir, "stage3_user_prompt.txt")
        manifest_path = os.path.join(step_dir, "frame_manifest.json")
        os.makedirs(step_dir, exist_ok=True)
        roi_content, representative_manifest = build_hand_step_roi_content(
            video_path=video_path,
            step=step,
            meta=meta,
        )
        prompt = build_hand_step_review_prompt(step, representative_manifest, int(meta["width"]), int(meta["height"]))
        write_text(system_path, HAND_STEP_REVIEW_SYSTEM_PROMPT)
        write_text(user_path, prompt)
        write_json(
            manifest_path,
            {
                "source_video": build_source_video_meta(video_path),
                "hand_step_id": step_id,
                "frames": representative_manifest,
                "images_materialized": False,
            },
        )
        last_content = ""
        last_errors: List[str] = []
        normalized: Optional[Dict[str, Any]] = None
        for attempt in range(1, max(1, int(max_retries)) + 1):
            user_content = [{"type": "text", "text": prompt}] + roi_content
            if attempt > 1:
                user_content = [{"type": "text", "text": build_retry_prefix(last_errors, last_content) + prompt}] + roi_content
            messages = [
                {"role": "system", "content": HAND_STEP_REVIEW_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            last_content, usage = call_chat_completion(client, api_cfg, messages, max_tokens=min(api_cfg.max_tokens, 8000))
            _add_usage(total_usage, usage)
            try:
                obj = json.loads(extract_json_from_response(last_content))
                normalized, last_errors = _normalize_review(
                    obj,
                    step=step,
                    width=int(meta["width"]),
                    height=int(meta["height"]),
                    representative_frame_ids=[
                        int(item["frame_id"]) for item in representative_manifest
                    ],
                )
            except Exception as exc:
                normalized = None
                last_errors = [f"JSON parse error: {exc}"]
            if normalized is not None:
                break
        write_text(raw_path, last_content)
        if normalized is None:
            raise RuntimeError(f"Stage 3 hand_step_id={step_id} failed: " + " | ".join(last_errors[:10]))
        write_json(result_path, normalized)

        tracks = {
            "left_hand": list(step.get("bbox_tracks", {}).get("left_hand", [])),
            "right_hand": list(step.get("bbox_tracks", {}).get("right_hand", [])),
        }
        for fallback in normalized.pop("fallback_bboxes", []):
            hand_name = fallback.pop("hand")
            tracks[hand_name].append(fallback)
        for hand_name in tracks:
            tracks[hand_name].sort(key=lambda x: int(x.get("frame_id", -1)))
        bbox_evaluations, bbox_usage = evaluate_all_bbox_tracks(
            video_path=video_path,
            tracks=tracks,
            client=client,
            api_cfg=api_cfg,
            max_retries=max_retries,
        )
        for usage_key in ("prompt_tokens", "completion_tokens", "total_tokens", "api_calls"):
            total_usage[usage_key] = int(total_usage.get(usage_key, 0)) + int(bbox_usage.get(usage_key, 0))
        reviewed = {
            "hand_step_id": step_id,
            "start_frame_id": int(step["start_frame_id"]),
            "end_frame_id": int(step["end_frame_id"]),
            "end_frame_id_exclusive": int(step["end_frame_id"]) + 1,
            "frame_ids": step.get("frame_ids", []),
            "detection_sources": step.get("detection_sources", []),
            "active_hands": step.get("active_hands", []),
            "bbox_tracks": tracks,
            "action_semantics": normalized["action_semantics"],
            "hand_object_relations": normalized["hand_object_relations"],
            "visibility": normalized["visibility"],
            "frame_visibility": normalized["frame_visibility"],
            "bbox_evaluations": bbox_evaluations,
            "vlm_frame_annotations": [
                stage1_frames_by_id[frame_id]
                for frame_id in range(int(step["start_frame_id"]), int(step["end_frame_id"]) + 1)
                if frame_id in stage1_frames_by_id
            ],
        }
        reviewed_steps.append(reviewed)
        plan_steps.append(
            {
                "step_id": step_id,
                "step_goal": normalized["action_semantics"]["overall"],
                "action_semantics": normalized["action_semantics"],
                "active_hands": step.get("active_hands", []),
                "hand_object_relations": normalized["hand_object_relations"],
                "visibility": normalized["visibility"],
                "frame_visibility": normalized["frame_visibility"],
                "bbox_evaluations": bbox_evaluations,
                "start_frame_0based": int(step["start_frame_id"]),
                "end_frame_0based_inclusive": int(step["end_frame_id"]),
                "end_frame_0based_exclusive": int(step["end_frame_id"]) + 1,
                "start_sec": float(step["start_frame_id"]) / float(meta["fps"]),
                "end_sec": float(int(step["end_frame_id"]) + 1) / float(meta["fps"]),
                "clip_materialized": False,
                "clip_path": None,
                "clip_relpath": None,
            }
        )
    hand_steps_payload = {
        "schema": HAND_STEPS_SCHEMA,
        "video_id": vid,
        "source_video": build_source_video_meta(video_path),
        "fps": float(meta["fps"]),
        "total_frames": int(meta["total_frames"]),
        "frame_index_base": 0,
        "endpoint_semantics": "inclusive",
        "media_artifacts_materialized": False,
        "hand_steps": reviewed_steps,
    }
    write_json(hand_steps_path, hand_steps_payload)
    high_level_goal = "Primary operator hand actions observed across the source video."
    draft_path = os.path.join(video_out, "stage1", "draft_plan.json")
    if os.path.exists(draft_path):
        high_level_goal = str(read_json(draft_path).get("high_level_goal") or high_level_goal)
    plan = {
        "source_video": build_source_video_meta(video_path),
        "high_level_goal": high_level_goal,
        "steps": plan_steps,
    }
    write_json(step_plan_path, plan)
    update_run_summary(
        run_summary_path,
        {
            "source_video": build_source_video_meta(video_path),
            "updated_at_utc": now_utc_iso(),
            "stage3": {
                "status": "completed",
                "mode": "text_only_hand_step_roi_review",
                "hand_steps_path": os.path.relpath(hand_steps_path, video_out),
                "step_plan_path": os.path.relpath(step_plan_path, video_out),
                "media_artifacts_materialized": False,
                "token_usage": total_usage,
            },
        },
    )
    logger.info(
        f"[stage3] video_id={vid} completed text-only hand_steps={len(reviewed_steps)} "
        f"elapsed={format_duration(time.perf_counter() - started)}"
    )
    return video_out, plan, total_usage
