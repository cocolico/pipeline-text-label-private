#!/usr/bin/env python3
"""Browser-based, zero-artifact hand-step visualization.

The server decodes the original RGB video, overlays upstream SAM3 and VLM
grounding bboxes on the same source frame, appends a semantic panel outside
the source image, and streams the result as in-memory MJPEG.  It never writes
rendered frames, images, or videos to local storage.
"""

from __future__ import annotations

import argparse
import bisect
import html
import json
import os
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except Exception as exc:  # pragma: no cover - checked again in main
    cv2 = None
    np = None
    _CV_IMPORT_ERROR = exc
else:
    _CV_IMPORT_ERROR = None


SAM3_COLOR = (60, 220, 60)  # BGR green
VLM_COLOR = (230, 80, 230)  # BGR magenta
PANEL_BG = (24, 27, 33)
PANEL_TEXT = (235, 238, 242)
PANEL_MUTED = (160, 169, 180)


@dataclass(frozen=True)
class FrameBox:
    hand: str
    bbox_xyxy: Tuple[int, int, int, int]
    source: str
    arm_visible: Optional[int] = None
    hand_completeness_score: Optional[float] = None


class HandStepIndex:
    """In-memory interval and exact-frame bbox index for hand_steps.json."""

    def __init__(self, payload: Dict[str, Any]) -> None:
        raw_steps = payload.get("hand_steps", [])
        if not isinstance(raw_steps, list):
            raise ValueError("hand_steps.json must contain a 'hand_steps' array")
        self.steps: List[Dict[str, Any]] = []
        self.starts: List[int] = []
        self.boxes_by_frame: Dict[int, List[FrameBox]] = {}
        for raw in raw_steps:
            if not isinstance(raw, dict):
                continue
            try:
                start = int(raw["start_frame_id"])
                end = int(raw["end_frame_id"])
            except Exception as exc:
                raise ValueError("Each hand_step needs integer start_frame_id/end_frame_id") from exc
            if start < 0 or end < start:
                raise ValueError(f"Invalid hand_step interval: [{start}, {end}]")
            step = dict(raw)
            step["start_frame_id"] = start
            step["end_frame_id"] = end
            self.steps.append(step)
        self.steps.sort(key=lambda item: (item["start_frame_id"], item["end_frame_id"]))
        self.starts = [int(item["start_frame_id"]) for item in self.steps]

        evaluation_lookup: Dict[Tuple[int, str, str], Dict[str, Any]] = {}
        for step in self.steps:
            for evaluation in step.get("bbox_evaluations", []) if isinstance(step.get("bbox_evaluations"), list) else []:
                if not isinstance(evaluation, dict):
                    continue
                try:
                    key = (
                        int(evaluation["frame_id"]), str(evaluation["hand"]),
                        self._normalize_source(evaluation.get("bbox_source")),
                    )
                except Exception:
                    continue
                evaluation_lookup[key] = evaluation
        for step in self.steps:
            tracks = step.get("bbox_tracks", {})
            if not isinstance(tracks, dict):
                continue
            for hand in ("left_hand", "right_hand"):
                entries = tracks.get(hand, [])
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    parsed = self._parse_box(entry, hand, evaluation_lookup)
                    if parsed is None:
                        continue
                    frame_id, box = parsed
                    existing = self.boxes_by_frame.setdefault(frame_id, [])
                    key = (box.hand, box.bbox_xyxy, box.source)
                    if not any((x.hand, x.bbox_xyxy, x.source) == key for x in existing):
                        existing.append(box)

    @staticmethod
    def _normalize_source(raw_source: Any) -> str:
        raw_source = str(raw_source or "").strip().lower()
        return "VLM_generated_bbox" if raw_source in {"vlm_grounding", "vlm_generated", "vlm"} else "SAM3_bbox"

    @classmethod
    def _parse_box(
        cls, entry: Any, hand: str,
        evaluation_lookup: Optional[Dict[Tuple[int, str, str], Dict[str, Any]]] = None,
    ) -> Optional[Tuple[int, FrameBox]]:
        if not isinstance(entry, dict):
            return None
        try:
            frame_id = int(entry["frame_id"])
            coords = tuple(int(round(float(x))) for x in entry["bbox_xyxy"])
        except Exception:
            return None
        if len(coords) != 4 or coords[2] <= coords[0] or coords[3] <= coords[1]:
            return None
        source = cls._normalize_source(entry.get("bbox_source"))
        evaluation = (evaluation_lookup or {}).get((frame_id, hand, source), {})
        arm_raw = evaluation.get("arm_visible") if isinstance(evaluation, dict) else None
        score_raw = evaluation.get("hand_completeness_score") if isinstance(evaluation, dict) else None
        arm_visible = int(arm_raw) if arm_raw in {0, 1, False, True} else None
        try:
            score = min(1.0, max(0.0, float(score_raw)))
        except (TypeError, ValueError):
            score = None
        return frame_id, FrameBox(
            hand=hand, bbox_xyxy=coords, source=source,
            arm_visible=arm_visible, hand_completeness_score=score,
        )

    def step_for_frame(self, frame_id: int) -> Optional[Dict[str, Any]]:
        pos = bisect.bisect_right(self.starts, int(frame_id)) - 1
        if pos < 0:
            return None
        # Normally intervals do not overlap. Walking backwards also makes the
        # visualizer deterministic for legacy files with accidental overlap.
        for idx in range(pos, -1, -1):
            step = self.steps[idx]
            if int(step["end_frame_id"]) >= frame_id:
                return step
            if int(step["start_frame_id"]) < frame_id:
                break
        return None

    def boxes_for_frame(self, frame_id: int) -> Sequence[FrameBox]:
        return self.boxes_by_frame.get(int(frame_id), ())

    def details_for_frame(self, frame_id: int) -> Dict[str, Any]:
        """Return all useful frame/step annotations without returning huge full tracks."""
        frame_id = int(frame_id)
        step = self.step_for_frame(frame_id)
        boxes = list(self.boxes_for_frame(frame_id))
        frame_annotation: Dict[str, Any] = {}
        exact_visibility: Dict[str, Any] = {}
        if isinstance(step, dict):
            for item in step.get("vlm_frame_annotations", []) if isinstance(step.get("vlm_frame_annotations"), list) else []:
                if isinstance(item, dict) and int(item.get("frame_id", -1)) == frame_id:
                    frame_annotation = item
                    break
            for item in step.get("frame_visibility", []) if isinstance(step.get("frame_visibility"), list) else []:
                if isinstance(item, dict) and int(item.get("frame_id", -1)) == frame_id:
                    exact_visibility = item
                    break
        box_records = [
            {
                "frame_id": frame_id,
                "hand": box.hand,
                "bbox_source": box.source,
                "bbox_xyxy": list(box.bbox_xyxy),
                "arm_visible": box.arm_visible,
                "hand_completeness_score": box.hand_completeness_score,
            }
            for box in boxes
        ]
        if not isinstance(step, dict):
            return {
                "frame_id": frame_id, "inside_hand_step": False,
                "vlm_frame_annotation": {}, "boxes": box_records,
                "bbox_evaluations": box_records, "step": None,
            }
        tracks = step.get("bbox_tracks", {}) if isinstance(step.get("bbox_tracks"), dict) else {}
        source_counts: Dict[str, int] = {}
        for hand in ("left_hand", "right_hand"):
            for item in tracks.get(hand, []) if isinstance(tracks.get(hand), list) else []:
                source = self._normalize_source(item.get("bbox_source")) if isinstance(item, dict) else "unknown"
                source_counts[source] = source_counts.get(source, 0) + 1
        step_summary = {
            "hand_step_id": step.get("hand_step_id", step.get("step_id")),
            "start_frame_id": step.get("start_frame_id"),
            "end_frame_id": step.get("end_frame_id"),
            "end_frame_id_exclusive": step.get("end_frame_id_exclusive"),
            "active_hands": step.get("active_hands", []),
            "detection_sources": step.get("detection_sources", []),
            "action_semantics": step.get("action_semantics", {}),
            "hand_object_relations": step.get("hand_object_relations", []),
            "visibility_summary": step.get("visibility", {}),
            "exact_frame_visibility_compatibility": exact_visibility,
            "bbox_track_counts": source_counts,
            "frame_count": int(step["end_frame_id"]) - int(step["start_frame_id"]) + 1,
        }
        return {
            "frame_id": frame_id,
            "inside_hand_step": True,
            "vlm_frame_annotation": frame_annotation,
            "boxes": box_records,
            "bbox_evaluations": box_records,
            "step": step_summary,
        }

    def step_summaries(self) -> List[Dict[str, Any]]:
        return [
            {
                "hand_step_id": step.get("hand_step_id", step.get("step_id")),
                "start_frame_id": int(step["start_frame_id"]),
                "end_frame_id": int(step["end_frame_id"]),
                "active_hands": step.get("active_hands", []),
                "detection_sources": step.get("detection_sources", []),
                "overall": (step.get("action_semantics", {}) or {}).get("overall", ""),
            }
            for step in self.steps
        ]


def infer_hand_steps_path(video_path: str) -> str:
    raw_dir = os.path.dirname(os.path.abspath(video_path))
    recording_dir = os.path.dirname(raw_dir)
    candidates = [
        os.path.join(recording_dir, "processed_text_label", "rgb", "stage3", "hand_steps.json"),
        os.path.join(recording_dir, "processed_text_label", "stage3", "hand_steps.json"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "Cannot infer hand_steps.json. Checked:\n" + "\n".join(f"- {x}" for x in candidates)
    )


def load_pipeline_artifacts(hand_steps_path: str) -> Dict[str, Any]:
    """Load useful sibling JSON products while trimming duplicated per-frame bulk."""
    stage3_dir = os.path.dirname(os.path.abspath(hand_steps_path))
    video_out = os.path.dirname(stage3_dir)
    candidates = {
        "final_plan": os.path.join(video_out, "final_plan.json"),
        "run_summary": os.path.join(video_out, "run_summary.json"),
        "draft_plan": os.path.join(video_out, "stage1", "draft_plan.json"),
        "vlm_hand_frames": os.path.join(video_out, "stage1", "vlm_hand_events.json"),
        "hand_step_boundaries": os.path.join(video_out, "stage2", "hand_step_boundaries.json"),
        "step_segments": os.path.join(video_out, "stage2", "step_segments.json"),
        "step_plan": os.path.join(stage3_dir, "step_plan.json"),
        "instruction_subtask": os.path.join(stage3_dir, "instruction_subtask.json"),
        "object_inventory": os.path.join(stage3_dir, "object_inventory.json"),
    }
    out: Dict[str, Any] = {}
    for name, path in candidates.items():
        if not os.path.isfile(path):
            out[name] = {"available": False, "path": path}
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            out[name] = {"available": False, "path": path, "error": str(exc)}
            continue
        # Frame annotations and boundary hand_steps are already available via
        # the exact-frame endpoint and hand_steps.json. Avoid duplicating many
        # megabytes in the browser's global-artifact request.
        if name == "vlm_hand_frames" and isinstance(data, dict):
            frame_annotations = data.get("frame_annotations", [])
            data = {key: value for key, value in data.items() if key != "frame_annotations"}
            data["frame_annotation_count"] = len(frame_annotations) if isinstance(frame_annotations, list) else 0
            data["frame_annotations_note"] = "Inspect exact records by moving to a frame in the viewer."
        if name == "hand_step_boundaries" and isinstance(data, dict):
            hand_steps = data.get("hand_steps", [])
            data = {key: value for key, value in data.items() if key != "hand_steps"}
            data["hand_step_count"] = len(hand_steps) if isinstance(hand_steps, list) else 0
            data["hand_steps_note"] = "Step summaries are shown in the left-hand list."
        out[name] = {"available": True, "path": path, "data": data}
    return out


def _scale_bbox(box: Tuple[int, int, int, int], scale: float, width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = (int(round(value * scale)) for value in box)
    x1 = min(max(0, x1), max(0, width - 1))
    y1 = min(max(0, y1), max(0, height - 1))
    x2 = min(max(x1 + 1, x2), width)
    y2 = min(max(y1 + 1, y2), height)
    return x1, y1, x2, y2


def _draw_box(frame: Any, box: FrameBox, scale: float) -> None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = _scale_bbox(box.bbox_xyxy, scale, width, height)
    color = VLM_COLOR if box.source == "VLM_generated_bbox" else SAM3_COLOR
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
    arm = "N/A" if box.arm_visible is None else ("yes" if box.arm_visible else "no")
    exposure = "N/A" if box.hand_completeness_score is None else f"{box.hand_completeness_score:.2f}"
    label = f"{box.source} | {box.hand} | arm:{arm} exposure:{exposure}"
    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    label_top = max(0, y1 - text_h - baseline - 8)
    cv2.rectangle(frame, (x1, label_top), (min(width - 1, x1 + text_w + 10), y1), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 5, max(text_h + 2, y1 - baseline - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (15, 15, 15),
        1,
        cv2.LINE_AA,
    )


def _wrap_text(text: str, max_width: int, font_scale: float, thickness: int) -> List[str]:
    words = str(text or "").replace("\n", " ").split()
    if not words:
        return ["-"]
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        trial = current + " " + word
        width = cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0]
        if width <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _draw_text_block(panel: Any, y: int, title: str, text: str, color: Tuple[int, int, int]) -> int:
    margin = 24
    cv2.putText(panel, title, (margin, y), cv2.FONT_HERSHEY_SIMPLEX, 0.66, color, 2, cv2.LINE_AA)
    y += 28
    for line in _wrap_text(text, panel.shape[1] - margin * 2, 0.56, 1):
        cv2.putText(panel, line, (margin, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, PANEL_TEXT, 1, cv2.LINE_AA)
        y += 24
    return y + 18


def visibility_summary(step: Optional[Dict[str, Any]], hand: str, frame_id: int) -> str:
    """Format an exact representative-frame score; never smear step summaries."""
    frame_visibility = step.get("frame_visibility", []) if isinstance(step, dict) else []
    record: Dict[str, Any] = {}
    if isinstance(frame_visibility, list):
        for item in frame_visibility:
            if not isinstance(item, dict):
                continue
            try:
                item_frame_id = int(item.get("frame_id"))
            except Exception:
                continue
            if item_frame_id == int(frame_id):
                record = item
                break
    if hand in {"left_hand", "right_hand"}:
        value = record.get(hand, {})
        if not isinstance(value, dict):
            value = {}
        arm_raw = value.get("arm_visible")
        score_raw = value.get("hand_completeness_score")
    else:
        left = record.get("left_hand", {}) if isinstance(record.get("left_hand"), dict) else {}
        right = record.get("right_hand", {}) if isinstance(record.get("right_hand"), dict) else {}
        active_hands = step.get("active_hands", []) if isinstance(step, dict) else []
        active_hands = [name for name in active_hands if name in {"left_hand", "right_hand"}]
        if not active_hands:
            active_hands = ["left_hand", "right_hand"]
        hand_records = {"left_hand": left, "right_hand": right}
        participating_records = [hand_records[name] for name in active_hands]
        arm_values = [item.get("arm_visible") for item in participating_records]
        arm_raw = 1 if any(value in {1, True} for value in arm_values) else (0 if record else None)
        scores: List[float] = []
        for item in participating_records:
            try:
                scores.append(float(item.get("hand_completeness_score")))
            except (TypeError, ValueError):
                pass
        score_raw = min(scores) if scores else None
    arm_text = "N/A" if arm_raw not in {0, 1, False, True} else ("yes" if int(arm_raw) == 1 else "no")
    try:
        score = float(score_raw)
    except (TypeError, ValueError):
        score_text = "N/A"
    else:
        score_text = f"{min(1.0, max(0.0, score)):.2f}"
    arm_label = "arm any" if hand == "overall" else "arm"
    exposure_label = "exposure"
    suffix = "" if record else " | representative frame: no"
    return f"{arm_label}: {arm_text} | {exposure_label}: {score_text}{suffix}"


def bbox_source_summary(boxes: Sequence[FrameBox], hand: str, source: str) -> str:
    box = next((item for item in boxes if item.hand == hand and item.source == source), None)
    if box is None:
        return f"{source}: N/A"
    arm = "N/A" if box.arm_visible is None else ("yes" if box.arm_visible else "no")
    exposure = "N/A" if box.hand_completeness_score is None else f"{box.hand_completeness_score:.2f}"
    return f"{source}: arm {arm}, exposure {exposure}"


def render_frame(
    source_frame: Any,
    *,
    frame_id: int,
    index: HandStepIndex,
    max_video_width: int = 1280,
    panel_width: int = 620,
) -> Any:
    """Return one annotated composite frame; no filesystem writes occur."""
    if cv2 is None or np is None:  # pragma: no cover
        raise RuntimeError(f"OpenCV/NumPy unavailable: {_CV_IMPORT_ERROR}")
    if source_frame is None or not hasattr(source_frame, "shape"):
        raise ValueError("source_frame is empty")
    src_h, src_w = source_frame.shape[:2]
    scale = min(1.0, float(max_video_width) / float(src_w)) if max_video_width > 0 else 1.0
    view_w = max(1, int(round(src_w * scale)))
    view_h = max(1, int(round(src_h * scale)))
    if scale != 1.0:
        view = cv2.resize(source_frame, (view_w, view_h), interpolation=cv2.INTER_AREA)
    else:
        view = source_frame.copy()

    boxes = index.boxes_for_frame(frame_id)
    for box in boxes:
        _draw_box(view, box, scale)

    panel = np.full((view_h, max(360, int(panel_width)), 3), PANEL_BG, dtype=np.uint8)
    step = index.step_for_frame(frame_id)
    step_id = "-" if step is None else str(step.get("hand_step_id", step.get("step_id", "-")))
    cv2.putText(panel, f"Source frame: {frame_id}", (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.72, PANEL_TEXT, 2, cv2.LINE_AA)
    cv2.putText(panel, f"hand_step: {step_id}", (24, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.64, PANEL_MUTED, 1, cv2.LINE_AA)
    cv2.line(panel, (24, 88), (panel.shape[1] - 24, 88), (70, 76, 86), 1)

    frame_annotation: Dict[str, Any] = {}
    if isinstance(step, dict):
        for item in step.get("vlm_frame_annotations", []) if isinstance(step.get("vlm_frame_annotations"), list) else []:
            if isinstance(item, dict) and int(item.get("frame_id", -1)) == int(frame_id):
                frame_annotation = item
                break
    semantics = frame_annotation.get("action_semantics", {}) if frame_annotation else (
        step.get("action_semantics", {}) if isinstance(step, dict) else {}
    )
    if not isinstance(semantics, dict):
        semantics = {}
    action_flag = frame_annotation.get("has_effective_hand_action") if frame_annotation else None
    action_text = "N/A" if action_flag not in {0, 1, False, True} else ("yes" if int(action_flag) else "no")
    cv2.putText(
        panel, f"VLM effective hand action: {action_text}", (24, 106),
        cv2.FONT_HERSHEY_SIMPLEX, 0.58, PANEL_TEXT, 1, cv2.LINE_AA,
    )
    y = 138
    y = _draw_text_block(
        panel,
        y,
        "left_hand | " + bbox_source_summary(boxes, "left_hand", "SAM3_bbox") +
        " | " + bbox_source_summary(boxes, "left_hand", "VLM_generated_bbox"),
        str(semantics.get("left_hand") or "No active hand_step"),
        (90, 190, 255),
    )
    y = _draw_text_block(
        panel,
        y,
        "right_hand | " + bbox_source_summary(boxes, "right_hand", "SAM3_bbox") +
        " | " + bbox_source_summary(boxes, "right_hand", "VLM_generated_bbox"),
        str(semantics.get("right_hand") or "No active hand_step"),
        (255, 180, 80),
    )
    _draw_text_block(
        panel,
        y,
        f"overall | {visibility_summary(step, 'overall', frame_id)}",
        str(semantics.get("overall") or "No active hand_step"),
        (220, 220, 100),
    )

    legend_y = max(110, view_h - 58)
    cv2.rectangle(panel, (24, legend_y), (44, legend_y + 16), SAM3_COLOR, -1)
    cv2.putText(panel, "SAM3_bbox", (54, legend_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, PANEL_TEXT, 1, cv2.LINE_AA)
    cv2.rectangle(panel, (200, legend_y), (220, legend_y + 16), VLM_COLOR, -1)
    cv2.putText(panel, "VLM_generated_bbox", (230, legend_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, PANEL_TEXT, 1, cv2.LINE_AA)
    cv2.putText(
        panel,
        f"boxes on this frame: {len(boxes)}",
        (24, legend_y + 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        PANEL_MUTED,
        1,
        cv2.LINE_AA,
    )
    return np.concatenate([view, panel], axis=1)


def _html_page(video_path: str, hand_steps_path: str) -> bytes:
    body = """<!doctype html>
<html><head><meta charset="utf-8"><title>TextLabel comprehensive visualization</title>
<style>
:root { color-scheme:dark; --bg:#0c0f14; --card:#171c24; --line:#2d3542; --muted:#9da8b7; --text:#edf1f7; --green:#3cdc5a; --magenta:#e650e6; --blue:#62a8ff; }
* { box-sizing:border-box; } body { margin:0; background:var(--bg); color:var(--text); font-family:Inter,system-ui,sans-serif; }
header { padding:14px 20px; background:#131820; border-bottom:1px solid var(--line); position:sticky; top:0; z-index:4; }
h1 { margin:0 0 7px; font-size:20px; } .meta { color:var(--muted); font-size:12px; overflow-wrap:anywhere; }
main { padding:16px; } .controls { display:flex; align-items:center; gap:9px; flex-wrap:wrap; margin-bottom:14px; padding:12px; background:var(--card); border:1px solid var(--line); border-radius:9px; }
button,select,input[type=number] { background:#222936; color:var(--text); border:1px solid #485366; border-radius:5px; padding:7px 10px; }
button:hover { background:#303a49; } #progress { flex:1 1 480px; min-width:240px; } #position { min-width:255px; font-variant-numeric:tabular-nums; }
.layout { display:grid; grid-template-columns:minmax(520px,1.65fr) minmax(390px,1fr); gap:14px; align-items:start; }
.viewer { position:sticky; top:126px; } #view { display:block; width:100%; height:auto; background:#000; border:1px solid var(--line); border-radius:8px; box-shadow:0 8px 28px #0008; }
.legend { display:flex; gap:18px; margin:9px 2px; color:var(--muted); font-size:13px; }.swatch { display:inline-block; width:14px; height:14px; margin-right:6px; vertical-align:-2px; }.sam3{background:var(--green)}.vlm{background:var(--magenta)}
.dashboard { display:grid; gap:10px; }.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:13px; min-width:0; }.card h2 { margin:0 0 10px; font-size:15px; color:#cdd8e6; }
.badges { display:flex; flex-wrap:wrap; gap:6px; }.badge { padding:3px 8px; border-radius:999px; background:#273142; font-size:12px; }.yes{background:#173b26;color:#76ec9a}.no{background:#452127;color:#ff9ba8}.na{color:var(--muted)}
.kv { display:grid; grid-template-columns:minmax(110px,.7fr) minmax(0,1.3fr); gap:6px 10px; font-size:13px; }.k { color:var(--muted); }.v { overflow-wrap:anywhere; }
.semantic { border-left:3px solid var(--blue); padding-left:9px; margin:7px 0; }.semantic strong { display:block; color:#a9cfff; font-size:12px; margin-bottom:3px; }
.bbox { border:1px solid var(--line); border-left:4px solid var(--green); border-radius:6px; padding:9px; margin:7px 0; font-size:12px; }.bbox.vlm-box { border-left-color:var(--magenta); }.bbox-head { display:flex; justify-content:space-between; gap:8px; font-weight:650; }.score { font-variant-numeric:tabular-nums; }
table { width:100%; border-collapse:collapse; font-size:12px; } th,td { text-align:left; vertical-align:top; border-bottom:1px solid var(--line); padding:6px; overflow-wrap:anywhere; } th { color:var(--muted); }
details summary { cursor:pointer; color:#b9c7da; } pre { white-space:pre-wrap; overflow-wrap:anywhere; max-height:420px; overflow:auto; font-size:11px; color:#b9c5d4; }
#stepList { max-height:260px; overflow:auto; }.step-row { padding:7px; border-bottom:1px solid var(--line); cursor:pointer; }.step-row:hover { background:#222a36; }.step-row.active { background:#24364e; }
@media(max-width:1050px){.layout{grid-template-columns:1fr}.viewer{position:static}}
</style></head><body>
<header><h1>TextLabel comprehensive visualization</h1><div class="meta">Original video: __VIDEO__</div><div class="meta">Annotations: __ANNOTATIONS__</div></header>
<main><div class="controls"><button id="play">Play</button><button id="prev">-1 frame</button><button id="next">+1 frame</button><input id="progress" type="range" min="0" max="0" value="0"><span id="position">frame 0 / 0</span><label>Speed <select id="speed"><option value="0.25">0.25x</option><option value="0.5">0.5x</option><option value="1" selected>1x</option><option value="2">2x</option><option value="4">4x</option><option value="8">8x</option><option value="16">16x</option><option value="32">32x</option><option value="64">64x</option><option value="128">128x</option><option value="256">256x</option></select></label><label>Go to <input id="goto" type="number" min="0" value="0"></label><button id="go">Go</button></div>
<div class="layout"><section class="viewer"><img id="view" alt="annotated RGB frame"><div class="legend"><span><i class="swatch sam3"></i>SAM3_bbox</span><span><i class="swatch vlm"></i>VLM_generated_bbox</span><span>Images/videos are not saved</span></div><div class="card"><h2>All hand_steps</h2><div id="stepList"></div></div></section>
<aside class="dashboard"><div class="card"><h2>Current frame</h2><div id="frameSummary" class="kv"></div></div><div class="card"><h2>Frame-level VLM decision</h2><div id="vlmDecision"></div></div><div class="card"><h2>Left / right / overall semantics</h2><div id="semantics"></div></div><div class="card"><h2>Bboxes and source-specific quality</h2><div id="boxes"></div></div><div class="card"><h2>Current hand_step</h2><div id="stepSummary"></div></div><div class="card"><h2>Hand-object relations</h2><div id="relations"></div></div><div class="card"><h2>Compatibility summaries</h2><div id="compat"></div></div><div class="card"><h2>Pipeline JSON artifacts</h2><select id="artifactSelect"></select><div id="artifactStatus" class="meta"></div><pre id="artifactJson"></pre></div><details class="card"><summary>Raw current-frame JSON</summary><pre id="rawJson"></pre></details><details class="card"><summary>Video / run metadata</summary><pre id="metaJson"></pre></details></aside></div></main>
<script>
const $=id=>document.getElementById(id), playButton=$('play'), progress=$('progress'), position=$('position'), view=$('view'), speed=$('speed'); let meta=null,playing=false,loading=false,timer=null,sequence=0,currentDetails=null;
const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const badge=(text,kind='')=>`<span class="badge ${kind}">${esc(text)}</span>`;
function kv(items){return items.map(([k,v])=>`<div class="k">${esc(k)}</div><div class="v">${v}</div>`).join('')}
function renderDetails(d){currentDetails=d;const a=d.vlm_frame_annotation||{},s=d.step||{},sem=a.action_semantics||s.action_semantics||{};const flag=a.has_effective_hand_action;
 $('frameSummary').innerHTML=kv([['frame_id',esc(d.frame_id)],['time',`${Number(d.time_sec||0).toFixed(3)} s`],['inside hand_step',badge(d.inside_hand_step?'yes':'no',d.inside_hand_step?'yes':'no')],['hand_step_id',esc(s.hand_step_id??'N/A')],['bbox count',esc((d.boxes||[]).length)]]);
 $('vlmDecision').innerHTML=`<div class="badges">${badge(flag===1?'effective action: yes':flag===0?'effective action: no':'effective action: N/A',flag===1?'yes':flag===0?'no':'na')}${badge(`confidence: ${a.confidence??'N/A'}`)}${(a.active_hands||[]).map(x=>badge(x)).join('')}</div>`;
 $('semantics').innerHTML=['left_hand','right_hand','overall'].map(k=>`<div class="semantic"><strong>${k}</strong>${esc(sem[k]||'N/A')}</div>`).join('');
 const boxes=d.boxes||[];$('boxes').innerHTML=boxes.length?boxes.map(b=>`<div class="bbox ${b.bbox_source==='VLM_generated_bbox'?'vlm-box':''}"><div class="bbox-head"><span>${esc(b.bbox_source)} · ${esc(b.hand)}</span><span>${esc(JSON.stringify(b.bbox_xyxy))}</span></div><div class="score">arm_visible: ${b.arm_visible===null?'N/A':b.arm_visible} · hand_completeness_score: ${b.hand_completeness_score===null?'N/A':Number(b.hand_completeness_score).toFixed(2)}</div></div>`).join(''):'<span class="na">No bbox on this frame</span>';
 $('stepSummary').innerHTML=d.inside_hand_step?`<div class="kv">${kv([['interval',`${s.start_frame_id} – ${s.end_frame_id} (inclusive)`],['frame count',esc(s.frame_count)],['active hands',(s.active_hands||[]).map(x=>badge(x)).join(' ')||'N/A'],['detection sources',(s.detection_sources||[]).map(x=>badge(x)).join(' ')||'N/A'],['bbox track counts',esc(JSON.stringify(s.bbox_track_counts||{}))]])}</div>`:'<span class="na">Outside every hand_step</span>';
 const rel=s.hand_object_relations||[];$('relations').innerHTML=rel.length?`<table><thead><tr><th>hand</th><th>relation</th><th>object</th><th>entity_ref</th></tr></thead><tbody>${rel.map(r=>`<tr><td>${esc(r.hand)}</td><td>${esc(r.relation)}</td><td>${esc(r.object)}</td><td>${esc(r.entity_ref??'N/A')}</td></tr>`).join('')}</tbody></table>`:'<span class="na">No hand-object relation</span>';
 $('compat').innerHTML=`<div class="kv">${kv([['step visibility',`<pre>${esc(JSON.stringify(s.visibility_summary||{},null,2))}</pre>`],['exact frame visibility',`<pre>${esc(JSON.stringify(s.exact_frame_visibility_compatibility||{},null,2))}</pre>`]])}</div>`;$('rawJson').textContent=JSON.stringify(d,null,2);highlightStep(s.hand_step_id);
}
function highlightStep(id){document.querySelectorAll('.step-row').forEach(x=>x.classList.toggle('active',String(x.dataset.id)===String(id)))}
function renderStepList(){ $('stepList').innerHTML=(meta.hand_steps||[]).map(s=>`<div class="step-row" data-id="${esc(s.hand_step_id)}" data-frame="${s.start_frame_id}"><strong>Step ${esc(s.hand_step_id)}</strong> · ${s.start_frame_id}–${s.end_frame_id}<br><span class="meta">${esc(s.overall||'')}</span></div>`).join('')||'<span class="na">No hand_steps</span>';document.querySelectorAll('.step-row').forEach(x=>x.onclick=()=>{playing=false;playButton.textContent='Play';showFrame(Number(x.dataset.frame))})}
function renderArtifacts(artifacts){const select=$('artifactSelect'),names=Object.keys(artifacts);select.innerHTML=names.map(name=>`<option value="${esc(name)}">${esc(name)}${artifacts[name].available?'':' (missing)'}</option>`).join('');const show=()=>{const item=artifacts[select.value]||{};$('artifactStatus').textContent=`${item.available?'available':'missing'} · ${item.path||''}`;$('artifactJson').textContent=JSON.stringify(item.data??item,null,2)};select.onchange=show;show()}
function showFrame(frame){if(!meta)return;frame=Math.max(0,Math.min(meta.total_frames-1,Math.round(frame)));progress.value=String(frame);$('goto').value=String(frame);position.textContent=`frame ${frame} / ${meta.total_frames-1} | ${(frame/meta.fps).toFixed(3)}s`;loading=true;const id=++sequence;const image=new Image();image.onload=()=>{if(id===sequence)view.src=image.src;loading=false;if(playing)scheduleNext()};image.onerror=()=>{loading=false;playing=false;playButton.textContent='Play'};image.src=`/frame.jpg?frame_id=${frame}&v=${id}`;fetch(`/annotation.json?frame_id=${frame}`,{cache:'no-store'}).then(r=>r.json()).then(d=>{if(id===sequence)renderDetails(d)})}
function scheduleNext(){clearTimeout(timer);if(!playing||loading)return;const rate=Number(speed.value),delay=1000/(meta.playback_fps*Math.min(rate,1)),frameAdvance=Math.max(1,Math.round(meta.playback_frame_stride*Math.max(rate,1)));timer=setTimeout(()=>{let next=Number(progress.value)+frameAdvance;if(next>=meta.total_frames)next=0;showFrame(next)},delay)}
playButton.onclick=()=>{playing=!playing;playButton.textContent=playing?'Pause':'Play';playing?scheduleNext():clearTimeout(timer)};progress.oninput=()=>{playing=false;playButton.textContent='Play';clearTimeout(timer);showFrame(Number(progress.value))};$('prev').onclick=()=>{playing=false;playButton.textContent='Play';showFrame(Number(progress.value)-1)};$('next').onclick=()=>{playing=false;playButton.textContent='Play';showFrame(Number(progress.value)+1)};$('go').onclick=()=>showFrame(Number($('goto').value));speed.onchange=()=>{if(playing)scheduleNext()};document.addEventListener('keydown',e=>{if(e.code==='Space'){e.preventDefault();playButton.click()}if(e.code==='ArrowLeft')$('prev').click();if(e.code==='ArrowRight')$('next').click()});
fetch('/metadata.json',{cache:'no-store'}).then(r=>r.json()).then(data=>{meta=data;progress.max=String(Math.max(0,data.total_frames-1));$('goto').max=progress.max;$('metaJson').textContent=JSON.stringify(data,null,2);renderStepList();showFrame(data.start_frame)});
fetch('/artifacts.json',{cache:'no-store'}).then(r=>r.json()).then(renderArtifacts);
</script></body></html>"""
    body = body.replace("__VIDEO__", html.escape(video_path)).replace("__ANNOTATIONS__", html.escape(hand_steps_path))
    return body.encode("utf-8")


class VisualizationApp:
    def __init__(
        self,
        *,
        video_path: str,
        hand_steps_path: str,
        index: HandStepIndex,
        render_fps: float,
        jpeg_quality: int,
        max_video_width: int,
        panel_width: int,
        loop: bool,
        start_frame: int,
    ) -> None:
        self.video_path = video_path
        self.hand_steps_path = hand_steps_path
        self.index = index
        self.render_fps = render_fps
        self.jpeg_quality = jpeg_quality
        self.max_video_width = max_video_width
        self.panel_width = panel_width
        self.loop = loop
        self.start_frame = start_frame
        self.artifacts = load_pipeline_artifacts(hand_steps_path)
        self._capture_lock = threading.Lock()
        self._capture = cv2.VideoCapture(self.video_path)
        if not self._capture.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
        self.source_fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if self.source_fps <= 0:
            self.source_fps = 30.0
        self.total_frames = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if self.total_frames <= 0:
            raise RuntimeError(f"Video has invalid frame count: {self.video_path}")
        playback_fps = self.source_fps if self.render_fps <= 0 else min(self.source_fps, self.render_fps)
        self.playback_frame_stride = max(1, int(round(self.source_fps / playback_fps)))
        self.playback_fps = self.source_fps / self.playback_frame_stride

    def metadata(self) -> Dict[str, Any]:
        return {
            "fps": float(self.source_fps),
            "total_frames": int(self.total_frames),
            "duration_sec": float(self.total_frames) / float(self.source_fps),
            "start_frame": min(max(0, int(self.start_frame)), self.total_frames - 1),
            "playback_fps": float(self.playback_fps),
            "playback_frame_stride": int(self.playback_frame_stride),
            "render_storage": "memory_only",
            "video_path": self.video_path,
            "hand_steps_path": self.hand_steps_path,
            "hand_step_count": len(self.index.steps),
            "hand_steps": self.index.step_summaries(),
        }

    def annotation(self, frame_id: int) -> Dict[str, Any]:
        frame_id = min(max(0, int(frame_id)), self.total_frames - 1)
        payload = self.index.details_for_frame(frame_id)
        payload["time_sec"] = float(frame_id) / float(self.source_fps)
        return payload

    def artifact_payload(self) -> Dict[str, Any]:
        return self.artifacts

    def render_jpeg(self, frame_id: int) -> bytes:
        frame_id = min(max(0, int(frame_id)), self.total_frames - 1)
        with self._capture_lock:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Cannot decode source frame_id={frame_id}")
        rendered = render_frame(
            frame,
            frame_id=frame_id,
            index=self.index,
            max_video_width=self.max_video_width,
            panel_width=self.panel_width,
        )
        ok_encode, encoded = cv2.imencode(
            ".jpg", rendered, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)]
        )
        if not ok_encode:
            raise RuntimeError(f"Cannot encode rendered frame_id={frame_id}")
        return encoded.tobytes()

    def close(self) -> None:
        with self._capture_lock:
            self._capture.release()

    def mjpeg(self) -> Iterable[bytes]:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if source_fps <= 0:
            source_fps = 30.0
        output_fps = source_fps if self.render_fps <= 0 else min(source_fps, self.render_fps)
        frame_stride = max(1, int(round(source_fps / output_fps)))
        period = float(frame_stride) / source_fps
        frame_id = max(0, int(self.start_frame))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        deadline = time.monotonic()
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    if not self.loop:
                        return
                    frame_id = max(0, int(self.start_frame))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                    deadline = time.monotonic()
                    continue
                if frame_id % frame_stride != 0:
                    frame_id += 1
                    continue
                rendered = render_frame(
                    frame,
                    frame_id=frame_id,
                    index=self.index,
                    max_video_width=self.max_video_width,
                    panel_width=self.panel_width,
                )
                ok_encode, encoded = cv2.imencode(
                    ".jpg", rendered, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)]
                )
                frame_id += 1
                if not ok_encode:
                    continue
                deadline += period
                wait_sec = deadline - time.monotonic()
                if wait_sec > 0:
                    time.sleep(wait_sec)
                else:
                    deadline = time.monotonic()
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-store\r\n\r\n"
                    + encoded.tobytes()
                    + b"\r\n"
                )
        finally:
            cap.release()


def make_handler(app: VisualizationApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "HandStepVisualizer/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[visualizer] {self.address_string()} - {fmt % args}")

        def do_GET(self) -> None:  # noqa: N802
            parsed_url = urlparse(self.path)
            path = parsed_url.path
            if path == "/":
                payload = _html_page(app.video_path, app.hand_steps_path)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if path == "/metadata.json":
                payload = json.dumps(app.metadata(), ensure_ascii=False).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if path == "/frame.jpg":
                query = parse_qs(parsed_url.query)
                try:
                    frame_id = int(query.get("frame_id", [str(app.start_frame)])[0])
                    payload = app.render_jpeg(frame_id)
                except Exception as exc:
                    self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if path == "/annotation.json":
                query = parse_qs(parsed_url.query)
                try:
                    frame_id = int(query.get("frame_id", [str(app.start_frame)])[0])
                    payload = json.dumps(app.annotation(frame_id), ensure_ascii=False).encode("utf-8")
                except Exception as exc:
                    self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if path == "/artifacts.json":
                payload = json.dumps(app.artifact_payload(), ensure_ascii=False).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if path == "/healthz":
                payload = b'{"status":"ok","render_storage":"memory_only"}'
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if path == "/stream.mjpg":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.end_headers()
                try:
                    for chunk in app.mjpeg():
                        self.wfile.write(chunk)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            self.send_error(HTTPStatus.NOT_FOUND)

    return Handler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stream original RGB with SAM3/VLM bboxes and hand-step semantics in a browser. "
            "Rendered media stays in memory and is never written to disk."
        )
    )
    parser.add_argument("--video", required=True, help="Original raw_data/rgb.mp4 path.")
    parser.add_argument(
        "--hand-steps-json",
        default="",
        help="Stage3 hand_steps.json; inferred from the video's sibling processed_text_label when omitted.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Listen host. Keep 127.0.0.1 when using SSH port forwarding.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--render-fps", type=float, default=0.0, help="0 uses source FPS; lower values reduce CPU/network load.")
    parser.add_argument("--jpeg-quality", type=int, default=82)
    parser.add_argument("--max-video-width", type=int, default=1280)
    parser.add_argument("--panel-width", type=int, default=620)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--no-loop", action="store_true", help="Stop the stream at end-of-video instead of looping.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if cv2 is None or np is None:
        raise SystemExit(f"OpenCV and NumPy are required: {_CV_IMPORT_ERROR}")
    video_path = os.path.abspath(os.path.expanduser(args.video))
    if not os.path.isfile(video_path):
        raise SystemExit(f"Video not found: {video_path}")
    hand_steps_path = (
        os.path.abspath(os.path.expanduser(args.hand_steps_json))
        if args.hand_steps_json
        else infer_hand_steps_path(video_path)
    )
    if not os.path.isfile(hand_steps_path):
        raise SystemExit(f"hand_steps.json not found: {hand_steps_path}")
    with open(hand_steps_path, "r", encoding="utf-8") as handle:
        index = HandStepIndex(json.load(handle))
    if not (1 <= int(args.port) <= 65535):
        raise SystemExit("--port must be in [1,65535]")
    if not (1 <= int(args.jpeg_quality) <= 100):
        raise SystemExit("--jpeg-quality must be in [1,100]")
    app = VisualizationApp(
        video_path=video_path,
        hand_steps_path=hand_steps_path,
        index=index,
        render_fps=max(0.0, float(args.render_fps)),
        jpeg_quality=int(args.jpeg_quality),
        max_video_width=max(320, int(args.max_video_width)),
        panel_width=max(360, int(args.panel_width)),
        loop=not bool(args.no_loop),
        start_frame=max(0, int(args.start_frame)),
    )
    server = ThreadingHTTPServer((str(args.host), int(args.port)), make_handler(app))
    print(f"Visualization URL: http://{args.host}:{args.port}/")
    print("Rendered frames are memory-only; no visualization video/image is written.")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.close()


if __name__ == "__main__":
    main()
