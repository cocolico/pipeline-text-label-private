#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from common import (
    DEFAULT_ARK_REASONING_EFFORT,
    add_api_cli_args,
    api_config_from_args,
    build_source_video_meta,
    ensure_video_out_dir_safe,
    format_duration,
    get_video_metadata,
    logger,
    now_utc_iso,
    read_json,
    update_run_summary,
    video_id_from_path,
    write_json,
)

if TYPE_CHECKING:
    from common import ApiConfig


STAGE5_SCHEMA_VERSION = "stage5-bbox-v9-bounded-visibility-evidence"
# Ark Grounding uses a 1000x1000 normalized grid with valid integers [0, 999].
OFFICIAL_COORD_MAX = 999
OFFICIAL_COORD_DIVISOR = 1000.0
HAND_LABELS = ("left_hand", "right_hand")
PRIMARY_OPERATOR_POLICY = "sustained_center_pair_instance_chain_v3"
ARK_HAND_CONFIRMED_NOT_VISIBLE_REASONS = frozenset(
    {"occluded", "out_of_frame", "absent"}
)
ARK_HAND_CONFIRMATION_MAX_SECONDS = 0.5
ARK_HAND_CONFIRMATION_MAX_SAMPLES_PER_RUN = 3
HAND_ADJACENT_MAX_CENTER_JUMP = 0.12
HAND_QA_MAX_AREA_RATIO = 4.0
HAND_QA_ZERO_IOU_JUMP = 0.08
_BBOX_TAG_RE = re.compile(
    r"<bbox>\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*</bbox>",
    re.IGNORECASE,
)


def _find_litchi_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if parent.name == "litchi":
            return parent
    return Path(__file__).resolve().parent


def _find_named_parent(name: str) -> Optional[Path]:
    for parent in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
        if parent.name == name:
            return parent
    return None


def default_hand_detector_path() -> str:
    # WiLoR project layout convention:
    #   <wilor_root>/pretrained_models/detector.pt
    #   <wilor_root>/pretrained_models/wilor_final.ckpt
    #   <wilor_root>/mano_data/MANO_RIGHT.pkl
    configured = os.environ.get("HAND_DETECTOR_PATH", "").strip()
    if configured:
        return os.path.abspath(os.path.expanduser(configured))
    candidates: List[Path] = []
    api_root = _find_named_parent("api")
    if api_root is not None:
        candidates.append(api_root / "models" / "pretrained_models" / "detector.pt")
        candidates.append(api_root / "models" / "wilor" / "pretrained_models" / "detector.pt")
    ego_root = _find_named_parent("ego_data_visualization")
    if ego_root is not None:
        candidates.append(ego_root / "hand_pipeline" / "local" / "pretrained_models" / "detector.pt")
    candidates.append(_find_litchi_root() / "WiLoR" / "pretrained_models" / "detector.pt")
    return str(next((path for path in candidates if path.is_file()), candidates[0]))


def default_bbox_python() -> str:
    configured = os.environ.get("BBOX_PYTHON", "").strip()
    if configured:
        return configured
    api_root = _find_named_parent("api")
    candidate = (api_root or (_find_litchi_root() / "api")) / ".venv_bbox_tracking" / "bin" / "python"
    return str(candidate)


def default_sam3_python() -> str:
    configured = os.environ.get("SAM3_PYTHON", "").strip()
    if configured:
        return os.path.abspath(os.path.expanduser(configured))
    return str(Path.home() / "miniconda3" / "envs" / "sam3" / "bin" / "python")


def default_sam3_repo() -> str:
    configured = os.environ.get("SAM3_REPO", "").strip()
    if configured:
        return os.path.abspath(os.path.expanduser(configured))
    candidates = [
        Path.home() / "caoheng" / "sam3",
        Path.home() / "sam3",
    ]
    return str(next((path for path in candidates if (path / "sam3").is_dir()), candidates[0]))


def default_sam3_checkpoint() -> str:
    configured = os.environ.get("SAM3_CHECKPOINT", "").strip()
    if configured:
        return os.path.abspath(os.path.expanduser(configured))
    candidates: List[Path] = []
    api_root = _find_named_parent("api")
    if api_root is not None:
        candidates.append(api_root / "models" / "sam3" / "sam3.pt")
    candidates.append(Path(default_sam3_repo()) / "ckpt" / "sam3.pt")
    return str(next((path for path in candidates if path.is_file()), candidates[0]))


@dataclass
class BBoxConfig:
    hand_detector: str = ""
    hand_conf: float = 0.20
    hand_iou: float = 0.50
    hand_batch_size: int = 16
    hand_gap_fill_frames: int = 3
    hand_device: str = "auto"
    device: str = "auto"
    backend: str = "sam3"
    sam3_python: str = ""
    sam3_repo: str = ""
    sam3_checkpoint: str = ""
    sam3_threshold: float = 0.50
    sam3_chunk_size: int = 500
    sam3_max_instances: int = 8
    sam3_interval_padding_frames: int = 15
    sam3_hand_fusion: bool = True
    sam3_legacy_fallback: bool = False
    api_hand_assist: bool = True
    api_hand_max_calls_per_step: int = 3
    api_hand_max_total_calls: int = 24
    api_hand_min_missing_run_frames: int = 4
    api_hand_low_conf_threshold: float = 0.35
    api_object_min_track_coverage: float = 0.60
    api_object_min_mean_confidence: float = 0.55
    api_grounding_thinking: str = "enabled"
    api_grounding_reasoning_effort: str = DEFAULT_ARK_REASONING_EFFORT
    local_grounding_model: str = "IDEA-Research/grounding-dino-tiny"
    local_box_threshold: float = 0.20
    local_text_threshold: float = 0.20
    disable_local_grounding: bool = False
    sam2_model: str = "facebook/sam2.1-hiera-base-plus"
    disable_sam2: bool = False
    api_grounding: bool = True
    api_max_calls_per_step: int = 2
    api_max_total_calls: int = 24
    allow_unresolved_objects: bool = False
    jpeg_quality: int = 95
    overlay: bool = True

    def __post_init__(self) -> None:
        if not self.hand_detector:
            self.hand_detector = default_hand_detector_path()
        if not self.sam3_python:
            self.sam3_python = default_sam3_python()
        if not self.sam3_repo:
            self.sam3_repo = default_sam3_repo()
        if not self.sam3_checkpoint:
            self.sam3_checkpoint = default_sam3_checkpoint()


def add_bbox_cli_args(parser: argparse.ArgumentParser, *, include_bbox_python: bool = False) -> None:
    if include_bbox_python:
        parser.add_argument(
            "--bbox-python",
            default=default_bbox_python(),
            help="Python 3.10 interpreter containing Stage 5 OpenCV/Ultralytics dependencies.",
        )
    parser.add_argument("--hand-detector", default=default_hand_detector_path())
    parser.add_argument("--hand-conf", type=float, default=0.20)
    parser.add_argument("--hand-iou", type=float, default=0.50)
    parser.add_argument("--hand-batch-size", type=int, default=16)
    parser.add_argument("--hand-gap-fill-frames", type=int, default=3)
    parser.add_argument("--hand-device", default="auto", choices=("auto", "mps", "cpu", "cuda"))
    parser.add_argument("--bbox-device", default="auto", choices=("auto", "mps", "cpu", "cuda"))
    parser.add_argument("--bbox-backend", default="sam3", choices=("sam3", "legacy"))
    parser.add_argument("--sam3-python", default=default_sam3_python())
    parser.add_argument("--sam3-repo", default=default_sam3_repo())
    parser.add_argument("--sam3-checkpoint", default=default_sam3_checkpoint())
    parser.add_argument("--sam3-threshold", type=float, default=0.50)
    parser.add_argument("--sam3-chunk-size", type=int, default=500)
    parser.add_argument("--sam3-max-instances", type=int, default=8)
    parser.add_argument("--sam3-interval-padding-frames", type=int, default=15)
    parser.add_argument(
        "--sam3-hand-fusion", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--sam3-legacy-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Explicitly permit unresolved SAM3 objects to use GroundingDINO/Ark plus SAM2.",
    )
    parser.add_argument(
        "--api-hand-assist",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Ark image grounding only for expected hands that SAM3/WiLoR miss or detect weakly.",
    )
    parser.add_argument("--api-hand-max-calls-per-step", type=int, default=3)
    parser.add_argument("--api-hand-max-total-calls", type=int, default=24)
    parser.add_argument("--api-hand-min-missing-run-frames", type=int, default=4)
    parser.add_argument("--api-hand-low-conf-threshold", type=float, default=0.35)
    parser.add_argument("--api-object-min-track-coverage", type=float, default=0.60)
    parser.add_argument("--api-object-min-mean-confidence", type=float, default=0.55)
    parser.add_argument(
        "--api-grounding-thinking",
        default="enabled",
        choices=("disabled", "auto", "enabled"),
    )
    parser.add_argument(
        "--api-grounding-reasoning-effort",
        default=os.environ.get("ARK_REASONING_EFFORT", DEFAULT_ARK_REASONING_EFFORT),
        choices=("low", "medium", "high"),
    )
    parser.add_argument("--local-grounding-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--local-box-threshold", type=float, default=0.20)
    parser.add_argument("--local-text-threshold", type=float, default=0.20)
    parser.add_argument("--disable-local-grounding", action="store_true")
    parser.add_argument("--sam2-model", default="facebook/sam2.1-hiera-base-plus")
    parser.add_argument("--disable-sam2", action="store_true")
    parser.add_argument(
        "--api-grounding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Ark image grounding only when a SAM3 object track is weak or unresolved.",
    )
    parser.add_argument("--api-max-calls-per-step", type=int, default=2)
    parser.add_argument("--api-max-total-calls", type=int, default=24)
    parser.add_argument(
        "--allow-unresolved-objects",
        action="store_true",
        help="Complete with null object tracks when every grounding method failed (default: fail).",
    )
    parser.add_argument("--bbox-jpeg-quality", type=int, default=95)
    parser.add_argument("--bbox-overlay", action=argparse.BooleanOptionalAction, default=True)


def bbox_config_from_args(args: Any) -> BBoxConfig:
    return BBoxConfig(
        hand_detector=str(getattr(args, "hand_detector", "") or ""),
        hand_conf=float(getattr(args, "hand_conf", 0.20)),
        hand_iou=float(getattr(args, "hand_iou", 0.50)),
        hand_batch_size=max(1, int(getattr(args, "hand_batch_size", 16))),
        hand_gap_fill_frames=max(0, int(getattr(args, "hand_gap_fill_frames", 3))),
        hand_device=str(getattr(args, "hand_device", "auto") or "auto"),
        device=str(getattr(args, "bbox_device", "auto") or "auto"),
        backend=str(getattr(args, "bbox_backend", "sam3") or "sam3"),
        sam3_python=str(getattr(args, "sam3_python", "") or ""),
        sam3_repo=str(getattr(args, "sam3_repo", "") or ""),
        sam3_checkpoint=str(getattr(args, "sam3_checkpoint", "") or ""),
        sam3_threshold=min(1.0, max(0.0, float(getattr(args, "sam3_threshold", 0.50)))),
        sam3_chunk_size=max(1, int(getattr(args, "sam3_chunk_size", 500))),
        sam3_max_instances=max(1, int(getattr(args, "sam3_max_instances", 8))),
        sam3_interval_padding_frames=max(
            0, int(getattr(args, "sam3_interval_padding_frames", 15))
        ),
        sam3_hand_fusion=bool(getattr(args, "sam3_hand_fusion", True)),
        sam3_legacy_fallback=bool(getattr(args, "sam3_legacy_fallback", False)),
        api_hand_assist=bool(getattr(args, "api_hand_assist", True)),
        api_hand_max_calls_per_step=max(
            0, int(getattr(args, "api_hand_max_calls_per_step", 3))
        ),
        api_hand_max_total_calls=max(
            0, int(getattr(args, "api_hand_max_total_calls", 24))
        ),
        api_hand_min_missing_run_frames=max(
            1, int(getattr(args, "api_hand_min_missing_run_frames", 4))
        ),
        api_hand_low_conf_threshold=min(
            1.0, max(0.0, float(getattr(args, "api_hand_low_conf_threshold", 0.35)))
        ),
        api_object_min_track_coverage=min(
            1.0, max(0.0, float(getattr(args, "api_object_min_track_coverage", 0.60)))
        ),
        api_object_min_mean_confidence=min(
            1.0, max(0.0, float(getattr(args, "api_object_min_mean_confidence", 0.55)))
        ),
        api_grounding_thinking=str(
            getattr(args, "api_grounding_thinking", "enabled") or "enabled"
        ),
        api_grounding_reasoning_effort=str(
            getattr(
                args,
                "api_grounding_reasoning_effort",
                DEFAULT_ARK_REASONING_EFFORT,
            )
            or DEFAULT_ARK_REASONING_EFFORT
        ),
        local_grounding_model=str(
            getattr(args, "local_grounding_model", "IDEA-Research/grounding-dino-tiny")
        ),
        local_box_threshold=float(getattr(args, "local_box_threshold", 0.20)),
        local_text_threshold=float(getattr(args, "local_text_threshold", 0.20)),
        disable_local_grounding=bool(getattr(args, "disable_local_grounding", False)),
        sam2_model=str(getattr(args, "sam2_model", "facebook/sam2.1-hiera-base-plus")),
        disable_sam2=bool(getattr(args, "disable_sam2", False)),
        api_grounding=bool(getattr(args, "api_grounding", True)),
        api_max_calls_per_step=max(0, int(getattr(args, "api_max_calls_per_step", 2))),
        api_max_total_calls=max(0, int(getattr(args, "api_max_total_calls", 24))),
        allow_unresolved_objects=bool(getattr(args, "allow_unresolved_objects", False)),
        jpeg_quality=min(100, max(1, int(getattr(args, "bbox_jpeg_quality", 95)))),
        overlay=bool(getattr(args, "bbox_overlay", True)),
    )


def _normalize_label(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _label_key(value: Any) -> str:
    return _normalize_label(value).casefold()


_OBJECT_LABEL_STOPWORDS = {"a", "an", "the", "this", "that"}
_OBJECT_COLOR_TOKENS = {
    "black", "blue", "brown", "clear", "green", "grey", "gray", "orange",
    "pink", "purple", "red", "silver", "white", "yellow",
}
_OBJECT_ORDINAL_TOKENS = {
    "first", "second", "third", "fourth", "left", "right", "upper", "lower",
}
_OBJECT_MATERIAL_GROUPS = {
    "glass": "glass",
    "metal": "metal",
    "stainless": "metal",
    "steel": "metal",
    "paper": "paper",
    "plastic": "plastic",
    "wood": "wood",
    "wooden": "wood",
}
_OBJECT_HEAD_GROUPS = {
    "jug": "pitcher",
    "pitcher": "pitcher",
}


def _object_label_tokens(value: Any) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", _label_key(value))
        if token not in _OBJECT_LABEL_STOPWORDS
    ]


def _object_label_match_score(a: Any, b: Any) -> float:
    a_key, b_key = _label_key(a), _label_key(b)
    if not a_key or not b_key:
        return 0.0
    if a_key == b_key:
        return 1.0
    a_tokens, b_tokens = _object_label_tokens(a), _object_label_tokens(b)
    if not a_tokens or not b_tokens:
        return 0.0

    def descriptor_values(tokens: Sequence[str], vocabulary: set[str]) -> set[str]:
        return {token for token in tokens if token in vocabulary}

    for vocabulary in (_OBJECT_COLOR_TOKENS, _OBJECT_ORDINAL_TOKENS):
        a_values = descriptor_values(a_tokens, vocabulary)
        b_values = descriptor_values(b_tokens, vocabulary)
        if a_values and b_values and a_values.isdisjoint(b_values):
            return 0.0
    a_materials = {_OBJECT_MATERIAL_GROUPS[t] for t in a_tokens if t in _OBJECT_MATERIAL_GROUPS}
    b_materials = {_OBJECT_MATERIAL_GROUPS[t] for t in b_tokens if t in _OBJECT_MATERIAL_GROUPS}
    if a_materials and b_materials and a_materials.isdisjoint(b_materials):
        return 0.0

    a_head = _OBJECT_HEAD_GROUPS.get(a_tokens[-1], a_tokens[-1])
    b_head = _OBJECT_HEAD_GROUPS.get(b_tokens[-1], b_tokens[-1])
    if a_head != b_head:
        return 0.0
    a_set, b_set = set(a_tokens), set(b_tokens)
    if a_set <= b_set or b_set <= a_set:
        return 0.85
    overlap = len(a_set & b_set) / float(max(1, len(a_set | b_set)))
    return 0.70 if overlap >= 0.60 else 0.0


def _slug(value: Any) -> str:
    text = _label_key(value)
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "object"


def _clamp_box(box: Sequence[float], width: int, height: int) -> Optional[List[int]]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except Exception:
        return None
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    x1 = min(max(0.0, x1), max(0.0, width - 1.0))
    y1 = min(max(0.0, y1), max(0.0, height - 1.0))
    x2 = min(max(0.0, x2), max(0.0, width - 1.0))
    y2 = min(max(0.0, y2), max(0.0, height - 1.0))
    ix1, iy1 = int(math.floor(x1)), int(math.floor(y1))
    ix2, iy2 = int(math.ceil(x2)), int(math.ceil(y2))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return [ix1, iy1, ix2, iy2]


def _box_area(box: Optional[Sequence[float]]) -> float:
    if not box or len(box) != 4:
        return 0.0
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def _box_iou(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> float:
    if not a or not b:
        return 0.0
    ix1 = max(float(a[0]), float(b[0]))
    iy1 = max(float(a[1]), float(b[1]))
    ix2 = min(float(a[2]), float(b[2]))
    iy2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = _box_area(a) + _box_area(b) - inter
    return inter / union if union > 0 else 0.0


def _box_center(box: Sequence[float]) -> Tuple[float, float]:
    return (
        (float(box[0]) + float(box[2])) / 2.0,
        (float(box[1]) + float(box[3])) / 2.0,
    )


def _bbox_transition_metrics(
    previous_box: Sequence[float],
    current_box: Sequence[float],
    width: int,
    height: int,
) -> Dict[str, float]:
    diagonal = max(1.0, math.hypot(float(width), float(height)))
    center_jump = math.hypot(
        _box_center(previous_box)[0] - _box_center(current_box)[0],
        _box_center(previous_box)[1] - _box_center(current_box)[1],
    ) / diagonal
    previous_area = max(1.0, _box_area(previous_box))
    current_area = max(1.0, _box_area(current_box))
    previous_width = max(1.0, float(previous_box[2]) - float(previous_box[0]))
    current_width = max(1.0, float(current_box[2]) - float(current_box[0]))
    previous_height = max(1.0, float(previous_box[3]) - float(previous_box[1]))
    current_height = max(1.0, float(current_box[3]) - float(current_box[1]))
    return {
        "center_jump_normalized": center_jump,
        "iou": _box_iou(previous_box, current_box),
        "area_ratio": max(previous_area, current_area) / min(previous_area, current_area),
        "width_ratio": max(previous_width, current_width) / min(previous_width, current_width),
        "height_ratio": max(previous_height, current_height) / min(previous_height, current_height),
    }


def _box_touches_image_border(
    box: Sequence[float], width: int, height: int, tolerance: float = 2.0
) -> bool:
    return bool(
        float(box[0]) <= tolerance
        or float(box[1]) <= tolerance
        or float(box[2]) >= float(width - 1) - tolerance
        or float(box[3]) >= float(height - 1) - tolerance
    )


def _hand_transition_is_plausible(
    previous_box: Sequence[float],
    current_box: Sequence[float],
    frame_gap: int,
    width: int,
    height: int,
) -> Tuple[bool, Dict[str, float]]:
    gap = max(1, int(frame_gap))
    metrics = _bbox_transition_metrics(previous_box, current_box, width, height)
    # Identity is held tightly across short gaps. Longer gaps permit reacquisition,
    # but never make an adjacent cross-image jump acceptable.
    motion_limit = min(
        0.24,
        HAND_ADJACENT_MAX_CENTER_JUMP + 0.025 * math.sqrt(float(gap - 1)),
    )
    metrics["motion_limit_normalized"] = motion_limit
    if gap <= 12 and metrics["iou"] < 0.02 and metrics["center_jump_normalized"] > motion_limit:
        return False, metrics
    if (
        gap <= 3
        and metrics["iou"] < 0.02
        and metrics["area_ratio"] > 6.0
        and metrics["center_jump_normalized"] > 0.06
        and not (
            _box_touches_image_border(previous_box, width, height)
            and _box_touches_image_border(current_box, width, height)
            and metrics["center_jump_normalized"] <= motion_limit
        )
    ):
        return False, metrics
    return True, metrics


def _locked_sam3_instance_transition_is_plausible(
    previous_box: Sequence[float],
    current_box: Sequence[float],
    frame_gap: int,
    width: int,
    height: int,
) -> Tuple[bool, Dict[str, float]]:
    gap = max(1, int(frame_gap))
    metrics = _bbox_transition_metrics(previous_box, current_box, width, height)
    motion_limit = min(0.28, 0.16 + 0.03 * math.sqrt(float(gap - 1)))
    metrics["motion_limit_normalized"] = motion_limit
    if gap <= 8 and metrics["iou"] < 0.01 and metrics["center_jump_normalized"] > motion_limit:
        return False, metrics
    if (
        gap <= 3
        and metrics["iou"] < 0.01
        and metrics["area_ratio"] > 8.0
        and metrics["center_jump_normalized"] > 0.08
        and not (
            _box_touches_image_border(previous_box, width, height)
            and _box_touches_image_border(current_box, width, height)
            and metrics["center_jump_normalized"] <= motion_limit
        )
    ):
        return False, metrics
    return True, metrics


def _point_proximity_score(
    point: Tuple[float, float],
    target: Tuple[float, float],
    width: int,
    height: int,
    radius_fraction: float,
) -> float:
    radius = max(1.0, float(radius_fraction) * math.hypot(float(width), float(height)))
    return max(0.0, 1.0 - math.hypot(point[0] - target[0], point[1] - target[1]) / radius)


def _hand_temporal_continuity(
    candidate: Dict[str, Any],
    previous: Optional[Dict[str, Any]],
    frame_idx: int,
    width: int,
    height: int,
) -> float:
    if not previous:
        return 0.0
    previous_box = (previous.get("detection") or {}).get("bbox")
    candidate_box = candidate.get("bbox")
    if not previous_box or not candidate_box:
        return 0.0
    age = max(1, int(frame_idx) - int(previous.get("frame_index", frame_idx - 1)))
    if age > 60:
        return 0.0
    max_motion = min(0.30, 0.07 + 0.012 * float(age))
    center_score = _point_proximity_score(
        _box_center(candidate_box),
        _box_center(previous_box),
        width,
        height,
        max_motion,
    )
    return max(_box_iou(candidate_box, previous_box), center_score)


def _primary_pair_features(
    left: Dict[str, Any],
    right: Dict[str, Any],
    width: int,
    height: int,
    reference_center: Optional[Tuple[float, float]] = None,
    previous: Optional[Dict[str, Dict[str, Any]]] = None,
    frame_idx: int = 0,
) -> Optional[Dict[str, Any]]:
    left_box, right_box = left.get("bbox"), right.get("bbox")
    if not left_box or not right_box:
        return None
    diagonal = max(1.0, math.hypot(float(width), float(height)))
    left_center, right_center = _box_center(left_box), _box_center(right_box)
    separation = math.hypot(
        left_center[0] - right_center[0], left_center[1] - right_center[1]
    ) / diagonal
    # A pair spanning nearly the full image is usually a cross-person combination.
    if separation > 0.68 or _box_iou(left_box, right_box) > 0.80:
        return None
    midpoint = (
        (left_center[0] + right_center[0]) / 2.0,
        (left_center[1] + right_center[1]) / 2.0,
    )
    image_center = (float(width) / 2.0, float(height) / 2.0)
    midpoint_center_score = _point_proximity_score(
        midpoint, image_center, width, height, 0.50
    )
    individual_center_score = (
        _point_proximity_score(left_center, image_center, width, height, 0.50)
        + _point_proximity_score(right_center, image_center, width, height, 0.50)
    ) / 2.0
    centrality = 0.80 * midpoint_center_score + 0.20 * individual_center_score
    coherence = max(0.0, 1.0 - separation / 0.68)
    confidence = (
        float(left.get("confidence", 0.0) or 0.0)
        + float(right.get("confidence", 0.0) or 0.0)
    ) / 2.0
    previous = previous or {}
    left_temporal = _hand_temporal_continuity(
        left, previous.get("left_hand"), frame_idx, width, height
    )
    right_temporal = _hand_temporal_continuity(
        right, previous.get("right_hand"), frame_idx, width, height
    )
    temporal = (left_temporal + right_temporal) / 2.0
    known_temporal = [
        value
        for label, value in (
            ("left_hand", left_temporal),
            ("right_hand", right_temporal),
        )
        if previous.get(label)
    ]
    identity_continuity = min(known_temporal) if known_temporal else 0.0
    reference_score = (
        _point_proximity_score(midpoint, reference_center, width, height, 0.25)
        if reference_center is not None
        else centrality
    )
    score = (
        0.60 * centrality
        + 0.12 * coherence
        + 0.12 * temporal
        + 0.08 * reference_score
        + 0.08 * confidence
    )
    return {
        "left_hand": left,
        "right_hand": right,
        "midpoint": midpoint,
        "centrality": centrality,
        "coherence": coherence,
        "temporal": temporal,
        "left_temporal": left_temporal,
        "right_temporal": right_temporal,
        "identity_continuity": identity_continuity,
        "reference_score": reference_score,
        "confidence": confidence,
        "score": score,
    }


def _weighted_median(values: Sequence[Tuple[float, float]]) -> float:
    ordered = sorted((float(value), max(0.0, float(weight))) for value, weight in values)
    if not ordered:
        return 0.0
    total = sum(weight for _value, weight in ordered)
    if total <= 0.0:
        return ordered[len(ordered) // 2][0]
    threshold = total / 2.0
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return ordered[-1][0]


def _primary_operator_reference(
    candidate_records: Sequence[Dict[str, List[Dict[str, Any]]]],
    width: int,
    height: int,
) -> Tuple[Optional[Tuple[float, float]], bool, Dict[str, Any]]:
    pair_samples: List[Dict[str, Any]] = []
    single_samples: List[Dict[str, Any]] = []
    frames_with_pair_candidates = 0
    for candidates in candidate_records:
        pair_options = [
            features
            for left in candidates.get("left_hand", [])
            for right in candidates.get("right_hand", [])
            if (features := _primary_pair_features(left, right, width, height)) is not None
        ]
        if pair_options:
            frames_with_pair_candidates += 1
            pair_samples.append(
                max(
                    pair_options,
                    key=lambda item: (
                        float(item["centrality"]),
                        float(item["coherence"]),
                        float(item["confidence"]),
                    ),
                )
            )
            continue
        image_center = (float(width) / 2.0, float(height) / 2.0)
        candidates_flat = [
            detection
            for label in HAND_LABELS
            for detection in candidates.get(label, [])
            if detection.get("bbox")
        ]
        if candidates_flat:
            best = max(
                candidates_flat,
                key=lambda detection: (
                    _point_proximity_score(
                        _box_center(detection["bbox"]),
                        image_center,
                        width,
                        height,
                        0.50,
                    ),
                    float(detection.get("confidence", 0.0) or 0.0),
                ),
            )
            single_samples.append(
                {
                    "midpoint": _box_center(best["bbox"]),
                    "centrality": _point_proximity_score(
                        _box_center(best["bbox"]),
                        image_center,
                        width,
                        height,
                        0.50,
                    ),
                    "confidence": float(best.get("confidence", 0.0) or 0.0),
                }
            )
    samples = pair_samples if pair_samples else single_samples
    reference_mode = "pair_centroid" if pair_samples else "single_hand_fallback"
    if not samples:
        return None, False, {
            "reference_mode": "none",
            "frames_with_pair_candidates": 0,
            "frames_with_central_pair_evidence": 0,
            "frames_with_single_hand_reference": 0,
        }
    weighted_x: List[Tuple[float, float]] = []
    weighted_y: List[Tuple[float, float]] = []
    central_count = 0
    for sample in samples:
        centrality = float(sample["centrality"])
        if centrality >= 0.65:
            central_count += 1
        weight = max(1e-4, centrality**6) * (0.50 + 0.50 * float(sample["confidence"]))
        weighted_x.append((float(sample["midpoint"][0]), weight))
        weighted_y.append((float(sample["midpoint"][1]), weight))
    reference = (_weighted_median(weighted_x), _weighted_median(weighted_y))
    required = max(1, int(math.ceil(0.03 * float(len(samples)))))
    return reference, central_count >= required, {
        "reference_mode": reference_mode,
        "frames_with_pair_candidates": frames_with_pair_candidates,
        "frames_with_central_pair_evidence": central_count,
        "frames_with_single_hand_reference": len(single_samples),
    }


def _select_primary_operator_sequence(
    candidate_records: Sequence[Dict[str, List[Dict[str, Any]]]],
    width: int,
    height: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    reference_center, has_central_evidence, reference_meta = _primary_operator_reference(
        candidate_records, width, height
    )
    records: List[Dict[str, Any]] = [dict() for _ in candidate_records]
    previous: Dict[str, Dict[str, Any]] = {}
    total_candidates = 0
    selected_candidates = 0
    frames_with_competing_hands = 0
    frames_with_pair_selected = 0
    frames_with_single_hand_selected = 0
    pair_identity_lock_frames = 0
    single_identity_lock_selections = 0
    rejected_temporal_identity_switches = 0
    temporal_rejection_sample: List[Dict[str, Any]] = []
    selected_centralities: List[float] = []

    def selected_record(
        detection: Dict[str, Any], score: float, centrality: float
    ) -> Dict[str, Any]:
        return {
            **detection,
            "primary_operator": True,
            "primary_operator_policy": PRIMARY_OPERATOR_POLICY,
            "primary_operator_selection_score": round(float(score), 6),
            "primary_operator_centrality": round(float(centrality), 6),
        }

    for frame_idx, candidates in enumerate(candidate_records):
        frame_candidate_count = sum(len(candidates.get(label, [])) for label in HAND_LABELS)
        total_candidates += frame_candidate_count
        if any(len(candidates.get(label, [])) > 1 for label in HAND_LABELS):
            frames_with_competing_hands += 1
        usable_candidates: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for label in HAND_LABELS:
            for detection in candidates.get(label, []):
                previous_entry = previous.get(label)
                previous_box = (previous_entry or {}).get("detection", {}).get("bbox")
                current_box = detection.get("bbox")
                if previous_box and current_box:
                    frame_gap = frame_idx - int(previous_entry.get("frame_index", frame_idx - 1))
                    plausible, transition = _hand_transition_is_plausible(
                        previous_box,
                        current_box,
                        frame_gap,
                        width,
                        height,
                    )
                    if not plausible:
                        rejected_temporal_identity_switches += 1
                        if len(temporal_rejection_sample) < 20:
                            temporal_rejection_sample.append(
                                {
                                    "frame_index_0based": frame_idx,
                                    "hand": label,
                                    "previous_frame_index_0based": int(
                                        previous_entry.get("frame_index", frame_idx - 1)
                                    ),
                                    **{
                                        key: round(float(value), 6)
                                        for key, value in transition.items()
                                    },
                                }
                            )
                        continue
                usable_candidates[label].append(detection)
        pair_options = [
            features
            for left in usable_candidates.get("left_hand", [])
            for right in usable_candidates.get("right_hand", [])
            if (
                features := _primary_pair_features(
                    left,
                    right,
                    width,
                    height,
                    reference_center,
                    previous,
                    frame_idx,
                )
            )
            is not None
        ]
        accepted_pair: Optional[Dict[str, Any]] = None
        best_pair_temporal = max(
            (float(option["identity_continuity"]) for option in pair_options),
            default=0.0,
        )
        pair_temporal_floor = max(0.30, best_pair_temporal - 0.08)
        temporally_continuous_pairs = [
            option
            for option in pair_options
            if float(option["identity_continuity"]) >= pair_temporal_floor
        ]
        ranked_pair_options = sorted(
            temporally_continuous_pairs or pair_options,
            key=lambda item: float(item["score"]),
            reverse=True,
        )
        if temporally_continuous_pairs:
            pair_identity_lock_frames += 1
        for option in ranked_pair_options:
            centrality = float(option["centrality"])
            temporal = float(option["temporal"])
            reference_score = float(option["reference_score"])
            if has_central_evidence and centrality < 0.45 and temporal < 0.35:
                continue
            if reference_center is not None and reference_score < 0.15 and centrality < 0.65 and temporal < 0.35:
                continue
            accepted_pair = option
            break
        if accepted_pair is not None:
            for label in HAND_LABELS:
                detection = accepted_pair[label]
                records[frame_idx][label] = selected_record(
                    detection,
                    float(accepted_pair["score"]),
                    float(accepted_pair["centrality"]),
                )
                previous[label] = {
                    "frame_index": frame_idx,
                    "detection": records[frame_idx][label],
                }
                selected_candidates += 1
            frames_with_pair_selected += 1
            selected_centralities.append(float(accepted_pair["centrality"]))
            continue

        selected_here = 0
        for label in HAND_LABELS:
            options: List[Tuple[float, float, float, float, Dict[str, Any]]] = []
            for detection in usable_candidates.get(label, []):
                box = detection.get("bbox")
                if not box:
                    continue
                center = _box_center(box)
                centrality = _point_proximity_score(
                    center,
                    (float(width) / 2.0, float(height) / 2.0),
                    width,
                    height,
                    0.50,
                )
                temporal = _hand_temporal_continuity(
                    detection, previous.get(label), frame_idx, width, height
                )
                reference_score = (
                    _point_proximity_score(center, reference_center, width, height, 0.25)
                    if reference_center is not None
                    else centrality
                )
                confidence = float(detection.get("confidence", 0.0) or 0.0)
                score = (
                    0.68 * centrality
                    + 0.20 * temporal
                    + 0.07 * reference_score
                    + 0.05 * confidence
                )
                options.append((score, centrality, temporal, reference_score, detection))
            best_single_temporal = max((option[2] for option in options), default=0.0)
            single_temporal_floor = max(0.30, best_single_temporal - 0.08)
            temporally_continuous = [
                option for option in options if option[2] >= single_temporal_floor
            ]
            ranked_options = sorted(
                temporally_continuous or options, key=lambda item: item[0], reverse=True
            )
            if temporally_continuous:
                single_identity_lock_selections += 1
            for score, centrality, temporal, reference_score, detection in ranked_options:
                if has_central_evidence and centrality < 0.45 and temporal < 0.35:
                    continue
                if reference_center is not None and reference_score < 0.12 and centrality < 0.65 and temporal < 0.35:
                    continue
                records[frame_idx][label] = selected_record(detection, score, centrality)
                previous[label] = {
                    "frame_index": frame_idx,
                    "detection": records[frame_idx][label],
                }
                selected_candidates += 1
                selected_here += 1
                selected_centralities.append(centrality)
                break
        if selected_here:
            frames_with_single_hand_selected += 1

    return records, {
        "policy": PRIMARY_OPERATOR_POLICY,
        "rule": (
            "Select one temporally consistent anatomical hand pair whose sustained "
            "manipulation centroid is closest to the image center; discard secondary edge hands."
        ),
        "reference_center_pixel": (
            [round(reference_center[0], 3), round(reference_center[1], 3)]
            if reference_center is not None
            else None
        ),
        "reference_center_normalized": (
            [
                round(reference_center[0] / max(1.0, float(width)), 6),
                round(reference_center[1] / max(1.0, float(height)), 6),
            ]
            if reference_center is not None
            else None
        ),
        "has_sustained_central_pair_evidence": has_central_evidence,
        **reference_meta,
        "total_wilor_candidate_boxes": total_candidates,
        "selected_primary_candidate_boxes": selected_candidates,
        "discarded_secondary_candidate_boxes": max(0, total_candidates - selected_candidates),
        "frames_with_competing_hands": frames_with_competing_hands,
        "frames_with_primary_pair_selected": frames_with_pair_selected,
        "frames_with_primary_single_hand_selected": frames_with_single_hand_selected,
        "pair_identity_lock_frames": pair_identity_lock_frames,
        "single_identity_lock_selections": single_identity_lock_selections,
        "rejected_temporal_identity_switches": rejected_temporal_identity_switches,
        "temporal_identity_rejection_sample": temporal_rejection_sample,
        "mean_selected_centrality": (
            round(sum(selected_centralities) / len(selected_centralities), 6)
            if selected_centralities
            else 0.0
        ),
    }


def _hand_object_interaction_score(
    object_box: Optional[Sequence[float]],
    hand_boxes: Sequence[Sequence[float]],
    width: int,
    height: int,
) -> float:
    if not object_box or not hand_boxes:
        return 0.0
    diagonal = max(1.0, math.hypot(float(width), float(height)))
    best = 0.0
    for hand_box in hand_boxes:
        dx = max(
            float(hand_box[0]) - float(object_box[2]),
            float(object_box[0]) - float(hand_box[2]),
            0.0,
        )
        dy = max(
            float(hand_box[1]) - float(object_box[3]),
            float(object_box[1]) - float(hand_box[3]),
            0.0,
        )
        edge_distance = math.hypot(dx, dy)
        proximity = max(0.0, 1.0 - edge_distance / (0.20 * diagonal))
        ix1 = max(float(object_box[0]), float(hand_box[0]))
        iy1 = max(float(object_box[1]), float(hand_box[1]))
        ix2 = min(float(object_box[2]), float(hand_box[2]))
        iy2 = min(float(object_box[3]), float(hand_box[3]))
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        contact = intersection / max(1.0, min(_box_area(object_box), _box_area(hand_box)))
        best = max(best, proximity, min(1.0, contact))
    return best


def _pixel_to_official(box: Optional[Sequence[int]], width: int, height: int) -> Optional[List[int]]:
    if box is None:
        return None
    x1, y1, x2, y2 = box
    values = (
        int(float(x1) * OFFICIAL_COORD_DIVISOR / float(width)),
        int(float(y1) * OFFICIAL_COORD_DIVISOR / float(height)),
        int(float(x2) * OFFICIAL_COORD_DIVISOR / float(width)),
        int(float(y2) * OFFICIAL_COORD_DIVISOR / float(height)),
    )
    return [min(OFFICIAL_COORD_MAX, max(0, value)) for value in values]


def _official_to_pixel(box: Sequence[int], width: int, height: int) -> Optional[List[int]]:
    if len(box) != 4:
        return None
    return _clamp_box(
        [
            int(int(box[0]) * width / OFFICIAL_COORD_DIVISOR),
            int(int(box[1]) * height / OFFICIAL_COORD_DIVISOR),
            int(int(box[2]) * width / OFFICIAL_COORD_DIVISOR),
            int(int(box[3]) * height / OFFICIAL_COORD_DIVISOR),
        ],
        width,
        height,
    )


def _resolve_device(requested: str) -> str:
    value = str(requested or "auto").strip().lower()
    if value != "auto":
        return value
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _resolve_hand_device(requested: str) -> str:
    value = str(requested or "auto").strip().lower()
    if value != "auto":
        return value
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    # Ultralytics 8.1 reports a known MPS PoseModel bug. Prefer CPU for reliable
    # handedness boxes while leaving GroundingDINO/SAM2 on MPS.
    return "cpu"


@contextmanager
def _preserve_environment_variable(name: str) -> Iterator[None]:
    existed = name in os.environ
    original = os.environ.get(name)
    try:
        yield
    finally:
        if existed:
            os.environ[name] = str(original)
        else:
            os.environ.pop(name, None)


def _step_bounds(step: Dict[str, Any], fps: float, total_frames: int) -> Tuple[int, int]:
    try:
        start = int(step.get("start_frame_0based"))
    except Exception:
        start = int(round(float(step.get("start_sec", 0.0) or 0.0) * fps))
    try:
        end_exclusive = int(step.get("end_frame_0based_exclusive"))
    except Exception:
        end_exclusive = int(round(float(step.get("end_sec", 0.0) or 0.0) * fps))
    start = min(max(0, start), max(0, total_frames - 1))
    end_exclusive = min(max(start + 1, end_exclusive), total_frames)
    return start, end_exclusive


def _candidate_frames(step: Dict[str, Any], fps: float, total_frames: int) -> List[int]:
    start, end_exclusive = _step_bounds(step, fps, total_frames)
    span = max(1, end_exclusive - start)
    candidates = {
        start,
        min(end_exclusive - 1, start + int(round(span * 0.25))),
        min(end_exclusive - 1, start + int(round(span * 0.50))),
        min(end_exclusive - 1, start + int(round(span * 0.75))),
        end_exclusive - 1,
    }
    for action in step.get("atomic_actions", []) or []:
        if not isinstance(action, dict):
            continue
        try:
            a_start = int(action.get("start_frame_0based"))
            a_end = int(action.get("end_frame_0based_exclusive"))
        except Exception:
            continue
        candidates.add(min(total_frames - 1, max(0, (a_start + max(a_start + 1, a_end) - 1) // 2)))
    center = start + span / 2.0
    return sorted(candidates, key=lambda frame_idx: (abs(frame_idx - center), frame_idx))


def _collect_object_specs(
    plan: Dict[str, Any], fps: float, total_frames: int
) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    steps = [step for step in plan.get("steps", []) if isinstance(step, dict)]
    steps_by_id: Dict[int, Dict[str, Any]] = {}
    registry: Dict[str, Dict[str, Any]] = {}
    for step in steps:
        try:
            step_id = int(step.get("step_id"))
        except Exception:
            continue
        steps_by_id[step_id] = step
        relations = step.get("hand_object_relations")
        if not isinstance(relations, list):
            raise RuntimeError(
                f"Stage 5 requires hand_object_relations to be an array for step_id={step_id}. "
                "Re-run the updated Stage 3 before Stage 5."
            )
        start, end_exclusive = _step_bounds(step, fps, total_frames)
        for relation in relations:
            if not isinstance(relation, dict):
                continue
            label = _normalize_label(relation.get("object"))
            if not label:
                continue
            key = _label_key(label)
            if key not in registry:
                fuzzy_matches = sorted(
                    (
                        (
                            max(
                                _object_label_match_score(label, alias)
                                for alias in existing.get("aliases", [existing["label"]])
                            ),
                            existing_key,
                        )
                        for existing_key, existing in registry.items()
                    ),
                    reverse=True,
                )
                if fuzzy_matches and fuzzy_matches[0][0] > 0.0:
                    key = fuzzy_matches[0][1]
            obj = registry.setdefault(
                key,
                {
                    "label": label,
                    "aliases": [],
                    "active_step_ids": [],
                    "active_intervals": [],
                    "hand_relations": [],
                    "candidate_frames": [],
                },
            )
            if label not in obj["aliases"]:
                obj["aliases"].append(label)
            if step_id not in obj["active_step_ids"]:
                obj["active_step_ids"].append(step_id)
                obj["active_intervals"].append([start, end_exclusive])
                obj["candidate_frames"].extend(_candidate_frames(step, fps, total_frames))
            relation_entry = {
                "step_id": step_id,
                "hand": _normalize_label(relation.get("hand")),
                "relation": _normalize_label(relation.get("relation")),
            }
            if relation_entry not in obj["hand_relations"]:
                obj["hand_relations"].append(relation_entry)

    specs: List[Dict[str, Any]] = []
    for index, (_key, obj) in enumerate(registry.items(), start=1):
        obj["label"] = max(
            obj["aliases"], key=lambda value: (len(_object_label_tokens(value)), len(value))
        )
        obj["track_id"] = f"object_{_slug(obj['label'])}_{index:03d}"
        obj["active_step_ids"] = sorted(set(int(v) for v in obj["active_step_ids"]))
        obj["candidate_frames"] = list(dict.fromkeys(int(v) for v in obj["candidate_frames"]))
        specs.append(obj)
    return specs, steps_by_id


def _extract_frames(video_path: str, frame_dir: Path, jpeg_quality: int) -> List[Path]:
    import cv2

    frame_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    paths: List[Path] = []
    try:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            path = frame_dir / f"{frame_idx:06d}.jpg"
            if not cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]):
                raise RuntimeError(f"Failed to write temporary frame: {path}")
            paths.append(path)
            frame_idx += 1
    finally:
        cap.release()
    if not paths:
        raise RuntimeError(f"Decoded zero frames from: {video_path}")
    return paths


def _frame_timestamps(
    video_path: str, total_frames: int, fps: float
) -> Tuple[List[float], str]:
    fallback = [frame_idx / fps for frame_idx in range(total_frames)]
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return fallback, "frame_index_div_fps"
    try:
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "frame=best_effort_timestamp_time",
                "-of",
                "json",
                video_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        frames = json.loads(completed.stdout).get("frames", [])
        timestamps = [float(frame["best_effort_timestamp_time"]) for frame in frames]
        if len(timestamps) != total_frames:
            raise ValueError(
                f"ffprobe timestamp count={len(timestamps)} metadata frames={total_frames}"
            )
        origin = timestamps[0]
        timestamps = [max(0.0, value - origin) for value in timestamps]
        if any(right < left for left, right in zip(timestamps, timestamps[1:])):
            raise ValueError("ffprobe timestamps are not monotonic")
        return timestamps, "ffprobe_best_effort_timestamp_time"
    except Exception as exc:
        logger.warning(
            f"[stage5] exact frame timestamps unavailable; using frame_index/fps: "
            f"{type(exc).__name__}: {exc}"
        )
        return fallback, "frame_index_div_fps"


def _fill_short_hand_gaps(records: List[Dict[str, Any]], max_gap: int) -> None:
    if max_gap <= 0:
        return
    for label in HAND_LABELS:
        known = [i for i, record in enumerate(records) if record.get(label, {}).get("bbox")]
        for left_idx, right_idx in zip(known, known[1:]):
            gap = right_idx - left_idx - 1
            if gap <= 0 or gap > max_gap:
                continue
            left = records[left_idx][label]
            right = records[right_idx][label]
            left_box, right_box = left["bbox"], right["bbox"]
            endpoint_scale = max(
                1.0,
                math.hypot(
                    float(left_box[2]) - float(left_box[0]),
                    float(left_box[3]) - float(left_box[1]),
                ),
                math.hypot(
                    float(right_box[2]) - float(right_box[0]),
                    float(right_box[3]) - float(right_box[1]),
                ),
            )
            endpoint_motion = math.hypot(
                _box_center(left_box)[0] - _box_center(right_box)[0],
                _box_center(left_box)[1] - _box_center(right_box)[1],
            )
            allowed_motion = 1.50 * endpoint_scale * float(gap + 1)
            if _box_iou(left_box, right_box) < 0.05 and endpoint_motion > allowed_motion:
                continue
            for offset in range(1, gap + 1):
                alpha = offset / float(gap + 1)
                box = [
                    int(round((1.0 - alpha) * left["bbox"][j] + alpha * right["bbox"][j]))
                    for j in range(4)
                ]
                interpolated = {
                    "bbox": box,
                    "confidence": min(float(left["confidence"]), float(right["confidence"])) * 0.80,
                    "source": "temporal_interpolation",
                    "bbox_extent": "amodal_estimated",
                    "primary_operator": True,
                    "primary_operator_policy": PRIMARY_OPERATOR_POLICY,
                }
                left_instance = left.get("sam3_instance_id")
                if left_instance and left_instance == right.get("sam3_instance_id"):
                    interpolated.update(
                        {
                            "sam3_instance_id": left_instance,
                            "primary_hand_chain_id": left.get("primary_hand_chain_id"),
                            "primary_instance_locked": True,
                            "identity_validation": "interpolated_between_same_locked_instance",
                        }
                    )
                else:
                    left_chain = left.get("primary_hand_chain_id")
                    if left_chain and left_chain == right.get("primary_hand_chain_id"):
                        interpolated.update(
                            {
                                "primary_hand_chain_id": left_chain,
                                "primary_instance_locked": True,
                                "identity_validation": (
                                    "interpolated_between_continuity_linked_instances"
                                ),
                            }
                        )
                records[left_idx + offset][label] = interpolated


def _detect_hands(
    video_path: str,
    total_frames: int,
    width: int,
    height: int,
    cfg: BBoxConfig,
    device: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    import cv2
    import torch
    from ultralytics import YOLO

    detector_path = os.path.abspath(os.path.expanduser(cfg.hand_detector))
    if not os.path.isfile(detector_path):
        raise FileNotFoundError(f"WiLoR hand detector not found: {detector_path}")
    # This repository-owned WiLoR checkpoint contains an Ultralytics PoseModel object.
    # PyTorch >=2.6 defaults torch.load to weights_only=True, which rejects that trusted
    # legacy format. Scope the compatibility override to this one local checkpoint load.
    original_torch_load = torch.load

    def trusted_checkpoint_load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = trusted_checkpoint_load
    try:
        detector = YOLO(detector_path)
    finally:
        torch.load = original_torch_load
    candidate_records: List[Dict[str, List[Dict[str, Any]]]] = [
        {} for _ in range(total_frames)
    ]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for hand detection: {video_path}")
    batch_frames: List[Any] = []
    batch_indices: List[int] = []

    def flush() -> None:
        if not batch_frames:
            return
        results = detector.predict(
            source=batch_frames,
            conf=float(cfg.hand_conf),
            iou=float(cfg.hand_iou),
            device=device,
            verbose=False,
        )
        for frame_idx, result in zip(batch_indices, results):
            boxes = result.boxes
            if boxes is None:
                continue
            candidates: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            xyxy_values = boxes.xyxy.detach().cpu().numpy().tolist()
            class_values = boxes.cls.detach().cpu().numpy().tolist()
            conf_values = boxes.conf.detach().cpu().numpy().tolist()
            for raw_box, raw_class, raw_conf in zip(xyxy_values, class_values, conf_values):
                label = "right_hand" if int(raw_class) == 1 else "left_hand"
                box = _clamp_box(raw_box, width, height)
                if box is None:
                    continue
                candidates[label].append(
                    {
                        "bbox": box,
                        "confidence": float(raw_conf),
                        "source": "wilor_yolo",
                        "bbox_extent": "observed_full",
                    }
                )
            for label, items in candidates.items():
                items.sort(key=lambda item: (float(item["confidence"]), _box_area(item["bbox"])), reverse=True)
                candidate_records[frame_idx][label] = items
        batch_frames.clear()
        batch_indices.clear()

    try:
        while True:
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx >= total_frames:
                break
            batch_frames.append(frame)
            batch_indices.append(frame_idx)
            if len(batch_frames) >= max(1, int(cfg.hand_batch_size)):
                flush()
        flush()
    finally:
        cap.release()
    records, selection_meta = _select_primary_operator_sequence(
        candidate_records, width, height
    )
    # Preserve the detector's anatomical class labels. Image position or geometric
    # continuity must not swap left/right identities when the hands cross.
    _fill_short_hand_gaps(records, int(cfg.hand_gap_fill_frames))
    return records, selection_meta


def _expanded_intervals(
    intervals: Sequence[Sequence[int]], total_frames: int, padding: int
) -> List[List[int]]:
    normalized: List[List[int]] = []
    for value in intervals:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            continue
        start = min(total_frames, max(0, int(value[0]) - padding))
        end = min(total_frames, max(start, int(value[1]) + padding))
        if end <= start:
            continue
        if normalized and start <= normalized[-1][1]:
            normalized[-1][1] = max(normalized[-1][1], end)
        else:
            normalized.append([start, end])
    return normalized


def _run_sam3_backend(
    frame_dir: Path,
    specs: Sequence[Dict[str, Any]],
    total_frames: int,
    cfg: BBoxConfig,
    stage5_dir: Path,
) -> Dict[str, Any]:
    interpreter = Path(cfg.sam3_python).expanduser().resolve()
    repo = Path(cfg.sam3_repo).expanduser().resolve()
    checkpoint = Path(cfg.sam3_checkpoint).expanduser().resolve()
    helper = Path(__file__).resolve().with_name("stage5_sam3_backend.py")
    required = {
        "SAM3 Python": interpreter,
        "SAM3 model_builder.py": repo / "sam3" / "model_builder.py",
        "SAM3 checkpoint": checkpoint,
        "SAM3 Stage 5 helper": helper,
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing SAM3 runtime files:\n" + "\n".join(missing))

    requests: List[Dict[str, Any]] = []
    if cfg.sam3_hand_fusion:
        requests.append(
            {
                "request_id": "__hands__",
                "prompt": "hand",
                "ranges": [[0, total_frames]],
                "max_instances": 4,
            }
        )
    for spec in specs:
        requests.append(
            {
                "request_id": str(spec["track_id"]),
                "prompt": str(spec["label"]),
                "ranges": _expanded_intervals(
                    spec.get("active_intervals", []),
                    total_frames,
                    int(cfg.sam3_interval_padding_frames),
                ),
                "max_instances": int(cfg.sam3_max_instances),
            }
        )
    request_path = stage5_dir / "sam3_requests.json"
    output_path = stage5_dir / "sam3_candidates.json"
    write_json(
        str(request_path),
        {
            "schema_version": "stage5-sam3-request-v1",
            "source_frames": str(frame_dir),
            "total_frames": total_frames,
            "requests": requests,
        },
    )
    cmd = [
        str(interpreter),
        str(helper),
        "--frames-dir",
        str(frame_dir),
        "--requests-json",
        str(request_path),
        "--output-json",
        str(output_path),
        "--sam3-repo",
        str(repo),
        "--checkpoint",
        str(checkpoint),
        "--threshold",
        str(cfg.sam3_threshold),
        "--chunk-size",
        str(cfg.sam3_chunk_size),
        "--max-instances",
        str(cfg.sam3_max_instances),
    ]
    env = os.environ.copy()
    env.setdefault("USE_PERFLIB", "0")
    completed = subprocess.run(cmd, check=False, env=env)
    if completed.returncode != 0:
        raise RuntimeError(f"SAM3 backend failed with exit code {completed.returncode}")
    payload = read_json(str(output_path))
    if payload.get("schema_version") != "stage5-sam3-candidates-v1":
        raise RuntimeError("SAM3 backend returned an unsupported schema")
    if int(payload.get("total_frames", -1)) != int(total_frames):
        raise RuntimeError(
            f"SAM3 frame count mismatch: expected={total_frames}, "
            f"actual={payload.get('total_frames')}"
        )
    return payload


def _sam3_request_frames(payload: Dict[str, Any], request_id: str) -> Dict[int, List[Dict[str, Any]]]:
    request = (payload.get("requests") or {}).get(request_id) or {}
    raw_frames = request.get("frames") or {}
    return {
        int(frame_idx): candidates
        for frame_idx, candidates in raw_frames.items()
        if isinstance(candidates, list)
    }


def _hand_match_score(hand_box: Sequence[float], sam3_box: Sequence[float]) -> float:
    intersection_x1 = max(float(hand_box[0]), float(sam3_box[0]))
    intersection_y1 = max(float(hand_box[1]), float(sam3_box[1]))
    intersection_x2 = min(float(hand_box[2]), float(sam3_box[2]))
    intersection_y2 = min(float(hand_box[3]), float(sam3_box[3]))
    intersection = max(0.0, intersection_x2 - intersection_x1) * max(
        0.0, intersection_y2 - intersection_y1
    )
    containment = intersection / max(1.0, min(_box_area(hand_box), _box_area(sam3_box)))
    hand_center = (
        (float(hand_box[0]) + float(hand_box[2])) / 2.0,
        (float(hand_box[1]) + float(hand_box[3])) / 2.0,
    )
    sam3_center = (
        (float(sam3_box[0]) + float(sam3_box[2])) / 2.0,
        (float(sam3_box[1]) + float(sam3_box[3])) / 2.0,
    )
    scale = max(1.0, math.sqrt(max(_box_area(hand_box), _box_area(sam3_box))))
    center_score = max(
        0.0,
        1.0 - math.hypot(hand_center[0] - sam3_center[0], hand_center[1] - sam3_center[1]) / scale,
    )
    return max(_box_iou(hand_box, sam3_box), containment, center_score)


def _match_hands_for_frame(
    hand_record: Dict[str, Any], candidates: Sequence[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    pairs: List[Tuple[float, str, int]] = []
    for label in HAND_LABELS:
        hand_box = hand_record.get(label, {}).get("bbox")
        if not hand_box:
            continue
        for index, candidate in enumerate(candidates):
            box = candidate.get("bbox_xyxy_pixel")
            if box:
                pairs.append((_hand_match_score(hand_box, box), label, index))
    matched: Dict[str, Dict[str, Any]] = {}
    used: set[int] = set()
    for score, label, index in sorted(pairs, reverse=True):
        if score < 0.10 or label in matched or index in used:
            continue
        matched[label] = candidates[index]
        used.add(index)
    return matched


def _sam3_segment_key(instance_id: str) -> str:
    value = str(instance_id or "")
    return value.split("_obj", 1)[0] if "_obj" in value else "global"


def _sam3_hand_segment_ranges(
    payload: Dict[str, Any], total_frames: int
) -> Dict[str, Tuple[int, int]]:
    request = (payload.get("requests") or {}).get("__hands__") or {}
    raw_ranges = request.get("ranges") or [[0, total_frames]]
    chunk_size = max(1, int(payload.get("chunk_size") or total_frames or 1))
    result: Dict[str, Tuple[int, int]] = {}
    segment_index = 0
    for raw_range in raw_ranges:
        if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
            continue
        range_start = min(total_frames, max(0, int(raw_range[0])))
        range_end = min(total_frames, max(range_start, int(raw_range[1])))
        for start in range(range_start, range_end, chunk_size):
            end = min(range_end, start + chunk_size)
            result[f"seg{segment_index:05d}"] = (start, end)
            segment_index += 1
    return result


def _assign_primary_sam3_hand_instances(
    hand_records: Sequence[Dict[str, Any]],
    frames: Dict[int, List[Dict[str, Any]]],
    width: int,
    height: int,
    segment_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    votes: Dict[str, Counter[str]] = defaultdict(Counter)
    match_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    instances_by_segment: Dict[str, set[str]] = defaultdict(set)
    frames_by_segment: Dict[str, set[int]] = defaultdict(set)
    instance_boxes: Dict[str, List[Tuple[int, Sequence[float]]]] = defaultdict(list)
    direct_matches = 0
    for frame_idx, candidates in frames.items():
        if not 0 <= frame_idx < len(hand_records):
            continue
        for candidate in candidates:
            instance_id = str(candidate.get("instance_id") or "")
            if instance_id:
                segment = _sam3_segment_key(instance_id)
                instances_by_segment[segment].add(instance_id)
                frames_by_segment[segment].add(frame_idx)
                box = candidate.get("bbox_xyxy_pixel")
                if box:
                    instance_boxes[instance_id].append((frame_idx, box))
        for label, candidate in _match_hands_for_frame(hand_records[frame_idx], candidates).items():
            instance_id = str(candidate.get("instance_id") or "")
            box = candidate.get("bbox_xyxy_pixel")
            hand_box = hand_records[frame_idx].get(label, {}).get("bbox")
            if instance_id and box and hand_box:
                votes[instance_id][label] += _hand_match_score(hand_box, box) * (
                    0.5 + float(candidate.get("confidence", 0.0) or 0.0)
                )
                match_counts[instance_id][label] += 1
                direct_matches += 1

    anchor_assignments: Dict[Tuple[str, str], str] = {}
    assignment_report: Dict[str, Any] = {}
    previous_terminal: Dict[str, Tuple[int, Sequence[float]]] = {}
    for segment, instance_ids in sorted(instances_by_segment.items()):
        segment_range = (segment_ranges or {}).get(segment)
        segment_frame_count = (
            max(0, int(segment_range[1]) - int(segment_range[0]))
            if segment_range is not None
            else (
                max(frames_by_segment[segment]) - min(frames_by_segment[segment]) + 1
                if frames_by_segment[segment]
                else 0
            )
        )
        min_matches = (
            1
            if segment_frame_count <= 3
            else max(3, int(math.ceil(0.01 * float(segment_frame_count))))
        )
        eligible: Dict[str, List[Optional[str]]] = {label: [None] for label in HAND_LABELS}
        continuity_meta: Dict[str, Dict[str, Any]] = defaultdict(dict)
        for label in HAND_LABELS:
            for instance_id in sorted(instance_ids):
                evidence_ok = (
                    int(match_counts[instance_id][label]) >= min_matches
                    and float(votes[instance_id][label]) >= 0.50
                )
                continuity_ok = False
                boundary_bonus = 0.0
                boundary = previous_terminal.get(label)
                current_track = sorted(instance_boxes.get(instance_id, []))
                if boundary and current_track:
                    previous_frame, previous_box = boundary
                    current_frame, current_box = current_track[0]
                    gap = max(1, int(current_frame) - int(previous_frame))
                    plausible, transition = _hand_transition_is_plausible(
                        previous_box, current_box, gap, width, height
                    )
                    motion_limit = max(
                        1e-6, float(transition["motion_limit_normalized"])
                    )
                    boundary_bonus = max(
                        float(transition["iou"]),
                        max(
                            0.0,
                            1.0
                            - float(transition["center_jump_normalized"])
                            / motion_limit,
                        ),
                    )
                    continuity_ok = bool(gap <= 12 and plausible and boundary_bonus >= 0.25)
                    continuity_meta[instance_id][label] = {
                        "previous_frame_index_0based": int(previous_frame),
                        "current_frame_index_0based": int(current_frame),
                        "plausible": bool(plausible),
                        "continuity_score": round(boundary_bonus, 6),
                        **{
                            key: round(float(value), 6)
                            for key, value in transition.items()
                        },
                    }
                    if gap <= 12 and not plausible:
                        continue
                if evidence_ok or continuity_ok:
                    eligible[label].append(instance_id)
        best_score = -1.0
        best_pair: Tuple[Optional[str], Optional[str]] = (None, None)
        for left_id in eligible["left_hand"]:
            for right_id in eligible["right_hand"]:
                if left_id is not None and left_id == right_id:
                    continue
                score = (
                    float(votes[left_id]["left_hand"]) if left_id is not None else 0.0
                ) + (
                    float(votes[right_id]["right_hand"]) if right_id is not None else 0.0
                )
                if left_id is not None:
                    score += 5.0 * float(
                        continuity_meta.get(left_id, {})
                        .get("left_hand", {})
                        .get("continuity_score", 0.0)
                    )
                if right_id is not None:
                    score += 5.0 * float(
                        continuity_meta.get(right_id, {})
                        .get("right_hand", {})
                        .get("continuity_score", 0.0)
                    )
                if score > best_score:
                    best_score = score
                    best_pair = (left_id, right_id)
        selected = dict(zip(HAND_LABELS, best_pair))
        for label, instance_id in selected.items():
            if instance_id is not None:
                anchor_assignments[(segment, label)] = instance_id
                track = sorted(instance_boxes.get(instance_id, []))
                if track:
                    previous_terminal[label] = track[-1]
        assignment_report[segment] = {
            "anchor_instance_ids": selected,
            "selected_assignment_score": round(max(0.0, best_score), 6),
            "frame_range_0based_half_open": list(segment_range) if segment_range else None,
            "minimum_matched_frames": min_matches,
            "boundary_continuity": dict(continuity_meta),
            "instance_evidence": {
                instance_id: {
                    label: {
                        "weighted_vote": round(float(votes[instance_id][label]), 6),
                        "matched_frames": int(match_counts[instance_id][label]),
                    }
                    for label in HAND_LABELS
                }
                for instance_id in sorted(instance_ids)
            },
        }
    instance_assignments: Dict[str, str] = {
        instance_id: label
        for (_segment, label), instance_id in anchor_assignments.items()
    }
    assignment_details: Dict[str, Dict[str, Any]] = {
        instance_id: {
            "hand": label,
            "assignment_kind": "segment_anchor",
            "segment": segment,
        }
        for (segment, label), instance_id in anchor_assignments.items()
    }

    def fragment_link(
        candidate_id: str, assigned_id: str
    ) -> Optional[Dict[str, Any]]:
        candidate_track = sorted(instance_boxes.get(candidate_id, []))
        assigned_track = sorted(instance_boxes.get(assigned_id, []))
        if not candidate_track or not assigned_track:
            return None
        candidate_start, candidate_end = candidate_track[0], candidate_track[-1]
        candidate_frames = {frame_idx for frame_idx, _box in candidate_track}
        assigned_frames = {frame_idx for frame_idx, _box in assigned_track}
        if candidate_frames & assigned_frames:
            return None
        if any(
            int(candidate_start[0]) <= frame_idx <= int(candidate_end[0])
            for frame_idx in assigned_frames
        ):
            return None
        boundaries: List[Tuple[str, Tuple[int, Sequence[float]], Tuple[int, Sequence[float]]]] = []
        before = [item for item in assigned_track if item[0] < candidate_start[0]]
        after = [item for item in assigned_track if item[0] > candidate_end[0]]
        if before:
            boundaries.append(("before", before[-1], candidate_start))
        if after:
            boundaries.append(("after", candidate_end, after[0]))
        if not boundaries:
            return None
        boundary_reports: List[Dict[str, Any]] = []
        for side, previous, current in boundaries:
            previous_frame, previous_box = previous
            current_frame, current_box = current
            gap = int(current_frame) - int(previous_frame)
            if gap > 12:
                continue
            plausible, transition = _hand_transition_is_plausible(
                previous_box, current_box, gap, width, height
            )
            if not plausible:
                return None
            motion_limit = max(1e-6, float(transition["motion_limit_normalized"]))
            continuity = max(
                float(transition["iou"]),
                max(
                    0.0,
                    1.0
                    - float(transition["center_jump_normalized"]) / motion_limit,
                ),
            )
            boundary_reports.append(
                {
                    "side": side,
                    "previous_frame_index_0based": int(previous_frame),
                    "current_frame_index_0based": int(current_frame),
                    "continuity_score": continuity,
                    **transition,
                }
            )
        if not boundary_reports:
            return None
        continuity = min(
            float(report["continuity_score"]) for report in boundary_reports
        )
        return {
            "linked_instance_id": assigned_id,
            "continuity_score": continuity,
            "boundary_checks": boundary_reports,
        }

    unassigned = set(instance_boxes) - set(instance_assignments)
    while unassigned:
        proposals: List[Tuple[float, str, str, Dict[str, Any]]] = []
        for instance_id in sorted(unassigned):
            per_label: List[Tuple[float, str, Dict[str, Any]]] = []
            for label in HAND_LABELS:
                links = [
                    link
                    for assigned_id, assigned_label in instance_assignments.items()
                    if assigned_label == label
                    and (link := fragment_link(instance_id, assigned_id)) is not None
                ]
                if not links:
                    continue
                best_link = max(links, key=lambda item: float(item["continuity_score"]))
                continuity = float(best_link["continuity_score"])
                label_matches = int(match_counts[instance_id][label])
                if label_matches <= 0 and continuity < 0.65:
                    continue
                evidence_bonus = min(0.20, 0.02 * float(label_matches))
                per_label.append((continuity + evidence_bonus, label, best_link))
            if not per_label:
                continue
            ranked = sorted(per_label, reverse=True)
            best_score, best_label, best_link = ranked[0]
            if len(ranked) > 1 and best_score - ranked[1][0] < 0.08:
                continue
            proposals.append((best_score, instance_id, best_label, best_link))
        if not proposals:
            break
        score, instance_id, label, link = max(proposals, key=lambda item: item[0])
        instance_assignments[instance_id] = label
        assignment_details[instance_id] = {
            "hand": label,
            "assignment_kind": "continuity_linked_fragment",
            "segment": _sam3_segment_key(instance_id),
            "assignment_score": round(float(score), 6),
            "link": {
                key: (
                    round(float(value), 6)
                    if isinstance(value, float)
                    else value
                )
                for key, value in link.items()
            },
        }
        unassigned.remove(instance_id)

    for segment, report in assignment_report.items():
        segment_ids = instances_by_segment.get(segment, set())
        report["selected_instance_ids"] = {
            label: sorted(
                instance_id
                for instance_id in segment_ids
                if instance_assignments.get(instance_id) == label
            )
            for label in HAND_LABELS
        }

    return instance_assignments, {
        "direct_matches": direct_matches,
        "segments": assignment_report,
        "instance_assignments": assignment_details,
        "selected_instance_count": len(instance_assignments),
        "continuity_linked_fragment_count": sum(
            1
            for details in assignment_details.values()
            if details["assignment_kind"] == "continuity_linked_fragment"
        ),
    }


def _prune_hand_records_outside_primary_instance(
    hand_records: List[Dict[str, Any]],
    width: int,
    height: int,
    max_reference_gap: int = 12,
) -> Dict[str, Any]:
    removed: List[Dict[str, Any]] = []
    removed_count = 0
    for label in HAND_LABELS:
        trusted = [
            frame_idx
            for frame_idx, record in enumerate(hand_records)
            if record.get(label, {}).get("bbox")
            and record.get(label, {}).get("primary_instance_locked")
        ]
        if not trusted:
            continue
        for frame_idx, frame_record in enumerate(hand_records):
            record = frame_record.get(label) or {}
            box = record.get("bbox")
            if not box or record.get("primary_instance_locked"):
                continue
            references = sorted(
                (
                    (abs(reference_idx - frame_idx), reference_idx)
                    for reference_idx in trusted
                    if abs(reference_idx - frame_idx) <= max(1, int(max_reference_gap))
                ),
                key=lambda item: item[0],
            )[:2]
            if not references:
                continue
            decisions: List[Tuple[bool, Dict[str, float], int]] = []
            for gap, reference_idx in references:
                reference_box = hand_records[reference_idx][label]["bbox"]
                plausible, metrics = _hand_transition_is_plausible(
                    reference_box,
                    box,
                    gap,
                    width,
                    height,
                )
                decisions.append((plausible, metrics, reference_idx))
            if any(decision[0] for decision in decisions):
                continue
            frame_record.pop(label, None)
            removed_count += 1
            if len(removed) < 50:
                removed.append(
                    {
                        "frame_index_0based": frame_idx,
                        "hand": label,
                        "reason": "inconsistent_with_nearby_locked_sam3_instance",
                        "references": [
                            {
                                "frame_index_0based": reference_idx,
                                **{
                                    key: round(float(value), 6)
                                    for key, value in metrics.items()
                                },
                            }
                            for _plausible, metrics, reference_idx in decisions
                        ],
                    }
                )
    return {"removed_count": removed_count, "removed_sample": removed}


def _fuse_sam3_hands(
    hand_records: List[Dict[str, Any]],
    payload: Dict[str, Any],
    max_gap: int,
    width: int,
    height: int,
) -> Dict[str, Any]:
    frames = _sam3_request_frames(payload, "__hands__")
    assignments, assignment_meta = _assign_primary_sam3_hand_instances(
        hand_records,
        frames,
        width,
        height,
        _sam3_hand_segment_ranges(payload, len(hand_records)),
    )

    replaced = 0
    filled = 0
    secondary_wilor_rejections = 0
    selected_instance_observations = 0
    selected_instance_transition_rejections = 0
    selected_instance_rejection_sample: List[Dict[str, Any]] = []
    concurrent_assigned_instance_conflicts = 0
    concurrent_conflict_sample: List[Dict[str, Any]] = []
    previous_locked: Dict[str, Tuple[int, Sequence[float], str]] = {}
    assigned_instance_ids = set(assignments)
    for frame_idx in sorted(frames):
        candidates = frames[frame_idx]
        if not 0 <= frame_idx < len(hand_records):
            continue
        record = hand_records[frame_idx]
        candidates_by_id = {
            str(candidate.get("instance_id") or ""): candidate
            for candidate in candidates
            if candidate.get("instance_id") and candidate.get("bbox_xyxy_pixel")
        }
        for label in HAND_LABELS:
            original_box = record.get(label, {}).get("bbox")
            label_candidates = [
                (instance_id, candidate)
                for instance_id, candidate in candidates_by_id.items()
                if assignments.get(instance_id) == label
            ]
            if len(label_candidates) > 1:
                concurrent_assigned_instance_conflicts += 1
                if len(concurrent_conflict_sample) < 50:
                    concurrent_conflict_sample.append(
                        {
                            "frame_index_0based": frame_idx,
                            "hand": label,
                            "sam3_instance_ids": sorted(
                                instance_id for instance_id, _candidate in label_candidates
                            ),
                        }
                    )
            instance_id: Optional[str] = None
            candidate: Optional[Dict[str, Any]] = None
            if label_candidates:
                instance_id, candidate = max(
                    label_candidates,
                    key=lambda item: (
                        _hand_match_score(
                            original_box, item[1]["bbox_xyxy_pixel"]
                        )
                        if original_box
                        else 0.0,
                        float(item[1].get("confidence", 0.0) or 0.0),
                    ),
                )
            if candidate is not None:
                box = candidate.get("bbox_xyxy_pixel")
                previous_instance = previous_locked.get(label)
                if previous_instance and box:
                    previous_frame, previous_box, previous_instance_id = previous_instance
                    transition_validator = (
                        _locked_sam3_instance_transition_is_plausible
                        if str(previous_instance_id) == str(instance_id)
                        else _hand_transition_is_plausible
                    )
                    plausible, transition = transition_validator(
                        previous_box, box, frame_idx - previous_frame, width, height
                    )
                    if not plausible:
                        selected_instance_transition_rejections += 1
                        if len(selected_instance_rejection_sample) < 50:
                            selected_instance_rejection_sample.append(
                                {
                                    "frame_index_0based": frame_idx,
                                    "previous_frame_index_0based": previous_frame,
                                    "hand": label,
                                    "sam3_instance_id": str(instance_id),
                                    **{
                                        key: round(float(value), 6)
                                        for key, value in transition.items()
                                    },
                                }
                            )
                        original_box = record.get(label, {}).get("bbox")
                        original_plausible = False
                        if original_box:
                            original_plausible, _ = _hand_transition_is_plausible(
                                previous_box,
                                original_box,
                                frame_idx - previous_frame,
                                width,
                                height,
                            )
                        if not original_plausible:
                            record.pop(label, None)
                        continue
                original = record.get(label) or {}
                original_box = original.get("bbox")
                direct_score = (
                    _hand_match_score(original_box, box)
                    if original_box and box
                    else 0.0
                )
                original_confidence = float(original.get("confidence", 0.0) or 0.0)
                candidate_confidence = float(candidate.get("confidence", 0.0) or 0.0)
                record[label] = {
                    "bbox": list(box),
                    "confidence": (
                        max(original_confidence, candidate_confidence)
                        if direct_score >= 0.10
                        else candidate_confidence
                    ),
                    "source": (
                        "sam3_box+wilor_handedness"
                        if direct_score >= 0.10
                        else "sam3_track+primary_instance_lock"
                    ),
                    "bbox_extent": "observed_full",
                    "primary_operator": True,
                    "primary_operator_policy": PRIMARY_OPERATOR_POLICY,
                    "sam3_instance_id": str(instance_id),
                    "primary_hand_chain_id": f"primary_operator_001:{label}",
                    "primary_instance_locked": True,
                }
                previous_locked[label] = (frame_idx, list(box), str(instance_id))
                selected_instance_observations += 1
                if original_box:
                    replaced += 1
                else:
                    filled += 1
                continue

            original_box = record.get(label, {}).get("bbox")
            if not original_box:
                continue
            secondary_match = max(
                (
                    _hand_match_score(original_box, candidate["bbox_xyxy_pixel"])
                    for instance_id, candidate in candidates_by_id.items()
                    if instance_id not in assigned_instance_ids
                ),
                default=0.0,
            )
            if secondary_match >= 0.10:
                record.pop(label, None)
                secondary_wilor_rejections += 1
    temporal_pruning = _prune_hand_records_outside_primary_instance(
        hand_records, width, height
    )
    _fill_short_hand_gaps(hand_records, max(0, int(max_gap)))
    return {
        "sam3_frames_with_hands": len(frames),
        "direct_matches": assignment_meta["direct_matches"],
        "stable_instances_with_handedness": assignment_meta["selected_instance_count"],
        "primary_instance_assignments": assignment_meta["segments"],
        "primary_instance_assignment_details": assignment_meta["instance_assignments"],
        "continuity_linked_fragment_count": assignment_meta[
            "continuity_linked_fragment_count"
        ],
        "wilor_boxes_replaced_by_sam3": replaced,
        "missing_wilor_boxes_filled_by_sam3": filled,
        "selected_primary_instance_observations": selected_instance_observations,
        "selected_instance_transition_rejections": selected_instance_transition_rejections,
        "selected_instance_transition_rejection_sample": selected_instance_rejection_sample,
        "concurrent_assigned_instance_conflicts": concurrent_assigned_instance_conflicts,
        "concurrent_assigned_instance_conflict_sample": concurrent_conflict_sample,
        "secondary_wilor_boxes_rejected": secondary_wilor_rejections,
        "temporal_identity_pruning": temporal_pruning,
    }


def _is_active_for_spec(spec: Dict[str, Any], frame_idx: int) -> bool:
    return any(
        int(start) <= frame_idx < int(end)
        for start, end in spec.get("active_intervals", [])
    )


def _relevant_hand_boxes_for_spec(
    spec: Dict[str, Any],
    frame_idx: int,
    hand_records: Sequence[Dict[str, Any]],
    step_bounds: Dict[int, Tuple[int, int]],
) -> List[Sequence[float]]:
    if not 0 <= frame_idx < len(hand_records):
        return []
    active_steps = {
        step_id
        for step_id, (start, end) in step_bounds.items()
        if start <= frame_idx < end
    }
    relations = [
        relation
        for relation in spec.get("hand_relations", [])
        if int(relation.get("step_id", -1)) in active_steps
    ]
    if not relations:
        relations = list(spec.get("hand_relations", []))
    labels: set[str] = set()
    for relation in relations:
        hand = _label_key(relation.get("hand"))
        if hand in {"left hand", "left_hand", "both hands", "both_hands"}:
            labels.add("left_hand")
        if hand in {"right hand", "right_hand", "both hands", "both_hands"}:
            labels.add("right_hand")
    if relations and not labels:
        labels.update(HAND_LABELS)
    return [
        hand_records[frame_idx][label]["bbox"]
        for label in sorted(labels)
        if hand_records[frame_idx].get(label, {}).get("bbox")
    ]


def _select_sam3_object_tracks(
    specs: Sequence[Dict[str, Any]],
    steps_by_id: Dict[int, Dict[str, Any]],
    payload: Dict[str, Any],
    hand_records: Sequence[Dict[str, Any]],
    fps: float,
    total_frames: int,
    width: int,
    height: int,
) -> Tuple[Dict[str, Dict[int, Dict[str, Any]]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    step_bounds = {
        step_id: _step_bounds(step, fps, total_frames)
        for step_id, step in steps_by_id.items()
    }
    raw_tracks: Dict[str, Dict[int, Dict[str, Any]]] = {}
    anchors: Dict[str, Dict[str, Any]] = {}
    selection_meta: Dict[str, Any] = {}
    for spec in specs:
        track_id = str(spec["track_id"])
        track_segments: Dict[str, Any] = {}
        frames = _sam3_request_frames(payload, track_id)
        instances: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
        for frame_idx, candidates in frames.items():
            for candidate in candidates:
                instance_id = str(candidate.get("instance_id") or "")
                if instance_id and candidate.get("bbox_xyxy_pixel"):
                    instances[instance_id][frame_idx] = candidate
        by_segment: Dict[str, List[str]] = defaultdict(list)
        for instance_id in instances:
            by_segment[instance_id.split("_obj", 1)[0]].append(instance_id)

        selected_ids: List[str] = []
        selected_raw: Dict[int, Dict[str, Any]] = {}
        anchor_candidates: List[Tuple[float, float, int, Dict[str, Any]]] = []
        for segment_id, instance_ids in sorted(by_segment.items()):
            ranked: List[Tuple[float, int, float, str]] = []
            per_instance_stats: Dict[str, Tuple[float, float, int, float]] = {}
            for instance_id in instance_ids:
                active_entries = [
                    (frame_idx, candidate)
                    for frame_idx, candidate in instances[instance_id].items()
                    if _is_active_for_spec(spec, frame_idx)
                ]
                if not active_entries:
                    continue
                confidences: List[float] = []
                interactions: List[float] = []
                duplicate_hand_frames = 0
                for frame_idx, candidate in active_entries:
                    confidence = float(candidate.get("confidence", 0.0) or 0.0)
                    hand_boxes = _relevant_hand_boxes_for_spec(
                        spec, frame_idx, hand_records, step_bounds
                    )
                    hand_iou = max(
                        (
                            _box_iou(candidate.get("bbox_xyxy_pixel"), hand_box)
                            for hand_box in hand_boxes
                        ),
                        default=0.0,
                    )
                    if hand_iou > 0.80:
                        duplicate_hand_frames += 1
                        continue
                    interaction = _hand_object_interaction_score(
                        candidate.get("bbox_xyxy_pixel"),
                        hand_boxes,
                        width,
                        height,
                    )
                    confidences.append(confidence)
                    interactions.append(interaction)
                if not confidences:
                    continue
                mean_confidence = sum(confidences) / len(confidences)
                max_interaction = max(interactions, default=0.0)
                mean_interaction = sum(interactions) / len(interactions)
                coverage = len(confidences)
                duplicate_fraction = duplicate_hand_frames / max(1, len(active_entries))
                score = (
                    0.45 * mean_confidence
                    + 0.35 * max_interaction
                    + 0.15 * mean_interaction
                    + 0.05 * min(1.0, coverage / 10.0)
                    - 0.60 * duplicate_fraction
                )
                ranked.append((score, coverage, mean_confidence, instance_id))
                per_instance_stats[instance_id] = (
                    max_interaction,
                    mean_interaction,
                    coverage,
                    duplicate_fraction,
                )
            if not ranked:
                continue
            _score, _coverage, _confidence, selected_id = max(ranked)
            selected_ids.append(selected_id)
            (
                max_interaction,
                _mean_interaction,
                _coverage,
                duplicate_fraction,
            ) = per_instance_stats[selected_id]
            for frame_idx, candidate in instances[selected_id].items():
                box = _clamp_box(candidate.get("bbox_xyxy_pixel", []), width, height)
                if box is None:
                    continue
                confidence = float(candidate.get("confidence", 0.0) or 0.0)
                hand_boxes = _relevant_hand_boxes_for_spec(
                    spec, frame_idx, hand_records, step_bounds
                )
                if max((_box_iou(box, hand_box) for hand_box in hand_boxes), default=0.0) > 0.80:
                    continue
                interaction = _hand_object_interaction_score(box, hand_boxes, width, height)
                selected_raw[frame_idx] = {
                    "bbox": box,
                    "confidence": confidence,
                    "source": "sam3_text_video",
                }
                if _is_active_for_spec(spec, frame_idx):
                    anchor_candidates.append((interaction, confidence, frame_idx, candidate))
            track_segments[segment_id] = {
                "selected_instance_id": selected_id,
                "selected_coverage": _coverage,
                "selected_max_interaction": round(max_interaction, 6),
                "selected_hand_duplicate_fraction": round(duplicate_fraction, 6),
            }
        if not selected_raw or not anchor_candidates:
            continue
        interaction, confidence, anchor_idx, anchor_candidate = max(
            anchor_candidates, key=lambda item: item[:3]
        )
        anchor_box = _clamp_box(anchor_candidate.get("bbox_xyxy_pixel", []), width, height)
        if anchor_box is None:
            continue
        raw_tracks[track_id] = selected_raw
        anchors[track_id] = {
            "frame_index_0based": anchor_idx,
            "bbox_xyxy_pixel": anchor_box,
            "confidence": confidence,
            "interaction_score": interaction,
            "source": "sam3_text_video",
            "visibility": "full",
            "sam3_instance_id": str(anchor_candidate.get("instance_id") or ""),
        }
        selection_meta[track_id] = {
            "label": str(spec["label"]),
            "selected_instance_ids": selected_ids,
            "observed_frames": len(selected_raw),
            "segments": track_segments,
        }
    return raw_tracks, anchors, selection_meta


def _empty_grounding_meta() -> Dict[str, Any]:
    return {
        "local_grounding_error": "",
        "api_usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "api_calls": 0,
            "api_attempts": 0,
            "cache_hits": 0,
        },
        "api_calls_by_step": {},
    }


def _write_anchor_artifacts(
    specs: Sequence[Dict[str, Any]],
    anchors: Dict[str, Dict[str, Any]],
    frame_paths: Sequence[Path],
    stage5_dir: Path,
    width: int,
    height: int,
) -> None:
    anchor_dir = stage5_dir / "anchors"
    anchor_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        anchor = anchors.get(str(spec["track_id"]))
        if anchor is None:
            continue
        frame_idx = int(anchor["frame_index_0based"])
        if not 0 <= frame_idx < len(frame_paths):
            continue
        image_name = f"{spec['track_id']}_frame_{frame_idx:06d}.jpg"
        image_path = anchor_dir / image_name
        if not image_path.exists():
            shutil.copyfile(frame_paths[frame_idx], image_path)
        anchor["image_relpath"] = str(Path("anchors") / image_name)
        anchor["bbox_xyxy_normalized_0_999"] = _pixel_to_official(
            anchor["bbox_xyxy_pixel"], width, height
        )


class LocalGrounder:
    def __init__(self, model_id: str, device: str, box_threshold: float, text_threshold: float):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.device = device
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.model.eval()

    def detect(self, frame_path: Path, labels: Sequence[str]) -> List[Dict[str, Any]]:
        import inspect

        from PIL import Image

        image = Image.open(frame_path).convert("RGB")
        prompt = ". ".join(_normalize_label(label) for label in labels if _normalize_label(label)) + "."
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        moved = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with self.torch.inference_mode():
            outputs = self.model(**moved)
        post_process = self.processor.post_process_grounded_object_detection
        threshold_name = (
            "box_threshold"
            if "box_threshold" in inspect.signature(post_process).parameters
            else "threshold"
        )
        result = post_process(
            outputs,
            inputs.input_ids,
            **{
                threshold_name: self.box_threshold,
                "text_threshold": self.text_threshold,
                "target_sizes": [image.size[::-1]],
            },
        )[0]
        text_labels = result.get("text_labels")
        if text_labels is None:
            text_labels = result.get("labels")
        if text_labels is None:
            text_labels = []
        boxes = result.get("boxes")
        if boxes is None:
            boxes = []
        scores = result.get("scores")
        if scores is None:
            scores = []
        detections: List[Dict[str, Any]] = []
        for label, box, score in zip(text_labels, boxes, scores):
            raw_label = str(label)
            if raw_label.isdigit() and int(raw_label) < len(labels):
                raw_label = str(labels[int(raw_label)])
            raw_box = box.detach().cpu().tolist() if hasattr(box, "detach") else list(box)
            raw_score = float(score.detach().cpu().item()) if hasattr(score, "detach") else float(score)
            detections.append({"label": _normalize_label(raw_label), "bbox": raw_box, "score": raw_score})
        return detections


class ArkGrounder:
    source = "ark_grounding"
    task = "manipulated_object"
    preserve_not_visible = False

    def __init__(
        self,
        api_cfg: "ApiConfig",
        cache_dir: Path,
        max_calls_per_step: int,
        max_total_calls: int = 0,
        thinking_type: str = "enabled",
        reasoning_effort: str = DEFAULT_ARK_REASONING_EFFORT,
    ):
        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_cfg.api_key,
            base_url=api_cfg.api_base_url,
            max_retries=max(0, int(api_cfg.api_call_retries)),
            timeout=max(1.0, float(api_cfg.api_call_timeout_sec)),
        )
        self.model = api_cfg.model_name
        self.max_tokens = min(1500, max(256, int(api_cfg.max_tokens)))
        self.thinking_type = str(thinking_type or "enabled")
        self.reasoning_effort = str(
            reasoning_effort or DEFAULT_ARK_REASONING_EFFORT
        )
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_calls_per_step = max(0, int(max_calls_per_step))
        self.max_total_calls = max(0, int(max_total_calls))
        self.attempts = 0
        self.calls_by_step: Counter[int] = Counter()
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "api_calls": 0,
            "api_attempts": 0,
            "cache_hits": 0,
        }

    @staticmethod
    def _prompt(
        labels: Sequence[str],
        contexts: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> str:
        contexts = list(contexts or [])
        targets = []
        for i, label in enumerate(labels, start=1):
            context = contexts[i - 1] if i - 1 < len(contexts) else {}
            targets.append(
                {
                    "target_id": f"target_{i:03d}",
                    "label": label,
                    **context,
                }
            )
        return f"""Ground the following task-relevant manipulated objects in this single image.

Targets:
{json.dumps(targets, ensure_ascii=False)}

All targets belong only to the primary operator whose sustained hand-pair manipulation workspace is closest to the image center. Ignore every object manipulated only by secondary hands near an image edge or in the background, even when that secondary object is visually clearer or has the same appearance. For duplicate or similar-looking instances, select the object participating in the primary operator's specified hand-object relation. The optional hand_bbox_normalized_0_999 fields identify the already-selected primary hand and are disambiguation hints, not target boxes. For every target, return exactly one JSON item. Estimate the complete physical object bounding box when it is partially occluded. If it is fully occluded or not visible, use visibility=not_visible and bbox=null. Do not confuse a held tool with the hand holding it.

Coordinates must use the official 1000x1000 normalized range [0,999], top-left origin. A visible box must use the exact string form <bbox>x1 y1 x2 y2</bbox>.

Return only a JSON array with this exact schema and no markdown:
[
  {{"target_id":"target_001","label":"same label as input","visibility":"full|partial|not_visible","bbox":"<bbox>x1 y1 x2 y2</bbox>"}}
]"""

    @staticmethod
    def _strip_json_fence(text: str) -> str:
        value = str(text or "").strip()
        if value.startswith("```"):
            lines = value.splitlines()[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            value = "\n".join(lines).strip()
        return value

    def ground(
        self,
        frame_path: Path,
        labels: Sequence[str],
        contexts: Sequence[Dict[str, Any]],
        step_id: int,
        width: int,
        height: int,
    ) -> Dict[str, Dict[str, Any]]:
        return self._ground_images(
            [("TARGET", frame_path)], labels, contexts, step_id, width, height
        )

    def ground_temporal(
        self,
        target_frame_path: Path,
        context_frames: Sequence[Tuple[str, Path]],
        labels: Sequence[str],
        contexts: Sequence[Dict[str, Any]],
        step_id: int,
        width: int,
        height: int,
    ) -> Dict[str, Dict[str, Any]]:
        previous = [item for item in context_frames if str(item[0]).startswith("PREVIOUS")]
        following = [item for item in context_frames if str(item[0]).startswith("NEXT")]
        return self._ground_images(
            [*previous, ("TARGET", target_frame_path), *following],
            labels,
            contexts,
            step_id,
            width,
            height,
        )

    def _ground_images(
        self,
        image_specs: Sequence[Tuple[str, Path]],
        labels: Sequence[str],
        contexts: Sequence[Dict[str, Any]],
        step_id: int,
        width: int,
        height: int,
    ) -> Dict[str, Dict[str, Any]]:
        if not labels:
            return {}
        prompt = self._prompt(labels, contexts)
        digest = hashlib.sha256()
        encoded_images: List[Tuple[str, Path, str]] = []
        for role, frame_path in image_specs:
            frame_bytes = frame_path.read_bytes()
            suffix = frame_path.suffix.lower()
            mime = "image/png" if suffix == ".png" else "image/jpeg"
            encoded_images.append(
                (
                    str(role),
                    frame_path,
                    f"data:{mime};base64,{base64.b64encode(frame_bytes).decode('ascii')}",
                )
            )
            digest.update(str(role).encode("utf-8"))
            digest.update(len(frame_bytes).to_bytes(8, "big"))
            digest.update(frame_bytes)
        digest.update(prompt.encode("utf-8"))
        digest.update(self.model.encode("utf-8"))
        digest.update(self.thinking_type.encode("ascii"))
        digest.update(self.reasoning_effort.encode("ascii"))
        digest.update(STAGE5_SCHEMA_VERSION.encode("ascii"))
        key = digest.hexdigest()
        cache_path = self.cache_dir / f"{key}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            self.usage["cache_hits"] += 1
            return dict(cached.get("detections") or {})
        if self.calls_by_step[int(step_id)] >= self.max_calls_per_step:
            return {}
        if self.max_total_calls and self.attempts >= self.max_total_calls:
            return {}

        self.calls_by_step[int(step_id)] += 1
        self.attempts += 1
        self.usage["api_attempts"] += 1
        content: List[Dict[str, Any]] = []
        if len(encoded_images) == 1:
            content.append(
                {"type": "image_url", "image_url": {"url": encoded_images[0][2]}}
            )
        else:
            for role, _frame_path, data_url in encoded_images:
                content.append(
                    {
                        "type": "text",
                        "text": (
                            f"{role} image. Only TARGET coordinates may be returned; "
                            "context images are identity evidence only."
                        ),
                    }
                )
                content.append({"type": "image_url", "image_url": {"url": data_url}})
        content.append({"type": "text", "text": prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            max_tokens=self.max_tokens,
            reasoning_effort=self.reasoning_effort,
            extra_body={"thinking": {"type": self.thinking_type}},
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
        )
        raw = response.choices[0].message.content
        if not isinstance(raw, str):
            raise RuntimeError("Ark grounding returned non-text content.")
        parsed = json.loads(self._strip_json_fence(raw))
        if not isinstance(parsed, list):
            raise RuntimeError("Ark grounding output must be a JSON array.")
        label_by_id = {f"target_{i:03d}": label for i, label in enumerate(labels, start=1)}
        detections: Dict[str, Dict[str, Any]] = {}
        for item in parsed:
            if not isinstance(item, dict):
                continue
            target_id = str(item.get("target_id") or "").strip()
            label = label_by_id.get(target_id, _normalize_label(item.get("label")))
            if not label:
                continue
            visibility = str(item.get("visibility") or "").strip().lower()
            bbox_value = item.get("bbox")
            if visibility == "not_visible" or bbox_value is None:
                if self.preserve_not_visible and visibility == "not_visible":
                    reason = str(item.get("not_visible_reason") or "unknown").strip().lower()
                    if reason not in {"occluded", "out_of_frame", "absent", "unknown"}:
                        reason = "unknown"
                    detections[_label_key(label)] = {
                        "label": label,
                        "bbox": None,
                        "score": 1.0,
                        "visibility": "not_visible",
                        "not_visible_reason": reason,
                        "source": self.source,
                    }
                continue
            match = _BBOX_TAG_RE.fullmatch(str(bbox_value).strip())
            if not match:
                continue
            official = [int(v) for v in match.groups()]
            if not all(0 <= value <= OFFICIAL_COORD_MAX for value in official):
                continue
            pixel = _official_to_pixel(official, width, height)
            if pixel is None:
                continue
            detections[_label_key(label)] = {
                "label": label,
                "bbox": pixel,
                "score": 1.0,
                "visibility": visibility if visibility in {"full", "partial"} else "full",
                "source": self.source,
                "bbox_normalized_0_999": official,
            }

        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
        self.usage["prompt_tokens"] += prompt_tokens
        self.usage["completion_tokens"] += completion_tokens
        self.usage["total_tokens"] += total_tokens
        self.usage["api_calls"] += 1
        cache_path.write_text(
            json.dumps(
                {
                    "model": self.model,
                    "task": self.task,
                    "thinking_type": self.thinking_type,
                    "reasoning_effort": self.reasoning_effort,
                    "frames": [
                        {"role": role, "path": str(frame_path)}
                        for role, frame_path, _data_url in encoded_images
                    ],
                    "labels": list(labels),
                    "contexts": list(contexts),
                    "raw": raw,
                    "detections": detections,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return detections


class ArkHandGrounder(ArkGrounder):
    source = "ark_hand_grounding"
    task = "anatomical_hand"
    preserve_not_visible = True

    @staticmethod
    def _prompt(
        labels: Sequence[str],
        contexts: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> str:
        contexts = list(contexts or [])
        targets = []
        for i, label in enumerate(labels, start=1):
            context = contexts[i - 1] if i - 1 < len(contexts) else {}
            targets.append(
                {
                    "target_id": f"target_{i:03d}",
                    "label": label,
                    **context,
                }
            )
        return f"""Ground the requested anatomical human hands in the TARGET first-person task image.

Targets:
{json.dumps(targets, ensure_ascii=False)}

A hand may be completely covered by a black, dark, or light work glove. Treat the glove as the hand and include the complete visible glove/hand through the wrist. Exclude the forearm and every held or touched object. Do not mistake a dark glove for an object or background.

PRIMARY-OPERATOR RULE: when multiple people's hands are visible, return only the one temporally consistent hand pair whose manipulation workspace is closest to the image center. Ignore all hands near the image edges or in the background, and never substitute a secondary person's hand when the primary hand is occluded. PREVIOUS/NEXT images, when present, are identity context only; the returned box must describe the TARGET image. The optional previous_known_bbox/next_known_bbox and other_hand_bbox fields are detections of that selected primary operator and must be used as identity-continuity hints.

The labels left_hand and right_hand mean the primary operator's anatomical left and right hands, not the left and right sides of the image. Use arm attachment, pose, task context, and the temporal hints to preserve that identity. Never copy a hint when it does not match the current image. Return one item for every target. If the requested primary hand is genuinely not visible, return visibility=not_visible, a specific not_visible_reason, and JSON bbox=null rather than returning an edge/background hand. For a visible hand use not_visible_reason=unknown.

Coordinates must use the official 1000x1000 normalized range [0,999], top-left origin. A visible box must use the exact string form <bbox>x1 y1 x2 y2</bbox>.

Return only a JSON array with this exact schema and no markdown:
[
  {{"target_id":"target_001","label":"same label as input","visibility":"full|partial|not_visible","not_visible_reason":"occluded|out_of_frame|absent|unknown","bbox":"<bbox>x1 y1 x2 y2</bbox>"}}
]"""


def _collect_hand_steps(specs: Sequence[Dict[str, Any]]) -> Dict[str, set[int]]:
    hand_steps: Dict[str, set[int]] = {label: set() for label in HAND_LABELS}
    for spec in specs:
        for relation in spec.get("hand_relations", []):
            hand = _label_key(relation.get("hand"))
            try:
                step_id = int(relation.get("step_id"))
            except Exception:
                continue
            if hand in {"left hand", "left_hand", "both hands", "both_hands"}:
                hand_steps["left_hand"].add(step_id)
            if hand in {"right hand", "right_hand", "both hands", "both_hands"}:
                hand_steps["right_hand"].add(step_id)
    return hand_steps


def _collect_expected_hand_steps(specs: Sequence[Dict[str, Any]]) -> Dict[str, set[int]]:
    expected = _collect_hand_steps(specs)
    for spec in specs:
        for relation in spec.get("hand_relations", []):
            if _label_key(relation.get("hand")) not in {"unknown hand", "unknown_hand"}:
                continue
            try:
                step_id = int(relation.get("step_id"))
            except Exception:
                continue
            # Laterality is unknown semantically, so either anatomical hand may need
            # recovery. This affects assist scheduling only, not relation semantics.
            expected["left_hand"].add(step_id)
            expected["right_hand"].add(step_id)
    return expected


def _resolve_duplicate_primary_hand_boxes(
    hand_records: List[Dict[str, Any]],
    specs: Sequence[Dict[str, Any]],
    steps_by_id: Dict[int, Dict[str, Any]],
    fps: float,
    iou_threshold: float = 0.75,
) -> Dict[str, Any]:
    expected_steps = _collect_expected_hand_steps(specs)
    step_bounds = {
        step_id: _step_bounds(step, fps, len(hand_records))
        for step_id, step in steps_by_id.items()
    }
    resolved_count = 0
    sample: List[Dict[str, Any]] = []
    source_priority = {
        "ark_hand_grounding": 4,
        "sam3_box+wilor_handedness": 3,
        "wilor_yolo": 2,
        "sam3_track+primary_instance_lock": 1,
        "temporal_interpolation": 0,
    }
    for frame_idx, record in enumerate(hand_records):
        left = record.get("left_hand") or {}
        right = record.get("right_hand") or {}
        left_box, right_box = left.get("bbox"), right.get("bbox")
        if not left_box or not right_box:
            continue
        overlap = _box_iou(left_box, right_box)
        if overlap <= float(iou_threshold):
            continue
        active_steps = {
            step_id
            for step_id, (start, end) in step_bounds.items()
            if start <= frame_idx < end
        }
        expected_labels = {
            label
            for label in HAND_LABELS
            if active_steps & expected_steps[label]
        }
        if len(expected_labels) == 1:
            kept_label = next(iter(expected_labels))
            decision_mode = "single_relation_expected_hand"
        else:
            kept_label = max(
                HAND_LABELS,
                key=lambda label: (
                    source_priority.get(str(record[label].get("source") or ""), -1),
                    float(record[label].get("confidence", 0.0) or 0.0),
                    _box_area(record[label].get("bbox")),
                ),
            )
            decision_mode = (
                "both_relations_evidence_priority"
                if len(expected_labels) == 2
                else "no_relation_evidence_priority"
            )
        dropped_label = "right_hand" if kept_label == "left_hand" else "left_hand"
        dropped = record.pop(dropped_label)
        resolved_count += 1
        if len(sample) < 50:
            sample.append(
                {
                    "frame_index_0based": frame_idx,
                    "iou": round(float(overlap), 6),
                    "active_step_ids": sorted(active_steps),
                    "expected_hands": sorted(expected_labels),
                    "kept_hand": kept_label,
                    "dropped_hand": dropped_label,
                    "kept_source": record[kept_label].get("source"),
                    "dropped_source": dropped.get("source"),
                    "decision_mode": decision_mode,
                }
            )
    return {
        "iou_threshold": float(iou_threshold),
        "resolved_duplicate_frame_count": resolved_count,
        "resolution_sample": sample,
    }


def _hand_detection_is_weak(
    records: Sequence[Dict[str, Any]], frame_idx: int, label: str, threshold: float
) -> bool:
    if not 0 <= frame_idx < len(records):
        return True
    detection = records[frame_idx].get(label) or {}
    return not detection.get("bbox") or float(detection.get("confidence", 0.0) or 0.0) < threshold


def _contiguous_runs(indices: Sequence[int]) -> List[Tuple[int, int]]:
    ordered = sorted(set(int(value) for value in indices))
    if not ordered:
        return []
    runs: List[Tuple[int, int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value != previous + 1:
            runs.append((start, previous + 1))
            start = value
        previous = value
    runs.append((start, previous + 1))
    return runs


def _nearest_known_hand_hint(
    records: Sequence[Dict[str, Any]],
    frame_idx: int,
    label: str,
    direction: int,
    threshold: float,
    width: int,
    height: int,
    max_distance: int = 60,
) -> Optional[Dict[str, Any]]:
    for distance in range(1, max(1, int(max_distance)) + 1):
        index = frame_idx + direction * distance
        if not 0 <= index < len(records):
            break
        detection = records[index].get(label) or {}
        box = detection.get("bbox")
        if box and float(detection.get("confidence", 0.0) or 0.0) >= threshold:
            normalized = _pixel_to_official(box, width, height)
            if normalized is not None:
                return {"frame_index_0based": index, "bbox_normalized_0_999": normalized}
    return None


def _ark_hand_anchor_in_primary_scope(
    records: Sequence[Dict[str, Any]],
    frame_idx: int,
    label: str,
    candidate_box: Sequence[int],
    threshold: float,
    width: int,
    height: int,
    max_distance: int = 60,
) -> Tuple[bool, Dict[str, Any]]:
    image_center = (float(width) / 2.0, float(height) / 2.0)
    single_centrality = _point_proximity_score(
        _box_center(candidate_box), image_center, width, height, 0.50
    )
    other_label = "right_hand" if label == "left_hand" else "left_hand"
    other_box = records[frame_idx].get(other_label, {}).get("bbox")
    pair_centrality = 0.0
    if other_box:
        candidate = {"bbox": list(candidate_box), "confidence": 1.0}
        other = {"bbox": list(other_box), "confidence": 1.0}
        pair = (
            _primary_pair_features(candidate, other, width, height)
            if label == "left_hand"
            else _primary_pair_features(other, candidate, width, height)
        )
        if pair is not None:
            pair_centrality = float(pair["centrality"])

    continuity = 0.0
    temporal_reference_count = 0
    for direction in (-1, 1):
        for distance in range(1, max(1, int(max_distance)) + 1):
            index = frame_idx + direction * distance
            if not 0 <= index < len(records):
                break
            known = records[index].get(label) or {}
            known_box = known.get("bbox")
            if not known_box or float(known.get("confidence", 0.0) or 0.0) < threshold:
                continue
            temporal_reference_count += 1
            motion_radius = min(0.35, 0.07 + 0.012 * float(distance))
            continuity = max(
                continuity,
                _box_iou(candidate_box, known_box),
                _point_proximity_score(
                    _box_center(candidate_box),
                    _box_center(known_box),
                    width,
                    height,
                    motion_radius,
                ),
            )
            break
    if temporal_reference_count:
        accepted = continuity >= 0.20 or (
            other_box is not None
            and pair_centrality >= 0.60
            and single_centrality >= 0.35
        )
        decision_mode = "temporal_identity_required"
    else:
        accepted = max(single_centrality, pair_centrality) >= 0.45
        decision_mode = "center_reacquisition_without_temporal_reference"
    return accepted, {
        "single_centrality": round(single_centrality, 6),
        "pair_centrality": round(pair_centrality, 6),
        "temporal_continuity": round(continuity, 6),
        "temporal_reference_count": temporal_reference_count,
        "decision_mode": decision_mode,
    }


def _run_sample_windows(start: int, end: int, fps: float) -> List[Tuple[int, int, int]]:
    """Return target frames and the bounded intervals each target can support."""
    if end <= start:
        return []
    length = end - start
    max_span = max(1, int(round(max(1.0, float(fps)) * ARK_HAND_CONFIRMATION_MAX_SECONDS)))
    count = min(
        ARK_HAND_CONFIRMATION_MAX_SAMPLES_PER_RUN,
        max(1, int(math.ceil(length / max_span))),
    )
    if length <= count * max_span:
        boundaries = [start + length * index // count for index in range(count)]
        boundaries.append(end)
        windows = list(zip(boundaries, boundaries[1:]))
    elif count == 1:
        window_start = start + (length - max_span) // 2
        windows = [(window_start, window_start + max_span)]
    else:
        available_offset = length - max_span
        windows = [
            (
                start + int(round(available_offset * index / (count - 1))),
                start
                + int(round(available_offset * index / (count - 1)))
                + max_span,
            )
            for index in range(count)
        ]
    return [
        (window_start + (window_end - window_start) // 2, window_start, window_end)
        for window_start, window_end in windows
        if window_end > window_start
    ]


def _assist_hands_with_ark(
    specs: Sequence[Dict[str, Any]],
    steps_by_id: Dict[int, Dict[str, Any]],
    frame_paths: Sequence[Path],
    hand_records: List[Dict[str, Any]],
    fps: float,
    width: int,
    height: int,
    cfg: BBoxConfig,
    api_cfg: "ApiConfig",
    stage5_dir: Path,
) -> Dict[str, Any]:
    empty_usage = _empty_grounding_meta()["api_usage"]
    if (
        not cfg.api_hand_assist
        or cfg.api_hand_max_calls_per_step <= 0
        or cfg.api_hand_max_total_calls <= 0
    ):
        return {
            "enabled": bool(cfg.api_hand_assist),
            "triggered": False,
            "api_usage": empty_usage,
            "calls_by_step": {},
            "accepted_anchors": [],
            "confirmed_not_visible_intervals": [],
            "confirmed_not_visible_frames": 0,
            "filled_frames": 0,
            "errors": [],
        }

    hand_steps = _collect_expected_hand_steps(specs)
    step_bounds = {
        step_id: _step_bounds(step, fps, len(frame_paths))
        for step_id, step in steps_by_id.items()
    }
    grounder = ArkHandGrounder(
        api_cfg,
        stage5_dir / "hand_grounding_cache",
        cfg.api_hand_max_calls_per_step,
        cfg.api_hand_max_total_calls,
        cfg.api_grounding_thinking,
        cfg.api_grounding_reasoning_effort,
    )
    schedule: List[Dict[str, Any]] = []
    trigger_runs: List[Dict[str, Any]] = []
    for step_id in sorted(step_bounds):
        expected_labels = [label for label in HAND_LABELS if step_id in hand_steps[label]]
        if not expected_labels:
            continue
        step_start, step_end = step_bounds[step_id]
        by_frame: Dict[int, Dict[str, Tuple[int, int]]] = defaultdict(dict)
        for label in expected_labels:
            weak_indices = [
                frame_idx
                for frame_idx in range(step_start, step_end)
                if _hand_detection_is_weak(
                    hand_records,
                    frame_idx,
                    label,
                    cfg.api_hand_low_conf_threshold,
                )
            ]
            for run_start, run_end in _contiguous_runs(weak_indices):
                run_length = run_end - run_start
                if run_length < cfg.api_hand_min_missing_run_frames:
                    continue
                trigger_runs.append(
                    {
                        "step_id": step_id,
                        "hand": label,
                        "start_frame_0based": run_start,
                        "end_frame_0based_exclusive": run_end,
                        "length_frames": run_length,
                    }
                )
                for frame_idx, window_start, window_end in _run_sample_windows(
                    run_start, run_end, fps
                ):
                    by_frame[frame_idx][label] = (window_start, window_end)
        ranked_frames = sorted(
            by_frame,
            key=lambda frame_idx: (
                -len(by_frame[frame_idx]),
                -max(end - start for start, end in by_frame[frame_idx].values()),
                frame_idx,
            ),
        )[: cfg.api_hand_max_calls_per_step]
        for frame_idx in sorted(ranked_frames):
            schedule.append(
                {
                    "step_id": step_id,
                    "frame_index_0based": frame_idx,
                    "targets": by_frame[frame_idx],
                }
            )

    proposals: Dict[str, Dict[int, Dict[str, Any]]] = {
        label: {} for label in HAND_LABELS
    }
    accepted_anchors: List[Dict[str, Any]] = []
    rejected_anchors: List[Dict[str, Any]] = []
    confirmed_not_visible_intervals: List[Dict[str, Any]] = []
    confirmed_not_visible_frames = 0
    errors: List[str] = []
    for request in schedule:
        step_id = int(request["step_id"])
        frame_idx = int(request["frame_index_0based"])
        targets: Dict[str, Tuple[int, int]] = request["targets"]
        labels = sorted(targets)
        contexts: List[Dict[str, Any]] = []
        temporal_frame_indices: set[int] = set()
        for label in labels:
            other_label = "right_hand" if label == "left_hand" else "left_hand"
            context: Dict[str, Any] = {
                "step_goal": _normalize_label(steps_by_id.get(step_id, {}).get("step_goal")),
                "operator_scope": "primary hand pair closest to image center only",
            }
            previous = _nearest_known_hand_hint(
                hand_records,
                frame_idx,
                label,
                -1,
                cfg.api_hand_low_conf_threshold,
                width,
                height,
            )
            following = _nearest_known_hand_hint(
                hand_records,
                frame_idx,
                label,
                1,
                cfg.api_hand_low_conf_threshold,
                width,
                height,
            )
            other_box = hand_records[frame_idx].get(other_label, {}).get("bbox")
            if previous:
                context["previous_known_bbox"] = previous
                temporal_frame_indices.add(int(previous["frame_index_0based"]))
            if following:
                context["next_known_bbox"] = following
                temporal_frame_indices.add(int(following["frame_index_0based"]))
            if other_box:
                context["other_hand_bbox"] = {
                    "label": other_label,
                    "bbox_normalized_0_999": _pixel_to_official(other_box, width, height),
                }
            contexts.append(context)
        context_frames = [
            (
                (
                    f"PREVIOUS frame {context_idx}"
                    if context_idx < frame_idx
                    else f"NEXT frame {context_idx}"
                ),
                frame_paths[context_idx],
            )
            for context_idx in sorted(temporal_frame_indices)
            if context_idx != frame_idx and 0 <= context_idx < len(frame_paths)
        ]
        request["context_frame_indices_0based"] = [
            context_idx for context_idx in sorted(temporal_frame_indices) if context_idx != frame_idx
        ]
        try:
            detections = grounder.ground_temporal(
                frame_paths[frame_idx],
                context_frames,
                labels,
                contexts,
                step_id,
                width,
                height,
            )
        except Exception as exc:
            message = f"step={step_id} frame={frame_idx}: {type(exc).__name__}: {exc}"
            errors.append(message)
            logger.warning(f"[stage5] Ark hand assist failed {message}")
            continue
        for label in labels:
            detection = detections.get(_label_key(label))
            if not detection:
                rejected_anchors.append(
                    {
                        "step_id": step_id,
                        "frame_index_0based": frame_idx,
                        "hand": label,
                        "bbox_xyxy_pixel": None,
                        "reason": "api_returned_not_visible_or_no_valid_bbox",
                        "run": list(targets[label]),
                    }
                )
                continue
            if detection.get("visibility") == "not_visible":
                run_start, run_end = targets[label]
                not_visible_reason = str(
                    detection.get("not_visible_reason") or "unknown"
                )
                if not_visible_reason not in ARK_HAND_CONFIRMED_NOT_VISIBLE_REASONS:
                    rejected_anchors.append(
                        {
                            "step_id": step_id,
                            "frame_index_0based": frame_idx,
                            "hand": label,
                            "bbox_xyxy_pixel": None,
                            "reason": "api_not_visible_reason_not_specific",
                            "reported_not_visible_reason": not_visible_reason,
                            "run": [run_start, run_end],
                        }
                    )
                    continue
                output_visibility = (
                    "occluded"
                    if not_visible_reason == "occluded"
                    else (
                        "not_present"
                        if not_visible_reason in {"out_of_frame", "absent"}
                        else "not_visible"
                    )
                )
                interval = {
                    "step_id": step_id,
                    "anchor_frame_index_0based": frame_idx,
                    "hand": label,
                    "run": [run_start, run_end],
                    "not_visible_reason": not_visible_reason,
                    "output_visibility": output_visibility,
                    "evidence": "ark_temporal_context",
                }
                confirmed_not_visible_intervals.append(interval)
                for interval_frame_idx in range(run_start, run_end):
                    existing = hand_records[interval_frame_idx].get(label) or {}
                    if existing.get("bbox") or existing.get("source") == "ark_hand_visibility_confirmation":
                        continue
                    hand_records[interval_frame_idx][label] = {
                        "bbox": None,
                        "confidence": 1.0,
                        "source": "ark_hand_visibility_confirmation",
                        "bbox_extent": None,
                        "visibility": output_visibility,
                        "not_visible_reason": not_visible_reason,
                        "primary_operator": True,
                        "primary_operator_policy": PRIMARY_OPERATOR_POLICY,
                        "identity_validation": "ark_temporal_context_not_visible",
                    }
                    confirmed_not_visible_frames += 1
                continue
            box = _clamp_box(detection.get("bbox", []), width, height)
            if box is None:
                rejected_anchors.append(
                    {
                        "step_id": step_id,
                        "frame_index_0based": frame_idx,
                        "hand": label,
                        "bbox_xyxy_pixel": None,
                        "reason": "api_returned_invalid_bbox",
                        "run": list(targets[label]),
                    }
                )
                continue
            area_ratio = _box_area(box) / max(1.0, float(width * height))
            other_label = "right_hand" if label == "left_hand" else "left_hand"
            other_box = hand_records[frame_idx].get(other_label, {}).get("bbox")
            if area_ratio < 0.0002 or area_ratio > 0.45 or (other_box and _box_iou(box, other_box) > 0.75):
                rejected_anchors.append(
                    {
                        "step_id": step_id,
                        "frame_index_0based": frame_idx,
                        "hand": label,
                        "bbox_xyxy_pixel": box,
                        "reason": "implausible_size_or_duplicate_other_hand",
                    }
                )
                continue
            in_primary_scope, scope_scores = _ark_hand_anchor_in_primary_scope(
                hand_records,
                frame_idx,
                label,
                box,
                cfg.api_hand_low_conf_threshold,
                width,
                height,
            )
            if not in_primary_scope:
                rejected_anchors.append(
                    {
                        "step_id": step_id,
                        "frame_index_0based": frame_idx,
                        "hand": label,
                        "bbox_xyxy_pixel": box,
                        "reason": "outside_primary_center_operator_scope",
                        "scope_scores": scope_scores,
                    }
                )
                continue
            run_start, run_end = targets[label]
            accepted_anchors.append(
                {
                    "step_id": step_id,
                    "frame_index_0based": frame_idx,
                    "hand": label,
                    "bbox_xyxy_pixel": box,
                    "visibility": detection.get("visibility", "full"),
                    "run": [run_start, run_end],
                    "operator_scope": "primary_center_operator_only",
                    "scope_scores": scope_scores,
                }
            )
            tracked = _optical_flow_track(
                frame_paths,
                frame_idx,
                box,
                width,
                height,
                run_start,
                run_end,
            )
            for tracked_idx, tracked_entry in tracked.items():
                if not run_start <= tracked_idx < run_end:
                    continue
                if not _hand_detection_is_weak(
                    hand_records,
                    tracked_idx,
                    label,
                    cfg.api_hand_low_conf_threshold,
                ):
                    continue
                tracked_box = _clamp_box(tracked_entry.get("bbox", []), width, height)
                if tracked_box is None:
                    continue
                distance = abs(tracked_idx - frame_idx)
                confidence = (
                    0.95
                    if tracked_idx == frame_idx
                    else max(
                        0.25,
                        min(0.80, float(tracked_entry.get("confidence", 0.0) or 0.0))
                        * (0.985 ** distance),
                    )
                )
                candidate = {
                    "bbox": tracked_box,
                    "confidence": confidence,
                    "source": (
                        "ark_hand_grounding"
                        if tracked_idx == frame_idx
                        else "ark_hand_grounding+optical_flow"
                    ),
                    "bbox_extent": (
                        "visible_partial"
                        if detection.get("visibility") == "partial"
                        else (
                            "observed_full"
                            if tracked_idx == frame_idx
                            else "tracked_estimated"
                        )
                    ),
                    "primary_operator": True,
                    "primary_operator_policy": PRIMARY_OPERATOR_POLICY,
                    "identity_validation": "ark_temporal_context_validated",
                }
                previous = proposals[label].get(tracked_idx)
                if previous is None or confidence > float(previous.get("confidence", 0.0) or 0.0):
                    proposals[label][tracked_idx] = candidate

    filled_frames = 0
    for label, by_frame in proposals.items():
        for frame_idx, proposal in by_frame.items():
            if _hand_detection_is_weak(
                hand_records,
                frame_idx,
                label,
                cfg.api_hand_low_conf_threshold,
            ):
                hand_records[frame_idx][label] = proposal
                filled_frames += 1
    post_ark_identity_pruning = _prune_hand_records_outside_primary_instance(
        hand_records, width, height
    )
    _fill_short_hand_gaps(hand_records, int(cfg.hand_gap_fill_frames))
    return {
        "enabled": True,
        "triggered": bool(trigger_runs),
        "trigger_runs": trigger_runs,
        "scheduled_requests": schedule,
        "accepted_anchors": accepted_anchors,
        "rejected_anchors": rejected_anchors,
        "confirmed_not_visible_intervals": confirmed_not_visible_intervals,
        "confirmed_not_visible_frames": confirmed_not_visible_frames,
        "filled_frames": filled_frames,
        "post_ark_identity_pruning": post_ark_identity_pruning,
        "api_usage": dict(grounder.usage),
        "calls_by_step": dict(grounder.calls_by_step),
        "errors": errors,
    }


def _sam3_object_quality(
    specs: Sequence[Dict[str, Any]],
    raw_tracks: Dict[str, Dict[int, Dict[str, Any]]],
    cfg: BBoxConfig,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    weak_specs: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {}
    for spec in specs:
        track_id = str(spec["track_id"])
        active_frames = {
            frame_idx
            for start, end in spec.get("active_intervals", [])
            for frame_idx in range(int(start), int(end))
        }
        raw = raw_tracks.get(track_id, {})
        active_entries = [
            entry
            for frame_idx, entry in raw.items()
            if frame_idx in active_frames and entry.get("bbox")
        ]
        coverage = len(active_entries) / max(1, len(active_frames))
        mean_confidence = (
            sum(float(entry.get("confidence", 0.0) or 0.0) for entry in active_entries)
            / len(active_entries)
            if active_entries
            else 0.0
        )
        reasons: List[str] = []
        if not raw:
            reasons.append("no_sam3_track")
        if coverage < cfg.api_object_min_track_coverage:
            reasons.append("low_active_frame_coverage")
        if mean_confidence < cfg.api_object_min_mean_confidence:
            reasons.append("low_mean_confidence")
        report[track_id] = {
            "label": str(spec["label"]),
            "active_frames": len(active_frames),
            "observed_active_frames": len(active_entries),
            "active_frame_coverage": round(coverage, 6),
            "mean_confidence": round(mean_confidence, 6),
            "ark_assist_reasons": reasons,
        }
        if reasons:
            weak_specs.append(spec)
    return weak_specs, report


def _validate_api_object_anchors(
    specs: Sequence[Dict[str, Any]],
    api_anchors: Dict[str, Dict[str, Any]],
    sam3_raw_tracks: Dict[str, Dict[int, Dict[str, Any]]],
    hand_records: Sequence[Dict[str, Any]],
    steps_by_id: Dict[int, Dict[str, Any]],
    fps: float,
    total_frames: int,
    width: int,
    height: int,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    specs_by_id = {str(spec["track_id"]): spec for spec in specs}
    step_bounds = {
        step_id: _step_bounds(step, fps, total_frames)
        for step_id, step in steps_by_id.items()
    }
    accepted: Dict[str, Dict[str, Any]] = {}
    report: Dict[str, Any] = {}
    for track_id, anchor in api_anchors.items():
        spec = specs_by_id.get(track_id)
        frame_idx = int(anchor.get("frame_index_0based", -1))
        api_box = _clamp_box(anchor.get("bbox_xyxy_pixel", []), width, height)
        sam3_entry = sam3_raw_tracks.get(track_id, {}).get(frame_idx) or {}
        sam3_box = sam3_entry.get("bbox")
        hand_boxes = (
            _relevant_hand_boxes_for_spec(
                spec, frame_idx, hand_records, step_bounds
            )
            if spec is not None
            else []
        )
        iou = _box_iou(api_box, sam3_box)
        api_interaction = _hand_object_interaction_score(
            api_box, hand_boxes, width, height
        )
        api_hand_iou = max(
            (_box_iou(api_box, hand_box) for hand_box in hand_boxes), default=0.0
        )
        sam3_interaction = _hand_object_interaction_score(
            sam3_box, hand_boxes, width, height
        )
        api_center_score = (
            _point_proximity_score(
                _box_center(api_box),
                (float(width) / 2.0, float(height) / 2.0),
                width,
                height,
                0.50,
            )
            if api_box is not None
            else 0.0
        )
        api_image_area_ratio = _box_area(api_box) / max(1.0, float(width * height))
        area_ratio = (
            _box_area(api_box) / max(1.0, _box_area(sam3_box))
            if sam3_box
            else None
        )
        rejection_reasons: List[str] = []
        if spec is None:
            rejection_reasons.append("unknown_track_id")
        elif not 0 <= frame_idx < total_frames or not _is_active_for_spec(spec, frame_idx):
            rejection_reasons.append("anchor_frame_outside_active_interval")
        if api_box is None:
            rejection_reasons.append("invalid_api_bbox")
        elif api_image_area_ratio < 0.00005 or api_image_area_ratio > 0.85:
            rejection_reasons.append("implausible_api_bbox_area")
        elif api_hand_iou > 0.80:
            rejection_reasons.append("api_bbox_duplicates_primary_hand_bbox")

        if api_box is not None and not rejection_reasons:
            if sam3_box:
                if (
                    iou < 0.05
                    and api_interaction < 0.20
                    and (
                        sam3_interaction > api_interaction
                        or area_ratio is None
                        or area_ratio < 0.25
                        or area_ratio > 4.0
                    )
                ):
                    rejection_reasons.append(
                        "inconsistent_with_sam3_and_primary_hand_evidence"
                    )
            elif hand_boxes:
                if api_interaction < 0.05:
                    rejection_reasons.append("no_primary_hand_interaction_evidence")
            elif api_center_score < 0.60:
                rejection_reasons.append(
                    "no_local_evidence_and_bbox_outside_primary_workspace"
                )

        reject = bool(rejection_reasons)
        report[track_id] = {
            "accepted": not reject,
            "frame_index_0based": frame_idx,
            "api_vs_sam3_iou": round(iou, 6),
            "api_hand_interaction": round(api_interaction, 6),
            "api_hand_max_iou": round(api_hand_iou, 6),
            "sam3_hand_interaction": round(sam3_interaction, 6),
            "primary_hand_boxes_available": len(hand_boxes),
            "api_center_score": round(api_center_score, 6),
            "api_image_area_ratio": round(api_image_area_ratio, 8),
            "api_vs_sam3_area_ratio": round(area_ratio, 6) if area_ratio is not None else None,
            "reason": (
                "accepted_api_anchor_with_local_evidence"
                if not reject
                else ";".join(rejection_reasons)
            ),
        }
        if not reject:
            accepted[track_id] = {**anchor, "bbox_xyxy_pixel": api_box}
    return accepted, report


def _merge_grounding_meta(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    usage_keys = {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "api_calls",
        "api_attempts",
        "cache_hits",
    }
    usage = {
        key: int((left.get("api_usage") or {}).get(key, 0) or 0)
        + int((right.get("api_usage") or {}).get(key, 0) or 0)
        for key in usage_keys
    }
    calls: Counter[int] = Counter()
    for source in (left, right):
        for step_id, count in (source.get("api_calls_by_step") or {}).items():
            calls[int(step_id)] += int(count)
    errors = [
        str(value)
        for value in (left.get("local_grounding_error"), right.get("local_grounding_error"))
        if value
    ]
    return {
        "local_grounding_error": "; ".join(errors),
        "api_usage": usage,
        "api_calls_by_step": dict(calls),
    }


def _best_matching_detection(
    label: str,
    detections: Sequence[Dict[str, Any]],
    hand_boxes: Sequence[Sequence[float]],
    width: int,
    height: int,
) -> Optional[Dict[str, Any]]:
    key = _label_key(label)
    tokens = set(re.findall(r"[a-z0-9]+", key))
    best: Optional[Dict[str, Any]] = None
    best_rank = (-1.0, -1.0, -1.0)
    for detection in detections:
        det_key = _label_key(detection.get("label"))
        det_tokens = set(re.findall(r"[a-z0-9]+", det_key))
        overlap = len(tokens & det_tokens) / float(max(1, len(tokens | det_tokens)))
        exact = 1.0 if det_key == key else 0.0
        score = float(detection.get("score", 0.0) or 0.0)
        interaction = _hand_object_interaction_score(
            detection.get("bbox"), hand_boxes, width, height
        )
        rank = (max(exact, overlap), interaction, score)
        if rank > best_rank:
            best_rank = rank
            best = {**detection, "interaction_score": interaction}
    if best is None or best_rank[0] < 0.20:
        return None
    return best


def _select_object_anchors(
    specs: List[Dict[str, Any]],
    steps_by_id: Dict[int, Dict[str, Any]],
    frame_paths: Sequence[Path],
    fps: float,
    width: int,
    height: int,
    hand_records: Sequence[Dict[str, Any]],
    cfg: BBoxConfig,
    device: str,
    api_cfg: "ApiConfig",
    stage5_dir: Path,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    step_bounds = {
        step_id: _step_bounds(step, fps, len(frame_paths))
        for step_id, step in steps_by_id.items()
    }

    def relevant_hand_boxes(spec: Dict[str, Any], frame_idx: int) -> List[Sequence[float]]:
        active_step_ids = {
            step_id
            for step_id, (start, end_exclusive) in step_bounds.items()
            if start <= frame_idx < end_exclusive
        }
        relations = [
            relation
            for relation in spec.get("hand_relations", [])
            if int(relation.get("step_id", -1)) in active_step_ids
        ]
        if not relations:
            relations = list(spec.get("hand_relations", []))
        labels: set[str] = set()
        for relation in relations:
            hand = _label_key(relation.get("hand"))
            if hand in {"left hand", "left_hand", "both hands", "both_hands"}:
                labels.add("left_hand")
            if hand in {"right hand", "right_hand", "both hands", "both_hands"}:
                labels.add("right_hand")
        if relations and not labels:
            labels.update(HAND_LABELS)
        if not 0 <= frame_idx < len(hand_records):
            return []
        return [
            hand_records[frame_idx][hand_label]["bbox"]
            for hand_label in sorted(labels)
            if hand_records[frame_idx].get(hand_label, {}).get("bbox")
        ]

    def ark_target_context(
        spec: Dict[str, Any], step_id: int, frame_idx: int
    ) -> Dict[str, Any]:
        relations = [
            {
                "hand": relation.get("hand"),
                "relation": relation.get("relation"),
            }
            for relation in spec.get("hand_relations", [])
            if int(relation.get("step_id", -1)) == int(step_id)
        ]
        hand_boxes: Dict[str, List[int]] = {}
        if 0 <= frame_idx < len(hand_records):
            for hand_label in HAND_LABELS:
                box = hand_records[frame_idx].get(hand_label, {}).get("bbox")
                if box:
                    normalized = _pixel_to_official(box, width, height)
                    if normalized is not None:
                        hand_boxes[hand_label] = normalized
        context: Dict[str, Any] = {
            "step_goal": _normalize_label(steps_by_id.get(step_id, {}).get("step_goal")),
            "hand_object_relations": relations,
        }
        if hand_boxes:
            context["hand_bbox_normalized_0_999"] = hand_boxes
        return context

    local_grounder: Optional[LocalGrounder] = None
    local_error = ""
    if not cfg.disable_local_grounding:
        try:
            logger.info(f"[stage5] loading local grounding model={cfg.local_grounding_model} device={device}")
            local_grounder = LocalGrounder(
                cfg.local_grounding_model,
                device,
                cfg.local_box_threshold,
                cfg.local_text_threshold,
            )
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
            logger.warning(f"[stage5] local grounding unavailable; Ark fallback remains enabled: {local_error}")

    ark_grounder: Optional[ArkGrounder] = None
    if cfg.api_grounding and cfg.api_max_calls_per_step > 0:
        ark_grounder = ArkGrounder(
            api_cfg,
            stage5_dir / "grounding_cache",
            cfg.api_max_calls_per_step,
            cfg.api_max_total_calls,
            cfg.api_grounding_thinking,
            cfg.api_grounding_reasoning_effort,
        )

    anchors: Dict[str, Dict[str, Any]] = {}
    local_cache: Dict[Tuple[int, str], Optional[Dict[str, Any]]] = {}
    for spec in specs:
        label = str(spec["label"])
        best: Optional[Dict[str, Any]] = None
        if local_grounder is not None:
            for frame_idx in spec["candidate_frames"]:
                if not 0 <= frame_idx < len(frame_paths):
                    continue
                cache_key = (frame_idx, _label_key(label))
                if cache_key not in local_cache:
                    try:
                        detections = local_grounder.detect(frame_paths[frame_idx], [label])
                        local_cache[cache_key] = _best_matching_detection(
                            label,
                            detections,
                            relevant_hand_boxes(spec, frame_idx),
                            width,
                            height,
                        )
                    except Exception as exc:
                        logger.warning(
                            f"[stage5] local grounding failed frame={frame_idx} label={label!r}: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        local_cache[cache_key] = None
                detection = local_cache[cache_key]
                if detection is None:
                    continue
                box = _clamp_box(detection.get("bbox", []), width, height)
                if box is None:
                    continue
                candidate = {
                    "frame_index_0based": int(frame_idx),
                    "bbox_xyxy_pixel": box,
                    "confidence": float(detection.get("score", 0.0) or 0.0),
                    "interaction_score": float(detection.get("interaction_score", 0.0) or 0.0),
                    "source": "local_grounding_dino",
                    "visibility": "full",
                }
                if best is None or (
                    candidate["interaction_score"], candidate["confidence"]
                ) > (best["interaction_score"], best["confidence"]):
                    best = candidate
                if (
                    candidate["confidence"] >= 0.75
                    and candidate["interaction_score"] >= 0.80
                ):
                    break
        has_named_hand = any(
            _label_key(relation.get("hand"))
            in {"left hand", "left_hand", "right hand", "right_hand", "both hands", "both_hands"}
            for relation in spec.get("hand_relations", [])
        )
        local_is_reliable = bool(
            best is not None
            and best["confidence"] >= cfg.local_box_threshold
            and (
                best["interaction_score"] >= 0.35
                if has_named_hand
                else best["confidence"] >= max(0.35, cfg.local_box_threshold)
            )
        )
        if local_is_reliable and best is not None:
            anchors[spec["track_id"]] = best

    unresolved = {spec["track_id"] for spec in specs if spec["track_id"] not in anchors}
    if unresolved and ark_grounder is not None:
        for step_id in sorted(steps_by_id):
            active_specs = [
                spec for spec in specs
                if spec["track_id"] in unresolved and int(step_id) in spec["active_step_ids"]
            ]
            if not active_specs:
                continue
            step_candidates = _candidate_frames(steps_by_id[step_id], fps, len(frame_paths))
            for frame_idx in step_candidates[: max(1, cfg.api_max_calls_per_step)]:
                pending_specs = [
                    spec for spec in active_specs if spec["track_id"] in unresolved
                ]
                labels = [str(spec["label"]) for spec in pending_specs]
                contexts = [
                    ark_target_context(spec, step_id, frame_idx) for spec in pending_specs
                ]
                if not labels or not 0 <= frame_idx < len(frame_paths):
                    continue
                try:
                    detections = ark_grounder.ground(
                        frame_paths[frame_idx], labels, contexts, step_id, width, height
                    )
                except Exception as exc:
                    logger.warning(
                        f"[stage5] Ark grounding failed step={step_id} frame={frame_idx}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue
                for spec in pending_specs:
                    detection = detections.get(_label_key(spec["label"]))
                    if not detection:
                        continue
                    anchors[spec["track_id"]] = {
                        "frame_index_0based": int(frame_idx),
                        "bbox_xyxy_pixel": list(detection["bbox"]),
                        "confidence": 1.0,
                        "source": "ark_grounding",
                        "visibility": detection.get("visibility", "full"),
                    }
                    unresolved.discard(spec["track_id"])
                if not any(spec["track_id"] in unresolved for spec in active_specs):
                    break

    anchor_dir = stage5_dir / "anchors"
    anchor_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        anchor = anchors.get(spec["track_id"])
        if anchor is None:
            continue
        frame_idx = int(anchor["frame_index_0based"])
        image_name = f"{spec['track_id']}_frame_{frame_idx:06d}.jpg"
        shutil.copyfile(frame_paths[frame_idx], anchor_dir / image_name)
        anchor["image_relpath"] = str(Path("anchors") / image_name)
        anchor["bbox_xyxy_normalized_0_999"] = _pixel_to_official(
            anchor["bbox_xyxy_pixel"], width, height
        )

    usage = ark_grounder.usage if ark_grounder is not None else {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "api_calls": 0,
        "api_attempts": 0,
        "cache_hits": 0,
    }
    calls_by_step = dict(ark_grounder.calls_by_step) if ark_grounder is not None else {}
    return anchors, {
        "local_grounding_error": local_error,
        "api_usage": usage,
        "api_calls_by_step": calls_by_step,
    }


def _mask_result(mask_logits: Any, width: int, height: int) -> Optional[Dict[str, Any]]:
    import numpy as np

    if hasattr(mask_logits, "detach"):
        raw = mask_logits.detach().float().cpu().numpy()
    else:
        raw = np.asarray(mask_logits)
    while raw.ndim > 2:
        raw = raw[0]
    mask = raw > 0.0
    ys, xs = np.where(mask)
    if xs.size < 16 or ys.size < 16:
        return None
    box = _clamp_box([xs.min(), ys.min(), xs.max(), ys.max()], width, height)
    if box is None:
        return None
    positive = raw[mask]
    mean_logit = float(positive.mean()) if positive.size else 0.0
    confidence = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, mean_logit))))
    return {"bbox": box, "confidence": confidence, "source": "sam2_mask"}


class Sam2ObjectTracker:
    def __init__(self, model_id: str, device: str):
        import torch
        from sam2.sam2_video_predictor import SAM2VideoPredictor

        self.torch = torch
        self.device = device
        self.predictor = SAM2VideoPredictor.from_pretrained(model_id, device=device)

    def track(
        self,
        frame_dir: Path,
        anchor_idx: int,
        anchor_box: Sequence[int],
        total_frames: int,
        width: int,
        height: int,
    ) -> Dict[int, Dict[str, Any]]:
        import numpy as np

        state = self.predictor.init_state(
            video_path=str(frame_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=self.device == "cpu",
        )
        results: Dict[int, Dict[str, Any]] = {}
        with self.torch.inference_mode():
            _, object_ids, logits = self.predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=int(anchor_idx),
                obj_id=1,
                box=np.asarray(anchor_box, dtype=np.float32),
            )
            if len(object_ids):
                anchor_result = _mask_result(logits[0], width, height)
                if anchor_result:
                    results[int(anchor_idx)] = anchor_result
            for reverse in (False, True):
                iterator = self.predictor.propagate_in_video(
                    state,
                    start_frame_idx=int(anchor_idx),
                    max_frame_num_to_track=total_frames,
                    reverse=reverse,
                )
                for frame_idx, object_ids, logits in iterator:
                    if not len(object_ids):
                        continue
                    result = _mask_result(logits[0], width, height)
                    if result is not None:
                        results[int(frame_idx)] = result
        try:
            self.predictor.reset_state(state)
        except Exception:
            pass
        return results


def _optical_flow_direction(
    frame_paths: Sequence[Path],
    anchor_idx: int,
    anchor_box: Sequence[int],
    direction: int,
    width: int,
    height: int,
    start_frame: int = 0,
    end_frame_exclusive: Optional[int] = None,
) -> Dict[int, Dict[str, Any]]:
    import cv2
    import numpy as np

    results: Dict[int, Dict[str, Any]] = {}
    prev_idx = int(anchor_idx)
    prev_box = np.asarray(anchor_box, dtype=np.float32)
    prev_gray = cv2.imread(str(frame_paths[prev_idx]), cv2.IMREAD_GRAYSCALE)
    if prev_gray is None:
        return results

    def features(gray: Any, box: Any) -> Any:
        mask = np.zeros_like(gray)
        x1, y1, x2, y2 = [int(v) for v in box]
        mask[max(0, y1): min(height, y2 + 1), max(0, x1): min(width, x2 + 1)] = 255
        return cv2.goodFeaturesToTrack(gray, maxCorners=120, qualityLevel=0.01, minDistance=5, mask=mask)

    prev_points = features(prev_gray, prev_box)
    lower = max(0, int(start_frame))
    upper = min(
        len(frame_paths),
        int(end_frame_exclusive) if end_frame_exclusive is not None else len(frame_paths),
    )
    next_idx = prev_idx + direction
    while lower <= next_idx < upper:
        gray = cv2.imread(str(frame_paths[next_idx]), cv2.IMREAD_GRAYSCALE)
        if gray is None or prev_points is None or len(prev_points) < 4:
            break
        next_points, status, _error = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            gray,
            prev_points,
            None,
            winSize=(31, 31),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if next_points is None or status is None:
            break
        good_prev = prev_points[status.reshape(-1) == 1].reshape(-1, 2)
        good_next = next_points[status.reshape(-1) == 1].reshape(-1, 2)
        if len(good_prev) < 4:
            break
        matrix, inliers = cv2.estimateAffinePartial2D(good_prev, good_next, method=cv2.RANSAC)
        if matrix is None:
            break
        corners = np.asarray(
            [[prev_box[0], prev_box[1], 1.0], [prev_box[2], prev_box[3], 1.0]],
            dtype=np.float32,
        )
        transformed = (matrix @ corners.T).T
        box = _clamp_box(
            [transformed[0, 0], transformed[0, 1], transformed[1, 0], transformed[1, 1]],
            width,
            height,
        )
        if box is None:
            break
        inlier_ratio = float(inliers.sum() / max(1, len(inliers))) if inliers is not None else 0.5
        results[next_idx] = {
            "bbox": box,
            "confidence": max(0.05, min(0.85, inlier_ratio)),
            "source": "optical_flow_fallback",
        }
        prev_idx = next_idx
        prev_box = np.asarray(box, dtype=np.float32)
        prev_gray = gray
        prev_points = features(prev_gray, prev_box)
        next_idx += direction
    return results


def _optical_flow_track(
    frame_paths: Sequence[Path],
    anchor_idx: int,
    anchor_box: Sequence[int],
    width: int,
    height: int,
    start_frame: int = 0,
    end_frame_exclusive: Optional[int] = None,
) -> Dict[int, Dict[str, Any]]:
    results = {
        int(anchor_idx): {
            "bbox": list(anchor_box),
            "confidence": 1.0,
            "source": "grounding_anchor",
        }
    }
    results.update(
        _optical_flow_direction(
            frame_paths,
            anchor_idx,
            anchor_box,
            1,
            width,
            height,
            start_frame,
            end_frame_exclusive,
        )
    )
    results.update(
        _optical_flow_direction(
            frame_paths,
            anchor_idx,
            anchor_box,
            -1,
            width,
            height,
            start_frame,
            end_frame_exclusive,
        )
    )
    return results


def _augment_track_with_optical_flow(
    raw_track: Dict[int, Dict[str, Any]],
    frame_paths: Sequence[Path],
    active_intervals: Sequence[Sequence[int]],
    anchor: Dict[str, Any],
    width: int,
    height: int,
    min_confidence: float,
    source_prefix: str,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    augmented = dict(raw_track)
    intervals = _expanded_intervals(active_intervals, len(frame_paths), 0)
    seeds: List[Dict[str, Any]] = []
    filled = 0
    replaced_low_confidence = 0
    anchor_idx = int(anchor.get("frame_index_0based", -1))
    anchor_box = _clamp_box(anchor.get("bbox_xyxy_pixel", []), width, height)

    for start, end in intervals:
        candidates: List[Tuple[float, int, Sequence[int], str]] = []
        if anchor_box is not None and start <= anchor_idx < end:
            candidates.append((1.25, anchor_idx, anchor_box, "grounding_anchor"))
        for frame_idx, entry in raw_track.items():
            box = _clamp_box(entry.get("bbox", []), width, height)
            if box is None or not start <= int(frame_idx) < end:
                continue
            candidates.append(
                (
                    float(entry.get("confidence", 0.0) or 0.0),
                    int(frame_idx),
                    box,
                    str(entry.get("source") or "tracker"),
                )
            )
        if not candidates:
            continue
        seed_confidence, seed_idx, seed_box, seed_source = max(
            candidates, key=lambda item: (item[0], -abs(item[1] - (start + end) / 2.0))
        )
        seeds.append(
            {
                "interval": [int(start), int(end)],
                "frame_index_0based": seed_idx,
                "source": seed_source,
                "confidence": round(float(seed_confidence), 6),
            }
        )
        tracked = _optical_flow_track(
            frame_paths,
            seed_idx,
            seed_box,
            width,
            height,
            int(start),
            int(end),
        )
        for frame_idx, tracked_entry in tracked.items():
            if not start <= frame_idx < end:
                continue
            existing = augmented.get(frame_idx) or {}
            existing_has_box = bool(existing.get("bbox"))
            existing_confidence = float(existing.get("confidence", 0.0) or 0.0)
            if existing_has_box and existing_confidence >= min_confidence:
                continue
            tracked_confidence = float(tracked_entry.get("confidence", 0.0) or 0.0)
            if existing_has_box and tracked_confidence <= existing_confidence:
                continue
            augmented[frame_idx] = {
                **tracked_entry,
                "source": f"{source_prefix}+optical_flow",
            }
            if existing_has_box:
                replaced_low_confidence += 1
            else:
                filled += 1
    return augmented, {
        "active_intervals": intervals,
        "seeds": seeds,
        "filled_missing_frames": filled,
        "replaced_low_confidence_frames": replaced_low_confidence,
    }


def _median(values: Iterable[float], fallback: float) -> float:
    ordered = sorted(float(v) for v in values if math.isfinite(float(v)) and float(v) > 0)
    if not ordered:
        return float(fallback)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _amodalize_track(
    raw: Dict[int, Dict[str, Any]],
    anchor_box: Sequence[int],
    total_frames: int,
    width: int,
    height: int,
    active_intervals: Sequence[Sequence[int]],
) -> List[Dict[str, Any]]:
    anchor_width = max(1.0, float(anchor_box[2]) - float(anchor_box[0]))
    anchor_height = max(1.0, float(anchor_box[3]) - float(anchor_box[1]))
    plausible_boxes: List[Sequence[float]] = [anchor_box]
    for entry in raw.values():
        box = entry.get("bbox")
        if not box:
            continue
        box_width = max(1.0, float(box[2]) - float(box[0]))
        box_height = max(1.0, float(box[3]) - float(box[1]))
        width_ratio = box_width / anchor_width
        height_ratio = box_height / anchor_height
        if 0.5 <= width_ratio <= 2.0 and 0.5 <= height_ratio <= 2.0:
            plausible_boxes.append(box)
    raw_widths = [float(box[2]) - float(box[0]) for box in plausible_boxes]
    raw_heights = [float(box[3]) - float(box[1]) for box in plausible_boxes]
    ref_width = _median(raw_widths, anchor_width)
    ref_height = _median(raw_heights, anchor_height)
    observed_frames = sorted(raw)
    first_observed = observed_frames[0] if observed_frames else None
    last_observed = observed_frames[-1] if observed_frames else None

    def is_active(frame_idx: int) -> bool:
        return any(int(start) <= frame_idx < int(end) for start, end in active_intervals)

    def missing_record(frame_idx: int) -> Dict[str, Any]:
        if first_observed is not None and first_observed < frame_idx < int(last_observed):
            visibility = "occluded"
        elif is_active(frame_idx):
            visibility = "occluded"
        else:
            visibility = "not_present"
        return {
            "visibility": visibility,
            "bbox": None,
            "bbox_extent": None,
            "source": "none",
            "confidence": 0.0,
        }

    output: List[Dict[str, Any]] = []
    for frame_idx in range(total_frames):
        entry = raw.get(frame_idx)
        if not entry or not entry.get("bbox"):
            output.append(missing_record(frame_idx))
            continue

        box = list(entry["bbox"])
        box_width = max(1.0, box[2] - box[0])
        box_height = max(1.0, box[3] - box[1])
        width_ratio = box_width / max(1.0, ref_width)
        height_ratio = box_height / max(1.0, ref_height)
        area_ratio = width_ratio * height_ratio
        if (
            width_ratio < 0.2
            or width_ratio > 5.0
            or height_ratio < 0.2
            or height_ratio > 5.0
            or area_ratio < 0.04
            or area_ratio > 25.0
        ):
            output.append(missing_record(frame_idx))
            continue
        touches_border = box[0] <= 1 or box[1] <= 1 or box[2] >= width - 2 or box[3] >= height - 2
        size_ratio = (box_width * box_height) / max(1.0, ref_width * ref_height)
        partial = touches_border or size_ratio < 0.65
        source = str(entry.get("source") or "tracker")
        if partial:
            center_x = (box[0] + box[2]) / 2.0
            center_y = (box[1] + box[3]) / 2.0
            estimated = _clamp_box(
                [
                    center_x - ref_width / 2.0,
                    center_y - ref_height / 2.0,
                    center_x + ref_width / 2.0,
                    center_y + ref_height / 2.0,
                ],
                width,
                height,
            )
            if estimated is not None:
                box = estimated
            bbox_extent = "amodal_estimated"
            visibility = "partial"
        else:
            bbox_extent = (
                "tracked_estimated" if "optical_flow" in source else "observed_full"
            )
            visibility = "estimated" if bbox_extent == "tracked_estimated" else "visible"
        output.append(
            {
                "visibility": visibility,
                "bbox": box,
                "bbox_extent": bbox_extent,
                "source": source,
                "confidence": float(entry.get("confidence", 0.0) or 0.0),
            }
        )
    return output


def _status_for_missing_hand(records: Sequence[Dict[str, Any]], frame_idx: int, label: str) -> str:
    previous = any(records[i].get(label, {}).get("bbox") for i in range(max(0, frame_idx - 15), frame_idx))
    following = any(
        records[i].get(label, {}).get("bbox")
        for i in range(frame_idx + 1, min(len(records), frame_idx + 16))
    )
    return "occluded" if previous and following else "not_present"


def _entity_entry(
    *,
    track_id: str,
    entity_type: str,
    label: str,
    record: Dict[str, Any],
    width: int,
    height: int,
    active_step_ids: Sequence[int],
    active_now: bool,
) -> Dict[str, Any]:
    box = record.get("bbox")
    return {
        "track_id": track_id,
        "entity_type": entity_type,
        "label": label,
        "visibility": str(record.get("visibility") or ("visible" if box else "not_present")),
        "bbox_xyxy_pixel": list(box) if box is not None else None,
        "bbox_xyxy_normalized_0_999": _pixel_to_official(box, width, height) if box is not None else None,
        "bbox_extent": record.get("bbox_extent"),
        "source": str(record.get("source") or "none"),
        "confidence": round(float(record.get("confidence", 0.0) or 0.0), 6),
        "sam3_instance_id": record.get("sam3_instance_id"),
        "primary_hand_chain_id": record.get("primary_hand_chain_id"),
        "primary_instance_locked": bool(record.get("primary_instance_locked", False)),
        "identity_validation": record.get("identity_validation"),
        "not_visible_reason": record.get("not_visible_reason"),
        "active_step_ids": [int(v) for v in active_step_ids],
        "is_active_manipulation": bool(active_now),
    }


def _longest_missing_run(
    indexed_entities: Sequence[Tuple[int, Dict[str, Any]]]
) -> int:
    longest = current = 0
    previous_frame: Optional[int] = None
    for frame_idx, entity in indexed_entities:
        if previous_frame is None or frame_idx != previous_frame + 1:
            current = 0
        if entity.get("bbox_xyxy_pixel") is None:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
        previous_frame = frame_idx
    return longest


def _longest_unexplained_missing_run(
    indexed_entities: Sequence[Tuple[int, Dict[str, Any]]]
) -> int:
    longest = current = 0
    previous_frame: Optional[int] = None
    for frame_idx, entity in indexed_entities:
        if previous_frame is None or frame_idx != previous_frame + 1:
            current = 0
        confirmed_negative = bool(
            str(entity.get("source") or "none")
            == "ark_hand_visibility_confirmation"
            and str(entity.get("not_visible_reason") or "unknown")
            in ARK_HAND_CONFIRMED_NOT_VISIBLE_REASONS
        )
        unexplained = bool(
            entity.get("bbox_xyxy_pixel") is None and not confirmed_negative
        )
        if unexplained:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
        previous_frame = frame_idx
    return longest


def _track_quality_report(
    frame_records: Sequence[Dict[str, Any]],
    width: int,
    height: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    by_track: Dict[str, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    for frame_record in frame_records:
        frame_idx = int(frame_record["frame_index_0based"])
        for entity in frame_record.get("entities", []):
            by_track[str(entity.get("track_id"))].append((frame_idx, entity))

    diagonal = max(1.0, math.hypot(float(width), float(height)))
    report: Dict[str, Any] = {}
    warnings: List[Dict[str, Any]] = []
    for track_id, indexed in sorted(by_track.items()):
        active = [item for item in indexed if bool(item[1].get("is_active_manipulation"))]
        evaluated = active or indexed
        observed = [item for item in evaluated if item[1].get("bbox_xyxy_pixel") is not None]
        direct = [
            item
            for item in observed
            if item[1].get("bbox_extent") in {"observed_full", "visible_partial"}
            and "optical_flow" not in str(item[1].get("source") or "")
            and str(item[1].get("source") or "") != "temporal_interpolation"
        ]
        confidences = [
            float(entity.get("confidence", 0.0) or 0.0)
            for _frame_idx, entity in observed
        ]
        visibility_counts = Counter(
            str(entity.get("visibility") or "unknown") for _frame_idx, entity in evaluated
        )
        source_counts = Counter(
            str(entity.get("source") or "none") for _frame_idx, entity in evaluated
        )
        entity_type = str(indexed[0][1].get("entity_type") or "unknown")
        max_jump = 0.0
        suspicious_jumps = 0
        max_area_ratio = 1.0
        abrupt_area_changes = 0
        border_clipped_area_changes = 0
        zero_iou_large_jumps = 0
        temporal_identity_breaks = 0
        sam3_instance_switches = 0
        linked_sam3_fragment_transitions = 0
        transition_warning_sample: List[Dict[str, Any]] = []
        previous_observed: Optional[Tuple[int, Dict[str, Any]]] = None
        for item in observed:
            if previous_observed is not None and item[0] == previous_observed[0] + 1:
                previous_box = previous_observed[1].get("bbox_xyxy_pixel")
                current_box = item[1].get("bbox_xyxy_pixel")
                transition = _bbox_transition_metrics(
                    previous_box, current_box, width, height
                )
                jump = float(transition["center_jump_normalized"])
                max_jump = max(max_jump, jump)
                max_area_ratio = max(max_area_ratio, float(transition["area_ratio"]))
                previous_chain = previous_observed[1].get("primary_hand_chain_id")
                current_chain = item[1].get("primary_hand_chain_id")
                same_primary_chain = bool(
                    previous_chain and previous_chain == current_chain
                )
                border_clipped_transition = bool(
                    entity_type == "hand"
                    and same_primary_chain
                    and _box_touches_image_border(previous_box, width, height)
                    and _box_touches_image_border(current_box, width, height)
                    and jump <= HAND_ADJACENT_MAX_CENTER_JUMP
                )
                if jump > 0.25:
                    suspicious_jumps += 1
                if (
                    entity_type == "hand"
                    and float(transition["area_ratio"]) > HAND_QA_MAX_AREA_RATIO
                    and float(transition["iou"]) < 0.10
                    and not border_clipped_transition
                ):
                    abrupt_area_changes += 1
                if (
                    border_clipped_transition
                    and float(transition["area_ratio"]) > HAND_QA_MAX_AREA_RATIO
                ):
                    border_clipped_area_changes += 1
                if (
                    entity_type == "hand"
                    and jump > HAND_QA_ZERO_IOU_JUMP
                    and float(transition["iou"]) < 0.01
                    and not border_clipped_transition
                ):
                    zero_iou_large_jumps += 1
                identity_break = bool(
                    entity_type == "hand"
                    and (
                        (
                            jump > HAND_ADJACENT_MAX_CENTER_JUMP
                            and float(transition["iou"]) < 0.02
                        )
                        or (
                            jump > HAND_QA_ZERO_IOU_JUMP
                            and float(transition["iou"]) < 0.01
                            and not border_clipped_transition
                        )
                        or (
                            float(transition["area_ratio"]) > HAND_QA_MAX_AREA_RATIO
                            and float(transition["iou"]) < 0.10
                            and not border_clipped_transition
                        )
                    )
                )
                previous_instance = previous_observed[1].get("sam3_instance_id")
                current_instance = item[1].get("sam3_instance_id")
                raw_instance_change = bool(
                    entity_type == "hand"
                    and previous_instance
                    and current_instance
                    and previous_instance != current_instance
                    and _sam3_segment_key(str(previous_instance))
                    == _sam3_segment_key(str(current_instance))
                )
                linked_fragment_transition = bool(
                    raw_instance_change
                    and previous_chain
                    and previous_chain == current_chain
                )
                instance_switch = bool(raw_instance_change and not linked_fragment_transition)
                if identity_break:
                    temporal_identity_breaks += 1
                if instance_switch:
                    sam3_instance_switches += 1
                if linked_fragment_transition:
                    linked_sam3_fragment_transitions += 1
                if (identity_break or instance_switch) and len(transition_warning_sample) < 20:
                    transition_warning_sample.append(
                        {
                            "previous_frame_index_0based": previous_observed[0],
                            "frame_index_0based": item[0],
                            "previous_sam3_instance_id": previous_instance,
                            "sam3_instance_id": current_instance,
                            **{
                                key: round(float(value), 6)
                                for key, value in transition.items()
                            },
                        }
                    )
            previous_observed = item

        evaluation_frames = len(evaluated)
        bbox_coverage = len(observed) / max(1, evaluation_frames)
        direct_coverage = len(direct) / max(1, evaluation_frames)
        report[track_id] = {
            "entity_type": entity_type,
            "label": str(indexed[0][1].get("label") or ""),
            "evaluation_scope": "active_step_intervals" if active else "full_video_fallback",
            "total_video_frames": len(indexed),
            "active_frames": len(active),
            "evaluated_frames": evaluation_frames,
            "frames_with_bbox": len(observed),
            "bbox_coverage": round(bbox_coverage, 6),
            "direct_observation_coverage": round(direct_coverage, 6),
            "longest_consecutive_missing_frames": _longest_missing_run(evaluated),
            "longest_unexplained_missing_frames": _longest_unexplained_missing_run(
                evaluated
            ),
            "mean_confidence_when_box_present": (
                round(sum(confidences) / len(confidences), 6) if confidences else 0.0
            ),
            "max_consecutive_center_jump_normalized": round(max_jump, 6),
            "suspicious_center_jump_count": suspicious_jumps,
            "max_consecutive_area_ratio": round(max_area_ratio, 6),
            "abrupt_area_change_count": abrupt_area_changes,
            "border_clipped_area_change_count": border_clipped_area_changes,
            "zero_iou_large_jump_count": zero_iou_large_jumps,
            "temporal_identity_break_count": temporal_identity_breaks,
            "sam3_instance_switch_count": sam3_instance_switches,
            "linked_sam3_fragment_transition_count": linked_sam3_fragment_transitions,
            "transition_warning_sample": transition_warning_sample,
            "visibility_counts": dict(visibility_counts),
            "source_counts": dict(source_counts),
        }
        if suspicious_jumps:
            warnings.append(
                {
                    "track_id": track_id,
                    "reason": "consecutive_bbox_center_jump_exceeds_0.25_image_diagonal",
                    "count": suspicious_jumps,
                }
            )
        if entity_type == "hand" and temporal_identity_breaks:
            warnings.append(
                {
                    "track_id": track_id,
                    "reason": "hand_temporal_identity_discontinuity",
                    "count": temporal_identity_breaks,
                }
            )
        if entity_type == "hand" and sam3_instance_switches:
            warnings.append(
                {
                    "track_id": track_id,
                    "reason": "hand_sam3_primary_instance_switch",
                    "count": sam3_instance_switches,
                }
            )
        if entity_type == "hand" and abrupt_area_changes:
            warnings.append(
                {
                    "track_id": track_id,
                    "reason": "hand_bbox_area_ratio_exceeds_4.0",
                    "count": abrupt_area_changes,
                }
            )
        if entity_type == "object" and active and bbox_coverage < 0.60:
            warnings.append(
                {
                    "track_id": track_id,
                    "reason": "object_active_interval_bbox_coverage_below_0.60",
                    "coverage": round(bbox_coverage, 6),
                }
            )
    return report, warnings


def _write_overlay(
    video_path: str,
    output_path: Path,
    frame_records: Sequence[Dict[str, Any]],
    fps: float,
    width: int,
    height: int,
) -> None:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source video for overlay: {video_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (int(width), int(height))
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create overlay video: {output_path}")
    palette = {
        "left_hand": (40, 220, 40),
        "right_hand": (255, 170, 20),
    }
    try:
        frame_idx = 0
        while frame_idx < len(frame_records):
            ok, frame = cap.read()
            if not ok:
                break
            for entity in frame_records[frame_idx].get("entities", []):
                box = entity.get("bbox_xyxy_pixel")
                if not box:
                    continue
                label = str(entity.get("label") or "entity")
                color = palette.get(label, (30, 80 + (hash(label) % 150), 240))
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"{label} {entity.get('confidence', 0.0):.2f}"
                cv2.putText(
                    frame,
                    text,
                    (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            cv2.putText(
                frame,
                f"frame {frame_idx}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()


def _can_resume_stage5(
    summary_path: Path,
    jsonl_path: Path,
    video_path: str,
    total_frames: int,
    final_plan_sha256: str,
    bbox_cfg: BBoxConfig,
) -> bool:
    if not summary_path.exists() or not jsonl_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("schema_version") != STAGE5_SCHEMA_VERSION:
            return False
        source = summary.get("source_video") or {}
        if os.path.abspath(str(source.get("path") or "")) != os.path.abspath(video_path):
            return False
        source_stat = summary.get("source_video_stat") or {}
        current_stat = os.stat(video_path)
        if int(source_stat.get("size_bytes", -1)) != int(current_stat.st_size):
            return False
        if int(source_stat.get("mtime_ns", -1)) != int(current_stat.st_mtime_ns):
            return False
        if str(summary.get("final_plan_sha256") or "") != final_plan_sha256:
            return False
        if summary.get("bbox_config") != asdict(bbox_cfg):
            return False
        quality_gate = summary.get("quality_gate") or {}
        if (
            quality_gate.get("status") != "passed"
            or not quality_gate.get("eligible_for_automatic_use")
        ):
            return False
        if int(summary.get("total_frames", -1)) != int(total_frames):
            return False
        with jsonl_path.open("r", encoding="utf-8") as handle:
            count = sum(1 for line in handle if line.strip())
        return count == int(total_frames)
    except Exception:
        return False


def run_stage5_for_video(
    video_path: str,
    output_root: str,
    api_cfg: "ApiConfig",
    bbox_cfg: BBoxConfig,
    overwrite: bool = False,
) -> str:
    started = time.perf_counter()
    video_path = os.path.abspath(video_path)
    video_id = video_id_from_path(video_path)
    video_out = os.path.join(output_root, video_id)
    ensure_video_out_dir_safe(video_out, video_path)
    stage5_dir = Path(video_out) / "stage5"
    stage5_dir.mkdir(parents=True, exist_ok=True)
    final_plan_path = Path(video_out) / "final_plan.json"
    if not final_plan_path.exists():
        raise FileNotFoundError(f"Stage 5 requires Stage 4 final_plan.json: {final_plan_path}")

    metadata = get_video_metadata(video_path)
    fps = float(metadata["fps"])
    total_frames = int(metadata["total_frames"])
    width = int(metadata["width"])
    height = int(metadata["height"])
    frame_timestamps, timestamp_source = _frame_timestamps(
        video_path, total_frames, fps
    )
    frame_jsonl_path = stage5_dir / "frame_bboxes.jsonl"
    summary_path = stage5_dir / "stage5_summary.json"
    run_summary_path = os.path.join(video_out, "run_summary.json")
    final_plan_sha256 = hashlib.sha256(final_plan_path.read_bytes()).hexdigest()
    if not overwrite and _can_resume_stage5(
        summary_path,
        frame_jsonl_path,
        video_path,
        total_frames,
        final_plan_sha256,
        bbox_cfg,
    ):
        logger.info(f"[stage5] video_id={video_id} resume: valid per-frame bbox output already exists")
        return video_out

    update_run_summary(
        run_summary_path,
        {
            "source_video": build_source_video_meta(video_path),
            "output_root": os.path.abspath(output_root),
            "updated_at_utc": now_utc_iso(),
            "stage5": {"status": "running", "started_at_utc": now_utc_iso()},
        },
    )
    plan = read_json(str(final_plan_path))
    specs, steps_by_id = _collect_object_specs(plan, fps, total_frames)
    device = _resolve_device(bbox_cfg.device)
    hand_device = _resolve_hand_device(bbox_cfg.hand_device)
    logger.info(
        f"[stage5] video_id={video_id} start frames={total_frames} fps={fps:.6f} "
        f"objects={len(specs)} backend={bbox_cfg.backend} device={device} "
        f"hand_device={hand_device}"
    )

    with tempfile.TemporaryDirectory(prefix=f"{video_id}_stage5_") as temp_root:
        frame_dir = Path(temp_root) / "frames"
        frame_paths = _extract_frames(video_path, frame_dir, bbox_cfg.jpeg_quality)
        if len(frame_paths) != total_frames:
            raise RuntimeError(
                f"Decoded frame count mismatch: metadata={total_frames}, decoded={len(frame_paths)}"
            )
        # Ultralytics may set CUDA_VISIBLE_DEVICES=-1 for a CPU detector. Keep that
        # process-local choice from hiding CUDA from the later SAM3 subprocess.
        with _preserve_environment_variable("CUDA_VISIBLE_DEVICES"):
            hand_records, primary_operator_meta = _detect_hands(
                video_path, total_frames, width, height, bbox_cfg, hand_device
            )
        anchors: Dict[str, Dict[str, Any]] = {}
        sam3_raw_tracks: Dict[str, Dict[int, Dict[str, Any]]] = {}
        sam3_error = ""
        sam3_meta: Dict[str, Any] = {}
        sam3_payload: Optional[Dict[str, Any]] = None
        hand_fusion_meta: Dict[str, Any] = {}
        hand_assist_meta: Dict[str, Any] = {
            "enabled": bool(bbox_cfg.api_hand_assist),
            "triggered": False,
            "api_usage": _empty_grounding_meta()["api_usage"],
            "calls_by_step": {},
        }
        duplicate_hand_resolution_meta: Dict[str, Any] = {
            "iou_threshold": 0.75,
            "resolved_duplicate_frame_count": 0,
            "resolution_sample": [],
        }
        object_quality_meta: Dict[str, Any] = {}
        api_object_validation_meta: Dict[str, Any] = {}
        optical_flow_augmentation_meta: Dict[str, Any] = {}
        grounding_meta = _empty_grounding_meta()
        legacy_track_ids: set[str] = set()
        api_assisted_object_ids: set[str] = set()
        weak_sam3_track_ids: set[str] = set()

        if bbox_cfg.backend == "sam3":
            try:
                logger.info(
                    f"[stage5] loading SAM3 checkpoint={bbox_cfg.sam3_checkpoint} "
                    f"threshold={bbox_cfg.sam3_threshold:.3f}"
                )
                sam3_payload = _run_sam3_backend(
                    frame_dir, specs, total_frames, bbox_cfg, stage5_dir
                )
                if bbox_cfg.sam3_hand_fusion:
                    hand_fusion_meta = _fuse_sam3_hands(
                        hand_records,
                        sam3_payload,
                        bbox_cfg.hand_gap_fill_frames,
                        width,
                        height,
                    )
            except Exception as exc:
                sam3_error = f"{type(exc).__name__}: {exc}"
                logger.warning(f"[stage5] SAM3 backend failed: {sam3_error}")

        if bbox_cfg.api_hand_assist:
            try:
                hand_assist_meta = _assist_hands_with_ark(
                    specs,
                    steps_by_id,
                    frame_paths,
                    hand_records,
                    fps,
                    width,
                    height,
                    bbox_cfg,
                    api_cfg,
                    stage5_dir,
                )
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                logger.warning(f"[stage5] Ark hand assist unavailable: {message}")
                hand_assist_meta["errors"] = [message]

        duplicate_hand_resolution_meta = _resolve_duplicate_primary_hand_boxes(
            hand_records, specs, steps_by_id, fps
        )

        if sam3_payload is not None:
            try:
                sam3_raw_tracks, sam3_anchors, sam3_selection = _select_sam3_object_tracks(
                    specs,
                    steps_by_id,
                    sam3_payload,
                    hand_records,
                    fps,
                    total_frames,
                    width,
                    height,
                )
                anchors.update(sam3_anchors)
                sam3_meta = {
                    **(sam3_payload.get("sam3") or {}),
                    "threshold": sam3_payload.get("threshold"),
                    "chunk_size": sam3_payload.get("chunk_size"),
                    "resolved_object_track_ids": sorted(sam3_raw_tracks),
                    "selection": sam3_selection,
                    "hand_fusion": hand_fusion_meta,
                }
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}"
                sam3_error = "; ".join(value for value in (sam3_error, detail) if value)
                logger.warning(f"[stage5] SAM3 object selection failed: {detail}")

        if bbox_cfg.backend == "sam3":
            weak_object_specs, object_quality_meta = _sam3_object_quality(
                specs, sam3_raw_tracks, bbox_cfg
            )
            weak_sam3_track_ids = {
                str(spec["track_id"]) for spec in weak_object_specs
            }
            if bbox_cfg.api_grounding and weak_object_specs:
                logger.info(
                    f"[stage5] Ark object assist triggered for {len(weak_object_specs)} "
                    "unresolved/weak SAM3 tracks"
                )
                api_only_cfg = replace(
                    bbox_cfg,
                    disable_local_grounding=True,
                    api_grounding=True,
                )
                api_anchors, api_object_meta = _select_object_anchors(
                    weak_object_specs,
                    steps_by_id,
                    frame_paths,
                    fps,
                    width,
                    height,
                    hand_records,
                    api_only_cfg,
                    device,
                    api_cfg,
                    stage5_dir,
                )
                api_anchors, api_object_validation_meta = _validate_api_object_anchors(
                    weak_object_specs,
                    api_anchors,
                    sam3_raw_tracks,
                    hand_records,
                    steps_by_id,
                    fps,
                    total_frames,
                    width,
                    height,
                )
                anchors.update(api_anchors)
                api_assisted_object_ids.update(api_anchors)
                grounding_meta = _merge_grounding_meta(
                    grounding_meta, api_object_meta
                )

        unresolved_specs = [
            spec for spec in specs if str(spec["track_id"]) not in anchors
        ]
        use_legacy = bbox_cfg.backend == "legacy" or (
            bbox_cfg.backend == "sam3"
            and bbox_cfg.sam3_legacy_fallback
            and bool(unresolved_specs)
        )
        if use_legacy:
            legacy_specs = specs if bbox_cfg.backend == "legacy" else unresolved_specs
            legacy_anchors, legacy_meta = _select_object_anchors(
                legacy_specs,
                steps_by_id,
                frame_paths,
                fps,
                width,
                height,
                hand_records,
                bbox_cfg,
                device,
                api_cfg,
                stage5_dir,
            )
            anchors.update(legacy_anchors)
            legacy_track_ids = set(legacy_anchors)
            grounding_meta = _merge_grounding_meta(grounding_meta, legacy_meta)

        unresolved_after_grounding = [
            str(spec["label"]) for spec in specs if str(spec["track_id"]) not in anchors
        ]
        if unresolved_after_grounding and not bbox_cfg.allow_unresolved_objects:
            raise RuntimeError(
                "Stage 5 could not obtain a reliable anchor bbox for manipulated objects: "
                + ", ".join(unresolved_after_grounding)
            )
        _write_anchor_artifacts(
            specs, anchors, frame_paths, stage5_dir, width, height
        )

        sam2_tracker: Optional[Sam2ObjectTracker] = None
        sam2_error = ""
        if not bbox_cfg.disable_sam2 and legacy_track_ids:
            try:
                logger.info(f"[stage5] loading SAM2 model={bbox_cfg.sam2_model} device={device}")
                sam2_tracker = Sam2ObjectTracker(bbox_cfg.sam2_model, device)
            except Exception as exc:
                sam2_error = f"{type(exc).__name__}: {exc}"
                logger.warning(f"[stage5] SAM2 unavailable; optical-flow fallback will be used: {sam2_error}")

        object_tracks: Dict[str, List[Dict[str, Any]]] = {}
        tracker_by_id: Dict[str, str] = {}
        for spec in specs:
            track_id = str(spec["track_id"])
            anchor = anchors.get(track_id)
            if anchor is None:
                object_tracks[track_id] = _amodalize_track(
                    {}, [0, 0, 1, 1], total_frames, width, height, spec["active_intervals"]
                )
                tracker_by_id[track_id] = "unresolved"
                continue
            anchor_idx = int(anchor["frame_index_0based"])
            anchor_box = list(anchor["bbox_xyxy_pixel"])
            if track_id in sam3_raw_tracks:
                raw_track = dict(sam3_raw_tracks[track_id])
                raw_track[anchor_idx] = {
                    "bbox": anchor_box,
                    "confidence": float(anchor.get("confidence", 1.0)),
                    "source": str(anchor.get("source") or "sam3_text_video"),
                }
                if track_id in weak_sam3_track_ids:
                    source_prefix = (
                        "ark_grounding"
                        if track_id in api_assisted_object_ids
                        else "sam3_anchor"
                    )
                    raw_track, augmentation = _augment_track_with_optical_flow(
                        raw_track,
                        frame_paths,
                        spec.get("active_intervals", []),
                        anchor,
                        width,
                        height,
                        bbox_cfg.api_object_min_mean_confidence,
                        source_prefix,
                    )
                    optical_flow_augmentation_meta[track_id] = augmentation
                object_tracks[track_id] = _amodalize_track(
                    raw_track,
                    anchor_box,
                    total_frames,
                    width,
                    height,
                    spec["active_intervals"],
                )
                tracker_by_id[track_id] = (
                    "sam3_text_video+ark_assist+optical_flow"
                    if track_id in api_assisted_object_ids
                    else (
                        "sam3_text_video+optical_flow"
                        if track_id in weak_sam3_track_ids
                        else "sam3_text_video"
                    )
                )
                continue
            raw_track: Dict[int, Dict[str, Any]] = {}
            if sam2_tracker is not None and track_id in legacy_track_ids:
                try:
                    raw_track = sam2_tracker.track(
                        frame_dir, anchor_idx, anchor_box, total_frames, width, height
                    )
                    tracker_by_id[track_id] = "sam2"
                except Exception as exc:
                    logger.warning(
                        f"[stage5] SAM2 track failed track_id={track_id}; using optical flow: "
                        f"{type(exc).__name__}: {exc}"
                    )
            if not raw_track:
                source_prefix = (
                    "ark_grounding"
                    if track_id in api_assisted_object_ids
                    else "grounding_anchor"
                )
                raw_track, augmentation = _augment_track_with_optical_flow(
                    {},
                    frame_paths,
                    spec.get("active_intervals", []),
                    anchor,
                    width,
                    height,
                    0.0,
                    source_prefix,
                )
                optical_flow_augmentation_meta[track_id] = augmentation
                tracker_by_id[track_id] = (
                    "ark_grounding+optical_flow"
                    if track_id in api_assisted_object_ids
                    else "optical_flow"
                )
            raw_track[anchor_idx] = {
                "bbox": anchor_box,
                "confidence": float(anchor.get("confidence", 1.0)),
                "source": str(anchor.get("source") or "grounding_anchor"),
            }
            object_tracks[track_id] = _amodalize_track(
                raw_track,
                anchor_box,
                total_frames,
                width,
                height,
                spec["active_intervals"],
            )

        frame_records: List[Dict[str, Any]] = []
        object_by_id = {str(spec["track_id"]): spec for spec in specs}
        hand_steps = _collect_hand_steps(specs)

        step_bounds = {
            step_id: _step_bounds(step, fps, total_frames)
            for step_id, step in steps_by_id.items()
        }

        def active_at(step_ids: Sequence[int], frame_idx: int) -> bool:
            return any(
                step_id in step_bounds
                and step_bounds[step_id][0] <= frame_idx < step_bounds[step_id][1]
                for step_id in step_ids
            )

        for frame_idx in range(total_frames):
            entities: List[Dict[str, Any]] = []
            for label in HAND_LABELS:
                detection = hand_records[frame_idx].get(label)
                if detection:
                    bbox_extent = str(detection.get("bbox_extent") or "")
                    visibility = str(detection.get("visibility") or "")
                    if not visibility:
                        if bbox_extent == "visible_partial":
                            visibility = "partial"
                        elif bbox_extent == "amodal_estimated":
                            visibility = "occluded"
                        elif bbox_extent == "tracked_estimated":
                            visibility = "estimated"
                        else:
                            visibility = "visible"
                    record = {
                        **detection,
                        "visibility": visibility,
                    }
                else:
                    record = {
                        "bbox": None,
                        "confidence": 0.0,
                        "source": "none",
                        "bbox_extent": None,
                        "visibility": _status_for_missing_hand(hand_records, frame_idx, label),
                    }
                steps_for_hand = sorted(hand_steps[label])
                entities.append(
                    _entity_entry(
                        track_id=f"hand_{label.split('_')[0]}_001",
                        entity_type="hand",
                        label=label,
                        record=record,
                        width=width,
                        height=height,
                        active_step_ids=steps_for_hand,
                        active_now=active_at(steps_for_hand, frame_idx),
                    )
                )
            for track_id, track in object_tracks.items():
                spec = object_by_id[track_id]
                entities.append(
                    _entity_entry(
                        track_id=track_id,
                        entity_type="object",
                        label=str(spec["label"]),
                        record=track[frame_idx],
                        width=width,
                        height=height,
                        active_step_ids=spec["active_step_ids"],
                        active_now=active_at(spec["active_step_ids"], frame_idx),
                    )
                )
            frame_records.append(
                {
                    "frame_index_0based": frame_idx,
                    "timestamp_sec": round(frame_timestamps[frame_idx], 9),
                    "image_size": {"width": width, "height": height},
                    "entities": entities,
                }
            )

        temp_jsonl = stage5_dir / ".frame_bboxes.jsonl.tmp"
        with temp_jsonl.open("w", encoding="utf-8") as handle:
            for record in frame_records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(temp_jsonl, frame_jsonl_path)

        tracks_payload = {
            "schema_version": STAGE5_SCHEMA_VERSION,
            "bbox_backend": bbox_cfg.backend,
            "primary_operator": primary_operator_meta,
            "sam3": sam3_meta,
            "coordinate_systems": {
                "pixel": {"format": "xyxy", "origin": "top_left", "image_size": [width, height]},
                "normalized": {"format": "xyxy", "origin": "top_left", "range": [0, 999]},
            },
            "tracks": [
                {
                    "track_id": "hand_left_001",
                    "entity_type": "hand",
                    "label": "left_hand",
                    "operator_scope": "primary_center_operator_only",
                    "active_step_ids": sorted(hand_steps["left_hand"]),
                    "ark_assist_enabled": bool(bbox_cfg.api_hand_assist),
                },
                {
                    "track_id": "hand_right_001",
                    "entity_type": "hand",
                    "label": "right_hand",
                    "operator_scope": "primary_center_operator_only",
                    "active_step_ids": sorted(hand_steps["right_hand"]),
                    "ark_assist_enabled": bool(bbox_cfg.api_hand_assist),
                },
            ]
            + [
                {
                    **spec,
                    "tracker": tracker_by_id.get(str(spec["track_id"]), "unresolved"),
                    "anchor": anchors.get(str(spec["track_id"])),
                    "sam3_quality": object_quality_meta.get(str(spec["track_id"])),
                    "ark_assisted": str(spec["track_id"]) in api_assisted_object_ids,
                }
                for spec in specs
            ],
        }
        write_json(str(stage5_dir / "tracks.json"), tracks_payload)

        source_counts: Counter[str] = Counter()
        visible_counts: Counter[str] = Counter()
        invalid_boxes: List[Dict[str, Any]] = []
        for frame_record in frame_records:
            for entity in frame_record["entities"]:
                source_counts[str(entity["source"])] += 1
                if entity["bbox_xyxy_pixel"] is not None:
                    visible_counts[str(entity["track_id"])] += 1
                    if _clamp_box(entity["bbox_xyxy_pixel"], width, height) != entity["bbox_xyxy_pixel"]:
                        invalid_boxes.append(
                            {"frame_index_0based": frame_record["frame_index_0based"], "track_id": entity["track_id"]}
                        )
        unresolved_labels = [
            str(spec["label"]) for spec in specs if str(spec["track_id"]) not in anchors
        ]
        track_quality, quality_warnings = _track_quality_report(
            frame_records, width, height
        )
        for track_id, quality in track_quality.items():
            if quality.get("entity_type") != "hand" or not quality.get("active_frames"):
                continue
            longest_missing = int(
                quality.get("longest_unexplained_missing_frames", 0) or 0
            )
            if longest_missing >= bbox_cfg.api_hand_min_missing_run_frames:
                quality_warnings.append(
                    {
                        "track_id": track_id,
                        "reason": "unexplained_expected_hand_missing_run_remains_after_all_assistance",
                        "longest_missing_frames": longest_missing,
                    }
                )
        hand_api_activity = hand_assist_meta.get("api_usage", {})
        if (
            hand_assist_meta.get("triggered")
            and int(hand_api_activity.get("api_calls", 0) or 0)
            + int(hand_api_activity.get("cache_hits", 0) or 0)
            > 0
            and not hand_assist_meta.get("accepted_anchors")
            and not hand_assist_meta.get("confirmed_not_visible_intervals")
        ):
            quality_warnings.append(
                {
                    "track_id": None,
                    "reason": "ark_hand_assist_produced_no_accepted_anchor",
                    "rejected_anchor_count": len(
                        hand_assist_meta.get("rejected_anchors", [])
                    ),
                }
            )
        combined_api_meta = _merge_grounding_meta(
            grounding_meta,
            {
                "api_usage": hand_assist_meta.get("api_usage", {}),
                "api_calls_by_step": hand_assist_meta.get("calls_by_step", {}),
                "local_grounding_error": "",
            },
        )
        api_usage = combined_api_meta["api_usage"]
        qa_summary = {
            "schema_version": STAGE5_SCHEMA_VERSION,
            "source_video": build_source_video_meta(video_path),
            "total_frames": total_frames,
            "fps": fps,
            "timestamp_source": timestamp_source,
            "frame_records": len(frame_records),
            "track_count": 2 + len(specs),
            "bbox_counts_by_track": dict(visible_counts),
            "source_counts": dict(source_counts),
            "invalid_bbox_count": len(invalid_boxes),
            "invalid_bbox_sample": invalid_boxes[:20],
            "unresolved_objects": unresolved_labels,
            "bbox_backend": bbox_cfg.backend,
            "primary_operator": primary_operator_meta,
            "sam3_error": sam3_error,
            "sam3": sam3_meta,
            "sam3_candidates_path": (
                "sam3_candidates.json"
                if bbox_cfg.backend == "sam3" and (stage5_dir / "sam3_candidates.json").exists()
                else None
            ),
            "legacy_fallback_track_ids": sorted(legacy_track_ids),
            "api_assisted_object_track_ids": sorted(api_assisted_object_ids),
            "sam3_object_quality": object_quality_meta,
            "optical_flow_augmentation": optical_flow_augmentation_meta,
            "api_object_anchor_validation": api_object_validation_meta,
            "ark_hand_assist": hand_assist_meta,
            "duplicate_hand_resolution": duplicate_hand_resolution_meta,
            "track_quality": track_quality,
            "quality_warnings": quality_warnings,
            "quality_gate": {
                "status": (
                    "failed"
                    if invalid_boxes or unresolved_labels
                    else ("warning" if quality_warnings else "passed")
                ),
                "structural_passed": not invalid_boxes and not unresolved_labels,
                "passed_without_warnings": (
                    not invalid_boxes and not unresolved_labels and not quality_warnings
                ),
                "eligible_for_automatic_use": (
                    not invalid_boxes and not unresolved_labels and not quality_warnings
                ),
                "requires_manual_review": bool(
                    invalid_boxes or unresolved_labels or quality_warnings
                ),
                "invalid_bbox_count": len(invalid_boxes),
                "unresolved_object_count": len(unresolved_labels),
                "warning_count": len(quality_warnings),
            },
            "local_grounding_error": grounding_meta.get("local_grounding_error", ""),
            "sam2_error": sam2_error,
            "api_calls_by_step": combined_api_meta.get("api_calls_by_step", {}),
            "api_object_calls_by_step": grounding_meta.get("api_calls_by_step", {}),
            "api_hand_calls_by_step": hand_assist_meta.get("calls_by_step", {}),
            "completed_at_utc": now_utc_iso(),
        }
        write_json(str(stage5_dir / "qa_summary.json"), qa_summary)
        write_json(
            str(stage5_dir / "api_usage.json"),
            {
                "model": api_cfg.model_name,
                "thinking_type": bbox_cfg.api_grounding_thinking,
                "reasoning_effort": bbox_cfg.api_grounding_reasoning_effort,
                "object_budget_max_calls_per_step": bbox_cfg.api_max_calls_per_step,
                "object_budget_max_total_calls": bbox_cfg.api_max_total_calls,
                "hand_budget_max_calls_per_step": bbox_cfg.api_hand_max_calls_per_step,
                "hand_budget_max_total_calls": bbox_cfg.api_hand_max_total_calls,
                "breakdown": {
                    "objects": grounding_meta.get("api_usage", {}),
                    "hands": hand_assist_meta.get("api_usage", {}),
                },
                **api_usage,
            },
        )
        if bbox_cfg.overlay:
            _write_overlay(
                video_path,
                stage5_dir / "bbox_overlay.mp4",
                frame_records,
                fps,
                width,
                height,
            )

    summary = {
        "schema_version": STAGE5_SCHEMA_VERSION,
        "source_video": build_source_video_meta(video_path),
        "source_video_stat": {
            "size_bytes": os.path.getsize(video_path),
            "mtime_ns": os.stat(video_path).st_mtime_ns,
        },
        "final_plan_sha256": final_plan_sha256,
        "total_frames": total_frames,
        "fps": fps,
        "timestamp_source": timestamp_source,
        "image_size": {"width": width, "height": height},
        "frame_bboxes_path": "frame_bboxes.jsonl",
        "tracks_path": "tracks.json",
        "qa_summary_path": "qa_summary.json",
        "api_usage_path": "api_usage.json",
        "bbox_backend": bbox_cfg.backend,
        "primary_operator_policy": PRIMARY_OPERATOR_POLICY,
        "primary_operator_qa_path": "qa_summary.json",
        "quality_gate": qa_summary["quality_gate"],
        "sam3_candidates_path": (
            "sam3_candidates.json"
            if bbox_cfg.backend == "sam3" and (stage5_dir / "sam3_candidates.json").exists()
            else None
        ),
        "overlay_path": "bbox_overlay.mp4" if bbox_cfg.overlay else None,
        "bbox_config": asdict(bbox_cfg),
        "device": device,
        "hand_device": hand_device,
        "completed_at_utc": now_utc_iso(),
    }
    write_json(str(summary_path), summary)
    api_usage = json.loads((stage5_dir / "api_usage.json").read_text(encoding="utf-8"))
    update_run_summary(
        run_summary_path,
        {
            "source_video": build_source_video_meta(video_path),
            "output_root": os.path.abspath(output_root),
            "updated_at_utc": now_utc_iso(),
            "stage5": {
                "status": (
                    "completed"
                    if qa_summary["quality_gate"]["status"] == "passed"
                    else (
                        "completed_with_warnings"
                        if qa_summary["quality_gate"]["status"] == "warning"
                        else "completed_quality_failed"
                    )
                ),
                "quality_gate": qa_summary["quality_gate"],
                "completed_at_utc": now_utc_iso(),
                "final_output_path": os.path.relpath(summary_path, video_out),
                "frame_bboxes_path": os.path.relpath(frame_jsonl_path, video_out),
                "token_usage": {
                    "prompt_tokens": int(api_usage.get("prompt_tokens", 0) or 0),
                    "completion_tokens": int(api_usage.get("completion_tokens", 0) or 0),
                    "total_tokens": int(api_usage.get("total_tokens", 0) or 0),
                    "api_calls": int(api_usage.get("api_calls", 0) or 0),
                },
            },
        },
    )
    logger.info(
        f"[stage5] video_id={video_id} completed frames={total_frames} "
        f"elapsed={format_duration(time.perf_counter() - started)}"
    )
    return video_out


def run_stage5_subprocess(
    *,
    bbox_python: str,
    video_path: str,
    output_root: str,
    api_cfg: "ApiConfig",
    bbox_cfg: BBoxConfig,
    overwrite: bool,
) -> str:
    interpreter = os.path.abspath(os.path.expanduser(str(bbox_python or "")))
    if not os.path.isfile(interpreter):
        raise FileNotFoundError(
            f"Stage 5 Python interpreter not found: {interpreter}. "
            "Create .venv_bbox_tracking or pass --bbox-python."
        )
    cmd = [
        interpreter,
        os.path.abspath(__file__),
        "--input-video",
        os.path.abspath(video_path),
        "--output-root",
        os.path.abspath(output_root),
        "--hand-detector",
        str(bbox_cfg.hand_detector),
        "--hand-conf",
        str(bbox_cfg.hand_conf),
        "--hand-iou",
        str(bbox_cfg.hand_iou),
        "--hand-batch-size",
        str(bbox_cfg.hand_batch_size),
        "--hand-gap-fill-frames",
        str(bbox_cfg.hand_gap_fill_frames),
        "--hand-device",
        str(bbox_cfg.hand_device),
        "--bbox-device",
        str(bbox_cfg.device),
        "--bbox-backend",
        str(bbox_cfg.backend),
        "--sam3-python",
        str(bbox_cfg.sam3_python),
        "--sam3-repo",
        str(bbox_cfg.sam3_repo),
        "--sam3-checkpoint",
        str(bbox_cfg.sam3_checkpoint),
        "--sam3-threshold",
        str(bbox_cfg.sam3_threshold),
        "--sam3-chunk-size",
        str(bbox_cfg.sam3_chunk_size),
        "--sam3-max-instances",
        str(bbox_cfg.sam3_max_instances),
        "--sam3-interval-padding-frames",
        str(bbox_cfg.sam3_interval_padding_frames),
        "--api-hand-max-calls-per-step",
        str(bbox_cfg.api_hand_max_calls_per_step),
        "--api-hand-max-total-calls",
        str(bbox_cfg.api_hand_max_total_calls),
        "--api-hand-min-missing-run-frames",
        str(bbox_cfg.api_hand_min_missing_run_frames),
        "--api-hand-low-conf-threshold",
        str(bbox_cfg.api_hand_low_conf_threshold),
        "--api-object-min-track-coverage",
        str(bbox_cfg.api_object_min_track_coverage),
        "--api-object-min-mean-confidence",
        str(bbox_cfg.api_object_min_mean_confidence),
        "--api-grounding-thinking",
        str(bbox_cfg.api_grounding_thinking),
        "--api-grounding-reasoning-effort",
        str(bbox_cfg.api_grounding_reasoning_effort),
        "--local-grounding-model",
        str(bbox_cfg.local_grounding_model),
        "--local-box-threshold",
        str(bbox_cfg.local_box_threshold),
        "--local-text-threshold",
        str(bbox_cfg.local_text_threshold),
        "--sam2-model",
        str(bbox_cfg.sam2_model),
        "--api-max-calls-per-step",
        str(bbox_cfg.api_max_calls_per_step),
        "--api-max-total-calls",
        str(bbox_cfg.api_max_total_calls),
        "--bbox-jpeg-quality",
        str(bbox_cfg.jpeg_quality),
    ]
    if bbox_cfg.disable_local_grounding:
        cmd.append("--disable-local-grounding")
    if bbox_cfg.disable_sam2:
        cmd.append("--disable-sam2")
    if bbox_cfg.allow_unresolved_objects:
        cmd.append("--allow-unresolved-objects")
    cmd.append("--sam3-hand-fusion" if bbox_cfg.sam3_hand_fusion else "--no-sam3-hand-fusion")
    cmd.append(
        "--sam3-legacy-fallback"
        if bbox_cfg.sam3_legacy_fallback
        else "--no-sam3-legacy-fallback"
    )
    cmd.append("--api-hand-assist" if bbox_cfg.api_hand_assist else "--no-api-hand-assist")
    cmd.append("--api-grounding" if bbox_cfg.api_grounding else "--no-api-grounding")
    cmd.append("--bbox-overlay" if bbox_cfg.overlay else "--no-bbox-overlay")
    if overwrite:
        cmd.append("--overwrite")
    env = os.environ.copy()
    env["ARK_API_KEY"] = str(api_cfg.api_key)
    env["API_BASE_URL"] = str(api_cfg.api_base_url)
    env["MODEL_NAME"] = str(api_cfg.model_name)
    completed = subprocess.run(cmd, check=False, env=env)
    if completed.returncode != 0:
        raise RuntimeError(f"Stage 5 subprocess failed with exit code {completed.returncode}.")
    return os.path.join(output_root, video_id_from_path(video_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 5: per-original-frame left/right hand and manipulated-object bounding boxes."
    )
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--output-root", required=True)
    add_api_cli_args(parser, include_no_embed_index=False)
    add_bbox_cli_args(parser, include_bbox_python=False)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    api_cfg = api_config_from_args(args)
    bbox_cfg = bbox_config_from_args(args)
    run_stage5_for_video(
        args.input_video,
        args.output_root,
        api_cfg,
        bbox_cfg,
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    main()
