#!/usr/bin/env python3
"""Standalone SAM3 text-prompt video backend for Stage 5.

This file intentionally runs in the dedicated SAM3 Conda environment.  It accepts
numeric JPEG/PNG frames plus semantic requests and returns every detected instance
with stable IDs inside each processed segment.  The parent Stage 5 process assigns
the semantic track, anatomical handedness, visibility, and final output schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def _numeric_frames(frame_dir: Path) -> List[Path]:
    paths = [
        path
        for path in frame_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and path.stem.isdigit()
    ]
    return sorted(paths, key=lambda path: int(path.stem))


def _merge_ranges(raw: Iterable[Sequence[Any]], total_frames: int) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    for value in raw:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            continue
        start = min(total_frames, max(0, int(value[0])))
        end = min(total_frames, max(start, int(value[1])))
        if end > start:
            ranges.append((start, end))
    if not ranges:
        return [(0, total_frames)]
    merged: List[List[int]] = []
    for start, end in sorted(ranges):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _clamp_box(box: Sequence[float], width: int, height: int) -> List[int] | None:
    x1, y1, x2, y2 = [float(value) for value in box]
    x1 = min(max(0.0, x1), width - 1.0)
    y1 = min(max(0.0, y1), height - 1.0)
    x2 = min(max(0.0, x2), width - 1.0)
    y2 = min(max(0.0, y2), height - 1.0)
    result = [int(x1), int(y1), int(round(x2)), int(round(y2))]
    return result if result[2] > result[0] and result[3] > result[1] else None


def _mask_box(mask: Any, width: int, height: int) -> Tuple[List[int] | None, int]:
    import cv2
    import numpy as np

    raw = mask.detach().cpu().numpy() if hasattr(mask, "detach") else np.asarray(mask)
    raw = np.squeeze(raw)
    if raw.ndim != 2:
        return None, 0
    binary = raw.astype(bool)
    if binary.shape != (height, width):
        binary = cv2.resize(
            binary.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    ys, xs = np.where(binary)
    if not len(xs):
        return None, 0
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], int(len(xs))


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return ""


def _checkpoint_sha256(checkpoint: Path) -> str:
    sidecar = checkpoint.with_suffix(checkpoint.suffix + ".sha256")
    if sidecar.is_file():
        value = sidecar.read_text(encoding="utf-8").strip().split()[0]
        if len(value) == 64:
            return value
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    value = digest.hexdigest()
    sidecar.write_text(f"{value}  {checkpoint.name}\n", encoding="utf-8")
    return value


def run(args: argparse.Namespace) -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    repo = Path(args.sam3_repo).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    frame_dir = Path(args.frames_dir).expanduser().resolve()
    if not (repo / "sam3" / "model_builder.py").is_file():
        raise FileNotFoundError(f"SAM3 repository is incomplete: {repo}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint}")
    sys.path.insert(0, str(repo))

    import cv2
    import numpy as np
    import torch
    from sam3.model_builder import build_sam3_video_predictor

    frames = _numeric_frames(frame_dir)
    if not frames:
        raise RuntimeError(f"No numeric JPEG/PNG frames found in {frame_dir}")
    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"Cannot read first frame: {frames[0]}")
    height, width = first.shape[:2]
    request_payload = json.loads(Path(args.requests_json).read_text(encoding="utf-8"))
    requests = request_payload.get("requests") or []
    if not requests:
        raise RuntimeError("SAM3 request file contains no requests")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    predictor = build_sam3_video_predictor(
        checkpoint_path=str(checkpoint),
        async_loading_frames=True,
        gpus_to_use=[int(args.gpu)],
    )
    output_requests: Dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="stage5_sam3_") as temp_root:
        for request_index, request in enumerate(requests):
            request_id = str(request["request_id"])
            prompt = " ".join(str(request["prompt"]).strip().split())
            max_instances = max(0, int(request.get("max_instances", args.max_instances)))
            ranges = _merge_ranges(request.get("ranges") or [], len(frames))
            frame_results: Dict[str, List[Dict[str, Any]]] = {}
            segment_index = 0
            for range_start, range_end in ranges:
                for start in range(range_start, range_end, int(args.chunk_size)):
                    end = min(range_end, start + int(args.chunk_size))
                    segment_dir = Path(temp_root) / f"r{request_index:03d}_s{segment_index:05d}"
                    segment_dir.mkdir()
                    for local_index, global_index in enumerate(range(start, end)):
                        source = frames[global_index]
                        target = segment_dir / f"{local_index:06d}{source.suffix.lower()}"
                        target.symlink_to(source)
                    session_id = None
                    try:
                        response = predictor.handle_request(
                            {
                                "type": "start_session",
                                "resource_path": str(segment_dir),
                                "offload_video_to_cpu": True,
                                "offload_state_to_cpu": True,
                            }
                        )
                        session_id = response["session_id"]
                        predictor.handle_request(
                            {
                                "type": "add_prompt",
                                "session_id": session_id,
                                "frame_index": 0,
                                "text": prompt,
                                "output_prob_thresh": float(args.threshold),
                            }
                        )
                        stream = predictor.handle_stream_request(
                            {
                                "type": "propagate_in_video",
                                "session_id": session_id,
                                "propagation_direction": "forward",
                                "output_prob_thresh": float(args.threshold),
                            }
                        )
                        for response in stream:
                            global_index = start + int(response["frame_index"])
                            raw = response["outputs"]
                            object_ids = np.asarray(raw["out_obj_ids"]).reshape(-1)
                            probabilities = np.asarray(raw["out_probs"]).reshape(-1)
                            boxes = (
                                np.asarray(raw["out_boxes_xywh"]).reshape(-1, 4)
                                if len(object_ids)
                                else np.zeros((0, 4), dtype=np.float32)
                            )
                            masks = raw.get("out_binary_masks")
                            order = np.argsort(-probabilities) if len(probabilities) else []
                            if max_instances > 0:
                                order = order[:max_instances]
                            candidates: List[Dict[str, Any]] = []
                            for candidate_index in order:
                                x, y, box_width, box_height = boxes[candidate_index]
                                box = _clamp_box(
                                    [
                                        x * width,
                                        y * height,
                                        (x + box_width) * width,
                                        (y + box_height) * height,
                                    ],
                                    width,
                                    height,
                                )
                                if box is None:
                                    continue
                                mask_box, mask_area = (None, 0)
                                if masks is not None:
                                    mask_box, mask_area = _mask_box(
                                        masks[candidate_index], width, height
                                    )
                                object_id = int(object_ids[candidate_index])
                                candidates.append(
                                    {
                                        "instance_id": f"seg{segment_index:05d}_obj{object_id}",
                                        "sam3_obj_id": object_id,
                                        "confidence": round(
                                            float(probabilities[candidate_index]), 6
                                        ),
                                        "bbox_xyxy_pixel": box,
                                        "visible_mask_bbox_xyxy_pixel": mask_box,
                                        "visible_mask_area_pixels": mask_area,
                                    }
                                )
                            if candidates:
                                frame_results[str(global_index)] = candidates
                    finally:
                        if session_id is not None:
                            predictor.handle_request(
                                {"type": "close_session", "session_id": session_id}
                            )
                        shutil.rmtree(segment_dir, ignore_errors=True)
                        torch.cuda.empty_cache()
                    segment_index += 1
            output_requests[request_id] = {
                "prompt": prompt,
                "ranges": ranges,
                "segments": segment_index,
                "frames_with_detections": len(frame_results),
                "frames": frame_results,
            }
            print(
                f"[sam3] request={request_id} prompt={prompt!r} "
                f"frames_with_detections={len(frame_results)} segments={segment_index}",
                flush=True,
            )

    payload = {
        "schema_version": "stage5-sam3-candidates-v1",
        "source_frames": str(frame_dir),
        "total_frames": len(frames),
        "image_size": {"width": width, "height": height},
        "threshold": float(args.threshold),
        "chunk_size": int(args.chunk_size),
        "sam3": {
            "repo": str(repo),
            "repo_commit": _git_commit(repo),
            "checkpoint": str(checkpoint),
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": _checkpoint_sha256(checkpoint),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "requests": output_requests,
    }
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    os.replace(temporary, output_path)
    print(f"[sam3] saved {output_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="SAM3 backend for Stage 5 bbox tracking")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--requests-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--sam3-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--max-instances", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
