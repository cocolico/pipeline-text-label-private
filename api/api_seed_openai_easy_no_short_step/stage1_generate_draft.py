#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from common import (
    _contains_frame_ref,
    _contains_time_ref,
    add_api_cli_args,
    add_sampling_cli_args,
    VIDEO_EXTS,
    api_config_from_args,
    build_source_video_meta,
    build_video_api_content,
    build_retry_prefix,
    call_chat_completion,
    collect_videos,
    default_output_root,
    ensure_frame_index_fps,
    ensure_video_out_dir_safe,
    extract_json_from_response,
    format_duration,
    guard_schema_fingerprint,
    initialize_api_client,
    logger,
    normalize_draft_plan,
    now_utc_iso,
    read_json,
    sample_video_to_frame_index,
    sampling_config_from_args,
    update_run_summary,
    video_id_from_path,
    write_frame_manifest,
    write_json,
    write_text,
)
from prompts import SYSTEM_PROMPT_STEP_PLANNER, build_stage1_user_prompt
from hand_step_pipeline import generate_stage1_vlm_hand_events, materialize_stage1_keep_clips

if TYPE_CHECKING:
    from common import ApiConfig, SamplingConfig


_STAGE1_ALLOWED_TOP_KEYS = {"high_level_goal", "steps"}
_STAGE1_ALLOWED_STEP_KEYS = {
    "step_id", "step_goal", "start_frame_id", "end_frame_id", "clip_path",
    "left_hand", "right_hand", "overall", "multiple_action_subjects",
}
_STAGE1_FORBIDDEN_KEYS = {
    "rationale",
    "counterfactual_challenge_question",
    "expected_challenge_outcome",
    "failure_reflecting",
    "critical_frames",
    "frame_index",
    "interaction",
    "keyframe_image_path",
}


def _find_forbidden_keys(obj: Any, path: str) -> List[str]:
    errors: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            next_path = f"{path}.{k}" if path else k
            if k in _STAGE1_FORBIDDEN_KEYS:
                errors.append(f"Forbidden key '{k}' found at: {next_path}")
            errors.extend(_find_forbidden_keys(v, next_path))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            errors.extend(_find_forbidden_keys(v, f"{path}[{i}]"))
    return errors


def _stage1_raw_schema_errors(plan: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(plan, dict):
        return ["Stage 1 output must be a JSON object."]

    errors.extend(_find_forbidden_keys(plan, ""))

    extra_top = sorted(set(plan.keys()) - _STAGE1_ALLOWED_TOP_KEYS)
    if extra_top:
        errors.append(f"Stage 1 top-level contains extra keys (not allowed): {extra_top}")

    steps = plan.get("steps")
    if not isinstance(steps, list):
        errors.append("Stage 1 'steps' must be a list.")
        return errors

    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            errors.append(f"steps[{i}] is not an object.")
            continue
        extra_step = sorted(set(step.keys()) - _STAGE1_ALLOWED_STEP_KEYS)
        if extra_step:
            errors.append(f"steps[{i}] contains extra keys (not allowed): {extra_step}")

    return errors


def _draft_hard_errors(draft: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    def _has_disallowed_ref(text: Any) -> bool:
        # Stage 1 prompt forbids both frame/image indices and timestamps (e.g., "Frame 12", "t=3.2s").
        if isinstance(text, list):
            return any(_has_disallowed_ref(x) for x in text)
        if isinstance(text, dict):
            return any(_has_disallowed_ref(v) for v in text.values())
        return _contains_frame_ref(text) or _contains_time_ref(text)

    goal = str(draft.get("high_level_goal", "")).strip()
    if not goal:
        errors.append("high_level_goal is missing/empty.")
    if _has_disallowed_ref(goal):
        errors.append("high_level_goal must not reference frame indices or timestamps (e.g., 'Frame 12', 't=3.2s').")

    steps = draft.get("steps", [])
    if not isinstance(steps, list):
        errors.append("steps is missing or is not a list.")
        return errors
    if not steps:
        return errors

    for i, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            errors.append(f"steps[{i}] is not an object.")
            continue
        sg = str(step.get("step_goal", "")).strip()
        if not sg:
            errors.append(f"steps[{i}].step_goal is empty.")
        if _has_disallowed_ref(sg):
            errors.append(f"steps[{i}].step_goal must not reference frame indices or timestamps.")
        if sg.startswith("unnamed_step_"):
            errors.append(f"steps[{i}].step_goal looks like a placeholder ('{sg}').")
        for key in ("left_hand", "right_hand", "overall"):
            value = str(step.get(key, "")).strip()
            if not value:
                errors.append(f"steps[{i}].{key} is empty.")
            elif any(ord(char) > 127 for char in value):
                errors.append(f"steps[{i}].{key} must be English ASCII text.")
        try:
            start = int(step.get("start_frame_id"))
            end = int(step.get("end_frame_id"))
            if start < 0 or end < start:
                raise ValueError
        except Exception:
            errors.append(f"steps[{i}] has an invalid frame interval.")
        if not str(step.get("clip_path", "")).strip():
            errors.append(f"steps[{i}].clip_path is empty.")

    return errors


def _can_resume_stage1(draft_path: str, manifest_path: str) -> bool:
    """Return True if cached Stage-1 outputs exist and pass strict validation."""
    if not (os.path.exists(draft_path) and os.path.exists(manifest_path)):
        return False
    try:
        draft = read_json(draft_path)
        manifest = read_json(manifest_path)
    except Exception:
        return False

    steps = draft.get("steps", [])
    if not (isinstance(steps, list) and isinstance(draft.get("high_level_goal"), str)):
        return False

    for idx, st in enumerate(steps, start=1):
        if not isinstance(st, dict):
            return False
        if any(k in st for k in _STAGE1_FORBIDDEN_KEYS):
            return False
        try:
            if int(st.get("step_id")) != idx:
                return False
        except Exception:
            return False

    if not (int(manifest.get("num_frames", 0)) > 0 and isinstance(manifest.get("frames"), list)):
        return False

    if _stage1_raw_schema_errors(draft):
        return False
    return not _draft_hard_errors(draft)


def _draft_from_step_semantics(events_payload: Dict[str, Any]) -> Dict[str, Any]:
    draft: Dict[str, Any] = {
        "high_level_goal": "Describe the primary operator's hand actions in each upstream keep=true step.",
        "steps": [],
    }
    for record in events_payload.get("step_semantics", []):
        semantics = record.get("action_semantics", {})
        draft["steps"].append({
            "step_id": int(record["step_id"]),
            "step_goal": str(semantics.get("overall", "")).strip(),
            "start_frame_id": int(record["start_frame_id"]),
            "end_frame_id": int(record["end_frame_id"]),
            "clip_path": str(record["clip_path"]),
            "left_hand": str(semantics.get("left_hand", "")).strip(),
            "right_hand": str(semantics.get("right_hand", "")).strip(),
            "overall": str(semantics.get("overall", "")).strip(),
            "multiple_action_subjects": bool(record.get("multiple_action_subjects", False)),
        })
    return draft


def run_stage1_for_video(
    video_path: str,
    output_root: str,
    api_cfg: ApiConfig,
    sampling_cfg: SamplingConfig,
    overwrite: bool,
    max_retries: int,
    *,
    allow_legacy_resume: bool = False,
) -> str:
    t_start = time.perf_counter()
    ensure_frame_index_fps(sampling_cfg, getattr(api_cfg, "video_fps", 10.0))
    vid = video_id_from_path(video_path)
    video_out = os.path.join(output_root, vid)
    ensure_video_out_dir_safe(video_out, video_path)
    stage1_dir = os.path.join(video_out, "stage1")
    manifest_path = os.path.join(stage1_dir, "frame_manifest.json")
    draft_path = os.path.join(stage1_dir, "draft_plan.json")
    raw_path = os.path.join(stage1_dir, "stage1_raw_response.txt")
    sys_prompt_path = os.path.join(stage1_dir, "stage1_system_prompt.txt")
    user_prompt_path = os.path.join(stage1_dir, "stage1_user_prompt.txt")
    vlm_hand_events_path = os.path.join(stage1_dir, "vlm_hand_events.json")
    step_clips_manifest_path = os.path.join(stage1_dir, "step_clips.json")
    run_summary_path = os.path.join(video_out, "run_summary.json")

    visualization_cache_complete = False
    if os.path.exists(step_clips_manifest_path):
        try:
            cached_clips = read_json(step_clips_manifest_path).get("steps", [])
            visualization_cache_complete = bool(cached_clips) and all(
                isinstance(item, dict)
                and bool(item.get("visualization_clip_path"))
                and os.path.exists(os.path.join(video_out, str(item["visualization_clip_path"])))
                and bool(item.get("bbox_repair_summary"))
                for item in cached_clips
            )
        except Exception:
            visualization_cache_complete = False

    will_resume = (
        not overwrite
        and _can_resume_stage1(draft_path, manifest_path)
        and os.path.exists(vlm_hand_events_path)
        and os.path.exists(step_clips_manifest_path)
        and visualization_cache_complete
    )
    schema_fp = guard_schema_fingerprint(
        run_summary_path,
        video_out,
        stage="Stage 1",
        overwrite=overwrite,
        allow_legacy_resume=allow_legacy_resume,
        will_resume=will_resume,
    )
    if will_resume:
        logger.info(f"[stage1] video_id={vid} resume: {os.path.relpath(video_out, output_root)}")
        return video_out

    logger.info(
        f"[stage1] video_id={vid} start: overwrite={bool(overwrite)} "
        f"frame_index_fps={float(getattr(sampling_cfg, 'frame_index_fps', 0.0) or 0.0)} src={os.path.abspath(video_path)}"
    )

    # Record source metadata early so reruns can detect/avoid output collisions even if Stage1 fails mid-run.
    update_run_summary(
        run_summary_path,
        {
            "source_video": build_source_video_meta(video_path),
            "output_root": os.path.abspath(output_root),
            "schema_fingerprint": schema_fp,
            "updated_at_utc": now_utc_iso(),
            "stage1": {"status": "running", "started_at_utc": now_utc_iso()},
        },
    )

    frames, dims = sample_video_to_frame_index(video_path, sampling_cfg)
    logger.info(f"[stage1] video_id={vid} frame_index_points={len(frames)} dims={dims}")
    write_frame_manifest(frames, None, manifest_path)

    # Stage 1's first and authoritative operation is deterministic: read the
    # upstream frame IDs, group only strictly adjacent keep=true IDs, and cut
    # one exact decoded-frame clip per run.  This happens before API client
    # initialization, so clip production never depends on a VLM request.
    keep_clips_payload = materialize_stage1_keep_clips(
        video_path=video_path,
        video_out=video_out,
    )

    has_effective_actions = bool(keep_clips_payload.get("steps"))
    client = None
    if has_effective_actions:
        client = initialize_api_client(api_cfg)
        if not client:
            raise SystemExit("Failed to initialize API client.")

    # Upload each previously materialized clip sequentially and describe it.
    write_text(sys_prompt_path, "Each Stage-1 keep=true clip is analyzed independently and sequentially by Doubao Seed.")
    write_text(user_prompt_path, "Return English left_hand, right_hand, and overall descriptions for each fixed upstream interval.")
    warnings: List[str] = [] if has_effective_actions else ["no_effective_action"]
    attempts = 0
    stage_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "api_calls": 0}
    events_payload, hand_event_usage = generate_stage1_vlm_hand_events(
        video_path=video_path,
        video_out=video_out,
        client=client,
        api_cfg=api_cfg,
        frames=frames,
        overwrite=overwrite,
        max_retries=max_retries,
    )
    for usage_key in ("prompt_tokens", "completion_tokens", "total_tokens", "api_calls"):
        stage_usage[usage_key] = int(stage_usage.get(usage_key, 0)) + int(hand_event_usage.get(usage_key, 0))
    normalized = _draft_from_step_semantics(events_payload)
    draft_errors = _stage1_raw_schema_errors(normalized) + _draft_hard_errors(normalized)
    if draft_errors:
        raise RuntimeError("Invalid Stage-1 draft_plan.json: " + " | ".join(draft_errors[:20]))
    last_content = json.dumps(normalized, ensure_ascii=False, indent=2)
    write_text(raw_path, last_content)
    write_json(draft_path, normalized)
    step_count = len(normalized.get("steps", [])) if isinstance(normalized, dict) else 0
    logger.info(
        f"[stage1] video_id={vid} completed: steps={step_count} attempts={attempts} warnings={len(warnings)} "
        f"elapsed={format_duration(time.perf_counter() - t_start)}"
    )

    update_run_summary(
        run_summary_path,
        {
            "source_video": build_source_video_meta(video_path),
            "output_root": os.path.abspath(output_root),
            "updated_at_utc": now_utc_iso(),
            "api_config": {
                "api_base_url": api_cfg.api_base_url,
                "model_provider_id": api_cfg.model_provider_id,
                "model_name": api_cfg.model_name,
                "reasoning_effort": api_cfg.reasoning_effort,
                "max_tokens": int(api_cfg.max_tokens),
                "temperature": float(getattr(api_cfg, "temperature", 0.2)),
                "api_call_retries": int(getattr(api_cfg, "api_call_retries", 1)),
                "api_call_retry_backoff_sec": float(getattr(api_cfg, "api_call_retry_backoff_sec", 1.0)),
            },
            "sampling_config": {
                "max_frames": sampling_cfg.max_frames,
                "resize_dimension": sampling_cfg.resize_dimension,
                "jpeg_quality": sampling_cfg.jpeg_quality,
                "frame_index_fps": sampling_cfg.frame_index_fps,
            },
            "stage1": {
                "status": "completed",
                "attempts": attempts,
                "manifest_path": os.path.relpath(manifest_path, video_out),
                "draft_plan_path": os.path.relpath(draft_path, video_out),
                "raw_response_path": os.path.relpath(raw_path, video_out),
                "vlm_hand_events_path": os.path.relpath(vlm_hand_events_path, video_out),
                "step_clips_manifest_path": os.path.relpath(step_clips_manifest_path, video_out),
                "step_clip_count": len(keep_clips_payload.get("steps", [])),
                "system_prompt_path": os.path.relpath(sys_prompt_path, video_out),
                "user_prompt_path": os.path.relpath(user_prompt_path, video_out),
                "warnings": warnings,
                "token_usage": stage_usage,
            },
        },
    )

    return video_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: generate draft plan (no critical_frames).")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-video", help="Path to one video file.")
    src.add_argument("--input-video-dir", help="Directory of videos to process.")
    parser.add_argument("--output-root", default=default_output_root(), help="Output root under ECCV/three_stage/...")

    add_api_cli_args(parser, include_no_embed_index=False)
    add_sampling_cli_args(parser, default_max_frames=100, default_jpeg_quality=95)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-legacy-resume",
        action="store_true",
        help="Allow resuming cached outputs whose run_summary.json lacks schema_fingerprint (legacy outputs).",
    )
    args = parser.parse_args()

    api_cfg = api_config_from_args(args)
    sampling_cfg = sampling_config_from_args(args)

    videos: List[str] = []
    if args.input_video:
        videos = [args.input_video]
    else:
        videos = collect_videos(args.input_video_dir, VIDEO_EXTS)

    if not videos:
        raise SystemExit("No videos found.")

    for vp in videos:
        run_stage1_for_video(
            vp,
            args.output_root,
            api_cfg,
            sampling_cfg,
            overwrite=args.overwrite,
            max_retries=args.max_retries,
            allow_legacy_resume=args.allow_legacy_resume,
        )


if __name__ == "__main__":
    main()
