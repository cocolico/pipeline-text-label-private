#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import shutil
import time
from concurrent.futures import CancelledError, FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Any, Dict, List, Optional, Sequence, Tuple

from common import (
    add_api_cli_args,
    add_sampling_cli_args,
    OutputDirCollisionError,
    VIDEO_EXTS,
    api_config_from_args,
    build_source_video_meta,
    collect_videos,
    default_output_root,
    ensure_frame_index_fps,
    ensure_video_out_dir_safe,
    format_duration,
    logger,
    now_utc_iso,
    print_cost_summary,
    read_json,
    sampling_config_from_args,
    source_video_id,
    source_video_path,
    three_stage_schema_fingerprint,
    update_run_summary,
    video_id_from_path,
    write_json,
)
from stage1_generate_draft import run_stage1_for_video
from stage2_localize_and_cut import run_stage2_for_video
from stage3_refine_and_keyframes import run_stage3_for_video
from stage4_atomic_actions import run_stage4_for_video
from stage5_bbox_tracking import (
    BBoxConfig,
    add_bbox_cli_args,
    bbox_config_from_args,
    run_stage5_subprocess,
)


def _one_line(text: str, *, max_len: int = 500) -> str:
    line = str(text or "").replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()
    if len(line) > max_len:
        return line[: max_len - 3] + "..."
    return line


def per_video_sibling_output_root(video_path: str) -> str:
    """Return the ``processed_text_label`` directory beside ``raw_data``.

    This layout is deliberately keyed by the source video directory instead
    of its filename.  It makes a recursive batch of many ``raw_data/rgb.mp4``
    files safe: every video receives an independent cache and run summary.
    """
    source_abs = os.path.abspath(video_path)
    raw_data_dir = os.path.dirname(source_abs)
    if os.path.basename(raw_data_dir) != "raw_data":
        raise ValueError(
            "--output-per-video-sibling requires an input video directly under a raw_data directory: "
            f"{source_abs}"
        )
    return os.path.join(os.path.dirname(raw_data_dir), "processed_text_label")


def output_root_for_video(video_path: str, shared_output_root: str, *, per_video_sibling: bool) -> str:
    if per_video_sibling:
        return per_video_sibling_output_root(video_path)
    return shared_output_root


def _append_schema_mismatch_txt(
    path: str,
    *,
    video_id: str,
    source_video: str,
    video_out: str,
    errors: Optional[List[str]] = None,
    warnings: Optional[List[str]] = None,
    note: str = "",
) -> None:
    if not path:
        return
    errors = errors or []
    warnings = warnings or []
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    is_new = not os.path.exists(path)
    try:
        with open(path, "a", encoding="utf-8") as txt_file:
            if is_new:
                txt_file.write("# four_stage schema mismatch records (post-validate failures)\n")
                txt_file.write("# columns: time_utc, video_id, source_video, video_out, errors, warnings, note, error_sample\n")
            err_sample = " | ".join(_one_line(err, max_len=200) for err in errors[:6])
            txt_file.write(
                "\t".join(
                    [
                        now_utc_iso(),
                        f"video_id={video_id}",
                        f"source_video={_one_line(os.path.abspath(source_video), max_len=300)}",
                        f"video_out={_one_line(os.path.abspath(video_out), max_len=300)}",
                        f"errors={len(errors)}",
                        f"warnings={len(warnings)}",
                        f"note={_one_line(note, max_len=200)}" if note else "note=",
                        f"error_sample={err_sample}",
                    ]
                )
                + "\n"
            )
    except Exception:
        # Best-effort logging; never fail the pipeline due to an auxiliary file write.
        return


def _get_status(summary: Any, key: str) -> str:
    if not isinstance(summary, dict):
        return ""
    obj = summary.get(key)
    if not isinstance(obj, dict):
        return ""
    raw = obj.get("status")
    if not isinstance(raw, str):
        return ""
    return raw.strip().lower()


def _get_error(summary: Any, key: str) -> str:
    if not isinstance(summary, dict):
        return ""
    obj = summary.get(key)
    if not isinstance(obj, dict):
        return ""
    raw = obj.get("error")
    if not isinstance(raw, str):
        return ""
    return raw.strip()


def _expected_stage_output(video_out: str, stage_id: str) -> str:
    if str(stage_id) == "1":
        return os.path.join(video_out, "stage1", "draft_plan.json")
    if str(stage_id) == "2":
        return os.path.join(video_out, "stage2", "step_segments.json")
    if str(stage_id) == "3":
        return os.path.join(video_out, "stage3", "step_plan.json")
    if str(stage_id) == "4":
        return os.path.join(video_out, "stage4", "atomic_plan_with_clips.json")
    if str(stage_id) == "5":
        return os.path.join(video_out, "stage5", "stage5_summary.json")
    return ""


def _missing_expected_outputs(video_out: str, stages: Sequence[str]) -> List[str]:
    stage_set = {str(s).strip() for s in (stages or []) if str(s).strip()}
    # If Stage 4 is requested, treat the stage4 final JSON as the completion signal.
    # If Stage 3 is the terminal stage, treat the stage3 final JSON as the completion signal.
    if "5" in stage_set:
        stage_set = {"5"}
    elif "4" in stage_set:
        stage_set = {"4"}
    elif "3" in stage_set:
        stage_set = {"3"}

    missing: List[str] = []
    for sid in sorted(stage_set):
        expected = _expected_stage_output(video_out, sid)
        if not expected:
            continue
        try:
            if not os.path.exists(expected) or os.path.getsize(expected) <= 0:
                missing.append(expected)
        except Exception:
            missing.append(expected)
    return missing


def _maybe_update_run_summary_source_video(video_out: str, *, source_video: str) -> None:
    """Best-effort: keep run_summary.json `source_video` aligned to the current input path.

    This is bookkeeping only and must NOT trigger any regeneration.
    """
    rs_path = os.path.join(video_out, "run_summary.json")
    if not os.path.exists(rs_path):
        return
    try:
        rs = read_json(rs_path)
    except Exception:
        return
    if not isinstance(rs, dict):
        return
    cur = os.path.abspath(source_video)
    old = source_video_path(rs.get("source_video"))
    if old and os.path.abspath(old) == cur:
        return
    try:
        update_run_summary(rs_path, {"source_video": build_source_video_meta(source_video)})
    except Exception:
        return


def _collect_previous_run_issues(
    *,
    video_path: str,
    output_root: str,
    stages: Sequence[str],
    post_validate: bool,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    vp = str(video_path)
    vid = video_id_from_path(vp)
    video_out = os.path.join(output_root, vid)
    rs_path = os.path.join(video_out, "run_summary.json")
    if not os.path.exists(rs_path):
        return None, []

    try:
        rs = read_json(rs_path)
    except Exception as e:
        meta = {
            "source_video": build_source_video_meta(vp),
            "video_out": os.path.abspath(video_out),
            "run_summary_path": os.path.abspath(rs_path),
        }
        issues = [
            {
                "key": "run_summary",
                "stage": "run_summary",
                "status": "unreadable",
                "error": f"{type(e).__name__}: {e}",
            }
        ]
        return meta, issues

    rs_source = rs.get("source_video")
    meta = {
        "source_video": build_source_video_meta(vp),
        "video_out": os.path.abspath(video_out),
        "run_summary_path": os.path.abspath(rs_path),
        "run_summary_source_video": build_source_video_meta(source_video_path(rs_source))
        if source_video_path(rs_source)
        else {},
    }

    stage_set = {str(s).strip() for s in (stages or []) if str(s).strip()}
    issues: List[Dict[str, Any]] = []
    for sid in ("1", "2", "3", "4", "5"):
        if sid not in stage_set:
            continue
        key = f"stage{sid}"
        status = _get_status(rs, key)
        err = _get_error(rs, key)
        if status in {"failed", "running"} or not status:
            issues.append(
                {
                    "key": key,
                    "stage": sid,
                    "status": status or "missing",
                    "error": err,
                }
            )
            continue
        expected = _expected_stage_output(video_out, sid)
        if expected and not os.path.exists(expected):
            issues.append(
                {
                    "key": key,
                    "stage": sid,
                    "status": "completed_but_missing_output",
                    "error": f"Missing expected output: {os.path.abspath(expected)}",
                }
            )

    # In retry-failed mode, consider post_validate only if it explicitly failed (do not select
    # every completed video just because it hasn't been validated before).
    if bool(post_validate) and "4" in stage_set:
        pv_status = _get_status(rs, "post_validate")
        if pv_status in {"failed", "running"}:
            issues.append(
                {
                    "key": "post_validate",
                    "stage": "post_validate",
                    "status": pv_status,
                    "error": _get_error(rs, "post_validate"),
                }
            )

    if not issues:
        return None, []
    return meta, issues


def _run_one_video_pipeline(
    *,
    video_path: str,
    output_root: str,
    stages: Sequence[str],
    api_cfg: Any,
    sampling_cfg: Any,
    overwrite: bool,
    allow_legacy_resume: bool,
    stage1_retries: int,
    stage2_retries: int,
    stage3_retries: int,
    stage4_retries: int,
    bbox_python: str,
    bbox_cfg: BBoxConfig,
    ffmpeg_bin: str,
    cut_mode: str,
    seek_slop_sec: float,
    crf: int,
    preset: str,
    keep_audio: bool,
    post_validate: bool,
    post_validate_fail_on_warnings: bool,
) -> Dict[str, Any]:
    """Run the pipeline for one video (optionally including Stage 5 and post-validate).

    Designed to be ProcessPool-friendly: takes only picklable inputs and returns a summary dict.
    """
    t0 = time.perf_counter()
    vp = str(video_path)
    vid = video_id_from_path(vp)
    video_out = os.path.join(output_root, vid)
    schema_fp = three_stage_schema_fingerprint()

    stage_failed = False
    failed_stage: Optional[str] = None
    stage_error: str = ""
    schema_mismatch: Optional[Dict[str, Any]] = None

    logger.info(f"[pipeline-worker] video_id={vid} start: {os.path.abspath(vp)}")

    for sid in ("1", "2", "3", "4", "5"):
        if sid not in set(stages):
            continue
        try:
            t_stage = time.perf_counter()
            logger.info(f"[pipeline-worker] video_id={vid} stage={sid} start")
            if sid == "1":
                run_stage1_for_video(
                    vp,
                    output_root,
                    api_cfg,
                    sampling_cfg,
                    overwrite=overwrite,
                    max_retries=int(stage1_retries),
                    allow_legacy_resume=allow_legacy_resume,
                )
            elif sid == "2":
                run_stage2_for_video(
                    vp,
                    output_root,
                    api_cfg,
                    ffmpeg_bin=ffmpeg_bin,
                    overwrite=overwrite,
                    max_retries=int(stage2_retries),
                    cut_mode=cut_mode,
                    seek_slop_sec=float(seek_slop_sec),
                    crf=int(crf),
                    preset=preset,
                    keep_audio=bool(keep_audio),
                    allow_legacy_resume=allow_legacy_resume,
                )
            elif sid == "3":
                run_stage3_for_video(
                    vp,
                    output_root,
                    api_cfg,
                    sampling_cfg,
                    overwrite=overwrite,
                    max_retries=int(stage3_retries),
                    allow_legacy_resume=allow_legacy_resume,
                )
            elif sid == "4":
                run_stage4_for_video(
                    vp,
                    output_root,
                    api_cfg,
                    sampling_cfg,
                    overwrite=overwrite,
                    max_retries=int(stage4_retries),
                    ffmpeg_bin=ffmpeg_bin,
                    allow_legacy_resume=allow_legacy_resume,
                )
            elif sid == "5":
                run_stage5_subprocess(
                    bbox_python=bbox_python,
                    video_path=vp,
                    output_root=output_root,
                    api_cfg=api_cfg,
                    bbox_cfg=bbox_cfg,
                    overwrite=overwrite,
                )
            logger.info(
                f"[pipeline-worker] video_id={vid} stage={sid} done in {format_duration(time.perf_counter() - t_stage)}"
            )
        except OutputDirCollisionError as e:
            stage_failed = True
            failed_stage = sid
            stage_error = f"{type(e).__name__}: {e}"
            logger.error(f"[pipeline-worker] video_id={vid} stage={sid} aborted: {stage_error}")
            # Do NOT write into run_summary.json here: the directory may belong to another source video.
            break
        except (Exception, SystemExit) as e:
            stage_failed = True
            failed_stage = sid
            stage_error = f"{type(e).__name__}: {e}"
            logger.exception(f"[pipeline-worker] video_id={vid} stage={sid} failed: {stage_error}")
            try:
                update_run_summary(
                    os.path.join(video_out, "run_summary.json"),
                    {
                        "source_video": build_source_video_meta(vp),
                        "output_root": os.path.abspath(output_root),
                        "schema_fingerprint": schema_fp,
                        "updated_at_utc": now_utc_iso(),
                        f"stage{sid}": {
                            "status": "failed",
                            "failed_at_utc": now_utc_iso(),
                            "error": stage_error,
                        },
                    },
                )
            except Exception:
                pass
            break

    video_ok = not stage_failed
    if video_ok and post_validate and "4" in set(stages):
        try:
            from validate_three_stage_output import validate_three_stage_video_output_dir

            v0 = time.perf_counter()
            logger.info(f"[pipeline-worker] video_id={vid} post-validate start")
            ok, errors, warnings = validate_three_stage_video_output_dir(video_out, check_deps=False)
            for w in warnings:
                logger.warning(f"[validate] video_id={vid}: {w}")
            ok_effective = bool(ok) and not (bool(post_validate_fail_on_warnings) and bool(warnings))
            if not ok_effective:
                video_ok = False
                schema_mismatch = {"errors": errors, "warnings": warnings, "note": "post_validate_schema_mismatch"}
                sample = (errors or [])[:20] or (warnings or [])[:20] or ["post_validate_failed"]
                raise RuntimeError(" | ".join(sample))
            logger.info(
                f"[pipeline-worker] video_id={vid} post-validate OK in {format_duration(time.perf_counter() - v0)} "
                f"(warnings={len(warnings)})"
            )
            try:
                update_run_summary(
                    os.path.join(video_out, "run_summary.json"),
                    {
                        "source_video": build_source_video_meta(vp),
                        "output_root": os.path.abspath(output_root),
                        "schema_fingerprint": schema_fp,
                        "updated_at_utc": now_utc_iso(),
                        "post_validate": {
                            "status": "completed",
                            "validated_at_utc": now_utc_iso(),
                            "warnings_count": len(warnings),
                            "warnings_sample": warnings[:10],
                        },
                    },
                )
            except Exception:
                pass
        except Exception as e:
            video_ok = False
            msg = f"{type(e).__name__}: {e}"
            logger.exception(f"[pipeline-worker] post-validate failed for video_id={vid}: {msg}")
            if schema_mismatch is None:
                schema_mismatch = {"errors": [msg], "warnings": [], "note": "post_validate_exception"}
            try:
                update_run_summary(
                    os.path.join(video_out, "run_summary.json"),
                    {
                        "source_video": build_source_video_meta(vp),
                        "output_root": os.path.abspath(output_root),
                        "schema_fingerprint": schema_fp,
                        "updated_at_utc": now_utc_iso(),
                        "post_validate": {
                            "status": "failed",
                            "failed_at_utc": now_utc_iso(),
                            "error": msg,
                        },
                    },
                )
            except Exception:
                pass

    elapsed_sec = time.perf_counter() - t0
    logger.info(
        f"[pipeline-worker] video_id={vid} done: ok={int(bool(video_ok))} elapsed={format_duration(elapsed_sec)}"
    )

    # Print cost summary after each video
    try:
        rs_path = os.path.join(video_out, "run_summary.json")
        if os.path.exists(rs_path):
            print_cost_summary(rs_path)
    except Exception:
        pass
    return {
        "source_video": build_source_video_meta(vp),
        "video_path": vp,
        "video_out": video_out,
        "ok": bool(video_ok),
        "stage_failed": bool(stage_failed),
        "failed_stage": failed_stage,
        "stage_error": stage_error,
        "schema_mismatch": schema_mismatch,
        "elapsed_sec": float(elapsed_sec),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline: full-video VLM hand-event discovery -> text-only hand_step fusion -> "
            "ROI-level semantic/visibility annotation. Stage 2 does not materialize clips."
        )
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-video", help="Path to one video file.")
    src.add_argument("--input-video-dir", help="Directory of videos to process.")
    parser.add_argument(
        "--input-min-age-sec",
        type=float,
        default=0.0,
        help="Skip input videos modified more recently than this many seconds.",
    )
    parser.add_argument(
        "--input-stability-check-sec",
        type=float,
        default=0.0,
        help=(
            "For directory batches, stat candidate videos twice this many seconds apart and "
            "skip files whose size or mtime changes."
        ),
    )
    parser.add_argument("--output-root", default=default_output_root(), help="Output root under ECCV/four_stage/...")
    parser.add_argument(
        "--output-per-video-sibling",
        action="store_true",
        help=(
            "For each input under <recording>/raw_data/, write to "
            "<recording>/processed_text_label/ instead of the shared --output-root. "
            "Use this for recursive batches of same-named raw_data/rgb.mp4 videos."
        ),
    )

    add_api_cli_args(parser, include_no_embed_index=True)
    add_sampling_cli_args(parser, default_max_frames=100, default_jpeg_quality=95)
    add_bbox_cli_args(parser, include_bbox_python=True)

    parser.add_argument("--ffmpeg-bin", default="ffmpeg", help="Legacy compatibility option; unused by text-only Stage 2.")
    parser.add_argument("--cut-mode", choices=["copy", "reencode"], default="reencode", help="Legacy compatibility option; Stage 2 never cuts video.")
    parser.add_argument("--seek-slop-sec", type=float, default=1.0)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--keep-audio", action="store_true")
    parser.add_argument("--stage1-retries", type=int, default=3)
    parser.add_argument("--stage2-retries", type=int, default=3)
    parser.add_argument("--stage3-retries", type=int, default=3)
    parser.add_argument("--stage4-retries", type=int, default=3)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help=(
            "Parallelize across videos with this many worker processes (ProcessPoolExecutor). "
            "Default=4."
        ),
    )

    parser.add_argument(
        "--stages",
        default="1,2,3,4",
        help=(
            "Comma-separated subset of stages to run. Default: 1,2,3,4 (no GPU required). "
            "Stage 4 decomposes steps into atomic actions. Stage 5 must be requested explicitly."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-legacy-resume",
        action="store_true",
        help="Allow resuming cached outputs whose run_summary.json lacks schema_fingerprint (legacy outputs).",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run dependency/collision checks and exit (does not call the model, does not cut clips).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing the next video if one video fails; writes failure info into <video_out>/run_summary.json.",
    )
    parser.add_argument(
        "--post-validate",
        action="store_true",
        help="After successful Stage 4, run `validate_three_stage_output.py` checks on the output folder.",
    )
    parser.add_argument(
        "--post-validate-fail-on-warnings",
        action="store_true",
        help="When --post-validate is enabled, treat validator warnings as failures (higher precision, more rejects).",
    )
    parser.add_argument(
        "--schema-mismatch-txt",
        default="",
        help=(
            "Optional: write post-validate schema-mismatch records to this .txt file (default: "
            "<output_root>/schema_mismatch_videos.txt when --post-validate is enabled)."
        ),
    )
    parser.add_argument(
        "--retry-failed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Default: enabled. When enabled, only process videos that are not yet complete for the requested stages "
            "(including previously-failed/incomplete videos), based on <output_root>/<video_id>/run_summary.json. "
            "Use --no-retry-failed to process all videos in the input list."
        ),
    )
    parser.add_argument(
        "--failed-report-json",
        default="",
        help=(
            "Optional JSON failure report. With --retry-failed, an empty value defaults to "
            "<output_root>/failed_videos.json. Pass '-' to disable report/list creation."
        ),
    )
    args = parser.parse_args()

    stages = {s.strip() for s in args.stages.split(",") if s.strip()}
    if not stages.issubset({"1", "2", "3", "4", "5"}):
        raise SystemExit(f"Invalid --stages: {args.stages}")

    api_cfg = api_config_from_args(args)
    sampling_cfg = sampling_config_from_args(args)
    bbox_cfg = bbox_config_from_args(args)
    ensure_frame_index_fps(sampling_cfg, getattr(api_cfg, "video_fps", 10.0))

    videos: List[str] = []
    if args.input_video:
        videos = [args.input_video]
    else:
        videos = collect_videos(args.input_video_dir, VIDEO_EXTS)
    if args.output_per_video_sibling:
        # This mode is purpose-built for raw recordings. Recursive discovery
        # must not feed the MP4 step clips and previews written below existing
        # processed_text_label directories back into the pipeline.
        videos = [
            vp
            for vp in videos
            if os.path.basename(vp) == "rgb.mp4" and os.path.basename(os.path.dirname(vp)) == "raw_data"
        ]
    if not videos:
        raise SystemExit("No videos found.")

    # Recurring ingestion must not open a file while an upstream recorder or
    # copier is still writing it. Age filtering catches newly-created files;
    # the optional second stat catches older files that are actively changing.
    min_age_sec = max(0.0, float(getattr(args, "input_min_age_sec", 0.0) or 0.0))
    stability_check_sec = max(
        0.0, float(getattr(args, "input_stability_check_sec", 0.0) or 0.0)
    )
    skipped_young = 0
    skipped_changing = 0
    if min_age_sec > 0:
        cutoff = time.time() - min_age_sec
        old_enough: List[str] = []
        for vp in videos:
            try:
                if os.path.getmtime(vp) <= cutoff:
                    old_enough.append(vp)
                else:
                    skipped_young += 1
            except OSError:
                skipped_changing += 1
        videos = old_enough

    if stability_check_sec > 0 and videos:
        before: Dict[str, Tuple[int, int]] = {}
        for vp in videos:
            try:
                st = os.stat(vp)
                before[vp] = (int(st.st_size), int(st.st_mtime_ns))
            except OSError:
                skipped_changing += 1
        logger.info(
            f"[pipeline] Waiting {stability_check_sec:g}s to verify that "
            f"{len(before)} input video(s) are no longer changing."
        )
        time.sleep(stability_check_sec)
        stable: List[str] = []
        for vp, first in before.items():
            try:
                st = os.stat(vp)
                second = (int(st.st_size), int(st.st_mtime_ns))
            except OSError:
                skipped_changing += 1
                continue
            if second == first and second[0] > 0:
                stable.append(vp)
            else:
                skipped_changing += 1
        videos = stable

    if skipped_young or skipped_changing:
        logger.info(
            "[pipeline] Input safety filter: "
            f"skipped_recent={skipped_young} skipped_changing_or_missing={skipped_changing}"
        )
    if not videos:
        logger.info("[pipeline] No stable, fully-written input videos are ready in this scan.")
        raise SystemExit(0)

    if args.input_video:
        logger.info(f"[pipeline] Discovered 1 video from --input-video: {os.path.abspath(args.input_video)}")
    else:
        logger.info(
            f"[pipeline] Discovered {len(videos)} videos under --input-video-dir: {os.path.abspath(args.input_video_dir)}"
        )

    def _video_output_root(vp: str) -> str:
        return output_root_for_video(
            vp,
            args.output_root,
            per_video_sibling=bool(args.output_per_video_sibling),
        )

    failed_report_json = str(getattr(args, "failed_report_json", "") or "").strip()
    failed_report_disabled = failed_report_json == "-"
    if failed_report_disabled:
        failed_report_json = ""
    if bool(getattr(args, "retry_failed", True)) and not failed_report_json and not failed_report_disabled:
        failed_report_json = os.path.join(args.output_root, "failed_videos.json")
    prev_failed_records: List[Dict[str, Any]] = []
    prev_incomplete_records: List[Dict[str, Any]] = []

    schema_fp = three_stage_schema_fingerprint()

    missing_inputs = [vp for vp in videos if not os.path.isfile(vp)]
    if missing_inputs:
        raise SystemExit("Missing/non-file inputs:\n" + "\n".join(f"- {p}" for p in missing_inputs))

    # A shared output root uses `video_id_from_path()` for `<video_out>/` and
    # cannot safely accept duplicate filename stems. Sibling roots are scoped
    # by source directory, so repeated raw_data/rgb.mp4 names are safe.
    vid_to_paths: dict[str, List[str]] = {}
    for vp in videos:
        vid_to_paths.setdefault(video_id_from_path(vp), []).append(vp)
    dup = {vid: ps for vid, ps in vid_to_paths.items() if len(ps) > 1}
    if dup and not args.output_per_video_sibling:
        lines: List[str] = ["Duplicate video_id detected (filename stem collision):"]
        for vid, ps in sorted(dup.items()):
            lines.append(f"- video_id={vid}:")
            for p in ps:
                lines.append(f"  - {p}")
        lines.append(
            "Rename colliding videos, use different --output-root values, or pass "
            "--output-per-video-sibling to avoid corrupt outputs."
        )
        raise SystemExit("\n".join(lines))

    if args.preflight_only:
        errs: List[str] = []
        try:
            import cv2  # noqa: F401
        except Exception as e:
            errs.append(f"Missing dependency: opencv-python (cv2). Install: pip install opencv-python. Detail: {e}")
        try:
            import requests  # noqa: F401
        except Exception as e:
            errs.append(f"Missing dependency: requests. Install: pip install requests. Detail: {e}")

        if "2" in stages:
            if shutil.which(args.ffmpeg_bin) is None and not os.path.exists(args.ffmpeg_bin):
                errs.append(
                    f"ffmpeg binary not found: '{args.ffmpeg_bin}'. Install ffmpeg or pass a valid path via --ffmpeg-bin."
                )
        if "5" in stages:
            bbox_python = os.path.abspath(os.path.expanduser(str(args.bbox_python)))
            if not os.path.isfile(bbox_python):
                errs.append(
                    f"Stage 5 Python interpreter not found: {bbox_python}. "
                    "Create .venv_bbox_tracking or pass --bbox-python."
                )
            detector_path = os.path.abspath(os.path.expanduser(str(bbox_cfg.hand_detector)))
            if not os.path.isfile(detector_path):
                errs.append(f"Stage 5 hand detector not found: {detector_path}")
            if bbox_cfg.backend == "sam3":
                sam3_python = os.path.abspath(os.path.expanduser(str(bbox_cfg.sam3_python)))
                sam3_repo = os.path.abspath(os.path.expanduser(str(bbox_cfg.sam3_repo)))
                sam3_checkpoint = os.path.abspath(
                    os.path.expanduser(str(bbox_cfg.sam3_checkpoint))
                )
                if not os.path.isfile(sam3_python):
                    errs.append(f"Stage 5 SAM3 Python not found: {sam3_python}")
                if not os.path.isfile(os.path.join(sam3_repo, "sam3", "model_builder.py")):
                    errs.append(f"Stage 5 SAM3 repository is incomplete: {sam3_repo}")
                if not os.path.isfile(sam3_checkpoint):
                    errs.append(f"Stage 5 SAM3 checkpoint not found: {sam3_checkpoint}")
        for vp in videos:
            output_root = _video_output_root(vp)
            vid = video_id_from_path(vp)
            video_out = os.path.join(output_root, vid)
            try:
                os.makedirs(output_root, exist_ok=True)
            except Exception as e:
                errs.append(f"Failed to create output root: {output_root}: {e}")
                continue
            try:
                ensure_video_out_dir_safe(video_out, vp)
            except OutputDirCollisionError as e:
                errs.append(str(e).strip())

        if errs:
            for e in errs:
                logger.error("[preflight] " + e.replace("\n", " | "))
            raise SystemExit(1)
        logger.info(f"[preflight] OK (schema_fingerprint={schema_fp})")
        raise SystemExit(0)

    total_input_videos = len(videos)

    # Default behavior: skip already-complete videos and only process videos that still need work.
    # Disable with --no-retry-failed (useful together with --overwrite when you want to force a full regen).
    if bool(getattr(args, "retry_failed", True)):
        selected: List[str] = []
        unprocessed = 0
        needs_work = 0
        complete = 0
        for vp in videos:
            output_root = _video_output_root(vp)
            vid = video_id_from_path(vp)
            video_out = os.path.join(output_root, vid)

            # New/unprocessed video: include it.
            if not os.path.exists(video_out):
                selected.append(vp)
                unprocessed += 1
                continue

            # Completion is determined by the existence of expected output files (especially Stage-4 final JSON),
            # not by run_summary.json `source_video` bookkeeping.
            missing = _missing_expected_outputs(video_out, tuple(sorted(stages)))
            if not missing:
                complete += 1
                _maybe_update_run_summary_source_video(video_out, source_video=vp)
                continue

            needs_work += 1
            selected.append(vp)
            prev_incomplete_records.append(
                {
                    "source_video": build_source_video_meta(vp),
                    "video_out": os.path.abspath(video_out),
                    "run_summary_path": os.path.abspath(os.path.join(video_out, "run_summary.json")),
                    "issues": [
                        {
                            "key": "expected_outputs",
                            "stage": "outputs",
                            "status": "missing",
                            "error": "Missing expected outputs: "
                            + ", ".join(os.path.abspath(p) for p in missing[:10])
                            + ("" if len(missing) <= 10 else f", ... (+{len(missing) - 10})"),
                        }
                    ],
                }
            )
        if prev_failed_records or prev_incomplete_records or unprocessed:
            sample = ", ".join(
                [source_video_id(r.get("source_video")) for r in prev_failed_records[:20] if source_video_id(r.get("source_video"))]
            )
            more = "" if len(prev_failed_records) <= 20 else f", ... (+{len(prev_failed_records) - 20})"
            logger.info(
                "[pipeline] retry-failed enabled: "
                f"selected {len(selected)}/{len(videos)} videos to run "
                f"(unprocessed={unprocessed}, needs_work={needs_work}, complete={complete}, prev_failed={len(prev_failed_records)}). "
                f"failed_sample: {sample}{more}"
            )
        else:
            logger.info("[pipeline] retry-failed enabled: everything is complete; nothing to do.")

        videos = selected

        if failed_report_json:
            write_json(
                failed_report_json,
                {
                    "output_root": os.path.abspath(args.output_root),
                    "retry_failed": True,
                    "stages": sorted(stages),
                    "total_input_videos": int(total_input_videos),
                    "selected_videos": int(len(videos)),
                    "previous_failed": prev_failed_records,
                    "previous_incomplete": prev_incomplete_records,
                    "this_run_failed": [],
                },
            )
            logger.info(f"[pipeline] Wrote failed report: {os.path.abspath(failed_report_json)}")

        if not videos:
            raise SystemExit(0)

    total = len(videos)
    run_started = time.perf_counter()
    ok_videos = 0
    failed_videos = 0

    schema_mismatch_txt = str(getattr(args, "schema_mismatch_txt", "") or "").strip()
    if args.post_validate and not schema_mismatch_txt:
        schema_mismatch_txt = os.path.join(args.output_root, "schema_mismatch_videos.txt")

    # Parallel mode (multi-video only).
    num_workers = int(getattr(args, "num_workers", 1) or 1)
    this_run_failed: List[Dict[str, Any]] = []
    if num_workers > 1 and not args.input_video:
        max_workers = min(max(1, num_workers), len(videos))
        logger.info(
            "[pipeline] Start: "
            f"videos={total} stages={','.join(sorted(stages))} "
            f"output_root={os.path.abspath(args.output_root)} "
            f"overwrite={bool(args.overwrite)} continue_on_error={bool(args.continue_on_error)} post_validate={bool(args.post_validate)}"
        )
        logger.info(
            "[pipeline] API: "
            f"base={api_cfg.api_base_url} provider={api_cfg.model_provider_id} model={api_cfg.model_name} "
            f"max_tokens={int(api_cfg.max_tokens)} temp={float(api_cfg.temperature)} "
            f"call_retries={int(api_cfg.api_call_retries)} backoff_sec={float(api_cfg.api_call_retry_backoff_sec)}"
        )
        logger.info(
            "[pipeline] Sampling: "
            f"frame_index_fps={float(getattr(sampling_cfg, 'frame_index_fps', 0.0) or 0.0)} "
            f"fallback_max_frames={int(sampling_cfg.max_frames)} jpeg_quality={int(sampling_cfg.jpeg_quality)} "
            f"embed_index_on_api_images={bool(api_cfg.embed_index_on_api_images)}"
        )
        logger.info(f"[pipeline] Parallel mode enabled: workers={max_workers} videos={total}")
        processed = 0
        stop_submitting = False

        def _submit(ex: ProcessPoolExecutor, vp0: str) -> Any:
            return ex.submit(
                _run_one_video_pipeline,
                video_path=vp0,
                output_root=_video_output_root(vp0),
                stages=tuple(sorted(stages)),
                api_cfg=api_cfg,
                sampling_cfg=sampling_cfg,
                overwrite=bool(args.overwrite),
                allow_legacy_resume=bool(args.allow_legacy_resume),
                stage1_retries=int(args.stage1_retries),
                stage2_retries=int(args.stage2_retries),
                stage3_retries=int(args.stage3_retries),
                stage4_retries=int(args.stage4_retries),
                bbox_python=str(args.bbox_python),
                bbox_cfg=bbox_cfg,
                ffmpeg_bin=str(args.ffmpeg_bin),
                cut_mode=str(args.cut_mode),
                seek_slop_sec=float(args.seek_slop_sec),
                crf=int(args.crf),
                preset=str(args.preset),
                keep_audio=bool(args.keep_audio),
                post_validate=bool(args.post_validate),
                post_validate_fail_on_warnings=bool(args.post_validate_fail_on_warnings),
            )

        remaining = iter(videos)
        futures: Dict[Any, str] = {}
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            for _ in range(max_workers):
                try:
                    vp0 = next(remaining)
                except StopIteration:
                    break
                futures[_submit(ex, vp0)] = vp0

            while futures:
                done, _pending = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
                for fut in done:
                    vp_done = futures.pop(fut, "")
                    try:
                        res = fut.result()
                    except CancelledError:
                        continue
                    except Exception as e:
                        processed += 1
                        failed_videos += 1
                        msg = f"{type(e).__name__}: {e}"
                        this_run_failed.append(
                            {
                                "source_video": build_source_video_meta(vp_done) if vp_done else {},
                                "video_out": "",
                                "failed_stage": "worker_crash",
                                "error": msg,
                            }
                        )
                        logger.exception(f"[pipeline] Worker crashed for {os.path.basename(vp_done)}: {msg}")
                        if not args.continue_on_error:
                            stop_submitting = True
                    else:
                        processed += 1
                        vid = source_video_id(res.get("source_video")) or video_id_from_path(vp_done)
                        ok = bool(res.get("ok"))
                        if ok:
                            ok_videos += 1
                        else:
                            failed_videos += 1
                            if not args.continue_on_error:
                                stop_submitting = True
                            failed_stage = str(res.get("failed_stage") or "").strip() or ""
                            stage_error = str(res.get("stage_error") or "").strip() or ""
                            mismatch = res.get("schema_mismatch")
                            if not failed_stage and mismatch:
                                failed_stage = "post_validate"
                            if not stage_error and isinstance(mismatch, dict):
                                sample = (mismatch.get("errors") or [])[:1] or (mismatch.get("warnings") or [])[:1]
                                stage_error = str(sample[0]) if sample else str(mismatch.get("note") or "post_validate_failed")
                            this_run_failed.append(
                                {
                                    "source_video": build_source_video_meta(vp_done) if vp_done else {},
                                    "video_out": os.path.abspath(str(res.get("video_out") or os.path.join(_video_output_root(vp_done), vid))),
                                    "failed_stage": failed_stage or "unknown",
                                    "error": stage_error or "failed",
                                }
                            )
                        mismatch = res.get("schema_mismatch")
                        if args.post_validate and mismatch:
                            _append_schema_mismatch_txt(
                                schema_mismatch_txt,
                                video_id=vid,
                                source_video=vp_done,
                                video_out=str(res.get("video_out") or os.path.join(_video_output_root(vp_done), vid)),
                                errors=list(mismatch.get("errors") or []),
                                warnings=list(mismatch.get("warnings") or []),
                                note=str(mismatch.get("note") or ""),
                            )

                    elapsed = time.perf_counter() - run_started
                    eta = (elapsed / processed) * (total - processed) if processed > 0 else 0.0
                    logger.info(
                        "[pipeline] Progress: "
                        f"{processed}/{total} processed (ok={ok_videos}, failed={failed_videos}) "
                        f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}"
                    )

                    if stop_submitting:
                        # Best-effort: cancel tasks that haven't started yet.
                        for pending_fut in list(futures.keys()):
                            pending_fut.cancel()
                        continue

                    try:
                        vp_next = next(remaining)
                    except StopIteration:
                        vp_next = ""
                    if vp_next:
                        futures[_submit(ex, vp_next)] = vp_next

        total_elapsed = time.perf_counter() - run_started
        logger.info(
            "[pipeline] Done: "
            f"processed={total} ok={ok_videos} failed={failed_videos} elapsed={format_duration(total_elapsed)}"
        )
        if failed_report_json:
            write_json(
                failed_report_json,
                {
                    "output_root": os.path.abspath(args.output_root),
                    "retry_failed": bool(getattr(args, "retry_failed", False)),
                    "stages": sorted(stages),
                    "total_input_videos": int(total_input_videos),
                    "selected_videos": int(total),
                    "previous_failed": prev_failed_records,
                    "previous_incomplete": prev_incomplete_records,
                    "this_run_failed": this_run_failed,
                },
            )
            logger.info(f"[pipeline] Wrote failed report: {os.path.abspath(failed_report_json)}")
        if not args.continue_on_error and failed_videos:
            raise SystemExit(1)
        return

    logger.info(
        "[pipeline] Start: "
        f"videos={total} stages={','.join(sorted(stages))} "
        f"output_root={os.path.abspath(args.output_root)} "
        f"overwrite={bool(args.overwrite)} continue_on_error={bool(args.continue_on_error)} post_validate={bool(args.post_validate)}"
    )
    logger.info(
        "[pipeline] API: "
        f"base={api_cfg.api_base_url} provider={api_cfg.model_provider_id} model={api_cfg.model_name} "
        f"max_tokens={int(api_cfg.max_tokens)} temp={float(api_cfg.temperature)} "
        f"call_retries={int(api_cfg.api_call_retries)} backoff_sec={float(api_cfg.api_call_retry_backoff_sec)}"
    )
    logger.info(
        "[pipeline] Sampling: "
        f"frame_index_fps={float(getattr(sampling_cfg, 'frame_index_fps', 0.0) or 0.0)} "
        f"fallback_max_frames={int(sampling_cfg.max_frames)} jpeg_quality={int(sampling_cfg.jpeg_quality)} "
        f"embed_index_on_api_images={bool(api_cfg.embed_index_on_api_images)}"
    )

    for idx, vp in enumerate(videos, start=1):
        output_root = _video_output_root(vp)
        vid = video_id_from_path(vp)
        video_out = os.path.join(output_root, vid)

        logger.info(f"[pipeline] ({idx}/{total}) video_id={vid} start: {os.path.abspath(vp)}")

        stage_failed = False
        for sid in ("1", "2", "3", "4", "5"):
            if sid not in stages:
                continue
            try:
                t0 = time.perf_counter()
                logger.info(f"[pipeline] ({idx}/{total}) video_id={vid} stage={sid} start")
                if sid == "1":
                    run_stage1_for_video(
                        vp,
                        output_root,
                        api_cfg,
                        sampling_cfg,
                        overwrite=args.overwrite,
                        max_retries=args.stage1_retries,
                        allow_legacy_resume=args.allow_legacy_resume,
                    )
                elif sid == "2":
                    run_stage2_for_video(
                        vp,
                        output_root,
                        api_cfg,
                        ffmpeg_bin=args.ffmpeg_bin,
                        overwrite=args.overwrite,
                        max_retries=args.stage2_retries,
                        cut_mode=args.cut_mode,
                        seek_slop_sec=args.seek_slop_sec,
                        crf=args.crf,
                        preset=args.preset,
                        keep_audio=args.keep_audio,
                        allow_legacy_resume=args.allow_legacy_resume,
                    )
                elif sid == "3":
                    run_stage3_for_video(
                        vp,
                        output_root,
                        api_cfg,
                        sampling_cfg,
                        overwrite=args.overwrite,
                        max_retries=args.stage3_retries,
                        allow_legacy_resume=args.allow_legacy_resume,
                    )
                elif sid == "4":
                    run_stage4_for_video(
                        vp,
                        output_root,
                        api_cfg,
                        sampling_cfg,
                        overwrite=args.overwrite,
                        max_retries=args.stage4_retries,
                        ffmpeg_bin=args.ffmpeg_bin,
                        allow_legacy_resume=args.allow_legacy_resume,
                    )
                elif sid == "5":
                    run_stage5_subprocess(
                        bbox_python=str(args.bbox_python),
                        video_path=vp,
                        output_root=output_root,
                        api_cfg=api_cfg,
                        bbox_cfg=bbox_cfg,
                        overwrite=bool(args.overwrite),
                    )
                dt = time.perf_counter() - t0
                logger.info(f"[pipeline] ({idx}/{total}) video_id={vid} stage={sid} done in {format_duration(dt)}")
            except OutputDirCollisionError as e:
                stage_failed = True
                msg = f"{type(e).__name__}: {e}"
                this_run_failed.append(
                    {
                        "source_video": build_source_video_meta(vp),
                        "video_out": os.path.abspath(video_out),
                        "failed_stage": sid,
                        "error": msg,
                    }
                )
                logger.error(f"[pipeline] video_id={vid} stage={sid} aborted: {msg}")
                # Do NOT write into run_summary.json here: the directory may belong to another source video.
                if args.continue_on_error:
                    break
                raise
            except (Exception, SystemExit) as e:
                stage_failed = True
                msg = f"{type(e).__name__}: {e}"
                logger.exception(f"[pipeline] video_id={vid} stage={sid} failed: {msg}")
                try:
                    update_run_summary(
                        os.path.join(video_out, "run_summary.json"),
                        {
                            "source_video": build_source_video_meta(vp),
                            "output_root": os.path.abspath(output_root),
                            "schema_fingerprint": schema_fp,
                            "updated_at_utc": now_utc_iso(),
                            f"stage{sid}": {
                                "status": "failed",
                                "failed_at_utc": now_utc_iso(),
                                "error": msg,
                            },
                        },
                    )
                except Exception:
                    pass
                this_run_failed.append(
                    {
                        "source_video": build_source_video_meta(vp),
                        "video_out": os.path.abspath(video_out),
                        "failed_stage": sid,
                        "error": msg,
                    }
                )
                if args.continue_on_error:
                    break
                raise

        if stage_failed:
            failed_videos += 1
            elapsed = time.perf_counter() - run_started
            processed = ok_videos + failed_videos
            eta = (elapsed / processed) * (total - processed) if processed > 0 else 0.0
            logger.info(
                "[pipeline] Progress: "
                f"{processed}/{total} processed (ok={ok_videos}, failed={failed_videos}) "
                f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}"
            )
            continue

        video_ok = True
        if args.post_validate and "4" in stages:
            logged_to_txt = False
            try:
                from validate_three_stage_output import validate_three_stage_video_output_dir

                v0 = time.perf_counter()
                logger.info(f"[pipeline] ({idx}/{total}) video_id={vid} post-validate start")
                ok, errors, warnings = validate_three_stage_video_output_dir(video_out, check_deps=False)
                for w in warnings:
                    logger.warning(f"[validate] video_id={vid}: {w}")
                ok_effective = bool(ok) and not (bool(args.post_validate_fail_on_warnings) and bool(warnings))
                if not ok_effective:
                    _append_schema_mismatch_txt(
                        schema_mismatch_txt,
                        video_id=vid,
                        source_video=vp,
                        video_out=video_out,
                        errors=errors,
                        warnings=warnings,
                        note="post_validate_schema_mismatch",
                    )
                    logged_to_txt = True
                    sample = (errors or [])[:20] or (warnings or [])[:20] or ["post_validate_failed"]
                    raise RuntimeError(" | ".join(sample))
                logger.info(
                    f"[pipeline] ({idx}/{total}) video_id={vid} post-validate OK in {format_duration(time.perf_counter() - v0)} "
                    f"(warnings={len(warnings)})"
                )
                try:
                    update_run_summary(
                        os.path.join(video_out, "run_summary.json"),
                        {
                            "source_video": build_source_video_meta(vp),
                            "output_root": os.path.abspath(output_root),
                            "schema_fingerprint": schema_fp,
                            "updated_at_utc": now_utc_iso(),
                            "post_validate": {
                                "status": "completed",
                                "validated_at_utc": now_utc_iso(),
                                "warnings_count": len(warnings),
                                "warnings_sample": warnings[:10],
                            },
                        },
                    )
                except Exception:
                    pass
            except Exception as e:
                video_ok = False
                msg = f"{type(e).__name__}: {e}"
                logger.exception(f"[pipeline] post-validate failed for video_id={vid}: {msg}")
                if not logged_to_txt:
                    _append_schema_mismatch_txt(
                        schema_mismatch_txt,
                        video_id=vid,
                        source_video=vp,
                        video_out=video_out,
                        errors=[msg],
                        warnings=[],
                        note="post_validate_exception",
                    )
                try:
                    update_run_summary(
                        os.path.join(video_out, "run_summary.json"),
                        {
                            "source_video": build_source_video_meta(vp),
                            "output_root": os.path.abspath(output_root),
                            "schema_fingerprint": schema_fp,
                            "updated_at_utc": now_utc_iso(),
                            "post_validate": {
                                "status": "failed",
                                "failed_at_utc": now_utc_iso(),
                                "error": msg,
                            },
                        },
                    )
                except Exception:
                    pass
                this_run_failed.append(
                    {
                        "source_video": build_source_video_meta(vp),
                        "video_out": os.path.abspath(video_out),
                        "failed_stage": "post_validate",
                        "error": msg,
                    }
                )
                if not args.continue_on_error:
                    raise

        if video_ok:
            ok_videos += 1
        else:
            failed_videos += 1

        # Print cost summary for this video
        try:
            rs_path_v = os.path.join(video_out, "run_summary.json")
            if os.path.exists(rs_path_v):
                print_cost_summary(rs_path_v)
        except Exception:
            pass
        elapsed = time.perf_counter() - run_started
        processed = ok_videos + failed_videos
        eta = (elapsed / processed) * (total - processed) if processed > 0 else 0.0
        logger.info(
            "[pipeline] Progress: "
            f"{processed}/{total} processed (ok={ok_videos}, failed={failed_videos}) "
            f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}"
        )

    total_elapsed = time.perf_counter() - run_started
    logger.info(
        "[pipeline] Done: "
        f"processed={total} ok={ok_videos} failed={failed_videos} elapsed={format_duration(total_elapsed)}"
    )
    if failed_report_json:
        write_json(
            failed_report_json,
            {
                "output_root": os.path.abspath(args.output_root),
                "retry_failed": bool(getattr(args, "retry_failed", False)),
                "stages": sorted(stages),
                "total_input_videos": int(total_input_videos),
                "selected_videos": int(total),
                "previous_failed": prev_failed_records,
                "previous_incomplete": prev_incomplete_records,
                "this_run_failed": this_run_failed,
            },
        )
        logger.info(f"[pipeline] Wrote failed report: {os.path.abspath(failed_report_json)}")


if __name__ == "__main__":
    main()
