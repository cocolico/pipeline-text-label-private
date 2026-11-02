#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


STAGE5_SCHEMA = "stage5-bbox-v9-bounded-visibility-evidence"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_file(path: Path, errors: list[str], label: str) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        errors.append(f"missing {label}: {path}")
        return False
    return True


def check_plan(video_out: Path, frame_count: int, errors: list[str]) -> dict[str, int]:
    plan_path = video_out / "final_plan.json"
    if not require_file(plan_path, errors, "final_plan.json"):
        return {"steps": 0, "atomic_actions": 0}
    try:
        plan = load_json(plan_path)
    except Exception as exc:
        errors.append(f"invalid final_plan.json: {exc}")
        return {"steps": 0, "atomic_actions": 0}

    steps = plan.get("steps") if isinstance(plan, dict) else None
    if not isinstance(steps, list) or not steps:
        errors.append("final_plan.json has no non-empty steps array")
        return {"steps": 0, "atomic_actions": 0}

    atomic_count = 0
    previous_step_end = 0
    for expected_step_id, step in enumerate(steps, 1):
        if not isinstance(step, dict):
            errors.append(f"step {expected_step_id} is not an object")
            continue
        if step.get("step_id") != expected_step_id:
            errors.append(f"non-sequential step_id at position {expected_step_id}")
        start = step.get("start_frame_0based")
        end = step.get("end_frame_0based_exclusive")
        if not isinstance(start, int) or not isinstance(end, int) or not (0 <= start < end <= frame_count):
            errors.append(f"invalid step frame bounds for step {expected_step_id}: {start}, {end}")
            continue
        if start < previous_step_end:
            errors.append(f"overlapping/non-monotonic step {expected_step_id}")
        previous_step_end = end
        if not isinstance(step.get("hand_object_relations"), list):
            errors.append(f"step {expected_step_id} lacks hand_object_relations array")
        clip = step.get("clip_relpath")
        if not isinstance(clip, str) or not (video_out / clip).is_file():
            errors.append(f"missing step clip for step {expected_step_id}: {clip}")

        actions = step.get("atomic_actions")
        if not isinstance(actions, list) or not actions:
            errors.append(f"step {expected_step_id} has no atomic_actions")
            continue
        atomic_count += len(actions)
        previous_atomic_end = start
        for expected_action_id, action in enumerate(actions, 1):
            if not isinstance(action, dict):
                errors.append(f"step {expected_step_id} atomic {expected_action_id} is not an object")
                continue
            if action.get("atomic_action_id") != expected_action_id:
                errors.append(f"non-sequential atomic_action_id in step {expected_step_id}")
            a_start = action.get("start_frame_0based")
            a_end = action.get("end_frame_0based_exclusive")
            if not isinstance(a_start, int) or not isinstance(a_end, int) or not (start <= a_start < a_end <= end):
                errors.append(
                    f"invalid atomic bounds step {expected_step_id} action {expected_action_id}: {a_start}, {a_end}"
                )
                continue
            if a_start < previous_atomic_end:
                errors.append(f"overlapping/non-monotonic atomic action in step {expected_step_id}")
            previous_atomic_end = a_end
            atomic_clip = action.get("clip_relpath")
            if not isinstance(atomic_clip, str) or not (video_out / atomic_clip).is_file():
                errors.append(
                    f"missing atomic clip step {expected_step_id} action {expected_action_id}: {atomic_clip}"
                )
    return {"steps": len(steps), "atomic_actions": atomic_count}


def check_stage5(video_out: Path, frame_count: int, errors: list[str]) -> dict[str, Any]:
    stage5 = video_out / "stage5"
    summary_path = stage5 / "stage5_summary.json"
    qa_path = stage5 / "qa_summary.json"
    rows_path = stage5 / "frame_bboxes.jsonl"
    overlay_path = stage5 / "bbox_overlay.mp4"
    for path, label in (
        (summary_path, "stage5_summary.json"),
        (qa_path, "qa_summary.json"),
        (rows_path, "frame_bboxes.jsonl"),
        (overlay_path, "bbox_overlay.mp4"),
    ):
        require_file(path, errors, label)
    if errors:
        return {"quality_status": "missing", "eligible": False, "frame_rows": 0}

    try:
        summary = load_json(summary_path)
        qa = load_json(qa_path)
    except Exception as exc:
        errors.append(f"invalid Stage 5 JSON: {exc}")
        return {"quality_status": "invalid", "eligible": False, "frame_rows": 0}

    if summary.get("schema_version") != STAGE5_SCHEMA:
        errors.append(f"wrong Stage 5 schema: {summary.get('schema_version')}")
    if summary.get("total_frames") != frame_count:
        errors.append(f"Stage 5 total_frames mismatch: {summary.get('total_frames')} != {frame_count}")
    gate = qa.get("quality_gate") if isinstance(qa, dict) else None
    status = gate.get("status") if isinstance(gate, dict) else None
    eligible = gate.get("eligible_for_automatic_use") if isinstance(gate, dict) else None
    if status != "passed" or eligible is not True:
        errors.append(f"Stage 5 quality gate not accepted: status={status}, eligible={eligible}")

    row_count = 0
    try:
        with rows_path.open("r", encoding="utf-8") as handle:
            for expected_index, line in enumerate(handle):
                row = json.loads(line)
                if row.get("frame_index_0based") != expected_index:
                    errors.append(
                        f"non-sequential Stage 5 frame at row {expected_index}: {row.get('frame_index_0based')}"
                    )
                    break
                row_count += 1
    except Exception as exc:
        errors.append(f"invalid frame_bboxes.jsonl: {exc}")
    if row_count != frame_count:
        errors.append(f"Stage 5 frame row mismatch: {row_count} != {frame_count}")
    return {"quality_status": status, "eligible": eligible is True, "frame_rows": row_count}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--phase", choices=("stage4", "final"), required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    with Path(args.manifest).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    results: list[dict[str, Any]] = []
    passed = 0
    for item in rows:
        video_id = item["video_id"]
        video_out = output_root / video_id
        errors: list[str] = []
        for rel in (
            "run_summary.json",
            "stage1/draft_plan.json",
            "stage2/step_segments.json",
            "stage3/step_plan.json",
            "stage4/atomic_plan_with_clips.json",
        ):
            require_file(video_out / rel, errors, rel)
        plan_stats = check_plan(video_out, int(item["frame_count"]), errors)
        stage5_stats: dict[str, Any] = {}
        if args.phase == "final":
            stage5_stats = check_stage5(video_out, int(item["frame_count"]), errors)
        ok = not errors
        if ok:
            passed += 1
        results.append(
            {
                "video_id": video_id,
                "ok": ok,
                "errors": errors,
                **plan_stats,
                **stage5_stats,
            }
        )

    report = {
        "phase": args.phase,
        "total_videos": len(rows),
        "passed_videos": passed,
        "all_passed": passed == len(rows),
        "videos": results,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("phase", "total_videos", "passed_videos", "all_passed")}))
    for result in results:
        if not result["ok"]:
            print(f"FAIL {result['video_id']}: {' | '.join(result['errors'][:5])}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
