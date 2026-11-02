#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


DEFAULT_ARK_API_KEY = ""
DEFAULT_ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_ARK_MODEL = "doubao-seed-2-1-pro-260628"
DEFAULT_ARK_REASONING_EFFORT = "medium"


SYSTEM_PROMPT = """
You are a strict embodied-action dataset auditor.
You judge whether a single video step clip and its annotation are SUPER-HIGH-QUALITY.
Return strict JSON only. No markdown. No extra text.
""".strip()


USER_PROMPT_TEMPLATE = """
You are given one short step clip and its current step-level + atomic-action annotation.
The video you see is already cut to exactly this step clip. Its local timeline starts at 0.0 seconds.
Any local timing fields in the annotation are relative to this step clip, not to the original full video.

Your task:
Decide whether this step is SUPER-HIGH-QUALITY and should be kept in a curated dataset.

Keep ONLY if all are true:
1. The clip contains clear, meaningful human operation/action execution, not waiting/countdown/idling/empty setup.
2. The step boundary is tight and rigorous: the clip starts near the start of the described step and ends after the described step is complete, without large unrelated leading/trailing content.
3. The step goal matches the visible action and is neither too broad nor misleading.
4. Atomic actions are aligned to visible phases: no obvious missing action, no severe over-splitting, no severe under-splitting, and no atomic action that describes events outside its own clip.
5. Captions/actions are visually grounded and do not hallucinate important objects or states.
6. The selected step would be useful as a clean training/evaluation example.

Reject if ANY of these appears:
- no-operation/waiting/timer/countdown dominates the clip
- action is too weak, peripheral, or merely a minor adjustment
- step is very broad with several loosely related subtasks
- start/end boundary is visibly inaccurate or includes unrelated neighboring actions
- atomic action segmentation is clearly wrong, too coarse, too fragmented, or not aligned with the clip
- textual annotation is not strictly supported by visible evidence

Score each category from 0 to 10:
- action_execution_quality
- step_boundary_quality
- step_goal_alignment
- atomic_boundary_alignment
- caption_grounding
- overall_quality

Use this strict threshold:
- keep_super_high_quality may be true ONLY if overall_quality >= 8.7
- and every category except overall_quality is >= 8.0
- and there is no severe issue.

Return exactly this JSON schema:
{
  "keep_super_high_quality": true/false,
  "overall_quality": number,
  "action_execution_quality": number,
  "step_boundary_quality": number,
  "step_goal_alignment": number,
  "atomic_boundary_alignment": number,
  "caption_grounding": number,
  "severe_issues": ["short issue label", "..."],
  "reason": "1-3 concise sentences explaining the decision",
  "best_visible_action_summary": "one concise sentence describing what is actually visible"
}

Current annotation JSON:
```json
__ANNOTATION_JSON__
```
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use Ark seed2.1pro to strictly audit selected step clips.")
    parser.add_argument(
        "--input-root",
        default="/Users/lululzz/Desktop/litchi/api/litchi_all_completed_outputs_20260705/high_quality_selected_steps_18_videos_20260706",
    )
    parser.add_argument(
        "--output-root",
        default="/Users/lululzz/Desktop/litchi/api/litchi_all_completed_outputs_20260705/super_high_quality_selected_steps_18_videos_seed21_audit_20260706",
    )
    parser.add_argument("--api-key", default=os.environ.get("ARK_API_KEY") or os.environ.get("API_KEY", DEFAULT_ARK_API_KEY))
    parser.add_argument("--api-base", default=os.environ.get("API_BASE_URL", DEFAULT_ARK_BASE_URL))
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", DEFAULT_ARK_MODEL))
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        default=os.environ.get("ARK_REASONING_EFFORT", DEFAULT_ARK_REASONING_EFFORT),
    )
    parser.add_argument("--video-fps", type=float, default=float(os.environ.get("VIDEO_FPS", "5.0")))
    parser.add_argument("--timeout-sec", type=float, default=float(os.environ.get("API_TIMEOUT_SEC", "420")))
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--audit-only", action="store_true", help="Only run/reuse audit JSON files; do not copy output.")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Debug limit; 0 means all.")
    return parser.parse_args()


def resolve_base_url(api_base: str) -> str:
    base = str(api_base or "").strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base[: -len("/chat/completions")]
    if base.endswith("/responses"):
        return base[: -len("/responses")]
    return base


def extract_json(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    parts: List[str] = []
    for item in getattr(response, "output", []) or []:
        for part in getattr(item, "content", []) or []:
            text = getattr(part, "text", None)
            if isinstance(text, str) and text.strip():
                parts.append(text)
    if parts:
        return "\n".join(parts).strip()
    if hasattr(response, "model_dump_json"):
        return response.model_dump_json()
    return str(response)


def upload_video(client: OpenAI, video_path: Path, video_fps: float, timeout_sec: float) -> str:
    with video_path.open("rb") as f:
        uploaded = client.with_options(timeout=float(timeout_sec)).files.create(
            file=f,
            purpose="user_data",
            extra_body={"preprocess_configs": {"video": {"fps": float(video_fps)}}},
        )
    file_id = str(getattr(uploaded, "id", "") or "")
    if not file_id:
        raise RuntimeError("Files API upload did not return file_id")

    deadline = time.time() + max(float(timeout_sec), 60.0)
    while True:
        file_obj = client.with_options(timeout=float(timeout_sec)).files.retrieve(file_id)
        status = str(getattr(file_obj, "status", "") or "").lower()
        if status in {"failed", "error", "cancelled", "canceled"}:
            raise RuntimeError(f"File processing failed: file_id={file_id}, status={status}")
        if status != "processing":
            return file_id
        if time.time() >= deadline:
            raise TimeoutError(f"Timed out waiting for uploaded file: file_id={file_id}, status={status}")
        time.sleep(2.0)


def delete_file_quietly(client: OpenAI, file_id: str, timeout_sec: float) -> None:
    try:
        client.with_options(timeout=min(float(timeout_sec), 60.0)).files.delete(file_id)
    except Exception:
        pass


def load_candidates(input_root: Path, limit: int = 0) -> List[Dict[str, Any]]:
    manifest = json.loads((input_root / "selection_manifest.json").read_text(encoding="utf-8"))
    out: List[Dict[str, Any]] = []
    for row in manifest:
        step_dir = input_root / row["copied_folder"]
        meta_path = step_dir / "step_selection_metadata.json"
        mp4s = sorted(p for p in step_dir.glob("*.mp4") if p.is_file())
        if len(mp4s) != 1:
            raise RuntimeError(f"Expected exactly one step mp4 in {step_dir}, found {len(mp4s)}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        out.append(
            {
                "video": row["video"],
                "video_folder": row["video_folder"],
                "step_id": int(row["step_id"]),
                "copied_folder": row["copied_folder"],
                "step_dir": str(step_dir),
                "step_clip": str(mp4s[0]),
                "metadata": meta,
            }
        )
    if limit and limit > 0:
        return out[: int(limit)]
    return out


def build_clip_local_annotation(candidate: Dict[str, Any]) -> Dict[str, Any]:
    step = dict(candidate["metadata"]["step"])
    step_start_sec = float(step.get("start_sec", 0.0) or 0.0)
    step_end_sec = float(step.get("end_sec", step_start_sec) or step_start_sec)
    step_start_frame = int(step.get("start_frame_0based", 0) or 0)
    duration_sec = max(0.0, step_end_sec - step_start_sec)

    atomic_actions: List[Dict[str, Any]] = []
    for action in step.get("atomic_actions", []) or []:
        if not isinstance(action, dict):
            continue
        local_action = {
            "atomic_action_id": action.get("atomic_action_id"),
            "actor": action.get("actor"),
            "action": action.get("action"),
            "patient": action.get("patient"),
            "caption": action.get("caption"),
        }
        try:
            local_action["local_start_sec"] = round(float(action.get("start_sec", step_start_sec)) - step_start_sec, 3)
            local_action["local_end_sec"] = round(float(action.get("end_sec", step_start_sec)) - step_start_sec, 3)
        except Exception:
            pass
        if action.get("start_frame_0based") is not None and action.get("end_frame_0based_exclusive") is not None:
            try:
                local_action["relative_start_frame_0based"] = int(action.get("start_frame_0based")) - step_start_frame
                local_action["relative_end_frame_0based_exclusive"] = int(action.get("end_frame_0based_exclusive")) - step_start_frame
            except Exception:
                pass
        atomic_actions.append(local_action)

    return {
        "video": candidate["video"],
        "step_id": candidate["step_id"],
        "step_clip_note": "This is an already cut step clip; local timeline starts at 0.0 seconds.",
        "step_duration_sec": round(duration_sec, 3),
        "step_goal": step.get("step_goal", ""),
        "atomic_actions": atomic_actions,
    }


def audit_one(args: argparse.Namespace, candidate: Dict[str, Any], audit_dir: Path, lock: threading.Lock) -> Dict[str, Any]:
    rel_key = f"{candidate['video_folder']}__step{candidate['step_id']:02d}"
    audit_path = audit_dir / f"{rel_key}.json"
    if audit_path.exists():
        return json.loads(audit_path.read_text(encoding="utf-8"))

    client = OpenAI(api_key=args.api_key, base_url=resolve_base_url(args.api_base), max_retries=0)
    annotation = build_clip_local_annotation(candidate)
    prompt = USER_PROMPT_TEMPLATE.replace(
        "__ANNOTATION_JSON__",
        json.dumps(annotation, ensure_ascii=False, indent=2),
    )
    last_error: Optional[str] = None
    for attempt in range(1, int(args.retries) + 1):
        file_id = ""
        try:
            file_id = upload_video(client, Path(candidate["step_clip"]), float(args.video_fps), float(args.timeout_sec))
            response = client.with_options(timeout=float(args.timeout_sec)).responses.create(
                model=args.model,
                instructions=SYSTEM_PROMPT,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_video", "file_id": file_id},
                            {"type": "input_text", "text": prompt},
                        ],
                    }
                ],
                max_output_tokens=int(args.max_output_tokens),
                stream=False,
                reasoning={"effort": args.reasoning_effort},
                extra_body={"thinking": {"type": "enabled"}},
            )
            raw_text = extract_response_text(response)
            judgment = extract_json(raw_text)
            result = normalize_judgment(candidate, judgment, raw_text)
            audit_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            with lock:
                print(
                    f"[audit] {rel_key}: keep={result['keep_super_high_quality']} "
                    f"overall={result['overall_quality']} reason={result['reason'][:120]}"
                )
            return result
        except Exception as exc:
            last_error = str(exc)
            with lock:
                print(f"[audit] {rel_key}: attempt {attempt}/{args.retries} failed: {last_error}")
            time.sleep(min(2 ** (attempt - 1), 8))
        finally:
            if file_id:
                delete_file_quietly(client, file_id, float(args.timeout_sec))
    result = {
        "video": candidate["video"],
        "video_folder": candidate["video_folder"],
        "step_id": candidate["step_id"],
        "copied_folder": candidate["copied_folder"],
        "step_clip": candidate["step_clip"],
        "keep_super_high_quality": False,
        "overall_quality": 0.0,
        "action_execution_quality": 0.0,
        "step_boundary_quality": 0.0,
        "step_goal_alignment": 0.0,
        "atomic_boundary_alignment": 0.0,
        "caption_grounding": 0.0,
        "severe_issues": ["api_audit_failed"],
        "reason": f"API audit failed after retries: {last_error}",
        "best_visible_action_summary": "",
        "raw_model_text": "",
    }
    audit_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def score_value(obj: Dict[str, Any], key: str) -> float:
    try:
        return float(obj.get(key, 0.0))
    except Exception:
        return 0.0


def normalize_judgment(candidate: Dict[str, Any], judgment: Dict[str, Any], raw_text: str) -> Dict[str, Any]:
    score_keys = [
        "overall_quality",
        "action_execution_quality",
        "step_boundary_quality",
        "step_goal_alignment",
        "atomic_boundary_alignment",
        "caption_grounding",
    ]
    scores = {key: max(0.0, min(10.0, score_value(judgment, key))) for key in score_keys}
    severe = judgment.get("severe_issues", [])
    if not isinstance(severe, list):
        severe = [str(severe)]
    severe = [str(x).strip() for x in severe if str(x).strip()]
    model_keep = bool(judgment.get("keep_super_high_quality", False))
    threshold_keep = (
        model_keep
        and scores["overall_quality"] >= 8.7
        and scores["action_execution_quality"] >= 8.0
        and scores["step_boundary_quality"] >= 8.0
        and scores["step_goal_alignment"] >= 8.0
        and scores["atomic_boundary_alignment"] >= 8.0
        and scores["caption_grounding"] >= 8.0
        and not severe
    )
    return {
        "video": candidate["video"],
        "video_folder": candidate["video_folder"],
        "step_id": candidate["step_id"],
        "copied_folder": candidate["copied_folder"],
        "step_clip": candidate["step_clip"],
        "keep_super_high_quality": bool(threshold_keep),
        **scores,
        "model_keep_super_high_quality": model_keep,
        "severe_issues": severe,
        "reason": str(judgment.get("reason", "")).strip(),
        "best_visible_action_summary": str(judgment.get("best_visible_action_summary", "")).strip(),
        "raw_model_text": raw_text,
    }


def copy_super_selection(input_root: Path, output_root: Path, candidates: List[Dict[str, Any]], results: List[Dict[str, Any]], overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise RuntimeError(f"Output root exists: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    result_by_folder = {r["copied_folder"]: r for r in results}
    candidate_by_folder = {c["copied_folder"]: c for c in candidates}
    kept = [r for r in results if r.get("keep_super_high_quality")]
    rejected = [r for r in results if not r.get("keep_super_high_quality")]

    video_names = sorted({c["video_folder"] for c in candidates})
    for video_folder in video_names:
        (output_root / video_folder).mkdir(parents=True, exist_ok=True)

    for r in kept:
        c = candidate_by_folder[r["copied_folder"]]
        src = input_root / c["copied_folder"]
        dst = output_root / c["copied_folder"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
        (dst / "seed21_super_quality_audit.json").write_text(json.dumps(r, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for video_folder in video_names:
        src_video = input_root / video_folder
        dst_video = output_root / video_folder
        for name in ["source_final_plan.json", "run_summary.json", "selected_steps.tsv", "selected_steps.json", "excluded_steps.tsv"]:
            src = src_video / name
            if src.exists():
                shutil.copy2(src, dst_video / name)

    (output_root / "seed21_audit_all_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_root / "super_selection_manifest.json").write_text(json.dumps(kept, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_root / "super_rejected_steps.json").write_text(json.dumps(rejected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with (output_root / "super_selection_manifest.tsv").open("w", encoding="utf-8") as f:
        f.write("video\tvideo_folder\tstep_id\toverall_quality\taction_execution_quality\tstep_boundary_quality\tstep_goal_alignment\tatomic_boundary_alignment\tcaption_grounding\tcopied_folder\treason\tstep_goal\n")
        for r in kept:
            c = candidate_by_folder[r["copied_folder"]]
            step_goal = c["metadata"]["step"].get("step_goal", "")
            f.write(
                f"{r['video']}\t{r['video_folder']}\t{r['step_id']}\t{r['overall_quality']:.2f}\t"
                f"{r['action_execution_quality']:.2f}\t{r['step_boundary_quality']:.2f}\t"
                f"{r['step_goal_alignment']:.2f}\t{r['atomic_boundary_alignment']:.2f}\t{r['caption_grounding']:.2f}\t"
                f"{r['copied_folder']}\t{str(r['reason']).replace(chr(9), ' ').replace(chr(10), ' ')}\t"
                f"{str(step_goal).replace(chr(9), ' ').replace(chr(10), ' ')}\n"
            )

    with (output_root / "super_rejected_steps.tsv").open("w", encoding="utf-8") as f:
        f.write("video\tvideo_folder\tstep_id\toverall_quality\tsevere_issues\treason\tcopied_folder\tstep_goal\n")
        for r in rejected:
            c = candidate_by_folder[r["copied_folder"]]
            step_goal = c["metadata"]["step"].get("step_goal", "")
            f.write(
                f"{r['video']}\t{r['video_folder']}\t{r['step_id']}\t{r['overall_quality']:.2f}\t"
                f"{'|'.join(r.get('severe_issues', []))}\t{str(r['reason']).replace(chr(9), ' ').replace(chr(10), ' ')}\t"
                f"{r['copied_folder']}\t{str(step_goal).replace(chr(9), ' ').replace(chr(10), ' ')}\n"
            )

    readme = f"""# Super-high-quality selected steps audited by seed2.1pro

Input root:
`{input_root}`

Output root:
`{output_root}`

Audit model: `{DEFAULT_ARK_MODEL}`

Strict keep rule:
- model must return `keep_super_high_quality=true`
- overall_quality >= 8.7
- all sub-scores >= 8.0
- no severe issues

Summary:
- Candidate steps audited: {len(results)}
- Super-high-quality steps kept: {len(kept)}
- Rejected after strict audit: {len(rejected)}

Important files:
- `super_selection_manifest.tsv`: kept super-high-quality steps
- `super_rejected_steps.tsv`: rejected steps and reasons
- `seed21_audit_all_results.json`: full model judgments
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    audit_dir = input_root / "seed21_super_quality_audit_cache"
    audit_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates(input_root, int(args.limit))

    lock = threading.Lock()
    results: List[Dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        future_map = {executor.submit(audit_one, args, c, audit_dir, lock): c for c in candidates}
        for fut in concurrent.futures.as_completed(future_map):
            results.append(fut.result())
    results.sort(key=lambda r: (r["video"], int(r["step_id"])))

    if not args.audit_only:
        copy_super_selection(input_root, output_root, candidates, results, bool(args.overwrite_output))

    kept = sum(1 for r in results if r.get("keep_super_high_quality"))
    print(f"audited={len(results)} kept={kept} rejected={len(results)-kept}")
    if not args.audit_only:
        print(f"output_root={output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
