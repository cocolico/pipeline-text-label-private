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
from typing import Any, Dict, List, Optional

from openai import OpenAI

from api.tools.super_filter_steps_with_seed21_audit import (
    DEFAULT_ARK_API_KEY,
    DEFAULT_ARK_BASE_URL,
    DEFAULT_ARK_MODEL,
    DEFAULT_ARK_REASONING_EFFORT,
    build_clip_local_annotation,
    delete_file_quietly,
    extract_response_text,
    load_candidates,
    resolve_base_url,
    upload_video,
)


SYSTEM_PROMPT = """
You are an expert embodied-action dataset auditor.
You judge whether a single step clip and its annotation are VERY-HIGH-QUALITY.
Return strict JSON only. No markdown. No extra text.
""".strip()


USER_PROMPT_TEMPLATE = """
You are given one already-cut step clip. The clip-local timeline starts at 0.0 seconds.
You are also given its step goal and atomic-action annotation, with local timing fields.

Goal:
Select a VERY-HIGH-QUALITY curated subset. This is less strict than a flawless "super" set:
- Keep if the clip has clear meaningful human operation and the core step/atomic annotation is strongly useful.
- Minor issues are acceptable: small leading/trailing context, small timestamp imprecision, minor hand laterality wording, or a caption detail that is slightly over-specific.
- Reject if the issue would make the step unreliable for training/evaluation.

KEEP if all are mostly true:
1. A clear operation/action is visible and is not dominated by waiting, countdown, static frames, or empty setup.
2. The step goal matches the visible action at the main task level.
3. Boundaries are reasonably tight: no large unrelated leading/trailing action; small context is acceptable.
4. Atomic actions are useful and mostly aligned with visible phases; minor coarse/fine granularity is acceptable if the main state changes are correct.
5. Captions/actions are mostly grounded; minor wording mistakes are acceptable, but no important hallucinated task state.

HARD REJECT if any are true:
- no clear operation is visible
- waiting/idling/countdown/static frames dominate the clip
- main step goal is wrong or describes a different action
- major object/action hallucination
- important action completion is not visible
- boundary includes a different neighboring task as a major part
- atomic actions are severely wrong or misleading

Score each category from 0 to 10:
- action_execution_quality
- step_boundary_quality
- step_goal_alignment
- atomic_boundary_alignment
- caption_grounding
- overall_quality

For this very-high-quality set, use this guidance:
- 9.0-10.0: excellent, clean, should keep.
- 8.0-8.9: strong, keep unless there is a hard reject issue.
- 7.0-7.9: usable high-quality with minor issues, keep if the action and goal are clearly correct.
- below 7.0: usually reject unless the visible action is unusually valuable and the weakness is minor.

Return exactly this JSON schema:
{
  "keep_very_high_quality": true/false,
  "hard_reject": true/false,
  "overall_quality": number,
  "action_execution_quality": number,
  "step_boundary_quality": number,
  "step_goal_alignment": number,
  "atomic_boundary_alignment": number,
  "caption_grounding": number,
  "issues": ["short issue label", "..."],
  "reason": "1-3 concise sentences explaining the decision",
  "best_visible_action_summary": "one concise sentence describing what is actually visible"
}

Current annotation JSON:
```json
__ANNOTATION_JSON__
```
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use Ark seed2.1pro to create an around-40 very-high-quality step subset.")
    parser.add_argument(
        "--input-root",
        default="/Users/lululzz/Desktop/litchi/api/litchi_all_completed_outputs_20260705/high_quality_selected_steps_18_videos_20260706",
    )
    parser.add_argument(
        "--output-root",
        default="/Users/lululzz/Desktop/litchi/api/litchi_all_completed_outputs_20260705/very_high_quality_selected_steps_18_videos_seed21_audit_around40_20260706",
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
    parser.add_argument("--target-count", type=int, default=40)
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


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


def score_value(obj: Dict[str, Any], key: str) -> float:
    try:
        return max(0.0, min(10.0, float(obj.get(key, 0.0))))
    except Exception:
        return 0.0


def normalize(candidate: Dict[str, Any], judgment: Dict[str, Any], raw_text: str) -> Dict[str, Any]:
    score_keys = [
        "overall_quality",
        "action_execution_quality",
        "step_boundary_quality",
        "step_goal_alignment",
        "atomic_boundary_alignment",
        "caption_grounding",
    ]
    scores = {k: score_value(judgment, k) for k in score_keys}
    issues = judgment.get("issues", [])
    if not isinstance(issues, list):
        issues = [str(issues)]
    issues = [str(x).strip() for x in issues if str(x).strip()]
    model_keep = bool(judgment.get("keep_very_high_quality", False))
    hard_reject = bool(judgment.get("hard_reject", False))
    eligible = (
        model_keep
        and not hard_reject
        and scores["overall_quality"] >= 7.0
        and scores["action_execution_quality"] >= 7.0
        and scores["step_goal_alignment"] >= 6.8
        and scores["step_boundary_quality"] >= 6.0
        and scores["atomic_boundary_alignment"] >= 6.0
        and scores["caption_grounding"] >= 6.0
    )
    return {
        "video": candidate["video"],
        "video_folder": candidate["video_folder"],
        "step_id": candidate["step_id"],
        "copied_folder": candidate["copied_folder"],
        "step_clip": candidate["step_clip"],
        "eligible_very_high_quality": bool(eligible),
        "model_keep_very_high_quality": model_keep,
        "hard_reject": hard_reject,
        **scores,
        "issues": issues,
        "reason": str(judgment.get("reason", "")).strip(),
        "best_visible_action_summary": str(judgment.get("best_visible_action_summary", "")).strip(),
        "raw_model_text": raw_text,
    }


def audit_one(args: argparse.Namespace, candidate: Dict[str, Any], audit_dir: Path, lock: threading.Lock) -> Dict[str, Any]:
    rel_key = f"{candidate['video_folder']}__step{candidate['step_id']:02d}"
    audit_path = audit_dir / f"{rel_key}.json"
    if audit_path.exists():
        return json.loads(audit_path.read_text(encoding="utf-8"))

    client = OpenAI(api_key=args.api_key, base_url=resolve_base_url(args.api_base), max_retries=0)
    annotation = build_clip_local_annotation(candidate)
    prompt = USER_PROMPT_TEMPLATE.replace("__ANNOTATION_JSON__", json.dumps(annotation, ensure_ascii=False, indent=2))
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
            result = normalize(candidate, extract_json(raw_text), raw_text)
            audit_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            with lock:
                print(
                    f"[very-audit] {rel_key}: eligible={result['eligible_very_high_quality']} "
                    f"overall={result['overall_quality']} hard={result['hard_reject']} reason={result['reason'][:100]}"
                )
            return result
        except Exception as exc:
            last_error = str(exc)
            with lock:
                print(f"[very-audit] {rel_key}: attempt {attempt}/{args.retries} failed: {last_error}")
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
        "eligible_very_high_quality": False,
        "model_keep_very_high_quality": False,
        "hard_reject": True,
        "overall_quality": 0.0,
        "action_execution_quality": 0.0,
        "step_boundary_quality": 0.0,
        "step_goal_alignment": 0.0,
        "atomic_boundary_alignment": 0.0,
        "caption_grounding": 0.0,
        "issues": ["api_audit_failed"],
        "reason": f"API audit failed after retries: {last_error}",
        "best_visible_action_summary": "",
        "raw_model_text": "",
    }
    audit_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def rank_key(row: Dict[str, Any]) -> tuple:
    min_sub = min(
        float(row.get("action_execution_quality", 0.0)),
        float(row.get("step_boundary_quality", 0.0)),
        float(row.get("step_goal_alignment", 0.0)),
        float(row.get("atomic_boundary_alignment", 0.0)),
        float(row.get("caption_grounding", 0.0)),
    )
    return (
        float(row.get("overall_quality", 0.0)),
        min_sub,
        float(row.get("action_execution_quality", 0.0)),
        float(row.get("step_boundary_quality", 0.0)),
        float(row.get("caption_grounding", 0.0)),
    )


def copy_selection(input_root: Path, output_root: Path, candidates: List[Dict[str, Any]], results: List[Dict[str, Any]], target_count: int, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise RuntimeError(f"Output root exists: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    eligible = [r for r in results if r.get("eligible_very_high_quality")]
    eligible.sort(key=rank_key, reverse=True)
    kept = eligible[: int(target_count)] if int(target_count) > 0 and len(eligible) > int(target_count) else eligible
    kept_keys = {r["copied_folder"] for r in kept}
    rejected = [r for r in results if r["copied_folder"] not in kept_keys]

    candidate_by_folder = {c["copied_folder"]: c for c in candidates}
    video_names = sorted({c["video_folder"] for c in candidates})
    for video_folder in video_names:
        (output_root / video_folder).mkdir(parents=True, exist_ok=True)

    for r in kept:
        c = candidate_by_folder[r["copied_folder"]]
        src = input_root / c["copied_folder"]
        dst = output_root / c["copied_folder"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
        (dst / "seed21_very_quality_audit.json").write_text(json.dumps(r, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for video_folder in video_names:
        src_video = input_root / video_folder
        dst_video = output_root / video_folder
        for name in ["source_final_plan.json", "run_summary.json", "selected_steps.tsv", "selected_steps.json", "excluded_steps.tsv"]:
            src = src_video / name
            if src.exists():
                shutil.copy2(src, dst_video / name)

    (output_root / "seed21_very_audit_all_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_root / "very_selection_manifest.json").write_text(json.dumps(kept, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_root / "very_rejected_steps.json").write_text(json.dumps(rejected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with (output_root / "very_selection_manifest.tsv").open("w", encoding="utf-8") as f:
        f.write("rank\tvideo\tvideo_folder\tstep_id\toverall_quality\taction_execution_quality\tstep_boundary_quality\tstep_goal_alignment\tatomic_boundary_alignment\tcaption_grounding\tcopied_folder\treason\tstep_goal\n")
        for i, r in enumerate(kept, 1):
            c = candidate_by_folder[r["copied_folder"]]
            step_goal = c["metadata"]["step"].get("step_goal", "")
            f.write(
                f"{i}\t{r['video']}\t{r['video_folder']}\t{r['step_id']}\t{r['overall_quality']:.2f}\t"
                f"{r['action_execution_quality']:.2f}\t{r['step_boundary_quality']:.2f}\t{r['step_goal_alignment']:.2f}\t"
                f"{r['atomic_boundary_alignment']:.2f}\t{r['caption_grounding']:.2f}\t{r['copied_folder']}\t"
                f"{str(r['reason']).replace(chr(9), ' ').replace(chr(10), ' ')}\t"
                f"{str(step_goal).replace(chr(9), ' ').replace(chr(10), ' ')}\n"
            )

    with (output_root / "very_rejected_steps.tsv").open("w", encoding="utf-8") as f:
        f.write("video\tvideo_folder\tstep_id\toverall_quality\thard_reject\teligible\tissues\treason\tcopied_folder\tstep_goal\n")
        for r in rejected:
            c = candidate_by_folder[r["copied_folder"]]
            step_goal = c["metadata"]["step"].get("step_goal", "")
            f.write(
                f"{r['video']}\t{r['video_folder']}\t{r['step_id']}\t{r['overall_quality']:.2f}\t"
                f"{r['hard_reject']}\t{r['eligible_very_high_quality']}\t{'|'.join(r.get('issues', []))}\t"
                f"{str(r['reason']).replace(chr(9), ' ').replace(chr(10), ' ')}\t{r['copied_folder']}\t"
                f"{str(step_goal).replace(chr(9), ' ').replace(chr(10), ' ')}\n"
            )

    readme = f"""# Very-high-quality selected steps audited by seed2.1pro

Input root:
`{input_root}`

Output root:
`{output_root}`

Audit model: `{DEFAULT_ARK_MODEL}`

Selection rule:
- second-pass seed2.1pro video audit with a very-high-quality rubric
- minor imperfections allowed only when the core visible action, step goal, and atomic annotations remain useful
- hard rejects remain excluded: no-operation, waiting-dominant, wrong main action, major hallucination, or severely misleading atomic annotation
- eligible steps are ranked by API scores and capped at target count {target_count}

Summary:
- Candidate steps audited: {len(results)}
- Eligible very-high-quality steps: {len(eligible)}
- Kept steps: {len(kept)}
- Rejected/not selected: {len(rejected)}

Important files:
- `very_selection_manifest.tsv`: kept very-high-quality steps
- `very_rejected_steps.tsv`: rejected/not-selected steps and reasons
- `seed21_very_audit_all_results.json`: full model judgments
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    audit_dir = input_root / "seed21_very_quality_audit_cache"
    audit_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates(input_root)
    lock = threading.Lock()
    results: List[Dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        future_map = {executor.submit(audit_one, args, c, audit_dir, lock): c for c in candidates}
        for fut in concurrent.futures.as_completed(future_map):
            results.append(fut.result())
    results.sort(key=lambda r: (r["video"], int(r["step_id"])))
    if not args.audit_only:
        copy_selection(input_root, output_root, candidates, results, int(args.target_count), bool(args.overwrite_output))
    eligible = sum(1 for r in results if r.get("eligible_very_high_quality"))
    print(f"audited={len(results)} eligible={eligible}")
    if not args.audit_only:
        print(f"output_root={output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
