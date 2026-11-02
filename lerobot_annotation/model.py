"""Bounded automatic video annotation. All debugging artifacts stay in work/."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

from .dataset import atomic_json, read_json, sha256, validate_segments

PROMPT = """You annotate ONE dual-arm Franka robot with Wuji dexterous hands.
cam0 is the user-designated primary view. Wrist cameras, if absent, provide no evidence.
Left/right means robot body identity, never screen position. If uncertain, say
'one robot hand' rather than inventing laterality. Background people are not the robot.
The task instruction is context, not proof of success. Do not infer a tomato
detached or was released unless visible evidence supports that conclusion.
Describe visible idle/repositioning periods explicitly when present.
Return JSON only: {"steps":[{"start_sec":0.0,"end_sec":1.0,
"text":"A concise English description of this manipulation phase, including
the supporting hand when visible.","evidence_status":"observed"}]}.
Times are clip-local seconds, half-open intervals. Ordered intervals must cover
the entire supplied clip exactly once, without gaps or overlap. Use semantic
action/state changes, not arbitrary equal-duration chunks. If an interval cannot
be described reliably, mark it uncertain; never invent an action to pass checks.
Only observed or uncertain are valid evidence_status values. Boundaries are
approximate observations; do not claim exact-frame contact timing.
"""


def windows(length, fps, window_seconds):
    width = max(1, round(fps * window_seconds))
    result = [[start, min(length, start + width)] for start in range(0, length, width)]
    if len(result) > 1 and result[-1][1] - result[-1][0] < round(fps * 5):
        result[-2][1] = result[-1][1]
        result.pop()
    return result


def normalize_response(raw, *, start, end, fps):
    value = raw.strip()
    if value.startswith("```") and value.endswith("```"):
        value = "\n".join(value.splitlines()[1:-1])
    payload = json.loads(value)
    if not isinstance(payload, dict) or set(payload) != {"steps"} or not isinstance(payload["steps"], list):
        raise ValueError("Expected exactly a steps array")
    result = []
    previous = 0.0
    duration = (end - start) / fps
    for row in payload["steps"]:
        if not isinstance(row, dict) or set(row) != {"start_sec", "end_sec", "text", "evidence_status"}:
            raise ValueError("Unexpected step schema")
        a, b = row["start_sec"], row["end_sec"]
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in (a, b)):
            raise ValueError("Invalid model timestamp")
        if not (0 <= a < b <= duration + 1e-6) or abs(a - previous) > 1e-6:
            raise ValueError("Model times do not cover clip continuously")
        result.append({"start_frame": start + round(a * fps),
                       "end_frame": min(end, start + round(b * fps)),
                       "text": row["text"], "evidence_status": row["evidence_status"]})
        previous = b
    if abs(previous - duration) > 1e-6:
        raise ValueError("Model omitted the clip tail")
    # Rebase for the shared complete-coverage validator.
    validate_segments([{**s, "start_frame": s["start_frame"] - start,
                        "end_frame": s["end_frame"] - start} for s in result], end - start)
    return result


class ArkAnnotator:
    def __init__(self, *, model, video_fps=5.0, attempts=2, max_calls=300,
                 api_base="https://ark.cn-beijing.volces.com/api/v3", timeout=420):
        if not model or attempts < 1 or max_calls < 1:
            raise ValueError("Model, positive attempts and call budget are required")
        self.model, self.video_fps, self.attempts = model, video_fps, attempts
        self.max_calls, self.api_base, self.timeout = max_calls, api_base, timeout
        self.calls = 0

    def verify_video(self, video, *, length, fps, shape):
        from demos.franka_wuji.run_demo import probe
        meta = probe(video)
        if (meta["frames"] != length or not math.isclose(meta["fps"], fps, rel_tol=1e-6)
                or [meta["height"], meta["width"], 3] != shape):
            raise ValueError(f"Video decoding disagrees with LeRobot metadata: {video}")

    def annotate(self, video, *, start, end, fps, task, work, source_hash):
        work = Path(work)
        work.mkdir(parents=True, exist_ok=True)
        config = {"model": self.model, "api_base": self.api_base, "video_fps": self.video_fps,
                  "source_sha256": source_hash, "start_frame": start, "end_frame": end,
                  "fps": fps, "task_context": task,
                  "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest()}
        cache = work / "labels.json"
        if (work / "config.json").exists() and read_json(work / "config.json") != config:
            raise ValueError("Window configuration changed; use a new work directory")
        atomic_json(work / "config.json", config)
        if cache.exists():
            labels = read_json(cache)
            validate_segments([{**s, "start_frame": s["start_frame"] - start,
                                "end_frame": s["end_frame"] - start} for s in labels], end - start)
            return labels
        key = os.environ.get("ARK_API_KEY") or os.environ.get("API_KEY")
        if not key:
            raise ValueError("Set ARK_API_KEY or API_KEY; credentials are never saved")
        # Reuse the existing project's upload/request/cleanup implementation.
        from api.api_seed_openai_easy_no_short_step.api_client_qwen_v2 import (
            create_api_client, build_request_payload_input, call_model_api,
        )
        from demos.franka_wuji.run_demo import probe
        clip = work / "clip.mp4"
        if not clip.exists():
            temporary = work / "clip.building.mp4"
            select = f"select=between(n\\,{start}\\,{end - 1}),setpts=N/({fps}*TB)"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-map", "0:v:0",
                            "-vf", select, "-fps_mode", "passthrough", "-c:v", "libx264",
                            "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                            "-an", str(temporary)], check=True)
            temporary.replace(clip)
        meta = probe(clip)
        if meta["frames"] != end - start or not math.isclose(meta["fps"], fps, rel_tol=1e-6):
            raise ValueError("Window decoding/frame count mismatch")
        user = (f"Duration={(end-start)/fps:.9f} seconds. FPS={fps}. "
                f"Task context, not ground truth: {json.dumps(task)}")
        (work / "system_prompt.txt").write_text(PROMPT)
        errors = ""
        client = create_api_client(self.api_base, key)
        # Persist attempts before calls. An interrupted paid call consumes budget.
        ledger = read_json(work / "attempts.json") if (work / "attempts.json").exists() else []
        for attempt in range(len(ledger), self.attempts):
            if self.calls >= self.max_calls:
                raise RuntimeError("API call budget reached")
            ledger.append({"attempt": attempt + 1, "status": "started"})
            atomic_json(work / "attempts.json", ledger)
            self.calls += 1
            try:
                messages = [{"role": "system", "content": PROMPT}, {"role": "user", "content": [
                    {"type": "input_text", "text": user + ("\nPrevious error: " + errors if errors else "")},
                    {"type": "input_video_path", "video_path": str(clip), "fps": self.video_fps}]}]
                raw, usage = call_model_api(client, model_name=self.model,
                    payload_input=build_request_payload_input(messages), video_fps=self.video_fps,
                    timeout_sec=self.timeout, max_output_tokens=8000, reasoning_effort="medium")
                (work / f"response_{attempt+1}.txt").write_text(raw)
                atomic_json(work / f"usage_{attempt+1}.json", usage)
                labels = normalize_response(raw, start=start, end=end, fps=fps)
                ledger[-1]["status"] = "passed"
                atomic_json(work / "attempts.json", ledger)
                atomic_json(cache, labels)
                return labels
            except Exception as exc:
                errors = str(exc)
                ledger[-1].update(status="failed", error_type=type(exc).__name__, error=errors)
                atomic_json(work / "attempts.json", ledger)
                if attempt + 1 < self.attempts:
                    time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"Automatic annotation failed after {self.attempts} attempts: {errors}")
