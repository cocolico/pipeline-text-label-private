from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from .dataset import atomic_json, inspect, read_json, sha256
from .pipeline import run
from .upload import upload


def main():
    parser = argparse.ArgumentParser(description="LeRobot v2.1 -> automatic step annotation -> LeRobot v2.1")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("inspect", help="Read-only metadata, Parquet schema and artifact preflight; no API")
    check.add_argument("--input", type=Path, required=True)
    check.add_argument("--report", type=Path)
    annotate = commands.add_parser("run", help="Automatic cam0 annotation and validated NEW dataset export")
    annotate.add_argument("--input", type=Path, required=True)
    annotate.add_argument("--output", type=Path, required=True)
    annotate.add_argument("--work", type=Path, required=True)
    annotate.add_argument("--model", default=os.environ.get("MODEL_NAME", ""))
    annotate.add_argument("--video-fps", type=float, default=5)
    annotate.add_argument("--window-seconds", type=float, default=30)
    annotate.add_argument("--attempts", type=int, default=2)
    annotate.add_argument("--max-api-calls", type=int, default=300)
    annotate.add_argument("--episode-limit", type=int,
                          help="Test only the first N episodes; output metadata describes this subset")
    annotate.add_argument("--upload-targets", type=Path, help="Optional explicit mounted receivers; no defaults")
    transfer = commands.add_parser("upload", help="Resume append-only upload after annotation completion")
    transfer.add_argument("--output", type=Path, required=True)
    transfer.add_argument("--work", type=Path, required=True)
    transfer.add_argument("--upload-targets", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "inspect":
        data = inspect(args.input)
        report = {"status": "passed", "scope": "metadata/schema/artifact preflight only; no full payload comparison",
                  "source": str(data["root"]), "episodes": len(data["episodes"]),
                  "frames": data["info"]["total_frames"], "cameras": data["cameras"],
                  "artifact_bytes": data["bytes"], "signature": data["signature"]}
        if args.report:
            report_path = args.report.resolve()
            if report_path == data["root"] or data["root"] in report_path.parents:
                parser.error("Inspection report must be outside the read-only input")
            atomic_json(args.report, report)
        print(json.dumps(report, indent=2)); return
    if args.command == "run":
        if not args.model:
            parser.error("Specify --model or MODEL_NAME")
        if any(not math.isfinite(v) or v <= 0 for v in [args.video_fps, args.window_seconds]):
            parser.error("Sampling/window must be finite and positive")
        output = run(args.input, args.output, args.work, model=args.model,
                     video_fps=args.video_fps, window_seconds=args.window_seconds,
                     attempts=args.attempts, max_calls=args.max_api_calls,
                     episode_limit=args.episode_limit)
    else:
        output = args.output.resolve()
        proof_path = args.work / "completed.json"
        if not proof_path.exists() or json.loads(proof_path.read_text())["config"]["output_root"] != str(output):
            parser.error("A completed annotation job is required before upload")
    if args.upload_targets:
        proof = read_json(args.work / "completed.json")
        actual = {str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file()}
        if actual != proof["validation"]["output_sha256"]:
            parser.error("Delivered files differ from the completed validation proof; upload blocked")
        targets = json.loads(args.upload_targets.read_text())
        if not isinstance(targets, list) or not targets:
            parser.error("Upload targets must be a nonempty list of explicit destinations")
        for i, target in enumerate(targets):
            if set(target) != {"name", "destination"}:
                parser.error("Each target requires name and destination")
            upload(output, target["destination"], args.work / "uploads" / str(i))
    print(json.dumps({"status": "completed", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
