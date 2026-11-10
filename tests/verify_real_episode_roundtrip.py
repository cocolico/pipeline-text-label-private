"""Disposable real-sensor transport check. NO model, NO semantic ground truth."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lerobot_annotation.dataset import atomic_json, inspect, jsonl, path_for, validate_output, export_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    source = inspect(args.input)
    report_path = args.report.resolve()
    if source["root"] == report_path or source["root"] in report_path.parents:
        parser.error("Report must be outside source")
    episode = source["episodes"][0]
    relatives = [path_for(source["info"], 0)] + [path_for(source["info"], 0, video_key=c) for c in source["cameras"]]
    required = sum((source["root"] / r).stat().st_size for r in relatives) * 2.5 + 128 * 1024 * 1024
    if shutil.disk_usage(tempfile.gettempdir()).free < required:
        raise OSError("Insufficient space even for the disposable episode-0 check")
    with tempfile.TemporaryDirectory(prefix="lerobot-real-roundtrip-") as temporary:
        root = Path(temporary);single = root / "input";(single / "meta").mkdir(parents=True)
        info = deepcopy(source["info"])
        info.update(total_episodes=1, total_frames=episode["length"], total_videos=len(source["cameras"]),
                    total_chunks=1, splits={"train": "0:1"})
        atomic_json(single / "meta/info.json", info)
        jsonl(single / "meta/episodes.jsonl", [episode])
        jsonl(single / "meta/episodes_stats.jsonl", [source["stats"][0]])
        jsonl(single / "meta/tasks.jsonl", source["tasks"])
        for relative in relatives:
            output = single / relative;output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source["root"] / relative, output)
        isolated = inspect(single)
        middle = episode["length"] // 2
        labels = {0: [{"start_frame": 0, "end_frame": middle,
                       "text": "TRANSPORT TEST ONLY: first frame range.", "evidence_status": "observed"},
                      {"start_frame": middle, "end_frame": episode["length"],
                       "text": "TRANSPORT TEST ONLY: second frame range.", "evidence_status": "observed"}]}
        export_dataset(isolated, root / "output", labels)
        verification = validate_output(isolated, root / "output", labels)
        verification.pop("output_sha256")
        report = {"test_kind": "real_sensor_roundtrip_with_synthetic_labels", "source": str(source["root"]),
                  "episode_index": 0, "frame_count": episode["length"], "api_calls": 0,
                  "remote_writes": 0, "semantic_annotation_verified": False,
                  "temporary_datasets_removed_after_test": True, "verification": verification}
        atomic_json(args.report, report)
        print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
