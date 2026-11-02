"""Synthetic labeler tests transport/contracts, NOT model semantic accuracy."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from lerobot_annotation.dataset import (atomic_json, export_dataset, inspect, jsonl,
                                        read_json, validate_output, validate_segments)
from lerobot_annotation.dataset import select_prefix
from lerobot_annotation.model import ArkAnnotator, normalize_response, windows
from lerobot_annotation.pipeline import publish_new_directory, require_space, run
from lerobot_annotation.upload import upload


def fixture(root):
    (root / "meta").mkdir(parents=True)
    features = {name: {"dtype": "int64", "shape": [1]} for name in
                ["frame_index", "episode_index", "index", "task_index"]}
    features.update({"timestamp": {"dtype": "float32", "shape": [1]},
                     "observation.state": {"dtype": "float32", "shape": [2]},
                     "action": {"dtype": "float32", "shape": [2]},
                     "observation.depths.cam0": {"dtype": "uint16", "shape": [2, 2]}})
    for cam in range(3):
        features[f"observation.images.cam{cam}"] = {"dtype": "video", "shape": [2, 2, 3]}
    info = {"codebase_version": "v2.1", "robot_type": "dual_fr3_wuji", "total_episodes": 1,
            "total_frames": 6, "total_tasks": 1, "total_videos": 3, "total_chunks": 1,
            "chunks_size": 1000, "fps": 30, "splits": {"train": "0:1"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": features}
    table = pa.table({"frame_index": pa.array(range(6), type=pa.int64()),
                      "episode_index": pa.array([0] * 6, type=pa.int64()),
                      "index": pa.array(range(6), type=pa.int64()),
                      "task_index": pa.array([0] * 6, type=pa.int64()),
                      "timestamp": pa.array([i / 30 for i in range(6)], type=pa.float32()),
                      "observation.state": pa.array([[i, i + 1] for i in range(6)], type=pa.list_(pa.float32(), 2)),
                      "action": pa.array([[i + 2, i + 3] for i in range(6)], type=pa.list_(pa.float32(), 2)),
                      "observation.depths.cam0": pa.array([[[i, 2], [3, 65535]] for i in range(6)],
                                                          type=pa.list_(pa.list_(pa.uint16(), 2), 2))})
    table = table.replace_schema_metadata({b"test": b"metadata-preserved"})
    file = root / "data/chunk-000/episode_000000.parquet"
    file.parent.mkdir(parents=True)
    pq.write_table(table, file, row_group_size=2)
    for cam in range(3):
        file = root / f"videos/chunk-000/observation.images.cam{cam}/episode_000000.mp4"
        file.parent.mkdir(parents=True)
        file.write_bytes(b"synthetic video bytes for transport test " + bytes([cam]))
    atomic_json(root / "meta/info.json", info)
    jsonl(root / "meta/tasks.jsonl", [{"task_index": 0, "task": "Original overall task"}])
    jsonl(root / "meta/episodes.jsonl", [{"episode_index": 0, "length": 6, "tasks": ["Original overall task"],
                                          "task_index": 0, "source_recording_id": "source-id"}])
    jsonl(root / "meta/episodes_stats.jsonl", [{"episode_index": 0, "stats": {
        "task_index": {"min": [0], "max": [0], "mean": [0], "std": [0], "count": [6]},
        "observation.state": {"kept": "unchanged"}, "observation.depths.cam0": {"kept": "unchanged"}}}])


def labels():
    return {0: [{"start_frame": 0, "end_frame": 2, "text": "Hold the truss.", "evidence_status": "observed"},
                {"start_frame": 2, "end_frame": 6, "text": "Move the tomato.", "evidence_status": "observed"}]}


class FakeEngine:
    def __init__(self): self.calls = 0
    def annotate(self, *args, **kwargs):
        self.calls += 1
        return labels()[0]


class RoundtripTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "input"
        fixture(self.source)

    def tearDown(self): self.temp.cleanup()

    def test_sensor_values_video_and_schema_preserved(self):
        source = inspect(self.source)
        out = self.root / "output"
        export_dataset(source, out, labels())
        result = validate_output(source, out, labels())
        self.assertTrue(result["all_non_task_values_equal"])
        self.assertEqual({p.name for p in out.iterdir()}, {"data", "meta", "videos"})
        table = pq.read_table(out / "data/chunk-000/episode_000000.parquet")
        self.assertEqual(table["task_index"].to_pylist(), [0, 0, 1, 1, 1, 1])
        self.assertEqual(read_json(out / "meta/info.json")["total_tasks"], 2)
        self.assertNotIn("task_index", json.loads((out / "meta/episodes.jsonl").read_text()))

    def test_invalid_and_uncertain_coverage_rejected(self):
        for change in [{"start_frame": 1}, {"end_frame": 7}, {"evidence_status": "uncertain"}]:
            steps = deepcopy(labels()[0]); steps[0].update(change)
            with self.assertRaises(ValueError): validate_segments(steps, 6)

    def test_existing_destination_never_overwritten(self):
        out = self.root / "output";out.mkdir()
        with self.assertRaises(FileExistsError): export_dataset(inspect(self.source), out, labels())

    def test_atomic_publication_does_not_replace_even_empty_directory(self):
        a, b = self.root / "a", self.root / "b";a.mkdir();b.mkdir()
        with self.assertRaises(FileExistsError): publish_new_directory(a, b)
        self.assertTrue(a.exists());self.assertTrue(b.exists())

    def test_low_disk_space_is_rejected_before_annotation(self):
        with patch("lerobot_annotation.pipeline.shutil.disk_usage") as usage:
            usage.return_value.free = 0
            with self.assertRaises(OSError):
                require_space(inspect(self.source), self.root / "output")

    def test_corrupted_non_task_value_blocks_validation(self):
        source = inspect(self.source);out = self.root / "output"
        export_dataset(source, out, labels())
        file = out / "data/chunk-000/episode_000000.parquet"
        table = pq.read_table(file)
        table = table.set_column(table.schema.get_field_index("index"), "index", pa.array([99] * 6, type=pa.int64()))
        pq.write_table(table, file)
        with self.assertRaises(ValueError): validate_output(source, out, labels())

    def test_job_runs_resumes_without_model_and_keeps_work_separate(self):
        engine = FakeEngine();out = self.root / "output";work = self.root / "work"
        run(self.source, out, work, model="test-only", annotator=engine)
        run(self.source, out, work, model="test-only", annotator=engine)
        self.assertEqual(engine.calls, 1)
        self.assertEqual(read_json(work / "status.json")["status"], "completed")
        self.assertFalse((out / "status.json").exists())

    def test_append_only_upload_and_resume(self):
        out = self.root / "output";export_dataset(inspect(self.source), out, labels())
        receiver = self.root / "receiver";receiver.mkdir()
        destination = receiver / "new-dataset";state = self.root / "upload-work"
        upload(out, destination, state)
        upload(out, destination, state)
        self.assertEqual(read_json(state / "transfer.json")["status"], "completed")
        self.assertEqual((destination / "meta/info.json").read_bytes(), (out / "meta/info.json").read_bytes())
        (destination / "meta/tasks.jsonl").write_text("conflicting contents")
        with self.assertRaises(ValueError): upload(out, destination, state)
        self.assertEqual((destination / "meta/tasks.jsonl").read_text(), "conflicting contents")

    def test_recover_after_publication_before_completion_record(self):
        engine = FakeEngine();out = self.root / "output";work = self.root / "work"
        run(self.source, out, work, model="test-only", annotator=engine)
        (work / "completed.json").unlink()
        run(self.source, out, work, model="test-only", annotator=engine)
        self.assertEqual(engine.calls, 1)
        self.assertTrue((work / "completed.json").exists())

    def test_unowned_remote_destination_rejected(self):
        out = self.root / "output";export_dataset(inspect(self.source), out, labels())
        receiver = self.root / "receiver";receiver.mkdir()
        with self.assertRaises(FileExistsError): upload(out, receiver, self.root / "state")

    def test_model_seconds_restore_source_frames_and_reject_uncertain(self):
        raw = json.dumps({"steps": [{"start_sec": 0, "end_sec": 1, "text": "Hold truss.", "evidence_status": "observed"}]})
        result = normalize_response(raw, start=900, end=930, fps=30)
        self.assertEqual((result[0]["start_frame"], result[0]["end_frame"]), (900, 930))
        with self.assertRaises(ValueError): normalize_response(raw.replace('"observed"', '"uncertain"'), start=900, end=930, fps=30)

    def test_decimal_duration_is_validated_on_frame_grid_without_clamping(self):
        raw = json.dumps({"steps": [{"start_sec": 0, "end_sec": 27.333,
                                   "text": "Move the tomato.", "evidence_status": "observed"}]})
        result = normalize_response(raw, start=2700, end=3520, fps=30)
        self.assertEqual(result[0]["end_frame"], 3520)
        with self.assertRaises(ValueError):
            normalize_response(raw.replace('27.333', '27.4'), start=2700, end=3520, fps=30)
        with self.assertRaises(ValueError):
            normalize_response(raw.replace('27.333', '27.0'), start=2700, end=3520, fps=30)

    def test_revalidate_saved_response_without_another_api_call(self):
        window = self.root / "saved-window";window.mkdir()
        (window / "response_1.txt").write_text(json.dumps({"steps": [{"start_sec": 0,
            "end_sec": 27.333, "text": "Move the tomato.", "evidence_status": "observed"}]}))
        # No API key or real clip is required to revalidate a saved response.
        engine = ArkAnnotator(model="test-only", attempts=1, max_calls=1)
        engine.calls = 1
        result = engine.annotate("unused", start=2700, end=3520, fps=30,
            task=["test"], work=window, source_hash="test-source")
        self.assertEqual(result[-1]["end_frame"], 3520)
        self.assertEqual(engine.calls, 1)
        self.assertFalse(read_json(window / "revalidated_response.json")["network_call_required"])

    def test_short_tail_is_not_a_separate_request(self):
        self.assertEqual(windows(912, 30, 30), [[0, 912]])

    def test_prefix_selection_changes_metadata_without_copying_input(self):
        source = inspect(self.source)
        selected = select_prefix(source, 1)
        self.assertEqual(selected["root"], source["root"])
        self.assertEqual(selected["info"]["total_episodes"], 1)
        self.assertEqual(selected["info"]["total_frames"], 6)
        self.assertEqual(selected["info"]["splits"], {"train": "0:1"})
        self.assertIs(selected["episodes"][0], source["episodes"][0])
        with self.assertRaises(ValueError): select_prefix(source, 0)
        with self.assertRaises(ValueError): select_prefix(source, 2)

    def test_uncertain_model_response_retries_then_uses_validated_cache(self):
        # Real client functions are patched; no file upload or model request.
        import api.api_seed_openai_easy_no_short_step.api_client_qwen_v2 as client
        window = self.root / "window";window.mkdir();(window / "clip.mp4").write_bytes(b"test")
        good = json.dumps({"steps": [{"start_sec": 0, "end_sec": 1, "text": "Hold the truss.", "evidence_status": "observed"}]})
        bad = good.replace('"observed"', '"uncertain"')
        engine = ArkAnnotator(model="test-only", attempts=2, max_calls=2)
        with patch.dict("os.environ", {"ARK_API_KEY": "test-placeholder-not-a-credential"}), \
             patch.object(client, "create_api_client"), \
             patch.object(client, "call_model_api", side_effect=[(bad, {}), (good, {})]) as calls, \
             patch("demos.franka_wuji.run_demo.probe", return_value={"frames": 30, "fps": 30}), \
             patch("lerobot_annotation.model.time.sleep"):
            kwargs = dict(start=0, end=30, fps=30, task=["original"], work=window, source_hash="test-hash")
            first = engine.annotate("unused-test-video", **kwargs)
            second = engine.annotate("unused-test-video", **kwargs)
            self.assertEqual(first, second)
            self.assertEqual(calls.call_count, 2)
            self.assertEqual(engine.calls, 2)
            self.assertEqual([r["status"] for r in read_json(window / "attempts.json")], ["failed", "passed"])

    def test_failed_automatic_annotation_does_not_publish_dataset(self):
        class UncertainEngine(FakeEngine):
            def annotate(self, *args, **kwargs):
                result = labels()[0];result[0]["evidence_status"] = "uncertain"
                return result
        output, work = self.root / "output", self.root / "work"
        with self.assertRaises(ValueError):
            run(self.source, output, work, model="test-only", annotator=UncertainEngine())
        self.assertFalse(output.exists())
        self.assertEqual(read_json(work / "status.json")["status"], "failed")


if __name__ == "__main__": unittest.main()
