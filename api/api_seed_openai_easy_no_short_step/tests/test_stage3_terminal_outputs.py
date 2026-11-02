import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import stage3_refine_and_keyframes as stage3


class Stage3TerminalOutputsTests(unittest.TestCase):
    def test_text_only_summary_tracks_success_resume_and_finalization_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_path = os.path.join(tmp, "demo.mp4")
            video_out = os.path.join(tmp, "out", "demo")
            stage3_dir = os.path.join(video_out, "stage3")
            os.makedirs(stage3_dir)
            summary_path = os.path.join(video_out, "run_summary.json")
            stage3.write_json(summary_path, {"stage1": {"status": "completed"}})
            plan = {"source_video": {}, "high_level_goal": "No effective action", "steps": []}
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "api_calls": 0}

            def refine(**kwargs):
                self.assertEqual(stage3.read_json(summary_path)["stage3"]["status"], "running")
                stage3.write_json(os.path.join(stage3_dir, "step_plan.json"), plan)
                stage3.write_json(os.path.join(stage3_dir, "hand_steps.json"), {"hand_steps": []})
                return video_out, plan, usage

            with patch.object(stage3, "refine_hand_steps_text_only", side_effect=refine):
                for _ in range(2):
                    stage3._run_stage3_text_only_hand_steps(
                        video_path=video_path, output_root=os.path.join(tmp, "out"),
                        api_cfg=None, overwrite=False, max_retries=1,
                    )
                    summary = stage3.read_json(summary_path)
                    self.assertEqual(summary["stage1"]["status"], "completed")
                    self.assertEqual(summary["stage3"]["status"], "completed")
                    self.assertEqual(summary["stage3"]["step_count"], 0)
                    self.assertEqual(summary["stage3"]["token_usage"], usage)
                    for key, value in summary["stage3"].items():
                        if key.endswith("_path"):
                            self.assertTrue(os.path.isfile(os.path.join(video_out, value)))
                with patch.object(stage3, "_generate_final_plan_without_stage4", side_effect=OSError("disk failure")):
                    with self.assertRaises(OSError):
                        stage3._run_stage3_text_only_hand_steps(
                            video_path=video_path, output_root=os.path.join(tmp, "out"),
                            api_cfg=None, overwrite=False, max_retries=1,
                        )
                summary = stage3.read_json(summary_path)
                self.assertEqual(summary["stage3"]["status"], "failed")
                self.assertEqual(summary["stage3"]["error_type"], "OSError")

    def test_empty_plan_writes_normal_empty_terminal_outputs(self) -> None:
        plan = {
            "source_video": {},
            "high_level_goal": "No effective hand action was detected.",
            "steps": [],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            stage3_dir = os.path.join(tmpdir, "stage3")
            os.makedirs(stage3_dir)
            stage3._generate_final_plan_without_stage4(tmpdir, plan, stage3_dir)
            with open(os.path.join(tmpdir, "final_plan.json"), "r", encoding="utf-8") as fh:
                final_plan = json.load(fh)
            with open(os.path.join(stage3_dir, "instruction_subtask.json"), "r", encoding="utf-8") as fh:
                subtasks = json.load(fh)
        self.assertEqual(final_plan["steps"], [])
        self.assertEqual(final_plan["object_inventory"], [])
        self.assertEqual(subtasks["sub_task"], [])

    def test_final_plan_geometry_is_restored_as_exact_upstream_copy(self) -> None:
        upstream_tracks = {
            "left_hand": [{
                "frame_id": 11,
                "bbox_xyxy": [10, 20, 30, 40],
                "bbox_source": "upstream_sam3",
            }],
            "right_hand": [],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            stage2_dir = os.path.join(tmpdir, "stage2")
            stage3_dir = os.path.join(tmpdir, "stage3")
            os.makedirs(stage2_dir); os.makedirs(stage3_dir)
            with open(os.path.join(stage2_dir, "hand_step_boundaries.json"), "w", encoding="utf-8") as fh:
                json.dump({"hand_steps": [{
                    "hand_step_id": 1,
                    "start_frame_id": 10,
                    "end_frame_id": 12,
                    "frame_ids": [10, 11, 12],
                    "bbox_tracks": upstream_tracks,
                }]}, fh)
            plan = {"source_video": {}, "high_level_goal": "goal", "steps": [{
                "step_id": 1,
                "step_goal": "action",
                "frame_ids": [999],
                "bbox_tracks": {"left_hand": [{"frame_id": 999, "bbox_xyxy": [1, 1, 2, 2]}]},
                "object_inventory": [],
                "hand_object_relations": [],
            }]}
            stage3._generate_final_plan_without_stage4(tmpdir, plan, stage3_dir)
            with open(os.path.join(tmpdir, "final_plan.json"), "r", encoding="utf-8") as fh:
                final_plan = json.load(fh)
        step = final_plan["steps"][0]
        self.assertEqual(step["frame_ids"], [10, 11, 12])
        self.assertEqual(step["bbox_tracks"], upstream_tracks)
        self.assertEqual(step["start_frame_0based"], 10)
        self.assertEqual(step["end_frame_0based_inclusive"], 12)
        self.assertEqual(step["geometry_source"], "upstream_hand_screening_exact_copy")

    def test_relation_validator_accepts_required_three_fields_without_entity_ref(self) -> None:
        """entity_ref is optional because the global inventory assigns it later."""
        normalized, errors = stage3._validate_hand_object_relations([
            {"object": "cup", "hand": "right hand", "relation": "holds"}
        ])
        self.assertEqual(errors, [])
        self.assertEqual(normalized, [{
            "object": "cup", "hand": "right hand", "relation": "holds", "entity_ref": ""
        }])

    def test_generate_final_plan_without_stage4_falls_back_when_entity_ref_missing(self) -> None:
        stage3_plan = {
            "source_video": {},
            "high_level_goal": "test goal",
            "steps": [
                {
                    "step_id": 1,
                    "step_goal": "pick up the cup",
                    "hand_object_relations": [
                        {"object": "cup", "hand": "right hand", "relation": "holds"}
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            stage3_dir = os.path.join(tmpdir, "stage3")
            os.makedirs(stage3_dir, exist_ok=True)
            stage3._generate_final_plan_without_stage4(tmpdir, stage3_plan, stage3_dir)
            inventory_path = os.path.join(stage3_dir, "object_inventory.json")
            self.assertTrue(os.path.exists(inventory_path))
            with open(inventory_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            self.assertTrue(payload["object_inventory"])
            self.assertTrue(payload["warnings"])
            self.assertEqual(payload["object_inventory"][0]["canonical_name"], "cup")
            self.assertEqual(payload["object_inventory"][0]["identity_scope"], "instance")

    def test_generate_final_plan_without_stage4_writes_generated_entity_refs_back_to_relations(self) -> None:
        stage3_plan = {
            "source_video": {},
            "high_level_goal": "test goal",
            "steps": [
                {
                    "step_id": 1,
                    "step_goal": "pick up the green cleaning cloth",
                    "hand_object_relations": [
                        {"object": "green cleaning cloth", "hand": "both hands", "relation": "grasping"}
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            stage3_dir = os.path.join(tmpdir, "stage3")
            os.makedirs(stage3_dir, exist_ok=True)
            stage3._generate_final_plan_without_stage4(tmpdir, stage3_plan, stage3_dir)

            with open(os.path.join(tmpdir, "final_plan.json"), "r", encoding="utf-8") as fh:
                final_plan = json.load(fh)
            rel = final_plan["steps"][0]["hand_object_relations"][0]
            self.assertEqual(rel["entity_ref"], "obj_green_cleaning_cloth_1")
            inventory_refs = {entry["entity_ref"] for entry in final_plan["object_inventory"]}
            self.assertIn(rel["entity_ref"], inventory_refs)

    def test_resume_writes_final_plan_and_object_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = os.path.join(tmpdir, "out")
            video_path = os.path.join(tmpdir, "demo.mp4")
            with open(video_path, "w", encoding="utf-8"):
                pass

            video_out = os.path.join(output_root, "demo")
            os.makedirs(os.path.join(video_out, "stage1"), exist_ok=True)
            os.makedirs(os.path.join(video_out, "stage2"), exist_ok=True)
            os.makedirs(os.path.join(video_out, "stage3"), exist_ok=True)

            with open(os.path.join(video_out, "stage1", "draft_plan.json"), "w", encoding="utf-8") as f:
                f.write('{"high_level_goal": "test goal"}')
            with open(os.path.join(video_out, "stage2", "step_segments.json"), "w", encoding="utf-8") as f:
                f.write('{"segments": []}')
            with open(os.path.join(video_out, "stage3", "step_plan.json"), "w", encoding="utf-8") as f:
                f.write(
                    '{"steps": [{"step_id": 1, "step_goal": "pick up the cup", "hand_object_relations": [{"object": "cup", "hand": "right hand", "relation": "holds", "entity_ref": "obj_1"}]}]}'
                )
            with open(os.path.join(video_out, "run_summary.json"), "w", encoding="utf-8") as f:
                f.write('{}')

            api_cfg = type("ApiCfg", (), {"video_fps": 10.0})()
            sampling_cfg = type("SamplingCfg", (), {"max_frames": 8, "resize_dimension": 256, "jpeg_quality": 90, "frame_index_fps": 10.0})()

            with patch.object(stage3, "ensure_frame_index_fps"), patch.object(stage3, "ensure_video_out_dir_safe"), patch.object(stage3, "guard_schema_fingerprint", return_value="fp"), patch.object(stage3, "_can_resume_stage3_step_plan", return_value=True), patch.object(stage3, "read_json", side_effect=lambda path: {
                os.path.join(video_out, "stage3", "step_plan.json"): {
                    "source_video": {},
                    "high_level_goal": "test goal",
                    "steps": [
                        {
                            "step_id": 1,
                            "step_goal": "pick up the cup",
                            "hand_object_relations": [
                                {"object": "cup", "hand": "right hand", "relation": "holds", "entity_ref": "obj_1"}
                            ],
                        }
                    ],
                },
                os.path.join(video_out, "stage2", "step_segments.json"): {"segments": []},
                os.path.join(video_out, "stage1", "draft_plan.json"): {"high_level_goal": "test goal"},
            }.get(path, {})), patch.object(stage3, "update_run_summary"), patch.object(stage3, "write_json", side_effect=stage3.write_json):
                stage3.run_stage3_for_video(
                    video_path,
                    output_root,
                    api_cfg,
                    sampling_cfg,
                    overwrite=False,
                    max_retries=1,
                    allow_legacy_resume=False,
                )

            self.assertTrue(os.path.exists(os.path.join(video_out, "final_plan.json")))
            self.assertTrue(os.path.exists(os.path.join(video_out, "stage3", "object_inventory.json")))


if __name__ == "__main__":
    unittest.main()
