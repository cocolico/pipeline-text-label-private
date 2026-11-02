from __future__ import annotations

import importlib.util
import json
import tempfile
import sys
import types
import unittest
from unittest.mock import patch
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_stage4():
    common = types.ModuleType("common")
    for name in (
        "add_api_cli_args",
        "add_sampling_cli_args",
        "api_config_from_args",
        "build_video_api_content",
        "build_retry_prefix",
        "build_source_video_meta",
        "can_open_video",
        "call_chat_completion",
        "collect_videos",
        "cut_video_segment_ffmpeg",
        "default_output_root",
        "ensure_frame_index_fps",
        "ensure_video_out_dir_safe",
        "extract_json_from_response",
        "format_duration",
        "get_video_metadata",
        "initialize_api_client",
        "guard_schema_fingerprint",
        "now_utc_iso",
        "read_json",
        "sample_video_to_frame_index",
        "sampling_config_from_args",
        "sanitize_filename",
        "stage4_atomic_clip_relpath",
        "update_run_summary",
        "video_id_from_path",
        "write_frame_manifest",
        "write_json",
        "write_text",
    ):
        setattr(common, name, lambda *args, **kwargs: None)
    common.VIDEO_EXTS = (".mp4", ".m4v", ".avi", ".mov", ".mkv", ".webm")
    common.logger = types.SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
    )

    spec = importlib.util.spec_from_file_location(
        "stage4_object_inventory_test_target", ROOT / "stage4_atomic_actions.py"
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"common": common, spec.name: module}):
        spec.loader.exec_module(module)
    return module


M = _load_stage4()


def relation(object_name, hand="right hand", rel="grasp"):
    return {"object": object_name, "hand": hand, "relation": rel}


def action(patient, actor="right_hand", act="move", caption="The hand moves the object."):
    return {"actor": actor, "action": act, "patient": patient, "caption": caption}


class CollectObjectMentionsTests(unittest.TestCase):
    def test_dedup_across_steps_and_fields(self):
        final_steps = [
            {
                "step_id": 1,
                "hand_object_relations": [relation("cup"), relation("plate")],
                "atomic_actions": [action("cup")],
            },
            {
                "step_id": 2,
                "hand_object_relations": [relation("cup")],
                "atomic_actions": [action("plate"), action("cup")],
            },
        ]
        mentions = M._collect_object_mentions(final_steps)
        self.assertEqual(mentions, ["cup", "plate"])

    def test_colon_multi_entity_patient_is_split(self):
        final_steps = [
            {
                "step_id": 1,
                "hand_object_relations": [],
                "atomic_actions": [action("cup:saucer")],
            }
        ]
        mentions = M._collect_object_mentions(final_steps)
        self.assertEqual(mentions, ["cup", "saucer"])

    def test_empty_when_no_mentions(self):
        final_steps = [{"step_id": 1, "hand_object_relations": [], "atomic_actions": []}]
        self.assertEqual(M._collect_object_mentions(final_steps), [])


class ValidateObjectInventoryTests(unittest.TestCase):
    def _valid_response(self):
        return {
            "object_inventory": [
                {"entity_ref": "obj_cup", "canonical_name": "cup", "identity_scope": "instance"},
                {"entity_ref": "obj_plate", "canonical_name": "plate", "identity_scope": "instance"},
            ],
            "mentions": {"cup": "obj_cup", "plate": "obj_plate"},
        }

    def test_accepts_valid_response(self):
        inventory, mention_to_ref, errors = M.validate_object_inventory(
            self._valid_response(), ["cup", "plate"]
        )
        self.assertEqual(errors, [])
        self.assertIsNotNone(inventory)
        self.assertEqual(mention_to_ref, {"cup": "obj_cup", "plate": "obj_plate"})

    def test_rejects_duplicate_entity_ref(self):
        resp = self._valid_response()
        resp["object_inventory"].append(
            {"entity_ref": "obj_cup", "canonical_name": "another cup", "identity_scope": "instance"}
        )
        inventory, mention_to_ref, errors = M.validate_object_inventory(resp, ["cup", "plate"])
        self.assertIsNone(inventory)
        self.assertIsNone(mention_to_ref)
        self.assertTrue(any("duplicate" in e.lower() for e in errors))

    def test_rejects_missing_mention_coverage(self):
        resp = self._valid_response()
        del resp["mentions"]["plate"]
        inventory, mention_to_ref, errors = M.validate_object_inventory(resp, ["cup", "plate"])
        self.assertIsNone(inventory)
        self.assertIsNone(mention_to_ref)
        self.assertTrue(any("missing" in e.lower() for e in errors))

    def test_rejects_invalid_identity_scope(self):
        resp = self._valid_response()
        resp["object_inventory"][0]["identity_scope"] = "maybe"
        inventory, mention_to_ref, errors = M.validate_object_inventory(resp, ["cup", "plate"])
        self.assertIsNone(inventory)
        self.assertIsNone(mention_to_ref)
        self.assertTrue(any("identity_scope" in e for e in errors))

    def test_rejects_mention_mapped_to_unknown_entity_ref(self):
        resp = self._valid_response()
        resp["mentions"]["cup"] = "obj_does_not_exist"
        inventory, mention_to_ref, errors = M.validate_object_inventory(resp, ["cup", "plate"])
        self.assertIsNone(inventory)
        self.assertIsNone(mention_to_ref)
        self.assertTrue(any("not found" in e.lower() for e in errors))


class ApplyEntityRefsTests(unittest.TestCase):
    def test_injects_entity_ref_without_altering_text_fields(self):
        final_steps = [
            {
                "step_id": 1,
                "hand_object_relations": [relation("cup")],
                "atomic_actions": [action("cup")],
            }
        ]
        mention_to_ref = {"cup": "obj_cup"}
        M._apply_entity_refs(final_steps, mention_to_ref)

        rel = final_steps[0]["hand_object_relations"][0]
        act = final_steps[0]["atomic_actions"][0]
        self.assertEqual(rel["object"], "cup")
        self.assertEqual(rel["entity_ref"], "obj_cup")
        self.assertEqual(act["patient"], "cup")
        self.assertEqual(act["entity_ref"], "obj_cup")

    def test_colon_multi_entity_patient_gets_colon_joined_entity_ref(self):
        final_steps = [
            {
                "step_id": 1,
                "hand_object_relations": [],
                "atomic_actions": [action("cup:saucer")],
            }
        ]
        mention_to_ref = {"cup": "obj_cup", "saucer": "obj_saucer"}
        M._apply_entity_refs(final_steps, mention_to_ref)

        act = final_steps[0]["atomic_actions"][0]
        self.assertEqual(act["patient"], "cup:saucer")
        self.assertEqual(act["entity_ref"], "obj_cup:obj_saucer")

    def test_stable_entity_ref_across_steps_despite_text_variation(self):
        final_steps = [
            {
                "step_id": 1,
                "hand_object_relations": [relation("vertical support wooden block")],
                "atomic_actions": [],
            },
            {
                "step_id": 3,
                "hand_object_relations": [],
                "atomic_actions": [action("scattered_loose_wooden_blocks")],
            },
        ]
        mention_to_ref = {
            "vertical support wooden block": "obj_wooden_blocks",
            "scattered_loose_wooden_blocks": "obj_wooden_blocks",
        }
        M._apply_entity_refs(final_steps, mention_to_ref)

        self.assertEqual(
            final_steps[0]["hand_object_relations"][0]["entity_ref"], "obj_wooden_blocks"
        )
        self.assertEqual(final_steps[1]["atomic_actions"][0]["entity_ref"], "obj_wooden_blocks")

    def test_missing_mapping_leaves_no_entity_ref(self):
        final_steps = [
            {
                "step_id": 1,
                "hand_object_relations": [relation("mystery_item")],
                "atomic_actions": [action("mystery_item")],
            }
        ]
        M._apply_entity_refs(final_steps, {})

        rel = final_steps[0]["hand_object_relations"][0]
        act = final_steps[0]["atomic_actions"][0]
        self.assertNotIn("entity_ref", rel)
        self.assertNotIn("entity_ref", act)

    def test_stale_refs_are_removed_when_mapping_is_missing(self):
        final_steps = [{
            "step_id": 1,
            "hand_object_relations": [dict(relation("cup"), entity_ref="stale")],
            "atomic_actions": [dict(action("cup"), entity_ref="stale")],
        }]
        M._apply_entity_refs(final_steps, {})
        self.assertNotIn("entity_ref", final_steps[0]["hand_object_relations"][0])
        self.assertNotIn("entity_ref", final_steps[0]["atomic_actions"][0])

    def test_consistency_validator_rejects_cross_step_drift(self):
        final_steps = [
            {"step_id": 1, "hand_object_relations": [dict(relation("cup"), entity_ref="obj_cup")], "atomic_actions": []},
            {"step_id": 2, "hand_object_relations": [], "atomic_actions": [dict(action("cup"), entity_ref="obj_other")]},
        ]
        inventory = [{"entity_ref": "obj_cup", "canonical_name": "cup", "identity_scope": "instance"}]
        errors = M._validate_applied_entity_refs(final_steps, {"cup": "obj_cup"}, inventory)
        self.assertTrue(any("Step 2" in error for error in errors))

    def test_syncs_resolved_refs_back_to_step_cache(self):
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "atomic_actions.json"
            path.write_text(json.dumps({"step_id": 1, "atomic_actions": [action("cup")]}), encoding="utf-8")
            final_steps = [{"step_id": 1, "atomic_actions": [dict(action("cup"), entity_ref="obj_cup")]}]
            old_read_json, old_write_json = M.read_json, M.write_json
            try:
                M.read_json = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
                M.write_json = lambda p, value: Path(p).write_text(json.dumps(value), encoding="utf-8")
                M._sync_entity_refs_to_step_caches(final_steps, {1: str(path)})
            finally:
                M.read_json, M.write_json = old_read_json, old_write_json
            cached = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(cached["atomic_actions"][0]["entity_ref"], "obj_cup")



class EmptyStage4Tests(unittest.TestCase):
    def _inputs(self, root, steps=None, segments=None):
        stage3 = root / "rgb" / "stage3"
        stage2 = root / "rgb" / "stage2"
        stage3.mkdir(parents=True)
        stage2.mkdir(parents=True)
        (stage3 / "step_plan.json").write_text(json.dumps({
            "high_level_goal": "No valid upstream steps", "steps": steps or [],
        }))
        (stage2 / "step_segments.json").write_text(json.dumps({"segments": segments or []}))
        return stage3 / "step_plan.json", stage2 / "step_segments.json"

    def test_screened_out_video_completes_without_model_calls_and_can_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, segments = self._inputs(root)
            summary = {}
            def write_json(path, value):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text(json.dumps(value))
            with patch.multiple(
                M,
                video_id_from_path=lambda _: "rgb",
                get_video_metadata=lambda _: {"fps": 30, "total_frames": 60, "duration_sec": 2},
                build_source_video_meta=lambda _: {"id": "rgb", "path": "rgb.mp4"},
                guard_schema_fingerprint=lambda *a, **k: "test-schema",
                read_json=lambda path: json.loads(Path(path).read_text()),
                write_json=write_json,
                update_run_summary=lambda path, value: summary.update(value),
            ), patch.object(M, "initialize_api_client") as client, patch.object(M.shutil, "which", return_value="ffmpeg"):
                cfg = types.SimpleNamespace(video_fps=5)
                M.run_stage4_for_video("rgb.mp4", tmp, cfg, cfg, False, 1)
                client.assert_not_called()
                self.assertEqual(summary["stage4"]["status"], "completed")
                self.assertEqual(summary["stage4"]["token_usage"]["api_calls"], 0)
                self.assertEqual(json.loads((root / "rgb" / "final_plan.json").read_text())["steps"], [])
                final = root / "rgb" / "stage4" / "atomic_plan_with_clips.json"
                inventory = root / "rgb" / "stage4" / "object_inventory.json"
                self.assertTrue(M._can_resume_stage4_final(str(final), str(segments), str(plan), str(inventory)))
                plan.write_text(json.dumps({"high_level_goal": "changed", "steps": []}))
                self.assertFalse(M._can_resume_stage4_final(str(final), str(segments), str(plan), str(inventory)))

    def test_empty_stage3_with_existing_segments_fails_before_model_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._inputs(Path(tmp), segments=[{"step_id": 1}])
            with patch.multiple(
                M, video_id_from_path=lambda _: "rgb",
                get_video_metadata=lambda _: {"fps": 30},
                read_json=lambda path: json.loads(Path(path).read_text()),
            ), patch.object(M, "initialize_api_client") as client, patch.object(M.shutil, "which", return_value="ffmpeg"):
                cfg = types.SimpleNamespace(video_fps=5)
                with self.assertRaisesRegex(RuntimeError, "Stage 2 has segments"):
                    M.run_stage4_for_video("rgb.mp4", tmp, cfg, cfg, False, 1)
                client.assert_not_called()


class Stage4ResumeTests(unittest.TestCase):
    def test_step_local_refs_do_not_invalidate_global_refs_but_semantic_changes_do(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage4 = root / "stage4"
            stage4.mkdir()
            final_path = stage4 / "atomic_plan_with_clips.json"
            inventory_path = stage4 / "object_inventory.json"
            plan_path = root / "step_plan.json"
            segments_path = root / "step_segments.json"
            plan_step = {
                "step_id": 1, "step_goal": "Lift the cup",
                "hand_object_relations": [{**relation("cup"), "entity_ref": "step_001_obj_001"}],
            }
            final_step = {
                **plan_step, "clip_relpath": "../stage1/cup.mp4",
                "hand_object_relations": [{**relation("cup"), "entity_ref": "obj_cup"}],
                "atomic_actions": [{
                    **action("cup"), "atomic_action_id": 1, "entity_ref": "obj_cup",
                    "start_frame_0based": 0, "end_frame_0based_exclusive": 30,
                    "end_frame_0based_inclusive": 29,
                }],
            }
            inventory = [{"entity_ref": "obj_cup", "canonical_name": "cup", "identity_scope": "instance"}]
            final = {"high_level_goal": "Lift the cup", "steps": [final_step], "object_inventory": inventory}
            plan = {"high_level_goal": "Lift the cup", "steps": [plan_step]}
            final_path.write_text(json.dumps(final))
            plan_path.write_text(json.dumps(plan))
            segments_path.write_text(json.dumps({"segments": [{"step_id": 1, "clip_relpath": "stage1/cup.mp4"}]}))
            inventory_path.write_text(json.dumps({"object_inventory": inventory, "mentions": {"cup": "obj_cup"}}))
            with patch.object(M, "read_json", side_effect=lambda path: json.loads(Path(path).read_text())):
                can_resume = lambda: M._can_resume_stage4_final(str(final_path), str(segments_path), str(plan_path), str(inventory_path))
                self.assertTrue(can_resume())
                plan["high_level_goal"] = "Wash the cup"
                plan_path.write_text(json.dumps(plan))
                self.assertFalse(can_resume())
                plan["high_level_goal"] = "Lift the cup"
                plan["steps"][0]["hand_object_relations"][0]["relation"] = "release"
                plan_path.write_text(json.dumps(plan))
                self.assertFalse(can_resume())
                plan["steps"][0]["hand_object_relations"][0]["relation"] = "grasp"
                plan_path.write_text(json.dumps(plan))
                final["steps"][0]["atomic_actions"][0]["entity_ref"] = "wrong_ref"
                final_path.write_text(json.dumps(final))
                self.assertFalse(can_resume())

if __name__ == "__main__":
    unittest.main()
